"""P-LCN: manuscript-aligned reference pipeline with synthetic demonstration data.

Multimodal fusion of well logs and cuttings thin-section images for
uncertainty-aware permeability prediction in volcanic reservoirs.
"""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from data_pipeline import (DATA_FILE, IMAGE_DIR, RAW_LOGS, TRAIN_WELLS, VAL_WELL,
                           BLIND_WELL, K_COL, PHI_COL, WELL_COL, DEPTH_COL,
                           IMAGE_SIZE, IMAGE_CHANNELS, LOG_WINDOW_L,
                           center_crop, load_image_pair, load_images_for_subset,
                           create_log_windows, load_and_split_by_well)
from evaluation import evaluate_predictions, summarize_runs, block_bootstrap_uncertainty

RANDOM_SEED = 2025
USE_RAW_IMAGES = True
KC_TAU = 0.60
BATCH_SIZE, MAX_EPOCHS, PATIENCE = 64, 200, 30
INIT_LR, LAMBDA_PHY = 2e-3, 1.0
LOG_EMB_DIM = IMG_EMB_DIM = 32
# Explicit L2 is separate from decoupled AdamW decay. This is a DEMO choice:
# the manuscript does not yet provide its verified historical coefficient.
LAMBDA_REG = 1e-4
WEIGHT_DECAY = 1e-4


@dataclass
class TrainingConfig:
    epochs: int = MAX_EPOCHS
    batch_size: int = BATCH_SIZE
    patience: int = PATIENCE
    warmup_steps: int = 65
    start_lr: float = 1e-6
    peak_lr: float = INIT_LR
    end_lr: float = 1e-6
    weight_decay: float = WEIGHT_DECAY
    lambda_phy: float = LAMBDA_PHY
    lambda_reg: float = LAMBDA_REG
    kc_tau: float = KC_TAU


def kc_ransac_filter(phi, logK, tau=KC_TAU, n_iter=1000, sample_frac=0.5,
                     random_state=RANDOM_SEED):
    """
    Asymmetric KC-RANSAC label filtering.

    Paper description:
      - Fit global trend log(K) = a * log(φ) + b in (logφ, logK) space
      - Compute residual r_KC = logK_core - (a*logφ + b)   [Eq.3]
      - Estimate inlier residual std σ_r via RANSAC
      - Mask samples where r_KC < -τ * σ_r  (ASYMMETRIC: only below-trend)
      - Retain all positive residuals (above-trend samples)

    Returns:
        mask: binary array, 1=retain, 0=reject
        a, b: fitted KC trend parameters
        sigma_r: inlier residual std
    """
    phi, logK = np.asarray(phi).reshape(-1), np.asarray(logK).reshape(-1)
    if len(phi) < 3 or len(phi) != len(logK) or not np.isfinite(phi).all() or not np.isfinite(logK).all():
        raise ValueError("KC fit needs matching finite arrays with at least three samples")
    if np.any(phi <= 0) or tau < 0 or n_iter < 1 or not 0 < sample_frac <= 1:
        raise ValueError("Invalid porosity, threshold or RANSAC configuration")
    rng = np.random.RandomState(random_state)

    # Work in (log φ, log K) space
    log_phi = np.log10(np.clip(phi, 1e-6, None))
    log_k = logK.copy()
    if np.ptp(log_phi) <= 1e-12:
        raise ValueError("KC slope cannot be identified from constant porosity")
    n = len(log_phi)
    n_sample = max(int(n * sample_frac), 2)

    best_inlier_count = -1
    best_a, best_b = None, None

    for _ in range(n_iter):
        # Random subset
        idx = rng.choice(n, size=n_sample, replace=False)
        x_s, y_s = log_phi[idx], log_k[idx]

        # Fit linear trend: logK = a * logφ + b
        A_mat = np.column_stack([x_s, np.ones(n_sample)])
        try:
            params, _, _, _ = np.linalg.lstsq(A_mat, y_s, rcond=None)
        except np.linalg.LinAlgError:
            continue
        a_c, b_c = params

        # Residuals on full dataset
        residuals = log_k - (a_c * log_phi + b_c)

        # Count inliers (within ±2σ for RANSAC consensus)
        med_abs_res = np.median(np.abs(residuals))
        sigma_est = max(med_abs_res / 0.6745, 1e-12)
        inlier_mask = np.abs(residuals) <= 2.0 * sigma_est
        inlier_count = inlier_mask.sum()

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_a, best_b = a_c, b_c

    # Final residuals with best model
    if best_a is None:
        raise ValueError("KC-RANSAC failed to find a valid trend")
    residuals_final = log_k - (best_a * log_phi + best_b)

    # Inlier σ_r estimation
    inlier_res = residuals_final[np.abs(residuals_final) <= 2.0 *
                                 max(np.median(np.abs(residuals_final)) / 0.6745, 1e-12)]
    sigma_r = np.std(inlier_res) if len(inlier_res) > 2 else np.std(residuals_final)
    sigma_r = max(float(sigma_r), 1e-12)

    # ASYMMETRIC masking: reject only samples BELOW the KC trend
    # r_KC < -τ * σ_r  →  rejected  (below-trend samples)
    # All positive residuals retained (above-trend)
    mask = np.ones(n, dtype=np.float32)
    reject_condition = residuals_final < -tau * sigma_r
    mask[reject_condition] = 0.0

    n_rejected = int((1 - mask).sum())
    pct_rejected = 100.0 * n_rejected / n
    print(f"  KC-RANSAC: trend logK = {best_a:.3f}*logphi + {best_b:.3f}")
    print(f"  KC-RANSAC: sigma_r = {sigma_r:.4f}, tau = {tau}")
    print(f"  KC-RANSAC: rejected {n_rejected}/{n} samples ({pct_rejected:.1f}%)")

    return mask, best_a, best_b, sigma_r


class DepthwiseSeparableConv1D(layers.Layer):
    """Single DSC block: DepthwiseConv1D → PointwiseConv1D → BN → Activation"""
    def __init__(self, filters, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.dw_conv = layers.DepthwiseConv1D(
            kernel_size=kernel_size, padding='same', use_bias=False)
        self.pw_conv = layers.Conv1D(filters, 1, use_bias=False)
        self.bn = layers.BatchNormalization()
        self.act = layers.Activation('relu')

    def call(self, x, training=False):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        x = self.bn(x, training=training)
        return self.act(x)


class LogExpertWithGating(layers.Layer):
    """
    Paper: "Log Expert with Lithology Gating"
    Input: (batch, L, n_curves) log window + (batch, n_litho) lithology probs
    Output: h_log_gated (batch, 32), sigma_log (batch, 1)
    """
    def __init__(self, emb_dim=LOG_EMB_DIM, **kwargs):
        super().__init__(**kwargs)
        self.emb_dim = emb_dim
        # 3 stacked DSC blocks (Paper: "three stacked DSC blocks")
        self.dsc1 = DepthwiseSeparableConv1D(16, kernel_size=3, name="dsc_1")
        self.dsc2 = DepthwiseSeparableConv1D(32, kernel_size=3, name="dsc_2")
        self.dsc3 = DepthwiseSeparableConv1D(emb_dim, kernel_size=3, name="dsc_3")
        self.gap = layers.GlobalAveragePooling1D()

        # Lithology gating (Eq.4)
        self.gate_dense = layers.Dense(self.emb_dim, name="gate_proj")

        # Positive learned fusion score; not a calibrated modality SD
        self.sigma_head = layers.Dense(1, name="sigma_log_raw")

    def call(self, inputs, training=False):
        log_input, lith_input = inputs  # (B,L,C), (B,n_litho)

        # 1D-CNN feature extraction
        h = self.dsc1(log_input, training=training)
        h = self.dsc2(h, training=training)
        h = self.dsc3(h, training=training)
        h_log = self.gap(h)  # (B, emb_dim=32)

        # Lithology gating (Eq.4): g = σ(W_g · P_litho + b_g)
        gate = tf.nn.sigmoid(self.gate_dense(lith_input))  # (B, 32)
        h_log_gated = h_log * gate  # element-wise (Eq.4: g ⊙ h_log)

        # Softplus keeps the learned fusion score positive
        sigma_log = tf.nn.softplus(self.sigma_head(h_log_gated)) + 1e-6

        return h_log_gated, sigma_log


# ── 3b. Image Expert ──────────────────────────────────────────
# Paper: "a truncated MobileNetV3-Small backbone"
# Input: PPL(3ch) + XPL(3ch) = 6-channel, center-cropped to 64x64

class ImageExpert(layers.Layer):
    """
    Paper: "Image Expert"

    When USE_RAW_IMAGES=True:
        - Accepts (B, 64, 64, 6) PPL+XPL concatenated input
        - 1×1 Conv2D adapter: 6ch → 3ch (learnable channel projection)
        - Truncated MobileNetV3-Small backbone (lightweight subset)
        - GlobalAveragePooling2D → Dense → 32-dim embedding

    When USE_RAW_IMAGES=False:
        - Accepts (B, micro_dim) pre-extracted features
        - MLP encoder → 32-dim embedding

    Output: h_img (batch, 32), sigma_img (batch, 1)
    """
    def __init__(self, emb_dim=IMG_EMB_DIM, use_raw_images=USE_RAW_IMAGES,
                 **kwargs):
        super().__init__(**kwargs)
        self.emb_dim = emb_dim
        self.use_raw_images = use_raw_images

        if use_raw_images:
            # Channel adapter: 6ch (PPL+XPL) → 3ch for MobileNetV3
            self.channel_adapter = layers.Conv2D(
                3, kernel_size=1, padding='same', use_bias=False,
                name="channel_adapter_6to3")

            # Truncated MobileNetV3-Small backbone
            # (Paper: "a truncated MobileNetV3-Small backbone")
            base = tf.keras.applications.MobileNetV3Small(
                input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                include_top=False,
                weights=None,
                minimalistic=False, include_preprocessing=False)

            # Select the truncation point for the lightweight backbone and
            # print the resolved layer for full transparency/reproducibility.
            target_layer_name = self._select_truncation_layer(base)
            truncated_output = base.get_layer(target_layer_name).output
            self.backbone = Model(
                inputs=base.input,
                outputs=truncated_output,
                name="MNv3_truncated")
            self.backbone.trainable = True

            self.pool = layers.GlobalAveragePooling2D()
            self.proj = layers.Dense(emb_dim, activation='relu',
                                     name="img_proj")
        else:
            # Fallback: MLP for pre-extracted features
            self.fc1 = layers.Dense(64, activation='swish')
            self.fc2 = layers.Dense(emb_dim, activation='swish')

        # Positive learned fusion-score head
        self.sigma_head = layers.Dense(1, name="sigma_img_raw")

    @staticmethod
    def _select_truncation_layer(base_model):
        """Resolve and print the layer at which the backbone is truncated."""
        # Inverted residual block IDs 0, 1, 2 = three bottlenecks.
        # End AFTER block 2's residual addition, not an SE operation.
        names = {layer.name for layer in base_model.layers}
        for candidate in ("expanded_conv_2_add", "expanded_conv_2/Add"):
            if candidate in names:
                return candidate
        raise RuntimeError("Cannot resolve third MobileNetV3-Small bottleneck endpoint")

    def call(self, x, training=False):
        if self.use_raw_images:
            # (B, 64, 64, 6) → adapter → (B, 64, 64, 3)
            x_3ch = self.channel_adapter(2.0 * x - 1.0)
            # MobileNetV3 backbone
            h = self.backbone(x_3ch, training=training)
            h = self.pool(h)
            h_img = self.proj(h)  # (B, 32)
        else:
            h = self.fc1(x)
            h_img = self.fc2(h)   # (B, 32)

        # Paper: "Softplus activation to ensure non-negativity"
        sigma_img = tf.nn.softplus(self.sigma_head(h_img)) + 1e-6
        return h_img, sigma_img


# ── 3c. Uncertainty-Aware Adaptive Fusion (IVW at feature level) ──
# Paper Eq.5-6: w_m = (1/σ²_m) / Σ(1/σ²_m)
#               z_fused = Σ w_m · h_m

class IVWFeatureFusion(layers.Layer):
    """
    Paper: "Uncertainty-Aware Adaptive Fusion"
    Fuses embeddings using normalized inverse-square learned scores.

    Input: h_log (B,D), sigma_log (B,1), h_img (B,D), sigma_img (B,1)
    Output: z_fused (B,D), w_log (B,1), w_img (B,1)
    """
    def call(self, inputs):
        h_log, sigma_log, h_img, sigma_img = inputs

        # Eq.5: w_m = (1/σ²_m) / Σ_m(1/σ²_m)
        var_log = tf.square(sigma_log) + 1e-6  # (B,1)
        var_img = tf.square(sigma_img) + 1e-6

        inv_var_log = 1.0 / var_log
        inv_var_img = 1.0 / var_img
        inv_var_sum = inv_var_log + inv_var_img

        w_log = inv_var_log / inv_var_sum  # (B,1)
        w_img = inv_var_img / inv_var_sum  # (B,1)

        # Eq.6: z_fused = w_log * h_log + w_img * h_img
        z_fused = w_log * h_log + w_img * h_img  # (B, D)

        return z_fused, w_log, w_img


# ── 3d. Porosity-Guided Cascaded Prediction Head ─────────────
# Paper Eq.7-9: z_fused → φ̂ → [z_fused; φ̂] → (μ_K, σ_total)

class CascadedPredictionHead(layers.Layer):
    """
    Paper: "Porosity-Guided Cascaded Prediction Head"

    Eq.7: φ̂ = f_φ(z_fused)
    Eq.8: z_aug = [z_fused; φ̂]
    Eq.9: (μ_K, σ_total) = f_K(z_aug)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Porosity head (Eq.7)
        self.phi_fc1 = layers.Dense(32, activation='relu')
        self.phi_out = layers.Dense(1, activation='sigmoid', name="phi_hat")

        # Permeability head (Eq.9)
        self.k_fc1 = layers.Dense(64, activation='relu')
        self.k_fc2 = layers.Dense(32, activation='relu')
        self.mu_out = layers.Dense(1, name="mu_K")
        self.log_var_out = layers.Dense(1, name="log_var_total")

    def call(self, z_fused, training=False):
        # Eq.7: porosity prediction
        phi_hat = self.phi_out(self.phi_fc1(z_fused))  # (B, 1)

        # Eq.8: physics-enhanced representation
        z_aug = tf.concat([z_fused, phi_hat], axis=-1)  # (B, D+1)

        # Eq.9: permeability + total uncertainty
        h_k = self.k_fc1(z_aug)
        h_k = self.k_fc2(h_k)
        mu_K = self.mu_out(h_k)              # (B, 1)
        # Numerical floor only; the returned log variance and SD agree exactly.
        raw_log_var = tf.clip_by_value(self.log_var_out(h_k), -20.0, 20.0)
        log_var_total = tf.math.log(tf.exp(raw_log_var) + 1e-6)
        sigma_total = tf.exp(0.5 * log_var_total)

        return mu_K, sigma_total, log_var_total, phi_hat


# ── 3e. Full P-LCN Model ─────────────────────────────────────

class PLCN(Model):
    """
    Full P-LCN model as described in Figure 4.

    Inputs:
        logs_input: (B, L, 5)       — windowed wireline curves
        img_input: (B, 64, 64, 6)   — PPL+XPL 6-channel images
        lith_input: (B, n_litho)    — lithology probability vector

    Outputs:
        mu_K, sigma_total, phi_hat, sigma_log, sigma_img, w_log, w_img
    """
    def __init__(self, emb_dim=LOG_EMB_DIM, **kwargs):
        super().__init__(**kwargs)
        self.log_expert = LogExpertWithGating(emb_dim=emb_dim)
        self.img_expert = ImageExpert(emb_dim=emb_dim)
        self.fusion = IVWFeatureFusion()
        self.cascade_head = CascadedPredictionHead()

    def call(self, inputs, training=False):
        logs_in, imgs_in, lith_in = inputs

        # Dual experts
        h_log, sigma_log = self.log_expert([logs_in, lith_in],
                                            training=training)
        h_img, sigma_img = self.img_expert(imgs_in, training=training)

        # Feature-level IVW fusion
        z_fused, w_log, w_img = self.fusion(
            [h_log, sigma_log, h_img, sigma_img])

        # Cascaded prediction
        mu_K, sigma_total, log_var_total, phi_hat = self.cascade_head(
            z_fused, training=training)

        return {
            "mu_K": mu_K,
            "sigma_total": sigma_total,
            "log_var_total": log_var_total,
            "phi_hat": phi_hat,
            "sigma_log": sigma_log,
            "sigma_img": sigma_img,
            "w_log": w_log,
            "w_img": w_img,
        }


def masked_heteroscedastic_nll(y_true, mu, log_var, mask):
    """Gaussian NLL, averaged over retained samples; zero for an empty mask."""
    y_true, mask = tf.reshape(y_true, tf.shape(mu)), tf.reshape(mask, tf.shape(mu))
    nll = 0.5 * (math.log(2.0 * math.pi) + log_var + tf.square(y_true - mu) * tf.exp(-log_var))
    return tf.math.divide_no_nan(tf.reduce_sum(mask * nll), tf.reduce_sum(mask))


def porosity_loss(phi_true, phi_pred):
    return tf.reduce_mean(tf.square(tf.reshape(phi_true, tf.shape(phi_pred)) - phi_pred))


class PLCNTrainer:
    def __init__(self, model, optimizer, lambda_phy=LAMBDA_PHY, lambda_reg=LAMBDA_REG):
        self.model, self.optimizer = model, optimizer
        self.lambda_phy, self.lambda_reg = lambda_phy, lambda_reg

    @tf.function(reduce_retracing=True)
    def train_step(self, logs, imgs, lith, y_k, y_phi, kc_mask):
        with tf.GradientTape() as tape:
            outputs = self.model([logs, imgs, lith], training=True)
            loss_perm = masked_heteroscedastic_nll(y_k, outputs["mu_K"], outputs["log_var_total"], kc_mask)
            loss_phi = porosity_loss(y_phi, outputs["phi_hat"])
            explicit_l2 = tf.add_n([tf.reduce_sum(tf.square(v)) for v in self.model.trainable_variables])
            total_loss = loss_perm + self.lambda_phy * loss_phi + self.lambda_reg * explicit_l2
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients((g, v) for g, v in zip(gradients, self.model.trainable_variables) if g is not None)
        return total_loss, loss_perm, loss_phi

    def evaluate(self, logs, imgs, lith, y_k, y_phi):
        # Batching avoids loading a whole well into accelerator memory at once.
        pieces = []
        for start in range(0, len(logs), 64):
            stop = start + 64
            out = self.model([logs[start:stop], imgs[start:stop], lith[start:stop]], training=False)
            pieces.append({key: value.numpy().reshape(-1) for key, value in out.items()})
        outputs = {key: np.concatenate([piece[key] for piece in pieces]) for key in pieces[0]}
        mu, sigma = outputs["mu_K"], outputs["sigma_total"]
        metrics = evaluate_predictions(np.asarray(y_k).reshape(-1), mu, sigma)
        error = np.abs(np.asarray(y_k).reshape(-1) - mu)
        corr = float(np.corrcoef(error, sigma)[0, 1]) if np.std(error) > 0 and np.std(sigma) > 0 else float("nan")
        return {"R2": metrics["r2"], "RMSE": metrics["rmse"], "MAE": metrics["mae"],
                "NLL": metrics["nll"], "Coverage_95PI": metrics["picp_95"], "Corr": corr,
                "mu": mu, "sigma": sigma, "phi_pred": outputs["phi_hat"],
                **{key: outputs[key] for key in ("w_log", "w_img", "sigma_log", "sigma_img")}}


class WarmupCosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, total_steps, warmup_steps=65, start_lr=1e-6, peak_lr=2e-3, end_lr=1e-6):
        super().__init__()
        self.total_steps, self.warmup_steps = total_steps, warmup_steps
        self.start_lr, self.peak_lr, self.end_lr = start_lr, peak_lr, end_lr

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = self.start_lr + (self.peak_lr - self.start_lr) * step / max(self.warmup_steps, 1)
        progress = tf.clip_by_value((step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1), 0., 1.)
        cosine = self.end_lr + 0.5 * (self.peak_lr - self.end_lr) * (1.0 + tf.cos(math.pi * progress))
        return tf.where(step < self.warmup_steps, warmup, cosine)

    def get_config(self):
        return dict(total_steps=self.total_steps, warmup_steps=self.warmup_steps,
                    start_lr=self.start_lr, peak_lr=self.peak_lr, end_lr=self.end_lr)


def run_training(data, seed=RANDOM_SEED, config=None):
    config = config or TrainingConfig()
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    tr, va = data["train"], data["val"]
    # Refitted using only training wells for each random-seed run.
    mask, a, b, residual_sd = kc_ransac_filter(tr["phi"], tr["logK"], tau=config.kc_tau, random_state=seed)
    model = PLCN(emb_dim=LOG_EMB_DIM, name="P_LCN")
    steps = math.ceil(len(tr["logK"]) / config.batch_size)
    schedule = WarmupCosineSchedule(config.epochs * steps, config.warmup_steps,
                                    config.start_lr, config.peak_lr, config.end_lr)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=schedule, weight_decay=config.weight_decay)
    trainer = PLCNTrainer(model, optimizer, config.lambda_phy, config.lambda_reg)
    arrays = [tr["logs"], tr["images"], tr["lith"], tr["logK"][:, None], tr["phi"][:, None], mask[:, None]]
    dataset = tf.data.Dataset.from_tensor_slices(tuple(arrays)).shuffle(len(mask), seed=seed).batch(config.batch_size)
    best_val_loss, best_weights, patience_counter, best_epoch = float("inf"), None, 0, 0
    history = []
    for epoch in range(config.epochs):
        losses = [float(trainer.train_step(*batch)[0]) for batch in dataset]
        val = trainer.evaluate(va["logs"], va["images"], va["lith"], va["logK"], va["phi"])
        value = val["NLL"]  # all validation samples; no KC mask or auxiliary/L2 term
        if not np.isfinite(value) or not np.isfinite(losses).all():
            raise FloatingPointError("Non-finite training/validation loss; no results will be certified")
        history.append(dict(epoch=epoch + 1, train_loss=float(np.mean(losses)), validation_nll=value))
        if value < best_val_loss:
            best_val_loss, best_weights, best_epoch = value, model.get_weights(), epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
        if epoch == 0 or (epoch + 1) % 20 == 0 or epoch + 1 == config.epochs:
            print(f"seed={seed}, epoch={epoch + 1}, validation NLL={value:.5f}", flush=True)
        if patience_counter >= config.patience:
            break
    model.set_weights(best_weights)
    trainer.run_metadata = dict(seed=seed, best_epoch=best_epoch, best_validation_nll=best_val_loss,
                                epochs_completed=len(history), kc_a=float(a), kc_b=float(b),
                                kc_residual_sd=float(residual_sd), masked_fraction=float(np.mean(1 - mask)),
                                history=history)
    trainer.kc_mask = mask
    return model, trainer


def prediction_frame(subset, result, seed, split, synthetic):
    frame = subset["frame"][["Well", "Depth"]].copy()
    for name in ("Lithology", "Caliper", "BitSize"):
        if name in subset["frame"]:
            frame[name] = subset["frame"][name].to_numpy()
    frame["seed"], frame["split"], frame["synthetic"] = seed, split, synthetic
    frame["truth_logK"] = subset["logK"]
    frame["mu_logK"], frame["sigma_total"] = result["mu"], result["sigma"]
    for key in ("w_log", "w_img", "sigma_log", "sigma_img", "phi_pred"):
        frame[key] = result[key]
    frame["lower_logK"] = result["mu"] - 1.96 * result["sigma"]
    frame["upper_logK"] = result["mu"] + 1.96 * result["sigma"]
    for key in ("lower", "upper"):
        frame[f"{key}_mD"] = np.power(10., frame[f"{key}_logK"])
    return frame


def save_json(path, value):
    # Strict JSON: undefined correlations become null, never invalid NaN tokens.
    def clean(v):
        if isinstance(v, dict): return {str(k): clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)): return [clean(x) for x in v]
        if isinstance(v, np.ndarray): return clean(v.tolist())
        if isinstance(v, np.generic): return clean(v.item())
        if isinstance(v, float) and not math.isfinite(v): return None
        return v
    Path(path).write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_FILE)
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--logs-file", type=Path, help="Optional dense well-log CSV/XLSX")
    parser.add_argument("--porosity-unit", choices=["fraction", "percent"], default="fraction")
    parser.add_argument("--log-window", type=int, default=5)
    parser.add_argument("--window-span-m", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(2025, 2030)))
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--block-length", type=int, default=5)
    for name, kind in [("epochs", int), ("batch_size", int), ("patience", int), ("warmup_steps", int),
                       ("start_lr", float), ("peak_lr", float), ("end_lr", float), ("weight_decay", float),
                       ("lambda_phy", float), ("lambda_reg", float), ("kc_tau", float)]:
        parser.add_argument("--" + name.replace("_", "-"), type=kind, default=getattr(TrainingConfig(), name))
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patience, args.block_length, args.bootstrap_resamples) < 1:
        parser.error("Epochs, batch size, patience and bootstrap settings must be positive")
    if args.bootstrap_resamples < 2:
        parser.error("Bootstrap needs at least two resamples")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Random seeds must be unique")
    if min(args.lambda_phy, args.lambda_reg, args.weight_decay, args.kc_tau, args.warmup_steps) < 0:
        parser.error("Loss weights, threshold, weight decay and warmup must be nonnegative")
    if min(args.start_lr, args.peak_lr, args.end_lr) <= 0:
        parser.error("Learning rates must be positive")
    return args


def main():
    args = parse_args()
    config = TrainingConfig(**{key: getattr(args, key) for key in TrainingConfig.__dataclass_fields__})
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a new --output-dir to preserve previous results: {output}")
    output.mkdir(parents=True, exist_ok=True)
    data = load_and_split_by_well(args.data, args.image_dir, args.logs_file,
                                 args.porosity_unit, args.log_window, args.window_span_m)
    synthetic = data["meta"]["synthetic"]
    print("SYNTHETIC DEMONSTRATION: these results do not reproduce the study." if synthetic else
          "User-provided data: retain authorization and provenance records.", flush=True)
    metrics, frames, runs = [], [], []
    for seed in args.seeds:
        model, trainer = run_training(data, seed, config)
        model.save_weights(output / f"seed_{seed}.weights.h5")
        runs.append(trainer.run_metadata)
        mask_frame = data["train"]["frame"][["Well", "Depth"]].copy()
        mask_frame["retained_in_permeability_loss"] = trainer.kc_mask.astype(int)
        mask_frame.to_csv(output / f"training_mask_{seed}.csv", index=False)
        for split in ("val", "blind"):
            subset = data[split]
            result = trainer.evaluate(subset["logs"], subset["images"], subset["lith"], subset["logK"], subset["phi"])
            metrics.append(dict(seed=seed, split=split, synthetic=synthetic,
                                **{key: result[key] for key in ("R2", "RMSE", "MAE", "NLL", "Coverage_95PI", "Corr")}))
            frames.append(prediction_frame(subset, result, seed, split, synthetic))
    pd.DataFrame(metrics).to_csv(output / "per_seed_metrics.csv", index=False)
    metric_names = {"R2": "r2", "RMSE": "rmse", "MAE": "mae", "NLL": "nll", "Coverage_95PI": "picp_95"}
    summary = {split: summarize_runs(pd.DataFrame([m for m in metrics if m["split"] == split]).rename(columns=metric_names))
               for split in ("val", "blind")}
    save_json(output / "metrics_summary.json", summary)
    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_csv(output / "per_seed_predictions.csv", index=False)
    # Tie-break by run order, never by blind-well performance.
    selected_seed = min(runs, key=lambda r: r["best_validation_nll"])["seed"]
    selected = predictions[predictions.seed == selected_seed]
    selected.to_csv(output / "selected_run_predictions.csv", index=False)
    v = selected[selected.split == "val"]
    diagnostics = block_bootstrap_uncertainty(v.Depth.to_numpy(), v.truth_logK.to_numpy(),
                        v.mu_logK.to_numpy(), v.sigma_total.to_numpy(),
                        block_length=args.block_length, n_resamples=args.bootstrap_resamples, seed=2025)
    save_json(output / "uncertainty_validation.json", dict(seed=selected_seed, synthetic=synthetic, **diagnostics))
    metadata = dict(config=asdict(config), preprocessing=data["meta"], runs=runs,
                    selected_seed=selected_seed, selection="minimum unmasked CS608 validation NLL",
                    aggregation="metrics computed per run; sample SD (ddof=1); no pooled-seed metrics",
                    lambda_reg_provenance="configurable public-demo default; original study value not established",
                    branch_scores="sigma_log/img are positive fusion scores, not calibrated modality SDs",
                    final_uncertainty="conditional final-head aleatoric SD in log10(mD); excludes parameter uncertainty",
                    bootstrap=dict(block_length=args.block_length, resamples=args.bootstrap_resamples),
                    tensorflow_version=tf.__version__, numpy_version=np.__version__,
                    data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
                    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save_json(output / "run_metadata.json", metadata)
    print(f"Saved complete run outputs to {output.resolve()}; representative seed={selected_seed}")


if __name__ == "__main__":
    main()
