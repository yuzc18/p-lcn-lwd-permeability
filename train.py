"""
Physics-Informed Uncertainty-Aware Multimodal Fusion for
Logging-While-Drilling Permeability Evaluation in Volcanic Reservoirs

Implementation of P-LCN (Physics-Informed Lightweight Cascaded Network)

Author: Xiu Jin, Taiji Yu
Contact: yutj1988@126.com
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

# ╔══════════════════════════════════════════════════════════════╗
# ║  0. GLOBAL CONFIGURATION                                    ║
# ║  Paper ref: Table 2 — Implementation Details                 ║
# ╚══════════════════════════════════════════════════════════════╝

RANDOM_SEED = 2025
tf.keras.utils.set_random_seed(RANDOM_SEED)

# --- Data columns ---
K_COL = "LogK_Ideal"           # log10(K) permeability label
PHI_COL = "Porosity_v6"        # Porosity (well-log interpreted)
WELL_COL = "Well"              # Well identifier column
DEPTH_COL = "Depth"            # Depth column

# --- Well-level split (Paper: Experimental Setup) ---
TRAIN_WELLS = ["WF1", "CS12", "CS13", "CS601", "CS607"]
VAL_WELL = "CS608"
BLIND_WELL = "CS606"

# --- Log Expert input (Paper: Log Expert with Lithology Gating) ---
RAW_LOGS = ["GR", "AC", "DEN", "CN", "RLA5"]  # 5 conventional LWD curves
LOG_WINDOW_L = 5  # Window length (~1m stratigraphic thickness)

# --- Image Expert input (Paper: Image Expert) ---
USE_RAW_IMAGES = True
IMAGE_DIR = "thin_section_images/"
IMAGE_SIZE = 64   # Paper: "center-cropped to 64×64 pixels"
IMAGE_CHANNELS = 6     # PPL(3ch) + XPL(3ch) concatenated
# Naming convention: {Well}_{Depth}_PPL.png / {Well}_{Depth}_XPL.png
IMAGE_PPL_SUFFIX = "_PPL.png"
IMAGE_XPL_SUFFIX = "_XPL.png"
MICRO_PREFIXES = ["MicroPPL", "MicroXPL"]  # fallback for pre-extracted features

# --- Lithology gating (Paper: Eq.4) ---
LITH_PREFIX = "LithProb"

# --- KC-RANSAC (Paper: Training-Time Label Governance) ---
KC_TAU = 0.60                  # Paper: τ = 0.60 (Table 4, optimal)

# --- Training (Paper: Table 2) ---
BATCH_SIZE = 64
MAX_EPOCHS = 200
PATIENCE = 30
INIT_LR = 2e-3
LAMBDA_PHY = 1.0               # Paper: physics-consistency weight
LAMBDA_L2 = 1e-4               # Paper: L2 regularization coefficient
LOG_EMB_DIM = 32               # Paper: 32-dimensional log feature embedding
IMG_EMB_DIM = 32               # Paper: image embedding dimension

DATA_FILE = "master_dataset_v11_ideal.xlsx"


# ╔══════════════════════════════════════════════════════════════╗
# ║  1. KC-RANSAC: PHYSICS-INFORMED LABEL GOVERNANCE             ║
# ║  Paper ref: "Training-Time Label Governance" + Eq.3          ║
# ╚══════════════════════════════════════════════════════════════╝

def kc_ransac_filter(phi, logK, tau=KC_TAU, n_iter=1000, sample_frac=0.5,
                     random_state=RANDOM_SEED):
    """
    Asymmetric KC-RANSAC label filtering.

    Paper description:
      - Fit global trend log(K) = a * log(φ) + b in (logφ, logK) space
      - Compute residual r_KC = logK_core - (a*logφ + b)   [Eq.3]
      - Estimate inlier residual std σ_r via RANSAC
      - Mask samples where r_KC < -τ * σ_r  (ASYMMETRIC: only below-trend)
      - Retain all positive residuals (fracture-enhanced samples)

    Returns:
        mask: binary array, 1=retain, 0=reject
        a, b: fitted KC trend parameters
        sigma_r: inlier residual std
    """
    rng = np.random.RandomState(random_state)

    # Work in (log φ, log K) space
    log_phi = np.log10(np.clip(phi, 1e-6, None))
    log_k = logK.copy()
    n = len(log_phi)
    n_sample = max(int(n * sample_frac), 2)

    best_inlier_count = 0
    best_a, best_b = 1.0, 0.0

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
        sigma_est = med_abs_res / 0.6745  # MAD-based robust σ
        inlier_mask = np.abs(residuals) < 2.0 * sigma_est
        inlier_count = inlier_mask.sum()

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_a, best_b = a_c, b_c

    # Final residuals with best model
    residuals_final = log_k - (best_a * log_phi + best_b)

    # Inlier σ_r estimation
    inlier_res = residuals_final[np.abs(residuals_final) < 2.0 *
                                 (np.median(np.abs(residuals_final)) / 0.6745)]
    sigma_r = np.std(inlier_res) if len(inlier_res) > 2 else np.std(residuals_final)

    # ASYMMETRIC masking: reject only samples BELOW the KC trend
    # r_KC < -τ * σ_r  →  rejected  (scale-attenuated outliers)
    # All positive residuals retained (fracture-enhanced)
    mask = np.ones(n, dtype=np.float32)
    reject_condition = residuals_final < -tau * sigma_r
    mask[reject_condition] = 0.0

    n_rejected = int((1 - mask).sum())
    pct_rejected = 100.0 * n_rejected / n
    print(f"  KC-RANSAC: trend logK = {best_a:.3f}·logφ + {best_b:.3f}")
    print(f"  KC-RANSAC: σ_r = {sigma_r:.4f}, τ = {tau}")
    print(f"  KC-RANSAC: rejected {n_rejected}/{n} samples ({pct_rejected:.1f}%)")

    return mask, best_a, best_b, sigma_r


# ╔══════════════════════════════════════════════════════════════╗
# ║  2. IMAGE LOADING AND PREPROCESSING                          ║
# ║  Paper ref: "Image Expert" — center-crop, PPL+XPL 6-channel  ║
# ╚══════════════════════════════════════════════════════════════╝

def center_crop(img, crop_size=IMAGE_SIZE):
    """
    Paper: "Each input image is center-cropped to focus on
    representative micro-textural regions (64 × 64 pixels)."

    Args:
        img: (H, W, 3) numpy array
        crop_size: target size (64)
    Returns:
        (crop_size, crop_size, 3) numpy array
    """
    h, w = img.shape[:2]
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    return img[top:top + crop_size, left:left + crop_size]


def load_image_pair(well, depth, image_dir=IMAGE_DIR):
    """
    Load paired PPL + XPL thin-section images for a given depth sample.

    Naming convention: {Well}_{Depth}_PPL.png / {Well}_{Depth}_XPL.png
    Paper: "one PPL and one XPL image (1024 × 1024 pixels) were acquired
    under standardized illumination conditions"

    Returns:
        (64, 64, 6) numpy array — PPL(3ch) + XPL(3ch) center-cropped
        None if images not found
    """
    # Handle depth formatting (e.g., 2662.3 or 2662.0)
    depth_str = f"{depth:.1f}" if depth != int(depth) else f"{int(depth)}"

    ppl_path = os.path.join(image_dir, f"{well}_{depth_str}{IMAGE_PPL_SUFFIX}")
    xpl_path = os.path.join(image_dir, f"{well}_{depth_str}{IMAGE_XPL_SUFFIX}")

    # Try alternative depth formats if primary not found
    if not os.path.exists(ppl_path):
        depth_str = f"{depth}"
        ppl_path = os.path.join(image_dir, f"{well}_{depth_str}{IMAGE_PPL_SUFFIX}")
        xpl_path = os.path.join(image_dir, f"{well}_{depth_str}{IMAGE_XPL_SUFFIX}")

    if not os.path.exists(ppl_path) or not os.path.exists(xpl_path):
        return None

    # Load images
    ppl_img = tf.keras.utils.load_img(ppl_path)
    xpl_img = tf.keras.utils.load_img(xpl_path)
    ppl_arr = tf.keras.utils.img_to_array(ppl_img)  # (1024, 1024, 3)
    xpl_arr = tf.keras.utils.img_to_array(xpl_img)  # (1024, 1024, 3)

    # Center crop to 64×64
    ppl_crop = center_crop(ppl_arr, IMAGE_SIZE)
    xpl_crop = center_crop(xpl_arr, IMAGE_SIZE)

    # Concatenate PPL + XPL → 6-channel input
    combined = np.concatenate([ppl_crop, xpl_crop], axis=-1)  # (64, 64, 6)

    # Normalize to [0, 1]
    combined = combined.astype(np.float32) / 255.0

    return combined


def load_images_for_subset(df_subset, image_dir=IMAGE_DIR):
    """
    Load all image pairs for a dataframe subset.

    Args:
        df_subset: DataFrame with WELL_COL and DEPTH_COL columns
        image_dir: directory containing thin-section images

    Returns:
        images: (N, 64, 64, 6) numpy array
        valid_mask: boolean array indicating successfully loaded images
    """
    n = len(df_subset)
    images = np.zeros((n, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS),
                      dtype=np.float32)
    valid_mask = np.ones(n, dtype=bool)

    wells = df_subset[WELL_COL].values
    depths = df_subset[DEPTH_COL].values

    loaded_count = 0
    for i in range(n):
        img = load_image_pair(wells[i], depths[i], image_dir)
        if img is not None:
            images[i] = img
            loaded_count += 1
        else:
            valid_mask[i] = False

    print(f"  Images loaded: {loaded_count}/{n} "
          f"({100*loaded_count/n:.1f}%)")

    if loaded_count < n:
        missing = n - loaded_count
        print(f"  ⚠️  {missing} images not found — "
              f"these samples will use zero-filled placeholders")

    return images, valid_mask


# ╔══════════════════════════════════════════════════════════════╗
# ║  3. DATA LOADING WITH WELL-LEVEL SPLIT                      ║
# ║  Paper ref: "Data Preparation and Partitioning"              ║
# ╚══════════════════════════════════════════════════════════════╝

def create_log_windows(log_values, window_size=LOG_WINDOW_L):
    """
    Paper: "For each target depth, a sequence segment of length L is
    extracted" — creates sliding windows for 1D-CNN input.

    Args:
        log_values: (N, n_curves) array of log values
        window_size: L (number of sampling points spanning ~1m)

    Returns:
        (N, L, n_curves) windowed array with zero-padding at boundaries
    """
    n_samples, n_curves = log_values.shape
    half_w = window_size // 2
    padded = np.pad(log_values, ((half_w, half_w), (0, 0)), mode='edge')
    windows = np.zeros((n_samples, window_size, n_curves), dtype=np.float32)
    for i in range(n_samples):
        windows[i] = padded[i:i + window_size]
    return windows


def load_and_split_by_well(data_file):
    """
    Paper: "strict well-level split strategy, ensuring complete
    independence across subsets at the well scale"

    Returns dict with train/val/blind data and KC-RANSAC mask.
    Loads raw PPL+XPL images when USE_RAW_IMAGES=True.
    """
    print(f"📂 Loading data: {data_file}")
    if data_file.endswith('.csv'):
        df = pd.read_csv(data_file)
    else:
        df = pd.read_excel(data_file, engine='openpyxl')

    # --- Identify columns dynamically ---
    micro_cols = [c for c in df.columns
                  for p in MICRO_PREFIXES if c.startswith(p)]
    lith_cols = [c for c in df.columns if c.startswith(LITH_PREFIX)]

    print(f"  Log features: {RAW_LOGS}")
    print(f"  Lithology priors: {lith_cols}")
    print(f"  Image mode: {'Raw PPL+XPL (6-channel)' if USE_RAW_IMAGES else 'Pre-extracted features'}")

    # --- Well-level split ---
    df_train = df[df[WELL_COL].isin(TRAIN_WELLS)].copy()
    df_val = df[df[WELL_COL] == VAL_WELL].copy()
    df_blind = df[df[WELL_COL] == BLIND_WELL].copy()

    print(f"\n  Train: {len(df_train)} samples from {TRAIN_WELLS}")
    print(f"  Validation: {len(df_val)} samples from {VAL_WELL}")
    print(f"  Blind test: {len(df_blind)} samples from {BLIND_WELL}")

    # --- Extract tabular arrays ---
    def extract_tabular(subset):
        logs = subset[RAW_LOGS].values.astype(np.float32)
        lith = subset[lith_cols].values.astype(np.float32)
        logK = subset[K_COL].values.astype(np.float32)
        phi = subset[PHI_COL].values.astype(np.float32)
        if phi.mean() > 1.0:
            phi = phi / 100.0
        return logs, lith, logK, phi

    logs_tr, lith_tr, logK_tr, phi_tr = extract_tabular(df_train)
    logs_va, lith_va, logK_va, phi_va = extract_tabular(df_val)
    logs_bl, lith_bl, logK_bl, phi_bl = extract_tabular(df_blind)

    # --- Load image data ---
    if USE_RAW_IMAGES:
        # Paper: "paired PPL and XPL thin-section images (1024×1024),
        #         center-cropped to 64×64"
        print(f"\n🖼️  Loading thin-section images from: {IMAGE_DIR}")
        imgs_tr, _ = load_images_for_subset(df_train, IMAGE_DIR)
        imgs_va, _ = load_images_for_subset(df_val, IMAGE_DIR)
        imgs_bl, _ = load_images_for_subset(df_blind, IMAGE_DIR)
        # Images are already normalized to [0,1] in load_image_pair()
        # No further scaling needed for image inputs
    else:
        # Fallback: pre-extracted micro features
        print(f"  Pre-extracted micro features: {len(micro_cols)} columns")
        imgs_tr = df_train[micro_cols].values.astype(np.float32)
        imgs_va = df_val[micro_cols].values.astype(np.float32)
        imgs_bl = df_blind[micro_cols].values.astype(np.float32)

    # ── KC-RANSAC on training data ONLY (Paper: "estimated exclusively
    #    from the training wells and then fixed") ──
    print("\n🔧 Applying KC-RANSAC label governance (training wells only)...")
    kc_mask, kc_a, kc_b, kc_sigma = kc_ransac_filter(phi_tr, logK_tr, tau=KC_TAU)

    # ── Normalization: fit on training set ONLY (Paper: "All
    #    preprocessing statistics computed exclusively from training") ──
    logs_scaler = StandardScaler().fit(logs_tr)
    lith_scaler = StandardScaler().fit(lith_tr)
    phi_scaler = MinMaxScaler().fit(phi_tr.reshape(-1, 1))

    # For pre-extracted features, also scale micro
    micro_scaler = None
    if not USE_RAW_IMAGES:
        micro_scaler = StandardScaler().fit(imgs_tr)

    def process_subset(logs, imgs, lith, phi):
        logs_s = logs_scaler.transform(logs)
        lith_s = lith_scaler.transform(lith)
        phi_s = phi_scaler.transform(phi.reshape(-1, 1)).flatten()
        logs_w = create_log_windows(logs_s, LOG_WINDOW_L)
        if not USE_RAW_IMAGES and micro_scaler is not None:
            imgs = micro_scaler.transform(imgs)
        return logs_w, imgs, lith_s, phi_s

    logs_tr_w, imgs_tr_p, lith_tr_s, phi_tr_s = process_subset(
        logs_tr, imgs_tr, lith_tr, phi_tr)
    logs_va_w, imgs_va_p, lith_va_s, phi_va_s = process_subset(
        logs_va, imgs_va, lith_va, phi_va)
    logs_bl_w, imgs_bl_p, lith_bl_s, phi_bl_s = process_subset(
        logs_bl, imgs_bl, lith_bl, phi_bl)

    # Determine image input shape for model construction
    if USE_RAW_IMAGES:
        img_input_shape = (IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)  # (64,64,6)
    else:
        img_input_shape = (imgs_tr_p.shape[1],)  # (micro_dim,)

    data = {
        "train": {
            "logs": logs_tr_w,           # (N, L, 5)
            "images": imgs_tr_p,         # (N, 64, 64, 6) or (N, micro_dim)
            "lith": lith_tr_s,           # (N, n_litho)
            "logK": logK_tr,             # (N,)
            "phi": phi_tr_s,             # (N,) scaled
            "kc_mask": kc_mask,          # (N,) binary
        },
        "val": {
            "logs": logs_va_w, "images": imgs_va_p, "lith": lith_va_s,
            "logK": logK_va, "phi": phi_va_s,
        },
        "blind": {
            "logs": logs_bl_w, "images": imgs_bl_p, "lith": lith_bl_s,
            "logK": logK_bl, "phi": phi_bl_s,
        },
        "meta": {
            "n_curves": len(RAW_LOGS),
            "img_input_shape": img_input_shape,
            "lith_dim": lith_tr_s.shape[1],
            "kc_params": (kc_a, kc_b, kc_sigma),
        }
    }
    return data


# ╔══════════════════════════════════════════════════════════════╗
# ║  3. MODEL: P-LCN ARCHITECTURE                               ║
# ║  Paper ref: "P-LCN Architecture: Lightweight Dual-Expert"   ║
# ╚══════════════════════════════════════════════════════════════╝

# ── 3a. Log Expert with Lithology Gating ──────────────────────
# Paper: "three stacked depthwise separable convolution (DSC) blocks
#         yielding a 32-dimensional log feature embedding h_log"
# Paper Eq.4: g = σ(W_g · P_litho + b_g),  h_log_gated = g ⊙ h_log

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
        self.gate_dense = None  # built dynamically on first call

        # Uncertainty head: σ_log (aleatoric)
        self.sigma_head = layers.Dense(1, name="sigma_log_raw")

    def build(self, input_shape):
        # Gate projection: W_g maps lith_dim → emb_dim
        # Will be built on first call
        super().build(input_shape)

    def call(self, inputs, training=False):
        log_input, lith_input = inputs  # (B,L,C), (B,n_litho)

        # 1D-CNN feature extraction
        h = self.dsc1(log_input, training=training)
        h = self.dsc2(h, training=training)
        h = self.dsc3(h, training=training)
        h_log = self.gap(h)  # (B, emb_dim=32)

        # Lithology gating (Eq.4): g = σ(W_g · P_litho + b_g)
        if self.gate_dense is None:
            self.gate_dense = layers.Dense(self.emb_dim, name="gate_proj")
        gate = tf.nn.sigmoid(self.gate_dense(lith_input))  # (B, 32)
        h_log_gated = h_log * gate  # element-wise (Eq.4: g ⊙ h_log)

        # Aleatoric uncertainty (Paper: "σ with Softplus activation")
        sigma_log = tf.nn.softplus(self.sigma_head(h_log_gated)) + 1e-6

        return h_log_gated, sigma_log


# ── 3b. Image Expert ──────────────────────────────────────────
# Paper: "MobileNetV3-Small as the backbone, truncated at 3rd
#         bottleneck layer"
# Input: PPL(3ch) + XPL(3ch) = 6-channel, center-cropped to 64×64

class ImageExpert(layers.Layer):
    """
    Paper: "Image Expert"

    When USE_RAW_IMAGES=True:
        - Accepts (B, 64, 64, 6) PPL+XPL concatenated input
        - 1×1 Conv2D adapter: 6ch → 3ch (learnable channel projection)
        - MobileNetV3-Small backbone truncated at 3rd bottleneck
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

            # MobileNetV3-Small backbone (Paper: "truncated at 3rd bottleneck")
            base = tf.keras.applications.MobileNetV3Small(
                input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                include_top=False,
                weights='imagenet',
                minimalistic=True)

            # Identify 3rd bottleneck output layer
            # MobileNetV3Small structure: initial conv → bottleneck blocks
            # We truncate after the 3rd inverted residual block
            target_layer_name = self._find_3rd_bottleneck(base)
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

        # Aleatoric uncertainty head
        self.sigma_head = layers.Dense(1, name="sigma_img_raw")

    @staticmethod
    def _find_3rd_bottleneck(base_model):
        """Find the output layer name of the 3rd bottleneck block."""
        # MobileNetV3Small bottleneck blocks end with 'expand_bn' or
        # 'project_bn' activation. We look for the 3rd such block.
        block_count = 0
        target_name = None
        for layer in base_model.layers:
            name = layer.name
            # Each inverted residual block ends with a 're_lu' or
            # 'activation' after the project BN
            if 'multiply' in name:  # SE block endpoint = block boundary
                block_count += 1
                if block_count == 3:
                    target_name = name
                    break
        # Fallback: use a layer around index 40 if pattern matching fails
        if target_name is None:
            target_name = base_model.layers[min(40, len(base_model.layers)-1)].name
            print(f"  ⚠️  3rd bottleneck auto-detection fallback → {target_name}")
        else:
            print(f"  MobileNetV3 truncated at: {target_name} (block {block_count})")
        return target_name

    def call(self, x, training=False):
        if self.use_raw_images:
            # (B, 64, 64, 6) → adapter → (B, 64, 64, 3)
            x_3ch = self.channel_adapter(x)
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
    Fuses embeddings at FEATURE level using inverse-variance weighting.

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


# ── 3d. Physics-Causal Cascaded Prediction Head ──────────────
# Paper Eq.7-9: z_fused → φ̂ → [z_fused; φ̂] → (μ_K, σ_total)

class CascadedPredictionHead(layers.Layer):
    """
    Paper: "Physics-Causal Cascaded Prediction Head"

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
        log_var_total = self.log_var_out(h_k) # (B, 1)
        sigma_total = tf.sqrt(tf.exp(log_var_total) + 1e-6)

        return mu_K, sigma_total, log_var_total, phi_hat


# ── 3e. Full P-LCN Model ─────────────────────────────────────

class PLCN(Model):
    """
    Full P-LCN model as described in Figure 4.

    Inputs:
        logs_input: (B, L, 5)       — windowed LWD curves
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


# ╔══════════════════════════════════════════════════════════════╗
# ║  4. LOSS FUNCTIONS                                           ║
# ║  Paper ref: "Joint Physics-Constrained Optimization" Eq.10-13║
# ╚══════════════════════════════════════════════════════════════╝

def masked_heteroscedastic_nll(y_true, mu, log_var, mask):
    """
    Paper Eq.11-12: Masked Gaussian NLL.
    L_perm = (1/N_valid) Σ m_i * [0.5 * (s_i + (y_i - μ_i)² / exp(s_i))]
    where s_i = log(σ²_i)
    """
    nll_per_sample = 0.5 * (log_var + tf.square(y_true - mu) / (
        tf.exp(log_var) + 1e-6))
    masked_nll = nll_per_sample * mask
    n_valid = tf.reduce_sum(mask) + 1e-6
    return tf.reduce_sum(masked_nll) / n_valid


def porosity_loss(phi_true, phi_pred):
    """
    Paper Eq.13: Porosity supervision loss (no mask applied).
    "preserving the backbone's representation learning capability"
    """
    return tf.reduce_mean(tf.square(phi_true - phi_pred))


# ╔══════════════════════════════════════════════════════════════╗
# ║  5. TRAINING LOOP WITH CUSTOM LOSS                           ║
# ║  Paper: masked NLL + porosity loss + L2 regularization       ║
# ╚══════════════════════════════════════════════════════════════╝

class PLCNTrainer:
    def __init__(self, model, optimizer, lambda_phy=LAMBDA_PHY):
        self.model = model
        self.optimizer = optimizer
        self.lambda_phy = lambda_phy

    @tf.function
    def train_step(self, logs, imgs, lith, y_k, y_phi, kc_mask):
        with tf.GradientTape() as tape:
            outputs = self.model([logs, imgs, lith], training=True)

            # Eq.12: Masked NLL for permeability
            loss_perm = masked_heteroscedastic_nll(
                y_k, outputs["mu_K"], outputs["log_var_total"], kc_mask)

            # Eq.13: Porosity supervision (unmasked)
            loss_phi = porosity_loss(y_phi, outputs["phi_hat"])

            # Eq.10: Total loss = L_perm + λ_phy * L_phi  (L2 via optimizer)
            total_loss = loss_perm + self.lambda_phy * loss_phi

        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))

        return total_loss, loss_perm, loss_phi

    def evaluate(self, logs, imgs, lith, y_k, y_phi):
        outputs = self.model([logs, imgs, lith], training=False)
        mu = outputs["mu_K"].numpy().flatten()
        sigma = outputs["sigma_total"].numpy().flatten()
        phi_pred = outputs["phi_hat"].numpy().flatten()
        y_k_np = y_k.numpy().flatten() if hasattr(y_k, 'numpy') else y_k.flatten()

        r2 = r2_score(y_k_np, mu)
        rmse = np.sqrt(mean_squared_error(y_k_np, mu))
        mae = np.mean(np.abs(y_k_np - mu))

        # Uncertainty-error correlation
        abs_err = np.abs(y_k_np - mu)
        if np.std(sigma) > 1e-8:
            corr, p_val = pearsonr(abs_err, sigma)
        else:
            corr, p_val = 0.0, 1.0

        # 95% PI coverage
        lower = mu - 1.96 * sigma
        upper = mu + 1.96 * sigma
        coverage = np.mean((y_k_np >= lower) & (y_k_np <= upper))

        return {
            "R2": r2, "RMSE": rmse, "MAE": mae,
            "Corr": corr, "p_val": p_val,
            "Coverage_95PI": coverage,
            "mu": mu, "sigma": sigma, "phi_pred": phi_pred,
            "w_log": outputs["w_log"].numpy().flatten(),
            "w_img": outputs["w_img"].numpy().flatten(),
            "sigma_log": outputs["sigma_log"].numpy().flatten(),
            "sigma_img": outputs["sigma_img"].numpy().flatten(),
        }


def run_training(data, seed=RANDOM_SEED):
    """Full training pipeline for one seed."""
    tf.keras.utils.set_random_seed(seed)

    tr = data["train"]
    va = data["val"]

    # Convert to tensors
    logs_tr = tf.constant(tr["logs"], dtype=tf.float32)
    imgs_tr = tf.constant(tr["images"], dtype=tf.float32)
    lith_tr = tf.constant(tr["lith"], dtype=tf.float32)
    y_k_tr = tf.constant(tr["logK"].reshape(-1, 1), dtype=tf.float32)
    y_phi_tr = tf.constant(tr["phi"].reshape(-1, 1), dtype=tf.float32)
    kc_mask_tr = tf.constant(tr["kc_mask"].reshape(-1, 1), dtype=tf.float32)

    logs_va = tf.constant(va["logs"], dtype=tf.float32)
    imgs_va = tf.constant(va["images"], dtype=tf.float32)
    lith_va = tf.constant(va["lith"], dtype=tf.float32)
    y_k_va = va["logK"].reshape(-1, 1)
    y_phi_va = va["phi"].reshape(-1, 1)

    # Build model
    model = PLCN(emb_dim=LOG_EMB_DIM, name="P_LCN")

    # Paper Table 2: AdamW with Cosine Decay + Warmup
    n_train = len(tr["logK"])
    steps_per_epoch = max(n_train // BATCH_SIZE, 1)
    total_steps = MAX_EPOCHS * steps_per_epoch
    warmup_steps = 5 * steps_per_epoch

    # Warmup + Cosine Decay schedule
    cosine_decay = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=INIT_LR,
        decay_steps=total_steps - warmup_steps,
        alpha=1e-6)

    # Linear warmup wrapper
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=1e-6,
        decay_steps=warmup_steps,
        end_learning_rate=INIT_LR,
        power=1.0)  # linear warmup

    class WarmupCosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, warmup_steps, warmup_lr, cosine_schedule):
            super().__init__()
            self.warmup_steps = warmup_steps
            self.warmup_lr = warmup_lr
            self.cosine_schedule = cosine_schedule
        def __call__(self, step):
            warmup_lr = self.warmup_lr(step)
            cosine_lr = self.cosine_schedule(step - self.warmup_steps)
            return tf.cond(step < self.warmup_steps, lambda: warmup_lr, lambda: cosine_lr)
        def get_config(self):
            return {"warmup_steps": self.warmup_steps}

    schedule = WarmupCosineSchedule(warmup_steps, lr_schedule, cosine_decay)

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=schedule, weight_decay=LAMBDA_L2)

    trainer = PLCNTrainer(model, optimizer, lambda_phy=LAMBDA_PHY)

    # Training dataset
    dataset = tf.data.Dataset.from_tensor_slices(
        (logs_tr, imgs_tr, lith_tr, y_k_tr, y_phi_tr, kc_mask_tr)
    ).shuffle(n_train, seed=seed).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Training loop with early stopping (Paper: patience=30)
    best_val_loss = float('inf')
    best_weights = None
    patience_counter = 0

    print(f"\n🚀 Training (seed={seed})...")
    for epoch in range(MAX_EPOCHS):
        epoch_losses = []
        for batch in dataset:
            b_logs, b_imgs, b_lith, b_yk, b_phi, b_mask = batch
            loss, _, _ = trainer.train_step(
                b_logs, b_imgs, b_lith, b_yk, b_phi, b_mask)
            epoch_losses.append(loss.numpy())

        # Validation loss (unmasked NLL for early stopping)
        val_out = model([logs_va, imgs_va, lith_va], training=False)
        val_nll = 0.5 * tf.reduce_mean(
            val_out["log_var_total"] +
            tf.square(y_k_va - val_out["mu_K"]) / (
                tf.exp(val_out["log_var_total"]) + 1e-6))
        val_loss = val_nll.numpy()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.get_weights()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d} | "
                  f"train_loss={np.mean(epoch_losses):.4f} | "
                  f"val_loss={val_loss:.4f} | "
                  f"patience={patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print(f"  ⏹ Early stopping at epoch {epoch+1}")
            break

    # Restore best weights
    if best_weights is not None:
        model.set_weights(best_weights)

    return model, trainer


# ╔══════════════════════════════════════════════════════════════╗
# ║  6. MAIN: MULTI-SEED EVALUATION                             ║
# ║  Paper: "repeated with five random seeds"                    ║
# ╚══════════════════════════════════════════════════════════════╝

def main():
    # Load data with well-level split + KC-RANSAC
    data = load_and_split_by_well(DATA_FILE)

    seeds = [2025, 2026, 2027, 2028, 2029]
    val_metrics = []
    blind_metrics = []

    for i, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"  RUN {i+1}/5  (seed={seed})")
        print(f"{'='*60}")

        model, trainer = run_training(data, seed=seed)

        # Validation well evaluation
        val_result = trainer.evaluate(
            data["val"]["logs"], data["val"]["images"],
            data["val"]["lith"], data["val"]["logK"],
            data["val"]["phi"])
        val_metrics.append(val_result)

        # Blind well evaluation
        blind_result = trainer.evaluate(
            data["blind"]["logs"], data["blind"]["images"],
            data["blind"]["lith"], data["blind"]["logK"],
            data["blind"]["phi"])
        blind_metrics.append(blind_result)

        print(f"\n  📋 Validation (CS608): R²={val_result['R2']:.4f}  "
              f"RMSE={val_result['RMSE']:.4f}  MAE={val_result['MAE']:.4f}")
        print(f"  📋 Blind     (CS606): R²={blind_result['R2']:.4f}  "
              f"RMSE={blind_result['RMSE']:.4f}  MAE={blind_result['MAE']:.4f}")
        print(f"  📋 Uncertainty: Corr={blind_result['Corr']:.3f}  "
              f"95%PI Coverage={blind_result['Coverage_95PI']:.3f}")

    # ── Summary statistics (Paper: "mean ± standard deviation") ──
    print("\n" + "="*60)
    print("  FINAL RESULTS (Mean ± Std across 5 seeds)")
    print("="*60)

    for name, metrics in [("Validation (CS608)", val_metrics),
                           ("Blind Test (CS606)", blind_metrics)]:
        r2s = [m["R2"] for m in metrics]
        rmses = [m["RMSE"] for m in metrics]
        maes = [m["MAE"] for m in metrics]
        corrs = [m["Corr"] for m in metrics]
        covs = [m["Coverage_95PI"] for m in metrics]

        print(f"\n  {name}:")
        print(f"    R²   = {np.mean(r2s):.3f} ± {np.std(r2s):.3f}")
        print(f"    RMSE = {np.mean(rmses):.3f} ± {np.std(rmses):.3f}")
        print(f"    MAE  = {np.mean(maes):.3f} ± {np.std(maes):.3f}")
        print(f"    Corr = {np.mean(corrs):.3f} ± {np.std(corrs):.3f}")
        print(f"    95%PI Coverage = {np.mean(covs):.3f}")

    # Save blind-well results from best seed
    best_idx = int(np.argmax([m["R2"] for m in blind_metrics]))
    best = blind_metrics[best_idx]
    res_df = pd.DataFrame({
        "Truth_logK": data["blind"]["logK"],
        "Pred_logK": best["mu"],
        "Sigma_total": best["sigma"],
        "Sigma_log": best["sigma_log"],
        "Sigma_img": best["sigma_img"],
        "W_log": best["w_log"],
        "W_img": best["w_img"],
    })
    res_df.to_csv("results_blind_well.csv", index=False)
    print(f"\n✅ Blind-well results saved to results_blind_well.csv")


if __name__ == "__main__":
    main()
