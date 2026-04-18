"""
Quick Test for P-LCN (Physics-Informed Lightweight Cascaded Network)

This script validates the full P-LCN pipeline using synthetic data,
including KC-RANSAC label filtering, model construction, training,
and uncertainty-aware inference. No real data is required.

Usage:
    python quick_test.py

Expected output:
    - KC-RANSAC filtering statistics
    - Training progress (5 epochs)
    - Prediction metrics (R², RMSE, uncertainty correlation)
    - All module checks passed

Author: Xiu Jin, Taiji Yu
Contact: yutj1988@126.com
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

# Import P-LCN components from train.py
from train import (
    kc_ransac_filter,
    create_log_windows,
    PLCN,
    PLCNTrainer,
    masked_heteroscedastic_nll,
    porosity_loss,
    LOG_EMB_DIM,
    LOG_WINDOW_L,
)

SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)


def generate_synthetic_data(n_train=200, n_test=50, n_lith=3):
    """
    Generate synthetic volcanic reservoir data mimicking the real dataset
    structure: LWD logs, thin-section images, lithology priors, porosity,
    and permeability with a Kozeny-Carman-like relationship.
    """
    print("=" * 60)
    print("  STEP 1: Generating synthetic volcanic reservoir data")
    print("=" * 60)

    def make_subset(n):
        # Simulate 5 LWD log curves (GR, AC, DEN, CN, RLA5)
        logs = np.random.randn(n, LOG_WINDOW_L, 5).astype(np.float32)

        # Simulate 6-channel PPL+XPL thin-section images (64x64)
        images = np.random.rand(n, 64, 64, 6).astype(np.float32)

        # Simulate lithology probability vector
        lith_raw = np.random.dirichlet([1.0] * n_lith, size=n).astype(np.float32)

        # Simulate porosity (0.01 ~ 0.35)
        phi = np.random.uniform(0.01, 0.35, size=n).astype(np.float32)

        # Simulate permeability via noisy Kozeny-Carman relationship
        # logK = a * log(phi) + b + noise
        logK = (2.5 * np.log10(np.clip(phi, 1e-6, None)) + 1.0
                + np.random.randn(n).astype(np.float32) * 0.3)

        return logs, images, lith_raw, phi, logK

    logs_tr, imgs_tr, lith_tr, phi_tr, logK_tr = make_subset(n_train)
    logs_te, imgs_te, lith_te, phi_te, logK_te = make_subset(n_test)

    print(f"  Train samples: {n_train}")
    print(f"  Test samples:  {n_test}")
    print(f"  Log input shape:   {logs_tr.shape}  (batch, window_L, curves)")
    print(f"  Image input shape: {imgs_tr.shape}  (batch, 64, 64, 6ch)")
    print(f"  Lith input shape:  {lith_tr.shape}  (batch, n_litho)")
    print(f"  Permeability range: [{logK_tr.min():.2f}, {logK_tr.max():.2f}] log10(mD)")

    return (logs_tr, imgs_tr, lith_tr, phi_tr, logK_tr,
            logs_te, imgs_te, lith_te, phi_te, logK_te)


def test_kc_ransac(phi, logK):
    """Test KC-RANSAC label filtering module."""
    print("\n" + "=" * 60)
    print("  STEP 2: Testing KC-RANSAC label governance")
    print("=" * 60)

    mask, a, b, sigma_r = kc_ransac_filter(
        phi, logK, tau=0.60, n_iter=200, random_state=SEED)

    n_retained = int(mask.sum())
    n_rejected = len(mask) - n_retained
    print(f"  Retained: {n_retained}, Rejected: {n_rejected}")

    # Verify asymmetric behavior: only below-trend samples rejected
    log_phi = np.log10(np.clip(phi, 1e-6, None))
    residuals = logK - (a * log_phi + b)
    rejected_residuals = residuals[mask == 0]
    if len(rejected_residuals) > 0:
        assert np.all(rejected_residuals < 0), \
            "FAIL: KC-RANSAC should only reject below-trend samples"
        print("  ✓ Asymmetric filtering verified (only below-trend rejected)")
    else:
        print("  ✓ No samples rejected (data is clean)")

    return mask


def test_model_forward_pass(logs, imgs, lith):
    """Test P-LCN model construction and forward pass."""
    print("\n" + "=" * 60)
    print("  STEP 3: Testing P-LCN model forward pass")
    print("=" * 60)

    model = PLCN(emb_dim=LOG_EMB_DIM, name="P_LCN_test")

    # Forward pass with small batch
    batch_logs = tf.constant(logs[:4], dtype=tf.float32)
    batch_imgs = tf.constant(imgs[:4], dtype=tf.float32)
    batch_lith = tf.constant(lith[:4], dtype=tf.float32)

    outputs = model([batch_logs, batch_imgs, batch_lith], training=False)

    # Verify output keys and shapes
    expected_keys = ["mu_K", "sigma_total", "log_var_total", "phi_hat",
                     "sigma_log", "sigma_img", "w_log", "w_img"]
    for key in expected_keys:
        assert key in outputs, f"FAIL: Missing output key '{key}'"
        assert outputs[key].shape[0] == 4, f"FAIL: Wrong batch dim for '{key}'"
    print(f"  ✓ All {len(expected_keys)} output keys present")

    # Verify uncertainty is positive
    assert tf.reduce_all(outputs["sigma_total"] > 0), \
        "FAIL: sigma_total must be positive"
    assert tf.reduce_all(outputs["sigma_log"] > 0), \
        "FAIL: sigma_log must be positive"
    assert tf.reduce_all(outputs["sigma_img"] > 0), \
        "FAIL: sigma_img must be positive"
    print("  ✓ All uncertainty outputs are positive")

    # Verify fusion weights sum to ~1
    w_sum = outputs["w_log"] + outputs["w_img"]
    assert tf.reduce_all(tf.abs(w_sum - 1.0) < 1e-4), \
        "FAIL: Fusion weights should sum to 1"
    print("  ✓ Fusion weights sum to 1.0")

    # Verify phi_hat in [0, 1] (sigmoid output)
    assert tf.reduce_all(outputs["phi_hat"] >= 0) and \
           tf.reduce_all(outputs["phi_hat"] <= 1), \
        "FAIL: phi_hat should be in [0, 1]"
    print("  ✓ Porosity prediction in valid range [0, 1]")

    n_params = model.count_params()
    print(f"  ✓ Model parameters: {n_params:,}")

    return model


def test_training_loop(model, logs_tr, imgs_tr, lith_tr,
                       logK_tr, phi_tr, kc_mask, n_epochs=5):
    """Test training loop with masked NLL loss."""
    print("\n" + "=" * 60)
    print(f"  STEP 4: Testing training loop ({n_epochs} epochs)")
    print("=" * 60)

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=1e-3, weight_decay=1e-4)
    trainer = PLCNTrainer(model, optimizer, lambda_phy=1.0)

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((
        tf.constant(logs_tr, dtype=tf.float32),
        tf.constant(imgs_tr, dtype=tf.float32),
        tf.constant(lith_tr, dtype=tf.float32),
        tf.constant(logK_tr.reshape(-1, 1), dtype=tf.float32),
        tf.constant(phi_tr.reshape(-1, 1), dtype=tf.float32),
        tf.constant(kc_mask.reshape(-1, 1), dtype=tf.float32),
    )).batch(32)

    losses = []
    for epoch in range(n_epochs):
        epoch_loss = []
        for batch in dataset:
            b_logs, b_imgs, b_lith, b_yk, b_phi, b_mask = batch
            loss, _, _ = trainer.train_step(
                b_logs, b_imgs, b_lith, b_yk, b_phi, b_mask)
            epoch_loss.append(loss.numpy())
        avg_loss = np.mean(epoch_loss)
        losses.append(avg_loss)
        print(f"  Epoch {epoch+1}/{n_epochs}  loss = {avg_loss:.4f}")

    # Verify loss is decreasing
    if losses[-1] < losses[0]:
        print("  ✓ Loss is decreasing (model is learning)")
    else:
        print("  ⚠ Loss did not decrease (may need more epochs)")

    return trainer


def test_inference(trainer, logs_te, imgs_te, lith_te, logK_te, phi_te):
    """Test inference and uncertainty quality."""
    print("\n" + "=" * 60)
    print("  STEP 5: Testing inference and uncertainty outputs")
    print("=" * 60)

    results = trainer.evaluate(logs_te, imgs_te, lith_te, logK_te, phi_te)

    print(f"  R²   = {results['R2']:.4f}")
    print(f"  RMSE = {results['RMSE']:.4f}")
    print(f"  MAE  = {results['MAE']:.4f}")
    print(f"  Uncertainty-Error Correlation = {results['Corr']:.4f}")
    print(f"  95% PI Coverage = {results['Coverage_95PI']:.3f}")

    # Verify outputs are valid
    assert not np.any(np.isnan(results["mu"])), "FAIL: NaN in predictions"
    assert not np.any(np.isnan(results["sigma"])), "FAIL: NaN in uncertainty"
    assert np.all(results["sigma"] > 0), "FAIL: Uncertainty must be positive"
    print("  ✓ No NaN values in outputs")
    print("  ✓ All uncertainty values are positive")

    # Verify fusion weight outputs
    assert np.all((results["w_log"] >= 0) & (results["w_log"] <= 1)), \
        "FAIL: w_log out of range"
    assert np.all((results["w_img"] >= 0) & (results["w_img"] <= 1)), \
        "FAIL: w_img out of range"
    print("  ✓ Fusion weights in valid range [0, 1]")

    return results


def main():
    print("\n" + "#" * 60)
    print("#  P-LCN Quick Test — Synthetic Data Validation")
    print("#" * 60)

    # Step 1: Generate data
    (logs_tr, imgs_tr, lith_tr, phi_tr, logK_tr,
     logs_te, imgs_te, lith_te, phi_te, logK_te) = generate_synthetic_data()

    # Step 2: KC-RANSAC
    kc_mask = test_kc_ransac(phi_tr, logK_tr)

    # Step 3: Model forward pass
    model = test_model_forward_pass(logs_tr, imgs_tr, lith_tr)

    # Step 4: Training
    trainer = test_training_loop(
        model, logs_tr, imgs_tr, lith_tr, logK_tr, phi_tr, kc_mask)

    # Step 5: Inference
    results = test_inference(
        trainer, logs_te, imgs_te, lith_te, logK_te, phi_te)

    # Summary
    print("\n" + "=" * 60)
    print("  ✅ ALL TESTS PASSED — P-LCN pipeline is functional")
    print("=" * 60)
    print("  Modules verified:")
    print("    • KC-RANSAC asymmetric label filtering")
    print("    • Log Expert (1D-CNN + DSC + Lithology Gating)")
    print("    • Image Expert (MobileNetV3-Small + 6ch adapter)")
    print("    • IVW feature-level fusion")
    print("    • Cascaded φ→K prediction head")
    print("    • Masked heteroscedastic NLL training")
    print("    • Uncertainty-aware inference")


if __name__ == "__main__":
    main()
