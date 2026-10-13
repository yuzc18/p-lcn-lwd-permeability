"""Fast offline forward/loss/training smoke test on in-memory synthetic inputs.

Run ``python quick_test.py``. This checks execution and numerical contracts,
not manuscript accuracy or calibration. The separate file-based demo exercises
well splitting, preprocessing, checkpoint selection and exported predictions.
"""

import argparse

import numpy as np
import tensorflow as tf

from train import LOG_EMB_DIM, LOG_WINDOW_L, PLCN, PLCNTrainer, kc_ransac_filter
from train import masked_heteroscedastic_nll, porosity_loss


def synthetic_inputs(n=32, seed=42):
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0.03, 0.22, n).astype(np.float32)
    signal = ((phi - 0.12) / 0.06)[:, None, None]
    logs = (signal + rng.normal(0, 0.3, (n, LOG_WINDOW_L, 5))).astype(np.float32)
    images = np.clip(phi[:, None, None, None] * 3
                     + rng.normal(0, 0.12, (n, 64, 64, 6)), 0, 1).astype(np.float32)
    lith = rng.dirichlet([2, 2, 2], n).astype(np.float32)
    log_k = (2.3 * np.log10(phi) + 2.8 + rng.normal(0, 0.2, n)).astype(np.float32)
    log_k[::9] -= 0.8
    return logs, images, lith, phi, log_k


def check_masked_losses():
    """Mask removes permeability supervision but not auxiliary supervision."""
    mu = tf.Variable([[0.0], [0.0]], dtype=tf.float32)
    phi_hat = tf.Variable([[0.0], [0.0]], dtype=tf.float32)
    mask = tf.constant([[1.0], [0.0]])
    with tf.GradientTape(persistent=True) as tape:
        loss_k = masked_heteroscedastic_nll(
            tf.constant([[1.0], [100.0]]), mu, tf.zeros_like(mu), mask)
        loss_phi = porosity_loss(tf.constant([[0.1], [0.3]]), phi_hat)
    changed = masked_heteroscedastic_nll(
        tf.constant([[1.0], [-100.0]]), mu, tf.zeros_like(mu), mask)
    np.testing.assert_allclose(loss_k.numpy(), changed.numpy(), atol=1e-7)
    np.testing.assert_allclose(tape.gradient(loss_k, mu).numpy()[1], 0, atol=1e-7)
    assert abs(tape.gradient(loss_phi, phi_hat).numpy()[1, 0]) > 0
    assert np.isfinite(loss_k.numpy()) and np.isfinite(loss_phi.numpy())
    del tape


def check_outputs(outputs):
    keys = ("mu_K", "sigma_total", "log_var_total", "phi_hat",
            "sigma_log", "sigma_img", "w_log", "w_img")
    for key in keys:
        assert key in outputs, f"Missing model output: {key}"
        assert np.all(np.isfinite(outputs[key].numpy())), f"Nonfinite output: {key}"
    for key in ("sigma_total", "sigma_log", "sigma_img"):
        assert np.all(outputs[key].numpy() > 0), f"Nonpositive output: {key}"
    for key in ("w_log", "w_img"):
        assert np.all((outputs[key].numpy() >= 0) & (outputs[key].numpy() <= 1))
    np.testing.assert_allclose((outputs["w_log"] + outputs["w_img"]).numpy(), 1, atol=1e-6)
    phi = outputs["phi_hat"].numpy()
    assert np.all((phi >= 0) & (phi <= 1)), "Porosity is a fraction"
    # Only final-head sigma_total is a permeability standard deviation.
    np.testing.assert_allclose(outputs["sigma_total"].numpy() ** 2,
                               np.exp(outputs["log_var_total"].numpy()), rtol=1e-5, atol=2e-6)


def run_smoke_test(steps=3):
    if steps < 1:
        raise ValueError("steps must be positive")
    tf.keras.utils.set_random_seed(42)
    check_masked_losses()
    logs, images, lith, phi, log_k = synthetic_inputs()
    mask, a, b, scale = kc_ransac_filter(phi, log_k, n_iter=100, random_state=42)
    residual = log_k - (a * np.log10(phi) + b)
    assert np.all(residual[mask == 0] < 0), "Only low-side labels may be masked"
    assert np.isfinite(scale) and scale > 0

    model = PLCN(emb_dim=LOG_EMB_DIM, name="P_LCN_smoke")
    inputs = [tf.constant(x[:8]) for x in (logs, images, lith)]
    check_outputs(model(inputs, training=False))
    trainer = PLCNTrainer(model, tf.keras.optimizers.AdamW(
        learning_rate=1e-3, weight_decay=1e-4), lambda_phy=1.0)
    for step in range(steps):
        start = (step % 4) * 8
        section = slice(start, start + 8)
        batch = [tf.constant(x[section]) for x in (logs, images, lith)]
        batch += [tf.constant(x[section, None]) for x in (log_k, phi, mask)]
        losses = trainer.train_step(*batch)
        assert all(np.isfinite(float(x.numpy())) for x in losses), "Nonfinite training loss"
        print(f"step {step + 1}/{steps}: total loss = {float(losses[0]):.6f}")
    check_outputs(model(inputs, training=False))
    test_logs, test_images, test_lith, test_phi, test_k = synthetic_inputs(8, seed=43)
    result = trainer.evaluate(test_logs, test_images, test_lith, test_k, test_phi)
    assert np.all(np.isfinite(result["mu"])) and np.all(np.isfinite(result["sigma"]))
    assert 0 <= result["Coverage_95PI"] <= 1
    print(f"Synthetic smoke metrics: R2={result['R2']:.3f}, RMSE={result['RMSE']:.3f}")
    print("ALL SMOKE CHECKS PASSED: finite outputs, valid weights, masked NLL and unmasked porosity loss.")
    print("These synthetic metrics are not manuscript results or evidence of uncertainty calibration.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3)
    run_smoke_test(parser.parse_args().steps)


if __name__ == "__main__":
    main()
