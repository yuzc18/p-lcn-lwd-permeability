"""Numerical/scientific contracts; run with unittest discovery."""

import unittest

import numpy as np
import tensorflow as tf

from quick_test import check_masked_losses, check_outputs, synthetic_inputs
from train import IVWFeatureFusion, PLCN, masked_heteroscedastic_nll


class ModelContractTests(unittest.TestCase):
    def test_mask_only_removes_permeability_supervision(self):
        check_masked_losses()

    def test_all_masked_batch_has_zero_permeability_loss(self):
        loss = masked_heteroscedastic_nll(
            tf.constant([[5.0], [-5.0]]), tf.zeros((2, 1)),
            tf.zeros((2, 1)), tf.zeros((2, 1)))
        self.assertAlmostEqual(float(loss), 0.0)

    def test_fusion_downweights_larger_score(self):
        fusion = IVWFeatureFusion()
        log_embedding = tf.constant([[1.0, 2.0]])
        image_embedding = tf.constant([[3.0, 4.0]])
        fused, w_log, w_img = fusion([
            log_embedding, tf.constant([[1.0]]),
            image_embedding, tf.constant([[2.0]])])
        self.assertGreater(float(w_log[0, 0]), float(w_img[0, 0]))
        np.testing.assert_allclose((w_log + w_img).numpy(), 1, atol=1e-6)
        np.testing.assert_allclose(fused.numpy(),
                                   (w_log * log_embedding + w_img * image_embedding).numpy())

    def test_model_outputs_and_third_bottleneck(self):
        tf.keras.utils.set_random_seed(42)
        logs, images, lith, _, _ = synthetic_inputs(2)
        model = PLCN()
        check_outputs(model([logs, images, lith], training=False))
        names = [layer.name for layer in model.img_expert.backbone.layers]
        self.assertIn("expanded_conv_2", names[-1])
        self.assertFalse(any("expanded_conv_3" in name for name in names))


if __name__ == "__main__":
    unittest.main()
