import unittest
import numpy as np
import tensorflow as tf
from train import PLCN, PLCNTrainer, WarmupCosineSchedule, kc_ransac_filter


class TrainingProtocolTests(unittest.TestCase):
    def test_exact_horizontal_kc_trend_and_degenerate_porosity(self):
        mask, a, b, scale = kc_ransac_filter(np.array([.1, .2, .3, .4]), np.zeros(4), n_iter=10)
        self.assertAlmostEqual(a, 0)
        self.assertAlmostEqual(b, 0)
        self.assertTrue(np.all(mask == 1))
        self.assertGreater(scale, 0)
        with self.assertRaises(ValueError):
            kc_ransac_filter(np.full(4, .2), np.arange(4.))

    def test_warmup_uses_optimizer_steps_and_short_smoke_is_finite(self):
        schedule = WarmupCosineSchedule(130)
        self.assertAlmostEqual(float(schedule(0)), 1e-6, places=10)
        self.assertAlmostEqual(float(schedule(65)), .002, places=8)
        self.assertAlmostEqual(float(schedule(130)), 1e-6, places=10)
        self.assertTrue(np.isfinite([float(WarmupCosineSchedule(3)(x)) for x in range(3)]).all())

    def test_explicit_l2_is_in_loss_independent_of_optimizer_decay(self):
        tf.keras.utils.set_random_seed(3)
        model = PLCN()
        trainer = PLCNTrainer(model, tf.keras.optimizers.AdamW(learning_rate=0., weight_decay=0.),
                              lambda_phy=1., lambda_reg=.002)
        logs = tf.zeros((2, 5, 5))
        images = tf.zeros((2, 64, 64, 6))
        lith = tf.ones((2, 3)) / 3
        total, perm, phi = trainer.train_step(logs, images, lith, tf.zeros((2, 1)),
                                              tf.ones((2, 1)) * .2, tf.ones((2, 1)))
        actual_l2 = sum(float(tf.reduce_sum(tf.square(v))) for v in model.trainable_variables)
        self.assertAlmostEqual(float(total - perm - phi), .002 * actual_l2, places=5)


if __name__ == "__main__":
    unittest.main()
