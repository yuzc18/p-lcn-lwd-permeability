import unittest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from data_pipeline import RAW_LOGS, depth_windows, center_crop, load_and_split_by_well


class DataTests(unittest.TestCase):
    def test_window_never_crosses_wells_and_tracks_target_depth(self):
        rows = []
        for well, offset in (("A", 0), ("B", 1000)):
            for depth in (1., 0., .5):
                rows.append(dict(Well=well, Depth=depth, **{c: depth + offset for c in RAW_LOGS}))
        source = pd.DataFrame(rows)
        targets = pd.DataFrame(dict(Well=["B", "A"], Depth=[.5, .5]))
        scaler = StandardScaler().fit(source[RAW_LOGS].to_numpy())
        windows = depth_windows(targets, source, scaler, 5, 1.)
        restored = scaler.inverse_transform(windows.reshape(-1, 5)).reshape(2, 5, 5)
        np.testing.assert_allclose(restored[0, :, 0], [1000, 1000.25, 1000.5, 1000.75, 1001], atol=.0001)
        np.testing.assert_allclose(restored[1, :, 0], [0, .25, .5, .75, 1], atol=.0001)

    def test_small_native_image_rejected(self):
        with self.assertRaises(ValueError):
            center_crop(np.zeros((32, 64, 3)))

    def test_target_outside_log_coverage_rejected(self):
        source = pd.DataFrame(dict(Well=["A", "A"], Depth=[0., 1.], **{c: [1., 2.] for c in RAW_LOGS}))
        scaler = StandardScaler().fit(source[RAW_LOGS].to_numpy())
        with self.assertRaisesRegex(ValueError, "outside log coverage"):
            depth_windows(pd.DataFrame(dict(Well=["A"], Depth=[2.])), source, scaler)

    def test_demo_pairing_and_training_only_scaler(self):
        data = load_and_split_by_well()
        tr, va, bl = [data[s] for s in ("train", "val", "blind")]
        wells = [set(x["frame"].Well) for x in (tr, va, bl)]
        self.assertFalse(wells[0] & wells[1] or wells[0] & wells[2] or wells[1] & wells[2])
        np.testing.assert_allclose(data["meta"]["log_mean"], tr["frame"][RAW_LOGS].mean().to_numpy())
        for subset in (tr, va, bl):
            np.testing.assert_allclose(subset["lith"].sum(1), 1, atol=1e-6)
            self.assertTrue(np.all((subset["phi"] > 0) & (subset["phi"] < 1)))
            np.testing.assert_allclose(subset["logs"][:, 2, :],
                                       (subset["frame"][RAW_LOGS].to_numpy() - data["meta"]["log_mean"]) / data["meta"]["log_scale"], atol=1e-6)
            self.assertEqual(subset["images"].shape, (len(subset["logK"]), 64, 64, 6))


if __name__ == "__main__":
    unittest.main()
