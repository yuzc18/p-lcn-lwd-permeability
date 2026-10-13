"""Saved-prediction CLI checks using synthetic CSV fixtures only."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from analyze_predictions import main


class AnalysisCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.root = Path(self.temp.name)
        records = []
        for seed in (1, 2):
            for depth in range(12):
                records.append({"seed": seed, "split": "blind", "Well": "test-well", "Depth": depth,
                                "truth_logK": depth / 10, "mu_logK": depth / 10 + seed * 0.1,
                                "sigma_total": 0.2 + depth * 0.01,
                                "Lithology": "lava" if depth % 2 else "tuff",
                                "Caliper": 10 if depth < 6 else 12, "BitSize": 10,
                                "synthetic": True})
        self.primary = pd.DataFrame(records)
        self.path = self.root / "predictions.csv"
        self.primary.to_csv(self.path, index=False)
        self.output = self.root / "analysis"

    def tearDown(self):
        self.temp.cleanup()

    def invoke(self, *extra):
        arguments = ["--predictions", str(self.path), "--output-dir", str(self.output),
                     "--block-length", "2", "--bootstrap-resamples", "20", *extra]
        with contextlib.redirect_stdout(io.StringIO()):
            return main(arguments)

    def baseline(self, frame=None):
        path = self.root / "baseline.csv"
        frame = self.primary.drop(columns="sigma_total") if frame is None else frame
        frame.to_csv(path, index=False)
        return str(path)

    def test_shuffled_baseline_without_sigma_is_paired_exactly(self):
        baseline = self.primary.drop(columns="sigma_total").sample(frac=1, random_state=6)
        result = self.invoke("--seed", "1", "--baseline", self.baseline(baseline))
        self.assertEqual(result["n_runs"], 2)
        self.assertTrue(result["synthetic"])
        groups = json.loads((self.output / "group_comparison.json").read_text())
        for group in groups["groups"]:
            self.assertEqual(group["rmse_reduction_mean"], 0)
            self.assertEqual(group["rmse_reduction_ci_95"], [0, 0])

    def test_multiple_seeds_require_explicit_or_recorded_choice(self):
        with self.assertRaisesRegex(SystemExit, "Several seeds"):
            self.invoke()
        self.assertFalse(self.output.exists())

    def test_metadata_uses_validation_nll_not_blind_performance(self):
        metadata = self.root / "run_metadata.json"
        metadata.write_text(json.dumps({"selected_seed": 2, "runs": [
            {"seed": 1, "best_validation_nll": 0.3},
            {"seed": 2, "best_validation_nll": 0.1}]}))
        result = self.invoke("--metadata", str(metadata))
        self.assertEqual(result["selected_seed"], 2)
        uncertainty = json.loads((self.output / "uncertainty_blind.json").read_text())
        self.assertEqual(uncertainty["seed"], 2)

    def test_invalid_metadata_selection_is_rejected(self):
        metadata = self.root / "run_metadata.json"
        metadata.write_text(json.dumps({"selected_seed": 1, "runs": [
            {"seed": 1, "best_validation_nll": 0.3},
            {"seed": 2, "best_validation_nll": 0.1}]}))
        with self.assertRaisesRegex(SystemExit, "minimum-validation-NLL"):
            self.invoke("--metadata", str(metadata))

    def test_missing_baseline_samples_are_not_silently_dropped(self):
        baseline = self.primary.loc[self.primary.Depth != 0].drop(columns="sigma_total")
        with self.assertRaisesRegex(SystemExit, "sample keys differ"):
            self.invoke("--seed", "1", "--baseline", self.baseline(baseline))
        self.assertFalse(self.output.exists())

    def test_different_truth_is_rejected(self):
        baseline = self.primary.copy()
        baseline.loc[baseline.Depth == 0, "truth_logK"] += 0.001
        with self.assertRaisesRegex(SystemExit, "truth_logK differs"):
            self.invoke("--seed", "1", "--baseline", self.baseline(baseline))

    def test_duplicate_sample_keys_are_rejected(self):
        pd.concat([self.primary, self.primary.iloc[:1]]).to_csv(self.path, index=False)
        with self.assertRaisesRegex(SystemExit, "duplicate"):
            self.invoke("--seed", "1")

    def test_bootstrap_does_not_cross_wells(self):
        self.primary.loc[self.primary.Depth >= 6, "Well"] = "another-well"
        self.primary.to_csv(self.path, index=False)
        with self.assertRaisesRegex(SystemExit, "one well per bootstrap"):
            self.invoke("--seed", "1")

    def test_existing_results_are_preserved(self):
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("existing result")
        with self.assertRaisesRegex(SystemExit, "preserve existing results"):
            self.invoke("--seed", "1")
        self.assertEqual(marker.read_text(), "existing result")


if __name__ == "__main__":
    unittest.main()
