"""Synthetic numerical checks; these do not reproduce manuscript results."""

import json
import unittest

import numpy as np
import pandas as pd

from evaluation import (
    block_bootstrap_uncertainty,
    evaluate_predictions,
    paired_group_comparison,
    summarize_runs,
)


class WholeWellMetricsTests(unittest.TestCase):
    def test_metrics_use_whole_well_and_gaussian_units(self):
        y = np.array([-2.0, 0.0, 1.0, 3.0])
        mu = np.array([-1.0, -1.0, 1.0, 2.0])
        sigma = np.array([0.25, 1.0, 0.5, 0.25])
        result = evaluate_predictions(y, mu, sigma)
        self.assertAlmostEqual(result["rmse"], np.sqrt(3.0 / 4.0))
        self.assertAlmostEqual(result["mae"], 0.75)
        self.assertAlmostEqual(result["r2"], 1.0 - 3.0 / 13.0)
        expected_nll = np.mean(0.5 * ((y - mu) / sigma) ** 2
                               + np.log(sigma) + 0.5 * np.log(2 * np.pi))
        self.assertAlmostEqual(result["nll"], expected_nll)
        self.assertEqual(result["picp_95"], 0.5)

    def test_five_run_summary_retains_exact_r2_rmse_identity(self):
        y = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
        records = [evaluate_predictions(y, y + shift, np.ones(5))
                   for shift in (0.1, 0.2, 0.3, 0.4, 0.5)]
        summary = summarize_runs(pd.DataFrame(records))
        mean = summary["rmse"]["mean"]
        sd = summary["rmse"]["std"]
        inferred_variance = (mean ** 2 + 4.0 / 5.0 * sd ** 2) / (1.0 - summary["r2"]["mean"])
        self.assertAlmostEqual(inferred_variance, np.var(y))
        self.assertAlmostEqual(sd, np.std([0.1, 0.2, 0.3, 0.4, 0.5], ddof=1))
        self.assertEqual(summary["rmse"]["n_valid"], 5)

    def test_constant_target_is_explicitly_undefined_and_json_safe(self):
        record = evaluate_predictions(np.ones(4), np.ones(4), np.ones(4))
        self.assertIsNone(record["r2"])
        summary = summarize_runs(pd.DataFrame([record]))
        self.assertIsNone(summary["r2"]["mean"])
        self.assertEqual(summary["r2"]["n_valid"], 0)
        self.assertIsNone(summary["rmse"]["std"])
        json.dumps(summary, allow_nan=False)

    def test_invalid_sigma_or_misaligned_samples_raise(self):
        with self.assertRaises(ValueError):
            evaluate_predictions([0, 1], [0, 1], [1, 0])
        with self.assertRaises(ValueError):
            evaluate_predictions([0, 1], [0], [1, 1])
        with self.assertRaises(ValueError):
            evaluate_predictions([0, np.nan], [0, 1], [1, 1])


class UncertaintyBootstrapTests(unittest.TestCase):
    def test_exact_linear_relation_has_exact_pearson_and_trend(self):
        depth = np.arange(30)
        sigma = np.linspace(0.1, 1.0, 30)
        y = np.zeros(30)
        mu = 0.5 * sigma
        result = block_bootstrap_uncertainty(depth, y, mu, sigma, 3, 60, 17)
        self.assertAlmostEqual(result["pearson"]["r"], 1.0)
        np.testing.assert_allclose(result["pearson"]["ci_95"], [1, 1], atol=1e-12)
        self.assertAlmostEqual(result["ols_trend"]["slope"], 0.5)
        np.testing.assert_allclose(result["ols_trend"]["pointwise_ci_95_lower"],
                                   0.5 * np.asarray(result["ols_trend"]["x"]), atol=1e-12)
        coverage = result["coverage"]
        self.assertAlmostEqual(coverage["ace"], np.mean(np.abs(np.asarray(coverage["nominal"])
                                                               - np.asarray(coverage["empirical"]))))
        self.assertEqual(sum(item["n_samples"] for item in result["bins"]), 30)
        self.assertEqual(coverage["picp_95"], evaluate_predictions(y, mu, sigma)["picp_95"])
        json.dumps(result, allow_nan=False)

    def test_depth_sorting_makes_input_row_permutation_irrelevant(self):
        depth = np.arange(24)
        y = np.sin(depth)
        sigma = np.linspace(0.1, 0.6, 24)
        mu = y + sigma * (1 + np.cos(depth))
        order = np.random.default_rng(6).permutation(24)
        original = block_bootstrap_uncertainty(depth, y, mu, sigma, 3, 40, 8)
        shuffled = block_bootstrap_uncertainty(depth[order], y[order], mu[order], sigma[order], 3, 40, 8)
        self.assertEqual(original, shuffled)

    def test_degenerate_uncertainty_has_null_statistics_not_nan(self):
        result = block_bootstrap_uncertainty(np.arange(10), np.arange(10), np.arange(10), np.ones(10), 2, 20)
        self.assertIsNone(result["pearson"]["r"])
        self.assertIsNone(result["pearson"]["ci_95"])
        self.assertIsNone(result["ols_trend"]["slope"])
        self.assertEqual(result["pearson"]["undefined_resamples"], 20)
        json.dumps(result, allow_nan=False)


class PairedGroupComparisonTests(unittest.TestCase):
    def test_identical_models_have_zero_paired_interval_and_no_improvements(self):
        n = 30
        depth = np.arange(n)
        y = np.sin(depth)
        predictions = np.stack([y + (depth + 1) * scale for scale in (0.01, 0.02, 0.03, 0.04, 0.05)])
        result = paired_group_comparison(y, predictions, predictions, depth,
                                         ["lava"] * n, np.where(depth % 2, 12.0, 10.0), 10.0,
                                         block_length=3, n_resamples=50)
        for group in result["groups"]:
            self.assertEqual(group["rmse_reduction_mean"], 0.0)
            self.assertEqual(group["rmse_reduction_ci_95"], [0.0, 0.0])
            self.assertEqual(group["samples_improved_percent"], 0.0)
            self.assertFalse(group["ci_excludes_zero"])

    def test_runwise_means_and_threshold_boundary(self):
        y = np.zeros(6)
        baseline = np.array([[1, 1, 3, 3, 2, 2], [3, 3, 1, 1, 4, 4]], dtype=float)
        plcn = baseline / 2.0
        result = paired_group_comparison(y, baseline, plcn, np.arange(6), ["tuff"] * 6,
                                         [10, 11, 10, 11, 12, 12], 10,
                                         threshold=0.1, block_length=1, n_resamples=50, seed=1)
        low, high = result["groups"]
        self.assertEqual(low["n_samples"], 4)
        self.assertEqual(high["n_samples"], 2)
        self.assertAlmostEqual(low["baseline_rmse_mean"], np.sqrt(5))
        self.assertAlmostEqual(high["baseline_rmse_mean"], 3.0)
        self.assertNotAlmostEqual(high["baseline_rmse_mean"], np.sqrt(10))
        for group in result["groups"]:
            self.assertAlmostEqual(group["relative_rmse_reduction_percent"], 50.0)
            self.assertAlmostEqual(group["samples_improved_percent"], 100.0)
        json.dumps(result, allow_nan=False)

    def test_empty_group_resamples_are_counted(self):
        n = 12
        result = paired_group_comparison(np.zeros(n), np.ones((2, n)), np.zeros((2, n)),
                                         np.arange(n), ["lava"] * 11 + ["tuff"],
                                         np.full(n, 10), 10, block_length=1, n_resamples=100, seed=7)
        rare = next(g for g in result["groups"] if g["lithology"] == "tuff"
                    and g["enlargement_category"] == "low_moderate")
        self.assertGreater(rare["empty_resamples"], 0)
        self.assertEqual(rare["empty_resamples"] + rare["valid_resamples"], 100)
        absent = next(g for g in result["groups"] if g["enlargement_category"] == "high")
        self.assertIsNone(absent["rmse_reduction_ci_95"])
        self.assertEqual(absent["empty_resamples"], 100)

    def test_unpaired_run_counts_raise(self):
        with self.assertRaises(ValueError):
            paired_group_comparison(np.zeros(6), np.zeros((2, 6)), np.zeros((3, 6)),
                                    np.arange(6), ["lava"] * 6, np.ones(6), 1)


if __name__ == "__main__":
    unittest.main()
