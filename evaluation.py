"""Evaluation of saved P-LCN predictions, without training dependencies.

All targets, predicted means and standard deviations must be in log10(mD).
Metrics are calculated over the entire supplied evaluation set, never averaged
over mini-batches. No training mask is applied to validation or test samples.

Bootstrap intervals describe depth-sampling variability conditional on the
supplied fitted models. They do not quantify variability between new wells or
retrain models. The moving-block bootstrap sorts by depth and samples contiguous
blocks of observations, not fixed physical lengths in metres.
"""

from statistics import NormalDist
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


METRIC_NAMES = ("r2", "rmse", "mae", "nll", "picp_95")
_STANDARD_NORMAL = NormalDist()
_Z95 = _STANDARD_NORMAL.inv_cdf(0.975)


def _vector(value: Any, name: str, length: Optional[int] = None) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim == 2 and result.shape[1] == 1:
        result = result[:, 0]
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if length is not None and result.size != length:
        raise ValueError(f"{name} must contain {length} samples")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains nonfinite values")
    return result


def _predictions(y: Any, mu: Any, sigma: Any):
    y = _vector(y, "y")
    mu = _vector(mu, "mu", len(y))
    sigma = _vector(sigma, "sigma", len(y))
    if np.any(sigma <= 0):
        raise ValueError("sigma must be strictly positive")
    return y, mu, sigma


def evaluate_predictions(y: Any, mu: Any, sigma: Any) -> Dict[str, Any]:
    """Return whole-array metrics for one fitted model and one evaluation well.

    ``nll`` is the unmasked mean Gaussian negative log-likelihood, including
    the 0.5*log(2*pi) constant. ``picp_95`` is a fraction (0--1), using the
    central nominal 95% Gaussian interval in log10 permeability. R2 is null
    when the target variance is zero; a perfect fit to a constant target does
    not make the coefficient of determination defined.
    """
    y, mu, sigma = _predictions(y, mu, sigma)
    error = y - mu
    mse = float(np.mean(error ** 2))
    target_variance = float(np.mean((y - np.mean(y)) ** 2))
    return {
        "n_samples": int(len(y)),
        "r2": None if target_variance == 0 else float(1.0 - mse / target_variance),
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(error))),
        "nll": float(np.mean(0.5 * (error / sigma) ** 2
                             + np.log(sigma) + 0.5 * np.log(2.0 * np.pi))),
        "picp_95": float(np.mean(np.abs(error) <= _Z95 * sigma)),
    }


def summarize_runs(metrics: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Summarize per-run whole-well metrics with sample SD (ddof=1).

    Each row must describe one run on the same evaluation samples. Metadata
    columns such as seed are ignored. Missing/undefined metrics are counted
    explicitly by ``n_valid``; one valid run has no estimable sample SD.
    This function cannot verify sample identity from metrics alone.
    """
    if not isinstance(metrics, pd.DataFrame) or len(metrics) == 0:
        raise ValueError("metrics must be a nonempty pandas DataFrame")
    names = [name for name in METRIC_NAMES if name in metrics.columns]
    if not names:
        raise ValueError(f"metrics must include at least one of {METRIC_NAMES}")
    summary = {}
    for name in names:
        values = pd.to_numeric(metrics[name], errors="raise").to_numpy(dtype=float)
        if np.any(np.isinf(values)):
            raise ValueError(f"{name} contains infinite values")
        valid = values[np.isfinite(values)]
        summary[name] = {
            "mean": float(np.mean(valid)) if len(valid) else None,
            "std": float(np.std(valid, ddof=1)) if len(valid) > 1 else None,
            "n_runs": int(len(values)),
            "n_valid": int(len(valid)),
        }
    return summary


def _bootstrap_config(n: int, block_length: int, n_resamples: int):
    for value, name in ((block_length, "block_length"), (n_resamples, "n_resamples")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    if not 1 <= block_length <= n:
        raise ValueError("block_length must be between 1 and the sample count")
    if n_resamples < 2:
        raise ValueError("n_resamples must be at least 2")


def _moving_block_indices(n: int, length: int, rng: np.random.Generator):
    """Noncircular moving blocks; concatenate and truncate to original n."""
    starts = rng.integers(0, n - length + 1, size=(n + length - 1) // length)
    return (starts[:, None] + np.arange(length)[None, :]).ravel()[:n]


def _percentile_ci(values: Any):
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    if valid.size < 2:
        return None
    return [float(v) for v in np.percentile(valid, [2.5, 97.5])]


def _pearson(x: np.ndarray, y: np.ndarray):
    xc, yc = x - np.mean(x), y - np.mean(y)
    denominator = float(np.sqrt(np.sum(xc ** 2) * np.sum(yc ** 2)))
    if denominator == 0:
        return None
    return float(np.clip(np.sum(xc * yc) / denominator, -1.0, 1.0))


def _ols(x: np.ndarray, y: np.ndarray):
    centered = x - np.mean(x)
    denominator = float(np.sum(centered ** 2))
    if denominator == 0:
        return None
    slope = float(np.sum(centered * (y - np.mean(y))) / denominator)
    intercept = float(np.mean(y) - slope * np.mean(x))
    return intercept, slope


def block_bootstrap_uncertainty(
    depth: Any, y: Any, mu: Any, sigma: Any,
    block_length: int = 5, n_resamples: int = 1000, seed: int = 2025,
) -> Dict[str, Any]:
    """Evaluate uncertainty for one supplied prediction vector.

    Coverage/ACE use the fixed nominal grid 0.05, 0.10, ..., 0.95. Up to ten
    equal-frequency uncertainty bins are fixed before resampling. Their RMSE
    confidence intervals use common whole-well depth-block resamples, retaining
    those bin assignments. Empty-bin resamples and undefined correlations/OLS
    fits are omitted and their counts are returned. The OLS band is pointwise,
    not a simultaneous band or an individual prediction interval.
    """
    y, mu, sigma = _predictions(y, mu, sigma)
    depth = _vector(depth, "depth", len(y))
    _bootstrap_config(len(y), block_length, n_resamples)
    order = np.argsort(depth, kind="stable")
    y, mu, sigma = y[order], mu[order], sigma[order]
    absolute_error = np.abs(y - mu)
    nominal = np.arange(1, 20, dtype=float) / 20.0
    z = np.asarray([_STANDARD_NORMAL.inv_cdf((1.0 + p) / 2.0) for p in nominal])
    empirical = np.mean(absolute_error[:, None] <= sigma[:, None] * z[None, :], axis=0)

    n_bins = min(10, len(y))
    uncertainty_order = np.argsort(sigma, kind="stable")
    members = np.array_split(uncertainty_order, n_bins)
    bin_ids = np.empty(len(y), dtype=int)
    bins = []
    for index, indices in enumerate(members):
        bin_ids[indices] = index
        bins.append({
            "bin": index + 1,
            "n_samples": int(len(indices)),
            "mean_sigma": float(np.mean(sigma[indices])),
            "rmse": float(np.sqrt(np.mean(absolute_error[indices] ** 2))),
        })

    x_grid = np.linspace(float(np.min(sigma)), float(np.max(sigma)), 100)
    fitted = _ols(sigma, absolute_error)
    pearson_samples = []
    trend_samples = []
    bin_samples = [[] for _ in bins]
    rng = np.random.default_rng(seed)
    for _ in range(n_resamples):
        indices = _moving_block_indices(len(y), block_length, rng)
        x, error = sigma[indices], absolute_error[indices]
        correlation = _pearson(x, error)
        if correlation is not None:
            pearson_samples.append(correlation)
        resampled_fit = _ols(x, error)
        if resampled_fit is not None:
            intercept, slope = resampled_fit
            trend_samples.append(intercept + slope * x_grid)
        sampled_bins = bin_ids[indices]
        for index in range(n_bins):
            selected = error[sampled_bins == index]
            if selected.size:
                bin_samples[index].append(float(np.sqrt(np.mean(selected ** 2))))

    for index, item in enumerate(bins):
        item["rmse_ci_95"] = _percentile_ci(bin_samples[index])
        item["valid_resamples"] = len(bin_samples[index])
        item["empty_resamples"] = n_resamples - len(bin_samples[index])
    band = np.percentile(trend_samples, [2.5, 97.5], axis=0) if len(trend_samples) >= 2 else None
    return {
        "n_samples": int(len(y)),
        "bootstrap": {
            "method": "depth-ordered noncircular moving-block bootstrap",
            "block_length_samples": int(block_length),
            "n_resamples": int(n_resamples), "seed": int(seed),
        },
        "pearson": {
            "r": _pearson(sigma, absolute_error),
            "ci_95": _percentile_ci(pearson_samples),
            "valid_resamples": len(pearson_samples),
            "undefined_resamples": n_resamples - len(pearson_samples),
        },
        "coverage": {
            "nominal": nominal.tolist(), "empirical": empirical.tolist(),
            "ace": float(np.mean(np.abs(empirical - nominal))),
            "picp_95": float(empirical[-1]),
        },
        "bins": bins,
        "ols_trend": {
            "x": x_grid.tolist(),
            "intercept": fitted[0] if fitted is not None else None,
            "slope": fitted[1] if fitted is not None else None,
            "fitted": (fitted[0] + fitted[1] * x_grid).tolist() if fitted is not None else None,
            "pointwise_ci_95_lower": band[0].tolist() if band is not None else None,
            "pointwise_ci_95_upper": band[1].tolist() if band is not None else None,
            "valid_resamples": len(trend_samples),
            "undefined_resamples": n_resamples - len(trend_samples),
        },
    }


def _run_predictions(value: Any, name: str, n: int):
    result = np.asarray(value, dtype=float)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] != n:
        raise ValueError(f"{name} must have shape (runs, {n})")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains nonfinite values")
    return result


def paired_group_comparison(
    y: Any, baseline_mu: Any, plcn_mu: Any, depth: Any, lithology: Any,
    caliper: Any, bit_diameter: Any, threshold: float = 0.1,
    block_length: int = 5, n_resamples: int = 1000, seed: int = 2025,
) -> Dict[str, Any]:
    """Compare paired runs within lithology/enlargement groups of one well.

    Predictions have shape (runs, samples) with identical run and sample order
    for both models. Caliper and bit diameter must use the same physical unit;
    bit diameter can be scalar or sample-wise. High enlargement means strictly
    E > threshold, where E = (caliper - bit_diameter) / bit_diameter.

    RMSE is calculated separately for each run, then averaged. Relative reduction
    is the difference between mean RMSEs divided by baseline mean RMSE. Improved
    percentage is the mean of the per-run percentages with strictly smaller
    absolute errors. A single depth-index resample is shared by both models and
    every run. Group labels stay attached to their original samples. Empty-group
    resamples are skipped and reported; confidence intervals are null when fewer
    than two valid resamples remain. Seed variability is not itself resampled.
    """
    y = _vector(y, "y")
    n = len(y)
    baseline = _run_predictions(baseline_mu, "baseline_mu", n)
    plcn = _run_predictions(plcn_mu, "plcn_mu", n)
    if baseline.shape != plcn.shape:
        raise ValueError("baseline_mu and plcn_mu must have the same number of paired runs")
    depth = _vector(depth, "depth", n)
    caliper = _vector(caliper, "caliper", n)
    bit = np.asarray(bit_diameter, dtype=float)
    if bit.ndim == 0:
        bit = np.full(n, float(bit))
    bit = _vector(bit, "bit_diameter", n)
    if np.any(bit <= 0) or np.any(caliper <= 0):
        raise ValueError("caliper and bit_diameter must be strictly positive")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")
    lithology = np.asarray(lithology, dtype=object)
    if lithology.ndim != 1 or len(lithology) != n or np.any(pd.isna(lithology)):
        raise ValueError("lithology must contain one nonmissing label per sample")
    lithology = np.asarray([str(label).strip() for label in lithology])
    if np.any(lithology == ""):
        raise ValueError("lithology labels must not be empty")
    _bootstrap_config(n, block_length, n_resamples)
    order = np.argsort(depth, kind="stable")
    y, baseline, plcn = y[order], baseline[:, order], plcn[:, order]
    lithology = lithology[order]
    enlargement = (caliper[order] - bit[order]) / bit[order]
    high = enlargement > threshold
    baseline_error, plcn_error = baseline - y[None, :], plcn - y[None, :]

    groups, memberships = [], []
    for category, is_high in (("low_moderate", False), ("high", True)):
        for label in sorted(set(lithology.tolist())):
            mask = (lithology == label) & (high == is_high)
            memberships.append(mask)
            count = int(np.sum(mask))
            item = {"lithology": label, "enlargement_category": category, "n_samples": count}
            if count:
                baseline_rmse = np.sqrt(np.mean(baseline_error[:, mask] ** 2, axis=1))
                plcn_rmse = np.sqrt(np.mean(plcn_error[:, mask] ** 2, axis=1))
                improved = 100.0 * np.mean(np.abs(plcn_error[:, mask]) < np.abs(baseline_error[:, mask]), axis=1)
                baseline_mean, plcn_mean = float(np.mean(baseline_rmse)), float(np.mean(plcn_rmse))
                reduction = baseline_mean - plcn_mean
                item.update({
                    "baseline_rmse_mean": baseline_mean, "plcn_rmse_mean": plcn_mean,
                    "rmse_reduction_mean": reduction,
                    "relative_rmse_reduction_percent": 100.0 * reduction / baseline_mean if baseline_mean > 0 else None,
                    "samples_improved_percent": float(np.mean(improved)),
                    "baseline_rmse_per_run": baseline_rmse.tolist(),
                    "plcn_rmse_per_run": plcn_rmse.tolist(),
                    "samples_improved_percent_per_run": improved.tolist(),
                })
            else:
                item.update({name: None for name in (
                    "baseline_rmse_mean", "plcn_rmse_mean", "rmse_reduction_mean",
                    "relative_rmse_reduction_percent", "samples_improved_percent",
                    "baseline_rmse_per_run", "plcn_rmse_per_run", "samples_improved_percent_per_run",
                )})
            groups.append(item)

    bootstrap_differences = [[] for _ in groups]
    rng = np.random.default_rng(seed)
    for _ in range(n_resamples):
        indices = _moving_block_indices(n, block_length, rng)
        for group_index, membership in enumerate(memberships):
            selected = indices[membership[indices]]
            if selected.size == 0:
                continue
            baseline_rmse = np.sqrt(np.mean(baseline_error[:, selected] ** 2, axis=1))
            plcn_rmse = np.sqrt(np.mean(plcn_error[:, selected] ** 2, axis=1))
            bootstrap_differences[group_index].append(float(np.mean(baseline_rmse - plcn_rmse)))
    for item, differences in zip(groups, bootstrap_differences):
        item["rmse_reduction_ci_95"] = _percentile_ci(differences)
        item["valid_resamples"] = len(differences)
        item["empty_resamples"] = n_resamples - len(differences)
        ci = item["rmse_reduction_ci_95"]
        item["ci_excludes_zero"] = None if ci is None else bool(ci[0] > 0 or ci[1] < 0)
    return {
        "n_samples": n, "n_runs": int(baseline.shape[0]),
        "enlargement_threshold": float(threshold),
        "bootstrap": {
            "method": "paired depth-ordered noncircular moving-block bootstrap",
            "block_length_samples": int(block_length), "n_resamples": int(n_resamples),
            "seed": int(seed), "resamples_shared_across_models_and_runs": True,
            "scope": "depth-sampling variability conditional on supplied fitted runs",
        },
        "groups": groups,
    }
