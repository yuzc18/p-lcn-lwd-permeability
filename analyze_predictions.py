"""Analyze saved whole-well predictions without importing TensorFlow.

Required CSV columns: seed, split, Well, Depth, truth_logK, mu_logK, sigma_total.
A baseline CSV uses the same schema but may omit sigma_total. All targets and
predictions are in log10(mD). For the optional lithology/enlargement comparison,
the primary CSV must also contain Lithology, Caliper and BitSize.

Example:
    python analyze_predictions.py --predictions outputs/run/per_seed_predictions.csv \
        --metadata outputs/run/run_metadata.json --split blind \
        --baseline outputs/baseline/per_seed_predictions.csv --output-dir outputs/analysis

For several seeds, supply either an explicit --seed or run_metadata.json with
the validation-NLL selection record. No seed is selected using held-out scores.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from evaluation import (
    block_bootstrap_uncertainty,
    evaluate_predictions,
    paired_group_comparison,
    summarize_runs,
)


_KEYS = ["Well", "Depth"]
_GROUP_COLUMNS = ["Lithology", "Caliper", "BitSize"]


def _numeric(frame, columns, source):
    for column in columns:
        try:
            values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{source}: {column} must be numeric") from exc
        if not np.isfinite(values).all():
            raise ValueError(f"{source}: {column} contains missing or nonfinite values")
        frame[column] = values


def _synthetic_status(frame, source):
    if "synthetic" not in frame:
        return None
    normalized = frame.synthetic.astype(str).str.lower()
    values = normalized.map({"true": True, "false": False, "1": True, "0": False,
                             "1.0": True, "0.0": False})
    if values.isna().any() or values.nunique() != 1:
        raise ValueError(f"{source}: synthetic must be one consistent Boolean value")
    return bool(values.iloc[0])


def _load_predictions(path: Path, split: str, require_sigma: bool):
    source = str(path)
    frame = pd.read_csv(path, dtype={"Well": str, "split": str, "Lithology": str})
    required = ["seed", "split", "Well", "Depth", "truth_logK", "mu_logK"]
    if require_sigma:
        required.append("sigma_total")
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{source}: missing columns {missing}")
    frame = frame.loc[frame.split == split].copy()
    if frame.empty:
        raise ValueError(f"{source}: no rows for split {split!r}")
    if frame.Well.isna().any() or (frame.Well.str.strip() == "").any():
        raise ValueError(f"{source}: Well contains missing or empty identifiers")
    if (frame.Well != frame.Well.str.strip()).any():
        raise ValueError(f"{source}: remove surrounding whitespace from Well identifiers explicitly")
    columns = ["seed", "Depth", "truth_logK", "mu_logK"]
    columns += [c for c in ("sigma_total", "Caliper", "BitSize") if c in frame]
    _numeric(frame, columns, source)
    if np.any(frame.seed != np.floor(frame.seed)):
        raise ValueError(f"{source}: seeds must be integers")
    frame["seed"] = frame.seed.astype(np.int64)
    if "sigma_total" in frame and (frame.sigma_total <= 0).any():
        raise ValueError(f"{source}: sigma_total must be strictly positive")
    if frame.duplicated(["seed"] + _KEYS).any():
        raise ValueError(f"{source}: duplicate (seed, Well, Depth) rows")
    if frame.Well.nunique() != 1:
        raise ValueError(f"{source}: analyze one well per bootstrap; split {split!r} contains several wells")
    if "Lithology" in frame and (frame.Lithology.isna().any()
                                  or (frame.Lithology.str.strip() == "").any()):
        raise ValueError(f"{source}: Lithology contains missing or empty labels")
    synthetic = _synthetic_status(frame, source)
    runs = {int(seed): group.sort_values(_KEYS, kind="stable").reset_index(drop=True)
            for seed, group in frame.groupby("seed", sort=True)}
    reference = next(iter(runs.values()))
    for seed, run in runs.items():
        if not run[_KEYS].equals(reference[_KEYS]):
            raise ValueError(f"{source}: seed {seed} has a different sample key set")
        for column in ["truth_logK"] + [c for c in _GROUP_COLUMNS if c in frame]:
            if not np.array_equal(run[column].to_numpy(), reference[column].to_numpy()):
                raise ValueError(f"{source}: {column} differs between seeds")
    return runs, synthetic


def _select_seed(seeds, requested_seed: Optional[int], metadata_path: Optional[Path]):
    if requested_seed is not None:
        if requested_seed not in seeds:
            raise ValueError(f"Requested seed {requested_seed} is absent from predictions")
        return requested_seed, "explicit --seed; selection provenance supplied by user"
    if metadata_path is not None:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        selected = metadata.get("selected_seed")
        if isinstance(selected, bool) or not isinstance(selected, int) or selected not in seeds:
            raise ValueError("Metadata selected_seed must identify an available integer seed")
        records = metadata.get("runs", [])
        if not isinstance(records, list) or len(records) != len(seeds):
            raise ValueError("Metadata must contain one validation-NLL run record per available seed")
        losses = {}
        for record in records:
            run_seed, loss = record.get("seed"), record.get("best_validation_nll")
            if isinstance(run_seed, bool) or not isinstance(run_seed, int) or run_seed in losses:
                raise ValueError("Metadata contains invalid or duplicate run seeds")
            if isinstance(loss, bool) or not isinstance(loss, (float, int)) or not np.isfinite(loss):
                raise ValueError("Metadata best_validation_nll must be finite for every run")
            losses[run_seed] = float(loss)
        if set(losses) != set(seeds) or losses[selected] != min(losses.values()):
            raise ValueError("Metadata selected_seed is not a minimum-validation-NLL run for these seeds")
        return selected, "minimum validation NLL from supplied run metadata"
    if len(seeds) == 1:
        return seeds[0], "only supplied seed"
    raise ValueError("Several seeds are present: supply --seed or --metadata with validation-NLL selection records")


def _save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def run_analysis(args):
    """Validate complete pairing, then write computed analysis artifacts."""
    output = args.output_dir
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Choose a new or empty --output-dir to preserve existing results: {output}")
    runs, synthetic = _load_predictions(args.predictions, args.split, require_sigma=True)
    seeds = sorted(runs)
    selected_seed, selection = _select_seed(seeds, args.seed, args.metadata)
    reference = runs[seeds[0]]
    baseline_runs = None
    baseline_synthetic = None
    if args.baseline is not None:
        baseline_runs, baseline_synthetic = _load_predictions(args.baseline, args.split, require_sigma=False)
        if set(baseline_runs) != set(runs):
            raise ValueError("Primary and baseline predictions must have exactly the same seeds")
        if synthetic is not None and baseline_synthetic is not None and synthetic != baseline_synthetic:
            raise ValueError("Primary and baseline synthetic-data flags disagree")
        missing = sorted(set(_GROUP_COLUMNS) - set(reference.columns))
        if missing:
            raise ValueError(f"Primary predictions need {missing} for group comparison")
        for seed in seeds:
            primary, baseline = runs[seed], baseline_runs[seed]
            if not primary[_KEYS].equals(baseline[_KEYS]):
                raise ValueError(f"Primary/baseline sample keys differ for seed {seed}; no rows will be dropped")
            for column in ["truth_logK"] + [c for c in _GROUP_COLUMNS if c in baseline]:
                if not np.array_equal(primary[column].to_numpy(), baseline[column].to_numpy()):
                    raise ValueError(f"Primary/baseline {column} differs for seed {seed}")

    records = [{"seed": seed, "split": args.split,
                **evaluate_predictions(run.truth_logK.to_numpy(), run.mu_logK.to_numpy(), run.sigma_total.to_numpy())}
               for seed, run in runs.items()]
    summary = summarize_runs(pd.DataFrame(records))
    selected = runs[selected_seed]
    diagnostics = block_bootstrap_uncertainty(
        selected.Depth.to_numpy(), selected.truth_logK.to_numpy(),
        selected.mu_logK.to_numpy(), selected.sigma_total.to_numpy(),
        block_length=args.block_length, n_resamples=args.bootstrap_resamples, seed=args.bootstrap_seed,
    )
    groups = None
    baseline_records = []
    if baseline_runs is not None:
        for seed, run in baseline_runs.items():
            y, mu = run.truth_logK.to_numpy(), run.mu_logK.to_numpy()
            mse = float(np.mean((y - mu) ** 2))
            variance = float(np.mean((y - y.mean()) ** 2))
            values = {"seed": seed, "split": args.split, "n_samples": len(y),
                      "r2": None if variance == 0 else 1.0 - mse / variance,
                      "rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(y - mu)))}
            if "sigma_total" in run:
                values.update(evaluate_predictions(y, mu, run.sigma_total.to_numpy()))
            baseline_records.append(values)
        groups = paired_group_comparison(
            reference.truth_logK.to_numpy(),
            np.stack([baseline_runs[seed].mu_logK.to_numpy() for seed in seeds]),
            np.stack([runs[seed].mu_logK.to_numpy() for seed in seeds]),
            reference.Depth.to_numpy(), reference.Lithology.to_numpy(),
            reference.Caliper.to_numpy(), reference.BitSize.to_numpy(),
            threshold=args.enlargement_threshold, block_length=args.block_length,
            n_resamples=args.bootstrap_resamples, seed=args.bootstrap_seed,
        )
    metadata = {
        "split": args.split, "well": str(reference.Well.iloc[0]),
        "seeds": seeds, "n_runs": len(seeds), "n_samples_per_run": len(reference),
        "selected_seed": selected_seed, "selection": selection,
        "synthetic": synthetic, "baseline_synthetic": baseline_synthetic,
        "predictions": str(args.predictions.resolve()),
        "predictions_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
        "baseline": str(args.baseline.resolve()) if args.baseline else None,
        "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest() if args.baseline else None,
        "selection_metadata": str(args.metadata.resolve()) if args.metadata else None,
        "selection_metadata_sha256": hashlib.sha256(args.metadata.read_bytes()).hexdigest() if args.metadata else None,
        "aggregation": "whole-well per-run metrics; sample SD (ddof=1); no pooled runs",
        "pairing": "exact seed, sample key and truth equality; no intersection or dropped rows",
        "bootstrap": {"block_length_samples": args.block_length,
                      "n_resamples": args.bootstrap_resamples, "seed": args.bootstrap_seed},
        "units": "targets, predicted means and SDs in log10(mD); PICP is a fraction",
    }
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output / "per_seed_metrics.csv", index=False)
    _save_json(output / "metrics_summary.json", summary)
    _save_json(output / f"uncertainty_{args.split}.json",
               {"seed": selected_seed, "selection": selection, "synthetic": synthetic, **diagnostics})
    if groups is not None:
        pd.DataFrame(baseline_records).to_csv(output / "baseline_per_seed_metrics.csv", index=False)
        _save_json(output / "baseline_metrics_summary.json", summarize_runs(pd.DataFrame(baseline_records)))
        _save_json(output / "group_comparison.json", {"synthetic": synthetic, **groups})
    _save_json(output / "analysis_metadata.json", metadata)
    print(f"Saved analysis for {metadata['well']} ({len(seeds)} runs) to {output.resolve()}")
    return metadata


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--split", default="blind", help="Exact split value in prediction CSVs")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--seed", type=int, help="Explicit representative run; never selected using test metrics")
    selection.add_argument("--metadata", type=Path, help="Training run_metadata.json with validation-NLL selection")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-length", type=int, default=5)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=2025)
    parser.add_argument("--enlargement-threshold", type=float, default=0.1)
    args = parser.parse_args(argv)
    if args.block_length < 1 or args.bootstrap_resamples < 2:
        parser.error("--block-length must be positive and --bootstrap-resamples must be at least 2")
    if not np.isfinite(args.enlargement_threshold) or args.enlargement_threshold < 0:
        parser.error("--enlargement-threshold must be finite and nonnegative")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        return run_analysis(args)
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit(f"Analysis failed: {exc}") from exc


if __name__ == "__main__":
    main()
