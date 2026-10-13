# P-LCN: Multimodal Permeability Prediction

Reference implementation for the manuscript:

> **Multimodal fusion of well logs and cuttings thin-section images for uncertainty-aware permeability prediction in volcanic reservoirs**

P-LCN combines well-log sequences with paired plane-polarized light (PPL) and cross-polarized light (XPL) images. The public repository includes a runnable **synthetic demonstration**, not the proprietary study dataset. Synthetic results do **not** reproduce or validate the manuscript's numerical results, tables, or figures.

## Installation and quick start

The reference environment is **Python 3.12, TensorFlow 2.16.2, and Keras 3**. Install the pinned dependencies in a virtual environment:

```bash
python -m venv .venv
```

Activate it with `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on Linux/macOS, then run:

```bash
python -m pip install -r requirements.txt
python generate_synthetic_data.py
python train.py --epochs 2 --seeds 2025 2026 --output-dir outputs/smoke --bootstrap-resamples 100
```

The deterministic generator creates 210 artificial depth samples in `data/synthetic/synthetic_demo_data.csv` and paired images in `data/synthetic/thin_section_images/`. The images are generated textures, not field photomicrographs. The short command above checks execution; two epochs and 100 bootstrap resamples are not the manuscript's experiment settings.

Run the default five-seed demonstration with:

```bash
python train.py
```

## Model and training protocol

| Manuscript component | Public implementation |
|---|---|
| Well-level partition and preprocessing | `data_pipeline.py`: fit log normalization on training wells; keep validation and test wells separate |
| KC-RANSAC screening | `train.py`: fit the porosity–permeability trend on training wells and mask large negative standardized residuals from the permeability loss only |
| Lithology-gated log expert | `train.py`: three depthwise separable 1D convolution blocks, with channel gating from supplied frozen-classifier probabilities |
| Image expert | `train.py`: paired 64 × 64 RGB crops, six-channel concatenation, learned 6-to-3 projection, and MobileNetV3-Small truncated at its third bottleneck |
| Adaptive fusion | `train.py`: positive branch scores and normalized, regularized inverse-square weights |
| Cascaded prediction | `train.py`: auxiliary porosity prediction concatenated with fused features, followed by a Gaussian permeability head |
| Training losses | `train.py`: masked Gaussian NLL, porosity MSE on all training samples, and configurable explicit L2 regularization |
| Evaluation and resampling | `evaluation.py`: regression metrics, interval diagnostics, and paired group-comparison utilities |
| Artificial inputs | `generate_synthetic_data.py`: deterministic synthetic logs, labels, probabilities, and paired images |

The branch scores control fusion; they are **not calibrated modality-specific uncertainty estimates**. The final head separately predicts the mean `mu` and aleatoric standard deviation `sigma_total` of log10 permeability. Nominal 95% intervals are `mu ± 1.96 * sigma_total` in log space; mD limits are obtained by exponentiating both endpoints with base 10. Coverage is measured, not assumed to equal 95%.

The fixed well split is:

| Subset | Wells | Samples reported in the manuscript's private dataset |
|---|---|---:|
| Training | WF1, CS12, CS13, CS601, CS607 | 858 |
| Validation | CS608 | 143 |
| Held-out test | CS606 | 258 |

These sample counts describe the private study, not the 210-row synthetic demonstration. Early stopping uses unmasked validation NLL, and the best validation checkpoint is restored within each run. The test well is not used for model or run selection.

Defaults include 200 maximum epochs, batch size 64, seeds 2025–2029, early-stopping patience 30, auxiliary porosity loss weight 1.0, and KC threshold 0.60. AdamW uses weight decay `1e-4`; linear warmup lasts 65 optimizer steps from `1e-6` to `2e-3`, followed by cosine decay.

The public demonstration sets `--lambda-reg 1e-4` for **explicit L2 loss**, separately from AdamW's weight decay. This is a documented demonstration default, **not a recovered value from the original experiment**: the current manuscript leaves the historical explicit L2 coefficient unresolved. Likewise, `--log-window 5` and `--window-span-m 1` specify the public window construction; the manuscript describes approximately 1 m but does not identify the historical point count. Preserve the actual settings and provenance when evaluating authorized study data.

## Input data

For data in the supported tabular schema:

```bash
python train.py --data data/private/samples.csv --image-dir data/private/thin_section_images --logs-file data/private/dense_logs.csv --porosity-unit fraction --output-dir outputs/private_run
```

`--logs-file` is optional. Run `python train.py --help` for the complete CLI.

| Column | Meaning |
|---|---|
| `Well`, `Depth` | Well identifier and depth in metres |
| `GR`, `DEN`, `CNL`, `AC`, `RLA5` | Log inputs in API, g/cm³, %, µs/m, and Ω·m, respectively |
| `LithProb_*` | Raw class probabilities supplied by a frozen lithology classifier; these are not standardized with the log curves |
| `LogK_Core` | Target log10 permeability, with permeability expressed in mD |
| `Porosity_WL` | Auxiliary log-interpreted porosity; use `--porosity-unit fraction` (default) or `--porosity-unit percent` to declare its units |

Images are named `{Well}_{Depth}_PPL.png` and `{Well}_{Depth}_XPL.png`, for example `CS12_2662.3_PPL.png`. Each RGB image is center-cropped to 64 × 64 pixels, and the registered PPL/XPL pair is concatenated along the channel dimension. Inputs must already have valid depth pairing; this repository does not infer circulation lag from acquisition records.

Log windows use evenly spaced physical depth offsets across the specified span. Interpolation is performed **within each well**, with edge clamping at its available depth limits; windows never cross a well boundary. Provide dense log observations through `--logs-file` when available. That file requires `Well`, `Depth`, and the five log columns, not permeability labels. Without it, the sample table supplies the log observations, so sparse sampling may limit the information contained in a 1 m window.

For real inputs, retain the provenance of `LithProb_*`: classifier fitting and its preprocessing must exclude held-out test-well information. The supplied synthetic probabilities are artificial demonstration inputs, not evidence about the original classifier's training data. The training script consumes these probabilities; it does not reproduce the earlier lithology-classification study.

## Outputs and interpretation

Each run writes to a new, empty `--output-dir` (default `outputs/demo/`):

| File | Contents |
|---|---|
| `per_seed_metrics.csv` | Separate validation and test metrics for each seed |
| `metrics_summary.json` | Across-run metric summaries |
| `per_seed_predictions.csv` | Per-sample predictions and uncertainty outputs, retaining seed identity |
| `selected_run_predictions.csv` | Predictions from the run selected by the lowest validation NLL |
| `seed_{seed}.weights.h5` | Restored best-checkpoint model weights for that seed |
| `uncertainty_validation.json` | Validation uncertainty diagnostics for the selected run |
| `run_metadata.json` | Configuration, selection, and data-processing metadata |
| `training_mask_{seed}.csv` | Per-depth permeability-loss mask for each seed's training-only KC fit |

For a selected-run visualization, use that run's `mu`, `sigma_total`, and fusion weights together. Do not combine a mean prediction across runs with one run's uncertainty band. The implementation keeps run-wise evaluation distinct from selected-run diagnostics.

`evaluation.py` also exposes paired, depth-ordered block-bootstrap utilities for group comparisons. They require predictions from both actual comparison models on matching samples. The default training command fits P-LCN; it does **not** claim to execute all manuscript baselines, ablations, robustness experiments, or the complete figure-generation workflow. A synthetic run is an execution check, not empirical support for the manuscript's accuracy or uncertainty claims.

Use `analyze_predictions.py` to analyze saved predictions without importing TensorFlow:

```bash
python analyze_predictions.py --predictions outputs/demo/per_seed_predictions.csv --metadata outputs/demo/run_metadata.json --split blind --output-dir outputs/analysis
```

To add a paired group comparison, supply `--baseline path/to/baseline_predictions.csv`. Both files must contain identical seeds, wells, depths and `truth_logK` values; each needs `mu_logK`, and the P-LCN file also needs `sigma_total`. Group analysis additionally needs `Lithology`, `Caliper`, and `BitSize`, with diameters in the same units. No baseline predictions are synthesized by this analysis script. Defaults are enlargement threshold 0.10, blocks of five consecutive depth samples, and 1,000 resamples. Intervals quantify depth-sampling variability conditional on the fitted models, not generalization uncertainty across new wells.

## Checks

```bash
python -m unittest discover -s tests
python quick_test.py
```

## Data access and license

The original well logs, core measurements, and thin-section photomicrographs are subject to data-provider and patent-related restrictions and are not included. Access requests should be directed to the corresponding author, subject to permission from the data provider. Keep authorized private inputs outside the public repository or under the ignored `data/private/` directory; do not upload them with demonstration outputs.

Code and the generated synthetic demonstration assets are provided under the [MIT License](LICENSE). This license does not grant access to, or rights over, the proprietary study data.
