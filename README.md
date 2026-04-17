# P-LCN: Physics-Informed Uncertainty-Aware Multimodal Fusion for LWD Permeability Evaluation

Official implementation of the paper:

> **Physics-Informed Uncertainty-Aware Multimodal Fusion for Logging-While-Drilling Permeability Evaluation in Volcanic Reservoirs**

## Architecture

```
LWD Logs (B,L,5) ──► 1D-CNN (3×DSC) ──► Lithology Gating ──► h_log, σ_log ─┐
                                                                              ├─► IVW Fusion ──► ẑ ──► φ̂ ──► [ẑ; φ̂] ──► μ_K, σ_total
PPL+XPL (B,64,64,6) ──► 1×1 Conv(6→3) ──► MobileNetV3-Small ──► h_img, σ_img ┘
```

**Key Modules:**
- **KC-RANSAC**: Kozeny–Carman-guided asymmetric label filtering (training-time only)
- **Lithology Gating**: Sigmoid-gated channel modulation on log features (Eq. 4)
- **IVW Fusion**: Inverse-variance weighting at the feature level (Eq. 5–6)
- **Cascaded Head**: Physics-causal φ → K prediction pathway (Eq. 7–9)
- **Heteroscedastic NLL**: Masked Gaussian negative log-likelihood loss (Eq. 11–12)

## Requirements

```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install tensorflow numpy pandas scikit-learn scipy openpyxl Pillow
```

## Data

The dataset comprises 1,259 samples from seven wells in the Huoshiling Formation, Wangfu Fault Depression, Songliao Basin. Due to proprietary constraints, raw data are not publicly released. Please contact the corresponding author for data access requests.

**Expected data format** (`master_dataset_v11_ideal.xlsx`):

| Column | Description |
|---|---|
| `Well` | Well identifier (WF1, CS12, CS13, CS601, CS607, CS608, CS606) |
| `Depth` | Measured depth (m) |
| `GR, AC, DEN, CN, RLA5` | Five conventional LWD curves |
| `LithProb_*` | Lithology probability vector from pre-trained classifier |
| `LogK_Ideal` | Core-measured permeability in log₁₀(mD) |
| `Porosity_v6` | Well-log interpreted porosity |

**Thin-section images** (`thin_section_images/` directory):

| File naming | Description |
|---|---|
| `{Well}_{Depth}_PPL.png` | Plane-polarized light image (1024×1024) |
| `{Well}_{Depth}_XPL.png` | Cross-polarized light image (1024×1024) |

Example: `CS12_2662.3_PPL.png`, `CS12_2662.3_XPL.png`

Each PPL+XPL pair is center-cropped to 64×64 and concatenated into a 6-channel input.

## Quick Test (no data required)

To verify the full P-LCN pipeline using synthetic data:

```bash
python quick_test.py
```

This script validates all core modules without requiring real data:
- KC-RANSAC asymmetric label filtering
- Log Expert (1D-CNN + DSC + Lithology Gating)
- Image Expert (MobileNetV3-Small with 6-channel adapter)
- IVW feature-level fusion
- Cascaded φ→K prediction head
- Masked heteroscedastic NLL training
- Uncertainty-aware inference

**Expected output:** Training progress for 5 epochs followed by prediction metrics and `✅ ALL TESTS PASSED`.

## Training with Real Data

```bash
python train.py
```

Results are reported as mean ± std across 5 random seeds with well-level data splitting:
- **Training**: WF1, CS12, CS13, CS601, CS607 (858 samples)
- **Validation**: CS608 (143 samples)
- **Blind test**: CS606 (258 samples, fully held out)

## Paper–Code Correspondence

| Paper Section | Code Location |
|---|---|
| KC-RANSAC (Eq. 3) | `kc_ransac_filter()` |
| Image preprocessing (center-crop, PPL+XPL) | `load_image_pair()`, `center_crop()` |
| Log Expert + Lithology Gating (Eq. 4) | `LogExpertWithGating` class |
| Image Expert (MobileNetV3-Small) | `ImageExpert` class |
| IVW Fusion (Eq. 5–6) | `IVWFeatureFusion` class |
| Cascaded Head (Eq. 7–9) | `CascadedPredictionHead` class |
| Masked NLL (Eq. 11–12) | `masked_heteroscedastic_nll()` |
| Porosity Loss (Eq. 13) | `porosity_loss()` |
| Well-level split | `load_and_split_by_well()` |
| Cosine Decay + Warmup (Table 2) | `WarmupCosineSchedule` class |

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Citation

```
[To be added upon publication]
```
