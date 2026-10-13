"""Well-level pairing and train-only preprocessing; no empirical study data."""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data/synthetic/synthetic_demo_data.csv"
IMAGE_DIR = DATA_FILE.parent / "thin_section_images"
RAW_LOGS = ["GR", "AC", "DEN", "CNL", "RLA5"]
TRAIN_WELLS = ["WF1", "CS12", "CS13", "CS601", "CS607"]
VAL_WELL, BLIND_WELL = "CS608", "CS606"
WELL_COL, DEPTH_COL, K_COL, PHI_COL = "Well", "Depth", "LogK_Core", "Porosity_WL"
IMAGE_SIZE, IMAGE_CHANNELS, LOG_WINDOW_L = 64, 6, 5


def read_table(path):
    path = Path(path)
    return pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)


def center_crop(img, crop_size=IMAGE_SIZE):
    h, w = img.shape[:2]
    if min(h, w) < crop_size:
        raise ValueError(f"Image {h}x{w} is smaller than native crop {crop_size}.")
    top, left = (h - crop_size) // 2, (w - crop_size) // 2
    return img[top:top + crop_size, left:left + crop_size]


def load_image_pair(well, depth, image_dir=IMAGE_DIR):
    # Preserve fractional depths; never round 3000.25 to a different sample.
    candidates = list(dict.fromkeys([format(float(depth), ".10g"), str(float(depth))]))
    for token in candidates:
        paths = [Path(image_dir) / f"{well}_{token}_{mode}.png" for mode in ("PPL", "XPL")]
        if all(p.is_file() for p in paths):
            arrays = []
            for path in paths:
                with Image.open(path) as im:
                    arrays.append(center_crop(np.asarray(im.convert("RGB"))))
            return np.concatenate(arrays, axis=-1).astype(np.float32) / 255.0
    raise FileNotFoundError(f"Missing paired PPL/XPL images: well={well}, depth={depth}, {image_dir}")


def load_images_for_subset(frame, image_dir=IMAGE_DIR):
    images = np.stack([load_image_pair(w, d, image_dir)
                       for w, d in frame[[WELL_COL, DEPTH_COL]].itertuples(index=False, name=None)])
    return images, np.ones(len(frame), dtype=bool)


def create_log_windows(log_values, window_size=LOG_WINDOW_L):
    """Index windows for an already sorted SINGLE well (not physical-depth interpolation)."""
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    padded = np.pad(log_values, ((window_size // 2, window_size // 2), (0, 0)), mode="edge")
    return np.stack([padded[i:i + window_size] for i in range(len(log_values))]).astype(np.float32)


def depth_windows(targets, source, scaler, window_size=5, span_m=1.0):
    """Interpolate fixed depth offsets within each well; clamp only at well edges."""
    if window_size < 3 or window_size % 2 == 0 or span_m <= 0:
        raise ValueError("Use an odd window_size >=3 and positive span_m")
    windows = np.empty((len(targets), window_size, len(RAW_LOGS)), dtype=np.float32)
    offsets = np.linspace(-span_m / 2, span_m / 2, window_size)
    for well, group in targets.groupby(WELL_COL, sort=False):
        logs = source[source[WELL_COL] == well].sort_values(DEPTH_COL)
        if logs.empty:
            raise ValueError(f"No log observations for {well}")
        d = logs[DEPTH_COL].to_numpy(float)
        if np.any(np.diff(d) <= 0):
            raise ValueError(f"Log depths must be unique within well {well}")
        target_depths = group[DEPTH_COL].to_numpy(float)
        if np.any((target_depths < d[0]) | (target_depths > d[-1])):
            raise ValueError(f"Target depth outside log coverage for {well}: [{d[0]}, {d[-1]}] m")
        values = scaler.transform(logs[RAW_LOGS].to_numpy(float))
        query = target_depths[:, None] + offsets
        for curve in range(len(RAW_LOGS)):
            windows[group.index, :, curve] = np.interp(query, d, values[:, curve])
    return windows


def _validate_numeric(frame, columns, label):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{label}: missing columns {sorted(missing)}")
    if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
        raise ValueError(f"{label}: numeric values must be finite; impute using training-only rules first")


def load_and_split_by_well(data_file=DATA_FILE, image_dir=None, logs_file=None,
                           porosity_unit="fraction", log_window=5, window_span_m=1.0):
    data_file = Path(data_file).resolve()
    image_dir = Path(image_dir) if image_dir else data_file.parent / "thin_section_images"
    df = read_table(data_file)
    if WELL_COL not in df:
        raise ValueError("Missing Well column")
    lith_cols = sorted(c for c in df.columns if c.startswith("LithProb"))
    if not lith_cols:
        raise ValueError("Supply frozen-classifier LithProb_* probabilities (synthetic in the demo)")
    _validate_numeric(df, [DEPTH_COL, K_COL, PHI_COL] + lith_cols, "samples")
    if df.duplicated([WELL_COL, DEPTH_COL]).any():
        raise ValueError("Each (Well, Depth) must identify one paired sample")
    expected = set(TRAIN_WELLS + [VAL_WELL, BLIND_WELL])
    if set(df[WELL_COL]) != expected:
        raise ValueError(f"Expected precisely these seven wells: {sorted(expected)}")
    probabilities = df[lith_cols].to_numpy(float)
    if (probabilities < 0).any() or (probabilities > 1).any() or not np.allclose(probabilities.sum(1), 1, atol=1e-5):
        raise ValueError("LithProb_* must be nonnegative probabilities summing to one")
    if porosity_unit not in ("fraction", "percent"):
        raise ValueError("porosity_unit must be fraction or percent")
    if porosity_unit == "percent":
        df[PHI_COL] = df[PHI_COL] / 100.0
    if not df[PHI_COL].between(0, 1, inclusive="neither").all():
        raise ValueError("Porosity must be in (0, 1) after explicit unit conversion")
    source = read_table(logs_file) if logs_file else df
    if WELL_COL not in source:
        raise ValueError("Log source needs Well column")
    _validate_numeric(source, [DEPTH_COL] + RAW_LOGS, "logs")
    if source.duplicated([WELL_COL, DEPTH_COL]).any():
        raise ValueError("Log source contains duplicate well/depth observations")
    train_source = source[source[WELL_COL].isin(TRAIN_WELLS)]
    scaler = StandardScaler().fit(train_source[RAW_LOGS].to_numpy(float))
    data = {}
    split_wells = {"train": TRAIN_WELLS, "val": [VAL_WELL], "blind": [BLIND_WELL]}
    for split, wells in split_wells.items():
        frame = df[df[WELL_COL].isin(wells)].sort_values([WELL_COL, DEPTH_COL]).reset_index(drop=True)
        images, _ = load_images_for_subset(frame, image_dir)
        data[split] = dict(logs=depth_windows(frame, source, scaler, log_window, window_span_m),
                           images=images, lith=frame[lith_cols].to_numpy(np.float32),
                           logK=frame[K_COL].to_numpy(np.float32),
                           phi=frame[PHI_COL].to_numpy(np.float32), frame=frame)
        print(f"{split}: {len(frame)} paired depth samples from {', '.join(wells)}")
    synthetic = "Synthetic" in df and df["Synthetic"].astype(str).str.lower().eq("true").all()
    data["meta"] = dict(n_curves=len(RAW_LOGS), lith_dim=len(lith_cols),
                        img_input_shape=[64, 64, 6], synthetic=bool(synthetic),
                        log_columns=RAW_LOGS, lithology_probability_columns=lith_cols,
                        log_mean=scaler.mean_.tolist(), log_scale=scaler.scale_.tolist(),
                        log_window=log_window, window_span_m=window_span_m,
                        depth_offsets_m=np.linspace(-window_span_m / 2, window_span_m / 2, log_window).tolist(),
                        window_method="within-well linear interpolation; edge clamping",
                        porosity_output_unit="fraction", data_path=str(data_file),
                        logs_path=str(Path(logs_file).resolve()) if logs_file else str(data_file))
    return data
