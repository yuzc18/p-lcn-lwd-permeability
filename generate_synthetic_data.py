"""Create the public, fully synthetic P-LCN execution example.

No measured logs, core labels, photomicrographs, or trained classifier outputs
are used. Shared artificial signals make the inputs nontrivial; they are not
tuned to reproduce the manuscript's accuracy, coverage, or group effects.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


SEED = 2025
IMAGE_SIZE = 64
DEPTH_SPACING_M = 0.25
LITHOLOGIES = ("Lava", "Breccia", "Tuff")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "synthetic"
WELL_CONFIG = {
    # Name: (synthetic sample count, synthetic interval start, split).
    # Well identifiers preserve the split interface, not the field records.
    "WF1": (30, 3089.0, "training"),
    "CS12": (25, 2550.0, "training"),
    "CS13": (35, 3000.0, "training"),
    "CS601": (30, 3055.0, "training"),
    "CS607": (25, 2875.0, "training"),
    "CS608": (25, 3550.0, "validation"),
    "CS606": (40, 3000.0, "blind"),
}


def depth_token(depth):
    """Keep the quarter-metre grid intact in filenames."""
    return f"{depth:.2f}".rstrip("0").rstrip(".")


def synthetic_image_pair(rng, porosity, lithology, size=IMAGE_SIZE):
    """Paired artificial RGB textures sharing geometry, not microscope data."""
    yy, xx = np.mgrid[:size, :size]
    ppl = np.full((size, size, 3), (176, 158, 129), dtype=np.float64)
    xpl = np.full((size, size, 3), (62, 65, 80), dtype=np.float64)
    for _ in range(12 + 4 * lithology):
        cx, cy = rng.uniform(0, size, 2)
        radius = rng.uniform(2.0, 7.0)
        grain = (xx - cx) ** 2 + (yy - cy) ** 2 < radius ** 2
        ppl[grain] = rng.uniform(110, 235, 3)
        xpl[grain] = rng.uniform(40, 240, 3)
    # The shared artificial porosity signal changes the number of dark patches.
    for _ in range(max(1, int(65 * porosity))):
        cx, cy = rng.uniform(0, size, 2)
        pore = (xx - cx) ** 2 + (yy - cy) ** 2 < rng.uniform(2, 5) ** 2
        ppl[pore] = (25, 28, 29)
        xpl[pore] = (12, 16, 22)
    noise = rng.normal(0, 3, (size, size, 3))
    return tuple(np.clip(a + noise, 0, 255).astype(np.uint8) for a in (ppl, xpl))


def generate_dataset(output_dir=DEFAULT_OUTPUT_DIR, seed=SEED):
    """Write deterministic CSV/images and a manifest; return their paths."""
    output_dir = Path(output_dir)
    image_dir = output_dir / "thin_section_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    image_hash = hashlib.sha256()

    for well_index, (well, (n, start, split)) in enumerate(WELL_CONFIG.items()):
        depths = start + DEPTH_SPACING_M * np.arange(n)
        position = np.linspace(0, 1, n)
        phase = 2 * np.pi * position + 0.3 * well_index
        lith = (np.floor(position * 4).astype(int) + well_index) % 3
        phi = np.clip(0.11 + 0.055 * np.sin(phase)
                      + 0.012 * (lith - 1) + rng.normal(0, 0.008, n), 0.025, 0.23)
        enlargement = np.clip(0.015 + 0.16 * np.exp(-0.5 * ((position - 0.6) / 0.13) ** 2)
                              + rng.normal(0, 0.008, n), 0, 0.3)
        log_noise = 1 + 5 * enlargement
        gr = 70 + 30 * (lith == 2) + 12 * (lith == 1) - 50 * phi + rng.normal(0, 4, n) * log_noise
        den = 2.72 - 1.5 * phi + rng.normal(0, 0.018, n) * log_noise
        cnl = 100 * phi + rng.normal(0, 1.1, n) * log_noise
        ac = 180 + 400 * phi + 5 * lith + rng.normal(0, 5, n) * log_noise
        rla5 = np.exp(6 - 10 * phi + 0.2 * lith + rng.normal(0, 0.15, n) * log_noise)
        log_k = 2.3 * np.log10(phi) + 2.8 + 0.22 * np.cos(phase) + rng.normal(0, 0.18, n)
        # Artificial low-side deviations exercise training-time label masking.
        log_k = log_k - (rng.random(n) < 0.08) * rng.uniform(0.7, 1.0, n)

        for i, depth in enumerate(depths):
            concentration = np.full(3, 1.0)
            concentration[lith[i]] = 9.0
            probabilities = rng.dirichlet(concentration)
            row = {
                "Synthetic": True, "Well": well, "Depth": float(depth),
                "GR": gr[i], "AC": ac[i], "DEN": den[i], "CNL": cnl[i], "RLA5": rla5[i],
                "Porosity_WL": phi[i], "LogK_Core": log_k[i],
                "Lithology": LITHOLOGIES[lith[i]],
                "Caliper": 8.5 * (1 + enlargement[i]), "BitSize": 8.5,
            }
            row.update({f"LithProb_{j}": probabilities[j] for j in range(3)})
            rows.append(row)
            for mode, pixels in zip(("PPL", "XPL"), synthetic_image_pair(rng, phi[i], lith[i])):
                name = f"{well}_{depth_token(depth)}_{mode}.png"
                target = image_dir / name
                Image.fromarray(pixels).save(target)
                image_hash.update(name.encode("utf-8"))
                image_hash.update(target.read_bytes())

    frame = pd.DataFrame(rows)
    csv_path = output_dir / "synthetic_demo_data.csv"
    frame.to_csv(csv_path, index=False, float_format="%.8f", lineterminator="\n")
    manifest = {
        "data_kind": "fully synthetic demonstration; not study measurements",
        "purpose": "execution and interface validation; not reproduction of manuscript results",
        "generator_seed": int(seed),
        "n_samples": len(frame), "n_images": 2 * len(frame),
        "image_shape": [IMAGE_SIZE, IMAGE_SIZE, 3],
        "depth_spacing_m": DEPTH_SPACING_M,
        "log_window": "5 points, 0.25 m spacing, 1 m between first and last points",
        "porosity_unit": "fraction", "permeability_target": "log10(K / mD)",
        "caliper_and_bit_size_unit": "inches",
        "lithology_probability_class_order": list(LITHOLOGIES),
        "lithology_probability_source": "synthetic Dirichlet draws, not a fitted classifier",
        "well_counts": {well: spec[0] for well, spec in WELL_CONFIG.items()},
        "well_roles": {well: spec[2] for well, spec in WELL_CONFIG.items()},
        "image_notes": "Paired artificial textures share geometry. No field photomicrographs are included.",
        "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "ordered_images_sha256": image_hash.hexdigest(),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return {"data_file": csv_path, "image_dir": image_dir, "manifest": manifest_path}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    paths = generate_dataset(args.output_dir, args.seed)
    print("Created 210 fully synthetic samples and 420 paired artificial RGB images.")
    for label, path in paths.items():
        print(f"{label}: {path.resolve()}")
    print("Synthetic metrics do not reproduce or validate the manuscript's empirical results.")


if __name__ == "__main__":
    main()
