"""
Generate Synthetic Dataset for P-LCN Code Validation

This script generates a synthetic dataset that mirrors the structure
of the real dataset described in the paper. It creates:
  1. An Excel file with well-log data, lithology probabilities, and labels
  2. Paired PPL/XPL thin-section images (64×64 pixels) for each sample

The synthetic data allows reviewers to validate that the code runs
correctly end-to-end. Metrics obtained from synthetic data will
differ from those reported in the paper, which were obtained using
the real dataset of 1,259 samples from seven wells.

Usage:
    python generate_synthetic_data.py

Output:
    - synthetic_demo_data.xlsx
    - thin_section_images/{Well}_{Depth}_PPL.png
    - thin_section_images/{Well}_{Depth}_XPL.png

Author: Xiu Jin, Taiji Yu
Contact: yutj1988@126.com
"""

import os
import numpy as np
import pandas as pd
from PIL import Image

SEED = 2025
np.random.seed(SEED)

# Well configuration matching paper Table 1
WELL_CONFIG = {
    # Well:  (n_samples, depth_start, depth_end, role)
    "WF1":   (30, 3009.0, 3120.0, "Training"),
    "CS12":  (25, 2550.0, 2700.0, "Training"),
    "CS13":  (35, 3000.0, 3250.0, "Training"),
    "CS601": (30, 3075.0, 3278.0, "Training"),
    "CS607": (25, 2865.0, 3005.0, "Training"),
    "CS608": (25, 3430.0, 3650.0, "Validation"),
    "CS606": (40, 3000.0, 3200.0, "Blind Test"),
}

N_LITH_CLASSES = 18  # Paper: lithology probability vector
IMAGE_SIZE = 64       # Paper: center-cropped to 64×64
OUTPUT_EXCEL = "synthetic_demo_data.xlsx"
OUTPUT_IMAGE_DIR = "thin_section_images"


def generate_synthetic_logs(n):
    """Generate synthetic LWD log curves with realistic ranges."""
    GR = np.random.uniform(20, 200, n)         # API
    AC = np.random.uniform(180, 350, n)        # μs/m
    DEN = np.random.uniform(2.1, 2.8, n)       # g/cm³
    CN = np.random.uniform(2, 35, n)           # %
    RLA5 = np.exp(np.random.uniform(1, 8, n))  # Ω·m (log-normal)
    return GR, AC, DEN, CN, RLA5


def generate_synthetic_labels(n, phi_range=(0.01, 0.30)):
    """Generate porosity and permeability with KC-like relationship."""
    phi = np.random.uniform(phi_range[0], phi_range[1], n)
    # Noisy Kozeny-Carman: logK = a * log(phi) + b + noise
    logK = (3.0 * np.log10(np.clip(phi, 1e-6, None)) + 2.0
            + np.random.randn(n) * 0.4)
    return phi, logK


def generate_synthetic_image(size=IMAGE_SIZE):
    """
    Generate a synthetic thin-section-like image (RGB, 64×64).
    Simulates mineral grains, pores, and texture patterns.
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)

    # Background: random mineral matrix
    base_color = np.random.randint(100, 200, 3)
    img[:, :] = base_color

    # Add random grain-like patches
    n_grains = np.random.randint(10, 30)
    for _ in range(n_grains):
        cx, cy = np.random.randint(0, size, 2)
        r = np.random.randint(2, 8)
        color = np.random.randint(50, 255, 3)
        y, x = np.ogrid[-cx:size - cx, -cy:size - cy]
        mask = x * x + y * y <= r * r
        img[mask] = color

    # Add noise for texture
    noise = np.random.randint(-20, 20, (size, size, 3))
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


def main():
    print("=" * 60)
    print("  Generating Synthetic Dataset for P-LCN Validation")
    print("=" * 60)

    # Create image directory
    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

    all_rows = []
    total_images = 0

    for well, (n_samples, d_start, d_end, role) in WELL_CONFIG.items():
        print(f"\n  {well} ({role}): {n_samples} samples, "
              f"depth {d_start}-{d_end} m")

        # Generate depths
        depths = np.linspace(d_start, d_end, n_samples)
        depths = np.round(depths, 1)

        # Generate log curves
        GR, AC, DEN, CN, RLA5 = generate_synthetic_logs(n_samples)

        # Generate labels
        phi, logK = generate_synthetic_labels(n_samples)

        # Generate lithology probabilities (Dirichlet distribution)
        lith_probs = np.random.dirichlet(
            [0.5] * N_LITH_CLASSES, size=n_samples)

        for i in range(n_samples):
            row = {
                "Well": well,
                "Depth": depths[i],
                "GR": GR[i],
                "AC": AC[i],
                "DEN": DEN[i],
                "CN": CN[i],
                "RLA5": RLA5[i],
                "Porosity_v6": phi[i],
                "LogK_Ideal": logK[i],
            }

            # Add lithology probabilities
            for j in range(N_LITH_CLASSES):
                row[f"LithProb_{j}"] = lith_probs[i, j]

            all_rows.append(row)

            # Generate and save PPL + XPL images
            depth_str = f"{depths[i]:.1f}"

            ppl_img = generate_synthetic_image(IMAGE_SIZE)
            xpl_img = generate_synthetic_image(IMAGE_SIZE)

            ppl_path = os.path.join(
                OUTPUT_IMAGE_DIR, f"{well}_{depth_str}_PPL.png")
            xpl_path = os.path.join(
                OUTPUT_IMAGE_DIR, f"{well}_{depth_str}_XPL.png")

            Image.fromarray(ppl_img).save(ppl_path)
            Image.fromarray(xpl_img).save(xpl_path)
            total_images += 2

    # Save Excel
    df = pd.DataFrame(all_rows)
    df.to_excel(OUTPUT_EXCEL, index=False, engine='openpyxl')

    print(f"\n{'=' * 60}")
    print(f"  ✅ Synthetic dataset generated successfully")
    print(f"{'=' * 60}")
    print(f"  Excel: {OUTPUT_EXCEL} ({len(df)} samples)")
    print(f"  Images: {OUTPUT_IMAGE_DIR}/ ({total_images} files)")
    print(f"\n  Well distribution:")
    for well, (n, _, _, role) in WELL_CONFIG.items():
        print(f"    {well:6s} ({role:10s}): {n} samples")
    print(f"\n  To run training with this data:")
    print(f"    python train.py")
    print(f"\n  Note: Metrics from synthetic data will differ from")
    print(f"  those reported in the paper (real data, 1,259 samples).")


if __name__ == "__main__":
    main()
