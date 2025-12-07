"""
Stage A2: Generate Gaussian+CLAHE images for the IU-CXR pathology sanity subset.

This script:
  ✓ Loads the pathology sanity subset CSV
  ✓ Reads each normalized IU-CXR PNG image
  ✓ Applies Gaussian blur (denoising) + OpenCV CLAHE
  ✓ Saves the enhanced image into a parallel folder
  ✓ Produces a 1:1 mapping of normalized → GAUSS+CLAHE images

Use this to compare:
  - MedCLIP performance on normalized images
  - MedCLIP performance on CLAHE images
  - MedCLIP performance on Gaussian+CLAHE images
"""

import os
import cv2
import pandas as pd
from tqdm import tqdm

# -------------------------------------------------------
# PATHS (edit here if your folder structure changes)
# -------------------------------------------------------

CSV_PATH = r"D:\MedVLMBench\EDA\eda_reports\sanity_subset_iucxr_v02.csv"

NORMALIZED_ROOT = r"D:\MedVLMBench\phase1\data\chestxray_iu\images\images_normalized"

# New output folder — Gaussian+CLAHE version of the same images
GAUSS_CLAHE_ROOT = r"D:\MedVLMBench\phase1\data\chestxray_iu\images\images_gauss_clahe"

os.makedirs(GAUSS_CLAHE_ROOT, exist_ok=True)

# -------------------------------------------------------
# LOAD CSV
# -------------------------------------------------------

print("Loading pathology sanity subset CSV:", CSV_PATH)
df = pd.read_csv(CSV_PATH)

if "Image_ID" not in df.columns:
    raise ValueError("CSV must contain column: 'Image_ID'")

print(f"Found {len(df)} images in pathology sanity subset.")

# -------------------------------------------------------
# CONFIGURE GAUSSIAN + CLAHE
# -------------------------------------------------------
# Gaussian blur: denoising with small sigma
GAUSSIAN_KERNEL = (5, 5)
GAUSSIAN_SIGMA = 1.0

# CLAHE: same as your previous CLAHE script (you can tweak these)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# -------------------------------------------------------
# PROCESS IMAGES
# -------------------------------------------------------

missing = []
failed = []
success = 0

print("\nStarting Gaussian+CLAHE generation...\n")

for _, row in tqdm(df.iterrows(), total=len(df), desc="Gauss+CLAHE Processing"):
    fname = row["Image_ID"]
    src_path = os.path.join(NORMALIZED_ROOT, fname)
    dst_path = os.path.join(GAUSS_CLAHE_ROOT, fname)

    if not os.path.exists(src_path):
        missing.append(src_path)
        continue

    # Read grayscale
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        failed.append(src_path)
        continue

    try:
        # 1) Gaussian blur
        img_blur = cv2.GaussianBlur(img, ksize=GAUSSIAN_KERNEL, sigmaX=GAUSSIAN_SIGMA)

        # 2) CLAHE on blurred image
        img_gauss_clahe = clahe.apply(img_blur)

        # 3) Save
        cv2.imwrite(dst_path, img_gauss_clahe)
        success += 1
    except Exception:
        failed.append(src_path)
        continue

# -------------------------------------------------------
# SUMMARY
# -------------------------------------------------------
print("\n========== GAUSS+CLAHE CREATION SUMMARY ==========")
print(f"Total images listed:    {len(df)}")
print(f"Successfully processed: {success}")
print(f"Missing images:         {len(missing)}")
print(f"Failed to process:      {len(failed)}")

if missing:
    print("\nExamples of missing images:")
    for x in missing[:5]:
        print("  ", x)

if failed:
    print("\nExamples of failed conversions:")
    for x in failed[:5]:
        print("  ", x)

print("\nDone! Gaussian+CLAHE-enhanced files saved to:")
print(GAUSS_CLAHE_ROOT)
