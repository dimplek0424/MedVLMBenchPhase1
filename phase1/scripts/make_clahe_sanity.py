"""
Stage A: Generate CLAHE-enhanced images for the IU-CXR sanity subset.

This script:
  ✓ Loads the sanity subset CSV (created during EDA)
  ✓ Reads each normalized IU-CXR PNG image
  ✓ Applies OpenCV CLAHE (Contrast Limited Adaptive Histogram Equalization)
  ✓ Saves the enhanced image into a parallel folder
  ✓ Produces a 1:1 mapping of normalized → CLAHE images

Use this to compare:
  - MedCLIP performance on normalized images
  - MedCLIP performance on CLAHE images
"""

import os
import cv2
import pandas as pd
from tqdm import tqdm


# -------------------------------------------------------
# PATHS (edit here if your folder structure changes)
# -------------------------------------------------------

# CSV listing the selected "sanity subset" images created earlier
CSV_PATH = r"D:\MedVLMBench\EDA\eda_reports\sanity_subset_iucxr_v02.csv"

# Normalized IU-CXR images (previous preprocessing step)
NORMALIZED_ROOT = r"D:\MedVLMBench\phase1\data\chestxray_iu\images\images_normalized"

# Output folder — CLAHE-enhanced version of the same images
CLAHE_ROOT = r"D:\MedVLMBench\phase1\data\chestxray_iu\images\images_clahe"


# Ensure the output directory exists
os.makedirs(CLAHE_ROOT, exist_ok=True)


# -------------------------------------------------------
# LOAD CSV
# -------------------------------------------------------

print("Loading sanity subset CSV:", CSV_PATH)
df = pd.read_csv(CSV_PATH)

# The CSV must have column "Image_ID"
if "Image_ID" not in df.columns:
    raise ValueError("CSV must contain column: 'Image_ID'")

print(f"Found {len(df)} images in sanity subset.")


# -------------------------------------------------------
# CONFIGURE CLAHE
# -------------------------------------------------------
# clipLimit: contrast limit (higher = more contrast)
# tileGridSize: size of grid tiles for local histogram equalization
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


# -------------------------------------------------------
# PROCESS IMAGES
# -------------------------------------------------------

missing = []   # Images listed in CSV but not found in normalized folder
failed  = []   # Images that OpenCV failed to process
success = 0    # Successful CLAHE conversions

print("\nStarting CLAHE generation...\n")

for _, row in tqdm(df.iterrows(), total=len(df), desc="CLAHE Processing"):

    fname = row["Image_ID"]  # Example: "3806_IM-1916-2001.dcm.png"

    # Full path to existing normalized image
    src_path = os.path.join(NORMALIZED_ROOT, fname)

    # Full path for new CLAHE image
    dst_path = os.path.join(CLAHE_ROOT, fname)

    # 1. Check if the image file exists
    if not os.path.exists(src_path):
        missing.append(src_path)
        continue

    # 2. Read the image as grayscale
    #    IU-CXR normalized images should already be single-channel
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        # This means OpenCV failed to decode the PNG
        failed.append(src_path)
        continue

    # 3. Apply CLAHE enhancement
    try:
        img_clahe = clahe.apply(img)  # Enhanced grayscale
    except Exception:
        failed.append(src_path)
        continue

    # 4. Save enhanced image to output folder
    cv2.imwrite(dst_path, img_clahe)
    success += 1


# -------------------------------------------------------
# SUMMARY
# -------------------------------------------------------
print("\n========== CLAHE CREATION SUMMARY ==========")
print(f"Total images listed:   {len(df)}")
print(f"Successfully processed: {success}")
print(f"Missing images:        {len(missing)}")
print(f"Failed to process:     {len(failed)}")

if missing:
    print("\nExamples of missing images:")
    for x in missing[:5]:
        print("  ", x)

if failed:
    print("\nExamples of failed conversions:")
    for x in failed[:5]:
        print("  ", x)

print("\nDone! CLAHE-enhanced files saved to:")
print(CLAHE_ROOT)
