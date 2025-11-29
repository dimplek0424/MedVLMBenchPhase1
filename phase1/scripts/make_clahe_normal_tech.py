"""
Generate CLAHE images for IU-CXR normals / tech-quality-unsatisfactory cases
===========================================================================

Reads:
    D:\MedVLMBench\EDA\eda_reports\sanity_normal_or_tech_iucxr_v01.csv

and uses:
    images_normalized/   as input
    images_clahe_no_findings/      as CLAHE output for "normal" cases
    images_clahe_tech_unsat/       as CLAHE output for tech-unsat cases

The CSV is expected to have (possibly with old names):
    uid or UID
    filename or Image_ID
    projection or View
    is_normal_meta
    is_tq_unsat
"""

import os
import cv2
import pandas as pd
from tqdm import tqdm

# -----------------------
# PATHS (update if needed)
# -----------------------
REPO = r"D:\MedVLMBench"
DATA_DIR = os.path.join(REPO, "phase1", "data", "chestxray_iu")

CSV_PATH = r"D:\MedVLMBench\EDA\eda_reports\sanity_normal_or_tech_iucxr_v01.csv"

NORM_DIR = os.path.join(DATA_DIR, "images", "images_normalized")

OUT_DIR_NORMAL = os.path.join(DATA_DIR, "images", "images_clahe_no_findings")
OUT_DIR_TQ     = os.path.join(DATA_DIR, "images", "images_clahe_tech_unsat")

os.makedirs(OUT_DIR_NORMAL, exist_ok=True)
os.makedirs(OUT_DIR_TQ, exist_ok=True)

# Toggle this if you only care about normals for now
PROCESS_NORMALS = True
PROCESS_TQ      = True


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to: UID, Image_ID, View."""
    cols = df.columns

    if "UID" not in cols and "uid" in cols:
        df = df.rename(columns={"uid": "UID"})
    if "Image_ID" not in cols and "filename" in cols:
        df = df.rename(columns={"filename": "Image_ID"})
    if "View" not in cols and "projection" in cols:
        df = df.rename(columns={"projection": "View"})

    required = ["UID", "Image_ID"]
    for c in required:
        if c not in df.columns:
            raise ValueError(
                f"CSV must contain '{c}' column (after normalization). "
                f"Found columns: {df.columns.tolist()}"
            )

    if "is_normal_meta" not in df.columns:
        df["is_normal_meta"] = False
    if "is_tq_unsat" not in df.columns:
        df["is_tq_unsat"] = False

    return df


def make_clahe_for_files(filenames, out_root, label):
    """
    Apply CLAHE to all given filenames and save under out_root.

    - Reads input from NORM_DIR
    - Writes PNGs with same filename under out_root
    """
    print(f"\n[{label}] Generating CLAHE images into: {out_root}")
    print(f"[{label}] # unique images to process: {len(filenames)}")

    # Standard chest X-ray-ish CLAHE settings (same as pathology script)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    skipped_missing = 0
    skipped_readerr = 0
    already_exist   = 0
    processed       = 0

    for fname in tqdm(filenames, desc=f"CLAHE {label}"):
        in_path  = os.path.join(NORM_DIR, fname)
        out_path = os.path.join(out_root, fname)

        # If already exists, skip (idempotent)
        if os.path.exists(out_path):
            already_exist += 1
            continue

        if not os.path.exists(in_path):
            print(f"[WARN][{label}] Missing normalized image: {in_path}")
            skipped_missing += 1
            continue

        # Read as grayscale
        img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARN][{label}] Failed to read image: {in_path}")
            skipped_readerr += 1
            continue

        # Apply CLAHE
        clahe_img = clahe.apply(img)

        # Ensure output directory exists (nested dirs are unlikely, but safe)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        ok = cv2.imwrite(out_path, clahe_img)
        if not ok:
            print(f"[WARN][{label}] Failed to write CLAHE image: {out_path}")
            skipped_readerr += 1
            continue

        processed += 1

    print(f"\n[{label}] DONE.")
    print(f"  processed images      : {processed}")
    print(f"  already existed       : {already_exist}")
    print(f"  missing source files  : {skipped_missing}")
    print(f"  read/write errors     : {skipped_readerr}")


def main():
    print("Loading CSV:", CSV_PATH)
    df = pd.read_csv(CSV_PATH)
    df = standardize_columns(df)

    # ---- Normal-only ----
    if PROCESS_NORMALS:
        normal_mask = (df["is_normal_meta"] == True) & (df["is_tq_unsat"] == False)
        df_normal = df[normal_mask].copy()

        filenames_normal = sorted(df_normal["Image_ID"].astype(str).unique())
        make_clahe_for_files(
            filenames=filenames_normal,
            out_root=OUT_DIR_NORMAL,
            label="Normal-only (no findings)",
        )

    # ---- Technical Quality Unsatisfactory ----
    if PROCESS_TQ:
        tq_mask = (df["is_tq_unsat"] == True)
        df_tq = df[tq_mask].copy()

        filenames_tq = sorted(df_tq["Image_ID"].astype(str).unique())
        make_clahe_for_files(
            filenames=filenames_tq,
            out_root=OUT_DIR_TQ,
            label="Technical-quality-unsatisfactory",
        )


if __name__ == "__main__":
    main()
