#!/usr/bin/env python
"""
This script will create a tiny CheXpert-style sanity subset from IU-CXR:

- At most 2 UIDs per CheXpert label (including "No Finding").
- For each selected UID, include ALL views (Frontal + Lateral).
- Exclude UIDs with tech-quality issues.
- Only include UIDs whose Problems text clearly contains a CheXpert label
  (substring match, case-insensitive). If no match → UID ignored.
- No complicated synonym mapping: if it's not close textually, we skip it.

This is meant for MedCLIP sanity runs, not for full benchmarking.
"""

import argparse
import os
from typing import List, Optional

import numpy as np
import pandas as pd


# Canonical CheXpert 14 observations (string match only)
CHEXPERT_LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

# Tokens in Problems that mean poor image quality → exclude UID
QUALITY_EXCLUDE_TOKENS = {
    "Technical Quality of Image Unsatisfactory",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create CheXpert14-style sanity subset (2 UIDs per label) "
            "from IU-CXR, keeping all views per UID."
        )
    )
    parser.add_argument(
        "--projections-csv",
        type=str,
        required=True,
        help="Path to indiana_projections.csv (uid, filename, projection, ...).",
    )
    parser.add_argument(
        "--reports-csv",
        type=str,
        required=True,
        help="Path to indiana_reports.csv (uid, Problems, ...).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        required=True,
        help="Path to output CSV for the sanity subset.",
    )
    parser.add_argument(
        "--images-root",
        type=str,
        default=None,
        help=(
            "Optional root dir for normalized images. "
            "If set, add `image_path = images_root / filename`."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    return parser.parse_args()


# ---------- Helpers ----------

def split_problems(problems: Optional[str]) -> List[str]:
    """Split Problems field into a list of non-empty, stripped tokens."""
    if not isinstance(problems, str):
        return []
    tokens = [t.strip() for t in problems.split(";")]
    return [t for t in tokens if t]


def has_quality_issue(problems_list: List[str]) -> bool:
    """Check if any token is a known technical-quality issue."""
    for t in problems_list:
        if t in QUALITY_EXCLUDE_TOKENS:
            return True
    return False


def match_chexpert_label(problems_list: List[str]) -> Optional[str]:
    """
    Find the *first* CheXpert label whose text appears as a substring
    in any Problems token (case-insensitive).

    Example:
        Problems: ["Pulmonary Atelectasis", "Scoliosis"]
        → matches "Atelectasis" (because "atelectasis" in "pulmonary atelectasis")

    If no CheXpert label matches → return None (UID excluded).
    """
    lower_tokens = [t.lower() for t in problems_list]

    for label in CHEXPERT_LABELS:
        key = label.lower()
        for tok in lower_tokens:
            if key in tok:
                return label
    return None


# ---------- Main ----------

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    print(f"[INFO] Loading projections from: {args.projections_csv}")
    df_proj = pd.read_csv(args.projections_csv)

    print(f"[INFO] Loading reports from:    {args.reports_csv}")
    df_rep = pd.read_csv(args.reports_csv)

    # Basic checks
    for col in ["uid", "filename", "projection"]:
        if col not in df_proj.columns:
            raise ValueError(f"Column `{col}` missing in projections CSV.")
    for col in ["uid", "Problems"]:
        if col not in df_rep.columns:
            raise ValueError(f"Column `{col}` missing in reports CSV.")

    # ---- 1) UID-level CheXpert label assignment ----
    print("[INFO] Computing Problems list per UID...")
    df_rep = df_rep.copy()
    df_rep["problems_list"] = df_rep["Problems"].apply(split_problems)
    df_rep["has_quality_issue"] = df_rep["problems_list"].apply(has_quality_issue)

    print("[INFO] Matching each UID to a CheXpert label (substring match)...")
    df_rep["chexpert_label"] = df_rep["problems_list"].apply(match_chexpert_label)

    # Filter out:
    #  - no CheXpert match
    #  - UIDs with tech-quality issues
    mask_valid = df_rep["chexpert_label"].notna() & (~df_rep["has_quality_issue"])
    df_uid = df_rep[mask_valid][["uid", "chexpert_label", "problems_list"]].copy()

    if df_uid.empty:
        raise RuntimeError(
            "No UIDs matched any CheXpert label after quality filtering. "
            "Check Problems text or CheXpert label strings."
        )

    print(f"[INFO] Valid UIDs with CheXpert match (after filtering): {len(df_uid)}")
    print("[INFO] UID count per CheXpert label (before sampling):")
    for label, n in df_uid["chexpert_label"].value_counts().items():
        print(f"    {label:30s} : {n:4d} UIDs")

    # ---- 2) Stratified sampling: up to 2 UIDs per CheXpert label ----
    sampled_uid_rows = []
    print("[INFO] Sampling up to 2 UIDs per CheXpert label...")

    for label in CHEXPERT_LABELS:
        df_c = df_uid[df_uid["chexpert_label"] == label]
        n_available = len(df_c)
        n_to_sample = min(2, n_available)

        if n_to_sample == 0:
            print(f"    {label:30s} : 0 UIDs (no match, skipped)")
            continue

        if n_to_sample == n_available:
            sampled = df_c
        else:
            sampled = df_c.sample(
                n=n_to_sample,
                replace=False,
                random_state=int(rng.integers(0, 2**32 - 1)),
            )

        sampled_uid_rows.append(sampled)
        print(f"    {label:30s} : {n_to_sample} UIDs selected (of {n_available})")

    if not sampled_uid_rows:
        raise RuntimeError("No UIDs selected after per-label sampling.")

    df_uid_sampled = pd.concat(sampled_uid_rows, axis=0).reset_index(drop=True)
    sampled_uids = set(df_uid_sampled["uid"].tolist())
    print(f"[INFO] Total sampled UIDs (across all labels): {len(sampled_uids)}")

    # ---- 3) Bring in ALL images (all views) for sampled UIDs ----
    print("[INFO] Collecting all projections (all views) for sampled UIDs...")
    df_proj_sel = df_proj[df_proj["uid"].isin(sampled_uids)].copy()
    print(f"[INFO] Image rows for sampled UIDs: {len(df_proj_sel)}")

    # Merge label info into image-level rows
    df_sanity = df_proj_sel.merge(
        df_uid_sampled[["uid", "chexpert_label", "problems_list"]],
        on="uid",
        how="left",
    )

    # Optional: add image_path
    if args.images_root:
        print(f"[INFO] Adding `image_path` with root: {args.images_root}")
        root = args.images_root
        df_sanity["image_path"] = df_sanity["filename"].apply(
            lambda fn: os.path.join(root, fn)
        )

    # Shuffle final rows
    df_sanity = df_sanity.sample(
        frac=1.0, replace=False, random_state=args.seed
    ).reset_index(drop=True)

    print(f"[INFO] Final sanity subset size (image rows): {len(df_sanity)}")
    print("[INFO] Final UID counts per CheXpert label:")
    for label, n in df_uid_sampled["chexpert_label"].value_counts().items():
        print(f"    {label:30s} : {n:4d} UIDs")

    # Save
    out_dir = os.path.dirname(args.output_csv)
    if out_dir and out_dir != "" and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Writing sanity CSV to: {args.output_csv}")
    df_sanity.to_csv(args.output_csv, index=False)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
