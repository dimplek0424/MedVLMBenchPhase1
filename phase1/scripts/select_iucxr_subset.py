"""
Select a diverse 20–40 image sanity subset from IU-CXR.

Usage (from repo root):
    python -m phase1.scripts.select_iucxr_subset

Assumptions:
- Kaggle IU-CXR CSVs live under: phase1/data/chestxray_iu/
    - indiana_reports.csv
    - indiana_projections.csv
- Images live under: phase1/data/chestxray_iu/images/  (or images-small/)
  and filenames in the CSV can be joined with that folder.

Edit IMAGE_ROOT / CSV paths below if your layout is different.
"""

import os
import re
import random
from typing import List, Dict

import numpy as np
import pandas as pd
from PIL import Image

# --------- CONFIG: EDIT IF NEEDED -----------------------------------------

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_ROOT = os.path.join(REPO_ROOT, "phase1", "data", "chestxray_iu")

REPORTS_CSV = os.path.join(DATA_ROOT, "indiana_reports.csv")
PROJ_CSV = os.path.join(DATA_ROOT, "indiana_projections.csv")

# set this to "images-small" or whatever your folder is actually called
IMAGE_ROOT = os.path.join(DATA_ROOT, "images")

OUTPUT_CSV = os.path.join(
    REPO_ROOT, "EDA", "eda_reports", "sanity_subset_iucxr_v01.csv"
)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Pathology categories we care about for the subset
PATHOLOGY_CATEGORIES = [
    "no_finding",
    "cardiomegaly",
    "effusion",
    "consolidation_pneumonia",
    "atelectasis",
    "infiltration_opacity",
    "multi_label",
]

# --------- 1. LOAD & MERGE METADATA ----------------------------------------


def load_and_merge_metadata() -> pd.DataFrame:
    """Load IU-CXR CSVs, merge them on 'uid', and create a 'caption' column."""
    if not os.path.exists(REPORTS_CSV):
        raise FileNotFoundError(f"Reports CSV not found: {REPORTS_CSV}")
    if not os.path.exists(PROJ_CSV):
        raise FileNotFoundError(f"Projections CSV not found: {PROJ_CSV}")

    reports = pd.read_csv(REPORTS_CSV)
    proj = pd.read_csv(PROJ_CSV)

    # Expect a common 'uid' column
    if "uid" not in reports.columns or "uid" not in proj.columns:
        raise KeyError("Expected 'uid' column in both CSVs")

    df = reports.merge(proj, on="uid", how="left", suffixes=("_rep", "_proj"))

    # Standardize some expected column names (adjust if your CSV differs)
    col_map = {}
    if "filename" in df.columns:
        col_map["filename"] = "image_id"
    if "projection" in df.columns:
        col_map["projection"] = "view"
    if "sex" in df.columns:
        col_map["sex"] = "gender"

    if col_map:
        df = df.rename(columns=col_map)

    # Combine findings + impression into caption
    for c in ["findings", "impression"]:
        if c not in df.columns:
            df[c] = ""

    df["findings"] = df["findings"].fillna("")
    df["impression"] = df["impression"].fillna("")

    df["caption_raw"] = (
        df["findings"].astype(str).str.strip()
        + ". "
        + df["impression"].astype(str).str.strip()
    )

    # Clean caption: remove 'nan', multiple spaces, etc.
    df["caption"] = (
        df["caption_raw"]
        .str.replace(r"\bnan\b", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # drop rows with completely empty captions
    df = df[df["caption"].str.len() > 0].copy()

    # Short helpers
    df["report_word_count"] = df["caption"].str.split().apply(len)
    df["report_sentences"] = df["caption"].str.count(r"\.") + 1

    return df


# --------- 2. SIMPLE PATHOLOGY TAGGING FROM TEXT --------------------------


def add_pathology_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple rule-based pathology flags from caption text.

    NOTE: This is intentionally simple + noisy.
    It is meant to help pick diverse examples, not serve as gold labels.
    """
    text = df["caption"].str.lower().fillna("")

    def contains_any(patterns: List[str]) -> pd.Series:
        regex = "|".join(patterns)
        return text.str.contains(regex, regex=True)

    df["no_finding"] = contains_any(
        [
            r"\bno acute cardiopulmonary disease\b",
            r"\bno acute disease\b",
            r"\bno active disease\b",
            r"\bno significant abnormalit(y|ies)\b",
            r"\bnormal chest\b",
            r"\bnormal study\b",
        ]
    )

    df["cardiomegaly"] = contains_any(
        [
            r"\bcardiomegaly\b",
            r"\benlarged cardiac silhouette\b",
            r"\bcardiac enlargement\b",
        ]
    )

    df["effusion"] = contains_any(
        [
            r"\bpleural effusion(s)?\b",
            r"\beffusion\b",
        ]
    )

    df["consolidation_pneumonia"] = contains_any(
        [
            r"\bconsolidation\b",
            r"\bpneumonia\b",
            r"\binfiltrate\b",
        ]
    )

    df["atelectasis"] = contains_any(
        [
            r"\batelectasis\b",
        ]
    )

    df["infiltration_opacity"] = contains_any(
        [
            r"\binfiltration\b",
            r"\bopacity\b",
            r"\bopacities\b",
        ]
    )

    # count how many of the above are true
    label_cols = [
        "cardiomegaly",
        "effusion",
        "consolidation_pneumonia",
        "atelectasis",
        "infiltration_opacity",
    ]

    df["num_positive_labels"] = df[label_cols].sum(axis=1)
    df["multi_label"] = df["num_positive_labels"] >= 2

    return df


# --------- 3. IMAGE QUALITY METRICS ---------------------------------------


def image_path_from_row(row: pd.Series) -> str:
    """
    Construct image path from row.

    Adjust this if your filename column is named differently or has extension.
    """
    if "image_id" in row:
        fname = row["image_id"]
    elif "filename" in row:
        fname = row["filename"]
    else:
        raise KeyError("Need 'image_id' or 'filename' column to locate images")

    # Some CSVs contain just the base name; append extension if needed.
    # Try as-is; if not found, try adding '.png' and '.jpg'.
    candidate_paths = [
        os.path.join(IMAGE_ROOT, str(fname)),
        os.path.join(IMAGE_ROOT, str(fname) + ".png"),
        os.path.join(IMAGE_ROOT, str(fname) + ".jpg"),
        os.path.join(IMAGE_ROOT, str(fname) + ".jpeg"),
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            return p
    return candidate_paths[0]  # fall back; may be missing on disk


def compute_image_stats(df: pd.DataFrame, max_images: int = 2000) -> pd.DataFrame:
    """
    Compute mean and std intensity for (up to) max_images rows.
    For speed, we only compute for a subset then use them when sampling.
    """
    df = df.copy()
    df["image_path"] = df.apply(image_path_from_row, axis=1)

    # Only sample up to 'max_images' rows for stats to keep it light
    if len(df) > max_images:
        df_sample = df.sample(n=max_images, random_state=RANDOM_SEED)
    else:
        df_sample = df

    mean_map: Dict[str, float] = {}
    std_map: Dict[str, float] = {}

    for idx, row in df_sample.iterrows():
        path = row["image_path"]
        if not os.path.exists(path):
            continue
        try:
            img = Image.open(path).convert("L")  # grayscale
            arr = np.array(img, dtype=np.float32) / 255.0
            mean_map[idx] = float(arr.mean())
            std_map[idx] = float(arr.std())
        except Exception:
            continue

    df["mean_intensity"] = df.index.map(mean_map)
    df["std_intensity"] = df.index.map(std_map)

    return df


# --------- 4. SAMPLING LOGIC ----------------------------------------------


def sample_by_label(df: pd.DataFrame, label: str, n: int) -> pd.DataFrame:
    """Sample up to n rows where df[label] is True."""
    if label not in df.columns:
        return df.iloc[0:0].copy()

    candidates = df[df[label]]

    # Prefer PA/AP views if 'view' column exists
    if "view" in candidates.columns:
        candidates = candidates[candidates["view"].isin(["PA", "AP", "Frontal"])]

    if len(candidates) == 0:
        return candidates

    if len(candidates) <= n:
        return candidates

    return candidates.sample(n=n, random_state=RANDOM_SEED)


def sample_technical_cases(df: pd.DataFrame, n_each: int = 2) -> pd.DataFrame:
    """
    Pick under-exposed, over-exposed, and low-contrast examples
    using mean/std intensity.
    """
    df_valid = df.dropna(subset=["mean_intensity", "std_intensity"])

    # Quantile thresholds
    low_brightness_thr = df_valid["mean_intensity"].quantile(0.1)
    high_brightness_thr = df_valid["mean_intensity"].quantile(0.9)
    low_contrast_thr = df_valid["std_intensity"].quantile(0.1)

    under = df_valid[df_valid["mean_intensity"] <= low_brightness_thr]
    over = df_valid[df_valid["mean_intensity"] >= high_brightness_thr]
    lowc = df_valid[df_valid["std_intensity"] <= low_contrast_thr]

    def pick(df_group: pd.DataFrame, n: int) -> pd.DataFrame:
        if len(df_group) <= n:
            return df_group
        return df_group.sample(n=n, random_state=RANDOM_SEED)

    dfs = [
        pick(under, n_each).assign(
            tech_reason="under_exposed"
        ),
        pick(over, n_each).assign(
            tech_reason="over_exposed"
        ),
        pick(lowc, n_each).assign(
            tech_reason="low_contrast"
        ),
    ]
    out = pd.concat(dfs, axis=0).drop_duplicates()
    return out


def build_sanity_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Combine pathology-based and technical sampling to get ~30 examples."""
    pieces = []

    # Pathology-based sampling: aim for 2–3 per category
    label_to_n = {
        "no_finding": 4,
        "cardiomegaly": 3,
        "effusion": 3,
        "consolidation_pneumonia": 3,
        "atelectasis": 3,
        "infiltration_opacity": 3,
        "multi_label": 4,
    }

    for label, n in label_to_n.items():
        sub = sample_by_label(df, label, n)
        if len(sub) > 0:
            sub = sub.copy()
            sub["selection_reason"] = f"pathology:{label}"
            pieces.append(sub)

    # Technical variation
    tech = sample_technical_cases(df, n_each=2)
    if len(tech) > 0:
        tech = tech.copy()
        tech["selection_reason"] = tech.get("selection_reason", "") + ";technical"

        pieces.append(tech)

    subset = pd.concat(pieces, axis=0).drop_duplicates()

    # If we ended up with > 40, subsample; if < 20, we just keep what we have
    if len(subset) > 40:
        subset = subset.sample(n=40, random_state=RANDOM_SEED)

    # Create clean columns for output table
    subset = subset.copy()

    # Basic demographics
    age_col = "age" if "age" in subset.columns else None
    gender_col = "gender" if "gender" in subset.columns else None

    out = pd.DataFrame()
    out["Image_ID"] = subset.get("image_id", subset.get("filename"))
    if "view" in subset.columns:
        out["View"] = subset["view"]
    else:
        out["View"] = ""

    if age_col:
        out["Age"] = subset[age_col]
    else:
        out["Age"] = np.nan

    if gender_col:
        out["Sex"] = subset[gender_col]
    else:
        out["Sex"] = ""

    # Pathology labels as a comma-separated list
    def row_labels(row) -> str:
        labels = []
        for lab in PATHOLOGY_CATEGORIES:
            if lab in row and bool(row[lab]):
                labels.append(lab)
        return ",".join(labels)

    out["Pathology_Labels"] = subset.apply(row_labels, axis=1)

    out["Report_Length_Words"] = subset["report_word_count"]
    out["Report_Length_Sentences"] = subset["report_sentences"]

    # Shortened findings/impression (optional)
    out["Findings_Short"] = subset["findings"].astype(str).str.slice(0, 200)
    out["Impression_Short"] = subset["impression"].astype(str).str.slice(0, 200)

    out["Mean_Intensity"] = subset["mean_intensity"]
    out["Std_Intensity"] = subset["std_intensity"]

    # Reason / notes
    out["Reason_Selected"] = subset["selection_reason"].fillna("").astype(str)
    if "tech_reason" in subset.columns:
        out["Reason_Selected"] = out["Reason_Selected"] + ";" + subset["tech_reason"].fillna("")

    return out.reset_index(drop=True)


# --------- 5. MAIN ---------------------------------------------------------


def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    print("Loading & merging IU-CXR metadata...")
    df = load_and_merge_metadata()
    print(f"  -> {len(df)} rows after caption cleaning")

    print("Adding simple pathology labels...")
    df = add_pathology_labels(df)

    print("Computing image intensity statistics (subset)...")
    df = compute_image_stats(df, max_images=2000)

    print("Building sanity subset...")
    subset = build_sanity_subset(df)
    print(f"  -> selected {len(subset)} images")

    print(f"Saving subset table to: {OUTPUT_CSV}")
    subset.to_csv(OUTPUT_CSV, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
