"""
IU-CXR sanity subset + integrity check (IU-aware, no normals, tech-unsat split)
-------------------------------------------------------------------------------

This script:

1. Loads IU-CXR reports + projections
2. Builds `caption` = indication + findings + impression
3. Separates and saves exams with:
     - "Technical Quality of Image Unsatisfactory" (MeSH or Problems)
     - "Normal" cases (MeSH or Problems)
4. Creates TWO main subsets:
     A) sanity_subset_iucxr_v02.csv
        - pathology-rich subset (no normals / no technical-unsatisfactory)
        - used for Stage-1 benchmarking in MedVLMBench

     B) sanity_normal_or_tech_iucxr_v01.csv
        - all exams where MeSH/Problems indicate "normal"
          OR "Technical Quality of Image Unsatisfactory"
        - ideal for probing model behaviour on clean normals & bad-quality exams

5. Checks image integrity:
     - missing files
     - orphan files
6. Builds 14 CheXpert-like pathology labels using:
     - MeSH + Problems + caption
   with conservative rules to avoid false positives (e.g., support_devices).

7. Samples UIDs (study-level), includes ALL images/views for each UID
   for the pathology-rich sanity subset.

Run from repo root:
    python -m phase1.scripts.select_iucxr_subset
"""

import os
from typing import Set
import pandas as pd
import numpy as np

# ================================
# CONFIG
# ================================
REPO = r"D:\MedVLMBench"
DATA_DIR = os.path.join(REPO, "phase1", "data", "chestxray_iu")
REPORTS_CSV = os.path.join(DATA_DIR, "indiana_reports.csv")
PROJ_CSV = os.path.join(DATA_DIR, "indiana_projections.csv")

# All normalized images live here
IMAGE_DIR = os.path.join(DATA_DIR, "images", "images_normalized")

# Output: EDA reports + subsets
OUT_DIR = os.path.join(REPO, "EDA", "eda_reports")
os.makedirs(OUT_DIR, exist_ok=True)

# Pathology-rich sanity subset (no normals / no tech-unsat)
OUT_SUBSET = os.path.join(OUT_DIR, "sanity_subset_iucxr_v02.csv")

# Integrity checks
OUT_MISSING = os.path.join(OUT_DIR, "iu_cxr_missing_files.csv")
OUT_ORPHAN = os.path.join(OUT_DIR, "iu_cxr_orphan_files.csv")
OUT_UNMATCHED = os.path.join(OUT_DIR, "iu_cxr_unmatched_csv_rows.csv")

# All technical-unsatisfactory rows (for transparency)
OUT_TQ_UNSAT = os.path.join(OUT_DIR, "iu_cxr_technical_unsatisfactory.csv")

# NEW: "Normal OR Technical Quality Unsatisfactory" subset
OUT_NORMAL_TECH = os.path.join(OUT_DIR, "sanity_normal_or_tech_iucxr_v01.csv")

np.random.seed(42)

# CheXpert-style 14-label set (binary indicators per label)
CHEXPERT_LABELS = [
    "no_finding",
    "enlarged_cardiomediastinum",
    "cardiomegaly",
    "lung_opacity",
    "lung_lesion",
    "edema",
    "consolidation",
    "pneumonia",
    "atelectasis",
    "pneumothorax",
    "pleural_effusion",
    "pleural_other",
    "fracture",
    "support_devices",
]


# ================================
# STEP 1 — LOAD & MERGE CSVs
# ================================
def load_iu_cxr() -> pd.DataFrame:
    """
    Load Indiana University (IU-CXR) reports + projections, merge on UID.

    Also:
      - builds free-text `caption` (indication + findings + impression)
      - computes basic text stats
      - flags:
          * is_tq_unsat   (technical quality unsatisfactory)
          * is_normal_meta (MeSH/Problems mention "normal")
      - saves:
          * OUT_TQ_UNSAT: all technical-quality-unsatisfactory rows
          * OUT_NORMAL_TECH: all "normal OR tech-unsat" rows
    """
    rep = pd.read_csv(REPORTS_CSV)
    proj = pd.read_csv(PROJ_CSV)

    if "uid" not in rep.columns or "uid" not in proj.columns:
        raise KeyError("Both CSVs must contain 'uid' column")

    # Merge report+projection tables on UID
    df = rep.merge(proj, on="uid", how="inner")

    # Build caption = indication + findings + impression
    df["indication"] = df["indication"].fillna("")
    df["findings"] = df["findings"].fillna("")
    df["impression"] = df["impression"].fillna("")

    df["caption"] = (
        df["indication"].astype(str).str.strip()
        + ". "
        + df["findings"].astype(str).str.strip()
        + ". "
        + df["impression"].astype(str).str.strip()
    )

    # Clean up "nan", duplicate dots, and extra whitespace
    df["caption"] = (
        df["caption"]
        .str.replace(r"\bnan\b", "", regex=True)
        .str.replace(r"\s+\.", ".", regex=True)
        .str.replace(r"\.\s*\.", ".", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip(". ")
    )

    # Drop rows with empty captions (no usable report text)
    df = df[df["caption"].str.len() > 0].copy()

    # Basic text stats for analysis/EDA
    df["report_word_count"] = df["caption"].str.split().apply(len)
    df["report_sentence_count"] = df["caption"].str.count(r"\.") + 1

    # Lowercase MeSH / Problems for downstream filtering
    mesh_lower = df["MeSH"].fillna("").str.lower()
    prob_lower = df["Problems"].fillna("").str.lower()

    # Rows with "Technical Quality of Image Unsatisfactory"
    is_tq_unsat = (
        mesh_lower.str.contains("technical quality of image unsatisfactory")
        | prob_lower.str.contains("technical quality of image unsatisfactory")
    )

    # Rows with MeSH or Problems explicitly marked as "normal"
    # (This is metadata-driven "normal", not free-text "normal chest" etc.)
    is_normal_meta = mesh_lower.str.contains("normal") | prob_lower.str.contains("normal")

    # Attach flags to df so other steps (sampling, labeling) can use them
    df["is_tq_unsat"] = is_tq_unsat
    df["is_normal_meta"] = is_normal_meta

    # Save all technical-unsatisfactory rows (for reference)
    df[is_tq_unsat].to_csv(OUT_TQ_UNSAT, index=False)

    # NEW: save "normal OR tech-unsat" rows as a separate subset
    norm_or_tq_mask = is_tq_unsat | is_normal_meta
    df_norm_tech = df[norm_or_tq_mask].copy()
    df_norm_tech.to_csv(OUT_NORMAL_TECH, index=False)
    print(f"[Subset] Normal/technical-unsatisfactory CSV -> {OUT_NORMAL_TECH}")
    print(f"         Rows in normal/tech subset: {len(df_norm_tech)}")

    # IMPORTANT:
    # We DO NOT drop these rows from `df` here.
    # The main pathology-rich subset will be constructed later by explicitly
    # excluding is_tq_unsat / is_normal_meta during UID sampling.

    return df


# ================================
# STEP 2 — PATH CHECKING
# ================================
def resolve_image_path(filename: str) -> str:
    """Map a filename to its full normalized image path."""
    return os.path.join(IMAGE_DIR, filename)


def check_image_integrity(df: pd.DataFrame):
    """
    Verify that every filename in the merged CSV has a corresponding file on disk,
    and report any orphan files in the folder that are not referenced by the CSV.
    """
    csv_files = set(df["filename"].astype(str))
    folder_files = set(os.listdir(IMAGE_DIR))

    missing = sorted(list(csv_files - folder_files))  # CSV references, file not found
    orphan = sorted(list(folder_files - csv_files))   # Files on disk but no CSV row

    unmatched_rows = df[df["filename"].isin(missing)].copy()

    pd.DataFrame({"missing_filename": missing}).to_csv(OUT_MISSING, index=False)
    pd.DataFrame({"orphan_file": orphan}).to_csv(OUT_ORPHAN, index=False)
    unmatched_rows.to_csv(OUT_UNMATCHED, index=False)

    print(f"[Integrity] Missing files: {len(missing)}")
    print(f"[Integrity] Orphan files: {len(orphan)}")
    print(f"[Integrity] CSV rows with missing images: {len(unmatched_rows)}")

    return missing


# ================================
# STEP 3 — IU-AWARE PATHOLOGY LABELS
# ================================
def add_iu_pathology_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pathology labels tuned for IU-CXR.

    Uses:
      - caption (indication + findings + impression)  -> free text
      - MeSH + Problems                               -> semi-structured codes

    Design goal:
      * Be expressive enough to approximate CheXpert labels.
      * Still conservative to avoid false positive labels.

    For each label, we:
      - define positive patterns
      - sometimes define explicit negative patterns (e.g., "no pneumothorax")
    """
    caption = df["caption"].fillna("").str.lower()
    mp = (df["MeSH"].fillna("") + ";" + df["Problems"].fillna("")).str.lower()

    def has_caption(pos_patterns, neg_patterns=None):
        """Heuristic match in free text with optional negative patterns."""
        if not pos_patterns:
            return pd.Series(False, index=caption.index)
        pos_pat = "|".join(pos_patterns)
        pos_m = caption.str.contains(pos_pat, regex=True)
        if neg_patterns:
            neg_pat = "|".join(neg_patterns)
            neg_m = caption.str.contains(neg_pat, regex=True)
            return pos_m & ~neg_m
        return pos_m

    def has_mp(patterns):
        """Heuristic match in MeSH/Problems fields."""
        if not patterns:
            return pd.Series(False, index=mp.index)
        pat = "|".join(patterns)
        return mp.str.contains(pat, regex=True)

    # 1. Enlarged cardiomediastinum / mediastinum widening
    df["enlarged_cardiomediastinum"] = has_mp(["mediastinum", "mediastinal widening"])

    # 2. Cardiomegaly / enlarged heart
    cardiop = has_mp(["cardiomegaly", "cardiac shadow/enlarged", "cardiac enlargement"])
    cardiop |= has_caption(
        ["cardiomegaly", "enlarged cardiac silhouette", "heart size is enlarged", "mildly enlarged heart"],
        ["no cardiomegaly", "heart size within normal limits", "cardiac silhouette is normal"],
    )
    df["cardiomegaly"] = cardiop

    # 3. Lung opacity / infiltrate / airspace disease
    opac = has_mp(["opacity", "airspace disease", "air space disease", "infiltrate"])
    opac |= has_caption(
        ["airspace disease", "parenchymal opacity", "infiltrate", "infiltrates", "focal opacity"],
        ["no focal opacity", "no focal infiltrate"],
    )
    df["lung_opacity"] = opac

    # 4. Lung lesion (nodule / mass)
    lesion = has_mp(["nodule", "mass", "lesion"])
    lesion |= has_caption(["pulmonary nodule", "lung nodule", "lung mass", "pulmonary mass"])
    df["lung_lesion"] = lesion

    # 5. Edema / congestion
    edema = has_mp(["pulmonary congestion"])
    edema |= has_caption(["pulmonary edema", "interstitial edema", "vascular congestion"])
    df["edema"] = edema

    # 6. Consolidation
    cons = has_mp(["airspace disease", "airspace opacity"])
    cons |= has_caption(
        ["consolidation", "airspace consolidation"],
        ["no consolidation", "no focal consolidation"],
    )
    df["consolidation"] = cons

    # 7. Pneumonia
    pneu = has_mp(["pneumonia"])
    pneu |= has_caption(
        ["pneumonia", "pneumonic process"],
        ["no evidence of pneumonia"],
    )
    df["pneumonia"] = pneu

    # 8. Atelectasis
    ate = has_mp(["atelectasis"])
    ate |= has_caption(
        ["atelectasis", "volume loss", "bibasilar atelectasis"],
        ["no atelectasis"],
    )
    df["atelectasis"] = ate

    # 9. Pneumothorax
    ptx = has_mp(["pneumothorax"])
    ptx |= has_caption(
        ["pneumothorax", "pleural air"],
        ["no pneumothorax"],
    )
    df["pneumothorax"] = ptx

    # 10. Pleural effusion
    plef = has_mp(["pleural effusion"])
    plef |= has_caption(
        ["pleural effusion", "effusion", "blunting of the costophrenic angle"],
        ["no pleural effusion", "no effusion", "no definite effusion"],
    )
    df["pleural_effusion"] = plef

    # 11. Other pleural disease
    pleo = has_mp(["pleural thickening", "pleural plaque", "pleural calcification"])
    pleo |= has_caption(
        ["pleural thickening", "pleural disease"],
        ["no pleural thickening"],
    )
    df["pleural_other"] = pleo

    # 12. Fracture
    frac = has_mp(["fracture"])
    frac |= has_caption(
        ["fracture"],
        ["no acute fracture", "no fracture"],
    )
    df["fracture"] = frac

    # 13. Support devices (tubes, lines, pacemaker, etc.)
    #
    # ⚠ To avoid false positives (like the uid=2894 case you observed),
    #    we take a conservative approach:
    #    - use ONLY MeSH/Problems codes;
    #    - DO NOT currently rely on caption-based keywords.
    #
    # You can always reintroduce caption-based patterns later with stricter rules.
    supp = has_mp([
        "tube, inserted",
        "catheters, indwelling",
        "pacemaker",
        "defibrillators, implantable",
        "prosthesis implantation",
    ])

    # Example (commented) if you later want *very strict* caption-based signals:
    # supp |= has_caption(
    #     [
    #         "endotracheal tube",
    #         "tracheostomy tube",
    #         "central venous catheter",
    #         "central line",
    #         "picc line",
    #         "pacemaker",
    #         "icd device",
    #         "defibrillator",
    #         "chest tube",
    #         "sternal wires",
    #     ],
    #     ["no tubes or lines", "no lines or tubes"],
    # )

    df["support_devices"] = supp

    # 14. No finding:
    #     - true ONLY if:
    #         * none of the other 13 labels fire
    #         * and caption looks clearly "normal"
    pathology_cols = [
        "enlarged_cardiomediastinum",
        "cardiomegaly",
        "lung_opacity",
        "lung_lesion",
        "edema",
        "consolidation",
        "pneumonia",
        "atelectasis",
        "pneumothorax",
        "pleural_effusion",
        "pleural_other",
        "fracture",
        "support_devices",
    ]
    any_path = df[pathology_cols].any(axis=1)

    normal_flag = caption.str.contains(
        "lungs are clear|no acute cardiopulmonary|no active cardiopulmonary|normal chest|normal study",
        regex=True,
    )

    df["no_finding"] = (~any_path) & normal_flag

    return df


# ================================
# STEP 4 — SAMPLE UIDs (pathology-rich subset)
# ================================
def sample_uids(df: pd.DataFrame, max_uids: int = 24) -> Set[str]:
    """
    Sample UIDs to get a pathology-diverse sanity subset.

    IMPORTANT:
    - We now KEEP normal / technical-unsatisfactory rows in `df`.
    - To build the pathology-rich subset for Stage-1, we must
      explicitly exclude these cases when sampling.
    """
    # Use only non-normal, non-technical-unsatisfactory rows for sampling
    if "is_tq_unsat" in df.columns and "is_normal_meta" in df.columns:
        base = df[~(df["is_tq_unsat"] | df["is_normal_meta"])].copy()
    else:
        # Fallback if flags are missing (should not happen if load_iu_cxr ran)
        base = df.copy()

    selected: Set[str] = set()

    # Heuristic: how many UIDs we try to grab for each pathology
    rules = {
        "cardiomegaly": 3,
        "pneumonia": 3,
        "pleural_effusion": 3,
        "atelectasis": 3,
        "consolidation": 2,
        "pneumothorax": 2,
        "edema": 2,
        "lung_opacity": 3,
        "lung_lesion": 2,
        "fracture": 2,
        "support_devices": 2,
    }

    # First pass: label-specific sampling
    for label, n in rules.items():
        if label not in base.columns:
            continue
        uids = base[base[label] == True]["uid"].unique()
        if len(uids) == 0:
            continue

        take = min(len(uids), n)
        chosen = np.random.choice(uids, size=take, replace=False)
        selected.update(chosen)

    # Second pass: fill remaining slots with any remaining UIDs (still from base)
    all_uids = base["uid"].unique()
    remaining_slots = max_uids - len(selected)
    if remaining_slots > 0:
        remaining = [u for u in all_uids if u not in selected]
        if len(remaining) > 0:
            take = min(len(remaining), remaining_slots)
            extra = np.random.choice(remaining, size=take, replace=False)
            selected.update(extra)

    return selected


# ================================
# STEP 5 — BUILD FINAL SUBSET
# ================================
def build_subset(df: pd.DataFrame, selected_uids: Set[str]) -> pd.DataFrame:
    """
    Build the final pathology-rich sanity subset (study-level).

    For each selected UID, include ALL images/views for that UID.
    Also aggregate pathology labels into a `Pathology_Labels_14` column
    with comma-separated label names.
    """
    sub = df[df["uid"].isin(selected_uids)].copy()

    def collect_labels(row):
        labs = []
        for L in CHEXPERT_LABELS:
            if L in row and bool(row[L]):
                labs.append(L)
        return ",".join(labs)

    sub["Pathology_Labels_14"] = sub.apply(collect_labels, axis=1)

    # Columns to keep in the subset CSV
    keep = [
        "uid",
        "filename",
        "projection",
        "MeSH",
        "Problems",
        "image",
        "indication",
        "comparison",
        "findings",
        "impression",
        "caption",
        "Pathology_Labels_14",
        "report_word_count",
        "report_sentence_count",
    ]

    keep = [c for c in keep if c in sub.columns]
    out = sub[keep].reset_index(drop=True)

    # Rename for convenience in notebooks and scripts
    out = out.rename(
        columns={
            "uid": "UID",
            "filename": "Image_ID",
            "projection": "View",
        }
    )

    return out


# ================================
# MAIN
# ================================
def main():
    print("Loading IU-CXR metadata...")
    df = load_iu_cxr()

    print("Running integrity check...")
    _ = check_image_integrity(df)

    print("Adding IU-CXR pathology labels (MeSH + Problems + caption)...")
    df = add_iu_pathology_labels(df)

    print("Sampling UIDs for pathology-rich sanity subset...")
    selected_uids = sample_uids(df, max_uids=24)
    print(f"Selected {len(selected_uids)} unique studies for pathology-rich subset")

    print("Building subset with full report fields...")
    final = build_subset(df, selected_uids)

    print(f"Saving pathology-rich sanity subset -> {OUT_SUBSET}")
    final.to_csv(OUT_SUBSET, index=False)

    print(f"Technical-unsatisfactory rows saved -> {OUT_TQ_UNSAT}")
    print(f"Normal/tech-unsatisfactory subset saved -> {OUT_NORMAL_TECH}")
    print("Done.")


if __name__ == "__main__":
    main()
