"""
MedCLIP Zero-Shot Disease Classification on IU-CXR Pathology-Rich Sanity Subset

For each image in a sanity CSV (e.g.:
    D:\MedVLMBench\EDA\eda_reports\sanity_subset_iucxr_v02.csv
)
we:
  - Use MedCLIP PromptClassifier (CheXpert mode) to get zero-shot probabilities
    for 5 CheXpert competition diseases:
        Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion
  - Build 0/1 ground-truth labels from Pathology_Labels_14:
        <Disease>_label = 1 if that disease appears in the text, else 0
  - Save everything to a predictions CSV.

IMPORTANT:
  * This script does NOT filter images. Every row from the sanity CSV is included.
  * Now supports CLI arguments so we can reuse the same script for
    - normalized images
    - CLAHE images
    - Gauss+CLAHE images
"""

import os
import time
from typing import Dict, List

import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
import argparse

from medclip import (
    MedCLIPModel,
    MedCLIPVisionModelViT,
    MedCLIPProcessor,
    PromptClassifier,
)
from medclip.prompts import (
    generate_chexpert_class_prompts,
    process_class_prompts,
)

# ----------- DEFAULT PATHS (used if CLI args are not provided) -----------

DEFAULT_CSV_PATH = r"D:\MedVLMBench\EDA\eda_reports\sanity_subset_iucxr_v02.csv"
DEFAULT_IMG_ROOT = r"D:\MedVLMBench\phase1\data\chestxray_iu\images\images_normalized"
DEFAULT_OUT_CSV = r"D:\MedVLMBench\phase1\results\medclip_disease_sanity_pathology_preds.csv"

os.makedirs(os.path.dirname(DEFAULT_OUT_CSV), exist_ok=True)

# 5 CheXpert competition diseases
DISEASE_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]

# Map text patterns from Pathology_Labels_14 → disease label 1/0
LABEL_PATTERNS = {
    "Atelectasis": ["atelectasis"],
    "Cardiomegaly": ["cardiomegaly"],
    "Consolidation": ["consolidation"],
    "Edema": ["edema"],
    "Pleural Effusion": ["pleural_effusion", "pleural effusion"],
}


def load_image_pil(path: str):
    """Load an image from disk as RGB PIL.Image, or None on error."""
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception as e:
        print(f"[WARN] Failed to load image {path}: {e}")
        return None


def build_binary_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    From Pathology_Labels_14, build 0/1 columns:
        <Disease>_label
    Rule:
      - If ANY matching substring for that disease appears in the text,
        mark the label as 1, else 0.
    """
    if "Pathology_Labels_14" not in df.columns:
        df["Pathology_Labels_14"] = ""

    def normalize_label_str(s):
        if pd.isna(s):
            return ""
        s = str(s).lower()
        s = s.replace(" ", "_")
        return s

    norm_text = df["Pathology_Labels_14"].apply(normalize_label_str)

    for disease in DISEASE_CLASSES:
        patterns = LABEL_PATTERNS[disease]
        col_name = f"{disease}_label"

        def has_disease(t):
            return int(any(p in t for p in patterns))

        df[col_name] = norm_text.apply(has_disease)

    return df


def parse_args():
    """
    Simple CLI parser.

    If you run without arguments:
        python -m phase1.scripts.eval_medclip_disease_sanity

    it will fall back to the DEFAULT_* paths above (your current behavior).

    If you want to override:
        python -m phase1.scripts.eval_medclip_disease_sanity ^
          --image-csv EDA\\eda_reports\\sanity_subset_iucxr_v02.csv ^
          --image-root D:\\MedVLMBench\\phase1\\data\\chestxray_iu\\images\\images_clahe ^
          --output-preds phase1\\results\\medclip_disease_sanity_clahe_preds.csv ^
          --device cpu
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image-csv",
        type=str,
        default=DEFAULT_CSV_PATH,
        help=f"Path to sanity subset CSV (default: {DEFAULT_CSV_PATH})",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default=DEFAULT_IMG_ROOT,
        help=(
            "Root folder with CXR images for THIS run "
            f"(default: {DEFAULT_IMG_ROOT})"
        ),
    )
    parser.add_argument(
        "--output-preds",
        type=str,
        default=DEFAULT_OUT_CSV,
        help=f"Where to save the prediction CSV (default: {DEFAULT_OUT_CSV})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device string (e.g. 'cpu' or 'cuda:0'). Default: cpu",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=6,
        help="Batch size for image processing (default: 6).",
    )

    return parser.parse_args()


def main():
    # ---------------- 0. Parse CLI arguments ----------------
    args = parse_args()

    csv_path = args.image_csv
    img_root = args.image_root
    out_csv = args.output_preds
    device_str = args.device
    batch_size = args.batch_size

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    device = torch.device(device_str)
    print("Using device:", device)
    print("Image CSV:", csv_path)
    print("Image root:", img_root)
    print("Output preds CSV:", out_csv)
    print("Batch size:", batch_size)

    # ---------------- 1. Load MedCLIP model ----------------
    print("Loading MedCLIP model (ViT backbone)...")
    model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    model.from_pretrained()
    model.eval()
    model.to(device)

    processor = MedCLIPProcessor()

    # ---------------- 2. Build CheXpert prompts + PromptClassifier ----------------
    print("\nBuilding CheXpert prompts and PromptClassifier (zero-shot disease head)...")
    # authors' helper: build a huge set of synthetic CheXpert prompts
    cls_prompts = generate_chexpert_class_prompts(n=10)
    # tokenize to BERT inputs
    prompt_inputs = process_class_prompts(cls_prompts)

    # Move prompt tensors to device (CPU for your runs, but this keeps it explicit)
    prompt_inputs = {
        k: {kk: vv.to(device) for kk, vv in v.items()}
        for k, v in prompt_inputs.items()
    }

    # Authors' zero-shot head: PromptClassifier
    clf = PromptClassifier(model, ensemble=True)
    disease_classes: List[str] = list(cls_prompts.keys())
    print("CheXpert zero-shot classes:", disease_classes)

    # ---------------- 3. Load pathology-rich sanity subset ----------------
    print("\nLoading pathology-rich sanity subset CSV:")
    print(" ", csv_path)
    df = pd.read_csv(csv_path)

    # Normalize column names (match earlier view script)
    if "UID" not in df.columns and "uid" in df.columns:
        df = df.rename(columns={"uid": "UID"})
    if "Image_ID" not in df.columns and "filename" in df.columns:
        df = df.rename(columns={"filename": "Image_ID"})

    for col in ["UID", "Image_ID"]:
        if col not in df.columns:
            raise ValueError(
                f"CSV must contain column '{col}'. "
                f"Available columns: {df.columns.tolist()}"
            )

    # Ensure Pathology_Labels_14 exists and build 0/1 labels for each disease
    df = build_binary_labels(df)

    print(f"Total rows in sanity subset (no filtering): {len(df)}")

    # ---------------- 4. Run zero-shot PromptClassifier on ALL rows ----------------
    all_uids: List[str] = []
    all_image_ids: List[str] = []
    all_path_labels: List[str] = []
    all_infer_time_ms: List[float] = []

    # disease -> list of probabilities
    prob_dict: Dict[str, List[float]] = {cls: [] for cls in disease_classes}

    print("\nRunning MedCLIP PromptClassifier (zero-shot diseases) on all sanity images...\n")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Disease inference"):
        uid = row["UID"]
        fname = row["Image_ID"]
        path_labels = row.get("Pathology_Labels_14", "")

        img_path = os.path.join(img_root, fname)

        all_uids.append(uid)
        all_image_ids.append(fname)
        all_path_labels.append(path_labels)

        if not os.path.exists(img_path):
            print(f"[WARN] Image missing: {img_path}")
            for cls in disease_classes:
                prob_dict[cls].append(float("nan"))
            all_infer_time_ms.append(float("nan"))
            continue

        img = load_image_pil(img_path)
        if img is None:
            for cls in disease_classes:
                prob_dict[cls].append(float("nan"))
            all_infer_time_ms.append(float("nan"))
            continue

        try:
            img_inputs = processor(images=img, return_tensors="pt")
            # authors' PromptClassifier expects prompt_inputs alongside pixel_values
            img_inputs["prompt_inputs"] = prompt_inputs
            img_inputs["pixel_values"] = img_inputs["pixel_values"].to(device)

            start_t = time.time()
            with torch.no_grad():
                out = clf(**img_inputs)  # authors' call; returns logits + class_names
            elapsed_ms = (time.time() - start_t) * 1000.0
            all_infer_time_ms.append(elapsed_ms)

            logits = out["logits"].squeeze(0)     # [num_classes]
            probs = torch.sigmoid(logits).cpu().numpy().tolist()
            out_classes = out["class_names"]      # order of logits

            # map class -> prob for this image
            class_to_prob = {c: p for c, p in zip(out_classes, probs)}

            # store probs in consistent order (disease_classes)
            for cls in disease_classes:
                prob_dict[cls].append(float(class_to_prob.get(cls, float("nan"))))

        except Exception as e:
            print(f"[WARN] Error processing {img_path}: {e}")
            for cls in disease_classes:
                prob_dict[cls].append(float("nan"))
            all_infer_time_ms.append(float("nan"))

    # ---------------- 5. Build result DataFrame ----------------
    result_df = pd.DataFrame({
        "UID": all_uids,
        "Image_ID": all_image_ids,
        "Pathology_Labels_14": all_path_labels,
        "Inference_Time_ms": all_infer_time_ms,
    })

    # Attach the 5 label columns (0/1) from df using Image_ID as key
    label_cols = [f"{d}_label" for d in DISEASE_CLASSES]
    label_map = df.set_index("Image_ID")[label_cols]
    result_df = result_df.join(label_map, on="Image_ID")

    # Add probability columns
    for cls in disease_classes:
        col_name = f"{cls}_prob"
        result_df[col_name] = prob_dict[cls]

    # ---------------- 6. Save CSV ----------------
    result_df.to_csv(out_csv, index=False)
    print(f"\nDisease prediction CSV (all sanity images) saved to:\n  {out_csv}")
    print("\nPreview of result_df (first 5 rows):")
    print(result_df.head())


if __name__ == "__main__":
    main()
