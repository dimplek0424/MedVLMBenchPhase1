"""
MedCLIP Zero-Shot CheXpert Disease Classification on an IU-CXR Subset

This script is designed to work with ANY IU-CXR subset CSV that has:
  - a column 'chexpert_label'  (one CheXpert observation per row)
  - and either:
        'image_path'
    OR  'Image_ID' / 'filename'  +  --image-root

It:
  ✓ Loads the subset CSV (RAW / CLAHE / Gauss+CLAHE – depends on image folder)
  ✓ Builds 0/1 ground-truth labels from `chexpert_label` for each CheXpert
    disease used by the official MedCLIP PromptClassifier
  ✓ Runs MedCLIP PromptClassifier (authors' zero-shot CheXpert head) on
    EVERY row in the CSV
  ✓ Adds, per row:
        <Disease>_label  (0/1, from chexpert_label)
        <Disease>_prob   (MedCLIP probability)
        Inference_Time_ms
  ✓ Saves an enriched CSV with all original columns + predictions

Notes:
  * We do NOT perform any resizing / normalization here.
    Images are treated as RAW (or RAW+preprocessed, e.g. CLAHE) on disk.
    MedCLIP's own processor takes care of resizing / normalization internally.
  * This script is agnostic to RAW vs CLAHE vs Gauss+CLAHE. That is entirely
    controlled by which image folder you point --image-root to (or by an
    `image_path` column inside the CSV).

Example usages (PowerShell):

  # RAW images
  python -m phase1.scripts.eval_medclip_disease_sanity_chexpert_subset `
    --image-csv    phase1\data\chestxray_iu\iu_sanity_uid_chexpert2perlabel_v01.csv `
    --image-root   D:\MedVLMBench\phase1\data\chestxray_iu\images\images_normalized `
    --output-preds phase1\results\medclip_chexpert_sanity_raw_preds.csv `
    --device cpu

  # CLAHE images
  python -m phase1.scripts.eval_medclip_disease_sanity_chexpert_subset `
    --image-csv    phase1\data\chestxray_iu\iu_sanity_uid_chexpert2perlabel_v01.csv `
    --image-root   D:\MedVLMBench\phase1\data\chestxray_iu\images\images_clahe `
    --output-preds phase1\results\medclip_chexpert_sanity_clahe_preds.csv `
    --device cpu

  # Gauss+CLAHE images
  python -m phase1.scripts.eval_medclip_disease_sanity_chexpert_subset `
    --image-csv    phase1\data\chestxray_iu\iu_sanity_uid_chexpert2perlabel_v01.csv `
    --image-root   D:\MedVLMBench\phase1\data\chestxray_iu\images\images_gauss_clahe `
    --output-preds phase1\results\medclip_chexpert_sanity_gauss_clahe_preds.csv `
    --device cpu
"""

import os
import time
import argparse
from typing import Dict, List

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm

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


def load_image_pil(path: str):
    """Load an image from disk as RGB PIL.Image, or None on error."""
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception as e:
        print(f"[WARN] Failed to load image {path}: {e}")
        return None


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image-csv",
        type=str,
        required=True,
        help="Path to subset CSV (must contain 'chexpert_label' and either "
             "'image_path' OR ['Image_ID' or 'filename'] + --image-root).",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default=None,
        help=(
            "Root folder with CXR images (used only if 'image_path' is NOT "
            "present in the CSV)."
        ),
    )
    parser.add_argument(
        "--output-preds",
        type=str,
        required=True,
        help="Where to save the enriched prediction CSV.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device string (e.g. 'cpu' or 'cuda:0'). Default: cpu.",
    )

    return parser.parse_args()


def main():
    # ---------------- 0. Parse CLI arguments ----------------
    args = parse_args()

    csv_path = args.image_csv
    img_root = args.image_root
    out_csv = args.output_preds
    device_str = args.device

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    device = torch.device(device_str)
    print("Using device:", device)
    print("Image CSV:", csv_path)
    print("Image root (if used):", img_root)
    print("Output preds CSV:", out_csv)

    # ---------------- 1. Load MedCLIP model ----------------
    print("\n[INFO] Loading MedCLIP model (ViT backbone)...")
    model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    model.from_pretrained()
    model.eval()
    model.to(device)

    processor = MedCLIPProcessor()

    # ---------------- 2. Build CheXpert prompts + PromptClassifier ----------------
    print("\n[INFO] Building CheXpert prompts and PromptClassifier (authors' zero-shot head)...")

    # Authors' helper: generate synthetic prompts per CheXpert disease class
    cls_prompts = generate_chexpert_class_prompts(n=10)
    # Tokenize to BERT inputs
    prompt_inputs = process_class_prompts(cls_prompts)

    # Move prompt tensors to device
    prompt_inputs = {
        k: {kk: vv.to(device) for kk, vv in v.items()}
        for k, v in prompt_inputs.items()
    }

    # Authors' zero-shot classifier
    clf = PromptClassifier(model, ensemble=True)

    # Class names exactly as returned by authors' prompt generator
    disease_classes: List[str] = list(cls_prompts.keys())
    print("[INFO] CheXpert zero-shot classes from authors' prompts:")
    for c in disease_classes:
        print("   -", c)

    # ---------------- 3. Load subset CSV ----------------
    print("\n[INFO] Loading subset CSV:")
    print("   ", csv_path)
    df = pd.read_csv(csv_path)

    if "chexpert_label" not in df.columns:
        raise ValueError(
            "Expected column 'chexpert_label' in the CSV. "
            f"Found columns: {df.columns.tolist()}"
        )

    has_image_path = "image_path" in df.columns

    # If there's no image_path, we need an ID column + image_root
    id_col = None
    if not has_image_path:
        if "Image_ID" in df.columns:
            id_col = "Image_ID"
        elif "filename" in df.columns:
            id_col = "filename"
        else:
            raise ValueError(
                "CSV must contain 'image_path' OR one of ['Image_ID', 'filename'] "
                "to resolve image paths.\n"
                f"Found columns: {df.columns.tolist()}"
            )
        if img_root is None:
            raise ValueError(
                "You must provide --image-root when the CSV does not have 'image_path'."
            )

    print(f"[INFO] Total rows in CSV (images): {len(df)}")
    print("[INFO] chexpert_label counts (all rows):")
    print(df["chexpert_label"].value_counts())

    # ---------------- 4. Build 0/1 labels from chexpert_label ----------------
    # For each CheXpert disease class, create a column:
    #   <Disease>_label = 1 if chexpert_label == Disease else 0
    # If a chexpert_label is not among disease_classes, it will simply have all-zeros.
    print("\n[INFO] Building 0/1 ground-truth labels from 'chexpert_label' for ALL rows...")
    for cls in disease_classes:
        col_name = f"{cls}_label"
        df[col_name] = (df["chexpert_label"] == cls).astype(int)

    # ---------------- 5. Run zero-shot PromptClassifier on ALL rows ----------------
    prob_dict: Dict[str, List[float]] = {cls: [] for cls in disease_classes}
    all_infer_time_ms: List[float] = []

    print("\n[INFO] Running MedCLIP PromptClassifier on ALL images in the subset...\n")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Disease inference"):
        # Resolve image path
        if has_image_path:
            img_path = row["image_path"]
        else:
            fname = row[id_col]
            img_path = os.path.join(img_root, fname)

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
            img_inputs["prompt_inputs"] = prompt_inputs
            img_inputs["pixel_values"] = img_inputs["pixel_values"].to(device)

            start_t = time.time()
            with torch.no_grad():
                out = clf(**img_inputs)  # authors' call; returns logits + class_names
            elapsed_ms = (time.time() - start_t) * 1000.0
            all_infer_time_ms.append(elapsed_ms)

            logits = out["logits"].squeeze(0)  # [num_classes]
            probs = torch.sigmoid(logits).cpu().numpy().tolist()
            out_classes = out["class_names"]

            class_to_prob = {c: p for c, p in zip(out_classes, probs)}
            for cls in disease_classes:
                prob_dict[cls].append(float(class_to_prob.get(cls, float("nan"))))

        except Exception as e:
            print(f"[WARN] Error processing {img_path}: {e}")
            for cls in disease_classes:
                prob_dict[cls].append(float("nan"))
            all_infer_time_ms.append(float("nan"))

    # ---------------- 6. Attach predictions to DataFrame ----------------
    print("\n[INFO] Attaching probabilities and inference times to DataFrame...")

    df["Inference_Time_ms"] = all_infer_time_ms
    for cls in disease_classes:
        col_name = f"{cls}_prob"
        df[col_name] = prob_dict[cls]

    # ---------------- 7. Save CSV ----------------
    df.to_csv(out_csv, index=False)
    print(f"\n[INFO] Disease prediction CSV (all rows) saved to:\n  {out_csv}")
    print("\n[INFO] Preview of enriched DataFrame (first 5 rows):")
    print(df.head())


if __name__ == "__main__":
    main()
