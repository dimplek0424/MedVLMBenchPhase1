"""
Stage 1: MedCLIP View Classification on IU-CXR Subsets
======================================================

Goal
----
Evaluate MedCLIP (ViT backbone) as a zero-shot classifier for
"Frontal vs Lateral" view prediction on three IU-CXR subsets:

  A) Pathology-rich sanity subset
     - CSV: sanity_subset_iucxr_v02.csv
     - No MeSH/Problems "normal"
     - No "Technical Quality of Image Unsatisfactory"
     - Used for core Stage-1 benchmarking

  B) Normal-only subset
     - CSV: sanity_normal_or_tech_iucxr_v01.csv
     - Filter: is_normal_meta == True and is_tq_unsat == False
     - Used to see how MedCLIP behaves on clean normals

  C) Technical Quality Unsatisfactory subset
     - CSV: sanity_normal_or_tech_iucxr_v01.csv
     - Filter: is_tq_unsat == True
     - Used to see how MedCLIP behaves on bad/unsatisfactory images

For each subset we:

- Load subset metadata (CSV)
- Filter to 'Frontal' / 'Lateral' views
- Run MedCLIP with two prompts:
      "a frontal chest x-ray radiograph"
      "a lateral chest x-ray radiograph"
- For each image, evaluate two variants:
      1) Normalized PNG from Kaggle IU-CXR
      2) CLAHE-enhanced version (generated separately)

We compare:
  * Accuracy (normalized vs CLAHE)
  * Confusion matrices (normalized & CLAHE)
  * Per-image inference times

We generate for each subset:
  1) A CSV with detailed per-image results
  2) A multi-page PDF report with summary, matrices, timing plots, and a table.

Why PIL instead of OpenCV here?
-------------------------------
For evaluation we only need to read the already-saved PNG
and feed it into MedCLIP's `MedCLIPProcessor`, which expects a
PIL Image. Using PIL keeps the pipeline consistent with
the original MedCLIP examples and avoids extra cv2 dependencies.

Inputs
------
- Pathology subset:
    CSV_PATH_PATHO:
        D:\\MedVLMBench\\EDA\\eda_reports\\sanity_subset_iucxr_v02.csv
    Columns (min):
        UID
        Image_ID
        View           ('Frontal' / 'Lateral')
        Pathology_Labels_14  (optional, used for context)

- Normal + Tech-quality subset:
    CSV_PATH_NORMAL_TECH:
        D:\\MedVLMBench\\EDA\\eda_reports\\sanity_normal_or_tech_iucxr_v01.csv
    Columns (min):
        uid / UID
        filename / Image_ID
        projection / View
        is_normal_meta
        is_tq_unsat

- Image roots:
    NORMALIZED_ROOT:
        D:\\MedVLMBench\\phase1\\data\\chestxray_iu\\images\\images_normalized
    CLAHE_ROOT:
        D:\\MedVLMBench\\phase1\\data\\chestxray_iu\\images\\images_clahe

Outputs
-------
- Per-subset CSVs (under phase1/results/):

    medclip_view_sanity_pathology_predictions.csv
    medclip_view_sanity_normal_predictions.csv
    medclip_view_sanity_tech_unsat_predictions.csv

    Columns:
        UID
        Image_ID
        GT_View
        Normal_Pred
        CLAHE_Pred
        Normal_Inference_Time_ms
        CLAHE_Inference_Time_ms
        Pathology_Labels_14   (if available; else empty)
        is_normal_meta        (if available; else False)
        is_tq_unsat           (if available; else False)
        Normal_Correct
        CLAHE_Correct

- Per-subset PDFs (under phase1/results/):

    medclip_view_sanity_pathology_report.pdf
    medclip_view_sanity_normal_report.pdf
    medclip_view_sanity_tech_unsat_report.pdf

    For each subset, pages:
        1. Text summary (accuracy, misclass, timing, explanation)
        2. Confusion matrix – Normalized images
        3. Confusion matrix – CLAHE images
        4. Accuracy bar chart (Normalized vs CLAHE)
        5. Histogram of per-image inference times
        6. Boxplot of per-image inference times
        7+. Paginated per-image result table (with wrapped text)

Notes
-----
- Everything runs on CPU (no CUDA required).
- We use the CPU/GPU-agnostic MedCLIP model code, but eval itself
  is done on `device = torch.device("cpu")` for reproducibility.
- We bypass the original CUDA-only forward and directly call:
      - `model.text_model(...)` for text embeddings
      - `model.vision_model(...)` for image embeddings
"""

import os
import time

import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image

import argparse
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor


# -----------------------
# PATHS (update if needed)
# -----------------------

# Pathology-rich sanity subset
CSV_PATH_PATHO = r"D:\MedVLMBench\EDA\eda_reports\sanity_subset_iucxr_v02.csv"

# Normal + Technical Quality Unsatisfactory combined subset
CSV_PATH_NORMAL_TECH = r"D:\MedVLMBench\EDA\eda_reports\sanity_normal_or_tech_iucxr_v01.csv"

NORMALIZED_ROOT = r"D:\MedVLMBench\phase1\data\chestxray_iu\images\images_normalized"
CLAHE_ROOT_PATHO = r"D:\MedVLMBench\phase1\data\chestxray_iu\images\images_clahe"
CLAHE_ROOT_NORMAL = r"D:\MedVLMBench\phase1\data\chestxray_iu\images\images_clahe_no_findings"
CLAHE_ROOT_TQ = r"D:\MedVLMBench\phase1\data\chestxray_iu\images\images_clahe_tech_unsat"

RESULTS_DIR = r"D:\MedVLMBench\phase1\results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_image_pil(path: str):
    """
    Load an image from disk using PIL and convert to RGB.

    Returns:
        PIL.Image.Image in RGB, or None if loading fails.
    """
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception as e:
        print(f"[WARN] Failed to load image {path}: {e}")
        return None


def wrap_text(text, width=25):
    """
    Simple text wrapper for table cells.

    - Converts non-string values to string.
    - Inserts line breaks every `width` characters.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text)
    if len(s) <= width:
        return s
    lines = []
    for i in range(0, len(s), width):
        lines.append(s[i:i + width])
    return "\n".join(lines)


def evaluate_subset(
    df: pd.DataFrame,
    subset_name: str,
    out_csv_path: str,
    out_pdf_path: str,
    model: MedCLIPModel,
    processor: MedCLIPProcessor,
    text_embeds: torch.Tensor,
    normalized_root: str,
    clahe_root: str,
):
    """
    Core evaluation routine for a single subset.

    Args:
        df: DataFrame with at least columns:
             - UID
             - Image_ID
             - View ('Frontal' or 'Lateral')
           Optionally:
             - Pathology_Labels_14
             - is_normal_meta
             - is_tq_unsat
        subset_name: label used for titles in the PDF.
        out_csv_path: where to save per-image CSV.
        out_pdf_path: where to save PDF report.
        model: MedCLIPModel (ViT).
        processor: MedCLIPProcessor.
        text_embeds: tensor [2, d] for "frontal" and "lateral" prompts.
    """
    print(f"\n================ Evaluating subset: {subset_name} ================")

    # Ensure expected column names
    # (some CSVs might use 'uid'/'filename'/'projection' instead of UID/Image_ID/View)
    if "UID" not in df.columns and "uid" in df.columns:
        df = df.rename(columns={"uid": "UID"})
    if "Image_ID" not in df.columns and "filename" in df.columns:
        df = df.rename(columns={"filename": "Image_ID"})
    if "View" not in df.columns and "projection" in df.columns:
        df = df.rename(columns={"projection": "View"})

    required_cols = ["UID", "Image_ID", "View"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"[{subset_name}] DataFrame must contain column: '{col}'. "
                f"Available columns: {df.columns.tolist()}"
            )

    # If pathology labels not present, create empty col
    if "Pathology_Labels_14" not in df.columns:
        df["Pathology_Labels_14"] = ""

    # If flags not present, default to False
    if "is_normal_meta" not in df.columns:
        df["is_normal_meta"] = False
    if "is_tq_unsat" not in df.columns:
        df["is_tq_unsat"] = False

    # Restrict to the two view classes we care about
    df = df[df["View"].isin(["Frontal", "Lateral"])].copy()
    print(f"[{subset_name}] Rows after View filter (Frontal/Lateral): {len(df)}")
    if len(df) == 0:
        print(f"[{subset_name}] No rows to evaluate after filtering. Skipping.")
        return

    print(f"[{subset_name}] Counts by view:\n{df['View'].value_counts()}")

    device = torch.device("cpu")
    print(f"[{subset_name}] Using device:", device)

    # Map between labels and indices for confusion matrices
    view2idx = {"Frontal": 0, "Lateral": 1}
    idx2view = {0: "Frontal", 1: "Lateral"}

    # Confusion matrices (rows = GT, cols = Pred) for [Frontal, Lateral]
    conf_norm = torch.zeros(2, 2, dtype=torch.int64)
    conf_clahe = torch.zeros(2, 2, dtype=torch.int64)

    # Accuracy counters
    total_norm = total_clahe = 0
    correct_norm = correct_clahe = 0

    # Per-image result storage
    uids = []
    image_ids = []
    gt_views = []
    norm_preds = []
    clahe_preds = []
    pathologies = []
    norm_times = []
    clahe_times = []
    is_normal_flags = []
    is_tq_flags = []

    print(f"\n[{subset_name}] Running MedCLIP on subset (normalized + CLAHE)...\n")

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Evaluating ({subset_name})"):
        uid       = row["UID"]
        fname     = row["Image_ID"]   # e.g., "3806_IM-1916-2001.dcm.png"
        gt_view   = row["View"]       # "Frontal" or "Lateral"
        pathology = row["Pathology_Labels_14"]
        norm_flag = bool(row.get("is_normal_meta", False))
        tq_flag   = bool(row.get("is_tq_unsat", False))

        gt_idx = view2idx[gt_view]

        norm_path  = os.path.join(normalized_root, fname)
        clahe_path = os.path.join(clahe_root, fname)

        norm_pred_view  = None
        clahe_pred_view = None
        norm_time_ms    = None
        clahe_time_ms   = None

        # ---------- Normalized image ----------
        if os.path.exists(norm_path):
            img = load_image_pil(norm_path)
            if img is not None:
                try:
                    img_inputs = processor(images=img, return_tensors="pt")
                    pixel_values = img_inputs["pixel_values"]

                    start_t = time.time()
                    with torch.no_grad():
                        img_embeds = model.vision_model(pixel_values, project=True)
                        img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
                        logits = img_embeds @ text_embeds.T
                        pred_idx = logits.argmax(dim=-1).item()
                    elapsed_ms = (time.time() - start_t) * 1000.0

                    norm_pred_view = idx2view[pred_idx]
                    norm_time_ms = elapsed_ms

                    total_norm += 1
                    if pred_idx == gt_idx:
                        correct_norm += 1
                    conf_norm[gt_idx, pred_idx] += 1
                except Exception as e:
                    print(f"[WARN][{subset_name}] Error processing normalized image {norm_path}: {e}")
        else:
            print(f"[WARN][{subset_name}] Normalized image missing: {norm_path}")

        # ---------- CLAHE image ----------
        if os.path.exists(clahe_path):
            img = load_image_pil(clahe_path)
            if img is not None:
                try:
                    img_inputs = processor(images=img, return_tensors="pt")
                    pixel_values = img_inputs["pixel_values"]

                    start_t = time.time()
                    with torch.no_grad():
                        img_embeds = model.vision_model(pixel_values, project=True)
                        img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
                        logits = img_embeds @ text_embeds.T
                        pred_idx = logits.argmax(dim=-1).item()
                    elapsed_ms = (time.time() - start_t) * 1000.0

                    clahe_pred_view = idx2view[pred_idx]
                    clahe_time_ms = elapsed_ms

                    total_clahe += 1
                    if pred_idx == gt_idx:
                        correct_clahe += 1
                    conf_clahe[gt_idx, pred_idx] += 1
                except Exception as e:
                    print(f"[WARN][{subset_name}] Error processing CLAHE image {clahe_path}: {e}")
        else:
            print(f"[WARN][{subset_name}] CLAHE image missing: {clahe_path}")

        uids.append(uid)
        image_ids.append(fname)
        gt_views.append(gt_view)
        norm_preds.append(norm_pred_view)
        clahe_preds.append(clahe_pred_view)
        norm_times.append(norm_time_ms)
        clahe_times.append(clahe_time_ms)
        pathologies.append(pathology)
        is_normal_flags.append(norm_flag)
        is_tq_flags.append(tq_flag)

    # ---------------- Metrics & confusion matrices ----------------
    acc_norm  = correct_norm  / total_norm  if total_norm  > 0 else 0.0
    acc_clahe = correct_clahe / total_clahe if total_clahe > 0 else 0.0

    print(f"\n[{subset_name}] ========= SUMMARY RESULTS =========")
    print(f"[{subset_name}] Normalized accuracy: {acc_norm:.4f}  ({correct_norm} / {total_norm})")
    print(f"[{subset_name}] Confusion matrix (Normalized) [rows=GT, cols=Pred; Frontal,Lateral]:\n{conf_norm}")
    print(f"\n[{subset_name}] CLAHE accuracy:      {acc_clahe:.4f}  ({correct_clahe} / {total_clahe})")
    print(f"[{subset_name}] Confusion matrix (CLAHE) [rows=GT, cols=Pred; Frontal,Lateral]:\n{conf_clahe}")

    # ---------------- Per-image DataFrame ----------------
    result_df = pd.DataFrame({
        "UID": uids,
        "Image_ID": image_ids,
        "GT_View": gt_views,
        "Normal_Pred": norm_preds,
        "CLAHE_Pred": clahe_preds,
        "Normal_Inference_Time_ms": norm_times,
        "CLAHE_Inference_Time_ms": clahe_times,
        "Pathology_Labels_14": pathologies,
        "is_normal_meta": is_normal_flags,
        "is_tq_unsat": is_tq_flags,
    })

    result_df["Normal_Correct"] = (result_df["Normal_Pred"] == result_df["GT_View"]).astype(int)
    result_df["CLAHE_Correct"]  = (result_df["CLAHE_Pred"] == result_df["GT_View"]).astype(int)

    result_df.to_csv(out_csv_path, index=False)
    print(f"\n[{subset_name}] Per-image prediction table saved to:\n  {out_csv_path}")

    mis_norm  = result_df[result_df["Normal_Pred"] != result_df["GT_View"]]
    mis_clahe = result_df[result_df["CLAHE_Pred"] != result_df["GT_View"]]

    print(f"[{subset_name}] # Misclassified (Normalized): {len(mis_norm)}")
    print(f"[{subset_name}] # Misclassified (CLAHE):     {len(mis_clahe)}")

    # ---------------- PDF report ----------------
    print(f"\n[{subset_name}] Generating PDF report at:\n  {out_pdf_path}")

    normal_times_series = result_df["Normal_Inference_Time_ms"].dropna()
    clahe_times_series  = result_df["CLAHE_Inference_Time_ms"].dropna()

    avg_norm_time   = normal_times_series.mean()   if len(normal_times_series) > 0 else float("nan")
    avg_clahe_time  = clahe_times_series.mean()    if len(clahe_times_series) > 0 else float("nan")
    med_norm_time   = normal_times_series.median() if len(normal_times_series) > 0 else float("nan")
    med_clahe_time  = clahe_times_series.median()  if len(clahe_times_series) > 0 else float("nan")
    max_norm_time   = normal_times_series.max()    if len(normal_times_series) > 0 else float("nan")
    max_clahe_time  = clahe_times_series.max()     if len(clahe_times_series) > 0 else float("nan")

    def confusion_caption(conf_mat, acc, variant_label: str) -> str:
        total = int(conf_mat.sum().item())
        ff = int(conf_mat[0, 0].item())  # GT Frontal → Pred Frontal
        fl = int(conf_mat[0, 1].item())  # GT Frontal → Pred Lateral
        lf = int(conf_mat[1, 0].item())  # GT Lateral → Pred Frontal
        ll = int(conf_mat[1, 1].item())  # GT Lateral → Pred Lateral

        lines = [
            f"{variant_label}: accuracy = {acc:.2%} on {total} images.",
            f"- GT Frontal: {ff} correctly predicted as Frontal, {fl} misclassified as Lateral.",
            f"- GT Lateral: {ll} correctly predicted as Lateral, {lf} misclassified as Frontal.",
        ]
        # Highlight systematic bias if any
        if fl > ff:
            lines.append("- Model tends to flip Frontal views into Lateral more often than vice versa.")
        if lf > ll:
            lines.append("- Model tends to flip Lateral views into Frontal more often than vice versa.")
        return "\n".join(lines)

    with PdfPages(out_pdf_path) as pdf:
        # ---------- Page 1: summary + overall interpretation ----------
        fig, ax = plt.subplots(figsize=(8.3, 6))  # A4-ish, helps avoid cut text
        ax.axis("off")

        summary_lines = [
            f"MedCLIP View Classification on IU-CXR ({subset_name})",
            "",
            f"Total samples (Frontal + Lateral): {len(result_df)}",
            "",
            f"Normalized accuracy: {acc_norm:.4f}  ({correct_norm} / {total_norm})",
            f"CLAHE accuracy:      {acc_clahe:.4f}  ({correct_clahe} / {total_clahe})",
            "",
            f"Misclassified (Normalized): {len(mis_norm)}",
            f"Misclassified (CLAHE):     {len(mis_clahe)}",
            "",
            "How to read the confusion matrices:",
            "- Rows = Ground Truth (GT) view, Columns = Predicted view.",
            "- Order of classes: [Frontal, Lateral].",
            "",
            "Timing (per-image, CPU-only):",
            f"- Normalized avg / median: {avg_norm_time:.2f} / {med_norm_time:.2f} ms",
            f"- CLAHE avg / median:      {avg_clahe_time:.2f} / {med_clahe_time:.2f} ms",
            f"- Max (normalized): {max_norm_time:.2f} ms",
            f"- Max (CLAHE):     {max_clahe_time:.2f} ms",
            "",
            "Overall interpretation:",
        ]

        # Simple high-level interpretation:
        if acc_norm >= acc_clahe:
            summary_lines.append(
                "- On this subset, normalized images perform slightly better or similar to CLAHE."
            )
        else:
            summary_lines.append(
                "- On this subset, CLAHE offers a small advantage over normalized images."
            )

        summary_lines.append(
            "- Off-diagonal counts in the confusion matrices indicate how often views are flipped "
            "between Frontal and Lateral; large off-diagonals reflect a strong bias toward one view."
        )

        ax.text(
            0.02, 0.98,
            "\n".join(summary_lines),
            va="top",
            fontsize=10,
        )
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
        pdf.savefig(fig)
        plt.close(fig)

        # ---------- Page 2: Normalized confusion matrix + caption ----------
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(conf_norm, cmap="Blues")  # nicer color scheme
        ax.set_title(f"Confusion Matrix - Normalized ({subset_name})")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pred Frontal", "Pred Lateral"], rotation=20, ha="right")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["GT Frontal", "GT Lateral"])

        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(conf_norm[i, j]),
                        ha="center", va="center", fontsize=10, color="black")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        caption = confusion_caption(conf_norm, acc_norm, "Normalized")
        fig.text(
            0.02, 0.02,
            caption,
            ha="left",
            va="bottom",
            fontsize=9,
        )
        fig.tight_layout(rect=[0.02, 0.10, 0.98, 0.90])  # keep caption visible
        pdf.savefig(fig)
        plt.close(fig)

        # ---------- Page 3: CLAHE confusion matrix + caption ----------
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(conf_clahe, cmap="Purples")
        ax.set_title(f"Confusion Matrix - CLAHE ({subset_name})")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pred Frontal", "Pred Lateral"], rotation=20, ha="right")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["GT Frontal", "GT Lateral"])

        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(conf_clahe[i, j]),
                        ha="center", va="center", fontsize=10, color="black")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        caption = confusion_caption(conf_clahe, acc_clahe, "CLAHE")
        fig.text(
            0.02, 0.02,
            caption,
            ha="left",
            va="bottom",
            fontsize=9,
        )
        fig.tight_layout(rect=[0.02, 0.10, 0.98, 0.90])
        pdf.savefig(fig)
        plt.close(fig)

        # ---------- Page 4: Accuracy bar chart + caption ----------
        fig, ax = plt.subplots(figsize=(6, 4.5))
        methods = ["Normalized", "CLAHE"]
        accuracies = [acc_norm, acc_clahe]
        bars = ax.bar(methods, accuracies)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"MedCLIP View Accuracy ({subset_name})")
        for bar, v in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9)

        caption = (
            "Interpretation: bars compare overall view-classification accuracy for normalized vs "
            "CLAHE images. The taller bar marks the better-performing preprocessing for this subset."
        )
        fig.text(0.02, 0.02, caption, ha="left", va="bottom", fontsize=9)
        fig.tight_layout(rect=[0.02, 0.10, 0.98, 0.92])
        pdf.savefig(fig)
        plt.close(fig)

        # ---------- Page 5: Histogram of per-image inference times ----------
        fig, ax = plt.subplots(figsize=(7, 4.5))
        normal_times = normal_times_series.values
        clahe_times = clahe_times_series.values

        ax.hist(
            [normal_times, clahe_times],
            bins=15,
            label=["Normalized", "CLAHE"],
            alpha=0.7,
        )
        ax.set_xlabel("Inference time per image (ms)")
        ax.set_ylabel("Count")
        ax.set_title(f"Inference time distribution (CPU) - {subset_name}")
        ax.legend()

        caption = (
            f"Interpretation: both distributions cluster around ~{med_norm_time:.1f}–"
            f"{med_clahe_time:.1f} ms. Overlap indicates similar latency for normalized "
            "and CLAHE images on this subset."
        )
        fig.text(0.02, 0.02, caption, ha="left", va="bottom", fontsize=9)
        fig.tight_layout(rect=[0.02, 0.10, 0.98, 0.92])
        pdf.savefig(fig)
        plt.close(fig)

        # ---------- Page 6: Boxplot of inference times ----------
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.boxplot(
            [normal_times, clahe_times],
            labels=["Normalized", "CLAHE"],
            showmeans=True,
        )
        ax.set_ylabel("Inference time per image (ms)")
        ax.set_title(f"Inference time boxplot - {subset_name}")

        caption = (
            "Interpretation: the boxplot summarizes median, IQR, and outliers for per-image "
            "inference time. Similar box heights and medians indicate comparable runtime "
            "for both preprocessing pipelines."
        )
        fig.text(0.02, 0.02, caption, ha="left", va="bottom", fontsize=9)
        fig.tight_layout(rect=[0.02, 0.10, 0.98, 0.92])
        pdf.savefig(fig)
        plt.close(fig)

        # ---------- Page 7+: Per-image table (wider + smaller text to avoid cuts) ----------
        cols_for_table = [
            "UID",
            "Image_ID",
            "GT_View",
            "Normal_Pred",
            "CLAHE_Pred",
            "Normal_Inference_Time_ms",
            "CLAHE_Inference_Time_ms",
            "Pathology_Labels_14",
            "is_normal_meta",
            "is_tq_unsat",
        ]

        table_df = result_df[cols_for_table].copy()

        table_df["Image_ID"] = table_df["Image_ID"].apply(lambda x: wrap_text(x, width=30))
        table_df["Pathology_Labels_14"] = table_df["Pathology_Labels_14"].apply(
            lambda x: wrap_text(x, width=40)
        )
        table_df["Normal_Inference_Time_ms"] = table_df["Normal_Inference_Time_ms"].apply(
            lambda x: wrap_text(f"{x:.2f}" if pd.notna(x) else "", width=10)
        )
        table_df["CLAHE_Inference_Time_ms"] = table_df["CLAHE_Inference_Time_ms"].apply(
            lambda x: wrap_text(f"{x:.2f}" if pd.notna(x) else "", width=10)
        )

        rows_per_page = 18  # slightly fewer rows → less clipping

        for start in range(0, len(table_df), rows_per_page):
            end = min(start + rows_per_page, len(table_df))
            sub_df = table_df.iloc[start:end]

            fig, ax = plt.subplots(figsize=(11.0, 6.5))
            ax.axis("off")
            ax.set_title(
                f"{subset_name} – Per-image predictions (rows {start + 1}–{end} of {len(table_df)})",
                loc="left",
            )

            # Give Image_ID & Pathology more width, shrink flags
            col_widths = [0.05, 0.25, 0.06, 0.07, 0.07, 0.09, 0.09, 0.22, 0.05, 0.05]

            table = ax.table(
                cellText=sub_df.values,
                colLabels=sub_df.columns,
                loc="center",
                cellLoc="left",
                colLoc="left",
                colWidths=col_widths,
            )
            table.auto_set_font_size(False)
            table.set_fontsize(6.5)
            table.scale(1.0, 1.15)  # a bit tighter vertically so nothing gets cut

            fig.tight_layout(rect=[0.01, 0.02, 0.99, 0.96])
            pdf.savefig(fig)
            plt.close(fig)

    print(f"[{subset_name}] PDF report generated successfully.")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 1 – MedCLIP view classification on IU-CXR subsets."
    )
    parser.add_argument(
        "--subset",
        choices=["pathology", "normal", "tech", "all"],
        default="all",
        help=(
            "Which subset(s) to evaluate: "
            "'pathology' = pathology-rich sanity subset, "
            "'normal'    = normal-only (MeSH/Problems), "
            "'tech'      = technical quality unsatisfactory, "
            "'all'      = run all three."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ----------------------------------------------------
    # 1. Load MedCLIP+Processor and precompute text embeds
    # ----------------------------------------------------
    device = torch.device("cpu")
    print("Using device:", device)

    print("Loading MedCLIP model (ViT backbone)...")
    model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    model.from_pretrained()
    model.eval()

    processor = MedCLIPProcessor()

    prompts = [
        "a frontal chest x-ray radiograph",
        "a lateral chest x-ray radiograph",
    ]
    txt_inputs = processor(text=prompts, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_embeds = model.text_model(
            input_ids=txt_inputs["input_ids"],
            attention_mask=txt_inputs["attention_mask"],
        )
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    # ----------------------------------------------------
    # 2. Run subsets based on CLI flag
    # ----------------------------------------------------
    if args.subset in ("pathology", "all"):
        print("\nLoading pathology-rich sanity subset CSV:")
        print(" ", CSV_PATH_PATHO)
        df_patho = pd.read_csv(CSV_PATH_PATHO)

        out_csv_patho = os.path.join(
            RESULTS_DIR, "medclip_view_sanity_pathology_predictions.csv"
        )
        out_pdf_patho = os.path.join(
            RESULTS_DIR, "medclip_view_sanity_pathology_report.pdf"
        )

        evaluate_subset(
            df=df_patho,
            subset_name="Pathology-rich sanity subset",
            out_csv_path=out_csv_patho,
            out_pdf_path=out_pdf_patho,
            model=model,
            processor=processor,
            text_embeds=text_embeds,
            normalized_root=NORMALIZED_ROOT,
            clahe_root=CLAHE_ROOT_PATHO,
        )

    if args.subset in ("normal", "all"):
        print("\nLoading Normal/Technical-quality subset CSV:")
        print(" ", CSV_PATH_NORMAL_TECH)
        df_normtech = pd.read_csv(CSV_PATH_NORMAL_TECH)

        df_normal_only = df_normtech[
            (df_normtech.get("is_normal_meta", False) == True)
            & (df_normtech.get("is_tq_unsat", False) == False)
        ].copy()

        out_csv_normal = os.path.join(
            RESULTS_DIR, "medclip_view_sanity_normal_predictions.csv"
        )
        out_pdf_normal = os.path.join(
            RESULTS_DIR, "medclip_view_sanity_normal_report.pdf"
        )

        evaluate_subset(
            df=df_normal_only,
            subset_name="Normal-only (MeSH/Problems)",
            out_csv_path=out_csv_normal,
            out_pdf_path=out_pdf_normal,
            model=model,
            processor=processor,
            text_embeds=text_embeds,
            normalized_root=NORMALIZED_ROOT,
            clahe_root=CLAHE_ROOT_NORMAL,
        )

    if args.subset in ("tech", "all"):
        # We re-use the same combined CSV
        if "df_normtech" not in locals():
            print("\nLoading Normal/Technical-quality subset CSV:")
            print(" ", CSV_PATH_NORMAL_TECH)
            df_normtech = pd.read_csv(CSV_PATH_NORMAL_TECH)

        df_tq_only = df_normtech[df_normtech.get("is_tq_unsat", False) == True].copy()

        out_csv_tq = os.path.join(
            RESULTS_DIR, "medclip_view_sanity_tech_unsat_predictions.csv"
        )
        out_pdf_tq = os.path.join(
            RESULTS_DIR, "medclip_view_sanity_tech_unsat_report.pdf"
        )

        evaluate_subset(
            df=df_tq_only,
            subset_name="Technical Quality Unsatisfactory",
            out_csv_path=out_csv_tq,
            out_pdf_path=out_pdf_tq,
            model=model,
            processor=processor,
            text_embeds=text_embeds,
            normalized_root=NORMALIZED_ROOT,
            clahe_root=CLAHE_ROOT_TQ,
        )

    # ----------------------------------------------------
    # 3. Normal + Technical Quality Unsatisfactory subset
    # ----------------------------------------------------
    print("\nLoading Normal/Technical-quality subset CSV:")
    print(" ", CSV_PATH_NORMAL_TECH)
    df_normtech = pd.read_csv(CSV_PATH_NORMAL_TECH)

    # Normal-only: is_normal_meta == True and not technical-unsatisfactory
    df_normal_only = df_normtech[
        (df_normtech.get("is_normal_meta", False) == True)
        & (df_normtech.get("is_tq_unsat", False) == False)
    ].copy()

    out_csv_normal = os.path.join(RESULTS_DIR, "medclip_view_sanity_normal_predictions.csv")
    out_pdf_normal = os.path.join(RESULTS_DIR, "medclip_view_sanity_normal_report.pdf")

    evaluate_subset(
        df=df_normal_only,
        subset_name="Normal-only (MeSH/Problems)",
        out_csv_path=out_csv_normal,
        out_pdf_path=out_pdf_normal,
        model=model,
        processor=processor,
        text_embeds=text_embeds,
        normalized_root=NORMALIZED_ROOT,
        clahe_root=CLAHE_ROOT_NORMAL,
    )

    # Technical quality unsatisfactory: is_tq_unsat == True
    df_tq_only = df_normtech[df_normtech.get("is_tq_unsat", False) == True].copy()

    out_csv_tq = os.path.join(RESULTS_DIR, "medclip_view_sanity_tech_unsat_predictions.csv")
    out_pdf_tq = os.path.join(RESULTS_DIR, "medclip_view_sanity_tech_unsat_report.pdf")

    evaluate_subset(
        df=df_tq_only,
        subset_name="Technical Quality Unsatisfactory",
        out_csv_path=out_csv_tq,
        out_pdf_path=out_pdf_tq,
        model=model,
        processor=processor,
        text_embeds=text_embeds,
        normalized_root=NORMALIZED_ROOT,
        clahe_root=CLAHE_ROOT_TQ,
    )


if __name__ == "__main__":
    main()
