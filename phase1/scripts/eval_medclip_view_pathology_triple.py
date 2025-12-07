"""
Stage 1 (Pathology-only, Triple Preprocessing):
MedCLIP View Classification on IU-CXR Pathology-Rich Sanity Subset

Compares MedCLIP zero-shot 'Frontal vs Lateral' view prediction on:
  1) Normalized IU-CXR images
  2) CLAHE-enhanced images
  3) Gaussian+CLAHE-enhanced images

Subset:
  - CSV: sanity_subset_iucxr_v02.csv
  - No MeSH/Problems "normal"
  - No "Technical Quality of Image Unsatisfactory"

Outputs:
  - CSV:  medclip_view_sanity_pathology_triple_predictions.csv
  - PDF:  medclip_view_sanity_pathology_triple_report.pdf
"""

import os
import time

import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor

# ---------------- PATHS ----------------

CSV_PATH_PATHO = r"D:\MedVLMBench\EDA\eda_reports\sanity_subset_iucxr_v02.csv"

NORMALIZED_ROOT = r"D:\MedVLMBench\phase1\data\chestxray_iu\images\images_normalized"
CLAHE_ROOT_PATHO = r"D:\MedVLMBench\phase1\data\chestxray_iu\images\images_clahe"
GAUSS_CLAHE_ROOT_PATHO = r"D:\MedVLMBench\phase1\data\chestxray_iu\images\images_gauss_clahe"

RESULTS_DIR = r"D:\MedVLMBench\phase1\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

OUT_CSV = os.path.join(RESULTS_DIR, "medclip_view_sanity_pathology_triple_predictions.csv")
OUT_PDF = os.path.join(RESULTS_DIR, "medclip_view_sanity_pathology_triple_report.pdf")


def load_image_pil(path: str):
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception as e:
        print(f"[WARN] Failed to load image {path}: {e}")
        return None


def wrap_text(text, width=25):
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text)
    if len(s) <= width:
        return s
    lines = []
    for i in range(0, len(s), width):
        lines.append(s[i:i + width])
    return "\n".join(lines)


def evaluate_pathology_triple(
    df: pd.DataFrame,
    model: MedCLIPModel,
    processor: MedCLIPProcessor,
    text_embeds: torch.Tensor,
    normalized_root: str,
    clahe_root: str,
    gauss_clahe_root: str,
    out_csv_path: str,
    out_pdf_path: str,
):
    subset_name = "Pathology-rich sanity subset (Triple preprocessing)"
    print(f"\n================ Evaluating: {subset_name} ================")

    # Normalize column names
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

    if "Pathology_Labels_14" not in df.columns:
        df["Pathology_Labels_14"] = ""

    if "is_normal_meta" not in df.columns:
        df["is_normal_meta"] = False
    if "is_tq_unsat" not in df.columns:
        df["is_tq_unsat"] = False

    df = df[df["View"].isin(["Frontal", "Lateral"])].copy()
    print(f"[{subset_name}] Rows after View filter (Frontal/Lateral): {len(df)}")
    print(f"[{subset_name}] Counts by view:\n{df['View'].value_counts()}")

    device = torch.device("cpu")
    print(f"[{subset_name}] Using device:", device)

    view2idx = {"Frontal": 0, "Lateral": 1}
    idx2view = {0: "Frontal", 1: "Lateral"}

    # Confusion matrices
    conf_norm = torch.zeros(2, 2, dtype=torch.int64)
    conf_clahe = torch.zeros(2, 2, dtype=torch.int64)
    conf_gauss = torch.zeros(2, 2, dtype=torch.int64)

    # Accuracy counters
    total_norm = total_clahe = total_gauss = 0
    correct_norm = correct_clahe = correct_gauss = 0

    # Per-image result storage
    uids = []
    image_ids = []
    gt_views = []
    norm_preds = []
    clahe_preds = []
    gauss_preds = []
    pathologies = []
    norm_times = []
    clahe_times = []
    gauss_times = []
    is_normal_flags = []
    is_tq_flags = []

    print(f"\n[{subset_name}] Running MedCLIP on (Normalized, CLAHE, Gauss+CLAHE)...\n")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating (pathology triple)"):
        uid = row["UID"]
        fname = row["Image_ID"]
        gt_view = row["View"]
        pathology = row["Pathology_Labels_14"]
        norm_flag = bool(row.get("is_normal_meta", False))
        tq_flag = bool(row.get("is_tq_unsat", False))

        gt_idx = view2idx[gt_view]

        norm_path = os.path.join(normalized_root, fname)
        clahe_path = os.path.join(clahe_root, fname)
        gauss_path = os.path.join(gauss_clahe_root, fname)

        norm_pred_view = None
        clahe_pred_view = None
        gauss_pred_view = None

        norm_time_ms = None
        clahe_time_ms = None
        gauss_time_ms = None

        # ---------- Normalized ----------
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
                    print(f"[WARN] Error processing normalized image {norm_path}: {e}")
        else:
            print(f"[WARN] Normalized image missing: {norm_path}")

        # ---------- CLAHE ----------
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
                    print(f"[WARN] Error processing CLAHE image {clahe_path}: {e}")
        else:
            print(f"[WARN] CLAHE image missing: {clahe_path}")

        # ---------- Gaussian+CLAHE ----------
        if os.path.exists(gauss_path):
            img = load_image_pil(gauss_path)
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

                    gauss_pred_view = idx2view[pred_idx]
                    gauss_time_ms = elapsed_ms

                    total_gauss += 1
                    if pred_idx == gt_idx:
                        correct_gauss += 1
                    conf_gauss[gt_idx, pred_idx] += 1
                except Exception as e:
                    print(f"[WARN] Error processing Gauss+CLAHE image {gauss_path}: {e}")
        else:
            print(f"[WARN] Gauss+CLAHE image missing: {gauss_path}")

        # store
        uids.append(uid)
        image_ids.append(fname)
        gt_views.append(gt_view)
        norm_preds.append(norm_pred_view)
        clahe_preds.append(clahe_pred_view)
        gauss_preds.append(gauss_pred_view)
        norm_times.append(norm_time_ms)
        clahe_times.append(clahe_time_ms)
        gauss_times.append(gauss_time_ms)
        pathologies.append(pathology)
        is_normal_flags.append(norm_flag)
        is_tq_flags.append(tq_flag)

    # ---------------- Metrics ----------------
    acc_norm = correct_norm / total_norm if total_norm > 0 else 0.0
    acc_clahe = correct_clahe / total_clahe if total_clahe > 0 else 0.0
    acc_gauss = correct_gauss / total_gauss if total_gauss > 0 else 0.0

    print(f"\n[{subset_name}] ========= SUMMARY RESULTS =========")
    print(f"Normalized accuracy:     {acc_norm:.4f}  ({correct_norm} / {total_norm})")
    print(f"CLAHE accuracy:          {acc_clahe:.4f}  ({correct_clahe} / {total_clahe})")
    print(f"Gauss+CLAHE accuracy:    {acc_gauss:.4f}  ({correct_gauss} / {total_gauss})")
    print("\nConfusion matrix (Normalized):\n", conf_norm)
    print("\nConfusion matrix (CLAHE):\n", conf_clahe)
    print("\nConfusion matrix (Gauss+CLAHE):\n", conf_gauss)

    # ---------------- Per-image DataFrame ----------------
    result_df = pd.DataFrame({
        "UID": uids,
        "Image_ID": image_ids,
        "GT_View": gt_views,
        "Normal_Pred": norm_preds,
        "CLAHE_Pred": clahe_preds,
        "GaussCLAHE_Pred": gauss_preds,
        "Normal_Inference_Time_ms": norm_times,
        "CLAHE_Inference_Time_ms": clahe_times,
        "GaussCLAHE_Inference_Time_ms": gauss_times,
        "Pathology_Labels_14": pathologies,
        "is_normal_meta": is_normal_flags,
        "is_tq_unsat": is_tq_flags,
    })

    result_df["Normal_Correct"] = (result_df["Normal_Pred"] == result_df["GT_View"]).astype(int)
    result_df["CLAHE_Correct"] = (result_df["CLAHE_Pred"] == result_df["GT_View"]).astype(int)
    result_df["GaussCLAHE_Correct"] = (result_df["GaussCLAHE_Pred"] == result_df["GT_View"]).astype(int)

    result_df.to_csv(out_csv_path, index=False)
    print(f"\nPer-image prediction table saved to:\n  {out_csv_path}")

    mis_norm = result_df[result_df["Normal_Pred"] != result_df["GT_View"]]
    mis_clahe = result_df[result_df["CLAHE_Pred"] != result_df["GT_View"]]
    mis_gauss = result_df[result_df["GaussCLAHE_Pred"] != result_df["GT_View"]]

    print(f"# Misclassified (Normalized):   {len(mis_norm)}")
    print(f"# Misclassified (CLAHE):        {len(mis_clahe)}")
    print(f"# Misclassified (Gauss+CLAHE):  {len(mis_gauss)}")

    # ---------------- PDF report ----------------
    print(f"\nGenerating PDF report at:\n  {out_pdf_path}")

    normal_times_series = result_df["Normal_Inference_Time_ms"].dropna()
    clahe_times_series = result_df["CLAHE_Inference_Time_ms"].dropna()
    gauss_times_series = result_df["GaussCLAHE_Inference_Time_ms"].dropna()

    avg_norm_time = normal_times_series.mean() if len(normal_times_series) > 0 else float("nan")
    avg_clahe_time = clahe_times_series.mean() if len(clahe_times_series) > 0 else float("nan")
    avg_gauss_time = gauss_times_series.mean() if len(gauss_times_series) > 0 else float("nan")

    med_norm_time = normal_times_series.median() if len(normal_times_series) > 0 else float("nan")
    med_clahe_time = clahe_times_series.median() if len(clahe_times_series) > 0 else float("nan")
    med_gauss_time = gauss_times_series.median() if len(gauss_times_series) > 0 else float("nan")

    max_norm_time = normal_times_series.max() if len(normal_times_series) > 0 else float("nan")
    max_clahe_time = clahe_times_series.max() if len(clahe_times_series) > 0 else float("nan")
    max_gauss_time = gauss_times_series.max() if len(gauss_times_series) > 0 else float("nan")

    def confusion_caption(conf_mat, acc, variant_label: str) -> str:
        total = int(conf_mat.sum().item())
        ff = int(conf_mat[0, 0].item())
        fl = int(conf_mat[0, 1].item())
        lf = int(conf_mat[1, 0].item())
        ll = int(conf_mat[1, 1].item())

        lines = [
            f"{variant_label}: accuracy = {acc:.2%} on {total} images.",
            f"- GT Frontal: {ff} correctly predicted as Frontal, {fl} misclassified as Lateral.",
            f"- GT Lateral: {ll} correctly predicted as Lateral, {lf} misclassified as Frontal.",
        ]
        if fl > ff:
            lines.append("- Model tends to flip Frontal views into Lateral more often than vice versa.")
        if lf > ll:
            lines.append("- Model tends to flip Lateral views into Frontal more often than vice versa.")
        return "\n".join(lines)

    with PdfPages(out_pdf_path) as pdf:
        # Page 1: summary
        fig, ax = plt.subplots(figsize=(8.3, 6))
        ax.axis("off")

        summary_lines = [
            "MedCLIP View Classification on IU-CXR",
            subset_name,
            "",
            f"Total samples (Frontal + Lateral): {len(result_df)}",
            "",
            f"Normalized accuracy:    {acc_norm:.4f}  ({correct_norm} / {total_norm})",
            f"CLAHE accuracy:         {acc_clahe:.4f}  ({correct_clahe} / {total_clahe})",
            f"Gauss+CLAHE accuracy:   {acc_gauss:.4f}  ({correct_gauss} / {total_gauss})",
            "",
            f"Misclassified (Normalized):   {len(mis_norm)}",
            f"Misclassified (CLAHE):        {len(mis_clahe)}",
            f"Misclassified (Gauss+CLAHE):  {len(mis_gauss)}",
            "",
            "Timing (per-image, CPU-only):",
            f"- Normalized avg/median/max:   {avg_norm_time:.2f} / {med_norm_time:.2f} / {max_norm_time:.2f} ms",
            f"- CLAHE avg/median/max:        {avg_clahe_time:.2f} / {med_clahe_time:.2f} / {max_clahe_time:.2f} ms",
            f"- Gauss+CLAHE avg/median/max:  {avg_gauss_time:.2f} / {med_gauss_time:.2f} / {max_gauss_time:.2f} ms",
            "",
            "High-level interpretation:",
        ]

        # quick high-level note
        best_acc = max(acc_norm, acc_clahe, acc_gauss)
        if best_acc == acc_gauss:
            summary_lines.append("- Gauss+CLAHE gives the best view classification accuracy on this subset.")
        elif best_acc == acc_clahe:
            summary_lines.append("- CLAHE gives the best view classification accuracy on this subset.")
        else:
            summary_lines.append("- Normalized images give the best view classification accuracy on this subset.")

        summary_lines.append(
            "- Compare confusion matrices to inspect whether errors are mainly Frontal→Lateral flips or vice versa."
        )

        ax.text(0.02, 0.98, "\n".join(summary_lines), va="top", fontsize=10)
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2–4: confusion matrices
        for conf_mat, acc, title_label, cmap in [
            (conf_norm,  acc_norm,  "Normalized",      "Blues"),
            (conf_clahe, acc_clahe, "CLAHE",           "Purples"),
            (conf_gauss, acc_gauss, "Gauss+CLAHE",     "Greens"),
        ]:
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(conf_mat, cmap=cmap)
            ax.set_title(f"Confusion Matrix - {title_label}")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Pred Frontal", "Pred Lateral"], rotation=20, ha="right")
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["GT Frontal", "GT Lateral"])

            for i in range(2):
                for j in range(2):
                    ax.text(j, i, int(conf_mat[i, j]),
                            ha="center", va="center", fontsize=10, color="black")

            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            caption = confusion_caption(conf_mat, acc, title_label)
            fig.text(0.02, 0.02, caption, ha="left", va="bottom", fontsize=9)
            fig.tight_layout(rect=[0.02, 0.10, 0.98, 0.90])
            pdf.savefig(fig)
            plt.close(fig)

        # Page 5: accuracy bar chart
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        methods = ["Normalized", "CLAHE", "Gauss+CLAHE"]
        accuracies = [acc_norm, acc_clahe, acc_gauss]
        bars = ax.bar(methods, accuracies)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Accuracy")
        ax.set_title("MedCLIP View Accuracy (Pathology sanity subset)")
        for bar, v in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9)

        caption = (
            "Comparison of overall view-classification accuracy across preprocessing pipelines.\n"
            "The tallest bar corresponds to the best-performing image preprocessing for this subset."
        )
        fig.text(0.02, 0.02, caption, ha="left", va="bottom", fontsize=9)
        fig.tight_layout(rect=[0.02, 0.10, 0.98, 0.92])
        pdf.savefig(fig)
        plt.close(fig)

        # Page 6: histogram of per-image inference times
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(
            [normal_times_series.values,
             clahe_times_series.values,
             gauss_times_series.values],
            bins=15,
            label=["Normalized", "CLAHE", "Gauss+CLAHE"],
            alpha=0.7,
        )
        ax.set_xlabel("Inference time per image (ms)")
        ax.set_ylabel("Count")
        ax.set_title("Inference time distribution (CPU)")
        ax.legend()

        caption = (
            "Histogram of per-image inference times. Overlapping distributions indicate similar runtime\n"
            "costs for all three preprocessing strategies on CPU."
        )
        fig.text(0.02, 0.02, caption, ha="left", va="bottom", fontsize=9)
        fig.tight_layout(rect=[0.02, 0.10, 0.98, 0.92])
        pdf.savefig(fig)
        plt.close(fig)

        # Page 7: boxplot of inference times
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.boxplot(
            [normal_times_series.values,
             clahe_times_series.values,
             gauss_times_series.values],
            labels=["Normalized", "CLAHE", "Gauss+CLAHE"],
            showmeans=True,
        )
        ax.set_ylabel("Inference time per image (ms)")
        ax.set_title("Inference time boxplot (CPU)")

        caption = (
            "Boxplot summarizing median, IQR, and outliers of inference time. Similar medians and spreads\n"
            "indicate that Gaussian+CLAHE does not introduce a major latency penalty relative to normalized/CLAHE."
        )
        fig.text(0.02, 0.02, caption, ha="left", va="bottom", fontsize=9)
        fig.tight_layout(rect=[0.02, 0.10, 0.98, 0.92])
        pdf.savefig(fig)
        plt.close(fig)

        # Page 8+: per-image table
        cols_for_table = [
            "UID",
            "Image_ID",
            "GT_View",
            "Normal_Pred",
            "CLAHE_Pred",
            "GaussCLAHE_Pred",
            "Normal_Inference_Time_ms",
            "CLAHE_Inference_Time_ms",
            "GaussCLAHE_Inference_Time_ms",
            "Pathology_Labels_14",
        ]
        table_df = result_df[cols_for_table].copy()
        table_df["Image_ID"] = table_df["Image_ID"].apply(lambda x: wrap_text(x, width=30))
        table_df["Pathology_Labels_14"] = table_df["Pathology_Labels_14"].apply(
            lambda x: wrap_text(x, width=40)
        )
        for col in [
            "Normal_Inference_Time_ms",
            "CLAHE_Inference_Time_ms",
            "GaussCLAHE_Inference_Time_ms",
        ]:
            table_df[col] = table_df[col].apply(
                lambda x: wrap_text(f"{x:.2f}" if pd.notna(x) else "", width=10)
            )

        rows_per_page = 18
        for start in range(0, len(table_df), rows_per_page):
            end = min(start + rows_per_page, len(table_df))
            sub_df = table_df.iloc[start:end]

            fig, ax = plt.subplots(figsize=(11.0, 6.5))
            ax.axis("off")
            ax.set_title(
                f"Per-image predictions (rows {start+1}–{end} of {len(table_df)})",
                loc="left",
            )

            col_widths = [
                0.05,  # UID
                0.22,  # Image_ID
                0.06,  # GT_View
                0.07,  # Normal_Pred
                0.07,  # CLAHE_Pred
                0.09,  # GaussCLAHE_Pred
                0.08,  # Normal time
                0.08,  # CLAHE time
                0.08,  # Gauss time
                0.20,  # Pathology_Labels_14
            ]

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
            table.scale(1.0, 1.15)

            fig.tight_layout(rect=[0.01, 0.02, 0.99, 0.96])
            pdf.savefig(fig)
            plt.close(fig)

    print(f"[{subset_name}] PDF report generated successfully at:\n  {out_pdf_path}")


def main():
    device = torch.device("cpu")
    print("Using device:", device)

    # 1. Load MedCLIP model
    print("Loading MedCLIP model (ViT backbone)...")
    model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    model.from_pretrained()
    model.eval()

    processor = MedCLIPProcessor()

    # 2. Prepare text embeddings
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

    # 3. Load pathology sanity subset
    print("\nLoading pathology-rich sanity subset CSV:")
    print(" ", CSV_PATH_PATHO)
    df_patho = pd.read_csv(CSV_PATH_PATHO)

    # 4. Evaluate triple-preprocessing variant
    evaluate_pathology_triple(
        df=df_patho,
        model=model,
        processor=processor,
        text_embeds=text_embeds,
        normalized_root= NORMALIZED_ROOT,
        clahe_root= CLAHE_ROOT_PATHO,
        gauss_clahe_root= GAUSS_CLAHE_ROOT_PATHO,
        out_csv_path= OUT_CSV,
        out_pdf_path= OUT_PDF,
    )


if __name__ == "__main__":
    main()