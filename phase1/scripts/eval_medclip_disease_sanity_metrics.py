"""
Zero-shot disease evaluation for MedCLIP on IU-CXR pathology-rich sanity subset.

Workflow:
  1. Run eval_medclip_disease_sanity.py (once per image variant) to produce a preds CSV.
  2. Run this script on that CSV to compute metrics and generate a PDF report.

Example:
  python -m phase1.scripts.eval_medclip_disease_sanity_metrics ^
    --preds-csv phase1\\results\\medclip_disease_sanity_normalized_preds.csv ^
    --tag normalized

Outputs:
  - phase1/results/medclip_disease_sanity_<tag>_zero_shot_metrics.csv
  - phase1/results/medclip_disease_sanity_<tag>_zero_shot_report.pdf
"""

import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# All metrics + PDF will be saved into this folder
RESULTS_DIR = r"D:\MedVLMBench\phase1\results"

# 5 CheXpert competition diseases
DISEASE_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]

# For safety: if *_label columns are missing, we re-derive them from Pathology_Labels_14
LABEL_PATTERNS = {
    "Atelectasis": ["atelectasis"],
    "Cardiomegaly": ["cardiomegaly"],
    "Consolidation": ["consolidation"],
    "Edema": ["edema"],
    "Pleural Effusion": ["pleural_effusion", "pleural effusion"],
}


# -------------------------------------------------------------------
# 1) Ensure labels exist
# -------------------------------------------------------------------
def build_binary_labels_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantee that <Disease>_label columns exist.

    If any of the 5 *_label columns are missing, we derive them
    from 'Pathology_Labels_14' with simple string matching.
    """
    if "Pathology_Labels_14" not in df.columns:
        raise ValueError(
            "Expected 'Pathology_Labels_14' in predictions CSV, "
            f"found columns: {df.columns.tolist()}"
        )

    def normalize_label_str(s):
        if pd.isna(s):
            return ""
        s = str(s).lower().replace(" ", "_")
        return s

    norm_text = df["Pathology_Labels_14"].apply(normalize_label_str)

    for disease in DISEASE_CLASSES:
        label_col = f"{disease}_label"
        if label_col in df.columns:
            # If you already saved labels in the eval script, we reuse them.
            continue

        patterns = LABEL_PATTERNS[disease]

        def has_disease(t):
            # 1 if ANY of the disease strings appears in the pathology text
            return int(any(p in t for p in patterns))

        df[label_col] = norm_text.apply(has_disease)

    return df


# -------------------------------------------------------------------
# 2) Per-disease metrics (binary classification using prob >= 0.5)
# -------------------------------------------------------------------
def compute_metrics_for_disease(y_true, y_prob, disease, thresh=0.5):
    """
    Compute metrics for a single disease:
      - counts: Positives, Negatives
      - AUROC, AUPRC
      - Acc, Precision, Recall, F1 at threshold (default 0.5)
      - TP, FP, TN, FN
    """
    # Drop rows where either label or prob is NaN
    mask = ~np.isnan(y_true) & ~np.isnan(y_prob)
    y_true = np.array(y_true[mask], dtype=int)
    y_prob = np.array(y_prob[mask], dtype=float)

    result = {
        "Disease": disease,
        "Positives": int(np.sum(y_true == 1)),
        "Negatives": int(np.sum(y_true == 0)),
        "AUROC": np.nan,
        "AUPRC": np.nan,
        "Acc@0.5": np.nan,
        "Prec@0.5": np.nan,
        "Rec@0.5": np.nan,
        "F1@0.5": np.nan,
        "TP": np.nan,
        "FP": np.nan,
        "TN": np.nan,
        "FN": np.nan,
    }

    if len(y_true) == 0:
        # No samples at all for this disease (unlikely but safe)
        return result

    # Need both classes present for AUROC/AUPRC
    if len(np.unique(y_true)) >= 2:
        try:
            result["AUROC"] = roc_auc_score(y_true, y_prob)
        except Exception:
            result["AUROC"] = np.nan

        try:
            result["AUPRC"] = average_precision_score(y_true, y_prob)
        except Exception:
            result["AUPRC"] = np.nan

    # Binary predictions at threshold
    y_pred = (y_prob >= thresh).astype(int)

    # Confusion matrix
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    except ValueError:
        # e.g., only one class present
        tn = fp = fn = tp = np.nan

    result["TP"] = tp
    result["FP"] = fp
    result["TN"] = tn
    result["FN"] = fn

    total = tn + fp + fn + tp
    if not np.isnan(total) and total > 0:
        result["Acc@0.5"] = (tp + tn) / total

    # Precision, Recall, F1
    try:
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        result["Prec@0.5"] = prec
        result["Rec@0.5"] = rec
        result["F1@0.5"] = f1
    except Exception:
        pass

    return result


# -------------------------------------------------------------------
# 3) Micro-average metrics across all diseases
# -------------------------------------------------------------------
def compute_micro_metrics(df: pd.DataFrame, thresh=0.5):
    """
    Flatten all disease labels into a single vector and compute:
      - Micro Acc, Prec, Rec, F1

    This addresses: "If I ignore which disease it is,
    how good are we overall at disease vs no-disease decisions?"
    """
    y_true_all = []
    y_pred_all = []

    for disease in DISEASE_CLASSES:
        label_col = f"{disease}_label"
        prob_col = f"{disease}_prob"
        if label_col not in df.columns or prob_col not in df.columns:
            continue

        y_true = df[label_col].values
        y_prob = df[prob_col].values
        mask = ~np.isnan(y_true) & ~np.isnan(y_prob)
        y_true = y_true[mask].astype(int)
        y_pred = (y_prob[mask] >= thresh).astype(int)

        y_true_all.append(y_true)
        y_pred_all.append(y_pred)

    if not y_true_all:
        return {}

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)

    acc = np.mean(y_true_all == y_pred_all)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_all, y_pred_all, average="binary", zero_division=0
    )

    return {
        "Micro_Acc@0.5": acc,
        "Micro_Prec@0.5": prec,
        "Micro_Rec@0.5": rec,
        "Micro_F1@0.5": f1,
        "Total_Labels": int(len(y_true_all)),
    }


# -------------------------------------------------------------------
# 4) Top-K accuracies across the 5 diseases
# -------------------------------------------------------------------
def compute_topk_accuracies(df: pd.DataFrame, k_list=[1, 3]):
    """
    Compute Top-K accuracies over the 5 CheXpert diseases.

    For each image:
      - Take the 5 probabilities [Atelectasis_prob, ..., Pleural Effusion_prob]
      - Take the 5 ground-truth labels [*_label]
      - If ALL labels are 0, skip that image (no target disease present)
      - For each K in k_list:
          * Sort diseases by prob descending
          * success_topK = 1 if ANY of the top-K diseases has label==1

    This directly matches:
      "If the highest-ranking disease (or top-3) is actually present, give 1."
    """
    topk_results = {}
    prob_cols = [f"{d}_prob" for d in DISEASE_CLASSES]
    label_cols = [f"{d}_label" for d in DISEASE_CLASSES]

    for c in prob_cols + label_cols:
        if c not in df.columns:
            print(f"[WARN] compute_topk_accuracies: missing column {c}, skipping Top-K.")
            return {}

    probs_mat = df[prob_cols].values  # [N, 5]
    labels_mat = df[label_cols].values  # [N, 5]

    # Only consider images with at least one positive label
    valid_mask = (labels_mat.sum(axis=1) > 0)
    probs_mat = probs_mat[valid_mask]
    labels_mat = labels_mat[valid_mask]

    if probs_mat.shape[0] == 0:
        print("[WARN] compute_topk_accuracies: no images with positive labels; Top-K undefined.")
        return {}

    for k in k_list:
        successes = []
        for probs, labels in zip(probs_mat, labels_mat):
            order = np.argsort(-probs)  # descending
            topk_idx = order[:k]
            success = int(labels[topk_idx].sum() > 0)
            successes.append(success)
        acc = float(np.mean(successes))
        topk_results[f"Top{k}_Acc"] = acc
        topk_results[f"Top{k}_N"] = int(len(successes))

    return topk_results


# -------------------------------------------------------------------
# 5) Global "any disease predicted correctly?" metrics
# -------------------------------------------------------------------
def compute_any_hit_metrics(df: pd.DataFrame, thresh=0.5):
    """
    Treat each image as a single multi-label sample.
    Question: Did we detect at least *one* of the true diseases?

    For each image:
      - any_true = (sum of 5 labels > 0)
      - any_pred = (sum of (prob >= thresh) across 5 diseases > 0)

    We then build a 2x2 confusion matrix between any_true vs any_pred.
    """
    label_cols = [f"{d}_label" for d in DISEASE_CLASSES]
    prob_cols = [f"{d}_prob" for d in DISEASE_CLASSES]

    for c in label_cols + prob_cols:
        if c not in df.columns:
            print(f"[WARN] compute_any_hit_metrics: missing column {c}, skipping any-hit metrics.")
            return {}, None

    labels_mat = df[label_cols].values  # [N,5]
    probs_mat = df[prob_cols].values   # [N,5]

    any_true = (labels_mat.sum(axis=1) > 0).astype(int)
    any_pred = ((probs_mat >= thresh).sum(axis=1) > 0).astype(int)

    # Remove rows where all probs are NaN (safety)
    nan_mask = np.isnan(probs_mat).all(axis=1)
    any_true = any_true[~nan_mask]
    any_pred = any_pred[~nan_mask]

    if any_true.size == 0:
        return {}, None

    tn, fp, fn, tp = confusion_matrix(any_true, any_pred).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec, rec, f1, _ = precision_recall_fscore_support(
        any_true, any_pred, average="binary", zero_division=0
    )

    stats = {
        "AnyHit_Acc": acc,
        "AnyHit_Prec": prec,
        "AnyHit_Rec": rec,
        "AnyHit_F1": f1,
        "AnyHit_TP": tp,
        "AnyHit_FP": fp,
        "AnyHit_TN": tn,
        "AnyHit_FN": fn,
    }

    cm = np.array([[tn, fp],
                   [fn, tp]], dtype=int)

    return stats, cm


# -------------------------------------------------------------------
# 6) PDF Report helpers
# -------------------------------------------------------------------
def wrap_text(s, width=40):
    """Simple helper to wrap long strings into multiple lines for tables."""
    if pd.isna(s):
        return ""
    s = str(s)
    if len(s) <= width:
        return s
    return "\n".join(s[i:i + width] for i in range(0, len(s), width))


def generate_pdf_report(
    df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    micro_stats: dict,
    topk_stats: dict,
    anyhit_stats: dict,
    anyhit_cm: np.ndarray,
    out_pdf: str,
):
    """
    PDF pages:
      1. Summary + micro metrics + top-k + any-hit summary
      2. AUROC bar chart
      3. F1 bar chart
      4. Per-disease metrics table
      5-?: Per-disease confusion matrices (with interpretation text, no overlap)
      + Global any-hit confusion matrix (with text, no overlap)
      + Inference time boxplot
      + Per-image table (short headers, larger figure, explicit column widths).
    """
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    n_images = len(df)

    with PdfPages(out_pdf) as pdf:
        # ---------------- Page 1: Summary ----------------
        fig, ax = plt.subplots(figsize=(8.3, 6))
        ax.axis("off")

        summary_lines = [
            "MedCLIP Zero-Shot Disease Classification on IU-CXR",
            "Pathology-Rich Sanity Subset (NO filtering; all sanity images)",
            "",
            f"Number of images: {n_images}",
            "",
            "Setup:",
            "- Start from the pathology-rich IU-CXR sanity subset CSV.",
            "- For each image, MedCLIP PromptClassifier produces zero-shot probabilities for:",
            "    Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion.",
            "- For each disease, ground-truth labels are derived from Pathology_Labels_14",
            "  using simple string matching; if ANY matching phrase is present, label=1.",
            "",
            "Evaluation:",
            "- Per-disease metrics: AUROC, AUPRC, Acc, Precision, Recall, F1 (threshold=0.5).",
            "- Micro-averaged metrics flatten all 5 diseases across all images.",
        ]

        if micro_stats:
            summary_lines += [
                "",
                "Micro-averaged performance (across all diseases & images, threshold 0.5):",
                f"- Micro Accuracy:  {micro_stats['Micro_Acc@0.5']:.3f}",
                f"- Micro Precision: {micro_stats['Micro_Prec@0.5']:.3f}",
                f"- Micro Recall:    {micro_stats['Micro_Rec@0.5']:.3f}",
                f"- Micro F1:        {micro_stats['Micro_F1@0.5']:.3f}",
                f"- Total disease-label decisions evaluated: {micro_stats['Total_Labels']}",
            ]

        if topk_stats:
            summary_lines += [
                "",
                "Top-K zero-shot disease ranking (only images with ≥1 positive label):",
                f"- Top-1 accuracy: {topk_stats.get('Top1_Acc', float('nan')):.3f} "
                f"(N={topk_stats.get('Top1_N', 0)})",
                f"- Top-3 accuracy: {topk_stats.get('Top3_Acc', float('nan')):.3f} "
                f"(N={topk_stats.get('Top3_N', 0)})",
            ]

        if anyhit_stats:
            summary_lines += [
                "",
                "Global 'any disease present & predicted' metric (threshold 0.5):",
                "- We mark an image as correctly classified if at least one of the 5 diseases",
                "  is present in the ground truth AND predicted as positive.",
                f"- Any-hit Accuracy:  {anyhit_stats['AnyHit_Acc']:.3f}",
                f"- Any-hit Precision: {anyhit_stats['AnyHit_Prec']:.3f}",
                f"- Any-hit Recall:    {anyhit_stats['AnyHit_Rec']:.3f}",
                f"- Any-hit F1:        {anyhit_stats['AnyHit_F1']:.3f}",
            ]

        ax.text(0.03, 0.97, "\n".join(summary_lines), va="top", fontsize=9)
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
        pdf.savefig(fig)
        plt.close(fig)

        if metrics_df is None or metrics_df.empty:
            return

        diseases = metrics_df["Disease"].tolist()

        # ---------------- Page 2: AUROC bar chart ----------------
        fig, ax = plt.subplots(figsize=(7, 5))
        aurocs = metrics_df["AUROC"].tolist()
        bars = ax.bar(diseases, aurocs)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("AUROC")
        ax.set_title("Per-disease AUROC (zero-shot, all sanity images)")
        for bar, v in zip(bars, aurocs):
            if pd.notna(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8)
        caption = (
            "AUROC measures how well the model ranks positive vs negative cases per disease.\n"
            "High AUROC indicates good separation even without probability calibration."
        )
        fig.text(0.02, 0.02, caption, ha="left", va="bottom", fontsize=8)
        fig.tight_layout(rect=[0.06, 0.10, 0.98, 0.93])
        pdf.savefig(fig)
        plt.close(fig)

        # ---------------- Page 3: F1 bar chart ----------------
        fig, ax = plt.subplots(figsize=(7, 5))
        f1s = metrics_df["F1@0.5"].tolist()
        bars = ax.bar(diseases, f1s)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("F1 score @ 0.5")
        ax.set_title("Per-disease F1 score (threshold 0.5)")
        for bar, v in zip(bars, f1s):
            if pd.notna(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8)
        caption = (
            "F1@0.5 combines precision and recall for each disease.\n"
            "It reflects how well zero-shot scores translate into present/absent decisions\n"
            "with a simple 0.5 decision threshold."
        )
        fig.text(0.02, 0.02, caption, ha="left", va="bottom", fontsize=8)
        fig.tight_layout(rect=[0.06, 0.10, 0.98, 0.93])
        pdf.savefig(fig)
        plt.close(fig)

        # ---------------- Page 4: Metrics table ----------------
        fig, ax = plt.subplots(figsize=(10.5, 4.5))
        ax.axis("off")
        ax.set_title("Per-disease zero-shot metrics (all sanity images)", loc="left")

        display_cols = [
            "Disease", "Positives", "Negatives",
            "AUROC", "AUPRC",
            "Acc@0.5", "Prec@0.5", "Rec@0.5", "F1@0.5",
            "TP", "FP", "TN", "FN",
        ]
        table_df = metrics_df[display_cols].copy()

        for col in ["AUROC", "AUPRC", "Acc@0.5", "Prec@0.5", "Rec@0.5", "F1@0.5"]:
            table_df[col] = table_df[col].apply(
                lambda x: f"{x:.3f}" if pd.notna(x) else "NA"
            )

        table = ax.table(
            cellText=table_df.values,
            colLabels=table_df.columns,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.25)

        caption = (
            "Metrics computed disease-wise using 0.5 as the decision threshold.\n"
            "This lets you compare MedCLIP behavior across the 5 CheXpert diseases."
        )
        fig.text(0.02, 0.02, caption, ha="left", va="bottom", fontsize=7)
        fig.tight_layout(rect=[0.03, 0.12, 0.98, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

        # ---------------- Per-disease confusion matrices ----------------
        for _, row in metrics_df.iterrows():
            disease = row["Disease"]
            tn, fp, fn, tp = row["TN"], row["FP"], row["FN"], row["TP"]
            if any(pd.isna(x) for x in [tn, fp, fn, tp]):
                continue

            cm = np.array([[tn, fp],
                           [fn, tp]], dtype=int)

            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(cm, cmap="Blues")

            ax.set_title(f"{disease}: Confusion Matrix (threshold=0.5)")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Pred 0", "Pred 1"])
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["GT 0", "GT 1"])

            for i in range(2):
                for j in range(2):
                    ax.text(j, i, int(cm[i, j]),
                            ha="center", va="center", fontsize=10, color="black")

            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            total_neg = tn + fp
            total_pos = fn + tp

            explanation_lines = [
                f"For {disease}, positives={total_pos}, negatives={total_neg}.",
                f"- TN={tn}: images without {disease} correctly predicted as negative.",
                f"- FP={fp}: images without {disease} incorrectly flagged as present.",
                f"- FN={fn}: images with {disease} missed by the model.",
                f"- TP={tp}: images with {disease} correctly detected.",
            ]
            if tn == 0 and fp > 0:
                explanation_lines.append(
                    "Here, the model predicts this disease for nearly every case "
                    "(no true negatives), so recall is high but specificity is poor."
                )

            # Reserve bottom ~22% of page for text → no overlap
            fig.text(0.03, 0.02, "\n".join(explanation_lines),
                     ha="left", va="bottom", fontsize=8)
            fig.tight_layout(rect=[0.10, 0.25, 0.98, 0.95])
            pdf.savefig(fig)
            plt.close(fig)

        # ---------------- Global any-hit confusion matrix ----------------
        if anyhit_stats and anyhit_cm is not None:
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(anyhit_cm, cmap="Greens")

            ax.set_title("Any-disease confusion matrix (threshold=0.5)")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Pred none", "Pred ≥1 disease"])
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["GT none", "GT ≥1 disease"])

            for i in range(2):
                for j in range(2):
                    ax.text(j, i, int(anyhit_cm[i, j]),
                            ha="center", va="center", fontsize=10, color="black")

            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            tn, fp, fn, tp = (anyhit_stats["AnyHit_TN"], anyhit_stats["AnyHit_FP"],
                              anyhit_stats["AnyHit_FN"], anyhit_stats["AnyHit_TP"])

            explanation_lines = [
                "We compress the multi-label task into a single decision per image:",
                "- GT ≥1 disease vs GT none (over the 5 CheXpert labels).",
                "- Pred ≥1 disease vs Pred none, using prob ≥ 0.5.",
                "",
                f"TN={tn}, FP={fp}, FN={fn}, TP={tp}",
                f"Accuracy: {anyhit_stats['AnyHit_Acc']:.3f}, "
                f"Precision: {anyhit_stats['AnyHit_Prec']:.3f}, "
                f"Recall: {anyhit_stats['AnyHit_Rec']:.3f}, "
                f"F1: {anyhit_stats['AnyHit_F1']:.3f}.",
                "",
                "This view answers: did the model detect at least one true disease on each image?",
            ]
            fig.text(0.03, 0.02, "\n".join(explanation_lines),
                     ha="left", va="bottom", fontsize=8)
            fig.tight_layout(rect=[0.10, 0.25, 0.98, 0.95])
            pdf.savefig(fig)
            plt.close(fig)

        # ---------------- Inference time boxplot ----------------
        if "Inference_Time_ms" in df.columns:
            time_series = df["Inference_Time_ms"].dropna().values
            if time_series.size > 0:
                fig, ax = plt.subplots(figsize=(6.5, 4.5))
                ax.boxplot(time_series, vert=True, showmeans=True)
                ax.set_ylabel("Inference time per image (ms)")
                ax.set_title("MedCLIP zero-shot disease head – CPU inference time")

                caption = (
                    "Boxplot summarizing median, IQR, outliers and mean of per-image inference time.\n"
                    "This helps compare disease-head latency with view-classification experiments."
                )
                fig.text(0.03, 0.02, caption,
                         ha="left", va="bottom", fontsize=8)
                fig.tight_layout(rect=[0.10, 0.15, 0.98, 0.92])
                pdf.savefig(fig)
                plt.close(fig)

        # ---------------- Per-image wide table ----------------
        # Use short column labels so headers don't get cut and a large landscape figure.
        wide_cols = ["UID", "Image_ID", "Pathology_Labels_14"]
        for d in DISEASE_CLASSES:
            wide_cols.append(f"{d}_label")
            wide_cols.append(f"{d}_prob")

        existing_cols = [c for c in wide_cols if c in df.columns]
        table_df = df[existing_cols].copy()

        table_df["Pathology_Labels_14"] = table_df["Pathology_Labels_14"].apply(
            lambda x: wrap_text(x, width=45)
        )

        # Format prob columns
        for d in DISEASE_CLASSES:
            prob_col = f"{d}_prob"
            if prob_col in table_df.columns:
                table_df[prob_col] = table_df[prob_col].apply(
                    lambda x: f"{x:.3f}" if pd.notna(x) else "NA"
                )

        # Short display names for headers: "Atelectasis_gt", "Atelectasis_p"
        display_colnames = []
        for col in table_df.columns:
            if col.endswith("_label"):
                base = col.replace("_label", "")
                display_colnames.append(f"{base}_gt")
            elif col.endswith("_prob"):
                base = col.replace("_prob", "")
                display_colnames.append(f"{base}_p")
            else:
                display_colnames.append(col)

        rows_per_page = 15
        n_rows = len(table_df)
        start_idx = 0
        while start_idx < n_rows:
            end_idx = min(start_idx + rows_per_page, n_rows)
            sub_df = table_df.iloc[start_idx:end_idx]

            # big landscape figure
            fig, ax = plt.subplots(figsize=(11.5, 7.5))
            ax.axis("off")
            ax.set_title(
                f"Per-image labels & scores (rows {start_idx + 1}–{end_idx} of {n_rows})",
                loc="left",
            )

            num_cols = sub_df.shape[1]
            col_widths = []
            for name in display_colnames:
                if name == "UID":
                    col_widths.append(0.05)
                elif name == "Image_ID":
                    col_widths.append(0.20)
                elif name == "Pathology_Labels_14":
                    col_widths.append(0.25)
                else:
                    # remaining width spread across disease cols
                    col_widths.append(max(0.04, (1.0 - 0.05 - 0.20 - 0.25) / (num_cols - 3)))

            table = ax.table(
                cellText=sub_df.values,
                colLabels=display_colnames,
                loc="center",
                cellLoc="left",
                colLoc="left",
                colWidths=col_widths,
            )
            table.auto_set_font_size(False)
            table.set_fontsize(6.5)
            table.scale(1.0, 1.2)

            fig.tight_layout(rect=[0.02, 0.06, 0.99, 0.95])
            pdf.savefig(fig)
            plt.close(fig)

            start_idx = end_idx

    print(f">> PDF report generated at:\n   {out_pdf}")


# -------------------------------------------------------------------
# 7) Main – glue it together and expose CLI
# -------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preds-csv",
        type=str,
        required=True,
        help="Path to preds CSV from eval_medclip_disease_sanity.py",
    )
    parser.add_argument(
        "--tag",
        type=str,
        required=True,
        help="Short tag to embed in output filenames, e.g. normalized / clahe / gaussclahe",
    )
    args = parser.parse_args()

    preds_csv = args.preds_csv
    tag = args.tag

    if not os.path.exists(preds_csv):
        raise FileNotFoundError(f"Prediction CSV not found at: {preds_csv}")

    metrics_csv = os.path.join(
        RESULTS_DIR, f"medclip_disease_sanity_{tag}_zero_shot_metrics.csv"
    )
    pdf_path = os.path.join(
        RESULTS_DIR, f"medclip_disease_sanity_{tag}_zero_shot_report.pdf"
    )

    print(">> Reading predictions from:")
    print("   ", preds_csv)
    df = pd.read_csv(preds_csv)

    print(">> Ensuring <Disease>_label columns exist...")
    df = build_binary_labels_if_missing(df)

    # Per-disease metrics
    print("\n>> Computing per-disease zero-shot metrics (AUROC/AUPRC + Acc/Prec/Rec/F1 @ 0.5)...")
    rows = []
    for disease in DISEASE_CLASSES:
        label_col = f"{disease}_label"
        prob_col = f"{disease}_prob"

        if label_col not in df.columns or prob_col not in df.columns:
            print(f"[WARN] Missing columns for {disease}; skipping.")
            continue

        y_true = df[label_col].values
        y_prob = df[prob_col].values

        stats = compute_metrics_for_disease(y_true, y_prob, disease, thresh=0.5)
        rows.append(stats)

    metrics_df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    metrics_df.to_csv(metrics_csv, index=False)

    print("\n>> Per-disease zero-shot metrics saved to:")
    print("   ", metrics_csv)
    if not metrics_df.empty:
        print("\nMetrics summary:")
        print(metrics_df.to_string(index=False))
    else:
        print("[WARN] No metrics computed.")

    # Micro-average
    print("\n>> Computing micro-averaged metrics across all diseases (threshold 0.5)...")
    micro_stats = compute_micro_metrics(df, thresh=0.5)

    # Top-K
    print("\n>> Computing Top-1 / Top-3 accuracies over CheXpert-5 diseases...")
    topk_stats = compute_topk_accuracies(df, k_list=[1, 3])

    # Any-hit metrics
    print("\n>> Computing global 'any disease present & predicted' metrics...")
    anyhit_stats, anyhit_cm = compute_any_hit_metrics(df, thresh=0.5)

    # PDF report
    print("\n>> Generating PDF report...")
    generate_pdf_report(
        df,
        metrics_df,
        micro_stats,
        topk_stats,
        anyhit_stats,
        anyhit_cm,
        pdf_path,
    )


if __name__ == "__main__":
    main()
