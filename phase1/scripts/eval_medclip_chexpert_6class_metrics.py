"""
6-class (CheXpert-5 + Others) zero-shot evaluation for MedCLIP on IU-CXR subsets.

Goal
----
Given a predictions CSV with:
  - chexpert_label          (single ground-truth label per row)
  - <Disease>_prob columns  (MedCLIP zero-shot scores for the 5 CheXpert classes)

we build a 6-way classification task:

  Ground truth classes:
    {Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion, Others}

  Predictions:
    argmax over the 5 MedCLIP probabilities  → always one of the 5 diseases
    (MedCLIP does *not* predict "Others"; this is intentional.)

This directly answers:
  "When the true label is outside the CheXpert-5 set, how often does
   MedCLIP incorrectly force it into one of the 5 disease buckets?"

Outputs
-------
  1) <results-dir>/medclip_chexpert_6class_<tag>_metrics.csv
     - Per-class support, accuracy, precision, recall, F1

  2) <results-dir>/medclip_chexpert_6class_<tag>_confusion_matrix.csv
     - 6x6 confusion matrix (rows=GT, cols=Pred)

  3) <results-dir>/medclip_chexpert_6class_<tag>_report.pdf
     - Summary page
     - Confusion matrix heatmap with explanation
     - Per-class F1 bar chart
     - Per-class metrics table
     - (Optional) inference time boxplot if Inference_Time_ms is present.

CLI usage
---------
  python -m phase1.scripts.eval_medclip_chexpert_6class_metrics ^
    --preds-csv D:\MedVLMBench\phase1\results\medclip_chexpert_sanity_raw_preds.csv ^
    --tag raw ^
    --results-dir D:\MedVLMBench\phase1\results
"""

import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# 5 CheXpert competition diseases (MedCLIP head)
CHEXPERT_DISEASES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]

# 6-class label set (CheXpert-5 + Others)
CLASS_NAMES_6 = CHEXPERT_DISEASES + ["Others"]


# -------------------------------------------------------------------
# Helper: build 6-way GT + predictions
# -------------------------------------------------------------------
def build_6class_labels(df: pd.DataFrame):
    """
    From a predictions CSV, construct:

      - gt_6: np.array of shape [N], values in CLASS_NAMES_6
      - pred_6: np.array of shape [N], values in CLASS_NAMES_6 (but never "Others")

    Requirements:
      - 'chexpert_label' column present.
      - For each disease d in CHEXPERT_DISEASES, column f"{d}_prob" exists.

    Any row where all 5 probabilities are NaN is skipped.
    """

    if "chexpert_label" not in df.columns:
        raise ValueError(
            "Expected 'chexpert_label' in preds CSV. "
            f"Found columns: {df.columns.tolist()}"
        )

    # Ensure all prob columns exist
    prob_cols = [f"{d}_prob" for d in CHEXPERT_DISEASES]
    for c in prob_cols:
        if c not in df.columns:
            raise ValueError(
                f"Missing probability column '{c}' in preds CSV. "
                "Please run the MedCLIP eval script first."
            )

    # Extract probs matrix [N,5]
    probs_mat = df[prob_cols].values
    gt_labels_raw = df["chexpert_label"].astype(str).values

    gt_6 = []
    pred_6 = []
    rows_used = 0

    for i, (gt_raw, probs_row) in enumerate(zip(gt_labels_raw, probs_mat)):
        # Skip rows where all probs are NaN (e.g., failed inference)
        if np.isnan(probs_row).all():
            continue

        # Ground truth:
        #   - If label is one of the 5 CheXpert classes, keep as is.
        #   - Otherwise, map to "Others".
        if gt_raw in CHEXPERT_DISEASES:
            gt = gt_raw
        else:
            gt = "Others"

        # Prediction: argmax over the 5 CheXpert probabilities
        # (MedCLIP never predicts "Others" explicitly.)
        max_idx = np.nanargmax(probs_row)  # safe because not all NaN
        pred = CHEXPERT_DISEASES[max_idx]

        gt_6.append(gt)
        pred_6.append(pred)
        rows_used += 1

    gt_6 = np.array(gt_6, dtype=object)
    pred_6 = np.array(pred_6, dtype=object)

    print(f"[INFO] Number of valid rows (used in 6-class eval): {rows_used}")
    return gt_6, pred_6


# -------------------------------------------------------------------
# Metrics computation
# -------------------------------------------------------------------
def compute_6class_metrics(gt_6, pred_6):
    """
    Compute:
      - overall accuracy
      - macro F1
      - per-class accuracy, precision, recall, F1, support
      - confusion matrix (6x6)

    Returns:
      metrics_df: DataFrame per class
      overall_stats: dict with 'Overall_Accuracy', 'Macro_F1'
      cm: confusion matrix np.array of shape [6,6]
    """

    cm = confusion_matrix(
        gt_6, pred_6, labels=CLASS_NAMES_6
    )  # rows=GT, cols=Pred

    total = cm.sum()
    correct = np.trace(cm)
    overall_acc = correct / total if total > 0 else np.nan

    # Macro precision/recall/F1
    prec, rec, f1, support = precision_recall_fscore_support(
        gt_6,
        pred_6,
        labels=CLASS_NAMES_6,
        average=None,
        zero_division=0,
    )

    macro_f1 = f1.mean() if len(f1) > 0 else np.nan

    metrics_rows = []
    for idx, cls in enumerate(CLASS_NAMES_6):
        cls_cm_row = cm[idx, :]
        cls_correct = cm[idx, idx]
        cls_total = cls_cm_row.sum()
        cls_acc = cls_correct / cls_total if cls_total > 0 else np.nan

        metrics_rows.append(
            {
                "Class": cls,
                "Support": int(support[idx]),
                "Accuracy": cls_acc,
                "Precision": prec[idx],
                "Recall": rec[idx],
                "F1": f1[idx],
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)

    overall_stats = {
        "Samples_Used": int(total),
        "Overall_Accuracy": overall_acc,
        "Macro_F1": macro_f1,
    }

    return metrics_df, overall_stats, cm


# -------------------------------------------------------------------
# Plotting / PDF helpers
# -------------------------------------------------------------------
def generate_6class_pdf_report(
    metrics_df: pd.DataFrame,
    overall_stats: dict,
    cm: np.ndarray,
    out_pdf: str,
    tag: str,
    preds_csv_path: str,
    df: pd.DataFrame,
):
    """
    Create a compact but readable PDF:

      Page 1: Summary + what "Others" means
      Page 2: 6x6 confusion matrix heatmap
      Page 3: Per-class F1 bar chart
      Page 4: Per-class metrics table
      Page 5: (optional) inference time boxplot if 'Inference_Time_ms' present
    """
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)

    with PdfPages(out_pdf) as pdf:
        # --------------------------------------------- Page 1: Summary
        fig, ax = plt.subplots(figsize=(8.3, 6.0))
        ax.axis("off")

        summary_lines = [
            "MedCLIP Zero-Shot Disease Classification",
            "6-class setting: CheXpert-5 + Others",
            "",
            f"Tag / image variant : {tag}",
            f"Preds CSV           : {preds_csv_path}",
            "",
            "Classes:",
            "  - CheXpert-5: Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion",
            "  - Others    : any chexpert_label outside these five",
            "",
            "How predictions are made:",
            "  - MedCLIP outputs probabilities only for the 5 CheXpert diseases.",
            "  - We take argmax over these 5 scores to get the predicted class.",
            "  - MedCLIP never predicts 'Others' (this is by design).",
            "",
            "Interpretation:",
            "  - If GT = Others, any prediction must be one of the 5 diseases.",
            "    This reveals how often MedCLIP mis-classifies non-CheXpert",
            "    pathologies into the limited disease set.",
            "",
            "Overall performance on this subset:",
            f"  - Samples used    : {overall_stats['Samples_Used']}",
            f"  - Overall accuracy: {overall_stats['Overall_Accuracy']:.3f}",
            f"  - Macro F1        : {overall_stats['Macro_F1']:.3f}",
        ]

        ax.text(
            0.03,
            0.97,
            "\n".join(summary_lines),
            va="top",
            ha="left",
            fontsize=9,
        )
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
        pdf.savefig(fig)
        plt.close(fig)

        # --------------------------------------------- Page 2: Confusion matrix
        fig, ax = plt.subplots(figsize=(7.5, 6.0))
        im = ax.imshow(cm, cmap="Blues")

        ax.set_title(
            "6-class confusion matrix (rows = GT, cols = Pred)",
            fontsize=11,
        )
        ax.set_xticks(np.arange(len(CLASS_NAMES_6)))
        ax.set_yticks(np.arange(len(CLASS_NAMES_6)))
        ax.set_xticklabels(
            [f"Pred\n{c}" for c in CLASS_NAMES_6],
            rotation=20,
            ha="right",
            fontsize=9,
        )
        ax.set_yticklabels(
            [f"GT {c}" for c in CLASS_NAMES_6],
            fontsize=9,
        )

        # Annotate cell counts
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    int(cm[i, j]),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                )

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        caption = (
            "Confusion matrix for the 6-class task (CheXpert-5 + Others).\n"
            "Diagonal cells show correct predictions. Off-diagonal cells show "
            "confusions between specific classes.\n"
            "In particular, the GT 'Others' row highlights how non-CheXpert "
            "conditions are forced into the 5 disease classes."
        )
        fig.text(
            0.02,
            0.02,
            caption,
            ha="left",
            va="bottom",
            fontsize=8,
        )
        fig.tight_layout(rect=[0.08, 0.14, 0.98, 0.93])
        pdf.savefig(fig)
        plt.close(fig)

        # --------------------------------------------- Page 3: F1 bar chart
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        f1s = metrics_df["F1"].values
        classes = metrics_df["Class"].tolist()

        bars = ax.bar(classes, f1s)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("F1 score", fontsize=10)
        ax.set_title("Per-class F1 (6-class CheXpert-5 + Others)", fontsize=11)
        ax.tick_params(axis="x", rotation=20)

        for bar, v in zip(bars, f1s):
            if not np.isnan(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + 0.01,
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        caption = (
            "F1 = 2 * (Precision * Recall) / (Precision + Recall).\n"
            "For 'Others', F1 shows how well MedCLIP avoids over-assigning "
            "non-CheXpert pathologies to the 5 target diseases."
        )
        fig.text(
            0.02,
            0.02,
            caption,
            ha="left",
            va="bottom",
            fontsize=8,
        )
        fig.tight_layout(rect=[0.06, 0.13, 0.98, 0.93])
        pdf.savefig(fig)
        plt.close(fig)

        # --------------------------------------------- Page 4: Per-class metrics table
        fig, ax = plt.subplots(figsize=(10.5, 4.8))
        ax.axis("off")
        ax.set_title(
            "Per-class metrics (6-class CheXpert-5 + Others)",
            loc="left",
            fontsize=11,
        )

        table_df = metrics_df.copy()
        # Format numeric columns
        for col in ["Accuracy", "Precision", "Recall", "F1"]:
            table_df[col] = table_df[col].apply(
                lambda x: f"{x:.3f}" if not np.isnan(x) else "NA"
            )

        table = ax.table(
            cellText=table_df.values,
            colLabels=table_df.columns,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.2)

        caption = (
            "Support = number of ground-truth samples for each class.\n"
            "Accuracy here is class-wise: fraction of samples for that class "
            "that are correctly predicted."
        )
        fig.text(
            0.02,
            0.02,
            caption,
            ha="left",
            va="bottom",
            fontsize=8,
        )
        fig.tight_layout(rect=[0.02, 0.10, 0.99, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

        # --------------------------------------------- Page 5: Inference time (optional)
        if "Inference_Time_ms" in df.columns:
            time_series = df["Inference_Time_ms"].dropna().values
            if time_series.size > 0:
                fig, ax = plt.subplots(figsize=(6.0, 4.8))
                ax.boxplot(time_series, vert=True, showmeans=True)
                ax.set_ylabel("Inference time per image (ms)", fontsize=10)
                ax.set_title(
                    "MedCLIP per-image inference latency (same CSV)",
                    fontsize=11,
                )

                caption = (
                    "Boxplot summarizing median, interquartile range, mean and "
                    "outliers for per-image inference time.\n"
                    "Useful when comparing different image variants (raw vs CLAHE vs Gauss+CLAHE)."
                )
                fig.text(
                    0.03,
                    0.02,
                    caption,
                    ha="left",
                    va="bottom",
                    fontsize=8,
                )
                fig.tight_layout(rect=[0.09, 0.15, 0.98, 0.92])
                pdf.savefig(fig)
                plt.close(fig)

    print(f"[INFO] 6-class PDF report saved to:\n       {out_pdf}")


# -------------------------------------------------------------------
# Main – CLI
# -------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Compute 6-class (CheXpert-5 + Others) metrics for MedCLIP on an IU-CXR subset "
            "and generate a PDF report."
        )
    )
    parser.add_argument(
        "--preds-csv",
        type=str,
        required=True,
        help="Path to MedCLIP predictions CSV (must contain chexpert_label and *_prob columns).",
    )
    parser.add_argument(
        "--tag",
        type=str,
        required=True,
        help="Short tag for output filenames, e.g. raw / clahe / gaussclahe.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=r"D:\MedVLMBench\phase1\results",
        help="Directory where metrics CSV, confusion matrix CSV, and PDF will be saved.",
    )

    args = parser.parse_args()
    preds_csv = args.preds_csv
    tag = args.tag
    results_dir = args.results_dir

    if not os.path.exists(preds_csv):
        raise FileNotFoundError(f"Prediction CSV not found at: {preds_csv}")

    os.makedirs(results_dir, exist_ok=True)

    print("[INFO] Reading predictions from:")
    print(f"       {preds_csv}")
    df = pd.read_csv(preds_csv)

    # Build 6-way GT and predictions
    print("[INFO] Building 6-way ground truth and predictions (CheXpert-5 + Others)...")
    gt_6, pred_6 = build_6class_labels(df)

    # Compute metrics
    print("[INFO] Computing 6-class metrics...")
    metrics_df, overall_stats, cm = compute_6class_metrics(gt_6, pred_6)

    # Save CSVs
    metrics_csv_path = os.path.join(
        results_dir, f"medclip_chexpert_6class_{tag}_metrics.csv"
    )
    cm_csv_path = os.path.join(
        results_dir, f"medclip_chexpert_6class_{tag}_confusion_matrix.csv"
    )

    metrics_df.to_csv(metrics_csv_path, index=False)
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES_6, columns=CLASS_NAMES_6)
    cm_df.to_csv(cm_csv_path)

    print("\n========== 6-class summary (CheXpert-5 + Others) ==========")
    print(f"Samples used      : {overall_stats['Samples_Used']}")
    print(f"Overall Accuracy  : {overall_stats['Overall_Accuracy']:.3f}")
    print(f"Macro F1          : {overall_stats['Macro_F1']:.3f}\n")

    print("Per-class metrics:")
    with pd.option_context("display.float_format", lambda x: f"{x:.3f}"):
        print(metrics_df.to_string(index=False))

    print("\n[INFO] 6-class per-class metrics saved to:")
    print(f"       {metrics_csv_path}")
    print("[INFO] 6-class confusion matrix saved to:")
    print(f"       {cm_csv_path}")

    # PDF report
    pdf_path = os.path.join(
        results_dir, f"medclip_chexpert_6class_{tag}_report.pdf"
    )
    print("\n[INFO] Generating 6-class PDF report...")
    generate_6class_pdf_report(
        metrics_df,
        overall_stats,
        cm,
        pdf_path,
        tag=tag,
        preds_csv_path=preds_csv,
        df=df,
    )


if __name__ == "__main__":
    main()
