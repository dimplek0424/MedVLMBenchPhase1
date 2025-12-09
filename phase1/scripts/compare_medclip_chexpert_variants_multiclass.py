"""
Compare MedCLIP CheXpert performance across 3 image variants (single-label, multiclass):

  1) RAW          (no CLAHE / Gaussian in this experiment)
  2) CLAHE
  3) Gauss+CLAHE

Assumptions:
  - Each prediction CSV comes from eval_medclip_disease_sanity_chexpert_subset.py
  - Columns include:
        chexpert_label
        Inference_Time_ms
        <Disease>_prob  for each MedCLIP CheXpert class
    (e.g., "Atelectasis_prob", "Cardiomegaly_prob", ...)

Evaluation mode (Option A, single-label):
  - For each row:
        true class  = chexpert_label
        pred class  = argmax over <Disease>_prob
  - Metrics:
        - Overall accuracy per variant
        - Per-class accuracy per variant
        - Mean / median inference time per variant

We do NOT use *_label columns here; we trust chexpert_label as the ground truth.

Example usage (PowerShell):

  python -m phase1.scripts.compare_medclip_chexpert_variants_multiclass `
    --raw-csv   phase1\\results\\medclip_chexpert_sanity_raw_preds.csv `
    --clahe-csv phase1\\results\\medclip_chexpert_sanity_clahe_preds.csv `
    --gauss-csv phase1\\results\\medclip_chexpert_sanity_gauss_clahe_preds.csv
"""

import argparse
import pandas as pd
import numpy as np


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--raw-csv",
        type=str,
        required=True,
        help="Prediction CSV for RAW images.",
    )
    ap.add_argument(
        "--clahe-csv",
        type=str,
        required=True,
        help="Prediction CSV for CLAHE images.",
    )
    ap.add_argument(
        "--gauss-csv",
        type=str,
        required=True,
        help="Prediction CSV for Gauss+CLAHE images.",
    )

    return ap.parse_args()


def detect_diseases(df):
    """
    Infer disease class names from *_prob columns.
    E.g. 'Atelectasis_prob' -> 'Atelectasis'
    """
    prob_cols = [c for c in df.columns if c.endswith("_prob")]
    if not prob_cols:
        raise ValueError("No *_prob columns found in CSV; cannot detect disease classes.")

    # Keep stable / sorted order
    prob_cols = sorted(prob_cols)
    diseases = [c[:-5] for c in prob_cols]  # strip "_prob"
    return diseases, prob_cols


def compute_multiclass_metrics(df, variant_name, diseases, prob_cols):
    """
    Single-label multiclass evaluation.

    - Uses 'chexpert_label' as ground truth.
    - Filters to rows where chexpert_label is one of the known diseases.
    - Pred class = argmax over prob_cols.
    - Returns overall accuracy, per-class accuracy, n_eval, and inference time stats.
    """
    if "chexpert_label" not in df.columns:
        raise ValueError(f"{variant_name}: CSV missing 'chexpert_label' column.")

    # Filter to rows where label is one of the modeled diseases
    df_eval = df[df["chexpert_label"].isin(diseases)].copy()
    if df_eval.empty:
        raise RuntimeError(f"{variant_name}: No rows with chexpert_label in {diseases}.")

    label_to_idx = {d: i for i, d in enumerate(diseases)}

    # True indices
    y_true_idx = df_eval["chexpert_label"].map(label_to_idx).values

    # Predicted indices: argmax over predicted probabilities
    probs = df_eval[prob_cols].values  # shape [N, C]
    y_pred_idx = probs.argmax(axis=1)

    correct = (y_true_idx == y_pred_idx).astype(int)
    overall_acc = correct.mean()

    # Per-class accuracy
    per_class_acc = {}
    per_class_n = {}
    for i, d in enumerate(diseases):
        mask = y_true_idx == i
        n_d = mask.sum()
        per_class_n[d] = int(n_d)
        if n_d == 0:
            per_class_acc[d] = np.nan
        else:
            per_class_acc[d] = correct[mask].mean()

    # Inference time stats
    if "Inference_Time_ms" in df_eval.columns:
        inf_mean = float(df_eval["Inference_Time_ms"].mean())
        inf_median = float(df_eval["Inference_Time_ms"].median())
    else:
        inf_mean = float("nan")
        inf_median = float("nan")

    return {
        "overall_acc": float(overall_acc),
        "per_class_acc": per_class_acc,
        "per_class_n": per_class_n,
        "n_eval": int(len(df_eval)),
        "inf_mean_ms": inf_mean,
        "inf_median_ms": inf_median,
    }


def main():
    args = parse_args()

    # Load all three
    df_raw = pd.read_csv(args.raw_csv)
    df_clahe = pd.read_csv(args.clahe_csv)
    df_gauss = pd.read_csv(args.gauss_csv)

    # Detect disease classes from RAW CSV (assume same across all)
    diseases, prob_cols = detect_diseases(df_raw)
    print("[INFO] Detected disease classes (from RAW CSV):")
    for d in diseases:
        print("  -", d)
    print()

    # Compute metrics per variant
    metrics = {}

    metrics["RAW"] = compute_multiclass_metrics(
        df_raw, "RAW", diseases, prob_cols
    )
    metrics["CLAHE"] = compute_multiclass_metrics(
        df_clahe, "CLAHE", diseases, prob_cols
    )
    metrics["Gauss+CLAHE"] = compute_multiclass_metrics(
        df_gauss, "Gauss+CLAHE", diseases, prob_cols
    )

    # ---------------- Print summary ----------------
    print("========== Overall Accuracy (single-label, multiclass) ==========")
    for variant in ["RAW", "CLAHE", "Gauss+CLAHE"]:
        m = metrics[variant]
        print(
            f"{variant:12s} | "
            f"overall_acc = {m['overall_acc']:.3f} | "
            f"n_eval = {m['n_eval']:3d} | "
            f"mean_inf = {m['inf_mean_ms']:.1f} ms | "
            f"median_inf = {m['inf_median_ms']:.1f} ms"
        )
    print()

    # Build per-class comparison table
    rows = []
    for d in diseases:
        row = {
            "Disease": d,
            "RAW_acc": metrics["RAW"]["per_class_acc"][d],
            "RAW_n":   metrics["RAW"]["per_class_n"][d],
            "CLAHE_acc": metrics["CLAHE"]["per_class_acc"][d],
            "CLAHE_n":   metrics["CLAHE"]["per_class_n"][d],
            "Gauss+CLAHE_acc": metrics["Gauss+CLAHE"]["per_class_acc"][d],
            "Gauss+CLAHE_n":   metrics["Gauss+CLAHE"]["per_class_n"][d],
        }
        rows.append(row)

    # Add overall row
    rows.append(
        {
            "Disease": "OVERALL",
            "RAW_acc": metrics["RAW"]["overall_acc"],
            "RAW_n":   metrics["RAW"]["n_eval"],
            "CLAHE_acc": metrics["CLAHE"]["overall_acc"],
            "CLAHE_n":   metrics["CLAHE"]["n_eval"],
            "Gauss+CLAHE_acc": metrics["Gauss+CLAHE"]["overall_acc"],
            "Gauss+CLAHE_n":   metrics["Gauss+CLAHE"]["n_eval"],
        }
    )

    df_res = pd.DataFrame(rows)

    print("========== Per-class Accuracy Comparison (single-label) ==========")
    # nice formatting: 3 decimals for acc
    with pd.option_context("display.float_format", lambda x: f"{x:.3f}"):
        print(df_res.to_string(index=False))


if __name__ == "__main__":
    main()
