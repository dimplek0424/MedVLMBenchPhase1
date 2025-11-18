#!/usr/bin/env python3
"""
Evaluate projection-type outputs (frontal vs lateral) from MedCLIP-style runs.

Reads a CSV with columns:
  image, p_frontal, p_lateral, pred
Joins with IU-CXR ground truth at:
  <data_dir>/indiana_projections.csv
…tolerating header variants {filename|image|image index|path} and {projection|view|view position}.

Saves:
  confusion_matrix.png, roc.png, pr.png, calibration_bins.csv, summary.json

Example (Kaggle):
  python scripts/evaluate_views.py \
    --csv     $OUTPUT_DIR/projection/iu_v03_medclip.csv \
    --data_dir $DATA_DIR \
    --outdir  $OUTPUT_DIR/eval_medclip
"""

import argparse, os, json, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
)

# ------------------------------
# ECE (Expected Calibration Error)
# ------------------------------
def ece(probs, correct, n_bins=10):
    probs = np.asarray(probs, dtype=float)
    correct = np.asarray(correct, dtype=int)
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val, rows = 0.0, []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        if mask.sum() == 0:
            rows.append((lo, hi, 0, np.nan, np.nan))
            continue
        conf_mean = float(probs[mask].mean())
        acc_mean = float(correct[mask].mean())
        ece_val += float(mask.mean()) * abs(conf_mean - acc_mean)
        rows.append((lo, hi, int(mask.sum()), conf_mean, acc_mean))
    df = pd.DataFrame(rows, columns=["bin_lo", "bin_hi", "count", "conf_mean", "acc_mean"])
    return float(ece_val), df


def map_view(v: str) -> int | None:
    v = str(v).strip().lower()
    if v in {"frontal", "ap", "pa"}: return 0
    if v in {"lateral", "lat"}:      return 1
    return None


def load_joined(csv_pred: str, data_dir: str) -> pd.DataFrame:
    """Returns df with columns: image, p_frontal, p_lateral, pred, label (0/1)."""
    preds = pd.read_csv(csv_pred).copy()
    if not {"image", "p_frontal", "p_lateral", "pred"}.issubset(preds.columns):
        raise ValueError(f"Expected columns [image,p_frontal,p_lateral,pred] in {csv_pred}")
    preds["fname"] = preds["image"].apply(lambda p: os.path.basename(str(p)).strip())

    gt_csv = os.path.join(data_dir, "indiana_projections.csv")
    gt = pd.read_csv(gt_csv).copy()
    cols = {c.lower(): c for c in gt.columns}
    img_col  = cols.get("filename") or cols.get("image") or cols.get("image index") or cols.get("path")
    view_col = cols.get("projection") or cols.get("view") or cols.get("view position")
    if not img_col or not view_col:
        raise ValueError(f"Unexpected GT columns in {gt_csv}: {gt.columns.tolist()}")

    gt["fname"] = gt[img_col].astype(str).apply(lambda p: os.path.basename(p).strip())
    gt["label"] = gt[view_col].map(map_view)
    gt = gt.dropna(subset=["label"]).copy()
    gt["label"] = gt["label"].astype(int)

    df = preds.merge(gt[["fname", "label"]], on="fname", how="inner")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Predictions CSV path.")
    ap.add_argument("--data_dir", default=os.environ.get("DATA_DIR", ""), help="Dataset root containing indiana_projections.csv")
    ap.add_argument("--outdir", required=True, help="Directory to save artifacts.")
    ap.add_argument("--positive", default="lateral", choices=["lateral", "frontal"], help="Which class is positive for ROC/PR.")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not args.data_dir:
        raise ValueError("Please provide --data_dir or set DATA_DIR environment variable.")

    # Join preds with ground truth
    df = load_joined(args.csv, args.data_dir)

    # Prepare targets/scores
    # label: 0=frontal,1=lateral
    y_true = df["label"].values
    p_lat  = df["p_lateral"].astype(float).values
    p_fro  = df["p_frontal"].astype(float).values
    if args.positive == "lateral":
        y_score = p_lat
    else:
        y_score = p_fro

    y_pred = (p_lat >= p_fro).astype(int)

    # Metrics
    acc   = accuracy_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_score)
    aupr  = average_precision_score(y_true, y_score)
    cm    = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print(f"rows={len(df)}  acc={acc:.4f}  auroc={auroc:.4f}  aupr={aupr:.4f}")
    print("Confusion matrix [rows=true 0,1; cols=pred 0,1]:\n", cm)

    # Save confusion matrix
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (0=frontal,1=lateral)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(outdir / "confusion_matrix.png", dpi=150)
    plt.close()

    # ROC & PR
    RocCurveDisplay.from_predictions(y_true, y_score)
    plt.title(f"ROC (pos={args.positive})")
    plt.tight_layout()
    plt.savefig(outdir / "roc.png", dpi=150)
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_true, y_score)
    plt.title(f"PR (pos={args.positive})")
    plt.tight_layout()
    plt.savefig(outdir / "pr.png", dpi=150)
    plt.close()

    # Calibration (ECE) using confidence of predicted class
    conf = np.maximum(p_lat, p_fro)
    correct = (y_pred == y_true).astype(int)
    ece_val, ece_df = ece(conf, correct, n_bins=10)
    ece_df.to_csv(outdir / "calibration_bins.csv", index=False)

    # Summary JSON
    summary = {
        "rows": int(len(df)),
        "accuracy": float(acc),
        "auroc": float(auroc),
        "aupr": float(aupr),
        "ece": float(ece_val),
        "artifacts": [
            "confusion_matrix.png", "roc.png", "pr.png", "calibration_bins.csv"
        ],
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
