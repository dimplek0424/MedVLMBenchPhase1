#!/usr/bin/env python3
"""
Evaluate projection-type classifier outputs (e.g., frontal vs lateral) from MedCLIP or similar models.

This script reads a CSV of per-image probabilities and predictions, 
computes accuracy, ROC/PR curves, calibration (ECE), pairwise cosine-similarity, 
and a 2D visualization (UMAP or t-SNE).

Typical usage (after running MedCLIP projection):
-------------------------------------------------
python scripts/evaluate_views.py \
  --csv results/projection/iu_v03_full_medclip.csv \
  --outdir results/eval/medclip \
  --col_image image --col_p1 p_frontal --col_p2 p_lateral \
  --col_pred pred --label1 frontal --label2 lateral
"""

import argparse, os, pathlib, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, precision_recall_curve
)

# Try to import UMAP (if unavailable, fall back to t-SNE)
HAS_UMAP = True
try:
    import umap
except Exception:
    HAS_UMAP = False
from sklearn.manifold import TSNE


# ---------------------------------------------------------------------------
# 📏 ECE (Expected Calibration Error)
# ---------------------------------------------------------------------------
def ece(probs, correct, n_bins=10):
    """
    Compute Expected Calibration Error for a binary classifier.

    Parameters:
    -----------
    probs   : array-like, shape (N,)
              Confidence scores of the predicted class.
    correct : array-like, shape (N,)
              1 if the prediction is correct, else 0.
    n_bins  : int
              Number of equal-width bins across [0,1].

    Returns:
    --------
    ece_val : float
        Weighted mean calibration gap.
    df      : pd.DataFrame
        Per-bin statistics for calibration plotting.
    """
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
    return ece_val, df


# ---------------------------------------------------------------------------
# 🚀 MAIN FUNCTION
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate MedCLIP-like binary projection classifier results.")
    ap.add_argument("--csv", required=True, help="Input CSV containing predictions and probabilities.")
    ap.add_argument("--col_image", default="image", help="Image filename column.")
    ap.add_argument("--col_p1", default="p_frontal", help="Probability column for class #1 (e.g., frontal).")
    ap.add_argument("--col_p2", default="p_lateral", help="Probability column for class #2 (e.g., lateral).")
    ap.add_argument("--col_pred", default="pred", help="Predicted label column.")
    ap.add_argument("--col_label", default="", help="Ground-truth label column (if empty, use heuristic).")
    ap.add_argument("--label1", default="frontal", help="Name for class 1.")
    ap.add_argument("--label2", default="lateral", help="Name for class 2.")
    ap.add_argument("--outdir", default="results/eval/medclip", help="Directory to save plots and outputs.")
    ap.add_argument("--sample_sim", type=int, default=50, help="Subset size for cosine-similarity & embeddings.")
    args = ap.parse_args()

    # Ensure output folder exists
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 🧾 LOAD CSV + sanity checks
    # -----------------------------------------------------------------------
    df = pd.read_csv(args.csv)
    for col in [args.col_image, args.col_p1, args.col_p2, args.col_pred]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {args.csv}")

    # -----------------------------------------------------------------------
    # 🧩 Handle Ground-Truth (heuristic if missing)
    # -----------------------------------------------------------------------
    if not args.col_label or args.col_label not in df.columns:
        # Simple heuristic: images containing "-1001" are frontal
        df["label"] = np.where(
            df[args.col_image].astype(str).str.contains("-1001"),
            args.label1, args.label2
        )
        col_label = "label"
    else:
        col_label = args.col_label

    # Convert to numeric arrays for metrics
    y_true = (df[col_label] == args.label1).astype(int).to_numpy()
    y_pred = (df[args.col_pred] == args.label1).astype(int).to_numpy()
    p1 = df[args.col_p1].astype(float).to_numpy()
    p2 = df[args.col_p2].astype(float).to_numpy()

    # -----------------------------------------------------------------------
    # 📉 Confusion Matrix
    # -----------------------------------------------------------------------
    cm = confusion_matrix(df[col_label], df[args.col_pred], labels=[args.label1, args.label2])
    fig = plt.figure(figsize=(4, 4))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[args.label1, args.label2]).plot(
        values_format="d", cmap="Blues", ax=plt.gca(), colorbar=False
    )
    plt.title("Confusion Matrix")
    plt.tight_layout()
    fig.savefig(outdir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    # -----------------------------------------------------------------------
    # 📈 ROC & PR Curves
    # -----------------------------------------------------------------------
    auc = roc_auc_score(y_true, p1)
    fpr, tpr, _ = roc_curve(y_true, p1)
    pre, rec, _ = precision_recall_curve(y_true, p1)

    plt.figure(figsize=(4, 4))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.legend(); plt.title("ROC Curve"); plt.tight_layout()
    plt.savefig(outdir / "roc.png", dpi=160); plt.close()

    plt.figure(figsize=(4, 4))
    plt.plot(rec, pre)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curve"); plt.tight_layout()
    plt.savefig(outdir / "pr.png", dpi=160); plt.close()

    # -----------------------------------------------------------------------
    # 📏 Calibration (ECE)
    # -----------------------------------------------------------------------
    conf_pred = np.maximum(p1, p2)  # confidence of predicted class
    correct = (df[args.col_pred] == df[col_label]).astype(int).to_numpy()
    ece_val, ece_df = ece(conf_pred, correct, n_bins=10)
    ece_df.to_csv(outdir / "calibration_bins.csv", index=False)

    # -----------------------------------------------------------------------
    # 🔍 Pairwise Cosine Similarity + Embeddings
    # -----------------------------------------------------------------------
    sub = df.sample(min(args.sample_sim, len(df)), random_state=42).reset_index(drop=True)
    V = sub[[args.col_p1, args.col_p2]].to_numpy(dtype=float)

    # Normalize each vector, compute cosine similarity matrix
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    S = Vn @ Vn.T
    pd.DataFrame(S, index=sub[args.col_image], columns=sub[args.col_image]).to_csv(
        outdir / "pairwise_cosine_similarity_subset.csv"
    )

    # -----------------------------------------------------------------------
    # 🌈 2D Visualization (UMAP → t-SNE fallback)
    # -----------------------------------------------------------------------
    if HAS_UMAP:
        Z = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, random_state=42).fit_transform(V)
        method = "UMAP"
    else:
        Z = TSNE(n_components=2, init="pca", perplexity=30, random_state=42).fit_transform(V)
        method = "t-SNE"

    labels_sub = np.where(sub[col_label] == args.label1, args.label1, args.label2)
    plt.figure(figsize=(5, 4))
    for cls, marker in [(args.label1, "o"), (args.label2, "s")]:
        mask = (labels_sub == cls)
        plt.scatter(Z[mask, 0], Z[mask, 1], label=cls, marker=marker, s=18, alpha=0.75)
    plt.legend(); plt.title(f"2D Projection of Probabilities ({method}) — n={len(sub)}")
    plt.tight_layout(); plt.savefig(outdir / "embedding_2d.png", dpi=160); plt.close()

    # -----------------------------------------------------------------------
    # 🧾 Summary Output
    # -----------------------------------------------------------------------
    accuracy = float((cm[0, 0] + cm[1, 1]) / max(1, cm.sum()))
    summary = {
        "csv": str(args.csv),
        "labels": [args.label1, args.label2],
        "counts": {
            args.label1: int((df[col_label] == args.label1).sum()),
            args.label2: int((df[col_label] == args.label2).sum())
        },
        "accuracy": accuracy,
        "roc_auc": float(auc),
        "ece": float(ece_val),
        "artifacts": [
            "confusion_matrix.png", "roc.png", "pr.png",
            "calibration_bins.csv", "pairwise_cosine_similarity_subset.csv",
            "embedding_2d.png"
        ]
    }

    # Write JSON summary for reproducibility
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()