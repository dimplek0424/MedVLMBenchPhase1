#!/usr/bin/env python3
import argparse, os, pathlib, json, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve, precision_recall_curve

# --- optional UMAP (auto fallback to t-SNE) ---
HAS_UMAP = True
try:
    import umap
except Exception:
    HAS_UMAP = False
from sklearn.manifold import TSNE

def ece(probs, labels, n_bins=10):
    """Simple ECE for binary probs of predicted class."""
    probs = np.asarray(probs); labels = np.asarray(labels).astype(int)
    bins = np.linspace(0,1,n_bins+1)
    ece_val, rows = 0.0, []
    for i in range(n_bins):
        m = (probs >= bins[i]) & (probs < bins[i+1] if i < n_bins-1 else probs <= bins[i+1])
        if m.sum() == 0:
            rows.append((bins[i], bins[i+1], 0, np.nan, np.nan))
            continue
        conf = probs[m].mean()
        acc  = labels[m].mean()
        ece_val += (m.mean()) * abs(conf - acc)
        rows.append((bins[i], bins[i+1], int(m.sum()), float(conf), float(acc)))
    return float(ece_val), pd.DataFrame(rows, columns=["bin_lo","bin_hi","count","conf_mean","acc_mean"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Predictions CSV path")
    ap.add_argument("--col-image", default="image")
    ap.add_argument("--col-p1", default="p_frontal", help="Probability column for class #1 (e.g., frontal)")
    ap.add_argument("--col-p2", default="p_lateral", help="Probability column for class #2 (e.g., lateral)")
    ap.add_argument("--col-pred", default="pred", help="Predicted label column (strings)")
    ap.add_argument("--col-label", default="", help="Ground-truth label column (strings). If empty, use heuristic.")
    ap.add_argument("--label1", default="frontal")
    ap.add_argument("--label2", default="lateral")
    ap.add_argument("--outdir", default="results/eval")
    ap.add_argument("--sample-sim", type=int, default=50, help="Subset size for cosine-sim table")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    # infer GT if absent (replace this with your true GT merge when ready)
    if not args.col_label or args.col_label not in df.columns:
        df["label"] = np.where(df[args.col-image].str.contains("-1001"), args.label1, args.label2)
        col_label = "label"
    else:
        col_label = args.col_label

    # numeric arrays
    y_true = (df[col_label] == args.label1).astype(int).to_numpy()
    y_pred = (df[args.col_pred] == args.label1).astype(int).to_numpy()
    p1 = df[args.col-p1].to_numpy(float)
    p2 = df[args.col-p2].to_numpy(float)

    # Confusion matrix
    cm = confusion_matrix(df[col_label], df[args.col_pred], labels=[args.label1, args.label2])
    fig = plt.figure(figsize=(4,4))
    ConfusionMatrixDisplay(cm, display_labels=[args.label1, args.label2]).plot(values_format="d", cmap="Blues", ax=plt.gca(), colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    fig.savefig(outdir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    # ROC / PR using class-1 probability
    auc = roc_auc_score(y_true, p1)
    fpr,tpr,_ = roc_curve(y_true, p1)
    pre,rec,_ = precision_recall_curve(y_true, p1)

    plt.figure(figsize=(4,4)); plt.plot(fpr,tpr); plt.plot([0,1],[0,1],'--'); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC AUC={auc:.3f}"); plt.tight_layout()
    plt.savefig(outdir / "roc.png", dpi=160); plt.close()

    plt.figure(figsize=(4,4)); plt.plot(rec,pre); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curve"); plt.tight_layout()
    plt.savefig(outdir / "pr.png", dpi=160); plt.close()

    # ECE on predicted class confidence
    conf_pred = np.maximum(p1, p2)
    ece_val, ece_df = ece(conf_pred, y_pred)  # ECE vs model’s predicted class
    ece_df.to_csv(outdir / "calibration_bins.csv", index=False)

    # Pairwise cosine similarity on a subset using [p1, p2] vectors
    sub = df.sample(min(args.sample_sim, len(df)), random_state=42).reset_index(drop=True)
    V = sub[[args.col-p1, args.col-p2]].to_numpy()
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    S = Vn @ Vn.T
    sim_path = outdir / "pairwise_cosine_similarity_subset.csv"
    pd.DataFrame(S, index=sub[args.col-image], columns=sub[args.col-image]).to_csv(sim_path)

    # 2D viz (UMAP if available, else t-SNE)
    X = V
    if HAS_UMAP:
        Z = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, random_state=42).fit_transform(X)
        method = "UMAP"
    else:
        Z = TSNE(n_components=2, init="pca", perplexity=30, random_state=42).fit_transform(X)
        method = "t-SNE"

    plt.figure(figsize=(5,4))
    for cls, marker in [(args.label1,"o"), (args.label2,"s")]:
        m = (df[col_label] == cls).to_numpy()
        plt.scatter(Z[m,0], Z[m,1], label=cls, marker=marker, s=18, alpha=0.75)
    plt.legend(); plt.title(f"2D projection of probabilities ({method})"); plt.tight_layout()
    plt.savefig(outdir / "embedding_2d.png", dpi=160); plt.close()

    # Summary JSON
    summary = {
        "csv": str(args.csv),
        "labels": [args.label1, args.label2],
        "counts": {args.label1: int((df[col_label]==args.label1).sum()),
                   args.label2: int((df[col_label]==args.label2).sum())},
        "accuracy": float((cm[0,0] + cm[1,1]) / max(1,cm.sum())),
        "roc_auc": float(auc),
        "ece": float(ece_val),
        "artifacts": ["confusion_matrix.png","roc.png","pr.png","calibration_bins.csv","pairwise_cosine_similarity_subset.csv","embedding_2d.png"]
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
