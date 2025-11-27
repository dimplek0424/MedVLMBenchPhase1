"""
Evaluate & compare MedCLIP vs BiomedCLIP (CPU-friendly)

Optional ground truth (create if you want label metrics):
  D:\MedVLMBench\POC_experimentation\outputs\ground_truth.csv
  Format:
    image_path,labels
    D:\...\img1.png,atelectasis
    D:\...\img2.png,pleural effusion;pneumonia

Outputs:
  D:\MedVLMBench\POC_experimentation\outputs\eval_compare_per_image.csv   (per-image table)
  D:\MedVLMBench\POC_experimentation\outputs\eval_summary.txt             (short summary report)
"""

import os, json, math, csv
from pathlib import Path
import numpy as np
import pandas as pd

# ---------- paths ----------
ROOT = Path(r"D:\MedVLMBench")
OUT_DIR = ROOT / "POC_experimentation" / "outputs"
MED_JSON = OUT_DIR / "medclip_zero_shot.json"
BIO_JSON = OUT_DIR / "biomedclip_zero_shot.json"
GT_CSV  = OUT_DIR / "ground_truth.csv"   # optional
PER_IMAGE_CSV = OUT_DIR / "eval_compare_per_image.csv"
SUMMARY_TXT   = OUT_DIR / "eval_summary.txt"


# ---------- helpers ----------
def entropy(p):
    """Shannon entropy in nats (use /log(2) if you want bits)."""
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())

def js_divergence(p, q):
    """Jensen–Shannon divergence (symmetric, bounded)."""
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    m = 0.5 * (p + q)
    def kl(a, b): return float((a * (np.log(a) - np.log(b))).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

def cosine_similarity(a, b):
    a = np.asarray(a); b = np.asarray(b)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return float("nan")
    return float((a @ b) / (na * nb))

def load_preds(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    labels = d["labels"]
    img2vec = {}
    for r in d["results"]:
        vec = np.array([r["all_scores"][lab] for lab in labels], dtype=float)
        # re-normalize to a prob simplex just in case
        s = vec.sum()
        if s > 0: vec = vec / s
        img2vec[r["image_path"]] = vec
    return labels, img2vec

def topk(labels, vec, k=3):
    idx = np.argsort(-vec)[:k]
    return [(labels[i], float(vec[i])) for i in idx]

def load_ground_truth(gt_csv, labels):
    """Return dict image_path -> binary vector (|labels|)."""
    if not gt_csv.exists(): return None
    lab2i = {lab: i for i, lab in enumerate(labels)}
    img2y = {}
    with open(gt_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            y = np.zeros(len(labels), dtype=int)
            labs = [s.strip() for s in row["labels"].split(";") if s.strip()]
            for lab in labs:
                if lab in lab2i: y[lab2i[lab]] = 1
            img2y[row["image_path"]] = y
    return img2y

def f1_micro_macro(Y_true, Y_pred):
    # micro
    tp = int(((Y_true == 1) & (Y_pred == 1)).sum())
    fp = int(((Y_true == 0) & (Y_pred == 1)).sum())
    fn = int(((Y_true == 1) & (Y_pred == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    micro = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
    # macro
    f1s = []
    for j in range(Y_true.shape[1]):
        tpj = int(((Y_true[:, j] == 1) & (Y_pred[:, j] == 1)).sum())
        fpj = int(((Y_true[:, j] == 0) & (Y_pred[:, j] == 1)).sum())
        fnj = int(((Y_true[:, j] == 1) & (Y_pred[:, j] == 0)).sum())
        precj = tpj / (tpj + fpj) if (tpj + fpj) else 0.0
        recj  = tpj / (tpj + fnj) if (tpj + fnj) else 0.0
        f1s.append(2*precj*recj/(precj+recj) if (precj+recj) else 0.0)
    macro = float(np.mean(f1s)) if f1s else 0.0
    return float(micro), float(macro)

def topk_accuracy(pred_matrix, gt_matrix, k):
    """Multi-label: counts an image correct if any GT label in top-k."""
    ranks = np.argsort(-pred_matrix, axis=1)
    hits = 0; denom = 0
    for i in range(pred_matrix.shape[0]):
        gt_idx = np.where(gt_matrix[i] == 1)[0]
        if gt_idx.size == 0: continue
        denom += 1
        if len(set(ranks[i, :k]).intersection(set(gt_idx))) > 0:
            hits += 1
    return hits/denom if denom else float("nan")


# ---------- load predictions ----------
if not MED_JSON.exists():
    raise SystemExit(f"[ERROR] Missing {MED_JSON}")
if not BIO_JSON.exists():
    raise SystemExit(f"[ERROR] Missing {BIO_JSON}")

labels_med, med = load_preds(MED_JSON)
labels_bio, bio = load_preds(BIO_JSON)

if labels_med != labels_bio:
    raise SystemExit("[ERROR] Label spaces differ between JSON files — align them first.")

labels = labels_med

# intersect on common images (robust vs ordering differences)
common_imgs = sorted(set(med.keys()) & set(bio.keys()))
if not common_imgs:
    raise SystemExit("[ERROR] No common images across the two JSONs.")

# ---------- build per-image comparison table ----------
rows = []
med_vecs, bio_vecs = [], []

for img in common_imgs:
    v_med = med[img]
    v_bio = bio[img]
    med_vecs.append(v_med)
    bio_vecs.append(v_bio)

    med_top1 = topk(labels, v_med, 1)[0]               # (label, score)
    bio_top1 = topk(labels, v_bio, 1)[0]
    med_top3 = [l for l, _ in topk(labels, v_med, 3)]
    bio_top3 = [l for l, _ in topk(labels, v_bio, 3)]
    top3_overlap = len(set(med_top3) & set(bio_top3))

    row = {
        "image_path": img,
        "med_top1_label": med_top1[0],
        "med_top1_score": med_top1[1],
        "bio_top1_label": bio_top1[0],
        "bio_top1_score": bio_top1[1],
        "top1_agree": int(med_top1[0] == bio_top1[0]),
        "top3_overlap": int(top3_overlap),
        "cosine_sim_probs": cosine_similarity(v_med, v_bio),
        "js_divergence": js_divergence(v_med, v_bio),
        "med_entropy": entropy(v_med),
        "bio_entropy": entropy(v_bio),
        "med_maxprob": float(v_med.max()),
        "bio_maxprob": float(v_bio.max()),
    }
    rows.append(row)

df = pd.DataFrame(rows)

# ---------- aggregate summary ----------
summary = {}
summary["n_images"] = len(common_imgs)
summary["top1_agreement_rate"] = float(df["top1_agree"].mean())
summary["avg_top3_overlap"] = float(df["top3_overlap"].mean())
summary["avg_cosine_similarity"] = float(df["cosine_sim_probs"].mean())
summary["avg_js_divergence"] = float(df["js_divergence"].mean())
summary["avg_med_entropy"] = float(df["med_entropy"].mean())
summary["avg_bio_entropy"] = float(df["bio_entropy"].mean())
summary["avg_med_maxprob"] = float(df["med_maxprob"].mean())
summary["avg_bio_maxprob"] = float(df["bio_maxprob"].mean())

# ---------- optional: ground-truth metrics ----------
if GT_CSV.exists():
    # align rows to GT order
    # read GT and keep only images that are in common set
    gt_rows = []
    with open(GT_CSV, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["image_path"] in common_imgs:
                gt_rows.append(r)

    if gt_rows:
        # build matrices in the SAME order as gt_rows
        lab2i = {lab:i for i, lab in enumerate(labels)}
        Y = []
        X_med = []
        X_bio = []
        for r in gt_rows:
            img = r["image_path"]
            y = np.zeros(len(labels), dtype=int)
            labs = [s.strip() for s in r["labels"].split(";") if s.strip()]
            for lab in labs:
                if lab in lab2i: y[lab2i[lab]] = 1
            Y.append(y)
            X_med.append(med[img])
            X_bio.append(bio[img])
        Y = np.array(Y, dtype=int)
        X_med = np.array(X_med, dtype=float)
        X_bio = np.array(X_bio, dtype=float)

        # thresholded F1 (micro/macro)
        THRESH = 0.5
        Ym = (X_med >= THRESH).astype(int)
        Yb = (X_bio >= THRESH).astype(int)

        micro_m, macro_m = f1_micro_macro(Y, Ym)
        micro_b, macro_b = f1_micro_macro(Y, Yb)
        summary["gt_thresh"] = THRESH
        summary["med_f1_micro"] = micro_m
        summary["med_f1_macro"] = macro_m
        summary["bio_f1_micro"] = micro_b
        summary["bio_f1_macro"] = macro_b

        # Top-k accuracy (treat GT as set; success if any GT label in top-k)
        for k in (1, 3):
            summary[f"med_top{k}_acc"] = topk_accuracy(X_med, Y, k)
            summary[f"bio_top{k}_acc"] = topk_accuracy(X_bio, Y, k)

# ---------- write outputs ----------
OUT_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(PER_IMAGE_CSV, index=False)

with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
    f.write("=== Per-image comparison saved to: {}\n".format(PER_IMAGE_CSV))
    f.write("=== Summary ===\n")
    for k, v in summary.items():
        f.write(f"{k}: {v}\n")

print(f"[SAVED] Per-image table -> {PER_IMAGE_CSV}")
print(f"[SAVED] Summary -> {SUMMARY_TXT}")
print("\nDone.")
