"""
Aggregate per-model CSVs and (optionally) compute accuracy/confusion if GT exists.

Run:
  python scripts/aggregate_projection.py \
      --config configs/dataset_iu_v03_full.yaml \
      --task   configs/task_projection_v01.yaml \
      --dir    results/projection
"""

import argparse
from pathlib import Path
import yaml, pandas as pd

from medvlm_core.metrics import projection_accuracy, projection_confusion
from medvlm_core.logging import write_json

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--task",   required=True)
    ap.add_argument("--dir",    required=True)
    return ap.parse_args()

def main():
    args = parse_args()
    cfg  = yaml.safe_load(open(args.config))
    task = yaml.safe_load(open(args.task))
    out_dir = Path(args.dir)

    m = pd.read_csv(out_dir/"iu_v03_full_medclip.csv")
    b = pd.read_csv(out_dir/"iu_v03_full_biomedclip.csv")
    c = pd.read_csv(out_dir/"iu_v03_full_chexzero.csv")

    m = m.rename(columns={"pred":"pred_medclip"})
    b = b.rename(columns={"pred":"pred_biomed"})
    c = c.rename(columns={"pred":"pred_chexzero"})

    merged = m.merge(b, on="image").merge(c, on="image")
    merged.to_csv(out_dir/"iu_v03_full_merged.csv", index=False)
    print("✅ merged:", out_dir/"iu_v03_full_merged.csv")

    # If weak labels provided, compute accuracy & confusion
    gt_path = task["task"].get("weak_labels_csv", None)
    summary = {}
    if gt_path and Path(gt_path).exists():
        gt = pd.read_csv(gt_path)  # expects columns: image_id, projection
        df = merged.merge(gt, left_on="image", right_on="image_id", how="left")
        df = df.dropna(subset=["projection"])

        y = df["projection"]
        summary = {
            "acc_medclip":  projection_accuracy(y, df["pred_medclip"]),
            "acc_biomed":   projection_accuracy(y, df["pred_biomed"]),
            "acc_chexzero": projection_accuracy(y, df["pred_chexzero"]),
            "cm_medclip":   projection_confusion(y, df["pred_medclip"]),
            "cm_biomed":    projection_confusion(y, df["pred_biomed"]),
            "cm_chexzero":  projection_confusion(y, df["pred_chexzero"]),
            "n": int(len(y))
        }
        write_json(summary, out_dir/"iu_v03_full_summary.json")
        print("✅ summary:", out_dir/"iu_v03_full_summary.json")
    else:
        print("ℹ️ No ground-truth CSV provided; skipped accuracy/confusion.")

if __name__ == "__main__":
    main()
