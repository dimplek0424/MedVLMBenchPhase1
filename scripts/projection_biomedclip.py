#!/usr/bin/env python
"""
Projection benchmark for BioMedCLIP (view classification: frontal vs lateral).

Usage:
  python scripts/projection_biomedclip.py \
    --config configs/dataset_iu_v03_full.yaml \
    --task   configs/task_projection_v01.yaml \
    --out    results/projection/iu_v03_full_biomedclip.csv
"""

import argparse, os, csv, torch
from pathlib import Path
import yaml

# --- make repo importable when launched from anywhere ---
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medvlm_core.dataloader import make_loader_from_cfg
from medvlm_core.io import get_dataset_paths


def build_biomedclip(device: str = "cuda"):
    """
    Load BioMedCLIP via open-clip.
    Override via env:
      BIOMEDCLIP_MODEL      (default: 'ViT-B-16')
      BIOMEDCLIP_PRETRAINED (default: 'biomed_clip')
      BIOMEDCLIP_HF_REPO    (optional HF repo id if needed)
    """
    import open_clip
    model_name = os.getenv("BIOMEDCLIP_MODEL", "ViT-B-16")
    pretrained = os.getenv("BIOMEDCLIP_PRETRAINED", "biomed_clip")
    hf_repo    = os.getenv("BIOMEDCLIP_HF_REPO")

    if hf_repo:
        model, _pt, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, pretrained_hf_repo_id=hf_repo, device=device
        )
    else:
        model, _pt, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, preprocess, tokenizer


@torch.no_grad()
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Respect DATA_DIR if set (Kaggle); override base_dir in config
    ds_cfg  = dict(cfg.get("dataset", {}))
    env_dir = os.environ.get("DATA_DIR")
    if env_dir:
        ds_cfg["base_dir"] = env_dir
    cfg["dataset"] = ds_cfg

    # Validate/resolve dataset paths (returns dict; we don't strictly need it later)
    # Optional: validate paths, but don't block if some keys (e.g., reports_csv) are absent for this task
    try:
        _ = get_dataset_paths(cfg["dataset"])
    except Exception as e:
        print("⚠️ Skipping dataset path validation:", repr(e))

    # Loader (split can be adjusted by your task if desired)
    loader = make_loader_from_cfg(cfg, split="test")

    # Model + text prompts
    model, preprocess, tokenizer = build_biomedclip(device)
    texts = ["a chest x-ray, frontal view", "a chest x-ray, lateral view"]
    text_tokens = tokenizer(texts).to(device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["image", "p_frontal", "p_lateral", "pred"])

        for batch in loader:
            # --- robust batch extraction across different dataloader returns ---
            images = None
            paths  = None

            if isinstance(batch, dict):
                images = batch.get("image") or batch.get("images")
                paths  = batch.get("image_path") or batch.get("paths") or batch.get("path")
            elif isinstance(batch, (list, tuple)):
                # Common patterns:
                #   (images, labels, meta{image_path: [...]})
                #   (images, paths)
                if len(batch) >= 3 and isinstance(batch[2], dict) and (
                    "image_path" in batch[2] or "paths" in batch[2] or "path" in batch[2]
                ):
                    images = batch[0]
                    meta   = batch[2]
                    paths  = meta.get("image_path") or meta.get("paths") or meta.get("path")
                elif len(batch) >= 2 and isinstance(batch[1], (list, tuple)):
                    images, paths = batch[0], batch[1]
                else:
                    images = batch[0]
                    paths  = [f"idx_{i}" for i in range(images.shape[0])]
            else:
                raise RuntimeError("Unsupported batch type from dataloader")

            if images is None:
                raise RuntimeError("Could not find 'image' tensor in batch")
            images = images.to(device)

            # --- BioMedCLIP forward ---
            img_feats = model.encode_image(images)
            txt_feats = model.encode_text(text_tokens)

            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)

            logits = img_feats @ txt_feats.t()   # [B, 2]
            probs  = logits.softmax(dim=-1)      # [B, 2]

            for pth, (pf, pl) in zip(paths, probs.tolist()):
                pred = "frontal" if pf >= pl else "lateral"
                writer.writerow([pth, f"{pf:.6f}", f"{pl:.6f}", pred])

    print(f"✅ Wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--task",   required=True)  # kept for symmetry; not used here
    ap.add_argument("--out",    required=True)
    args = ap.parse_args()
    main(args)
