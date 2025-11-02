#!/usr/bin/env python
"""
Projection benchmark for BioMedCLIP (view classification: frontal vs lateral).
Usage:
  python scripts/projection_biomedclip.py --config configs/dataset_iu_v03_full.yaml \
      --task configs/task_projection_v01.yaml --out results/projection/iu_v03_full_biomedclip.csv
"""

import argparse, os, csv, math, torch
from pathlib import Path
from medvlm_core.dataloader import make_loader_from_cfg
from medvlm_core.io import get_dataset_paths
import yaml

def build_biomedclip(device: str = "cuda"):
    """
    Tries to load BioMedCLIP via open-clip. You can override model/ckpt via env:
      BIOMEDCLIP_MODEL (default: 'ViT-B-16')
      BIOMEDCLIP_PRETRAINED (default: 'biomed_clip')
    If your Kaggle image can’t find weights, set BIOMEDCLIP_HF_REPO (e.g., microsoft/BiomedCLIP-PubMedBERT-base-uncased-ViT-B-16)
    and we’ll try to load via open-clip’s hf repo arg.
    """
    import open_clip
    model_name = os.getenv("BIOMEDCLIP_MODEL", "ViT-B-16")
    pretrained = os.getenv("BIOMEDCLIP_PRETRAINED", "biomed_clip")
    hf_repo = os.getenv("BIOMEDCLIP_HF_REPO", None)

    if hf_repo:
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, pretrained_hf_repo_id=hf_repo, device=device
        )
        tokenizer = open_clip.get_tokenizer(model_name)
    else:
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        tokenizer = open_clip.get_tokenizer(model_name)

    model.eval()
    return model, preprocess_val, tokenizer

@torch.no_grad()
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # common dataset loader (expects cfg["dataset"] with base_dir, splits, etc.)
    ds_paths = get_dataset_paths(cfg["dataset"], os.environ.get("DATA_DIR"))
    loader = make_loader_from_cfg(cfg, split="test")  # or the split your task config defines

    model, preprocess, tokenizer = build_biomedclip(device)

    # Prompts (edit as desired)
    texts = ["a chest x-ray, frontal view", "a chest x-ray, lateral view"]
    text_tokens = tokenizer(texts).to(device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "p_frontal", "p_lateral", "pred"])

        for batch in loader:
            # Expect batch["image_path"] or similar from your dataloader;
            # adjust if your loader returns (img_tensor, label, meta)
            images = batch["image"]  # tensor [B, C, H, W] from your loader
            paths  = batch["image_path"]
            images = images.to(device)

            # BioMedCLIP forward
            # open-clip uses encode_image / encode_text
            img_feats = model.encode_image(images)
            txt_feats = model.encode_text(text_tokens)

            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)

            # cosine sims -> softmax
            logits = img_feats @ txt_feats.t()  # [B, 2]
            probs = logits.softmax(dim=-1)      # [B, 2]

            for pth, (pf, pl) in zip(paths, probs.tolist()):
                pred = "frontal" if pf >= pl else "lateral"
                writer.writerow([pth, f"{pf:.6f}", f"{pl:.6f}", pred])

    print(f"✅ Wrote {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--task", required=True)  # kept for symmetry; not used here
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args)