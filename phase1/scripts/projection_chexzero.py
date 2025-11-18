#!/usr/bin/env python
"""
Projection benchmark for CheXzero (view classification: frontal vs lateral).
Usage:
  python scripts/projection_chexzero.py --config configs/dataset_iu_v03_full.yaml \
      --task configs/task_projection_v01.yaml --out results/projection/iu_v03_full_chexzero.csv
"""
import argparse, os, csv, torch, yaml
from pathlib import Path
from medvlm_core.dataloader import make_loader_from_cfg
from medvlm_core.io import get_dataset_paths

@torch.no_grad()
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    ds_paths = get_dataset_paths(cfg["dataset"], os.environ.get("DATA_DIR"))
    loader = make_loader_from_cfg(cfg, split="test")

    # Load CheXzero CLIP model
    # We install/import from git (rajpurkarlab/chexzero) in the Kaggle notebook.
    from chexzero.model import load_clip  # this is the usual entry in their repo
    model, preprocess = load_clip(device=device)  # returns CLIP model + PIL transform
    model.eval()

    # Build text embeddings once
    texts = ["a chest x-ray, frontal view", "a chest x-ray, lateral view"]
    import clip  # CheXzero wraps OpenAI CLIP
    text_tokens = clip.tokenize(texts).to(device)
    text_embeds = model.encode_text(text_tokens)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "p_frontal", "p_lateral", "pred"])

        for batch in loader:
            # Your loader should give paths (and optionally preloaded tensors).
            # For CheXzero we’ll re-open images and apply their preprocess to be safe.
            paths = batch["image_path"]
            from PIL import Image

            imgs = []
            for p in paths:
                im = Image.open(p).convert("RGB")
                imgs.append(preprocess(im))

            images = torch.stack(imgs, dim=0).to(device)

            img_embeds = model.encode_image(images)
            img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)

            logits = img_embeds @ text_embeds.t()  # [B, 2]
            probs = logits.softmax(dim=-1)

            for pth, (pf, pl) in zip(paths, probs.tolist()):
                pred = "frontal" if pf >= pl else "lateral"
                writer.writerow([pth, f"{pf:.6f}", f"{pl:.6f}", pred])

    print(f"✅ Wrote {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args)
