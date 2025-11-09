# scripts/projection_medclip.py
"""
MedCLIP projection benchmark (frontal vs lateral) with CLIP-faithful preprocessing.

- Reads dataset config (paths + CSVs) and loader config.
- Enumerates IU-CXR images via projections CSV (expects columns: image, view).
- Builds absolute paths: base_dir / images_subdir / <image>
- DataLoader returns raw RGB PIL images; CLIPProcessor applies resize/crop/normalize (224).
- Zero-shot: two prompts → cosine-sim logits → softmax → p_frontal/p_lateral.
- Writes CSV: [image, p_frontal, p_lateral, pred]

Usage (local):
  python scripts/projection_medclip.py \
    --config configs/dataset_iu_v03_full.yaml \
    --out    D:/MedVLMPhase1/outputs/projection/iu_v03_medclip.csv

Usage (Kaggle):
  export DATA_DIR=/kaggle/input/chest-xrays-indiana-university
  export OUTPUT_DIR=/kaggle/working/outputs
  python scripts/projection_medclip.py \
    --config configs/dataset_iu_v03_full.kaggle.yaml \
    --out    $OUTPUT_DIR/projection/iu_v03_medclip.csv
"""

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

from medvlm_core.dataloader import make_loader_from_cfg

try:
    import yaml
    _HAVE_YAML = True
except Exception:
    _HAVE_YAML = False


# ---------------------------
# IO helpers
# ---------------------------
def load_cfg(path: str) -> Dict:
    if path.endswith(".json"):
        with open(path, "r") as f:
            return json.load(f)
    if path.endswith(".yaml") or path.endswith(".yml"):
        if not _HAVE_YAML:
            raise RuntimeError("Please install pyyaml to use YAML configs.")
        with open(path, "r") as f:
            return yaml.safe_load(f)
    raise ValueError(f"Unsupported config format: {path}")


def getenv_or_literal(value: str) -> str:
    """Expands ${ENV} in config values if present."""
    if value is None:
        return value
    v = str(value)
    if v.startswith("${") and v.endswith("}"):
        env_name = v[2:-1]
        return os.environ.get(env_name, "")
    return v


def build_image_list(cfg: Dict) -> Tuple[List[str], List[int]]:
    """
    Reads projections_csv; expects columns: image, view
    - view ∈ {frontal, lateral, AP, PA, LATERAL} (we map to {0,1})
    Returns absolute image paths and integer labels: 0=frontal, 1=lateral
    """
    ds = cfg["dataset"]
    base_dir = Path(getenv_or_literal(ds["base_dir"]))
    images_subdir = ds["images_subdir"]
    projections_csv = ds["projections_csv"]

    csv_path = base_dir / projections_csv
    images_root = base_dir / images_subdir

    if not csv_path.exists():
        raise FileNotFoundError(f"projections_csv not found: {csv_path}")
    if not images_root.exists():
        raise FileNotFoundError(f"images_subdir not found: {images_root}")

    paths: List[str] = []
    labels: List[int] = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        # tolerate different header casings
        cols = {k.lower(): k for k in reader.fieldnames or []}
        img_col = cols.get("image") or cols.get("img") or cols.get("filename") or None
        view_col = cols.get("view") or cols.get("projection") or None
        if not img_col or not view_col:
            raise ValueError("projections_csv must have columns like {image, view}")

        for row in reader:
            rel = row[img_col].strip()
            view = row[view_col].strip().lower()
            # Normalize common synonyms
            if view in ("frontal", "ap", "pa"):
                y = 0
            elif view in ("lateral", "lat"):
                y = 1
            else:
                # skip unknown labels
                continue
            full = images_root / rel
            if full.exists():
                paths.append(str(full))
                labels.append(y)

    if len(paths) == 0:
        raise RuntimeError("No images matched from projections_csv; check paths/columns.")

    return paths, labels


# ---------------------------
# Collate: CLIPProcessor
# ---------------------------
def make_collate_fn(processor: CLIPProcessor):
    """
    Input batch is a list of tuples: (PIL.Image (RGB), label:int, path:str)
    Returns: pixel_values [B,3,224,224], labels [B], paths [B]
    """
    def _collate(batch):
        imgs, labels, paths = zip(*batch)
        proc = processor(images=list(imgs), return_tensors="pt")
        pixel_values = proc["pixel_values"]  # [B,3,224,224], CLIP mean/std in place
        labels = torch.tensor(labels, dtype=torch.long)
        return pixel_values, labels, list(paths)
    return _collate


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="dataset YAML/JSON")
    ap.add_argument("--out", required=True, help="output CSV path")
    return ap.parse_args()


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    args = parse_args()

    # seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    cfg = load_cfg(args.config)
    loader_cfg = cfg.get("loader", {})
    out_csv_path = Path(getenv_or_literal(args.out))
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Build image list + integer labels from projections CSV
    image_paths, labels = build_image_list(cfg)

    # Prepare model and processor (CLIP ViT-B/16 → matches MedCLIP vision preprocessing)
    model_name = "openai/clip-vit-base-patch16"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).eval()

    # Device handling
    dev_pref = (cfg.get("runtime") or {}).get("device", "auto")
    if dev_pref == "cuda" or (dev_pref == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = model.to(device)

    # DataLoader (dataset returns raw RGB; processor runs in collate)
    loader = make_loader_from_cfg(image_paths, labels, loader_cfg)
    loader.collate_fn = make_collate_fn(processor)

    # Pre-encode text prompts once (zero-shot)
    prompt_texts = [
        "a frontal chest x-ray radiograph",
        "a lateral chest x-ray radiograph",
    ]
    text_inputs = processor(text=prompt_texts, padding=True, return_tensors="pt").to(device)

    # Speed knob: autotune convolution algorithms for fixed 224×224
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Inference loop
    rows = []
    n = 0
    t0 = time.time()

    # Use AMP on GPU for speed
    amp_ctx = torch.cuda.amp.autocast if device.type == "cuda" else torch.cpu.amp.autocast
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    with torch.no_grad():
        for pixel_values, y, paths in loader:
            pixel_values = pixel_values.to(device, non_blocking=True)

            if n == 0:
                print(f"[sanity] pixel_values.shape={tuple(pixel_values.shape)} (expect [B,3,224,224])")

            with amp_ctx(dtype=amp_dtype):
                image_features = model.get_image_features(pixel_values=pixel_values)
                text_features = model.get_text_features(**text_inputs)
                # cosine-sim style logits
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                logits = image_features @ text_features.t()  # [B,2]
                probs = torch.softmax(logits, dim=-1)        # [:,0]=frontal, [:,1]=lateral

            probs_np = probs.detach().float().cpu().numpy()
            for i, p in enumerate(paths):
                rows.append({
                    "image": p,
                    "p_frontal": float(probs_np[i, 0]),
                    "p_lateral": float(probs_np[i, 1]),
                    "pred": "frontal" if probs_np[i, 0] >= probs_np[i, 1] else "lateral",
                })
            n += len(paths)

    # Write CSV
    with open(out_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "p_frontal", "p_lateral", "pred"])
        w.writeheader()
        w.writerows(rows)

    dt = time.time() - t0
    print(f"[done] wrote: {out_csv_path} | rows={n} | wall={dt:.2f}s")
