# scripts/zeroshot_disease_medclip.py
"""
Zero-shot disease classification (multi-label) using CLIP text prompts.

- Reuses the same data loader pipeline as projection_medclip.py (RGB + CLIPProcessor 224).
- For each disease, we build one or more prompts (synonyms). The class score is the
  mean cosine-similarity softmax over all prompts, aggregated per class.
- Outputs a CSV with: image, <class1>, <class2>, ..., top1, top1_prob

Usage (Kaggle):
  export DATA_DIR=/kaggle/input/chest-xrays-indiana-university
  export OUTPUT_DIR=/kaggle/working/outputs
  python scripts/zeroshot_disease_medclip.py \
    --config configs/dataset_iu_v03_full.kaggle.yaml \
    --out    $OUTPUT_DIR/disease/iu_v03_medclip_zeroshot.csv
"""

# ---- import hardening (must be first) ----
from pathlib import Path
import sys, os
repo_root = str(Path(__file__).resolve().parents[1])
if repo_root not in sys.path:
    sys.path.append(repo_root)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass
# -----------------------------------------

import argparse, csv, json, time, random
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


# ---------- config helpers ----------
def load_cfg(path: str) -> Dict:
    if path.endswith((".yaml",".yml")):
        if not _HAVE_YAML: raise RuntimeError("Please pip install pyyaml")
        return yaml.safe_load(open(path,"r"))
    if path.endswith(".json"):
        return json.load(open(path,"r"))
    raise ValueError(f"Unsupported config format: {path}")

def getenv_or_literal(v: str) -> str:
    if v is None: return v
    s = str(v)
    if s.startswith("${") and s.endswith("}"):
        return os.environ.get(s[2:-1], "")
    return s

def build_image_list(cfg: Dict) -> Tuple[List[str], List[int]]:
    """
    Reuse projections CSV solely to enumerate images (labels are not used here).
    Tolerates {image|img|filename}.
    """
    ds = cfg["dataset"]
    base_dir = Path(getenv_or_literal(ds["base_dir"]))
    root = base_dir / ds["images_subdir"]
    csv_path = base_dir / ds["projections_csv"]
    if not root.exists(): raise FileNotFoundError(f"images_subdir not found: {root}")
    if not csv_path.exists(): raise FileNotFoundError(f"projections_csv not found: {csv_path}")

    paths = []
    import csv as _csv
    with open(csv_path, "r", newline="") as f:
        reader = _csv.DictReader(f)
        cols = {k.lower(): k for k in (reader.fieldnames or [])}
        img_col = cols.get("image") or cols.get("img") or cols.get("filename")
        if not img_col: raise ValueError("projections_csv must have image/filename column")
        for row in reader:
            rel = row[img_col].strip()
            full = root / rel
            if full.exists():
                paths.append(str(full))
    if not paths: raise RuntimeError("No images enumerated from projections_csv")
    return paths, [0]*len(paths)  # dummy labels (unused)


# ---------- prompts ----------
DEFAULT_CLASSES = {
    "No Finding": [
        "a normal frontal chest x-ray radiograph with no abnormal findings",
        "a normal chest radiograph without pathology"
    ],
    "Atelectasis": [
        "a chest x-ray showing atelectasis",
        "collapse of lung segments on a chest radiograph"
    ],
    "Cardiomegaly": [
        "an enlarged cardiac silhouette on chest x-ray (cardiomegaly)",
        "a chest radiograph with cardiomegaly"
    ],
    "Consolidation": [
        "pulmonary consolidation visible on chest x-ray",
        "airspace consolidation in the lung fields on radiograph"
    ],
    "Edema": [
        "pulmonary edema pattern on chest x-ray",
        "interstitial or alveolar edema on chest radiograph"
    ],
    "Effusion": [
        "pleural effusion on chest x-ray",
        "blunting of costophrenic angle from pleural effusion"
    ],
    "Emphysema": [
        "hyperinflated lungs with emphysema on chest radiograph",
        "barrel chest emphysema appearance on x-ray"
    ],
    "Fibrosis": [
        "pulmonary fibrosis pattern on chest x-ray",
        "reticular opacities suggesting fibrosis on radiograph"
    ],
    "Infiltration": [
        "diffuse pulmonary infiltration on chest x-ray",
        "hazy lung infiltrates on radiograph"
    ],
    "Mass": [
        "pulmonary mass noted on chest x-ray",
        "a solitary lung mass on chest radiograph"
    ],
    "Nodule": [
        "pulmonary nodule on chest x-ray",
        "a small lung nodule on radiograph"
    ],
    "Pleural_Thickening": [
        "pleural thickening visible on chest x-ray",
        "irregular pleural thickening on radiograph"
    ],
    "Pneumonia": [
        "lobar pneumonia pattern on chest x-ray",
        "consolidation consistent with pneumonia on radiograph"
    ],
    "Pneumothorax": [
        "pneumothorax on chest x-ray with absent lung markings",
        "a visible pleural line indicating pneumothorax on radiograph"
    ]
}


# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="dataset YAML/JSON")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--classes_json", default="", help="optional JSON file overriding DEFAULT_CLASSES")
    ap.add_argument("--batch_text", type=int, default=64, help="batch size for encoding text prompts")
    return ap.parse_args()


# ---------- main ----------
if __name__ == "__main__":
    args = parse_args()
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    cfg = load_cfg(args.config)
    out_csv = Path(getenv_or_literal(args.out))
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # enumerate images (labels unused)
    image_paths, _ = build_image_list(cfg)

    # device
    dev_pref = (cfg.get("runtime") or {}).get("device", "auto")
    device = torch.device("cuda" if (dev_pref=="cuda" or (dev_pref=="auto" and torch.cuda.is_available())) else "cpu")

    # model + processor
    model_name = "openai/clip-vit-base-patch16"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).eval().to(device)

    # dataloader
    loader_cfg = cfg.get("loader", {})
    dl = make_loader_from_cfg(image_paths, [0]*len(image_paths), loader_cfg)  # dummy labels
    from contextlib import nullcontext
    amp_ctx = torch.cuda.amp.autocast if device.type=="cuda" else nullcontext
    dl.collate_fn = (lambda batch: (
        processor(images=[b[0] for b in batch], return_tensors="pt")["pixel_values"],
        [b[2] for b in batch]  # paths
    ))

    # classes/prompts
    classes = DEFAULT_CLASSES.copy()
    if args.classes_json:
        classes = json.load(open(args.classes_json, "r"))
    class_names = list(classes.keys())
    all_prompts = [p for k in class_names for p in classes[k]]
    class_offsets = {}  # name -> (start, end)
    idx = 0
    for k in class_names:
        start = idx; idx += len(classes[k]); end = idx
        class_offsets[k] = (start, end)

    # encode all prompts once
    with torch.no_grad():
        text_embeds = []
        for i in range(0, len(all_prompts), args.batch_text):
            chunk = all_prompts[i:i+args.batch_text]
            t = processor(text=chunk, padding=True, return_tensors="pt").to(device)
            e = model.get_text_features(**t)
            e = e / e.norm(dim=-1, keepdim=True)
            text_embeds.append(e)
        text_embeds = torch.cat(text_embeds, dim=0)  # [P, D]

    # inference loop
    rows = []
    n = 0
    t0 = time.time()
    with torch.no_grad():
        for pixel_values, paths in dl:
            pixel_values = pixel_values.to(device, non_blocking=True)
            if n == 0:
                print(f"[sanity] pixel_values.shape={tuple(pixel_values.shape)} (expect [B,3,224,224])")

            with amp_ctx():
                im = model.get_image_features(pixel_values=pixel_values)
                im = im / im.norm(dim=-1, keepdim=True)  # [B, D]
                # sim to all prompts
                logits = im @ text_embeds.t()            # [B, P]
                # convert prompt-sims to per-class probs:
                #   1) per-image softmax over all prompts
                probs_all = torch.softmax(logits, dim=-1)  # [B, P]
                #   2) per-class aggregate = mean over its prompt group
                probs_per_class = []
                for k in class_names:
                    a, b = class_offsets[k]
                    probs_per_class.append(probs_all[:, a:b].mean(dim=1, keepdim=True))
                probs_per_class = torch.cat(probs_per_class, dim=1)  # [B, C]

            P = probs_per_class.detach().cpu().float().numpy()
            for r, p in zip(paths, P):
                top_idx = int(np.argmax(p))
                rows.append({
                    "image": r,
                    **{class_names[j]: float(p[j]) for j in range(len(class_names))},
                    "top1": class_names[top_idx],
                    "top1_prob": float(p[top_idx]),
                })
            n += len(paths)

    dt = time.time() - t0
    out_fields = ["image"] + class_names + ["top1", "top1_prob"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[done] wrote: {out_csv} | rows={n} | wall={dt:.2f}s")
