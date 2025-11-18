"""
MedCLIP zero-shot demo (CPU-only, Windows 11)
Fixes in this version:
- Force CPU map_location when loading checkpoints (avoids CUDA deserialization errors)
- Use a manual "processor" (Bio_ClinicalBERT tokenizer + CLIP-style image transforms)
- Filter tokenizer outputs to ONLY pass `input_ids` and `attention_mask`
- Set explicit max_length to silence truncation warnings

Paths assumed:
  D:\MedVLMPhase1\data\chestxray_iu\images\images_normalized\**\*.jpg|jpeg|png
"""

import os
import json
import glob
from pathlib import Path
from typing import List

from PIL import Image
import torch
import torch.nn.functional as F

# MedCLIP core (image & text encoders)
from medclip import MedCLIPModel

# Manual processor bits
from transformers import AutoTokenizer
from torchvision import transforms

# ----------------------------
# 0) Configuration
# ----------------------------
IMG_ROOT = r"D:\MedVLMPhase1\data\chestxray_iu\images\images_normalized"
IMG_PATTERNS: List[str] = [
    IMG_ROOT + r"\**\*.jpg",
    IMG_ROOT + r"\**\*.jpeg",
    IMG_ROOT + r"\**\*.png",
]
LABELS = [
    "no finding",
    "cardiomegaly",
    "pulmonary edema",
    "pneumonia",
    "atelectasis",
    "pleural effusion",
    "consolidation",
    "pneumothorax",
]
TEMPLATE = "a chest x-ray showing {}"
N_IMAGES = 5
SAVE_JSON = True
OUT_JSON = r"D:\MedVLMPhase1\outputs\medclip_zero_shot.json"

# ----------------------------
# 1) Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ----------------------------
# 2) Load MedCLIP safely on CPU
# ----------------------------
print("[INFO] Loading MedCLIP weights with CPU map_location...")
_original_torch_load = torch.load  # keep ref

def _cpu_only_load(*args, **kwargs):
    kwargs["map_location"] = torch.device("cpu")
    return _original_torch_load(*args, **kwargs)

torch.load = _cpu_only_load
try:
    model = MedCLIPModel()                             # instantiate
    model.from_pretrained("medclip-vit-base-patch16")  # DO NOT reassign (returns None)
finally:
    torch.load = _original_torch_load                  # restore

model = model.to(device).eval()
print("[INFO] MedCLIP model ready.")

# ----------------------------
# 3) Manual processor: tokenizer + image transforms
# ----------------------------
print("[INFO] Building manual processor (tokenizer + image transforms)...")
# Text tower tokenizer (Bio_ClinicalBERT)
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

# CLIP normalization used for ViT-B/16
clip_mean = (0.48145466, 0.4578275, 0.40821073)
clip_std  = (0.26862954, 0.26130258, 0.27577711)
image_tx = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=clip_mean, std=clip_std),
])
print("[INFO] Manual processor ready.")

# ----------------------------
# 4) Collect a few images (recursive)
# ----------------------------
img_paths: List[str] = []
for pat in IMG_PATTERNS:
    img_paths.extend(glob.glob(pat, recursive=True))
img_paths = img_paths[:N_IMAGES]

if not img_paths:
    raise SystemExit(
        f"[ERROR] No images found under: {IMG_ROOT}\n"
        "  (Searched recursively for .jpg/.jpeg/.png)"
    )

print(f"[INFO] Found {len(img_paths)} image(s):")
for p in img_paths:
    print("       -", p)

# ----------------------------
# 5) Encode images -> embeddings
# ----------------------------
print("[INFO] Encoding images...")
images = [Image.open(p).convert("RGB") for p in img_paths]
img_batch = torch.stack([image_tx(im) for im in images]).to(device)

with torch.no_grad():
    img_feats = model.encode_image(img_batch)   # [B, D]
    img_feats = F.normalize(img_feats, dim=-1)  # L2-normalize for cosine sim

print("[INFO] Image encoding done.")

# ----------------------------
# 6) Encode label prompts (text) -> embeddings  [CPU-safe patch]
#    - Pass ONLY input_ids + attention_mask
#    - Set max_length
#    - Temporarily monkey-patch Tensor.cuda to NO-OP (lib forces .cuda())
# ----------------------------
print("[INFO] Encoding text prompts...")
prompts = [TEMPLATE.format(lab) for lab in LABELS]
tok = tokenizer(
    prompts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=64,
)

# Keep only fields MedCLIP expects
txt_inputs = {k: v for k, v in tok.items() if k in ("input_ids", "attention_mask")}

# --- CPU-only patch: make .cuda() a no-op during encode_text ---
_orig_tensor_cuda = torch.Tensor.cuda
def _noop_cuda(self, *args, **kwargs):
    # return the tensor itself unchanged (stays on CPU)
    return self

torch.Tensor.cuda = _noop_cuda
try:
    with torch.no_grad():
        txt_feats = model.encode_text(**txt_inputs)  # [L, D], now CPU-safe
        txt_feats = F.normalize(txt_feats, dim=-1)
finally:
    # Restore original .cuda behavior immediately after
    torch.Tensor.cuda = _orig_tensor_cuda

print("[INFO] Text encoding done.")

# ----------------------------
# 7) Similarity matrix -> probabilities
# ----------------------------
sims = img_feats @ txt_feats.T              # [B, L] cosine similarities
probs = torch.softmax(sims, dim=-1).cpu()   # row-wise softmax

# ----------------------------
# 8) Print top-5 labels per image
# ----------------------------
results = []
for i, p in enumerate(img_paths):
    row = probs[i].tolist()
    pairs = list(zip(LABELS, row))
    top5 = sorted(pairs, key=lambda x: x[1], reverse=True)[:5]

    print(f"\n[RESULT] Image: {p}")
    for lab, sc in top5:
        print(f"  {lab:20s} {sc:.4f}")

    results.append({
        "image_path": p,
        "top5": [{"label": lab, "score": float(sc)} for lab, sc in top5],
        "all_scores": {lab: float(score) for lab, score in pairs},
    })

# ----------------------------
# 9) (Optional) Save a JSON artifact
# ----------------------------
if SAVE_JSON:
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "labels": LABELS,
            "template": TEMPLATE,
            "results": results,
        }, f, indent=2)
    print(f"\n[SAVED] Wrote zero-shot results to: {OUT_JSON}")
