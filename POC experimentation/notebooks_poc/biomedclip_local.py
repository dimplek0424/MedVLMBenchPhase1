"""
BiomedCLIP zero-shot demo (CPU-only, Windows-safe)
Strategy:
- Monkey-patch torch.nn.Module.load_state_dict to force strict=False during model load.
- Use HF hub model id to build the right architecture via open_clip.
- Restore all patches after loading; inference is normal (CPU).

Output JSON matches medclip_demo.py so the shared evaluator works.
"""

import os, json, glob
from typing import List
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F
import open_clip

# -------------------- config --------------------
IMG_ROOT = r"D:\MedVLMPhase1\data\chestxray_iu\images\images_normalized"
IMG_PATTERNS = [IMG_ROOT + r"\**\*.jpg",
                IMG_ROOT + r"\**\*.jpeg",
                IMG_ROOT + r"\**\*.png"]
LABELS = ["no finding","cardiomegaly","pulmonary edema","pneumonia",
          "atelectasis","pleural effusion","consolidation","pneumothorax"]
TEMPLATE = "a chest x-ray showing {}"
N_IMAGES = 5
OUT_JSON = r"D:\MedVLMPhase1\outputs\biomedclip_zero_shot.json"
MODEL_ID = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
# ------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ---- Patch: force non-strict load_state_dict during model construction ----
orig_load_state_dict = torch.nn.Module.load_state_dict
def _non_strict_load(self, state_dict, strict=True):
    # ignore 'strict' requested by callers; always load with strict=False
    return orig_load_state_dict(self, state_dict, strict=False)

torch.nn.Module.load_state_dict = _non_strict_load
try:
    print("[INFO] Creating model & transforms from HF (non-strict load patched)...")
    # This constructs the right text+vision towers & tries to load weights.
    model, preprocess = open_clip.create_model_from_pretrained(f"hf-hub:{MODEL_ID}", device=device)
    tokenizer = open_clip.get_tokenizer(f"hf-hub:{MODEL_ID}")
finally:
    # Always restore original behavior (important!)
    torch.nn.Module.load_state_dict = orig_load_state_dict

model = model.to(device).eval()
print("[INFO] BiomedCLIP model ready.")

# ---- Gather a few images (recursive) ----
img_paths: List[str] = []
for pat in IMG_PATTERNS:
    img_paths.extend(glob.glob(pat, recursive=True))
img_paths = img_paths[:N_IMAGES]
if not img_paths:
    raise SystemExit(f"[ERROR] No images under: {IMG_ROOT}")

print(f"[INFO] Found {len(img_paths)} image(s).")

# ---- Encode images & text ----
images = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in img_paths]).to(device)
texts  = [TEMPLATE.format(lab) for lab in LABELS]
text_inputs = tokenizer(texts).to(device)

with torch.no_grad():
    img_feats = model.encode_image(images)
    txt_feats = model.encode_text(text_inputs)
    img_feats = F.normalize(img_feats, dim=-1)
    txt_feats = F.normalize(txt_feats, dim=-1)
    sims = img_feats @ txt_feats.t()
    probs = torch.softmax(sims, dim=-1).cpu().numpy()

# ---- Print + save (same schema as medclip) ----
results = []
for i, p in enumerate(img_paths):
    pairs = list(zip(LABELS, probs[i].tolist()))
    top5 = sorted(pairs, key=lambda x: x[1], reverse=True)[:5]
    print(f"\n[RESULT] Image: {p}")
    for lab, sc in top5:
        print(f"  {lab:20s} {sc:.4f}")
    results.append({
        "image_path": p,
        "top5": [{"label": lab, "score": float(sc)} for lab, sc in top5],
        "all_scores": {lab: float(score) for lab, score in pairs},
    })

os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump({"labels": LABELS, "template": TEMPLATE, "results": results}, f, indent=2)
print(f"\n[SAVED] {OUT_JSON}")
