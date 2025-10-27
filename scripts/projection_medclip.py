# make repo root importable when running as a script
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

"""
Projection-type benchmark (frontal vs lateral) using MedCLIP on FULL IU dataset.

Run:
  conda activate medvlm
  python scripts/projection_medclip.py \
      --config configs/dataset_iu_v03_full.yaml \
      --task   configs/task_projection_v01.yaml \
      --out    results/projection/iu_v03_full_medclip.csv

Output:
  - CSV: image,p_frontal,p_lateral,pred,latency_sec
  - JSON manifest alongside CSV (config hash, git commit, device, etc.)
"""

import argparse
from pathlib import Path
import yaml, torch
import numpy as np

# ⬇️ We only need MedCLIPModel (NOT MedCLIPProcessor)
from medclip import MedCLIPModel

# Core utils (your local package)
from medvlm_core.seeds import set_all
from medvlm_core.dataloader import make_loader_from_cfg
from medvlm_core.logging import write_csv, write_json, git_commit_short, config_hash
from medvlm_core.timer import wallclock, gpu_mem_mb

# ⬇️ Use tokenizer + image processor explicitly (HF)
from transformers import AutoTokenizer, CLIPProcessor


# === Helper: CHW float[0,1] → HWC uint8 RGB (Option B boundary) ===
def chw01_to_hwc_uint8(img_chw_float01: torch.Tensor) -> np.ndarray:
    """
    Convert a (3,H,W) float32 tensor in [0,1] → (H,W,3) uint8 RGB array.

    Why: MedCLIP expects CLIP-style pixel_values, which we obtain via
    a Hugging Face image processor that works on HWC uint8 RGB or PIL.
    """
    arr = img_chw_float01.detach().cpu().numpy().transpose(1, 2, 0)  # CHW → HWC
    arr = np.clip(arr * 255.0, 0, 255).astype("uint8")
    return arr


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--task",   required=True)
    ap.add_argument("--out",    required=True)
    return ap.parse_args()


def choose_device(device_cfg: str) -> str:
    if device_cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_cfg


def main():
    args = parse_args()
    cfg  = yaml.safe_load(open(args.config, "r"))
    task = yaml.safe_load(open(args.task,   "r"))

    set_all(42)
    device_cfg = cfg.get("runtime", {}).get("device", "auto")
    device = choose_device(device_cfg)

    # ----- Data (OpenCV → NumPy via our dataloader) -----
    loader = make_loader_from_cfg(cfg)

    # --- Force CPU map_location for MedCLIP weight loading (library lacks it) ---
    _orig_torch_load = torch.load  # keep original
    def _cpu_torch_load(*args, **kwargs):
        kwargs.setdefault("map_location", torch.device("cpu"))
        return _orig_torch_load(*args, **kwargs)

    # ----- Model -----
    model_id = "umich-hai/medclip-vit-base-patch16"
    model = MedCLIPModel()

    # Temporarily patch torch.load so MedCLIP weights map to CPU
    torch.load = _cpu_torch_load
    try:
        model.from_pretrained(model_id)   # downloads & loads on CPU safely
    finally:
        torch.load = _orig_torch_load     # restore original torch.load

    model = model.to(device).eval()

    # ----- Text & Image processors (replace MedCLIPProcessor) -----
    # Text branch uses Bio_ClinicalBERT tokenizer
    tokenizer     = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    # Vision branch uses CLIP ViT-B/16 image preprocessor
    img_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    queries = task["task"]["labels"]  # ["frontal chest x-ray", "lateral chest x-ray"]

    rows = [("image","p_frontal","p_lateral","pred","latency_sec")]
    mem0 = gpu_mem_mb()

    with torch.no_grad():
        for names, batch_np in loader:
            # batch_np: (B,C,H,W) float32 in [0,1] – convert to torch
            # Accept both NumPy and Torch; DataLoader may already collate to tensors
            if isinstance(batch_np, torch.Tensor):
                batch = batch_np.to(device)
            else:
                batch = torch.from_numpy(batch_np).to(device)

            # Ensure 3 channels (MedCLIP/CLIP expects RGB)
            if batch.shape[1] == 1:
                batch = batch.repeat(1, 3, 1, 1)

            for i, name in enumerate(names):
                img_t = batch[i:i+1]  # (1,3,H,W)

                # === Option B boundary: CHW float[0,1] → HWC uint8 RGB ===
                img_np = chw01_to_hwc_uint8(img_t[0])

                # 1) text → Bio_ClinicalBERT tokenizer
                text_inputs = tokenizer(
                    queries,                 # ["frontal chest x-ray", "lateral chest x-ray"]
                    padding=True,
                    return_tensors="pt"
                )

                # 2) image → CLIP image processor (resizes/normalizes to pixel_values)
                img_inputs = img_processor(
                    images=img_np,           # HWC uint8 RGB
                    return_tensors="pt"
                )

                # 3) merge & move to device
                inputs = {**text_inputs, **img_inputs}
                inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

                # Forward (MedCLIP returns logits_per_image: similarity(image, texts))
                with wallclock() as t:
                    out = model(**inputs)
                dt = round(t(), 4)

                probs = out["logits_per_image"].softmax(dim=1)[0].tolist()
                pred  = "frontal" if probs[0] >= probs[1] else "lateral"
                rows.append((name, probs[0], probs[1], pred, dt))

    # Write outputs
    out_csv = Path(args.out)
    write_csv(rows, out_csv)

    manifest = {
        "script": "projection_medclip.py",
        "git_commit": git_commit_short(),
        "config_hash": config_hash(cfg),
        "config": cfg, "task": task, "device": device,
        "gpu_max_mem_mb": gpu_mem_mb() - mem0
    }
    write_json(manifest, out_csv.with_suffix(".json"))
    print("✅ wrote:", out_csv)


if __name__ == "__main__":
    main()
