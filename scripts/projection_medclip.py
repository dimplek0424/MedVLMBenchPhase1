#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ──────────────────────────────────────────────────────────────────────────────
# Projection-type benchmark (frontal vs lateral) using MedCLIP on FULL IU dataset
#
# Run from repo root (Kaggle or local):
#   python scripts/projection_medclip.py \
#       --config configs/dataset_iu_v03_full.yaml \
#       --task   configs/task_projection_v01.yaml \
#       --out    results/projection/iu_v03_full_medclip.csv
#
# Output:
#   - CSV  : image,p_frontal,p_lateral,pred,latency_sec
#   - JSON : manifest (config hash, git commit, device, GPU mem delta)
#
# Design notes:
#   • We avoid Hugging Face auth/gated models entirely. MedCLIP’s own
#     `from_pretrained()` (no args) downloads weights from GCS.
#   • We patch torch.load → map_location="cpu" during weight load to avoid
#     CUDA incompatibility on Kaggle; then move the model to device.
#   • We use HF AutoTokenizer + CLIPProcessor explicitly instead of the
#     MedCLIPProcessor (more stable across versions).
#   • Data comes from your medvlm_core DataLoader (float[0,1], CHW).
#     We convert each sample to HWC uint8 RGB before CLIPProcessor.
# ──────────────────────────────────────────────────────────────────────────────

# -- Make repo root importable so "medvlm_core.*" works when run as a script
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path
import yaml
import numpy as np
import torch

# MedCLIP
from medclip import MedCLIPModel
from transformers import AutoTokenizer, CLIPProcessor

# repo utilities
from medvlm_core.seeds import set_all
from medvlm_core.dataloader import make_loader_from_cfg
from medvlm_core.logging import write_csv, write_json, git_commit_short, config_hash
from medvlm_core.timer import wallclock, gpu_mem_mb


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def chw01_to_hwc_uint8(img_chw_float01: torch.Tensor) -> np.ndarray:
    """
    Convert a single image from (C,H,W) float32 in [0,1] → (H,W,3) uint8 RGB.

    Why:
      CLIPProcessor expects PIL/HWC uint8 or similar; our loader gives CHW [0,1].
    """
    arr = img_chw_float01.detach().cpu().numpy().transpose(1, 2, 0)  # CHW → HWC
    arr = np.clip(arr * 255.0, 0, 255).astype("uint8")
    return arr


def choose_device(device_cfg: str) -> str:
    """Resolve 'auto' to 'cuda' when available; otherwise return given cfg."""
    if device_cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_cfg


def tolerant_load_medclip() -> MedCLIPModel:
    """
    Load MedCLIP weights via its internal GCS zip (no HF calls, no token).
    We temporarily patch torch.load to force map_location='cpu' for safety.
    """
    _orig_torch_load = torch.load

    def _cpu_torch_load(*args, **kwargs):
        kwargs.setdefault("map_location", torch.device("cpu"))
        return _orig_torch_load(*args, **kwargs)

    model = MedCLIPModel()
    torch.load = _cpu_torch_load
    try:
        state = model.from_pretrained(return_state_dict=True)  # get state_dict
        # If the lib returns a state_dict, load it with strict=False
        if isinstance(state, dict):
            model.load_state_dict(state, strict=False)
    except Exception as e:
        print("⚠️ Tolerant load triggered:", e)
        try:
            # retry normally but tolerate unexpected keys
            model.from_pretrained()
        except Exception as e2:
            print("Fallback load failed:", e2)
            raise
    finally:
        torch.load = orig_load

    return model


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config (dataset, loader, runtime)")
    ap.add_argument("--task",   required=True, help="YAML task spec (contains 'task.labels')")
    ap.add_argument("--out",    required=True, help="Output CSV path")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg  = yaml.safe_load(open(args.config, "r"))
    task = yaml.safe_load(open(args.task,   "r"))

    # Repro, device
    set_all(42)
    device = choose_device(cfg.get("runtime", {}).get("device", "auto"))

    # Data: your convenience wrapper uses get_dataset_paths() internally
    loader = make_loader_from_cfg(cfg)

    # Model: load on CPU, then move to device
    model = tolerant_load_medclip().to(device).eval()

    # Tokenizer (text) + image processor (vision)
    tokenizer     = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    img_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    # Two class prompts, e.g. ["frontal chest x-ray", "lateral chest x-ray"]
    queries = task["task"]["labels"]

    # Results table
    rows = [("image", "p_frontal", "p_lateral", "pred", "latency_sec")]
    mem0 = gpu_mem_mb()

    with torch.no_grad():
        for names, batch_np in loader:
            # DataLoader may return numpy or torch tensors; normalize to torch.Tensor
            batch = batch_np if isinstance(batch_np, torch.Tensor) else torch.from_numpy(batch_np)
            batch = batch.to(device)

            # Ensure 3 channels for CLIP (RGB)
            if batch.shape[1] == 1:
                batch = batch.repeat(1, 3, 1, 1)

            # Process images one-by-one (keeps code simple & robust)
            for i, name in enumerate(names):
                # CHW [0,1] → HWC uint8
                img_np = chw01_to_hwc_uint8(batch[i])

                # 1) Text inputs (labels as prompts)
                text_inputs = tokenizer(queries, padding=True, return_tensors="pt")

                # 2) Image inputs (CLIP normalization, resize, etc.)
                img_inputs = img_processor(images=img_np, return_tensors="pt")

                # 3) Merge and move to device
                inputs = {**text_inputs, **img_inputs}
                inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

                # Forward pass & timing
                with wallclock() as t:
                    out = model(**inputs)
                dt = round(t(), 4)

                # Handle both dict and object style returns
                logits = out["logits_per_image"] if isinstance(out, dict) else getattr(out, "logits_per_image", None)
                if logits is None:
                    raise RuntimeError("MedCLIP output missing 'logits_per_image'.")

                probs = logits.softmax(dim=1)[0].tolist()  # [p_frontal, p_lateral]
                pred  = "frontal" if probs[0] >= probs[1] else "lateral"
                rows.append((name, probs[0], probs[1], pred, dt))

    # Write outputs
    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)  # defensive
    write_csv(rows, out_csv)

    manifest = {
        "script": "projection_medclip.py",
        "git_commit": git_commit_short(),
        "config_hash": config_hash(cfg),
        "config": cfg,
        "task": task,
        "device": device,
        "gpu_max_mem_mb": gpu_mem_mb() - mem0
    }
    write_json(manifest, out_csv.with_suffix(".json"))
    print("✅ wrote:", out_csv)


if __name__ == "__main__":
    main()
