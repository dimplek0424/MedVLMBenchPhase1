"""
BioMedCLIP projection-type benchmark (frontal vs lateral).

Run:
  python scripts/projection_biomedclip.py \
      --config configs/dataset_iu_v03_full.yaml \
      --task   configs/task_projection_v01.yaml \
      --out    results/projection/iu_v03_full_biomedclip.csv
"""

import argparse
from pathlib import Path
import yaml, torch
from transformers import CLIPModel, CLIPProcessor

from medvlm_core.seeds import set_all
from medvlm_core.dataloader import make_loader
from medvlm_core.logging import write_csv, write_json, git_commit_short, config_hash
from medvlm_core.timer import wallclock, gpu_mem_mb

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--task",   required=True)
    ap.add_argument("--out",    required=True)
    return ap.parse_args()

def choose_device(d: str) -> str:
    return "cuda" if (d == "auto" and torch.cuda.is_available()) else ("cpu" if d == "auto" else d)

def main():
    args = parse_args()
    cfg  = yaml.safe_load(open(args.config))
    task = yaml.safe_load(open(args.task))
    set_all(42)
    device = choose_device(cfg["runtime"]["device"])

    loader = make_loader(
        root=cfg["dataset"]["root"],
        images_rel_dir=cfg["dataset"]["images_dir"],
        size=tuple(cfg["preprocess"]["size"]),
        use_clahe=bool(cfg["preprocess"]["use_clahe"]),
        mode=cfg["preprocess"]["mode"],
        batch_size=int(cfg["runtime"]["batch_size"]),
        num_workers=int(cfg["runtime"]["num_workers"])
    )

    model_id = "microsoft/BiomedCLIP-PubMedBERT_256-vit-base-224"
    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    proc  = CLIPProcessor.from_pretrained(model_id)

    queries = task["task"]["labels"]  # ["frontal chest x-ray", "lateral chest x-ray"]
    rows = [("image","p_frontal","p_lateral","pred","latency_sec")]
    mem0 = gpu_mem_mb()

    with torch.no_grad():
        for names, batch_np in loader:
            batch = torch.from_numpy(batch_np).to(device)
            if batch.shape[1] == 1:
                batch = batch.repeat(1,3,1,1)  # CLIP expects 3-channel

            for i, name in enumerate(names):
                img_t = batch[i:i+1]  # (1,3,H,W)

                # CLIPProcessor expects PIL or numpy by default; but it can take tensors
                inputs = proc(text=queries, images=img_t, return_tensors="pt", padding=True)
                inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k,v in inputs.items()}

                with wallclock() as t:
                    out = model(**inputs)
                dt = round(t(), 4)

                # logits_per_image: similarity of image to texts
                probs = out.logits_per_image.softmax(dim=1)[0].tolist()
                pred  = "frontal" if probs[0] >= probs[1] else "lateral"
                rows.append((name, probs[0], probs[1], pred, dt))

    out_csv = Path(args.out)
    write_csv(rows, out_csv)

    write_json({
        "script":"projection_biomedclip.py",
        "git_commit": git_commit_short(),
        "config_hash": config_hash(cfg),
        "config": cfg, "task": task, "device": device,
        "gpu_max_mem_mb": gpu_mem_mb() - mem0
    }, out_csv.with_suffix(".json"))
    print("✅ wrote:", out_csv)

if __name__ == "__main__":
    main()
