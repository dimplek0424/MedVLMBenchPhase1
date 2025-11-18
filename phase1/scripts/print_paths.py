# scripts/print_paths.py
# --- Make repo root importable (so "medvlm_core" resolves) ---
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import yaml
from medvlm_core.io import get_dataset_paths

def main():
    cfg_path = pathlib.Path("configs/dataset_iu_v03_full.yaml")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths = get_dataset_paths(cfg)
    print("BASE_DIR :", paths["base_dir"])
    print("IMAGES   :", paths["images_dir"])
    print("REPORTS  :", paths["reports_csv"])
    print("PROJECTS :", paths["projections_csv"])

if __name__ == "__main__":
    main()
