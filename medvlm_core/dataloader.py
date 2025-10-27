"""
medvlm_core.dataloader
----------------------
PyTorch Dataset/DataLoader built on OpenCV utilities for CXR datasets.
"""

from pathlib import Path
from typing import Tuple
import numpy as np
from torch.utils.data import Dataset, DataLoader

from .io import imread, to_float01, resize
from .transforms import clahe_gray01

import os
from medvlm_core.io import get_dataset_paths

class CXRDataset(Dataset):
    """
    Minimal dataset for IU Chest X-ray images.

    Returns
    -------
    (filename, image_tensor_np)
        filename: str
        image_tensor_np: np.ndarray float32 in CHW format, normalized to [0,1]
    """
    def __init__(
        self,
        root: str | Path,
        images_rel_dir: str,          # e.g., "images/images_normalized"
        size: Tuple[int, int] = (224, 224),
        use_clahe: bool = True,
        mode: str = "gray"            # "gray" (1ch) or "rgb" (3ch)
    ):
        self.root = Path(root)
        self.dir = self.root / images_rel_dir
        self.paths = sorted(self.dir.glob("*.png"))
        self.size, self.use_clahe, self.mode = size, use_clahe, mode

        if not self.paths:
            raise FileNotFoundError(f"No PNGs found under: {self.dir}")

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        img = imread(p, mode=("gray" if self.mode == "gray" else "rgb"))
        img = to_float01(img)
        if self.use_clahe and self.mode == "gray":
            img = clahe_gray01(img)

        img = resize(img, self.size)  # (H,W) or (H,W,3)

        # Convert to CHW float32 for PyTorch compatibility
        if img.ndim == 2:  # gray → (1,H,W)
            img = img[None, ...]
        else:              # rgb → (3,H,W)
            img = np.transpose(img, (2, 0, 1))

        return p.name, img.astype("float32")


def make_loader(
    root: str | Path,
    images_rel_dir: str,
    size: Tuple[int, int],
    use_clahe: bool,
    mode: str,
    batch_size: int,
    num_workers: int
) -> DataLoader:
    """
    Factory to build a non-shuffling DataLoader for deterministic evaluation.
    """
    ds = CXRDataset(root, images_rel_dir, size=size, use_clahe=use_clahe, mode=mode)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

def make_loader_from_cfg(cfg) -> DataLoader:
    """
    Convenience wrapper: builds a DataLoader directly from the YAML config.
    It resolves paths via get_dataset_paths(cfg) and uses loader settings.
    """
    paths = get_dataset_paths(cfg)

    # loader settings with safe defaults
    loader_cfg = cfg.get("loader", {})
    img_size = int(loader_cfg.get("img_size", 224))
    batch_size = int(loader_cfg.get("batch_size", 32))
    num_workers = int(loader_cfg.get("num_workers", 2))
    grayscale = bool(loader_cfg.get("grayscale", True))
    mode = "gray" if grayscale else "rgb"
    use_clahe = True  # keep your current behavior; adjust if you have a cfg flag

    # Prefer a relative path if images_dir is inside base_dir; else keep absolute
    try:
        images_rel = os.path.relpath(paths["images_dir"], start=paths["base_dir"])
    except Exception:
        images_rel = paths["images_dir"]

    return make_loader(
        root=paths["base_dir"],
        images_rel_dir=images_rel,
        size=(img_size, img_size),
        use_clahe=use_clahe,
        mode=mode,
        batch_size=batch_size,
        num_workers=num_workers
    )