# medvlm_core/dataloader.py
# Centralized data loading for IU CXR.
# If clip_preprocess=True, we DO NOT resize/normalize here.
# All CLIP transforms are applied in scripts/projection_medclip.py via CLIPProcessor.

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any

import csv
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

@dataclass
class LoaderCfg:
    batch_size: int = 32
    num_workers: int = 2
    prefetch_factor: int = 2
    persistent_workers: bool = True
    pin_memory: bool = True
    clip_preprocess: bool = True
    grayscale: bool = False
    use_clahe: bool = False
    img_size: int = 224

class IUChestXrayDataset(Dataset):
    """
    Returns (PIL.Image (RGB), label:int, full_image_path:str).
    If clip_preprocess=True -> DO NOT resize/normalize here.
    """
    def __init__(self, image_paths: List[str], labels: List[int], cfg: LoaderCfg):
        assert len(image_paths) == len(labels), "images and labels length mismatch"
        self.paths = image_paths
        self.labels = labels
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.paths)

    def _read_rgb(self, p: str) -> Image.Image:
        img = Image.open(p)
        if img.mode != "RGB":  # many CXRs are single-channel or 'L'
            img = img.convert("RGB")
        return img

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        label = self.labels[idx]
        img = self._read_rgb(path)

        # IMPORTANT: when clip_preprocess=True, all resizing/normalizing happens later.
        if self.cfg.clip_preprocess:
            return img, label, path

        # Legacy branch: keep image as RGB; if you ever add transforms here,
        # ensure they are mutually exclusive with CLIPProcessor usage.
        return img, label, path


def make_loader_from_cfg(
    image_paths: List[str],
    labels: List[int],
    loader_cfg: Dict[str, Any],
) -> DataLoader:
    cfg = LoaderCfg(**loader_cfg)
    ds = IUChestXrayDataset(image_paths, labels, cfg)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.prefetch_factor,
        persistent_workers=cfg.persistent_workers,
        pin_memory=cfg.pin_memory,
    )
