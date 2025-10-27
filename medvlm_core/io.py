"""
OpenCV I/O utilities for loading and preparing medical images
(especially Chest X-rays) in a deterministic and reproducible way.

Why this module:
----------------
Scientific reproducibility in medical imaging requires:
  • Explicit image read modes (grayscale vs RGB)
  • Deterministic normalization to [0,1]
  • Controlled interpolation during resize
  • Clear exception handling for missing/corrupt files

These small details ensure that two researchers running the same code
on different systems produce identical tensors for downstream models.
"""

from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
from typing import Literal, Tuple
import os
from typing import Dict, Any

def get_dataset_paths(cfg: Dict[str, Any]) -> Dict[str, str]:
    """
    Resolve dataset paths from YAML config with env-aware defaults.
    Expects keys:
      cfg['dataset']['base_dir']
      cfg['dataset']['images_subdir']
      cfg['dataset']['reports_csv']
      (optional) cfg['dataset'].get('projections_csv')
    """
    ds = cfg.get("dataset", {})
    base_dir = os.environ.get("DATA_DIR") or ds.get("base_dir")
    images_subdir = ds.get("images_subdir", "")
    reports_csv_name = ds.get("reports_csv")
    projections_csv_name = ds.get("projections_csv", None)

    if not base_dir:
        raise ValueError("Config missing: dataset.base_dir")
    if not reports_csv_name:
        raise ValueError("Config missing: dataset.reports_csv")

    images_dir = os.path.join(base_dir, images_subdir) if images_subdir else base_dir
    reports_csv = os.path.join(base_dir, reports_csv_name)
    projections_csv = (
        os.path.join(base_dir, projections_csv_name) if projections_csv_name else None
    )

    return {
        "base_dir": base_dir,
        "images_dir": images_dir,
        "reports_csv": reports_csv,
        "projections_csv": projections_csv or "",
    }

def imread(path: str | Path,
           mode: Literal["gray", "rgb"] = "gray") -> np.ndarray:
    """
    Read an image from disk using OpenCV.

    Parameters
    ----------
    path : str or Path
        Full path to the image file (.png, .jpg, .jpeg, etc.)
    mode : {'gray', 'rgb'}, optional (default='gray')
        'gray' → loads as single-channel grayscale (shape: H×W)
        'rgb'  → loads as color image converted from OpenCV’s
                  default BGR format to standard RGB (shape: H×W×3)

    Returns
    -------
    np.ndarray
        Image array in uint8 format.

    Raises
    ------
    FileNotFoundError
        If the image cannot be opened (corrupted or missing file).
    ValueError
        If mode is not one of {'gray', 'rgb'}.

    Notes
    -----
    - OpenCV’s cv2.imread() silently returns None on failure,
      so we explicitly raise FileNotFoundError to make failures visible.
    - Using OpenCV ensures consistent pixel decoding across systems,
      which is crucial in medical image reproducibility studies.
    """
    p = str(path)

    # Load in grayscale mode
    if mode == "gray":
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Unable to read image: {p}")
        return img

    # Load in color (BGR) and convert to RGB
    elif mode == "rgb":
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Unable to read image: {p}")
        # Convert from OpenCV default (BGR) → scientific convention (RGB)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Invalid mode → explicit exception
    raise ValueError("mode must be 'gray' or 'rgb'")


def to_float01(img: np.ndarray) -> np.ndarray:
    """
    Normalize image to float32 range [0,1].

    Parameters
    ----------
    img : np.ndarray
        Input image, either uint8 [0,255] or float.

    Returns
    -------
    np.ndarray (dtype=float32)
        Image normalized to [0,1] with an epsilon safeguard.

    Explanation
    -----------
    Normalization ensures that pixel intensity distributions are
    model-independent. This step also protects against overflow
    when converting between uint8 and float during augmentation.

    The epsilon (1e-6) avoids divide-by-zero errors for uniform images.
    """
    img = img.astype(np.float32)
    mn, mx = img.min(), img.max()
    return (img - mn) / (mx - mn + 1e-6)


def resize(img: np.ndarray,
           size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Resize image to a fixed resolution for model input.

    Parameters
    ----------
    img : np.ndarray
        Input image (H×W or H×W×C).
    size : tuple of int, optional (default=(224,224))
        Target (width, height) used by most ViT/ResNet architectures.

    Returns
    -------
    np.ndarray
        Resized image (dtype preserved).

    Explanation
    -----------
    - Uses cv2.INTER_AREA which is the scientifically preferred
      interpolation for downsampling because it approximates
      pixel area relation and avoids aliasing artifacts.
    - For upsampling, INTER_LINEAR is sometimes used, but for
      consistent benchmarks, INTER_AREA provides stable results.
    """
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)
