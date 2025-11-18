"""
medvlm_core.transforms
======================
Standardized, clinically interpretable image preprocessing transforms
for radiology images — implemented with OpenCV.

Current scope:
--------------
• CLAHE (Contrast Limited Adaptive Histogram Equalization)
  for controlled local contrast enhancement.

Scientific rationale:
---------------------
Chest X-rays (CXRs) often have poor contrast or variable illumination
caused by acquisition parameters or digitization artifacts.
CLAHE improves local contrast while preventing over-amplification
of noise in homogeneous regions (e.g., lungs, mediastinum).

This file is designed for modularity — each transform is pure,
deterministic, and easily unit-tested.
"""

import cv2
import numpy as np
from typing import Tuple


def clahe_gray01(img_gray01: np.ndarray,
                 clip_limit: float = 2.0,
                 tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to a grayscale image normalized to [0,1].

    Parameters
    ----------
    img_gray01 : np.ndarray
        Input grayscale image, assumed already normalized to [0,1].
        Shape: (H, W). Type: float32 or float64.
    clip_limit : float, optional (default=2.0)
        Threshold for contrast limiting. Lower values = milder enhancement,
        higher values = stronger contrast (risking noise amplification).
        Typical medical range: 1.5–3.0.
    tile_grid_size : tuple of int, optional (default=(8,8))
        Size of local tiles used for histogram equalization.
        Each tile is processed independently, then combined using
        bilinear interpolation by OpenCV.

    Returns
    -------
    np.ndarray
        CLAHE-enhanced image, dtype=float32, range [0,1].

    Raises
    ------
    AssertionError
        If the input image is not 2-D or not normalized properly.

    Explanation
    -----------
    Steps performed:
    1. Validate input (2D grayscale, [0,1]).
    2. Convert float [0,1] → uint8 [0,255] for OpenCV’s CLAHE API.
    3. Apply cv2.createCLAHE with the given parameters.
    4. Convert result back to float [0,1].
    5. Return reproducible enhancement suitable for downstream models.

    CLAHE improves visualization of fine-grained structures such as:
      - pleural lines
      - cardiac borders
      - costophrenic angles
      - interstitial markings

    References
    ----------
    - Zuiderveld K. “Contrast Limited Adaptive Histogram Equalization.”
      In *Graphics Gems IV*, 1994.
    - OpenCV Documentation: https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
    """
    # --- 1. Validate input ---
    assert img_gray01.ndim == 2, "CLAHE expects a 2-D grayscale image"
    if img_gray01.max() > 1.0 or img_gray01.min() < 0.0:
        raise ValueError("Image must be normalized to [0,1] before CLAHE")

    # --- 2. Convert float [0,1] → uint8 [0,255] ---
    img_u8 = (img_gray01 * 255.0).clip(0, 255).astype(np.uint8)

    # --- 3. Apply CLAHE ---
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    img_eq = clahe.apply(img_u8)

    # --- 4. Convert back to float [0,1] ---
    img_eq = img_eq.astype(np.float32) / 255.0

    # --- 5. Return result ---
    return img_eq


# ---------------------------------------------------------------------------
# Example usage (for testing or tutorial notebooks)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from medvlm_core.io import imread, to_float01

    # Example: load a single CXR and visualize the effect of CLAHE
    path = "D:/MedVLMPhase1/data/chestxray_iu/images/images_normalized/00012345_000.png"

    img = imread(path, mode="gray")       # Load grayscale
    img = to_float01(img)                 # Normalize to [0,1]
    img_clahe = clahe_gray01(img, clip_limit=2.0, tile_grid_size=(8,8))

    # Plot side-by-side for qualitative verification
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1); plt.imshow(img, cmap='gray'); plt.title("Original")
    plt.subplot(1,2,2); plt.imshow(img_clahe, cmap='gray'); plt.title("CLAHE-Enhanced")
    plt.tight_layout(); plt.show()
