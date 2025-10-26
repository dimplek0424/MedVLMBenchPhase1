"""
medvlm_core.seeds
-----------------
Deterministic seeds for scientific reproducibility.
"""

import os, random, numpy as np
try:
    import torch
except ImportError:
    torch = None

def set_all(seed: int = 42, deterministic_cudnn: bool = True) -> None:
    """
    Set seeds for Python, NumPy, and (optionally) PyTorch.

    Parameters
    ----------
    seed : int
        Random seed value.
    deterministic_cudnn : bool
        If True, disable CuDNN autotuner for determinism (slower, but reproducible).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
