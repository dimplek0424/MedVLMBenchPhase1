"""
medvlm_core.timer
-----------------
Lightweight timing context manager and GPU memory helper.
"""

import time, contextlib
try:
    import torch
except ImportError:
    torch = None

@contextlib.contextmanager
def wallclock():
    """
    Usage:
        with wallclock() as t:
            ... your op ...
        dt = t()  # seconds (float)
    """
    t0 = time.time()
    yield lambda: time.time() - t0

def gpu_mem_mb() -> float:
    """
    Return max allocated GPU memory (MB) if torch CUDA is available, else 0.0.
    """
    if torch is None or not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024 ** 2))
