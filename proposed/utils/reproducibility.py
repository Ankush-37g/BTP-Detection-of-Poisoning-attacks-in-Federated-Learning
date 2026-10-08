"""
Reproducibility utilities.

Design decision: Every experiment must set seeds for Python, NumPy, and PyTorch
before any computation begins. This ensures that the same --seed value
produces the same results across runs on the same hardware.

Note: MPS-based execution may still have minor non-determinism in some
PyTorch ops. Where deterministic execution is critical, move that operation
to CPU. This is documented in docs/environment.md.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set all random seeds for reproducibility.

    Sets seeds for:
        - Python's built-in random module
        - NumPy
        - PyTorch (CPU and GPU)
        - os.environ PYTHONHASHSEED

    Args:
        seed: Integer seed value. Use the same seed across experiments
              that should be comparable.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Enable deterministic mode on CUDA where possible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Note: MPS does not have a separate seed API in PyTorch 2.x.
    # torch.manual_seed() also seeds MPS operations.
