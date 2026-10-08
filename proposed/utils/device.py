"""
Device selection utility.

Priority order: MPS (Apple Silicon) → CUDA → CPU.

Design decision: This function is the single source of truth for device selection.
Every module that needs a device should call get_device() rather than
hardcoding 'cuda' or 'cpu'. This ensures the entire codebase works on
any hardware without modification.
"""

import torch


def get_device() -> torch.device:
    """
    Return the best available compute device.

    Priority:
        1. MPS  — Apple Silicon GPU (used on development machine)
        2. CUDA — NVIDIA GPU
        3. CPU  — fallback

    Returns:
        torch.device
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def device_name() -> str:
    """Human-readable description of the selected device."""
    d = get_device()
    if d.type == "mps":
        return "Apple Silicon MPS"
    elif d.type == "cuda":
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    else:
        return "CPU"
