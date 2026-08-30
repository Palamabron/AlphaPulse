import os
import random

import numpy as np


def set_global_seed(seed: int, *, seed_torch: bool = True) -> None:
    """Lock all RNG sources to a single seed for reproducibility.

    Sets Python ``random``, NumPy, and PyTorch (if available).
    Also sets ``PYTHONHASHSEED`` in the environment so that any child
    processes (e.g. Ray workers) inherit a fixed hash seed.  Note: this has
    no effect on the current interpreter, which reads the variable at startup.

    Args:
        seed: Integer seed value. Must be non-negative.
        seed_torch: Whether to import and seed PyTorch. Disable this in processes
            that only train non-PyTorch models.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if not seed_torch:
        return

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
