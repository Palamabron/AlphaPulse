import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Lock all RNG sources to a single seed for reproducibility.

    Sets Python ``random``, NumPy, and PyTorch (if available).
    Also fixes ``PYTHONHASHSEED`` so dictionary ordering is deterministic
    within the current process.

    Args:
        seed: Integer seed value. Must be non-negative.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
