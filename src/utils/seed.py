"""Global reproducibility helper. Every entry-point script (src/train.py,
src/train_classifier.py, src/sampling.py, src/evaluate_metrics.py) calls
set_seed() once, as the first thing in main(), before any randomness is
drawn (weight init, MNIST DataLoader shuffling, x1 ~ N(0,I) noise draws,
Hutchinson-trace eps, ...) -- so a run with the same --seed is repeatable
end to end.
"""

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seeds Python's random, NumPy, and Torch (CPU + every CUDA device
    seen via manual_seed_all), and pins cuDNN to its deterministic
    (non-benchmark) algorithms so convolution results don't vary run to
    run. torch.manual_seed also seeds the default generator DataLoader
    shuffling draws from, so a fixed seed reproduces batch order too."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
