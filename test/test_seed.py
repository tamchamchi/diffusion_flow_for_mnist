# tests/test_seed.py
import random

import numpy as np
import torch

from src.utils.seed import set_seed


def test_set_seed_reproduces_torch_random_numbers():
    set_seed(0)
    a = torch.randn(5)
    set_seed(0)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_set_seed_reproduces_numpy_and_python_random():
    set_seed(0)
    np_a, py_a = np.random.rand(5), [random.random() for _ in range(5)]
    set_seed(0)
    np_b, py_b = np.random.rand(5), [random.random() for _ in range(5)]
    assert (np_a == np_b).all()
    assert py_a == py_b


def test_different_seeds_diverge():
    set_seed(0)
    a = torch.randn(5)
    set_seed(1)
    b = torch.randn(5)
    assert not torch.equal(a, b)


def test_set_seed_pins_cudnn_to_deterministic_algorithms():
    set_seed(0)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
