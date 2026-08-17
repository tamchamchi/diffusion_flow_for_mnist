"""Shared interface for all 5 generative-modeling methods.

Every method wraps the *same* UNet architecture (build_default_unet) and the
same time convention (t in [T_MIN, 1], t=0 <-> data, t=1 <-> noise, see
src/schedules.py). Subclasses differ only in (a) which conditional path they
train on, (b) what the network head is trained to predict, and (c) how that
head's raw output is converted into a probability-flow-ODE velocity, so that
sampling (src/sampling.py) is identical across all five.
"""
from abc import ABC, abstractmethod

import torch
from torch import nn

from src.models import build_default_unet
from src.schedules import T_MIN


class Method(ABC):
    name: str

    def __init__(self, net: nn.Module | None = None) -> None:
        self.net = net if net is not None else build_default_unet()

    def parameters(self):
        return self.net.parameters()

    def to(self, device: torch.device) -> "Method":
        self.net.to(device)
        return self

    def train(self) -> "Method":
        self.net.train()
        return self

    def eval(self) -> "Method":
        self.net.eval()
        return self

    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """t ~ U(T_MIN, 1), shared across every method."""
        return torch.rand(batch_size, device=device) * (1.0 - T_MIN) + T_MIN

    @abstractmethod
    def loss(self, x0: torch.Tensor) -> torch.Tensor:
        """Training loss for one batch of clean data x0 in [-1, 1]."""

    @abstractmethod
    def velocity(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """dx/dt for the probability-flow ODE at (x, t). src/sampling.py
        integrates this from t=1 (noise) to t=T_MIN (data) for every method."""