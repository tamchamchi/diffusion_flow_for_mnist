"""Shared interface for all 5 generative-modeling methods.

Every method wraps the *same* UNet architecture (build_default_unet) and the
same time convention (t in [T_MIN, 1 - T_MIN], t=0 <-> noise, t=1 <-> data,
see src/schedules.py). Subclasses differ only in (a) which conditional path
they train on, (b) what the network head is trained to predict, and (c) how
that head's raw output is converted into a probability-flow-ODE velocity, so
that sampling (src/sampling.py) is identical across all five.
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
        """t ~ U(T_MIN, 1 - T_MIN), shared across every method. The upper
        margin keeps t away from t=1 (data), where every VP-path quantity's
        singularity lives (see src/schedules.py); the lower margin isn't
        strictly needed (nothing blows up at t=0/noise) but keeps training
        and sampling's t ranges identical."""
        return torch.rand(batch_size, device=device) * (1.0 - 2 * T_MIN) + T_MIN

    @abstractmethod
    def loss(self, x1: torch.Tensor) -> torch.Tensor:
        """x1: (B, 1, 28, 28) clean data in [-1, 1] -> scalar training loss."""

    @abstractmethod
    def velocity(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 28, 28), t: (B,) -> (B, 1, 28, 28) dx/dt for the
        probability-flow ODE at (x, t). src/sampling.py integrates this
        from t=T_MIN (noise) to t=1-T_MIN (data) for every method."""