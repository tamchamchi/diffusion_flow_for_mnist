"""SM-Diffusion: continuous-time DDPM loss. The network predicts the noise
epsilon added by the shared variance-preserving (VP) forward process; this is
the standard DDPM objective (Ho et al. 2020) written in continuous time so it
shares T_MIN/VPPath with every other diffusion-based method here.
"""

import torch

from src.methods.base import Method
from src.schedules import VPPath, beta, expand_t


class NoiseMatchingDiffusion(Method):
    """DDPM: variance-preserving conditional path, network predicts noise."""

    name = "ddpm"
    path = VPPath

    def loss(self, x1: torch.Tensor) -> torch.Tensor:
        """x1: (B, 1, 28, 28) clean images -> scalar MSE between the
        network's noise prediction and the noise x0 actually mixed in at a
        random t (Ho et al. 2020's simple objective, in continuous time)."""
        t = self.sample_time(x1.shape[0], x1.device)  # (B,)
        te = expand_t(t, x1)  # (B,1,1,1), broadcastable against x1
        x0 = torch.randn_like(x1)  # (B, 1, 28, 28), the noise being predicted
        x_t = self.path.alpha(te) * x0 + self.path.sigma(te) * x1
        eps_pred = self.net(x_t, t)
        return torch.mean((eps_pred - x0) ** 2)

    def velocity(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 28, 28), t: (B,) -> (B, 1, 28, 28) dx/dt.
        Probability-flow ODE, derived the same way as ScoreMatching.velocity
        below: dx/dt = 1/2*beta(t)*(x + score(x,t)), with score(x,t) =
        -eps/alpha(t) since alpha(t) is x0=noise's coefficient here."""
        te = expand_t(t, x)
        eps_pred = self.net(x, t)
        alpha = self.path.alpha(te).clamp_min(1e-3)  # extra safety near t=1 (data)
        b = beta(te)
        return 0.5 * b * x - 0.5 * (b / alpha) * eps_pred
