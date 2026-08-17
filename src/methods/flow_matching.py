"""Flow Matching methods. Both regress the network output directly onto the
closed-form conditional velocity field
    u_t(x_t|x0,x1) = alpha_dot(t) * x0 + sigma_dot(t) * x1
so `velocity()` is just `self.net(x, t)` for both — the network already
outputs a probability-flow-ODE drift, no reconstruction needed.
"""

import torch

from src.methods.base import Method
from src.schedules import OTPath, VPPath, expand_t


class FlowMatchingOT(Method):
    """FM-OT: linear / optimal-transport conditional path."""

    name = "fm_ot"
    path = OTPath

    def loss(self, x0: torch.Tensor) -> torch.Tensor:
        """x0: (B, 1, 28, 28) clean images -> scalar MSE between the
        network's velocity prediction and the closed-form conditional
        velocity at a random t."""
        t = self.sample_time(x0.shape[0], x0.device)  # (B,)
        te = expand_t(t, x0)  # (B,1,1,1), broadcastable against x0
        x1 = torch.randn_like(x0)  # (B, 1, 28, 28), the noise endpoint
        x_t = self.path.alpha(te) * x0 + self.path.sigma(te) * x1
        target = self.path.alpha_dot(te) * x0 + self.path.sigma_dot(te) * x1
        v_pred = self.net(x_t, t)
        return torch.mean((v_pred - target) ** 2)

    def velocity(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 28, 28), t: (B,) -> (B, 1, 28, 28) dx/dt. The net
        already outputs a velocity directly, so no conversion is needed."""
        return self.net(x, t)


class FlowMatchingDiffusion(FlowMatchingOT):
    """FM-Diffusion: identical loss/velocity code as FM-OT; only the
    conditional path changes to the shared variance-preserving (VP) schedule
    (added in Task 5)."""

    name = "fm_diffusion"
    path = VPPath
