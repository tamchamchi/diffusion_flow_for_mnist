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
        t = self.sample_time(x0.shape[0], x0.device)
        te = expand_t(t, x0)
        x1 = torch.randn_like(x0)
        x_t = self.path.alpha(te) * x0 + self.path.sigma(te) * x1
        target = self.path.alpha_dot(te) * x0 + self.path.sigma_dot(te) * x1
        v_pred = self.net(x_t, t)
        return torch.mean((v_pred - target) ** 2)

    def velocity(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(x, t)


class FlowMatchingDiffusion(FlowMatchingOT):
    """FM-Diffusion: identical loss/velocity code as FM-OT; only the
    conditional path changes to the shared variance-preserving (VP) schedule
    (added in Task 5)."""

    name = "fm_diffusion"
    path = VPPath
