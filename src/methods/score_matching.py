"""Score-based methods. The network predicts the score of the shared VP path
directly: s_theta(x_t, t) ≈ -x0 / alpha(t) = grad_x log p_t(x_t | x1), since
alpha(t) is the noise endpoint x0's coefficient in this module's convention
(t=0 noise, t=1 data; see src/schedules.py). ScoreMatching and ScoreFlow
(Task 8) share everything except the loss weight lambda(t): ScoreMatching
uses alpha(t)^2 (Song & Ermon 2019's weight that keeps the regression target
O(1) instead of blowing up as alpha -> 0, i.e. as t -> 1/data); ScoreFlow
(Task 8) uses beta(t).
"""

import torch

from src.methods.base import Method
from src.schedules import VPPath, beta, expand_t


class ScoreMatching(Method):
    """SM: variance-preserving conditional path, network predicts the score
    grad_x log p_t(x_t), weighted by alpha(t)^2 (Song & Ermon 2019)."""

    name = "score"
    path = VPPath

    def _weight(self, t: torch.Tensor) -> torch.Tensor:
        """t: any shape -> matching-shape loss weight lambda(t); overridden
        by ScoreFlow below to swap the weighting scheme."""
        return self.path.alpha(t) ** 2

    def loss(self, x1: torch.Tensor) -> torch.Tensor:
        """x1: (B, 1, 28, 28) clean images -> scalar weighted MSE between
        the network's score prediction and the closed-form conditional
        score -x0/alpha(t) at a random t (denoising score matching)."""
        t = self.sample_time(x1.shape[0], x1.device)  # (B,)
        te = expand_t(t, x1)  # (B,1,1,1), broadcastable against x1
        x0 = torch.randn_like(x1)  # (B, 1, 28, 28), the noise endpoint
        alpha = self.path.alpha(te)
        x_t = alpha * x0 + self.path.sigma(te) * x1
        target_score = -x0 / alpha
        score_pred = self.net(x_t, t)
        weight = self._weight(te)
        return torch.mean(weight * (score_pred - target_score) ** 2)

    def velocity(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 28, 28), t: (B,) -> (B, 1, 28, 28) dx/dt.
        Probability-flow ODE for this module's (t=0 noise, t=1 data) VP
        path: dx/dt = 1/2*beta(t)*(x + score(x,t)) -- see
        src/methods/noise_matching.py's velocity() for the same derivation
        in the eps-parameterized form."""
        te = expand_t(t, x)
        score_pred = self.net(x, t)
        b = beta(te)
        return 0.5 * b * (x + score_pred)


class ScoreFlow(ScoreMatching):
    """SF: identical loss/velocity code as ScoreMatching; only the loss
    weight changes to beta(t) (Song et al. 2021's maximum-likelihood
    weight g(t)^2, in this module's t=0-noise/t=1-data convention)."""

    name = "score_flow"

    def _weight(self, t: torch.Tensor) -> torch.Tensor:
        """Likelihood-style weight beta(t) -- see the module docstring
        above."""
        return beta(t)
