"""Score-based methods. The network predicts the score of the shared VP path
directly: s_theta(x_t, t) ≈ -x1 / sigma(t) = grad_x log p_t(x_t | x0).
ScoreMatching and ScoreFlow (Task 8) share everything except the loss weight
lambda(t): ScoreMatching uses sigma(t)^2 (Song & Ermon 2019's weight that
keeps the regression target O(1) instead of blowing up as sigma -> 0);
ScoreFlow (Task 8) uses beta(1-t).
"""

import torch

from src.methods.base import Method
from src.schedules import VPPath, beta, expand_t


class ScoreMatching(Method):
    name = "score"
    path = VPPath

    def _weight(self, t: torch.Tensor) -> torch.Tensor:
        return self.path.sigma(t) ** 2

    def loss(self, x0: torch.Tensor) -> torch.Tensor:
        t = self.sample_time(x0.shape[0], x0.device)
        te = expand_t(t, x0)
        x1 = torch.randn_like(x0)
        sigma = self.path.sigma(te)
        x_t = self.path.alpha(te) * x0 + sigma * x1
        target_score = -x1 / sigma
        score_pred = self.net(x_t, t)
        weight = self._weight(te)
        return torch.mean(weight * (score_pred - target_score) ** 2)

    def velocity(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Probability-flow ODE for the VP-SDE: dx/dt = -0.5*beta(t)*(x + score(x,t)).
        te = expand_t(t, x)
        score_pred = self.net(x, t)
        b = beta(te)
        return -0.5 * b * (x + score_pred)


class ScoreFlow(ScoreMatching):
    """SF: identical loss/velocity code as ScoreMatching; only the loss
    weight changes to beta(1-t) (a likelihood-style weight evaluated at the
    reversed time index — see the spec table in this plan's header)."""

    name = "score_continuous"

    def _weight(self, t: torch.Tensor) -> torch.Tensor:
        return beta(1.0 - t)
