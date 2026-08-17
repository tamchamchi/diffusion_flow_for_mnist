"""Shared time convention and Gaussian probability paths for all 5 methods.

Time convention: t in [T_MIN, 1]. t≈0 <-> clean data x0, t=1 <-> standard
Gaussian noise x1 ~ N(0, I). Every method builds its training pair as

    x_t = alpha(t) * x0 + sigma(t) * x1

This module is the single source of truth for alpha(t), sigma(t) and their
time-derivatives, the same role build_default_unet() plays for the
architecture, so every method's target/velocity conversion agrees exactly.
"""

import torch

# Avoids the t=0 singularity shared by every VP-path quantity (alpha_dot,
# sigma_dot, and therefore every score/velocity formula blow up as t -> 0).
# Standard practice in score-based generative modeling (Song et al. 2021).
T_MIN = 1e-3

BETA_MIN = 0.1
BETA_MAX = 20.0


def beta(t: torch.Tensor) -> torch.Tensor:
    """Linear noise schedule shared by every diffusion-based method (VP-SDE)."""
    return BETA_MIN + t * (BETA_MAX - BETA_MIN)


def _integral_beta(t: torch.Tensor) -> torch.Tensor:
    """Closed form of integral_0^t beta(s) ds for the linear schedule above."""
    return BETA_MIN * t + 0.5 * (BETA_MAX - BETA_MIN) * t**2


def expand_t(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape a (B,) time tensor to broadcast against an image tensor (B,C,H,W)."""
    return t.view(-1, *([1] * (x.dim() - 1)))


class OTPath:
    """Linear / optimal-transport conditional path used by FM-OT.

    alpha(t) = 1 - t, sigma(t) = t, so the conditional velocity
    u_t(x_t|x0,x1) = alpha_dot*x0 + sigma_dot*x1 = x1 - x0 is constant in t.

    Every method below is elementwise: t can be any shape (typically a
    (B,) batch of times or a (B,1,1,1) tensor already broadcast against an
    image via expand_t) and the return shape matches t exactly.
    """

    @staticmethod
    def alpha(t: torch.Tensor) -> torch.Tensor:
        """Data-signal coefficient in x_t = alpha(t)*x0 + sigma(t)*x1."""
        return 1.0 - t

    @staticmethod
    def sigma(t: torch.Tensor) -> torch.Tensor:
        """Noise coefficient in x_t = alpha(t)*x0 + sigma(t)*x1."""
        return t

    @staticmethod
    def alpha_dot(t: torch.Tensor) -> torch.Tensor:
        """d(alpha)/dt, used to build the conditional target velocity."""
        return -torch.ones_like(t)

    @staticmethod
    def sigma_dot(t: torch.Tensor) -> torch.Tensor:
        """d(sigma)/dt, used to build the conditional target velocity."""
        return torch.ones_like(t)


class VPPath:
    """Variance-preserving conditional path shared by FM-Diffusion,
    SM-Diffusion (DDPM loss), Score Matching, and Score Flow:
    alpha(t)^2 + sigma(t)^2 = 1 for every t.

    Same elementwise/broadcast convention as OTPath: t and the return
    value always have matching shapes.
    """

    @staticmethod
    def alpha(t: torch.Tensor) -> torch.Tensor:
        """Data-signal coefficient in x_t = alpha(t)*x0 + sigma(t)*x1;
        exp(-1/2 * integral_0^t beta) per the VP-SDE closed form."""
        return torch.exp(-0.5 * _integral_beta(t))

    @staticmethod
    def sigma(t: torch.Tensor) -> torch.Tensor:
        """Noise coefficient, sqrt(1 - alpha(t)^2) so alpha^2 + sigma^2 = 1
        holds exactly (clamped away from 0 for stability near t=0)."""
        a = VPPath.alpha(t)
        return torch.sqrt((1.0 - a**2).clamp_min(1e-12))

    @staticmethod
    def alpha_dot(t: torch.Tensor) -> torch.Tensor:
        """d(alpha)/dt = -1/2 * beta(t) * alpha(t)."""
        return -0.5 * beta(t) * VPPath.alpha(t)

    @staticmethod
    def sigma_dot(t: torch.Tensor) -> torch.Tensor:
        """d(sigma)/dt, derived from alpha^2 + sigma^2 = 1 by implicit
        differentiation: sigma_dot = -alpha*alpha_dot / sigma."""
        a, s = VPPath.alpha(t), VPPath.sigma(t)
        return 0.5 * beta(t) * a**2 / s
