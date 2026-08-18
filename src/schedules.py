"""Shared time convention and Gaussian probability paths for all 5 methods.

Time convention: t in [T_MIN, 1 - T_MIN]. t≈0 <-> standard Gaussian noise
x0 ~ N(0, I), t=1 <-> clean data x1. Every method builds its training pair as

    x_t = alpha(t) * x0 + sigma(t) * x1

so alpha(t) is the *noise* coefficient (1 at t=0, ~0 at t=1) and sigma(t) is
the *data* coefficient (~0 at t=0, 1 at t=1) -- the reverse of which
variable each letter multiplies compared to the more common "t=0 data, t=1
noise" diffusion convention, chosen here so sampling integrates t forward,
0 -> 1, noise -> data (matching the original Flow Matching / rectified-flow
papers). This module is the single source of truth for alpha(t), sigma(t)
and their time-derivatives, the same role build_default_unet() plays for
the architecture, so every method's target/velocity conversion agrees
exactly.

VPPath below is built from the *physically* forward VP-SDE schedule (noise
strictly accumulating as its own internal time runs 0 -> 1, data -> noise,
same as Song et al. 2021 / Ho et al. 2020) evaluated at the reversed index
(1 - t), then swapped into the alpha/sigma roles above -- see the
derivation note on VPPath.sigma. That's why alpha(t) here is the *derived*
sqrt(1 - sigma(t)^2) quantity (its exact-zero singularity lands at t=1,
approached but never reached) while sigma(t) is the exp-closed-form one.
"""

import torch

# Avoids the t=1 singularity every VP-path quantity has at the data end
# (alpha(t) -> 0 as t -> 1; anything dividing by alpha, e.g. score targets
# and the eps-parameterized velocity ODE, blows up there). Also used as a
# symmetric margin at the t=0 (noise) end even though nothing actually
# blows up there, purely so training's t ~ U(.) and sampling's t_grid cover
# the same range. Standard practice in score-based generative modeling
# (Song et al. 2021).
T_MIN = 1e-3

BETA_MIN = 0.1
BETA_MAX = 20.0


def beta(t: torch.Tensor) -> torch.Tensor:
    """Noise-injection rate as a function of this module's t (t=0 noise,
    t=1 data): BETA_MAX at t=0, BETA_MIN at t=1 -- i.e. decreasing in t,
    the mirror image of the textbook VP-SDE schedule (which increases along
    its own data(0)->noise(1) time). Shared by every diffusion-based
    method's velocity ODE and by ScoreFlow's loss weight."""
    return BETA_MAX - t * (BETA_MAX - BETA_MIN)


def _integral_beta_reversed(t: torch.Tensor) -> torch.Tensor:
    """Closed form of integral_0^(1-t) beta_phys(s) ds, where beta_phys(s)
    = BETA_MIN + s*(BETA_MAX-BETA_MIN) is the textbook *increasing*
    VP-SDE schedule along its own internal (data->noise) time. This is the
    total noise accumulated between "this module's t" and t=1 (data) --
    used to build VPPath.sigma below. Not the same as integrating `beta`
    above directly; see VPPath.sigma's docstring for the full derivation."""
    tau = 1.0 - t
    return BETA_MIN * tau + 0.5 * (BETA_MAX - BETA_MIN) * tau**2


def expand_t(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape a (B,) time tensor to broadcast against an image tensor (B,C,H,W)."""
    return t.view(-1, *([1] * (x.dim() - 1)))


class OTPath:
    """Linear / optimal-transport conditional path used by FM-OT.

    alpha(t) = 1 - t, sigma(t) = t, so the conditional velocity
    u_t(x_t|x0,x1) = alpha_dot*x0 + sigma_dot*x1 = x1 - x0 is constant in t
    -- the standard rectified-flow target, noise -> data.

    Every method below is elementwise: t can be any shape (typically a
    (B,) batch of times or a (B,1,1,1) tensor already broadcast against an
    image via expand_t) and the return shape matches t exactly.
    """

    @staticmethod
    def alpha(t: torch.Tensor) -> torch.Tensor:
        """Noise-signal coefficient in x_t = alpha(t)*x0 + sigma(t)*x1."""
        return 1.0 - t

    @staticmethod
    def sigma(t: torch.Tensor) -> torch.Tensor:
        """Data-signal coefficient in x_t = alpha(t)*x0 + sigma(t)*x1."""
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
    def sigma(t: torch.Tensor) -> torch.Tensor:
        """Data-signal coefficient: exp(-1/2 * integral_0^(1-t) beta_phys)
        per the VP-SDE closed form evaluated at the reversed index 1-t --
        i.e. this *is* the textbook forward-process alpha(tau) (tau=0
        data, tau=1 noise) with tau=1-t substituted in, relabeled as this
        module's sigma because here it multiplies the data endpoint x1.
        1 at t=1 (data), ~0 (never exactly, so safe to evaluate at t=0)
        as t -> 0 (noise)."""
        return torch.exp(-0.5 * _integral_beta_reversed(t))

    @staticmethod
    def alpha(t: torch.Tensor) -> torch.Tensor:
        """Noise-signal coefficient, sqrt(1 - sigma(t)^2) so alpha^2 +
        sigma^2 = 1 holds exactly. ~1 at t=0 (noise), and *exactly* 0 at
        t=1 (data) -- the path's only true singularity, guarded by T_MIN
        (clamped for numerical stability near that boundary)."""
        s = VPPath.sigma(t)
        return torch.sqrt((1.0 - s**2).clamp_min(1e-12))

    @staticmethod
    def sigma_dot(t: torch.Tensor) -> torch.Tensor:
        """d(sigma)/dt = 1/2 * beta(t) * sigma(t), directly from sigma's
        own exp(-1/2 * integral) definition above (chain rule through the
        1-t integration bound)."""
        return 0.5 * beta(t) * VPPath.sigma(t)

    @staticmethod
    def alpha_dot(t: torch.Tensor) -> torch.Tensor:
        """d(alpha)/dt, derived from alpha^2 + sigma^2 = 1 by implicit
        differentiation: alpha_dot = -sigma*sigma_dot / alpha."""
        a, s = VPPath.alpha(t), VPPath.sigma(t)
        return -0.5 * beta(t) * s**2 / a
