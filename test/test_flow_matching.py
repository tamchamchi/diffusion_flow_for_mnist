# tests/test_flow_matching.py
import torch

from src.methods.flow_matching import FlowMatchingOT


def test_fm_ot_loss_is_finite_scalar_with_grad():
    """
    Verify that the FM‑OT loss function returns a finite scalar and supports
    backpropagation.

    This test checks three critical properties:
        1. The loss is a zero‑dimensional tensor (scalar) so that it can be
           used directly by optimizers.
        2. The loss value is finite (not NaN or Inf), which guards against
           numerical instabilities in the probability path formulas (e.g.
           division by near‑zero sigma).
        3. Calling `loss.backward()` actually populates gradients for the
           network parameters, ensuring the computational graph is intact and
           training can proceed.
    """
    method = FlowMatchingOT()
    x1 = torch.randn(4, 1, 28, 28)
    loss = method.loss(x1)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in method.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_fm_ot_velocity_matches_raw_net_output():
    """
    Verify that the velocity output of FlowMatchingOT is exactly the raw
    network output, with no extra transformation.

    In the FM‑OT method, the conditional vector field is trained directly,
    meaning the model's forward pass already returns the velocity field
    u_t(x_t). No conversion from score or noise is needed. This test ensures
    that:

        1. The method.velocity() call is simply a passthrough to self.net().
        2. Any future changes do not inadvertently add a transformation that
           would break the expected interface used by sampling and evaluation.
    """
    method = FlowMatchingOT()
    x = torch.randn(2, 1, 28, 28)
    t = torch.full((2,), 0.3)
    with torch.no_grad():
        assert torch.allclose(method.velocity(x, t), method.net(x, t))
