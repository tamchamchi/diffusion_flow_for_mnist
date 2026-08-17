# 5 Generative-Modeling Methods on MNIST — Implementation Plan

> **For agentic workers:** This repo does not currently have the
> `superpowers` plugin's `subagent-driven-development` / `executing-plans`
> skills registered. Execute task-by-task in this session ("Inline
> Execution" style: implement → test → commit, checkpoint between tasks),
> or dispatch one general-purpose sub-agent per task if you prefer isolation
> — either way, do not start task N+1 until task N's tests are green and
> committed. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and compare 5 generative-modeling methods on MNIST
(FM-OT, FM-Diffusion, SM-Diffusion/DDPM-loss, Score Matching, Score Flow),
all sharing one UNet architecture, one continuous time convention, and one
Probability-Flow-ODE sampler.

**Architecture:** Every method is a small `Method` subclass that wraps the
existing `build_default_unet()` (unchanged) and differs only in (a) which
Gaussian conditional path it trains on — the linear/OT path or the shared
variance-preserving (VP) path — (b) what the network head is trained to
regress (velocity, noise, or score) and with what loss weight, and (c) a
`velocity()` method that converts that head's raw output into a
probability-flow-ODE drift `dx/dt`. Because `velocity()` is the only thing
the sampler touches, one `src/sampling.py` (built on the `torchdiffeq`
dependency already in `environment.yml`) draws samples from all 5 methods
identically.

**Tech Stack:** Python 3.11, PyTorch, torchvision (MNIST), einops,
torchdiffeq (ODE integration), tqdm, pytest — all already declared in
`environment.yml`. No new dependencies.

**Spec:** This plan's spec is the Vietnamese method table the user provided
in-conversation (no separate spec file exists yet):

| # | Method | Family | Head predicts | Loss weight |
|---|--------|--------|----------------|-------------|
| 1 | FM-OT | Flow matching | velocity, linear/OT path | none (uniform) |
| 2 | FM-Diffusion | Flow matching | velocity, VP path | none (uniform) |
| 3 | SM-Diffusion (DDPM loss) | Diffusion | noise ε | none (uniform) |
| 4 | Score Matching (SM) | Diffusion | score | σ(t)² |
| 5 | Score Flow (SF) | Diffusion | score | β(1−t) |

Checkpoint directory names are fixed by the existing `.env.example`
(`CKPT_ROOT` sub-directories already named there): `ckpt_flow`,
`ckpt_flow_diff`, `ckpt_ddpm`, `ckpt_score`, `ckpt_score_continuous`.

## Global Constraints

- All 5 methods MUST reuse `build_default_unet()` from
  [src/models/unet.py](../../../src/models/unet.py) unmodified — it is
  already documented there as "the single source of truth ... so every
  method (ddpm/ddim/score/flow) instantiates an identical net."
- All 5 methods share one time convention: `t ∈ [T_MIN, 1]`, `t≈0` ↔ clean
  data `x0`, `t=1` ↔ standard Gaussian noise `x1 ~ N(0,I)`, with
  `T_MIN = 1e-3` (avoids the shared `t=0` singularity — see Task 1).
- All 5 methods are sampled with one Probability-Flow-ODE integrator
  (`src/sampling.py`, `torchdiffeq`), integrating `t: 1 → T_MIN`. No
  method gets a bespoke sampler.
- Checkpoints save to `$CKPT_ROOT/<dir>/model.pt` where `<dir>` is exactly
  one of `ckpt_flow`, `ckpt_flow_diff`, `ckpt_ddpm`, `ckpt_score`,
  `ckpt_score_continuous` (from `.env.example`), keyed by method name
  `fm_ot`, `fm_diffusion`, `ddpm`, `score`, `score_continuous` respectively.
- Data: MNIST, normalized to `[-1, 1]` (`transforms.Normalize((0.5,),(0.5,))`),
  28×28 grayscale, matching the UNet's `in_channels=out_channels=1`.
- Stack is Python 3.11 + PyTorch + torchvision + einops + torchdiffeq + tqdm
  + pytest, exactly as pinned in `environment.yml` — do not add packages.

---

## File Structure

```
pyproject.toml                     # NEW — pytest config (rootdir imports)
src/
  schedules.py                     # NEW — shared time convention: OTPath, VPPath, beta(t)
  data.py                          # NEW — MNIST DataLoader, [-1,1] normalization
  sampling.py                      # NEW — shared Probability-Flow-ODE sampler
  train.py                         # NEW — generic training CLI (--method ...)
  evaluate.py                      # NEW — sample-grid generation from a checkpoint
  methods/
    base.py                        # MODIFY (currently empty) — abstract Method class
    flow_matching.py                # NEW — FlowMatchingOT, FlowMatchingDiffusion
    noise_matching.py               # MODIFY (currently a stub) — NoiseMatchingDiffusion (DDPM loss)
    score_matching.py               # NEW — ScoreMatching, ScoreFlow
  models/
    unet.py                         # UNCHANGED — build_default_unet() reused by every method
scripts/
  train_fm_ot.sh                    # NEW
  train_fm_diffusion.sh             # NEW
  train_ddpm.sh                     # NEW
  train_score.sh                    # NEW
  train_score_continuous.sh         # NEW
tests/
  test_schedules.py                 # NEW
  test_data.py                      # NEW
  test_base.py                      # NEW
  test_flow_matching.py             # NEW
  test_noise_matching.py            # NEW
  test_score_matching.py            # NEW
  test_sampling.py                  # NEW
  test_train.py                     # NEW
  test_evaluate.py                  # NEW
```

Design rationale: `schedules.py` is the single source of truth for
`alpha(t)`, `sigma(t)` and their time-derivatives, the same role
`unet.py:build_default_unet` already plays for architecture — every method
module imports from it instead of redefining path math, so the 5 methods
can never silently drift onto different time conventions.

---

### Task 1: Shared time convention (`src/schedules.py`)

**Files:**
- Create: `pyproject.toml`
- Create: `src/schedules.py`
- Test: `tests/test_schedules.py`

**Interfaces:**
- Produces: `T_MIN: float`, `BETA_MIN: float`, `BETA_MAX: float`,
  `beta(t: Tensor) -> Tensor`, `expand_t(t: Tensor, x: Tensor) -> Tensor`,
  `class OTPath` and `class VPPath`, each with staticmethods
  `alpha(t) -> Tensor`, `sigma(t) -> Tensor`, `alpha_dot(t) -> Tensor`,
  `sigma_dot(t) -> Tensor`. Every later task imports these names.

- [ ] **Step 1: Create `pyproject.toml` so `tests/` can import `src.*`**

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_schedules.py
import torch

from src.schedules import T_MIN, OTPath, VPPath, beta, expand_t


def test_ot_path_boundaries():
    t = torch.tensor([0.0, 1.0])
    assert torch.allclose(OTPath.alpha(t), torch.tensor([1.0, 0.0]))
    assert torch.allclose(OTPath.sigma(t), torch.tensor([0.0, 1.0]))


def test_ot_path_derivatives_are_constant():
    t = torch.tensor([0.2, 0.7])
    assert torch.allclose(OTPath.alpha_dot(t), torch.tensor([-1.0, -1.0]))
    assert torch.allclose(OTPath.sigma_dot(t), torch.tensor([1.0, 1.0]))


def test_vp_path_is_variance_preserving():
    t = torch.linspace(T_MIN, 1.0, 10)
    variance = VPPath.alpha(t) ** 2 + VPPath.sigma(t) ** 2
    assert torch.allclose(variance, torch.ones_like(variance), atol=1e-5)


def test_vp_path_boundaries():
    t_min = torch.tensor([T_MIN])
    t_max = torch.tensor([1.0])
    assert torch.allclose(VPPath.alpha(t_min), torch.tensor([1.0]), atol=1e-3)
    assert VPPath.sigma(t_max).item() > 0.999
    assert VPPath.alpha(t_max).item() < 0.01


def test_vp_path_derivatives_match_autograd():
    t = torch.linspace(T_MIN, 1.0 - 1e-3, 8, requires_grad=True)
    alpha = VPPath.alpha(t)
    sigma = VPPath.sigma(t)
    (autograd_alpha_dot,) = torch.autograd.grad(alpha.sum(), t, retain_graph=True)
    (autograd_sigma_dot,) = torch.autograd.grad(sigma.sum(), t)
    with torch.no_grad():
        assert torch.allclose(VPPath.alpha_dot(t), autograd_alpha_dot, atol=1e-4)
        assert torch.allclose(VPPath.sigma_dot(t), autograd_sigma_dot, atol=1e-4)


def test_beta_is_linear():
    t = torch.tensor([0.0, 0.5, 1.0])
    expected = 0.1 + t * (20.0 - 0.1)
    assert torch.allclose(beta(t), expected)


def test_expand_t_broadcasts_against_image_tensor():
    t = torch.tensor([0.1, 0.2, 0.3])
    x = torch.zeros(3, 1, 28, 28)
    te = expand_t(t, x)
    assert te.shape == (3, 1, 1, 1)
    assert (te * x).shape == x.shape
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_schedules.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.schedules'`

- [ ] **Step 4: Implement `src/schedules.py`**

```python
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
    """

    @staticmethod
    def alpha(t: torch.Tensor) -> torch.Tensor:
        return 1.0 - t

    @staticmethod
    def sigma(t: torch.Tensor) -> torch.Tensor:
        return t

    @staticmethod
    def alpha_dot(t: torch.Tensor) -> torch.Tensor:
        return -torch.ones_like(t)

    @staticmethod
    def sigma_dot(t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)


class VPPath:
    """Variance-preserving conditional path shared by FM-Diffusion,
    SM-Diffusion (DDPM loss), Score Matching, and Score Flow:
    alpha(t)^2 + sigma(t)^2 = 1 for every t.
    """

    @staticmethod
    def alpha(t: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * _integral_beta(t))

    @staticmethod
    def sigma(t: torch.Tensor) -> torch.Tensor:
        a = VPPath.alpha(t)
        return torch.sqrt((1.0 - a**2).clamp_min(1e-12))

    @staticmethod
    def alpha_dot(t: torch.Tensor) -> torch.Tensor:
        return -0.5 * beta(t) * VPPath.alpha(t)

    @staticmethod
    def sigma_dot(t: torch.Tensor) -> torch.Tensor:
        a, s = VPPath.alpha(t), VPPath.sigma(t)
        return 0.5 * beta(t) * a**2 / s
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_schedules.py -v`
Expected: PASS (7 tests)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/schedules.py tests/test_schedules.py
git commit -m "feat: add shared time convention and probability paths"
```

---

### Task 2: MNIST data pipeline (`src/data.py`)

**Files:**
- Create: `src/data.py`
- Modify: `.gitignore` (ignore the downloaded dataset directory)
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `get_mnist_loader(root: str = "./data", batch_size: int = 128,
  train: bool = True, download: bool = True, num_workers: int = 2) ->
  DataLoader`, yielding `(x0, label)` batches with `x0.shape == (B,1,28,28)`
  and values in `[-1,1]`. `src/train.py` (Task 10) calls this directly.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data.py
import os

import numpy as np
import pytest
import torch
from PIL import Image

from src.data import _TRANSFORM, get_mnist_loader


def test_transform_maps_to_minus_one_one_and_correct_shape():
    img = Image.fromarray(np.zeros((28, 28), dtype=np.uint8))
    x = _TRANSFORM(img)
    assert x.shape == (1, 28, 28)
    assert x.min().item() >= -1.0 - 1e-6
    assert x.max().item() <= 1.0 + 1e-6


def test_transform_black_pixel_maps_to_minus_one():
    img = Image.fromarray(np.zeros((28, 28), dtype=np.uint8))
    x = _TRANSFORM(img)
    assert torch.allclose(x, -torch.ones_like(x), atol=1e-6)


RUN_INTEGRATION = os.environ.get("RUN_INTEGRATION_TESTS") == "1"


@pytest.mark.skipif(not RUN_INTEGRATION, reason="downloads real MNIST; set RUN_INTEGRATION_TESTS=1")
def test_get_mnist_loader_batch_shape(tmp_path):
    loader = get_mnist_loader(root=str(tmp_path), batch_size=4, train=True)
    x0, label = next(iter(loader))
    assert x0.shape == (4, 1, 28, 28)
    assert label.shape == (4,)
    assert x0.min().item() >= -1.0 - 1e-6
    assert x0.max().item() <= 1.0 + 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data'`

- [ ] **Step 3: Implement `src/data.py`**

```python
"""MNIST data pipeline shared by every training script."""
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Maps [0,1] pixel range to [-1,1], matching the UNet's expected input range
# (all 5 methods define x1 ~ N(0,I) as "full noise", so data must be centered).
_TRANSFORM = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)


def get_mnist_loader(
    root: str = "./data",
    batch_size: int = 128,
    train: bool = True,
    download: bool = True,
    num_workers: int = 2,
) -> DataLoader:
    dataset = datasets.MNIST(
        root=root, train=train, download=download, transform=_TRANSFORM
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        drop_last=train,
    )
```

- [ ] **Step 4: Ignore the downloaded dataset directory**

Add to `.gitignore` (near the other project-generated-output entries):

```
# Downloaded datasets
/data/
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_data.py -v`
Expected: PASS (2 tests run; the integration test is skipped unless
`RUN_INTEGRATION_TESTS=1`)

- [ ] **Step 6: Commit**

```bash
git add src/data.py .gitignore tests/test_data.py
git commit -m "feat: add MNIST data pipeline"
```

---

### Task 3: Base `Method` interface (`src/methods/base.py`)

**Files:**
- Modify: `src/methods/base.py` (currently empty)
- Test: `tests/test_base.py`

**Interfaces:**
- Consumes: `build_default_unet` from `src/models/unet.py`; `T_MIN` from
  `src/schedules.py` (Task 1).
- Produces: `class Method(ABC)` with `__init__(net=None)`, `parameters()`,
  `to(device)`, `train()`, `eval()`, `sample_time(batch_size, device) ->
  Tensor` shape `(B,)`, and abstract methods `loss(x0: Tensor) -> Tensor`
  and `velocity(x: Tensor, t: Tensor) -> Tensor`. Every method in Tasks
  4–8 subclasses this; `src/sampling.py` (Task 9) type-hints against it.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_base.py
import pytest
import torch

from src.methods.base import Method
from src.schedules import T_MIN


class _DummyMethod(Method):
    """Minimal concrete subclass, used only to exercise Method's shared code."""

    def loss(self, x0: torch.Tensor) -> torch.Tensor:
        t = self.sample_time(x0.shape[0], x0.device)
        return (self.net(x0, t) ** 2).mean()

    def velocity(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(x, t)


def test_method_is_abstract():
    with pytest.raises(TypeError):
        Method()


def test_sample_time_is_within_bounds():
    method = _DummyMethod()
    t = method.sample_time(batch_size=1000, device=torch.device("cpu"))
    assert t.shape == (1000,)
    assert t.min().item() >= T_MIN
    assert t.max().item() <= 1.0


def test_train_eval_toggle_net_mode():
    method = _DummyMethod()
    method.train()
    assert method.net.training is True
    method.eval()
    assert method.net.training is False


def test_loss_and_velocity_run_end_to_end():
    method = _DummyMethod()
    x0 = torch.randn(2, 1, 28, 28)
    loss = method.loss(x0)
    assert loss.dim() == 0
    assert torch.isfinite(loss)

    t = torch.full((2,), 0.5)
    v = method.velocity(x0, t)
    assert v.shape == x0.shape
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_base.py -v`
Expected: FAIL (`Method` doesn't exist / `ImportError`)

- [ ] **Step 3: Implement `src/methods/base.py`**

```python
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

from src.models.unet import build_default_unet
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_base.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/methods/base.py tests/test_base.py
git commit -m "feat: add shared Method interface"
```

---

### Task 4: FM-OT (`src/methods/flow_matching.py`)

**Files:**
- Create: `src/methods/flow_matching.py`
- Test: `tests/test_flow_matching.py`

**Interfaces:**
- Consumes: `Method` (Task 3), `OTPath`, `expand_t` (Task 1).
- Produces: `class FlowMatchingOT(Method)`, `name = "fm_ot"`. Used by
  `src/train.py` (Task 10) as `METHODS["fm_ot"]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_flow_matching.py
import torch

from src.methods.flow_matching import FlowMatchingOT


def test_fm_ot_loss_is_finite_scalar_with_grad():
    method = FlowMatchingOT()
    x0 = torch.randn(4, 1, 28, 28)
    loss = method.loss(x0)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in method.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_fm_ot_velocity_matches_raw_net_output():
    method = FlowMatchingOT()
    x = torch.randn(2, 1, 28, 28)
    t = torch.full((2,), 0.3)
    with torch.no_grad():
        assert torch.allclose(method.velocity(x, t), method.net(x, t))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_flow_matching.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.methods.flow_matching'`

- [ ] **Step 3: Implement `src/methods/flow_matching.py`**

```python
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
```

Note: `FlowMatchingDiffusion` is included in this file already so Task 5 is
a documentation/test-only task — see below.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_flow_matching.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/methods/flow_matching.py tests/test_flow_matching.py
git commit -m "feat: add FM-OT flow matching method"
```

---

### Task 5: FM-Diffusion (`src/methods/flow_matching.py`)

**Files:**
- Modify: `tests/test_flow_matching.py` (append tests; `FlowMatchingDiffusion`
  itself was already written in Task 4 since it's a 3-line subclass — this
  task's job is proving it actually behaves like a distinct, VP-path method)

**Interfaces:**
- Consumes: `FlowMatchingDiffusion` (already defined in Task 4's file).
- Produces: nothing new — confirms `METHODS["fm_diffusion"]` (Task 10) is
  correct and distinct from `fm_ot`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_flow_matching.py
from src.methods.flow_matching import FlowMatchingDiffusion
from src.schedules import VPPath


def test_fm_diffusion_uses_vp_path():
    assert FlowMatchingDiffusion.path is VPPath
    assert FlowMatchingDiffusion.name == "fm_diffusion"


def test_fm_diffusion_loss_is_finite_scalar_with_grad():
    method = FlowMatchingDiffusion()
    x0 = torch.randn(4, 1, 28, 28)
    loss = method.loss(x0)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in method.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_fm_ot_and_fm_diffusion_targets_differ():
    torch.manual_seed(0)
    x0 = torch.randn(8, 1, 28, 28)
    x1 = torch.randn(8, 1, 28, 28)
    t = torch.full((8,), 0.5)
    from src.schedules import OTPath, expand_t

    te = expand_t(t, x0)
    ot_target = OTPath.alpha_dot(te) * x0 + OTPath.sigma_dot(te) * x1
    vp_target = VPPath.alpha_dot(te) * x0 + VPPath.sigma_dot(te) * x1
    assert not torch.allclose(ot_target, vp_target)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_flow_matching.py -v -k fm_diffusion`
Expected: at this point it should actually already PASS, because
`FlowMatchingDiffusion` was written in Task 4. Confirm this explicitly — if
it fails, `src/methods/flow_matching.py` is missing the class from Task 4
Step 3 and must be fixed before continuing.

- [ ] **Step 3: (no implementation change expected)**

If Step 2 passed, there is nothing to implement — this task exists to give
FM-Diffusion its own reviewable test-gate per the plan's task-sizing rule,
even though its 3-line implementation piggybacks on Task 4.

- [ ] **Step 4: Run the full file to verify everything passes**

Run: `pytest tests/test_flow_matching.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add tests/test_flow_matching.py
git commit -m "test: verify FM-Diffusion uses VP path distinct from FM-OT"
```

---

### Task 6: SM-Diffusion / DDPM loss (`src/methods/noise_matching.py`)

**Files:**
- Modify: `src/methods/noise_matching.py` (currently only a docstring stub)
- Test: `tests/test_noise_matching.py`

**Interfaces:**
- Consumes: `Method` (Task 3), `VPPath`, `beta`, `expand_t` (Task 1).
- Produces: `class NoiseMatchingDiffusion(Method)`, `name = "ddpm"`. Used by
  `src/train.py` (Task 10) as `METHODS["ddpm"]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_noise_matching.py
import torch

from src.methods.noise_matching import NoiseMatchingDiffusion
from src.schedules import VPPath, beta, expand_t


def test_ddpm_loss_is_finite_scalar_with_grad():
    method = NoiseMatchingDiffusion()
    x0 = torch.randn(4, 1, 28, 28)
    loss = method.loss(x0)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in method.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_ddpm_velocity_matches_closed_form():
    method = NoiseMatchingDiffusion()
    x = torch.randn(2, 1, 28, 28)
    t = torch.full((2,), 0.4)

    # Replace the network with a fixed, known "prediction" so the formula
    # can be checked exactly rather than through an untrained black box.
    known_eps = torch.randn(2, 1, 28, 28)
    method.net = lambda x_in, t_in: known_eps

    v = method.velocity(x, t)

    te = expand_t(t, x)
    b = beta(te)
    sigma = VPPath.sigma(te).clamp_min(1e-3)
    expected = -0.5 * b * x + 0.5 * (b / sigma) * known_eps
    assert torch.allclose(v, expected, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_noise_matching.py -v`
Expected: FAIL — `NoiseMatchingDiffusion` doesn't exist yet.

- [ ] **Step 3: Implement `src/methods/noise_matching.py`**

```python
"""SM-Diffusion: continuous-time DDPM loss. The network predicts the noise
epsilon added by the shared variance-preserving (VP) forward process; this is
the standard DDPM objective (Ho et al. 2020) written in continuous time so it
shares T_MIN/VPPath with every other diffusion-based method here.
"""
import torch

from src.methods.base import Method
from src.schedules import VPPath, beta, expand_t


class NoiseMatchingDiffusion(Method):
    name = "ddpm"
    path = VPPath

    def loss(self, x0: torch.Tensor) -> torch.Tensor:
        t = self.sample_time(x0.shape[0], x0.device)
        te = expand_t(t, x0)
        x1 = torch.randn_like(x0)
        x_t = self.path.alpha(te) * x0 + self.path.sigma(te) * x1
        eps_pred = self.net(x_t, t)
        return torch.mean((eps_pred - x1) ** 2)

    def velocity(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Probability-flow ODE for the VP-SDE: dx/dt = f(x,t) - 0.5*g(t)^2*score(x,t)
        # with f(x,t) = -0.5*beta(t)*x, g(t)^2 = beta(t), score = -eps/sigma.
        te = expand_t(t, x)
        eps_pred = self.net(x, t)
        sigma = self.path.sigma(te).clamp_min(1e-3)  # extra safety near t=T_MIN
        b = beta(te)
        return -0.5 * b * x + 0.5 * (b / sigma) * eps_pred
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_noise_matching.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/methods/noise_matching.py tests/test_noise_matching.py
git commit -m "feat: add SM-Diffusion (continuous-time DDPM loss) method"
```

---

### Task 7: Score Matching (`src/methods/score_matching.py`)

**Files:**
- Create: `src/methods/score_matching.py`
- Test: `tests/test_score_matching.py`

**Interfaces:**
- Consumes: `Method` (Task 3), `VPPath`, `beta`, `expand_t` (Task 1).
- Produces: `class ScoreMatching(Method)`, `name = "score"`, with
  `_weight(t) -> Tensor` returning `sigma(t)**2`. Used by `src/train.py`
  (Task 10) as `METHODS["score"]`; subclassed by Task 8's `ScoreFlow`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_score_matching.py
import torch

from src.methods.score_matching import ScoreMatching
from src.schedules import VPPath, beta, expand_t


def test_score_matching_weight_is_sigma_squared():
    method = ScoreMatching()
    t = torch.tensor([0.1, 0.5, 0.9])
    assert torch.allclose(method._weight(t), VPPath.sigma(t) ** 2)


def test_score_matching_loss_is_finite_scalar_with_grad():
    method = ScoreMatching()
    x0 = torch.randn(4, 1, 28, 28)
    loss = method.loss(x0)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in method.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_score_matching_velocity_matches_closed_form():
    method = ScoreMatching()
    x = torch.randn(2, 1, 28, 28)
    t = torch.full((2,), 0.4)

    known_score = torch.randn(2, 1, 28, 28)
    method.net = lambda x_in, t_in: known_score

    v = method.velocity(x, t)

    te = expand_t(t, x)
    b = beta(te)
    expected = -0.5 * b * (x + known_score)
    assert torch.allclose(v, expected, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_score_matching.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.methods.score_matching'`

- [ ] **Step 3: Implement `src/methods/score_matching.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_score_matching.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/methods/score_matching.py tests/test_score_matching.py
git commit -m "feat: add Score Matching (sigma^2-weighted) method"
```

---

### Task 8: Score Flow (`src/methods/score_matching.py`)

**Files:**
- Modify: `src/methods/score_matching.py`
- Modify: `tests/test_score_matching.py`

**Interfaces:**
- Consumes: `ScoreMatching` (Task 7).
- Produces: `class ScoreFlow(ScoreMatching)`, `name = "score_continuous"`,
  overriding only `_weight(t) -> beta(1-t)`. Used by `src/train.py`
  (Task 10) as `METHODS["score_continuous"]`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_score_matching.py
from src.methods.score_matching import ScoreFlow


def test_score_flow_weight_is_beta_of_one_minus_t():
    method = ScoreFlow()
    t = torch.tensor([0.1, 0.5, 0.9])
    assert torch.allclose(method._weight(t), beta(1.0 - t))


def test_score_flow_loss_is_finite_scalar_with_grad():
    method = ScoreFlow()
    x0 = torch.randn(4, 1, 28, 28)
    loss = method.loss(x0)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in method.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_score_flow_velocity_matches_closed_form():
    # ScoreFlow inherits velocity() from ScoreMatching unchanged.
    method = ScoreFlow()
    x = torch.randn(2, 1, 28, 28)
    t = torch.full((2,), 0.4)

    known_score = torch.randn(2, 1, 28, 28)
    method.net = lambda x_in, t_in: known_score

    v = method.velocity(x, t)

    te = expand_t(t, x)
    b = beta(te)
    expected = -0.5 * b * (x + known_score)
    assert torch.allclose(v, expected, atol=1e-5)


def test_score_matching_and_score_flow_weights_differ():
    t = torch.tensor([0.1, 0.5, 0.9])
    sm_weight = ScoreMatching()._weight(t)
    sf_weight = ScoreFlow()._weight(t)
    assert not torch.allclose(sm_weight, sf_weight)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_score_matching.py -v -k score_flow`
Expected: FAIL — `ImportError: cannot import name 'ScoreFlow'`

- [ ] **Step 3: Implement `ScoreFlow` (append to `src/methods/score_matching.py`)**

```python
class ScoreFlow(ScoreMatching):
    """SF: identical loss/velocity code as ScoreMatching; only the loss
    weight changes to beta(1-t) (a likelihood-style weight evaluated at the
    reversed time index — see the spec table in this plan's header)."""

    name = "score_continuous"

    def _weight(self, t: torch.Tensor) -> torch.Tensor:
        return beta(1.0 - t)
```

Note on training stability: near `t = T_MIN`, `beta(1-t)` is close to
`BETA_MAX = 20` while the target score `-x1/sigma(t)` is also large (small
`sigma`), so early-training loss/gradients for `ScoreFlow` can spike more
than `ScoreMatching`'s. `src/train.py` (Task 10) exposes an optional
`--grad-clip` flag for this reason — enable it for `score_continuous` runs
if you observe loss spikes or NaNs.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_score_matching.py -v`
Expected: PASS (7 tests total in this file)

- [ ] **Step 5: Commit**

```bash
git add src/methods/score_matching.py tests/test_score_matching.py
git commit -m "feat: add Score Flow (beta(1-t)-weighted) method"
```

---

### Task 9: Probability Flow ODE sampler (`src/sampling.py`)

**Files:**
- Create: `src/sampling.py`
- Test: `tests/test_sampling.py`

**Interfaces:**
- Consumes: `Method` (Task 3), `T_MIN` (Task 1), `torchdiffeq.odeint`.
- Produces: `sample(method: Method, num_samples: int, shape: tuple[int,int,int]
  = (1,28,28), num_steps: int = 50, device="cpu") -> Tensor` of shape
  `(num_samples, *shape)`. Used by `src/evaluate.py` (Task 12).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sampling.py
import math

import torch
from torch import nn

from src.methods.base import Method
from src.methods.flow_matching import FlowMatchingOT
from src.sampling import sample
from src.schedules import T_MIN


class _LinearDecayMethod(Method):
    """Test double with a known analytic solution: dx/dt = -x has solution
    x(t) = x(1) * exp(-(t-1)) = x(1) * exp(1-t). Used to verify the
    integrator itself, independent of any trained network."""

    def __init__(self):
        super().__init__(net=nn.Identity())

    def loss(self, x0: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("not needed for this test")

    def velocity(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -x


def test_sampler_matches_analytic_solution_for_linear_ode():
    torch.manual_seed(0)
    method = _LinearDecayMethod()
    x1 = torch.randn(5, 1, 4, 4)

    # Monkeypatch: sample() draws its own random x1 internally, so instead we
    # replicate its integration call directly to compare against x1 exactly.
    from torchdiffeq import odeint

    t_grid = torch.linspace(1.0, T_MIN, 2000)

    def ode_func(t, x):
        return method.velocity(x, t.expand(x.shape[0]))

    traj = odeint(ode_func, x1, t_grid, method="euler")
    x0_numeric = traj[-1]
    x0_analytic = x1 * math.exp(1.0 - T_MIN)
    assert torch.allclose(x0_numeric, x0_analytic, atol=1e-2)


def test_sample_returns_correct_shape_and_is_finite():
    method = FlowMatchingOT()
    images = sample(method, num_samples=3, num_steps=5, device="cpu")
    assert images.shape == (3, 1, 28, 28)
    assert torch.isfinite(images).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sampling.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.sampling'`

- [ ] **Step 3: Implement `src/sampling.py`**

```python
"""Shared Probability-Flow-ODE sampler used to draw samples from any of the
five methods. Every Method.velocity(x, t) already returns dx/dt in the same
(t≈0 data, t=1 noise) convention, so this integrator is method-agnostic.
"""
import torch
from torchdiffeq import odeint

from src.methods.base import Method
from src.schedules import T_MIN


@torch.no_grad()
def sample(
    method: Method,
    num_samples: int,
    shape: tuple[int, int, int] = (1, 28, 28),
    num_steps: int = 50,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Integrate dx/dt = method.velocity(x, t) from t=1 (noise) to t=T_MIN
    (data) with a fixed-step Euler solver, identical for all 5 methods."""
    method.eval()
    x1 = torch.randn(num_samples, *shape, device=device)
    t_grid = torch.linspace(1.0, T_MIN, num_steps, device=device)

    def ode_func(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        t_batch = t.expand(x.shape[0])
        return method.velocity(x, t_batch)

    trajectory = odeint(ode_func, x1, t_grid, method="euler")
    return trajectory[-1]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sampling.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/sampling.py tests/test_sampling.py
git commit -m "feat: add shared Probability-Flow-ODE sampler"
```

---

### Task 10: Training CLI (`src/train.py`)

**Files:**
- Create: `src/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `get_mnist_loader` (Task 2), all 5 `Method` subclasses (Tasks
  4–8), `Method` (Task 3).
- Produces: `METHODS: dict[str, type[Method]]` and `CKPT_DIRNAME: dict[str,
  str]` module-level constants (used by `src/evaluate.py`, Task 12), plus a
  `python -m src.train --method ... ` CLI. Checkpoint saved to
  `$CKPT_ROOT/<CKPT_DIRNAME[method]>/model.pt`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train.py
import os
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.train import CKPT_DIRNAME, METHODS, run_training


def test_methods_and_ckpt_dirname_have_matching_keys():
    assert set(METHODS) == set(CKPT_DIRNAME) == {
        "fm_ot",
        "fm_diffusion",
        "ddpm",
        "score",
        "score_continuous",
    }


def test_ckpt_dirnames_match_env_example():
    assert CKPT_DIRNAME == {
        "fm_ot": "ckpt_flow",
        "fm_diffusion": "ckpt_flow_diff",
        "ddpm": "ckpt_ddpm",
        "score": "ckpt_score",
        "score_continuous": "ckpt_score_continuous",
    }


@pytest.mark.parametrize("method_name", list(METHODS))
def test_run_training_one_step_writes_checkpoint(tmp_path, monkeypatch, method_name):
    monkeypatch.setenv("CKPT_ROOT", str(tmp_path))
    # Tiny synthetic dataset so this test needs no MNIST download and no GPU.
    x = torch.randn(4, 1, 28, 28)
    y = torch.zeros(4, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=2)

    run_training(method_name=method_name, loader=loader, epochs=1, lr=1e-3, grad_clip=None)

    ckpt_path = Path(tmp_path) / CKPT_DIRNAME[method_name] / "model.pt"
    assert ckpt_path.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.train'`

- [ ] **Step 3: Implement `src/train.py`**

```python
"""Generic trainer shared by all 5 methods; only --method changes which
Method subclass (and therefore loss/path/head semantics) is trained.
"""
import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import get_mnist_loader
from src.methods.base import Method
from src.methods.flow_matching import FlowMatchingDiffusion, FlowMatchingOT
from src.methods.noise_matching import NoiseMatchingDiffusion
from src.methods.score_matching import ScoreFlow, ScoreMatching

METHODS: dict[str, type[Method]] = {
    "fm_ot": FlowMatchingOT,
    "fm_diffusion": FlowMatchingDiffusion,
    "ddpm": NoiseMatchingDiffusion,
    "score": ScoreMatching,
    "score_continuous": ScoreFlow,
}

# Matches the CKPT_ROOT sub-directories already named in .env.example.
CKPT_DIRNAME: dict[str, str] = {
    "fm_ot": "ckpt_flow",
    "fm_diffusion": "ckpt_flow_diff",
    "ddpm": "ckpt_ddpm",
    "score": "ckpt_score",
    "score_continuous": "ckpt_score_continuous",
}


def run_training(
    method_name: str,
    loader: DataLoader,
    epochs: int,
    lr: float,
    grad_clip: float | None,
    device: torch.device | str = "cpu",
) -> None:
    device = torch.device(device)
    method = METHODS[method_name]().to(device)
    optim = torch.optim.Adam(method.parameters(), lr=lr)

    ckpt_root = Path(os.environ["CKPT_ROOT"]) / CKPT_DIRNAME[method_name]
    ckpt_root.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        method.train()
        running = 0.0
        n_seen = 0
        for x0, _ in tqdm(loader, desc=f"{method_name} epoch {epoch}"):
            x0 = x0.to(device)
            loss = method.loss(x0)
            optim.zero_grad()
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(method.parameters(), grad_clip)
            optim.step()
            running += loss.item() * x0.shape[0]
            n_seen += x0.shape[0]
        print(f"[{method_name}] epoch {epoch}: loss={running / n_seen:.4f}")
        torch.save(method.net.state_dict(), ckpt_root / "model.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=None,
        help="Max grad norm; recommended for score_continuous (see Task 8 notes).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = get_mnist_loader(
        root=args.data_root, batch_size=args.batch_size, train=True
    )
    run_training(
        method_name=args.method,
        loader=loader,
        epochs=args.epochs,
        lr=args.lr,
        grad_clip=args.grad_clip,
        device=device,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_train.py -v`
Expected: PASS (7 tests: 2 static + 5 parametrized)

- [ ] **Step 5: Commit**

```bash
git add src/train.py tests/test_train.py
git commit -m "feat: add generic training CLI for all 5 methods"
```

---

### Task 11: Per-method training scripts (`scripts/train_*.sh`)

**Files:**
- Create: `scripts/train_fm_ot.sh`
- Create: `scripts/train_fm_diffusion.sh`
- Create: `scripts/train_ddpm.sh`
- Create: `scripts/train_score.sh`
- Create: `scripts/train_score_continuous.sh`

**Interfaces:**
- Consumes: `src/train.py`'s `--method` CLI (Task 10), `.env`'s
  `CKPT_ROOT` (existing, see `.env.example`).
- Produces: five `bash scripts/train_<name>.sh` entry points, each a thin
  wrapper so a user never has to remember the exact `--method` string for a
  given checkpoint directory.

- [ ] **Step 1: Create the five scripts**

```bash
#!/usr/bin/env bash
# scripts/train_fm_ot.sh
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
python -m src.train --method fm_ot "$@"
```

```bash
#!/usr/bin/env bash
# scripts/train_fm_diffusion.sh
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
python -m src.train --method fm_diffusion "$@"
```

```bash
#!/usr/bin/env bash
# scripts/train_ddpm.sh
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
python -m src.train --method ddpm "$@"
```

```bash
#!/usr/bin/env bash
# scripts/train_score.sh
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
python -m src.train --method score "$@"
```

```bash
#!/usr/bin/env bash
# scripts/train_score_continuous.sh
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
python -m src.train --method score_continuous --grad-clip 1.0 "$@"
```

- [ ] **Step 2: Make them executable**

```bash
chmod +x scripts/train_fm_ot.sh scripts/train_fm_diffusion.sh \
  scripts/train_ddpm.sh scripts/train_score.sh scripts/train_score_continuous.sh
```

- [ ] **Step 3: Verify each script's `--method` argument is valid**

Run (no `.env` required for this dry check — `--help` exits before reading
`CKPT_ROOT`):

```bash
for f in scripts/train_*.sh; do
  grep -oP '(?<=--method )\S+' "$f"
done
```

Expected output: `fm_ot`, `fm_diffusion`, `ddpm`, `score`,
`score_continuous` — one per line, each a key in `src.train.METHODS`
(verified by Task 10's `test_methods_and_ckpt_dirname_have_matching_keys`).

- [ ] **Step 4: Commit**

```bash
git add scripts/
git commit -m "feat: add per-method training entry-point scripts"
```

---

### Task 12: Sample-grid evaluation (`src/evaluate.py`)

**Files:**
- Create: `src/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `sample` (Task 9), `METHODS`, `CKPT_DIRNAME` (Task 10).
- Produces: `python -m src.evaluate --method ... --out samples.png` CLI
  that loads a checkpoint and writes a sample grid.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluate.py
import os
from pathlib import Path

import torch

from src.evaluate import generate_and_save
from src.train import CKPT_DIRNAME, METHODS


def test_generate_and_save_writes_png(tmp_path, monkeypatch):
    monkeypatch.setenv("CKPT_ROOT", str(tmp_path))
    method_name = "fm_ot"
    ckpt_dir = Path(tmp_path) / CKPT_DIRNAME[method_name]
    ckpt_dir.mkdir(parents=True)

    # Save an untrained-but-valid checkpoint so this test needs no training run.
    net = METHODS[method_name]().net
    torch.save(net.state_dict(), ckpt_dir / "model.pt")

    out_path = tmp_path / "samples.png"
    generate_and_save(
        method_name=method_name,
        num_samples=2,
        num_steps=3,
        out_path=out_path,
        device="cpu",
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evaluate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.evaluate'`

- [ ] **Step 3: Implement `src/evaluate.py`**

```python
"""Draw a sample grid from a trained checkpoint using the shared PF-ODE
sampler (src/sampling.py), for qualitative comparison across all 5 methods.
"""
import argparse
import os
from pathlib import Path

import torch
import torchvision.utils as vutils

from src.sampling import sample
from src.train import CKPT_DIRNAME, METHODS


def generate_and_save(
    method_name: str,
    num_samples: int,
    num_steps: int,
    out_path: Path,
    device: torch.device | str = "cpu",
) -> None:
    device = torch.device(device)
    method = METHODS[method_name]().to(device)
    ckpt_path = Path(os.environ["CKPT_ROOT"]) / CKPT_DIRNAME[method_name] / "model.pt"
    method.net.load_state_dict(torch.load(ckpt_path, map_location=device))

    images = sample(
        method, num_samples=num_samples, num_steps=num_steps, device=device
    )
    images = (images.clamp(-1, 1) + 1) / 2  # [-1,1] -> [0,1] for saving

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(images, out_path, nrow=8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--out", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_path = Path(args.out or f"samples_{args.method}.png")
    generate_and_save(
        method_name=args.method,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        out_path=out_path,
        device=device,
    )
    print(f"Saved {args.num_samples} samples to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_evaluate.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Run the full test suite**

Run: `pytest -v`
Expected: PASS (all tests across Tasks 1–12; integration tests requiring
real MNIST download remain skipped unless `RUN_INTEGRATION_TESTS=1`)

- [ ] **Step 6: Commit**

```bash
git add src/evaluate.py tests/test_evaluate.py
git commit -m "feat: add sample-grid evaluation script"
```

---

## Manual End-to-End Verification (after Task 12)

The unit suite never actually downloads MNIST or trains for real. Once all
12 tasks are green, do one real run to confirm the full pipeline works
end-to-end (requires internet access for the MNIST download and a writable
`CKPT_ROOT`):

```bash
cp .env.example .env   # then edit CKPT_ROOT to a real writable path
bash scripts/train_fm_ot.sh --epochs 1
python -m src.evaluate --method fm_ot --num-samples 16 --num-steps 50 --out /tmp/fm_ot_samples.png
```

Expected: training prints one epoch's average loss and writes
`$CKPT_ROOT/ckpt_flow/model.pt`; evaluation writes a 4×4 grid PNG of
recognizable-ish digit shapes (one epoch is not enough for good samples —
this step is a pipeline smoke test, not a quality bar). Repeat for the other
4 scripts to confirm every method trains and samples without error before
committing to longer training runs for the actual comparison.

---

## Self-Review

**Spec coverage:**
- FM-OT → Task 4. ✅
- FM-Diffusion → Task 5 (implementation in Task 4's file, dedicated test
  gate in Task 5). ✅
- SM-Diffusion (DDPM loss, predicts ε) → Task 6. ✅
- Score Matching (predicts score, weight σ_t²) → Task 7. ✅
- Score Flow (predicts score, weight β(1−t)) → Task 8. ✅
- Shared UNet architecture → enforced by every method calling
  `build_default_unet()` through `Method.__init__` (Task 3); never
  subclassed or modified. ✅
- Shared time convention → `src/schedules.py` (Task 1) is the only place
  `alpha`, `sigma`, `beta`, `T_MIN` are defined; every method imports from
  it. ✅
- Sampling via Probability Flow ODE for all methods → `src/sampling.py`
  (Task 9), consumed identically by `src/evaluate.py` (Task 12) for every
  `method_name`. ✅

**Placeholder scan:** no TBD/TODO, no "add appropriate handling" steps, no
"similar to Task N" call-outs without code — every step has runnable code or
an exact shell command with expected output.

**Type consistency:** `Method.loss(x0: Tensor) -> Tensor` and
`Method.velocity(x: Tensor, t: Tensor) -> Tensor` (Task 3) are used with
identical signatures in every subclass (Tasks 4–8), `src/sampling.py`
(Task 9), and the closed-form tests that monkeypatch `method.net`. The
`METHODS` / `CKPT_DIRNAME` dict keys (Task 10) are asserted equal in
`test_methods_and_ckpt_dirname_have_matching_keys` and reused verbatim in
Task 11's scripts and Task 12's `generate_and_save`.
