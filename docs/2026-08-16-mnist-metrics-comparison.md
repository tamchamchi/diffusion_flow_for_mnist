# NLL / FID / NFE Comparison Across the 5 Methods — Implementation Plan

> **For agentic workers:** This repo does not currently have the
> `superpowers` plugin's `subagent-driven-development` / `executing-plans`
> skills registered. Execute task-by-task in this session ("Inline
> Execution" style: implement → test → commit, checkpoint between tasks),
> or dispatch one general-purpose sub-agent per task if you prefer isolation
> — either way, do not start task N+1 until task N's tests are green and
> committed. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare the 5 already-implemented generative methods
(`fm_ot`, `fm_diffusion`, `ddpm`, `score`, `score_continuous`, all in
`src/methods/`, trained via `src/train.py`) on 3 metrics: negative
log-likelihood in bits-per-dimension (NLL/BPD), Frechet Inception Distance
(FID; Heusel et al. 2017), and the average number of function evaluations
(NFE) an adaptive-tolerance ODE solver needs to draw a sample — matching
the evaluation protocol used in the score-based generative modeling
literature (e.g. Song et al. 2021).

**Architecture:** Every metric is built on the one hook every method
already exposes — `Method.velocity(x, t)` (the probability-flow-ODE drift,
see `src/methods/base.py`). NFE comes from counting calls to `velocity()`
inside a new adaptive-step `torchdiffeq` integration
(`sample_adaptive()`, extending `src/sampling.py`). NLL/BPD comes from
integrating the *augmented* ODE `(x, log-det)` through `velocity()` with a
Hutchinson trace estimator (the standard continuous-normalizing-flow
likelihood trick, Grathwohl et al. 2018 "FFJORD"), converted to bits/dim
via the exact change-of-variables constant for this repo's `pixel/255`
normalization (`src/data.py`). FID reuses a pretrained torchvision
Inception-v3 as a fixed feature extractor. A per-method CLI
(`src/evaluate_metrics.py`) computes all 3 metrics for one checkpoint and
writes a JSON report; a final script reads all 5 reports into one Markdown
comparison table.

**Tech Stack:** Same as the existing plan — Python 3.11, PyTorch,
torchvision, torchdiffeq, tqdm, pytest — plus `scipy` (for
`scipy.linalg.sqrtm` in the FID formula; already present transitively via
`scikit-learn` in the `flow` conda env, added explicitly to
`environment.yml` in Task 1 for reproducibility) and torchvision's
pretrained `Inception_V3_Weights` (downloaded on first use, same caveat as
the MNIST download already in `src/data.py`).

**Spec:** This plan's spec is the user's in-conversation request: "so sánh
các phương pháp bằng 3 độ đo: NLL, FID, NFE — reporting negative
log-likelihood (NLL) in units of bits per dimension (BPD), sample quality
as measured by the Frechet Inception Distance (FID; Heusel et al. 2017),
and averaged number of function evaluations (NFE) required for the
adaptive solver to reach its prespecified numerical tolerance, averaged
over 50k samples." No separate spec file exists; this plan carries the
requirement inline (see prior plan
`docs/2026-08-16-mnist-five-generative-methods.md` for the already-built
methods this plan evaluates).

## Global Constraints

- Every metric MUST go through `Method.velocity(x, t)` (`src/methods/base.py`)
  — no metric may call `method.net` directly, so metric correctness stays
  tied to each method's already-implemented `velocity()`.
- Reuse `src.train.METHODS: dict[str, type[Method]]` and
  `src.train.CKPT_DIRNAME: dict[str, str]` — do not redefine these dicts
  anywhere else.
- Checkpoints are **not** named `model.pt`: `src/train.py`'s
  `run_training()` saves to `f"model_{epochs}.pt"` every epoch (so the
  filename encodes the `--epochs` value the run was started with, not the
  current epoch). Evaluation code must glob `model_*.pt` and take the
  lexicographically-last match — never assume a fixed filename.
- FID and NFE are measured over `--num-fid-samples` generated images,
  **default 50,000** (the spec's "averaged over 50k samples"). NLL/BPD is
  measured over `--num-nll-samples` held-out real test images, **default
  10,000** (the full MNIST test split) — this is a held-out-likelihood
  metric, not a generated-sample metric, so it does not use the 50k figure.
- Bits-per-dimension MUST use the exact constant for this repo's actual
  normalization (`src/data.py`'s `pixel/255`-based `_TRANSFORM`, **not**
  the commonly-quoted `pixel/256` textbook constant) — see Task 3.
- No changes to `src/schedules.py`, `src/methods/*.py`, `src/models/*.py`,
  or `src/train.py` — this plan is purely additive (new `src/metrics/`
  package + 2 new top-level scripts + one addition to `src/sampling.py`).
- Same stack constraint as before: Python 3.11 + the `flow` conda env
  (`environment.yml`); the only new addition is `scipy` (Task 1).

---

## File Structure

```
environment.yml                    # MODIFY — add scipy (Task 1)
src/
  sampling.py                      # MODIFY — add sample_adaptive() with NFE counting
  metrics/
    __init__.py                    # NEW — empty package marker
    fid.py                         # NEW — Inception features, activation stats, FID formula
    likelihood.py                  # NEW — augmented-ODE NLL/BPD via Hutchinson trace estimator
  evaluate_metrics.py               # NEW — per-method CLI: NLL + FID + NFE -> metrics.json
  compare_methods.py                # NEW — reads all 5 metrics.json -> one Markdown table
scripts/
  evaluate.sh                       # NEW — python -m src.evaluate_metrics --method $1 ...
  compare_all.sh                    # NEW — evaluate.sh for all 5 methods, then compare_methods
tests/
  test_fid.py                       # NEW
  test_sampling.py                  # NEW (adaptive-sampler tests; see note in Task 2)
  test_likelihood.py                # NEW
  test_evaluate_metrics.py          # NEW
  test_compare_methods.py           # NEW
```

Design rationale: `src/metrics/` is a new package (mirroring the existing
`src/methods/` and `src/models/` packages) because FID and likelihood are
each a self-contained, independently testable piece of math, neither of
which any `Method` subclass needs to know about — keeping them out of
`src/methods/` avoids coupling model definitions to evaluation code.

---

### Task 1: FID module (`src/metrics/fid.py`)

**Files:**
- Create: `src/metrics/__init__.py`
- Create: `src/metrics/fid.py`
- Modify: `environment.yml` (add `scipy` to the `pip:` list)
- Test: `tests/test_fid.py`

**Interfaces:**
- Produces: `get_inception_feature_extractor(device="cpu") -> (nn.Module,
  Callable[[Tensor], Tensor])`; `compute_activation_statistics(images:
  Tensor[N,1,28,28], extractor, transform, device="cpu", batch_size=256) ->
  (mu: np.ndarray[2048], sigma: np.ndarray[2048,2048])`;
  `fid_from_statistics(mu1, sigma1, mu2, sigma2, eps=1e-6) -> float`;
  `compute_or_load_real_statistics(loader, cache_path: Path, extractor,
  transform, device="cpu", num_samples=50_000) -> (mu, sigma)`. Used by
  `src/evaluate_metrics.py` (Task 4).

- [ ] **Step 1: Add `scipy` to `environment.yml`**

```yaml
  pip:
    - scikit-learn
    - pytest
    - scipy
```

- [ ] **Step 2: Create `src/metrics/__init__.py`**

```python
"""Evaluation metrics shared across all 5 methods: NLL/BPD (likelihood.py),
FID (fid.py). Every metric is computed purely from a trained Method's
velocity(x, t) hook — see src/methods/base.py.
"""
```

- [ ] **Step 3: Write the failing test**

```python
# tests/test_fid.py
import numpy as np
import torch
from torch import nn

from src.metrics.fid import (
    compute_activation_statistics,
    compute_or_load_real_statistics,
    fid_from_statistics,
)


def test_fid_from_statistics_is_zero_for_identical_distributions():
    mu = np.array([1.0, 2.0, 3.0])
    sigma = np.eye(3) * 2.0
    fid = fid_from_statistics(mu, sigma, mu, sigma)
    assert abs(fid) < 1e-4


def test_fid_from_statistics_matches_closed_form_for_diagonal_gaussians():
    mu1 = np.array([0.0, 0.0])
    mu2 = np.array([3.0, 4.0])
    sigma1 = np.diag([1.0, 4.0])
    sigma2 = np.diag([9.0, 1.0])

    fid = fid_from_statistics(mu1, sigma1, mu2, sigma2)

    # Closed form for commuting (here diagonal) covariances:
    # FID = ||mu1-mu2||^2 + sum_i (sqrt(s1_i) - sqrt(s2_i))^2
    mean_term = np.sum((mu1 - mu2) ** 2)
    trace_term = np.sum((np.sqrt(np.diag(sigma1)) - np.sqrt(np.diag(sigma2))) ** 2)
    expected = mean_term + trace_term
    assert abs(fid - expected) < 1e-4


class _FakeExtractor(nn.Module):
    """4-dim mock feature extractor so these tests never download the real
    ImageNet-pretrained Inception-v3 weights."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3 * 8 * 8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.flatten(1))


def _fake_transform(images: torch.Tensor) -> torch.Tensor:
    images = (images.clamp(-1, 1) + 1) / 2
    images = images.repeat(1, 3, 1, 1)
    return torch.nn.functional.interpolate(images, size=(8, 8), mode="bilinear")


def test_compute_activation_statistics_shapes():
    extractor = _FakeExtractor()
    images = torch.randn(10, 1, 28, 28)
    mu, sigma = compute_activation_statistics(
        images, extractor, _fake_transform, batch_size=4
    )
    assert mu.shape == (4,)
    assert sigma.shape == (4, 4)


def test_compute_or_load_real_statistics_caches_after_first_call(tmp_path):
    extractor = _FakeExtractor()
    images = torch.randn(6, 1, 28, 28)
    loader = [(images, torch.zeros(6))]  # a DataLoader stand-in: any iterable works
    cache_path = tmp_path / "real_stats.npz"

    mu1, sigma1 = compute_or_load_real_statistics(
        loader, cache_path, extractor, _fake_transform, num_samples=6
    )
    assert cache_path.exists()

    class _ExplodingLoader:
        def __iter__(self):
            raise AssertionError("cache was not used")

    mu2, sigma2 = compute_or_load_real_statistics(
        _ExplodingLoader(), cache_path, extractor, _fake_transform, num_samples=6
    )
    assert np.allclose(mu1, mu2)
    assert np.allclose(sigma1, sigma2)
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest tests/test_fid.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.metrics'`

- [ ] **Step 5: Implement `src/metrics/fid.py`**

```python
"""Frechet Inception Distance (FID; Heusel et al. 2017) for MNIST samples.

FID compares 2048-dim Inception-v3 pool features of real vs. generated
images. MNIST is 28x28 grayscale, so every image is resized to Inception's
expected 299x299 and channel-replicated to 3 channels before extraction.
"""
from pathlib import Path
from typing import Callable

import numpy as np
import scipy.linalg
import torch
from torch import nn
from torchvision.models import Inception_V3_Weights, inception_v3


def get_inception_feature_extractor(
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, Callable[[torch.Tensor], torch.Tensor]]:
    """Returns (model, transform) where model(transform(images)) yields
    (N, 2048) pooled features. Requires internet access the first time (to
    download ImageNet-pretrained Inception-v3 weights)."""
    weights = Inception_V3_Weights.DEFAULT
    model = inception_v3(weights=weights, aux_logits=True)
    model.fc = nn.Identity()  # expose the 2048-dim pooled features directly
    model.eval()
    model.to(device)
    preprocess = weights.transforms()

    def transform(images: torch.Tensor) -> torch.Tensor:
        # images: (N, 1, 28, 28) in [-1, 1] -> (N, 3, 299, 299) Inception input
        images = (images.clamp(-1, 1) + 1) / 2  # [-1,1] -> [0,1]
        images = images.repeat(1, 3, 1, 1)
        return preprocess(images)

    return model, transform


@torch.no_grad()
def compute_activation_statistics(
    images: torch.Tensor,
    extractor: nn.Module,
    transform: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device | str = "cpu",
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """(N,1,28,28) images in [-1,1] -> (mu, sigma) of their extractor features."""
    features = []
    for i in range(0, images.shape[0], batch_size):
        batch = images[i : i + batch_size].to(device)
        feats = extractor(transform(batch))
        features.append(feats.cpu().numpy())
    features_np = np.concatenate(features, axis=0)
    mu = features_np.mean(axis=0)
    sigma = np.cov(features_np, rowvar=False)
    return mu, sigma


def fid_from_statistics(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Frechet distance between N(mu1,sigma1) and N(mu2,sigma2)."""
    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean, _ = scipy.linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def compute_or_load_real_statistics(
    loader,
    cache_path: Path,
    extractor: nn.Module,
    transform: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device | str = "cpu",
    num_samples: int = 50_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Real-image activation statistics are identical for every method being
    compared, so this caches them to disk and computes them only once."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        data = np.load(cache_path)
        return data["mu"], data["sigma"]

    collected = []
    n_seen = 0
    for x0, _ in loader:
        if n_seen >= num_samples:
            break
        collected.append(x0)
        n_seen += x0.shape[0]
    images = torch.cat(collected, dim=0)[:num_samples]

    mu, sigma = compute_activation_statistics(images, extractor, transform, device)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, mu=mu, sigma=sigma)
    return mu, sigma
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_fid.py -v`
Expected: PASS (4 tests)

- [ ] **Step 7: Commit**

```bash
git add environment.yml src/metrics/__init__.py src/metrics/fid.py tests/test_fid.py
git commit -m "feat: add FID module (Inception features, activation stats, FID formula)"
```

---

### Task 2: Adaptive-solver sampler with NFE counting (`src/sampling.py`)

**Files:**
- Modify: `src/sampling.py`
- Test: `tests/test_sampling.py`

**Interfaces:**
- Consumes: `Method` (`src/methods/base.py`), `T_MIN`
  (`src/schedules.py`), the existing `cast`/`odeint` imports already in
  `src/sampling.py`.
- Produces: `sample_adaptive(method: Method, num_samples: int, shape:
  tuple[int,int,int] = (1,28,28), solver: str = "dopri5", rtol: float =
  1e-5, atol: float = 1e-5, device="cpu") -> tuple[Tensor, int]` — the
  generated batch and the NFE the solver needed. Used by
  `src/evaluate_metrics.py` (Task 4). Does not modify the existing
  `sample()` function (used for qualitative sample grids elsewhere) or its
  signature.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sampling.py
import torch
from torch import nn

from src.methods.base import Method
from src.methods.flow_matching import FlowMatchingOT
from src.sampling import sample_adaptive


class _LinearDecayMethod(Method):
    """Test double with a known analytic solution: dx/dt = -x. Used to
    verify the adaptive integrator itself, independent of any trained
    network."""

    def __init__(self):
        super().__init__(net=nn.Identity())

    def loss(self, x0: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("not needed for this test")

    def velocity(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -x


def test_sample_adaptive_returns_correct_shape_and_positive_nfe():
    torch.manual_seed(0)
    method = _LinearDecayMethod()
    images, nfe = sample_adaptive(
        method, num_samples=5, shape=(1, 4, 4), rtol=1e-8, atol=1e-8
    )
    assert images.shape == (5, 1, 4, 4)
    assert torch.isfinite(images).all()
    assert nfe > 0


def test_sample_adaptive_nfe_matches_manual_call_count():
    torch.manual_seed(0)
    method = _LinearDecayMethod()
    call_count = 0
    original_velocity = method.velocity

    def counting_velocity(x, t):
        nonlocal call_count
        call_count += 1
        return original_velocity(x, t)

    method.velocity = counting_velocity
    _, nfe = sample_adaptive(
        method, num_samples=3, shape=(1, 4, 4), rtol=1e-6, atol=1e-6
    )
    assert nfe == call_count


def test_sample_adaptive_works_for_a_real_method():
    method = FlowMatchingOT()
    images, nfe = sample_adaptive(method, num_samples=2, rtol=1e-2, atol=1e-2)
    assert images.shape == (2, 1, 28, 28)
    assert torch.isfinite(images).all()
    assert nfe > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sampling.py -v`
Expected: FAIL — `ImportError: cannot import name 'sample_adaptive'`

- [ ] **Step 3: Append `sample_adaptive` to `src/sampling.py`**

```python
def sample_adaptive(
    method: Method,
    num_samples: int,
    shape: tuple[int, int, int] = (1, 28, 28),
    solver: str = "dopri5",
    rtol: float = 1e-5,
    atol: float = 1e-5,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, int]:
    """Integrate dx/dt = method.velocity(x, t) from t=1 (noise) to t=T_MIN
    (data) with an adaptive-step solver run to (rtol, atol), returning both
    the generated batch and the number of function evaluations (NFE) the
    solver needed to reach that tolerance."""
    method.eval()
    x1 = torch.randn(num_samples, *shape, device=device)
    t_grid = torch.tensor([1.0, T_MIN], device=device)

    nfe = 0

    def ode_func(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        nonlocal nfe
        nfe += 1
        t_batch = t.expand(x.shape[0])
        return method.velocity(x, t_batch)

    trajectory = cast(
        torch.Tensor,
        odeint(ode_func, x1, t_grid, method=solver, rtol=rtol, atol=atol),
    )
    return trajectory[-1], nfe
```

Add `@torch.no_grad()` above the `def sample_adaptive(` line, matching the
existing `sample()` function immediately above it in the same file.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sampling.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/sampling.py tests/test_sampling.py
git commit -m "feat: add adaptive-solver sampler with NFE counting"
```

---

### Task 3: NLL/BPD via augmented-ODE likelihood (`src/metrics/likelihood.py`)

**Files:**
- Create: `src/metrics/likelihood.py`
- Test: `tests/test_likelihood.py`

**Interfaces:**
- Consumes: `Method`, `T_MIN` (`src/schedules.py`), `torchdiffeq.odeint`.
- Produces: `standard_normal_log_prob(x: Tensor) -> Tensor` (per-sample,
  nats); `dequantize(x0: Tensor) -> Tensor`; `log_prob_to_bpd(log_p:
  Tensor, dim: int) -> Tensor`; `compute_log_prob(method: Method, x0:
  Tensor, solver="dopri5", rtol=1e-5, atol=1e-5) -> Tensor` (per-sample
  nats); `compute_bpd(method: Method, x0: Tensor, num_mc_samples: int = 1,
  solver="dopri5", rtol=1e-5, atol=1e-5) -> Tensor` (per-sample bits/dim).
  Used by `src/evaluate_metrics.py` (Task 4).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_likelihood.py
import math

import torch

from src.metrics.likelihood import (
    BPD_CONSTANT,
    PIXEL_SCALE,
    compute_bpd,
    compute_log_prob,
    dequantize,
    log_prob_to_bpd,
    standard_normal_log_prob,
)
from src.methods.base import Method
from src.methods.flow_matching import FlowMatchingOT
from src.schedules import T_MIN


def test_standard_normal_log_prob_matches_manual_formula():
    x = torch.tensor([[0.0, 0.0], [1.0, -1.0]])
    d = 2
    expected = -0.5 * (x**2).sum(dim=1) - 0.5 * d * math.log(2 * math.pi)
    assert torch.allclose(standard_normal_log_prob(x), expected)


def test_dequantize_adds_noise_within_one_pixel_step():
    x0 = torch.zeros(5, 1, 4, 4)
    x_deq = dequantize(x0)
    assert (x_deq >= 0).all()
    assert (x_deq < PIXEL_SCALE).all()


def test_bpd_constant_matches_pixel_255_normalization():
    # src.data._TRANSFORM maps pixel/255 -> [-1,1], i.e. x = 2*pixel/255 - 1,
    # NOT the textbook pixel/256 convention, so BPD_CONSTANT != 7 exactly.
    assert math.isclose(BPD_CONSTANT, math.log2(255.0 / 2.0), rel_tol=1e-9)


def test_log_prob_to_bpd_offset_formula():
    dim = 784
    log_p = torch.zeros(3)
    bpd = log_prob_to_bpd(log_p, dim)
    assert torch.allclose(bpd, torch.full((3,), BPD_CONSTANT))


class _ZeroVelocity(Method):
    def loss(self, x0: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def velocity(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


def test_compute_log_prob_matches_prior_for_zero_velocity():
    # v=0 => x never moves => x(1) == x0, and a constant-zero field has zero
    # divergence, so log p_model(x0) == log p_1(x0) exactly, no Monte Carlo
    # noise at all (Hutchinson's estimator is exact for a zero field).
    method = _ZeroVelocity(net=torch.nn.Identity())
    x0 = torch.randn(4, 1, 4, 4)
    log_p = compute_log_prob(method, x0, rtol=1e-8, atol=1e-8)
    assert torch.allclose(log_p, standard_normal_log_prob(x0), atol=1e-5)


class _LinearVelocity(Method):
    def loss(self, x0: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def velocity(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -x


def test_compute_log_prob_matches_closed_form_for_linear_velocity():
    # v=-x has Jacobian -I everywhere, so Hutchinson's estimator with
    # Rademacher eps is exact (eps^T(-I)eps = -||eps||^2 = -dim always),
    # giving a closed form to check against: x(1) = x0*exp(-(1-T_MIN)),
    # logdet = -dim*(1-T_MIN) (both constant along the whole trajectory).
    method = _LinearVelocity(net=torch.nn.Identity())
    torch.manual_seed(0)
    x0 = torch.randn(4, 1, 4, 4)
    log_p = compute_log_prob(method, x0, rtol=1e-8, atol=1e-8)

    dim = x0[0].numel()
    x1_analytic = x0 * math.exp(-(1.0 - T_MIN))
    logdet_analytic = -dim * (1.0 - T_MIN)
    expected = standard_normal_log_prob(x1_analytic) + logdet_analytic
    assert torch.allclose(log_p, expected, atol=1e-3)


def test_compute_bpd_is_finite_and_shaped_per_sample():
    torch.manual_seed(0)
    method = FlowMatchingOT()
    x0 = torch.randn(2, 1, 28, 28).clamp(-1, 1)
    bpd = compute_bpd(method, x0, num_mc_samples=2, rtol=1e-3, atol=1e-3)
    assert bpd.shape == (2,)
    assert torch.isfinite(bpd).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_likelihood.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.metrics.likelihood'`

- [ ] **Step 3: Implement `src/metrics/likelihood.py`**

```python
"""NLL / bits-per-dimension via the instantaneous change-of-variables
formula for continuous normalizing flows (Grathwohl et al. 2018,
"FFJORD"), integrated along the shared probability-flow ODE
(Method.velocity, src/methods/base.py) from t=T_MIN (data) to t=1 (noise),
using a Hutchinson trace estimator for the divergence.
"""
import math

import torch
from torchdiffeq import odeint

from src.methods.base import Method
from src.schedules import T_MIN

# src.data._TRANSFORM maps pixel/255 (not the textbook pixel/256) to
# [-1, 1]: x = 2 * (pixel/255) - 1, so one raw pixel step is this big in
# the model's x-space.
PIXEL_SCALE = 2.0 / 255.0
BPD_CONSTANT = math.log2(1.0 / PIXEL_SCALE)  # = log2(255/2) ~= 6.994353


def standard_normal_log_prob(x: torch.Tensor) -> torch.Tensor:
    """log N(x; 0, I), per sample (summed over all non-batch dims), nats."""
    dim = x[0].numel()
    flat = x.flatten(1)
    return -0.5 * (flat**2).sum(dim=1) - 0.5 * dim * math.log(2 * math.pi)


def dequantize(x0: torch.Tensor) -> torch.Tensor:
    """Uniform dequantization directly in x-space (standard variational
    bound for evaluating a continuous density model on discrete pixels;
    Theis et al. 2016)."""
    return x0 + torch.rand_like(x0) * PIXEL_SCALE


def log_prob_to_bpd(log_p: torch.Tensor, dim: int) -> torch.Tensor:
    return -log_p / (dim * math.log(2)) + BPD_CONSTANT


def compute_log_prob(
    method: Method,
    x0: torch.Tensor,
    solver: str = "dopri5",
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> torch.Tensor:
    """log p_model(x0) in nats, per sample."""
    eps = torch.randint(0, 2, x0.shape, device=x0.device, dtype=x0.dtype) * 2 - 1

    def augmented_ode(t: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]):
        x, _ = state
        with torch.enable_grad():
            x_req = x.detach().requires_grad_(True)
            t_batch = t.expand(x_req.shape[0])
            v = method.velocity(x_req, t_batch)
            (vjp,) = torch.autograd.grad((v * eps).sum(), x_req)
        trace = (vjp * eps).flatten(1).sum(dim=1)
        return v.detach(), trace

    logdet0 = torch.zeros(x0.shape[0], device=x0.device)
    t_grid = torch.tensor([T_MIN, 1.0], device=x0.device)
    x1, logdet = odeint(
        augmented_ode, (x0, logdet0), t_grid, method=solver, rtol=rtol, atol=atol
    )
    x1_final, logdet_final = x1[-1], logdet[-1]
    return standard_normal_log_prob(x1_final) + logdet_final


def compute_bpd(
    method: Method,
    x0: torch.Tensor,
    num_mc_samples: int = 1,
    solver: str = "dopri5",
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> torch.Tensor:
    """Per-sample bits/dim, averaged over num_mc_samples independent
    dequantization + Hutchinson-trace noise draws."""
    dim = x0[0].numel()
    estimates = []
    for _ in range(num_mc_samples):
        x_deq = dequantize(x0)
        log_p = compute_log_prob(method, x_deq, solver=solver, rtol=rtol, atol=atol)
        estimates.append(log_prob_to_bpd(log_p, dim))
    return torch.stack(estimates).mean(dim=0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_likelihood.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/metrics/likelihood.py tests/test_likelihood.py
git commit -m "feat: add NLL/BPD via augmented-ODE likelihood (FFJORD-style)"
```

---

### Task 4: Per-method evaluation CLI (`src/evaluate_metrics.py`)

**Files:**
- Create: `src/evaluate_metrics.py`
- Test: `tests/test_evaluate_metrics.py`

**Interfaces:**
- Consumes: `compute_bpd` (Task 3); `sample_adaptive` (Task 2);
  `compute_activation_statistics`, `compute_or_load_real_statistics`,
  `fid_from_statistics`, `get_inception_feature_extractor` (Task 1);
  `METHODS`, `CKPT_DIRNAME` (`src/train.py`); `get_mnist_loader`
  (`src/data.py`).
- Produces: `find_latest_checkpoint(ckpt_dir: Path) -> Path`;
  `run_evaluation(method, real_mu, real_sigma, extractor, transform,
  test_loader, num_fid_samples, num_nll_samples, rtol, atol, solver,
  device, batch_size) -> dict` (the testable, dependency-injected core);
  `evaluate_method(method_name, num_fid_samples=50_000,
  num_nll_samples=10_000, rtol=1e-5, atol=1e-5, solver="dopri5",
  batch_size=500, device="cpu") -> dict` (the disk/network-facing CLI
  wrapper, writes `$CKPT_ROOT/<dir>/metrics.json`). Used by
  `src/compare_methods.py` (Task 5).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluate_metrics.py
import torch
from torch import nn

from src.evaluate_metrics import run_evaluation
from src.metrics.fid import compute_activation_statistics
from src.methods.flow_matching import FlowMatchingOT


class _FakeExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3 * 8 * 8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.flatten(1))


def _fake_transform(images: torch.Tensor) -> torch.Tensor:
    images = (images.clamp(-1, 1) + 1) / 2
    images = images.repeat(1, 3, 1, 1)
    return torch.nn.functional.interpolate(images, size=(8, 8), mode="bilinear")


def test_run_evaluation_returns_expected_report_shape():
    torch.manual_seed(0)
    method = FlowMatchingOT()
    extractor = _FakeExtractor()

    real_images = torch.randn(8, 1, 28, 28).clamp(-1, 1)
    real_mu, real_sigma = compute_activation_statistics(
        real_images, extractor, _fake_transform
    )

    test_loader = [(torch.randn(4, 1, 28, 28).clamp(-1, 1), torch.zeros(4))]

    report = run_evaluation(
        method=method,
        real_mu=real_mu,
        real_sigma=real_sigma,
        extractor=extractor,
        transform=_fake_transform,
        test_loader=test_loader,
        num_fid_samples=4,
        num_nll_samples=4,
        rtol=1e-2,
        atol=1e-2,
        solver="dopri5",
        device="cpu",
        batch_size=4,
    )

    assert report["method"] == "fm_ot"
    assert isinstance(report["nll_bpd"], float)
    assert isinstance(report["fid"], float)
    assert isinstance(report["avg_nfe"], float)
    assert report["num_fid_samples"] == 4
    assert report["num_nll_samples"] == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evaluate_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.evaluate_metrics'`

- [ ] **Step 3: Implement `src/evaluate_metrics.py`**

```python
"""Compute NLL (bits/dim), FID, and NFE for one trained method, matching the
evaluation protocol described in Song et al. 2021 ("Score-Based Generative
Modeling through SDEs" appendix): NLL is measured on held-out real data;
FID and NFE are measured while drawing --num-fid-samples generated images
with an adaptive-tolerance Probability-Flow-ODE solver.
"""
import argparse
import json
import os
from pathlib import Path
from typing import Callable

import torch
from torch import nn

from src.data import get_mnist_loader
from src.metrics.fid import (
    compute_activation_statistics,
    compute_or_load_real_statistics,
    fid_from_statistics,
    get_inception_feature_extractor,
)
from src.metrics.likelihood import compute_bpd
from src.methods.base import Method
from src.sampling import sample_adaptive
from src.train import CKPT_DIRNAME, METHODS


def find_latest_checkpoint(ckpt_dir: Path) -> Path:
    candidates = sorted(Path(ckpt_dir).glob("model_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"no checkpoint found in {ckpt_dir}")
    return candidates[-1]


def run_evaluation(
    method: Method,
    real_mu,
    real_sigma,
    extractor: nn.Module,
    transform: Callable[[torch.Tensor], torch.Tensor],
    test_loader,
    num_fid_samples: int,
    num_nll_samples: int,
    rtol: float,
    atol: float,
    solver: str,
    device: torch.device | str,
    batch_size: int,
) -> dict:
    """The testable core: every I/O-bound dependency (checkpoint, real
    statistics, Inception weights, MNIST test loader) is passed in rather
    than loaded here, so tests can substitute lightweight fakes."""
    device = torch.device(device)
    method.eval()

    # --- NLL (bits/dim) on held-out real data ---
    bpd_values = []
    n_seen = 0
    for x0, _ in test_loader:
        if n_seen >= num_nll_samples:
            break
        x0 = x0.to(device)
        bpd_values.append(compute_bpd(method, x0, solver=solver, rtol=rtol, atol=atol))
        n_seen += x0.shape[0]
    nll_bpd = torch.cat(bpd_values)[:num_nll_samples].mean().item()

    # --- FID + NFE over num_fid_samples generated images ---
    generated = []
    nfe_values = []
    n_generated = 0
    while n_generated < num_fid_samples:
        n = min(batch_size, num_fid_samples - n_generated)
        images, nfe = sample_adaptive(
            method, num_samples=n, rtol=rtol, atol=atol, solver=solver, device=device
        )
        generated.append(images.cpu())
        nfe_values.append(nfe)
        n_generated += n
    generated_images = torch.cat(generated, dim=0)

    gen_mu, gen_sigma = compute_activation_statistics(
        generated_images, extractor, transform, device
    )
    fid = fid_from_statistics(real_mu, real_sigma, gen_mu, gen_sigma)
    avg_nfe = sum(nfe_values) / len(nfe_values)

    return {
        "method": method.name,
        "nll_bpd": nll_bpd,
        "fid": fid,
        "avg_nfe": avg_nfe,
        "num_fid_samples": n_generated,
        "num_nll_samples": min(n_seen, num_nll_samples),
    }


def evaluate_method(
    method_name: str,
    num_fid_samples: int = 50_000,
    num_nll_samples: int = 10_000,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    solver: str = "dopri5",
    batch_size: int = 500,
    device: torch.device | str = "cpu",
) -> dict:
    device = torch.device(device)
    ckpt_root = Path(os.environ["CKPT_ROOT"])
    ckpt_dir = ckpt_root / CKPT_DIRNAME[method_name]

    method = METHODS[method_name]().to(device)
    ckpt_path = find_latest_checkpoint(ckpt_dir)
    method.net.load_state_dict(torch.load(ckpt_path, map_location=device))

    extractor, transform = get_inception_feature_extractor(device)
    real_mu, real_sigma = compute_or_load_real_statistics(
        get_mnist_loader(batch_size=batch_size, train=True, download=True),
        cache_path=ckpt_root / "real_activation_stats.npz",
        extractor=extractor,
        transform=transform,
        device=device,
        num_samples=num_fid_samples,
    )
    test_loader = get_mnist_loader(batch_size=batch_size, train=False, download=True)

    report = run_evaluation(
        method=method,
        real_mu=real_mu,
        real_sigma=real_sigma,
        extractor=extractor,
        transform=transform,
        test_loader=test_loader,
        num_fid_samples=num_fid_samples,
        num_nll_samples=num_nll_samples,
        rtol=rtol,
        atol=atol,
        solver=solver,
        device=device,
        batch_size=batch_size,
    )
    report["checkpoint"] = str(ckpt_path)

    report_path = ckpt_dir / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--num-nll-samples", type=int, default=10_000)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--solver", type=str, default="dopri5")
    parser.add_argument("--batch-size", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = evaluate_method(
        method_name=args.method,
        num_fid_samples=args.num_fid_samples,
        num_nll_samples=args.num_nll_samples,
        rtol=args.rtol,
        atol=args.atol,
        solver=args.solver,
        batch_size=args.batch_size,
        device=device,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_evaluate_metrics.py -v`
Expected: PASS (1 test — it exercises real `compute_bpd`/`sample_adaptive`
code paths with tiny sizes and loose tolerances, so allow it a bit longer
than the other unit tests, typically well under a minute on CPU)

- [ ] **Step 5: Commit**

```bash
git add src/evaluate_metrics.py tests/test_evaluate_metrics.py
git commit -m "feat: add per-method NLL/FID/NFE evaluation CLI"
```

---

### Task 5: Cross-method comparison table (`src/compare_methods.py`)

**Files:**
- Create: `src/compare_methods.py`
- Test: `tests/test_compare_methods.py`

**Interfaces:**
- Consumes: `CKPT_DIRNAME`, `METHODS` (`src/train.py`); reads
  `metrics.json` files written by `evaluate_method` (Task 4).
- Produces: `load_reports(ckpt_root: Path) -> list[dict]`;
  `render_markdown_table(reports: list[dict]) -> str`. CLI writes to
  `--out` (default `comparison.md`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_compare_methods.py
import json

from src.compare_methods import load_reports, render_markdown_table
from src.train import CKPT_DIRNAME


def test_load_reports_reads_present_reports_and_fills_missing(tmp_path):
    # Only fm_ot has a metrics.json; the other 4 are missing.
    fm_ot_dir = tmp_path / CKPT_DIRNAME["fm_ot"]
    fm_ot_dir.mkdir(parents=True)
    (fm_ot_dir / "metrics.json").write_text(
        json.dumps({"method": "fm_ot", "nll_bpd": 1.23, "fid": 45.6, "avg_nfe": 78.9})
    )

    reports = load_reports(tmp_path)

    assert len(reports) == 5
    by_method = {r["method"]: r for r in reports}
    assert by_method["fm_ot"]["nll_bpd"] == 1.23
    assert by_method["ddpm"]["nll_bpd"] is None


def test_render_markdown_table_formats_present_and_missing_rows():
    reports = [
        {"method": "fm_ot", "nll_bpd": 1.234, "fid": 45.67, "avg_nfe": 78.9},
        {"method": "ddpm", "nll_bpd": None, "fid": None, "avg_nfe": None},
    ]
    table = render_markdown_table(reports)

    assert "| fm_ot | 1.234 | 45.670 | 78.9 |" in table
    assert "| ddpm | — | — | — |" in table
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_compare_methods.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.compare_methods'`

- [ ] **Step 3: Implement `src/compare_methods.py`**

```python
"""Reads every trained method's metrics.json (src/evaluate_metrics.py) and
renders one Markdown table comparing NLL/BPD, FID, and avg. NFE.
"""
import argparse
import json
import os
from pathlib import Path

from src.train import CKPT_DIRNAME, METHODS


def load_reports(ckpt_root: Path) -> list[dict]:
    ckpt_root = Path(ckpt_root)
    reports = []
    for method_name, dirname in CKPT_DIRNAME.items():
        path = ckpt_root / dirname / "metrics.json"
        if path.exists():
            reports.append(json.loads(path.read_text()))
        else:
            reports.append(
                {"method": method_name, "nll_bpd": None, "fid": None, "avg_nfe": None}
            )
    return reports


def render_markdown_table(reports: list[dict]) -> str:
    lines = ["| Method | NLL (BPD) | FID | Avg. NFE |", "|---|---|---|---|"]
    for r in reports:
        nll = f"{r['nll_bpd']:.3f}" if r.get("nll_bpd") is not None else "—"
        fid = f"{r['fid']:.3f}" if r.get("fid") is not None else "—"
        nfe = f"{r['avg_nfe']:.1f}" if r.get("avg_nfe") is not None else "—"
        lines.append(f"| {r['method']} | {nll} | {fid} | {nfe} |")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="comparison.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_root = Path(os.environ["CKPT_ROOT"])
    reports = load_reports(ckpt_root)
    table = render_markdown_table(reports)
    Path(args.out).write_text(table)
    print(table)


if __name__ == "__main__":
    main()
```

Note: `METHODS` is imported but unused by name in this file's logic (only
`CKPT_DIRNAME`'s keys are iterated) — it is imported anyway to keep the
"one place defines the 5 method names" contract visible/greppable in every
file that touches all 5 methods; if a linter flags it as unused, prefer
`from src.train import CKPT_DIRNAME` alone.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_compare_methods.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/compare_methods.py tests/test_compare_methods.py
git commit -m "feat: add cross-method NLL/FID/NFE comparison table"
```

---

### Task 6: Evaluation shell scripts (`scripts/evaluate.sh`, `scripts/compare_all.sh`)

**Files:**
- Create: `scripts/evaluate.sh`
- Create: `scripts/compare_all.sh`

**Interfaces:**
- Consumes: `src/evaluate_metrics.py`'s `--method` CLI (Task 4),
  `src/compare_methods.py`'s CLI (Task 5), `.env`'s `CKPT_ROOT` (existing),
  the `logs/` convention already used by `scripts/train_*.sh`.

- [ ] **Step 1: Create the two scripts**

```bash
#!/usr/bin/env bash
# scripts/evaluate.sh
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
python -m src.evaluate_metrics --method "$1" "${@:2}" 2>&1 | tee "logs/eval_$1.txt"
```

```bash
#!/usr/bin/env bash
# scripts/compare_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
for m in fm_ot fm_diffusion ddpm score score_continuous; do
  bash scripts/evaluate.sh "$m" "$@"
done
python -m src.compare_methods --out comparison.md
```

- [ ] **Step 2: Make them executable**

```bash
chmod +x scripts/evaluate.sh scripts/compare_all.sh
```

- [ ] **Step 3: Verify the method loop matches `src.train.METHODS`**

Run:

```bash
grep -oP '(?<=for m in )[a-z_ ]+(?= in)' scripts/compare_all.sh
```

Expected output: `fm_ot fm_diffusion ddpm score score_continuous` — must be
exactly the 5 keys of `src.train.METHODS`
(`tests/test_train.py::test_methods_and_ckpt_dirname_have_matching_keys`
already guards this set from the prior plan).

- [ ] **Step 4: Run the full test suite**

Run: `pytest -v`
Expected: PASS (all tests across both plans' tasks)

- [ ] **Step 5: Commit**

```bash
git add scripts/evaluate.sh scripts/compare_all.sh
git commit -m "feat: add evaluation entry-point scripts"
```

---

## Manual End-to-End Verification (after Task 6)

The unit suite never runs a real 50k-sample evaluation — that is compute-
heavy (an adaptive-solver ODE integration per generated image batch, an
Inception-v3 forward pass per image, and a full augmented-ODE likelihood
solve per NLL sample) and expected to take a long time even on a GPU.
Smoke-test the pipeline at small scale first, once at least one method has
a real checkpoint (see the prior plan's own manual verification section):

```bash
bash scripts/evaluate.sh fm_ot --num-fid-samples 16 --num-nll-samples 16
```

Expected: prints a JSON report with finite `nll_bpd`, `fid`, `avg_nfe`
values and writes it to `$CKPT_ROOT/ckpt_flow/metrics.json`; the first run
also downloads Inception-v3 weights (needs internet) and creates
`$CKPT_ROOT/real_activation_stats.npz`. Only after this smoke test passes,
scale up:

```bash
bash scripts/compare_all.sh
cat comparison.md
```

This runs the full default (`--num-fid-samples 50000
--num-nll-samples 10000`) protocol for all 5 methods sequentially — budget
real wall-clock time for this (hours, GPU recommended) — and writes the
final comparison table to `comparison.md`.

---

## Self-Review

**Spec coverage:**
- NLL in bits/dim → Task 3 (`compute_bpd`), wired into the per-method
  report in Task 4. ✅
- FID (Heusel et al. 2017) → Task 1 (`fid_from_statistics` +
  Inception-v3 features), wired into Task 4. ✅
- Average NFE for an adaptive solver reaching a prespecified tolerance →
  Task 2 (`sample_adaptive`, `rtol`/`atol` params, NFE = call count),
  averaged across generation batches in Task 4. ✅
- "Averaged over 50k samples" → `--num-fid-samples` default `50_000` in
  Task 4, applied to both the FID generation loop and the NFE averaging
  (both draw from the same `sample_adaptive` calls). ✅
- Comparison across all 5 methods → Task 5 renders one table from all 5
  `CKPT_DIRNAME` entries, Task 6 automates running all 5. ✅

**Placeholder scan:** no TBD/TODO, no "add appropriate handling" steps, no
"similar to Task N" call-outs without code — every step has runnable code
or an exact shell command with expected output.

**Type consistency:** `Method.velocity(x: Tensor, t: Tensor) -> Tensor`
(pre-existing, `src/methods/base.py`) is the only hook every new function
touches — `sample_adaptive` (Task 2) calls it inside `odeint`'s `ode_func`;
`compute_log_prob` (Task 3) calls it inside the augmented-state
`augmented_ode`. `run_evaluation`'s dict keys (`method`, `nll_bpd`, `fid`,
`avg_nfe`, `num_fid_samples`, `num_nll_samples`, Task 4) are read back
verbatim by `render_markdown_table` (Task 5) via `.get(...)`, so a renamed
key in one would be caught by Task 5's tests failing to find data (both
tasks' tests independently pin the exact key names, so a mismatch fails
loudly rather than silently rendering "—" for real data).
