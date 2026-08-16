"""Compute NLL (bits/dim), FID, and NFE for one trained method, matching the
evaluation protocol described in Song et al. 2021 ("Score-Based Generative
Modeling through SDEs" appendix): NLL is measured on held-out real data;
FID and NFE are measured while drawing --num-fid-samples generated images
with an adaptive-tolerance Probability-Flow-ODE solver.
"""

import argparse
import json
import os
from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn

from src.data import get_mnist_loader
from src.methods.base import Method
from src.metrics.fid import (
    compute_activation_statistics,
    compute_or_load_real_statistics,
    fid_from_statistics,
    get_inception_feature_extractor,
)
from src.metrics.likelihood import compute_bpd
from src.sampling import sample_adaptive
from src.train import CKPT_DIRNAME, METHODS


def _epoch_of(ckpt_path: Path) -> int:
    """model_{N}.pt -> N (see src/train.py:run_training, which saves one
    checkpoint per epoch under this name)."""
    return int(Path(ckpt_path).stem.rsplit("_", 1)[-1])


def find_all_checkpoints(ckpt_dir: Path) -> list[Path]:
    """Every model_*.pt checkpoint in ckpt_dir, sorted by epoch number.

    Sorting must be numeric, not lexicographic: "model_10.pt" needs to sort
    after "model_9.pt", which plain string sorting gets wrong.
    """
    candidates = sorted(Path(ckpt_dir).glob("model_*.pt"), key=_epoch_of)
    if not candidates:
        raise FileNotFoundError(f"no checkpoint found in {ckpt_dir}")
    return candidates


def find_latest_checkpoint(ckpt_dir: Path) -> Path:
    return find_all_checkpoints(ckpt_dir)[-1]


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
    ckpt_path: str | Path | None = None,
    num_fid_samples: int = 50_000,
    num_nll_samples: int = 10_000,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    solver: str = "dopri5",
    batch_size: int = 500,
    device: torch.device | str = "cpu",
) -> dict:
    """Evaluate one checkpoint (default: the latest one on disk for
    method_name) and write its report to <ckpt_dir>/metrics.json. For a
    full per-epoch history use evaluate_all_epochs() instead."""
    device = torch.device(device)
    ckpt_root = Path(os.environ["CKPT_ROOT"])
    ckpt_dir = ckpt_root / CKPT_DIRNAME[method_name]
    resolved_ckpt = (
        Path(ckpt_path) if ckpt_path is not None else find_latest_checkpoint(ckpt_dir)
    )

    method = METHODS[method_name]().to(device)
    method.net.load_state_dict(torch.load(resolved_ckpt, map_location=device))

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
    report["checkpoint"] = str(resolved_ckpt)
    report["epoch"] = _epoch_of(resolved_ckpt)

    (ckpt_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    return report


def evaluate_all_epochs(
    method_name: str,
    num_fid_samples: int = 50_000,
    num_nll_samples: int = 10_000,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    solver: str = "dopri5",
    batch_size: int = 500,
    device: torch.device | str = "cpu",
) -> list[dict]:
    """Evaluate every model_*.pt checkpoint saved for method_name (one per
    training epoch — src/train.py:run_training saves a new one every
    epoch), so NLL/FID/NFE can be tracked against epoch. Writes one
    metrics_epoch{N}.json per checkpoint plus a combined
    metrics_history.json (list of reports, ordered by epoch); the last
    entry is also written to metrics.json, same file evaluate_method()
    writes, so src/compare_methods.py keeps reading the latest epoch."""
    device = torch.device(device)
    ckpt_root = Path(os.environ["CKPT_ROOT"])
    ckpt_dir = ckpt_root / CKPT_DIRNAME[method_name]
    checkpoints = find_all_checkpoints(ckpt_dir)

    method = METHODS[method_name]().to(device)
    extractor, transform = get_inception_feature_extractor(device)
    real_mu, real_sigma = compute_or_load_real_statistics(
        get_mnist_loader(batch_size=batch_size, train=True, download=True),
        cache_path=ckpt_root / "real_activation_stats.npz",
        extractor=extractor,
        transform=transform,
        device=device,
        num_samples=num_fid_samples,
    )

    history = []
    for ckpt in checkpoints:
        method.net.load_state_dict(torch.load(ckpt, map_location=device))
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
        report["checkpoint"] = str(ckpt)
        report["epoch"] = _epoch_of(ckpt)
        (ckpt_dir / f"metrics_epoch{report['epoch']}.json").write_text(
            json.dumps(report, indent=2)
        )
        history.append(report)

    (ckpt_dir / "metrics_history.json").write_text(json.dumps(history, indent=2))
    (ckpt_dir / "metrics.json").write_text(json.dumps(history[-1], indent=2))
    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--num-nll-samples", type=int, default=10_000)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--solver", type=str, default="dopri5")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Evaluate one specific checkpoint file instead of the latest.",
    )
    parser.add_argument(
        "--all-epochs",
        action="store_true",
        help="Evaluate every saved epoch checkpoint (not just the latest), "
        "writing metrics_epoch{N}.json + metrics_history.json. "
        "Mutually exclusive with --ckpt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.all_epochs:
        if args.ckpt is not None:
            raise SystemExit("--all-epochs and --ckpt are mutually exclusive")
        history = evaluate_all_epochs(
            method_name=args.method,
            num_fid_samples=args.num_fid_samples,
            num_nll_samples=args.num_nll_samples,
            rtol=args.rtol,
            atol=args.atol,
            solver=args.solver,
            batch_size=args.batch_size,
            device=device,
        )
        print(json.dumps(history, indent=2))
    else:
        report = evaluate_method(
            method_name=args.method,
            ckpt_path=args.ckpt,
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
