"""Compute NLL (bits/dim), FID, and NFE for one trained method, matching the
evaluation protocol described in Song et al. 2021 ("Score-Based Generative
Modeling through SDEs" appendix): NLL is measured on held-out real data;
FID and NFE are measured while drawing --num-fid-samples generated images
with an adaptive-tolerance Probability-Flow-ODE solver.
"""

import argparse
import json
import logging
import os
import sys
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
    get_mnist_cnn_feature_extractor,
)
from src.metrics.likelihood import compute_bpd
from src.sampling import sample_adaptive
from src.train import CKPT_DIRNAME, METHODS, _epoch_of
from src.train import find_all_checkpoints as _find_all_checkpoints_or_empty
from src.train import find_latest_checkpoint as _find_latest_checkpoint_or_none

logger = logging.getLogger(__name__)


def configure_logging(level: str = "INFO") -> None:
    """Called once, from main() -- library functions below only ever log
    through the module-level `logger`, never configure handlers themselves,
    so importing this module (e.g. from tests) never has side effects."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def find_all_checkpoints(ckpt_dir: Path) -> list[Path]:
    """Same discovery/sort as src.train (single source of truth for the
    model_*.pt naming convention), but evaluation has no checkpoint to fall
    back to, so an empty result is an error here rather than "start fresh"."""
    candidates = _find_all_checkpoints_or_empty(ckpt_dir)
    if not candidates:
        raise FileNotFoundError(f"no checkpoint found in {ckpt_dir}")
    return candidates


def load_feature_extractor(
    feature_extractor: str,
    classifier_ckpt: str | Path | None,
    device: torch.device | str,
) -> tuple[nn.Module, Callable[[torch.Tensor], torch.Tensor], str]:
    """Resolves --feature-extractor into (model, transform, stats_tag).
    stats_tag namespaces the cached real-image statistics file (the two
    extractors live in different feature spaces, so their statistics --
    and any FID computed from them -- are not comparable or interchangeable)."""
    if feature_extractor == "inception":
        logger.info("Loading Inception-v3 feature extractor")
        model, transform = get_inception_feature_extractor(device)
        return model, transform, "inception"
    if feature_extractor == "mnist_cnn":
        ckpt = Path(classifier_ckpt) if classifier_ckpt is not None else (
            Path(os.environ["CKPT_ROOT"]) / "ckpt_classifier" / "mnist_cnn.pt"
        )
        if not ckpt.exists():
            raise FileNotFoundError(
                f"no MNIST classifier checkpoint at {ckpt}; train one first "
                "with `python -m src.train_classifier`, or pass --classifier-ckpt"
            )
        logger.info("Loading MNIST-CNN feature extractor")
        model, transform = get_mnist_cnn_feature_extractor(ckpt, device)
        return model, transform, "mnist_cnn"
    raise ValueError(f"unknown --feature-extractor {feature_extractor!r}")


def find_latest_checkpoint(ckpt_dir: Path) -> Path:
    latest = _find_latest_checkpoint_or_none(ckpt_dir)
    if latest is None:
        raise FileNotFoundError(f"no checkpoint found in {ckpt_dir}")
    return latest


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
    logger.info(
        "[%s] NLL: evaluating up to %d held-out test samples (solver=%s, rtol=%.1e, atol=%.1e)",
        method.name, num_nll_samples, solver, rtol, atol,
    )
    bpd_values = []
    n_seen = 0
    for x0, _ in test_loader:
        if n_seen >= num_nll_samples:
            break
        x0 = x0.to(device)
        batch_bpd = compute_bpd(method, x0, solver=solver, rtol=rtol, atol=atol)
        bpd_values.append(batch_bpd)
        n_seen += x0.shape[0]
        running_mean = torch.cat(bpd_values).mean().item()
        logger.info(
            "[%s] NLL progress: %d/%d samples, running mean BPD=%.4f",
            method.name, n_seen, num_nll_samples, running_mean,
        )
    nll_bpd = torch.cat(bpd_values)[:num_nll_samples].mean().item()
    logger.info("[%s] NLL done: %d samples, BPD=%.4f", method.name, min(n_seen, num_nll_samples), nll_bpd)

    # --- FID + NFE over num_fid_samples generated images ---
    logger.info(
        "[%s] FID/NFE: generating %d samples (solver=%s, rtol=%.1e, atol=%.1e)",
        method.name, num_fid_samples, solver, rtol, atol,
    )
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
        logger.info(
            "[%s] FID sampling progress: %d/%d generated (this batch NFE=%d, avg NFE so far=%.1f)",
            method.name, n_generated, num_fid_samples, nfe, sum(nfe_values) / len(nfe_values),
        )
    generated_images = torch.cat(generated, dim=0)

    logger.info("[%s] Extracting Inception features for %d generated images", method.name, n_generated)
    gen_mu, gen_sigma = compute_activation_statistics(
        generated_images, extractor, transform, device
    )
    fid = fid_from_statistics(real_mu, real_sigma, gen_mu, gen_sigma)
    avg_nfe = sum(nfe_values) / len(nfe_values)
    logger.info("[%s] FID done: %.4f (avg NFE=%.1f over %d batches)", method.name, fid, avg_nfe, len(nfe_values))

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
    feature_extractor: str = "inception",
    classifier_ckpt: str | Path | None = None,
) -> dict:
    """Evaluate one checkpoint (default: the latest one on disk for
    method_name) and write its report to <ckpt_dir>/metrics.json. For a
    full per-epoch history use evaluate_epochs() instead."""
    device = torch.device(device)
    ckpt_root = Path(os.environ["CKPT_ROOT"])
    ckpt_dir = ckpt_root / CKPT_DIRNAME[method_name]
    resolved_ckpt = (
        Path(ckpt_path) if ckpt_path is not None else find_latest_checkpoint(ckpt_dir)
    )
    logger.info("[%s] Loading checkpoint %s", method_name, resolved_ckpt)

    method = METHODS[method_name]().to(device)
    method.net.load_state_dict(torch.load(resolved_ckpt, map_location=device, weights_only=True))

    extractor, transform, stats_tag = load_feature_extractor(feature_extractor, classifier_ckpt, device)
    stats_filename = "real_activation_stats.npz" if stats_tag == "inception" else f"real_activation_stats_{stats_tag}.npz"
    real_mu, real_sigma = compute_or_load_real_statistics(
        get_mnist_loader(batch_size=batch_size, train=True, download=True),
        cache_path=ckpt_root / stats_filename,
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
    report["feature_extractor"] = stats_tag

    report_path = ckpt_dir / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("[%s] Wrote report to %s", method_name, report_path)
    return report


def evaluate_epochs(
    method_name: str,
    epochs: list[int] | None = None,
    num_fid_samples: int = 50_000,
    num_nll_samples: int = 10_000,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    solver: str = "dopri5",
    batch_size: int = 500,
    device: torch.device | str = "cpu",
    feature_extractor: str = "inception",
    classifier_ckpt: str | Path | None = None,
) -> list[dict]:
    """Evaluate model_*.pt checkpoints saved for method_name (one per
    training epoch — src/train.py:run_training saves a new one every
    epoch), so NLL/FID/NFE can be tracked against epoch.

    epochs: which epoch checkpoints to evaluate, e.g. [50, 80, 100]. None
    (the default) evaluates every checkpoint found on disk. Raises
    FileNotFoundError if an explicitly requested epoch has no checkpoint.

    Writes one metrics_epoch{N}.json per evaluated checkpoint plus a
    combined metrics_history.json (list of reports, ordered by epoch); the
    last entry is also written to metrics.json, same file evaluate_method()
    writes, so src/compare_methods.py keeps reading the latest epoch."""
    device = torch.device(device)
    ckpt_root = Path(os.environ["CKPT_ROOT"])
    ckpt_dir = ckpt_root / CKPT_DIRNAME[method_name]
    all_checkpoints = find_all_checkpoints(ckpt_dir)

    if epochs is None:
        checkpoints = all_checkpoints
    else:
        by_epoch = {_epoch_of(c): c for c in all_checkpoints}
        missing = sorted(e for e in epochs if e not in by_epoch)
        if missing:
            raise FileNotFoundError(
                f"no checkpoint found for epoch(s) {missing} in {ckpt_dir}; "
                f"available epochs: {sorted(by_epoch)}"
            )
        checkpoints = [by_epoch[e] for e in sorted(set(epochs))]

    logger.info(
        "[%s] Evaluating %d checkpoint(s): epochs %s",
        method_name, len(checkpoints), [_epoch_of(c) for c in checkpoints],
    )

    method = METHODS[method_name]().to(device)
    extractor, transform, stats_tag = load_feature_extractor(feature_extractor, classifier_ckpt, device)
    stats_filename = "real_activation_stats.npz" if stats_tag == "inception" else f"real_activation_stats_{stats_tag}.npz"
    real_mu, real_sigma = compute_or_load_real_statistics(
        get_mnist_loader(batch_size=batch_size, train=True, download=True),
        cache_path=ckpt_root / stats_filename,
        extractor=extractor,
        transform=transform,
        device=device,
        num_samples=num_fid_samples,
    )

    history = []
    for i, ckpt in enumerate(checkpoints, start=1):
        epoch = _epoch_of(ckpt)
        logger.info(
            "[%s] === Checkpoint %d/%d: %s (epoch %d) ===",
            method_name, i, len(checkpoints), ckpt.name, epoch,
        )
        method.net.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        test_loader = get_mnist_loader(
            batch_size=batch_size, train=False, download=True
        )

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
        report["epoch"] = epoch
        report["feature_extractor"] = stats_tag
        (ckpt_dir / f"metrics_epoch{epoch}.json").write_text(json.dumps(report, indent=2))
        history.append(report)
        logger.info(
            "[%s] Checkpoint %d/%d (epoch %d) done: NLL=%.4f FID=%.4f avg_NFE=%.1f",
            method_name, i, len(checkpoints), epoch,
            report["nll_bpd"], report["fid"], report["avg_nfe"],
        )

    (ckpt_dir / "metrics_history.json").write_text(json.dumps(history, indent=2))
    (ckpt_dir / "metrics.json").write_text(json.dumps(history[-1], indent=2))
    logger.info(
        "[%s] Done: wrote metrics_history.json with %d entries to %s",
        method_name, len(history), ckpt_dir,
    )
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
        "--epochs",
        type=int,
        nargs="*",
        default=None,
        metavar="EPOCH",
        help="Evaluate specific epoch checkpoints, e.g. --epochs 50 80 100 "
        "-- writes metrics_epoch{N}.json + metrics_history.json. Pass "
        "--epochs with no values to evaluate every checkpoint found. Omit "
        "entirely to evaluate only the latest checkpoint (default). "
        "Mutually exclusive with --ckpt.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="e.g. 'cuda:0', 'cuda:1', 'cuda:2', or 'cpu'. "
        "Default: cuda:0 if a GPU is available, else cpu.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity for progress tracking (default: INFO).",
    )
    parser.add_argument(
        "--feature-extractor",
        type=str,
        default="inception",
        choices=["inception", "mnist_cnn"],
        help="FID feature space. 'inception' (default) is the literature-"
        "comparable ImageNet-pretrained Inception-v3. 'mnist_cnn' uses a "
        "small CNN trained on MNIST itself (src/train_classifier.py), a "
        "more domain-appropriate but not cross-paper-comparable option.",
    )
    parser.add_argument(
        "--classifier-ckpt",
        type=str,
        default=None,
        help="Checkpoint for --feature-extractor mnist_cnn. Default: "
        "$CKPT_ROOT/ckpt_classifier/mnist_cnn.pt (src/train_classifier.py's "
        "output). Ignored for --feature-extractor inception.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Starting evaluation: method=%s device=%s", args.method, device)

    if args.epochs is not None:
        if args.ckpt is not None:
            raise SystemExit("--epochs and --ckpt are mutually exclusive")
        epochs = args.epochs or None  # `--epochs` with no values -> evaluate all
        history = evaluate_epochs(
            method_name=args.method,
            epochs=epochs,
            num_fid_samples=args.num_fid_samples,
            num_nll_samples=args.num_nll_samples,
            rtol=args.rtol,
            atol=args.atol,
            solver=args.solver,
            batch_size=args.batch_size,
            device=device,
            feature_extractor=args.feature_extractor,
            classifier_ckpt=args.classifier_ckpt,
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
            feature_extractor=args.feature_extractor,
            classifier_ckpt=args.classifier_ckpt,
        )
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
