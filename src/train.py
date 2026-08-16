import argparse
import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import get_mnist_loader
from src.methods.base import Method
from src.methods.flow_matching import FlowMatchingDiffusion, FlowMatchingOT
from src.methods.noise_matching import NoiseMatchingDiffusion
from src.methods.score_matching import ScoreFlow, ScoreMatching

load_dotenv()

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


def _epoch_of(ckpt_path: Path) -> int:
    """model_{N}.pt -> N (this is the only place that name is constructed,
    in run_training below; src/evaluate_metrics.py imports this rather than
    re-deriving it)."""
    return int(Path(ckpt_path).stem.rsplit("_", 1)[-1])


def find_all_checkpoints(ckpt_dir: Path) -> list[Path]:
    """Every model_*.pt checkpoint in ckpt_dir, sorted by epoch number.

    Sorting must be numeric, not lexicographic: "model_10.pt" needs to sort
    after "model_9.pt". Returns [] if ckpt_dir doesn't exist yet or has no
    checkpoints -- callers decide whether that's an error (evaluation) or
    just "start from scratch" (training's --resume).
    """
    if not Path(ckpt_dir).exists():
        return []
    return sorted(Path(ckpt_dir).glob("model_*.pt"), key=_epoch_of)


def find_latest_checkpoint(ckpt_dir: Path) -> Path | None:
    checkpoints = find_all_checkpoints(ckpt_dir)
    return checkpoints[-1] if checkpoints else None


def run_training(
    method_name: str,
    loader: DataLoader,
    epochs: int,
    lr: float,
    grad_clip: float | None,
    device: torch.device | str = "cpu",
    show: bool = True,
    resume: bool = False,
) -> None:
    device = torch.device(device)
    method = METHODS[method_name]().to(device)
    optim = torch.optim.Adam(method.parameters(), lr=lr)

    ckpt_root = Path(os.environ["CKPT_ROOT"]) / CKPT_DIRNAME[method_name]
    ckpt_root.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    if resume:
        latest = find_latest_checkpoint(ckpt_root)
        if latest is not None:
            method.net.load_state_dict(
                torch.load(latest, map_location=device, weights_only=True)
            )
            start_epoch = _epoch_of(latest) + 1
            print(f"[{method_name}] resuming from {latest.name}, starting at epoch {start_epoch}")
        else:
            print(f"[{method_name}] --resume set but no checkpoint found in {ckpt_root}; starting from epoch 0")

    if start_epoch >= epochs:
        print(
            f"[{method_name}] already trained through epoch {start_epoch - 1} "
            f">= --epochs {epochs}; nothing to do"
        )
        return

    # Note: only the network weights are checkpointed (not optimizer state),
    # so a resumed run restarts Adam's moment estimates from scratch each
    # time. Checkpoints are saved every epoch, so the gap this reoptimizes
    # over is at most one epoch.
    for epoch in range(start_epoch, epochs):
        method.train()
        running = 0.0
        n_seen = 0
        for x0, _ in tqdm(
            loader, desc=f"{method_name} epoch {epoch}", disable=not show
        ):
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
        torch.save(method.net.state_dict(), ckpt_root / f"model_{epoch}.pt")


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
        help="Max grad norm; recommended for score_continuous.",
    )
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show training progress bar (default: enabled).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="e.g. 'cuda:0', 'cuda:1', 'cuda:2', or 'cpu'. "
        "Default: cuda:0 if a GPU is available, else cpu.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest model_*.pt checkpoint in "
        "$CKPT_ROOT/<method's ckpt dir>, continuing at the next epoch "
        "instead of starting over at epoch 0 (e.g. after a crash). "
        "No-op if no checkpoint exists yet.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
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
        show=args.show,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
