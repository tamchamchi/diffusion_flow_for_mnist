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
    show: bool = True,
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
        torch.save(method.net.state_dict(), ckpt_root / f"model_{epochs}.pt")


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
        show=args.show,
    )


if __name__ == "__main__":
    main()
