"""Trains the MNIST digit classifier (src/models/mnist_cnn.py) used by
src/metrics/fid.py as an alternative, MNIST-domain FID feature extractor
(see --feature-extractor mnist_cnn in src/evaluate_metrics.py).

This is a plain supervised classification run -- unrelated to the 5
generative methods trained by src/train.py -- so it gets its own script
and its own checkpoint directory rather than joining src/train.py's
METHODS registry.
"""

import argparse
import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import get_mnist_loader
from src.models import build_default_mnist_cnn

load_dotenv()

# Matches the CKPT_ROOT convention in .env.example / src/train.py.
CKPT_DIRNAME = "ckpt_classifier"


@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module, loader: DataLoader, device: torch.device | str
) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.shape[0]
    return correct / total


def run_training(
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device | str = "cpu",
    show: bool = True,
) -> Path:
    """Trains for `epochs` epochs, printing train loss and test accuracy
    each epoch, and saves the final weights to
    $CKPT_ROOT/ckpt_classifier/mnist_cnn.pt (feature extraction only ever
    needs the final, best-trained classifier -- unlike the generative
    methods, there's no need for a full per-epoch checkpoint history)."""
    device = torch.device(device)
    model = build_default_mnist_cnn().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    ckpt_dir = Path(os.environ["CKPT_ROOT"]) / CKPT_DIRNAME
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    test_acc = None
    for epoch in range(epochs):
        model.train()
        running, n_seen = 0.0, 0
        for x, y in tqdm(
            train_loader, desc=f"classifier epoch {epoch}", disable=not show
        ):
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running += loss.item() * x.shape[0]
            n_seen += x.shape[0]
        test_acc = evaluate_accuracy(model, test_loader, device)
        print(
            f"[classifier] epoch {epoch}: loss={running / n_seen:.4f} test_acc={test_acc:.4f}"
        )

    ckpt_path = ckpt_dir / "mnist_cnn.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"[classifier] saved final weights to {ckpt_path} (test_acc={test_acc:.4f})")
    return ckpt_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-root", type=str, default="./data")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = get_mnist_loader(
        root=args.data_root, batch_size=args.batch_size, train=True
    )
    test_loader = get_mnist_loader(
        root=args.data_root, batch_size=args.batch_size, train=False
    )
    run_training(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        show=args.show,
    )


if __name__ == "__main__":
    main()
