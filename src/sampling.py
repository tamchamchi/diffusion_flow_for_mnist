"""Shared Probability-Flow-ODE sampler used to draw samples from any of the
five methods. Every Method.velocity(x, t) already returns dx/dt in the same
(t≈0 data, t=1 noise) convention, so this integrator is method-agnostic.
"""

import argparse
import os
from pathlib import Path

import torch
import torchvision.utils as vutils
from torchdiffeq import odeint

from src.methods.base import Method
from src.schedules import T_MIN
from src.train import CKPT_DIRNAME, METHODS


@torch.no_grad()
def sample(
    method: Method,
    num_samples: int,
    solver: str = "euler",
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

    trajectory = odeint(ode_func, x1, t_grid, method=solver)
    return trajectory[-1]  # type: ignore


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

    images = sample(method, num_samples=num_samples, num_steps=num_steps, device=device)
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