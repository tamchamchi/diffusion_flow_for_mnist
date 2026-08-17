"""Shared Probability-Flow-ODE sampler used to draw samples from any of the
five methods. Every Method.velocity(x, t) already returns dx/dt in the same
(t≈0 data, t=1 noise) convention, so this integrator is method-agnostic.
"""

import argparse
import os
from pathlib import Path

import torch
import torchvision.utils as vutils
from dotenv import load_dotenv
from torchdiffeq import odeint

from src.methods.base import Method
from src.schedules import T_MIN
from src.train import CKPT_DIRNAME, METHODS, find_latest_checkpoint

load_dotenv()


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
    (data) with a fixed-step Euler solver, identical for all 5 methods.
    Returns (num_samples, *shape) generated images in [-1, 1]."""
    method.eval()
    x1 = torch.randn(num_samples, *shape, device=device)
    t_grid = torch.linspace(1.0, T_MIN, num_steps, device=device)

    def ode_func(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        t_batch = t.expand(x.shape[0])
        return method.velocity(x, t_batch)

    trajectory = odeint(ode_func, x1, t_grid, method=solver)
    return trajectory[-1]  # type: ignore


@torch.no_grad()
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
    the generated batch -- (num_samples, *shape) images in [-1, 1] -- and
    the number of function evaluations (NFE) the solver needed to reach
    that tolerance."""
    method.eval()
    x1 = torch.randn(num_samples, *shape, device=device)
    t_grid = torch.tensor([1.0, T_MIN], device=device)

    nfe = 0

    def ode_func(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        nonlocal nfe
        nfe += 1
        t_batch = t.expand(x.shape[0])
        return method.velocity(x, t_batch)

    trajectory = odeint(ode_func, x1, t_grid, method=solver, rtol=rtol, atol=atol)
    return trajectory[-1], nfe  # type: ignore


def generate_and_save(
    method_name: str,
    num_samples: int,
    num_steps: int,
    out_path: Path,
    ckpt_path: str | None = None,
    device: torch.device | str = "cpu",
) -> None:
    """Loads method_name's checkpoint (latest one on disk if ckpt_path is
    None), draws num_samples images with the fixed-step sample() above,
    and saves them as one num_samples/8-row grid PNG at out_path."""
    device = torch.device(device)
    method = METHODS[method_name]().to(device)
    if ckpt_path is None:
        ckpt_root = Path(os.environ["CKPT_ROOT"]) / CKPT_DIRNAME[method_name]  # type: ignore
        ckpt_path = find_latest_checkpoint(ckpt_root)  # type: ignore
    else:
        ckpt_path = Path(ckpt_path)  # type: ignore
    method.net.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True)  # type: ignore
    )

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
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out or f"samples_{args.method}.png")
    generate_and_save(
        method_name=args.method,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        out_path=out_path,
        ckpt_path=args.ckpt,
        device=args.device,
    )
    print(f"Saved {args.num_samples} samples to {out_path}")


if __name__ == "__main__":
    main()
