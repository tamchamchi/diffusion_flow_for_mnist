"""Animates how each method's samples change over training: one frame per
requested epoch, all 5 methods stacked in that frame, using src.sampling's
fixed-step sample() to draw the images.

The same noise x1 is reused for every (method, epoch) cell (via set_seed
right before each sample() call) so what changes frame-to-frame is purely
the network's denoising of the *same* starting noise -- both a fair
side-by-side comparison across methods and a clean "watch training
converge" animation for a single method.
"""

import argparse
import os
from pathlib import Path

import torch
import torchvision.utils as vutils
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

from src.sampling import sample
from src.train import CKPT_DIRNAME, METHODS
from src.utils.seed import set_seed

load_dotenv()

_LABELS: dict[str, str] = {
    "fm_ot": "FM-OT",
    "fm_diffusion": "FM-Diff",
    "ddpm": "DDPM",
    "score": "Score",
    "score_flow": "Score-Flow",
}

_UPSCALE = 3  # 28px MNIST digits are too small to read at native size in a gif
_ROW_LABEL_WIDTH = 110
_TITLE_HEIGHT = 30
_ROW_GAP = 4
_FONT_SIZE = 20


def _grid_for_checkpoint(
    method_name: str,
    ckpt_path: Path,
    num_samples: int,
    num_steps: int,
    seed: int,
    device: torch.device | str,
) -> Image.Image:
    """Loads one checkpoint, draws num_samples images (same noise every
    call, via set_seed), and returns them as one upscaled PIL row image."""
    method = METHODS[method_name]().to(device)  # type: ignore
    method.net.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True)
    )

    set_seed(seed)  # same x1 ~ N(0,I) for every (method, epoch) cell
    images = sample(method, num_samples=num_samples, num_steps=num_steps, device=device)
    images = (images.clamp(-1, 1) + 1) / 2  # [-1,1] -> [0,1]

    grid = vutils.make_grid(images, nrow=num_samples, padding=2)  # (3, H, W)
    grid_img = Image.fromarray((grid.permute(1, 2, 0).numpy() * 255).astype("uint8"))
    return grid_img.resize(
        (grid_img.width * _UPSCALE, grid_img.height * _UPSCALE),
        Image.NEAREST,  # type: ignore
    )


def _compose_frame(
    epoch: int, row_images: dict[str, Image.Image], methods: list[str]
) -> Image.Image:
    """Stacks one row image per method under a title banner naming the
    epoch, with each row prefixed by its method's label."""
    row_h = next(iter(row_images.values())).height
    row_w = next(iter(row_images.values())).width
    frame = Image.new(
        "RGB",
        (_ROW_LABEL_WIDTH + row_w, _TITLE_HEIGHT + len(methods) * (row_h + _ROW_GAP)),
        color=(255, 255, 255),
    )
    draw = ImageDraw.Draw(frame)
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        _FONT_SIZE,
    )
    draw.text((8, 6), f"epoch {epoch}", fill=(0, 0, 0), font=font)

    y = _TITLE_HEIGHT
    for method_name in methods:
        draw.text(
            (8, y + row_h // 2 - 6),
            _LABELS.get(method_name, method_name),
            fill=(0, 0, 0),
            font=font,
        )
        frame.paste(row_images[method_name], (_ROW_LABEL_WIDTH, y))
        y += row_h + _ROW_GAP
    return frame


def make_training_gif(
    ckpt_root: Path,
    epochs: list[int],
    out_path: Path,
    methods: list[str] | None = None,
    num_samples: int = 8,
    num_steps: int = 50,
    seed: int = 42,
    duration_ms: int = 400,
    device: torch.device | str = "cpu",
) -> Path:
    """Renders one frame per epoch (all `methods` stacked in it, each row
    num_samples images sampled from that epoch's checkpoint) and saves an
    animated GIF to out_path. Epochs with no checkpoint on disk for a
    given method are skipped for that method with a printed warning --
    every other method/epoch still renders."""
    methods = methods or list(CKPT_DIRNAME)
    frames = []
    for epoch in epochs:
        row_images = {}
        for method_name in methods:
            ckpt_path = ckpt_root / CKPT_DIRNAME[method_name] / f"model_{epoch}.pt"
            if not ckpt_path.exists():
                print(
                    f"[make_gif] skipping {method_name} epoch {epoch}: no checkpoint at {ckpt_path}"
                )
                continue
            row_images[method_name] = _grid_for_checkpoint(
                method_name, ckpt_path, num_samples, num_steps, seed, device
            )
        if len(row_images) != len(methods):
            print(
                f"[make_gif] epoch {epoch}: missing {set(methods) - row_images.keys()}, skipping this frame"
            )
            continue
        frames.append(_compose_frame(epoch, row_images, methods))
        print(
            f"[make_gif] rendered epoch {epoch} ({len(frames)}/{len(epochs)} frames so far)"
        )

    if not frames:
        raise RuntimeError(
            "no frames rendered -- no epoch had checkpoints for every requested method"
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"[make_gif] saved {len(frames)}-frame GIF to {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=list(range(0, 350, 10)),
        help="Checkpoint epochs to animate, in order. Default: 0 10 20 ... 350",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=None,
        choices=sorted(CKPT_DIRNAME),
        help="Default: all 5 methods.",
    )
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--duration-ms", type=int, default=400, help="Per-frame display time."
    )
    parser.add_argument("--out", type=str, default="figs/training_progress.gif")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_root = Path(os.environ["CKPT_ROOT"])
    make_training_gif(
        ckpt_root=ckpt_root,
        epochs=args.epochs,
        out_path=Path(args.out),
        methods=args.methods,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        seed=args.seed,
        duration_ms=args.duration_ms,
        device=args.device,
    )


if __name__ == "__main__":
    main()
