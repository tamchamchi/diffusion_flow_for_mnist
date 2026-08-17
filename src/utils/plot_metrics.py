"""Plots FID vs. training epoch for every method, reading the
metrics_history*.json files src/evaluate_metrics.py:evaluate_epochs()
writes to each method's checkpoint directory.
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
from dotenv import load_dotenv

from src.train import CKPT_DIRNAME
from src.utils.metrics_paths import metrics_history_filename

load_dotenv()

# Categorical palette, fixed hue order (never reassigned per-plot) --
# same 5 slots used everywhere else FID is broken down by method.
_COLORS: dict[str, str] = {
    "fm_ot": "#2a78d6",
    "fm_diffusion": "#eb6834",
    "ddpm": "#1baf7a",
    "score": "#eda100",
    "score_continuous": "#e87ba4",
}
# Marker shape doubles as a colorblind-safe secondary encoding for identity,
# alongside the legend -- never color alone.
_MARKERS: dict[str, str] = {
    "fm_ot": "o",
    "fm_diffusion": "s",
    "ddpm": "^",
    "score": "D",
    "score_continuous": "v",
}
_LABELS: dict[str, str] = {
    "fm_ot": "FM-OT",
    "fm_diffusion": "FM-Diffusion",
    "ddpm": "DDPM",
    "score": "Score",
    "score_continuous": "Score (continuous)",
}
_EXTRACTOR_LABELS: dict[str, str] = {
    "inception": "Inception-v3",
    "mnist_cnn": "MNIST-CNN",
}


def load_fid_by_epoch(
    ckpt_dir: Path, epochs: list[int], feature_extractor: str
) -> list[float | None]:
    """Reads the metrics_history*.json file matching `feature_extractor`
    (src/utils/metrics_paths.py -- "inception" or "mnist_cnn", two
    different feature spaces whose FID numbers are not comparable, so each
    lives in its own file rather than being mixed into one) and returns
    one FID value per requested epoch. The `feature_extractor` field on
    each entry is checked too, as a second line of defense: reports
    written before that field existed are treated as "inception", the
    only extractor available at the time. None for any epoch with no
    matching entry, so the line plots a gap there instead of a wrong point."""
    history_path = ckpt_dir / metrics_history_filename(feature_extractor)
    if not history_path.exists():
        return [None] * len(epochs)
    history = json.loads(history_path.read_text())
    fid_by_epoch = {
        r["epoch"]: r["fid"]
        for r in history
        if r.get("feature_extractor", "inception") == feature_extractor
    }
    return [fid_by_epoch.get(e) for e in epochs]


def plot_fid_vs_epoch(
    ckpt_root: Path,
    epochs: list[int],
    ylim: tuple[float, float],
    out_path: Path,
    feature_extractor: str = "inception",
    methods: list[str] | None = None,
) -> Path:
    """Draws one line per method (FID on Y, epoch on X), zoomed to `ylim`,
    and saves it to out_path. feature_extractor selects which FID feature
    space to plot ("inception" or "mnist_cnn", see load_fid_by_epoch) --
    every method is plotted in the same feature space, since that's the
    only way the resulting FID numbers are comparable across the lines."""
    methods = methods or list(CKPT_DIRNAME)
    extractor_label = _EXTRACTOR_LABELS.get(feature_extractor, feature_extractor)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    off_chart: list[str] = []
    no_data: list[str] = []
    for method in methods:
        ckpt_dir = ckpt_root / CKPT_DIRNAME[method]
        fid_values = load_fid_by_epoch(ckpt_dir, epochs, feature_extractor)
        ax.plot(
            epochs,
            fid_values,  # type: ignore
            label=_LABELS.get(method, method),
            color=_COLORS.get(method, "#52514e"),
            marker=_MARKERS.get(method, "o"),
            markersize=7,
            linewidth=2,
        )
        seen = [v for v in fid_values if v is not None]
        if not seen:
            # No report at all for this (method, feature_extractor) pair --
            # a bare legend entry with an empty line looks like a bug, so
            # say why instead of leaving a silent mystery.
            no_data.append(_LABELS.get(method, method))
        elif all(v < ylim[0] or v > ylim[1] for v in seen):
            # Has data, but every value sits outside ylim -- same "legend
            # entry with no visible mark" problem, different cause.
            off_chart.append(f"{_LABELS.get(method, method)} (min {min(seen):.0f})")

    notes = []
    if no_data:
        notes.append(f"No {extractor_label} data: " + ", ".join(no_data))
    if off_chart:
        notes.append("Off-chart: " + ", ".join(off_chart))
    if notes:
        ax.text(
            0.98,
            0.02,
            "\n".join(notes),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#52514e",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"FID ({extractor_label})")
    ax.set_xticks(epochs)
    ax.set_ylim(*ylim)
    ax.set_title(
        f"FID vs. training epoch, {extractor_label} features "
        f"(zoomed to [{ylim[0]}, {ylim[1]}])"
    )
    ax.grid(True, color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", type=int, nargs="+", default=[50, 100, 150, 200, 250, 350]
    )
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        metavar=("YMIN", "YMAX"),
        help="Y-axis zoom range, e.g. --ylim 120 200. Required -- there's no "
        "sensible default across arbitrary methods/epochs, so callers must "
        "pick a range that fits the FID values they're plotting.",
        default=(50, 200),
    )
    parser.add_argument("--out", type=str, default="figs/fid_vs_epoch.png")
    parser.add_argument(
        "--feature-extractor",
        type=str,
        default="inception",
        choices=sorted(_EXTRACTOR_LABELS),
        help="Which FID feature space to plot (must match the "
        "--feature-extractor used when the metrics were computed -- see "
        "src/evaluate_metrics.py). Default: inception.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=None,
        choices=sorted(CKPT_DIRNAME),
        help="Default: all 5 methods.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_root = Path(os.environ["CKPT_ROOT"])
    out_path = plot_fid_vs_epoch(
        ckpt_root=ckpt_root,
        epochs=args.epochs,
        ylim=tuple(args.ylim),
        out_path=Path(args.out),
        feature_extractor=args.feature_extractor,
        methods=args.methods,
    )
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
