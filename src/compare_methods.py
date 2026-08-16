"""Reads every trained method's metrics.json (src/evaluate_metrics.py) and
renders one Markdown table comparing NLL/BPD, FID, and avg. NFE.
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from src.train import CKPT_DIRNAME

load_dotenv()


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
