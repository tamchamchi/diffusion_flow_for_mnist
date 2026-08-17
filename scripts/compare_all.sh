#!/usr/bin/env bash
# scripts/compare_all.sh
#
# Runs scripts/evaluate.sh for all 5 methods with the same settings, then
# renders the cross-method comparison table (src/compare_methods.py).
#
# Any extra args appended after "$0" are forwarded to every evaluate.sh
# call and win over the defaults below (argparse takes the last occurrence
# of a repeated flag), e.g.:
#   bash scripts/compare_all.sh --device cuda:1 --num-fid-samples 5000
set -euo pipefail
cd "$(dirname "$0")/.."

for m in fm_ot fm_diffusion ddpm score score_continuous; do
  bash scripts/evaluate.sh "$m" \
    --device cuda:2 \
    --num-nll-samples 2000 \
    --num-fid-samples 2000 \
    --epochs 50 100 150 200 250 350 \
    "$@"
done

set -a; source .env; set +a
python -m src.compare_methods --out comparison.md
