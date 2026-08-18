#!/usr/bin/env bash
# scripts/evaluate.sh
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
python -m src.evaluate_metrics --method "$1" "${@:2}" 2>&1 | tee -a "logs/eval_$1.txt"