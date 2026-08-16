#!/usr/bin/env bash
# scripts/train_score_continuous.sh
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
python -m src.train --method score_continuous --grad-clip 1.0 "$@" 2>&1 | tee "logs/score_continuous.txt"