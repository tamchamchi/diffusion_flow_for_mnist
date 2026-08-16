#!/usr/bin/env bash
# scripts/train_score.sh
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
mkdir -p logs
python -m src.train --method score "$@" 2>&1 | tee "logs/score.txt"