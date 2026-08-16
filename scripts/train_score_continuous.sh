#!/usr/bin/env bash
# scripts/train_score_continuous.sh
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
mkdir -p logs
{
  echo "===== $(date -Is) : python -m src.train --method score_continuous --grad-clip 1.0 "$@" ====="
  python -m src.train --method score_continuous --grad-clip 1.0 "$@"
} 2>&1 | tee -a "logs/score_continuous.txt"