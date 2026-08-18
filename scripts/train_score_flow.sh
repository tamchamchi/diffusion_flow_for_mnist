#!/usr/bin/env bash
# scripts/train_score_flow.sh
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
mkdir -p logs
{
  echo "===== $(date -Is) : python -m src.train --method score_flow --grad-clip 1.0 "$@" ====="
  python -m src.train --method score_flow --grad-clip 1.0 "$@"
} 2>&1 | tee -a "logs/score_flow.txt"