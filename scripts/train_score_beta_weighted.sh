#!/usr/bin/env bash
# scripts/train_score_beta_weighted.sh
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
mkdir -p logs
{
  echo "===== $(date -Is) : python -m src.train --method score_beta_weighted "$@" ====="
  python -m src.train --method score_beta_weighted "$@"
} 2>&1 | tee -a "logs/score_beta_weighted.txt"
