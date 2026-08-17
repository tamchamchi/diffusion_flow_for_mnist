#!/usr/bin/env bash
# scripts/train_score_edm.sh
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
mkdir -p logs
{
  echo "===== $(date -Is) : python -m src.train --method score_edm "$@" ====="
  python -m src.train --method score_edm "$@"
} 2>&1 | tee -a "logs/score_edm.txt"
