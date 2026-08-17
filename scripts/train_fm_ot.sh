#!/usr/bin/env bash
# scripts/train_fm_ot.sh
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
mkdir -p logs
{
  echo "===== $(date -Is) : python -m src.train --method fm_ot "$@" ====="
  python -m src.train --method fm_ot "$@"
} 2>&1 | tee -a "logs/fm_ot.txt"