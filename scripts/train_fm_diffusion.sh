#!/usr/bin/env bash
# scripts/train_fm_diffusion.sh
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
mkdir -p logs
{
  echo "===== $(date -Is) : python -m src.train --method fm_diffusion "$@" ====="
  python -m src.train --method fm_diffusion "$@"
} 2>&1 | tee -a "logs/fm_diffusion.txt"