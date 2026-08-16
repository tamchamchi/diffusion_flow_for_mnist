#!/usr/bin/env bash
# scripts/train_ddpm.sh
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
mkdir -p logs
{
  echo "===== $(date -Is) : python -m src.train --method ddpm "$@" ====="
  python -m src.train --method ddpm "$@"
} 2>&1 | tee -a "logs/ddpm.txt"