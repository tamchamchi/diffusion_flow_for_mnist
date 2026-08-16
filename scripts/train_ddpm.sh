#!/usr/bin/env bash
# scripts/train_ddpm.sh
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
python -m src.train --method ddpm "$@" 2>&1 | tee "logs/ddpm.txt"