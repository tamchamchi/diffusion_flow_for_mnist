#!/usr/bin/env bash
# scripts/train_fm_ot.sh
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
python -m src.train --method fm_ot "$@" 2>&1 | tee "logs/fm_ot.txt"