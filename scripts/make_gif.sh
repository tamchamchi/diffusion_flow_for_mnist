#!/usr/bin/env bash
# scripts/make_gif.sh
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
python -m src.utils.make_gif "$@"
