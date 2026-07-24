#!/usr/bin/env bash
set -euo pipefail
source scripts/repro/setup_env.sh
bash scripts/repro/download_weights_pvc.sh
python -m scripts.repro.smoke_test
