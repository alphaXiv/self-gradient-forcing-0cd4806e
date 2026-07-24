#!/usr/bin/env bash
set -euo pipefail
source scripts/repro/setup_env.sh
pip install -q matplotlib open_clip_torch timm scipy
export TORCH_HOME="$SGF_SHARED/torch-cache"
python -m scripts.repro.compute_metrics
