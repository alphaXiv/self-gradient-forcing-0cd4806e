#!/usr/bin/env bash
set -euo pipefail
source scripts/repro/setup_env.sh
for m in .done_wan13b .done_sgf_ckpts; do
  [ -f "$SGF_SHARED/$m" ] || { echo "ERROR: weights not ready ($m missing)"; exit 1; }
done
python -m scripts.repro.probe_recon_memory
