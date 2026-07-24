#!/usr/bin/env bash
set -euo pipefail
source scripts/repro/setup_env.sh
for m in .done_wan13b .done_sgf_ckpts; do
  [ -f "$SGF_SHARED/$m" ] || { echo "ERROR: weights not ready ($m missing)"; exit 1; }
done

mkdir -p "$SGF_SHARED/checkpoints/init_normalized"
python scripts/repro/normalize_ckpt.py \
  checkpoints/init/framewise/ar_diffusion.pt \
  "$SGF_SHARED/checkpoints/init_normalized/model.pt"

export NUM_OUTPUT_FRAMES=241 SEED=0 USE_EMA=1
OUTPUT_ROOT="$SGF_SHARED/outputs/eval/init" \
  bash scripts/infer_self_gradient_forcing.sh framewise "$SGF_SHARED/checkpoints/init_normalized/model.pt"
OUTPUT_ROOT="$SGF_SHARED/outputs/eval/released" \
  bash scripts/infer_self_gradient_forcing.sh framewise checkpoints/framewise/ar/model.pt
echo "[eval-refs] DONE"; ls -la "$SGF_SHARED/outputs/eval/init" "$SGF_SHARED/outputs/eval/released"
