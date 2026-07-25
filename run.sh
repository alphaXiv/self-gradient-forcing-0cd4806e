#!/usr/bin/env bash
set -euo pipefail
source scripts/repro/setup_env.sh

STEP=$(python - <<'PY'
import os, re
def steps(d):
    if not os.path.isdir(d):
        return set()
    return {int(m.group(1)) for f in os.listdir(d)
            if (m := re.match(r"checkpoint_model_(\d+)$", f))
            and os.path.exists(os.path.join(d, f, "model.pt"))}
s = os.environ["SGF_SHARED"]
common = steps(f"{s}/outputs/train_sgf") & steps(f"{s}/outputs/train_sf")
assert common, "no common checkpoint step"
print(max(common))
PY
)
echo "[eval-short] matched checkpoint step $STEP"
PAD_STEP=$(printf %06d "$STEP")

export NUM_OUTPUT_FRAMES=21 SEED=0

USE_EMA=0 OUTPUT_ROOT="$SGF_SHARED/outputs/eval/init_f21" \
  bash scripts/infer_self_gradient_forcing.sh framewise "$SGF_SHARED/checkpoints/init_normalized/model.pt"
USE_EMA=1 OUTPUT_ROOT="$SGF_SHARED/outputs/eval/sf_f21" \
  bash scripts/infer_self_gradient_forcing.sh framewise "$SGF_SHARED/outputs/train_sf/checkpoint_model_$PAD_STEP/model.pt"
USE_EMA=1 OUTPUT_ROOT="$SGF_SHARED/outputs/eval/sgf_f21" \
  bash scripts/infer_self_gradient_forcing.sh framewise "$SGF_SHARED/outputs/train_sgf/checkpoint_model_$PAD_STEP/model.pt"
USE_EMA=1 OUTPUT_ROOT="$SGF_SHARED/outputs/eval/released_f21" \
  bash scripts/infer_self_gradient_forcing.sh framewise "checkpoints/framewise/ar/model.pt"
echo "[eval-short] DONE"
find "$SGF_SHARED/outputs/eval" -maxdepth 1 -name "*_f21" | sort
