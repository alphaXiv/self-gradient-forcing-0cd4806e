#!/usr/bin/env bash
set -euo pipefail
source scripts/repro/setup_env.sh

# Highest checkpoint step present in BOTH training arms (matched update count).
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
assert common, "no common checkpoint step between train_sgf and train_sf"
print(max(common))
PY
)
echo "[eval-trained] using matched checkpoint step $STEP"

export NUM_OUTPUT_FRAMES=963 SEED=0 USE_EMA=1
OUTPUT_ROOT="$SGF_SHARED/outputs/eval/sgf_f963" \
  bash scripts/infer_self_gradient_forcing.sh framewise \
  "$SGF_SHARED/outputs/train_sgf/checkpoint_model_$(printf %06d "$STEP")/model.pt"
OUTPUT_ROOT="$SGF_SHARED/outputs/eval/sf_f963" \
  bash scripts/infer_self_gradient_forcing.sh framewise \
  "$SGF_SHARED/outputs/train_sf/checkpoint_model_$(printf %06d "$STEP")/model.pt"
echo "[eval-trained] DONE step=$STEP"
ls -la "$SGF_SHARED/outputs/eval/sgf_f963" "$SGF_SHARED/outputs/eval/sf_f963"
