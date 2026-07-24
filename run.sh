#!/usr/bin/env bash
set -euo pipefail
source scripts/repro/setup_env.sh
pip install -q open_clip_torch matplotlib scikit-image
for m in .done_wan13b .done_sgf_ckpts; do
  [ -f "$SGF_SHARED/$m" ] || { echo "ERROR: weights not ready ($m missing)"; exit 1; }
done

EVAL_ROOT="$SGF_SHARED/outputs/eval"
mkdir -p "$EVAL_ROOT"

# Highest common checkpoint step across the two training runs -> matched budget.
common_step() {
  comm -12 \
    <(ls -1 "$SGF_SHARED/outputs/train_sgf" 2>/dev/null | grep checkpoint_model | sort) \
    <(ls -1 "$SGF_SHARED/outputs/train_sf" 2>/dev/null | grep checkpoint_model | sort) \
    | tail -1
}
STEP_DIR="$(common_step)"
[ -n "$STEP_DIR" ] || { echo "ERROR: no common checkpoint between train_sgf and train_sf"; exit 1; }
echo "[eval] matched checkpoint: $STEP_DIR"

declare -A CKPTS=(
  [init]="checkpoints/init/framewise/ar_diffusion.pt"
  [sf]="$SGF_SHARED/outputs/train_sf/$STEP_DIR/model.pt"
  [sgf]="$SGF_SHARED/outputs/train_sgf/$STEP_DIR/model.pt"
  [released]="checkpoints/framewise/ar/model.pt"
)
declare -A EMA=([init]=0 [sf]=1 [sgf]=1 [released]=1)
HORIZONS="${HORIZONS:-963}"

for cond in init sf sgf released; do
  for frames in $HORIZONS; do
    out="$EVAL_ROOT/${cond}_f${frames}"
    if compgen -G "$out/*.mp4" > /dev/null; then
      echo "[eval] skip existing $out"
      continue
    fi
    echo "[eval] generating cond=$cond frames=$frames"
    USE_EMA="${EMA[$cond]}" NUM_OUTPUT_FRAMES="$frames" OUTPUT_FOLDER="$out" SEED=0 \
      bash scripts/infer_self_gradient_forcing.sh framewise "${CKPTS[$cond]}" prompts/test_prompt.txt
  done
done

python -m scripts.repro.eval_metrics --root "$EVAL_ROOT" --strips_dir "$EVAL_ROOT/strips"
echo "[eval] DONE"
