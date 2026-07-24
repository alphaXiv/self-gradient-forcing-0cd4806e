#!/usr/bin/env bash
set -euo pipefail
source scripts/repro/setup_env.sh
pip install -q open_clip_torch matplotlib scikit-image timm scipy
export TORCH_HOME="$SGF_SHARED/torch-cache"
for m in .done_wan13b .done_sgf_ckpts; do
  [ -f "$SGF_SHARED/$m" ] || { echo "ERROR: weights not ready ($m missing)"; exit 1; }
done

EVAL_ROOT="$SGF_SHARED/outputs/eval"
mkdir -p "$EVAL_ROOT"

declare -A CKPTS=(
  [init]="checkpoints/init/framewise/ar_diffusion.pt"
  [released]="checkpoints/framewise/ar/model.pt"
)
declare -A EMA=([init]=0 [released]=1)

for cond in init released; do
  out="$EVAL_ROOT/${cond}_f963"
  if compgen -G "$out/*.mp4" > /dev/null; then
    echo "[eval960] skip existing $out"
    continue
  fi
  echo "[eval960] generating cond=$cond frames=963 (~240s)"
  USE_EMA="${EMA[$cond]}" NUM_OUTPUT_FRAMES=963 OUTPUT_FOLDER="$out" SEED=0 \
    bash scripts/infer_self_gradient_forcing.sh framewise "${CKPTS[$cond]}" prompts/test_prompt.txt
done

python -m scripts.repro.eval_metrics --root "$EVAL_ROOT" --strips_dir "$EVAL_ROOT/strips"
echo "[eval960] DONE"
