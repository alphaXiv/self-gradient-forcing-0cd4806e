#!/usr/bin/env bash
set -euo pipefail
source scripts/repro/setup_env.sh
for m in .done_wan13b .done_wan14b .done_sgf_ckpts; do
  [ -f "$SGF_SHARED/$m" ] || { echo "ERROR: weights not ready ($m missing)"; exit 1; }
done
export NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 NUM_GPUS=8
bash scripts/train_self_gradient_forcing_framewise.sh configs/repro_sgf_framewise.yaml "$SGF_SHARED/outputs/train_sgf" --step-time-log-interval 5
