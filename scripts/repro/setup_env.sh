#!/usr/bin/env bash
# Shared environment bootstrap for reproduction runs (sourced by run.sh).
# Blackwell (sm_120) note: no flash-attn — wan/modules/attention.py falls back
# to torch SDPA; the causal model uses torch-native flex_attention.
set -euo pipefail

export SGF_SHARED="${SGF_SHARED:-/shared}"
export PIP_CACHE_DIR="$SGF_SHARED/pip-cache"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$PIP_CACHE_DIR" "$SGF_SHARED/outputs"

pip install -q \
    omegaconf easydict einops ftfy tqdm sentencepiece lmdb wandb \
    "diffusers==0.31.0" "transformers>=4.49.0" accelerate \
    imageio imageio-ffmpeg "av==13.1.0" opencv-python-headless \
    "huggingface_hub[cli]" hf_transfer

# Repo expects wan_models/ and checkpoints/ relative to the project root.
ln -sfn "$SGF_SHARED/wan_models" wan_models
ln -sfn "$SGF_SHARED/checkpoints" checkpoints
if [ -f "$SGF_SHARED/prompts/vidprom_filtered_extended.txt" ]; then
    ln -sf "$SGF_SHARED/prompts/vidprom_filtered_extended.txt" prompts/vidprom_filtered_extended.txt
fi

nvidia-smi --query-gpu=name,memory.total --format=csv || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'devices', torch.cuda.device_count())"
