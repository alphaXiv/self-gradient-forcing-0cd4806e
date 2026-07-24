#!/usr/bin/env bash
# Idempotent download of all weights to the shared PVC (done-markers skip work).
set -euo pipefail

S="${SGF_SHARED:-/shared}"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$S/wan_models" "$S/checkpoints" "$S/prompts"

if [ ! -f "$S/.done_wan13b" ]; then
    hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir "$S/wan_models/Wan2.1-T2V-1.3B"
    touch "$S/.done_wan13b"
fi

if [ ! -f "$S/.done_wan14b" ]; then
    hf download Wan-AI/Wan2.1-T2V-14B --local-dir "$S/wan_models/Wan2.1-T2V-14B"
    touch "$S/.done_wan14b"
fi

if [ ! -f "$S/.done_sgf_ckpts" ]; then
    hf download JunhaoZhuang/Self_Gradient_Forcing init/framewise/ar_diffusion.pt --local-dir "$S/checkpoints"
    hf download JunhaoZhuang/Self_Gradient_Forcing framewise/ar/model.pt --local-dir "$S/checkpoints"
    hf download JunhaoZhuang/Self_Gradient_Forcing vidprom_filtered_extended.txt --local-dir "$S/prompts"
    touch "$S/.done_sgf_ckpts"
fi

echo "[download] complete:"
du -sh "$S/wan_models"/* "$S/checkpoints"/* 2>/dev/null || true
