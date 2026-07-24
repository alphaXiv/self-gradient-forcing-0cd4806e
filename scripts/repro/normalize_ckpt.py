"""Normalize any checkpoint format to {"generator": sd, "generator_ema": sd}
so inference.py can load it with or without --use_ema."""
import sys

import torch


def extract(sd):
    for key in ("generator_ema", "generator", "model"):
        if key in sd:
            sd = sd[key]
            break
    return {k.replace("model._fsdp_wrapped_module.", "model.", 1) if
            k.startswith("model._fsdp_wrapped_module.") else k: v for k, v in sd.items()}


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    sd = extract(torch.load(src, map_location="cpu", weights_only=False))
    torch.save({"generator": sd, "generator_ema": sd}, dst)
    print(f"[normalize] {src} -> {dst} ({len(sd)} tensors)")
