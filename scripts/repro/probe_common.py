"""Shared helpers for single-GPU reproduction probes (no FSDP)."""
import torch
from omegaconf import OmegaConf

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder


def load_config(path="configs/self_gradient_forcing_framewise.yaml",
                default="configs/default_config.yaml"):
    cfg = OmegaConf.merge(OmegaConf.load(default), OmegaConf.load(path))
    return cfg


def build_generator(cfg, device, ckpt_path=None, dtype=torch.bfloat16):
    gen = WanDiffusionWrapper(**cfg.model_kwargs, is_causal=True)
    if ckpt_path:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        for key in ("generator", "generator_ema", "model"):
            if key in sd:
                sd = sd[key]
                break
        sd = {k.replace("model._fsdp_wrapped_module.", "model.", 1) if
              k.startswith("model._fsdp_wrapped_module.") else k: v for k, v in sd.items()}
        gen.load_state_dict(sd, strict=True)
        print(f"[probe] loaded checkpoint {ckpt_path}")
    gen = gen.to(device=device, dtype=dtype)
    return gen


def warped_denoising_steps(cfg, scheduler, device):
    steps = torch.tensor(cfg.denoising_step_list, dtype=torch.long, device=device)
    if cfg.warp_denoising_step:
        timesteps = torch.cat((scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))).to(device)
        steps = timesteps[1000 - steps]
    return steps


def encode_prompt(prompt, device, dtype=torch.bfloat16):
    te = WanTextEncoder()
    te = te.to(device=device, dtype=dtype)
    with torch.no_grad():
        cond = te(text_prompts=[prompt])
    del te
    torch.cuda.empty_cache()
    return {k: v.to(dtype) for k, v in cond.items()}


def gpu_mem_report(tag):
    peak = torch.cuda.max_memory_allocated() / 1024**3
    cur = torch.cuda.memory_allocated() / 1024**3
    print(f"[mem] {tag}: peak_alloc={peak:.2f}GiB current={cur:.2f}GiB", flush=True)
    return peak
