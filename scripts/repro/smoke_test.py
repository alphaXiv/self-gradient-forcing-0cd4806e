"""Smoke test: env + weights + a 3-frame two-pass SGF rollout on one GPU."""
import torch

from pipeline import SelfGradientForcingTrainingPipeline
from scripts.repro.probe_common import (
    load_config, build_generator, warped_denoising_steps, encode_prompt, gpu_mem_report,
)


def main():
    device = torch.device("cuda")
    print("[smoke] device:", torch.cuda.get_device_name(0), flush=True)

    # flex_attention compile sanity on this arch
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    q = torch.randn(1, 2, 128, 64, device=device, dtype=torch.bfloat16)
    mask = create_block_mask(lambda b, h, qi, ki: qi >= ki, 1, 2, 128, 128, device=device)
    out = torch.compile(flex_attention)(q, q, q, block_mask=mask)
    print("[smoke] flex_attention compiled OK", out.shape, flush=True)

    cfg = load_config()
    gen = build_generator(cfg, device, ckpt_path="checkpoints/init/framewise/ar_diffusion.pt")
    cond = encode_prompt("A corgi runs across a sunny meadow, camera tracking.", device)
    scheduler = gen.get_scheduler()
    steps = warped_denoising_steps(cfg, scheduler, device)
    print("[smoke] warped denoising steps:", steps.tolist(), flush=True)

    pipe = SelfGradientForcingTrainingPipeline(
        denoising_step_list=steps,
        scheduler=scheduler,
        generator=gen,
        num_frame_per_block=cfg.num_frame_per_block,
        num_max_frames=3,
        context_noise=cfg.context_noise,
        per_rank_exit_step=cfg.per_rank_exit_step,
        self_gradient_forcing_match_context=cfg.self_gradient_forcing_match_context,
        self_gradient_forcing_cache_mode=cfg.self_gradient_forcing_cache_mode,
    )
    noise = torch.randn(1, 3, 16, 60, 104, device=device, dtype=torch.bfloat16)
    torch.cuda.reset_peak_memory_stats()
    out, t_from, t_to = pipe.inference_with_trajectory(noise=noise, **cond)
    loss = out.float().mean()
    loss.backward()
    grad_norm = sum(p.grad.norm().item() ** 2 for p in gen.parameters() if p.grad is not None) ** 0.5
    gpu_mem_report("3-frame two-pass + backward")
    print(f"[smoke] output {tuple(out.shape)} t_from={t_from} t_to={t_to} grad_norm={grad_norm:.4f}", flush=True)
    assert grad_norm > 0
    print("[smoke] PASS", flush=True)


if __name__ == "__main__":
    main()
