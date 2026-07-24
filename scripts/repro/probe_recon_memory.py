"""Claim 2 probe: (a) pass-2 parallel reconstruction fidelity vs the pass-1
sampled serial exit prediction (paper: ~1.41% relative L2); (b) peak-memory
scaling of the two-pass design vs differentiating through the rollout.

Variants per rollout length F:
  sgf_two_pass          — release behavior, backward through pass 2 only
  diff_rollout_true     — pass-1 loop with grad, cache writes tracked (the
                          "keep the cache differentiable" strawman; expected to
                          fail on in-place cache mutation or blow up in memory)
  diff_rollout_detached — grad through every rollout forward, cache writes
                          detached: a lower bound on full differentiation cost
"""
import gc
import json
import torch

from pipeline import SelfGradientForcingTrainingPipeline
from wan.modules.causal_model import CausalWanSelfAttention
from scripts.repro.probe_common import (
    load_config, build_generator, warped_denoising_steps, encode_prompt,
)


class Capture(torch.nn.Module):
    def __init__(self, gen):
        super().__init__()
        self.gen = gen
        self.clean_x = None
        self.noisy_at_t = None

    def forward(self, **kw):
        if kw.get("clean_x") is not None:
            self.clean_x = kw["clean_x"].detach()
            self.noisy_at_t = kw["noisy_image_or_video"].detach()
        return self.gen(**kw)


def make_pipe(gen_like, cfg, steps, scheduler, num_frames):
    return SelfGradientForcingTrainingPipeline(
        denoising_step_list=steps,
        scheduler=scheduler,
        generator=gen_like,
        num_frame_per_block=cfg.num_frame_per_block,
        num_max_frames=num_frames,
        context_noise=cfg.context_noise,
        per_rank_exit_step=cfg.per_rank_exit_step,
        self_gradient_forcing_match_context=cfg.self_gradient_forcing_match_context,
        self_gradient_forcing_cache_mode=cfg.self_gradient_forcing_cache_mode,
    )


def diff_rollout(gen, pipe, noise, cond, detach_cache_writes: bool):
    """Pass-1 rollout with gradients enabled end-to-end."""
    for m in gen.model.modules():
        if isinstance(m, CausalWanSelfAttention):
            m.kv_cache_write_detach = detach_cache_writes
    batch_size, num_frames = noise.shape[:2]
    device, dtype = noise.device, noise.dtype
    pipe._initialize_kv_cache(batch_size=batch_size, dtype=dtype, device=device)
    pipe._initialize_crossattn_cache(batch_size=batch_size, dtype=dtype, device=device)
    exits = []
    current_start_frame = 0
    for _ in range(num_frames // pipe.num_frame_per_block):
        cur = pipe.num_frame_per_block
        noisy_input = noise[:, current_start_frame:current_start_frame + cur]
        for index, current_timestep in enumerate(pipe.denoising_step_list):
            timestep = torch.ones([batch_size, cur], device=device, dtype=torch.int64) * current_timestep
            _, x0 = gen(
                noisy_image_or_video=noisy_input,
                conditional_dict=cond,
                timestep=timestep,
                kv_cache=pipe.kv_cache1,
                crossattn_cache=pipe.crossattn_cache,
                current_start=current_start_frame * pipe.frame_seq_length,
            )
            if index < len(pipe.denoising_step_list) - 1:
                next_t = pipe.denoising_step_list[index + 1]
                noisy_input = pipe.scheduler.add_noise(
                    x0.flatten(0, 1), torch.randn_like(x0.flatten(0, 1)),
                    next_t * torch.ones([batch_size * cur], device=device, dtype=torch.long),
                ).unflatten(0, x0.shape[:2])
        exits.append(x0)
        ctx_t = torch.zeros([batch_size, cur], device=device, dtype=torch.int64)
        gen(
            noisy_image_or_video=x0,
            conditional_dict=cond,
            timestep=ctx_t,
            kv_cache=pipe.kv_cache1,
            crossattn_cache=pipe.crossattn_cache,
            current_start=current_start_frame * pipe.frame_seq_length,
        )
        current_start_frame += cur
    for m in gen.model.modules():
        if isinstance(m, CausalWanSelfAttention):
            m.kv_cache_write_detach = False
    return torch.cat(exits, dim=1)


def measure(fn, tag):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        fn()
        peak = torch.cuda.max_memory_allocated() / 1024**3
        return {"status": "ok", "peak_gib": round(peak, 2)}
    except torch.cuda.OutOfMemoryError:
        return {"status": "OOM", "peak_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2)}
    except RuntimeError as e:
        return {"status": "runtime_error", "error": str(e)[:300],
                "peak_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2)}
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def main():
    device = torch.device("cuda")
    cfg = load_config()
    gen = build_generator(cfg, device, ckpt_path="checkpoints/init/framewise/ar_diffusion.pt")
    gen.enable_gradient_checkpointing()
    cond = encode_prompt("A sailboat crosses a calm bay at sunset, gentle waves.", device)
    scheduler = gen.get_scheduler()
    steps = warped_denoising_steps(cfg, scheduler, device)

    # ---- (a) reconstruction fidelity at F=21, several seeds -----------------
    rel_errs = []
    for seed in range(6):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cap = Capture(gen)
        pipe = make_pipe(cap, cfg, steps, scheduler, 21)
        noise = torch.randn(1, 21, 16, 60, 104, device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            out, _, _ = pipe.inference_with_trajectory(noise=noise, **cond)
        p1 = cap.clean_x.float()   # pass-1 exit predictions (context_noise=0)
        p2 = out.float()
        rel = (p2 - p1).norm() / p1.norm()
        per_frame = ((p2 - p1).flatten(2).norm(dim=2) / p1.flatten(2).norm(dim=2)).squeeze(0)
        rel_errs.append(rel.item())
        print(f"[probe2-recon] seed={seed} rel_l2={rel.item():.6f} "
              f"per_frame={[round(v, 4) for v in per_frame.tolist()]}", flush=True)
    mean_err = sum(rel_errs) / len(rel_errs)
    print(f"[probe2-recon-summary] mean_rel_l2={mean_err:.6f} "
          f"min={min(rel_errs):.6f} max={max(rel_errs):.6f} (paper: 0.0141)", flush=True)

    # ---- (b) memory scaling -------------------------------------------------
    results = {}
    for F in (3, 6, 9, 12, 21):
        row = {}
        torch.manual_seed(100 + F)
        noise = torch.randn(1, F, 16, 60, 104, device=device, dtype=torch.bfloat16)

        def sgf_run():
            pipe = make_pipe(gen, cfg, steps, scheduler, F)
            out, _, _ = pipe.inference_with_trajectory(noise=noise, **cond)
            out.float().pow(2).mean().backward()
            gen.model.zero_grad(set_to_none=True)
        row["sgf_two_pass"] = measure(sgf_run, f"sgf F={F}")

        def diff_true():
            pipe = make_pipe(gen, cfg, steps, scheduler, F)
            out = diff_rollout(gen, pipe, noise, cond, detach_cache_writes=False)
            out.float().pow(2).mean().backward()
            gen.model.zero_grad(set_to_none=True)
        row["diff_rollout_true"] = measure(diff_true, f"diff-true F={F}")

        def diff_detached():
            pipe = make_pipe(gen, cfg, steps, scheduler, F)
            out = diff_rollout(gen, pipe, noise, cond, detach_cache_writes=True)
            out.float().pow(2).mean().backward()
            gen.model.zero_grad(set_to_none=True)
        row["diff_rollout_detached"] = measure(diff_detached, f"diff-detached F={F}")

        results[F] = row
        print(f"[probe2-mem] F={F} " + json.dumps(row), flush=True)

    print("[probe2-mem-final] " + json.dumps(results), flush=True)
    assert mean_err < 0.05, "reconstruction error unexpectedly large"
    print("[probe2] DONE", flush=True)


if __name__ == "__main__":
    main()
