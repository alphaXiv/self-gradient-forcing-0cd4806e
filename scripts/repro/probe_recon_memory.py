"""Claim 2 probe: (A) pass-2 parallel reconstruction fidelity vs the pass-1
serial exit predictions (paper: ~1.41% relative L2, bf16 roundoff scale);
(B) peak-memory/wall-time scaling of SGF two-pass vs frozen-cache Self Forcing
vs the naive differentiable-rollout probe as the rollout length grows."""
import json
import time

import torch

from scripts.repro.probe_common import (
    load_config, build_generator, warped_denoising_steps, encode_prompt,
)
from scripts.repro.rollout_lib import serial_rollout, sgf_pass2

PROMPT = "A red vintage car drives along a coastal road at sunset, waves crashing."
EXIT_IDX = 2


def rel_l2(a, b):
    a, b = a.float(), b.float()
    return (a - b).norm().item() / b.norm().item()


def main():
    device = torch.device("cuda")
    cfg = load_config()
    gen = build_generator(cfg, device, ckpt_path="checkpoints/init/framewise/ar_diffusion.pt")
    gen.enable_gradient_checkpointing()
    cond = encode_prompt(PROMPT, device)
    scheduler = gen.get_scheduler()
    steps = warped_denoising_steps(cfg, scheduler, device)

    # ---------------- A: reconstruction fidelity (21 frames, 3 seeds) ----------------
    errs = []
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        noise = torch.randn(1, 21, 16, 60, 104, device=device, dtype=torch.bfloat16)
        exit_idx = seed % len(steps)
        rec = serial_rollout(gen, scheduler, steps, noise, cond, exit_idx, mode="frozen")
        with torch.no_grad():
            out, _ = sgf_pass2(gen, rec["noisy_at_t"], rec["x_ctx_hat"], cond, steps[exit_idx])
        err = rel_l2(out, rec["x_hat"])
        per_frame = [rel_l2(out[:, i], rec["x_hat"][:, i]) for i in range(21)]
        errs.append(err)
        print(f"[claim2A] seed={seed} exit_idx={exit_idx} rel_l2={err * 100:.3f}%  "
              f"per-frame%: {[f'{e * 100:.2f}' for e in per_frame]}", flush=True)
    print(f"[claim2A] mean pass-2 reconstruction rel_l2 = {sum(errs) / len(errs) * 100:.3f}% "
          f"(paper reports ~1.41%)", flush=True)

    # ---------------- B: memory scaling ----------------
    results = []
    for F in (6, 12, 21):
        for mode in ("sgf", "self_forcing", "differentiable"):
            gen.model.zero_grad(set_to_none=True)
            gen.model.block_mask = None
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.manual_seed(0)
            noise = torch.randn(1, F, 16, 60, 104, device=device, dtype=torch.bfloat16)
            t0 = time.time()
            try:
                if mode == "sgf":
                    rec = serial_rollout(gen, scheduler, steps, noise, cond, EXIT_IDX, mode="frozen")
                    with torch.enable_grad():
                        out, _ = sgf_pass2(gen, rec["noisy_at_t"], rec["x_ctx_hat"], cond, steps[EXIT_IDX])
                        out.float().pow(2).mean().backward()
                elif mode == "self_forcing":
                    rec = serial_rollout(gen, scheduler, steps, noise, cond, EXIT_IDX, mode="self_forcing")
                    torch.cat(rec["sf_outputs"], dim=1).float().pow(2).mean().backward()
                else:
                    rec = serial_rollout(gen, scheduler, steps, noise, cond, EXIT_IDX, mode="differentiable")
                    torch.cat(rec["diff_outputs"], dim=1).float().pow(2).mean().backward()
                peak = torch.cuda.max_memory_allocated() / 1024 ** 3
                row = {"frames": F, "mode": mode, "peak_gib": round(peak, 2),
                       "wall_s": round(time.time() - t0, 1), "oom": False}
            except torch.OutOfMemoryError:
                row = {"frames": F, "mode": mode, "peak_gib": None,
                       "wall_s": round(time.time() - t0, 1), "oom": True}
                torch.cuda.empty_cache()
            results.append(row)
            print(f"[claim2B] {json.dumps(row)}", flush=True)

    print("[claim2B] RESULTS " + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
