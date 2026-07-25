"""Claim 2 probe: (A) pass-2 parallel reconstruction fidelity vs the pass-1
serial exit predictions (paper: ~1.41% relative L2, bf16 roundoff scale);
(B) peak-memory/wall-time scaling of SGF two-pass vs frozen-cache Self Forcing
vs the naive differentiable-rollout probe as the rollout length grows."""
import argparse
import json
import subprocess
import sys
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


def measure_one(frames, mode):
    device = torch.device("cuda")
    cfg = load_config()
    gen = build_generator(cfg, device, ckpt_path="checkpoints/init/framewise/ar_diffusion.pt")
    gen.enable_gradient_checkpointing()
    cond = encode_prompt(PROMPT, device)
    scheduler = gen.get_scheduler()
    steps = warped_denoising_steps(cfg, scheduler, device)
    torch.manual_seed(0)
    noise = torch.randn(1, frames, 16, 60, 104, device=device, dtype=torch.bfloat16)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        if mode == "sgf":
            rec = serial_rollout(gen, scheduler, steps, noise, cond, EXIT_IDX, mode="frozen")
            with torch.enable_grad():
                out, _ = sgf_pass2(gen, rec["noisy_at_t"], rec["x_ctx_hat"], cond, steps[EXIT_IDX])
                out.float().pow(2).mean().backward()
        elif mode.startswith("self_forcing"):
            rec = serial_rollout(gen, scheduler, steps, noise, cond, EXIT_IDX, mode=mode)
            torch.cat(rec["sf_outputs"], dim=1).float().pow(2).mean().backward()
        else:
            rec = serial_rollout(gen, scheduler, steps, noise, cond, EXIT_IDX, mode="differentiable")
            torch.cat(rec["diff_outputs"], dim=1).float().pow(2).mean().backward()
        row = {"frames": frames, "mode": mode,
               "peak_gib": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
               "wall_s": round(time.time() - t0, 1), "oom": False}
    except torch.OutOfMemoryError:
        row = {"frames": frames, "mode": mode,
               "peak_gib": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
               "wall_s": round(time.time() - t0, 1), "oom": True}
    except RuntimeError as e:
        row = {"frames": frames, "mode": mode, "peak_gib": None, "oom": False,
               "wall_s": round(time.time() - t0, 1), "error": str(e)[:200]}
    print("[claim2B-row] " + json.dumps(row), flush=True)


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

    # ---------------- B: memory scaling (one subprocess per point) ----------
    del gen
    torch.cuda.empty_cache()
    results = []
    for F in (6, 12, 21):
        for mode in ("sgf", "self_forcing", "self_forcing_clone", "differentiable"):
            out = subprocess.run(
                [sys.executable, "-m", "scripts.repro.probe_recon_memory",
                 "--frames", str(F), "--mode", mode],
                capture_output=True, text=True)
            row = None
            for line in out.stdout.splitlines():
                if line.startswith("[claim2B-row] "):
                    row = json.loads(line[len("[claim2B-row] "):])
            if row is None:
                tail = (out.stdout + out.stderr)[-400:]
                row = {"frames": F, "mode": mode, "peak_gib": None, "oom": False,
                       "error": "subprocess died", "tail": tail}
            results.append(row)
            print(f"[claim2B] {json.dumps(row)}", flush=True)

    print("[claim2B] RESULTS " + json.dumps(results), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--mode", default=None)
    args = ap.parse_args()
    if args.frames is not None:
        measure_one(args.frames, args.mode)
    else:
        main()
