"""Claim 1 probe: do future-frame losses reach the context KV-writing path?

Compares, on identical rollouts from the released framewise init:
- SGF two-pass: gradient of a future-frames-only loss w.r.t. the recorded
  context latents fed to pass 2 (the KV-write inputs) -> expected NONZERO for
  early (context) frames.
- Frozen-cache Self Forcing: same loss through grad-enabled exit-step forwards
  reading the frozen cache -> gradient w.r.t. the cache-write inputs is
  structurally ABSENT; k/v projection params still get gradients via the
  read path only.
"""
import torch

from scripts.repro.probe_common import (
    load_config, build_generator, warped_denoising_steps, encode_prompt, gpu_mem_report,
)
from scripts.repro.rollout_lib import serial_rollout, sgf_pass2

F = 8
SPLIT = 4          # loss covers frames SPLIT..F-1 only
EXIT_IDX = 2
PROMPT = "A chef in a white apron kneads dough on a wooden table, warm kitchen light."


def kv_write_grad_norm(gen):
    tot = 0.0
    for blk in gen.model.blocks:
        for lin in (blk.self_attn.k, blk.self_attn.v):
            if lin.weight.grad is not None:
                tot += lin.weight.grad.float().norm().item() ** 2
    return tot ** 0.5


def main():
    torch.manual_seed(0)
    device = torch.device("cuda")
    cfg = load_config()
    gen = build_generator(cfg, device, ckpt_path="checkpoints/init/framewise/ar_diffusion.pt")
    gen.enable_gradient_checkpointing()
    cond = encode_prompt(PROMPT, device)
    scheduler = gen.get_scheduler()
    steps = warped_denoising_steps(cfg, scheduler, device)
    noise = torch.randn(1, F, 16, 60, 104, device=device, dtype=torch.bfloat16)

    # ---------------- SGF two-pass ----------------
    torch.cuda.reset_peak_memory_stats()
    rec = serial_rollout(gen, scheduler, steps, noise, cond, EXIT_IDX, mode="frozen")
    for name in ("x_hat", "x_ctx_hat", "noisy_at_t"):
        assert not rec[name].requires_grad, f"pass-1 record {name} must be stop-gradient"
    print("[claim1] pass-1 records are stop-gradient (x_hat/x_ctx_hat/noisy_at_t): OK", flush=True)
    # pass-1 leaves got no grads even though they require grad (rollout is no-grad)
    assert all(l.grad is None for l in rec["ctx_leaves"])

    gen.model.requires_grad_(True)
    gen.model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        out, ctx_leaf = sgf_pass2(gen, rec["noisy_at_t"], rec["x_ctx_hat"], cond,
                                  steps[EXIT_IDX], ctx_requires_grad=True)
        loss = out[:, SPLIT:].float().pow(2).mean()
        loss.backward()
    ctx_grads = [ctx_leaf.grad[:, i].float().norm().item() for i in range(F)]
    kv_sgf = kv_write_grad_norm(gen)
    print(f"[claim1][SGF] per-frame d(loss_future)/d(context_latent) norms: "
          f"{[f'{g:.3e}' for g in ctx_grads]}", flush=True)
    print(f"[claim1][SGF] context frames (<{SPLIT}) grad norm sum: {sum(ctx_grads[:SPLIT]):.3e}  "
          f"kv-proj param grad norm: {kv_sgf:.3e}", flush=True)
    gpu_mem_report("SGF two-pass probe")
    assert all(g > 0 for g in ctx_grads[:SPLIT]), "SGF: future loss must reach context latents"

    # ---------------- frozen-cache Self Forcing ----------------
    gen.model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    rec_sf = serial_rollout(gen, scheduler, steps, noise, cond, EXIT_IDX, mode="self_forcing")
    sf_out = torch.cat(rec_sf["sf_outputs"], dim=1)
    loss_sf = sf_out[:, SPLIT:].float().pow(2).mean()
    loss_sf.backward()
    sf_ctx_grads = [l.grad for l in rec_sf["ctx_leaves"]]
    kv_sf = kv_write_grad_norm(gen)
    n_none = sum(g is None for g in sf_ctx_grads)
    print(f"[claim1][SF ] context-latent grads absent (None) for {n_none}/{F} frames; "
          f"kv-proj param grad norm via read path: {kv_sf:.3e}", flush=True)
    gpu_mem_report("Self Forcing probe")
    assert n_none == F, "frozen-cache SF: no autograd path to cache-write inputs expected"
    assert kv_sf > 0

    print("[claim1] PASS: SGF restores future-loss gradients to the context "
          "KV-writing path; frozen-cache Self Forcing has no such path.", flush=True)


if __name__ == "__main__":
    main()
