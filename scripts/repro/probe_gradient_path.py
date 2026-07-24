"""Claim 1 probe: does a future-frame-only loss send gradients into the
context-KV writing path?

SGF pass 2 (context K/V grad-enabled) vs frozen-context control
(context_kv_stop_grad: reads supervised, writes not). The recorded pass-1
tensors must be stop-gradient in both.

Measured via a leaf tensor substituted for pass-2 clean_x: grad(loss_future,
clean_x_leaf) is the gradient reaching the context-encoding input.
"""
import json
import torch

from pipeline import SelfGradientForcingTrainingPipeline
from wan.modules.causal_model import CausalWanSelfAttention
from scripts.repro.probe_common import (
    load_config, build_generator, warped_denoising_steps, encode_prompt, gpu_mem_report,
)

NUM_FRAMES = 21
LOSS_FRAMES = {"future_last7": (14, 21), "mid_single": (10, 11), "first_frame": (0, 1)}


class Pass2Interceptor(torch.nn.Module):
    """Wraps the generator; swaps pass-2 clean_x for a grad leaf."""

    def __init__(self, gen):
        super().__init__()
        self.gen = gen
        self.clean_leaf = None
        self.pass1_all_stop_grad = True

    def forward(self, **kw):
        if kw.get("kv_cache") is not None:
            # pass-1 rollout call: everything must be grad-free
            if kw["noisy_image_or_video"].requires_grad:
                self.pass1_all_stop_grad = False
            return self.gen(**kw)
        if kw.get("clean_x") is not None:
            assert not kw["clean_x"].requires_grad, "recorded context latents must be stop-grad"
            assert not kw["noisy_image_or_video"].requires_grad, "recorded noisy exit states must be stop-grad"
            leaf = kw["clean_x"].detach().clone().requires_grad_(True)
            self.clean_leaf = leaf
            kw = {**kw, "clean_x": leaf}
        return self.gen(**kw)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("gen"), name)


def run_variant(gen, cfg, cond, device, stop_grad_control: bool, seed: int):
    for m in gen.model.modules():
        if isinstance(m, CausalWanSelfAttention):
            m.context_kv_stop_grad = stop_grad_control
    gen.model.block_mask = None  # rebuild TF mask per variant

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    wrapped = Pass2Interceptor(gen)
    scheduler = gen.get_scheduler()
    steps = warped_denoising_steps(cfg, scheduler, device)
    pipe = SelfGradientForcingTrainingPipeline(
        denoising_step_list=steps,
        scheduler=scheduler,
        generator=wrapped,
        num_frame_per_block=cfg.num_frame_per_block,
        num_max_frames=NUM_FRAMES,
        context_noise=cfg.context_noise,
        per_rank_exit_step=cfg.per_rank_exit_step,
        self_gradient_forcing_match_context=cfg.self_gradient_forcing_match_context,
        self_gradient_forcing_cache_mode=cfg.self_gradient_forcing_cache_mode,
    )
    noise = torch.randn(1, NUM_FRAMES, 16, 60, 104, device=device, dtype=torch.bfloat16)
    out, _, _ = pipe.inference_with_trajectory(noise=noise, **cond)
    assert wrapped.pass1_all_stop_grad, "pass-1 rollout saw grad-tracking inputs"

    res = {"variant": "frozen_context_control" if stop_grad_control else "sgf",
           "pass1_stop_grad_ok": wrapped.pass1_all_stop_grad}
    leaf = wrapped.clean_leaf

    for name, (a, b) in LOSS_FRAMES.items():
        loss = out[:, a:b].float().pow(2).mean()
        (gleaf,) = torch.autograd.grad(loss, leaf, retain_graph=True, allow_unused=True)
        if gleaf is None:
            res[f"ctx_grad/{name}"] = None
        else:
            per_frame = gleaf.float().flatten(2).norm(dim=2).squeeze(0)
            res[f"ctx_grad/{name}"] = [round(v, 8) for v in per_frame.tolist()]

    # parameter-level: future-only loss -> k/v projection weights
    gen.model.zero_grad(set_to_none=True)
    loss = out[:, 14:].float().pow(2).mean()
    loss.backward()
    kv_sq, total_sq = 0.0, 0.0
    for n, p in gen.model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.float().norm().item() ** 2
        total_sq += g
        if "self_attn.k." in n or "self_attn.v." in n:
            kv_sq += g
    res["param_grad_norm/kv_proj"] = kv_sq ** 0.5
    res["param_grad_norm/total"] = total_sq ** 0.5
    gen.model.zero_grad(set_to_none=True)
    return res


def main():
    device = torch.device("cuda")
    cfg = load_config()
    gen = build_generator(cfg, device, ckpt_path="checkpoints/init/framewise/ar_diffusion.pt")
    gen.enable_gradient_checkpointing()
    cond = encode_prompt("A corgi runs across a sunny meadow, camera tracking.", device)

    results = []
    for control in (False, True):
        torch.cuda.reset_peak_memory_stats()
        r = run_variant(gen, cfg, cond, device, stop_grad_control=control, seed=1234)
        r["peak_mem_gib"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
        results.append(r)
        print("[probe1-result] " + json.dumps(r), flush=True)
        gpu_mem_report(r["variant"])
        torch.cuda.empty_cache()

    sgf, ctl = results
    ctx = sgf["ctx_grad/future_last7"]
    assert ctx is not None and sum(ctx[:14]) > 0, "SGF: future loss must reach context frames"
    cg = ctl["ctx_grad/future_last7"]
    assert cg is None or sum(cg) == 0, "control: no gradient may reach context input"
    assert ctl["param_grad_norm/total"] > 0, "control: read-path param grads must exist"
    print("[probe1] PASS: SGF restores context-write gradients; frozen control has none; "
          "read path supervised in both.", flush=True)


if __name__ == "__main__":
    main()
