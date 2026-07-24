"""Serial rollout mirroring SelfGradientForcingTrainingPipeline pass 1, with
three gradient modes for the reproduction probes:

- "frozen":         pure no-grad rollout (SGF pass 1 / cache construction)
- "self_forcing":   like frozen, but the sampled exit-step forward of each block
                    runs with grad enabled reading the frozen KV cache
                    (frame-wise frozen-cache Self Forcing baseline)
- "differentiable": every forward grad-enabled with a graph-retaining list cache
                    (the naive differentiate-through-the-rollout probe; the
                    stock cache writes in-place into no-grad buffers, which
                    silently detaches, so this mode swaps in a list cache)
"""
import torch

from scripts.repro.diff_cache_patch import differentiable_kv_cache

NUM_BLOCKS_TRANSFORMER = 30
FRAME_SEQ_LEN = 1560


def init_kv_cache(batch_size, num_max_frames, dtype, device):
    return [{
        "k": torch.zeros([batch_size, num_max_frames * FRAME_SEQ_LEN, 12, 128], dtype=dtype, device=device),
        "v": torch.zeros([batch_size, num_max_frames * FRAME_SEQ_LEN, 12, 128], dtype=dtype, device=device),
        "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
        "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
    } for _ in range(NUM_BLOCKS_TRANSFORMER)]


def init_diff_kv_cache():
    return [{"k_list": [], "v_list": []} for _ in range(NUM_BLOCKS_TRANSFORMER)]


def snapshot_kv_cache(kv_cache):
    """Freeze the cache counters for a grad-enabled read: gradient-checkpoint
    recompute re-reads the cache at backward time, when the counters have
    advanced past this frame (the read window would leak future frames).
    K/V buffers are shared: slots <= current frame hold their final context
    writes already, and the recomputed forward rewrites its own slot before
    reading, so values match the original read."""
    return [{"k": c["k"], "v": c["v"],
             "global_end_index": c["global_end_index"].clone(),
             "local_end_index": c["local_end_index"].clone()} for c in kv_cache]


def init_crossattn_cache(batch_size, dtype, device):
    return [{
        "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
        "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
        "is_init": False,
    } for _ in range(NUM_BLOCKS_TRANSFORMER)]


def serial_rollout(gen, scheduler, denoising_step_list, noise, cond,
                   exit_idx, mode="frozen", context_noise=0):
    """Frame-wise rollout (num_frame_per_block=1). Returns a dict with
    recorded exit predictions, context latents, noisy inputs at the exit step,
    per-block grad-enabled outputs (self_forcing mode), and per-frame context
    leaves (differentiable mode uses grad-chained contexts instead)."""
    B, F, C, H, W = noise.shape
    device, dtype = noise.device, noise.dtype
    steps = denoising_step_list
    n_steps = len(steps)

    diff = mode == "differentiable"
    if diff:
        kv_cache = init_diff_kv_cache()
    else:
        kv_cache = init_kv_cache(B, F, dtype, device)
    ca_cache = init_crossattn_cache(B, dtype, device)

    x_hat = torch.zeros_like(noise)          # exit predictions (pass-1 record)
    x_ctx_hat = torch.zeros_like(noise)      # context latents written to cache
    noisy_at_t = torch.zeros_like(noise)     # noisy input at exit step
    sf_outputs = []                          # grad-enabled exit outputs (self_forcing)
    diff_outputs = []                        # grad-chained exit outputs (differentiable)
    ctx_leaves = []                          # leaf tensors fed to cache writes

    grad_ctx = torch.enable_grad if diff else torch.no_grad
    patch_ctx = differentiable_kv_cache() if diff else _null_ctx()

    # Checkpoint recompute re-reads the (mutated) list cache -> disable
    # checkpointing for the naive differentiable rollout.
    ckpt_state = getattr(gen.model, "gradient_checkpointing", False)
    if diff:
        gen.model.gradient_checkpointing = False

    with grad_ctx(), patch_ctx:
        for i in range(F):
            noisy_input = noise[:, i:i + 1]
            x0_exit = None
            for index, current_timestep in enumerate(steps):
                timestep = torch.ones([B, 1], device=device, dtype=torch.int64) * current_timestep
                if index == exit_idx:
                    noisy_at_t[:, i:i + 1] = noisy_input.detach()
                if mode == "self_forcing" and index == exit_idx:
                    with torch.enable_grad():
                        _, x0 = gen(noisy_image_or_video=noisy_input,
                                    conditional_dict=cond, timestep=timestep,
                                    kv_cache=snapshot_kv_cache(kv_cache),
                                    crossattn_cache=ca_cache,
                                    current_start=i * FRAME_SEQ_LEN)
                    sf_outputs.append(x0)
                    x0 = x0.detach()
                else:
                    _, x0 = gen(noisy_image_or_video=noisy_input,
                                conditional_dict=cond, timestep=timestep,
                                kv_cache=kv_cache, crossattn_cache=ca_cache,
                                current_start=i * FRAME_SEQ_LEN)
                if index == exit_idx:
                    x0_exit = x0
                if index < n_steps - 1:
                    next_t = steps[index + 1]
                    noisy_input = scheduler.add_noise(
                        x0.flatten(0, 1).detach() if not diff else x0.flatten(0, 1),
                        torch.randn_like(x0.flatten(0, 1)),
                        next_t * torch.ones([B], device=device, dtype=torch.long),
                    ).unflatten(0, (B, 1))

            x_hat[:, i:i + 1] = x0_exit.detach()
            if diff:
                diff_outputs.append(x0_exit)

            # cache update at the clean context timestep (t_ctx = context_noise)
            context_timestep = torch.ones([B, 1], device=device, dtype=torch.int64) * context_noise
            x_ctx = scheduler.add_noise(
                x0_exit.flatten(0, 1) if diff else x0_exit.flatten(0, 1).detach(),
                torch.randn_like(x0_exit.flatten(0, 1)),
                context_timestep.flatten(0, 1).long(),
            ).unflatten(0, (B, 1))
            if not diff:
                x_ctx = x_ctx.detach().requires_grad_(True)   # leaf probe: can grads reach the write?
            ctx_leaves.append(x_ctx)
            x_ctx_hat[:, i:i + 1] = x_ctx.detach()

            gen(noisy_image_or_video=x_ctx, conditional_dict=cond,
                timestep=context_timestep, kv_cache=kv_cache,
                crossattn_cache=ca_cache, current_start=i * FRAME_SEQ_LEN)

    gen.model.gradient_checkpointing = ckpt_state
    return {
        "x_hat": x_hat, "x_ctx_hat": x_ctx_hat, "noisy_at_t": noisy_at_t,
        "sf_outputs": sf_outputs, "diff_outputs": diff_outputs,
        "ctx_leaves": ctx_leaves,
    }


def sgf_pass2(gen, noisy_at_t, x_ctx_hat, cond, train_t, context_noise=0,
              ctx_requires_grad=False):
    """SGF pass 2: parallel teacher-forced reconstruction over all frames."""
    B, F = noisy_at_t.shape[:2]
    device = noisy_at_t.device
    tf_timestep = train_t * torch.ones([B, F], device=device, dtype=torch.int64)
    aug_t = torch.ones([B, F], device=device, dtype=torch.int64) * context_noise
    pass2_clean = x_ctx_hat.detach().clone()
    if ctx_requires_grad:
        pass2_clean.requires_grad_(True)
    gen.model.block_mask = None   # mask is cached per num_frames
    _, output = gen(noisy_image_or_video=noisy_at_t, conditional_dict=cond,
                    timestep=tf_timestep, clean_x=pass2_clean, aug_t=aug_t)
    return output, pass2_clean


class _null_ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
