"""Monkeypatch making the causal self-attention KV cache differentiable.

The stock cache writes roped K / V in-place into preallocated no-grad buffers
(wan/modules/causal_model.py), which detaches them from autograd. For the
naive differentiate-through-the-rollout probe we swap in a list cache whose
entries stay graph-connected; reads concatenate the lists. Valid for rollouts
up to sink+local_attn frames (<= 21 here), where the streaming window spans
the whole cache, so visibility matches the stock read path exactly.
"""
import math
from contextlib import contextmanager

import torch

from wan.modules import causal_model as cm


def _diff_forward(self, x, seq_lens, grid_sizes, freqs, block_mask,
                  kv_cache=None, current_start=0, cache_start=None):
    if kv_cache is None or "k_list" not in kv_cache:
        return _orig_forward(self, x, seq_lens, grid_sizes, freqs, block_mask,
                             kv_cache, current_start, cache_start)
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    q = self.norm_q(self.q(x)).view(b, s, n, d)
    k = self.norm_k(self.k(x)).view(b, s, n, d)
    v = self.v(x).view(b, s, n, d)

    frame_seqlen = math.prod(grid_sizes[0][1:]).item()
    start_frame = current_start // frame_seqlen
    roped_query = cm.causal_rope_apply(q, grid_sizes, freqs, start_frame=start_frame).type_as(v)
    roped_key = cm.causal_rope_apply(k, grid_sizes, freqs, start_frame=start_frame).type_as(v)

    # Overwrite same-frame entries (multiple denoise steps per frame re-write
    # the same slot in the stock cache; here the last write before advancing
    # is the context write, matching stock behavior for reads by later frames).
    kv_cache["k_list"] = kv_cache["k_list"][:start_frame]
    kv_cache["v_list"] = kv_cache["v_list"][:start_frame]
    k_read = torch.cat(kv_cache["k_list"] + [roped_key], dim=1)
    v_read = torch.cat(kv_cache["v_list"] + [v], dim=1)
    kv_cache["k_list"].append(roped_key)
    kv_cache["v_list"].append(v)

    out = cm.attention(roped_query, k_read, v_read)
    return self.o(out.flatten(2))


_orig_forward = cm.CausalWanSelfAttention.forward


@contextmanager
def differentiable_kv_cache():
    cm.CausalWanSelfAttention.forward = _diff_forward
    try:
        yield
    finally:
        cm.CausalWanSelfAttention.forward = _orig_forward
