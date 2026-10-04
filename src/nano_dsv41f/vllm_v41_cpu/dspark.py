"""One-stage DSpark/Markov block proposal, matching the JAX reference geometry."""
from __future__ import annotations

import torch

from .mhc_engram import mhc_mixes, pre_mix, post_mix
from .model_ops import apply_moe
from .rope_ops import linear, partial_rope, rope_kwargs
from .sparse_attention import latent_attention


def record_context(model, cache, features, position):
    """Keep selected block-input features in a fixed-size SWA ring."""
    window = model.config.attention.local_window
    main = model._norm(linear(features, model._w("dspark.main_proj.weight")), "dspark.main_norm")
    kv = model._norm(linear(main, model._w("dspark.kv.weight")), "dspark.kv_norm")
    kv = partial_rope(kv, position, rotary_dim=model.config.attention.rope.rope_head_dim,
                      **rope_kwargs(model.config, 0))
    if cache.draft_kv is None:
        cache.draft_kv = kv.new_empty((1, window, kv.shape[-1]))
        cache.draft_positions = position.new_empty((1, window))
    slot = (cache.length - 1) % window
    cache.draft_kv[:, slot:slot+1].copy_(kv)
    cache.draft_positions[:, slot:slot+1].copy_(position)


@torch.inference_mode()
def draft_block(model, cache, *, return_base_logits=False):
    """Propose a full block with sequential greedy Markov corrections.

    The draft block attends bidirectionally within itself and to the target SWA
    window. Only the target verifier may commit these proposals to its KV cache.
    """
    dc, ac = model.config.dspark, model.config.attention
    if dc.n_layers != 1 or cache.draft_kv is None:
        raise ValueError("DSpark needs one stage and collected target features")
    if int(cache.position_buffer[0, cache.length-1]) != cache.length-1:
        raise ValueError("DSpark serving requires a single unpacked conversation")
    n = min(cache.length, ac.local_window)
    draft_ids = cache.input_ids.new_full((1, dc.block_size), dc.noise_token_id)
    draft_ids[:, 0] = cache.input_ids[:, -1]
    streams = model._w("embed")[draft_ids].unsqueeze(-2).expand(
        -1, -1, model.config.mhc_streams, -1).clone()
    incoming = torch.zeros(streams.shape[:-1], device=model.device)
    incoming[..., 0] = 1
    def mixes(x, name):
        p = f"dspark.{name}"
        return mhc_mixes(x, model._w(f"{p}.weight"), model._w(f"{p}.base"),
            model._w(f"{p}.scale"), sinkhorn_iters=model.config.mhc_sinkhorn_iters,
            eps=model.config.mhc_eps, norm_eps=model.config.norm_eps)
    pre, post, comb = mixes(streams, "mhc_attn")
    x = model._norm(pre_mix(streams, incoming), "dspark.attn_norm")
    qr = model._norm(linear(x, model._w("dspark.q_a.weight")), "dspark.q_norm")
    q = linear(qr, model._w("dspark.q_b.weight")).reshape(1, dc.block_size, ac.n_heads, ac.head_dim)
    pos = torch.arange(cache.length, cache.length+dc.block_size, device=model.device)[None]
    kwargs = dict(rotary_dim=ac.rope.rope_head_dim, **rope_kwargs(model.config, 0))
    q = partial_rope(q, pos, **kwargs)
    context_kv = cache.draft_kv[:, :n]
    draft_kv = model._norm(linear(x, model._w("dspark.kv.weight")), "dspark.kv_norm")
    draft_kv = partial_rope(draft_kv, pos, **kwargs)
    kv = torch.cat((context_kv, draft_kv), dim=1)
    mask = torch.ones((1, dc.block_size, kv.shape[1]), device=model.device, dtype=torch.bool)
    out, lse = latent_attention(q, kv, mask)
    if "nano.dspark.attn_sink" in model.weights:
        total = torch.logaddexp(lse, model._w("dspark.attn_sink").float()[None, None])
        out = out * torch.exp(lse-total).unsqueeze(-1)
    out = partial_rope(out, pos, inverse=True, **kwargs)
    grouped = out.reshape(1, dc.block_size, ac.o_groups, ac.n_heads//ac.o_groups*ac.head_dim)
    low = torch.einsum("...gd,gdr->...gr", grouped.to(model.dtype), model._w("dspark.wo_a"))
    out = linear(low.flatten(-2), model._w("dspark.wo_b.weight"))
    streams = post_mix(streams, out, comb, post)
    fpre, fpost, fcomb = mixes(streams, "mhc_ffn")
    fx = model._norm(pre_mix(streams, pre), "dspark.ffn_norm")
    fout = apply_moe(model, fx, prefix="dspark.moe", top_k=dc.experts_per_token)
    streams = post_mix(streams, fout, fcomb, fpost)
    collapsed = pre_mix(streams, fpre)
    base = linear(model._norm(collapsed, "dspark.norm"), model._w("lm_head"))
    prev = cache.input_ids[:, -1]
    proposals = []
    for i in range(dc.block_size):
        logits = base[:, i] + linear(model._w("dspark.markov_embed")[prev],
                                    model._w("dspark.markov_head.weight"))
        prev = logits.argmax(-1)
        proposals.append(prev)
    result = torch.stack(proposals, dim=1)
    return (result, base) if return_base_logits else result
