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
    if cache.transaction is not None:
        cache.transaction.draft = (kv, position, window)
    cache.write_draft(kv, position, window, cache.length - kv.shape[1])


def draft_from_kv(model, context_kv, context_mask, anchor_ids, anchor_positions, *, return_router_indices=False):
    """Differentiable draft block; used by both training and cached inference."""
    dc, ac = model.config.dspark, model.config.attention
    if dc.n_layers != 1:
        raise ValueError("nano DSpark supports exactly one stage")
    batch = anchor_ids.shape[0]
    draft_ids = anchor_ids.new_full((batch, dc.block_size), dc.noise_token_id)
    draft_ids[:, 0] = anchor_ids
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
    q = linear(qr, model._w("dspark.q_b.weight")).reshape(batch, dc.block_size, ac.n_heads, ac.head_dim)
    pos = anchor_positions[:, None] + 1 + torch.arange(dc.block_size, device=model.device)[None]
    kwargs = dict(rotary_dim=ac.rope.rope_head_dim, **rope_kwargs(model.config, 0))
    q = partial_rope(q, pos, **kwargs)
    draft_kv = model._norm(linear(x, model._w("dspark.kv.weight")), "dspark.kv_norm")
    draft_kv = partial_rope(draft_kv, pos, **kwargs)
    kv = torch.cat((context_kv, draft_kv), dim=1)
    mask = torch.cat((context_mask, torch.ones((batch, dc.block_size), device=model.device, dtype=torch.bool)), dim=-1)
    mask = mask[:, None, :].expand(-1, dc.block_size, -1)
    out, lse = latent_attention(q, kv, mask)
    if "nano.dspark.attn_sink" in model.weights:
        total = torch.logaddexp(lse, model._w("dspark.attn_sink").float()[None, None])
        out = out * torch.exp(lse-total).unsqueeze(-1)
    out = partial_rope(out, pos, inverse=True, **kwargs)
    grouped = out.reshape(batch, dc.block_size, ac.o_groups, ac.n_heads//ac.o_groups*ac.head_dim)
    low = torch.einsum("...gd,gdr->...gr", grouped.to(model.dtype), model._w("dspark.wo_a"))
    out = linear(low.flatten(-2), model._w("dspark.wo_b.weight"))
    streams = post_mix(streams, out, comb, post)
    fpre, fpost, fcomb = mixes(streams, "mhc_ffn")
    fx = model._norm(pre_mix(streams, pre), "dspark.ffn_norm")
    fout = apply_moe(model, fx, prefix="dspark.moe", top_k=dc.experts_per_token,
                     return_router_indices=return_router_indices)
    if return_router_indices:
        fout, router_indices = fout
    streams = post_mix(streams, fout, fcomb, fpost)
    collapsed = pre_mix(streams, fpre)
    base = linear(model._norm(collapsed, "dspark.norm"), model._w("lm_head"))
    return (base, collapsed, router_indices) if return_router_indices else (base, collapsed)


def draft_from_features(model, features, context_positions, context_mask,
                        anchor_ids, anchor_positions, previous_ids, *, return_router_indices=False):
    """Teacher-forced training over independent SWA windows (one per anchor)."""
    main = model._norm(linear(features, model._w("dspark.main_proj.weight")), "dspark.main_norm")
    kv = model._norm(linear(main, model._w("dspark.kv.weight")), "dspark.kv_norm")
    kv = partial_rope(kv, context_positions,
        rotary_dim=model.config.attention.rope.rope_head_dim, **rope_kwargs(model.config, 0))
    result = draft_from_kv(model, kv, context_mask, anchor_ids, anchor_positions,
                           return_router_indices=return_router_indices)
    base, hidden = result[:2]
    markov = model._w("dspark.markov_embed")[previous_ids]
    logits = base + linear(markov, model._w("dspark.markov_head.weight"))
    confidence = linear(torch.cat((hidden, markov), dim=-1),
                        model._w("dspark.confidence.weight")).squeeze(-1)
    return (logits, confidence, result[2]) if return_router_indices else (logits, confidence)


@torch.inference_mode()
def draft_block(model, cache, *, return_base_logits=False):
    """Greedy Markov proposals; only target verification can commit cache tokens."""
    dc, ac = model.config.dspark, model.config.attention
    if cache.draft_kv is None:
        raise ValueError("DSpark needs collected target features")
    if int(cache.position_buffer[0, cache.length-1]) != cache.length-1:
        raise ValueError("DSpark serving requires a single unpacked conversation")
    n = min(cache.length, ac.local_window)
    base, _ = draft_from_kv(model, cache.draft_kv[:, :n],
        torch.ones((1, n), dtype=torch.bool, device=model.device),
        cache.input_ids[:, -1], cache.position_buffer[:, cache.length-1])
    prev = cache.input_ids[:, -1]
    proposals = []
    for i in range(dc.block_size):
        logits = base[:, i] + linear(model._w("dspark.markov_embed")[prev],
                                    model._w("dspark.markov_head.weight"))
        prev = logits.argmax(-1)
        proposals.append(prev)
    result = torch.stack(proposals, dim=1)
    return (result, base) if return_base_logits else result
