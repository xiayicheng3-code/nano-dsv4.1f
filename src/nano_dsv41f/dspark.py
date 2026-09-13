from __future__ import annotations

import jax
import jax.numpy as jnp

from .csa2 import grouped_output_projection
from .layers import init_embedding, init_linear, init_rms_norm, linear, rms_norm
from .mhc import init_mhc_generator, make_identity_pre_mix, mhc_mixes, post_mix, pre_mix
from .moe import apply_moe, init_moe
from .rope import apply_partial_rope, rope_kwargs_for_layer


def init_dspark(key: jax.Array, config) -> dict[str, object]:
    """Initialize a one-stage DSpark head with its own Transformer/MoE parameters.

    Token embeddings and LM head are intentionally *not* duplicated; callers pass the
    backbone embedding and prediction head, matching the released draft model's reuse of
    the target vocabulary spaces.
    """
    dc = config.dspark
    ac = config.attention
    keys = iter(jax.random.split(key, 18))
    params: dict[str, object] = {
        "main_proj": init_linear(
            next(keys), config.d_model * len(dc.target_layer_ids), config.d_model
        ),
        "main_norm": init_rms_norm(config.d_model),
        "attn_norm": init_rms_norm(config.d_model),
        "ffn_norm": init_rms_norm(config.d_model),
        "mhc_attn": init_mhc_generator(next(keys), config.mhc_streams, config.d_model),
        "mhc_ffn": init_mhc_generator(next(keys), config.mhc_streams, config.d_model),
        "q_a": init_linear(next(keys), config.d_model, ac.q_rank),
        "q_norm": init_rms_norm(ac.q_rank),
        "q_b": init_linear(next(keys), ac.q_rank, ac.n_heads * ac.head_dim),
        "kv": init_linear(next(keys), config.d_model, ac.head_dim),
        "kv_norm": init_rms_norm(ac.head_dim),
        "wo_a": jax.random.normal(
            next(keys),
            (
                ac.o_groups,
                (ac.n_heads // ac.o_groups) * ac.head_dim,
                ac.o_rank,
            ),
            dtype=jnp.float32,
        ) * (((ac.n_heads // ac.o_groups) * ac.head_dim) ** -0.5),
        "wo_b": init_linear(next(keys), ac.o_groups * ac.o_rank, config.d_model),
        "moe": init_moe(
            next(keys), config.d_model, config.d_ff, dc.n_routed_experts
        ),
        "norm": init_rms_norm(config.d_model),
        # Released vanilla Markov head: W1 token embedding then W2 vocabulary projection.
        "markov_embed": init_embedding(
            next(keys), config.vocab_size, dc.markov_rank, scale=0.02
        ),
        "markov_head": init_linear(next(keys), dc.markov_rank, config.vocab_size),
        "confidence": init_linear(
            next(keys), config.d_model + dc.markov_rank, 1
        ),
    }
    if ac.attention_sink:
        params["attn_sink"] = jnp.full(
            (ac.n_heads,), ac.attention_sink_init, dtype=jnp.float32
        )
    return params


def _dspark_mask(
    anchor_positions: jax.Array,
    block_keep_mask: jax.Array,
    *,
    seq_len: int,
    block_size: int,
) -> jax.Array:
    """DeepSpec mask: context before anchor OR any position in the same draft block."""
    bsz, n_blocks = anchor_positions.shape
    q_block = jnp.arange(n_blocks * block_size) // block_size
    context_pos = jnp.arange(seq_len)
    draft_pos = jnp.arange(n_blocks * block_size)

    # [B,Q,S]: context strictly precedes the corresponding anchor.
    anchor_for_q = anchor_positions[:, q_block]
    context_ok = context_pos[None, None, :] < anchor_for_q[:, :, None]

    # [Q,A*block]: all draft slots in the query's own block are mutually visible.
    draft_block = draft_pos // block_size
    same_draft = q_block[:, None] == draft_block[None, :]
    same_draft = jnp.broadcast_to(same_draft[None, :, :], (bsz,) + same_draft.shape)

    keep_q = block_keep_mask[:, q_block]
    return jnp.concatenate([context_ok, same_draft], axis=-1) & keep_q[:, :, None]


def _latent_attention(q: jax.Array, kv: jax.Array, mask: jax.Array) -> tuple[jax.Array, jax.Array]:
    scale = q.shape[-1] ** -0.5
    logits = jnp.einsum("bqhd,bkd->bhqk", q, kv) * scale
    valid = mask[:, None, :, :]
    any_valid = jnp.any(mask, axis=-1)
    masked = jnp.where(valid, logits, -1e30)
    prob = jax.nn.softmax(masked, axis=-1)
    prob = jnp.where(any_valid[:, None, :, None], prob, 0.0)
    out = jnp.einsum("bhqk,bkd->bqhd", prob, kv)
    lse = jax.nn.logsumexp(masked, axis=-1)
    lse = jnp.where(any_valid[:, None, :], lse, -jnp.inf)
    return out, jnp.swapaxes(lse, 1, 2)


def _draft_inputs(
    embed: jax.Array,
    input_ids: jax.Array,
    anchor_positions: jax.Array,
    block_keep_mask: jax.Array,
    *,
    noise_token_id: int,
    block_size: int,
) -> tuple[jax.Array, jax.Array]:
    """Released noise-input construction: anchor token then mask/noise tokens."""
    bsz, n_blocks = anchor_positions.shape
    ids = jnp.full(
        (bsz, n_blocks, block_size), noise_token_id, dtype=input_ids.dtype
    )
    anchor_token = jnp.take_along_axis(input_ids, anchor_positions, axis=1)
    ids = ids.at[:, :, 0].set(jnp.where(block_keep_mask, anchor_token, noise_token_id))
    return embed[ids].reshape(bsz, n_blocks * block_size, -1), ids


def apply_markov_teacher_forced(
    base_logits: jax.Array,
    prev_token_ids: jax.Array,
    params: dict[str, object],
) -> tuple[jax.Array, jax.Array]:
    """Add the released vanilla Markov low-rank vocabulary bias."""
    markov = params["markov_embed"][prev_token_ids]
    return base_logits + linear(markov, params["markov_head"]), markov


def apply_dspark(
    params: dict[str, object],
    config,
    *,
    embed: jax.Array,
    lm_head: jax.Array,
    input_ids: jax.Array,
    context_features: jax.Array,
    anchor_positions: jax.Array,
    block_keep_mask: jax.Array | None = None,
    teacher_prev_ids: jax.Array | None = None,
) -> dict[str, jax.Array]:
    """One-stage DSpark training/reference forward.

    Args:
      context_features: [B,T,len(target_layer_ids)*D] concatenated target-layer states.
      anchor_positions: [B,A] fixed-shape sampled anchors.
      teacher_prev_ids: optional [B,A,block] previous-token ids for teacher-forced Markov.

    The attention pattern exactly follows DeepSpec's released DSpark mask: draft queries
    can see target context strictly before their anchor and all draft tokens from their own
    proposal block, enabling block-parallel Transformer computation. The Markov correction
    remains sequential at sampling time but can be teacher-forced here.
    """
    dc = config.dspark
    ac = config.attention
    if context_features.shape[-1] != config.d_model * len(dc.target_layer_ids):
        raise ValueError("context_features last dim does not match DSpark target layers")
    if block_keep_mask is None:
        block_keep_mask = jnp.ones_like(anchor_positions, dtype=bool)

    main_x = rms_norm(
        linear(context_features, params["main_proj"]),
        params["main_norm"],
        eps=config.norm_eps,
    )
    draft, draft_ids = _draft_inputs(
        embed,
        input_ids,
        anchor_positions,
        block_keep_mask,
        noise_token_id=dc.noise_token_id,
        block_size=dc.block_size,
    )
    streams = jnp.repeat(draft[..., None, :], config.mhc_streams, axis=-2)
    incoming_pre = make_identity_pre_mix(streams)

    # One released-style DSpark Transformer layer.
    residual = streams
    attn_pre, attn_post, attn_comb = mhc_mixes(
        streams,
        params["mhc_attn"],
        sinkhorn_iters=config.mhc_sinkhorn_iters,
        eps=config.mhc_eps,
    )
    x = rms_norm(
        pre_mix(streams, incoming_pre), params["attn_norm"], eps=config.norm_eps
    )
    qr = rms_norm(linear(x, params["q_a"]), params["q_norm"], eps=config.norm_eps)
    q = linear(qr, params["q_b"]).reshape(
        *x.shape[:-1], ac.n_heads, ac.head_dim
    )

    bsz, seq_len = input_ids.shape
    n_blocks = anchor_positions.shape[1]
    offsets = jnp.arange(dc.block_size, dtype=jnp.int32)[None, None, :]
    draft_positions = (anchor_positions[..., None] + offsets).reshape(
        bsz, n_blocks * dc.block_size
    )
    # DSpark is an MTP/SWA-only stage, so use ordinary (non-compressed) RoPE.
    rope_kwargs = rope_kwargs_for_layer(
        compress_ratio=0,
        rope_theta=ac.rope.rope_theta,
        compress_rope_theta=ac.rope.compress_rope_theta,
        original_seq_len=ac.rope.original_seq_len,
        factor=ac.rope.rope_factor,
        beta_fast=ac.rope.beta_fast,
        beta_slow=ac.rope.beta_slow,
    )
    q = apply_partial_rope(
        q, draft_positions, rotary_dim=ac.rope.rope_head_dim, **rope_kwargs
    )

    context_pos = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
    context_pos = jnp.broadcast_to(context_pos, (bsz, seq_len))
    context_kv = rms_norm(linear(main_x, params["kv"]), params["kv_norm"], eps=config.norm_eps)
    context_kv = apply_partial_rope(
        context_kv, context_pos, rotary_dim=ac.rope.rope_head_dim, **rope_kwargs
    )
    draft_kv = rms_norm(linear(x, params["kv"]), params["kv_norm"], eps=config.norm_eps)
    draft_kv = apply_partial_rope(
        draft_kv, draft_positions, rotary_dim=ac.rope.rope_head_dim, **rope_kwargs
    )
    kv = jnp.concatenate([context_kv, draft_kv], axis=1)
    mask = _dspark_mask(
        anchor_positions,
        block_keep_mask,
        seq_len=seq_len,
        block_size=dc.block_size,
    )
    o, lse = _latent_attention(q, kv, mask)
    if "attn_sink" in params:
        sink = params["attn_sink"][None, None, :]
        total_lse = jnp.logaddexp(lse, sink)
        o = o * jnp.exp(lse - total_lse)[..., None]
        lse = total_lse
    o = apply_partial_rope(
        o,
        draft_positions,
        rotary_dim=ac.rope.rope_head_dim,
        inverse=True,
        **rope_kwargs,
    )
    attn_out = grouped_output_projection(
        o,
        params,
        n_heads=ac.n_heads,
        n_groups=ac.o_groups,
        o_rank=ac.o_rank,
    )
    streams = post_mix(residual, attn_out, attn_comb, attn_post)

    residual = streams
    ffn_pre, ffn_post, ffn_comb = mhc_mixes(
        streams,
        params["mhc_ffn"],
        sinkhorn_iters=config.mhc_sinkhorn_iters,
        eps=config.mhc_eps,
    )
    ffn_x = rms_norm(
        pre_mix(streams, attn_pre), params["ffn_norm"], eps=config.norm_eps
    )
    ffn_out, moe_aux = apply_moe(
        ffn_x,
        params["moe"],
        top_k=dc.experts_per_token,
        route_scale=config.route_scale,
        swiglu_limit=config.swiglu_limit,
        eps=config.route_eps,
    )
    streams = post_mix(residual, ffn_out, ffn_comb, ffn_post)
    hidden = rms_norm(
        pre_mix(streams, ffn_pre), params["norm"], eps=config.norm_eps
    )
    base_logits = jnp.einsum("bqd,dv->bqv", hidden, lm_head)

    base_logits = base_logits.reshape(bsz, n_blocks, dc.block_size, -1)
    hidden_blocks = hidden.reshape(bsz, n_blocks, dc.block_size, -1)
    if teacher_prev_ids is None:
        # Teacher-forced default: position 0 is conditioned on the anchor token; later
        # positions use the actual next source tokens when available, clamped at sequence end.
        label_pos = anchor_positions[..., None] + offsets
        prev_pos = jnp.maximum(label_pos, 0)
        safe = jnp.minimum(prev_pos, seq_len - 1)
        teacher_prev_ids = jnp.take_along_axis(
            input_ids[:, None, :], safe, axis=-1
        )
        teacher_prev_ids = teacher_prev_ids.at[:, :, 0].set(
            jnp.take_along_axis(input_ids, anchor_positions, axis=1)
        )
    corrected, markov = apply_markov_teacher_forced(
        base_logits, teacher_prev_ids, params
    )
    confidence = linear(
        jnp.concatenate([hidden_blocks, markov], axis=-1), params["confidence"]
    )[..., 0]
    if not dc.confidence_head:
        confidence = jnp.zeros(base_logits.shape[:-1], dtype=base_logits.dtype)

    return {
        "base_logits": base_logits,
        "draft_logits": corrected,
        "draft_hidden": hidden_blocks,
        "draft_input_ids": draft_ids.reshape(bsz, n_blocks, dc.block_size),
        "confidence": confidence,
        "attention_lse": lse.reshape(bsz, n_blocks, dc.block_size, ac.n_heads),
        "router_indices": moe_aux["router_indices"].reshape(
            bsz, n_blocks, dc.block_size, dc.experts_per_token
        ),
    }


def sample_markov_block(
    base_logits: jax.Array,
    first_prev_token_ids: jax.Array,
    params: dict[str, object],
    *,
    temperature: float = 0.0,
    key: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Sequential vanilla-Markov sampling over a block, matching released DeepSpec."""
    bsz, block_size, _ = base_logits.shape
    prev = first_prev_token_ids
    samples = []
    corrected = []
    keys = None if key is None else jax.random.split(key, block_size)
    for i in range(block_size):
        markov = params["markov_embed"][prev]
        logits = base_logits[:, i] + linear(markov, params["markov_head"])
        corrected.append(logits)
        if temperature <= 0.0:
            nxt = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        else:
            if keys is None:
                raise ValueError("sampling key is required when temperature > 0")
            nxt = jax.random.categorical(keys[i], logits / temperature, axis=-1).astype(jnp.int32)
        samples.append(nxt)
        prev = nxt
    return jnp.stack(samples, axis=1), jnp.stack(corrected, axis=1)
