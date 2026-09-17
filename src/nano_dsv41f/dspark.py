from __future__ import annotations

import jax
import jax.numpy as jnp

from .csa2 import grouped_output_projection
from .layers import init_embedding, init_linear, init_rms_norm, linear, rms_norm
from .mhc import init_mhc_generator, make_identity_pre_mix, mhc_mixes, post_mix, pre_mix
from .moe import apply_moe, init_moe
from .quantization import fake_fp8_e4m3
from .rope import apply_partial_rope, rope_kwargs_for_layer


def init_dspark(key: jax.Array, config) -> dict[str, object]:
    """Initialize the one-stage V4.1-style DSpark draft block."""
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
            (ac.o_groups, (ac.n_heads // ac.o_groups) * ac.head_dim, ac.o_rank),
            dtype=jnp.float32,
        ) * (((ac.n_heads // ac.o_groups) * ac.head_dim) ** -0.5),
        "wo_b": init_linear(next(keys), ac.o_groups * ac.o_rank, config.d_model),
        # DSpark owns a distinct routed-expert population.
        "moe": init_moe(next(keys), config.d_model, config.d_ff, dc.n_routed_experts),
        "norm": init_rms_norm(config.d_model),
        # Released DSpark Markov head: token -> low rank -> vocab.
        "markov_embed": init_embedding(
            next(keys), config.vocab_size, dc.markov_rank, scale=0.02
        ),
        "markov_head": init_linear(next(keys), dc.markov_rank, config.vocab_size),
        # Released V4.1 confidence head always concatenates hidden + Markov embedding.
        "confidence": init_linear(next(keys), config.d_model + dc.markov_rank, 1),
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
    window_size: int,
) -> jax.Array:
    """Stateless equivalent of V4.1 DSpark's SWA-cache + whole-draft-block indices.

    For an anchor at raw position `a`, context keys cover `[a-window+1, a]`: the current
    target hidden is inserted into the SWA cache before the draft block runs. Every draft
    query additionally sees every slot in its own proposal block, intentionally non-causal
    inside that block.
    """
    bsz, n_blocks = anchor_positions.shape
    q_block = jnp.arange(n_blocks * block_size) // block_size
    context_pos = jnp.arange(seq_len)
    draft_flat = jnp.arange(n_blocks * block_size)
    anchor_for_q = anchor_positions[:, q_block]
    context_ok = (
        (context_pos[None, None, :] <= anchor_for_q[:, :, None])
        & (context_pos[None, None, :] >= anchor_for_q[:, :, None] - (window_size - 1))
    )
    draft_block = draft_flat // block_size
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
    """Build `[anchor token, noise, noise, ...]` exactly as released forward_embed."""
    bsz, n_blocks = anchor_positions.shape
    ids = jnp.full((bsz, n_blocks, block_size), noise_token_id, dtype=input_ids.dtype)
    anchor_token = jnp.take_along_axis(input_ids, anchor_positions, axis=1)
    ids = ids.at[:, :, 0].set(jnp.where(block_keep_mask, anchor_token, noise_token_id))
    return embed[ids].reshape(bsz, n_blocks * block_size, -1), ids


def apply_markov_teacher_forced(
    base_logits: jax.Array,
    prev_token_ids: jax.Array,
    params: dict[str, object],
) -> tuple[jax.Array, jax.Array]:
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
    """Stateless training/reference form of one released V4.1 DSpark stage.

    `anchor_positions` lets one ordinary JAX call emulate several decode anchors while
    retaining the released decode geometry: the target hidden at the anchor is already in
    the 128-token SWA context, while draft Q positions begin at `anchor + 1`; all draft
    positions within a proposal block are mutually visible.
    """
    dc = config.dspark
    ac = config.attention
    if dc.n_layers != 1:
        raise NotImplementedError("the nano DSpark reference currently implements one stage")
    if context_features.shape[-1] != config.d_model * len(dc.target_layer_ids):
        raise ValueError("context_features last dim does not match DSpark target layers")
    if block_keep_mask is None:
        block_keep_mask = jnp.ones_like(anchor_positions, dtype=bool)

    main_x = rms_norm(
        linear(context_features, params["main_proj"]), params["main_norm"], eps=config.norm_eps
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

    residual = streams
    attn_pre, attn_post, attn_comb = mhc_mixes(
        streams,
        params["mhc_attn"],
        sinkhorn_iters=config.mhc_sinkhorn_iters,
        eps=config.mhc_eps,
        norm_eps=config.norm_eps,
    )
    x = rms_norm(pre_mix(streams, incoming_pre), params["attn_norm"], eps=config.norm_eps)
    qr = rms_norm(linear(x, params["q_a"]), params["q_norm"], eps=config.norm_eps)
    q = linear(qr, params["q_b"]).reshape(*x.shape[:-1], ac.n_heads, ac.head_dim)

    bsz, seq_len = input_ids.shape
    n_blocks = anchor_positions.shape[1]
    offsets = jnp.arange(dc.block_size, dtype=jnp.int32)[None, None, :]
    # Current target token sits at `anchor`; drafts predict positions anchor+1 ... +block.
    draft_positions = (anchor_positions[..., None] + 1 + offsets).reshape(
        bsz, n_blocks * dc.block_size
    )
    rope_kwargs = rope_kwargs_for_layer(
        compress_ratio=0,
        rope_theta=ac.rope.rope_theta,
        compress_rope_theta=ac.rope.compress_rope_theta,
        original_seq_len=ac.rope.original_seq_len,
        factor=ac.rope.rope_factor,
        beta_fast=ac.rope.beta_fast,
        beta_slow=ac.rope.beta_slow,
    )
    q = apply_partial_rope(q, draft_positions, rotary_dim=ac.rope.rope_head_dim, **rope_kwargs)

    context_pos = jnp.broadcast_to(
        jnp.arange(seq_len, dtype=jnp.int32)[None, :], (bsz, seq_len)
    )
    context_kv = rms_norm(
        linear(main_x, params["kv"]), params["kv_norm"], eps=config.norm_eps
    )
    context_kv = apply_partial_rope(
        context_kv, context_pos, rotary_dim=ac.rope.rope_head_dim, **rope_kwargs
    )
    draft_kv = rms_norm(linear(x, params["kv"]), params["kv_norm"], eps=config.norm_eps)
    draft_kv = apply_partial_rope(
        draft_kv, draft_positions, rotary_dim=ac.rope.rope_head_dim, **rope_kwargs
    )
    if config.quantization.swa_fp8_qat:
        context_kv = fake_fp8_e4m3(
            context_kv,
            block_size=config.quantization.swa_fp8_block_size,
            ste=True,
        )
        draft_kv = fake_fp8_e4m3(
            draft_kv,
            block_size=config.quantization.swa_fp8_block_size,
            ste=True,
        )
    kv = jnp.concatenate([context_kv, draft_kv], axis=1)
    mask = _dspark_mask(
        anchor_positions,
        block_keep_mask,
        seq_len=seq_len,
        block_size=dc.block_size,
        window_size=ac.local_window,
    )
    o, lse = _latent_attention(q, kv, mask)
    if "attn_sink" in params:
        total_lse = jnp.logaddexp(lse, params["attn_sink"][None, None, :])
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
        o, params, n_heads=ac.n_heads, n_groups=ac.o_groups, o_rank=ac.o_rank
    )
    streams = post_mix(residual, attn_out, attn_comb, attn_post)

    residual = streams
    ffn_pre, ffn_post, ffn_comb = mhc_mixes(
        streams,
        params["mhc_ffn"],
        sinkhorn_iters=config.mhc_sinkhorn_iters,
        eps=config.mhc_eps,
        norm_eps=config.norm_eps,
    )
    ffn_x = rms_norm(pre_mix(streams, attn_pre), params["ffn_norm"], eps=config.norm_eps)
    ffn_out, moe_aux = apply_moe(
        ffn_x,
        params["moe"],
        top_k=dc.experts_per_token,
        route_scale=config.route_scale,
        swiglu_limit=config.swiglu_limit,
        eps=config.route_eps,
    )
    streams = post_mix(residual, ffn_out, ffn_comb, ffn_post)

    # Released forward_head collapses mHC first, normalizes only for vocabulary logits,
    # and feeds the *unnormalized collapsed hidden* to the confidence head.
    collapsed = pre_mix(streams, ffn_pre)
    logits_hidden = rms_norm(collapsed, params["norm"], eps=config.norm_eps)
    base_logits = jnp.einsum("bqd,dv->bqv", logits_hidden, lm_head)
    base_logits = base_logits.reshape(bsz, n_blocks, dc.block_size, -1)
    collapsed_blocks = collapsed.reshape(bsz, n_blocks, dc.block_size, -1)

    if teacher_prev_ids is None:
        # Markov step i conditions on token at anchor+i; step 0 therefore uses the anchor.
        prev_pos = anchor_positions[..., None] + offsets
        safe = jnp.minimum(prev_pos, seq_len - 1)
        source = jnp.broadcast_to(input_ids[:, None, :], (bsz, n_blocks, seq_len))
        teacher_prev_ids = jnp.take_along_axis(source, safe, axis=-1)
    corrected, markov = apply_markov_teacher_forced(base_logits, teacher_prev_ids, params)
    confidence = linear(
        jnp.concatenate([collapsed_blocks, markov], axis=-1), params["confidence"]
    )[..., 0]
    if not dc.confidence_head:
        confidence = jnp.zeros(base_logits.shape[:-1], dtype=jnp.float32)

    return {
        "base_logits": base_logits,
        "draft_logits": corrected,
        "draft_hidden": collapsed_blocks,
        "draft_input_ids": draft_ids.reshape(bsz, n_blocks, dc.block_size),
        "confidence": confidence.astype(jnp.float32),
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
    """Sequential released-style Markov correction/sampling over one proposal block."""
    _, block_size, _ = base_logits.shape
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
            nxt = jax.random.categorical(keys[i], logits / max(temperature, 1e-5), axis=-1).astype(jnp.int32)
        samples.append(nxt)
        prev = nxt
    return jnp.stack(samples, axis=1), jnp.stack(corrected, axis=1)
