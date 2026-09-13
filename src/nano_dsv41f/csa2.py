from __future__ import annotations

from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp

from .compression import learned_group_compress
from .indexer_scorer import build_index_k, init_indexer
from .layers import init_linear, init_rms_norm, linear, rms_norm
from .quantization import fake_fp8_e4m3, fake_mxfp4_e2m1
from .rope import apply_partial_rope, rope_kwargs_for_layer

CSA2Mode = Literal["swa", "full", "reindex", "reuse"]


class SharedCSA2State(NamedTuple):
    """Cross-layer state published by the most recent compressed-KV source layer.

    `latent` is the normalized compressed representation *before* RoPE. The main cache
    `kv` is partial-RoPE'd (and optionally fake-quantized) from that latent. `index_k`
    is independently projected from `latent`, exactly because the indexer must not consume
    an already-rotated main-cache representation.
    """

    kv: jax.Array
    latent: jax.Array
    index_k: jax.Array | None
    segment_ids: jax.Array
    group_start_positions: jax.Array
    source_layer: jax.Array
    compress_ratio: jax.Array


def _segment_local_positions_1d(segment_ids: jax.Array) -> jax.Array:
    starts = jnp.concatenate([jnp.array([True]), segment_ids[1:] != segment_ids[:-1]])
    pos = jnp.arange(segment_ids.shape[0], dtype=jnp.int32)
    start_values = jnp.where(starts, pos, 0)
    start_pos = jax.lax.associative_scan(jnp.maximum, start_values)
    return pos - start_pos


def segment_local_positions(segment_ids: jax.Array) -> jax.Array:
    if segment_ids.ndim != 2:
        raise ValueError("segment_ids must be [batch, tokens]")
    return jax.vmap(_segment_local_positions_1d)(segment_ids)


def _init_grouped_wo_a(
    key: jax.Array,
    *,
    n_heads: int,
    head_dim: int,
    n_groups: int,
    o_rank: int,
) -> jax.Array:
    heads_per_group = n_heads // n_groups
    in_dim = heads_per_group * head_dim
    return jax.random.normal(
        key, (n_groups, in_dim, o_rank), dtype=jnp.float32
    ) * (in_dim**-0.5)


def init_csa2_attention(
    key: jax.Array,
    *,
    dim: int,
    n_heads: int,
    head_dim: int,
    q_rank: int,
    o_rank: int,
    o_groups: int,
    owns_global_kv: bool,
    is_index_source: bool,
    compression_ratio: int,
    attention_sink: bool,
    attention_sink_init: float,
    index_n_heads: int,
    index_head_dim: int,
) -> dict[str, object]:
    if n_heads % o_groups:
        raise ValueError("n_heads must be divisible by o_groups")
    keys = iter(jax.random.split(key, 14))
    params: dict[str, object] = {
        "q_a": init_linear(next(keys), dim, q_rank),
        "q_norm": init_rms_norm(q_rank),
        "q_b": init_linear(next(keys), q_rank, n_heads * head_dim),
        # SWA uses one latent KV vector shared by all Q heads, as in MLA/MQA.
        "local_kv": init_linear(next(keys), dim, head_dim),
        "local_kv_norm": init_rms_norm(head_dim),
        # V4.1's first output projection is independent per head group.
        "wo_a": _init_grouped_wo_a(
            next(keys),
            n_heads=n_heads,
            head_dim=head_dim,
            n_groups=o_groups,
            o_rank=o_rank,
        ),
        "wo_b": init_linear(next(keys), o_groups * o_rank, dim),
    }
    if attention_sink:
        params["attn_sink"] = jnp.full(
            (n_heads,), attention_sink_init, dtype=jnp.float32
        )
    if owns_global_kv:
        params["global_kv"] = init_linear(next(keys), dim, head_dim)
        if compression_ratio > 1:
            params["global_gate"] = init_linear(next(keys), dim, head_dim)
        params["global_norm"] = init_rms_norm(head_dim)
    if is_index_source:
        params["indexer"] = init_indexer(
            next(keys),
            q_rank=q_rank,
            model_dim=dim,
            main_head_dim=head_dim,
            n_heads=index_n_heads,
            head_dim=index_head_dim,
            owns_k=owns_global_kv,
        )
    return params


def _rope_kwargs(attn_cfg, compress_ratio: int) -> dict[str, float | int]:
    rc = attn_cfg.rope
    return rope_kwargs_for_layer(
        compress_ratio=compress_ratio,
        rope_theta=rc.rope_theta,
        compress_rope_theta=rc.compress_rope_theta,
        original_seq_len=rc.original_seq_len,
        factor=rc.rope_factor,
        beta_fast=rc.beta_fast,
        beta_slow=rc.beta_slow,
    )


def _latent_attention(
    q: jax.Array,
    kv: jax.Array,
    mask: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Dense correctness path for MQA-style latent attention."""
    scale = q.shape[-1] ** -0.5
    logits = jnp.einsum("bthd,bsd->bhts", q, kv) * scale
    valid = mask[:, None, :, :]
    any_valid = jnp.any(mask, axis=-1)
    masked = jnp.where(valid, logits, -1e30)
    probs = jax.nn.softmax(masked, axis=-1)
    probs = jnp.where(any_valid[:, None, :, None], probs, 0.0)
    out = jnp.einsum("bhts,bsd->bthd", probs, kv)
    lse = jax.nn.logsumexp(masked, axis=-1)
    lse = jnp.where(any_valid[:, None, :], lse, -jnp.inf)
    return out, jnp.swapaxes(lse, 1, 2)


def _build_global_state(
    source: jax.Array,
    segment_ids: jax.Array,
    params: dict[str, object],
    *,
    compression_ratio: int,
    source_layer: int,
    config,
) -> SharedCSA2State:
    latent = linear(source, params["global_kv"])
    if compression_ratio > 1:
        gate = linear(source, params["global_gate"])
        latent = learned_group_compress(latent, gate, ratio=compression_ratio)
    latent = rms_norm(latent, params["global_norm"], eps=config.norm_eps)

    local_pos = segment_local_positions(segment_ids)
    group_pos = local_pos[:, ::compression_ratio]
    group_segments = segment_ids[:, ::compression_ratio]
    rope_kwargs = _rope_kwargs(config.attention, compression_ratio)

    # Main compressed cache is rotated first, then quantized. Keep `latent` pre-RoPE for K-indexing.
    kv = apply_partial_rope(
        latent,
        group_pos,
        rotary_dim=config.attention.rope.rope_head_dim,
        **rope_kwargs,
    )
    if config.quantization.main_kv_fp4_qat:
        kv = fake_mxfp4_e2m1(
            kv,
            block_size=config.quantization.main_kv_block_size,
            scale_format=config.quantization.main_kv_scale_format,
            ste=True,
        )

    index_k = None
    if "indexer" in params:
        index_k = build_index_k(
            latent,
            group_pos,
            params["indexer"],
            rope_dim=config.attention.rope.rope_head_dim,
            rope_kwargs=rope_kwargs,
            norm_eps=config.norm_eps,
            fp4_qat=config.quantization.indexer_fp4_qat,
            fp4_block_size=config.quantization.indexer_block_size,
            fp4_scale_format=config.quantization.indexer_scale_format,
        )

    return SharedCSA2State(
        kv=kv,
        latent=latent,
        index_k=index_k,
        segment_ids=group_segments,
        group_start_positions=group_pos,
        source_layer=jnp.asarray(source_layer, dtype=jnp.int32),
        compress_ratio=jnp.asarray(compression_ratio, dtype=jnp.int32),
    )


def _local_mask(segment_ids: jax.Array, local_window: int) -> jax.Array:
    pos = segment_local_positions(segment_ids)
    same = segment_ids[:, :, None] == segment_ids[:, None, :]
    causal = pos[:, None, :] <= pos[:, :, None]
    recent = pos[:, None, :] >= pos[:, :, None] - (local_window - 1)
    return same & causal & recent


def _global_mask(
    segment_ids: jax.Array,
    state: SharedCSA2State,
    local_window: int,
) -> jax.Array:
    """Compressed history outside SWA, retaining the fixed-128 boundary overlap."""
    q_pos = segment_local_positions(segment_ids)
    same = segment_ids[:, :, None] == state.segment_ids[:, None, :]
    old_enough = state.group_start_positions[:, None, :] <= q_pos[:, :, None] - local_window
    return same & old_enough


def _merge_with_sink(
    local_out: jax.Array,
    local_lse: jax.Array,
    global_out: jax.Array | None,
    global_lse: jax.Array | None,
    sink: jax.Array | None,
) -> tuple[jax.Array, jax.Array]:
    if global_out is None:
        branch_lse = local_lse
        merged = local_out
    else:
        assert global_lse is not None
        branch_lse = jnp.logaddexp(local_lse, global_lse)
        lw = jnp.exp(local_lse - branch_lse)[..., None]
        gw = jnp.exp(global_lse - branch_lse)[..., None]
        merged = lw * local_out + gw * global_out

    if sink is None:
        return merged, branch_lse
    total_lse = jnp.logaddexp(branch_lse, sink[None, None, :])
    # Sink contributes denominator mass only; it has no value vector.
    return merged * jnp.exp(branch_lse - total_lse)[..., None], total_lse


def grouped_output_projection(
    o: jax.Array,
    params: dict[str, object],
    *,
    n_heads: int,
    n_groups: int,
    o_rank: int,
) -> jax.Array:
    heads_per_group = n_heads // n_groups
    grouped = o.reshape(*o.shape[:-2], n_groups, heads_per_group * o.shape[-1])
    low_rank = jnp.einsum("...gd,gdr->...gr", grouped, params["wo_a"])
    return linear(low_rank.reshape(*low_rank.shape[:-2], n_groups * o_rank), params["wo_b"])


def apply_csa2_attention(
    x: jax.Array,
    segment_ids: jax.Array,
    params: dict[str, object],
    state: SharedCSA2State | None,
    *,
    layer_id: int,
    mode: CSA2Mode,
    owns_global_kv: bool,
    compression_ratio: int,
    config,
    global_source: jax.Array | None = None,
) -> tuple[jax.Array, SharedCSA2State | None, dict[str, jax.Array | None]]:
    """Dense semantic reference for SWA/Full/Reindex/Reuse V4.1 attention.

    Every layer computes its own Q and SWA latent. Compressed layers additionally consume
    the most recently published main-KV state. A Full layer can publish from an explicit
    CED `global_source`, while still querying with its normal `x` input.
    """
    ac = config.attention
    q_pos = segment_local_positions(segment_ids)
    rope_kwargs = _rope_kwargs(ac, compression_ratio)

    qr = rms_norm(linear(x, params["q_a"]), params["q_norm"], eps=config.norm_eps)
    q = linear(qr, params["q_b"]).reshape(*x.shape[:-1], ac.n_heads, ac.head_dim)
    q = apply_partial_rope(
        q,
        q_pos,
        rotary_dim=ac.rope.rope_head_dim,
        **rope_kwargs,
    )

    local_kv = rms_norm(
        linear(x, params["local_kv"]), params["local_kv_norm"], eps=config.norm_eps
    )
    local_kv = apply_partial_rope(
        local_kv,
        q_pos,
        rotary_dim=ac.rope.rope_head_dim,
        **rope_kwargs,
    )
    if config.quantization.swa_fp8_qat:
        local_kv = fake_fp8_e4m3(
            local_kv,
            block_size=config.quantization.swa_fp8_block_size,
            ste=True,
        )
    local_out, local_lse = _latent_attention(q, local_kv, _local_mask(segment_ids, ac.local_window))

    if mode == "swa":
        if owns_global_kv:
            raise ValueError("SWA-only layer cannot own compressed global KV")
        if global_source is not None:
            raise ValueError("SWA-only layer cannot take global_source")
        global_out = global_lse = None
    else:
        if owns_global_kv:
            source = x if global_source is None else global_source
            if source.shape != x.shape:
                raise ValueError("global_source must match x [batch,tokens,dim]")
            state = _build_global_state(
                source,
                segment_ids,
                params,
                compression_ratio=compression_ratio,
                source_layer=layer_id,
                config=config,
            )
        elif global_source is not None:
            raise ValueError("global_source only applies to a compressed-KV source")
        if state is None:
            raise ValueError(f"CSA2 {mode} layer {layer_id} has no shared compressed state")
        global_out, global_lse = _latent_attention(
            q, state.kv, _global_mask(segment_ids, state, ac.local_window)
        )

    sink = params.get("attn_sink")
    merged, total_lse = _merge_with_sink(
        local_out, local_lse, global_out, global_lse, sink
    )

    # Because latent K is also V, the RoPE subspace survives the weighted value sum.
    # V4.1 explicitly rotates that tail back before the grouped low-rank output projection.
    merged = apply_partial_rope(
        merged,
        q_pos,
        rotary_dim=ac.rope.rope_head_dim,
        inverse=True,
        **rope_kwargs,
    )
    out = grouped_output_projection(
        merged,
        params,
        n_heads=ac.n_heads,
        n_groups=ac.o_groups,
        o_rank=ac.o_rank,
    )
    return out, state, {
        "qr": qr,
        "q": q,
        "index_hidden": x,
        "total_lse": total_lse,
        "global_lse": global_lse,
        "index_k": None if state is None else state.index_k,
    }
