from __future__ import annotations

from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp

from .compression import learned_group_compress
from .layers import init_linear, init_rms_norm, linear, rms_norm

CSA2Mode = Literal["full", "reindex", "reuse"]


class SharedCSA2State(NamedTuple):
    """Cross-layer compressed KV published by a CSA2 source layer."""

    kv: jax.Array
    segment_ids: jax.Array
    group_start_positions: jax.Array
    source_layer: jax.Array


def _segment_local_positions_1d(segment_ids: jax.Array) -> jax.Array:
    starts = jnp.concatenate(
        [jnp.array([True]), segment_ids[1:] != segment_ids[:-1]]
    )
    pos = jnp.arange(segment_ids.shape[0], dtype=jnp.int32)
    start_values = jnp.where(starts, pos, 0)
    start_pos = jax.lax.associative_scan(jnp.maximum, start_values)
    return pos - start_pos


def segment_local_positions(segment_ids: jax.Array) -> jax.Array:
    if segment_ids.ndim != 2:
        raise ValueError("segment_ids must be [batch, tokens]")
    return jax.vmap(_segment_local_positions_1d)(segment_ids)


def init_csa2_attention(
    key: jax.Array,
    *,
    dim: int,
    n_heads: int,
    head_dim: int,
    q_rank: int,
    o_rank: int,
    owns_global_kv: bool,
) -> dict[str, object]:
    keys = iter(jax.random.split(key, 10))
    params: dict[str, object] = {
        # Small MLA-like query bottleneck.
        "q_a": init_linear(next(keys), dim, q_rank),
        "q_norm": init_rms_norm(q_rank),
        "q_b": init_linear(next(keys), q_rank, n_heads * head_dim),
        # One latent KV vector per raw token, shared by all query heads.
        "local_kv": init_linear(next(keys), dim, head_dim),
        "local_kv_norm": init_rms_norm(head_dim),
        # Low-rank output path; the released grouped wo_a layout is deferred.
        "out_a": init_linear(next(keys), n_heads * head_dim, o_rank),
        "out_b": init_linear(next(keys), o_rank, dim),
    }
    if owns_global_kv:
        params.update(
            {
                "global_kv": init_linear(next(keys), dim, head_dim),
                # V4.1 compression uses one learned pooling score per latent channel.
                "global_gate": init_linear(next(keys), dim, head_dim),
                "global_norm": init_rms_norm(head_dim),
            }
        )
    return params


def _latent_attention(
    q: jax.Array,
    kv: jax.Array,
    mask: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Dense correctness path for latent attention, including empty global rows."""
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
    eps: float,
) -> SharedCSA2State:
    latent = linear(source, params["global_kv"])
    gate = linear(source, params["global_gate"])
    latent = learned_group_compress(latent, gate, ratio=compression_ratio)
    latent = rms_norm(latent, params["global_norm"], eps=eps)
    local_pos = segment_local_positions(segment_ids)
    return SharedCSA2State(
        kv=latent,
        segment_ids=segment_ids[:, ::compression_ratio],
        group_start_positions=local_pos[:, ::compression_ratio],
        source_layer=jnp.asarray(source_layer, dtype=jnp.int32),
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
    """Older compressed history with the fixed-SWA boundary convention.

    A compressed group becomes part of the global branch when its *first* raw token is
    at least `local_window` behind the query. For r=2 this deliberately permits the one
    boundary token to have both a raw SWA representation and a compressed representation
    on alternating query positions, matching the fixed-128 decision made for this repo.
    """
    q_pos = segment_local_positions(segment_ids)
    same = segment_ids[:, :, None] == state.segment_ids[:, None, :]
    old_enough = (
        state.group_start_positions[:, None, :]
        <= q_pos[:, :, None] - local_window
    )
    return same & old_enough


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
    n_heads: int,
    head_dim: int,
    local_window: int,
    norm_eps: float,
) -> tuple[jax.Array, SharedCSA2State, dict[str, jax.Array]]:
    """Apply dense local+compressed attention while preserving CSA2 ownership semantics.

    `full` source layers publish a new compressed KV bank. `reuse` layers consume it.
    `reindex` layers also consume the same bank; their distinct indexer is intentionally
    trained/evaluated outside this dense backbone path until sparse-aware training exists.
    """
    qr = rms_norm(linear(x, params["q_a"]), params["q_norm"], eps=norm_eps)
    q = linear(qr, params["q_b"]).reshape(
        *x.shape[:-1], n_heads, head_dim
    )

    local_kv = rms_norm(
        linear(x, params["local_kv"]), params["local_kv_norm"], eps=norm_eps
    )
    local_out, local_lse = _latent_attention(
        q, local_kv, _local_mask(segment_ids, local_window)
    )

    if owns_global_kv:
        state = _build_global_state(
            x,
            segment_ids,
            params,
            compression_ratio=compression_ratio,
            source_layer=layer_id,
            eps=norm_eps,
        )
    if state is None:
        raise ValueError(
            f"CSA2 {mode} layer {layer_id} has no shared global KV source"
        )

    global_out, global_lse = _latent_attention(
        q, state.kv, _global_mask(segment_ids, state, local_window)
    )

    # Exact shared denominator without concatenating the two score matrices.
    total_lse = jnp.logaddexp(local_lse, global_lse)
    local_weight = jnp.exp(local_lse - total_lse)[..., None]
    global_weight = jnp.exp(global_lse - total_lse)[..., None]
    merged = local_weight * local_out + global_weight * global_out

    merged = merged.reshape(*x.shape[:-1], n_heads * head_dim)
    out = linear(linear(merged, params["out_a"]), params["out_b"])
    return out, state, {
        "q": q,
        "total_lse": total_lse,
        "global_lse": global_lse,
    }
