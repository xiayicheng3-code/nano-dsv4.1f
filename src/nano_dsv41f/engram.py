from __future__ import annotations

import jax
import jax.numpy as jnp

from .layers import init_linear


def _shift_right(x: jax.Array, amount: int, fill: int) -> jax.Array:
    if amount == 0:
        return x
    pad = jnp.full(x.shape[:-1] + (amount,), fill, dtype=x.dtype)
    return jnp.concatenate([pad, x[..., :-amount]], axis=-1)


def ngram_hash_ids(
    input_ids: jax.Array,
    segment_ids: jax.Array,
    *,
    table_size: int,
    max_ngram_size: int,
    n_hash_heads: int,
    pad_token_id: int = 0,
    seed: int = 0,
) -> jax.Array:
    """Build per-position n-gram hash ids without crossing packed boundaries."""
    if input_ids.shape != segment_ids.shape:
        raise ValueError("input_ids and segment_ids must have the same shape")
    n_cols = (max_ngram_size - 1) * n_hash_heads
    if n_cols <= 0 or table_size < n_cols:
        raise ValueError("engram table is too small for requested hash columns")

    bucket_size = table_size // n_cols
    columns = []
    col = 0
    for ngram_size in range(2, max_ngram_size + 1):
        for head in range(n_hash_heads):
            h = jnp.zeros_like(input_ids, dtype=jnp.uint32)
            for shift in range(ngram_size):
                tok = _shift_right(input_ids, shift, pad_token_id)
                seg = _shift_right(segment_ids, shift, -1)
                valid = seg == segment_ids
                tok = jnp.where(valid, tok, pad_token_id).astype(jnp.uint32)
                mix_value = (
                    0x9E3779B1
                    + 0x85EBCA6B * (head + 1)
                    + 0x27D4EB2D * (shift + 1)
                    + seed
                ) & 0xFFFFFFFF
                mix = jnp.uint32(mix_value)
                h = jnp.bitwise_xor(h * jnp.uint32(16777619), tok + mix)
            bucket = (h % jnp.uint32(bucket_size)).astype(jnp.int32)
            columns.append(bucket + col * bucket_size)
            col += 1
    return jnp.stack(columns, axis=-1)


def init_engram(
    key: jax.Array,
    *,
    table_size: int,
    head_dim: int,
    n_hash_cols: int,
    n_streams: int,
    dim: int,
) -> dict[str, jax.Array]:
    kt, kw = jax.random.split(key)
    table = jax.random.normal(
        kt, (table_size, head_dim), dtype=jnp.float32
    ) * 0.02
    wkv = init_linear(kw, n_hash_cols * head_dim, dim * (n_streams + 1))
    return {
        "table": table,
        "wkv": wkv["weight"],
        "q_weight": jnp.ones((n_streams, dim), dtype=jnp.float32),
        "k_weight": jnp.ones((n_streams, dim), dtype=jnp.float32),
    }


def apply_engram(
    streams: jax.Array,
    hash_ids: jax.Array,
    params: dict[str, jax.Array],
    *,
    eps: float = 1e-6,
    token_mask: jax.Array | None = None,
) -> jax.Array:
    """Gate one shared memory value into each mHC residual stream.

    Table/projection payload follows the stored matrix dtype (BF16 on v5e), while the
    normalization and scalar gate stay FP32. The injected result returns to the residual
    dtype so the large downstream projections remain low precision.
    """
    looked_up = params["table"][hash_ids]
    flat = looked_up.reshape(*looked_up.shape[:-2], -1)
    wkv = params["wkv"]
    kv = jnp.einsum("...d,df->...f", flat.astype(wkv.dtype), wkv)

    n_streams, dim = streams.shape[-2:]
    key = kv[..., : n_streams * dim].reshape(
        *kv.shape[:-1], n_streams, dim
    )
    value = kv[..., n_streams * dim :]

    h = streams.astype(jnp.float32)
    key_fp32 = key.astype(jnp.float32)
    weight = (
        params["q_weight"].astype(jnp.float32)
        * params["k_weight"].astype(jnp.float32)
    )
    rstd = jax.lax.rsqrt(jnp.mean(jnp.square(h), axis=-1) + eps) * jax.lax.rsqrt(
        jnp.mean(jnp.square(key_fp32), axis=-1) + eps
    )
    dot = jnp.sum(h * weight * key_fp32, axis=-1) * rstd * (dim**-0.5)
    signed_sqrt = jnp.sign(dot) * jnp.sqrt(jnp.maximum(jnp.abs(dot), 1e-6))
    gate = jax.nn.sigmoid(signed_sqrt)
    if token_mask is not None:
        gate = jnp.where(token_mask[..., None], gate, 0.0)

    injected = (
        gate.astype(streams.dtype)[..., None]
        * value.astype(streams.dtype)[..., None, :]
    )
    return (streams + injected).astype(streams.dtype)
