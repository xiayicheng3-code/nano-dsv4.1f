from __future__ import annotations

import jax.nn as jnn
import jax.numpy as jnp


def learned_group_compress(
    x: jnp.ndarray,
    score: jnp.ndarray,
    *,
    ratio: int,
) -> jnp.ndarray:
    """Learned pooling over contiguous token groups.

    Compression weights are normalized in FP32, then cast back to the payload dtype before
    the weighted reduction. This mirrors the TPU mixed-precision policy used elsewhere:
    numerically sensitive control math stays FP32 without promoting the stored latent.
    """
    if ratio not in (1, 2):
        raise ValueError("current reference supports compression ratio 1 or 2")
    if x.shape[-2] % ratio:
        raise ValueError("token dimension must be divisible by ratio")
    if score.shape not in (x.shape[:-1], x.shape):
        raise ValueError("score must have shape x.shape[:-1] or x.shape")
    if ratio == 1:
        return x

    t, d = x.shape[-2:]
    grouped_x = x.reshape(*x.shape[:-2], t // ratio, ratio, d)
    if score.shape == x.shape:
        grouped_score = score.reshape(
            *score.shape[:-2], t // ratio, ratio, d
        ).astype(jnp.float32)
        weight = jnn.softmax(grouped_score, axis=-2).astype(x.dtype)
    else:
        grouped_score = score.reshape(
            *score.shape[:-1], t // ratio, ratio
        ).astype(jnp.float32)
        weight = jnn.softmax(grouped_score, axis=-1).astype(x.dtype)[..., None]
    return jnp.sum(grouped_x * weight, axis=-2).astype(x.dtype)


def completed_group_mask(length: int, ratio: int) -> jnp.ndarray:
    """Tiny reference mask for when each compression group has causally completed."""
    if length % ratio:
        raise ValueError("length must be divisible by ratio")
    q = jnp.arange(length)[:, None]
    group_last_token = (
        jnp.arange(length // ratio)[None, :] + 1
    ) * ratio - 1
    return group_last_token <= q
