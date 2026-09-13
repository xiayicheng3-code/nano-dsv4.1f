from __future__ import annotations

import jax.nn as jnn
import jax.numpy as jnp


def learned_group_compress(
    x: jnp.ndarray,
    score: jnp.ndarray,
    *,
    ratio: int,
) -> jnp.ndarray:
    """Reference learned pooling over contiguous token groups.

    Args:
      x: [..., T, D]
      score: [..., T] scalar pooling logits aligned with x
      ratio: 1 or 2 in the current educational model

    Returns:
      [..., T // ratio, D]

    Every packed segment must be padded independently to a multiple of `ratio` before
    calling this function. Compression must never form a group across segment boundaries.
    """
    if ratio not in (1, 2):
        raise ValueError("current reference supports compression ratio 1 or 2")
    if x.shape[-2] % ratio:
        raise ValueError("token dimension must be divisible by ratio")
    if score.shape != x.shape[:-1]:
        raise ValueError("score must have shape x.shape[:-1]")
    if ratio == 1:
        return x

    t, d = x.shape[-2:]
    grouped_x = x.reshape(*x.shape[:-2], t // ratio, ratio, d)
    grouped_score = score.reshape(*score.shape[:-1], t // ratio, ratio)
    weight = jnn.softmax(grouped_score, axis=-1)
    return jnp.sum(grouped_x * weight[..., None], axis=-2)


def completed_group_mask(length: int, ratio: int) -> jnp.ndarray:
    """Map token positions to whether each compression group is causally complete.

    Returns [T, T//r]. This is a tiny reference helper for tests. Long-context kernels
    should compute the equivalent frontier from ids instead of materializing this array.
    """
    if length % ratio:
        raise ValueError("length must be divisible by ratio")
    q = jnp.arange(length)[:, None]
    group_last_token = (jnp.arange(length // ratio)[None, :] + 1) * ratio - 1
    return group_last_token <= q
