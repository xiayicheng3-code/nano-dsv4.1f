from __future__ import annotations

import jax.numpy as jnp


def sinkhorn(matrix: jnp.ndarray, iters: int = 4, eps: float = 1e-6) -> jnp.ndarray:
    """Small differentiable doubly-stochastic normalization used by the mHC reference."""
    x = jnp.exp(matrix - jnp.max(matrix, axis=(-2, -1), keepdims=True))
    for _ in range(iters):
        x = x / jnp.maximum(jnp.sum(x, axis=-1, keepdims=True), eps)
        x = x / jnp.maximum(jnp.sum(x, axis=-2, keepdims=True), eps)
    return x


def pre_mix(streams: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """Collapse multiple residual streams into one sublayer input.

    streams: [..., S, D]
    weights: [..., S]
    """
    return jnp.einsum("...s,...sd->...d", weights, streams)


def post_mix(
    streams: jnp.ndarray,
    branch_output: jnp.ndarray,
    combine: jnp.ndarray,
    branch_weights: jnp.ndarray,
) -> jnp.ndarray:
    """Single-pass residual update used by the educational mHC implementation.

    combine is a doubly-stochastic [..., S, S] matrix that mixes old residual streams.
    branch_weights distributes the newly computed branch output back to the S streams.
    """
    mixed_old = jnp.einsum("...ij,...jd->...id", combine, streams)
    return mixed_old + branch_weights[..., :, None] * branch_output[..., None, :]


def mhc_step(
    streams: jnp.ndarray,
    branch_fn,
    pre_logits: jnp.ndarray,
    combine_logits: jnp.ndarray,
    post_logits: jnp.ndarray,
    *,
    sinkhorn_iters: int = 4,
) -> jnp.ndarray:
    """Transparent reference composition for a Single-Pass mHC-like residual branch.

    This file intentionally isolates the residual-stream mechanics from the parameter
    generator. DeepSeek's released model learns the pre/post/combine coefficients from
    the current residual state; the model module will provide that generator separately.
    """
    pre = jnp.softmax(pre_logits, axis=-1)
    post = jnp.softmax(post_logits, axis=-1)
    combine = sinkhorn(combine_logits, iters=sinkhorn_iters)
    branch_input = pre_mix(streams, pre)
    branch_output = branch_fn(branch_input)
    return post_mix(streams, branch_output, combine, post)
