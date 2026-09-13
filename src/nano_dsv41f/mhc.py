from __future__ import annotations

import jax
import jax.numpy as jnp


def sinkhorn(
    matrix: jnp.ndarray,
    iters: int = 4,
    eps: float = 1e-6,
) -> jnp.ndarray:
    """Differentiable doubly-stochastic normalization for the mHC reference."""
    x = jnp.exp(matrix - jnp.max(matrix, axis=(-2, -1), keepdims=True))
    for _ in range(iters):
        x = x / jnp.maximum(jnp.sum(x, axis=-1, keepdims=True), eps)
        x = x / jnp.maximum(jnp.sum(x, axis=-2, keepdims=True), eps)
    return x


def pre_mix(streams: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """Collapse residual streams into one sublayer input."""
    return jnp.einsum("...s,...sd->...d", weights, streams)


def post_mix(
    streams: jnp.ndarray,
    branch_output: jnp.ndarray,
    combine: jnp.ndarray,
    branch_weights: jnp.ndarray,
) -> jnp.ndarray:
    """Mix old streams and distribute the new branch output back to all streams."""
    mixed_old = jnp.einsum("...ij,...jd->...id", combine, streams)
    return mixed_old + branch_weights[..., :, None] * branch_output[..., None, :]


def make_identity_pre_mix(streams: jnp.ndarray) -> jnp.ndarray:
    """The released model starts by reading only residual stream zero."""
    weights = jnp.zeros(streams.shape[:-1], dtype=jnp.float32)
    return weights.at[..., 0].set(1.0)


def init_mhc_generator(
    key: jax.Array,
    n_streams: int,
    dim: int,
) -> dict[str, jax.Array]:
    """Initialize the state-conditioned pre/post/combine coefficient generator."""
    out_dim = (2 + n_streams) * n_streams
    weight = (
        jax.random.normal(
            key, (n_streams * dim, out_dim), dtype=jnp.float32
        )
        * 0.01
    )
    base = jnp.zeros((out_dim,), dtype=jnp.float32)
    # Start the residual-stream combination near identity rather than a uniform soup.
    comb_start = 2 * n_streams
    base = base.at[comb_start:].set(
        (2.0 * jnp.eye(n_streams, dtype=jnp.float32)).reshape(-1)
    )
    return {
        "weight": weight,
        "base": base,
        "scale": jnp.ones((3,), dtype=jnp.float32),
    }


def mhc_mixes(
    streams: jnp.ndarray,
    params: dict[str, jax.Array],
    *,
    sinkhorn_iters: int = 4,
    eps: float = 1e-6,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Generate pre/post/comb coefficients from the current multi-stream residual.

    DeepSeek computes all three coefficient sets from one projection of the flattened
    residual state. We keep that dataflow explicit; production Mega-mHC fusion is omitted.
    """
    n_streams = streams.shape[-2]
    flat = streams.astype(jnp.float32).reshape(*streams.shape[:-2], -1)
    flat = flat * jax.lax.rsqrt(
        jnp.mean(jnp.square(flat), axis=-1, keepdims=True) + eps
    )
    raw = jnp.einsum("...d,do->...o", flat, params["weight"]) + params["base"]

    pre_raw = raw[..., :n_streams] * params["scale"][0]
    post_raw = raw[..., n_streams : 2 * n_streams] * params["scale"][1]
    comb_raw = raw[..., 2 * n_streams :].reshape(
        *raw.shape[:-1], n_streams, n_streams
    )
    comb_raw = comb_raw * params["scale"][2]

    return (
        jax.nn.softmax(pre_raw, axis=-1),
        jax.nn.softmax(post_raw, axis=-1),
        sinkhorn(comb_raw, iters=sinkhorn_iters, eps=eps),
    )


def mhc_step(
    streams: jnp.ndarray,
    branch_fn,
    pre_logits: jnp.ndarray,
    combine_logits: jnp.ndarray,
    post_logits: jnp.ndarray,
    *,
    sinkhorn_iters: int = 4,
) -> jnp.ndarray:
    """Legacy explicit-coefficient helper retained for small isolated experiments."""
    pre = jax.nn.softmax(pre_logits, axis=-1)
    post = jax.nn.softmax(post_logits, axis=-1)
    combine = sinkhorn(combine_logits, iters=sinkhorn_iters)
    branch_input = pre_mix(streams, pre)
    branch_output = branch_fn(branch_input)
    return post_mix(streams, branch_output, combine, post)
