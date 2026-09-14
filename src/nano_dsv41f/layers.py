from __future__ import annotations

import jax
import jax.numpy as jnp


def init_linear(
    key: jax.Array,
    in_dim: int,
    out_dim: int,
    *,
    scale: float | None = None,
) -> dict[str, jax.Array]:
    if in_dim <= 0 or out_dim <= 0:
        raise ValueError("linear dimensions must be positive")
    std = in_dim**-0.5 if scale is None else scale
    weight = jax.random.normal(key, (in_dim, out_dim), dtype=jnp.float32) * std
    return {"weight": weight}


def linear(x: jax.Array, params: dict[str, jax.Array]) -> jax.Array:
    """Linear projection whose compute dtype follows the stored matrix dtype.

    CPU/reference initialization leaves matrices in FP32. The v5e initializer can store
    matrix/table parameters as BF16; explicitly casting the activation here prevents an
    FP32 residual/control path from silently promoting a large MXU matmul back to FP32.
    """
    weight = params["weight"]
    x_compute = x.astype(weight.dtype) if x.dtype != weight.dtype else x
    return jnp.einsum("...d,df->...f", x_compute, weight)


def init_rms_norm(dim: int) -> dict[str, jax.Array]:
    return {"weight": jnp.ones((dim,), dtype=jnp.float32)}


def rms_norm(
    x: jax.Array,
    params: dict[str, jax.Array],
    *,
    eps: float = 1e-6,
) -> jax.Array:
    # Statistics and scale application stay FP32; the normalized activation returns to the
    # residual stream dtype so BF16 TPU paths do not get permanently promoted.
    xf = x.astype(jnp.float32)
    scale = jax.lax.rsqrt(jnp.mean(jnp.square(xf), axis=-1, keepdims=True) + eps)
    return (xf * scale * params["weight"].astype(jnp.float32)).astype(x.dtype)


def init_embedding(
    key: jax.Array,
    vocab_size: int,
    dim: int,
    *,
    scale: float = 0.02,
) -> jax.Array:
    return jax.random.normal(key, (vocab_size, dim), dtype=jnp.float32) * scale
