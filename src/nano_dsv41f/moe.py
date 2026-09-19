from __future__ import annotations

import jax
import jax.numpy as jnp


def expert_dot_precision(dtype):
    """FP32 payload needs FP32-quality products; BF16 keeps native MXU compute."""
    return jax.lax.Precision.HIGHEST if dtype == jnp.float32 else jax.lax.Precision.DEFAULT


def _init_expert_stack(
    key: jax.Array,
    n: int,
    dim: int,
    hidden: int,
) -> dict[str, jax.Array]:
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "w1": jax.random.normal(k1, (n, dim, hidden), dtype=jnp.float32) * dim**-0.5,
        "w2": jax.random.normal(k2, (n, hidden, dim), dtype=jnp.float32) * hidden**-0.5,
        "w3": jax.random.normal(k3, (n, dim, hidden), dtype=jnp.float32) * dim**-0.5,
    }


def _init_expert(key: jax.Array, dim: int, hidden: int) -> dict[str, jax.Array]:
    stack = _init_expert_stack(key, 1, dim, hidden)
    return {name: value[0] for name, value in stack.items()}


def init_moe(
    key: jax.Array,
    dim: int,
    hidden: int,
    n_experts: int,
) -> dict[str, object]:
    kr, ke, ks = jax.random.split(key, 3)
    return {
        "router_weight": (
            jax.random.normal(kr, (dim, n_experts), dtype=jnp.float32) * dim**-0.5
        ),
        # Correction bias affects expert selection only; output mixture uses unbiased score.
        "router_bias": jnp.zeros((n_experts,), dtype=jnp.float32),
        "experts": _init_expert_stack(ke, n_experts, dim, hidden),
        "shared": _init_expert(ks, dim, hidden),
    }


def _expert_forward(
    x: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w3: jax.Array,
    swiglu_limit: float,
) -> jax.Array:
    """Expert payload follows expert matrix dtype (BF16 on the v5e path)."""
    compute_dtype = w1.dtype
    x_compute = x.astype(compute_dtype)
    precision = expert_dot_precision(compute_dtype)
    gate = jnp.einsum("...d,...df->...f", x_compute, w1, precision=precision)
    up = jnp.einsum("...d,...df->...f", x_compute, w3, precision=precision)
    if swiglu_limit > 0:
        gate = jnp.minimum(gate, jnp.asarray(swiglu_limit, dtype=gate.dtype))
        up = jnp.clip(
            up,
            jnp.asarray(-swiglu_limit, dtype=up.dtype),
            jnp.asarray(swiglu_limit, dtype=up.dtype),
        )
    hidden = jax.nn.silu(gate) * up
    return jnp.einsum("...f,...fd->...d", hidden, w2, precision=precision)


def _expert_forward_fused(x, w1, w2, w3, swiglu_limit):
    """Resident-weight expert: one wide gate/up GEMM, then one down GEMM."""
    precision = expert_dot_precision(w1.dtype)
    gate_up = jnp.matmul(x.astype(w1.dtype), jnp.concatenate((w1, w3), axis=-1), precision=precision)
    gate, up = jnp.split(gate_up, 2, axis=-1)
    if swiglu_limit > 0:
        gate = jnp.minimum(gate, jnp.asarray(swiglu_limit, gate.dtype))
        up = jnp.clip(up, -swiglu_limit, swiglu_limit)
    return jnp.matmul(jax.nn.silu(gate) * up, w2, precision=precision)


def route_tokens(
    x: jax.Array,
    params: dict[str, object],
    *,
    top_k: int,
    route_scale: float = 1.0,
    eps: float = 1e-20,
) -> tuple[jax.Array, jax.Array]:
    """Route in FP32 even when expert matrices/residual payload use BF16."""
    logits = jnp.einsum(
        "...d,de->...e",
        x.astype(jnp.float32),
        params["router_weight"].astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST,
    )
    raw = jnp.sqrt(jax.nn.softplus(logits))
    selection = raw + params["router_bias"].astype(jnp.float32)
    _, indices = jax.lax.top_k(selection, top_k)
    weights = jnp.take_along_axis(raw, indices, axis=-1)
    weights = weights / jnp.maximum(jnp.sum(weights, axis=-1, keepdims=True), eps)
    return weights * route_scale, indices


def _router_loads(
    indices: jax.Array,
    token_mask: jax.Array | None,
    n_experts: int,
) -> jax.Array:
    """Count only real-token assignments in the routing tensor's own sharding domain."""
    if token_mask is None:
        token_mask = jnp.ones(indices.shape[:-1], dtype=bool)
    if token_mask.shape != indices.shape[:-1]:
        raise ValueError("token_mask must match MoE token dimensions")
    assignment_mask = jnp.broadcast_to(token_mask[..., None], indices.shape)
    return jnp.bincount(
        indices.reshape(-1),
        weights=assignment_mask.reshape(-1).astype(jnp.int32),
        length=n_experts,
    )


def apply_moe(
    x: jax.Array,
    params: dict[str, object],
    *,
    top_k: int,
    route_scale: float = 1.0,
    swiglu_limit: float = 7.0,
    eps: float = 1e-20,
    token_mask: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Readable routed-MoE reference with one always-on shared expert."""
    weights, indices = route_tokens(
        x, params, top_k=top_k, route_scale=route_scale, eps=eps
    )
    experts = params["experts"]
    router_loads = _router_loads(indices, token_mask, int(experts["w1"].shape[0]))
    w1 = experts["w1"][indices]
    w2 = experts["w2"][indices]
    w3 = experts["w3"][indices]
    selected = _expert_forward(x[..., None, :], w1, w2, w3, swiglu_limit)
    routed = jnp.sum(
        weights.astype(selected.dtype)[..., None] * selected,
        axis=-2,
    )

    shared = params["shared"]
    shared_out = _expert_forward(
        x, shared["w1"], shared["w2"], shared["w3"], swiglu_limit
    )
    out = routed.astype(x.dtype) + shared_out.astype(x.dtype)
    return out.astype(x.dtype), {
        "router_indices": indices,
        "router_weights": weights,
        "router_loads": router_loads,
    }
