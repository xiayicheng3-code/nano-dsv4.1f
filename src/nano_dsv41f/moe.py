from __future__ import annotations

import jax
import jax.numpy as jnp


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
        # Mirrors no-aux routing semantics: this bias changes selection only.
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
    gate = jnp.einsum("...d,...df->...f", x, w1)
    up = jnp.einsum("...d,...df->...f", x, w3)
    if swiglu_limit > 0:
        gate = jnp.minimum(gate, swiglu_limit)
        up = jnp.clip(up, -swiglu_limit, swiglu_limit)
    hidden = jax.nn.silu(gate) * up
    return jnp.einsum("...f,...fd->...d", hidden, w2)


def route_tokens(
    x: jax.Array,
    params: dict[str, object],
    *,
    top_k: int,
    route_scale: float = 1.0,
) -> tuple[jax.Array, jax.Array]:
    logits = jnp.einsum(
        "...d,de->...e", x.astype(jnp.float32), params["router_weight"]
    )
    # V4.1 uses sqrt(softplus(.)) routing scores.
    raw = jnp.sqrt(jax.nn.softplus(logits))
    selection = raw + params["router_bias"]
    _, indices = jax.lax.top_k(selection, top_k)
    # The correction bias selects experts but does not scale expert outputs.
    weights = jnp.take_along_axis(raw, indices, axis=-1)
    weights = weights / jnp.maximum(jnp.sum(weights, axis=-1, keepdims=True), 1e-9)
    return weights * route_scale, indices


def apply_moe(
    x: jax.Array,
    params: dict[str, object],
    *,
    top_k: int,
    route_scale: float = 1.0,
    swiglu_limit: float = 7.0,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Reference sparse MoE with one shared expert.

    Parameter gathers keep compute proportional to selected experts in this readable path.
    A TPU expert-parallel implementation will replace this gather pattern later.
    """
    weights, indices = route_tokens(
        x, params, top_k=top_k, route_scale=route_scale
    )
    experts = params["experts"]
    w1 = experts["w1"][indices]
    w2 = experts["w2"][indices]
    w3 = experts["w3"][indices]
    selected = _expert_forward(x[..., None, :], w1, w2, w3, swiglu_limit)
    routed = jnp.sum(weights[..., None] * selected, axis=-2)

    shared = params["shared"]
    shared_out = _expert_forward(
        x, shared["w1"], shared["w2"], shared["w3"], swiglu_limit
    )
    return routed + shared_out, {
        "router_indices": indices,
        "router_weights": weights,
    }
