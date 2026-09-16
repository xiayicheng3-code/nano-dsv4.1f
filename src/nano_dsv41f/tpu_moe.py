from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from .moe import _expert_forward, apply_moe as apply_moe_reference, route_tokens
from .tpu_native import TPUNativeState, _moe_param_specs, manual_v5e_mesh, moe_capacity


def apply_moe_v5e_multi(
    x: jax.Array,
    params: dict[str, object],
    *,
    top_k: int,
    route_scale: float,
    swiglu_limit: float,
    eps: float,
    state: TPUNativeState,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Static-capacity expert parallelism with E / chips resident experts per chip.

    The current v5e path intentionally keeps the simple all-gather + psum communication
    scheme. This function only removes the one-expert-per-chip restriction: expert matrices
    stay sharded on the expert axis while each chip evaluates its contiguous local expert
    group. A future ragged/all-to-all dispatcher can replace the communication pattern
    without changing routing semantics.
    """
    if x.ndim != 3:
        raise ValueError("v5e EP MoE expects x=[batch,tokens,dim]")

    n_experts = int(params["experts"]["w1"].shape[0])
    manual = manual_v5e_mesh(state.mesh, state.options.manual_axis_name)
    shards = int(manual.size)
    if n_experts < shards or n_experts % shards:
        return apply_moe_reference(
            x,
            params,
            top_k=top_k,
            route_scale=route_scale,
            swiglu_limit=swiglu_limit,
            eps=eps,
        )

    local_experts = n_experts // shards
    batch, tokens, dim = map(int, x.shape)
    capacity = moe_capacity(
        batch * tokens,
        top_k=top_k,
        n_experts=n_experts,
        capacity_factor=state.options.moe_capacity_factor,
        multiple=state.options.moe_capacity_multiple,
    )
    axis = state.options.manual_axis_name
    param_specs = _moe_param_specs(axis)

    @jax.shard_map(
        mesh=manual,
        in_specs=(P(None, axis, None), param_specs),
        out_specs=(
            P(None, axis, None),
            P(None, axis, None),
            P(None, axis, None),
            P(axis),
            P(axis),
        ),
        check_vma=False,
    )
    def _mapped(local_x, local_params):
        local_weights, local_indices = route_tokens(
            local_x,
            local_params,
            top_k=top_k,
            route_scale=route_scale,
            eps=eps,
        )
        global_x = jax.lax.all_gather(local_x, axis, axis=1, tiled=True)
        global_weights = jax.lax.all_gather(local_weights, axis, axis=1, tiled=True)
        global_indices = jax.lax.all_gather(local_indices, axis, axis=1, tiled=True)

        n_global = int(global_x.shape[0] * global_x.shape[1])
        n_assignments = n_global * top_k
        flat_x = global_x.reshape(n_global, dim)
        flat_weights = global_weights.reshape(n_assignments)
        flat_indices = global_indices.reshape(n_assignments)

        chip = jax.lax.axis_index(axis)
        expert_base = chip * local_experts
        local_ids = expert_base + jnp.arange(local_experts, dtype=flat_indices.dtype)
        matches = flat_indices[None, :] == local_ids[:, None]

        selected_valid, selected_assignment = jax.lax.top_k(
            matches.astype(jnp.int32), capacity
        )
        selected_token = selected_assignment // top_k
        selected_x = flat_x[selected_token]

        experts = local_params["experts"]
        selected_out = jax.vmap(
            lambda xe, w1, w2, w3: _expert_forward(
                xe, w1, w2, w3, swiglu_limit
            )
        )(
            selected_x,
            experts["w1"],
            experts["w2"],
            experts["w3"],
        )
        selected_weights = jnp.take(flat_weights, selected_assignment)
        selected_out = selected_out * (
            selected_weights[..., None].astype(selected_out.dtype)
            * selected_valid[..., None].astype(selected_out.dtype)
        )

        def scatter_one(token_ids, values):
            return jnp.zeros((n_global, dim), dtype=values.dtype).at[token_ids].add(values)

        local_contribution = jax.vmap(scatter_one)(
            selected_token, selected_out
        ).sum(axis=0)
        contribution = local_contribution.reshape(global_x.shape)
        routed_global = jax.lax.psum(contribution, axis)

        local_tokens = int(local_x.shape[1])
        start = chip * local_tokens
        routed_local = jax.lax.dynamic_slice_in_dim(
            routed_global, start, local_tokens, axis=1
        )

        shared = local_params["shared"]
        shared_out = _expert_forward(
            local_x,
            shared["w1"],
            shared["w2"],
            shared["w3"],
            swiglu_limit,
        )
        out = routed_local.astype(local_x.dtype) + shared_out.astype(local_x.dtype)
        loads = jnp.sum(matches.astype(jnp.int32), axis=-1)
        overflow = jnp.maximum(loads - capacity, 0)
        return out.astype(local_x.dtype), local_weights, local_indices, loads, overflow

    out, weights, indices, loads, overflow = _mapped(x, params)
    return out, {
        "router_indices": indices,
        "router_weights": weights,
        "expert_loads": loads,
        "expert_overflow": overflow,
        "expert_capacity": jnp.asarray(capacity, dtype=jnp.int32),
        "experts_per_chip": jnp.asarray(local_experts, dtype=jnp.int32),
    }
