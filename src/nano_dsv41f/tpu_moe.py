"""Dropless, tiled expert parallelism with resident expert weights."""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from .moe import _expert_forward_fused, apply_moe as apply_moe_reference, route_tokens
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
    """Compact assignments into MXU-sized tiles; process every routed assignment.

    Capacity is a tile size, never a token-dropping limit. Overflow tiles execute only
    when needed. Their static scan bound supports reverse-mode AD; rematerialization
    keeps expert activations bounded by one tile. No collectives live inside the
    data-dependent branches, so chips with different expert loads cannot deadlock.
    """
    if x.ndim != 3:
        raise ValueError("v5e EP MoE expects x=[batch,tokens,dim]")
    n_experts = int(params["experts"]["w1"].shape[0])
    manual = manual_v5e_mesh(state.mesh, state.options.manual_axis_name)
    shards = int(manual.size)
    if n_experts < shards or n_experts % shards:
        return apply_moe_reference(
            x, params, top_k=top_k, route_scale=route_scale,
            swiglu_limit=swiglu_limit, eps=eps,
        )
    local_experts = n_experts // shards
    batch, tokens, dim = map(int, x.shape)
    n_global = batch * tokens
    # Top-K returns distinct experts: one expert can receive at most n_global tokens.
    capacity = min(n_global, moe_capacity(
        n_global, top_k=top_k, n_experts=n_experts,
        capacity_factor=state.options.moe_capacity_factor,
        multiple=state.options.moe_capacity_multiple,
    ))
    rounds = math.ceil(n_global / capacity)
    padded_slots = rounds * capacity
    axis = state.options.manual_axis_name

    @jax.shard_map(
        mesh=manual,
        in_specs=(P(None, axis, None), _moe_param_specs(axis)),
        out_specs=(P(None, axis, None), P(None, axis, None),
                   P(None, axis, None), P(axis), P(axis)),
        check_vma=False,
    )
    def _mapped(local_x, local_params):
        local_weights, local_indices = route_tokens(
            local_x, local_params, top_k=top_k, route_scale=route_scale, eps=eps,
        )
        global_x = jax.lax.all_gather(local_x, axis, axis=1, tiled=True)
        weights = jax.lax.all_gather(local_weights, axis, axis=1, tiled=True).reshape(-1)
        indices = jax.lax.all_gather(local_indices, axis, axis=1, tiled=True).reshape(-1)
        flat_x = global_x.reshape(n_global, dim)
        local_ids = jax.lax.axis_index(axis) * local_experts + jnp.arange(local_experts)
        matches = indices[None, :] == local_ids[:, None]
        loads = jnp.sum(matches, axis=-1, dtype=jnp.int32)
        slots = jnp.cumsum(matches, axis=-1, dtype=jnp.int32) - 1
        slots = jnp.where(matches, slots, padded_slots)  # out-of-bounds writes are dropped
        assignment_ids = jnp.broadcast_to(jnp.arange(indices.size), matches.shape)
        expert_ids = jnp.broadcast_to(jnp.arange(local_experts)[:, None], matches.shape)
        compact = jnp.full((local_experts, padded_slots), -1, dtype=jnp.int32)
        compact = compact.at[expert_ids, slots].set(assignment_ids, mode="drop")
        experts = local_params["experts"]

        def tile(round_id, contribution):
            def evaluate(acc):
                selected = jax.lax.dynamic_slice_in_dim(
                    compact, round_id * capacity, capacity, axis=1,
                )
                valid = selected >= 0
                safe = jnp.maximum(selected, 0)
                selected_token = safe // top_k
                selected_out = jax.vmap(
                    lambda xe, w1, w2, w3: _expert_forward_fused(
                        xe, w1, w2, w3, swiglu_limit,
                    )
                )(flat_x[selected_token], experts["w1"], experts["w2"], experts["w3"])
                values = selected_out.astype(jnp.float32) * jnp.where(
                    valid, weights[safe], 0.0,
                )[..., None]
                # One accumulator rather than [local_experts, n_global, dim].
                return acc.at[selected_token.reshape(-1)].add(values.reshape(-1, dim))
            return jax.lax.cond(jnp.max(loads) > round_id * capacity,
                                evaluate, lambda acc: acc, contribution)

        contribution = jax.lax.fori_loop(
            0, rounds, jax.checkpoint(tile),
            jnp.zeros((n_global, dim), dtype=jnp.float32),
        ).reshape(global_x.shape)
        routed_local = jax.lax.psum_scatter(contribution, axis, scatter_dimension=1, tiled=True)
        shared = local_params["shared"]
        shared_out = _expert_forward_fused(
            local_x, shared["w1"], shared["w2"], shared["w3"], swiglu_limit,
        )
        out = routed_local.astype(local_x.dtype) + shared_out.astype(local_x.dtype)
        return out, local_weights, local_indices, loads, jnp.maximum(loads - capacity, 0)

    out, weights, indices, loads, overflow = _mapped(x, params)
    return out, {
        "router_indices": indices, "router_weights": weights,
        "expert_loads": loads,
        # Kept for monitoring: excess over the first tile, NOT dropped assignments.
        "expert_overflow": overflow,
        "expert_dropped": jnp.zeros_like(loads),
        "expert_capacity": jnp.asarray(capacity, dtype=jnp.int32),
        "experts_per_chip": jnp.asarray(local_experts, dtype=jnp.int32),
    }
