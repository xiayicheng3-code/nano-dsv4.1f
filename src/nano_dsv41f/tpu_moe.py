"""Dropless expert parallelism using Tokamax 0.0.12 grouped matrix products."""
from __future__ import annotations

from functools import lru_cache
import sys

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from .moe import _expert_forward_fused, _router_loads, expert_dot_precision, route_tokens
from .tpu_native import TPUNativeState, _moe_param_specs, manual_v5e_mesh


def ragged_dot_implementation(state: TPUNativeState) -> str:
    """Select explicitly: TPU runs Mosaic; CPU correctness tests run XLA."""
    implementation = state.options.moe_ragged_implementation
    if implementation == "auto":
        device = state.mesh.devices.flat[0]
        return "mosaic" if device.platform == "tpu" else "xla"
    return implementation


@lru_cache(maxsize=1)
def _tokamax():
    import tokamax
    from absl import flags

    # Tokamax 0.0.12 otherwise lazily parses all of sys.argv and rejects our
    # argparse/notebook/pytest options. Leave application options to their owner.
    if not flags.FLAGS.is_parsed():
        flags.FLAGS(sys.argv, known_only=True)
    return tokamax


def _ragged_dot(lhs, rhs, group_sizes, *, implementation):
    tokamax = _tokamax()

    rows, inner = lhs.shape
    outputs = rhs.shape[-1]
    # 0.0.12 Mosaic requires non-expert dimensions >=128. Padding also supports
    # nano shapes which are not multiples of its default 128-wide kernel tiles.
    if implementation == "mosaic":
        pad = lambda n: (-n) % 128
        lhs = jnp.pad(lhs, ((0, pad(rows)), (0, pad(inner))))
        rhs = jnp.pad(rhs, ((0, 0), (0, pad(inner)), (0, pad(outputs))))
    valid = jnp.arange(lhs.shape[0]) < jnp.sum(group_sizes)
    # Mosaic can leave rows beyond sum(group_sizes) unwritten. Mask both sides
    # of EVERY dot, including its input gradient, before any nonlinear operation.
    lhs = jnp.where(valid[:, None], lhs, 0)
    # Tokamax DEFAULT maps FP32 operands to a single BF16 product on TPU.
    # Explicit precision also propagates through Tokamax's input/weight VJPs.
    out = tokamax.ragged_dot(
        lhs, rhs, group_sizes, precision=expert_dot_precision(rhs.dtype),
        implementation=implementation,
    )
    return jnp.where(valid[:, None], out, 0)[:rows, :outputs]


@jax.named_call
def ragged_expert_forward(x, experts, group_sizes, *, swiglu_limit, implementation):
    """Two grouped GEMMs with Tokamax's input/weight VJPs and clipped SwiGLU."""
    gate_up = _ragged_dot(
        x.astype(experts["w1"].dtype),
        jnp.concatenate((experts["w1"], experts["w3"]), axis=-1),
        group_sizes, implementation=implementation,
    )
    gate, up = jnp.split(gate_up, 2, axis=-1)
    if swiglu_limit > 0:
        gate = jnp.minimum(gate, jnp.asarray(swiglu_limit, gate.dtype))
        up = jnp.clip(up, -swiglu_limit, swiglu_limit)
    return _ragged_dot(
        jax.nn.silu(gate) * up, experts["w2"], group_sizes,
        implementation=implementation,
    )


def apply_moe_v5e_multi(
    x: jax.Array,
    params: dict[str, object],
    *,
    top_k: int,
    route_scale: float,
    swiglu_limit: float,
    eps: float,
    state: TPUNativeState,
    token_mask: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Sort local expert assignments, execute ragged GEMMs, and reduce-scatter.

    All assignments fit in a static packed buffer. Runtime group sizes determine
    which rows the Mosaic kernel computes; there is no per-expert capacity or tile
    loop in this adapter. Weight shards remain resident. The readable MoE in moe.py
    is an independent numerical oracle, never a silent native-backend fallback.
    """
    if x.ndim != 3:
        raise ValueError("v5e EP MoE expects x=[batch,tokens,dim]")
    if token_mask is None:
        token_mask = jnp.ones(x.shape[:2], dtype=bool)
    if token_mask.shape != x.shape[:2]:
        raise ValueError("token_mask must match x [batch,tokens]")
    n_experts = int(params["experts"]["w1"].shape[0])
    if not 0 < top_k <= n_experts:
        raise ValueError("top_k must be in [1, n_experts]")
    manual = manual_v5e_mesh(state.mesh, state.options.manual_axis_name)
    shards = int(manual.size)
    if n_experts < shards or n_experts % shards:
        raise ValueError("native MoE needs n_experts divisible by the expert mesh size")
    local_experts = n_experts // shards
    batch, tokens, dim = map(int, x.shape)
    if tokens % shards:
        raise ValueError("token dimension must be divisible by the expert mesh size")
    n_global = batch * tokens
    # Top-K experts are distinct. This bound holds even if all tokens choose the
    # same local experts; unlike an average-load capacity, it cannot drop tokens.
    packed_rows = n_global * min(top_k, local_experts)
    axis = state.options.manual_axis_name
    implementation = ragged_dot_implementation(state)

    @jax.shard_map(
        mesh=manual,
        in_specs=(P(None, axis, None), P(None, axis), _moe_param_specs(axis)),
        out_specs=(P(None, axis, None), P(None, axis, None),
                   P(None, axis, None), P(axis), P()),
        check_vma=False,
    )
    def _mapped(local_x, local_token_mask, local_params):
        local_weights, local_indices = route_tokens(
            local_x, local_params, top_k=top_k, route_scale=route_scale, eps=eps,
        )
        # Preserve the tp-domain router accounting fix: only [E] crosses meshes.
        router_loads = jax.lax.psum(
            _router_loads(local_indices, local_token_mask, n_experts), axis
        )
        with jax.named_scope("moe_all_gather"):
            global_x = jax.lax.all_gather(local_x, axis, axis=1, tiled=True)
            weights = jax.lax.all_gather(local_weights, axis, axis=1, tiled=True).reshape(-1)
            indices = jax.lax.all_gather(local_indices, axis, axis=1, tiled=True).reshape(-1)
        flat_x = global_x.reshape(n_global, dim)
        with jax.named_scope("moe_dispatch_sort"):
            relative_ids = indices - jax.lax.axis_index(axis) * local_experts
            owned = (relative_ids >= 0) & (relative_ids < local_experts)
            keys = jnp.where(owned, relative_ids, local_experts)
            loads = jnp.bincount(keys, length=local_experts + 1)[:local_experts]
            # Tokamax 0.0.12 rejects group_offset. Compact local groups from row zero
            # instead, placing non-owned assignments in a masked suffix.
            _, assignment_ids = jax.lax.sort_key_val(keys, jnp.arange(indices.size), is_stable=True)
            assignment_ids = assignment_ids[:packed_rows]
            token_ids = assignment_ids // top_k
            valid = jnp.arange(packed_rows) < jnp.sum(loads)

        def evaluate(_):
            return ragged_expert_forward(
                flat_x[token_ids], local_params["experts"], loads,
                swiglu_limit=swiglu_limit, implementation=implementation,
            )

        # Empty chips skip both GEMMs. All collectives remain outside this branch.
        selected_out = jax.lax.cond(
            jnp.any(loads > 0), evaluate,
            lambda _: jnp.zeros((packed_rows, dim), local_params["experts"]["w2"].dtype),
            operand=None,
        )
        with jax.named_scope("moe_combine_scatter"):
            values = jnp.where(valid[:, None], selected_out.astype(jnp.float32), 0)
            values *= jnp.where(valid, weights[assignment_ids], 0)[:, None]
            contribution = jnp.zeros((n_global, dim), jnp.float32).at[token_ids].add(values)
        contribution = contribution.reshape(global_x.shape)
        routed_local = jax.lax.psum_scatter(contribution, axis, scatter_dimension=1, tiled=True)
        shared = local_params["shared"]
        shared_out = _expert_forward_fused(
            local_x, shared["w1"], shared["w2"], shared["w3"], swiglu_limit,
        )
        out = routed_local.astype(local_x.dtype) + shared_out.astype(local_x.dtype)
        return out, local_weights, local_indices, loads, router_loads

    out, weights, indices, loads, router_loads = _mapped(x, token_mask, params)
    router_loads = jax.lax.with_sharding_constraint(router_loads, NamedSharding(state.mesh, P()))
    return out, {
        "router_indices": indices, "router_weights": weights,
        "router_loads": router_loads, "expert_loads": loads,
        # Legacy diagnostic keys: ragged dispatch has no per-expert overflow cap.
        "expert_overflow": jnp.zeros_like(loads),
        "expert_dropped": jnp.zeros_like(loads),
        "expert_capacity": jnp.asarray(n_global, dtype=jnp.int32),
        "expert_packed_rows": jnp.asarray(packed_rows, dtype=jnp.int32),
        "moe_mosaic": jnp.asarray(implementation == "mosaic", dtype=jnp.int32),
        "experts_per_chip": jnp.asarray(local_experts, dtype=jnp.int32),
    }
