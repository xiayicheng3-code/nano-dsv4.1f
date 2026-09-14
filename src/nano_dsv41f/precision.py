from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from .model import init_model
from .tpu import named_shardings, parameter_partition_specs


def is_payload_parameter(x: jax.Array) -> bool:
    """Large matrix/table tensors use the TPU payload dtype; vectors stay FP32."""
    return bool(jnp.issubdtype(x.dtype, jnp.floating) and x.ndim >= 2)


def cast_payload_parameters(params, dtype=jnp.bfloat16):
    """Cast matrix/table payloads while retaining FP32 norms, biases and scalar controls."""
    dtype = jnp.dtype(dtype)
    if dtype not in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float32)):
        raise ValueError("current TPU policy supports bfloat16 or float32 payloads")
    return jax.tree_util.tree_map(
        lambda x: x.astype(dtype) if is_payload_parameter(x) else x,
        params,
    )


def init_model_sharded_mixed_precision(
    key: jax.Array,
    config,
    mesh,
    *,
    payload_dtype=jnp.bfloat16,
):
    """Initialize directly into final shards with BF16 matrix/table payloads.

    The unsharded FP32 model never exists as a runtime array. `eval_shape` only creates
    abstract metadata; the jitted initializer creates, casts and writes each result into its
    final NamedSharding. This matters on 16-GB/chip v5e when scaling the nano dimensions.
    """
    abstract_fp32 = jax.eval_shape(lambda k: init_model(k, config), key)
    specs = parameter_partition_specs(abstract_fp32, config, mesh)
    shardings = named_shardings(specs, mesh)
    key = jax.device_put(key, NamedSharding(mesh, P()))

    def initialize(k):
        return cast_payload_parameters(init_model(k, config), payload_dtype)

    init_fn = jax.jit(initialize, out_shardings=shardings)
    return init_fn(key), specs, shardings


def precision_summary(params) -> dict[str, int]:
    """Count parameter elements by dtype for notebook HBM sanity checks."""
    out: dict[str, int] = {}
    for leaf in jax.tree_util.tree_leaves(params):
        key = str(leaf.dtype)
        out[key] = out.get(key, 0) + int(leaf.size)
    return out
