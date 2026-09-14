from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .model import init_model
from .optimizer import OptimizerLeafState, classify_parameter, init_optimizer_state
from .training import pretrain_step


@dataclass(frozen=True)
class V5EHardware:
    """Public single-chip / single-host v5e properties used for notebook guardrails.

    The Cloud TPU product page reports 16 GB HBM and 800 GiB/s, while JAX's Pallas
    hardware table reports the same generation as 17 GB / 820 GB/s using different unit
    conventions. We keep the Cloud values here because they are the safer HBM budget.
    """

    chips_per_host: int = 8
    topology: tuple[int, int] = (2, 4)
    hbm_gb_per_chip: float = 16.0
    hbm_gibps_per_chip: float = 800.0
    ici_bidirectional_gbps_per_chip: float = 400.0
    vmem_mib_per_tensorcore: int = 128
    bf16_tflops_per_chip: float = 197.0
    int8_tops_per_chip: float = 393.0
    native_fp8: bool = False


V5E = V5EHardware()
V5E_AXIS_NAMES = ("x", "y")


def _path_parts(path) -> tuple[str, ...]:
    parts: list[str] = []
    for key in path:
        if hasattr(key, "key"):
            parts.append(str(key.key))
        elif hasattr(key, "idx"):
            parts.append(str(key.idx))
        elif hasattr(key, "name"):
            parts.append(str(key.name))
        else:
            parts.append(str(key))
    return tuple(parts)


def runtime_report() -> dict[str, Any]:
    devices = tuple(jax.devices())
    return {
        "jax_version": jax.__version__,
        "platform": devices[0].platform if devices else "none",
        "device_kind": devices[0].device_kind if devices else "none",
        "device_count": len(devices),
        "local_device_count": jax.local_device_count(),
        "process_count": jax.process_count(),
        "process_index": jax.process_index(),
    }


def validate_v5e_runtime(*, require_eight_chips: bool = True) -> tuple[str, ...]:
    """Return human-readable runtime warnings; raise only for hard topology mismatch."""
    report = runtime_report()
    if report["platform"] != "tpu":
        raise RuntimeError(
            f"expected a TPU runtime, got platform={report['platform']!r}; "
            "select TPU in the Kaggle accelerator settings"
        )
    if require_eight_chips and report["device_count"] != V5E.chips_per_host:
        raise RuntimeError(
            f"this notebook is tuned for v5e-8 ({V5E.chips_per_host} chips), "
            f"but JAX sees {report['device_count']} devices"
        )

    warnings: list[str] = []
    kind = str(report["device_kind"]).lower()
    if "v5" not in kind:
        warnings.append(
            f"device_kind={report['device_kind']!r} does not look like v5e; "
            "the code can still run, but the tuning constants are v5e-specific"
        )
    if report["process_count"] != 1:
        warnings.append(
            "v5e-8 is expected to be a single-host target; multi-process JAX changes "
            "input placement and should be treated as a separate experiment"
        )
    return tuple(warnings)


def make_v5e_mesh(
    *,
    devices: tuple[jax.Device, ...] | list[jax.Device] | None = None,
    strict: bool = True,
) -> Mesh:
    """Construct a topology-aware mesh using JAX's physical-device ordering helper.

    On the Kaggle target this is a 2x4 explicit mesh. For one-device CPU CI, `strict=False`
    yields a one-axis mesh so sharding helpers remain executable without pretending the CPU
    runner is a TPU.
    """
    devs = tuple(jax.devices() if devices is None else devices)
    if len(devs) == V5E.chips_per_host:
        return jax.make_mesh(V5E.topology, V5E_AXIS_NAMES, devices=devs)
    if strict:
        raise ValueError(
            f"v5e mesh requires exactly {V5E.chips_per_host} devices, got {len(devs)}"
        )
    if not devs:
        raise ValueError("cannot build a mesh without devices")
    return jax.make_mesh((len(devs),), ("x",), devices=devs)


def axes_for_shard_count(mesh: Mesh, shards: int, *, strict: bool = True):
    """Map a semantic shard count onto one or both physical mesh axes.

    For the v5e 2x4 mesh this gives 2 -> 'x', 4 -> 'y', 8 -> ('x','y'). The exact
    single-axis match prefers the last mesh axis so 4-way DSpark EP naturally lands on y.
    """
    if shards <= 0:
        raise ValueError("shard count must be positive")
    if shards == 1:
        return None

    names = tuple(mesh.axis_names)
    sizes = tuple(int(x) for x in mesh.devices.shape)
    mesh_size = int(mesh.size)
    if shards == mesh_size:
        return names[0] if len(names) == 1 else names

    exact = [name for name, size in zip(names, sizes) if size == shards]
    if exact:
        return exact[-1]

    if strict:
        raise ValueError(
            f"cannot map {shards}-way sharding onto mesh shape {sizes}; "
            "use 1, an exact mesh-axis size, or the full mesh size"
        )
    return None


def semantic_axes(config, mesh: Mesh, *, strict: bool = True) -> dict[str, Any]:
    pc = config.parallelism
    return {
        "vocab": axes_for_shard_count(mesh, pc.vocab_shard, strict=strict),
        "engram": axes_for_shard_count(mesh, pc.engram_table_shard, strict=strict),
        "experts": axes_for_shard_count(mesh, pc.expert_shard, strict=strict),
        "dspark_experts": axes_for_shard_count(
            mesh, pc.dspark_expert_shard, strict=strict
        ),
        "context": axes_for_shard_count(
            mesh, pc.attention_context_shard, strict=strict
        ),
        "heads": axes_for_shard_count(
            mesh, pc.attention_head_shard, strict=strict
        ),
        "indexer_context": axes_for_shard_count(
            mesh, pc.indexer_context_shard, strict=strict
        ),
    }


def _axis_factor(mesh: Mesh, axis) -> int:
    if axis is None:
        return 1
    axis_sizes = {
        name: int(size) for name, size in zip(mesh.axis_names, mesh.devices.shape)
    }
    if isinstance(axis, tuple):
        out = 1
        for name in axis:
            out *= axis_sizes[name]
        return out
    return axis_sizes[axis]


def _validate_dim_sharding(
    value,
    dim: int,
    axis,
    mesh: Mesh,
    *,
    path: tuple[str, ...],
) -> None:
    if axis is None:
        return
    factor = _axis_factor(mesh, axis)
    if int(value.shape[dim]) % factor:
        raise ValueError(
            f"{'/'.join(path)} shape {value.shape}: dimension {dim} is not divisible by "
            f"its {factor}-way sharding"
        )


def parameter_partition_spec(path, value, config, mesh: Mesh, *, strict: bool = True) -> P:
    """Assign capacity/compute sharding by parameter semantics, not one global TP flag."""
    parts = _path_parts(path)
    axes = semantic_axes(config, mesh, strict=strict)
    ndim = int(value.ndim)
    replicated = P(*([None] * ndim))
    if ndim == 0:
        return P()

    def spec_with(dim: int, axis) -> P:
        _validate_dim_sharding(value, dim, axis, mesh, path=parts)
        entries = [None] * ndim
        entries[dim] = axis
        return P(*entries)

    # Vocabulary capacity: row-shard embeddings, column-shard [D,V] prediction matrices.
    if parts and parts[0] == "embed":
        return spec_with(0, axes["vocab"])
    if parts and parts[0] == "lm_head":
        return spec_with(ndim - 1, axes["vocab"])
    if parts[:2] == ("dspark", "markov_embed"):
        return spec_with(0, axes["vocab"])
    if parts[:2] == ("dspark", "markov_head") and parts[-1] == "weight":
        return spec_with(ndim - 1, axes["vocab"])

    # Engram capacity lives in hash-table rows. Projection/gates stay replicated.
    if "engram" in parts and parts[-1] == "table":
        return spec_with(0, axes["engram"])

    # Routed expert stacks are [E,...]. Router logits are sharded over the expert axis.
    if "moe" in parts:
        expert_axis = (
            axes["dspark_experts"] if "dspark" in parts else axes["experts"]
        )
        if "experts" in parts:
            return spec_with(0, expert_axis)
        if parts[-1] == "router_weight":
            return spec_with(ndim - 1, expert_axis)
        if parts[-1] == "router_bias":
            return spec_with(0, expert_axis)

    # Optional attention TP. Nano defaults to 1 so MLA projections stay replicated while
    # context parallelism consumes all eight chips. These rules allow explicit TP ablations.
    head_axis = axes["heads"]
    if head_axis is not None:
        if "q_b" in parts and parts[-1] == "weight":
            return spec_with(ndim - 1, head_axis)
        if parts[-1] == "attn_sink":
            return spec_with(0, head_axis)
        if parts[-1] == "wo_a" and ndim == 3:
            return spec_with(0, head_axis)
        if "wo_b" in parts and parts[-1] == "weight":
            return spec_with(0, head_axis)

    return replicated


def parameter_partition_specs(params, config, mesh: Mesh, *, strict: bool = True):
    items, treedef = jax.tree_util.tree_flatten_with_path(params)
    specs = [
        parameter_partition_spec(path, value, config, mesh, strict=strict)
        for path, value in items
    ]
    return treedef.unflatten(specs)


def named_shardings(specs, mesh: Mesh):
    return jax.tree_util.tree_map(
        lambda spec: NamedSharding(mesh, spec),
        specs,
        is_leaf=lambda x: isinstance(x, P),
    )


def init_model_sharded(key: jax.Array, config, mesh: Mesh):
    """Initialize directly into final shards; never materialize the full model on chip 0."""
    abstract = jax.eval_shape(lambda k: init_model(k, config), key)
    specs = parameter_partition_specs(abstract, config, mesh)
    shardings = named_shardings(specs, mesh)
    key = jax.device_put(key, NamedSharding(mesh, P()))
    init_fn = jax.jit(lambda k: init_model(k, config), out_shardings=shardings)
    return init_fn(key), specs, shardings


def optimizer_state_named_shardings(params, param_specs, config, mesh: Mesh):
    """Place momentum beside each parameter; Adam second moments follow the same shard."""
    p_items, treedef = jax.tree_util.tree_flatten_with_path(params)
    s_items, _ = jax.tree_util.tree_flatten_with_path(
        param_specs, is_leaf=lambda x: isinstance(x, P)
    )
    if len(p_items) != len(s_items):
        raise ValueError("parameter/spec pytrees do not align")

    leaves: list[OptimizerLeafState] = []
    for (path, param), (_, spec) in zip(p_items, s_items):
        if classify_parameter(path, param, config) == "adamw":
            second = NamedSharding(mesh, spec)
        else:
            # Muon/Sinkhorn use an empty sentinel for `second`; keep that tiny leaf
            # replicated instead of inventing a parameter-shaped optimizer allocation.
            second = NamedSharding(mesh, P())
        leaves.append(
            OptimizerLeafState(
                NamedSharding(mesh, spec),
                second,
            )
        )
    return treedef.unflatten(leaves)


def init_optimizer_state_sharded(params, param_specs, config, mesh: Mesh):
    param_shardings = named_shardings(param_specs, mesh)
    state_shardings = optimizer_state_named_shardings(
        params, param_specs, config, mesh
    )
    init_fn = jax.jit(
        lambda p: init_optimizer_state(p, config),
        # A single positional pytree argument still needs a singleton tuple here.
        in_shardings=(param_shardings,),
        out_shardings=state_shardings,
    )
    return init_fn(params), state_shardings


def batch_named_sharding(config, mesh: Mesh, *, strict: bool = True) -> NamedSharding:
    axis = semantic_axes(config, mesh, strict=strict)["context"]
    return NamedSharding(mesh, P(None, axis))


def validate_sequence_length(seq_len: int, config, mesh: Mesh) -> tuple[str, ...]:
    """Hard shape checks plus v5e-specific performance warnings for a packed batch."""
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    axes = semantic_axes(config, mesh)
    context_shards = _axis_factor(mesh, axes["context"])
    if seq_len % context_shards:
        raise ValueError(
            f"seq_len={seq_len} must be divisible by context sharding={context_shards}"
        )
    if (
        config.csa2.context_compression_ratio > 1
        and seq_len % config.csa2.context_compression_ratio
    ):
        raise ValueError("sequence length must be divisible by the r=2 context compressor")

    warnings: list[str] = []
    local_tokens = seq_len // context_shards
    if local_tokens % 128:
        warnings.append(
            f"each context shard has {local_tokens} tokens; 128-token multiples are a "
            "cleaner starting point for v5e MXU/Pallas tiling"
        )
    if config.attention.local_window % 128:
        warnings.append(
            f"local_window={config.attention.local_window} is not a 128-token multiple; "
            "correctness is fine, but the first Pallas kernel should benchmark padded tiles"
        )
    return tuple(warnings)


def put_training_batch(
    input_ids,
    segment_ids,
    token_mask,
    config,
    mesh: Mesh,
):
    if input_ids.shape != segment_ids.shape or input_ids.shape != token_mask.shape:
        raise ValueError(
            "input_ids, segment_ids and token_mask must have identical [B,T] shapes"
        )
    validate_sequence_length(int(input_ids.shape[1]), config, mesh)
    sharding = batch_named_sharding(config, mesh)
    return (
        jax.device_put(input_ids, sharding),
        jax.device_put(segment_ids, sharding),
        jax.device_put(token_mask, sharding),
    )


def compile_pretrain_step(
    params,
    optimizer_state,
    param_specs,
    config,
    train_config,
    mesh: Mesh,
    *,
    include_indexer: bool,
    n_segments: int | None,
):
    """Compile one static base or late-indexer step for the v5e mesh.

    We intentionally compile two executables rather than branch on the training step inside
    XLA. That keeps selected-row teacher work out of the base executable.
    """
    if include_indexer and n_segments is None:
        raise ValueError("n_segments is required for the late-indexer executable")
    param_shardings = named_shardings(param_specs, mesh)
    state_shardings = optimizer_state_named_shardings(
        params, param_specs, config, mesh
    )
    batch_sharding = batch_named_sharding(config, mesh)
    scalar_sharding = NamedSharding(mesh, P())

    def step_fn(p, opt, ids, seg, step, mask):
        return pretrain_step(
            p,
            opt,
            config,
            train_config,
            ids,
            segment_ids=seg,
            step=step,
            token_mask=mask,
            include_indexer=include_indexer,
            n_segments=n_segments,
        )

    return jax.jit(
        step_fn,
        in_shardings=(
            param_shardings,
            state_shardings,
            batch_sharding,
            batch_sharding,
            scalar_sharding,
            batch_sharding,
        ),
        out_shardings=(param_shardings, state_shardings, None),
        donate_argnums=(0, 1),
    )


def global_tree_nbytes(tree) -> int:
    return sum(
        int(x.size) * int(jnp.dtype(x.dtype).itemsize)
        for x in jax.tree_util.tree_leaves(tree)
    )


def per_device_parameter_nbytes(params, specs, mesh: Mesh) -> int:
    p_leaves = jax.tree_util.tree_leaves(params)
    s_leaves = jax.tree_util.tree_leaves(
        specs, is_leaf=lambda x: isinstance(x, P)
    )
    if len(p_leaves) != len(s_leaves):
        raise ValueError("parameter/spec pytrees do not align")
    total = 0
    for value, spec in zip(p_leaves, s_leaves):
        factor = 1
        for axis in spec:
            factor *= _axis_factor(mesh, axis)
        total += (
            int(value.size) * int(jnp.dtype(value.dtype).itemsize)
        ) // factor
    return total


def memory_report(params, specs, mesh: Mesh) -> dict[str, float]:
    global_bytes = global_tree_nbytes(params)
    per_device_bytes = per_device_parameter_nbytes(params, specs, mesh)
    budget = V5E.hbm_gb_per_chip * 1_000_000_000
    return {
        "global_parameter_gb": global_bytes / 1e9,
        "per_device_parameter_gb": per_device_bytes / 1e9,
        "per_device_parameter_hbm_fraction": per_device_bytes / budget,
        "v5e_hbm_budget_gb": V5E.hbm_gb_per_chip,
    }
