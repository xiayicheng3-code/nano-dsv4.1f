from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
import math
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from .csa2 import (
    SharedCSA2State,
    _build_global_state,
    _rope_kwargs,
    grouped_output_projection,
    segment_local_positions,
)
from .layers import linear, rms_norm
from .moe import apply_moe as apply_moe_reference
from .quantization import fake_fp8_e4m3
from .rope import apply_partial_rope
from .splash import _splash_modules


@dataclass(frozen=True)
class TPUNativeConfig:
    """Execution knobs for the v5e-native training backend."""

    use_splash_attention: bool = True
    use_expert_parallel_moe: bool = True
    force_block_remat: bool = True
    moe_ragged_implementation: Literal["auto", "mosaic", "xla"] = "auto"
    # Deprecated compatibility fields; Tokamax ragged dispatch has no capacity cap.
    moe_capacity_factor: float = 1.5
    moe_capacity_multiple: int = 128
    manual_axis_name: str = "tp"
    splash_interpret: bool = False
    need_teacher_lse: bool = False
    # Set from ModelConfig.parallelism by compile_pretrain_step_native.
    attention_data_shards: int = 1
    # Experimental controls. Defaults preserve the measured production path.
    splash_batch_mode: Literal["vmap", "sequential"] = "vmap"
    splash_block_q_dkv: int | None = None

    def __post_init__(self) -> None:
        if self.splash_batch_mode not in ("vmap", "sequential"):
            raise ValueError("splash_batch_mode must be vmap or sequential")
        if self.splash_block_q_dkv is not None and self.splash_block_q_dkv not in (128, 256, 512):
            raise ValueError("experimental splash_block_q_dkv must be 128, 256 or 512")
        if self.moe_ragged_implementation not in ("auto", "mosaic", "xla"):
            raise ValueError("moe_ragged_implementation must be auto, mosaic or xla")
        if self.moe_capacity_factor < 1.0:
            raise ValueError("moe_capacity_factor must be >= 1")
        if self.moe_capacity_multiple <= 0:
            raise ValueError("moe_capacity_multiple must be positive")
        if not self.manual_axis_name:
            raise ValueError("manual_axis_name must be non-empty")
        if self.attention_data_shards <= 0:
            raise ValueError("attention_data_shards must be positive")


@dataclass(frozen=True)
class TPUNativeState:
    mesh: Mesh
    options: TPUNativeConfig


_ACTIVE_NATIVE: ContextVar[TPUNativeState | None] = ContextVar(
    "nano_dsv41f_tpu_native", default=None
)
_MANUAL_MESH_CACHE: dict[tuple[int, str], Mesh] = {}
_SPLASH_RUNNER_CACHE: dict[tuple[Any, ...], Any] = {}


def active_tpu_backend() -> TPUNativeState | None:
    return _ACTIVE_NATIVE.get()


@contextmanager
def tpu_native_context(mesh: Mesh, options: TPUNativeConfig):
    token = _ACTIVE_NATIVE.set(TPUNativeState(mesh=mesh, options=options))
    try:
        yield
    finally:
        _ACTIVE_NATIVE.reset(token)


def manual_v5e_mesh(mesh: Mesh, axis_name: str = "tp") -> Mesh:
    """Flatten the topology-aware mesh into one shard_map-local SPMD axis.

    The flat mesh remains Auto to surrounding JAX.  ``shard_map`` itself owns the manual
    interpretation of ``tp``; this prevents manual-axis annotations from leaking into
    ordinary model matmuls such as the LM head.
    """
    key = (id(mesh), axis_name)
    cached = _MANUAL_MESH_CACHE.get(key)
    if cached is not None:
        return cached
    devices = np.asarray(mesh.devices, dtype=object).reshape(-1)
    manual = Mesh(devices, (axis_name,), axis_types=(AxisType.Auto,))
    _MANUAL_MESH_CACHE[key] = manual
    return manual


def _moe_param_specs(axis_name: str):
    return {
        "router_weight": P(),
        "router_bias": P(),
        "experts": {
            "w1": P(axis_name, None, None),
            "w2": P(axis_name, None, None),
            "w3": P(axis_name, None, None),
        },
        "shared": {"w1": P(), "w2": P(), "w3": P()},
    }


def apply_moe_v5e(
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
    """Compatibility entry point for the unified dropless multi-expert kernel."""
    from .tpu_moe import apply_moe_v5e_multi

    return apply_moe_v5e_multi(
        x, params, top_k=top_k, route_scale=route_scale,
        swiglu_limit=swiglu_limit, eps=eps, state=state, token_mask=token_mask,
    )


def apply_moe_dispatch(
    x: jax.Array,
    params: dict[str, object],
    *,
    top_k: int,
    route_scale: float = 1.0,
    swiglu_limit: float = 7.0,
    eps: float = 1e-20,
    token_mask: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    state = active_tpu_backend()
    if state is not None and state.options.use_expert_parallel_moe:
        return apply_moe_v5e(
            x,
            params,
            top_k=top_k,
            route_scale=route_scale,
            swiglu_limit=swiglu_limit,
            eps=eps,
            state=state,
            token_mask=token_mask,
        )
    return apply_moe_reference(
        x,
        params,
        top_k=top_k,
        route_scale=route_scale,
        swiglu_limit=swiglu_limit,
        eps=eps,
        token_mask=token_mask,
    )


def combined_csa2_mask(
    seq_len: int,
    *,
    local_window: int,
    global_kv_len: int = 0,
    compression_ratio: int = 1,
    padded_global_kv_len: int | None = None,
) -> np.ndarray:
    """Static local+compressed-global mask consumed by one Splash softmax."""
    if min(seq_len, local_window) <= 0:
        raise ValueError("sequence length and local window must be positive")
    q = np.arange(seq_len, dtype=np.int32)[:, None]
    local_k = np.arange(seq_len, dtype=np.int32)[None, :]
    local = (local_k <= q) & (local_k >= q - (local_window - 1))
    if global_kv_len <= 0:
        return local
    if compression_ratio <= 0:
        raise ValueError("compression_ratio must be positive")
    padded = global_kv_len if padded_global_kv_len is None else padded_global_kv_len
    if padded < global_kv_len:
        raise ValueError("padded global KV length cannot shrink the real KV")
    global_k = np.arange(global_kv_len, dtype=np.int32)[None, :] * compression_ratio
    global_mask = global_k <= q - local_window
    if padded > global_kv_len:
        global_mask = np.pad(
            global_mask,
            ((0, 0), (0, padded - global_kv_len)),
            constant_values=False,
        )
    return np.concatenate((local, global_mask), axis=1)


def _round_up(value: int, multiple: int) -> int:
    return math.ceil(value / multiple) * multiple


def _splash_runner(
    state: TPUNativeState,
    *,
    seq_len: int,
    kv_len: int,
    n_heads: int,
    head_dim: int,
    mask_array: np.ndarray,
    save_residuals: bool,
):
    axis = state.options.manual_axis_name
    dp = state.options.attention_data_shards
    if int(state.mesh.size) % dp:
        raise ValueError("attention DP must divide the device mesh")
    shards = int(state.mesh.size) // dp
    if dp == 1:
        manual = manual_v5e_mesh(state.mesh, axis)
        data_axis = None
    else:
        from .tpu import axes_for_shard_count
        manual = state.mesh
        axis = axes_for_shard_count(manual, shards)
        data_axis = axes_for_shard_count(manual, dp)
        if axis == data_axis:
            raise ValueError("attention CP and DP require distinct mesh axes")
    if seq_len % shards:
        raise ValueError("Splash query length must divide evenly over context shards")
    if (seq_len // shards) % 128:
        raise ValueError(
            f"TPU Splash needs 128-row Q tiles per shard; got {seq_len // shards}"
        )

    cache_key = (
        id(state.mesh),
        axis,
        data_axis,
        seq_len,
        kv_len,
        n_heads,
        head_dim,
        hash(mask_array.tobytes()),
        save_residuals,
        state.options.splash_interpret,
        state.options.splash_batch_mode,
        state.options.splash_block_q_dkv,
    )
    cached = _SPLASH_RUNNER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    splash, mask_lib = _splash_modules()
    base_mask = mask_lib.NumpyMask(mask_array.astype(np.bool_, copy=False))
    mask = mask_lib.MultiHeadMask(tuple(base_mask for _ in range(n_heads)))
    kernel = splash.make_splash_mqa(
        mask,
        head_shards=1,
        q_seq_shards=shards,
        save_residuals=save_residuals,
        interpret=state.options.splash_interpret,
        **({"block_sizes": replace(splash.BlockSizes.get_default(),
                                  block_q_dkv=state.options.splash_block_q_dkv)}
           if state.options.splash_block_q_dkv is not None else {}),
    )
    kernel_spec = kernel.manual_sharding_spec(NamedSharding(manual, P(None, axis)))
    q_spec = P(data_axis, None, axis, None)
    kv_spec = P(data_axis, None, None)
    q_segment_spec = P(data_axis, axis)
    kv_segment_spec = P(data_axis, None)
    lse_spec = P(data_axis, None, axis)
    out_specs = (q_spec, lse_spec) if save_residuals else q_spec

    @jax.shard_map(
        mesh=manual,
        in_specs=(
            kernel_spec,
            q_spec,
            kv_spec,
            kv_spec,
            q_segment_spec,
            kv_segment_spec,
            P(),
        ),
        out_specs=out_specs,
        check_vma=False,
    )
    def _mapped(kernel_arg, q_bhtd, k_btd, v_btd, q_segments, kv_segments, sinks):
        sink_dtype = sinks.dtype

        def raw_one(row_kernel, qh, k, v, row_sinks, q_seg, kv_seg):
            result = row_kernel(
                qh * jnp.asarray(head_dim**-0.5, dtype=qh.dtype),
                k,
                v,
                segment_ids=splash.SegmentIds(q=q_seg, kv=kv_seg),
                sinks=row_sinks,
            )
            if save_residuals:
                out, (lse,) = result
                return out, lse
            return result

        # JAX 0.10.2 Splash returns dsinks in the attention output dtype (BF16),
        # even for FP32 sink inputs. A scan transpose must accumulate it into an
        # FP32 carry and rejects the mismatched type. Normalize the cotangent at
        # the custom-VJP boundary, before scan/vmap perform their reductions.
        # Forward sink values and the Splash kernel are unchanged. Apply this to
        # both schedules so the experiment uses the same derivative contract.
        @jax.custom_vjp
        def typed_one(*args):
            return raw_one(*args)

        def typed_fwd(*args):
            result, pullback = jax.vjp(raw_one, *args)
            return result, pullback

        def typed_bwd(pullback, cotangent):
            grads = list(pullback(cotangent))
            grads[4] = grads[4].astype(sink_dtype)
            return tuple(grads)

        typed_one.defvjp(typed_fwd, typed_bwd)

        def one(qh, k, v, q_seg, kv_seg):
            # Pass the kernel pytree explicitly: closing over its traced mask
            # arrays can leak tracers when the full model is rematerialized.
            return typed_one(kernel_arg, qh, k, v, sinks, q_seg, kv_seg)

        inputs = (q_bhtd, k_btd, v_btd, q_segments, kv_segments)
        if state.options.splash_batch_mode == "sequential":
            return jax.lax.map(lambda xs: one(*xs), inputs)
        return jax.vmap(one)(*inputs)

    def apply(q, kv, q_segments, kv_segments, sinks, *, value=None):
        if q.shape[0] % dp:
            raise ValueError("Splash batch rows must be divisible by attention DP")
        q_bhtd = jnp.swapaxes(q, 1, 2)
        # Production uses tied K/V. The replay can also check their separate VJPs.
        result = _mapped(kernel, q_bhtd, kv, kv if value is None else value,
                         q_segments, kv_segments, sinks)
        if save_residuals:
            out_bhtd, lse_bht = result
            return jnp.swapaxes(out_bhtd, 1, 2), jnp.swapaxes(lse_bht, 1, 2)
        return jnp.swapaxes(result, 1, 2)

    _SPLASH_RUNNER_CACHE[cache_key] = apply
    return apply


def _combined_splash_attention(
    q: jax.Array,
    local_kv: jax.Array,
    segment_ids: jax.Array,
    global_state: SharedCSA2State | None,
    *,
    compression_ratio: int,
    local_window: int,
    sink: jax.Array | None,
    state: TPUNativeState,
) -> tuple[jax.Array, jax.Array | None]:
    seq_len = int(q.shape[1])
    if global_state is None:
        kv = local_kv
        kv_segments = segment_ids
        mask = combined_csa2_mask(seq_len, local_window=local_window)
    else:
        real_global = int(global_state.kv.shape[1])
        padded_global = _round_up(real_global, 128)
        pad = padded_global - real_global
        global_kv = global_state.kv
        global_segments = global_state.segment_ids
        if pad:
            global_kv = jnp.pad(global_kv, ((0, 0), (0, pad), (0, 0)))
            global_segments = jnp.pad(
                global_segments,
                ((0, 0), (0, pad)),
                constant_values=-1,
            )
        kv = jnp.concatenate((local_kv, global_kv), axis=1)
        kv_segments = jnp.concatenate((segment_ids, global_segments), axis=1)
        mask = combined_csa2_mask(
            seq_len,
            local_window=local_window,
            global_kv_len=real_global,
            compression_ratio=compression_ratio,
            padded_global_kv_len=padded_global,
        )

    sinks = (
        jnp.asarray(sink, dtype=jnp.float32)
        if sink is not None
        else jnp.full((q.shape[-2],), -1e30, dtype=jnp.float32)
    )
    runner = _splash_runner(
        state,
        seq_len=seq_len,
        kv_len=int(kv.shape[1]),
        n_heads=int(q.shape[-2]),
        head_dim=int(q.shape[-1]),
        mask_array=mask,
        save_residuals=False,
    )
    out = runner(q, kv, segment_ids, kv_segments, sinks)

    # Late distillation reconstructs the denominator only for selected query rows.
    # Running Splash again for every query just to obtain LSE doubled forward work.
    return out, None


def apply_csa2_attention_v5e(
    x: jax.Array,
    segment_ids: jax.Array,
    params: dict[str, object],
    shared_state: SharedCSA2State | None,
    *,
    layer_id: int,
    mode: Literal["swa", "full", "reindex", "reuse"],
    owns_global_kv: bool,
    compression_ratio: int,
    config,
    global_source: jax.Array | None = None,
    compute_indexer: bool = True,
    state: TPUNativeState,
):
    if compute_indexer:
        from .csa2 import apply_csa2_attention as reference_attention

        return reference_attention(
            x,
            segment_ids,
            params,
            shared_state,
            layer_id=layer_id,
            mode=mode,
            owns_global_kv=owns_global_kv,
            compression_ratio=compression_ratio,
            config=config,
            global_source=global_source,
            compute_indexer=compute_indexer,
        )

    ac = config.attention
    q_pos = segment_local_positions(segment_ids)
    rope_kwargs = _rope_kwargs(ac, compression_ratio)
    qr = rms_norm(linear(x, params["q_a"]), params["q_norm"], eps=config.norm_eps)
    q = linear(qr, params["q_b"]).reshape(*x.shape[:-1], ac.n_heads, ac.head_dim)
    q = apply_partial_rope(
        q,
        q_pos,
        rotary_dim=ac.rope.rope_head_dim,
        **rope_kwargs,
    )

    local_kv = rms_norm(
        linear(x, params["local_kv"]), params["local_kv_norm"], eps=config.norm_eps
    )
    local_kv = apply_partial_rope(
        local_kv,
        q_pos,
        rotary_dim=ac.rope.rope_head_dim,
        **rope_kwargs,
    )
    if config.quantization.swa_fp8_qat:
        local_kv = fake_fp8_e4m3(
            local_kv,
            block_size=config.quantization.swa_fp8_block_size,
            ste=True,
        )

    global_valid = None
    if mode == "swa":
        if owns_global_kv or global_source is not None:
            raise ValueError("SWA-only layer cannot own/take compressed global KV")
        active_global = None
    else:
        if owns_global_kv:
            source = x if global_source is None else global_source
            if source.shape != x.shape:
                raise ValueError("global_source must match x [batch,tokens,dim]")
            shared_state = _build_global_state(
                source,
                segment_ids,
                params,
                compression_ratio=compression_ratio,
                source_layer=layer_id,
                build_index_k_cache=False,
                config=config,
            )
        elif global_source is not None:
            raise ValueError("global_source only applies to a compressed-KV source")
        if shared_state is None:
            raise ValueError(f"CSA2 {mode} layer {layer_id} has no shared compressed state")
        active_global = shared_state
        # Do not materialize [B,T,K] for a loss that uses only [B,Q,K].

    merged, total_lse = _combined_splash_attention(
        q,
        local_kv,
        segment_ids,
        active_global,
        compression_ratio=max(compression_ratio, 1),
        local_window=ac.local_window,
        sink=params.get("attn_sink"),
        state=state,
    )
    merged = apply_partial_rope(
        merged,
        q_pos,
        rotary_dim=ac.rope.rope_head_dim,
        inverse=True,
        **rope_kwargs,
    )
    out = grouped_output_projection(
        merged,
        params,
        n_heads=ac.n_heads,
        n_groups=ac.o_groups,
        o_rank=ac.o_rank,
    )

    index_aux = {
        "index_scores": None,
        "index_topk_indices": None
        if shared_state is None
        else shared_state.latest_topk_indices,
        "index_topk_values": None
        if shared_state is None
        else shared_state.latest_topk_values,
        "index_candidate_mask": None
        if shared_state is None
        else shared_state.candidate_mask,
    }
    active_state = None if mode == "swa" else shared_state
    return out, shared_state, {
        "qr": qr,
        "q": q,
        "index_hidden": x,
        "total_lse": total_lse,
        "global_lse": None,
        "main_kv": None if active_state is None else active_state.kv,
        "compressed_latent": None if active_state is None else active_state.latent,
        "global_positions": None
        if active_state is None
        else active_state.group_start_positions,
        "global_segment_ids": None
        if active_state is None
        else active_state.segment_ids,
        "global_valid": global_valid,
        "local_kv": local_kv if state.options.need_teacher_lse else None,
        "attn_sink": params.get("attn_sink"),
        "index_k": None if active_state is None else active_state.index_k,
        **index_aux,
    }


def apply_csa2_attention_dispatch(*args, **kwargs):
    state = active_tpu_backend()
    if state is not None and state.options.use_splash_attention:
        return apply_csa2_attention_v5e(*args, **kwargs, state=state)
    from .csa2 import apply_csa2_attention as reference_attention

    return reference_attention(*args, **kwargs)


def install_model_dispatch() -> None:
    """Install context-sensitive native/reference call targets into ``model`` once."""
    from . import model as model_module

    if model_module.apply_moe is not apply_moe_dispatch:
        model_module.apply_moe = apply_moe_dispatch
    if model_module.apply_csa2_attention is not apply_csa2_attention_dispatch:
        model_module.apply_csa2_attention = apply_csa2_attention_dispatch


class _NativeLowered:
    def __init__(self, lowered, mesh: Mesh, options: TPUNativeConfig):
        self._lowered = lowered
        self._mesh = mesh
        self._options = options

    def compile(self, *args, **kwargs):
        with tpu_native_context(self._mesh, self._options):
            return self._lowered.compile(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._lowered, name)


class NativeCompiledStep:
    """Jitted step wrapper that activates native dispatch while tracing/executing."""

    def __init__(self, jitted, mesh: Mesh, options: TPUNativeConfig):
        self._jitted = jitted
        self._mesh = mesh
        self._options = options

    def __call__(self, *args, **kwargs):
        with tpu_native_context(self._mesh, self._options):
            return self._jitted(*args, **kwargs)

    def lower(self, *args, **kwargs):
        with tpu_native_context(self._mesh, self._options):
            lowered = self._jitted.lower(*args, **kwargs)
        return _NativeLowered(lowered, self._mesh, self._options)

    def __getattr__(self, name):
        return getattr(self._jitted, name)


def compile_pretrain_step_native(
    params,
    optimizer_state,
    param_specs,
    config,
    train_config,
    mesh: Mesh,
    *,
    include_indexer: bool,
    n_segments: int | None = None,
    native_config: TPUNativeConfig | None = None,
):
    """Build the ordinary static pretrain executable under the v5e-native backend."""
    from .config import RematConfig
    from .tpu import compile_pretrain_step as compile_reference_step

    options = TPUNativeConfig() if native_config is None else native_config
    pc = config.parallelism
    if options.use_splash_attention:
        if pc.attention_head_shard != 1:
            raise ValueError("native Splash currently supports attention_head_shard=1")
        if pc.attention_context_shard * pc.attention_data_shard != int(mesh.size):
            raise ValueError("native attention requires CP * DP = mesh size")
    options = replace(options, attention_data_shards=pc.attention_data_shard)
    install_model_dispatch()
    options = replace(options, need_teacher_lse=bool(include_indexer))
    effective_config = config
    if options.force_block_remat and config.remat.policy != "block":
        effective_config = replace(config, remat=RematConfig(policy="block"))

    jitted = compile_reference_step(
        params,
        optimizer_state,
        param_specs,
        effective_config,
        train_config,
        mesh,
        include_indexer=include_indexer,
        n_segments=n_segments,
    )
    return NativeCompiledStep(jitted, mesh, options)
