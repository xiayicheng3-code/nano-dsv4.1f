from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P


def _splash_modules():
    """Import experimental TPU Splash lazily so CPU/reference users do not depend on it."""
    from jax.experimental.pallas.ops.tpu.splash_attention import (  # noqa: PLC0415
        splash_attention_kernel as splash,
    )
    from jax.experimental.pallas.ops.tpu.splash_attention import (  # noqa: PLC0415
        splash_attention_mask as mask_lib,
    )

    return splash, mask_lib


def _axis_factor(mesh, axis) -> int:
    sizes = {name: int(size) for name, size in zip(mesh.axis_names, mesh.devices.shape)}
    if axis is None:
        return 1
    if isinstance(axis, tuple):
        out = 1
        for name in axis:
            out *= sizes[name]
        return out
    return sizes[axis]


def _default_sequence_axis(mesh):
    names = tuple(mesh.axis_names)
    if not names:
        return None
    return names[0] if len(names) == 1 else names


def dense_local_mqa_reference(
    q: jax.Array,
    kv: jax.Array,
    segment_ids: jax.Array,
    *,
    local_window: int,
) -> tuple[jax.Array, jax.Array]:
    """Dense local MQA reference with model-layout q=[B,T,H,D], kv=[B,T,D]."""
    if q.ndim != 4 or kv.ndim != 3 or segment_ids.ndim != 2:
        raise ValueError("expected q=[B,T,H,D], kv=[B,T,D], segment_ids=[B,T]")
    if q.shape[:2] != kv.shape[:2] or q.shape[:2] != segment_ids.shape:
        raise ValueError("batch/token dimensions must match")
    if q.shape[-1] != kv.shape[-1]:
        raise ValueError("Q and latent KV head dimensions must match")
    if local_window <= 0:
        raise ValueError("local_window must be positive")

    t = q.shape[1]
    pos = jnp.arange(t, dtype=jnp.int32)
    causal_local = (
        (pos[None, :] <= pos[:, None])
        & (pos[None, :] >= pos[:, None] - (local_window - 1))
    )
    same_segment = segment_ids[:, :, None] == segment_ids[:, None, :]
    valid = same_segment & causal_local[None, :, :]

    # BF16 operands hit the low-precision MXU path; logits/softmax stay FP32.
    logits = jnp.einsum("bthd,bsd->bhts", q, kv).astype(jnp.float32)
    logits = logits * jnp.float32(q.shape[-1] ** -0.5)
    masked = jnp.where(valid[:, None, :, :], logits, jnp.float32(-1e30))
    probs = jax.nn.softmax(masked, axis=-1)
    out = jnp.einsum(
        "bhts,bsd->bthd", probs.astype(kv.dtype), kv
    ).astype(q.dtype)
    lse = jax.nn.logsumexp(masked, axis=-1)
    return out, jnp.swapaxes(lse, 1, 2)


def make_v5e_sharded_local_mqa(
    mesh,
    *,
    seq_len: int,
    n_heads: int,
    head_dim: int,
    local_window: int,
    sequence_axis: Any | None = None,
    block_sizes=None,
    interpret: bool = False,
):
    """Build packed-safe local Splash MQA with sequence-sharded Q and replicated latent KV.

    Public callable layout:
      q:           [B,T,H,D]  (T sharded over `sequence_axis`)
      kv:          [B,T,D]    (resharded/replicated for the Splash call)
      segment_ids: [B,T]
      returns:     out [B,T,H,D], lse [B,T,H]

    The full latent KV replication is deliberate for MLA: it trades a small compact-KV
    all-gather for perfectly balanced Q-row compute. This prototype only replaces the local
    branch; the asymmetric compressed-global branch remains dense until its zero-row/cropping
    semantics are handled separately.
    """
    if min(seq_len, n_heads, head_dim, local_window) <= 0:
        raise ValueError("Splash dimensions/window must be positive")
    if local_window > seq_len:
        raise ValueError("local_window cannot exceed seq_len")

    splash, mask_lib = _splash_modules()
    seq_axis = _default_sequence_axis(mesh) if sequence_axis is None else sequence_axis
    q_seq_shards = _axis_factor(mesh, seq_axis)
    if seq_len % q_seq_shards:
        raise ValueError(
            f"seq_len={seq_len} must divide evenly over q_seq_shards={q_seq_shards}"
        )

    base_mask = mask_lib.LocalMask(
        shape=(seq_len, seq_len),
        window_size=(local_window - 1, 0),
        offset=0,
    )
    mask = mask_lib.MultiHeadMask(tuple(base_mask for _ in range(n_heads)))
    kwargs = dict(
        head_shards=1,
        q_seq_shards=q_seq_shards,
        save_residuals=True,
        interpret=interpret,
    )
    if block_sizes is not None:
        kwargs["block_sizes"] = block_sizes
    kernel = splash.make_splash_mqa(mask, **kwargs)

    # `manual_sharding_spec` describes the preprocessed static mask/kernel pytree. Its two
    # axes are [heads, q_sequence]. K/V sequence sharding is intentionally unsupported by
    # Splash's manual partitioner and therefore replicated below.
    kernel_spec = kernel.manual_sharding_spec(
        NamedSharding(mesh, P(None, seq_axis))
    )
    q_spec = P(None, None, seq_axis, None)  # [B,H,T,D]
    kv_spec = P()  # [B,T,D] replicated on every chip
    q_segment_spec = P(None, seq_axis)
    kv_segment_spec = P()
    lse_spec = P(None, None, seq_axis)  # [B,H,T]

    @jax.shard_map(
        mesh=mesh,
        in_specs=(
            kernel_spec,
            q_spec,
            kv_spec,
            kv_spec,
            q_segment_spec,
            kv_segment_spec,
        ),
        out_specs=(q_spec, lse_spec),
        check_vma=False,
    )
    def _mapped(kernel_arg, q_bhtd, k_btd, v_btd, q_segments, kv_segments):
        scale = jnp.asarray(head_dim**-0.5, dtype=q_bhtd.dtype)

        def one(qh, k, v, q_seg, kv_seg):
            out, (lse,) = kernel_arg(
                qh * scale,
                k,
                v,
                segment_ids=splash.SegmentIds(q=q_seg, kv=kv_seg),
            )
            return out, lse

        return jax.vmap(one)(q_bhtd, k_btd, v_btd, q_segments, kv_segments)

    def apply(q: jax.Array, kv: jax.Array, segment_ids: jax.Array):
        if q.shape != (q.shape[0], seq_len, n_heads, head_dim):
            raise ValueError(
                f"q must have [B,{seq_len},{n_heads},{head_dim}], got {q.shape}"
            )
        if kv.shape != (q.shape[0], seq_len, head_dim):
            raise ValueError(
                f"kv must have [B,{seq_len},{head_dim}], got {kv.shape}"
            )
        if segment_ids.shape != (q.shape[0], seq_len):
            raise ValueError("segment_ids must match q's [B,T] prefix")
        q_bhtd = jnp.swapaxes(q, 1, 2)
        out_bhtd, lse_bht = _mapped(
            kernel,
            q_bhtd,
            kv,
            kv,
            segment_ids,
            segment_ids,
        )
        return jnp.swapaxes(out_bhtd, 1, 2), jnp.swapaxes(lse_bht, 1, 2)

    return apply
