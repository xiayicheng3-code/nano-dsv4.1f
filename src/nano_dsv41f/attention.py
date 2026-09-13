from __future__ import annotations

import jax
import jax.numpy as jnp


def compressed_global_mask(
    q_ids: jnp.ndarray,
    kv_ids: jnp.ndarray,
    q_segment_ids: jnp.ndarray,
    kv_segment_ids: jnp.ndarray,
    *,
    compression_ratio: int,
) -> jnp.ndarray:
    """Reference mask for packed rectangular global attention.

    This intentionally materializes only for tests/reference code. The TPU path should
    compute the same predicate tile-wise inside Splash/Pallas:

        same_segment & (q_id >= r * kv_id)

    The relation remains valid globally when every packed segment is padded to a multiple
    of `r` and Q/K spans are cropped so each segment satisfies Q_len == r * K_len.
    """
    q_ids = jnp.asarray(q_ids)
    kv_ids = jnp.asarray(kv_ids)
    same_segment = q_segment_ids[:, None] == kv_segment_ids[None, :]
    causal = q_ids[:, None] >= compression_ratio * kv_ids[None, :]
    return same_segment & causal


def alternating_local_window_mask(
    q_ids: jnp.ndarray,
    kv_ids: jnp.ndarray,
    q_segment_ids: jnp.ndarray,
    kv_segment_ids: jnp.ndarray,
    *,
    base_window: int = 128,
    compression_ratio: int = 2,
) -> jnp.ndarray:
    """Disjoint local side of the educational dense CSA2 approximation.

    For r=2 and a 128-token crop, local coverage alternates between 127 and 128 tokens
    so the boundary always lies on a compression-group boundary. This is *not* exact
    DeepSeek semantics: the released model keeps a fixed SWA window and permits overlap
    between raw-local and compressed representations.
    """
    if compression_ratio == 1:
        lower = q_ids - (base_window - 1)
    elif compression_ratio == 2:
        # Even q: 127 tokens; odd q: 128 tokens.
        lower = q_ids - (base_window - 2) - (q_ids & 1)
    else:
        raise ValueError("reference implementation currently supports r in {1, 2}")

    same_segment = q_segment_ids[:, None] == kv_segment_ids[None, :]
    causal = kv_ids[None, :] <= q_ids[:, None]
    in_window = kv_ids[None, :] >= lower[:, None]
    return same_segment & causal & in_window


def merge_attention_outputs(
    local_out: jnp.ndarray,
    local_lse: jnp.ndarray,
    global_out: jnp.ndarray,
    global_lse: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Exactly merge two attention domains that share a softmax denominator.

    `*_out` are already normalized within their own domains. `*_lse` are corresponding
    log-sum-exp values. Shapes may have trailing singleton dimensions; broadcasting is
    intentional.
    """
    total_lse = jnp.logaddexp(local_lse, global_lse)
    local_weight = jnp.exp(local_lse - total_lse)
    global_weight = jnp.exp(global_lse - total_lse)
    while local_weight.ndim < local_out.ndim:
        local_weight = local_weight[..., None]
        global_weight = global_weight[..., None]
    out = local_weight * local_out + global_weight * global_out
    return out, total_lse


def dense_attention_reference(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    mask: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Small CPU-test reference; never intended for long-context training."""
    scale = q.shape[-1] ** -0.5
    logits = jnp.einsum("qhd,khd->hqk", q, k) * scale
    masked = jnp.where(mask[None, :, :], logits, -jnp.inf)
    lse = jax.nn.logsumexp(masked, axis=-1)
    probs = jax.nn.softmax(masked, axis=-1)
    out = jnp.einsum("hqk,khd->qhd", probs, v)
    return out, jnp.swapaxes(lse, 0, 1)
