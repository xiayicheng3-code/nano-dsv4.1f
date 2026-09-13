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

    With every packed segment padded to a multiple of `r` and the 128-token global-Q
    crop paired with a `128 / r` tail crop of compressed KV, each segment satisfies
    `Q_len == r * K_len`. For r=2, this causal boundary can overlap the fixed 128-token
    local window by one raw token on alternating query positions. That overlap is kept
    intentionally because the released model uses fixed-width SWA rather than a disjoint
    local/global partition.
    """
    q_ids = jnp.asarray(q_ids)
    kv_ids = jnp.asarray(kv_ids)
    same_segment = q_segment_ids[:, None] == kv_segment_ids[None, :]
    causal = q_ids[:, None] >= compression_ratio * kv_ids[None, :]
    return same_segment & causal


def fixed_local_window_mask(
    q_ids: jnp.ndarray,
    kv_ids: jnp.ndarray,
    q_segment_ids: jnp.ndarray,
    kv_segment_ids: jnp.ndarray,
    *,
    window: int = 128,
) -> jnp.ndarray:
    """Fixed-width causal SWA reference mask.

    This deliberately permits representation overlap with compressed/global attention.
    For r=2, a compression group that straddles the local/global boundary may represent a
    raw token that is also present in the 128-token SWA branch. We preserve that behavior
    rather than alternating between 127/128 local tokens merely to make the domains
    disjoint.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    same_segment = q_segment_ids[:, None] == kv_segment_ids[None, :]
    causal = kv_ids[None, :] <= q_ids[:, None]
    in_window = kv_ids[None, :] >= (q_ids[:, None] - (window - 1))
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

    If one historical region is represented in both domains, the two representations are
    still distinct KV entries and both participate in the shared denominator. This matches
    the intended fixed-SWA + compressed-global reference semantics.
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
