from __future__ import annotations

import jax
import jax.numpy as jnp


def eligibility_position(
    *,
    local_window: int,
    retrieve_top_k: int,
    compression_ratio: int,
    rule: str,
) -> int:
    """Minimum segment-local query position used by the educational warmup.

    ``local_plus_topk`` preserves the originally proposed 128+512=640 rule. The
    ratio-aware variant waits until the compressed global bank can actually contain more
    than Top-K candidates: 128 + r*512 (1152 at r=2).
    """
    if local_window <= 0 or retrieve_top_k <= 0 or compression_ratio <= 0:
        raise ValueError("window/top-k/compression ratio must be positive")
    if rule == "local_plus_topk":
        return local_window + retrieve_top_k
    if rule == "local_plus_ratio_topk":
        return local_window + compression_ratio * retrieve_top_k
    raise ValueError(f"unknown eligibility rule: {rule}")


def latest_teacher_indices(
    segment_ids: jnp.ndarray,
    *,
    n_segments: int,
    min_local_position: int | None = None,
    local_window: int | None = None,
    retrieve_top_k: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return one fixed-shape latest eligible teacher query per packed segment.

    `min_local_position` is the preferred explicit threshold. The window/top-k pair is kept
    for backwards compatibility with the original 640-history helper.
    """
    if segment_ids.ndim != 1:
        raise ValueError("segment_ids must be rank-1")
    if n_segments <= 0:
        raise ValueError("n_segments must be positive")
    if min_local_position is None:
        if local_window is None or retrieve_top_k is None:
            raise ValueError(
                "provide min_local_position or both local_window and retrieve_top_k"
            )
        min_local_position = local_window + retrieve_top_k

    pos = jnp.arange(segment_ids.shape[0], dtype=jnp.int32)
    starts = jnp.concatenate(
        [jnp.array([True]), segment_ids[1:] != segment_ids[:-1]]
    )
    segment_start = jnp.stack(
        [
            jnp.max(jnp.where((segment_ids == s) & starts, pos, -1))
            for s in range(n_segments)
        ]
    )
    segment_end = jnp.stack(
        [
            jnp.max(jnp.where(segment_ids == s, pos, -1))
            for s in range(n_segments)
        ]
    )
    local_end = segment_end - segment_start
    valid = (segment_start >= 0) & (local_end >= min_local_position)
    return jnp.where(valid, segment_end, 0), valid


def latest_teacher_indices_batched(
    segment_ids: jnp.ndarray,
    *,
    n_segments: int,
    min_local_position: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Batch-vmap the fixed-shape selector, returning [B, n_segments]."""
    if segment_ids.ndim != 2:
        raise ValueError("segment_ids must be [batch,tokens]")
    return jax.vmap(
        lambda row: latest_teacher_indices(
            row,
            n_segments=n_segments,
            min_local_position=min_local_position,
        )
    )(segment_ids)


def dense_teacher_mass(
    q: jnp.ndarray,
    k: jnp.ndarray,
    total_lse: jnp.ndarray,
    valid_k: jnp.ndarray,
) -> jnp.ndarray:
    """Teacher attention mass over selected queries and compressed candidates.

    Shapes may be unbatched or batched:
      q: [..., Q, H, D]
      k: [..., K, D]
      total_lse: [..., Q, H] from complete local + global + sink denominator
      valid_k: [..., Q, K]

    Returns [..., Q, K], summing the main-attention probability mass across Q heads.
    """
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and shared latent k must use the same head dimension")
    if q.shape[:-3] != k.shape[:-2]:
        raise ValueError("q and k batch-prefix dimensions must match")
    scale = q.shape[-1] ** -0.5
    logits = jnp.einsum("...qhd,...kd->...qhk", q, k) * scale
    probs = jnp.exp(logits - total_lse[..., None])
    probs = jnp.where(valid_k[..., :, None, :], probs, 0.0)
    return jnp.sum(probs, axis=-2)


def indexer_cross_entropy_from_mass(
    index_scores: jnp.ndarray,
    teacher_mass: jnp.ndarray,
    valid_k: jnp.ndarray,
    query_valid: jnp.ndarray | None = None,
    eps: float = 1e-9,
) -> jnp.ndarray:
    """Cross entropy against normalized teacher mass without materializing normalization.

    Empty fixed-shape teacher slots are replaced with a harmless finite row before
    ``logsumexp``. This matters for JAX autodiff: computing an all-``-inf`` logsumexp and
    masking it afterwards can still leak NaN gradients from an otherwise invalid slot.
    """
    if index_scores.shape != teacher_mass.shape or index_scores.shape != valid_k.shape:
        raise ValueError("student scores, teacher mass and valid_k must have same shape")
    row_has_key = jnp.any(valid_k, axis=-1)
    masked_scores = jnp.where(valid_k, index_scores, -jnp.inf)
    safe_scores = jnp.where(row_has_key[..., None], masked_scores, 0.0)
    mass = jnp.where(valid_k, teacher_mass, 0.0)
    mass_sum = jnp.sum(mass, axis=-1)
    weighted_score = jnp.sum(
        mass * jnp.where(valid_k, index_scores, 0.0), axis=-1
    )
    per_query = jax.nn.logsumexp(safe_scores, axis=-1) - weighted_score / jnp.maximum(
        mass_sum, eps
    )
    finite_teacher = (mass_sum > eps) & row_has_key
    valid = finite_teacher if query_valid is None else (finite_teacher & query_valid)
    denom = jnp.maximum(jnp.sum(valid), 1)
    return jnp.sum(jnp.where(valid, per_query, 0.0)) / denom


def mean_all_served_teacher_mass(
    teacher_masses: tuple[jnp.ndarray, ...],
) -> jnp.ndarray:
    """Aggregate all served-layer teachers in a small retrieval-sharing group."""
    if not teacher_masses:
        raise ValueError("at least one served-layer teacher is required")
    return jnp.mean(jnp.stack(teacher_masses, axis=0), axis=0)


def indexer_stage_active(
    step: jnp.ndarray,
    total_steps: int,
    start_fraction: float,
    end_fraction: float,
) -> jnp.ndarray:
    progress = step.astype(jnp.float32) / float(total_steps)
    return (progress >= start_fraction) & (progress <= end_fraction)
