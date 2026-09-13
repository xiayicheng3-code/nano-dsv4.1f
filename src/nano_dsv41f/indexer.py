from __future__ import annotations

import jax
import jax.numpy as jnp


def latest_teacher_indices(
    segment_ids: jnp.ndarray,
    *,
    n_segments: int,
    local_window: int,
    retrieve_top_k: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return one fixed-shape latest eligible teacher query per packed segment."""
    if segment_ids.ndim != 1:
        raise ValueError("segment_ids must be rank-1")
    if n_segments <= 0:
        raise ValueError("n_segments must be positive")

    pos = jnp.arange(segment_ids.shape[0], dtype=jnp.int32)
    starts = jnp.concatenate([jnp.array([True]), segment_ids[1:] != segment_ids[:-1]])
    segment_start = jnp.stack(
        [jnp.max(jnp.where((segment_ids == s) & starts, pos, -1)) for s in range(n_segments)]
    )
    segment_end = jnp.stack(
        [jnp.max(jnp.where(segment_ids == s, pos, -1)) for s in range(n_segments)]
    )
    local_end = segment_end - segment_start
    valid = (segment_start >= 0) & (local_end >= (local_window + retrieve_top_k))
    return jnp.where(valid, segment_end, 0), valid


def dense_teacher_mass(
    q: jnp.ndarray,
    k: jnp.ndarray,
    total_lse: jnp.ndarray,
    valid_k: jnp.ndarray,
) -> jnp.ndarray:
    """Teacher attention mass over compressed/global candidates.

    Args:
      q: [Q,H,D] already-RoPE'd main attention queries.
      k: [K,D] shared MLA latent K (also already RoPE'd / fake-quantized as configured).
      total_lse: [Q,H] LSE from the complete local + global + sink denominator.
      valid_k: [Q,K] legality mask.

    The returned [Q,K] mass sums the main-attention probability assigned by all Q heads.
    This dense reference is only for selected teacher queries; TPU code can stream K tiles.
    """
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and shared latent k must use the same head dimension")
    scale = q.shape[-1] ** -0.5
    logits = jnp.einsum("qhd,kd->qhk", q, k) * scale
    probs = jnp.exp(logits - total_lse[..., None])
    probs = jnp.where(valid_k[:, None, :], probs, 0.0)
    return jnp.sum(probs, axis=1)


def indexer_cross_entropy_from_mass(
    index_scores: jnp.ndarray,
    teacher_mass: jnp.ndarray,
    valid_k: jnp.ndarray,
    query_valid: jnp.ndarray | None = None,
    eps: float = 1e-9,
) -> jnp.ndarray:
    """Cross entropy against normalized teacher mass without materializing normalization."""
    masked_scores = jnp.where(valid_k, index_scores, -jnp.inf)
    mass = jnp.where(valid_k, teacher_mass, 0.0)
    mass_sum = jnp.sum(mass, axis=-1)
    weighted_score = jnp.sum(mass * jnp.where(valid_k, index_scores, 0.0), axis=-1)
    per_query = jax.nn.logsumexp(masked_scores, axis=-1) - weighted_score / jnp.maximum(
        mass_sum, eps
    )
    finite_teacher = mass_sum > eps
    valid = finite_teacher if query_valid is None else (finite_teacher & query_valid)
    denom = jnp.maximum(jnp.sum(valid), 1)
    return jnp.sum(jnp.where(valid, per_query, 0.0)) / denom


def mean_all_served_teacher_mass(
    teacher_masses: tuple[jnp.ndarray, ...],
) -> jnp.ndarray:
    """Aggregate all served-layer teachers in the nano 2-3 layer reuse group."""
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
