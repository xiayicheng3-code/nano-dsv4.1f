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
    """Return fixed-shape teacher-query indices: one slot per packed segment.

    A slot is valid only when that segment contains a query with at least
    `local_window + retrieve_top_k` prior/local positions under our educational policy.
    Invalid slots point to zero and must be masked by the returned `valid` array.

    The result shape depends only on `n_segments`, making it friendly to JIT/static graphs.
    """
    if segment_ids.ndim != 1:
        raise ValueError("segment_ids must be rank-1")
    if n_segments <= 0:
        raise ValueError("n_segments must be positive")

    pos = jnp.arange(segment_ids.shape[0], dtype=jnp.int32)
    starts = jnp.concatenate(
        [jnp.array([True]), segment_ids[1:] != segment_ids[:-1]]
    )
    # O(S * n_segments) reference implementation. Packing metadata can precompute these
    # host-side later; the important property is fixed output shape.
    segment_start = jnp.stack(
        [jnp.max(jnp.where((segment_ids == s) & starts, pos, -1)) for s in range(n_segments)]
    )
    segment_end = jnp.stack(
        [jnp.max(jnp.where(segment_ids == s, pos, -1)) for s in range(n_segments)]
    )
    local_end = segment_end - segment_start
    valid = (segment_start >= 0) & (local_end >= (local_window + retrieve_top_k))
    indices = jnp.where(valid, segment_end, 0)
    return indices, valid


def dense_teacher_mass(
    q: jnp.ndarray,
    k: jnp.ndarray,
    total_lse: jnp.ndarray,
    valid_k: jnp.ndarray,
) -> jnp.ndarray:
    """Reference teacher mass over candidate KV positions for selected queries.

    Args:
      q: [Q, H, D] selected main-attention queries.
      k: [K, H, D] shared/compressed main-attention keys.
      total_lse: [Q, H] LSE from the *complete* local+global attention domain.
      valid_k: [Q, K] legality mask for the compressed/global candidates.

    Returns:
      Unnormalized teacher mass [Q, K], summed across main-attention heads.

    This intentionally materializes QxK for small reference tests only. The TPU path
    should stream tiles and accumulate the distillation statistics without retaining QxK.
    """
    scale = q.shape[-1] ** -0.5
    logits = jnp.einsum("qhd,khd->qhk", q, k) * scale
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
    """Distill an indexer without requiring a normalized teacher distribution.

    For each query q:

        L = logsumexp(I) - sum_j u_j I_j / sum_j u_j

    where u_j is the main attention mass assigned to compressed candidate j. This has the
    same gradient as cross entropy against the normalized teacher distribution while
    avoiding a separately materialized normalized target.
    """
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


def indexer_stage_active(
    step: jnp.ndarray,
    total_steps: int,
    start_fraction: float,
    end_fraction: float,
) -> jnp.ndarray:
    """Static-graph-compatible gate for late pre/mid-training indexer distillation."""
    progress = step.astype(jnp.float32) / float(total_steps)
    return (progress >= start_fraction) & (progress <= end_fraction)
