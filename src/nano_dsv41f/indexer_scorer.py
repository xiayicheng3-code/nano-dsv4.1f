from __future__ import annotations

import jax.nn as jnn
import jax.numpy as jnp


def dense_index_scores(
    q: jnp.ndarray,
    k: jnp.ndarray,
    head_weight: jnp.ndarray,
    *,
    relu_scores: bool = True,
) -> jnp.ndarray:
    """Small reference scorer for the sparse-retrieval indexer.

    Args:
      q: [Q, H, D]
      k: [K, H, D]
      head_weight: [H], learned mixing weights over indexer heads

    Returns:
      [Q, K] retrieval scores.

    This mirrors the useful architectural idea: low-dimensional multi-head query/key
    similarities are combined into one token-retrieval score. It is intentionally dense
    and should only be used for correctness tests / late-stage teacher queries.
    """
    if q.ndim != 3 or k.ndim != 3:
        raise ValueError("q and k must have shape [tokens, heads, dim]")
    if q.shape[1:] != k.shape[1:]:
        raise ValueError("q/k head dimensions must match")
    if head_weight.shape != (q.shape[1],):
        raise ValueError("head_weight must have shape [heads]")

    scale = q.shape[-1] ** -0.5
    score = jnp.einsum("qhd,khd->qhk", q, k) * scale
    if relu_scores:
        score = jnn.relu(score)
    return jnp.einsum("qhk,h->qk", score, head_weight)


def masked_top_k(
    scores: jnp.ndarray,
    valid: jnp.ndarray,
    *,
    k: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Reference Top-K retrieval with a legality mask.

    The TPU training path initially does not put this hard Top-K on the backbone forward
    path. This helper exists for retrieval-quality evaluation and later sparse experiments.
    """
    if scores.shape != valid.shape:
        raise ValueError("scores and valid mask must have identical shape")
    if k <= 0 or k > scores.shape[-1]:
        raise ValueError("k must be in [1, scores.shape[-1]]")
    masked = jnp.where(valid, scores, -jnp.inf)
    values, indices = jnp.linalg.eigh(jnp.zeros((1, 1))) if False else (None, None)
    # lax.top_k is used through jax.lax to keep the operation JIT friendly.
    import jax

    values, indices = jax.lax.top_k(masked, k)
    return values, indices


def attention_mass_recall(
    teacher_mass: jnp.ndarray,
    retrieved_indices: jnp.ndarray,
    *,
    eps: float = 1e-9,
) -> jnp.ndarray:
    """Fraction of dense teacher attention mass retained by retrieved candidates."""
    chosen = jnp.take_along_axis(teacher_mass, retrieved_indices, axis=-1)
    return jnp.sum(chosen, axis=-1) / jnp.maximum(jnp.sum(teacher_mass, axis=-1), eps)
