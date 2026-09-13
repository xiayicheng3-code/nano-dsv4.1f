from __future__ import annotations

import jax
import jax.nn as jnn
import jax.numpy as jnp

from .layers import init_linear, init_rms_norm, linear, rms_norm
from .quantization import fake_mxfp4_e2m1
from .rope import apply_partial_rope


def init_indexer(
    key: jax.Array,
    *,
    q_rank: int,
    model_dim: int,
    main_head_dim: int,
    n_heads: int,
    head_dim: int,
    owns_k: bool,
) -> dict[str, object]:
    """Initialize the V4.1-style sparse-attention indexer.

    Query heads are projected from the main attention Q-LoRA latent, while a single shared
    index K is projected from the *pre-RoPE compressed latent*. Head mixture weights are
    query-dependent and projected from the layer hidden state.
    """
    keys = iter(jax.random.split(key, 5))
    params: dict[str, object] = {
        "wq_b": init_linear(next(keys), q_rank, n_heads * head_dim),
        "weights_proj": init_linear(next(keys), model_dim, n_heads),
    }
    if owns_k:
        params.update(
            {
                "wk": init_linear(next(keys), main_head_dim, head_dim),
                "k_norm": init_rms_norm(head_dim),
            }
        )
    return params


def build_index_k(
    latent_pre_rope: jnp.ndarray,
    positions: jnp.ndarray,
    params: dict[str, object],
    *,
    rope_dim: int,
    rope_kwargs: dict[str, float | int],
    norm_eps: float,
    fp4_qat: bool = False,
    fp4_block_size: int = 16,
    fp4_scale_format: str = "e8m0",
) -> jnp.ndarray:
    """Build the shared cross-layer index K from a compressed source latent."""
    if "wk" not in params:
        raise ValueError("this indexer does not own an index-K projection")
    k = rms_norm(linear(latent_pre_rope, params["wk"]), params["k_norm"], eps=norm_eps)
    k = apply_partial_rope(k, positions, rotary_dim=rope_dim, **rope_kwargs)
    if fp4_qat:
        k = fake_mxfp4_e2m1(
            k,
            block_size=fp4_block_size,
            scale_format=fp4_scale_format,
            ste=True,
        )
    return k


def project_index_q_and_weights(
    qr: jnp.ndarray,
    hidden: jnp.ndarray,
    positions: jnp.ndarray,
    params: dict[str, object],
    *,
    n_heads: int,
    head_dim: int,
    rope_dim: int,
    rope_kwargs: dict[str, float | int],
    fp4_qat: bool = False,
    fp4_block_size: int = 16,
    fp4_scale_format: str = "e8m0",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Project indexer Q and query-dependent per-head mixing weights."""
    q = linear(qr, params["wq_b"]).reshape(*qr.shape[:-1], n_heads, head_dim)
    q = apply_partial_rope(q, positions, rotary_dim=rope_dim, **rope_kwargs)
    if fp4_qat:
        q = fake_mxfp4_e2m1(
            q,
            block_size=fp4_block_size,
            scale_format=fp4_scale_format,
            ste=True,
        )
    # Released V4.1 uses dynamic weights with this dimension-dependent scale.
    weights = linear(hidden.astype(jnp.float32), params["weights_proj"])
    weights = weights * (head_dim**-0.5 * n_heads**-0.5)
    return q, weights


def dense_index_scores(
    q: jnp.ndarray,
    k: jnp.ndarray,
    head_weight: jnp.ndarray,
    *,
    relu_scores: bool = True,
) -> jnp.ndarray:
    """Reference dense scorer for the released indexer equation.

    Args:
      q: [Q, H, D] or [..., Q, H, D]
      k: [K, D] or [..., K, D], shared across index heads
      head_weight: [Q, H] / [..., Q, H], or legacy static [H]

    Returns:
      [Q, K] / [..., Q, K] retrieval scores.
    """
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q/k index dimensions must match")
    score = jnp.einsum("...qhd,...kd->...qhk", q, k)
    if relu_scores:
        score = jnn.relu(score)
    if head_weight.ndim == 1:
        return jnp.einsum("...qhk,h->...qk", score, head_weight)
    if head_weight.shape[-2:] != q.shape[-3:-1]:
        raise ValueError("dynamic head_weight must match q's [query, head] axes")
    return jnp.einsum("...qhk,...qh->...qk", score, head_weight)


def apply_indexer_scores(
    qr: jnp.ndarray,
    hidden: jnp.ndarray,
    index_k: jnp.ndarray,
    q_positions: jnp.ndarray,
    params: dict[str, object],
    *,
    n_heads: int,
    head_dim: int,
    rope_dim: int,
    rope_kwargs: dict[str, float | int],
    fp4_qat: bool = False,
    fp4_block_size: int = 16,
    fp4_scale_format: str = "e8m0",
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    q, weights = project_index_q_and_weights(
        qr,
        hidden,
        q_positions,
        params,
        n_heads=n_heads,
        head_dim=head_dim,
        rope_dim=rope_dim,
        rope_kwargs=rope_kwargs,
        fp4_qat=fp4_qat,
        fp4_block_size=fp4_block_size,
        fp4_scale_format=fp4_scale_format,
    )
    return dense_index_scores(q, index_k, weights), {
        "index_q": q,
        "index_head_weights": weights,
    }


def masked_top_k(
    scores: jnp.ndarray,
    valid: jnp.ndarray,
    *,
    k: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    if scores.shape != valid.shape:
        raise ValueError("scores and valid mask must have identical shape")
    if k <= 0 or k > scores.shape[-1]:
        raise ValueError("k must be in [1, scores.shape[-1]]")
    return jax.lax.top_k(jnp.where(valid, scores, -jnp.inf), k)


def select_candidate_blocks(
    logits: jnp.ndarray,
    compress_lens: jnp.ndarray,
    *,
    topk_blocks: int,
    block_size: int,
) -> jnp.ndarray:
    """Released-style hierarchical decoder candidate mask.

    Scores are max-pooled by block. The newest reachable (possibly incomplete) block is
    forced to +inf so it is retained, then top blocks are expanded back to token positions.
    This is a dense reference; later sparse kernels can consume the boolean mask.
    """
    if logits.ndim < 2:
        raise ValueError("logits must end in [query, compressed_position]")
    if topk_blocks <= 0 or block_size <= 0:
        raise ValueError("topk_blocks and block_size must be positive")
    n = logits.shape[-1]
    n_blocks = (n + block_size - 1) // block_size
    pad = n_blocks * block_size - n
    padded = jnp.pad(logits, [(0, 0)] * (logits.ndim - 1) + [(0, pad)], constant_values=-jnp.inf)
    block_score = jnp.max(padded.reshape(*logits.shape[:-1], n_blocks, block_size), axis=-1)

    # compress_lens is per leading query row; block index of newest reachable candidate.
    newest = jnp.maximum((compress_lens - 1) // block_size, 0)
    newest_oh = jax.nn.one_hot(newest, n_blocks, dtype=bool)
    while newest_oh.ndim < block_score.ndim:
        newest_oh = newest_oh[..., None, :]
    newest_oh = jnp.broadcast_to(newest_oh, block_score.shape)
    block_score = jnp.where(newest_oh, jnp.inf, block_score)

    k = min(topk_blocks, n_blocks)
    values, indices = jax.lax.top_k(block_score, k)
    selected = jnp.any(jax.nn.one_hot(indices, n_blocks, dtype=bool), axis=-2)
    selected = selected & jnp.isfinite(values)[..., :, None].any(axis=-2)
    # Always retain the explicitly pinned newest block even though its top-k value is +inf.
    selected = selected | newest_oh
    token_mask = jnp.repeat(selected, block_size, axis=-1)[..., :n]
    positions = jnp.arange(n)
    valid_len = positions < compress_lens[..., None]
    while valid_len.ndim < token_mask.ndim:
        valid_len = valid_len[..., None, :]
    return token_mask & valid_len


def attention_mass_recall(
    teacher_mass: jnp.ndarray,
    retrieved_indices: jnp.ndarray,
    *,
    eps: float = 1e-9,
) -> jnp.ndarray:
    chosen = jnp.take_along_axis(teacher_mass, retrieved_indices, axis=-1)
    return jnp.sum(chosen, axis=-1) / jnp.maximum(jnp.sum(teacher_mass, axis=-1), eps)
