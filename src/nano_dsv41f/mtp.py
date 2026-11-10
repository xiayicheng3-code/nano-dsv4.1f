"""Multi-token-prediction auxiliary loss through the DSpark draft head.

During SFT the DSpark block reads the backbone's last-layer features at sampled anchors
and predicts the next `block_size` tokens. Gradients flow into the backbone (and the
shared embedding / LM head), as in DeepSeek-V3-style MTP; the separate post-SFT DSpark
distillation (`scripts/train_dspark.py`) instead keeps the backbone frozen.

Anchors are sampled, not exhaustive: with 32K rows, scoring every position would add a
second full-vocabulary head over the whole batch. A fixed per-row budget keeps shapes
static and cost bounded. An anchor is eligible only when all `block_size` future targets
are real, same-segment, supervised (assistant) tokens.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .dspark import apply_dspark


def eligible_mtp_anchors(
    segment_ids: jax.Array,
    *,
    block_size: int,
    token_mask: jax.Array | None = None,
    target_mask: jax.Array | None = None,
) -> jax.Array:
    """[B, T] bool: anchor `a` can supervise targets `a+1 ... a+block_size`."""
    bsz, seq_len = segment_ids.shape
    real = jnp.ones_like(segment_ids, dtype=bool) if token_mask is None else token_mask.astype(bool)
    target = real if target_mask is None else (target_mask.astype(bool) & real)
    ok = real
    for k in range(1, block_size + 1):
        if k >= seq_len:
            return jnp.zeros((bsz, seq_len), dtype=bool)
        future = (
            (segment_ids[:, k:] == segment_ids[:, :-k]) & target[:, k:]
        )
        ok = ok & jnp.pad(future, ((0, 0), (0, k)), constant_values=False)
    return ok


def sample_mtp_anchors(
    eligible: jax.Array,
    *,
    anchors_per_row: int,
    step: jax.Array | int,
    seed: int = 1701,
) -> tuple[jax.Array, jax.Array]:
    """Uniform sample without replacement per row; returns positions and a keep mask."""
    n = min(int(anchors_per_row), int(eligible.shape[1]))
    key = jax.random.fold_in(jax.random.PRNGKey(seed), jnp.asarray(step, jnp.int32))
    scores = jax.random.uniform(key, eligible.shape, dtype=jnp.float32)
    scores = jnp.where(eligible, scores, -1.0)
    values, positions = jax.lax.top_k(scores, n)
    return positions.astype(jnp.int32), values >= 0.0


def mtp_loss(
    params,
    config,
    input_ids: jax.Array,
    *,
    context_features: jax.Array,
    segment_ids: jax.Array,
    token_mask: jax.Array | None = None,
    target_mask: jax.Array | None = None,
    anchors_per_row: int = 256,
    step: jax.Array | int = 0,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Position-weighted next-`block_size` cross entropy of the Markov-corrected drafts.

    Weights follow the DSpark paper's `exp(-i / block_size)` (zero-based `i`), normalized
    over every kept target, so nearer predictions dominate as in the distillation recipe.
    """
    block = config.dspark.block_size
    eligible = eligible_mtp_anchors(
        segment_ids, block_size=block, token_mask=token_mask, target_mask=target_mask
    )
    anchors, keep = sample_mtp_anchors(
        eligible, anchors_per_row=anchors_per_row, step=step
    )
    draft = apply_dspark(
        params["dspark"],
        config,
        embed=params["embed"],
        lm_head=params["lm_head"],
        input_ids=input_ids,
        context_features=context_features,
        anchor_positions=anchors,
        block_keep_mask=keep,
        segment_ids=segment_ids,
        windowed=True,
        dense_experts=True,
    )
    logits = draft["draft_logits"].astype(jnp.float32)  # [B, n, block, V]
    offsets = jnp.arange(1, block + 1, dtype=jnp.int32)
    target_pos = jnp.minimum(anchors[..., None] + offsets, input_ids.shape[1] - 1)
    labels = jnp.take_along_axis(
        input_ids, target_pos.reshape(input_ids.shape[0], -1), axis=1
    ).reshape(target_pos.shape)
    picked = jnp.take_along_axis(logits, labels[..., None], axis=-1)[..., 0]
    nll = jax.nn.logsumexp(logits, axis=-1) - picked
    weight = jnp.exp(-jnp.arange(block, dtype=jnp.float32) / block)[None, None, :]
    weight = weight * keep[..., None].astype(jnp.float32)
    denom = jnp.maximum(jnp.sum(weight), 1e-6)
    loss = jnp.sum(nll * weight) / denom
    correct = (jnp.argmax(logits, axis=-1) == labels) & keep[..., None]
    kept = jnp.maximum(jnp.sum(keep), 1)
    return loss, {
        "mtp_blocks": jnp.sum(keep.astype(jnp.int32)),
        "mtp_eligible": jnp.sum(eligible.astype(jnp.int32)),
        # Teacher-forced top-1 accuracy per draft depth (depth 0 = next token).
        "mtp_accuracy": jnp.sum(correct, axis=(0, 1)).astype(jnp.float32) / kept,
    }
