from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .csa2 import segment_local_positions
from .indexer import (
    dense_teacher_mass,
    eligibility_position,
    indexer_cross_entropy_from_mass,
    latest_teacher_indices_batched,
    mean_all_served_teacher_mass,
)
from .indexer_scorer import apply_indexer_scores, build_index_k
from .model import apply_model, build_layer_specs
from .rope import rope_kwargs_for_layer


@dataclass(frozen=True)
class IndexerGroup:
    """One index decision and the backbone layers that reuse it."""

    index_source_layer: int
    kv_source_layer: int
    served_layers: tuple[int, ...]
    compression_ratio: int


def build_indexer_groups(config) -> tuple[IndexerGroup, ...]:
    """Derive Full/Reindex reuse groups directly from the configured layer schedule."""
    specs = build_layer_specs(config)
    index_sources = [i for i, spec in enumerate(specs) if spec.is_index_source]
    groups: list[IndexerGroup] = []
    latest_kv_source = -1
    for i, spec in enumerate(specs):
        if spec.owns_global_kv:
            latest_kv_source = i
        if not spec.is_index_source:
            continue
        if latest_kv_source < 0:
            raise ValueError("index source appears before any compressed-KV source")
        next_source = next((j for j in index_sources if j > i), len(specs))
        groups.append(
            IndexerGroup(
                index_source_layer=i,
                kv_source_layer=latest_kv_source,
                served_layers=tuple(range(i, next_source)),
                compression_ratio=spec.compression_ratio,
            )
        )
    return tuple(groups)


def _batched_gather_tokens(x: jax.Array, indices: jax.Array) -> jax.Array:
    """Gather [B,T,...] at fixed-shape [B,Q] physical token indices."""
    if x.ndim < 2 or indices.ndim != 2 or x.shape[0] != indices.shape[0]:
        raise ValueError("expected x=[B,T,...] and indices=[B,Q]")
    gather = indices.reshape(indices.shape + (1,) * (x.ndim - 2))
    gather = jnp.broadcast_to(gather, indices.shape + x.shape[2:])
    return jnp.take_along_axis(x, gather, axis=1)


def _rope_kwargs(config, compression_ratio: int) -> dict[str, float | int]:
    rc = config.attention.rope
    return rope_kwargs_for_layer(
        compress_ratio=compression_ratio,
        rope_theta=rc.rope_theta,
        compress_rope_theta=rc.compress_rope_theta,
        original_seq_len=rc.original_seq_len,
        factor=rc.rope_factor,
        beta_fast=rc.beta_fast,
        beta_slow=rc.beta_slow,
    )


def _teacher_layers(group: IndexerGroup, policy: str) -> tuple[int, ...]:
    served = group.served_layers
    if policy == "all_served":
        return served
    if policy == "full_only":
        return (served[0],)
    if policy == "full_last":
        return (served[0],) if len(served) == 1 else (served[0], served[-1])
    raise ValueError(f"unknown teacher layer policy: {policy}")


def _candidate_mask_for_selected_queries(
    scores: jax.Array,
    valid: jax.Array,
    global_positions: jax.Array,
    *,
    compression_ratio: int,
    topk_blocks: int,
    block_size: int,
) -> jax.Array:
    """Build packed-safe hierarchical candidates for only the selected query rows."""
    if scores.shape != valid.shape:
        raise ValueError("scores and valid must have identical [B,Q,K] shape")
    n_k = scores.shape[-1]
    n_blocks = (n_k + block_size - 1) // block_size
    compressed_pos = global_positions // compression_ratio
    block_ids = jnp.clip(compressed_pos // block_size, 0, n_blocks - 1)
    one_hot = jax.nn.one_hot(block_ids, n_blocks, dtype=bool)  # [B,K,NB]

    masked = jnp.where(valid, scores, -jnp.inf)
    block_scores = jnp.max(
        jnp.where(one_hot[:, None, :, :], masked[..., None], -jnp.inf),
        axis=-2,
    )
    has_valid = jnp.any(valid, axis=-1)
    newest = jnp.max(jnp.where(valid, block_ids[:, None, :], -1), axis=-1)
    newest_oh = jax.nn.one_hot(jnp.maximum(newest, 0), n_blocks, dtype=bool)
    newest_oh = newest_oh & has_valid[..., None]
    block_scores = jnp.where(newest_oh, jnp.inf, block_scores)

    k = min(topk_blocks, n_blocks)
    _, block_indices = jax.lax.top_k(block_scores, k)
    selected_blocks = jnp.any(
        jax.nn.one_hot(block_indices, n_blocks, dtype=bool), axis=-2
    )
    selected_blocks = selected_blocks | newest_oh
    candidate = jnp.any(
        selected_blocks[:, :, None, :] & one_hot[:, None, :, :], axis=-1
    )
    return candidate & valid


def _selected_teacher_mass(
    layer_aux: dict[str, object],
    *,
    indices: jax.Array,
    main_kv: jax.Array,
) -> jax.Array:
    q = jax.lax.stop_gradient(_batched_gather_tokens(layer_aux["q"], indices))
    lse = jax.lax.stop_gradient(
        _batched_gather_tokens(layer_aux["total_lse"], indices)
    )
    valid = _batched_gather_tokens(layer_aux["global_valid"], indices)
    return jax.lax.stop_gradient(dense_teacher_mass(q, main_kv, lse, valid))


def selective_indexer_distillation_loss(
    params,
    config,
    input_ids: jax.Array,
    *,
    segment_ids: jax.Array,
    n_segments: int,
) -> tuple[jax.Array, dict[str, object]]:
    """Late-stage fixed-shape indexer distillation without full T-by-K scoring.

    The backbone runs once with dense main attention and ``compute_indexer=False``. For
    each Full/Reindex group, this function then:

    1. picks one latest eligible query slot per packed segment;
    2. builds the shared index K once from the owning compressor latent;
    3. scores only those selected student Q rows;
    4. reconstructs teacher global attention mass for selected rows from every configured
       served layer using the complete local+global+sink LSE;
    5. applies cross entropy over the legal (and, for later Reindex, hierarchical) pool.

    With ``detach_backbone_inputs=True`` the student still trains indexer-specific
    ``wk/k_norm/wq_b/weights_proj`` parameters, but its auxiliary gradients do not perturb
    the dense backbone. The teacher branch is always stop-gradient.
    """
    tc = config.indexer_training
    if not tc.enabled:
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        return zero, {"group_losses": {}, "active_queries": zero}
    if tc.teacher_queries != "latest_eligible":
        raise NotImplementedError(
            "selective reference currently implements teacher_queries='latest_eligible'"
        )
    if segment_ids.shape != input_ids.shape:
        raise ValueError("segment_ids must match input_ids")
    if n_segments <= 0:
        raise ValueError("n_segments must be a positive static integer")

    # The logits are intentionally unused; under jit/XLA their materialization can be DCE'd
    # when the caller only consumes this auxiliary loss.
    _, backbone_aux = apply_model(
        params,
        config,
        input_ids,
        segment_ids=segment_ids,
        compute_indexer=False,
    )
    layers = backbone_aux["layers"]
    local_positions = segment_local_positions(segment_ids)

    group_losses: dict[str, jax.Array] = {}
    query_counts: dict[str, jax.Array] = {}
    student_shapes: dict[str, jax.Array] = {}
    candidate_pool = None
    candidate_indices = None
    candidate_kv_source = None

    for group in build_indexer_groups(config):
        min_pos = eligibility_position(
            local_window=config.attention.local_window,
            retrieve_top_k=config.indexer.top_k,
            compression_ratio=group.compression_ratio,
            rule=tc.eligibility_rule,
        )
        query_indices, query_valid = latest_teacher_indices_batched(
            segment_ids,
            n_segments=n_segments,
            min_local_position=min_pos,
        )
        q_positions = _batched_gather_tokens(local_positions, query_indices)

        source_aux = layers[group.index_source_layer]
        kv_aux = layers[group.kv_source_layer]
        latent = kv_aux["compressed_latent"]
        global_positions = kv_aux["global_positions"]
        main_kv = kv_aux["main_kv"]
        if latent is None or global_positions is None or main_kv is None:
            raise ValueError("indexer group has no compressed source state")

        if tc.detach_backbone_inputs:
            latent_for_student = jax.lax.stop_gradient(latent)
        else:
            latent_for_student = latent

        kv_indexer_params = params["blocks"][group.kv_source_layer]["attn"][
            "indexer"
        ]
        student_params = params["blocks"][group.index_source_layer]["attn"][
            "indexer"
        ]
        index_k = build_index_k(
            latent_for_student,
            global_positions,
            kv_indexer_params,
            rope_dim=config.attention.rope.rope_head_dim,
            rope_kwargs=_rope_kwargs(config, group.compression_ratio),
            norm_eps=config.norm_eps,
            fp4_qat=config.quantization.indexer_fp4_qat,
            fp4_block_size=config.quantization.indexer.block_size
            if hasattr(config.quantization, "indexer")
            else config.quantization.indexer_block_size,
            fp4_scale_format=config.quantization.indexer_scale_format,
        )

        qr = _batched_gather_tokens(source_aux["qr"], query_indices)
        hidden = _batched_gather_tokens(source_aux["index_hidden"], query_indices)
        if tc.detach_backbone_inputs:
            qr = jax.lax.stop_gradient(qr)
            hidden = jax.lax.stop_gradient(hidden)

        student_scores, _ = apply_indexer_scores(
            qr,
            hidden,
            index_k,
            q_positions,
            student_params,
            n_heads=config.indexer.n_heads,
            head_dim=config.indexer.head_dim,
            rope_dim=config.attention.rope.rope_head_dim,
            rope_kwargs=_rope_kwargs(config, group.compression_ratio),
            fp4_qat=config.quantization.indexer_fp4_qat,
            fp4_block_size=config.quantization.indexer_block_size,
            fp4_scale_format=config.quantization.indexer_scale_format,
        )

        student_valid = _batched_gather_tokens(
            source_aux["global_valid"], query_indices
        ) & query_valid[..., None]

        if (
            candidate_pool is not None
            and config.indexer.candidate_source_layer < group.index_source_layer
            and candidate_kv_source == group.kv_source_layer
        ):
            # The default decoder source/reindex groups use the same r=1 selected-query
            # coordinates. Keep the assumption explicit rather than silently misaligning.
            if candidate_indices is None:
                raise AssertionError("candidate indices missing")
            same_slots = jnp.all(query_indices == candidate_indices, axis=-1)
            student_valid = student_valid & candidate_pool & same_slots[..., None]

        teacher_masses = tuple(
            _selected_teacher_mass(
                layers[layer_id],
                indices=query_indices,
                main_kv=jax.lax.stop_gradient(main_kv),
            )
            for layer_id in _teacher_layers(group, tc.teacher_layers)
        )
        teacher_mass = mean_all_served_teacher_mass(teacher_masses)
        loss = indexer_cross_entropy_from_mass(
            student_scores,
            teacher_mass,
            student_valid,
            query_valid=query_valid,
        )
        key = f"L{group.index_source_layer}"
        group_losses[key] = loss
        query_counts[key] = jnp.sum(query_valid.astype(jnp.int32))
        student_shapes[key] = jnp.asarray(student_scores.shape, dtype=jnp.int32)

        if group.index_source_layer == config.indexer.candidate_source_layer:
            candidate_pool = _candidate_mask_for_selected_queries(
                student_scores,
                student_valid,
                global_positions,
                compression_ratio=group.compression_ratio,
                topk_blocks=config.indexer.candidate_topk_blocks,
                block_size=config.indexer.candidate_block_size,
            )
            candidate_pool = jax.lax.stop_gradient(candidate_pool)
            candidate_indices = query_indices
            candidate_kv_source = group.kv_source_layer

    if not group_losses:
        total = jnp.asarray(0.0, dtype=jnp.float32)
    else:
        total = jnp.mean(jnp.stack(tuple(group_losses.values())))
    total = total * tc.loss_weight
    active_queries = sum(query_counts.values(), jnp.asarray(0, dtype=jnp.int32))
    return total, {
        "group_losses": group_losses,
        "active_queries": active_queries,
        "student_score_shapes": student_shapes,
    }
