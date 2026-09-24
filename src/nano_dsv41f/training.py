from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .csa2 import segment_local_positions
from .indexer import (
    budgeted_teacher_indices,
    dense_teacher_mass,
    eligible_teacher_queries,
    eligibility_position,
    indexer_cross_entropy_from_mass,
    latest_teacher_indices_batched,
    mean_all_served_teacher_mass,
)
from .indexer_scorer import apply_indexer_scores, build_index_k
from .model import apply_model, build_layer_specs
from .optimizer import optimizer_step
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
    indices_valid: jax.Array | None = None,
    segment_ids: jax.Array | None = None,
    local_window: int | None = None,
) -> jax.Array:
    q = jax.lax.stop_gradient(_batched_gather_tokens(layer_aux["q"], indices))
    if layer_aux["total_lse"] is None:
        return selected_teacher_mass_native(
            q, main_kv, jax.lax.stop_gradient(layer_aux["local_kv"]),
            indices, segment_ids, indices_valid, local_window=local_window,
            sink=layer_aux.get("attn_sink"),
        )
    lse = jax.lax.stop_gradient(
        _batched_gather_tokens(layer_aux["total_lse"], indices)
    )
    valid = _batched_gather_tokens(layer_aux["global_valid"], indices)
    return jax.lax.stop_gradient(dense_teacher_mass(q, main_kv, lse, valid))


def selected_global_valid(layer_aux, indices, segment_ids, local_window):
    """Construct only selected rows of the compressed-history mask."""
    q_pos = _batched_gather_tokens(segment_local_positions(segment_ids), indices)
    q_seg = _batched_gather_tokens(segment_ids, indices)
    return (
        (q_seg[..., None] == layer_aux["global_segment_ids"][:, None, :])
        & (layer_aux["global_positions"][:, None, :] <= q_pos[..., None] - local_window)
    )


def selected_teacher_mass_native(q, main_kv, local_kv, indices, segment_ids, valid,
                                 *, local_window, sink):
    """Exact selected-row local + global + sink denominator, FP32 accumulation.

    q is already RoPE/QAT transformed, as are both KV banks. Only W recent local
    tokens are gathered; global logits are reused for both LSE and teacher mass.
    """
    local_indices = indices[..., None] - jnp.arange(local_window - 1, -1, -1)
    safe = jnp.maximum(local_indices, 0)
    local_keys = jax.vmap(lambda kv, ix: kv[ix])(local_kv, safe)
    local_segments = jax.vmap(lambda seg, ix: seg[ix])(segment_ids, safe)
    q_segments = _batched_gather_tokens(segment_ids, indices)
    local_valid = (local_indices >= 0) & (local_segments == q_segments[..., None])
    # Splash scales Q before its FP32 dot accumulation; mirror that rounding.
    scaled_q = q * jnp.asarray(q.shape[-1] ** -0.5, dtype=q.dtype)
    local_logits = jnp.einsum("bqhd,bqwd->bqhw", scaled_q, local_keys,
                             preferred_element_type=jnp.float32)
    global_logits = jnp.einsum("bqhd,bkd->bqhk", scaled_q, main_kv,
                              preferred_element_type=jnp.float32)
    local_lse = jax.nn.logsumexp(jnp.where(local_valid[:, :, None, :], local_logits, -jnp.inf), axis=-1)
    global_lse = jax.nn.logsumexp(jnp.where(valid[:, :, None, :], global_logits, -jnp.inf), axis=-1)
    total = jnp.logaddexp(local_lse, global_lse)
    if sink is not None:
        total = jnp.logaddexp(total, sink.astype(jnp.float32))
    mass = jnp.sum(jnp.where(valid[:, :, None, :],
                            jnp.exp(global_logits - total[..., None]), 0.0), axis=-2)
    return jax.lax.stop_gradient(mass)


def _native_moe_diagnostics(backbone_aux, config) -> dict[str, jax.Array]:
    """Expose enough routed-EP state for TPU smoke tests to detect silent fallback."""
    native_layers = tuple(
        layer for layer in backbone_aux["layers"] if "expert_loads" in layer
    )
    if not native_layers:
        return {
            "native_moe_layers": jnp.asarray(0, dtype=jnp.int32),
            "expert_loads": jnp.zeros((config.n_experts,), dtype=jnp.int32),
            "expert_loads_by_layer": jnp.zeros((0, config.n_experts), dtype=jnp.int32),
            "expert_overflow": jnp.zeros((config.n_experts,), dtype=jnp.int32),
            "expert_dropped": jnp.zeros((config.n_experts,), dtype=jnp.int32),
            "expert_capacity": jnp.asarray(0, dtype=jnp.int32),
            "expert_packed_rows": jnp.asarray(0, dtype=jnp.int32),
            "moe_mosaic_layers": jnp.asarray(0, dtype=jnp.int32),
            "experts_per_chip": jnp.asarray(0, dtype=jnp.int32),
        }
    loads = jnp.sum(
        jnp.stack(tuple(layer["expert_loads"] for layer in native_layers), axis=0),
        axis=0,
    )
    overflow = jnp.sum(
        jnp.stack(tuple(layer["expert_overflow"] for layer in native_layers), axis=0),
        axis=0,
    )
    return {
        "native_moe_layers": jnp.asarray(len(native_layers), dtype=jnp.int32),
        "expert_loads": loads,
        "expert_loads_by_layer": jnp.stack(tuple(layer["expert_loads"] for layer in native_layers)),
        "expert_overflow": overflow,
        "expert_dropped": jnp.sum(jnp.stack(tuple(layer["expert_dropped"] for layer in native_layers)), axis=0),
        "expert_capacity": native_layers[0]["expert_capacity"],
        "expert_packed_rows": native_layers[0]["expert_packed_rows"],
        "moe_mosaic_layers": sum(layer["moe_mosaic"] for layer in native_layers),
        "experts_per_chip": native_layers[0].get(
            "experts_per_chip", jnp.asarray(1, dtype=jnp.int32)
        ),
    }


def causal_lm_loss(
    logits: jax.Array,
    input_ids: jax.Array,
    segment_ids: jax.Array,
    *,
    token_mask: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Packed next-token cross entropy that never predicts across segment boundaries."""
    if logits.ndim != 3 or input_ids.ndim != 2 or segment_ids.shape != input_ids.shape:
        raise ValueError("expected logits=[B,T,V] and ids/segments=[B,T]")
    if logits.shape[:2] != input_ids.shape or input_ids.shape[1] < 2:
        raise ValueError("logit/token shapes must align and sequence length must be >= 2")
    if token_mask is not None and token_mask.shape != input_ids.shape:
        raise ValueError("token_mask must match input_ids")

    pred = logits[:, :-1].astype(jnp.float32)
    labels = input_ids[:, 1:]
    valid = segment_ids[:, :-1] == segment_ids[:, 1:]
    if token_mask is not None:
        valid = valid & token_mask[:, :-1] & token_mask[:, 1:]
    # Avoid materializing a second full [B,T,V] log-probability array.
    target = jnp.take_along_axis(pred, labels[..., None], axis=-1)[..., 0]
    nll = jax.nn.logsumexp(pred, axis=-1) - target
    count = jnp.sum(valid.astype(jnp.int32))
    loss = jnp.sum(jnp.where(valid, nll, 0.0)) / jnp.maximum(count, 1)
    return loss, count


def _selective_indexer_from_backbone_aux(
    params,
    config,
    *,
    segment_ids: jax.Array,
    token_mask: jax.Array | None,
    n_segments: int | None,
    backbone_aux: dict[str, object],
    step: jax.Array | int = 0,
) -> tuple[jax.Array, dict[str, object]]:
    """Compute selected-row indexer loss from an already executed dense backbone."""
    tc = config.indexer_training
    if not tc.enabled:
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        return zero, {
            "raw_loss": zero,
            "group_losses": {},
            "query_counts": {},
            "eligible_query_counts": {},
            "student_key_counts": {},
            "teacher_query_indices": {},
            "teacher_query_valid": {},
            "active_queries": jnp.asarray(0, jnp.int32),
            "student_score_shapes": {},
        }
    if tc.teacher_queries == "latest_eligible" and (n_segments is None or n_segments <= 0):
        raise ValueError("latest_eligible requires a positive static n_segments")

    layers = backbone_aux["layers"]
    local_positions = segment_local_positions(segment_ids)
    group_losses: dict[str, jax.Array] = {}
    query_counts: dict[str, jax.Array] = {}
    eligible_query_counts: dict[str, jax.Array] = {}
    student_key_counts: dict[str, jax.Array] = {}
    student_shapes: dict[str, jax.Array] = {}
    teacher_query_indices: dict[str, jax.Array] = {}
    teacher_query_valid: dict[str, jax.Array] = {}
    candidate_pool = None
    candidate_indices = None
    candidate_kv_source = None
    index_k_cache = {}
    query_cache = {}

    for group in build_indexer_groups(config):
        min_pos = eligibility_position(
            local_window=config.attention.local_window,
            retrieve_top_k=config.indexer.top_k,
            compression_ratio=group.compression_ratio,
            rule=tc.eligibility_rule,
        )
        if min_pos not in query_cache:
            eligible = eligible_teacher_queries(
                segment_ids, min_local_position=min_pos, token_mask=token_mask,
            )
            if tc.teacher_queries == "sampled":
                selected = budgeted_teacher_indices(
                    segment_ids, query_budget=tc.query_budget,
                    min_local_position=min_pos, token_mask=token_mask,
                    step=step, seed=tc.query_seed,
                )
            elif tc.teacher_queries == "all_eligible":
                indices = jnp.broadcast_to(jnp.arange(segment_ids.shape[1]), segment_ids.shape)
                selected = (jnp.where(eligible, indices, 0), eligible)
            else:  # Explicit legacy ablation; never the default.
                selected = latest_teacher_indices_batched(
                    segment_ids, n_segments=n_segments,
                    min_local_position=min_pos, token_mask=token_mask,
                )
            query_cache[min_pos] = (*selected, jnp.sum(eligible.astype(jnp.int32)))
        query_indices, query_valid, eligible_count = query_cache[min_pos]
        q_positions = _batched_gather_tokens(local_positions, query_indices)

        source_aux = layers[group.index_source_layer]
        kv_aux = layers[group.kv_source_layer]
        latent = kv_aux["compressed_latent"]
        global_positions = kv_aux["global_positions"]
        main_kv = kv_aux["main_kv"]
        if latent is None or global_positions is None or main_kv is None:
            raise ValueError("indexer group has no compressed source state")

        latent_for_student = (
            jax.lax.stop_gradient(latent) if tc.detach_backbone_inputs else latent
        )
        kv_indexer_params = params["blocks"][group.kv_source_layer]["attn"]["indexer"]
        student_params = params["blocks"][group.index_source_layer]["attn"]["indexer"]
        if group.kv_source_layer not in index_k_cache:
            index_k_cache[group.kv_source_layer] = build_index_k(
                latent_for_student,
                global_positions,
                kv_indexer_params,
                rope_dim=config.attention.rope.rope_head_dim,
                rope_kwargs=_rope_kwargs(config, group.compression_ratio),
                norm_eps=config.norm_eps,
                fp4_qat=config.quantization.indexer_fp4_qat,
                fp4_block_size=config.quantization.indexer_block_size,
                fp4_scale_format=config.quantization.indexer_scale_format,
            )
        index_k = index_k_cache[group.kv_source_layer]

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
        full_history_valid = selected_global_valid(
            source_aux, query_indices, segment_ids, config.attention.local_window,
        )
        student_valid = full_history_valid & query_valid[..., None]

        if (
            tc.apply_candidate_mask
            and candidate_pool is not None
            and config.indexer.candidate_source_layer < group.index_source_layer
            and candidate_kv_source == group.kv_source_layer
        ):
            if candidate_indices is None:
                raise AssertionError("candidate indices missing")
            same_slots = query_indices == candidate_indices
            student_valid = student_valid & candidate_pool & same_slots[..., None]

        teacher_masses = tuple(
            _selected_teacher_mass(
                layers[layer_id],
                indices=query_indices,
                main_kv=jax.lax.stop_gradient(main_kv),
                indices_valid=full_history_valid,
                segment_ids=segment_ids,
                local_window=config.attention.local_window,
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
        eligible_query_counts[key] = eligible_count
        student_key_counts[key] = jnp.sum(student_valid.astype(jnp.int32), axis=-1)
        student_shapes[key] = jnp.asarray(student_scores.shape, dtype=jnp.int32)
        teacher_query_indices[key] = query_indices
        teacher_query_valid[key] = query_valid

        if tc.apply_candidate_mask and group.index_source_layer == config.indexer.candidate_source_layer:
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

    raw = (
        jnp.mean(jnp.stack(tuple(group_losses.values())))
        if group_losses
        else jnp.asarray(0.0, dtype=jnp.float32)
    )
    weighted = raw * tc.loss_weight
    active_queries = sum(query_counts.values(), jnp.asarray(0, dtype=jnp.int32))
    return weighted, {
        "raw_loss": raw,
        "group_losses": group_losses,
        "query_counts": query_counts,
        "eligible_query_counts": eligible_query_counts,
        "student_key_counts": student_key_counts,
        "teacher_query_indices": teacher_query_indices,
        "teacher_query_valid": teacher_query_valid,
        "active_queries": active_queries,
        "student_score_shapes": student_shapes,
    }


def selective_indexer_distillation_loss(
    params,
    config,
    input_ids: jax.Array,
    *,
    segment_ids: jax.Array,
    n_segments: int | None = None,
    token_mask: jax.Array | None = None,
    step: jax.Array | int = 0,
) -> tuple[jax.Array, dict[str, object]]:
    """Standalone selected-row indexer loss; useful for tests and retrieval-only tuning."""
    if segment_ids.shape != input_ids.shape:
        raise ValueError("segment_ids must match input_ids")
    if token_mask is not None and token_mask.shape != input_ids.shape:
        raise ValueError("token_mask must match input_ids")
    _, backbone_aux = apply_model(
        params,
        config,
        input_ids,
        segment_ids=segment_ids,
        token_mask=token_mask,
        compute_indexer=False,
    )
    return _selective_indexer_from_backbone_aux(
        params,
        config,
        segment_ids=segment_ids,
        token_mask=token_mask,
        n_segments=n_segments,
        backbone_aux=backbone_aux,
        step=step,
    )


def pretrain_loss(
    params,
    config,
    input_ids: jax.Array,
    *,
    segment_ids: jax.Array,
    token_mask: jax.Array | None = None,
    include_indexer: bool = False,
    n_segments: int | None = None,
    step: jax.Array | int = 0,
) -> tuple[jax.Array, dict[str, object]]:
    """One-backbone-pass packed LM objective, optionally with late indexer distillation.

    `include_indexer` is intended to be a **static** choice: compile one base function and
    one late-indexer function, then let the Python training driver select between them by
    step. This avoids embedding a large dynamic branch inside one XLA graph.
    """
    logits, backbone_aux = apply_model(
        params,
        config,
        input_ids,
        segment_ids=segment_ids,
        token_mask=token_mask,
        compute_indexer=False,
    )
    lm, lm_tokens = causal_lm_loss(
        logits, input_ids, segment_ids, token_mask=token_mask
    )
    if include_indexer and config.indexer_training.enabled:
        index_loss, index_aux = _selective_indexer_from_backbone_aux(
            params,
            config,
            segment_ids=segment_ids,
            token_mask=token_mask,
            n_segments=n_segments,
            backbone_aux=backbone_aux,
            step=step,
        )
    else:
        index_loss = jnp.asarray(0.0, dtype=jnp.float32)
        index_aux = {
            "raw_loss": jnp.asarray(0.0, dtype=jnp.float32),
            "group_losses": {},
            "query_counts": {},
            "eligible_query_counts": {},
            "student_key_counts": {},
            "teacher_query_indices": {},
            "teacher_query_valid": {},
            "active_queries": jnp.asarray(0, dtype=jnp.int32),
            "student_score_shapes": {},
        }
    total = lm + index_loss
    # MoE owns token-level routing in its own sharding domain and exports only [E]
    # real-token counts. Do not combine tp-sharded router indices with x/y masks here.
    router_loads = jnp.stack(
        tuple(layer["router_loads"] for layer in backbone_aux["layers"])
    )
    return total, {
        "lm_loss": lm,
        "lm_tokens": lm_tokens,
        "indexer_loss": index_loss,
        "indexer": index_aux,
        "router_loads": router_loads,
        **_native_moe_diagnostics(backbone_aux, config),
    }


def indexer_phase_enabled(step: int, train_config, config) -> bool:
    """Python-side phase selector used to choose between two separately-jitted steps."""
    if not config.indexer_training.enabled:
        return False
    progress = step / float(train_config.total_steps)
    return (
        config.indexer_training.start_fraction
        <= progress
        <= config.indexer_training.end_fraction
    )


def pretrain_step(
    params,
    optimizer_state,
    config,
    train_config,
    input_ids: jax.Array,
    *,
    segment_ids: jax.Array,
    step: jax.Array,
    token_mask: jax.Array | None = None,
    include_indexer: bool = False,
    n_segments: int | None = None,
):
    """Reference gradient/update step for base or late-indexer pretraining phases.

    `include_indexer` should be closed over as a static bool before `jax.jit`. DSpark is
    intentionally frozen here because its separate draft objective is not part of this
    language-model pretraining step.
    """

    def loss_fn(p):
        return pretrain_loss(
            p,
            config,
            input_ids,
            segment_ids=segment_ids,
            token_mask=token_mask,
            include_indexer=include_indexer,
            n_segments=n_segments,
            step=step,
        )

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    train_indexer = bool(include_indexer and config.indexer_training.enabled)
    new_params, new_state, optimizer_metrics = optimizer_step(
        params,
        grads,
        optimizer_state,
        step=step,
        config=config,
        train_config=train_config,
        train_indexer=train_indexer,
        train_dspark=False,
    )
    new_params = update_router_biases(new_params, metrics["router_loads"],
                                     speed=config.router_bias_update_speed)
    return new_params, new_state, {
        "loss": loss,
        **metrics,
        **optimizer_metrics,
    }


def update_router_biases(params, loads, *, speed):
    """Aux-loss-free text routing controller; one independent bias per layer/expert."""
    if not speed:
        return params
    blocks = []
    for i, block in enumerate(params["blocks"]):
        bias = block["moe"]["router_bias"]
        correction = speed * jnp.sign(jnp.mean(loads[i].astype(jnp.float32)) - loads[i])
        blocks.append({**block, "moe": {**block["moe"], "router_bias": bias + correction}})
    return {**params, "blocks": tuple(blocks)}
