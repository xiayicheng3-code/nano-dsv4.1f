from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp

from .config import ModelConfig
from .csa2 import SharedCSA2State, apply_csa2_attention, init_csa2_attention
from .dspark import apply_dspark, init_dspark
from .engram import apply_engram, init_engram, ngram_hash_ids
from .layers import init_embedding, init_rms_norm, rms_norm
from .mhc import (
    init_mhc_generator,
    make_identity_pre_mix,
    mhc_mixes,
    post_mix,
    pre_mix,
)
from .moe import apply_moe, init_moe


@dataclass(frozen=True)
class LayerSpec:
    layer_id: int
    half: Literal["context", "generation"]
    mode: Literal["swa", "full", "reindex", "reuse"]
    owns_global_kv: bool
    is_index_source: bool
    compression_ratio: int
    has_engram: bool


def build_layer_specs(config: ModelConfig) -> tuple[LayerSpec, ...]:
    """Build the explicit nano CED/CSA2 schedule from hyperparameters.

    DeepSeek's released code includes a useful five-layer tiny default. Our seven-layer
    default keeps that anchor but extends the decoder by one retrieval-sharing group:

        context:    L0 SWA | L1 Full(r=2) -> L2 Reuse
        generation: L3 Full(r=1) -> L4 Reuse -> L5 Reindex -> L6 Reuse

    Deeper generation settings continue the same Reindex/Reuse cadence without changing
    block code.
    """
    specs: list[LayerSpec] = []
    cc = config.csa2

    for i in range(cc.context_layers):
        if i < cc.context_swa_only_layers:
            mode, owns, index_source, ratio = "swa", False, False, 0
        else:
            relative = i - cc.context_swa_only_layers
            is_source = relative % cc.context_retriever_group_size == 0
            mode = "full" if is_source else "reuse"
            owns = is_source
            index_source = is_source
            ratio = cc.context_compression_ratio
        specs.append(
            LayerSpec(
                layer_id=i,
                half="context",
                mode=mode,
                owns_global_kv=owns,
                is_index_source=index_source,
                compression_ratio=ratio,
                has_engram=i in config.engram.layer_ids,
            )
        )

    base = cc.context_layers
    for j in range(cc.generation_layers):
        layer_id = base + j
        if j == 0:
            mode, owns, index_source = "full", True, True
        elif j % cc.generation_retriever_group_size == 0:
            mode, owns, index_source = "reindex", False, True
        else:
            mode, owns, index_source = "reuse", False, False
        specs.append(
            LayerSpec(
                layer_id=layer_id,
                half="generation",
                mode=mode,
                owns_global_kv=owns,
                is_index_source=index_source,
                compression_ratio=cc.generation_compression_ratio,
                has_engram=layer_id in config.engram.layer_ids,
            )
        )
    return tuple(specs)


def _init_block(
    key: jax.Array, config: ModelConfig, spec: LayerSpec
) -> dict[str, object]:
    keys = iter(jax.random.split(key, 10))
    ac = config.attention
    block: dict[str, object] = {
        "attn_norm": init_rms_norm(config.d_model),
        "ffn_norm": init_rms_norm(config.d_model),
        "mhc_attn": init_mhc_generator(
            next(keys), config.mhc_streams, config.d_model
        ),
        "mhc_ffn": init_mhc_generator(
            next(keys), config.mhc_streams, config.d_model
        ),
        "attn": init_csa2_attention(
            next(keys),
            dim=config.d_model,
            n_heads=ac.n_heads,
            head_dim=ac.head_dim,
            q_rank=ac.q_rank,
            o_rank=ac.o_rank,
            o_groups=ac.o_groups,
            owns_global_kv=spec.owns_global_kv,
            is_index_source=spec.is_index_source,
            compression_ratio=spec.compression_ratio,
            attention_sink=ac.attention_sink,
            attention_sink_init=ac.attention_sink_init,
            index_n_heads=config.indexer.n_heads,
            index_head_dim=config.indexer.head_dim,
        ),
        "moe": init_moe(
            next(keys), config.d_model, config.d_ff, config.n_experts
        ),
    }
    if config.engram.enabled and spec.has_engram:
        n_cols = (
            config.engram.max_ngram_size - 1
        ) * config.engram.n_hash_heads
        block["engram"] = init_engram(
            next(keys),
            table_size=config.engram.table_size,
            head_dim=config.engram.head_dim,
            n_hash_cols=n_cols,
            n_streams=config.mhc_streams,
            dim=config.d_model,
        )
    return block


def init_model(key: jax.Array, config: ModelConfig) -> dict[str, object]:
    specs = build_layer_specs(config)
    keys = jax.random.split(key, len(specs) + 5)
    model: dict[str, object] = {
        "embed": init_embedding(keys[0], config.vocab_size, config.d_model),
        "blocks": tuple(
            _init_block(keys[i + 1], config, spec)
            for i, spec in enumerate(specs)
        ),
        "final_norm": init_rms_norm(config.d_model),
        "lm_head": init_embedding(
            keys[-2], config.vocab_size, config.d_model
        ).T,
    }
    if config.dspark.enabled:
        model["dspark"] = init_dspark(keys[-1], config)
    return model


def _apply_block(
    streams: jax.Array,
    incoming_pre_mix: jax.Array,
    segment_ids: jax.Array,
    params: dict[str, object],
    state: SharedCSA2State | None,
    config: ModelConfig,
    spec: LayerSpec,
    *,
    global_source: jax.Array | None = None,
    compute_indexer: bool = False,
) -> tuple[jax.Array, jax.Array, SharedCSA2State | None, dict[str, object]]:
    """One Single-Pass-mHC block with optional attention-only rematerialization."""
    residual = streams
    attn_pre, attn_post, attn_comb = mhc_mixes(
        streams,
        params["mhc_attn"],
        sinkhorn_iters=config.mhc_sinkhorn_iters,
        eps=config.mhc_eps,
        norm_eps=config.norm_eps,
    )
    attn_input = rms_norm(
        pre_mix(streams, incoming_pre_mix),
        params["attn_norm"],
        eps=config.norm_eps,
    )

    # Capture architecture/config metadata in the closure so remat only sees array pytrees.
    # Separate late-stage teacher distillation lives outside this remat boundary.
    def attention_forward(attn_x, attn_params, shared_state, ced_source):
        return apply_csa2_attention(
            attn_x,
            segment_ids,
            attn_params,
            shared_state,
            layer_id=spec.layer_id,
            mode=spec.mode,
            owns_global_kv=spec.owns_global_kv,
            compression_ratio=spec.compression_ratio,
            config=config,
            global_source=ced_source,
            compute_indexer=compute_indexer,
        )

    if config.remat.policy == "attention":
        attention_forward = jax.checkpoint(attention_forward)

    attn_out, state, attn_aux = attention_forward(
        attn_input, params["attn"], state, global_source
    )
    streams = post_mix(residual, attn_out, attn_comb, attn_post)

    residual = streams
    ffn_pre, ffn_post, ffn_comb = mhc_mixes(
        streams,
        params["mhc_ffn"],
        sinkhorn_iters=config.mhc_sinkhorn_iters,
        eps=config.mhc_eps,
        norm_eps=config.norm_eps,
    )
    ffn_input = rms_norm(
        pre_mix(streams, attn_pre),
        params["ffn_norm"],
        eps=config.norm_eps,
    )
    ffn_out, moe_aux = apply_moe(
        ffn_input,
        params["moe"],
        top_k=config.experts_per_token,
        route_scale=config.route_scale,
        swiglu_limit=config.swiglu_limit,
        eps=config.route_eps,
    )
    streams = post_mix(residual, ffn_out, ffn_comb, ffn_post)
    return streams, ffn_pre, state, {**attn_aux, **moe_aux}


def apply_model(
    params: dict[str, object],
    config: ModelConfig,
    input_ids: jax.Array,
    *,
    segment_ids: jax.Array | None = None,
    token_mask: jax.Array | None = None,
    compute_indexer: bool = False,
) -> tuple[jax.Array, dict[str, object]]:
    """Run the dense semantic backbone reference and expose training intermediates.

    ``compute_indexer=False`` is the normal dense pretraining path. It deliberately skips
    full-sequence index-K construction and T-by-K index scoring; late-stage distillation can
    score only selected query rows. Set it to ``True`` for retrieval diagnostics / explicit
    Full-Reindex-Reuse state-machine tests.

    Rematerialization policy is architectural plumbing, not model semantics:

    - ``none``: retain normal autodiff intermediates;
    - ``attention``: replay CSA2 attention math but retain MoE forward activations;
    - ``block``: replay the attention + MoE Transformer block from its mHC block boundary.

    Engram injection remains outside the block remat boundary because its table lookup is
    cheap relative to attention/MoE and its memory concern is parameter/table sharding.
    """
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch,tokens]")
    if segment_ids is None:
        segment_ids = jnp.zeros_like(input_ids, dtype=jnp.int32)
    if segment_ids.shape != input_ids.shape:
        raise ValueError("segment_ids must match input_ids")

    specs = build_layer_specs(config)
    streams = params["embed"][input_ids]
    streams = jnp.repeat(
        streams[..., None, :], config.mhc_streams, axis=-2
    )
    incoming_pre = make_identity_pre_mix(streams)
    state: SharedCSA2State | None = None
    layer_aux: list[dict[str, object]] = []
    context_final = None
    dspark_targets: list[jax.Array] = []
    target_ids = (
        set(config.dspark.target_layer_ids) if config.dspark.enabled else set()
    )

    for spec, block in zip(specs, params["blocks"]):
        generation_global_source = None
        if spec.half == "generation" and context_final is None:
            context_final = pre_mix(streams, incoming_pre)
            generation_global_source = context_final

        if "engram" in block:
            hashes = ngram_hash_ids(
                input_ids,
                segment_ids,
                table_size=config.engram.table_size,
                max_ngram_size=config.engram.max_ngram_size,
                n_hash_heads=config.engram.n_hash_heads,
                pad_token_id=config.engram.pad_token_id,
                seed=spec.layer_id * 97,
            )
            streams = apply_engram(
                streams,
                hashes,
                block["engram"],
                eps=config.mhc_eps,
                token_mask=token_mask,
            )

        # Released V4.1/DeepSpec extracts target features at the selected block input;
        # with mHC the public reference averages residual streams before concatenation.
        if spec.layer_id in target_ids:
            dspark_targets.append(jnp.mean(streams, axis=-2))

        def block_forward(s, pre, block_params, shared_state, ced_source):
            return _apply_block(
                s,
                pre,
                segment_ids,
                block_params,
                shared_state,
                config,
                spec,
                global_source=ced_source,
                compute_indexer=compute_indexer,
            )

        if config.remat.policy == "block":
            block_forward = jax.checkpoint(block_forward)

        streams, incoming_pre, state, aux = block_forward(
            streams,
            incoming_pre,
            block,
            state,
            generation_global_source,
        )
        layer_aux.append(aux)

    hidden = rms_norm(
        pre_mix(streams, incoming_pre),
        params["final_norm"],
        eps=config.norm_eps,
    )
    logits = jnp.einsum("btd,dv->btv", hidden, params["lm_head"])
    dspark_context = (
        jnp.concatenate(dspark_targets, axis=-1)
        if dspark_targets
        else None
    )
    return logits, {
        "context_final": context_final,
        "final_hidden": hidden,
        "final_global_source_layer": None
        if state is None
        else state.source_layer,
        "layers": tuple(layer_aux),
        "dspark_context_features": dspark_context,
    }


def apply_model_dspark(
    params: dict[str, object],
    config: ModelConfig,
    input_ids: jax.Array,
    *,
    anchor_positions: jax.Array,
    segment_ids: jax.Array | None = None,
    block_keep_mask: jax.Array | None = None,
    teacher_prev_ids: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, object], dict[str, jax.Array]]:
    """Convenience reference forward: backbone first, then the one-stage DSpark head."""
    if not config.dspark.enabled or "dspark" not in params:
        raise ValueError("DSpark is disabled or uninitialized")
    logits, aux = apply_model(
        params,
        config,
        input_ids,
        segment_ids=segment_ids,
        compute_indexer=False,
    )
    context = aux["dspark_context_features"]
    if context is None:
        raise ValueError("backbone produced no DSpark target features")
    draft = apply_dspark(
        params["dspark"],
        config,
        embed=params["embed"],
        lm_head=params["lm_head"],
        input_ids=input_ids,
        context_features=context,
        anchor_positions=anchor_positions,
        block_keep_mask=block_keep_mask,
        teacher_prev_ids=teacher_prev_ids,
    )
    return logits, aux, draft
