from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp

from .config import ModelConfig
from .csa2 import SharedCSA2State, apply_csa2_attention, init_csa2_attention
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
    mode: Literal["full", "reindex", "reuse"]
    owns_global_kv: bool
    compression_ratio: int
    has_engram: bool


def build_layer_specs(config: ModelConfig) -> tuple[LayerSpec, ...]:
    """Build the small architecture that preserves the CED/CSA2 state transitions.

    Default 4+4 layout:

        context:    Full -> Reuse -> Full -> Reuse
        generation: Full -> Reuse -> Reindex -> Reuse

    The first generation Full layer overwrites the shared global-KV state using the
    final context-side representation. Later generation layers reuse that same bank;
    Reindex changes retrieval scoring, not main KV ownership.
    """
    specs: list[LayerSpec] = []
    group = config.csa2.retriever_group_size

    for i in range(config.csa2.context_layers):
        mode = "full" if i % group == 0 else "reuse"
        specs.append(
            LayerSpec(
                layer_id=i,
                half="context",
                mode=mode,
                owns_global_kv=mode == "full",
                compression_ratio=config.csa2.context_compression_ratio,
                has_engram=i in config.engram.layer_ids,
            )
        )

    base = config.csa2.context_layers
    for j in range(config.csa2.generation_layers):
        layer_id = base + j
        if j == 0:
            mode, owns = "full", True
        elif j % group == 0:
            mode, owns = "reindex", False
        else:
            mode, owns = "reuse", False
        specs.append(
            LayerSpec(
                layer_id=layer_id,
                half="generation",
                mode=mode,
                owns_global_kv=owns,
                compression_ratio=config.csa2.generation_compression_ratio,
                has_engram=layer_id in config.engram.layer_ids,
            )
        )
    return tuple(specs)


def _init_block(
    key: jax.Array,
    config: ModelConfig,
    spec: LayerSpec,
) -> dict[str, object]:
    keys = iter(jax.random.split(key, 8))
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
            n_heads=config.attention.n_heads,
            head_dim=config.attention.head_dim,
            q_rank=config.attention.q_rank,
            o_rank=config.attention.o_rank,
            owns_global_kv=spec.owns_global_kv,
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
    keys = jax.random.split(key, len(specs) + 3)
    return {
        "embed": init_embedding(keys[0], config.vocab_size, config.d_model),
        "blocks": tuple(
            _init_block(keys[i + 1], config, spec)
            for i, spec in enumerate(specs)
        ),
        "final_norm": init_rms_norm(config.d_model),
        "lm_head": init_embedding(
            keys[-1], config.vocab_size, config.d_model
        ).T,
    }


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
) -> tuple[
    jax.Array,
    jax.Array,
    SharedCSA2State,
    dict[str, jax.Array],
]:
    """One Single-Pass-mHC Transformer block.

    This keeps the released cross-sublayer mHC timing: attention consumes the previous
    FFN's pre-mix; this attention's newly generated pre-mix is consumed by the FFN; and
    this FFN's pre-mix is returned for the next block's attention.
    """
    residual = streams
    attn_pre, attn_post, attn_comb = mhc_mixes(
        streams,
        params["mhc_attn"],
        sinkhorn_iters=config.mhc_sinkhorn_iters,
        eps=config.norm_eps,
    )
    attn_input = rms_norm(
        pre_mix(streams, incoming_pre_mix),
        params["attn_norm"],
        eps=config.norm_eps,
    )
    attn_out, state, attn_aux = apply_csa2_attention(
        attn_input,
        segment_ids,
        params["attn"],
        state,
        layer_id=spec.layer_id,
        mode=spec.mode,
        owns_global_kv=spec.owns_global_kv,
        compression_ratio=spec.compression_ratio,
        n_heads=config.attention.n_heads,
        head_dim=config.attention.head_dim,
        local_window=config.attention.local_window,
        norm_eps=config.norm_eps,
        global_source=global_source,
    )
    streams = post_mix(residual, attn_out, attn_comb, attn_post)

    residual = streams
    ffn_pre, ffn_post, ffn_comb = mhc_mixes(
        streams,
        params["mhc_ffn"],
        sinkhorn_iters=config.mhc_sinkhorn_iters,
        eps=config.norm_eps,
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
) -> tuple[jax.Array, dict[str, object]]:
    """Run the readable dense-training reference model.

    `segment_ids` supports packed examples. For r=2 source groups, every packed segment
    must begin/end on an even physical token boundary; the data pipeline owns that padding
    invariant. Sparse Top-K is intentionally not used by this backbone forward yet.
    """
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, tokens]")
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
    layer_aux = []
    context_final = None

    for spec, block in zip(specs, params["blocks"]):
        generation_global_source = None
        if spec.half == "generation" and context_final is None:
            # Snapshot exactly once at the CED boundary. This is the source consumed by
            # the first generation-side Full layer's shared global-KV projection.
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
                eps=config.norm_eps,
                token_mask=token_mask,
            )

        streams, incoming_pre, state, aux = _apply_block(
            streams,
            incoming_pre,
            segment_ids,
            block,
            state,
            config,
            spec,
            global_source=generation_global_source,
        )
        layer_aux.append(aux)

    hidden = rms_norm(
        pre_mix(streams, incoming_pre),
        params["final_norm"],
        eps=config.norm_eps,
    )
    logits = jnp.einsum("btd,dv->btv", hidden, params["lm_head"])
    return logits, {
        "context_final": context_final,
        "final_hidden": hidden,
        "final_global_source_layer": state.source_layer,
        "layers": tuple(layer_aux),
    }
