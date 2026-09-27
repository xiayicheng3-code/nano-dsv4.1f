from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .rope_ops import (
    learned_group_compress,
    linear,
    partial_rope,
    rope_kwargs,
    segment_local_positions,
)
from .sparse_attention import TorchCSA2State, latent_attention, run_indexer


@dataclass
class OwnerCache:
    """Persistent compressed-KV state for one Full/source layer."""

    state: TorchCSA2State
    pending_latent: torch.Tensor | None = None
    pending_gate: torch.Tensor | None = None
    pending_segment: torch.Tensor | None = None
    pending_position: torch.Tensor | None = None


@dataclass
class NanoDecodeCache:
    """Unpaged correctness cache that mirrors the V4.1 serving state machine."""

    input_ids: torch.Tensor
    segment_ids: torch.Tensor
    local_kv: dict[int, torch.Tensor] = field(default_factory=dict)
    owners: dict[int, OwnerCache] = field(default_factory=dict)

    @classmethod
    def empty(cls, *, device: torch.device) -> "NanoDecodeCache":
        return cls(
            input_ids=torch.empty((1, 0), dtype=torch.long, device=device),
            segment_ids=torch.empty((1, 0), dtype=torch.long, device=device),
        )

    @property
    def length(self) -> int:
        return int(self.input_ids.shape[1])


def _empty_owner_state(
    model: Any,
    *,
    layer_id: int,
    compression_ratio: int,
    compute_indexer: bool,
) -> TorchCSA2State:
    ac = model.config.attention
    ic = model.config.indexer
    device = model.device
    dtype = model.dtype
    prefix = f"blocks.{layer_id}.attn"
    owns_index_k = (
        compute_indexer
        and f"nano.{prefix}.indexer.wk.weight" in model.weights
    )
    return TorchCSA2State(
        kv=torch.empty((1, 0, ac.head_dim), device=device, dtype=dtype),
        latent=torch.empty((1, 0, ac.head_dim), device=device, dtype=dtype),
        index_k=(
            torch.empty((1, 0, ic.head_dim), device=device, dtype=dtype)
            if owns_index_k
            else None
        ),
        segment_ids=torch.empty((1, 0), device=device, dtype=torch.long),
        group_start_positions=torch.empty(
            (1, 0), device=device, dtype=torch.long
        ),
        source_layer=layer_id,
        compress_ratio=compression_ratio,
    )


def _append_tensor(old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
    return torch.cat((old, new), dim=1)


def _append_owner_group(
    model: Any,
    owner: OwnerCache,
    latent: torch.Tensor,
    segment: torch.Tensor,
    position: torch.Tensor,
    *,
    layer_id: int,
    compression_ratio: int,
    compute_indexer: bool,
) -> None:
    prefix = f"blocks.{layer_id}.attn"
    latent = model._norm(latent, f"{prefix}.global_norm")
    kwargs = rope_kwargs(model.config, compression_ratio)
    kv = partial_rope(
        latent,
        position,
        rotary_dim=model.config.attention.rope.rope_head_dim,
        **kwargs,
    )
    owner.state.latent = _append_tensor(owner.state.latent, latent)
    owner.state.kv = _append_tensor(owner.state.kv, kv)
    owner.state.segment_ids = _append_tensor(owner.state.segment_ids, segment)
    owner.state.group_start_positions = _append_tensor(
        owner.state.group_start_positions, position
    )

    if compute_indexer and owner.state.index_k is not None:
        index_k = linear(latent, model._w(f"{prefix}.indexer.wk.weight"))
        index_k = model._norm(index_k, f"{prefix}.indexer.k_norm")
        index_k = partial_rope(
            index_k,
            position,
            rotary_dim=model.config.attention.rope.rope_head_dim,
            **kwargs,
        )
        owner.state.index_k = _append_tensor(owner.state.index_k, index_k)


def update_owner_cache(
    model: Any,
    cache: NanoDecodeCache,
    source: torch.Tensor,
    *,
    layer_id: int,
    compression_ratio: int,
    segment: torch.Tensor,
    position: torch.Tensor,
    compute_indexer: bool,
) -> TorchCSA2State:
    """Append one source token, completing a ratio-2 group only when causal."""
    prefix = f"blocks.{layer_id}.attn"
    owner = cache.owners.get(layer_id)
    if owner is None:
        owner = OwnerCache(
            state=_empty_owner_state(
                model,
                layer_id=layer_id,
                compression_ratio=compression_ratio,
                compute_indexer=compute_indexer,
            )
        )
        cache.owners[layer_id] = owner
    elif compute_indexer and owner.state.index_k is None:
        # A cache created in dense mode cannot later reconstruct historical index K.
        raise ValueError(
            "decode cache was created without index-K; start a fresh cache for sparse retrieval"
        )

    # Full/Reindex/Reuse metadata belongs to the current query, not the next token.
    owner.state.latest_topk_indices = None
    owner.state.latest_topk_values = None
    owner.state.index_source_layer = -1
    owner.state.candidate_mask = None

    latent = linear(source, model._w(f"{prefix}.global_kv.weight"))
    if compression_ratio == 1:
        _append_owner_group(
            model,
            owner,
            latent,
            segment,
            position,
            layer_id=layer_id,
            compression_ratio=compression_ratio,
            compute_indexer=compute_indexer,
        )
        return owner.state
    if compression_ratio != 2:
        raise ValueError("incremental CPU cache supports compression ratio 1 or 2")

    gate = linear(source, model._w(f"{prefix}.global_gate.weight"))
    if owner.pending_latent is None:
        owner.pending_latent = latent
        owner.pending_gate = gate
        owner.pending_segment = segment
        owner.pending_position = position
        return owner.state

    assert owner.pending_gate is not None
    assert owner.pending_segment is not None
    assert owner.pending_position is not None
    if not torch.equal(owner.pending_segment, segment):
        raise ValueError(
            "ratio-2 packed decode requires segment boundaries aligned to compression groups"
        )
    pair_latent = torch.cat((owner.pending_latent, latent), dim=1)
    pair_gate = torch.cat((owner.pending_gate, gate), dim=1)
    compressed = learned_group_compress(pair_latent, pair_gate, ratio=2)
    _append_owner_group(
        model,
        owner,
        compressed,
        owner.pending_segment,
        owner.pending_position,
        layer_id=layer_id,
        compression_ratio=2,
        compute_indexer=compute_indexer,
    )
    owner.pending_latent = None
    owner.pending_gate = None
    owner.pending_segment = None
    owner.pending_position = None
    return owner.state


def _current_global_valid(
    state: TorchCSA2State,
    segment: torch.Tensor,
    position: torch.Tensor,
    local_window: int,
) -> torch.Tensor:
    same = segment.unsqueeze(-1) == state.segment_ids.unsqueeze(1)
    old_enough = state.group_start_positions.unsqueeze(1) <= (
        position.unsqueeze(-1) - local_window
    )
    return same & old_enough


def _reuse_mask(
    state: TorchCSA2State,
    valid: torch.Tensor,
) -> torch.Tensor | None:
    if state.latest_topk_indices is None or state.latest_topk_values is None:
        return None
    selected = torch.zeros_like(valid)
    selected.scatter_(
        -1,
        state.latest_topk_indices,
        torch.isfinite(state.latest_topk_values),
    )
    return selected


def attention_step(
    model: Any,
    cache: NanoDecodeCache,
    x: torch.Tensor,
    *,
    layer_id: int,
    mode: str,
    owns_global_kv: bool,
    compression_ratio: int,
    state: TorchCSA2State | None,
    segment: torch.Tensor,
    position: torch.Tensor,
    global_source: torch.Tensor | None,
    compute_indexer: bool,
    sparse_retrieval: bool,
) -> tuple[torch.Tensor, TorchCSA2State | None, dict[str, Any]]:
    ac = model.config.attention
    prefix = f"blocks.{layer_id}.attn"
    kwargs = rope_kwargs(model.config, compression_ratio)

    qr = model._norm(
        linear(x, model._w(f"{prefix}.q_a.weight")), f"{prefix}.q_norm"
    )
    q = linear(qr, model._w(f"{prefix}.q_b.weight")).reshape(
        *x.shape[:-1], ac.n_heads, ac.head_dim
    )
    q = partial_rope(
        q, position, rotary_dim=ac.rope.rope_head_dim, **kwargs
    )

    local_kv = model._norm(
        linear(x, model._w(f"{prefix}.local_kv.weight")),
        f"{prefix}.local_kv_norm",
    )
    local_kv = partial_rope(
        local_kv, position, rotary_dim=ac.rope.rope_head_dim, **kwargs
    )
    previous = cache.local_kv.get(layer_id)
    cache.local_kv[layer_id] = (
        local_kv if previous is None else _append_tensor(previous, local_kv)
    )
    history_kv = cache.local_kv[layer_id]
    history_segment = cache.segment_ids
    history_position = segment_local_positions(history_segment)
    local_valid = (
        (history_segment.unsqueeze(1) == segment.unsqueeze(-1))
        & (history_position.unsqueeze(1) <= position.unsqueeze(-1))
        & (
            history_position.unsqueeze(1)
            >= position.unsqueeze(-1) - (ac.local_window - 1)
        )
    )
    local_out, local_lse = latent_attention(q, history_kv, local_valid)

    global_out = None
    global_lse = None
    global_valid = None
    retrieval_mask = None
    if mode != "swa":
        if owns_global_kv:
            source = x if global_source is None else global_source
            state = update_owner_cache(
                model,
                cache,
                source,
                layer_id=layer_id,
                compression_ratio=compression_ratio,
                segment=segment,
                position=position,
                compute_indexer=compute_indexer,
            )
        elif global_source is not None:
            raise ValueError("global_source only applies to a compressed-KV source")
        if state is None:
            raise ValueError(f"CSA2 {mode} layer {layer_id} has no shared state")

        global_valid = _current_global_valid(
            state, segment, position, ac.local_window
        )
        if (
            compute_indexer
            and mode in ("full", "reindex")
            and state.index_k is not None
        ):
            state, retrieval_mask = run_indexer(
                model,
                qr,
                x,
                position,
                global_valid,
                layer_id,
                compression_ratio,
                state,
            )
        elif mode == "reuse":
            retrieval_mask = _reuse_mask(state, global_valid)

        attn_valid = (
            global_valid & retrieval_mask
            if sparse_retrieval and retrieval_mask is not None
            else global_valid
        )
        global_out, global_lse = latent_attention(q, state.kv, attn_valid)

    if global_out is None:
        merged, branch_lse = local_out, local_lse
    else:
        assert global_lse is not None
        branch_lse = torch.logaddexp(local_lse, global_lse)
        local_weight = torch.nan_to_num(
            torch.exp(local_lse - branch_lse)
        ).unsqueeze(-1)
        global_weight = torch.nan_to_num(
            torch.exp(global_lse - branch_lse)
        ).unsqueeze(-1)
        merged = local_weight * local_out + global_weight * global_out

    sink_key = f"nano.{prefix}.attn_sink"
    if sink_key in model.weights:
        sink = model._w(f"{prefix}.attn_sink").float()[None, None, :]
        total_lse = torch.logaddexp(branch_lse, sink)
        merged = merged * torch.nan_to_num(
            torch.exp(branch_lse - total_lse)
        ).unsqueeze(-1)
    else:
        total_lse = branch_lse

    merged = partial_rope(
        merged,
        position,
        rotary_dim=ac.rope.rope_head_dim,
        inverse=True,
        **kwargs,
    )
    heads_per_group = ac.n_heads // ac.o_groups
    grouped = merged.reshape(
        *merged.shape[:-2],
        ac.o_groups,
        heads_per_group * ac.head_dim,
    )
    low_rank = torch.einsum(
        "...gd,gdr->...gr", grouped, model._w(f"{prefix}.wo_a")
    )
    out = linear(
        low_rank.reshape(
            *low_rank.shape[:-2], ac.o_groups * ac.o_rank
        ),
        model._w(f"{prefix}.wo_b.weight"),
    )
    return out, state, {
        "total_lse": total_lse,
        "global_lse": global_lse,
        "global_valid": global_valid,
        "retrieval_mask": retrieval_mask,
    }
