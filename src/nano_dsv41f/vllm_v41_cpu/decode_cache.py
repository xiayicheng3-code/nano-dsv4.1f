from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .rope_ops import (
    learned_group_compress,
    linear,
    partial_rope,
    rope_kwargs,
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
    buffers: dict[str, torch.Tensor] = field(default_factory=dict)
    capacity: int = 0

    def append(self, name: str, value: torch.Tensor) -> None:
        old = getattr(self.state, name)
        start = old.shape[1]
        if start + value.shape[1] > self.capacity:
            raise ValueError("compressed cache capacity exceeded")
        if name not in self.buffers:
            self.buffers[name] = value.new_empty((1, self.capacity, *value.shape[2:]))
        buf = self.buffers[name]
        buf.narrow(1, start, value.shape[1]).copy_(value)
        setattr(self.state, name, buf.narrow(1, 0, start + value.shape[1]))


@dataclass
class NanoDecodeCache:
    """Single-sequence arena: fixed global capacity and bounded local rings.

    Views grow logically; their backing storage never grows. This is not a Mamba
    recurrence: compressed global attention still stores O(capacity) state.
    """

    input_buffer: torch.Tensor
    segment_buffer: torch.Tensor
    position_buffer: torch.Tensor
    capacity: int
    length: int = 0
    local_kv: dict[int, torch.Tensor] = field(default_factory=dict)
    local_segments: dict[int, torch.Tensor] = field(default_factory=dict)
    local_positions: dict[int, torch.Tensor] = field(default_factory=dict)
    local_metadata_length: int = -1
    owners: dict[int, OwnerCache] = field(default_factory=dict)
    next_logits: torch.Tensor | None = None
    mode: tuple[bool, bool] | None = None
    draft_kv: torch.Tensor | None = None
    draft_positions: torch.Tensor | None = None
    collect_draft: bool = False

    @classmethod
    def empty(cls, *, device: torch.device, capacity: int = 32768) -> "NanoDecodeCache":
        if capacity < 1:
            raise ValueError("cache capacity must be positive")
        ids = torch.empty((1, capacity), dtype=torch.long, device=device)
        return cls(ids, torch.empty_like(ids), torch.empty_like(ids), capacity)

    @property
    def input_ids(self):
        return self.input_buffer.narrow(1, 0, self.length)

    @property
    def segment_ids(self):
        return self.segment_buffer.narrow(1, 0, self.length)

    def append_token(self, token, segment):
        if self.length >= self.capacity:
            raise ValueError("cache capacity exceeded; reset or shorten the conversation")
        i = self.length
        position = torch.zeros_like(segment) if not i else torch.where(
            segment == self.segment_buffer[:, i-1:i],
            self.position_buffer[:, i-1:i] + 1, 0)
        self.input_buffer[:, i:i+1].copy_(token)
        self.segment_buffer[:, i:i+1].copy_(segment)
        self.position_buffer[:, i:i+1].copy_(position)
        self.length += 1
        return position

    def append_local(self, layer, kv, segment, position, window):
        if layer not in self.local_kv:
            self.local_kv[layer] = kv.new_empty((1, window, kv.shape[-1]))
            # Metadata is identical for every local layer; share one allocation.
            self.local_segments[layer] = next(iter(self.local_segments.values())) if self.local_segments else segment.new_empty((1, window))
            self.local_positions[layer] = next(iter(self.local_positions.values())) if self.local_positions else position.new_empty((1, window))
        slot = (self.length - 1) % window
        self.local_kv[layer].narrow(1, slot, 1).copy_(kv)
        if self.local_metadata_length != self.length:
            self.local_segments[layer].narrow(1, slot, 1).copy_(segment)
            self.local_positions[layer].narrow(1, slot, 1).copy_(position)
            self.local_metadata_length = self.length
        n = min(self.length, window)
        return (self.local_kv[layer].narrow(1, 0, n), self.local_segments[layer].narrow(1, 0, n),
                self.local_positions[layer].narrow(1, 0, n))

    def snapshot(self):
        # Global KV is append-only. Save only lengths and the small rolling state.
        owners = {}
        for k, owner in self.owners.items():
            pending = {name: None if getattr(owner, name) is None else getattr(owner, name).clone()
                       for name in ("pending_latent", "pending_gate", "pending_segment", "pending_position")}
            owners[k] = (owner.state.kv.shape[1], pending)
        return dict(length=self.length, owners=owners,
                    rings={name: {k: v.clone() for k, v in getattr(self, name).items()}
                           for name in ("local_kv", "local_segments", "local_positions")},
                    next_logits=self.next_logits,
                    draft_kv=None if self.draft_kv is None else self.draft_kv.clone(),
                    draft_positions=None if self.draft_positions is None else self.draft_positions.clone())

    def restore(self, saved):
        self.length = saved["length"]
        self.local_metadata_length = self.length
        self.next_logits = saved["next_logits"]
        for name, rings in saved["rings"].items():
            for k, v in rings.items():
                getattr(self, name)[k].copy_(v)
        for k, (length, pending) in saved["owners"].items():
            owner = self.owners[k]
            for name, buf in owner.buffers.items():
                setattr(owner.state, name, buf[:, :length])
            for name, value in pending.items():
                setattr(owner, name, value)
        for name in ("draft_kv", "draft_positions"):
            if saved[name] is not None:
                getattr(self, name).copy_(saved[name])


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
    owner.append("latent", latent)
    owner.append("kv", kv)
    owner.append("segment_ids", segment)
    owner.append("group_start_positions", position)

    if compute_indexer and owner.state.index_k is not None:
        index_k = linear(latent, model._w(f"{prefix}.indexer.wk.weight"))
        index_k = model._norm(index_k, f"{prefix}.indexer.k_norm")
        index_k = partial_rope(
            index_k,
            position,
            rotary_dim=model.config.attention.rope.rope_head_dim,
            **kwargs,
        )
        owner.append("index_k", index_k)


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
            ),
            capacity=(cache.capacity + compression_ratio - 1) // compression_ratio,
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
    history_kv, history_segment, history_position = cache.append_local(
        layer_id, local_kv, segment, position, ac.local_window)
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
        # Only the selected global latents enter the expensive main-head attention.
        if sparse_retrieval and retrieval_mask is not None and state.latest_topk_indices is not None:
            indices = state.latest_topk_indices[:, 0]
            selected_kv = state.kv.gather(1, indices.unsqueeze(-1).expand(-1, -1, state.kv.shape[-1]))
            selected_valid = attn_valid.gather(-1, indices.unsqueeze(1))
            global_out, global_lse = latent_attention(q, selected_kv, selected_valid)
        else:
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
        "...gd,gdr->...gr", grouped.to(model.dtype), model._w(f"{prefix}.wo_a")
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
