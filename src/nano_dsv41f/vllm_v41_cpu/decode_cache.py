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
    transaction: Any = field(default=None, repr=False)

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

    def append_tokens(self, tokens, segments):
        start, size = self.length, tokens.shape[1]
        if start + size > self.capacity:
            raise ValueError("cache capacity exceeded; reset or shorten the conversation")
        positions = segment_local_positions(segments)
        if start:
            # Add the previous position only to the leading continuing segment.
            continuing = (segments == self.segment_buffer[:, start-1:start]).long().cumprod(1)
            positions = positions + continuing * (self.position_buffer[:, start-1:start] + 1)
        self.input_buffer[:, start:start+size].copy_(tokens)
        self.segment_buffer[:, start:start+size].copy_(segments)
        self.position_buffer[:, start:start+size].copy_(positions)
        self.length += size
        return positions

    def append_token(self, token, segment):
        return self.append_tokens(token, segment)

    def write_local(self, layer, kv, segment, position, window, start):
        if layer not in self.local_kv:
            self.local_kv[layer] = kv.new_empty((1, window, kv.shape[-1]))
            self.local_segments[layer] = next(iter(self.local_segments.values())) if self.local_segments else segment.new_empty((1, window))
            self.local_positions[layer] = next(iter(self.local_positions.values())) if self.local_positions else position.new_empty((1, window))
        size = kv.shape[1]
        if not size:
            return
        # Retain the last window with unique ring indices even for large chunks.
        skip = max(0, size - window)
        slots = torch.arange(start + skip, start + size, device=kv.device) % window
        self.local_kv[layer].index_copy_(1, slots, kv[:, skip:])
        self.local_segments[layer].index_copy_(1, slots, segment[:, skip:])
        self.local_positions[layer].index_copy_(1, slots, position[:, skip:])
        self.local_metadata_length = self.length

    def append_local(self, layer, kv, segment, position, window):
        start = self.length - kv.shape[1]
        if start:
            # Read old metadata from the token arena: rings share metadata, so an
            # earlier layer may already have overwritten its slots in this chunk.
            indices = torch.arange(max(0, start-window+1), start, device=kv.device)
            old_kv = self.local_kv[layer].index_select(1, indices % window)
            history = (torch.cat((old_kv, kv), 1),
                torch.cat((self.segment_buffer.index_select(1, indices), segment), 1),
                torch.cat((self.position_buffer.index_select(1, indices), position), 1))
        else:
            history = (kv, segment, position)
        if self.transaction is not None:
            self.transaction.locals[layer] = (kv, segment, position, window)
        self.write_local(layer, kv, segment, position, window, start)
        return history

    def write_draft(self, kv, position, window, start):
        if self.draft_kv is None:
            self.draft_kv = kv.new_empty((1, window, kv.shape[-1]))
            self.draft_positions = position.new_empty((1, window))
        size = kv.shape[1]
        if size:
            skip = max(0, size-window)
            slots = torch.arange(start+skip, start+size, device=kv.device) % window
            self.draft_kv.index_copy_(1, slots, kv[:, skip:])
            self.draft_positions.index_copy_(1, slots, position[:, skip:])

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
    """Append a source chunk, retaining an unfinished compression group."""
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

    owner.state.latest_topk_indices = None
    owner.state.latest_topk_values = None
    owner.state.index_source_layer = -1
    owner.state.candidate_mask = None
    latent = linear(source, model._w(f"{prefix}.global_kv.weight"))
    gate = None
    pending = int(owner.pending_latent is not None)
    if compression_ratio == 2:
        gate = linear(source, model._w(f"{prefix}.global_gate.weight"))
        if pending:
            latent = torch.cat((owner.pending_latent, latent), 1)
            gate = torch.cat((owner.pending_gate, gate), 1)
            segment = torch.cat((owner.pending_segment, segment), 1)
            position = torch.cat((owner.pending_position, position), 1)
    elif compression_ratio != 1:
        raise ValueError("incremental cache supports compression ratio 1 or 2")
    complete = latent.shape[1] // compression_ratio * compression_ratio
    if compression_ratio == 2 and complete:
        if not torch.equal(segment[:, :complete:2], segment[:, 1:complete:2]):
            raise ValueError("ratio-2 packed decode requires segment boundaries aligned to compression groups")
    if cache.transaction is not None:
        cache.transaction.owners[layer_id] = (owner.state.kv.shape[1], pending,
            latent, gate, segment, position, compression_ratio)
    if complete:
        compressed = (learned_group_compress(latent[:, :complete], gate[:, :complete], 2)
                      if compression_ratio == 2 else latent)
        _append_owner_group(model, owner, compressed, segment[:, :complete:compression_ratio],
            position[:, :complete:compression_ratio], layer_id=layer_id,
            compression_ratio=compression_ratio, compute_indexer=compute_indexer)
    for name, value in (("latent", latent), ("gate", gate), ("segment", segment), ("position", position)):
        setattr(owner, "pending_"+name,
                value[:, -1:].clone() if complete < latent.shape[1] and value is not None else None)
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
            indices = state.latest_topk_indices
            batch, queries, count = indices.shape
            selected_kv = state.kv[:, None].expand(-1, queries, -1, -1).gather(
                2, indices[..., None].expand(-1, -1, -1, state.kv.shape[-1]))
            selected_valid = attn_valid.gather(-1, indices)
            global_out, global_lse = latent_attention(
                q.reshape(batch*queries, 1, ac.n_heads, ac.head_dim),
                selected_kv.reshape(batch*queries, count, ac.head_dim),
                selected_valid.reshape(batch*queries, 1, count))
            global_out = global_out.reshape(batch, queries, ac.n_heads, ac.head_dim)
            global_lse = global_lse.reshape(batch, queries, ac.n_heads)

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


class CacheTransaction:
    """Speculate once, then retain a prefix without recomputing the transformer.

    Global buffers are append-only. Only bounded rings/pending compression state
    need copying; journal entries hold the new chunk projections, not history.
    """
    def __init__(self, cache):
        if cache.transaction is not None or cache.length == 0:
            raise ValueError("speculation requires a non-empty cache and no active transaction")
        self.cache = cache
        self.saved = cache.snapshot()
        self.locals = {}
        self.owners = {}
        self.draft = None
        cache.transaction = self

    @torch.inference_mode()
    def commit(self, count, logits):
        cache, start = self.cache, self.saved['length']
        size = cache.length - start
        if cache.transaction is not self or not 0 <= count <= size:
            raise ValueError("invalid speculative commit")
        cache.transaction = None
        if count == size:
            return
        cache.restore(self.saved)
        cache.length = start + count
        if count:
            for layer, (kv, segment, position, window) in self.locals.items():
                cache.write_local(layer, kv[:, :count], segment[:, :count], position[:, :count], window, start)
            for layer, (old, pending, latent, gate, segment, position, ratio) in self.owners.items():
                owner = cache.owners[layer]
                rows = pending + count
                groups = rows // ratio
                for name, buf in owner.buffers.items():
                    setattr(owner.state, name, buf[:, :old+groups])
                for name, value in (("latent", latent), ("gate", gate), ("segment", segment), ("position", position)):
                    setattr(owner, "pending_"+name,
                        value[:, rows-1:rows].clone() if rows % ratio and value is not None else None)
            if self.draft is not None:
                kv, position, window = self.draft
                cache.write_draft(kv[:, :count], position[:, :count], window, start)
            cache.next_logits = logits[:, count-1].float().clone()
        # Discard query-specific selections from the speculative suffix.
        for owner in cache.owners.values():
            owner.state.latest_topk_indices = owner.state.latest_topk_values = None
            owner.state.candidate_mask = None
            owner.state.index_source_layer = -1
