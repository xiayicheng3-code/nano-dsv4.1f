from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .rope_ops import (
    learned_group_compress,
    linear,
    partial_rope,
    rope_kwargs,
    segment_local_positions,
)


@dataclass
class TorchCSA2State:
    kv: torch.Tensor
    latent: torch.Tensor
    index_k: torch.Tensor | None
    segment_ids: torch.Tensor
    group_start_positions: torch.Tensor
    source_layer: int
    compress_ratio: int
    latest_topk_indices: torch.Tensor | None = None
    latest_topk_values: torch.Tensor | None = None
    index_source_layer: int = -1
    candidate_mask: torch.Tensor | None = None


def latent_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, qlen, n_heads, head_dim = q.shape
    if kv.shape[-2] == 0:
        out = q.new_zeros((bsz, qlen, n_heads, head_dim))
        lse = torch.full(
            (bsz, qlen, n_heads),
            -torch.inf,
            dtype=torch.float32,
            device=q.device,
        )
        return out, lse

    logits = torch.einsum(
        "bthd,bsd->bhts", q.float(), kv.float()
    ) * (head_dim**-0.5)
    valid = mask.unsqueeze(1)
    any_valid = mask.any(dim=-1)
    masked = torch.where(valid, logits, torch.full_like(logits, -1e30))
    probs = torch.softmax(masked, dim=-1)
    probs = torch.where(
        any_valid[:, None, :, None], probs, torch.zeros_like(probs)
    )
    out = torch.einsum("bhts,bsd->bthd", probs.to(kv.dtype), kv)
    lse = torch.logsumexp(masked, dim=-1)
    lse = torch.where(
        any_valid[:, None, :], lse, torch.full_like(lse, -torch.inf)
    )
    return out.to(q.dtype), lse.transpose(1, 2)


def local_mask(segment_ids: torch.Tensor, local_window: int) -> torch.Tensor:
    pos = segment_local_positions(segment_ids)
    same = segment_ids.unsqueeze(-1) == segment_ids.unsqueeze(-2)
    causal = pos.unsqueeze(-2) <= pos.unsqueeze(-1)
    recent = pos.unsqueeze(-2) >= pos.unsqueeze(-1) - (local_window - 1)
    return same & causal & recent


def global_mask(
    segment_ids: torch.Tensor,
    state: TorchCSA2State,
    local_window: int,
) -> torch.Tensor:
    pos = segment_local_positions(segment_ids)
    same = segment_ids.unsqueeze(-1) == state.segment_ids.unsqueeze(-2)
    old_enough = state.group_start_positions.unsqueeze(-2) <= (
        pos.unsqueeze(-1) - local_window
    )
    return same & old_enough


def build_global_state(
    model: Any,
    source: torch.Tensor,
    segment_ids: torch.Tensor,
    layer_id: int,
    compression_ratio: int,
    *,
    compute_indexer: bool,
) -> TorchCSA2State:
    prefix = f"blocks.{layer_id}.attn"
    latent = linear(source, model._w(f"{prefix}.global_kv.weight"))
    complete = source.shape[-2]
    if compression_ratio > 1:
        gate = linear(source, model._w(f"{prefix}.global_gate.weight"))
        latent = learned_group_compress(latent, gate, compression_ratio)
        complete = (source.shape[-2] // compression_ratio) * compression_ratio
    latent = model._norm(latent, f"{prefix}.global_norm")

    pos = segment_local_positions(segment_ids)
    group_pos = pos[:, :complete:compression_ratio]
    group_segments = segment_ids[:, :complete:compression_ratio]
    kwargs = rope_kwargs(model.config, compression_ratio)
    kv = partial_rope(
        latent,
        group_pos,
        rotary_dim=model.config.attention.rope.rope_head_dim,
        **kwargs,
    )

    index_k = None
    wk_key = f"nano.{prefix}.indexer.wk.weight"
    if compute_indexer and wk_key in model.weights:
        index_k = linear(latent, model._w(f"{prefix}.indexer.wk.weight"))
        index_k = model._norm(index_k, f"{prefix}.indexer.k_norm")
        index_k = partial_rope(
            index_k,
            group_pos,
            rotary_dim=model.config.attention.rope.rope_head_dim,
            **kwargs,
        )

    return TorchCSA2State(
        kv=kv,
        latent=latent,
        index_k=index_k,
        segment_ids=group_segments,
        group_start_positions=group_pos,
        source_layer=layer_id,
        compress_ratio=compression_ratio,
    )


def indexer_scores(
    model: Any,
    qr: torch.Tensor,
    hidden: torch.Tensor,
    q_pos: torch.Tensor,
    layer_id: int,
    state: TorchCSA2State,
    compression_ratio: int,
) -> torch.Tensor:
    if state.index_k is None:
        raise ValueError("index-source layer requires shared index K")
    prefix = f"blocks.{layer_id}.attn.indexer"
    ic = model.config.indexer
    q = linear(qr, model._w(f"{prefix}.wq_b.weight")).reshape(
        *qr.shape[:-1], ic.n_heads, ic.head_dim
    )
    q = partial_rope(
        q,
        q_pos,
        rotary_dim=model.config.attention.rope.rope_head_dim,
        **rope_kwargs(model.config, compression_ratio),
    )
    weights = linear(
        hidden.float(), model._w(f"{prefix}.weights_proj.weight")
    )
    weights = weights * (ic.head_dim**-0.5 * ic.n_heads**-0.5)
    score = torch.einsum(
        "...qhd,...kd->...qhk", q.float(), state.index_k.float()
    )
    return torch.einsum(
        "...qhk,...qh->...qk", F.relu(score), weights.float()
    )


def hierarchical_candidate_mask(
    model: Any,
    scores: torch.Tensor,
    valid: torch.Tensor,
    state: TorchCSA2State,
    compression_ratio: int,
) -> torch.Tensor:
    ic = model.config.indexer
    n_k = scores.shape[-1]
    if n_k == 0:
        return valid
    n_blocks = (n_k + ic.candidate_block_size - 1) // ic.candidate_block_size
    block_ids = (
        state.group_start_positions
        // compression_ratio
        // ic.candidate_block_size
    ).clamp(0, n_blocks - 1)
    masked = torch.where(valid, scores, torch.full_like(scores, -torch.inf))
    scatter_ids = block_ids[:, None, :].expand_as(masked)
    block_scores = torch.full(
        (*scores.shape[:-1], n_blocks),
        -torch.inf,
        device=scores.device,
        dtype=scores.dtype,
    )
    block_scores.scatter_reduce_(
        2, scatter_ids, masked, reduce="amax", include_self=True
    )
    newest = torch.where(
        valid, scatter_ids, torch.full_like(scatter_ids, -1)
    ).amax(dim=-1)
    newest = newest.clamp(min=0)
    block_scores.scatter_(-1, newest.unsqueeze(-1), torch.inf)

    k = min(ic.candidate_topk_blocks, n_blocks)
    selected_ids = torch.topk(block_scores, k=k, dim=-1).indices
    selected_blocks = torch.zeros_like(block_scores, dtype=torch.bool)
    selected_blocks.scatter_(-1, selected_ids, True)
    selected_blocks.scatter_(-1, newest.unsqueeze(-1), True)
    candidate = selected_blocks.gather(-1, scatter_ids)
    return candidate & valid


def run_indexer(
    model: Any,
    qr: torch.Tensor,
    hidden: torch.Tensor,
    q_pos: torch.Tensor,
    valid: torch.Tensor,
    layer_id: int,
    compression_ratio: int,
    state: TorchCSA2State,
) -> tuple[TorchCSA2State, torch.Tensor | None]:
    if valid.shape[-1] == 0:
        return state, None
    scores = indexer_scores(
        model, qr, hidden, q_pos, layer_id, state, compression_ratio
    )
    candidate = state.candidate_mask
    scoring_valid = valid
    if (
        candidate is not None
        and layer_id != model.config.indexer.candidate_source_layer
    ):
        scoring_valid = scoring_valid & candidate
    masked = torch.where(
        scoring_valid, scores, torch.full_like(scores, -torch.inf)
    )
    k = min(model.config.indexer.top_k, scores.shape[-1])
    values, indices = torch.topk(masked, k=k, dim=-1)

    if layer_id == model.config.indexer.candidate_source_layer:
        candidate = hierarchical_candidate_mask(
            model, scores, valid, state, compression_ratio
        )
    state.latest_topk_indices = indices
    state.latest_topk_values = values
    state.index_source_layer = layer_id
    state.candidate_mask = candidate

    selected = torch.zeros_like(valid)
    selected.scatter_(-1, indices, torch.isfinite(values))
    return state, selected


def attention_forward(
    model: Any,
    x: torch.Tensor,
    segment_ids: torch.Tensor,
    layer_id: int,
    mode: str,
    owns_global_kv: bool,
    compression_ratio: int,
    state: TorchCSA2State | None,
    *,
    global_source: torch.Tensor | None,
    compute_indexer: bool,
    sparse_retrieval: bool,
) -> tuple[torch.Tensor, TorchCSA2State | None, dict[str, Any]]:
    ac = model.config.attention
    prefix = f"blocks.{layer_id}.attn"
    q_pos = segment_local_positions(segment_ids)
    kwargs = rope_kwargs(model.config, compression_ratio)

    qr = model._norm(
        linear(x, model._w(f"{prefix}.q_a.weight")), f"{prefix}.q_norm"
    )
    q = linear(qr, model._w(f"{prefix}.q_b.weight")).reshape(
        *x.shape[:-1], ac.n_heads, ac.head_dim
    )
    q = partial_rope(
        q, q_pos, rotary_dim=ac.rope.rope_head_dim, **kwargs
    )

    local_kv = model._norm(
        linear(x, model._w(f"{prefix}.local_kv.weight")),
        f"{prefix}.local_kv_norm",
    )
    local_kv = partial_rope(
        local_kv, q_pos, rotary_dim=ac.rope.rope_head_dim, **kwargs
    )
    local_out, local_lse = latent_attention(
        q, local_kv, local_mask(segment_ids, ac.local_window)
    )

    global_out = None
    global_lse = None
    global_valid = None
    retrieval_mask = None
    if mode != "swa":
        if owns_global_kv:
            source = x if global_source is None else global_source
            state = build_global_state(
                model,
                source,
                segment_ids,
                layer_id,
                compression_ratio,
                compute_indexer=compute_indexer,
            )
        elif global_source is not None:
            raise ValueError("global_source only applies to a compressed-KV source")
        if state is None:
            raise ValueError(f"CSA2 {mode} layer {layer_id} has no shared state")

        global_valid = global_mask(segment_ids, state, ac.local_window)
        if (
            compute_indexer
            and mode in ("full", "reindex")
            and state.index_k is not None
        ):
            state, retrieval_mask = run_indexer(
                model,
                qr,
                x,
                q_pos,
                global_valid,
                layer_id,
                compression_ratio,
                state,
            )
        elif mode == "reuse" and state.latest_topk_indices is not None:
            retrieval_mask = torch.zeros_like(global_valid)
            finite = torch.isfinite(state.latest_topk_values)
            retrieval_mask.scatter_(-1, state.latest_topk_indices, finite)

        attn_mask = (
            global_valid & retrieval_mask
            if sparse_retrieval and retrieval_mask is not None
            else global_valid
        )
        global_out, global_lse = latent_attention(q, state.kv, attn_mask)

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
        q_pos,
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
