from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .mhc_engram import ngram_hash_ids
from .rope_ops import linear


def _inject_engram(
    model: Any,
    streams: torch.Tensor,
    hashes: torch.Tensor,
    layer_id: int,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    prefix = f"blocks.{layer_id}.engram"
    looked_up = model._w(f"{prefix}.table")[hashes]
    flat = looked_up.reshape(*looked_up.shape[:-2], -1)
    kv = linear(flat, model._w(f"{prefix}.wkv"))

    n_streams, dim = streams.shape[-2:]
    key = kv[..., : n_streams * dim].reshape(
        *kv.shape[:-1], n_streams, dim
    )
    value = kv[..., n_streams * dim :]

    hidden = streams.float()
    key_fp32 = key.float()
    weight = (
        model._w(f"{prefix}.q_weight").float()
        * model._w(f"{prefix}.k_weight").float()
    )
    rstd = torch.rsqrt(
        hidden.square().mean(dim=-1) + model.config.mhc_eps
    ) * torch.rsqrt(
        key_fp32.square().mean(dim=-1) + model.config.mhc_eps
    )
    dot = (
        (hidden * weight * key_fp32).sum(dim=-1)
        * rstd
        * (dim**-0.5)
    )
    signed_sqrt = torch.sign(dot) * torch.sqrt(
        torch.clamp(dot.abs(), min=1e-6)
    )
    gate = torch.sigmoid(signed_sqrt)
    if token_mask is not None:
        gate = torch.where(
            token_mask.unsqueeze(-1), gate, torch.zeros_like(gate)
        )
    return streams + (
        gate.to(streams.dtype).unsqueeze(-1)
        * value.to(streams.dtype).unsqueeze(-2)
    )


def apply_engram(
    model: Any,
    streams: torch.Tensor,
    input_ids: torch.Tensor,
    segment_ids: torch.Tensor,
    layer_id: int,
    token_mask: torch.Tensor | None,
) -> torch.Tensor:
    ec = model.config.engram
    hashes = ngram_hash_ids(
        input_ids,
        segment_ids,
        table_size=ec.table_size,
        max_ngram_size=ec.max_ngram_size,
        n_hash_heads=ec.n_hash_heads,
        pad_token_id=ec.pad_token_id,
        seed=layer_id * 97,
    )
    return _inject_engram(
        model, streams, hashes, layer_id, token_mask=token_mask
    )


def apply_engram_step(
    model: Any,
    streams: torch.Tensor,
    input_ids_history: torch.Tensor,
    segment_ids_history: torch.Tensor,
    layer_id: int,
) -> torch.Tensor:
    """Inject Engram memory for only the newest autoregressive token.

    Hashing reads the short token history so n-grams remain exactly packed-sequence safe,
    while the table lookup and gate are evaluated only for the current decode row.
    """
    ec = model.config.engram
    hashes = ngram_hash_ids(
        input_ids_history,
        segment_ids_history,
        table_size=ec.table_size,
        max_ngram_size=ec.max_ngram_size,
        n_hash_heads=ec.n_hash_heads,
        pad_token_id=ec.pad_token_id,
        seed=layer_id * 97,
    )[..., -1:, :]
    return _inject_engram(model, streams, hashes, layer_id)


def apply_moe(model: Any, x: torch.Tensor, layer_id: int) -> torch.Tensor:
    prefix = f"blocks.{layer_id}.moe"
    logits = torch.matmul(
        x.float(), model._w(f"{prefix}.router_weight").float()
    )
    raw = torch.sqrt(F.softplus(logits))
    selection = raw + model._w(f"{prefix}.router_bias").float()
    indices = torch.topk(
        selection, k=model.config.experts_per_token, dim=-1
    ).indices
    weights = raw.gather(-1, indices)
    weights = weights / weights.sum(
        dim=-1, keepdim=True
    ).clamp_min(model.config.route_eps)
    weights = weights * model.config.route_scale

    w1 = model._w(f"{prefix}.experts.w1")[indices]
    w2 = model._w(f"{prefix}.experts.w2")[indices]
    w3 = model._w(f"{prefix}.experts.w3")[indices]
    x_compute = x.to(w1.dtype)
    gate = torch.einsum("...d,...kdf->...kf", x_compute, w1)
    up = torch.einsum("...d,...kdf->...kf", x_compute, w3)
    if model.config.swiglu_limit > 0:
        limit = torch.tensor(
            model.config.swiglu_limit,
            device=x.device,
            dtype=gate.dtype,
        )
        gate = torch.minimum(gate, limit)
        up = up.clamp(-model.config.swiglu_limit, model.config.swiglu_limit)
    hidden = F.silu(gate) * up
    selected = torch.einsum("...kf,...kfd->...kd", hidden, w2)
    routed = (
        weights.to(selected.dtype).unsqueeze(-1) * selected
    ).sum(dim=-2)

    shared_w1 = model._w(f"{prefix}.shared.w1")
    shared_w2 = model._w(f"{prefix}.shared.w2")
    shared_w3 = model._w(f"{prefix}.shared.w3")
    shared_x = x.to(shared_w1.dtype)
    shared_gate = torch.matmul(shared_x, shared_w1)
    shared_up = torch.matmul(shared_x, shared_w3)
    if model.config.swiglu_limit > 0:
        limit = torch.tensor(
            model.config.swiglu_limit,
            device=x.device,
            dtype=shared_gate.dtype,
        )
        shared_gate = torch.minimum(shared_gate, limit)
        shared_up = shared_up.clamp(
            -model.config.swiglu_limit, model.config.swiglu_limit
        )
    shared = torch.matmul(F.silu(shared_gate) * shared_up, shared_w2)
    return (routed.to(x.dtype) + shared.to(x.dtype)).to(x.dtype)
