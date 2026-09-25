from __future__ import annotations

import torch


def sinkhorn(matrix: torch.Tensor, iters: int, eps: float) -> torch.Tensor:
    x = torch.softmax(matrix.float(), dim=-1) + eps
    x = x / (x.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        x = x / (x.sum(dim=-1, keepdim=True) + eps)
        x = x / (x.sum(dim=-2, keepdim=True) + eps)
    return x


def mhc_mixes(
    streams: torch.Tensor,
    weight: torch.Tensor,
    base: torch.Tensor,
    scale: torch.Tensor,
    *,
    sinkhorn_iters: int,
    eps: float,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_streams = streams.shape[-2]
    flat = streams.float().reshape(*streams.shape[:-2], -1)
    flat = flat * torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + norm_eps)
    raw = torch.matmul(flat, weight.float())
    base = base.float()
    scale = scale.float()
    pre_raw = raw[..., :n_streams] * scale[0] + base[:n_streams]
    post_raw = (
        raw[..., n_streams : 2 * n_streams] * scale[1]
        + base[n_streams : 2 * n_streams]
    )
    comb_raw = (
        raw[..., 2 * n_streams :] * scale[2] + base[2 * n_streams :]
    ).reshape(*raw.shape[:-1], n_streams, n_streams)
    return (
        torch.sigmoid(pre_raw) + eps,
        2.0 * torch.sigmoid(post_raw),
        sinkhorn(comb_raw, sinkhorn_iters, eps),
    )


def pre_mix(streams: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.einsum(
        "...s,...sd->...d", weights.float(), streams.float()
    ).to(streams.dtype)


def post_mix(
    streams: torch.Tensor,
    branch_output: torch.Tensor,
    combine: torch.Tensor,
    branch_weights: torch.Tensor,
) -> torch.Tensor:
    mixed_old = torch.einsum(
        "...ij,...id->...jd", combine.float(), streams.float()
    )
    branch = (
        branch_weights.float().unsqueeze(-1)
        * branch_output.float().unsqueeze(-2)
    )
    return (mixed_old + branch).to(streams.dtype)


def ngram_hash_ids(
    input_ids: torch.Tensor,
    segment_ids: torch.Tensor,
    *,
    table_size: int,
    max_ngram_size: int,
    n_hash_heads: int,
    pad_token_id: int,
    seed: int,
) -> torch.Tensor:
    """Build packed-safe V4.1 Engram hashes with uint32 wraparound semantics."""
    n_cols = (max_ngram_size - 1) * n_hash_heads
    if n_cols <= 0 or table_size < n_cols:
        raise ValueError("engram table is too small for requested hash columns")
    bucket_size = table_size // n_cols

    def shift_right(x: torch.Tensor, amount: int, fill: int) -> torch.Tensor:
        if amount == 0:
            return x
        pad = torch.full(
            (*x.shape[:-1], amount), fill, dtype=x.dtype, device=x.device
        )
        return torch.cat((pad, x[..., :-amount]), dim=-1)

    columns: list[torch.Tensor] = []
    col = 0
    for ngram_size in range(2, max_ngram_size + 1):
        for head in range(n_hash_heads):
            h = torch.zeros_like(input_ids, dtype=torch.int64)
            for shift in range(ngram_size):
                tok = shift_right(input_ids, shift, pad_token_id)
                seg = shift_right(segment_ids, shift, -1)
                tok = torch.where(seg == segment_ids, tok, pad_token_id).to(torch.int64)
                mix = (
                    0x9E3779B1
                    + 0x85EBCA6B * (head + 1)
                    + 0x27D4EB2D * (shift + 1)
                    + seed
                ) & 0xFFFFFFFF
                h = torch.bitwise_xor(
                    (h * 16777619) & 0xFFFFFFFF,
                    (tok + mix) & 0xFFFFFFFF,
                )
            columns.append((h % bucket_size) + col * bucket_size)
            col += 1
    return torch.stack(columns, dim=-1).long()
