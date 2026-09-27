from __future__ import annotations

import math

import torch

from ..config import ModelConfig


def linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.matmul(x.to(weight.dtype), weight)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    scale = torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps)
    return (xf * scale * weight.float()).to(x.dtype)


def segment_local_positions(segment_ids: torch.Tensor) -> torch.Tensor:
    if segment_ids.ndim != 2:
        raise ValueError("segment_ids must have shape [batch,tokens]")
    bsz, length = segment_ids.shape
    absolute = torch.arange(length, device=segment_ids.device, dtype=torch.long)
    absolute = absolute.unsqueeze(0).expand(bsz, -1)
    starts = torch.ones_like(segment_ids, dtype=torch.bool)
    if length > 1:
        starts[:, 1:] = segment_ids[:, 1:] != segment_ids[:, :-1]
    start_values = torch.where(starts, absolute, torch.zeros_like(absolute))
    start_pos = torch.cummax(start_values, dim=1).values
    return absolute - start_pos


def rope_kwargs(config: ModelConfig, compress_ratio: int) -> dict[str, float | int]:
    rc = config.attention.rope
    if compress_ratio == 0:
        return {
            "base": rc.rope_theta,
            "original_seq_len": 0,
            "factor": 1.0,
            "beta_fast": rc.beta_fast,
            "beta_slow": rc.beta_slow,
        }
    return {
        "base": rc.compress_rope_theta,
        "original_seq_len": rc.original_seq_len,
        "factor": rc.rope_factor,
        "beta_fast": rc.beta_fast,
        "beta_slow": rc.beta_slow,
    }


def yarn_inv_freq(
    rotary_dim: int,
    *,
    base: float,
    original_seq_len: int,
    factor: float,
    beta_fast: int,
    beta_slow: int,
    device: torch.device,
) -> torch.Tensor:
    idx = torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32)
    freqs = 1.0 / (base ** (idx / float(rotary_dim)))
    if original_seq_len <= 0 or factor <= 1.0:
        return freqs

    def corrected_dim(rotations: float) -> float:
        return rotary_dim * math.log(
            original_seq_len / (rotations * 2.0 * math.pi)
        ) / (2.0 * math.log(base))

    low = max(math.floor(corrected_dim(beta_fast)), 0)
    high = min(math.ceil(corrected_dim(beta_slow)), rotary_dim - 1)
    ramp = (
        torch.arange(rotary_dim // 2, device=device, dtype=torch.float32) - low
    ) / max(high - low, 1e-3)
    ramp = ramp.clamp(0.0, 1.0)
    smooth = 1.0 - ramp
    return freqs / factor * (1.0 - smooth) + freqs * smooth


def partial_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    *,
    rotary_dim: int,
    inverse: bool = False,
    **kwargs: float | int,
) -> torch.Tensor:
    """Apply V4.1 partial RoPE to the last ``rotary_dim`` channels."""
    if rotary_dim <= 0 or rotary_dim > x.shape[-1] or rotary_dim % 2:
        raise ValueError("rotary_dim must be positive, even, and fit the head dimension")
    inv = yarn_inv_freq(
        rotary_dim,
        base=float(kwargs["base"]),
        original_seq_len=int(kwargs["original_seq_len"]),
        factor=float(kwargs["factor"]),
        beta_fast=int(kwargs["beta_fast"]),
        beta_slow=int(kwargs["beta_slow"]),
        device=x.device,
    )
    angle = positions.float().unsqueeze(-1) * inv
    cos, sin = angle.cos(), angle.sin()
    if inverse:
        sin = -sin

    prefix = x[..., :-rotary_dim]
    tail = x[..., -rotary_dim:]
    pair = tail.reshape(*tail.shape[:-1], rotary_dim // 2, 2)
    while cos.ndim < pair.ndim - 1:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
    real, imag = pair[..., 0], pair[..., 1]
    rotated = torch.stack(
        (real * cos - imag * sin, real * sin + imag * cos), dim=-1
    ).reshape_as(tail)
    return torch.cat((prefix, rotated.to(x.dtype)), dim=-1)


def learned_group_compress(
    x: torch.Tensor,
    score: torch.Tensor,
    ratio: int,
) -> torch.Tensor:
    """V4.1 learned contiguous compression, retaining only completed groups."""
    if ratio not in (1, 2):
        raise ValueError("CPU reference supports compression ratio 1 or 2")
    if ratio == 1:
        return x
    complete = (x.shape[-2] // ratio) * ratio
    x = x[..., :complete, :]
    score = score[..., :complete, :]
    t, dim = x.shape[-2:]
    grouped_x = x.reshape(*x.shape[:-2], t // ratio, ratio, dim)
    grouped_score = score.reshape(
        *score.shape[:-2], t // ratio, ratio, dim
    ).float()
    weight = torch.softmax(grouped_score, dim=-2).to(x.dtype)
    return (grouped_x * weight).sum(dim=-2).to(x.dtype)
