from __future__ import annotations

import math

import jax.numpy as jnp


def yarn_inv_freq(
    rotary_dim: int,
    *,
    base: float,
    original_seq_len: int = 0,
    factor: float = 1.0,
    beta_fast: int = 32,
    beta_slow: int = 1,
) -> jnp.ndarray:
    """DeepSeek YaRN inverse frequencies, matching the released reference formula.

    `original_seq_len == 0` disables YaRN and returns ordinary RoPE frequencies.
    """
    if rotary_dim <= 0 or rotary_dim % 2:
        raise ValueError("rotary_dim must be a positive even integer")
    idx = jnp.arange(0, rotary_dim, 2, dtype=jnp.float32)
    freqs = 1.0 / (base ** (idx / float(rotary_dim)))
    if original_seq_len <= 0 or factor <= 1.0:
        return freqs

    def corrected_dim(rotations: float) -> float:
        return rotary_dim * math.log(
            original_seq_len / (rotations * 2.0 * math.pi)
        ) / (2.0 * math.log(base))

    low = max(math.floor(corrected_dim(beta_fast)), 0)
    high = min(math.ceil(corrected_dim(beta_slow)), rotary_dim - 1)
    ramp = (jnp.arange(rotary_dim // 2, dtype=jnp.float32) - low) / max(
        high - low, 1e-3
    )
    ramp = jnp.clip(ramp, 0.0, 1.0)
    smooth = 1.0 - ramp
    return freqs / factor * (1.0 - smooth) + freqs * smooth


def rope_cos_sin(
    positions: jnp.ndarray,
    rotary_dim: int,
    *,
    base: float,
    original_seq_len: int = 0,
    factor: float = 1.0,
    beta_fast: int = 32,
    beta_slow: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    inv_freq = yarn_inv_freq(
        rotary_dim,
        base=base,
        original_seq_len=original_seq_len,
        factor=factor,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
    )
    angle = positions.astype(jnp.float32)[..., None] * inv_freq
    return jnp.cos(angle), jnp.sin(angle)


def apply_partial_rope(
    x: jnp.ndarray,
    positions: jnp.ndarray,
    *,
    rotary_dim: int,
    base: float,
    original_seq_len: int = 0,
    factor: float = 1.0,
    beta_fast: int = 32,
    beta_slow: int = 1,
    inverse: bool = False,
) -> jnp.ndarray:
    """Apply V4.1-style RoPE to the *last* `rotary_dim` channels.

    Adjacent tail channels are interpreted as complex pairs. `inverse=True` negates the
    rotation angle; V4.1 applies this to the attention output before grouped `wo_a`.

    `positions` must match the token-prefix shape of `x`; any head axes live between the
    token axes and the final channel axis and are broadcast automatically.
    """
    if rotary_dim <= 0 or rotary_dim > x.shape[-1] or rotary_dim % 2:
        raise ValueError("rotary_dim must be even and fit inside the final dimension")

    cos, sin = rope_cos_sin(
        positions,
        rotary_dim,
        base=base,
        original_seq_len=original_seq_len,
        factor=factor,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
    )
    if inverse:
        sin = -sin

    prefix = x[..., :-rotary_dim]
    tail = x[..., -rotary_dim:]
    pair = tail.reshape(*tail.shape[:-1], rotary_dim // 2, 2)

    # Insert singleton axes between token positions and complex-pair dimension until
    # cos/sin broadcast over optional attention heads.
    while cos.ndim < pair.ndim - 1:
        cos = cos[..., None, :]
        sin = sin[..., None, :]

    real = pair[..., 0]
    imag = pair[..., 1]
    rotated = jnp.stack(
        [real * cos - imag * sin, real * sin + imag * cos], axis=-1
    ).reshape(tail.shape)
    return jnp.concatenate([prefix, rotated.astype(x.dtype)], axis=-1)


def rope_kwargs_for_layer(
    *,
    compress_ratio: int,
    rope_theta: float,
    compress_rope_theta: float,
    original_seq_len: int,
    factor: float,
    beta_fast: int,
    beta_slow: int,
) -> dict[str, float | int]:
    """Return the positional regime used by one attention/indexer layer.

    Pure SWA (`compress_ratio == 0`) uses plain base RoPE. Compressed layers use the
    compressed theta and optionally YaRN, matching the released V4.1 reference.
    """
    if compress_ratio == 0:
        return {
            "base": rope_theta,
            "original_seq_len": 0,
            "factor": 1.0,
            "beta_fast": beta_fast,
            "beta_slow": beta_slow,
        }
    return {
        "base": compress_rope_theta,
        "original_seq_len": original_seq_len,
        "factor": factor,
        "beta_fast": beta_fast,
        "beta_slow": beta_slow,
    }
