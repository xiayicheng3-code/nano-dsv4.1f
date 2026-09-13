from __future__ import annotations

import jax
import jax.numpy as jnp


_E2M1_LEVELS = jnp.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=jnp.float32)


def _ste(original: jnp.ndarray, quantized: jnp.ndarray, enabled: bool) -> jnp.ndarray:
    if not enabled:
        return quantized.astype(original.dtype)
    return original + jax.lax.stop_gradient(quantized.astype(original.dtype) - original)


def fake_e2m1(x: jnp.ndarray) -> jnp.ndarray:
    """Numerically emulate signed FP4 E2M1 values (not packed storage)."""
    xf = x.astype(jnp.float32)
    mag = jnp.abs(xf)[..., None]
    idx = jnp.argmin(jnp.abs(mag - _E2M1_LEVELS), axis=-1)
    q = jnp.take(_E2M1_LEVELS, idx)
    return jnp.copysign(q, xf)


def fake_e4m3(x: jnp.ndarray) -> jnp.ndarray:
    """Small software E4M3FN emulator sufficient for QAT scale/value experiments."""
    xf = x.astype(jnp.float32)
    sign = jnp.sign(xf)
    a = jnp.abs(xf)
    tiny = jnp.float32(2.0**-9)  # smallest E4M3 subnormal step
    normal_min = jnp.float32(2.0**-6)

    sub = jnp.round(a / tiny) * tiny
    safe = jnp.maximum(a, normal_min)
    exponent = jnp.floor(jnp.log2(safe))
    exponent = jnp.clip(exponent, -6.0, 8.0)
    base = jnp.exp2(exponent)
    mantissa = jnp.round((safe / base - 1.0) * 8.0) / 8.0
    normal = base * (1.0 + mantissa)
    q = jnp.where(a < normal_min, sub, normal)
    q = jnp.clip(q, 0.0, 448.0)
    return sign * q


def fake_e8m0_scale(x: jnp.ndarray) -> jnp.ndarray:
    """Power-of-two shared scale used by MX-style E8M0 scaling."""
    xf = jnp.maximum(x.astype(jnp.float32), jnp.finfo(jnp.float32).tiny)
    return jnp.exp2(jnp.round(jnp.log2(xf)))


def fake_e4m3_scale(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.maximum(jnp.abs(fake_e4m3(jnp.abs(x))), jnp.float32(2.0**-9))


def _blockify(x: jnp.ndarray, block_size: int) -> tuple[jnp.ndarray, tuple[int, ...]]:
    if block_size <= 0 or x.shape[-1] % block_size:
        raise ValueError("last dimension must be divisible by block_size")
    shape = x.shape
    return x.reshape(*shape[:-1], shape[-1] // block_size, block_size), shape


def fake_mxfp4_e2m1(
    x: jnp.ndarray,
    *,
    block_size: int,
    scale_format: str = "e4m3",
    ste: bool = True,
) -> jnp.ndarray:
    """FP4 E2M1 fake-QAT with one software shared scale per channel block.

    V4.1 compressed main KV uses E4M3 scales with block size 16. Its indexer Q/K uses
    an E8M0-style scale with a larger block in the released model. This routine supports
    both so the nano dimensions can choose compatible block sizes explicitly.
    """
    blocks, shape = _blockify(x.astype(jnp.float32), block_size)
    raw_scale = jnp.max(jnp.abs(blocks), axis=-1, keepdims=True) / 6.0
    raw_scale = jnp.maximum(raw_scale, jnp.finfo(jnp.float32).tiny)
    if scale_format == "e4m3":
        scale = fake_e4m3_scale(raw_scale)
    elif scale_format == "e8m0":
        scale = fake_e8m0_scale(raw_scale)
    else:
        raise ValueError("scale_format must be 'e4m3' or 'e8m0'")
    q = fake_e2m1(jnp.clip(blocks / scale, -6.0, 6.0)) * scale
    q = q.reshape(shape)
    return _ste(x, q, ste)


def fake_fp8_e4m3(
    x: jnp.ndarray,
    *,
    block_size: int,
    ste: bool = True,
) -> jnp.ndarray:
    """Block-scaled E4M3 fake quantization for the SWA-KV educational path."""
    blocks, shape = _blockify(x.astype(jnp.float32), block_size)
    raw_scale = jnp.max(jnp.abs(blocks), axis=-1, keepdims=True) / 448.0
    scale = fake_e8m0_scale(jnp.maximum(raw_scale, jnp.finfo(jnp.float32).tiny))
    q = fake_e4m3(jnp.clip(blocks / scale, -448.0, 448.0)) * scale
    q = q.reshape(shape)
    return _ste(x, q, ste)
