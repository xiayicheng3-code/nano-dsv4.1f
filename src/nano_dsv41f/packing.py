from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import jax.numpy as jnp


@dataclass(frozen=True)
class PackedSegments:
    lengths: tuple[int, ...]
    compression_ratio: int

    def __post_init__(self) -> None:
        if self.compression_ratio <= 0:
            raise ValueError("compression_ratio must be positive")
        if any(length <= 0 for length in self.lengths):
            raise ValueError("all packed segment lengths must be positive")
        if any(length % self.compression_ratio for length in self.lengths):
            raise ValueError(
                "each segment must be padded to a multiple of compression_ratio"
            )

    @property
    def total_tokens(self) -> int:
        return sum(self.lengths)

    def token_segment_ids(self) -> jnp.ndarray:
        return jnp.concatenate(
            [jnp.full((length,), i, dtype=jnp.int32) for i, length in enumerate(self.lengths)]
        )

    def compressed_segment_ids(self) -> jnp.ndarray:
        r = self.compression_ratio
        return jnp.concatenate(
            [
                jnp.full((length // r,), i, dtype=jnp.int32)
                for i, length in enumerate(self.lengths)
            ]
        )


def padded_to_multiple(length: int, multiple: int) -> int:
    if length < 0 or multiple <= 0:
        raise ValueError("length must be non-negative and multiple positive")
    return ((length + multiple - 1) // multiple) * multiple


def eligible_query_mask(
    segment_ids: jnp.ndarray,
    *,
    local_window: int,
    retrieve_top_k: int,
) -> jnp.ndarray:
    """Marks token positions whose *segment-local* history reaches local+retrieval size.

    The first eligible query has local position `local_window + retrieve_top_k` when
    local position zero denotes the first token in the segment. This intentionally
    requires enough causal history for both the local window and K retrieved items.
    """
    if segment_ids.ndim != 1:
        raise ValueError("segment_ids must be rank-1")

    starts = jnp.concatenate(
        [
            jnp.array([True]),
            segment_ids[1:] != segment_ids[:-1],
        ]
    )
    global_positions = jnp.arange(segment_ids.shape[0], dtype=jnp.int32)
    start_positions = jnp.maximum.accumulate(jnp.where(starts, global_positions, 0))
    local_positions = global_positions - start_positions
    return local_positions >= (local_window + retrieve_top_k)


def latest_eligible_queries(
    segment_ids: jnp.ndarray,
    *,
    local_window: int,
    retrieve_top_k: int,
) -> jnp.ndarray:
    """Returns one boolean-selected teacher query per segment when one is eligible.

    This is the default educational approximation used to keep indexer distillation
    cheap. It is deliberately not claimed to reproduce DeepSeek's private recipe.
    """
    eligible = eligible_query_mask(
        segment_ids,
        local_window=local_window,
        retrieve_top_k=retrieve_top_k,
    )
    n = segment_ids.shape[0]
    is_segment_end = jnp.concatenate(
        [segment_ids[:-1] != segment_ids[1:], jnp.array([True])]
    )

    # A segment end is selected iff that segment has reached the eligibility threshold.
    # Since eligibility is monotonic within a contiguous segment, checking the end is enough.
    return is_segment_end & eligible


def segment_lengths(segment_ids: jnp.ndarray) -> jnp.ndarray:
    """Convenience helper used by tests and notebook visualizations."""
    if segment_ids.ndim != 1:
        raise ValueError("segment_ids must be rank-1")
    if segment_ids.size == 0:
        return jnp.zeros((0,), dtype=jnp.int32)
    max_id = int(jnp.max(segment_ids))
    return jnp.stack([jnp.sum(segment_ids == i) for i in range(max_id + 1)])
