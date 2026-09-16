from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .packing import PackedSegments, padded_to_multiple


@dataclass(frozen=True)
class PackedTokenBatch:
    """One fixed-length packed token row plus alignment metadata.

    Padding inserted inside a segment keeps that segment id so r=2 compression groups
    never straddle documents. `token_mask` is therefore the source of truth for real
    tokens; losses and late-indexer teacher selection must honor it.
    """

    input_ids: np.ndarray
    segment_ids: np.ndarray
    token_mask: np.ndarray
    real_lengths: tuple[int, ...]
    physical_lengths: tuple[int, ...]
    compression_ratio: int

    @property
    def n_segments(self) -> int:
        return len(self.real_lengths)

    @property
    def real_tokens(self) -> int:
        return int(sum(self.real_lengths))

    @property
    def lm_tokens(self) -> int:
        return int(sum(max(length - 1, 0) for length in self.real_lengths))

    @property
    def packed_segments(self) -> PackedSegments:
        return PackedSegments(self.physical_lengths, self.compression_ratio)


def pack_token_sequences(
    sequences: Sequence[Sequence[int] | np.ndarray],
    *,
    seq_len: int,
    pad_token_id: int,
    compression_ratio: int,
) -> PackedTokenBatch:
    """Pack complete documents into one fixed row without cross-document targets.

    Each document receives only the minimum right padding needed to align its physical
    length to `compression_ratio`. Any remaining row tail is attached to the final segment
    as masked padding, which preserves compression alignment while keeping the number of
    semantic segments fixed. The caller should pass `n_segments=len(sequences)` to the
    late-indexer executable.
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if compression_ratio <= 0:
        raise ValueError("compression_ratio must be positive")
    if seq_len % compression_ratio:
        raise ValueError("seq_len must be divisible by compression_ratio")
    if not sequences:
        raise ValueError("at least one sequence is required")

    arrays = tuple(np.asarray(seq, dtype=np.int32).reshape(-1) for seq in sequences)
    if any(arr.size == 0 for arr in arrays):
        raise ValueError("packed sequences must be non-empty")

    real_lengths = tuple(int(arr.size) for arr in arrays)
    physical = [padded_to_multiple(length, compression_ratio) for length in real_lengths]
    used = sum(physical)
    if used > seq_len:
        raise ValueError(
            f"ratio-aligned documents need {used} tokens but seq_len={seq_len}"
        )

    tail = seq_len - used
    if tail % compression_ratio:
        raise AssertionError("aligned sequence lengths left a misaligned tail")
    physical[-1] += tail
    physical_lengths = tuple(physical)

    ids = np.full((seq_len,), pad_token_id, dtype=np.int32)
    segments = np.empty((seq_len,), dtype=np.int32)
    mask = np.zeros((seq_len,), dtype=bool)

    cursor = 0
    for segment_id, (arr, real_len, physical_len) in enumerate(
        zip(arrays, real_lengths, physical_lengths)
    ):
        stop = cursor + physical_len
        ids[cursor : cursor + real_len] = arr
        segments[cursor:stop] = segment_id
        mask[cursor : cursor + real_len] = True
        cursor = stop

    if cursor != seq_len:
        raise AssertionError("packer did not fill the requested fixed sequence length")

    # Validate the invariant used by CSA2's strided compression metadata.
    PackedSegments(physical_lengths, compression_ratio)
    return PackedTokenBatch(
        input_ids=ids[None, :],
        segment_ids=segments[None, :],
        token_mask=mask[None, :],
        real_lengths=real_lengths,
        physical_lengths=physical_lengths,
        compression_ratio=compression_ratio,
    )
