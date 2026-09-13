from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class AttentionConfig:
    d_model: int = 1024
    n_heads: int = 16
    head_dim: int = 64
    local_window: int = 128
    compression_ratio: int = 2
    retrieve_top_k: int = 512

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.n_heads <= 0 or self.head_dim <= 0:
            raise ValueError("attention dimensions must be positive")
        if self.local_window <= 0:
            raise ValueError("local_window must be positive")
        if self.compression_ratio not in (1, 2):
            raise ValueError("educational implementation currently supports r in {1, 2}")
        if self.local_window % self.compression_ratio != 0:
            raise ValueError("local_window must align to compression_ratio")


@dataclass(frozen=True)
class IndexerTrainingConfig:
    enabled: bool = True
    start_fraction: float = 0.55
    end_fraction: float = 0.90
    teacher_queries: Literal["latest_eligible", "all_eligible"] = "latest_eligible"
    teacher_layers: Literal["full_only", "full_last", "all_served"] = "full_last"
    loss_weight: float = 0.05

    def __post_init__(self) -> None:
        if not (0.0 <= self.start_fraction <= self.end_fraction <= 1.0):
            raise ValueError("require 0 <= start_fraction <= end_fraction <= 1")
        if self.loss_weight < 0:
            raise ValueError("loss_weight must be non-negative")


@dataclass(frozen=True)
class RematConfig:
    policy: Literal["none", "attention", "block"] = "attention"


@dataclass(frozen=True)
class ParallelismConfig:
    # These names describe tensor semantics, not independent physical meshes.
    vocab_shard: int = 8
    engram_table_shard: int = 8
    expert_shard: int = 8
    attention_context_shard: int = 8
    attention_head_shard: int = 1
    indexer_context_shard: int = 8


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 32_768
    n_layers: int = 12
    d_model: int = 1024
    d_ff: int = 3072
    n_experts: int = 8
    experts_per_token: int = 2
    mhc_streams: int = 4
    attention: AttentionConfig = AttentionConfig()
    indexer_training: IndexerTrainingConfig = IndexerTrainingConfig()
    remat: RematConfig = RematConfig()
    parallelism: ParallelismConfig = ParallelismConfig()


@dataclass(frozen=True)
class TrainConfig:
    total_steps: int = 10_000
    seq_len: int = 4096
    learning_rate: float = 3e-4

    def progress(self, step: int) -> float:
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        return min(max(step / self.total_steps, 0.0), 1.0)
