from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class AttentionConfig:
    d_model: int = 512
    n_heads: int = 8
    head_dim: int = 64
    q_rank: int = 128
    o_rank: int = 128
    local_window: int = 128
    retrieve_top_k: int = 512

    def __post_init__(self) -> None:
        if min(self.d_model, self.n_heads, self.head_dim, self.q_rank, self.o_rank) <= 0:
            raise ValueError("attention dimensions must be positive")
        if self.local_window <= 0 or self.retrieve_top_k <= 0:
            raise ValueError("attention windows/top-k must be positive")


@dataclass(frozen=True)
class CSA2Config:
    # Four layers on each side are enough to show Full/Reuse and Full/Reindex/Reuse.
    context_layers: int = 4
    generation_layers: int = 4
    retriever_group_size: int = 2
    context_compression_ratio: int = 2
    generation_compression_ratio: int = 1

    def __post_init__(self) -> None:
        if self.context_layers <= 0 or self.generation_layers <= 0:
            raise ValueError("both CED halves need at least one layer")
        if self.retriever_group_size <= 0:
            raise ValueError("retriever_group_size must be positive")
        if self.context_compression_ratio not in (1, 2):
            raise ValueError("context compression ratio must be 1 or 2")
        if self.generation_compression_ratio not in (1, 2):
            raise ValueError("generation compression ratio must be 1 or 2")


@dataclass(frozen=True)
class EngramConfig:
    enabled: bool = True
    # V4.1 places Engram only in its first/context half. The nano model keeps two
    # spread-out insertion points so the mechanism is visible without huge tables.
    layer_ids: tuple[int, ...] = (1, 3)
    table_size: int = 32_768
    max_ngram_size: int = 4
    n_hash_heads: int = 2
    head_dim: int = 32
    pad_token_id: int = 0

    def __post_init__(self) -> None:
        if self.table_size <= 0 or self.head_dim <= 0 or self.n_hash_heads <= 0:
            raise ValueError("Engram table/head dimensions must be positive")
        if self.max_ngram_size < 2:
            raise ValueError("Engram needs at least bigram hashing")


@dataclass(frozen=True)
class IndexerTrainingConfig:
    enabled: bool = True
    start_fraction: float = 0.55
    end_fraction: float = 0.90
    teacher_queries: Literal["latest_eligible", "all_eligible"] = "latest_eligible"
    # Nano groups are typically two served layers, so all-served is affordable and
    # is equivalent to full+last for the default group size.
    teacher_layers: Literal["full_only", "full_last", "all_served"] = "all_served"
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
    d_model: int = 512
    d_ff: int = 768
    n_experts: int = 8
    experts_per_token: int = 2
    route_scale: float = 1.5
    swiglu_limit: float = 7.0
    mhc_streams: int = 4
    mhc_sinkhorn_iters: int = 20
    norm_eps: float = 1e-6
    attention: AttentionConfig = AttentionConfig()
    csa2: CSA2Config = CSA2Config()
    engram: EngramConfig = EngramConfig()
    indexer_training: IndexerTrainingConfig = IndexerTrainingConfig()
    remat: RematConfig = RematConfig()
    parallelism: ParallelismConfig = ParallelismConfig()

    def __post_init__(self) -> None:
        if self.attention.d_model != self.d_model:
            raise ValueError("attention.d_model must equal model d_model")
        if self.vocab_size <= 0 or self.d_model <= 0 or self.d_ff <= 0:
            raise ValueError("model dimensions must be positive")
        if self.n_experts < self.experts_per_token or self.experts_per_token <= 0:
            raise ValueError("require n_experts >= experts_per_token > 0")
        if self.mhc_streams <= 0 or self.mhc_sinkhorn_iters <= 0:
            raise ValueError("mHC settings must be positive")
        if any(layer < 0 or layer >= self.n_layers for layer in self.engram.layer_ids):
            raise ValueError("Engram layer ids must refer to nano model layers")

    @property
    def n_layers(self) -> int:
        return self.csa2.context_layers + self.csa2.generation_layers


@dataclass(frozen=True)
class TrainConfig:
    total_steps: int = 10_000
    seq_len: int = 4096
    learning_rate: float = 3e-4

    def progress(self, step: int) -> float:
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        return min(max(step / self.total_steps, 0.0), 1.0)
