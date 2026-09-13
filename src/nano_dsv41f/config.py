from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class RopeConfig:
    # Released V4.1 uses 64/512 rotary dimensions. Nano keeps the same partial-RoPE idea.
    rope_head_dim: int = 8
    rope_theta: float = 10_000.0
    compress_rope_theta: float = 160_000.0
    # Set to zero to disable YaRN in short educational runs; set e.g. 4096 to study extension.
    original_seq_len: int = 0
    rope_factor: float = 16.0
    beta_fast: int = 32
    beta_slow: int = 1


@dataclass(frozen=True)
class QuantizationConfig:
    # Software fake-quant references. TPU Pallas kernels can replace execution later.
    main_kv_fp4_qat: bool = False
    main_kv_block_size: int = 16
    main_kv_scale_format: Literal["e4m3"] = "e4m3"
    indexer_fp4_qat: bool = False
    indexer_block_size: int = 16
    indexer_scale_format: Literal["e8m0"] = "e8m0"
    swa_fp8_qat: bool = False
    swa_fp8_block_size: int = 32

    def __post_init__(self) -> None:
        if min(self.main_kv_block_size, self.indexer_block_size, self.swa_fp8_block_size) <= 0:
            raise ValueError("quantization block sizes must be positive")


@dataclass(frozen=True)
class AttentionConfig:
    d_model: int = 512
    n_heads: int = 8
    head_dim: int = 64
    q_rank: int = 128
    o_rank: int = 128
    # V4.1 uses grouped low-rank wo_a. G=2 keeps the mechanism visible at nano scale.
    o_groups: int = 2
    local_window: int = 128
    retrieve_top_k: int = 512
    attention_sink: bool = True
    attention_sink_init: float = 0.0
    rope: RopeConfig = RopeConfig()

    def __post_init__(self) -> None:
        if min(self.d_model, self.n_heads, self.head_dim, self.q_rank, self.o_rank) <= 0:
            raise ValueError("attention dimensions must be positive")
        if self.n_heads % self.o_groups:
            raise ValueError("n_heads must be divisible by o_groups")
        if self.rope.rope_head_dim <= 0 or self.rope.rope_head_dim > self.head_dim:
            raise ValueError("rope_head_dim must be in [1, head_dim]")
        if self.rope.rope_head_dim % 2:
            raise ValueError("rope_head_dim must be even")
        if self.local_window <= 0 or self.retrieve_top_k <= 0:
            raise ValueError("attention windows/top-k must be positive")


@dataclass(frozen=True)
class CSA2Config:
    # This mirrors the released runnable reference topology at nano scale:
    #   L0 SWA | L1 Full(r=2) -> L2 Reuse | L3 Full(r=1) -> L4 Reuse
    # The CED split is therefore 3 encoder/context layers + 2 decoder/generation layers.
    context_layers: int = 3
    generation_layers: int = 2
    context_swa_only_layers: int = 1
    context_retriever_group_size: int = 2
    generation_retriever_group_size: int = 2
    context_compression_ratio: int = 2
    generation_compression_ratio: int = 1

    def __post_init__(self) -> None:
        if self.context_layers <= 0 or self.generation_layers <= 0:
            raise ValueError("both CED halves need at least one layer")
        if not (0 <= self.context_swa_only_layers < self.context_layers):
            raise ValueError("context_swa_only_layers must leave at least one compressed context layer")
        if min(self.context_retriever_group_size, self.generation_retriever_group_size) <= 0:
            raise ValueError("retriever group sizes must be positive")
        if self.context_compression_ratio not in (1, 2):
            raise ValueError("context compression ratio must be 1 or 2")
        if self.generation_compression_ratio not in (1, 2):
            raise ValueError("generation compression ratio must be 1 or 2")


@dataclass(frozen=True)
class IndexerConfig:
    n_heads: int = 4
    head_dim: int = 16
    top_k: int = 512
    # Hierarchical retrieval is implemented but disabled in the minimal 5-layer default.
    candidate_source_layer: int = -1
    candidate_topk_blocks: int = 16
    candidate_block_size: int = 8

    def __post_init__(self) -> None:
        if min(self.n_heads, self.head_dim, self.top_k) <= 0:
            raise ValueError("indexer dimensions/top-k must be positive")
        if self.candidate_source_layer >= 0 and min(self.candidate_topk_blocks, self.candidate_block_size) <= 0:
            raise ValueError("candidate hierarchy sizes must be positive when enabled")


@dataclass(frozen=True)
class EngramConfig:
    enabled: bool = True
    # Keep Engram in the causal-encoder half, as in V4.1.
    layer_ids: tuple[int, ...] = (1,)
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
class DSparkConfig:
    enabled: bool = True
    # User-selected nano scaling: one released-style DSpark Transformer stage.
    n_layers: int = 1
    block_size: int = 5
    noise_token_id: int = 0
    target_layer_ids: tuple[int, ...] = (3, 4)
    markov_rank: int = 32
    n_routed_experts: int = 4
    experts_per_token: int = 1
    confidence_head: bool = True
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if self.enabled:
            if self.n_layers <= 0 or self.block_size <= 0 or self.markov_rank <= 0:
                raise ValueError("enabled DSpark needs positive layer/block/Markov sizes")
            if not self.target_layer_ids:
                raise ValueError("DSpark needs at least one target backbone layer")
            if self.n_routed_experts < self.experts_per_token or self.experts_per_token <= 0:
                raise ValueError("invalid DSpark expert counts")


@dataclass(frozen=True)
class IndexerTrainingConfig:
    enabled: bool = True
    start_fraction: float = 0.55
    end_fraction: float = 0.90
    teacher_queries: Literal["latest_eligible", "all_eligible"] = "latest_eligible"
    teacher_layers: Literal["full_only", "full_last", "all_served"] = "all_served"
    loss_weight: float = 0.05

    def __post_init__(self) -> None:
        if not (0.0 <= self.start_fraction <= self.end_fraction <= 1.0):
            raise ValueError("require 0 <= start_fraction <= end_fraction <= 1")
        if self.loss_weight < 0:
            raise ValueError("loss_weight must be non-negative")


@dataclass(frozen=True)
class OptimizerConfig:
    # Report-faithful parameter partitioning. Learning-rate schedule is kept explicit below.
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-20
    adam_weight_decay: float = 0.1
    muon_momentum: float = 0.95
    muon_weight_decay: float = 0.1
    muon_update_rms: float = 0.18
    muon_fast_steps: int = 8
    muon_stable_steps: int = 2
    sinkhorn_momentum: float = 0.95
    sinkhorn_update_rms: float = 0.18
    sinkhorn_iters: int = 11
    sinkhorn_eps: float = 1e-12
    sinkhorn_row_mask_tau: float = 1e-3
    headwise_qk_muon: bool = True


@dataclass(frozen=True)
class RematConfig:
    policy: Literal["none", "attention", "block"] = "attention"


@dataclass(frozen=True)
class ParallelismConfig:
    # Semantic sharding names: several can reuse the same physical 8-chip TPU mesh axis.
    vocab_shard: int = 8
    engram_table_shard: int = 8
    expert_shard: int = 8
    dspark_expert_shard: int = 8
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
    route_eps: float = 1e-20
    swiglu_limit: float = 7.0
    mhc_streams: int = 4
    mhc_sinkhorn_iters: int = 20
    # V4.1 language RMSNorm and mHC use deliberately different epsilons.
    norm_eps: float = 1e-20
    mhc_eps: float = 1e-6
    attention: AttentionConfig = AttentionConfig()
    csa2: CSA2Config = CSA2Config()
    indexer: IndexerConfig = IndexerConfig()
    quantization: QuantizationConfig = QuantizationConfig()
    engram: EngramConfig = EngramConfig()
    dspark: DSparkConfig = DSparkConfig()
    indexer_training: IndexerTrainingConfig = IndexerTrainingConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
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
        if self.indexer.head_dim < self.attention.rope.rope_head_dim:
            raise ValueError("indexer head_dim must fit the shared partial-RoPE dimension")
        if any(layer < 0 or layer >= self.n_layers for layer in self.engram.layer_ids):
            raise ValueError("Engram layer ids must refer to backbone layers")
        if any(layer < 0 or layer >= self.n_layers for layer in self.dspark.target_layer_ids):
            raise ValueError("DSpark target_layer_ids must refer to backbone layers")

    @property
    def n_layers(self) -> int:
        return self.csa2.context_layers + self.csa2.generation_layers


@dataclass(frozen=True)
class TrainConfig:
    total_steps: int = 10_000
    seq_len: int = 4096
    learning_rate: float = 2.6e-4
    warmup_steps: int = 500
    min_learning_rate: float = 2.6e-5
    cosine_decay_start_fraction: float = 0.90

    def progress(self, step: int) -> float:
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        return min(max(step / self.total_steps, 0.0), 1.0)
