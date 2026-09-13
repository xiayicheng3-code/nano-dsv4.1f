"""nano-dsv4.1f educational DeepSeek-V4.1-Flash reference package."""

from .config import (
    AttentionConfig,
    CSA2Config,
    DSparkConfig,
    EngramConfig,
    IndexerConfig,
    IndexerTrainingConfig,
    ModelConfig,
    OptimizerConfig,
    ParallelismConfig,
    QuantizationConfig,
    RematConfig,
    RopeConfig,
    TrainConfig,
)
from .dspark import apply_dspark, sample_markov_block
from .model import apply_model, apply_model_dspark, build_layer_specs, init_model
from .optimizer import init_optimizer_state, optimizer_step, parameter_rule_map
from .training import (
    build_indexer_groups,
    causal_lm_loss,
    indexer_phase_enabled,
    pretrain_loss,
    pretrain_step,
    selective_indexer_distillation_loss,
)

__all__ = [
    "AttentionConfig",
    "CSA2Config",
    "DSparkConfig",
    "EngramConfig",
    "IndexerConfig",
    "IndexerTrainingConfig",
    "ModelConfig",
    "OptimizerConfig",
    "ParallelismConfig",
    "QuantizationConfig",
    "RematConfig",
    "RopeConfig",
    "TrainConfig",
    "apply_dspark",
    "apply_model",
    "apply_model_dspark",
    "build_indexer_groups",
    "build_layer_specs",
    "causal_lm_loss",
    "indexer_phase_enabled",
    "init_model",
    "init_optimizer_state",
    "optimizer_step",
    "parameter_rule_map",
    "pretrain_loss",
    "pretrain_step",
    "sample_markov_block",
    "selective_indexer_distillation_loss",
]
