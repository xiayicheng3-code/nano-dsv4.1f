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
from .precision import (
    cast_payload_parameters,
    init_model_sharded_mixed_precision,
    precision_summary,
)
from .profiling import (
    collective_counts,
    compile_diagnostics,
    compiled_cost_report,
    compiled_memory_report,
)
from .tpu import (
    V5E,
    batch_named_sharding,
    compile_pretrain_step as compile_pretrain_step_reference,
    init_model_sharded,
    init_optimizer_state_sharded,
    make_v5e_mesh,
    memory_report,
    parameter_partition_specs,
    put_training_batch,
    runtime_report,
    semantic_axes,
    validate_sequence_length,
    validate_v5e_runtime,
)
from .tpu_native import TPUNativeConfig, compile_pretrain_step_native
from . import tpu_native_trace_safety as _tpu_native_trace_safety
from .training import (
    build_indexer_groups,
    causal_lm_loss,
    indexer_phase_enabled,
    pretrain_loss,
    pretrain_step,
    selective_indexer_distillation_loss,
)

# Kaggle/v5e callers get the memory-bounded native executable by default. The old
# auto-sharded dense compiler remains explicitly available for reference/ablation work.
compile_pretrain_step = compile_pretrain_step_native

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
    "TPUNativeConfig",
    "TrainConfig",
    "V5E",
    "apply_dspark",
    "apply_model",
    "apply_model_dspark",
    "batch_named_sharding",
    "build_indexer_groups",
    "build_layer_specs",
    "cast_payload_parameters",
    "causal_lm_loss",
    "collective_counts",
    "compile_diagnostics",
    "compile_pretrain_step",
    "compile_pretrain_step_native",
    "compile_pretrain_step_reference",
    "compiled_cost_report",
    "compiled_memory_report",
    "indexer_phase_enabled",
    "init_model",
    "init_model_sharded",
    "init_model_sharded_mixed_precision",
    "init_optimizer_state",
    "init_optimizer_state_sharded",
    "make_v5e_mesh",
    "memory_report",
    "optimizer_step",
    "parameter_partition_specs",
    "parameter_rule_map",
    "precision_summary",
    "pretrain_loss",
    "pretrain_step",
    "put_training_batch",
    "runtime_report",
    "sample_markov_block",
    "selective_indexer_distillation_loss",
    "semantic_axes",
    "validate_sequence_length",
    "validate_v5e_runtime",
]
