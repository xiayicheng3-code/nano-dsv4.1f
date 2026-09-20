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
from .chat_protocol import (
    TokenizerContract,
    apply_tokenizer_contract,
    nano_v41_tokenizer_contract,
    validate_model_token_ids,
)
from .agent_data import CleanPolicy, TraceNormalizationError, normalize_agent_trace
from .reasoning_effort import (
    assign_length_guided_reasoning_effort,
    character_reasoning_length,
    missing_reasoning_efforts,
    reasoning_effort_histogram,
    reasoning_text,
    render_v41_reasoning_effort_prompt,
    tokenizer_reasoning_length,
)
from .dspark import apply_dspark, sample_markov_block
from .model import apply_model, apply_model_dspark, build_layer_specs, init_model
from .hf_export import (
    build_deepseek_v41_probe_config,
    build_nano_hf_config,
    export_portable_checkpoint,
    flatten_parameter_tree,
    runtime_compatibility_report,
)
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
from .sequence_packing import PackedTokenBatch, pack_token_sequences
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
from .tpu_moe import apply_moe_v5e_multi
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
    "CleanPolicy",
    "TokenizerContract",
    "TraceNormalizationError",
    "CSA2Config",
    "DSparkConfig",
    "EngramConfig",
    "IndexerConfig",
    "IndexerTrainingConfig",
    "ModelConfig",
    "OptimizerConfig",
    "PackedTokenBatch",
    "ParallelismConfig",
    "QuantizationConfig",
    "RematConfig",
    "RopeConfig",
    "TPUNativeConfig",
    "TrainConfig",
    "V5E",
    "apply_dspark",
    "apply_tokenizer_contract",
    "apply_model",
    "apply_model_dspark",
    "assign_length_guided_reasoning_effort",
    "batch_named_sharding",
    "build_deepseek_v41_probe_config",
    "build_indexer_groups",
    "build_layer_specs",
    "build_nano_hf_config",
    "cast_payload_parameters",
    "causal_lm_loss",
    "character_reasoning_length",
    "collective_counts",
    "compile_diagnostics",
    "compile_pretrain_step",
    "compile_pretrain_step_native",
    "compile_pretrain_step_reference",
    "compiled_cost_report",
    "compiled_memory_report",
    "export_portable_checkpoint",
    "flatten_parameter_tree",
    "indexer_phase_enabled",
    "init_model",
    "init_model_sharded",
    "init_model_sharded_mixed_precision",
    "init_optimizer_state",
    "init_optimizer_state_sharded",
    "make_v5e_mesh",
    "memory_report",
    "missing_reasoning_efforts",
    "nano_v41_tokenizer_contract",
    "normalize_agent_trace",
    "optimizer_step",
    "pack_token_sequences",
    "parameter_partition_specs",
    "parameter_rule_map",
    "precision_summary",
    "pretrain_loss",
    "pretrain_step",
    "put_training_batch",
    "reasoning_effort_histogram",
    "reasoning_text",
    "render_v41_reasoning_effort_prompt",
    "runtime_compatibility_report",
    "runtime_report",
    "sample_markov_block",
    "selective_indexer_distillation_loss",
    "semantic_axes",
    "tokenizer_reasoning_length",
    "validate_model_token_ids",
    "validate_sequence_length",
    "validate_v5e_runtime",
]
