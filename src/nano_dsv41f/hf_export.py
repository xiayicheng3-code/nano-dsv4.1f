from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping

from .chat_protocol import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    nano_v41_tokenizer_contract,
    validate_model_token_ids,
)
from .config import ModelConfig
from .model import build_layer_specs


def flatten_parameter_tree(tree: Any, prefix: str = "nano") -> dict[str, Any]:
    """Flatten the JAX parameter pytree into stable safetensors keys."""
    out: dict[str, Any] = {}

    def visit(value: Any, path: list[str]) -> None:
        if isinstance(value, Mapping):
            for key in sorted(value, key=lambda x: str(x)):
                visit(value[key], path + [str(key)])
            return
        if isinstance(value, (tuple, list)):
            for i, item in enumerate(value):
                visit(item, path + [str(i)])
            return
        name = ".".join(path)
        if not name:
            raise ValueError("parameter tree cannot be a single unnamed leaf")
        out[name] = value

    visit(tree, [prefix] if prefix else [])
    return out


def parameter_manifest(params: Any) -> dict[str, dict[str, Any]]:
    flat = flatten_parameter_tree(params)
    manifest: dict[str, dict[str, Any]] = {}
    for name, value in flat.items():
        shape = tuple(int(x) for x in getattr(value, "shape", ()))
        manifest[name] = {
            "shape": list(shape),
            "dtype": str(getattr(value, "dtype", type(value).__name__)),
        }
    return manifest


def build_nano_hf_config(
    config: ModelConfig,
    *,
    max_position_embeddings: int = 4096,
) -> dict[str, Any]:
    """HF-shaped metadata for the future out-of-tree NanoDeepseekV41 runtime."""
    contract = nano_v41_tokenizer_contract(config.vocab_size)
    specs = build_layer_specs(config)
    return {
        "architectures": ["NanoDeepseekV41ForCausalLM"],
        "model_type": "nano_deepseek_v41",
        "dtype": "bfloat16",
        "vocab_size": config.vocab_size,
        "hidden_size": config.d_model,
        "num_hidden_layers": config.n_layers,
        "num_attention_heads": config.attention.n_heads,
        "head_dim": config.attention.head_dim,
        "qk_rope_head_dim": config.attention.rope.rope_head_dim,
        "q_lora_rank": config.attention.q_rank,
        "o_lora_rank": config.attention.o_rank,
        "o_groups": config.attention.o_groups,
        "sliding_window": config.attention.local_window,
        "max_position_embeddings": max_position_embeddings,
        "bos_token_id": BOS_TOKEN_ID,
        "eos_token_id": EOS_TOKEN_ID,
        "pad_token_id": PAD_TOKEN_ID,
        "compress_ratios": [
            0 if spec.mode == "swa" else spec.compression_ratio for spec in specs
        ],
        "kv_source_layer_ids": [
            spec.layer_id for spec in specs if spec.owns_global_kv
        ],
        "index_source_layer_ids": [
            spec.layer_id for spec in specs if spec.is_index_source
        ],
        "candidate_source_layer_id": config.indexer.candidate_source_layer,
        "candidate_topk_blocks": config.indexer.candidate_topk_blocks,
        "candidate_block_size": config.indexer.candidate_block_size,
        "nano_config": asdict(config),
        "nano_tokenizer_contract": contract.as_dict(),
        "nano_parameter_format": "nano-dsv41f-portable-v1",
    }


def build_deepseek_v41_probe_config(
    config: ModelConfig,
    *,
    max_position_embeddings: int = 4096,
) -> dict[str, Any]:
    """Official-looking V4.1 config used only to probe runtime assumptions.

    This intentionally does *not* claim that the official production V4.1 kernels can
    execute nano dimensions. See ``runtime_compatibility_report``.
    """
    specs = build_layer_specs(config)
    text = {
        "model_type": "deepseek_v41_text",
        "vocab_size": config.vocab_size,
        "hidden_size": config.d_model,
        "moe_intermediate_size": config.d_ff,
        "num_hidden_layers": config.n_layers,
        "num_attention_heads": config.attention.n_heads,
        "num_key_value_heads": 1,
        "head_dim": config.attention.head_dim,
        "qk_rope_head_dim": config.attention.rope.rope_head_dim,
        "q_lora_rank": config.attention.q_rank,
        "o_lora_rank": config.attention.o_rank,
        "o_groups": config.attention.o_groups,
        "rms_norm_eps": config.norm_eps,
        "sliding_window": config.attention.local_window,
        "max_position_embeddings": max_position_embeddings,
        "rope_theta": config.attention.rope.rope_theta,
        "compress_rope_theta": config.attention.rope.compress_rope_theta,
        "n_routed_experts": config.n_experts,
        "n_shared_experts": 1,
        "num_experts_per_tok": config.experts_per_token,
        "scoring_func": "sqrtsoftplus",
        "routed_scaling_factor": config.route_scale,
        "compress_ratios": [
            0 if spec.mode == "swa" else spec.compression_ratio for spec in specs
        ],
        "kv_source_layer_ids": [
            spec.layer_id for spec in specs if spec.owns_global_kv
        ],
        "index_source_layer_ids": [
            spec.layer_id for spec in specs if spec.is_index_source
        ],
        "index_n_heads": config.indexer.n_heads,
        "index_head_dim": config.indexer.head_dim,
        "index_topk": config.indexer.top_k,
        "candidate_source_layer_id": config.indexer.candidate_source_layer,
        "candidate_topk_blocks": config.indexer.candidate_topk_blocks,
        "candidate_block_size": config.indexer.candidate_block_size,
        "hc_mult": config.mhc_streams,
        "hc_sinkhorn_iters": config.mhc_sinkhorn_iters,
        "hc_eps": config.mhc_eps,
        "engram_layer_ids": list(config.engram.layer_ids) if config.engram.enabled else [],
        "engram_num_embeddings": (
            [config.engram.table_size for _ in config.engram.layer_ids]
            if config.engram.enabled
            else []
        ),
        "engram_max_ngram_size": config.engram.max_ngram_size,
        "engram_vocab_size": config.vocab_size,
        "engram_n_heads": config.engram.n_hash_heads,
        "engram_head_dim": config.engram.head_dim,
        "engram_pad_token_id": config.engram.pad_token_id,
        "num_nextn_predict_layers": config.dspark.n_layers if config.dspark.enabled else 0,
        "dspark_block_size": config.dspark.block_size,
        "dspark_noise_token_id": config.dspark.noise_token_id,
        "dspark_target_layer_ids": list(config.dspark.target_layer_ids),
        "dspark_markov_rank": config.dspark.markov_rank,
        "dspark_n_routed_experts": config.dspark.n_routed_experts,
        "dspark_num_experts_per_tok": config.dspark.experts_per_token,
    }
    rc = config.attention.rope
    if rc.original_seq_len > 0:
        text["rope_scaling"] = {
            "rope_type": "yarn",
            "factor": rc.rope_factor,
            "beta_fast": rc.beta_fast,
            "beta_slow": rc.beta_slow,
            "original_max_position_embeddings": rc.original_seq_len,
        }

    return {
        "architectures": ["DeepseekV41ForCausalLM"],
        "model_type": "deepseek_v41",
        "dtype": "bfloat16",
        "bos_token_id": BOS_TOKEN_ID,
        "eos_token_id": EOS_TOKEN_ID,
        "pad_token_id": PAD_TOKEN_ID,
        "text_config": text,
        "_nano_probe_only": True,
        "_nano_note": (
            "Shape/config probe for official runtimes; weights remain in nano portable "
            "layout and require the NanoDeepseekV41 out-of-tree adapter."
        ),
    }


def runtime_compatibility_report(config: ModelConfig) -> dict[str, Any]:
    """Machine-readable blockers for loading the nano model as stock DeepSeek V4.1."""
    token_issues = list(validate_model_token_ids(config))
    blockers: list[dict[str, str]] = []

    if config.attention.head_dim != 512 or config.attention.rope.rope_head_dim != 64:
        blockers.append(
            {
                "component": "vllm.deepseek_v41.compressor",
                "severity": "hard",
                "reason": (
                    "Current stock vLLM V4.1 compressor kernels assert head_dim=512 "
                    "and qk_rope_head_dim=64; nano uses "
                    f"{config.attention.head_dim}/{config.attention.rope.rope_head_dim}."
                ),
                "resolution": (
                    "Use an out-of-tree NanoDeepseekV41 attention/compressor backend "
                    "or add generic eager kernels."
                ),
            }
        )
    blockers.extend(
        [
            {
                "component": "weights",
                "severity": "hard",
                "reason": (
                    "Portable safetensor keys mirror the JAX tree, not the production "
                    "DeepSeek checkpoint naming/packing."
                ),
                "resolution": "Use the nano runtime adapter's weight loader.",
            },
            {
                "component": "engram",
                "severity": "medium",
                "reason": (
                    "Nano Engram uses a compact hash-table layout rather than the released "
                    "V4.1 production table/compressed-vocab layout."
                ),
                "resolution": "Keep nano Engram semantics in the custom runtime.",
            },
            {
                "component": "dspark",
                "severity": "medium",
                "reason": (
                    "Nano intentionally implements one DSpark stage with four routed experts, "
                    "not the released three-stage production draft stack."
                ),
                "resolution": "Disable speculative serving initially or use a nano draft adapter.",
            },
        ]
    )
    if token_issues:
        blockers.append(
            {
                "component": "tokenizer",
                "severity": "hard-before-training",
                "reason": "; ".join(token_issues),
                "resolution": (
                    "Freeze the nano V4.1 tokenizer contract and update model IDs before "
                    "starting any token-ID-dependent training."
                ),
            }
        )

    return {
        "stock_deepseek_v41_runtime_compatible": not blockers,
        "recommended_runtime_path": (
            "portable checkpoint -> NanoDeepseekV41 out-of-tree vLLM/SGLang adapter"
        ),
        "blockers": blockers,
    }


def export_portable_checkpoint(
    params: Any,
    output_dir: str | Path,
    config: ModelConfig,
    *,
    max_position_embeddings: int = 4096,
    metadata: dict[str, str] | None = None,
) -> Path:
    """Write a lossless safetensors checkpoint plus nano/HF interoperability metadata."""
    try:
        from safetensors.flax import save_file
    except ImportError as exc:
        raise RuntimeError(
            "Checkpoint export needs the optional export dependency: "
            "pip install 'nano-dsv41f[export]'"
        ) from exc

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tensors = flatten_parameter_tree(params)
    checkpoint = output / "model.safetensors"
    save_file(
        tensors,
        str(checkpoint),
        metadata=metadata or {"format": "nano-dsv41f-portable-v1"},
    )

    files = {
        "config.json": build_nano_hf_config(
            config, max_position_embeddings=max_position_embeddings
        ),
        "deepseek_v41_probe_config.json": build_deepseek_v41_probe_config(
            config, max_position_embeddings=max_position_embeddings
        ),
        "runtime_compatibility.json": runtime_compatibility_report(config),
        "nano_parameter_manifest.json": parameter_manifest(params),
        "nano_tokenizer_contract.json": nano_v41_tokenizer_contract(
            config.vocab_size
        ).as_dict(),
        "generation_config.json": {
            "bos_token_id": BOS_TOKEN_ID,
            "eos_token_id": EOS_TOKEN_ID,
            "pad_token_id": PAD_TOKEN_ID,
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 0.95,
        },
    }
    for filename, payload in files.items():
        (output / filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return checkpoint
