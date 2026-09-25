from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from ..config import (
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
)


def model_config_from_export(payload: Mapping[str, Any]) -> ModelConfig:
    """Rebuild ``ModelConfig`` from ``config.json`` written by ``hf_export``."""
    raw = payload.get("nano_config", payload)
    if not isinstance(raw, Mapping):
        raise ValueError("config payload must contain a mapping-valued nano_config")

    attention_raw = dict(raw.get("attention", {}))
    attention_raw["rope"] = RopeConfig(**dict(attention_raw.get("rope", {})))
    return ModelConfig(
        **{
            key: value
            for key, value in raw.items()
            if key
            not in {
                "attention",
                "csa2",
                "indexer",
                "quantization",
                "engram",
                "dspark",
                "indexer_training",
                "optimizer",
                "remat",
                "parallelism",
            }
        },
        attention=AttentionConfig(**attention_raw),
        csa2=CSA2Config(**dict(raw.get("csa2", {}))),
        indexer=IndexerConfig(**dict(raw.get("indexer", {}))),
        quantization=QuantizationConfig(**dict(raw.get("quantization", {}))),
        engram=EngramConfig(**dict(raw.get("engram", {}))),
        dspark=DSparkConfig(**dict(raw.get("dspark", {}))),
        indexer_training=IndexerTrainingConfig(**dict(raw.get("indexer_training", {}))),
        optimizer=OptimizerConfig(**dict(raw.get("optimizer", {}))),
        remat=RematConfig(**dict(raw.get("remat", {}))),
        parallelism=ParallelismConfig(**dict(raw.get("parallelism", {}))),
    )


def to_torch(
    value: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Move a portable JAX/safetensors leaf to the CPU inference dtype."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device=device)
    else:
        tensor = torch.as_tensor(np.asarray(value), device=device)
    if tensor.is_floating_point():
        tensor = tensor.to(dtype=dtype)
    return tensor.contiguous()
