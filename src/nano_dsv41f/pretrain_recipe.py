"""Shared 8K pretraining recipe: stress runs and training must use this factory."""
from dataclasses import asdict, replace
import hashlib
import json

from .chat_protocol import apply_tokenizer_contract
from .config import ModelConfig, RematConfig, TrainConfig
from .tpu_native import TPUNativeConfig

# Equal routed capacity E*F=6144; active compute and shared width are NOT equal.
EXPERT_PROFILES = {"baseline": (8, 768, 2), "narrow24": (24, 256, 2), "narrow48": (48, 128, 4)}


def pretrain_recipe(*, profile="baseline", cp=8, dp=1, experts=None, width=None, top_k=None):
    if cp * dp != 8 or (cp, dp) not in ((8, 1), (4, 2), (2, 4)):
        raise ValueError("supported v5e-8 attention layouts: CP8/DP1, CP4/DP2, CP2/DP4")
    e, f, k = EXPERT_PROFILES[profile]
    config = apply_tokenizer_contract(ModelConfig())
    config = replace(
        config, n_experts=e if experts is None else experts,
        d_ff=f if width is None else width, experts_per_token=k if top_k is None else top_k,
        remat=RematConfig(policy="block"),
        parallelism=replace(config.parallelism, attention_context_shard=cp,
                            attention_data_shard=dp, indexer_context_shard=cp),
        indexer_training=replace(config.indexer_training, apply_candidate_mask=False),
    )
    if config.d_ff % 128:
        raise ValueError("stress recipe expert width must be a multiple of 128")
    train = TrainConfig(seq_len=8192)
    native = TPUNativeConfig(moe_ragged_implementation="mosaic", attention_data_shards=dp)
    return config, train, native


def recipe_manifest(config, train, native):
    data = {"model": asdict(config), "train": asdict(train), "native": asdict(native)}
    digest = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    return {**data, "sha256": digest}
