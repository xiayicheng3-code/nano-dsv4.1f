from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, PartitionSpec as P

from nano_dsv41f.config import (
    AttentionConfig,
    CSA2Config,
    DSparkConfig,
    EngramConfig,
    IndexerConfig,
    ModelConfig,
    ParallelismConfig,
    RematConfig,
    RopeConfig,
    TrainConfig,
)
from nano_dsv41f.precision import (
    init_model_sharded_mixed_precision,
    precision_summary,
)
from nano_dsv41f.tpu import (
    axes_for_shard_count,
    compile_pretrain_step,
    init_optimizer_state_sharded,
    make_v5e_mesh,
    parameter_partition_spec,
    put_training_batch,
    validate_sequence_length,
)


class FakeV5EMesh:
    axis_names = ("x", "y")
    devices = np.empty((2, 4), dtype=object)
    size = 8


def tiny_single_device_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        d_model=32,
        d_ff=48,
        n_experts=4,
        experts_per_token=2,
        mhc_streams=2,
        mhc_sinkhorn_iters=2,
        attention=AttentionConfig(
            d_model=32,
            n_heads=4,
            head_dim=8,
            q_rank=16,
            o_rank=8,
            o_groups=2,
            local_window=4,
            rope=RopeConfig(rope_head_dim=2, original_seq_len=0),
        ),
        csa2=CSA2Config(
            context_layers=3,
            generation_layers=4,
            context_swa_only_layers=1,
            context_retriever_group_size=2,
            generation_retriever_group_size=2,
            context_compression_ratio=2,
            generation_compression_ratio=1,
        ),
        indexer=IndexerConfig(
            n_heads=2,
            head_dim=4,
            top_k=8,
            candidate_source_layer=3,
            candidate_topk_blocks=2,
            candidate_block_size=2,
        ),
        engram=EngramConfig(
            enabled=True,
            layer_ids=(1,),
            table_size=128,
            max_ngram_size=3,
            n_hash_heads=2,
            head_dim=8,
        ),
        dspark=DSparkConfig(enabled=False),
        remat=RematConfig(policy="none"),
        parallelism=ParallelismConfig(
            vocab_shard=1,
            engram_table_shard=1,
            expert_shard=1,
            dspark_expert_shard=1,
            attention_context_shard=1,
            attention_head_shard=1,
            indexer_context_shard=1,
        ),
    )


def test_v5e_2x4_semantic_axis_mapping():
    mesh = FakeV5EMesh()
    assert axes_for_shard_count(mesh, 1) is None
    assert axes_for_shard_count(mesh, 2) == "x"
    assert axes_for_shard_count(mesh, 4) == "y"
    assert axes_for_shard_count(mesh, 8) == ("x", "y")
    with pytest.raises(ValueError):
        axes_for_shard_count(mesh, 3)


def test_fallback_runtime_mesh_uses_auto_axes():
    mesh = make_v5e_mesh(strict=False)
    assert all(axis_type == AxisType.Auto for axis_type in mesh.axis_types)


def test_parameter_specs_reuse_physical_mesh_by_semantic_role():
    cfg = ModelConfig()
    mesh = FakeV5EMesh()
    f32 = jnp.float32

    embed = parameter_partition_spec(
        ("embed",), jax.ShapeDtypeStruct((cfg.vocab_size, cfg.d_model), f32), cfg, mesh
    )
    expert = parameter_partition_spec(
        ("blocks", "0", "moe", "experts", "w1"),
        jax.ShapeDtypeStruct((cfg.n_experts, cfg.d_model, cfg.d_ff), f32),
        cfg,
        mesh,
    )
    dspark_expert = parameter_partition_spec(
        ("dspark", "moe", "experts", "w1"),
        jax.ShapeDtypeStruct(
            (cfg.dspark.n_routed_experts, cfg.d_model, cfg.d_ff), f32
        ),
        cfg,
        mesh,
    )
    q_b = parameter_partition_spec(
        ("blocks", "0", "attn", "q_b", "weight"),
        jax.ShapeDtypeStruct(
            (cfg.attention.q_rank, cfg.attention.n_heads * cfg.attention.head_dim), f32
        ),
        cfg,
        mesh,
    )

    assert embed == P(("x", "y"), None)
    assert expert == P(("x", "y"), None, None)
    assert dspark_expert == P("y", None, None)
    assert q_b == P(None, None)  # MLA TP is deliberately off by default.


def test_v5e_sequence_guardrails():
    cfg = ModelConfig()
    mesh = FakeV5EMesh()
    assert validate_sequence_length(4096, cfg, mesh) == ()
    with pytest.raises(ValueError):
        validate_sequence_length(4097, cfg, mesh)


def test_single_device_ci_can_compile_mixed_precision_sharded_pretrain_step():
    cfg = tiny_single_device_config()
    train_cfg = TrainConfig(total_steps=8, seq_len=8, warmup_steps=1)
    mesh = make_v5e_mesh(strict=False)

    params, specs, _ = init_model_sharded_mixed_precision(
        jax.random.PRNGKey(0), cfg, mesh, payload_dtype=jnp.bfloat16
    )
    summary = precision_summary(params)
    assert summary.get("bfloat16", 0) > 0
    assert summary.get("float32", 0) > 0
    assert params["embed"].dtype == jnp.bfloat16
    assert params["blocks"][0]["attn_norm"]["weight"].dtype == jnp.float32

    opt_state, _ = init_optimizer_state_sharded(params, specs, cfg, mesh)
    step_fn = compile_pretrain_step(
        params,
        opt_state,
        specs,
        cfg,
        train_cfg,
        mesh,
        include_indexer=False,
        n_segments=None,
    )

    ids = jnp.arange(8, dtype=jnp.int32)[None, :] % cfg.vocab_size
    segments = jnp.zeros_like(ids)
    mask = jnp.ones_like(ids, dtype=bool)
    ids, segments, mask = put_training_batch(ids, segments, mask, cfg, mesh)

    new_params, new_state, metrics = step_fn(
        params,
        opt_state,
        ids,
        segments,
        jnp.asarray(0, dtype=jnp.int32),
        mask,
    )
    jax.block_until_ready(metrics["loss"])
    assert jnp.isfinite(metrics["loss"])
    assert new_params["embed"].dtype == jnp.bfloat16
    assert jax.tree_util.tree_structure(new_params) == jax.tree_util.tree_structure(params)
    assert jax.tree_util.tree_structure(new_state) == jax.tree_util.tree_structure(opt_state)
