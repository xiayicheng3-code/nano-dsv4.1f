from dataclasses import replace

import jax
import jax.numpy as jnp

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
)
from nano_dsv41f.indexer import eligibility_position
from nano_dsv41f.indexer_scorer import select_candidate_blocks
from nano_dsv41f.model import (
    apply_model,
    apply_model_dspark,
    build_layer_specs,
    init_model,
)
from nano_dsv41f.moe import init_moe, route_tokens
from nano_dsv41f.optimizer import parameter_rule_map, sinkhorn_balance
from nano_dsv41f.quantization import fake_e8m0_scale
from nano_dsv41f.rope import apply_partial_rope
from nano_dsv41f.training import selective_indexer_distillation_loss


def tiny_config(*, dspark=True) -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        d_model=32,
        d_ff=48,
        n_experts=4,
        experts_per_token=2,
        mhc_streams=2,
        mhc_sinkhorn_iters=4,
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
        dspark=DSparkConfig(
            enabled=dspark,
            n_layers=1,
            block_size=3,
            noise_token_id=0,
            target_layer_ids=(4, 5, 6),
            markov_rank=8,
            n_routed_experts=2,
            experts_per_token=1,
            confidence_head=True,
        ),
        # Unit tests exercise model semantics on one CPU. v5e sharding is tested
        # separately in tests/test_tpu.py and must not constrain tiny dimensions here.
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


def test_layer_specs_capture_swa_ced_full_reindex_reuse_groups():
    specs = build_layer_specs(tiny_config())
    assert [s.mode for s in specs] == [
        "swa",
        "full",
        "reuse",
        "full",
        "reuse",
        "reindex",
        "reuse",
    ]
    assert [s.owns_global_kv for s in specs] == [
        False,
        True,
        False,
        True,
        False,
        False,
        False,
    ]
    assert [s.is_index_source for s in specs] == [
        False,
        True,
        False,
        True,
        False,
        True,
        False,
    ]
    assert [s.compression_ratio for s in specs] == [0, 2, 2, 1, 1, 1, 1]


def test_partial_rope_inverse_restores_tail_and_leaves_nope_untouched():
    x = jax.random.normal(jax.random.PRNGKey(8), (2, 5, 3, 8))
    positions = jnp.tile(jnp.arange(5)[None, :], (2, 1))
    y = apply_partial_rope(x, positions, rotary_dim=2, base=10_000.0)
    z = apply_partial_rope(
        y, positions, rotary_dim=2, base=10_000.0, inverse=True
    )
    assert jnp.allclose(y[..., :-2], x[..., :-2])
    assert jnp.allclose(z, x, atol=1e-5)


def test_dense_lm_forward_skips_full_indexer_scoring_by_default():
    cfg = tiny_config(dspark=False)
    params = init_model(jax.random.PRNGKey(4), cfg)
    ids = jnp.arange(16, dtype=jnp.int32)[None, :] % cfg.vocab_size
    segments = jnp.array([[0] * 8 + [1] * 8], dtype=jnp.int32)
    _, aux = apply_model(params, cfg, ids, segment_ids=segments)
    assert all(layer["index_scores"] is None for layer in aux["layers"])
    assert all(layer["index_k"] is None for layer in aux["layers"])


def test_reference_model_diagnostic_indexer_exercises_reindex_state_machine():
    cfg = tiny_config()
    params = init_model(jax.random.PRNGKey(0), cfg)
    ids = jnp.arange(16, dtype=jnp.int32)[None, :] % cfg.vocab_size
    segments = jnp.array([[0] * 8 + [1] * 8], dtype=jnp.int32)

    logits, aux = apply_model(
        params, cfg, ids, segment_ids=segments, compute_indexer=True
    )

    assert logits.shape == (1, 16, cfg.vocab_size)
    assert jnp.all(jnp.isfinite(logits))
    assert aux["context_final"].shape == (1, 16, cfg.d_model)
    assert int(aux["final_global_source_layer"]) == cfg.csa2.context_layers
    assert len(aux["layers"]) == cfg.n_layers
    assert aux["dspark_context_features"].shape == (
        1,
        16,
        cfg.d_model * len(cfg.dspark.target_layer_ids),
    )

    assert aux["layers"][1]["index_scores"] is not None
    assert aux["layers"][2]["index_scores"] is None
    assert jnp.array_equal(
        aux["layers"][1]["index_topk_indices"],
        aux["layers"][2]["index_topk_indices"],
    )

    assert aux["layers"][3]["index_scores"] is not None
    assert aux["layers"][3]["index_candidate_mask"] is not None
    assert aux["layers"][4]["index_scores"] is None
    assert jnp.array_equal(
        aux["layers"][3]["index_topk_indices"],
        aux["layers"][4]["index_topk_indices"],
    )

    assert aux["layers"][5]["index_scores"] is not None
    assert aux["layers"][5]["index_candidate_mask"] is not None
    assert aux["layers"][6]["index_scores"] is None
    assert jnp.array_equal(
        aux["layers"][5]["index_topk_indices"],
        aux["layers"][6]["index_topk_indices"],
    )


def test_selective_distillation_uses_all_eligible_rows_when_budget_covers_them():
    cfg = tiny_config(dspark=False)
    params = init_model(jax.random.PRNGKey(41), cfg)
    ids = jnp.arange(28, dtype=jnp.int32)[None, :] % cfg.vocab_size
    segments = jnp.array([[0] * 14 + [1] * 14], dtype=jnp.int32)

    loss, aux = selective_indexer_distillation_loss(
        params, cfg, ids, segment_ids=segments, n_segments=2
    )
    assert jnp.isfinite(loss)
    assert int(aux["active_queries"]) == 12
    assert tuple(aux["student_score_shapes"]["L1"]) == (1, 28, 14)
    assert tuple(aux["student_score_shapes"]["L3"]) == (1, 28, 28)
    assert tuple(aux["student_score_shapes"]["L5"]) == (1, 28, 28)
    for key, indices in aux["teacher_query_indices"].items():
        assert set(map(int, indices[aux["teacher_query_valid"][key]])) == {12, 13, 26, 27}


def test_ratio_aware_indexer_eligibility_delays_r2_encoder_only():
    cfg = tiny_config(dspark=False)
    params = init_model(jax.random.PRNGKey(42), cfg)
    ids = jnp.arange(28, dtype=jnp.int32)[None, :] % cfg.vocab_size
    segments = jnp.array([[0] * 14 + [1] * 14], dtype=jnp.int32)
    ratio_cfg = replace(
        cfg,
        indexer_training=replace(
            cfg.indexer_training, eligibility_rule="local_plus_ratio_topk"
        ),
    )
    assert eligibility_position(
        local_window=4,
        retrieve_top_k=8,
        compression_ratio=1,
        rule="local_plus_ratio_topk",
    ) == 12
    assert eligibility_position(
        local_window=4,
        retrieve_top_k=8,
        compression_ratio=2,
        rule="local_plus_ratio_topk",
    ) == 20
    loss, aux = selective_indexer_distillation_loss(
        params, ratio_cfg, ids, segment_ids=segments, n_segments=2
    )
    assert jnp.isfinite(loss)
    assert int(aux["active_queries"]) == 8


def test_reference_model_can_be_jitted():
    cfg = tiny_config(dspark=False)
    params = init_model(jax.random.PRNGKey(3), cfg)
    ids = jnp.arange(8, dtype=jnp.int32)[None, :] % cfg.vocab_size
    segments = jnp.array(
        [[0, 0, 0, 0, 1, 1, 1, 1]], dtype=jnp.int32
    )
    forward = jax.jit(
        lambda p, x, s: apply_model(p, cfg, x, segment_ids=s)[0]
    )
    logits = forward(params, ids, segments)
    assert logits.shape == (1, 8, cfg.vocab_size)
    assert jnp.all(jnp.isfinite(logits))


def test_remat_policies_preserve_loss_and_support_backward():
    base = tiny_config(dspark=False)
    params = init_model(jax.random.PRNGKey(13), base)
    ids = jnp.arange(8, dtype=jnp.int32)[None, :] % base.vocab_size
    segments = jnp.array(
        [[0, 0, 0, 0, 1, 1, 1, 1]], dtype=jnp.int32
    )

    losses = []
    for policy in ("none", "attention", "block"):
        cfg = replace(base, remat=RematConfig(policy=policy))

        def loss_fn(p):
            logits, _ = apply_model(p, cfg, ids, segment_ids=segments)
            return jnp.mean(jnp.square(logits))

        loss, grads = jax.value_and_grad(loss_fn)(params)
        losses.append(loss)
        assert jnp.isfinite(loss)
        assert all(
            bool(jnp.all(jnp.isfinite(g)))
            for g in jax.tree_util.tree_leaves(grads)
        )

    assert jnp.allclose(jnp.stack(losses), losses[0], atol=1e-6)


def test_dspark_uses_separate_expert_count_and_returns_block_logits():
    cfg = tiny_config()
    params = init_model(jax.random.PRNGKey(21), cfg)
    ids = jnp.arange(12, dtype=jnp.int32)[None, :] % cfg.vocab_size
    anchors = jnp.array([[6]], dtype=jnp.int32)
    _, _, draft = apply_model_dspark(
        params, cfg, ids, anchor_positions=anchors
    )
    assert draft["draft_logits"].shape == (
        1,
        1,
        cfg.dspark.block_size,
        cfg.vocab_size,
    )
    assert draft["confidence"].shape == (1, 1, cfg.dspark.block_size)
    assert draft["router_indices"].shape[-1] == cfg.dspark.experts_per_token
    assert params["dspark"]["moe"]["router_bias"].shape == (
        cfg.dspark.n_routed_experts,
    )
    assert params["blocks"][0]["moe"]["router_bias"].shape == (
        cfg.n_experts,
    )


def test_optimizer_partition_is_visible_and_uses_headwise_q_muon():
    cfg = tiny_config(dspark=False)
    params = init_model(jax.random.PRNGKey(30), cfg)
    rules = parameter_rule_map(params, cfg)
    assert rules["embed"] == "sinkhorn"
    assert rules["lm_head"] == "sinkhorn"
    assert rules["blocks/1/attn/q_b/weight"] == "headwise_muon"
    assert rules["blocks/1/attn/q_norm/weight"] == "adamw"
    assert rules["blocks/1/attn/wo_a"] == "muon"
    assert rules["blocks/1/engram/table"] == "sinkhorn"


def test_sinkhorn_runs_exactly_alternating_normalizations():
    x = jnp.arange(1, 13, dtype=jnp.float32).reshape(3, 4)
    one = sinkhorn_balance(x, iters=1, eps=1e-12, row_mask_tau=0.0)
    two = sinkhorn_balance(x, iters=2, eps=1e-12, row_mask_tau=0.0)
    assert jnp.allclose(jnp.linalg.norm(one, axis=1), 1.0, atol=1e-5)
    assert jnp.allclose(jnp.linalg.norm(two, axis=0), 1.0, atol=1e-5)


def test_ue8m0_scale_rounds_required_exponent_up():
    x = jnp.array([1.0, 1.01, 1.9, 2.0, 2.01], dtype=jnp.float32)
    got = fake_e8m0_scale(x)
    expected = jnp.array([1.0, 2.0, 2.0, 2.0, 4.0], dtype=jnp.float32)
    assert jnp.array_equal(got, expected)


def test_candidate_blocks_pin_newest_reachable_block():
    logits = jnp.array([[[9.0, 8.0, 1.0, 0.0, -2.0, -3.0]]])
    lens = jnp.array([[6]], dtype=jnp.int32)
    mask = select_candidate_blocks(
        logits, lens, topk_blocks=2, block_size=2
    )
    assert mask.shape == logits.shape
    assert bool(mask[0, 0, 0])
    assert bool(mask[0, 0, 4])


def test_router_weights_normalize_over_selected_experts():
    params = init_moe(jax.random.PRNGKey(1), 8, 12, 4)
    x = jax.random.normal(jax.random.PRNGKey(2), (2, 3, 8))
    weights, indices = route_tokens(
        x, params, top_k=2, route_scale=1.5, eps=1e-20
    )
    assert indices.shape == (2, 3, 2)
    assert jnp.allclose(jnp.sum(weights, axis=-1), 1.5)
