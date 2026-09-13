import jax
import jax.numpy as jnp

from nano_dsv41f.config import (
    AttentionConfig,
    CSA2Config,
    DSparkConfig,
    EngramConfig,
    IndexerConfig,
    ModelConfig,
    RopeConfig,
)
from nano_dsv41f.indexer_scorer import select_candidate_blocks
from nano_dsv41f.model import apply_model, apply_model_dspark, build_layer_specs, init_model
from nano_dsv41f.moe import init_moe, route_tokens
from nano_dsv41f.optimizer import parameter_rule_map
from nano_dsv41f.rope import apply_partial_rope


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
            retrieve_top_k=8,
            rope=RopeConfig(rope_head_dim=2, original_seq_len=0),
        ),
        csa2=CSA2Config(
            context_layers=3,
            generation_layers=2,
            context_swa_only_layers=1,
            context_retriever_group_size=2,
            generation_retriever_group_size=2,
            context_compression_ratio=2,
            generation_compression_ratio=1,
        ),
        indexer=IndexerConfig(n_heads=2, head_dim=4, top_k=8),
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
            target_layer_ids=(3, 4),
            markov_rank=8,
            n_routed_experts=2,
            experts_per_token=1,
            confidence_head=True,
        ),
    )


def test_layer_specs_capture_swa_ced_and_two_layer_retriever_groups():
    specs = build_layer_specs(tiny_config())
    assert [s.mode for s in specs] == ["swa", "full", "reuse", "full", "reuse"]
    assert [s.owns_global_kv for s in specs] == [False, True, False, True, False]
    assert [s.is_index_source for s in specs] == [False, True, False, True, False]
    assert [s.compression_ratio for s in specs] == [0, 2, 2, 1, 1]


def test_partial_rope_inverse_restores_tail_and_leaves_nope_untouched():
    x = jax.random.normal(jax.random.PRNGKey(8), (2, 5, 3, 8))
    positions = jnp.tile(jnp.arange(5)[None, :], (2, 1))
    y = apply_partial_rope(x, positions, rotary_dim=2, base=10_000.0)
    z = apply_partial_rope(y, positions, rotary_dim=2, base=10_000.0, inverse=True)
    assert jnp.allclose(y[..., :-2], x[..., :-2])
    assert jnp.allclose(z, x, atol=1e-5)


def test_reference_model_forward_is_finite_on_packed_segments():
    cfg = tiny_config()
    params = init_model(jax.random.PRNGKey(0), cfg)
    ids = jnp.arange(16, dtype=jnp.int32)[None, :] % cfg.vocab_size
    segments = jnp.array([[0] * 8 + [1] * 8], dtype=jnp.int32)

    logits, aux = apply_model(params, cfg, ids, segment_ids=segments)

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


def test_reference_model_can_be_jitted():
    cfg = tiny_config(dspark=False)
    params = init_model(jax.random.PRNGKey(3), cfg)
    ids = jnp.arange(8, dtype=jnp.int32)[None, :] % cfg.vocab_size
    segments = jnp.array([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=jnp.int32)
    forward = jax.jit(lambda p, x, s: apply_model(p, cfg, x, segment_ids=s)[0])
    logits = forward(params, ids, segments)
    assert logits.shape == (1, 8, cfg.vocab_size)
    assert jnp.all(jnp.isfinite(logits))


def test_dspark_uses_separate_expert_count_and_returns_block_logits():
    cfg = tiny_config()
    params = init_model(jax.random.PRNGKey(21), cfg)
    ids = jnp.arange(12, dtype=jnp.int32)[None, :] % cfg.vocab_size
    anchors = jnp.array([[6]], dtype=jnp.int32)
    _, _, draft = apply_model_dspark(params, cfg, ids, anchor_positions=anchors)
    assert draft["draft_logits"].shape == (1, 1, cfg.dspark.block_size, cfg.vocab_size)
    assert draft["confidence"].shape == (1, 1, cfg.dspark.block_size)
    assert draft["router_indices"].shape[-1] == cfg.dspark.experts_per_token
    assert params["dspark"]["moe"]["router_bias"].shape == (
        cfg.dspark.n_routed_experts,
    )
    assert params["blocks"][0]["moe"]["router_bias"].shape == (cfg.n_experts,)


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


def test_candidate_blocks_pin_newest_reachable_block():
    # Highest score is in block 0, but partial newest block 2 must also survive.
    logits = jnp.array([[[9.0, 8.0, 1.0, 0.0, -2.0, -3.0]]])
    lens = jnp.array([[6]], dtype=jnp.int32)
    mask = select_candidate_blocks(logits, lens, topk_blocks=2, block_size=2)
    assert mask.shape == logits.shape
    assert bool(mask[0, 0, 0])
    assert bool(mask[0, 0, 4])


def test_router_weights_normalize_over_selected_experts():
    params = init_moe(jax.random.PRNGKey(1), 8, 12, 4)
    x = jax.random.normal(jax.random.PRNGKey(2), (2, 3, 8))
    weights, indices = route_tokens(x, params, top_k=2, route_scale=1.5, eps=1e-20)
    assert indices.shape == (2, 3, 2)
    assert jnp.allclose(jnp.sum(weights, axis=-1), 1.5)
