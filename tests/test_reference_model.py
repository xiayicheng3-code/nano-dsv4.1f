import jax
import jax.numpy as jnp

from nano_dsv41f.config import (
    AttentionConfig,
    CSA2Config,
    EngramConfig,
    ModelConfig,
)
from nano_dsv41f.csa2 import apply_csa2_attention, init_csa2_attention
from nano_dsv41f.model import apply_model, build_layer_specs, init_model
from nano_dsv41f.moe import init_moe, route_tokens


def tiny_config() -> ModelConfig:
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
            o_rank=16,
            local_window=4,
            retrieve_top_k=8,
        ),
        csa2=CSA2Config(
            context_layers=4,
            generation_layers=4,
            retriever_group_size=2,
            context_compression_ratio=2,
            generation_compression_ratio=1,
        ),
        engram=EngramConfig(
            enabled=True,
            layer_ids=(1, 3),
            table_size=128,
            max_ngram_size=3,
            n_hash_heads=2,
            head_dim=8,
        ),
    )


def test_layer_specs_capture_nano_ced_and_csa2_modes():
    specs = build_layer_specs(tiny_config())
    assert [s.mode for s in specs] == [
        "full",
        "reuse",
        "full",
        "reuse",
        "full",
        "reuse",
        "reindex",
        "reuse",
    ]
    assert [s.owns_global_kv for s in specs] == [
        True,
        False,
        True,
        False,
        True,
        False,
        False,
        False,
    ]
    assert [s.compression_ratio for s in specs] == [
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        1,
    ]


def test_full_layer_can_publish_global_kv_from_explicit_ced_source():
    params = init_csa2_attention(
        jax.random.PRNGKey(11),
        dim=8,
        n_heads=2,
        head_dim=4,
        q_rank=4,
        o_rank=4,
        owns_global_kv=True,
    )
    x = jnp.zeros((1, 8, 8), dtype=jnp.float32)
    context_source = jnp.ones((1, 8, 8), dtype=jnp.float32)
    segments = jnp.zeros((1, 8), dtype=jnp.int32)

    _, state_from_x, _ = apply_csa2_attention(
        x,
        segments,
        params,
        None,
        layer_id=4,
        mode="full",
        owns_global_kv=True,
        compression_ratio=1,
        n_heads=2,
        head_dim=4,
        local_window=4,
        norm_eps=1e-6,
    )
    _, state_from_context, _ = apply_csa2_attention(
        x,
        segments,
        params,
        None,
        layer_id=4,
        mode="full",
        owns_global_kv=True,
        compression_ratio=1,
        n_heads=2,
        head_dim=4,
        local_window=4,
        norm_eps=1e-6,
        global_source=context_source,
    )

    assert not jnp.allclose(state_from_x.kv, state_from_context.kv)
    assert int(state_from_context.source_layer) == 4


def test_reference_model_forward_is_finite_on_packed_segments():
    cfg = tiny_config()
    params = init_model(jax.random.PRNGKey(0), cfg)
    ids = jnp.arange(16, dtype=jnp.int32)[None, :] % cfg.vocab_size
    # Every segment length is divisible by r=2, so compression groups never cross packs.
    segments = jnp.array([[0] * 8 + [1] * 8], dtype=jnp.int32)

    logits, aux = apply_model(params, cfg, ids, segment_ids=segments)

    assert logits.shape == (1, 16, cfg.vocab_size)
    assert jnp.all(jnp.isfinite(logits))
    assert aux["context_final"].shape == (1, 16, cfg.d_model)
    # The first generation-side Full layer owns the CED global KV bank.
    assert int(aux["final_global_source_layer"]) == cfg.csa2.context_layers
    assert len(aux["layers"]) == cfg.n_layers


def test_reference_model_can_be_jitted():
    cfg = ModelConfig(
        vocab_size=32,
        d_model=16,
        d_ff=24,
        n_experts=2,
        experts_per_token=1,
        mhc_streams=2,
        mhc_sinkhorn_iters=2,
        attention=AttentionConfig(
            d_model=16,
            n_heads=2,
            head_dim=8,
            q_rank=8,
            o_rank=8,
            local_window=4,
            retrieve_top_k=4,
        ),
        csa2=CSA2Config(
            context_layers=2,
            generation_layers=2,
            retriever_group_size=2,
            context_compression_ratio=2,
            generation_compression_ratio=1,
        ),
        engram=EngramConfig(
            enabled=True,
            layer_ids=(1,),
            table_size=64,
            max_ngram_size=3,
            n_hash_heads=1,
            head_dim=8,
        ),
    )
    params = init_model(jax.random.PRNGKey(3), cfg)
    ids = jnp.arange(8, dtype=jnp.int32)[None, :]
    segments = jnp.array([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=jnp.int32)
    forward = jax.jit(
        lambda p, x, s: apply_model(p, cfg, x, segment_ids=s)[0]
    )
    logits = forward(params, ids, segments)
    assert logits.shape == (1, 8, cfg.vocab_size)
    assert jnp.all(jnp.isfinite(logits))


def test_router_weights_normalize_over_selected_experts():
    params = init_moe(jax.random.PRNGKey(1), 8, 12, 4)
    x = jax.random.normal(jax.random.PRNGKey(2), (2, 3, 8))
    weights, indices = route_tokens(x, params, top_k=2, route_scale=1.5)
    assert indices.shape == (2, 3, 2)
    assert jnp.allclose(jnp.sum(weights, axis=-1), 1.5)
