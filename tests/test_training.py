import jax
import jax.numpy as jnp

from nano_dsv41f.config import ModelConfig, TrainConfig
from nano_dsv41f.optimizer import init_optimizer_state, optimizer_step
from nano_dsv41f.training import causal_lm_loss, indexer_phase_enabled


def test_packed_causal_lm_loss_never_crosses_segment_boundary():
    ids = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
    segments = jnp.array([[0, 0, 1, 1]], dtype=jnp.int32)
    logits = jnp.zeros((1, 4, 4), dtype=jnp.float32)
    logits = logits.at[0, 0, 1].set(8.0)  # valid 0 -> 1
    logits = logits.at[0, 2, 3].set(8.0)  # valid 2 -> 3

    loss_a, count_a = causal_lm_loss(logits, ids, segments)
    # Position 1 would predict token 2, but that transition crosses packed examples.
    poisoned = logits.at[0, 1].set(jnp.array([100.0, -100.0, -100.0, -100.0]))
    loss_b, count_b = causal_lm_loss(poisoned, ids, segments)

    assert int(count_a) == 2
    assert int(count_b) == 2
    assert jnp.allclose(loss_a, loss_b)


def test_optimizer_phase_freeze_prevents_weight_decay_on_inactive_families():
    cfg = ModelConfig()
    train_cfg = TrainConfig(total_steps=10, warmup_steps=1)
    params = {
        "blocks": (
            {
                "attn": {
                    "indexer": {
                        "wq_b": {"weight": jnp.ones((2, 2), dtype=jnp.float32)}
                    }
                },
                "plain": {"weight": jnp.ones((2, 2), dtype=jnp.float32)},
            },
        ),
        "dspark": {"draft_weight": jnp.ones((2, 2), dtype=jnp.float32)},
    }
    grads = jax.tree_util.tree_map(jnp.zeros_like, params)
    state = init_optimizer_state(params, cfg)
    updated, _, _ = optimizer_step(
        params,
        grads,
        state,
        step=jnp.asarray(0, dtype=jnp.int32),
        config=cfg,
        train_config=train_cfg,
        train_indexer=False,
        train_dspark=False,
    )

    assert jnp.array_equal(
        updated["blocks"][0]["attn"]["indexer"]["wq_b"]["weight"],
        params["blocks"][0]["attn"]["indexer"]["wq_b"]["weight"],
    )
    assert jnp.array_equal(
        updated["dspark"]["draft_weight"], params["dspark"]["draft_weight"]
    )
    # An active matrix with zero gradient still receives Muon's decoupled weight decay.
    assert not jnp.array_equal(
        updated["blocks"][0]["plain"]["weight"],
        params["blocks"][0]["plain"]["weight"],
    )


def test_python_phase_selector_matches_configured_interval():
    cfg = ModelConfig()
    train_cfg = TrainConfig(total_steps=100)
    assert not indexer_phase_enabled(54, train_cfg, cfg)
    assert indexer_phase_enabled(55, train_cfg, cfg)
    assert indexer_phase_enabled(90, train_cfg, cfg)
    assert not indexer_phase_enabled(91, train_cfg, cfg)
