from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from nano_dsv41f.config import TrainConfig
from nano_dsv41f.dspark import apply_dspark
from nano_dsv41f.model import apply_model, init_model
from nano_dsv41f.moe import apply_moe, apply_moe_dense_small
from nano_dsv41f.mtp import eligible_mtp_anchors, mtp_loss, sample_mtp_anchors
from nano_dsv41f.optimizer import init_optimizer_state
from nano_dsv41f.training import pretrain_loss, pretrain_step

from test_reference_model import tiny_config


def _setup(seq_len=24, seed=3):
    cfg = tiny_config()
    params = init_model(jax.random.PRNGKey(seed), cfg)
    ids = jax.random.randint(jax.random.PRNGKey(seed + 1), (2, seq_len), 3, cfg.vocab_size)
    _, aux = apply_model(params, cfg, ids, compute_indexer=False)
    return cfg, params, ids, aux["dspark_context_features"]


def _draft(cfg, params, ids, ctx, anchors, keep, **kwargs):
    return apply_dspark(params["dspark"], cfg, embed=params["embed"], lm_head=params["lm_head"],
                        input_ids=ids, context_features=ctx, anchor_positions=anchors,
                        block_keep_mask=keep, **kwargs)


def test_dense_small_moe_matches_reference_gather():
    cfg = tiny_config()
    params = init_model(jax.random.PRNGKey(0), cfg)["dspark"]["moe"]
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 7, cfg.d_model))
    ref, ref_aux = apply_moe(x, params, top_k=2, route_scale=1.5, swiglu_limit=3.0)
    new, new_aux = apply_moe_dense_small(x, params, top_k=2, route_scale=1.5, swiglu_limit=3.0)
    np.testing.assert_allclose(new, ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(new_aux["router_indices"], ref_aux["router_indices"])


def test_windowed_dense_dspark_matches_reference_forward():
    cfg, params, ids, ctx = _setup()
    # Includes an anchor near the start (partial window) and a dropped block.
    anchors = jnp.array([[1, 9, 17], [0, 12, 18]], dtype=jnp.int32)
    keep = jnp.array([[True, True, True], [True, False, True]])
    ref = _draft(cfg, params, ids, ctx, anchors, keep)
    new = _draft(cfg, params, ids, ctx, anchors, keep, windowed=True, dense_experts=True)
    for key in ("draft_logits", "base_logits", "confidence", "draft_hidden"):
        np.testing.assert_allclose(new[key], ref[key], rtol=2e-4, atol=2e-4, err_msg=key)
    finite = np.isfinite(ref["attention_lse"])
    np.testing.assert_array_equal(np.isfinite(new["attention_lse"]), finite)
    np.testing.assert_allclose(np.asarray(new["attention_lse"])[finite],
                               np.asarray(ref["attention_lse"])[finite], rtol=2e-4, atol=2e-4)


def test_segment_confined_window_ignores_previous_conversation():
    cfg, params, ids, ctx = _setup()
    seg = jnp.array([[0] * 10 + [1] * 14] * 2)
    anchors = jnp.array([[11], [11]], dtype=jnp.int32)
    keep = jnp.ones((2, 1), bool)
    base = _draft(cfg, params, ids, ctx, anchors, keep, segment_ids=seg, windowed=True)
    # Changing features of the previous segment (inside the raw window) changes nothing.
    ctx2 = ctx.at[:, 8:10].set(ctx[:, 8:10] + 5.0)
    again = _draft(cfg, params, ids, ctx2, anchors, keep, segment_ids=seg, windowed=True)
    np.testing.assert_allclose(again["draft_logits"], base["draft_logits"], rtol=1e-6, atol=1e-6)
    unconfined = _draft(cfg, params, ids, ctx2, anchors, keep, windowed=True)
    assert not np.allclose(unconfined["draft_logits"], base["draft_logits"])


def test_anchor_eligibility_requires_supervised_same_segment_future():
    block = 2
    seg = jnp.array([[0, 0, 0, 0, 1, 1, 1, 1]])
    target = jnp.array([[0, 1, 1, 1, 0, 1, 1, 0]], bool)
    ok = eligible_mtp_anchors(seg, block_size=block, target_mask=target)
    # a=0 -> targets 1,2; a=1 -> 2,3; a=4 -> 5,6. Others cross a segment or hit a non-target.
    np.testing.assert_array_equal(ok[0], [1, 1, 0, 0, 1, 0, 0, 0])
    pos, keep = sample_mtp_anchors(ok, anchors_per_row=5, step=0)
    assert int(keep.sum()) == 3
    assert set(np.asarray(pos)[np.asarray(keep)].tolist()) == {0, 1, 4}


def test_mtp_loss_is_finite_and_trains_backbone_and_dspark():
    cfg, params, ids, _ = _setup()
    seg = jnp.zeros_like(ids)

    def total(p, weight):
        loss, metrics = pretrain_loss(p, cfg, ids, segment_ids=seg, mtp_weight=weight,
                                      mtp_anchors_per_row=4)
        return loss, metrics

    (loss0, m0), g0 = jax.value_and_grad(total, has_aux=True)(params, 0.0)
    (loss1, m1), g1 = jax.value_and_grad(total, has_aux=True)(params, 0.3)
    assert float(m0["mtp_loss"]) == 0.0 and float(m1["mtp_loss"]) > 0
    np.testing.assert_allclose(loss1, loss0 + 0.3 * m1["mtp_loss"], rtol=1e-5)
    assert int(m1["mtp_blocks"]) == 8
    dspark_grad = jax.tree_util.tree_leaves(g1["dspark"])
    assert all(np.all(np.isfinite(g)) for g in jax.tree_util.tree_leaves(g1))
    assert any(float(jnp.abs(g).max()) > 0 for g in dspark_grad)
    assert all(float(jnp.abs(g).max()) == 0 for g in jax.tree_util.tree_leaves(g0["dspark"]))
    # DSpark reads the *inputs* of layers 4-6, so layer 4's weights receive MTP gradient
    # while the final layer's cannot (its output never feeds the draft head).
    def max_diff(i):
        diff = jax.tree_util.tree_map(lambda a, b: jnp.abs(a - b).max(), g1["blocks"][i], g0["blocks"][i])
        return max(float(x) for x in jax.tree_util.tree_leaves(diff))
    assert max_diff(4) > 0
    assert max_diff(len(params["blocks"]) - 1) == 0


def test_pretrain_step_updates_dspark_only_when_mtp_enabled():
    cfg, params, ids, _ = _setup()
    seg = jnp.zeros_like(ids)
    state = init_optimizer_state(params, cfg)
    train = TrainConfig(total_steps=10, seq_len=ids.shape[1], warmup_steps=1)
    for weight, should_change in ((0.0, False), (0.2, True)):
        new, _, metrics = pretrain_step(params, state, cfg, train, ids, segment_ids=seg,
                                        step=jnp.asarray(1), mtp_weight=weight,
                                        mtp_anchors_per_row=4)
        changed = any(not np.array_equal(a, b) for a, b in zip(
            jax.tree_util.tree_leaves(new["dspark"]), jax.tree_util.tree_leaves(params["dspark"])))
        assert changed == should_change
        assert np.isfinite(float(metrics["loss"]))
