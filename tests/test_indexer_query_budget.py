from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nano_dsv41f.config import IndexerConfig, IndexerTrainingConfig
from nano_dsv41f.indexer import budgeted_teacher_indices, eligible_teacher_queries
from nano_dsv41f.model import apply_model, init_model
from nano_dsv41f.training import _selective_indexer_from_backbone_aux
from test_reference_model import tiny_config


@pytest.mark.parametrize("budget", [1, 7, 128])
def test_global_budget_handles_uneven_rows_packing_and_padding(budget):
    segments = jnp.array([[7] * 8 + [19] * 8, [21] * 16, [30] * 4 + [31] * 12])
    mask = jnp.ones_like(segments, dtype=bool).at[0, 7].set(False).at[1].set(False)
    eligible = np.asarray(eligible_teacher_queries(segments, min_local_position=4, token_mask=mask))
    indices, valid = jax.jit(lambda step: budgeted_teacher_indices(
        segments, query_budget=budget, min_local_position=4, token_mask=mask, step=step,
    ))(jnp.asarray(9))
    indices, valid = np.asarray(indices), np.asarray(valid)
    assert indices.shape == valid.shape == (3, min(budget, 16))
    assert valid.sum() == min(budget, eligible.sum())
    assert not valid[1].any()
    for row in range(3):
        selected = indices[row, valid[row]]
        assert len(set(selected.tolist())) == len(selected)
        assert eligible[row, selected].all()
        assert (indices[row, ~valid[row]] == 0).all()


def test_sampling_refreshes_with_step_is_reproducible_and_does_not_recompile():
    segments = jnp.zeros((1, 64), jnp.int32)
    traces = []
    @jax.jit
    def select(step):
        traces.append(True)
        return budgeted_teacher_indices(segments, query_budget=8, min_local_position=12,
                                        seed=37, step=step)
    first, valid = select(jnp.asarray(0))
    again, _ = select(jnp.asarray(0))
    second, _ = select(jnp.asarray(1))
    np.testing.assert_array_equal(first, again)
    assert valid.all()
    assert set(np.asarray(first).ravel()) != set(np.asarray(second).ravel())
    assert len(traces) == 1
    seen = set()
    for step in range(64):
        seen.update(np.asarray(select(jnp.asarray(step))[0]).ravel())
    assert seen == set(range(12, 64))


@pytest.mark.parametrize("budget", [0, -1, 1.5, True])
def test_invalid_budget_is_rejected(budget):
    with pytest.raises(ValueError, match="query_budget"):
        IndexerTrainingConfig(query_budget=budget)


@pytest.mark.parametrize('blocks', [16, 64])
def test_candidate_pool_must_leave_room_for_topk_selection(blocks):
    with pytest.raises(ValueError, match='candidate pool capacity must exceed top_k'):
        IndexerConfig(candidate_topk_blocks=blocks)
    # An explicitly disabled hierarchy does not constrain retrieval width.
    IndexerConfig(candidate_source_layer=-1, candidate_topk_blocks=blocks)


def test_no_eligible_queries_has_only_safe_dummy_slots():
    indices, valid = budgeted_teacher_indices(jnp.zeros((2, 8), jnp.int32),
                                              query_budget=16, min_local_position=12)
    assert not valid.any()
    assert (indices == 0).all()


@pytest.mark.parametrize("batch", [1, 3])
def test_context_sharded_query_budget_matches_global_selection(batch):
    if len(jax.devices()) < 8:
        pytest.skip("Run with XLA_FLAGS=--xla_force_host_platform_device_count=8")
    from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P
    mesh = Mesh(np.asarray(jax.devices()[:8]).reshape(2, 4), ("x", "y"),
                axis_types=(AxisType.Auto, AxisType.Auto))
    sharding = NamedSharding(mesh, P(None, ("x", "y")))
    segments = jnp.broadcast_to(jnp.arange(64)[None] // 32, (batch, 64))
    mask = jnp.ones_like(segments, dtype=bool).at[:, 31].set(False)
    def select(seg, real, step):
        return budgeted_teacher_indices(seg, query_budget=7, min_local_position=12,
                                        token_mask=real, step=step)
    expected = jax.jit(select)(segments, mask, jnp.asarray(13))
    actual = jax.jit(select, in_shardings=(sharding, sharding, NamedSharding(mesh, P())))(
        jax.device_put(segments, sharding), jax.device_put(mask, sharding), jnp.asarray(13))
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)
    assert int(actual[1].sum()) == 7


@pytest.fixture(scope="module")
def backbone():
    cfg = tiny_config(dspark=False)
    params = init_model(jax.random.key(71), cfg)
    ids = jnp.arange(28, dtype=jnp.int32)[None, :] % cfg.vocab_size
    segments = jnp.array([[0] * 14 + [1] * 14])
    mask = jnp.ones_like(ids, dtype=bool).at[0, 13].set(False)
    _, aux = apply_model(params, cfg, ids, segment_ids=segments, token_mask=mask,
                         compute_indexer=False)
    return cfg, params, segments, mask, aux


def objective(config, segments, mask, aux, *, step=11):
    return jax.jit(jax.value_and_grad(lambda p: _selective_indexer_from_backbone_aux(
        p, config, segment_ids=segments, token_mask=mask, n_segments=None,
        backbone_aux=aux, step=jnp.asarray(step),
    ), has_aux=True))


def test_budget_covering_all_queries_matches_exhaustive_loss_and_gradients(backbone):
    cfg, params, segments, mask, aux = backbone
    actual, actual_grad = objective(cfg, segments, mask, aux)(params)
    exhaustive = replace(cfg, indexer_training=replace(cfg.indexer_training, teacher_queries="all_eligible"))
    expected, expected_grad = objective(exhaustive, segments, mask, aux)(params)
    np.testing.assert_allclose(actual[0], expected[0], rtol=2e-6, atol=1e-7)
    assert int(actual[1]["active_queries"]) == 9  # 3 non-padding queries * 3 retrievers
    nonzero_paths = []
    for (path, a), b in zip(jax.tree_util.tree_flatten_with_path(actual_grad)[0], jax.tree.leaves(expected_grad)):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-7)
        assert np.isfinite(a).all()
        if np.any(np.asarray(a) != 0):
            nonzero_paths.append(str(path))
    assert nonzero_paths and all("indexer" in p for p in nonzero_paths)
    for layer in (1, 3, 5):
        assert np.any(actual_grad["blocks"][layer]["attn"]["indexer"]["wq_b"] != 0)
    for layer in (1, 3):
        assert np.any(actual_grad["blocks"][layer]["attn"]["indexer"]["wk"] != 0)


def test_default_l5_uses_full_history_and_candidate_mask_is_opt_in():
    cfg = tiny_config(dspark=False)
    params = init_model(jax.random.key(71), cfg)
    ids = jnp.arange(28, dtype=jnp.int32)[None, :] % cfg.vocab_size
    # Enough same-document history to make the corrected pool selective.
    segments = jnp.zeros_like(ids)
    mask = jnp.ones_like(ids, dtype=bool)
    _, aux = apply_model(params, cfg, ids, segment_ids=segments, token_mask=mask,
                         compute_indexer=False)
    assert not cfg.indexer_training.apply_candidate_mask
    (loss, unmasked), _ = objective(cfg, segments, mask, aux)(params)
    restricted_cfg = replace(cfg, indexer_training=replace(cfg.indexer_training, apply_candidate_mask=True))
    (restricted_loss, restricted), grads = objective(restricted_cfg, segments, mask, aux)(params)
    assert np.isfinite(loss) and np.isfinite(restricted_loss)
    valid = np.asarray(unmasked["teacher_query_valid"]["L5"])
    for key in ("L1", "L3", "L5"):
        np.testing.assert_array_equal(unmasked["teacher_query_indices"][key], unmasked["teacher_query_indices"]["L3"])
    full_counts = np.asarray(unmasked["student_key_counts"]["L5"])[valid]
    l3_counts = np.asarray(unmasked["student_key_counts"]["L3"])[valid]
    restricted_counts = np.asarray(restricted["student_key_counts"]["L5"])[valid]
    np.testing.assert_array_equal(full_counts, l3_counts)
    assert (restricted_counts <= full_counts).all()
    assert (restricted_counts < full_counts).any()
    assert (restricted_counts <= cfg.indexer.candidate_topk_blocks * cfg.indexer.candidate_block_size).all()
    assert (restricted_counts > 0).all()
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(grads))


def test_empty_budget_selection_has_zero_loss_and_finite_zero_gradients(backbone):
    cfg, params, segments, mask, aux = backbone
    (loss, metrics), grads = objective(cfg, segments, jnp.zeros_like(mask), aux)(params)
    assert float(loss) == 0 and int(metrics["active_queries"]) == 0
    assert all(np.isfinite(x).all() and not np.any(x) for x in jax.tree.leaves(grads))
