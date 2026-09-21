import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dataclasses import replace
from jax.sharding import AxisType, Mesh
from nano_dsv41f.tpu_native import TPUNativeConfig, TPUNativeState

@pytest.mark.parametrize('dp', [2, 4])
@pytest.mark.parametrize('ratio', [1, 2])
def test_attention_dp_cp_matches_dense_forward_backward(dp, ratio):
    if len(jax.devices()) < 8:
        pytest.skip("requires eight CPU devices")
    from nano_dsv41f.csa2 import SharedCSA2State
    from nano_dsv41f.tpu_native import _combined_splash_attention, combined_csa2_mask
    mesh = Mesh(np.asarray(jax.devices()[:8]).reshape(2, 4), ('x', 'y'), axis_types=(AxisType.Auto, AxisType.Auto))
    batch, length = 4, 512
    state = TPUNativeState(mesh, TPUNativeConfig(splash_interpret=True, attention_data_shards=dp, need_teacher_lse=True))
    q = jax.random.normal(jax.random.key(31), (batch, length, 1, 128)) * .1
    kv = jax.random.normal(jax.random.key(32), (batch, length, 128)) * .1
    gkv = jax.random.normal(jax.random.key(33), (batch, length // ratio, 128)) * .1
    seg = jnp.tile(jnp.repeat(jnp.arange(4), length // 4)[None], (batch, 1))
    sink = jnp.array([.2])
    def run(q, kv, gkv, sink):
        shared = SharedCSA2State(kv=gkv, latent=gkv, index_k=None,
            segment_ids=seg[:, ::ratio], group_start_positions=jnp.tile((jnp.arange(length)[None] % (length // 4)), (batch, 1))[:, ::ratio],
            source_layer=jnp.array(1), compress_ratio=jnp.array(ratio),
            latest_topk_indices=None, latest_topk_values=None,
            index_source_layer=jnp.array(-1), candidate_mask=None)
        return _combined_splash_attention(q, kv, seg, shared, compression_ratio=ratio,
                                          local_window=8, sink=sink, state=state)[0]
    mask = jnp.asarray(combined_csa2_mask(length, local_window=8, global_kv_len=length // ratio,
                                        compression_ratio=ratio))[None]
    mask &= seg[:, :, None] == jnp.concatenate((seg, seg[:, ::ratio]), -1)[:, None, :]
    def reference(q, kv, gkv, sink):
        keys = jnp.concatenate((kv, gkv), 1)
        logits = jnp.einsum('bthd,bkd->bthk', q * jnp.float32(128 ** -.5), keys)
        logits = jnp.where(mask[:, :, None, :], logits, -jnp.inf)
        logits = jnp.concatenate((logits, jnp.broadcast_to(sink[None, None, :, None], (batch, length, 1, 1))), -1)
        prob = jax.nn.softmax(logits, -1)[..., :-1]
        return jnp.einsum('bthk,bkd->bthd', prob, keys)
    np.testing.assert_allclose(jax.jit(run)(q, kv, gkv, sink), reference(q, kv, gkv, sink), atol=3e-6, rtol=5e-5)
    def grads(fn):
        return jax.jit(jax.grad(lambda *a: jnp.square(fn(*a)).sum(), argnums=(0, 1, 2, 3)))(q, kv, gkv, sink)
    for actual, expected in zip(grads(run), grads(reference)):
        np.testing.assert_allclose(actual, expected, atol=5e-6, rtol=3e-4)


def test_cp2_dp4_full_late_optimizer_step_matches_reference():
    """Exercise DP reduction, EP8 reshards, global query budget, and the optimizer together."""
    if len(jax.devices()) < 8:
        pytest.skip("requires eight CPU devices")
    from test_reference_model import tiny_config
    from nano_dsv41f import (TrainConfig, compile_pretrain_step, init_model_sharded,
        init_optimizer_state_sharded, make_v5e_mesh, put_training_batch)
    from nano_dsv41f.training import pretrain_step
    cfg = tiny_config(dspark=False)
    cfg = replace(cfg, n_experts=8,
        parallelism=replace(cfg.parallelism, vocab_shard=8, engram_table_shard=8,
                            expert_shard=8, attention_context_shard=2,
                            attention_data_shard=4, indexer_context_shard=2),
        indexer_training=replace(cfg.indexer_training, query_budget=8))
    mesh = make_v5e_mesh()
    params, specs, _ = init_model_sharded(jax.random.key(101), cfg, mesh)
    opt, _ = init_optimizer_state_sharded(params, specs, cfg, mesh)
    ids = np.random.default_rng(100).integers(0, cfg.vocab_size, (4, 256), dtype=np.int32)
    segments = np.tile(np.repeat(np.arange(2, dtype=np.int32), 128)[None], (4, 1))
    mask = np.ones_like(ids, dtype=bool)
    mask[:, 127] = False
    ids, segments, mask = put_training_batch(ids, segments, mask, cfg, mesh)
    step_id = jnp.asarray(6000, jnp.int32)
    train = TrainConfig(seq_len=256)
    # Independent dense oracle, with no active native dispatch context.
    host_params = jax.tree.map(lambda x: jnp.asarray(np.asarray(x)), params)
    host_opt = jax.tree.map(lambda x: jnp.asarray(np.asarray(x)), opt)
    ref = jax.jit(lambda p, o: pretrain_step(p, o, cfg, train,
        jnp.asarray(np.asarray(ids)), segment_ids=jnp.asarray(np.asarray(segments)),
        token_mask=jnp.asarray(np.asarray(mask)), step=step_id, include_indexer=True))
    expected = ref(host_params, host_opt)
    native = compile_pretrain_step(params, opt, specs, cfg, train, mesh,
        include_indexer=True, native_config=TPUNativeConfig(splash_interpret=True))
    actual = native(params, opt, ids, segments, step_id, mask)
    jax.block_until_ready(actual)
    np.testing.assert_allclose(actual[2]["loss"], expected[2]["loss"], atol=2e-5, rtol=2e-5)
    np.testing.assert_array_equal(actual[2]["router_loads"], expected[2]["router_loads"])
    physical = np.asarray(actual[2]["expert_loads_by_layer"])
    assert physical.shape == (cfg.n_layers, cfg.n_experts)
    np.testing.assert_array_equal(physical.sum(axis=0), actual[2]["expert_loads"])
    np.testing.assert_array_equal(physical.sum(axis=1),
                                  np.full(cfg.n_layers, ids.size * cfg.experts_per_token))
    np.testing.assert_array_equal(np.asarray(actual[2]["router_loads"]).sum(axis=1),
                                  np.full(cfg.n_layers, int(mask.sum()) * cfg.experts_per_token))
    assert int(actual[2]["indexer"]["active_queries"]) == 24
    for a, b in zip(jax.tree.leaves(actual[:2]), jax.tree.leaves(expected[:2])):
        np.testing.assert_allclose(a, b, atol=3e-5, rtol=5e-3)
