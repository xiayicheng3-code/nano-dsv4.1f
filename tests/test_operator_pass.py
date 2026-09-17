from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh

from nano_dsv41f.moe import apply_moe, init_moe
from nano_dsv41f.tpu_moe import apply_moe_v5e_multi
from nano_dsv41f.tpu_native import TPUNativeConfig, TPUNativeState
from nano_dsv41f.training import selected_teacher_mass_native
from nano_dsv41f.optimizer import hybrid_newton_schulz


@pytest.mark.parametrize('shards', [1, 8])
@pytest.mark.parametrize('skewed', [False, True])
def test_dropless_expert_forward_and_backward(shards, skewed):
    if len(jax.devices()) < shards:
        pytest.skip('Run with XLA_FLAGS=--xla_force_host_platform_device_count=8')
    mesh = Mesh(np.asarray(jax.devices()[:shards]), ('x',), axis_types=(AxisType.Auto,))
    state = TPUNativeState(mesh, TPUNativeConfig(moe_capacity_factor=1.0, moe_capacity_multiple=2))
    x = jax.random.normal(jax.random.key(1), (2, 16, 4)) * 0.3
    p = init_moe(jax.random.key(2), 4, 8, 16)
    if skewed:
        p['router_bias'] = jnp.arange(16, dtype=jnp.float32) * 100
    kw = dict(top_k=2, route_scale=1.5, swiglu_limit=10., eps=1e-20)
    def ref(x, p): return apply_moe(x, p, **kw)[0]
    def native(x, p): return apply_moe_v5e_multi(x, p, state=state, **kw)[0]
    ref_y = jax.jit(ref)(x, p)
    got, aux = jax.jit(lambda x, p: apply_moe_v5e_multi(x, p, state=state, **kw))(x, p)
    np.testing.assert_allclose(got, ref_y, atol=2e-6, rtol=2e-5)
    assert int(aux['expert_loads'].sum()) == 64
    assert int(aux['expert_dropped'].sum()) == 0
    if skewed:
        assert int(aux['expert_overflow'].sum()) > 0
    def objective(fn, x, p): return jnp.sum(jnp.sin(fn(x, p)))
    expected = jax.jit(jax.grad(lambda x, p: objective(ref, x, p), argnums=(0, 1)))(x, p)
    actual = jax.jit(jax.grad(lambda x, p: objective(native, x, p), argnums=(0, 1)))(x, p)
    for a, b in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
        np.testing.assert_allclose(a, b, atol=3e-6, rtol=3e-4)


@pytest.mark.parametrize('ratio', [1, 2])
def test_selected_teacher_matches_complete_softmax(ratio):
    q = jax.random.normal(jax.random.key(3), (1, 12, 2, 4))
    kv = jax.random.normal(jax.random.key(4), (1, 12, 4))
    gkv = jax.random.normal(jax.random.key(5), (1, 12 // ratio, 4))
    segments = jnp.array([[0] * 6 + [1] * 6])
    sink = jnp.array([0.4, -0.3])
    ix = jnp.array([[0, 5, 6, 11]])  # includes queries with no global history
    pos = jnp.arange(12)
    local_valid = ((pos[None, :] <= pos[:, None]) & (pos[None, :] > pos[:, None] - 3))[None]
    local_valid &= segments[:, :, None] == segments[:, None, :]
    global_valid = (pos[::ratio][None, :] <= pos[:, None] - 3)[None]
    global_valid &= segments[:, :, None] == segments[:, None, ::ratio]
    l = jnp.einsum('bthd,bkd->bthk', q * 0.5, kv)
    g = jnp.einsum('bthd,bkd->bthk', q * 0.5, gkv)
    all_logits = jnp.concatenate((jnp.where(local_valid[:, :, None], l, -jnp.inf),
                                  jnp.where(global_valid[:, :, None], g, -jnp.inf),
                                  jnp.broadcast_to(sink[None, None, :, None], (1, 12, 2, 1))), -1)
    full = jax.nn.softmax(all_logits, -1)[..., 12:-1].sum(-2)
    got = selected_teacher_mass_native(q[:, ix[0]], gkv, kv, ix, segments,
                                       global_valid[:, ix[0]], local_window=3, sink=sink)
    np.testing.assert_allclose(got, full[:, ix[0]], atol=1e-6, rtol=2e-6)


def test_factored_muon_matches_original_iterations():
    x = jax.random.normal(jax.random.key(6), (8, 12))
    y = x / jnp.linalg.norm(x)
    for a, b, c in [(3.4445, -4.775, 2.0315)] * 8 + [(2., -1.5, .5)] * 2:
        gram = y @ y.T
        y = a * y + b * (gram @ y) + c * ((gram @ gram) @ y)
    np.testing.assert_allclose(hybrid_newton_schulz(x), y, atol=2e-5, rtol=2e-4)


def test_mhc_matches_released_split_and_stream_orientation():
    from nano_dsv41f.mhc import mhc_mixes, post_mix
    streams = jnp.array([[[1., 2.], [3., 5.]]])
    p = {'weight': jnp.arange(32, dtype=jnp.float32).reshape(4, 8) * .01,
         'base': jnp.arange(8, dtype=jnp.float32) * .2,
         'scale': jnp.array([.3, .7, 1.4])}
    pre, post, comb = mhc_mixes(streams, p, sinkhorn_iters=4)
    flat = np.asarray(streams).reshape(1, 4)
    raw = (flat / np.sqrt(np.mean(flat**2, axis=-1, keepdims=True) + 1e-20)) @ np.asarray(p['weight'])
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    np.testing.assert_allclose(pre, sigmoid(raw[:, :2] * .3 + np.asarray(p['base'][:2])) + 1e-6, atol=1e-6)
    np.testing.assert_allclose(post, 2 * sigmoid(raw[:, 2:4] * .7 + np.asarray(p['base'][2:4])), atol=1e-6)
    z = (raw[:, 4:] * 1.4 + np.asarray(p['base'][4:])).reshape(1, 2, 2)
    z = np.exp(z - z.max(-1, keepdims=True)); z = z / z.sum(-1, keepdims=True) + 1e-6
    z /= z.sum(-2, keepdims=True) + 1e-6
    for _ in range(3):
        z /= z.sum(-1, keepdims=True) + 1e-6
        z /= z.sum(-2, keepdims=True) + 1e-6
    np.testing.assert_allclose(comb, z, atol=1e-6)
    asymmetric = jnp.array([[[.1, .9], [.3, .7]]])
    got = post_mix(streams, jnp.zeros((1, 2)), asymmetric, jnp.zeros((1, 2)))
    np.testing.assert_allclose(got, np.swapaxes(asymmetric, -1, -2) @ np.asarray(streams), atol=1e-6)


def test_bias_controller_and_optimizer_no_decay_for_control_parameters():
    from nano_dsv41f.config import ModelConfig, TrainConfig
    from nano_dsv41f.training import update_router_biases
    from nano_dsv41f.optimizer import init_optimizer_state, optimizer_step
    p = {'blocks': ({'moe': {'router_bias': jnp.array([1., 2.])},
                       'mhc': {'base': jnp.ones(2), 'scale': jnp.ones(3)},
                       'attn_norm': {'weight': jnp.ones(2)}},)}
    cfg = ModelConfig()
    opt = init_optimizer_state(p, cfg)
    got, state, _ = optimizer_step(p, jax.tree.map(jnp.zeros_like, p), opt,
                                   step=jnp.array(0), config=cfg,
                                   train_config=TrainConfig(warmup_steps=1))
    np.testing.assert_array_equal(got['blocks'][0]['moe']['router_bias'], [1., 2.])
    np.testing.assert_array_equal(got['blocks'][0]['mhc']['base'], [1., 1.])
    assert np.all(np.asarray(got['blocks'][0]['attn_norm']['weight']) < 1.)
    balanced = update_router_biases(got, jnp.array([[10, 0]]), speed=.001)
    np.testing.assert_allclose(balanced['blocks'][0]['moe']['router_bias'], [.999, 2.001])


@pytest.mark.parametrize('ratio', [1, 2])
def test_splash_combined_packed_forward_backward_interpreter(ratio):
    from nano_dsv41f.csa2 import SharedCSA2State
    from nano_dsv41f.tpu_native import _combined_splash_attention, combined_csa2_mask
    mesh = Mesh(np.asarray(jax.devices()[:1]), ('x',), axis_types=(AxisType.Auto,))
    state = TPUNativeState(mesh, TPUNativeConfig(splash_interpret=True))
    q = jax.random.normal(jax.random.key(31), (1, 128, 1, 128)) * .1
    kv = jax.random.normal(jax.random.key(32), (1, 128, 128)) * .1
    gkv = jax.random.normal(jax.random.key(33), (1, 128 // ratio, 128)) * .1
    seg = jnp.array([[0] * 64 + [1] * 64], jnp.int32)
    sink = jnp.array([.2])
    def run(q, kv, gkv, sink):
        shared = SharedCSA2State(kv=gkv, latent=gkv, index_k=None,
            segment_ids=seg[:, ::ratio], group_start_positions=(jnp.arange(128)[None] % 64)[:, ::ratio],
            source_layer=jnp.array(1), compress_ratio=jnp.array(ratio),
            latest_topk_indices=None, latest_topk_values=None,
            index_source_layer=jnp.array(-1), candidate_mask=None)
        return _combined_splash_attention(q, kv, seg, shared, compression_ratio=ratio,
                                          local_window=8, sink=sink, state=state)[0]
    mask = jnp.asarray(combined_csa2_mask(128, local_window=8, global_kv_len=128 // ratio,
                                        compression_ratio=ratio))[None]
    mask &= seg[:, :, None] == jnp.concatenate((seg, seg[:, ::ratio]), -1)[:, None, :]
    def reference(q, kv, gkv, sink):
        keys = jnp.concatenate((kv, gkv), 1)
        logits = jnp.einsum('bthd,bkd->bthk', q * jnp.float32(128 ** -.5), keys)
        logits = jnp.where(mask[:, :, None, :], logits, -jnp.inf)
        logits = jnp.concatenate((logits, jnp.broadcast_to(sink[None, None, :, None], (1, 128, 1, 1))), -1)
        prob = jax.nn.softmax(logits, -1)[..., :-1]
        return jnp.einsum('bthk,bkd->bthd', prob, keys)
    np.testing.assert_allclose(jax.jit(run)(q, kv, gkv, sink), reference(q, kv, gkv, sink), atol=3e-6, rtol=5e-5)
    def grads(fn):
        return jax.jit(jax.grad(lambda *a: jnp.square(fn(*a)).sum(), argnums=(0, 1, 2, 3)))(q, kv, gkv, sink)
    for actual, expected in zip(grads(run), grads(reference)):
        np.testing.assert_allclose(actual, expected, atol=5e-6, rtol=3e-4)


def test_generated_notebooks_share_exact_bootstrap():
    import ast
    from pathlib import Path
    import nbformat
    root = Path(__file__).resolve().parents[1]
    source = (root / 'scripts/kaggle_bootstrap.py').read_text()
    ast.parse(source)
    for name in ('nano_dsv41f_kaggle.ipynb', 'nano_dsv41f_combined_smoke.ipynb'):
        nb = nbformat.read(root / 'notebooks' / name, as_version=4)
        nbformat.validate(nb)
        code = [cell.source for cell in nb.cells if cell.cell_type == 'code']
        assert code[0] == source
        for cell in code:
            ast.parse(cell)
    # A warmed notebook must fail before installing libraries or touching the checkout.
    with pytest.raises(RuntimeError, match='Restart the Kaggle session'):
        exec(compile(source, 'kaggle_bootstrap.py', 'exec'), {})


def test_native_late_loss_and_gradients_match_reference_backbone():
    from test_reference_model import tiny_config
    from nano_dsv41f.config import RematConfig
    from nano_dsv41f.model import init_model
    from nano_dsv41f.training import pretrain_loss
    from nano_dsv41f.tpu_native import install_model_dispatch, tpu_native_context
    cfg = replace(tiny_config(dspark=False), remat=RematConfig(policy='none'))
    params = init_model(jax.random.key(42), cfg)
    ids = (jnp.arange(128)[None] % cfg.vocab_size).astype(jnp.int32)
    seg = jnp.array([[0] * 64 + [1] * 64], jnp.int32)
    mask = jnp.ones_like(ids, dtype=bool).at[:, 63].set(False)
    def objective(p):
        return pretrain_loss(p, cfg, ids, segment_ids=seg, token_mask=mask,
                             include_indexer=True, n_segments=2)
    expected, expected_grad = jax.jit(jax.value_and_grad(objective, has_aux=True))(params)
    install_model_dispatch()
    mesh = Mesh(np.asarray(jax.devices()[:1]), ('x',), axis_types=(AxisType.Auto,))
    with tpu_native_context(mesh, TPUNativeConfig(splash_interpret=True, need_teacher_lse=True,
                                                moe_capacity_factor=1., moe_capacity_multiple=8)):
        actual, actual_grad = jax.jit(jax.value_and_grad(objective, has_aux=True))(params)
    np.testing.assert_allclose(actual[0], expected[0], atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(actual[1]['indexer_loss'], expected[1]['indexer_loss'], atol=2e-6, rtol=2e-5)
    assert int(actual[1]['lm_tokens']) == 125
    assert np.asarray(actual[1]['router_loads']).sum() == 127 * cfg.experts_per_token * cfg.n_layers
    for a, b in zip(jax.tree.leaves(actual_grad), jax.tree.leaves(expected_grad)):
        np.testing.assert_allclose(a, b, atol=2e-5, rtol=3e-3)
