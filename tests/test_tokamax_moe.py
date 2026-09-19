import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh

from nano_dsv41f.moe import _expert_forward, apply_moe, init_moe
from nano_dsv41f.tpu_moe import apply_moe_v5e_multi, ragged_expert_forward
from nano_dsv41f.tpu_native import TPUNativeConfig, TPUNativeState


@pytest.fixture
def mosaic_interpreter(monkeypatch):
    """Run 0.0.12's actual forward/VJP kernels in the Pallas CPU interpreter.

    This release's device guard does not recognize interpret=True. Supply v5e
    geometry and allow interpreter instances on CPU only within these tests.
    Production still uses Tokamax's unmodified device guard and real Mosaic.
    """
    from jax._src.pallas.mosaic import tpu_info
    from tokamax._src.ops.ragged_dot import api
    from tokamax._src.ops.ragged_dot.pallas_mosaic_tpu import PallasMosaicTpuRaggedDot
    info = tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V5E, 1)
    monkeypatch.setattr(tpu_info, "get_tpu_info", lambda: info)
    original = PallasMosaicTpuRaggedDot.supported_on
    monkeypatch.setattr(PallasMosaicTpuRaggedDot, "supported_on",
                        lambda self, device: self.interpret or original(self, device))
    monkeypatch.setattr(api, "IMPLEMENTATIONS", {
        **api.IMPLEMENTATIONS, "mosaic_tpu": PallasMosaicTpuRaggedDot(interpret=True),
    })


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("sizes", [(7, 0, 11), (0, 0, 0)])
def test_mosaic_ragged_payload_and_gradients_with_padding(mosaic_interpreter, dtype, sizes):
    # Exercise both channel padding (<128) and an unused token suffix.
    x = jax.random.normal(jax.random.key(60), (31, 16)) * 1.3
    experts = init_moe(jax.random.key(61), 16, 19, 3)["experts"]
    experts = jax.tree.map(lambda a: a.astype(dtype), experts)
    x = x.astype(dtype)
    groups = jnp.asarray(sizes, jnp.int32)
    ids = np.pad(np.repeat(np.arange(3), sizes), (0, 31 - sum(sizes)))
    valid = jnp.arange(31) < sum(sizes)
    def reference(x, p):
        y = _expert_forward(x, p['w1'][ids], p['w2'][ids], p['w3'][ids], .7)
        return jnp.where(valid[:, None], y, 0)
    def native(x, p):
        return ragged_expert_forward(x, p, groups, swiglu_limit=.7, implementation='mosaic')
    def evaluate(fn):
        def loss(x, p):
            y = fn(x, p)
            return jnp.sin(y.astype(jnp.float32)).sum(), y
        return jax.jit(jax.value_and_grad(loss, argnums=(0, 1), has_aux=True))(x, experts)
    actual, expected = evaluate(native), evaluate(reference)
    atol, rtol = (3e-2, 6e-2) if dtype == jnp.bfloat16 else (3e-5, 3e-4)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
        assert np.isfinite(a).all()
        np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)
    assert np.all(np.asarray(actual[1][0][sum(sizes):]) == 0)
    assert all(np.all(np.asarray(v[1]) == 0) for v in actual[1][1].values())


def test_mosaic_runtime_preflight(mosaic_interpreter):
    from nano_dsv41f.runtime import ragged_dot_preflight
    assert ragged_dot_preflight()["parity"] == "PASS"


@pytest.mark.parametrize('dtype', [jnp.float32, jnp.bfloat16])
def test_mosaic_precision_reaches_forward_and_both_vjps(mosaic_interpreter, monkeypatch, dtype):
    # CPU dot numerics do not emulate TPU DEFAULT's BF16 truncation. Inspect the
    # actual Mosaic forward/dlhs/drhs calls so CPU parity cannot hide this again.
    from functools import wraps
    from tokamax._src.ops.ragged_dot.pallas_mosaic_tpu import PallasMosaicTpuRaggedDot
    from nano_dsv41f.tpu_moe import _ragged_dot
    original = PallasMosaicTpuRaggedDot._fwd
    seen = []
    @wraps(original)
    def record(self, *args, **kwargs):
        seen.append((kwargs['precision'], str(kwargs['ragged_dot_dimension_numbers'])))
        return original(self, *args, **kwargs)
    monkeypatch.setattr(PallasMosaicTpuRaggedDot, '_fwd', record)
    x = jnp.ones((31, 16), dtype)
    w = jnp.ones((3, 16, 24), dtype)
    sizes = jnp.array([7, 0, 11], jnp.int32)
    with jax.default_matmul_precision('bfloat16'):
        result = jax.jit(jax.value_and_grad(
            lambda x, w: _ragged_dot(x, w, sizes, implementation='mosaic').astype(jnp.float32).sum(),
            argnums=(0, 1),
        ))(x, w)
    assert all(np.isfinite(a).all() for a in jax.tree.leaves(result))
    expected = jax.lax.Precision.HIGHEST if dtype == jnp.float32 else jax.lax.Precision.DEFAULT
    assert len({dimensions for _, dimensions in seen}) == 3
    assert all(precision == (expected, expected) for precision, _ in seen)


@pytest.mark.parametrize("n_experts", [8, 16])
@pytest.mark.parametrize("skewed", [False, True])
def test_ragged_ep_bf16_input_and_all_parameter_gradients(n_experts, skewed):
    if len(jax.devices()) < 8:
        pytest.skip('Run with XLA_FLAGS=--xla_force_host_platform_device_count=8')
    mesh = Mesh(np.asarray(jax.devices()[:8]).reshape(2, 4), ('x', 'y'),
                axis_types=(AxisType.Auto, AxisType.Auto))
    state = TPUNativeState(mesh, TPUNativeConfig())
    x = (jax.random.normal(jax.random.key(63), (2, 16, 8)) * .2).astype(jnp.bfloat16)
    p = init_moe(jax.random.key(64), 8, 16, n_experts)
    p = {**p, 'experts': jax.tree.map(lambda a: a.astype(jnp.bfloat16), p['experts']),
         'shared': jax.tree.map(lambda a: a.astype(jnp.bfloat16), p['shared'])}
    if skewed:
        p['router_bias'] = jnp.arange(n_experts, dtype=jnp.float32) * 100
    mask = jnp.ones((2, 16), bool).at[:, -1].set(False)
    kw = dict(top_k=2, route_scale=1.5, swiglu_limit=.2, eps=1e-20, token_mask=mask)
    def ref(x, p): return apply_moe(x, p, **kw)
    def native(x, p): return apply_moe_v5e_multi(x, p, state=state, **kw)
    def evaluate(fn):
        def objective(x, p):
            y, aux = fn(x, p)
            return jnp.square(y.astype(jnp.float32)).sum(), (y, aux)
        return jax.jit(jax.value_and_grad(objective, argnums=(0, 1), has_aux=True))(x, p)
    actual, expected = evaluate(native), evaluate(ref)
    for a, b in zip(jax.tree.leaves((actual[0][0], actual[0][1][0], actual[1])),
                    jax.tree.leaves((expected[0][0], expected[0][1][0], expected[1]))):
        a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
        assert np.isfinite(a).all()
        np.testing.assert_allclose(a, b, atol=2e-3, rtol=.06)
    aux = actual[0][1][1]
    np.testing.assert_array_equal(aux['router_loads'], expected[0][1][1]['router_loads'])
    assert int(aux['expert_dropped'].sum()) == 0
    assert int(aux['expert_loads'].sum()) == x.shape[0] * x.shape[1] * 2
    assert int(aux['expert_packed_rows']) == 32 * min(2, n_experts // 8)
    assert int(aux['router_loads'].sum()) == 60
    if skewed:
        assert (np.asarray(aux['expert_loads']) == 0).sum() == n_experts - 2


def test_invalid_expert_mesh_does_not_silently_fall_back():
    if len(jax.devices()) < 8:
        pytest.skip('Run with XLA_FLAGS=--xla_force_host_platform_device_count=8')
    mesh = Mesh(np.asarray(jax.devices()[:8]), ('x',), axis_types=(AxisType.Auto,))
    with pytest.raises(ValueError, match='divisible'):
        apply_moe_v5e_multi(jnp.ones((1, 8, 4)), init_moe(jax.random.key(65), 4, 8, 3),
                           top_k=2, route_scale=1., swiglu_limit=7., eps=1e-20,
                           state=TPUNativeState(mesh, TPUNativeConfig()))


def test_ragged_ep_mosaic_interpreter_with_empty_chips(mosaic_interpreter):
    if len(jax.devices()) < 8:
        pytest.skip('Run with XLA_FLAGS=--xla_force_host_platform_device_count=8')
    mesh = Mesh(np.asarray(jax.devices()[:8]), ('x',), axis_types=(AxisType.Auto,))
    state = TPUNativeState(mesh, TPUNativeConfig(moe_ragged_implementation='mosaic'))
    x = jax.random.normal(jax.random.key(66), (1, 8, 4)) * .2
    p = init_moe(jax.random.key(67), 4, 8, 16)
    p['router_bias'] = jnp.arange(16, dtype=jnp.float32) * 100
    kw = dict(top_k=2, route_scale=1.5, swiglu_limit=.7, eps=1e-20)
    def evaluate(fn):
        def objective(x, p):
            y, _ = fn(x, p)
            return jnp.sin(y).sum(), y
        return jax.jit(jax.value_and_grad(objective, argnums=(0, 1), has_aux=True))(x, p)
    actual = evaluate(lambda x, p: apply_moe_v5e_multi(x, p, state=state, **kw))
    expected = evaluate(lambda x, p: apply_moe(x, p, **kw))
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        assert np.isfinite(a).all()
        np.testing.assert_allclose(a, b, atol=2e-5, rtol=3e-4)
