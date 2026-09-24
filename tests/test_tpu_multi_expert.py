import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh

from nano_dsv41f.moe import apply_moe, init_moe
from nano_dsv41f.tpu_moe import apply_moe_v5e_multi
from nano_dsv41f.tpu_native import TPUNativeConfig, TPUNativeState


def test_multi_expert_local_group_matches_reference_on_single_device():
    devices = np.asarray(jax.devices()[:1], dtype=object)
    mesh = Mesh(devices, ("x",), axis_types=(AxisType.Auto,))
    state = TPUNativeState(
        mesh,
        TPUNativeConfig(
            moe_capacity_factor=10.0,
            moe_capacity_multiple=1,
            manual_axis_name="tp",
        ),
    )

    key_x, key_moe = jax.random.split(jax.random.PRNGKey(7))
    x = jax.random.normal(key_x, (1, 4, 4), dtype=jnp.float32)
    params = init_moe(key_moe, dim=4, hidden=8, n_experts=2)

    expected, _ = apply_moe(
        x,
        params,
        top_k=2,
        route_scale=1.0,
        swiglu_limit=10.0,
        eps=1e-20,
    )
    got, aux = apply_moe_v5e_multi(
        x,
        params,
        top_k=2,
        route_scale=1.0,
        swiglu_limit=10.0,
        eps=1e-20,
        state=state,
    )

    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol=1e-5, atol=1e-5)
    assert np.asarray(aux["expert_loads"]).shape == (2,)
    assert int(np.asarray(aux["expert_loads"]).sum()) == 8
    assert int(np.asarray(aux["expert_overflow"]).sum()) == 0
    assert int(np.asarray(aux["experts_per_chip"])) == 2


@pytest.mark.parametrize("skewed", [False, True])
def test_48_expert_top4_forward_backward_on_eight_devices(skewed):
    if len(jax.devices()) < 8:
        pytest.skip("requires eight CPU devices")
    mesh = Mesh(np.asarray(jax.devices()[:8]), ("x",), axis_types=(AxisType.Auto,))
    state = TPUNativeState(mesh, TPUNativeConfig())
    x = jax.random.normal(jax.random.key(121), (2, 16, 4)) * 0.2
    params = init_moe(jax.random.key(122), dim=4, hidden=8, n_experts=48)
    if skewed:
        params["router_bias"] = jnp.arange(48, dtype=jnp.float32) * 1000
    kw = dict(top_k=4, route_scale=1.5, swiglu_limit=10., eps=1e-20)
    def reference(x, p):
        return apply_moe(x, p, **kw)[0]
    def native(x, p):
        return apply_moe_v5e_multi(x, p, state=state, **kw)[0]
    actual, aux = jax.jit(lambda x, p: apply_moe_v5e_multi(x, p, state=state, **kw))(x, params)
    np.testing.assert_allclose(actual, jax.jit(reference)(x, params), atol=3e-6, rtol=2e-5)
    assert int(aux["expert_loads"].sum()) == 128
    assert int(aux["expert_packed_rows"]) == 128
    assert int(aux["expert_dropped"].sum()) == 0
    if skewed:
        np.testing.assert_array_equal(aux["expert_loads"][-4:], np.full(4, 32))
    def gradient(fn):
        return jax.jit(jax.grad(lambda x, p: jnp.sin(fn(x, p)).sum(), argnums=(0, 1)))(x, params)
    for a, b in zip(jax.tree.leaves(gradient(native)), jax.tree.leaves(gradient(reference))):
        np.testing.assert_allclose(a, b, atol=3e-6, rtol=3e-4)
