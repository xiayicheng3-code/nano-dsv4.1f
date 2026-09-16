import jax
import jax.numpy as jnp
import numpy as np
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
