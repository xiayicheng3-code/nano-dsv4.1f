"""Physical TPU output/all-gradient gates for smaller MoE buffers and overflow."""
import argparse
from dataclasses import replace
from pathlib import Path
import sys

from run_pretrain_stress import save_report


def check(divisor, *, validate=True, implementation='mosaic'):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from nano_dsv41f import make_v5e_mesh, validate_v5e_runtime
    from nano_dsv41f.moe import init_moe
    from nano_dsv41f.tpu_native import TPUNativeConfig, TPUNativeState
    from nano_dsv41f.tpu_moe import apply_moe_v5e_multi
    from attention_replay_utils import error_metrics
    if validate:
        validate_v5e_runtime()
    mesh = make_v5e_mesh()
    options = TPUNativeConfig(moe_ragged_implementation=implementation)
    # Production expert dimensions, reduced token count; one chip owns six experts.
    x = (jax.random.normal(jax.random.key(63), (4, 256, 512)) * .2).astype(jnp.bfloat16)
    params = init_moe(jax.random.key(64), 512, 128, 48)
    params = {**params, 'experts': jax.tree.map(lambda a: a.astype(jnp.bfloat16), params['experts']),
              'shared': jax.tree.map(lambda a: a.astype(jnp.bfloat16), params['shared'])}
    results = []
    for skewed in (False, True):
        p = {**params, 'router_bias': jnp.arange(48, dtype=jnp.float32) * 100} if skewed else params
        def evaluate(d):
            state = TPUNativeState(mesh, replace(options, moe_buffer_divisor=d))
            def objective(x, p):
                y, aux = apply_moe_v5e_multi(x, p, state=state, top_k=4,
                    route_scale=1.5, swiglu_limit=10., eps=1e-20)
                return jnp.square(y.astype(jnp.float32)).mean(), (y, aux['expert_loads'])
            return jax.device_get(jax.jit(jax.value_and_grad(objective, argnums=(0, 1), has_aux=True))(x, p))
        reference, candidate = evaluate(1), evaluate(divisor)
        def leaves(r):
            return jax.tree.leaves((r[0][0], r[0][1][0], r[1]))
        errors = [error_metrics(a, b) for a, b in zip(leaves(candidate), leaves(reference))]
        loads = np.asarray(candidate[0][1][1]).reshape(8, 6).sum(axis=-1)
        capacity = 4 * 256 * 4 // divisor
        fallback = int((loads > capacity).sum())
        passed = (all(e['passed'] for e in errors) and
                  np.array_equal(candidate[0][1][1], reference[0][1][1]) and
                  int(loads.sum()) == 4 * 256 * 4 and
                  (not skewed or divisor == 1 or fallback > 0))
        results.append(dict(skewed=skewed, passed=passed, errors=errors,
                            capacity=capacity, chip_loads=loads.tolist(), fallback_chips=fallback))
        print('MoE numerical gate:', divisor, 'skewed:', skewed, 'passed:', passed, flush=True)
    return dict(status='passed' if all(r['passed'] for r in results) else 'failed',
                divisor=divisor, cases=results)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--divisor', type=int, choices=(1, 2, 4), required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    try:
        result = check(a.divisor)
    except Exception as e:
        save_report(a.output, dict(status='failed', exception=repr(e)))
        raise
    save_report(a.output, result)
    if result['status'] != 'passed':
        sys.exit(1)


if __name__ == '__main__':
    main()
