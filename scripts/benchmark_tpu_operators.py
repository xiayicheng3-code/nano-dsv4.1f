"""Small operator parity + synchronized forward/backward timings on the target TPU."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from nano_dsv41f.moe import apply_moe, init_moe
from nano_dsv41f.runtime import pallas_preflight
from nano_dsv41f.tpu import make_v5e_mesh
from nano_dsv41f.tpu_moe import apply_moe_v5e_multi
from nano_dsv41f.tpu_native import TPUNativeConfig, TPUNativeState
from nano_dsv41f.tpu_native import _moe_param_specs, manual_v5e_mesh
from jax.sharding import NamedSharding, PartitionSpec as P


def parity_diagnostics(actual, expected, *, dtype):
    """Name every output/gradient leaf and retain evidence before failing a gate."""
    atol, rtol = (2e-3, .06) if dtype == jnp.bfloat16 else (2e-6, 5e-4)
    relative_limit = .06 if dtype == jnp.bfloat16 else .001
    leaves, structure = jax.tree_util.tree_flatten_with_path(actual)
    ref_leaves, ref_structure = jax.tree_util.tree_flatten_with_path(expected)
    if structure != ref_structure:
        raise ValueError('Native/reference result trees differ')
    diagnostics = []
    for (path, a), (_, b) in zip(leaves, ref_leaves):
        af, bf = np.asarray(a, np.float32), np.asarray(b, np.float32)
        finite = bool(np.isfinite(af).all() and np.isfinite(bf).all())
        difference = float(np.max(np.abs(af - bf))) if finite else None
        relative = float(np.linalg.norm(af - bf) / max(np.linalg.norm(bf), 1e-12)) if finite else None
        mismatches = int(np.count_nonzero(~np.isclose(af, bf, atol=atol, rtol=rtol)))
        diagnostics.append({'path': jax.tree_util.keystr(path), 'shape': list(af.shape),
                            'finite': finite, 'max_abs_error': difference,
                            'relative_l2_error': relative, 'mismatched_elements': mismatches,
                            'passed': finite and mismatches == 0 and relative < relative_limit})
    return {'parity': 'PASS' if all(d['passed'] for d in diagnostics) else 'FAIL',
            'atol': atol, 'rtol': rtol, 'relative_l2_limit': relative_limit,
            'leaves': diagnostics}


def measure(fn, args, *, steps):
    start = time.perf_counter()
    compiled = jax.jit(fn).lower(*args).compile()
    compile_seconds = time.perf_counter() - start
    result = compiled(*args)
    jax.block_until_ready(result)
    for _ in range(2):
        jax.block_until_ready(compiled(*args))
    times = []
    for _ in range(steps):
        start = time.perf_counter()
        jax.block_until_ready(compiled(*args))
        times.append(time.perf_counter() - start)
    return result, {'compile_seconds': compile_seconds, 'seconds': times,
                    'median_seconds': float(np.median(times))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('operator-benchmark.json'))
    parser.add_argument('--steps', type=int, default=10)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    runtime = pallas_preflight()
    mesh = make_v5e_mesh()
    state = TPUNativeState(mesh, TPUNativeConfig(moe_ragged_implementation='mosaic'))
    # Small enough for the deliberately inefficient reference to be a safe oracle.
    x = jax.random.normal(jax.random.key(17), (1, 256, 128), dtype=jnp.float32) * .1
    params = init_moe(jax.random.key(18), 128, 128, 16)
    kwargs = dict(top_k=2, route_scale=1.5, swiglu_limit=10., eps=1e-20)
    report = {'runtime': runtime,
              'native_backend': 'tokamax-0.0.12/mosaic/ragged_dot',
              'precision_policy': {'float32_payload': 'HIGHEST', 'bfloat16_payload': 'DEFAULT',
                                   'router': 'HIGHEST'},
              'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'shape': {'batch': 1, 'tokens': 256, 'dim': 128, 'expert_dim': 128, 'experts': 16, 'top_k': 2},
              'cases': {}}
    for skew in (False, True):
        p = {**params, 'router_bias': (jnp.arange(16) * 100.).astype(jnp.float32) if skew else params['router_bias']}
        for dtype in (jnp.float32, jnp.bfloat16):
            xp = x.astype(dtype)
            pp = {**p, 'experts': jax.tree.map(lambda z: z.astype(dtype), p['experts']),
                  'shared': jax.tree.map(lambda z: z.astype(dtype), p['shared'])}
            manual = manual_v5e_mesh(mesh)
            xp = jax.device_put(xp, NamedSharding(manual, P(None, 'tp', None)))
            pp = jax.device_put(pp, jax.tree.map(lambda spec: NamedSharding(manual, spec),
                                                _moe_param_specs('tp')))
            def ref(x, p): return apply_moe(x, p, **kwargs)[0]
            def native(x, p): return apply_moe_v5e_multi(x, p, state=state, **kwargs)[0]
            def with_grad(fn):
                def loss(x, p):
                    y = fn(x, p)
                    return jnp.mean(jnp.square(y.astype(jnp.float32))), y
                return jax.value_and_grad(loss, argnums=(0, 1), has_aux=True)
            expected, rt = measure(with_grad(ref), (xp, pp), steps=args.steps)
            actual, nt = measure(with_grad(native), (xp, pp), steps=args.steps)
            name = f'{jnp.dtype(dtype)}_skew={skew}'
            parity = parity_diagnostics(actual, expected, dtype=dtype)
            report['cases'][name] = {**parity, 'reference': rt, 'native': nt,
                                    'reference_over_native': rt['median_seconds'] / nt['median_seconds']}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
            if parity['parity'] != 'PASS':
                failed = [leaf['path'] for leaf in parity['leaves'] if not leaf['passed']]
                raise AssertionError(f'{name}: parity failed at {failed}; diagnostics saved to {args.output}')
            print(name, report['cases'][name], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print('saved report:', args.output)


if __name__ == '__main__':
    main()
