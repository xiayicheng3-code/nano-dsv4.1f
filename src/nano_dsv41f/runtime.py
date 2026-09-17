"""Fail before model initialization if the loaded TPU client cannot run Pallas."""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import os


def package_versions():
    def get(name):
        try:
            return version(name)
        except PackageNotFoundError:
            return None
    return {name: get(name) for name in ('jax', 'jaxlib', 'libtpu')}


def pallas_preflight(*, require_eight_chips=True):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.extend import backend
    from .tpu import runtime_report, validate_v5e_runtime
    from .splash import _splash_modules

    validate_v5e_runtime(require_eight_chips=require_eight_chips)
    report = {**runtime_report(), 'packages': package_versions(),
              'platform_version': backend.get_backend().platform_version,
              'TPU_LIBRARY_PATH': os.environ.get('TPU_LIBRARY_PATH')}
    print('TPU/Pallas preflight:', report, flush=True)
    splash, masks = _splash_modules()
    kernel = splash.make_splash_mqa(
        masks.MultiHeadMask((masks.CausalMask((128, 128)),)),
        head_shards=1, q_seq_shards=1,
    )
    q = jnp.ones((1, 128, 128), dtype=jnp.bfloat16) * 0.01
    kv = jnp.ones((128, 128), dtype=jnp.bfloat16) * 0.01
    sinks = jnp.zeros((1,), dtype=jnp.float32)
    segments = splash.SegmentIds(q=jnp.zeros(128, jnp.int32), kv=jnp.zeros(128, jnp.int32))
    def loss(q, kv, sinks):
        return jnp.mean(kernel(q, kv, kv, segment_ids=segments, sinks=sinks).astype(jnp.float32))
    try:
        result = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2)))(q, kv, sinks)
        jax.block_until_ready(result)
        if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(result)):
            raise RuntimeError('Splash forward/backward returned non-finite values')
    except Exception as exc:
        raise RuntimeError(
            'Splash forward/backward preflight failed before model initialization. '
            'Restart the Kaggle session and run the bootstrap cell first; it installs '
            'requirements-tpu.txt. Check the loaded platform_version above, not only pip '
            'metadata, and any custom TPU_LIBRARY_PATH override. Original error: ' + str(exc)
        ) from exc
    print('Splash forward + backward + packed segments + sink: PASS', flush=True)
    return report
