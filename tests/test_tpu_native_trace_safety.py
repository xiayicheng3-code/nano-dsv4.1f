from __future__ import annotations

import jax
import jax.numpy as jnp

from nano_dsv41f import tpu_native
from nano_dsv41f import tpu_native_trace_safety as trace_safety


def test_splash_runner_cache_does_not_leak_tracers_across_retraces(monkeypatch):
    """Regression for the Kaggle UnexpectedTracerError after compile diagnostics.

    This fake runner deliberately reproduces the dangerous pattern: construction saves a
    closure that captures a traced intermediate in the same global cache used by the native
    Splash path. A second JIT trace with a different input shape would fail if that closure
    survived globally. The trace-safety shim must clear it before and after each build.
    """

    def fake_runner(x):
        cached = tpu_native._SPLASH_RUNNER_CACHE.get("runner")
        if cached is not None:
            return cached
        captured = x + 1.0

        def runner(z):
            return z + captured

        tpu_native._SPLASH_RUNNER_CACHE["runner"] = runner
        return runner

    monkeypatch.setattr(trace_safety, "_ORIGINAL_SPLASH_RUNNER", fake_runner)
    tpu_native._SPLASH_RUNNER_CACHE.clear()
    trace_safety.install_trace_safe_splash_runner()

    def transformed(x):
        runner = tpu_native._splash_runner(x)
        return runner(x)

    compiled = jax.jit(transformed)
    first = compiled(jnp.asarray(1.0, dtype=jnp.float32))
    second = compiled(jnp.asarray([2.0, 3.0], dtype=jnp.float32))

    assert float(first) == 3.0
    assert jnp.allclose(second, jnp.asarray([5.0, 7.0], dtype=jnp.float32))
    assert not tpu_native._SPLASH_RUNNER_CACHE
