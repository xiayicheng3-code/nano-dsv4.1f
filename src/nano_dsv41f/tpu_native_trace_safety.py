"""Trace-safety shim for TPU Splash runner construction.

Splash builds a preprocessed kernel pytree while JAX is tracing the model. Caching the
resulting callable across traces can retain DynamicJaxprTracer leaves in global Python
state, which then triggers ``UnexpectedTracerError`` on the next retrace. The native
backend's static mask construction is cheap relative to compilation, so the safe default
is to rebuild the runner per trace and never retain it globally.
"""

from __future__ import annotations

from typing import Any

from . import tpu_native as _tpu_native


_ORIGINAL_SPLASH_RUNNER = _tpu_native._splash_runner


def _trace_safe_splash_runner(*args: Any, **kwargs: Any):
    """Build a Splash runner for the current trace without cross-trace global state."""
    _tpu_native._SPLASH_RUNNER_CACHE.clear()
    try:
        return _ORIGINAL_SPLASH_RUNNER(*args, **kwargs)
    finally:
        # The returned closure stays alive in the current Python stack long enough to be
        # called immediately, but no traced kernel/closure remains reachable globally.
        _tpu_native._SPLASH_RUNNER_CACHE.clear()


def install_trace_safe_splash_runner() -> None:
    """Install the no-cross-trace-cache wrapper exactly once."""
    if _tpu_native._splash_runner is not _trace_safe_splash_runner:
        _tpu_native._splash_runner = _trace_safe_splash_runner


install_trace_safe_splash_runner()
