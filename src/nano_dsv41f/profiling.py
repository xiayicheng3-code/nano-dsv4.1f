from __future__ import annotations

from collections import Counter
from typing import Any


_COLLECTIVE_TOKENS = {
    "all_gather": ("all_gather", "all-gather"),
    "all_reduce": ("all_reduce", "all-reduce"),
    "all_to_all": ("all_to_all", "all-to-all"),
    "collective_permute": ("collective_permute", "collective-permute"),
    "reduce_scatter": ("reduce_scatter", "reduce-scatter"),
}


class DiagnosticExecutable:
    """Expose AOT diagnostics while preserving the caller's execution wrapper.

    A native TPU step is not just a raw ``jax.stages.Compiled`` object: its Python wrapper
    activates context-sensitive Splash/MoE dispatch before entering the jitted function.
    Calling the raw AOT object returned by ``lower().compile()`` can therefore bypass that
    wrapper and, with JAX captured constants, produce a flattened-input signature mismatch.

    Attribute access is delegated to the AOT executable so callers can still inspect memory,
    cost analysis and other compiled metadata. Calling the object delegates to the original
    jitted/wrapped function, which preserves its execution semantics.
    """

    def __init__(self, compiled, runner):
        self._compiled = compiled
        self._runner = runner

    def __call__(self, *args, **kwargs):
        return self._runner(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._compiled, name)



def collective_counts(lowered_or_text: Any) -> dict[str, int]:
    """Best-effort StableHLO collective count for notebook comparisons.

    JAX explicitly treats lowered text as a debugging aid rather than a stable machine API,
    so this helper is intentionally diagnostic. It is useful for before/after comparisons
    on one pinned Kaggle runtime, not for asserting exact compiler behavior in unit tests.
    """
    text = (
        lowered_or_text
        if isinstance(lowered_or_text, str)
        else lowered_or_text.as_text()
    )
    lowered = text.lower()
    counts: dict[str, int] = {}
    for name, tokens in _COLLECTIVE_TOKENS.items():
        counts[name] = max(lowered.count(token) for token in tokens)
    return counts



def compiled_memory_report(compiled) -> dict[str, float] | None:
    """Return compiler-estimated memory in GiB, including donation alias savings."""
    stats = compiled.memory_analysis()
    if stats is None:
        return None
    argument = int(stats.argument_size_in_bytes)
    output = int(stats.output_size_in_bytes)
    temp = int(stats.temp_size_in_bytes)
    alias = int(stats.alias_size_in_bytes)
    total = argument + output + temp - alias
    gib = float(1024**3)
    return {
        "argument_gib": argument / gib,
        "output_gib": output / gib,
        "temporary_gib": temp / gib,
        "alias_gib": alias / gib,
        "estimated_total_gib": total / gib,
    }



def compiled_cost_report(compiled) -> dict[str, float]:
    """Normalize the backend's optional cost-analysis dictionaries into one flat summary."""
    raw = compiled.cost_analysis()
    if not raw:
        return {}
    if isinstance(raw, list):
        rows = raw
    else:
        rows = [raw]
    out: Counter[str] = Counter()
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key, value in row.items():
            try:
                out[str(key)] += float(value)
            except (TypeError, ValueError):
                pass
    return dict(out)



def compile_diagnostics(jitted_fn, *args):
    """Lower/compile without executing and return a safe callable plus diagnostics.

    The returned object delegates diagnostics to the AOT executable, but delegates execution
    to ``jitted_fn`` itself. This matters for context-sensitive native TPU wrappers and is
    harmless for ordinary ``jax.jit`` callables.
    """
    lowered = jitted_fn.lower(*args)
    compiled = lowered.compile()
    executable = DiagnosticExecutable(compiled, jitted_fn)
    return executable, {
        "collectives": collective_counts(lowered),
        "memory": compiled_memory_report(compiled),
        "cost": compiled_cost_report(compiled),
    }
