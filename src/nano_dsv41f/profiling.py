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
    """Lower/compile without executing and return the object plus lightweight diagnostics."""
    lowered = jitted_fn.lower(*args)
    compiled = lowered.compile()
    return compiled, {
        "collectives": collective_counts(lowered),
        "memory": compiled_memory_report(compiled),
        "cost": compiled_cost_report(compiled),
    }
