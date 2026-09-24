"""CPU-only analysis of the preregistered attention experiment."""
from __future__ import annotations
import numpy as np


def error_metrics(actual, expected):
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    if not finite:
        return {"finite": False, "normalized_rms": None, "max_absolute": None, "passed": False}
    error = float(np.linalg.norm((a - b).ravel()) / max(np.linalg.norm(b.ravel()), 1e-12))
    return {"finite": True, "normalized_rms": error,
            "max_absolute": float(np.max(np.abs(a - b))), "passed": error <= .01}


def select_rows(array, rows, composition, *, shared=False):
    if shared:
        return array
    if composition == "repeated":
        return np.repeat(array[:1], rows, axis=0)
    if composition != "distinct" or len(array) < rows:
        raise ValueError("distinct replay requires enough captured rows")
    return array[:rows]


def decide_h1(pairs):
    """Each item holds times for one fresh-process replicate; never pool samples."""
    if len(pairs) < 3 or any(not p.get("passed", False) for p in pairs):
        return {"decision": "inconclusive", "reason": "need >=3 numerically valid complete repeats"}
    ratios = [{"baseline_s2": p["vmap8"] / (2 * p["vmap4"]),
               "sequential_s2": p["sequential8"] / (2 * p["sequential4"]),
               "b8_speedup": p["vmap8"] / p["sequential8"],
               "b4_regression": p["sequential4"] / p["vmap4"] - 1} for p in pairs]
    spread = max(max(p[key] for p in pairs) / min(p[key] for p in pairs) - 1
                 for key in ("vmap4", "vmap8"))
    decision = "inconclusive"
    if spread <= .05:
        if all(r["baseline_s2"] <= 1.10 for r in ratios):
            decision = "evidence_against_standalone_H1"
        elif all(r["baseline_s2"] >= 1.20 and
                 r["sequential_s2"] - 1 <= .5 * (r["baseline_s2"] - 1) and
                 p["sequential8"] <= .9 * p["vmap8"] and
                 r["b4_regression"] <= .05 for r, p in zip(ratios, pairs)):
            decision = "supports_H1_scheduling_remedy"
    return {"decision": decision, "pairs": ratios, "baseline_process_spread": spread,
            "VMEM_mechanism": "unresolved; timing alone is insufficient"}
