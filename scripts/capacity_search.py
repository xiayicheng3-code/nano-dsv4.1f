"""Bounded row search. Only explicit device-memory failures establish an upper bound."""
import re


def capacity_outcome(report, log_tail=""):
    if report.get("status") == "passed" and report.get("returncode", 0) == 0:
        return "fit"
    if report.get("timed_out") or report.get("status") in ("timeout", "terminated"):
        return "unknown"
    # Require both an exhaustion marker and explicit HBM/device-memory context.
    # Generic RESOURCE_EXHAUSTED, VMEM tiling errors, host MemoryError, and SIGKILL
    # alone are not evidence of a per-row HBM limit. Inspect exception + log tail.
    # A recoverable allocation warning earlier in the log must not override the
    # final exception (for example a later numerical failure).
    text = str(report.get("exception") or log_tail)
    for line in text.lower().splitlines():
        device = re.search(r"\bhbm\b|device memory", line)
        exhausted = re.search(r"resource_exhausted|out of memory|ran out of memory|"
                              r"failed to allocate|allocation failed|exceeds? (?:the )?(?:available |capacity|limit)|"
                              r"not enough|insufficient", line)
        if device and exhausted:
            return "device_oom"
    return "unknown"


def search_capacity(probe, *, start_rows, max_rows, row_multiple, on_update=lambda state: None):
    """Double then bisect in DP-sized increments; report evidence, not universal capacity.

    probe(rows) returns fit/device_oom/unknown. Unknown stops the search without
    turning a timeout, numerical failure, or kernel error into a memory bound.
    The search assumes fit is monotonic with batch size for this fixed workload.
    """
    if row_multiple < 1 or start_rows < 1 or max_rows < start_rows:
        raise ValueError("require positive row_multiple and 0 < start_rows <= max_rows")
    if start_rows % row_multiple or max_rows % row_multiple:
        raise ValueError("start_rows and max_rows must be multiples of attention DP")
    state = {"status": "running", "largest_passed_rows": None,
             "smallest_device_oom_rows": None, "row_multiple": row_multiple,
             "max_rows_cap": max_rows, "attempts": [],
             "assumption": "Fit is monotonic with batch size for this exact workload/runtime.",
             "scope": "Observed short-run microbatch boundary; not a universal training guarantee."}

    def attempt(rows):
        state["current_rows"] = rows
        on_update(state)
        outcome = probe(rows)
        if outcome not in ("fit", "device_oom", "unknown"):
            raise ValueError(f"invalid probe outcome: {outcome}")
        state["attempts"].append({"rows": rows, "outcome": outcome})
        if outcome == "fit":
            state["largest_passed_rows"] = max(state["largest_passed_rows"] or 0, rows)
        elif outcome == "device_oom":
            state["smallest_device_oom_rows"] = min(state["smallest_device_oom_rows"] or rows, rows)
        else:
            state["status"] = "inconclusive"
        on_update(state)
        return outcome

    lo, hi = 0, None
    units, cap = start_rows // row_multiple, max_rows // row_multiple
    while hi is None:
        outcome = attempt(units * row_multiple)
        if outcome == "unknown":
            return state
        if outcome == "device_oom":
            hi = units
        else:
            lo = units
            if lo == cap:
                state["status"] = "cap_reached"
                on_update(state)
                return state
            units = min(cap, units * 2)
    while hi - lo > 1:
        units = (lo + hi) // 2
        outcome = attempt(units * row_multiple)
        if outcome == "unknown":
            return state
        if outcome == "device_oom":
            hi = units
        else:
            lo = units
    state["status"] = "boundary_observed" if lo else "no_fit_observed"
    on_update(state)
    return state
