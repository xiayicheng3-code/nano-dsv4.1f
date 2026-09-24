import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("capacity_search", ROOT / "scripts/capacity_search.py")
capacity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capacity)


@pytest.mark.parametrize("start,limit,expected_upper", [(4, 20, 24), (32, 12, 16), (4, 0, 4)])
def test_finds_adjacent_observed_boundaries_even_when_start_does_not_fit(start, limit, expected_upper):
    result = capacity.search_capacity(lambda rows: "fit" if rows <= limit else "device_oom",
                                     start_rows=start, max_rows=32, row_multiple=4)
    assert result["largest_passed_rows"] == (limit or None)
    assert result["smallest_device_oom_rows"] == expected_upper
    assert result["status"] == ("boundary_observed" if limit else "no_fit_observed")
    assert all(item["rows"] % 4 == 0 for item in result["attempts"])


def test_cap_pass_does_not_claim_maximum():
    result = capacity.search_capacity(lambda rows: "fit", start_rows=4, max_rows=28, row_multiple=4)
    assert result["status"] == "cap_reached"
    assert result["largest_passed_rows"] == 28
    assert result["smallest_device_oom_rows"] is None


def test_unknown_failure_preserves_lower_bound_without_inventing_upper_bound():
    result = capacity.search_capacity(lambda rows: "fit" if rows <= 8 else "unknown",
                                     start_rows=4, max_rows=32, row_multiple=4)
    assert result["status"] == "inconclusive"
    assert result["largest_passed_rows"] == 8
    assert result["smallest_device_oom_rows"] is None
    assert result["attempts"][-1] == {"rows": 16, "outcome": "unknown"}


@pytest.mark.parametrize("text,outcome", [
    ("RESOURCE_EXHAUSTED: Ran out of memory in memory space hbm.", "device_oom"),
    ("Failed to allocate device memory", "device_oom"),
    ("RESOURCE_EXHAUSTED: VMEM allocation failed", "unknown"),
    ("MemoryError: numpy could not allocate array", "unknown"),
    ("RESOURCE_EXHAUSTED: thread quota", "unknown"),
    ("Killed", "unknown"),
])
def test_only_explicit_device_memory_failures_bound_capacity(text, outcome):
    assert capacity.capacity_outcome({"status": "failed", "exception": text}) == outcome


def test_timeout_overrides_allocator_diagnostic_and_successful_retry_is_fit():
    log = "RESOURCE_EXHAUSTED: HBM allocation failed"
    assert capacity.capacity_outcome({"status": "timeout", "timed_out": True}, log) == "unknown"
    assert capacity.capacity_outcome({"status": "terminated", "returncode": -9}, log) == "unknown"
    assert capacity.capacity_outcome({"status": "failed", "exception": "nonfinite loss"}, log) == "unknown"
    assert capacity.capacity_outcome({"status": "passed", "returncode": 0}, log) == "fit"


def test_invalid_row_alignment_rejected_before_probe():
    with pytest.raises(ValueError, match="multiples"):
        capacity.search_capacity(lambda _: pytest.fail("must not execute"),
                                 start_rows=4, max_rows=30, row_multiple=4)
