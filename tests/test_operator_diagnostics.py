"""Keep benchmark failure reports strict and JSON-serializable."""
import importlib.util
import json
from pathlib import Path

import jax.numpy as jnp
import pytest


spec = importlib.util.spec_from_file_location(
    'operator_benchmark', Path(__file__).parents[1] / 'scripts/benchmark_tpu_operators.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


@pytest.mark.parametrize('bad_value', [1.01, float('nan'), float('inf')])
def test_reports_failed_gradient_without_losing_output_result(bad_value):
    expected = {'output': jnp.ones(2), 'gradient': jnp.ones(2)}
    actual = {**expected, 'gradient': jnp.array([1., bad_value])}
    report = benchmark.parity_diagnostics(actual, expected, dtype=jnp.float32)
    assert report['parity'] == 'FAIL'
    assert report['atol'] == 2e-6
    assert report['rtol'] == 5e-4
    failed = [leaf['path'] for leaf in report['leaves'] if not leaf['passed']]
    assert failed == ["['gradient']"]
    json.dumps(report, allow_nan=False)


def test_identical_results_pass():
    result = (jnp.array(1.), {'output': jnp.ones(2)})
    assert benchmark.parity_diagnostics(result, result, dtype=jnp.float32)['parity'] == 'PASS'
