"""Verify preset isolation, failure handling and buffer-load accounting."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_moe_buffer_experiment as experiment
from run_pretrain_stress import routing_summary


def report(case):
    d = case['moe_buffer_divisor']
    r = dict(status='passed', arguments=case.copy(), commit='same', batch_bank=[{'sha256': 'same'}], phase_results={})
    for p in ('base', 'late'):
        step = dict(step=500 if p == 'base' else 6000, loss=3., lm_loss=3., indexer_loss=0.,
                    routing={'physical_dispatch': {'loads_by_layer': [[4, 4]],
                        'chip_loads_by_layer': [[4, 4]], 'fallback_chip_layers': 0}})
        r['phase_results'][p] = dict(steps=[step], median_seconds=1. if d == 1 else .8,
                                     diagnostics={'memory': {'estimated_total_gib': 3.}})
    return r


@pytest.mark.parametrize('failure', [None, 'preflight4', 'worker2', 'baseline'])
def test_failure_isolation_and_fixed_plan(tmp_path, monkeypatch, failure):
    calls = []
    def launch(cmd, output, timeout):
        d = int(cmd[cmd.index('--divisor')+1])
        return dict(status='failed' if failure == 'preflight4' and d == 4 else 'passed')
    def worker(case, **kwargs):
        d = case['moe_buffer_divisor'];calls.append(d)
        assert case['batch_rows'] == 4 and (case['cp'], case['dp']) == (2, 4)
        assert case['splash_batch_mode'] == 'sequential'
        assert case['splash_block_q_dkv'] == 128
        assert case['splash_compressed_block_q_dkv'] == case['splash_global_block_q_dkv'] == 1024
        assert case['trace_steps'] == 3
        r = report(case)
        if (failure == 'worker2' and d == 2) or (failure == 'baseline' and d == 1): r['status'] = 'timeout'
        return r
    monkeypatch.setattr(experiment, 'launch', launch)
    monkeypatch.setattr(experiment, 'run_case', worker)
    args = SimpleNamespace(output=tmp_path/'run', data=tmp_path/'data', repeats=3, warmup=3, steps=12, timeout=3600)
    s = experiment.run(args)
    if failure is None:
        assert calls == [1, 2, 4, 2, 4, 1, 4, 1, 2]
        assert s['comparison']['divisor_passing_screen'] in (2, 4)
    elif failure == 'preflight4':
        assert calls.count(4) == 0 and calls.count(2) == 3
    elif failure == 'worker2':
        assert calls.count(2) == 1 and calls.count(4) == 3
    else:
        assert calls == [1]
    assert json.loads((args.output/'summary.json').read_text())['status'] == s['status']


def test_pair_rejects_other_knob_changes_and_numerical_drift():
    cases = experiment.plan(Path('data'))
    a,b = [report(x['case']) for x in cases[:2]]
    assert experiment.pair(a,b,'base')['passed']
    altered = copy.deepcopy(b)
    altered['arguments']['splash_compressed_block_q_dkv'] = 512
    assert not experiment.pair(a,altered,'base')['passed']
    b['phase_results']['base']['steps'][0]['loss'] += .02
    assert not experiment.pair(a,b,'base')['passed']


def test_fallback_diagnostics_use_physical_loads():
    m = dict(experts_per_chip=2, expert_packed_rows=5,
             router_loads=np.array([[2, 2, 0, 0]]), expert_loads_by_layer=np.array([[3, 3, 0, 0]]))
    r = routing_summary(m)['physical_dispatch']
    assert r['fallback_chip_layers'] == 1
    assert r['fallback_by_layer_chip'] == [[True, False]]


def test_preset_notebook_is_valid_and_only_runs_buffer_experiment(monkeypatch):
    import os
    import nbformat
    import build_moe_buffer_notebook as b
    b.build()
    nb = nbformat.read(b.ROOT/'notebooks/nano_dsv41f_moe_buffers.ipynb', as_version=4)
    nbformat.validate(nb)
    with monkeypatch.context() as patch:
        patch.setattr(os, 'environ', {**os.environ, 'NANO_DSV41F_REF': 'old'})
        exec(nb.cells[1].source, {})
        assert os.environ['NANO_DSV41F_REF'] != 'old'
    code = '\n'.join(c.source for c in nb.cells if c.cell_type == 'code')
    assert 'scripts/run_moe_buffer_experiment.py' in code
    assert 'scripts/run_final_attention_tuning.py' not in code
    for c in nb.cells:
        if c.cell_type == 'code': compile(c.source, '<notebook>', 'exec')
