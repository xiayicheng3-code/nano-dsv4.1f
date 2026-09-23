"""Validate automatic selection, numerical gates and unattended failure isolation."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_final_attention_tuning as final


def replay_report():
    result = {'comparisons': [], 'cases': []}
    for family in final.FAMILIES:
        for tile, time in ((512, .8), (1024, .7), (2048, .69)):
            result['comparisons'].append(dict(family=family, candidate_tile=tile, complete=True, passed_repeats=3))
            for repeat in range(3):
                variants = {}
                for name, seconds in (('sequential', 1.), (f'tile{tile}', time)):
                    variants[name] = dict(combined={'median_seconds': seconds},
                        errors={n: {'passed': True, 'finite': True} for n in ('output', 'dq', 'dkv', 'dsinks')},
                        component_errors=[{'passed': True, 'finite': True} for _ in range(4)])
                result['cases'].append(dict(status='passed', variants=variants,
                    arguments=dict(family=family, tile=tile, repeat=repeat)))
    return result


def full_report(entry):
    time = 1. if entry['label'] == 'baseline128' else .9
    step = dict(step=503, loss=3., lm_loss=3., indexer_loss=0.,
                routing={'physical_dispatch': {'loads_by_layer': [[3, 5], [4, 4]]}})
    phase = dict(steps=[step], median_seconds=time, diagnostics={'memory': {'estimated_total_gib': 10.}})
    return dict(**entry, status='passed', commit='same', batch_bank=[{'sha256': f'rows-{entry["rows"]}'}],
                arguments=entry['case'], phase_results={'base': deepcopy(phase), 'late': deepcopy(phase)})


def test_selection_prefers_smaller_near_ties_and_deduplicates():
    data = replay_report()
    selected = final.select_tiles(data)
    assert selected['selected_by_family'] == {'compressed': 1024, 'global': 1024}
    assert len(selected['profiles']) == 3
    for row in data['cases']:
        if row['arguments']['tile'] > 512:
            row['status'] = 'failed'
    selected = final.select_tiles(data)
    assert selected['selected_by_family'] == {'compressed': 512, 'global': 512}
    assert set(selected['profiles']) == {'baseline128', 'fallback512'}


@pytest.mark.parametrize('problem', ['failed_gradient', 'missing_components', 'incomplete', 'unstable', 'small_gain'])
def test_invalid_or_unconvincing_large_tiles_cannot_win(problem):
    data = replay_report()
    for r in data['cases']:
        tile = r['arguments']['tile']
        if tile == 512 or r['arguments']['repeat'] != 0:
            continue
        variant = r['variants'][f'tile{tile}']
        if problem == 'failed_gradient':
            variant['errors']['dkv']['passed'] = False
        elif problem == 'missing_components':
            variant['component_errors'] = []
        elif problem == 'incomplete':
            r['arguments']['repeat'] = 1
        elif problem == 'unstable':
            variant['combined']['median_seconds'] *= 1.1
        else:
            variant['combined']['median_seconds'] = .99
    assert final.select_tiles(data)['selected_by_family'] == {'compressed': 512, 'global': 512}


def test_no_valid_tile_skips_full_model(tmp_path, monkeypatch):
    def replay(*args, **kwargs):
        path = tmp_path / 'run/replay'
        path.mkdir()
        (path / 'summary.json').write_text(json.dumps({'cases': [], 'comparisons': []}))
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(final.subprocess, 'run', replay)
    monkeypatch.setattr(final, 'run_case', lambda *a, **k: pytest.fail('must skip full model'))
    args = SimpleNamespace(output=tmp_path/'run', bank=tmp_path/'bank', inputs=tmp_path/'inputs',
        repeats=3, warmup=3, steps=12, seed=7, replay_timeout=1800, full_timeout=3600)
    assert final.run(args)['status'] == 'inconclusive'


@pytest.mark.parametrize('failed_label', ['selected', 'baseline128'])
def test_full_model_failure_isolated_by_profile_and_row_count(tmp_path, monkeypatch, failed_label):
    data = replay_report()
    calls = []
    def replay(command, **kwargs):
        assert command[command.index('--tiles') + 1] == '128,512,1024,2048'
        path = tmp_path / 'run/replay'
        path.mkdir()
        (path / 'summary.json').write_text(json.dumps(data))
        return SimpleNamespace(returncode=0)
    profiles = final.select_tiles(data)['profiles']
    def worker(case, **kwargs):
        assert case['splash_batch_mode'] == 'sequential'
        assert case['splash_block_q_dkv'] == 128
        assert case['trace_steps'] == 0 and case['phase'] == 'both'
        label = next(k for k, p in profiles.items() if p['compressed'] == case['splash_compressed_block_q_dkv'])
        calls.append((label, case['batch_rows']))
        entry = dict(case=case, label=label, repeat=0, rows=case['batch_rows'])
        report = full_report(entry)
        if label == failed_label and case['batch_rows'] == 8:
            report['status'] = 'timeout'
        return report
    monkeypatch.setattr(final.subprocess, 'run', replay)
    monkeypatch.setattr(final, 'run_case', worker)
    args = SimpleNamespace(output=tmp_path/'run', bank=tmp_path/'bank', inputs=tmp_path/'inputs',
        repeats=3, warmup=3, steps=12, seed=7, replay_timeout=1800, full_timeout=3600)
    summary = final.run(args)
    assert summary['status'] == 'completed_with_failures'
    assert len([c for c in calls if c[1] == 4]) == 9
    assert calls.count((failed_label, 8)) == 1
    if failed_label == 'selected':
        assert calls.count(('fallback512', 8)) == 3
        assert summary['full_comparison']['decisions'][1]['profile_passing_screen'] == 'fallback512'
    else:
        assert len([c for c in calls if c[1] == 8]) == 1
    assert json.loads((args.output/'summary.json').read_text())['stage'] == 'complete'


@pytest.mark.parametrize('problem', ['loss', 'routing', 'inputs', 'source', 'seed'])
def test_fast_but_invalid_full_model_result_is_rejected(problem):
    entry = dict(label='baseline128', repeat=0, rows=8, case={'seed': 7})
    a = full_report(entry)
    b = full_report({**entry, 'label': 'selected'})
    if problem == 'loss':
        b['phase_results']['base']['steps'][0]['loss'] += .02
    elif problem == 'routing':
        b['phase_results']['base']['steps'][0]['routing']['physical_dispatch']['loads_by_layer'][0] = [5, 3]
    elif problem == 'inputs':
        b['batch_bank'] = []
    elif problem == 'source':
        b['commit'] = 'other'
    else:
        b['arguments'] = {'seed': 8}
    assert not final.compare_pair(a, b, 'base')['passed']


def test_family_overrides_keep_local_tile_unchanged():
    from nano_dsv41f.tpu_native import TPUNativeConfig, _splash_tile
    options = TPUNativeConfig(splash_block_q_dkv=128,
        splash_compressed_block_q_dkv=1024, splash_global_block_q_dkv=2048)
    assert [_splash_tile(options, f) for f in ('local', 'compressed', 'global')] == [128, 1024, 2048]
    assert _splash_tile(TPUNativeConfig(splash_block_q_dkv=512), 'global') == 512
    with pytest.raises(ValueError):
        TPUNativeConfig(splash_global_block_q_dkv=4096)


def test_notebook_is_preset_and_runs_combined_supervisor(monkeypatch):
    import os
    import nbformat
    import build_final_attention_notebook as builder
    builder.build()
    notebook = nbformat.read(builder.ROOT/'notebooks/nano_dsv41f_final_attention_tuning.ipynb', as_version=4)
    with monkeypatch.context() as patch:
        patch.setattr(os, 'environ', {**os.environ, 'NANO_DSV41F_REF': 'old', 'NANO_ATTN_TILES': '128,256'})
        scope = {}
        exec(notebook.cells[1].source, scope)
        assert os.environ['NANO_ATTN_TILES'] == '128,512,1024,2048'
        assert os.environ['NANO_ATTN_ROWS'] == '4,8'
        assert os.environ['NANO_DSV41F_REF'] != 'old'
    cells = [c.source for c in notebook.cells if c.cell_type == 'code']
    for code in cells:
        compile(code, '<notebook>', 'exec')
    assert 'scripts/run_final_attention_tuning.py' in '\n'.join(cells)
    assert 'final-attention-reports.zip' in cells[-1]


@pytest.mark.parametrize('ratio', [1, 2])
def test_bf16_large_tiles_preserve_output_and_all_gradients(ratio):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from nano_dsv41f import make_v5e_mesh
    from run_attention_replay import make_function, operations
    from attention_replay_utils import error_metrics
    if jax.device_count() < 8:
        pytest.skip('requires eight CPU devices')
    # Two local rows exercise sequential scheduling; CP2 leaves 2048 queries.
    batch, length, dim = 8, 4096, 64
    kvlen = length + length // ratio
    meta = {'arrays': {'q': {'shape': [batch, length, 1, dim]},
                       'k': {'shape': [batch, kvlen, dim]}}, 'local_window': 128, 'ratio': ratio}
    rng = np.random.default_rng(23)
    q = jnp.asarray(rng.normal(size=(batch, length, 1, dim)) * .1, jnp.bfloat16)
    kv = jnp.asarray(rng.normal(size=(batch, kvlen, dim)) * .1, jnp.bfloat16)
    ct = jnp.asarray(rng.normal(size=q.shape), jnp.bfloat16)
    seg = jnp.tile(jnp.repeat(jnp.arange(4), length // 4)[None], (batch, 1))
    args = (q, kv, jnp.array([.234567], jnp.float32), seg,
            jnp.concatenate((seg, seg[:, ::ratio]), axis=1), ct)
    mesh = make_v5e_mesh()
    expected = operations(make_function(mesh, meta, 'sequential', tile=512, interpret=True))[3](*args)
    for tile in (1024, 2048):
        fwd, prepare, back, combined = operations(make_function(mesh, meta, 'sequential', tile=tile, interpret=True))
        actual = combined(*args)
        assert actual[1][-1].dtype == jnp.float32
        for x, y in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
            assert error_metrics(x, y)['passed']
        _, pb = prepare(*args[:-1])
        separate = (fwd(*args[:-1]), back(pb, ct))
        for x, y in zip(jax.tree.leaves(separate), jax.tree.leaves(actual)):
            assert error_metrics(x, y)['passed']
    jax.clear_caches()
