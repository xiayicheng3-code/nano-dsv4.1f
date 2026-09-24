"""One-variable buffer comparison at the validated four-row attention settings."""
import argparse
import copy
import json
import os
from pathlib import Path
from statistics import median
import sys

from run_attention_experiment import launch
from run_final_attention_tuning import compare_pair
from run_pretrain_stress import save_report
from run_stress_suite import run_case


def plan(data, repeats=3, warmup=3, steps=12):
    out = []
    for repeat in range(repeats):
        # Rotate so every capacity appears once in each execution position.
        order = (1, 2, 4)
        order = order[repeat % 3:] + order[:repeat % 3]
        for divisor in order:
            case = dict(profile='narrow48', top_k=4, cp=2, dp=4, batch_rows=4,
                splash_batch_mode='sequential', splash_block_q_dkv=128,
                splash_compressed_block_q_dkv=1024, splash_global_block_q_dkv=1024,
                moe_buffer_divisor=divisor, data=str(data), data_batches=1, seed=7,
                phase='both', warmup=warmup, steps=steps, trace_steps=3)
            out.append(dict(divisor=divisor, repeat=repeat, case=case))
    return out


def pair(a, b, phase):
    # Reuse input/commit/step/loss/routing checks, allowing exactly this intervention.
    a, b = copy.deepcopy(a), copy.deepcopy(b)
    ignored = {'output', 'trace_dir', 'moe_buffer_divisor'}
    settings = [{k: v for k, v in r.get('arguments', {}).items() if k not in ignored} for r in (a, b)]
    if settings[0] != settings[1]:
        return dict(passed=False, reason='non-buffer worker settings differ')
    for r in (a, b):
        r['arguments'].pop('moe_buffer_divisor', None)
    return compare_pair(a, b, phase)


def summarize(reports, repeats):
    groups = []
    for divisor in (1, 2, 4):
        rs = [r for r in reports if r['divisor'] == divisor]
        complete = (len(rs) == repeats and {r['repeat'] for r in rs} == set(range(repeats))
                    and all(r['status'] == 'passed' for r in rs))
        g = dict(divisor=divisor, buffer_rows=131072//divisor, complete=complete, phases={})
        if complete:
            for phase in ('base', 'late'):
                ps = [r['phase_results'][phase] for r in rs]
                times = [p['median_seconds'] for p in ps]
                steps = [s for p in ps for s in p['steps'] + p.get('trace', {}).get('steps', [])]
                memory = [(p.get('diagnostics', {}).get('memory') or {}).get('estimated_total_gib') for p in ps]
                item = dict(median_seconds=median(times), process_spread=max(times)/min(times)-1,
                    physical_tokens_per_second=32768/median(times),
                    max_compiler_estimate_gib=max((m for m in memory if m is not None), default=None),
                    max_chip_assignments=max(v for s in steps for layer in
                        s['routing']['physical_dispatch']['chip_loads_by_layer'] for v in layer),
                    fallback_chip_layers=sum(s['routing']['physical_dispatch'].get('fallback_chip_layers', 0) for s in steps))
                if divisor != 1:
                    checks = []
                    for r in rs:
                        base = next((b for b in reports if b['divisor'] == 1 and b['repeat'] == r['repeat']), None)
                        checks.append(pair(base, r, phase) if base else dict(passed=False, reason='missing baseline'))
                    item['paired_checks'] = checks
                    base_times = [c['baseline_seconds'] for c in checks if 'baseline_seconds' in c]
                    item['meets_screen'] = (item['process_spread'] <= .05 and len(base_times) == repeats
                        and max(base_times)/min(base_times)-1 <= .05 and
                        all(c['passed'] and c.get('time_reduction_fraction', -1) >= .05 for c in checks))
                g['phases'][phase] = item
        groups.append(g)
    eligible = [g for g in groups if g['divisor'] != 1 and g['complete'] and
                all(p['meets_screen'] for p in g['phases'].values())]
    winner = min(eligible, key=lambda g: sum(p['median_seconds'] for p in g['phases'].values()), default=None)
    return dict(groups=groups, divisor_passing_screen=winner['divisor'] if winner else None,
        note='Three complete repeats, >=5% time reduction in every pair and both phases, <=5% process spread; '
             'matched-step loss/routing screen. No automatic adoption. Compiler memory includes both branches; '
             'a smaller fast buffer need not lower reserved peak memory. Short initialized runs do not certify long training.')


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    pre = a.output/'preflight';pre.mkdir()
    full = a.output/'full-model';full.mkdir()
    summary = dict(status='running', preflights=[], reports=[], skipped=[])
    eligible = set()
    for divisor in (1, 2, 4):
        report = launch([sys.executable, '-u', 'scripts/run_moe_buffer_preflight.py',
                        '--divisor', str(divisor), '--output', str(pre/f'divisor-{divisor}.json')],
                        pre/f'divisor-{divisor}.json', a.timeout)
        report['divisor'] = divisor
        summary['preflights'].append(report)
        if report['status'] == 'passed': eligible.add(divisor)
        save_report(a.output/'summary.json', summary)
    entries = plan(a.data, a.repeats, a.warmup, a.steps)
    save_report(a.output/'plan.json', entries)
    failed = set()
    for i, entry in enumerate(entries):
        d = entry['divisor']
        if 1 not in eligible or 1 in failed or d not in eligible or d in failed:
            summary['skipped'].append(entry)
            continue
        case = {**entry['case'], 'trace_dir': str(full/'traces')}
        r = run_case(case, index=i, output_dir=full, timeout_seconds=a.timeout,
                     environment=dict(os.environ, JAX_PLATFORMS='tpu'))
        r.update(divisor=d, repeat=entry['repeat'])
        summary['reports'].append(r)
        if r['status'] != 'passed': failed.add(d)
        summary['comparison'] = summarize(summary['reports'], a.repeats)
        save_report(a.output/'summary.json', summary)
    summary['comparison'] = summarize(summary['reports'], a.repeats)
    summary['status'] = 'complete' if eligible == {1, 2, 4} and not failed else 'completed_with_failures'
    save_report(a.output/'summary.json', summary)
    print(json.dumps(summary['comparison'], indent=2))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--steps', type=int, default=12)
    p.add_argument('--timeout', type=int, default=3600)
    a = p.parse_args()
    if not a.data.is_file() or a.repeats < 3 or a.warmup < 3 or a.steps < 12 or a.timeout <= 0:
        p.error('require input file, >=3 repeats/warmups, >=12 timed steps, positive timeout')
    run(a)


if __name__ == '__main__':
    main()
