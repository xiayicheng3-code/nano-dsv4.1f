#!/usr/bin/env python3
"""Generate the T4 notebook that tests whether a loop-onset direction causally delays looping.

Input (Kaggle): the shipped chat bundle `xiayicheng3gmailcom/nanodsv4-1f/PyTorch/chat-dspark`.
The notebook runs on T4 x2:

1. ``repetition_direction.py collect --label onset`` on 200 held-out smol-smoltalk prompts;
2. ``repetition_direction.py steer``: inference-time ablation / subtraction of that direction,
   with norm-matched random-direction controls, split across both GPUs;
3. a paired per-prompt comparison of each intervention with the unsteered model.
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
REF = 'main'

GPU_SPLIT = (
    ('none', 'ablate_top2', 'subtract_x1', 'subtract_x2', 'random_subtract_x1'),
    ('ablate_all', 'subtract_x4', 'random_subtract_x2', 'random_subtract_x4', 'random_ablate_all'),
)


def build(out_dir=None, *, ref=REF, prompts=200):
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [
        md('''# nano-dsv4.1f: does a loop-onset direction cause looping?

The chat model loops (78% of greedy reply tokens sit inside a loop). A direction that separates
*any* repeated token from fresh text could be removed from the FFN outputs without changing
looping ([write-up](https://github.com/xiayicheng3-code/nano-dsv4.1f/blob/main/docs/experiments/2026-11-10-repetition-direction-edit.md)):
it reads out "already in a loop". This notebook targets the **entry** into a loop instead and
tests causality at inference time before any weight edit.

Select **GPU T4 x2**, Internet on, and attach the model `xiayicheng3gmailcom/nanodsv4-1f`
(variation `chat-dspark`).'''),
        code('''import os, subprocess, sys, json, time, threading
from pathlib import Path
SOURCE_REF = ''' + repr(ref) + '''
PROMPTS = ''' + repr(prompts) + '''
WORK = Path('/kaggle/working')
REPO = Path('/kaggle/temp/nano-dsv4.1f')
if not REPO.exists():
    subprocess.run(['git', 'clone', '-q', 'https://github.com/xiayicheng3-code/nano-dsv4.1f.git', str(REPO)], check=True)
subprocess.run(['git', '-C', str(REPO), 'fetch', '-q', 'origin', SOURCE_REF], check=True)
subprocess.run(['git', '-C', str(REPO), 'checkout', '-q', '--detach', 'FETCH_HEAD'], check=True)
COMMIT = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
print('source commit:', COMMIT)
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', '-q', 'jax-cuda12-plugin', 'jax-cuda12-pjrt'], check=False)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'jax==0.10.2', 'jaxlib==0.10.2', 'tokamax==0.0.12',
                'ml_dtypes', 'safetensors>=0.6', 'tokenizers>=0.21', 'datasets', 'deepseek-recipe==0.1.1'], check=True)
ENV = dict(os.environ, JAX_PLATFORMS='cpu', TOKENIZERS_PARALLELISM='false', HF_HUB_DISABLE_PROGRESS_BARS='1',
           PYTHONPATH=os.pathsep.join([str(REPO / 'src'), str(REPO / 'scripts')]))
SCRIPT = REPO / 'scripts/repetition_direction.py'
LOGS = WORK / 'logs'; LOGS.mkdir(exist_ok=True)
def run(args, log, gpu=0):
    started = time.time()
    with open(LOGS / log, 'w') as f:
        rc = subprocess.run([sys.executable, '-u', SCRIPT, *map(str, args)], stdout=f, stderr=subprocess.STDOUT,
                            env={**ENV, 'CUDA_VISIBLE_DEVICES': str(gpu)}).returncode
    print(f'[gpu{gpu}] {log}: exit {rc} after {(time.time() - started) / 60:.1f} min', flush=True)
    if rc:
        print((LOGS / log).read_text()[-3000:])
        raise RuntimeError(log)'''),
        code('''bundles = sorted({p.parent for p in Path('/kaggle/input').rglob('export_manifest.json')})
assert len(bundles) == 1, bundles
BUNDLE = bundles[0]
import hashlib
print(BUNDLE, hashlib.sha256((BUNDLE / 'model.safetensors').read_bytes()).hexdigest())'''),
        md('## 1. Loop-onset direction'),
        code('''ONSET = WORK / 'onset'
run(['collect', '--bundle', BUNDLE, '--output', ONSET, '--prompts', PROMPTS, '--max-new-tokens', 192,
     '--label', 'onset', '--min-run', 8, '--device', 'cuda'], 'collect_onset.log')
report = json.loads((ONSET / 'collect_report.json').read_text())
print(json.dumps({k: v for k, v in report.items() if k != 'layers'}, indent=2))
import pandas as pd
pd.DataFrame(report['layers']).T'''),
        md('## 2. Inference-time steering, split across both GPUs'),
        code('''sys.path.insert(0, str(REPO / 'scripts'))
import repetition_direction as R
ranked = sorted(report['layers'], key=lambda l: -report['layers'][l]['cohens_d_heldout'])
TOP2 = [int(l) for l in ranked[:2]]
names = [c['name'] for c in R.steer_configs(report, TOP2)]
split = ''' + repr(GPU_SPLIT) + '''
split = [[f"ablate_L{'-'.join(map(str, TOP2))}" if n == 'ablate_top2' else n for n in group] for group in split]
assert sorted(sum(split, [])) == sorted(names), (split, names)
print('top-2 layers', TOP2)
threads = [threading.Thread(target=run, args=(['steer', '--bundle', BUNDLE, '--directions', ONSET, '--layers', *TOP2,
                                               '--only', *group, '--device', 'cuda', '--max-new-tokens', 192,
                                               '--pairs', 200, '--output', WORK / f'steer_gpu{g}'],
                                              f'steer_gpu{g}.log', g))
           for g, group in enumerate(split)]
for t in threads: t.start()
for t in threads: t.join()'''),
        md('## 3. Paired comparison with the unsteered model (32 prompts, bootstrap 95% CI)'),
        code('''import numpy as np
configs = {}
for g in range(len(split)):
    configs.update(json.loads((WORK / f'steer_gpu{g}' / 'steer_results.json').read_text())['configs'])
base = configs['none']
def paired(metric, mode, name):
    d = np.array([e[metric] - b[metric] for e, b in zip(configs[name][mode]['samples'], base[mode]['samples'])], float)
    rng = np.random.default_rng(0)
    boot = [rng.choice(d, d.size).mean() for _ in range(5000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return f'{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]'
rows = []
for name in names:
    c = configs[name]
    row = {'config': name, 'nll': c['nll']['nll'], 'd_nll': c['nll']['nll'] - base['nll']['nll']}
    for mode in ('greedy', 't07'):
        s = c[mode]['summary']
        row.update({f'{mode}_loop_rate': s['loop_rate'], f'{mode}_median_first_loop': s['median_first_loop'],
                    f'{mode}_repeat4': s['repeat4'], f'{mode}_distinct2': s['distinct2']})
        if name != 'none':
            row[f'{mode}_d_first_loop'] = paired('first_loop', mode, name)
            row[f'{mode}_d_repeat4'] = paired('repeat4', mode, name)
    rows.append(row)
table = pd.DataFrame(rows).set_index('config')
summary = {'source_commit': COMMIT, 'bundle': str(BUNDLE), 'top2_layers': TOP2,
           'collect_report': report, 'table': table.reset_index().to_dict('records')}
(WORK / 'steering_summary.json').write_text(json.dumps(summary, indent=2))
(WORK / 'steering_summary.md').write_text(table.round(4).to_markdown())
table.round(3)'''),
        code('''best = min((n for n in names if n != 'none' and not n.startswith('random')),
           key=lambda n: -configs[n]['t07']['summary']['median_first_loop'])
print('longest T=0.7 median time to first loop:', best)
for i in range(4):
    print('\\n###', base['t07']['samples'][i]['prompt'])
    for n in ('none', best):
        print(f'--- {n}:', configs[n]['t07']['samples'][i]['reply'][:600])'''),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.kernelspec = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    for i, c in enumerate(nb.cells):
        c.id = f'repetition-steer-{i:02d}'
        if c.cell_type == 'code':
            compile(c.source, 'repetition-steer', 'exec')
    nbf.validate(nb)
    path = Path(out_dir or ROOT / 'notebooks') / 'nano_dsv41f_repetition_steering_t4.ipynb'
    nbf.write(nb, path)
    return path


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--out-dir', type=Path, default=None)
    p.add_argument('--ref', default=REF)
    p.add_argument('--prompts', type=int, default=200)
    a = p.parse_args()
    print(build(a.out_dir, ref=a.ref, prompts=a.prompts))
