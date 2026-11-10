#!/usr/bin/env python3
"""Generate the GPU notebook that evaluates the chat-SFT MTP ablation.

Inputs (Kaggle): the two TPU chat-SFT runs' outputs (baseline and MTP arms) and the
earlier agent/reasoning SFT bundle as a reference. The notebook exports both SFT
checkpoints to portable bundles, then on T4 x2:

1. runs the same zero-shot lm-evaluation-harness suite as the standard-evals notebook;
2. samples 32 fixed chat prompts greedily and with seeded T=0.7 / top-p 0.9 sampling;
3. collects the held-out losses recorded by both TPU runs.
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
REF = 'main'


def build(out_dir=None, *, ref=REF):
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [
        md('''# nano-dsv4.1f chat SFT: MTP-in-SFT ablation evaluation

Compares three checkpoints that share the same pretrain + mid-train weights:

| name | SFT data | MTP loss |
|---|---|---|
| `old_sft` | reasoning + agent traces (earlier release) | no |
| `chat_baseline` | smol-smoltalk chat (95%) + reasoning (5%), 441M tokens | no |
| `chat_mtp` | identical data, seed and schedule | DSpark MTP, weight 0.1 |

Select **GPU T4 x2** and Internet on. Attach the two TPU run outputs
(`nano-dsv4-1f-chat-sft-baseline`, `nano-dsv4-1f-chat-sft-mtp`) and the model
`xiayicheng3gmailcom/nanodsv4-1f` (the earlier SFT bundle).'''),
        code('''import os, subprocess, sys, json, time
from pathlib import Path
SOURCE_REF = ''' + repr(ref) + '''
WORK = Path('/kaggle/working')
REPO = Path('/kaggle/temp/nano-dsv4.1f')
if not REPO.exists():
    subprocess.run(['git', 'clone', '-q', 'https://github.com/xiayicheng3-code/nano-dsv4.1f.git', str(REPO)], check=True)
subprocess.run(['git', '-C', str(REPO), 'fetch', '-q', 'origin', SOURCE_REF], check=True)
subprocess.run(['git', '-C', str(REPO), 'checkout', '-q', '--detach', 'FETCH_HEAD'], check=True)
print('source commit:', subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip())
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', '-q', 'jax-cuda12-plugin', 'jax-cuda12-pjrt'], check=False)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'lm_eval[hf]==0.4.13', 'jax==0.10.2', 'jaxlib==0.10.2',
                'tokamax==0.0.12', 'ml_dtypes', 'tabulate', 'safetensors>=0.6', 'tokenizers>=0.21',
                'deepseek-recipe==0.1.1'], check=True)
ENV = dict(os.environ, JAX_PLATFORMS='cpu', TOKENIZERS_PARALLELISM='false', HF_HUB_DISABLE_PROGRESS_BARS='1',
           PYTHONPATH=os.pathsep.join([str(REPO / 'src'), str(REPO / 'scripts')]))
def run(args, **kw):
    print('$', ' '.join(map(str, args))[:300], flush=True)
    return subprocess.run([sys.executable, '-u', *map(str, args)], env={**ENV, **kw.pop('env', {})}, check=True, **kw)'''),
        md('## Export both chat-SFT checkpoints'),
        code('''INPUT = Path('/kaggle/input')
BUNDLES = WORK / 'bundles'; BUNDLES.mkdir(exist_ok=True)
def checkpoints(tag):
    hits = sorted({p.parent for p in INPUT.rglob('sft/checkpoints/latest.json') if f'chat-sft-{tag}-' in str(p)})
    if len(hits) != 1:
        raise FileNotFoundError(f'expected one {tag} SFT run, found {hits}')
    return hits[0]
models = {}
for tag in ('baseline', 'mtp'):
    out = BUNDLES / f'chat_{tag}'
    if not (out / 'model.safetensors').exists():
        run([REPO / 'scripts/export_posttrain_checkpoint.py', '--checkpoint', checkpoints(tag), '--output', out])
    models[f'chat_{tag}'] = out
old = sorted({p.parent for p in INPUT.rglob('export_manifest.json') if 'chat-sft' not in str(p)})
assert len(old) == 1, old
models = {'old_sft': old[0], **models}
summaries = {}
for tag in ('baseline', 'mtp'):
    summary = json.loads((checkpoints(tag).parents[1] / 'summary.json').read_text())
    summaries[f'chat_{tag}'] = summary
    print(tag, summary['status'], summary['plan'])
print(models)'''),
        md('## Held-out loss recorded by the TPU runs'),
        code('''import pandas as pd
rows = []
for name, s in summaries.items():
    for e in s['evaluations']:
        rows.append({'model': name, 'pool': e['pool'], 'step': e['step'], 'loss': e['loss']})
val = pd.DataFrame(rows).pivot_table(index='step', columns=['pool', 'model'], values='loss')
val.to_csv(WORK / 'heldout_loss.csv')
val.round(4)'''),
        md('## Benchmarks (lm-evaluation-harness, zero-shot, full sets) and chat samples, one model per GPU'),
        code('''import threading, queue, torch
RESULTS = WORK / 'results'; RESULTS.mkdir(exist_ok=True)
LOGS = WORK / 'logs'; LOGS.mkdir(exist_ok=True)
jobs = queue.Queue()
for name, path in models.items():
    jobs.put(('bench', name, [REPO / 'scripts/nano_lm_eval.py', '--bundle', path, '--name', name,
                              '--output', RESULTS / f'bench_{name}.json']))
bundle_args = [x for name, path in models.items() for x in ('--bundle', f'{name}={path}')]
jobs.put(('samples', 'greedy', [REPO / 'scripts/chat_samples.py', *bundle_args, '--max-new-tokens', 512,
                                '--output', RESULTS / 'samples_greedy']))
jobs.put(('samples', 't07', [REPO / 'scripts/chat_samples.py', *bundle_args, '--max-new-tokens', 512,
                             '--temperature', 0.7, '--top-p', 0.9, '--seed', 0, '--output', RESULTS / 'samples_t07']))
def worker(gpu):
    while True:
        try:
            kind, name, args = jobs.get_nowait()
        except queue.Empty:
            return
        started = time.time()
        with open(LOGS / f'{kind}_{name}.log', 'w') as log:
            rc = subprocess.run([sys.executable, '-u', *map(str, args)], stdout=log, stderr=subprocess.STDOUT,
                                env={**ENV, 'CUDA_VISIBLE_DEVICES': str(gpu)}).returncode
        print(f'[gpu{gpu}] {kind} {name}: exit {rc} after {(time.time() - started) / 60:.1f} min', flush=True)
threads = [threading.Thread(target=worker, args=(g,)) for g in range(max(1, torch.cuda.device_count()))]
for t in threads: t.start()
for t in threads: t.join()'''),
        code('''sys.path.insert(0, str(REPO / 'scripts'))
import nano_lm_eval as E
bench = {n: json.loads((RESULTS / f'bench_{n}.json').read_text()) for n in models if (RESULTS / f'bench_{n}.json').exists()}
table = pd.DataFrame(E.primary_table(bench, E.DEFAULT_TASKS)).set_index('model')
show = list(E.DEFAULT_TASKS) + ['avg_acc']
pretty = table[show].copy()
for t in show:
    pretty[t] = pretty[t].map(lambda v: '—' if pd.isna(v) else (f'{v:.3f}' if t == 'wikitext' else f'{100 * v:.1f}'))
(WORK / 'benchmarks.md').write_text(pretty.to_markdown())
json.dump(bench, open(WORK / 'benchmarks.json', 'w'), indent=2)
pretty'''),
        code('''for mode in ('greedy', 't07'):
    path = RESULTS / f'samples_{mode}.json'
    if path.exists():
        res = json.loads(path.read_text())
        print(mode, pd.DataFrame({n: r['summary'] for n, r in res.items()}).round(3).to_string(), sep='\\n')
print((RESULTS / 'samples_t07.md').read_text()[:6000] if (RESULTS / 'samples_t07.md').exists() else 'no samples')'''),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.kernelspec = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    for i, c in enumerate(nb.cells):
        c.id = f'ablation-eval-{i:02d}'
        if c.cell_type == 'code':
            compile(c.source, 'ablation-eval', 'exec')
    nbf.validate(nb)
    path = Path(out_dir or ROOT / 'notebooks') / 'nano_dsv41f_chat_sft_ablation_eval.ipynb'
    nbf.write(nb, path)
    return path


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--out-dir', type=Path, default=None)
    p.add_argument('--ref', default=REF)
    a = p.parse_args()
    print(build(a.out_dir, ref=a.ref))
