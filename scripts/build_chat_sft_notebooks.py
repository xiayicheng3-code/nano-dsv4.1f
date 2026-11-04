#!/usr/bin/env python3
"""Generate the CPU chat-corpus notebook and the TPU chat-SFT (MTP ablation) notebook."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
REF = 'main'


def save(name, cells, out_dir=None):
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.kernelspec = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    for i, c in enumerate(nb.cells):
        c.id = f'{name}-{i:02d}'
        if c.cell_type == 'code':
            compile(c.source, name, 'exec')
    nbf.validate(nb)
    path = Path(out_dir or ROOT / 'notebooks') / f'nano_dsv41f_{name}.ipynb'
    nbf.write(nb, path)
    return path


def build(out_dir=None, *, ref=REF, max_train_records=0, mtp_weight=0.0, max_steps=0):
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cpu = save('prepare_chat_sft_cpu', [
        md('''# Prepare the 32K chat SFT corpus (CPU)

Select **CPU** and turn **Internet on**. Attach the frozen tokenizer dataset
`xiayicheng3gmailcom/nano-dsv41f-tokenizer-fineweb`.

Streams [HuggingFaceTB/smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk),
the Apache-2.0 SmolTalk subset Hugging Face built for sub-1B models (SmolLM2-135M/360M-Instruct).
It contains no function-calling data. Each conversation is rendered with the
DeepSeek V4.1 encoder in **non-thinking** mode, tokenized with the frozen nano
tokenizer and packed into 32K rows with **assistant-only** loss masks. The dataset's
test split becomes validation, so both SFT arms are scored on the same held-out chats.
The resolved dataset commit is written to `chat_sft_manifest.json`.

Save the output (`chat-sft-corpus/`) and attach it to the TPU notebook. A few
subsets carry upstream terms (OpenHermes / Llama-3.1-generated rows), so keep the
saved corpus private rather than republishing it.'''),
        code('''import os
from pathlib import Path
import subprocess
import sys
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
SOURCE_REF = ''' + repr(ref) + '''
MAX_TRAIN_RECORDS = ''' + repr(max_train_records) + '''  # 0 = whole split
OUTPUT = Path('/kaggle/working/chat-sft-corpus')
ROOT = Path('/kaggle/working/nano-dsv4.1f')
if not ROOT.exists():
    subprocess.run(['git', 'clone', '-q', 'https://github.com/xiayicheng3-code/nano-dsv4.1f.git', str(ROOT)], check=True)
subprocess.run(['git', '-C', str(ROOT), 'fetch', '-q', 'origin', SOURCE_REF], check=True)
subprocess.run(['git', '-C', str(ROOT), 'checkout', '-q', '--detach', 'FETCH_HEAD'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'jax[cpu]==0.10.2', 'tokamax==0.0.12',
                'datasets>=3.0', 'tokenizers>=0.21', 'deepseek-recipe==0.1.1', 'huggingface_hub'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', str(ROOT), '--no-deps'], check=True)
os.chdir(ROOT)
os.environ['PYTHONPATH'] = os.pathsep.join([str(ROOT / 'src'), str(ROOT / 'scripts')])
print('Source commit:', subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())'''),
        code('''import json
tokenizers = [p for p in Path('/kaggle/input').rglob('tokenizer.json')
              if 'nano-dsv41f-tokenizer-fineweb' in str(p)]
if len(tokenizers) != 1:
    raise FileNotFoundError('Attach the nano-dsv41f-tokenizer-fineweb dataset')
command = [sys.executable, '-u', 'scripts/prepare_chat_sft_corpus.py', '--output', str(OUTPUT),
           '--tokenizer', str(tokenizers[0]), '--max-train-records', str(MAX_TRAIN_RECORDS)]
subprocess.run(command, check=True)
manifest = json.loads((OUTPUT / 'chat_sft_manifest.json').read_text())
for split, stats in manifest['splits'].items():
    print(split, {k: v for k, v in stats.items() if k not in ('shards', 'subset_tokens')})
    total = sum(stats['subset_tokens'].values())
    for subset, tokens in sorted(stats['subset_tokens'].items(), key=lambda kv: -kv[1]):
        print(f'   {subset:28s} {tokens / total:6.1%}')
print('dataset revision:', manifest['dataset']['resolved_revision'])'''),
    ], out_dir)

    tpu = save('chat_sft_tpu', [
        md('''# TPU v5e-8: 32K chat SFT from the mid-trained checkpoint (MTP ablation)

Select **TPU v5e-8**, enable Internet, and attach:

1. the **chat SFT corpus** saved by `nano_dsv41f_prepare_chat_sft_cpu.ipynb`;
2. `xiayicheng3gmailcom/nanodsv4-1f-sftcheckpoint`, which holds the completed
   **mid-training checkpoint** (`midtrain/checkpoints`, step 85,085) used as the start;
3. `xiayicheng3gmailcom/nanodsv4-1fposttraincorpus` for the optional reasoning slice.

This replaces the earlier reasoning/agent SFT with chat data and keeps that stage's
recipe: fresh optimizer, 2.6e-5 → 2.6e-6 cosine LR, YaRN ×4 to 32K and the
candidate-masked indexer auxiliary. `REASONING_FRACTION` keeps a small share of the
existing reasoning traces so thinking mode keeps its format; agent traces are dropped.

**Ablation:** run this notebook twice with identical settings except `MTP_WEIGHT`:
`0.0` (baseline) and `0.1` (DSpark multi-token-prediction loss with gradients into
the backbone, 256 sampled anchors per 32K row). Compare held-out chat loss
(logged at step 0 and every 500 updates), benchmarks and samples.

A one-pass run is about 3,400 updates; the previous 32K SFT measured ~2.6 s/update,
so a run should finish in one session. Checkpoints are saved before compilation and
every 500 updates; set `RESUME_CHECKPOINT` to continue a paused run.'''),
        code('''import os
import time
SESSION_STARTED = time.time()
SESSION_HOURS = 8.0
MTP_WEIGHT = ''' + repr(mtp_weight) + '''  # 0.0 = baseline arm, 0.1 = MTP arm; keep unchanged on resume
MTP_ANCHORS = 256
REASONING_FRACTION = 0.10
SFT_TOKENS = 0  # 0 = one pass over the chat corpus at this mix
MAX_STEPS = ''' + repr(max_steps) + '''  # >0 only for a smoke test
RESUME_CHECKPOINT = ''  # later sessions: this run's training/sft/checkpoints directory
os.environ['NANO_DSV41F_REF'] = ''' + repr(ref)),
        code((ROOT / 'scripts/kaggle_bootstrap.py').read_text()),
        code('''import json
from datetime import datetime, timezone
def find_one(pattern, keep=lambda p: True, what='input'):
    hits = sorted({p.parent for p in Path('/kaggle/input').rglob(pattern) if keep(p)})
    if len(hits) != 1:
        raise FileNotFoundError(f'Expected one {what}, found {len(hits)}: {hits[:4]}')
    return hits[0]
chat_corpus = find_one('chat_sft_manifest.json', what='chat SFT corpus')
midtrain = find_one('latest.json', lambda p: p.parent.name == 'checkpoints' and p.parent.parent.name == 'midtrain',
                    'midtrain checkpoint directory')
reasoning = find_one('posttrain_manifest.json', what='v3 posttrain corpus') if REASONING_FRACTION else None
arm = 'mtp' if MTP_WEIGHT else 'baseline'
RUN_ROOT = Path('/kaggle/working') / (f'chat-sft-{arm}-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'))
RUN_ROOT.mkdir(exist_ok=False)
OUTPUT = RUN_ROOT / 'training'
subprocess.run(['git', 'archive', '--format=tar.gz', '-o', str(RUN_ROOT / 'source.tar.gz'), 'HEAD'], check=True)
command = [sys.executable, '-u', 'scripts/run_chat_sft.py', '--chat-corpus', str(chat_corpus),
           '--output', str(OUTPUT), '--mtp-weight', str(MTP_WEIGHT), '--mtp-anchors', str(MTP_ANCHORS),
           '--reasoning-fraction', str(REASONING_FRACTION), '--sft-tokens', str(SFT_TOKENS),
           '--max-steps', str(MAX_STEPS),
           '--deadline-unix', str(SESSION_STARTED + SESSION_HOURS * 3600)]
command += ['--resume', RESUME_CHECKPOINT] if RESUME_CHECKPOINT else ['--midtrain', str(midtrain)]
if reasoning:
    command += ['--reasoning-corpus', str(reasoning)]
print(' '.join(command))
with (RUN_ROOT / 'train.log').open('w', buffering=1) as log:
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in p.stdout:
            if 'Autotuning cache miss' in line:
                continue  # Tokamax autotune notices; kept out of the log for readability
            print(line, end='', flush=True); log.write(line)
        rc = p.wait()
    except KeyboardInterrupt:
        p.terminate()
        print('Waiting for the runner to checkpoint between updates')
        rc = p.wait()
summary = json.loads((OUTPUT / 'summary.json').read_text()) if (OUTPUT / 'summary.json').exists() else {}
print(json.dumps({k: v for k, v in summary.items() if k != 'error'}, indent=2))
if rc:
    print(summary.get('error', ''))
    raise RuntimeError('Training failed; inspect train.log. The previous committed checkpoint is retained.')'''),
        md('''## Outputs

`training/summary.json` holds the plan, held-out losses and the final checkpoint;
`metrics.jsonl` the training curve (including MTP loss and per-depth draft accuracy);
`sft-compile-*.json` the compiler memory/collective diagnostics. Save this version's
output; the final `sft/checkpoints` directory exports with
`scripts/export_posttrain_checkpoint.py` like the previous SFT.'''),
    ], out_dir)
    return cpu, tpu


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir', type=Path, default=None)
    p.add_argument('--ref', default=REF)
    p.add_argument('--max-train-records', type=int, default=0)
    p.add_argument('--mtp-weight', type=float, default=0.0)
    p.add_argument('--max-steps', type=int, default=0)
    a = p.parse_args()
    for path in build(a.out_dir, ref=a.ref, max_train_records=a.max_train_records,
                      mtp_weight=a.mtp_weight, max_steps=a.max_steps):
        print(path)
