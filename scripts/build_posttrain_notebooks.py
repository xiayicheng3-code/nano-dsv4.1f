"""Generate the CPU corpus notebook and resumable TPU midtrain -> SFT notebook."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
REF = 'codex/midtrain8k-sft16k'


def save(name, cells):
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.kernelspec = {'display_name':'Python 3','language':'python','name':'python3'}
    for i, c in enumerate(nb.cells):
        c.id = f'{name}-{i:02d}'
        if c.cell_type == 'code': compile(c.source, name, 'exec')
    nbf.validate(nb)
    path = ROOT / 'notebooks' / f'nano_dsv41f_{name}.ipynb'
    nbf.write(nb, path)
    return path


def build():
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cpu = save('prepare_midtrain8k_sft16k_cpu', [
        md('''# Prepare 8K midtrain + 16K SFT on CPU

Select **CPU**, turn **Internet on**, attach the tokenizer dataset
`xiayicheng3gmailcom/nano-dsv41f-tokenizer-fineweb`, and enable the Kaggle secret
`HF_TOKEN` after accepting the official Salesforce xLAM dataset access conditions.

Builds a **600M-token midtrain allocation** (480M documents / 30M reasoning /
90M agents), with 10% preparation headroom. Canonical full histories are saved
before the independent 8192/16384 packing passes. SFT uses the same selected
source pool with **16K complete prefixes**, assistant-only labels, and a separate
1:2 reasoning/agent sampler. Its one-pass budget is computed after materialization.

To reuse the unused FineWeb pretrain tail, also attach the pretrain-tokenized
dataset and completed pretrain checkpoint; set both paths below. The checkpoint's
exact shuffle seed/cursor selects unconsumed rows. Leaving both blank streams new
FineWeb-Edu instead (not guaranteed disjoint from pretraining).

Tool observations are preserved. Overlong conversations end at a complete assistant
turn; this retains the original prefix, never a disconnected tail. Overlong single
reasoning answers and Pivot targets that cannot fit are dropped. Canonical JSONL
is retained for later repacking. A stable task hash holds out 2% of trace tasks
before packing, shared across both stages. OpenResearcher requires a conservative exact final-answer match to its reference;
completion status alone is not accepted as correctness. This does not verify every action.

Source exhaustion is reported; the TPU runner refuses an underfilled 600M mix
instead of repeating data or substituting another pool silently.'''),
        code('''import os
from pathlib import Path
import subprocess
import sys
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
os.environ['RAYON_NUM_THREADS'] = str(max(1, (os.cpu_count() or 2) - 1))
SOURCE_REF = ''' + repr(REF) + '''
TOKENIZER_ROOT = Path('/kaggle/input/datasets/xiayicheng3gmailcom/nano-dsv41f-tokenizer-fineweb')
PRETRAIN_CORPUS = ''
PRETRAIN_CHECKPOINT = ''  # directory containing manifest.json or latest.json
MIDTRAIN_TOKENS = 600_000_000
OUTPUT = Path('/kaggle/working/posttrain-corpus')
# Resume CPU preparation from saved outputs by copying posttrain-corpus to OUTPUT.
# Completed source manifests are reused only with identical build settings.
ROOT = Path('/kaggle/working/nano-dsv4.1f')
if not ROOT.exists():
    subprocess.run(['git','clone','--depth','1','https://github.com/xiayicheng3-code/nano-dsv4.1f.git',str(ROOT)],check=True)
if subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain','--untracked-files=no'],text=True).strip():
    raise RuntimeError('Preserve tracked edits before updating this checkout')
subprocess.run(['git','-C',str(ROOT),'fetch','--depth','1','origin',SOURCE_REF],check=True)
subprocess.run(['git','-C',str(ROOT),'checkout','--detach','FETCH_HEAD'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','jax[cpu]==0.10.2',
                'datasets>=3.0','tokenizers>=0.21','deepseek-recipe==0.1.1','huggingface_hub'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','-e',str(ROOT),'--no-deps'],check=True)
os.chdir(ROOT)
os.environ['PYTHONPATH'] = str(ROOT / 'src')
print('Source commit:',subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())'''),
        code('''import json
from kaggle_secrets import UserSecretsClient
try:
    os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')
except Exception as exc:
    raise RuntimeError('Enable the HF_TOKEN Kaggle secret for the gated xLAM source') from exc
if not TOKENIZER_ROOT.exists():
    matches = list(Path('/kaggle/input').rglob('nano-dsv41f-tokenizer-fineweb'))
    if len(matches) != 1: raise FileNotFoundError('Attach the tokenizer dataset')
    TOKENIZER_ROOT = matches[0]
files = list(TOKENIZER_ROOT.rglob('tokenizer.json'))
if len(files) != 1: raise ValueError('Expected one tokenizer.json')
TOKENIZER = files[0]
# Check gated access before spending CPU time on the document corpus.
from datasets import load_dataset
xlam_probe = load_dataset('Salesforce/xlam-function-calling-60k',split='train',streaming=True)
next(iter(xlam_probe))
del xlam_probe
print('Official xLAM access verified')
if bool(PRETRAIN_CORPUS) != bool(PRETRAIN_CHECKPOINT):
    raise ValueError('Provide both pretrain corpus and checkpoint, or leave both blank')
command = [sys.executable,'-u','scripts/prepare_posttrain_corpus.py',
    '--tokenizer',str(TOKENIZER),'--output',str(OUTPUT),
    '--midtrain-tokens',str(MIDTRAIN_TOKENS),'--headroom','1.10',
    '--tokenize-batch-size','16','--shard-rows','128','--seed','1701']
if PRETRAIN_CORPUS:
    command += ['--pretrain-corpus',PRETRAIN_CORPUS,'--pretrain-checkpoint',PRETRAIN_CHECKPOINT]
# Does not print the token or include it in command arguments.
with Path('/kaggle/working/prepare-posttrain.log').open('a',buffering=1) as log:
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,text=True,bufsize=1)
    for line in p.stdout:
        print(line,end='',flush=True); log.write(line)
    if p.wait(): raise RuntimeError('Corpus preparation failed; inspect prepare-posttrain.log')'''),
        code('''sys.path.insert(0,str(ROOT / 'scripts'))
from posttrain_input import inspect, capacities, sft_budget
_, manifest, fingerprint = inspect(OUTPUT)
print('Corpus checksum:',fingerprint)
for stage in ('midtrain','sft'):
    print(stage, 'train capacity:', capacities(manifest,stage))
    for source in manifest['sources']:
        v = source['views'][stage]['train']
        print(source['source']['key'], {
            k:v.get(k,0) for k in ('records','real_tokens','supervised_tokens','genuine_over_8k_records')})
        if source['exhausted_before_target']:
            print('  Source exhausted before requested quota:',source['collection'])
print('Initial one-pass SFT raw-token budget:',sft_budget(manifest))
long_records = sum(s['views']['sft']['train'].get('genuine_over_8k_records',0) for s in manifest['sources'])
if not long_records: raise RuntimeError('No genuine >8K traces survived: inspect source length rejection statistics')
for pool, weight in manifest['pool_mix']['midtrain'].items():
    available = capacities(manifest,'midtrain')[pool]
    required = MIDTRAIN_TOKENS*weight + 4*8192
    if available < required:
        raise RuntimeError(f'{pool} underfilled: {available:,} < {required:,.0f}. Review source audit before training.')
print('Save Version -> Save & Run All. Save the posttrain-corpus output as a Kaggle dataset.')'''),
        md('''## Outputs

Attach the saved corpus dataset and your completed pretrain checkpoint to the TPU
notebook. Keep `posttrain_manifest.json`, `tokenizer.json`, `documents/`, `midtrain/`,
`sft/`, and `canonical/` together. Uncompressed numeric shards favor fast CPU writes
and TPU reads; no extra ZIP copy is made. Canonical/source manifests preserve licenses,
provenance, rejection counts and task-split metadata. Saving outputs does not publish
them; choose dataset visibility and attribution when you create the Kaggle dataset.''')])
    tpu = save('midtrain8k_sft16k_tpu', [
        md('''# TPU v5e-8: 8K midtrain -> 16K SFT

Select **TPU v5e-8**, enable Internet, attach the prepared posttrain corpus and the
**completed 2.4B pretrain checkpoint**. Set their paths below. This notebook runs
both remaining stages in order, with no new model initialization for training.

1. **Midtrain:** 600M nonpadding tokens, 8K rows, 80/5/15 document/reasoning/agent.
   Restores pretrained parameters **and optimizer**, preserving the original 3B LR
   horizon and global update counter. Candidate masking stays off.
2. **SFT:** 16K rows, 1:2 reasoning/agent, assistant-only targets. Resets optimizer
   and uses a new low-LR schedule. Enables compressed-layer YaRN with original
   length 8192 / factor 2 and the planned SFT candidate mask. Pure local RoPE stays
   unchanged. `SFT_TOKENS=0` computes a one-pass budget from actual pool capacity;
   it does not mean train zero tokens or concatenate every source indiscriminately.

The 16K workload has **not yet been measured on your TPU**. First compilation and
update validate the native path, loss-mask count, finite loss and zero dropped
MoE assignments. A checkpoint is saved before compilation and after the first
successful update. There is no automatic fallback to 8K if 16K cannot compile.

An eight-hour deadline includes setup, leaving margin before Kaggle's nine-hour
limit. The runner pauses safely between updates and resumes the current stage,
shuffle cursors and optimizer. Both stages may require more than one session.'''),
        code('''import os
import time
SESSION_STARTED = time.time()
SESSION_HOURS = 8.0
CORPUS = '/kaggle/input/your-posttrain-corpus-dataset'
PRETRAINED_CHECKPOINT = '/kaggle/input/your-pretrain-output/training/checkpoints'
RESUME_CHECKPOINT = ''  # later sessions: current midtrain/sft checkpoint directory
SFT_TOKENS = 0  # derive one-pass budget at fixed 1:2 ratio; keep unchanged on resume
SFT_LR = 2.6e-5
os.environ['NANO_DSV41F_REF'] = ''' + repr(REF)),
        code((ROOT / 'scripts/kaggle_bootstrap.py').read_text()),
        code('''import json
from datetime import datetime,timezone
if not Path(CORPUS).is_dir(): raise FileNotFoundError('Set CORPUS to the attached CPU notebook output')
checkpoint = RESUME_CHECKPOINT or PRETRAINED_CHECKPOINT
if not Path(checkpoint).is_dir(): raise FileNotFoundError('Set the checkpoint path')
RUN_ROOT = Path('/kaggle/working') / ('posttrain-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'))
RUN_ROOT.mkdir(exist_ok=False)
OUTPUT = RUN_ROOT / 'training'
subprocess.run(['git','archive','--format=tar.gz','-o',str(RUN_ROOT/'source.tar.gz'),'HEAD'],check=True)
command = [sys.executable,'-u','scripts/run_posttrain.py',
    '--corpus',CORPUS,'--output',str(OUTPUT),
    '--resume' if RESUME_CHECKPOINT else '--pretrained',checkpoint,
    '--sft-tokens',str(SFT_TOKENS),'--sft-lr',str(SFT_LR),
    '--deadline-unix',str(SESSION_STARTED+SESSION_HOURS*3600)]
with (RUN_ROOT/'train.log').open('w',buffering=1) as log:
    p = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    try:
        for line in p.stdout:
            print(line,end='',flush=True); log.write(line)
        rc = p.wait()
    except KeyboardInterrupt:
        p.terminate()
        print('Waiting for the runner to checkpoint between updates')
        rc = p.wait()
summary = json.loads((OUTPUT/'summary.json').read_text()) if (OUTPUT/'summary.json').exists() else {}
print(json.dumps(summary,indent=2))
if rc: raise RuntimeError('Training failed; inspect train.log. The previous committed checkpoint is retained.')'''),
        md('''## Resume / completion

Save these outputs as a Kaggle dataset. `summary.json` reports `paused`, `completed`
or `failed` and the exact latest checkpoint path. On a fresh session, attach the
saved output, set `RESUME_CHECKPOINT` to that directory (or its parent containing
`latest.json`), and keep the source version, corpus, SFT budget, LR and seed unchanged.
The source commit is printed by bootstrap and archived; pin `NANO_DSV41F_REF` to
that commit for future resumes if the branch moves.

Midtrain and SFT checkpoints have separate directories. `completed` means both
stages finished. Each checkpoint includes the effective model/YaRN configuration,
full optimizer state and data cursor. Keep the final SFT recipe with the weights
for inference. QAT and DSpark retain the existing recipe settings; this notebook
does not introduce an additional training stage.'''),
        code('''from IPython.display import FileLink,display
for name in ('summary.json','metrics.jsonl','launch.json'):
    path = OUTPUT/name
    if path.exists(): display(FileLink(str(path)))
display(FileLink(str(RUN_ROOT/'train.log')))
print('Latest checkpoint:', summary.get('checkpoint','No committed checkpoint yet'))
print('Save Version outputs to preserve checkpoint directories and source.tar.gz.')''')])
    return cpu,tpu


if __name__ == '__main__':
    for path in build(): print(path)
