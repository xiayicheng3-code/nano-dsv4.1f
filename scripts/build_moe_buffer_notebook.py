"""Build the preset four-row MoE buffer experiment notebook."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def build():
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    nb.cells = [md('''# Reduced MoE buffers — preset experiment

Use a fresh Kaggle TPU v5e-8 session with Internet and the same corpus/tokenizer
datasets attached. **Run All**; only the buffer experiment runs.

Fixed settings: **4 global rows ×8192**, CP2/DP4 attention, EP8 MoE,
sequential attention, compressed/global `block_q_dkv=1024`, local tile128,
48 experts ×128 width, top-4 plus the shared expert, BF16 payloads and FP32
controls, the existing optimizer, block rematerialization and indexer policy.
These are validated settings; the architecture and remat policy are not claimed
optimal and are held fixed to isolate this intervention.

Compare per-chip fast-buffer capacities **131072 / 65536 / 32768** assignments.
If a chip's load exceeds the smaller buffer, that chip executes the original
full-buffer branch. No assignments are dropped. Both branches are compiled;
peak reserved memory can remain large even if the fast path does less work.

Three numerical preflight workers check normal and deliberately skewed routing,
outputs, input gradients and all parameter gradients. Only passing candidates
advance to nine full-model workers: three configurations ×three process repeats,
each with base and late phases, three warmups and twelve timed steps per phase.
Three additional steps are traced **after** timing in every phase/repeat so all
paired runs have the same number of updates. Traces can identify whether the
combine/scatter scope became faster. The same initialized seed and resident
corpus batch are used across configurations. This is not long-training validation.

The report records actual maximum chip loads and fallback chip-layer counts.
Candidate failures are isolated; baseline failure stops meaningful comparisons.
Production defaults remain unchanged.'''),
    code('''import os
SETTINGS = {
    'NANO_DSV41F_REF': 'main',
    'NANO_PROFILE_CORPUS': '/kaggle/input/datasets/xiayicheng3gmailcom/nanodsv4-1f-pretrain-tokenized',
    'NANO_PROFILE_TOKENIZER': '/kaggle/input/datasets/xiayicheng3gmailcom/nano-dsv41f-tokenizer-fineweb',
    'NANO_BUFFER_REPEATS': '3', 'NANO_BUFFER_WARMUP': '3',
    'NANO_BUFFER_STEPS': '12', 'NANO_BUFFER_TIMEOUT': '3600',
    'NANO_PROFILE_DATA_SEED': '1701',
}
# Explicitly replace stale source/budget settings. Model settings live in the fixed plan.
os.environ.update(SETTINGS)
print(SETTINGS)'''),
    code((ROOT/'scripts/kaggle_bootstrap.py').read_text()),
    code('''import json
from datetime import datetime, timezone
OUTPUT = Path('/kaggle/working') / ('moe-buffer-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'))
OUTPUT.mkdir(parents=True, exist_ok=False)
(OUTPUT/'environment.json').write_text(json.dumps(SETTINGS, indent=2))
subprocess.run([sys.executable, '-u', 'scripts/prepare_profile_data.py',
    '--corpus', SETTINGS['NANO_PROFILE_CORPUS'], '--tokenizer', SETTINGS['NANO_PROFILE_TOKENIZER'],
    '--output', str(OUTPUT/'inputs'), '--rows', '4,8', '--batches', '1',
    '--seed', SETTINGS['NANO_PROFILE_DATA_SEED']], check=True)'''),
    code('''command = [sys.executable, '-u', 'scripts/run_moe_buffer_experiment.py',
    '--data', str(OUTPUT/'inputs/rows-4.npz'), '--output', str(OUTPUT/'experiment')]
for flag in ('repeats', 'warmup', 'steps', 'timeout'):
    command.extend(['--' + flag, SETTINGS['NANO_BUFFER_' + flag.upper()]])
result = subprocess.run(command, check=False)
print('Supervisor return code:', result.returncode)
summary_path = OUTPUT/'experiment/summary.json'
if summary_path.exists():
    summary = json.loads(summary_path.read_text())
    print('Status:', summary['status'])
    print(json.dumps(summary.get('comparison'), indent=2))
else:
    print('No summary was produced; inspect the notebook output.')'''),
    md('''## Export

Download the compact ZIP first or share the public Kaggle output. It contains
raw timings, source/settings, preflight numerical checks, loss/routing screens,
fallback counts, compiler memory estimates and worker logs. Full traces and HLO
remain separately in the Kaggle outputs for detailed attribution.

The automatic timing screen requires ≥5% improvement in each paired repeat in
both phases and ≤5% process spread. All raw results are retained, including smaller
improvements. Loss/routing comparisons are short-trajectory screens, not a complete
parameter-equivalence certificate. No default is automatically adopted.'''),
    code('''import zipfile
from IPython.display import FileLink, display
report_zip = OUTPUT/'moe-buffer-reports.zip'
with zipfile.ZipFile(report_zip, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(OUTPUT.rglob('*')):
        if path.is_file() and path.suffix in ('.json', '.log') and 'traces' not in path.parts:
            archive.write(path, path.relative_to(OUTPUT))
display(FileLink(str(report_zip)))
print('Trace and HLO outputs:', OUTPUT/'experiment/full-model')''')]
    path = ROOT/'notebooks/nano_dsv41f_moe_buffers.ipynb'
    for i, cell in enumerate(nb.cells):
        cell.id = f'moe-buffer-{i:02d}'
        if cell.cell_type == 'code': compile(cell.source, str(path), 'exec')
    nbf.validate(nb)
    nbf.write(nb, path)
    print(path)


if __name__ == '__main__':
    build()
