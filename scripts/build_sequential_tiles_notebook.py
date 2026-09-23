"""Generate the small, preset sequential-attention tile experiment notebook."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def build():
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.cells = [md('''# Sequential attention: 128 / 256 / 512 query tiles

Run all cells in a **fresh Kaggle TPU v5e-8 session**, with Internet enabled and
the same tokenizer and pretokenized corpus attached. No settings need changing.

This focused follow-up tests **`block_q_dkv`**, the query tile in the K/V-gradient
kernel. All measured variants use **sequential attention, 8 global rows, CP2/DP4**.
Only compressed and global attention are timed. Other tile settings stay unchanged.
Actual BF16 Q/K/V, FP32 sinks and document masks are captured from the same
seven-layer, 48-expert/top-4 initialized model. Distinct corpus rows are frozen.

Each candidate is paired with a fresh 128 baseline: 2 families × 2 candidates ×
3 independent process repeats = **12 timed workers**, plus 4 untimed preflights.
Variant and candidate order alternate across repeats. Output and Q/shared-KV/sink
gradient checks precede timing; the normalized RMS tolerance remains 0.01.
A failed 512 preflight is retained in the report and does not block the 256 pairs.

**No old batching sweep, local-attention sweep, full-model run or TPU trace capture
is launched.** This measures whether larger tiles help the sequential kernels;
the winning setting still needs a focused full-model validation before adoption.
Compiler HBM estimates do not establish VMEM usage or spilling.'''),
    code('''import os
# Edit this dictionary if needed. Explicit assignment prevents stale settings from
# the previous broad experiment (including its pinned commit) from carrying over.
SETTINGS = {
    'NANO_DSV41F_REF': 'codex/attention-replay-mixed-precision',
    'NANO_PROFILE_CORPUS': '/kaggle/input/datasets/xiayicheng3gmailcom/nanodsv4-1f-pretrain-tokenized',
    'NANO_PROFILE_TOKENIZER': '/kaggle/input/datasets/xiayicheng3gmailcom/nano-dsv41f-tokenizer-fineweb',
    'NANO_ATTN_ROWS': '8',
    'NANO_ATTN_FAMILIES': 'compressed,global',
    'NANO_ATTN_TILES': '128,256,512',
    'NANO_ATTN_REPEATS': '3',
    'NANO_ATTN_WARMUP': '3',
    'NANO_ATTN_STEPS': '12',
    'NANO_ATTN_TRACE_STEPS': '0',
    'NANO_ATTN_TILE_TEST': '0',
    'NANO_ATTN_FULL_MODEL': '0',
    'NANO_ATTN_TIMEOUT': '1800',
    'NANO_PROFILE_DATA_SEED': '1701',
    'NANO_PROFILE_MODEL_SEED': '7',
}
os.environ.update(SETTINGS)
print(SETTINGS)'''),
    code((ROOT / "scripts/kaggle_bootstrap.py").read_text()),
    md('''## Capture fixed model inputs

Input preparation and capture are untimed. The parent notebook process never
initializes JAX. Corpus/tokenizer hashes and captured tensor metadata are saved.'''),
    code('''import json
from datetime import datetime, timezone
OUTPUT = Path('/kaggle/working') / ('sequential-tiles-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'))
OUTPUT.mkdir(parents=True, exist_ok=False)
(OUTPUT / 'environment.json').write_text(json.dumps(SETTINGS, indent=2))
subprocess.run([sys.executable, '-u', 'scripts/prepare_profile_data.py',
    '--corpus', SETTINGS['NANO_PROFILE_CORPUS'],
    '--tokenizer', SETTINGS['NANO_PROFILE_TOKENIZER'], '--output', str(OUTPUT / 'inputs'),
    '--rows', SETTINGS['NANO_ATTN_ROWS'], '--batches', '1',
    '--seed', SETTINGS['NANO_PROFILE_DATA_SEED']], check=True)
subprocess.run([sys.executable, '-u', 'scripts/capture_attention_replay.py',
    '--data', str(OUTPUT / 'inputs' / ('rows-' + SETTINGS['NANO_ATTN_ROWS'] + '.npz')),
    '--output', str(OUTPUT / 'bank'), '--rows', SETTINGS['NANO_ATTN_ROWS'],
    '--seed', SETTINGS['NANO_PROFILE_MODEL_SEED']], check=True, timeout=3600)'''),
    md('''## Run only sequential tile comparisons

Each successful pair gets 3 warmups and 12 synchronized unprofiled samples for
combined forward+VJP, forward-only and backward-only. Use the combined time as the
primary measurement; forward and backward separately need not sum to it.
Compilation or numerical failures remain visible as incomplete comparisons.'''),
    code('''command = [sys.executable, '-u', 'scripts/run_sequential_tiles.py',
    '--bank', str(OUTPUT / 'bank'), '--output', str(OUTPUT / 'replay')]
for flag, setting in [('rows', 'ROWS'), ('families', 'FAMILIES'), ('tiles', 'TILES'),
                      ('repeats', 'REPEATS'), ('warmup', 'WARMUP'), ('steps', 'STEPS'),
                      ('timeout', 'TIMEOUT')]:
    command.extend(['--' + flag, SETTINGS['NANO_ATTN_' + setting]])
# Trace capture is deliberately disabled in this small experiment.
command.extend(['--trace-steps', '0'])
result = subprocess.run(command, check=False)
print('Experiment return code:', result.returncode)
summary_path = OUTPUT / 'replay' / 'summary.json'
if summary_path.exists():
    summary = json.loads(summary_path.read_text())
    print(json.dumps({k: v for k, v in summary.items() if k not in ('cases', 'preflights')}, indent=2))
    for preflight in summary['preflights']:
        if preflight['status'] != 'passed':
            print('FAILED PREFLIGHT:', preflight['arguments'], preflight.get('stage'), preflight.get('exception'))
else:
    print('No summary was produced; check the notebook error output.')'''),
    md('''## Download compact reports

The ZIP includes settings, input manifests, raw timings, numerical checks,
compiler HLO and failure logs. Frozen arrays and traces are excluded.'''),
    code('''import zipfile
from IPython.display import FileLink, display
report_zip = OUTPUT / 'sequential-tile-reports.zip'
with zipfile.ZipFile(report_zip, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(OUTPUT.rglob('*')):
        if path.is_file() and path.suffix in ('.json', '.log', '.gz') and 'traces' not in path.parts:
            archive.write(path, path.relative_to(OUTPUT))
print('Report size (MiB):', round(report_zip.stat().st_size / 2**20, 2))
display(FileLink(str(report_zip)))
print('All outputs:', OUTPUT)''')]
    path = ROOT / "notebooks/nano_dsv41f_sequential_tiles.ipynb"
    for i, cell in enumerate(nb.cells):
        cell.id = f"sequential-tiles-{i:02d}"
        if cell.cell_type == "code":
            compile(cell.source, str(path), "exec")
    nbf.validate(nb)
    nbf.write(nb, path)
    print(path)


if __name__ == "__main__":
    build()
