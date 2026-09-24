"""Build the unattended final tile-search and full-model-validation notebook."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def build():
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.cells = [md('''# Final attention tuning — one unattended experiment

Use a **fresh Kaggle TPU v5e-8 session**, enable Internet, attach the same two
datasets, and **run all cells**. All experiment settings are preset.

1. Capture actual BF16 model activations, FP32 sinks and document masks from your
   tokenized corpus using the seven-layer, 48×128-expert/top-4 model.
2. At **8 rows**, compare sequential `block_q_dkv=512,1024,2048` independently
   against a fresh **128 control** for compressed and global attention.
3. Automatically select a valid tile **per family**, then benchmark full training
   at **4 and 8 rows** in both base and late-indexer phases. Compare all-128,
   a validated 512 fallback, and the selected family settings; deduplicate matches.
   **Local/windowed attention stays at 128 in every full-model configuration.**
4. Export one compact ZIP with raw measurements, numerical checks and the final
   comparison. Production defaults are never changed automatically.

The replay stage has 6 untimed preflights and up to 18 timed paired workers.
The full-model stage has **12–18 workers**, each measuring both phases. Every
configuration has 3 process repeats, 3 warmups and 12 timed steps. Compilation
dominates elapsed time; expect a longer run than the isolated tile notebook.
Trace capture and the previous batching/capacity sweeps are disabled.

Numerical or compilation failure excludes only the affected replay candidate.
Other candidates continue. A full-model worker failure skips repeats of that
configuration/row count while preserving the other configurations. Baseline
failure blocks comparisons at that row count. Partial reports remain available.
Timeouts alone are not classified as OOM.'''),
    code('''import os
SETTINGS = {
    'NANO_DSV41F_REF': 'codex/attention-replay-mixed-precision',
    'NANO_PROFILE_CORPUS': '/kaggle/input/datasets/xiayicheng3gmailcom/nanodsv4-1f-pretrain-tokenized',
    'NANO_PROFILE_TOKENIZER': '/kaggle/input/datasets/xiayicheng3gmailcom/nano-dsv41f-tokenizer-fineweb',
    'NANO_ATTN_ROWS': '4,8',
    'NANO_ATTN_TILES': '128,512,1024,2048',
    'NANO_ATTN_FAMILIES': 'compressed,global',
    'NANO_ATTN_REPEATS': '3',
    'NANO_ATTN_WARMUP': '3',
    'NANO_ATTN_STEPS': '12',
    'NANO_ATTN_TRACE_STEPS': '0',
    'NANO_ATTN_TIMEOUT': '1800',
    'NANO_ATTN_FULL_TIMEOUT': '3600',
    'NANO_PROFILE_DATA_SEED': '1701',
    'NANO_PROFILE_MODEL_SEED': '7',
}
# Rows, tiles, families and trace policy describe the fixed design of this supervisor.
# The budgets, paths and seeds below are passed to the workers.
# Clear the effect of earlier notebook settings, including an old pinned commit.
os.environ.update(SETTINGS)
print(SETTINGS)'''),
    code((ROOT / "scripts/kaggle_bootstrap.py").read_text()),
    md('''## Prepare nested input batches and capture frozen tensors

The four-row batch is the prefix of the eight-row batch. Data sampling, token
identity checks, activation capture, compilation and warmup are outside timing.
These are initialized weights, not a trained checkpoint. Full-model workers
update weights and router state on a fixed resident batch; disk I/O is excluded.'''),
    code('''import json
from datetime import datetime, timezone
OUTPUT = Path('/kaggle/working') / ('final-attention-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'))
OUTPUT.mkdir(parents=True, exist_ok=False)
(OUTPUT / 'environment.json').write_text(json.dumps(SETTINGS, indent=2))
subprocess.run([sys.executable, '-u', 'scripts/prepare_profile_data.py',
    '--corpus', SETTINGS['NANO_PROFILE_CORPUS'],
    '--tokenizer', SETTINGS['NANO_PROFILE_TOKENIZER'], '--output', str(OUTPUT / 'inputs'),
    '--rows', '4,8', '--batches', '1', '--seed', SETTINGS['NANO_PROFILE_DATA_SEED']], check=True)
subprocess.run([sys.executable, '-u', 'scripts/capture_attention_replay.py',
    '--data', str(OUTPUT / 'inputs' / 'rows-8.npz'), '--output', str(OUTPUT / 'bank'),
    '--rows', '8', '--seed', SETTINGS['NANO_PROFILE_MODEL_SEED']], check=True, timeout=3600)'''),
    md('''## Run the bounded search and full-model validation

Replay checks output and Q/shared-KV/sink gradients with the existing normalized
RMS error limit of 0.01. Selection requires all 3 paired repeats to pass, process
spread ≤5% for both arms, and at least 5% improvement in each repeat. Among settings
within 2% of the best median time ratio, prefer the smaller tile. The 512 fallback
must independently pass numerical checks and the stability criterion for both
families. Invalid large-tile results cannot override a valid smaller candidate.

The full-model summary uses a separate screen: ≥5% lower step time in **both**
phases across all repeats, ≤5% process spread, maximum matched-step loss difference
≤0.01 and per-layer routing histogram L1 difference / assignments ≤0.10.
These are short-trajectory screening limits, not certification of complete
parameter/optimizer equivalence. Raw loss/routing and memory data are retained.
Timings alone still cannot establish a VMEM/spilling mechanism.'''),
    code('''command = [sys.executable, '-u', 'scripts/run_final_attention_tuning.py',
    '--bank', str(OUTPUT / 'bank'), '--inputs', str(OUTPUT / 'inputs'),
    '--output', str(OUTPUT / 'experiment')]
for flag, setting in [('repeats', 'NANO_ATTN_REPEATS'), ('warmup', 'NANO_ATTN_WARMUP'),
                      ('steps', 'NANO_ATTN_STEPS'), ('seed', 'NANO_PROFILE_MODEL_SEED'),
                      ('replay-timeout', 'NANO_ATTN_TIMEOUT'), ('full-timeout', 'NANO_ATTN_FULL_TIMEOUT')]:
    command.extend(['--' + flag, SETTINGS[setting]])
result = subprocess.run(command, check=False)
print('Supervisor return code:', result.returncode)
summary_path = OUTPUT / 'experiment' / 'summary.json'
if summary_path.exists():
    summary = json.loads(summary_path.read_text())
    print('Stage:', summary.get('stage'), 'Status:', summary.get('status'))
    print(json.dumps(summary.get('selection'), indent=2))
    print(json.dumps(summary.get('full_comparison'), indent=2))
else:
    print('No combined summary; export partial reports and inspect the error output.')'''),
    md('''## Export results even if a candidate failed

Download this ZIP first, or share the public Kaggle output. It contains raw timings,
settings, provenance, numerical checks, HLO from replay, and worker failure logs.
Large arrays and traces are excluded. Failed comparisons are not performance wins.'''),
    code('''import zipfile
from IPython.display import FileLink, display
report_zip = OUTPUT / 'final-attention-reports.zip'
with zipfile.ZipFile(report_zip, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(OUTPUT.rglob('*')):
        if path.is_file() and path.suffix in ('.json', '.log', '.gz') and 'traces' not in path.parts:
            archive.write(path, path.relative_to(OUTPUT))
print('Report size (MiB):', round(report_zip.stat().st_size / 2**20, 2))
display(FileLink(str(report_zip)))
print('All outputs:', OUTPUT)''')]
    path = ROOT / "notebooks/nano_dsv41f_final_attention_tuning.ipynb"
    for i, cell in enumerate(nb.cells):
        cell.id = f"final-attention-{i:02d}"
        if cell.cell_type == "code":
            compile(cell.source, str(path), "exec")
    nbf.validate(nb)
    nbf.write(nb, path)
    print(path)


if __name__ == "__main__":
    build()
