"""Generate the Kaggle notebook for the preregistered attention hypothesis test."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def build():
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.cells = [md('''# nano-dsv4.1f — attention batching and VMEM hypothesis

Start a **fresh TPU v5e-8 session**, enable Internet, and attach the tokenizer and
pretokenized corpus. This notebook tests why 4 rows had higher tokens/s than 8/24.
**Four rows is a throughput reference, not a demonstrated capacity limit.**

The source model is the seven-layer pretraining recipe: 48 × 128 experts, top-4,
BF16 payloads, attention CP2/DP4 and MoE EP8. A real-corpus forward pass captures
actual Q/K/V, segment masks and sinks at layers 0/1/3 (local, compressed and
uncompressed global attention). All seven backbone layers run during capture;
vocabulary logits and optimizer work are excluded from this untimed capture.
Weights are initialized from seed 7, not a trained checkpoint.

We replay the frozen tensors as runtime arguments with fixed cotangents. Each
family uses the production tied K=V buffers, `save_residuals=False` mode and exact 8K shapes.
A repeats one row on every replica; B uses distinct captured corpus rows.
At each shape, compare production `vmap` with one-local-row-at-a-time `lax.map`.
Only attention scheduling changes; this is not gradient accumulation.

Three fresh-process repeats alternate variant and row-count order. Forward and
Q/shared-KV/sink VJPs must pass the registered numerical gate before timing. Forward,
backward-only (materialized residuals), and combined forward+VJP are measured
separately. Combined timing is the primary H1 metric; these isolated timings do
not include full-block rematerialization, MoE or optimizer work.

**H4 / VMEM remains unresolved from timing alone.** An optional intervention changes
only `block_q_dkv` from the pinned default 128 to 256. This increases one working
block dimension; it is a pressure/scheduling probe, not a promised optimization.
Inspect compiler placement and usable spill/DMA evidence with the exported HLO
and traces. Compiler HBM estimates do not measure VMEM occupancy.

Protocol: [registered hypotheses](https://github.com/xiayicheng3-code/nano-dsv4.1f/blob/codex/pretrain-stress-8k/docs/experiments/2026-09-21-pretrain-profile.md).
This experiment does not change production defaults or launch pretraining.'''),
    code('''import os
DEFAULTS = {
    'NANO_DSV41F_REF': 'codex/pretrain-stress-8k',
    'NANO_PROFILE_CORPUS': '/kaggle/input/datasets/xiayicheng3gmailcom/nanodsv4-1f-pretrain-tokenized',
    'NANO_PROFILE_TOKENIZER': '/kaggle/input/datasets/xiayicheng3gmailcom/nano-dsv41f-tokenizer-fineweb',
    'NANO_ATTN_ROWS': '4,8',                 # optionally 4,8,24
    'NANO_ATTN_FAMILIES': 'local,compressed,global',
    'NANO_ATTN_COMPOSITIONS': 'repeated,distinct',
    'NANO_ATTN_REPEATS': '3',
    'NANO_ATTN_WARMUP': '3',
    'NANO_ATTN_STEPS': '12',
    'NANO_ATTN_TRACE_STEPS': '3',            # additional steps, first repeat only
    'NANO_ATTN_TILE_TEST': '0',             # optional H4: vmap, block_q_dkv 128 -> 256
    'NANO_ATTN_FULL_MODEL': '0',            # optional full training A/B after replay
    'NANO_ATTN_TIMEOUT': '1800',            # per isolated replay worker
    'NANO_ATTN_FULL_TIMEOUT': '3600',
    'NANO_PROFILE_DATA_SEED': '1701',
    'NANO_PROFILE_MODEL_SEED': '7',
}
for key, value in DEFAULTS.items():
    os.environ.setdefault(key, value)
print({key: os.environ[key] for key in DEFAULTS})'''),
    code((ROOT / "scripts/kaggle_bootstrap.py").read_text()),
    md('''## Check semantics before allocating the TPU workload

A small FP32 dense oracle checks all three attention families, both schedules,
the tile intervention, and Q/K/V/sink gradients on eight virtual CPU devices.
The fixed tolerance is `atol=2e-5, rtol=2e-4`. This validates semantics; it does not
predict physical TPU performance. Full-size TPU A/B checks additionally require
finite values and normalized RMS error ≤0.01 for every output/gradient tensor.
Neither test changes tolerances based on timing results.'''),
    code('''subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pytest>=8'], check=True)
cpu_env = dict(os.environ, JAX_PLATFORMS='cpu',
               XLA_FLAGS='--xla_force_host_platform_device_count=8')
subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_attention_replay.py'],
               env=cpu_env, check=True, timeout=600)'''),
    md('''## Capture a frozen bank from real text

The sampler finds the nested compact corpus, validates tokenizer identity and
reconstructs packed segment masks. Input tensors retain their actual dtypes;
BF16 values are stored losslessly as FP32 in NPZ and restored before replay.
The manifest records the code, recipe, seeds, layer IDs and tensor hashes.
Repeated inputs use captured row zero on every replica; distinct inputs use
nested prefixes of the same frozen bank. No input I/O or transfers are timed.'''),
    code('''import json
from datetime import datetime, timezone
ROWS = [int(x) for x in os.environ['NANO_ATTN_ROWS'].split(',')]
if not {4, 8}.issubset(ROWS) or any(x not in (4, 8, 24) for x in ROWS):
    raise ValueError('Use rows 4,8 or 4,8,24')
OUTPUT = Path('/kaggle/working') / ('attention-hypothesis-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'))
OUTPUT.mkdir(parents=True, exist_ok=False)
(OUTPUT / 'environment.json').write_text(json.dumps({k: os.environ[k] for k in DEFAULTS}, indent=2))
subprocess.run([
    sys.executable, '-u', 'scripts/prepare_profile_data.py',
    '--corpus', os.environ['NANO_PROFILE_CORPUS'],
    '--tokenizer', os.environ['NANO_PROFILE_TOKENIZER'],
    '--output', str(OUTPUT / 'inputs'), '--rows', ','.join(map(str, ROWS)),
    '--batches', '1', '--seed', os.environ['NANO_PROFILE_DATA_SEED'],
], check=True)
subprocess.run([
    sys.executable, '-u', 'scripts/capture_attention_replay.py',
    '--data', str(OUTPUT / 'inputs' / f'rows-{max(ROWS)}.npz'),
    '--output', str(OUTPUT / 'bank'), '--rows', str(max(ROWS)),
    '--seed', os.environ['NANO_PROFILE_MODEL_SEED'],
], check=True, timeout=int(os.environ['NANO_ATTN_FULL_TIMEOUT']))
print('Frozen bank:', OUTPUT / 'bank' / 'manifest.json')'''),
    md('''## Run paired attention experiments

The default matrix has **36 worker processes**: 3 families × 2 compositions ×
2 row counts × 3 repeats. Each worker compiles both schedules, checks equivalence,
then collects 12 synchronized samples after 3 warmups. Compilation can dominate
elapsed notebook time. The parent process never initializes a TPU client.

Trace capture follows all unprofiled measurements and occurs only in repeat zero.
Each worker saves partial JSON, a log and compressed optimized HLO. Failed cases
remain visible and prevent a confident decision. A timeout or killed process
alone is not classified as VMEM/HBM OOM. Both combined executables are resident
within a pair; reported memory is not an isolated production capacity estimate.'''),
    code('''command = [sys.executable, '-u', 'scripts/run_attention_experiment.py',
           '--bank', str(OUTPUT / 'bank'), '--output', str(OUTPUT / 'replay')]
for flag, name in [
    ('rows', 'ROWS'), ('families', 'FAMILIES'), ('compositions', 'COMPOSITIONS'),
    ('repeats', 'REPEATS'), ('warmup', 'WARMUP'), ('steps', 'STEPS'),
    ('trace-steps', 'TRACE_STEPS'), ('tile-test', 'TILE_TEST'), ('timeout', 'TIMEOUT')]:
    command.extend(['--' + flag, os.environ['NANO_ATTN_' + name]])
result = subprocess.run(command, check=False)
print('Replay return code:', result.returncode)
print('Reports:', OUTPUT / 'replay')'''),
    md('''## Read the decision report

For each family, `S2 = time(8 rows) / (2 × time(4 rows))` uses combined forward+VJP.
H1 support requires baseline S2 ≥1.20 in every repeat, halving excess `(S2−1)`,
at least 10% lower 8-row time, ≤5% 4-row regression, and ≤5% baseline process spread.
Baseline S2 ≤1.10 consistently is evidence against standalone H1. Other outcomes
are inconclusive; samples are not pooled across processes.

The backward-only timings include materialized residual handling. Combined timing
may optimize differently; do not expect `forward + backward` to equal combined.
A successful row loop alone cannot prove VMEM spilling. An isolated result also
cannot select a training batch size or estimate the maximum fitting rows.'''),
    code('''from IPython.display import display, Markdown, FileLink
summary = json.loads((OUTPUT / 'replay' / 'summary.json').read_text())
print(json.dumps({k: v for k, v in summary.items() if k != 'cases'}, indent=2))
lines = ['| Family | Inputs | Rows | Repeat | Variant | Combined ms | Forward ms | Backward ms |',
         '|---|---|---:|---:|---|---:|---:|---:|']
for case in summary['cases']:
    a = case.get('arguments', {})
    for variant, values in case.get('variants', {}).items():
        def ms(key):
            value = values.get(key, {}).get('median_seconds')
            return '—' if value is None else f'{1000 * value:.3f}'
        lines.append(f"| {a.get('family')} | {a.get('composition')} | {a.get('rows')} | {a.get('repeat')} | {variant} | {ms('combined')} | {ms('forward')} | {ms('backward')} |")
display(Markdown('\\n'.join(lines)))
display(FileLink(str(OUTPUT / 'replay' / 'summary.json')))'''),
    md('''## Optional: verify full training-step benefit

Set `NANO_ATTN_FULL_MODEL=1` before running this cell after reviewing replay checks.
This runs the existing full pretraining worker from the same initialization seed,
optimizer initialization and fixed corpus batch for each schedule. It retains the
entire global batch and one optimizer update per step. The only experimental knob
is attention row scheduling; MoE stays EP8 and uses all global rows together.

Both base and late phases are measured at 4 and 8 rows, with three paired process
repeats. Parameters evolve identically in intent but floating-point differences
can propagate; inspect matched-step losses/routing as well as throughput. Tiny
full-step dense-oracle tests cover optimizer semantics in the repository. This
screen does not automatically certify full-sized parameter/optimizer equivalence
or choose a deployment memory margin. Require reproducible ≥5% full-step benefit,
numerical validation, and acceptable memory headroom before adopting a variant.

If four rows remains the best throughput point, it can be the **global microbatch**:
6 accumulated microbatches would give 24 rows / 196,608 physical tokens per optimizer
update. Accumulation is not implemented by this experiment. A training implementation
must normalize by valid LM tokens, update optimizer/schedule once, and explicitly
handle router balancing state across microbatches. Six ordinary optimizer steps
are not equivalent to one accumulated step.'''),
    code('''if os.environ['NANO_ATTN_FULL_MODEL'] == '1':
    if any(case.get('status') != 'passed' for case in summary['cases']):
        raise RuntimeError('Resolve replay failures before full-model testing.')
    full = OUTPUT / 'full-model'
    full.mkdir(exist_ok=False)
    cases = []
    for repeat in range(int(os.environ['NANO_ATTN_REPEATS'])):
        modes = ['vmap', 'sequential'] if repeat % 2 == 0 else ['sequential', 'vmap']
        for rows in ([4, 8] if repeat % 2 == 0 else [8, 4]):
            for mode in modes:
                cases.append(dict(profile='narrow48', top_k=4, cp=2, dp=4,
                    batch_rows=rows, splash_batch_mode=mode,
                    data=str(OUTPUT / 'inputs' / f'rows-{rows}.npz'), data_batches=1,
                    seed=int(os.environ['NANO_PROFILE_MODEL_SEED']),
                    phase='both', warmup=int(os.environ['NANO_ATTN_WARMUP']),
                    steps=int(os.environ['NANO_ATTN_STEPS']),
                    trace_steps=int(os.environ['NANO_ATTN_TRACE_STEPS']) if repeat == 0 else 0,
                    trace_dir=str(full / 'traces')))
    (full / 'plan.json').write_text(json.dumps(cases, indent=2))
    subprocess.run([sys.executable, '-u', 'scripts/run_stress_suite.py',
        '--plan', str(full / 'plan.json'), '--output-dir', str(full),
        '--timeout-seconds', os.environ['NANO_ATTN_FULL_TIMEOUT']], check=False)
    print('Full-model reports:', full / 'summary.json')
else:
    print('Optional full-model screen skipped; set NANO_ATTN_FULL_MODEL=1 to run it.')'''),
    md('''## Export compact results and upload-sized trace parts

Upload `attention-reports.zip` first: it contains decisions, raw timings, numerical
errors, logs, HLO and provenance. The frozen arrays and sampled corpus remain in
Kaggle output, excluded from this compact archive. Traces are written as separate
ZIP parts of at most 240 MiB. If needed, concatenate the parts in filename order
into one ZIP before extraction. The split manifest includes hashes and order.
Public Kaggle output also lets us retrieve individual files without a large upload.'''),
    code('''import hashlib
import zipfile
report_zip = OUTPUT / 'attention-reports.zip'
trace_zip = OUTPUT / 'attention-traces.zip'
with zipfile.ZipFile(report_zip, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for path in sorted(OUTPUT.rglob('*')):
        if path.is_file() and path.suffix in ('.json', '.log', '.gz') and 'traces' not in path.parts:
            z.write(path, path.relative_to(OUTPUT))
display(FileLink(str(report_zip)))
traces = [p for p in OUTPUT.rglob('*') if p.is_file() and 'traces' in p.parts]
if traces:
    with zipfile.ZipFile(trace_zip, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for path in traces:
            z.write(path, path.relative_to(OUTPUT))
    parts = []
    with trace_zip.open('rb') as source:
        index = 0
        while chunk := source.read(240 * 1024**2):
            path = OUTPUT / f'attention-traces.zip.part{index:03d}'
            path.write_bytes(chunk)
            parts.append(dict(file=path.name, bytes=len(chunk), sha256=hashlib.sha256(chunk).hexdigest()))
            display(FileLink(str(path)))
            index += 1
    trace_zip.unlink()
    (OUTPUT / 'trace-parts.json').write_text(json.dumps(parts, indent=2))
    display(FileLink(str(OUTPUT / 'trace-parts.json')))
print('All output:', OUTPUT)''')]
    for i, cell in enumerate(nb.cells):
        cell.id = f"attention-hypothesis-{i:02d}"
    path = ROOT / "notebooks/nano_dsv41f_attention_hypothesis.ipynb"
    nbf.write(nb, path)
    nbf.validate(nb)
    for cell in nb.cells:
        if cell.cell_type == "code":
            compile(cell.source, str(path), "exec")
    print(path)


if __name__ == "__main__":
    build()
