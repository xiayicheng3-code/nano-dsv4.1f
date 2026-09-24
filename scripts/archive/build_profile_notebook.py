"""Generate the real-corpus TPU profiling notebook."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def build():
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.cells = [
        nbf.v4.new_markdown_cell("""# nano-dsv4.1f — real-corpus TPU profiling

Select **TPU v5e-8**, enable Internet, attach the tokenizer and pretokenized datasets,
and start a fresh session. This compares **4 / 8 / 24 global rows**, each 8192 tokens,
with seven layers, **48 experts × width 128 / top-4, attention CP2/DP4, MoE EP8**.
The full pretraining recipe, optimizer, remat, frozen allocated DSpark, and both
base/late-indexer objectives match the stress notebook. QAT and candidate masking are off.

We measure unprofiled step time first, then capture additional warmed-up steps with
JAX/XProf. The output separates per-layer real-token routing from physical dispatch,
which includes padding. This tests whether dispatch, expert GEMMs, attention, loss,
or collectives explain the throughput drop; it does not assume the largest allocation
is the slowest operation.

Weights start from the same random seed in each case and evolve with optimizer steps.
The late phase uses schedule step 6000, **not a checkpoint trained for 6000 steps**.
Real corpus inputs improve realism, but cannot predict trained expert specialization.
Compilation takes minutes for each shape and phase."""),
        nbf.v4.new_code_cell("""import os
# Change these defaults here, before bootstrap. Existing environment values take precedence.
DEFAULTS = {
    'NANO_DSV41F_REF': 'codex/pretrain-stress-8k',
    'NANO_PROFILE_CORPUS': '/kaggle/input/datasets/xiayicheng3gmailcom/nanodsv4-1f-pretrain-tokenized',
    'NANO_PROFILE_TOKENIZER': '/kaggle/input/datasets/xiayicheng3gmailcom/nano-dsv41f-tokenizer-fineweb',
    'NANO_PROFILE_ROWS': '4,8,24',
    'NANO_PROFILE_DATA_BATCHES': '3',
    'NANO_PROFILE_DATA_SEED': '1701',
    'NANO_PROFILE_MODEL_SEED': '7',
    'NANO_PROFILE_WARMUP': '3',
    'NANO_PROFILE_STEPS': '12',
    'NANO_PROFILE_TRACE_STEPS': '3',
    'NANO_PROFILE_PHASE': 'both',
    'NANO_PROFILE_TIMEOUT': '3600',
}
for key, value in DEFAULTS.items():
    os.environ.setdefault(key, value)
print({key: os.environ[key] for key in DEFAULTS})"""),
        nbf.v4.new_code_cell((ROOT / "scripts/kaggle_bootstrap.py").read_text()),
        nbf.v4.new_markdown_cell("""## Prepare a shared sample bank

The dataset mount can contain the nested `nano-dsv41f-pretrain-3b-8k` directory;
the sampler finds its completed compact-format manifest automatically. It validates
the tokenizer SHA-256 and special IDs. No retokenization or corpus download is needed.
Only selected rows are read through memory maps, from the training split.

The default bank contains 3 × 24 randomly sampled rows from across the corpus.
For each bank slot, smaller batches use the same prefix of the 24-row batch. All
cases share sample/model seeds. Workers preload the small bank on device and cycle
it during warmup, timing and tracing, excluding disk I/O and host-to-device transfer.
Reports identify every sampled source row, batch slot and array hash. The bank is
reused for this systems experiment; this is not a pretraining data loop."""),
        nbf.v4.new_code_cell("""import json
from datetime import datetime, timezone

ROWS = [int(x) for x in os.environ['NANO_PROFILE_ROWS'].split(',')]
OUTPUT = Path('/kaggle/working') / ('pretrain-profile-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'))
OUTPUT.mkdir(parents=True, exist_ok=False)
subprocess.run([
    sys.executable, '-u', 'scripts/prepare_profile_data.py',
    '--corpus', os.environ['NANO_PROFILE_CORPUS'],
    '--tokenizer', os.environ['NANO_PROFILE_TOKENIZER'],
    '--output', str(OUTPUT / 'inputs'), '--rows', ','.join(map(str, ROWS)),
    '--batches', os.environ['NANO_PROFILE_DATA_BATCHES'],
    '--seed', os.environ['NANO_PROFILE_DATA_SEED'],
], check=True)

CASES = [dict(
    profile='narrow48', top_k=4, cp=2, dp=4, batch_rows=rows,
    data=str(OUTPUT / 'inputs' / f'rows-{rows}.npz'),
    data_batches=int(os.environ['NANO_PROFILE_DATA_BATCHES']),
    seed=int(os.environ['NANO_PROFILE_MODEL_SEED']),
    warmup=int(os.environ['NANO_PROFILE_WARMUP']), steps=int(os.environ['NANO_PROFILE_STEPS']),
    trace_steps=int(os.environ['NANO_PROFILE_TRACE_STEPS']), trace_dir=str(OUTPUT / 'traces'),
    phase=os.environ['NANO_PROFILE_PHASE'], routing='normal',
) for rows in ROWS]
PLAN = OUTPUT / 'plan.json'
PLAN.write_text(json.dumps(CASES, indent=2))
print(PLAN.read_text())"""),
        nbf.v4.new_markdown_cell("""## Time and trace isolated workers

Each case has its own TPU process. Both compiled phase executables remain resident
within that process, matching the earlier stress test. Baseline timing excludes
compilation, warmup, profiler capture, input transfers and JSON output.
Additional traced steps run after timing; their overhead is never included in tokens/s.
Per-step loads include evolving weights and router correction biases.

The profiler writes XPlane files and the worker saves optimized HLO for both phases.
Failure reports retain the last stage and any completed timing measurements. A trace
export failure is reported as a failed case, with timing results preserved."""),
        nbf.v4.new_code_cell("""result = subprocess.run([
    sys.executable, '-u', 'scripts/run_stress_suite.py', '--plan', str(PLAN),
    '--output-dir', str(OUTPUT), '--timeout-seconds', os.environ['NANO_PROFILE_TIMEOUT'],
], check=False)
print('suite return code:', result.returncode, 'reports:', OUTPUT)"""),
        nbf.v4.new_markdown_cell("""## Compare cost per token and worst-layer loads

Compare microseconds/token, not just step seconds. `LM tokens/s` excludes padding and
cross-document targets; physical tokens/s counts all 8192 slots. Load ratios are
max/mean **within each layer**, then the maximum across measured steps and layers;
they are not ratios of seven-layer sums. The chip ratio groups six resident experts
per EP chip. Full vectors and buffer utilization remain in each step's JSON.

For causal diagnosis, inspect XProf **HLO Op Profile / HLO Op Stats**, the trace viewer
and roofline analysis for each phase, normalizing operation costs by global token count.
Search HLO names for `backbone_layer_`, `moe_all_gather`, `moe_dispatch_sort`,
`ragged_expert_forward`, `moe_combine_scatter`, Splash and the LM loss.
Check time spent in gathers/scatters/sorts, expert forward/backward GEMMs, collectives,
attention and vocabulary projection/loss. Compiler fusion may combine source scopes;
source labels alone do not assign exact exclusive runtime. Confirm TPU device events
are present in the trace before drawing conclusions from it.

Install XProf in a **separate analysis environment** with `pip install xprof`, extract
the archive, then run `xprof --logdir /path/to/extracted/traces --port 8791`.
This avoids changing the pinned Kaggle JAX/TPU packages merely to view a profile.
See [JAX profiling](https://docs.jax.dev/en/latest/profiling.html)."""),
        nbf.v4.new_code_cell("""from IPython.display import display, Markdown, FileLink
summary_path = OUTPUT / 'summary.json'
if summary_path.exists():
    reports = json.loads(summary_path.read_text())
    lines = ['| Rows | Phase | Status | Median s | µs/physical token | LM tokens/s | Worst expert ratio | Worst chip ratio |',
             '|---:|---|---|---:|---:|---:|---:|---:|']
    for report in reports:
        for phase, values in report.get('phase_results', {}).items():
            measured = [s for s in values.get('steps', []) if not s['warmup']]
            def worst(key):
                ratios = [v for s in measured for v in s['routing']['physical_dispatch'][key]]
                return max(ratios) if ratios else float('nan')
            def fmt(key):
                v = values.get(key)
                return '—' if v is None else f'{v:,.3f}'
            lines.append(f\"| {report['case']['batch_rows']} | {phase} | {report['status']} | {fmt('median_seconds')} | {fmt('microseconds_per_physical_token')} | {fmt('lm_tokens_per_second')} | {worst('max_over_mean_by_layer'):.3f} | {worst('chip_max_over_mean_by_layer'):.3f} |\")
    display(Markdown('\\n'.join(lines)))
    display(FileLink(str(summary_path)))
else:
    print('No summary yet; inspect worker JSON/logs:', list(OUTPUT.glob('case-*')))"""),
        nbf.v4.new_code_cell("""import shutil
archive = shutil.make_archive(str(OUTPUT), 'zip', root_dir=OUTPUT)
display(FileLink(archive))
print('Download the ZIP for traces, HLO, timing/routing reports, and sample provenance.')"""),
    ]
    for index, cell in enumerate(nb.cells):
        cell.id = f"pretrain-profile-{index:02d}"
    path = ROOT / "notebooks/nano_dsv41f_pretrain_profile.ipynb"
    nbf.write(nb, path)
    print(path)


if __name__ == "__main__":
    build()
