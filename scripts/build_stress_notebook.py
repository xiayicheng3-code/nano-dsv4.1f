"""Generate the full pretraining stress notebook, keeping code in maintained scripts."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def build():
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.cells = [
        nbf.v4.new_markdown_cell("""# nano-dsv4.1f — full 8K pretraining stress test

Select **TPU v5e-8**, enable Internet, start a fresh session, and run top-to-bottom.
This runs the actual seven-layer pretraining backbone, 32,768 vocabulary, hybrid optimizer,
BF16 payloads, block remat, Engram, Tokamax 0.0.12, and base + late-indexer objectives.
DSpark parameters/state remain allocated and frozen as in pretraining; its separate training
objective is outside this test. Candidate masking and QAT remain off.

The maintained recipe is `nano_dsv41f.pretrain_recipe.pretrain_recipe`.
It preserves the repo's training schedule while setting rows to 8192 tokens and matching
the frozen tokenizer's PAD/noise IDs. The default architecture is the current baseline;
narrow expert profiles are candidates, not a silently changed training decision.

Default: compare CP8/DP1 against CP2/DP4 at the **same four-row global batch**, then test
48 width-128 experts with top-4 routing. EP remains 8 and its reshards are included in end-to-end timings.
Each case runs two warmup + ten measured steps per phase. Compilation can take minutes.
Real TPU timings and HBM cannot be inferred from CPU validation."""),
        nbf.v4.new_code_cell("import os\nos.environ.setdefault('NANO_DSV41F_REF', 'codex/pretrain-stress-8k')"),
        nbf.v4.new_code_cell((ROOT / "scripts/kaggle_bootstrap.py").read_text()),
        nbf.v4.new_markdown_cell("""## Choose the comparison

Every case uses the full model and 8K rows. `batch_rows` is the global microbatch, not rows
per chip; it must divide evenly over DP. CP2/DP2 would use four devices and is rejected on
this eight-device recipe. Try CP4/DP2 if you want two data replicas using all eight.

`baseline=8×768, top-2`, `narrow24=24×256, top-2`, `narrow48=48×128, top-4`.
They have equal routed expert parameters; narrow24 and narrow48 each activate 512 routed
FFN units/token, with different dispatch costs and shared-expert/DSpark sizes. Use
`experts`, `width`, `top_k` for explicit alternatives. A lower step time does not establish
equal learning quality.

Start with the three cases below. Add the commented cases to investigate a specific
memory/batch limit. `long` uses full-length documents for long global histories; `packed`
tests odd segment boundaries and padding; `skewed` forces extreme expert imbalance.
Optionally add `data='/kaggle/input/.../shard.npz'` with input_ids/segment_ids/token_mask
to use real packed rows. It must contain enough rows and match the tokenizer vocabulary.
Synthetic tokens need no tokenizer download; tokenization and data I/O are outside timing."""),
        nbf.v4.new_code_cell("""import json
from datetime import datetime, timezone

CASES = [
    dict(profile='baseline', cp=8, dp=1, batch_rows=4),
    dict(profile='baseline', cp=2, dp=4, batch_rows=4),
    dict(profile='narrow48', top_k=4, cp=2, dp=4, batch_rows=4),
    # dict(profile='baseline', cp=4, dp=2, batch_rows=4),
    # dict(profile='narrow24', cp=2, dp=4, batch_rows=4),
    # dict(profile='narrow48', cp=2, dp=4, batch_rows=8),
    # dict(profile='narrow48', cp=2, dp=4, batch_rows=4, layout='packed'),
    # dict(profile='narrow48', cp=2, dp=4, batch_rows=4, routing='skewed'),
]
for case in CASES:
    case.setdefault('warmup', 2)
    case.setdefault('steps', 10)
    case.setdefault('phase', 'both')

OUTPUT = Path('/kaggle/working') / ('pretrain-stress-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'))
OUTPUT.mkdir(parents=True, exist_ok=False)
PLAN = OUTPUT / 'plan.json'
PLAN.write_text(json.dumps(CASES, indent=2))
print(PLAN.read_text())"""),
        nbf.v4.new_markdown_cell("""## Run isolated cases

A preflight checks Splash and Tokamax forward/backward first. Each model case owns a fresh
TPU process; failures are recorded and the supervisor proceeds to the next case. Inside a
case, both compiled phase executables stay resident so the late transition is measured.
The late phase starts at schedule step 6000, with the same initialized/briefly trained
weights; this is a workload test, not a checkpoint representing 6000 training steps.

Reports are saved before initialization, compile, and every step. If the entire Kaggle VM
is killed, inspect the last saved case file/log after restarting. A signal or timeout does
not identify whether failure came from host RAM, device HBM, or another cause."""),
        nbf.v4.new_code_cell("""result = subprocess.run([
    sys.executable, '-u', 'scripts/run_stress_suite.py',
    '--plan', str(PLAN), '--output-dir', str(OUTPUT), '--timeout-seconds', '3600',
], check=False)
print('suite return code:', result.returncode)
print('reports:', OUTPUT)"""),
        nbf.v4.new_markdown_cell("""## Read the results

Compile seconds are separate. Median/p95 and tokens/s exclude warmups and synchronize
the complete optimizer step. Compiler memory is an executable estimate, not an observed
total HBM peak; device memory counters are saved when supported. The estimate excludes
some runtime/executable/cache overhead, so retain measured headroom before increasing batch.
Compare identical global batches, packing, phase, and recipes. A passed short run does not
guarantee every later routing distribution fits; test the skewed case for the selected recipe.
`summary.json` contains the exact commit, complete config hash, runtime, sharding, losses,
query counts, expert loads, host peak RSS, and available device memory statistics."""),
        nbf.v4.new_code_cell("""from IPython.display import display, Markdown, FileLink
summary_path = OUTPUT / 'summary.json'
if summary_path.exists():
    reports = json.loads(summary_path.read_text())
    table = ['| Case | Status | Phase | Compile s | Median s | p95 s | LM tokens/s | Compiler GiB |',
             '|---|---|---|---:|---:|---:|---:|---:|']
    for report in reports:
        phases = report.get('phase_results', {})
        for phase, values in (phases.items() or [('—', {})]):
            def fmt(key):
                value = values.get(key)
                return '—' if value is None else f'{value:,.3f}'
            mem = (values.get('diagnostics', {}).get('memory') or {}).get('estimated_total_gib')
            memory = '—' if mem is None else f'{mem:.2f}'
            table.append(f\"| {report['name']} | {report['status']} | {phase} | {fmt('compile_seconds')} | {fmt('median_seconds')} | {fmt('p95_seconds')} | {fmt('lm_tokens_per_second')} | {memory} |\")
    display(Markdown('\\n'.join(table)))
    display(FileLink(str(summary_path)))
else:
    print('No suite summary yet; inspect per-case JSON/logs:', list(OUTPUT.glob('*')))"""),
        nbf.v4.new_markdown_cell("""## Optional: find the row-capacity boundary

Set `RUN_CAPACITY_SEARCH=True` for the selected recipe. Each probe runs both training
phases in a fresh process. It tries 4, 8, 16, ... rows up to the chosen cap, then bisects
between a passing batch and an explicit HBM/device-memory failure in DP-sized increments.
Every different batch shape requires compilation, so this can take substantially longer
than the comparison above. `MAX_ROWS=32` is a search cap, not a prediction that 32 fits.

`capacity.json` reports the largest observed passing batch and smallest explicit device
OOM. Timeouts, host kills, numerical failures and unidentified compiler errors stop with
an **inconclusive** result; they are not treated as HBM evidence. A reached cap establishes
only that the cap passed. Bisection assumes fit is monotonic with rows for this workload;
compiler algorithm changes can violate that assumption.

This measures **global microbatch capacity**, before gradient accumulation. The largest
fitting batch may not maximize tokens/s. Validate the chosen training batch with longer
normal and skewed runs and representative packed data; allow headroom for checkpointing
or evaluation buffers that the training loop may retain. Synthetic long rows exercise
long-history attention but do not cover every future routing distribution."""),
        nbf.v4.new_code_cell("""RUN_CAPACITY_SEARCH = False
MAX_ROWS = 32
CAPACITY_CASE = dict(profile='narrow48', top_k=4, cp=2, dp=4, batch_rows=4,
                     phase='both', layout='long', routing='normal', warmup=2, steps=10)

if RUN_CAPACITY_SEARCH:
    CAPACITY_OUTPUT = Path('/kaggle/working') / ('capacity-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'))
    CAPACITY_OUTPUT.mkdir(parents=True, exist_ok=False)
    capacity_plan = CAPACITY_OUTPUT / 'plan.json'
    capacity_plan.write_text(json.dumps([CAPACITY_CASE], indent=2))
    result = subprocess.run([
        sys.executable, '-u', 'scripts/run_stress_suite.py', '--plan', str(capacity_plan),
        '--output-dir', str(CAPACITY_OUTPUT), '--capacity-max-rows', str(MAX_ROWS),
        '--timeout-seconds', '3600',
    ], check=False)
    capacity_report = CAPACITY_OUTPUT / 'capacity.json'
    if capacity_report.exists():
        capacity = json.loads(capacity_report.read_text())
        display(Markdown(f\"**{capacity['status']}** — largest passed: {capacity['largest_passed_rows']} rows; smallest explicit device OOM: {capacity['smallest_device_oom_rows']} rows.\"))
        display(FileLink(str(capacity_report)))
    print('capacity return code:', result.returncode, 'reports:', CAPACITY_OUTPUT)
else:
    print('Capacity search is disabled; enable it after choosing a recipe/layout.')"""),
    ]
    for index, cell in enumerate(nb.cells):
        cell.id = f"pretrain-stress-{index:02d}"
    output = ROOT / "notebooks/nano_dsv41f_pretrain_stress.ipynb"
    nbf.write(nb, output)
    print(output)


if __name__ == "__main__":
    build()
