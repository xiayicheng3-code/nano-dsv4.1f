# Attention batching and VMEM experiment

Status: initial TPU attempt failed before timing; mixed-precision correction
CPU-validated, corrected TPU run pending. See the
[2026-09-22 failure analysis](experiments/2026-09-22-attention-replay-failure.md).
Use [the Kaggle notebook](../notebooks/nano_dsv41f_attention_hypothesis.ipynb) on a
fresh TPU v5e-8 session, with the same two attached datasets as the real-corpus
profile. Run the default matrix first. The notebook fetches
`codex/attention-replay-mixed-precision` and records its resolved commit.

This implements the [registered H1–H4 protocol](experiments/2026-09-21-pretrain-profile.md).
The existing `vmap` and default Splash tiles remain the production defaults.
Both schedules now normalize the returned sink cotangent to the sink input dtype;
forward values and kernel math are unchanged. The baseline is remeasured with
this shared correction.

## What is controlled

- Source: the shared seven-layer narrow48/top-4 pretraining recipe, CP2/DP4,
  EP8, BF16 parameters, 8K rows, packed real-corpus inputs and seed 7.
- Untimed capture executes all seven backbone layers at initialized weights.
  Layers 0, 1 and 3 provide representative local-only, compressed global and
  uncompressed global attention inputs. Vocabulary logits and optimizer are not
  needed for capture. This is not a trained or briefly updated checkpoint.
- Frozen Q, shared K/V, segment IDs, sinks and fixed random output cotangents
  retain their original dtypes. Checksums, layer IDs, recipe and corpus source
  rows are exported. BF16 NPZ storage uses exact FP32 representations.
- Primary replay preserves **tied K=V**, including shared-buffer aliasing and
  the sum of K/V gradients. Independent K/V gradients are separately checked
  against a small dense oracle. This avoids changing the production graph
  merely to measure separate derivatives.
- The model currently uses `save_residuals=False` for all three families, in
  both base and late attention. No extra LSE-output variant is substituted.
- Repeated-row inputs duplicate the same frozen row across all replicas;
  distinct-row inputs use nested prefixes of the same frozen bank. Nothing
  updates weights, router state or optimizer during replay.
- A changes only local row scheduling: `vmap` versus elementwise `lax.map`
  inside the existing attention `shard_map`. Global training microbatch and
  MoE workload are unchanged by this option.

The notebook records normalized RMS error and max absolute error for output,
Q gradient, shared-KV gradient and sink gradient. Both variants must be finite
and pass the ≤0.01 normalized RMS gate before either is timed. The small FP32
oracle uses `atol=2e-5, rtol=2e-4`; no tolerance is chosen after seeing timings.
The eight-CPU-device tests also cover independent Q/K/V/sink gradients and a
complete late optimizer step with two local rows under sequential attention.

## Timing and interpretation

Default: 3 families × 2 compositions × 2 shapes × 3 fresh-process repeats =
36 worker processes. Each process checks and times both variants. Shape order
and variant order alternate. Workers preserve failures and partial JSON/logs.
Expect compilation to take substantially longer than the timed samples.

Three untimed TPU preflight workers first check the 8-row shape for each family,
using the real captured dtypes. Combined, forward-only and backward-only paths
must all compile and pass numerical checks. A failed preflight stops the sweep,
saves partial reports and remains visible in the notebook. Preflight results are
excluded from all timing summaries. The CPU tests also explicitly cover BF16
Q/K/V with FP32 sink gradients; the earlier all-FP32 checks missed this failure.

Each configuration has at least 3 warmups and 12 synchronized unprofiled samples.
Combined forward+VJP is the primary S2 metric. Forward-only and backward-only
are separate measurements; backward receives materialized VJP residuals as
runtime arguments. Their sum need not equal the optimized combined execution.
Combined replay does not reproduce full-block rematerialization; the optional
full-model stage is needed for training conclusions. Three further profiled
steps are captured after all timing, only in repeat zero to limit output size.

Both combined executables are resident during an A/B pair. HBM memory estimates
and device counters are diagnostics, not VMEM measurements or a capacity search.
The H1 decision function applies the preregistered thresholds separately per
family and preserves process-level medians. Missing cases, numerical failures,
insufficient repeats, or >5% baseline process spread make the result inconclusive.
H2 compares distinct and repeated data; H3 requires the optional full-model screen.

The optional H4 intervention keeps `vmap` and changes **only `block_q_dkv` from
128 to 256**, leaving other tiles/layouts at pinned JAX defaults. This probes a
larger working block; it does not promise a speedup. A changed slowdown or an
explicit VMEM allocation failure would motivate examining local-memory pressure,
but timing alone cannot establish spill/transfer causality. Optimized HLO is
exported as gzip, with XPlane traces for compiler/DMA analysis. If counters are
missing or zero placeholders, report the mechanism as unresolved. `S(1)` placement
annotations alone do not establish spilling. There is no automatic H4 verdict.

## Environment variables

Set these in the notebook's first cell; existing environment values take precedence.

| Variable | Default | Purpose |
|---|---|---|
| `NANO_ATTN_ROWS` | `4,8` | Add `24` for a larger-batch follow-up |
| `NANO_ATTN_FAMILIES` | `local,compressed,global` | All three production attention families |
| `NANO_ATTN_COMPOSITIONS` | `repeated,distinct` | Content control and real-document comparison |
| `NANO_ATTN_REPEATS` | `3` | Independent paired process repeats |
| `NANO_ATTN_WARMUP` | `3` | Warmups per measured operation |
| `NANO_ATTN_STEPS` | `12` | Synchronized unprofiled samples |
| `NANO_ATTN_TRACE_STEPS` | `3` | Additional traced steps, first repeat only; `0` disables tracing |
| `NANO_ATTN_TILE_TEST` | `0` | `1` adds the single-tile H4 probe at 4/8 rows |
| `NANO_ATTN_FULL_MODEL` | `0` | `1` enables optional base+late full-model A/B |
| `NANO_ATTN_TIMEOUT` | `1800` | Seconds per replay worker |
| `NANO_ATTN_FULL_TIMEOUT` | `3600` | Capture / full training worker timeout |
| `NANO_PROFILE_MODEL_SEED` | `7` | Shared initialized snapshot |
| `NANO_PROFILE_DATA_SEED` | `1701` | Shared corpus sample |
| `NANO_PROFILE_CORPUS` | `/kaggle/input/datasets/xiayicheng3gmailcom/nanodsv4-1f-pretrain-tokenized` | Nested compact corpus discovered automatically |
| `NANO_PROFILE_TOKENIZER` | `/kaggle/input/datasets/xiayicheng3gmailcom/nano-dsv41f-tokenizer-fineweb` | Tokenizer identity validation |

The optional full-model screen uses the existing training worker with
`--splash-batch-mode vmap` or `--splash-batch-mode sequential`, identical seeds,
optimizer initialization and batches. It measures both phases at 4/8 rows in
three process pairs. It keeps one optimizer update per whole batch and lets
weights evolve. Inspect matched-step losses/routing and at least 5% reproducible
step-time benefit; isolated parity does not certify full-sized parameter/optimizer
parity or deployment headroom. No optimization is automatically adopted.

## If four rows remains fastest

Four global rows is currently a performance reference, not a capacity boundary:
24 rows already ran. Attention-only sequential scheduling may allow a larger
global batch while retaining one-local-row kernel behavior.

If that does not help, four rows can remain the global **microbatch**. Gradient
accumulation could process six microbatches for 24 effective rows per optimizer
update: 196,608 physical token slots. That feature is not implemented by this
notebook. It requires valid-token-weighted loss/gradients, a single optimizer
and schedule update, and explicit handling of router balancing state. Timing six
ordinary four-row updates is not a valid accumulated-step benchmark.

## Outputs and local checks

Upload `attention-reports.zip` first. It includes numerical results, raw timings,
H1/H2 decisions, HLO, logs and provenance. Arrays stay in Kaggle output. The trace
ZIP is split into at most 240 MiB parts with an ordered SHA-256 manifest, so every
part fits the 512 MB upload limit. Reassemble in filename order before extraction.
Public Kaggle output also supports retrieving individual files.

```bash
python scripts/build_attention_notebook.py
XLA_FLAGS=--xla_force_host_platform_device_count=8 PYTHONPATH=src \
  python -m pytest -q tests/test_attention_replay.py tests/test_attention_parallelism.py
```

References: [JAX TPU memory and register spilling](https://docs.jax.dev/en/latest/pallas/tpu/details.html),
[TPU pipelining](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html),
[`lax.map`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.map.html).
