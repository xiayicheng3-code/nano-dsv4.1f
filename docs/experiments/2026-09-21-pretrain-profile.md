# 8K pretraining profile: observations and next-experiment hypotheses

Status: **baseline measured; causal interventions below are proposed and unrun**.
Recorded after inspecting the 2026-09-21 Kaggle run. This note makes the predictions
explicit before running the next benchmark. It does not change the training recipe.

## Evidence and reproducibility

- Public run: [xiayicheng3gmailcom/nano-dsv4-1f](https://www.kaggle.com/code/xiayicheng3gmailcom/nano-dsv4-1f/output), output version ID `351468121`.
- Output directory: `pretrain-profile-20260921-110434`.
- Measured code: `0b52e74dd5fa578aabff926752a684ef03c7d547`.
- Hardware: eight TPU v5e devices, one host. JAX/jaxlib `0.10.2`,
  libtpu `0.0.42.1`, Tokamax `0.0.12`; offline XProf analysis `2.23.2`.
- Seven backbone layers, d_model 512, 32768 vocabulary, 48 width-128 experts,
  top-4, attention CP2/DP4, MoE EP8, 8192-token rows, BF16 payloads,
  block rematerialization and the full optimizer. DSpark allocated/frozen;
  QAT and candidate masking off.
- Three sampled real-corpus batches, nested prefixes across batch sizes;
  corpus seed 1701, model seed 7. Three warmups, twelve unprofiled measured
  updates, then three additional traced updates per phase.
- Both phases use initialized/briefly updated parameters. Late schedule step 6000
  is not a checkpoint trained for 6000 steps. Data is already on device during timing.

The [compact measurements](2026-09-21-pretrain-profile.json) preserve unrounded
values, config/batch hashes, package versions, measured routing maxima and hashes
of the summary and analyzed XProf exports. Raw traces remain in the public output;
the repository does not contain the multi-gigabyte trace payloads.

## What was observed

| Global rows | Local rows per attention replica | Base median ms | Late median ms | Base physical tokens/s | Late physical tokens/s |
|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 400.39 | 428.82 | 81,840 | 76,414 |
| 8 | 2 | 953.07 | 1,010.97 | 68,763 | 64,825 |
| 24 | 6 | 2,937.85 | 3,115.61 | 66,922 | 63,104 |

All six phase/batch combinations passed. These are short-run fit observations,
not a new maximum-capacity search. The base compiler memory estimates were
2.22 / 4.29 / 12.24 GiB; they are not measured total device peaks.

Base-phase source-scope attribution from XProf:

| Scope | B=4 ms/device/step | B=8 | B=24 | B=8 / (2 × B=4) |
|---|---:|---:|---:|---:|
| Splash attention | 109.57 | 353.89 | 994.90 | 1.615 |
| MoE combine/scatter | 83.47 | 185.94 | 561.32 | 1.114 |
| Expert GEMMs and activation scope | 41.76 | 91.33 | 282.83 | 1.094 |
| MoE all-gather scope | 19.67 | 38.96 | 126.22 | 0.990 |
| MoE sort/dispatch scope | 18.49 | 37.10 | 114.01 | 1.003 |
| Other/unattributed scopes | 117.69 | 235.07 | 862.88 | 0.999 |

Attribution sums `total_self_time` across HLO entries with the indicated source
scope, dividing by 8 devices × 3 traced steps. It is average device work, not
critical-path wall time. Fusion can affect source attribution. The expert scope
includes surrounding activation work; the listed MoE scopes do not capture every
MoE-related operation. In particular, some dispatch gathers, backward scatters,
reduce-scatters and shared-expert operations are in `other`.

The summed base device work is 390.64 / 942.29 / 2942.17 ms; XProf reports mean
traced steps of 406.20 / 960.34 / 2956.29 ms. The earlier table reports different,
unprofiled steps. Profiling, evolving parameters and sample order preclude treating
these as identical observations.

The 4→8-row excess above linear scaling is approximately 161.0 ms of summed device
work, of which the Splash source scope contributes about 134.8 ms. This localizes
most of the observed scaling regression to attention. It is not proof of an
84% recoverable end-to-end speedup or a diagnosis of memory bandwidth saturation.
For the analyzed late traces, Splash is 109.72 ms at B=4 and 985.86 ms at B=24,
showing a similar pattern. The B=8 late trace was captured but not converted in
this analysis; its compact HLO field is explicitly null.

Real-text routing is skewed. In the B=24 late measured steps, the worst physical
expert max/mean is 10.095 and the worst chip max/mean is 2.290. These are maxima
over steps and layers, not seven-layer aggregate ratios. Routing imbalance can
matter, but these observations do not establish that it causes the attention
regression. Expanded MoE buffers remain a major capacity concern independently.

## Conclusions we can and cannot draw

**Supported:** increasing rows reduces throughput in this run; Splash is the
largest contributor to the extra normalized device time between B=4 and B=8.
A larger allocation by itself does not explain declining tokens/s.

**Leading explanation to test:** the change from one local attention row to a
batched Splash call creates a less efficient execution path. The implementation
uses `jax.vmap(one)` inside the attention `shard_map`; the HLO has a singleton
batch dimension removed at B=4 and explicit local batch sizes 2/6 at B=8/24.
This is a correlation, not a causal demonstration.

**Still unknown:** whether this is intrinsic batching overhead, local-memory or
pipeline behavior, mask/data composition, or interactions with the surrounding
compiled/rematerialized graph. HBM/MXU utilization fields in the analyzed overview
were zero and are not usable evidence of actual zero utilization or saturation.

## Hypotheses registered before the next run

The hypotheses can coexist. Thresholds below are engineering decision rules,
chosen in advance; they are not statistical significance tests.

| ID | Falsifiable claim | Intervention and prediction | Evidence against / limit |
|---|---|---|---|
| H1: batched execution | Multiple local rows in the current Splash batched path cause a material per-token penalty even with identical row content. | Repeat the same fixed row to build B=4 and B=8 at CP2/DP4. Compare existing `vmap` with elementwise `lax.map` over local rows, holding the attention math fixed. The baseline penalty should persist; the alternative should reduce it. | If repeated-row isolated `vmap` has ≤10% per-token penalty, the standalone batching claim is not supported. If both implementations have the same cliff, the proposed scheduling remedy is not supported. A loop can add overhead, so a failed remedy does not rule out every batching mechanism. |
| H2: input composition | Different packed masks or activations account for a material part of the observed attention penalty. | Compare repeated-row inputs with the original distinct-row bank at each fixed shape and frozen weights. Support requires at least a 10% fixed-shape time/token difference and S2 falling from ≥1.20 on distinct rows to ≤1.10 on repeated rows across paired repeats. | A similar cliff on identical repeated rows makes different documents unnecessary to explain it. This control changes values/masks together; it does not by itself distinguish mask effects from activation effects. |
| H3: surrounding graph | The problematic behavior requires full-step scheduling/remat or other model work, rather than the isolated attention operation alone. | Replay fixed attention inputs outside the model, then compare a full-model A/B changing only local-row attention scheduling. A cliff present only in the full graph supports a graph interaction. | Reproduction in the isolated attention forward/VJP weakens the claim that MoE, optimizer or full-model residency is necessary. Isolation must preserve the kernel, layout, precision and residual mode. |
| H4: local-memory/pipeline mechanism | A local-memory placement or DMA/pipeline change explains the batching penalty. | After H1–H3 localization, collect compiler memory-placement/spill evidence and usable DMA/compute counters, then vary one legal Splash tile or buffering setting with the same inputs and math. Predict a reproducible decrease in both the implicated stall/transfer measure and time/token. | A capacity estimate, `S(1)` annotation, or faster row loop alone is insufficient. If counters or compiler evidence are unavailable, leave this mechanism unresolved; do not label it bandwidth-bound. |

MoE dispatch and routing deserve their own subsequent experiment with fixed
assignments and forward/backward equivalence. Changing top-k, expert width/count,
or dropping tokens would confound that comparison and is outside this first test.

## Smallest useful next experiment

Start with B=4 and B=8, where the observed discontinuity is largest. Keep CP2/DP4;
changing CP also changes local Q length, KV replication and communication.

1. **Freeze the inputs.** Save Q, K/V, masks, sink values and a fixed output
   cotangent from a named model snapshot/batch. Record hashes and layer IDs.
   Benchmark each distinct attention family used by the recipe: local-only,
   compressed global history and uncompressed global history. Preserve each
   family's exact shapes and residual mode. For the repeated-row control, put
   the same row on all four data replicas and repeat it twice at B=8, so both
   per-row content and CP rank assignments stay controlled.
2. **Change one execution choice.** A is the current local `vmap`; B invokes
   the same per-row kernel through `lax.map(lambda xs: one(*xs), inputs)`.
   This is a proposed implementation, not a guaranteed TPU-supported speedup.
   Keep kernel tiles, sharding, precision, masks and attention sinks identical.
   Use runtime array arguments, not captured constant tensors that can fold away.
3. **Check equivalence before timing.** Check forward outputs, residuals when
   requested, and VJPs for Q, K/V and sinks. Use small-shape dense-oracle checks
   and full-shape A/B comparisons. Before seeing performance results, record
   tolerances: FP32 oracle checks `atol=2e-5, rtol=2e-4`; full BF16 A/B normalized
   RMS error ≤0.01 for each output/gradient tensor, with all values finite.
   Define normalized RMS error as `norm(actual-reference) / max(norm(reference), 1e-12)`.
   Record maximum absolute error too. Exclude a variant that fails; any tolerance
   change requires a documented reason and re-registration before timing.
4. **Measure forward and backward separately, plus together.** Account for
   forward recomputation from block remat when relating isolated results to a
   full training step. A forward-only improvement is insufficient. No optimizer
   updates in this replay; both variants receive identical tensors/cotangents.
5. **Use paired repeats.** Three fresh-process A/B pairs, alternating execution
   order; at least three warmups and twelve synchronized unprofiled samples per
   configuration. Report each process's median, p95 and µs/token, and all raw
   times. Extend warmup if compilation remains visible; record the actual count.
   Capture three additional traced steps separately. Preload arrays, synchronize
   outputs and keep file writes outside timing. Do not pool samples across
   processes as independent experimental replicates.
6. **Advance conditionally.** Run B=24 and the original distinct-row bank for a
   promising alternative. Then perform a full-model B=8 A/B from identical
   parameter/optimizer snapshots and the same batch. Preserve the global MoE
   workload and one optimizer update over the complete batch. Processing only
   attention rows sequentially must not silently become several smaller training
   updates. Finally confirm both base and late phases before adopting a change.

Define `S2 = attention_time(B=8) / (2 * attention_time(B=4))`, using unprofiled
forward+VJP time on the repeated-row workload. Evaluate each family separately;
report all families even if only one improves. The earlier 1.615 is a traced
whole-backbone scope ratio and is a motivation, not the isolated benchmark target.

Decision rules for each family:

- H1 receives strong support if baseline S2 ≥1.20 in all three pairs, the alternative
  reduces `(S2 - 1)` by at least half, and B=8 time/token falls by at least 10%
  in every pair, with numerical checks passing and B=4 regression no worse than 5%.
- Baseline S2 ≤1.10 in all repeats is evidence against the standalone claim for
  that family. The interval 1.10–1.20, conflicting repeats or >5% between-process
  baseline spread is inconclusive; repeat under controlled runtime conditions
  before choosing a winner.
- An isolated win becomes a useful training optimization only after the full-model
  A/B improves unprofiled step time by at least 5% reproducibly, preserves numerical
  behavior, and does not create an OOM or consume the chosen deployment headroom.
  Report absolute memory estimates/counters; do not infer a safe maximum from them.

If H1 fails, that is a useful result: investigate H2/H3 according to whether the
cliff disappeared on repeated data or only when isolated. H4 is a follow-up
mechanism test, not an explanation to assume in advance. Keep the 4-row recipe
as the current performance reference while these tests remain unrun.

## Reproduce the offline table

Download the public output using `kaggle kernels output
xiayicheng3gmailcom/nano-dsv4-1f -p /path/to/dest`. Record the output version:
the unversioned command may fetch a newer run in the future. For each selected
XPlane, export `hlo_stats` using XProf 2.23.2, in a separate analysis environment:

```python
from pathlib import Path
from xprof.convert import raw_to_tool_data

payload, content_type = raw_to_tool_data.xspace_to_tool_data(
    [str(xplane_path)], 'hlo_stats', {})
if payload is None:
    raise RuntimeError('XProf conversion failed')
if isinstance(payload, bytes):
    payload = payload.decode()
Path(f'{case}-{phase}-hlo_stats.json').write_text(payload)
```

Then run the standard-library-only extractor:

```bash
python scripts/summarize_pretrain_profile.py \
  --run-dir /path/to/pretrain-profile-20260921-110434 \
  --xprof-dir /path/to/exported-hlo-json \
  --output /path/to/compact-profile.json
```

Confirm the XPlane contains eight device traces and the three intended steps
before applying the normalization. Missing HLO exports remain null. JSON
serialization can change export hashes while preserving numeric contents; compare
numeric fields as well as hashes. XProf's internal conversion API can change,
so retain the recorded analysis version.

References: [JAX profiling](https://docs.jax.dev/en/latest/profiling.html),
[`lax.map` semantics](https://docs.jax.dev/en/latest/_autosummary/jax.lax.map.html),
[TPU pipelining](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html).
