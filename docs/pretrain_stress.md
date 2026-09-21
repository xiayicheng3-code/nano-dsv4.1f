# Full 8K pretraining stress test

Measured results and the preregistered next experiment are in the
[2026-09-21 profiling note](experiments/2026-09-21-pretrain-profile.md).

The controlled follow-up is now implemented in
`notebooks/nano_dsv41f_attention_hypothesis.ipynb`; see the
[attention experiment guide](attention_hypothesis.md). Physical TPU results for
that intervention are still pending.

## Real-corpus profiling follow-up

Use `notebooks/nano_dsv41f_pretrain_profile.ipynb` for the 4/8/24-row throughput
investigation. It runs narrow48/top-4, attention CP2/DP4 and MoE EP8 with the same
seven-layer training recipe. The existing stress/capacity notebook remains available.
Regenerate the profiling notebook with `python scripts/build_profile_notebook.py`.

Attach the two Kaggle datasets. The notebook's first cell sets these defaults:

| Environment variable | Default | Meaning |
|---|---|---|
| `NANO_DSV41F_REF` | `codex/pretrain-stress-8k` | Code branch, before bootstrap |
| `NANO_PROFILE_CORPUS` | `/kaggle/input/datasets/xiayicheng3gmailcom/nanodsv4-1f-pretrain-tokenized` | Dataset mount; nested corpus root is discovered |
| `NANO_PROFILE_TOKENIZER` | `/kaggle/input/datasets/xiayicheng3gmailcom/nano-dsv41f-tokenizer-fineweb` | Tokenizer directory or tokenizer.json |
| `NANO_PROFILE_ROWS` | `4,8,24` | Global rows, positive multiples of DP4 |
| `NANO_PROFILE_DATA_BATCHES` | `3` | Number of sampled batches cycled on device |
| `NANO_PROFILE_DATA_SEED` | `1701` | Corpus sampling seed |
| `NANO_PROFILE_MODEL_SEED` | `7` | Model initialization seed |
| `NANO_PROFILE_WARMUP` | `3` | Excluded warmup steps per phase |
| `NANO_PROFILE_STEPS` | `12` | Unprofiled measured steps per phase |
| `NANO_PROFILE_TRACE_STEPS` | `3` | Additional profiled steps; 0 disables traces |
| `NANO_PROFILE_PHASE` | `both` | `both`, `base`, or `late` |
| `NANO_PROFILE_TIMEOUT` | `3600` | Timeout seconds per isolated worker |

No extra XLA/TPU environment flags are required. The canonical bootstrap selects
`JAX_PLATFORMS=tpu` and installs the pinned stack. Keep the notebook process free
of a JAX client; preprocessing and training each run in child processes.

The sampler understands `nano-dsv41f-pretrain-compact-v1`: memory-mapped uint16
tokens plus per-document lengths and row offsets. It validates the completed
manifest, tokenizer hash/vocabulary/special IDs, segment lengths, BOS/EOS and PAD.
It samples global training rows uniformly without replacement across shards,
without reading the full corpus into RAM. Each smaller batch takes a nested prefix
of the same maximum-size sampled batch. Input NPZs, sample provenance and hashes
are saved. Every worker preloads its small batch bank on device; transfer and disk
time are excluded, while the bank's device residency contributes to HBM usage.

Timing precedes trace capture. Median/p95, seconds/row and microseconds/physical
token exclude both warmup and profiling. LM throughput is the median of per-step
valid-target throughput because different packed batches can have different masks.
Per-step routing reports retain [layer, expert] real-token and physical-dispatch
counts, coefficient of variation, idle experts, chip loads/imbalance and packed
buffer utilization. Physical dispatch includes padding in the current MoE kernel.
Six consecutive experts belong to each EP chip in narrow48. Source scopes identify
backbone layers and MoE gather, sort and combine operations in HLO/profiler views.

Both phases use initialized/briefly updated weights. Late schedule step 6000 does
not restore a trained checkpoint. These measurements describe early routing on
real text, not mature specialization or worst-case OOM capacity. Packed corpus
attention workloads can also differ from synthetic full-document rows.

Download the final ZIP: it includes summary/per-step JSON, worker logs, optimized
HLO, XPlane traces and sampled input provenance. In a separate analysis environment,
install `xprof`, then run `xprof --logdir /path/to/extracted/traces --port 8791`.
Use HLO Op Profile/Stats, roofline and trace views to compare operation time per
token across shapes. Confirm TPU events are present; a host-only trace cannot
establish a device bottleneck. Profiling export failures preserve measured timing
but mark the case failed. Source scopes can be fused and should not be interpreted
as independent exclusive times. See the [JAX profiling guide](https://docs.jax.dev/en/latest/profiling.html).

Run `notebooks/nano_dsv41f_pretrain_stress.ipynb` on a fresh Kaggle TPU v5e-8
session with Internet enabled. Its first cell selects the implementation branch;
the bootstrap prints the resolved commit and installs the existing pinned runtime.
The notebook can be imported directly from its GitHub file URL.

## Exact configuration and scope

`nano_dsv41f.pretrain_recipe.pretrain_recipe()` is the shared factory for the
stress run and subsequent pretraining. It keeps `ModelConfig`'s full seven-layer
CED/CSA2 topology, d_model=512, eight attention heads of width 64, Q/O ranks 128,
G=2, SWA=128, top-512 retrieval, mHC=4, Engram, and 32,768 vocabulary.
It explicitly selects block remat, the frozen tokenizer's PAD/noise IDs,
Tokamax Mosaic, and 8192-token rows. The default learning-rate schedule and hybrid
optimizer are retained. Reports serialize all nested configs and a SHA-256 digest.

Base steps start at the end of LR warmup (step 500); late steps start at step 6000.
This changes the schedule input, not the history of the initialized checkpoint.
The late step includes the actual auxiliary indexer objective with the global
128-query budget per retriever and unrestricted legal history. Candidate masking
stays off. The corrected 1024-position SFT candidate pool is inherited from PR #4.
QAT stays off. DSpark parameters and optimizer state remain allocated/frozen;
DSpark's independent draft objective is not part of the current pretraining step.

This is a synthetic workload by default, not a learning-quality experiment.
It reuses deterministic random token rows across steps; weights, router correction
biases, optimizer state and sampled indexer queries evolve normally. Optional NPZ
input accepts the data pipeline's `input_ids`, `segment_ids`, `token_mask` arrays.
Disk I/O and tokenization occur before timing. Four global rows mean 32,768 physical
tokens per optimizer step, so adopting this batch for training also changes the
total tokens needed by the current 10,000-step corpus schedule.

## Attention layout

| Attention CP | Attention DP | Devices | Local Q length at 8K | Local rows at global B=4 |
|---:|---:|---:|---:|---:|
| 8 | 1 | 8 | 1024 | 4 |
| 4 | 2 | 8 | 2048 | 2 |
| 2 | 4 | 8 | 4096 | 1 |

All local Q lengths meet the current 128-row Splash tile requirement. The sequence
length alone does not require CP8. At a fixed global batch these three layouts
have the same local Q-token count; lower CP changes per-row Q length, KV replication,
and communication, not simply the amount of local query work. Splash's latent KV is
replicated within each CP group and batch-sharded across DP groups. This is the existing
replicated-KV attention strategy, not a ring-attention implementation.

Previously `_splash_runner` flattened every mesh into CP8 and ignored smaller
configured context factors. It now explicitly maps batch and query dimensions to
disjoint physical DP/CP axes. Native attention rejects unsupported head sharding or
CP×DP mismatches instead of reporting a layout the kernel does not actually execute.

CP2/DP2 accounts for four devices. Using all eight would require another factor of
two (for example attention head parallelism, not implemented here), or DP4. MoE's
EP8 is a **different layout of the same eight devices**; it does not multiply the
attention device count. It gathers activations across its flat eight-device mesh
and keeps expert weights resident. Attention/MoE reshards remain in the measured
complete step. Lower attention CP does not reduce the current MoE's global dispatch
buffer or introduce a new MoE communication algorithm.

## Expert candidates

| Profile | Routed experts | Width | Top-k | Routed width per token | Total allocated parameters |
|---|---:|---:|---:|---:|---:|
| baseline | 8 | 768 | 2 | 1536 | 122,069,260 |
| narrow24 | 24 | 256 | 2 | 512 | 112,689,532 |
| narrow48 | 48 | 128 | 4 | 512 | 110,416,420 |

All preserve E×F=6144 and the routed SwiGLU weight capacity per layer
(3×D×E×F). Total parameters differ because the current config shares `d_ff` with
the always-on shared expert and DSpark FFN; router size also changes. Counts above
include allocated frozen DSpark. No backbone depth/attention dimension is reduced.

For a moderate narrowing start with 24×256; 48×128 is the stronger sparsity
candidate. Width 128 meets the current ragged kernel's tile dimensions, while
widths below 128 require padding and do not guarantee proportional speedups.
More experts mean fewer assignments per expert and potentially worse kernel
utilization. With global B=4 and T=8192, uniform routing gives about 2731 assignments per
expert at E=48/K=4, versus 8192 at E=8/K=2, before packing masks.

The narrow48 profile now defaults to top-4, matching narrow24/top-2 in routed
active width (512) and routed weight capacity. Their shared-expert width and
routing/dispatch costs still differ, so this is not an equal-total-compute comparison.
Both execute less routed FFN work than the baseline. Faster steps cannot establish
equal learning quality. Override `--top-k 2` to repeat the old narrow48 experiment.

The current EP adapter allocates `B*T*min(K, local_experts)` packed rows per chip.
At E=48/EP8 there are six local experts, so top-4 uses `4*B*T` packed rows versus
`2*B*T` for top-2. This doubles those dispatch buffers, not the entire model HBM
footprint. Re-measure capacity after changing K. `--experts` and `--width` expose
explicit alternatives without changing the baseline.

## Measurement and failures

Each case owns a fresh process. Within `phase=both`, retain the base executable
while compiling/running the late executable to expose transition HBM pressure.
Record two warmup steps and ten measured steps by default. Both parameters and
optimizer state are donated by the existing compiler wrapper. Synchronize the
complete output tree before stopping the timer, following the
[JAX benchmarking guidance](https://docs.jax.dev/en/latest/benchmarking.html).
The reported compile time is explicit lowering/AOT compilation; any additional
first-call dispatch compilation appears in the excluded warmup timings.

Compiler memory is arguments + outputs + temporaries - aliases. It is an executable
estimate, not a total observed peak. Reports also record all available device
memory counters after initialization/compile/steps, host peak RSS, and comparison
with the [published 16 GB per-chip capacity](https://docs.cloud.google.com/tpu/docs/v5e).
Some runtimes expose no device counters. Multiple resident executables, allocator
overhead, and runtime buffers can consume memory beyond the individual estimate.

The worker atomically saves a report before every risky stage and after each
completed step. The supervisor records exit signals/timeouts and preserves logs,
then continues with the next case. Host or VM death can prevent the supervisor
from finishing; the last saved worker stage remains useful. Neither a signal nor
a timeout alone proves an HBM OOM. A passed short run establishes only that those
steps fit. Use `routing=skewed` to force every token into the final K experts and
`layout=packed` to exercise odd segment lengths and padding for the selected recipe.

## Validation

CPU tests compare CP2/DP4 and CP4/DP2 Splash forward/backward with the independent
dense oracle, including packed segments, r=1/r=2 global KV, and the attention sink.
A full seven-layer tiny-dimension late optimizer test checks interaction with EP8,
global query budgets and parameter/optimizer updates. These tests validate semantics;
only the Kaggle notebook can establish physical TPU compile time, step time, and HBM.

```bash
XLA_FLAGS=--xla_force_host_platform_device_count=8 python -m pytest \
  tests/test_attention_parallelism.py tests/test_pretrain_stress.py -q
python scripts/build_stress_notebook.py
```


## Automatic row-capacity search

The final notebook section enables a bounded search for one selected model/layout.
It is opt-in because every batch size needs a fresh full-model compilation. Default
search candidate: 48×128/top-4, CP2/DP4, start B=4, cap B=32, both phases, long rows.
The cap is user-configurable and is not a predicted TPU capacity.

The supervisor's `--capacity-max-rows N` flag doubles batch size until an explicit
HBM/device-memory failure or the cap, then bisects in DP-sized increments. Starting
above the fitting range also searches down to the smallest legal microbatch. A
passing cap reports `cap_reached`; it does not claim a maximum. Adjacent observed
pass/OOM sizes report `boundary_observed`. Non-memory failures, host kills, VMEM
kernel errors and timeouts stop with `inconclusive`, preserving prior evidence.
The method assumes fit is monotonic with rows for this workload; shape-dependent
compiler choices can violate that assumption. Every attempted row count is saved.

A fixed-row test alone establishes that row count fits, not an accurate maximum.
The automatic search measures a boundary under the tested runtime, remat, precision,
packing, routing, objectives and executable-retention policy. A short synthetic run
cannot guarantee the same boundary for all trained router distributions, checkpoint
or evaluation buffers, or longer runs. Validate the chosen batch with more steps,
representative packed data and forced skew. Keep the training microbatch below the
observed failure boundary with measured headroom. Compare tokens/s separately:
maximum fitting rows and fastest rows per second need not coincide.

All row counts are global microbatch rows. Gradient accumulation increases the
number of rows per optimizer update without retaining every microbatch's forward
activations simultaneously; its accumulator buffers still require memory. This
stress worker performs one microbatch per optimizer update and does not benchmark
gradient accumulation.
