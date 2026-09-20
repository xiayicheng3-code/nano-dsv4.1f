# Full 8K pretraining stress test

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
| narrow48 | 48 | 128 | 2 | 256 | 110,416,420 |

All preserve E×F=6144 and the routed SwiGLU weight capacity per layer
(3×D×E×F). Total parameters differ because the current config shares `d_ff` with
the always-on shared expert and DSpark FFN; router size also changes. Counts above
include allocated frozen DSpark. No backbone depth/attention dimension is reduced.

For a moderate narrowing start with 24×256; 48×128 is the stronger sparsity
candidate. Width 128 meets the current ragged kernel's tile dimensions, while
widths below 128 require padding and do not guarantee proportional speedups.
More experts mean fewer assignments per expert and potentially worse kernel
utilization. With global B=4, T=8192 and K=2, uniform routing gives about 1365
assignments per expert at E=48, versus 8192 at E=8, before packing masks.

At top-2 the narrow candidates execute much less routed FFN compute. Faster steps
cannot be interpreted as equal quality. `--top-k` can test wider selection, but
increasing K also enlarges dispatch/sort work and worst-case packed buffers.
`--experts` and `--width` expose explicit alternatives without changing defaults.

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
