# nano-dsv4.1f

A TPU-first educational reimplementation of the architectural/training ideas in **DeepSeek-V4.1-Flash**, deliberately scaled so they can be read, modified, trained and profiled on a Kaggle TPU.

> This is not a checkpoint-compatible miniature of the production model. `docs/implementation_scope.md` records which mechanisms are faithful semantic references, which are software approximations, and which production systems pieces are intentionally deferred.

## Default nano architecture

DeepSeek's released inference code contains a useful **five-layer tiny topology anchor**:

```text
L0 SWA | L1 Full(r=2) -> L2 Reuse | L3 Full(r=1) -> L4 Reuse
```

That anchor never gets deep enough to demonstrate the decoder's separate **Reindex** stage, so our default adds exactly one two-layer retrieval group:

```text
context / causal encoder
  L0  SWA only
  L1  Full(r=2, KV+index source) -> L2 Reuse

CED handoff
  final context representation -> generation KV source

generation / causal decoder
  L3  Full(r=1, KV+index source, candidate source) -> L4 Reuse
  L5  Reindex(r=1, same KV/index-K)                 -> L6 Reuse

speculation
  DSpark: 1 block-parallel Transformer layer + separate MoE + Markov + confidence
```

L5 deliberately does **not** create a new KV bank. It changes the retrieval decision over the index-K owned by L3. L3 also publishes hierarchical candidate blocks, and L5 rescans that candidate set before L6 reuses its new selection.

DSpark consumes the last three backbone layer features (`L4/L5/L6`) by default, echoing the production model's use of its final three target layers while keeping the draft network itself to one educational Transformer stage.

All dimensions, group counts, RoPE settings, compression ratios, expert counts, DSpark rank/block size, optimizer constants, rematerialization policy and sharding intentions live in configuration dataclasses rather than being buried in kernels.

## Implemented V4.1 details

The readable JAX reference includes:

- **CED / CSA2:** SWA / Full / Reindex / Reuse, shared compressed-KV lifetimes, r=1/r=2 learned compression, fixed-128 local overlap and exact local/global shared-softmax LSE merge.
- **MLA-style attention:** Q low-rank bottleneck, one latent K/V shared by Q heads, learned attention sink, partial RoPE on the last channels, compressed positional regime, inverse RoPE on attention output, and grouped low-rank `wo_a` (`G=2` by default).
- **Sparse indexer:** Q from main `qr`, K from pre-RoPE compressed latent, partial RoPE, query-dependent head weights, ReLU head scores, cross-layer K reuse and released-style hierarchical candidate-block selection.
- **Single-Pass mHC:** multi-stream residuals, state-conditioned pre/post/combine coefficients, Sinkhorn stream mixing and cross-sublayer timing; mHC epsilon is separate from language RMSNorm epsilon.
- **MoE:** sqrt-softplus router, selection-only correction bias, routed + shared expert, clipped SwiGLU; DSpark has an independently configured expert count.
- **Engram:** segment-safe causal n-gram hashing, conditional hashed memory and per-stream gated injection.
- **Low-precision reference:** software E2M1/E4M3/UE8M0 fake quantization, compressed-KV FP4 QAT path, indexer MXFP4-style path and optional SWA FP8 path. Packed MXFP4/Pallas storage is a later kernel milestone.
- **Hybrid optimizer reference:** inspectable AdamW / Muon / head-wise Muon / Sinkhorn-balanced parameter partitioning and explicit LR schedule.
- **DSpark:** projected selected-layer target context, `[anchor, noise, ...]` draft blocks, block-parallel attention mask, one draft Transformer layer, separate MoE, vanilla low-rank Markov correction, sequential Markov sampling helper and confidence head.
- **Activation rematerialization:** configurable `none`, `attention`, and `block` policies wired with `jax.checkpoint`.

## Cheap late-stage indexer training

The ordinary dense LM forward intentionally runs with **no full indexer scoring**: it does not build index-K or materialize a `T x K` retrieval matrix. `compute_indexer=True` exists only as a diagnostic/evaluation path for checking Full/Reindex/Reuse semantics.

Late in pre/mid-training, `training.py` runs a separate selective auxiliary objective:

```text
for each index source (L1, L3, L5):
  one fixed query slot per packed segment
  -> build shared index-K once
  -> score only selected student Q rows
  -> reconstruct teacher mass from served layers' main Q/K + complete LSE
  -> CE over legal candidates
```

The default follows the proposed `local_window + top_k` eligibility rule: `128 + 512 = 640` raw-token query position. We also expose a ratio-aware ablation, `local_window + r * top_k`; for the encoder's `r=2`, that waits until roughly `1152` raw positions, when a 512-entry retrieval limit actually becomes selective.

Nano retrievers serve only two layers by default, so `all_served` and `Full + last` use the same teacher set. Teacher attention is always stop-gradient. By default student inputs from the dense backbone are detached too, so the auxiliary loss trains the indexer-specific `wk/k_norm/wq_b/weights_proj` without perturbing the backbone; this is our educational/stability choice, not a claimed DeepSeek recipe, and it is configurable.

## TPU v5e-8 baseline

The repository now contains a real GSPMD/mixed-precision baseline rather than only sharding intentions. The target is a single-host **2×4 v5e-8** mesh:

- `jax.make_mesh((2, 4), ("x", "y"))` keeps physical topology visible;
- vocabulary rows, Engram table rows and routed experts can each reuse the full 8-chip physical mesh in their own modules;
- nano DSpark uses 4-way expert sharding because it has four routed experts by default;
- attention starts with **8-way context/Q-sequence sharding and no head TP**;
- large matrix/table payload parameters are initialized directly into their final shards as **BF16**;
- norm, bias, sink, router/control vectors and optimizer accumulators remain FP32;
- base-LM and late-indexer training are compiled as separate static executables;
- parameters and optimizer state are donated across training steps;
- compiler diagnostics report estimated memory, cost-analysis fields and same-runtime StableHLO collective counts.

This is a **placement/compilation baseline**, not a claim that expert parallelism is already efficient. The current readable MoE still performs data-dependent expert gathers, so XLA may insert costly communication around expert-sharded parameters. The first v5e profiling run should determine whether MoE dispatch or attention is the larger systems bottleneck before replacing either one.

A standalone `splash.py` prototype also exercises the intended MLA-friendly local attention layout: Q rows are sequence-sharded while the compact latent K/V is replicated, packed examples use Splash `SegmentIds`, and `save_residuals=True` exposes LSE for the exact local/global softmax merge. It deliberately remains outside the backbone until it passes numerical checks on an actual Kaggle v5e runtime.

## Kaggle notebook workflow

`scripts/build_notebook.py` generates a self-contained Kaggle notebook from the maintained Python sources. The generated notebook:

1. checks that the selected accelerator is TPU before modifying the environment;
2. **does not upgrade JAX/libtpu** and installs this package with `--no-deps`;
3. constructs the topology-aware v5e mesh;
4. initializes BF16 payload / FP32 control parameters directly into final shards;
5. initializes sharded optimizer state and compiles base/late-indexer executables;
6. uses a `T=1024` smoke batch so 8-way context sharding gives 128 Q tokens per chip;
7. prints compiler memory/cost/collective diagnostics;
8. compares the standalone sharded Splash local-MQA output and LSE against the dense reference before it is allowed into the backbone.

Normal Python remains canonical; notebook cells are generated views, not a second implementation. CI also generates and validates the notebook structure so the Kaggle artifact cannot silently drift away from the package.

## Source layout

```text
src/nano_dsv41f/
  config.py          architecture/training/sharding hyperparameters
  rope.py            partial/inverse RoPE + compressed/YaRN frequency reference
  quantization.py    software FP4/FP8 fake-QAT reference
  layers.py          small JAX primitives + dtype-preserving linear/RMSNorm
  packing.py         packed-example/query-selection utilities
  attention.py       standalone mask/LSE helpers
  compression.py     learned r=1/r=2 compression with FP32 control math
  csa2.py            SWA + compressed MLA + optional Full/Reindex/Reuse diagnostics
  indexer.py         fixed-shape teacher selection + distillation math
  indexer_scorer.py  released-style indexer + hierarchy reference
  training.py        packed causal LM + selective late-stage indexer objective
  mhc.py             Single-Pass mHC mechanics + BF16 payload boundaries
  moe.py             routed + shared-expert MoE, FP32 routing/BF16 expert payload
  engram.py          conditional hashed memory, FP32 gate/BF16 payload
  dspark.py          one-stage released-style DSpark reference
  optimizer.py       hybrid optimizer rules and update kernels
  model.py           end-to-end backbone + DSpark wiring + remat boundaries
  tpu.py             v5e mesh, PartitionSpec, direct-to-shard init and JIT helpers
  precision.py       BF16 payload / FP32 control parameter policy
  profiling.py       AOT memory/cost/collective diagnostics
  splash.py          standalone sharded Splash local-MQA prototype
scripts/
  build_notebook.py  generate the self-contained Kaggle TPU notebook
```

## Still a systems project

The next milestones are deliberately hardware-driven rather than cosmetic:

- run the generated notebook on a real Kaggle v5e-8 and record HBM, StableHLO collectives and steady-state step time;
- promote local SplashAttention into the backbone only after output/LSE parity is verified;
- replace compressed-global dense attention with a packed-safe Splash/Pallas path that handles rows with no global history;
- implement real token dispatch/all-to-all if profiler evidence shows the current expert-sharded dynamic gather is expensive;
- implement packed MXFP4 main-KV storage with on-chip software dequantization rather than materializing dequantized KV in HBM;
- benchmark `none` / `attention` / `block` rematerialization on actual v5e HBM and throughput;
- add the DSpark training/verification path and sparse-aware GPU continuation only when those experiments become useful.

See `docs/implementation_scope.md` for the detailed fidelity matrix.

## References

Primary references are the DeepSeek-V4.1-Flash technical report/released configuration and inference implementation, vLLM's V4.1 implementation for deployment details, JAX's TPU/Pallas/SplashAttention implementation, and DeepSeek's released DeepSpec DSpark code.

This project is unaffiliated with DeepSeek.
