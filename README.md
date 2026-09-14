# nano-dsv4.1f

**A from-scratch JAX reimplementation of DeepSeek-V4.1-Flash ideas, scaled into a model small enough to train and profile on a Kaggle TPU v5e-8.**

Rather than only shrinking layer counts, this project keeps the architectural pieces that make V4.1 interesting—CED/CSA2 cache sharing, Full/Reindex/Reuse retrieval, partial + inverse RoPE, grouped low-rank output projection, Single-Pass mHC, routed MoE, Engram memory, DSpark speculative decoding, and the hybrid optimizer—and then asks a systems question:

> **How much of a modern frontier-model architecture can be reproduced, trained, and optimized on a free 8-chip TPU?**

> **Status — active work in progress.** The readable reference architecture, packed-LM/indexer training path, mixed-precision sharding baseline, and Kaggle notebook are implemented. Real v5e profiling, data training runs, and optimized sparse/MoE kernels are the next milestones. Interfaces and hyperparameters may still change substantially.

This is **not** a checkpoint-compatible miniature of the 552B production model, and it does not claim to reproduce DeepSeek's private training recipe. `docs/implementation_scope.md` separates released behavior, faithful semantic references, educational approximations, and intentionally deferred systems work.

## Project at a glance

| Area | What this repo implements |
| --- | --- |
| Model architecture | 7-layer nano CED/CSA2 backbone with SWA, Full, Reindex and Reuse states |
| Attention | MLA-style shared latent K/V, partial + inverse RoPE, attention sink, grouped low-rank `wo_a`, local/global shared-softmax merge |
| Sparse retrieval | Cross-layer index-K reuse, dynamic multi-head indexer scoring, hierarchical candidate blocks, late selective distillation |
| Residual / FFN | Single-Pass mHC + routed/shared-expert MoE + clipped SwiGLU |
| Memory | Engram-style hashed n-gram memory with packed-sequence-safe hashing |
| Speculation | One-stage DSpark reference with separate MoE, Markov correction and confidence prediction |
| Optimizer | Inspectable AdamW / Muon / head-wise Muon / Sinkhorn-balanced parameter rules |
| TPU systems | v5e-8 2×4 mesh, semantic `PartitionSpec`s, direct-to-shard BF16 initialization, FP32 control paths, rematerialization, compiler diagnostics |
| Notebook | Checked-in Kaggle smoke notebook plus a generator for a self-contained notebook |

**Tech:** Python · JAX · XLA/GSPMD · TPU v5e · SplashAttention/Pallas experimentation · MoE · sparse attention · mixed precision · speculative decoding

## Why I built it

Large-model architecture papers often make individual ideas look simple in isolation, while the hard part is how they interact in an executable system. This repo is an attempt to reconstruct those interactions explicitly and make them small enough to inspect:

- preserve cross-layer KV/index lifetimes instead of replacing CSA2 with ordinary attention;
- preserve mHC's multiple residual streams rather than collapsing it into a normal residual connection;
- keep optimizer behavior parameter-family-specific rather than hiding everything behind one Optax transform;
- separate dense correctness paths from hardware-specific kernels so numerical behavior can be checked before optimization;
- expose architecture, quantization, rematerialization and sharding choices as configuration instead of hard-coding one experiment;
- use actual compiler/HBM/collective measurements to decide what to optimize next.

The goal is therefore as much **ML systems / accelerator engineering** as model reproduction.

## Current status

- [x] End-to-end readable JAX backbone and causal-LM loss
- [x] CED handoff and CSA2 `SWA / Full / Reindex / Reuse` state machine
- [x] Partial/inverse RoPE, attention sink and grouped low-rank output projection
- [x] Single-Pass mHC, MoE and Engram reference implementations
- [x] One-stage DSpark forward / Markov / confidence reference
- [x] Hybrid optimizer implementation and training-phase parameter freezing
- [x] Packed-sequence-safe late indexer distillation without a full `T×K` training-time score matrix
- [x] v5e-aware GSPMD placement, BF16 payload / FP32 control policy and direct-to-shard initialization
- [x] CPU/JAX reference tests and notebook generation/validation
- [x] Checked-in Kaggle v5e smoke notebook
- [ ] First recorded Kaggle v5e-8 HBM / compile / step-time profile
- [ ] Promote SplashAttention into the backbone after numerical parity on TPU
- [ ] Efficient compressed-global sparse attention path
- [ ] Efficient expert token dispatch / all-to-all if profiling shows MoE communication is a bottleneck
- [ ] Real training dataset + pretrained tokenizer pipeline
- [ ] Training curves, retrieval diagnostics and DSpark acceptance measurements

The GitHub Actions workflow is intentionally **manual-only**; most correctness tests are run locally, while TPU-specific validation belongs in Kaggle rather than a CPU CI runner.

## Default nano architecture

DeepSeek's released inference code contains a useful **five-layer tiny topology anchor**:

```text
L0 SWA | L1 Full(r=2) -> L2 Reuse | L3 Full(r=1) -> L4 Reuse
```

That anchor never gets deep enough to demonstrate the decoder's separate **Reindex** stage, so this project adds exactly one two-layer retrieval group:

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

DSpark consumes the last three backbone layer features (`L4/L5/L6`) by default, echoing the production model's final-target-layer pattern while keeping the draft network itself to one educational Transformer stage.

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

The default follows the proposed `local_window + top_k` eligibility rule: `128 + 512 = 640` raw-token query position. A ratio-aware ablation, `local_window + r * top_k`, waits until roughly `1152` raw positions for encoder `r=2`, when a 512-entry retrieval limit actually becomes selective.

Nano retrievers serve only two layers by default, so `all_served` and `Full + last` use the same teacher set. Teacher attention is always stop-gradient. By default student inputs from the dense backbone are detached too, so the auxiliary loss trains the indexer-specific `wk/k_norm/wq_b/weights_proj` without perturbing the backbone. This is an educational/stability choice, not a claimed DeepSeek recipe, and is configurable.

## TPU v5e-8 baseline

The repository contains a real GSPMD/mixed-precision baseline rather than only sharding intentions. The target is a single-host **2×4 v5e-8** mesh:

- `jax.make_mesh((2, 4), ("x", "y"))` keeps physical topology visible;
- vocabulary rows, Engram table rows and routed experts can each reuse the full 8-chip physical mesh in their own modules;
- nano DSpark uses 4-way expert sharding because it has four routed experts by default;
- attention starts with **8-way context/Q-sequence sharding and no head TP**;
- large matrix/table payload parameters are initialized directly into final shards as **BF16**;
- norm, bias, sink, router/control vectors and optimizer accumulators remain FP32;
- base-LM and late-indexer training compile as separate static executables;
- parameters and optimizer state are donated across training steps;
- compiler diagnostics report estimated memory, cost-analysis fields and same-runtime StableHLO collective counts.

This is a **placement/compilation baseline**, not a claim that expert parallelism is already efficient. The readable MoE still performs data-dependent expert gathers, so XLA may insert costly communication around expert-sharded parameters. The first real v5e profile should determine whether MoE dispatch or attention is the larger systems bottleneck before replacing either one.

A standalone `splash.py` prototype exercises the intended MLA-friendly local-attention layout: Q rows are sequence-sharded while compact latent K/V is replicated, packed examples use Splash `SegmentIds`, and `save_residuals=True` exposes LSE for exact local/global softmax merging. It deliberately remains outside the backbone until numerical checks pass on an actual Kaggle v5e runtime.

## Try it on Kaggle

A lightweight notebook is checked in at [`notebooks/nano_dsv41f_kaggle.ipynb`](notebooks/nano_dsv41f_kaggle.ipynb). Run it from a checkout of this repository with a TPU accelerator selected.

For a single-file self-contained notebook, generate one with:

```bash
python scripts/build_notebook.py
```

The smoke notebook is intentionally hardware-first rather than dataset-first. It:

1. verifies the TPU runtime without upgrading JAX/libtpu;
2. constructs the topology-aware v5e mesh;
3. initializes BF16 payload / FP32 control parameters directly into final shards;
4. initializes sharded optimizer state and compiles base/late-indexer executables;
5. runs a `T=1024` synthetic batch so each of 8 context shards receives 128 tokens;
6. prints compiler memory/cost/collective diagnostics;
7. compares standalone sharded Splash local-MQA output/LSE against the dense reference;
8. attempts the first compiled training step.

Once this smoke test is stable, the next layer is a pretrained ~32K tokenizer plus packed real data. Token efficiency is deliberately secondary to keeping embeddings/softmax and HBM reasonable for the nano model.

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

## Next experiments

The remaining milestones are deliberately hardware/data-driven rather than cosmetic:

- run the generated notebook on a real Kaggle v5e-8 and record HBM, StableHLO collectives and steady-state step time;
- promote local SplashAttention into the backbone only after output/LSE parity is verified;
- replace compressed-global dense attention with a packed-safe Splash/Pallas path that handles rows with no global history;
- implement real token dispatch/all-to-all if profiling shows expert-sharded dynamic gather is expensive;
- implement packed MXFP4 main-KV storage with on-chip software dequantization rather than materializing dequantized KV in HBM;
- benchmark `none` / `attention` / `block` rematerialization on actual v5e HBM and throughput;
- attach a pretrained tokenizer + packed corpus and publish training curves / retrieval diagnostics;
- add DSpark training/verification metrics once the backbone run is stable.

See `docs/implementation_scope.md` for the detailed fidelity matrix.

## References

Primary references are the DeepSeek-V4.1-Flash technical report/released configuration and inference implementation, vLLM's V4.1 implementation for deployment details, JAX's TPU/Pallas/SplashAttention implementation, and DeepSeek's released DeepSpec DSpark code.

This project is unaffiliated with DeepSeek.
