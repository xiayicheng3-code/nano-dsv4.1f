# nano-dsv4.1f

A TPU-first educational reimplementation of the architectural/training ideas in **DeepSeek-V4.1-Flash**, deliberately scaled so they can be read, modified, trained and profiled on a Kaggle TPU.

> This is not a checkpoint-compatible miniature of the production model. `docs/implementation_scope.md` records which mechanisms are faithful semantic references, which are software approximations, and which production systems pieces are intentionally deferred.

## Default nano architecture

DeepSeek's released inference code contains a useful **five-layer tiny topology anchor**:

```text
L0 SWA | L1 Full(r=2) -> L2 Reuse | L3 Full(r=1) -> L4 Reuse
```

That anchor is excellent for checking our scaling choices, but it never gets deep enough to demonstrate the decoder's separate **Reindex** stage. Our default therefore extends it by exactly one two-layer decoder retrieval group:

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

The key extra idea is that **L5 does not create a new KV bank**. It computes a new retrieval decision over the index-K owned by L3. By default L3 also publishes hierarchical candidate blocks, and L5 rescans only within that candidate set before producing the Top-K reused by L6.

DSpark consumes the last three backbone layer features (`L4/L5/L6`) by default, echoing the production model's use of its final three target layers while keeping the draft network itself to one educational Transformer stage.

All dimensions, group counts, RoPE settings, compression ratios, expert counts, DSpark rank/block size, optimizer constants and sharding intentions live in configuration dataclasses rather than being buried in kernels.

## Implemented V4.1 details

The readable JAX reference includes:

- **CED / CSA2:** SWA / Full / Reindex / Reuse, shared compressed-KV lifetimes, r=1/r=2 learned compression, fixed-128 local overlap and exact local/global shared-softmax LSE merge.
- **MLA-style attention:** Q low-rank bottleneck, one latent K/V shared by Q heads, learned attention sink, **partial RoPE on the last channels**, compressed positional regime, **inverse RoPE on attention output**, and **grouped low-rank `wo_a`** (`G=2` by default).
- **Sparse indexer:** Q from main `qr`, K from pre-RoPE compressed latent, partial RoPE, query-dependent head weights, ReLU head scores, cross-layer K reuse and released-style hierarchical candidate-block selection.
- **Indexer training experiment:** late-stage, latest-eligible-query distillation; all 2–3 layers served by a nano retriever can be teachers without industrial-scale replay.
- **Single-Pass mHC:** multi-stream residuals, state-conditioned pre/post/combine coefficients, Sinkhorn stream mixing and cross-sublayer timing; mHC epsilon is separate from language RMSNorm epsilon.
- **MoE:** sqrt-softplus router, selection-only correction bias, routed + shared expert, clipped SwiGLU; DSpark has an independently configured expert count.
- **Engram:** segment-safe causal n-gram hashing, conditional hashed memory and per-stream gated injection.
- **Low-precision reference:** software E2M1/E4M3/E8M0 fake quantization, compressed-KV FP4 QAT path, indexer FP4 path and optional SWA FP8 path. Packed MXFP4/Pallas storage is a later kernel milestone.
- **Hybrid optimizer reference:** inspectable AdamW / Muon / head-wise Muon / Sinkhorn-balanced parameter partitioning and explicit LR schedule.
- **DSpark:** projected selected-layer target context, `[anchor, noise, ...]` draft blocks, block-parallel attention mask, one draft Transformer layer, separate MoE, vanilla low-rank Markov correction, sequential Markov sampling helper and confidence head.

## Why the retriever teacher is still cheap

Default retrieval is 512 global candidates plus 128 local positions. We only distill a packed sequence once its query has at least 640 causal positions and select the **latest eligible query** by default. Nano retriever groups serve only two layers by default, so `all_served` and `Full + last` have the same teacher set.

The added decoder Reindex group does not make each retriever more expensive: it creates a **second two-layer retrieval lifetime**, rather than stretching one retriever across many layers.

This query-subsampling rule is our educational experiment, not a claim about DeepSeek's private training recipe.

## Source layout

```text
src/nano_dsv41f/
  config.py          Every architecture/training/sharding hyperparameter
  rope.py            partial/inverse RoPE + compressed/YaRN frequency reference
  quantization.py    software FP4/FP8 fake-QAT reference
  layers.py          small JAX primitives
  packing.py         packed-example/query-selection utilities
  attention.py       standalone mask/LSE helpers
  compression.py     learned r=1/r=2 compression
  csa2.py            SWA + compressed MLA + Full/Reindex/Reuse state machine
  indexer.py         late-stage teacher/distillation utilities
  indexer_scorer.py  released-style indexer + hierarchy reference
  mhc.py             Single-Pass mHC mechanics
  moe.py             routed + shared-expert MoE
  engram.py          conditional hashed memory
  dspark.py          one-stage released-style DSpark reference
  optimizer.py       hybrid optimizer rules and update kernels
  model.py           end-to-end backbone + DSpark wiring
scripts/
  build_notebook.py  compile maintained source into a Kaggle notebook
```

Normal Python is canonical. Notebook cells are generated views of this codebase, not a second implementation.

## Still a systems project

The current dense path establishes semantics before TPU specialization. Major next milestones are:

- SplashAttention/Pallas kernels for long-context local + compressed attention;
- packed MXFP4 cache storage with software dequantization on-chip;
- real `NamedSharding`/collectives for context, experts, vocabulary and Engram tables on TPU v5e-8;
- activation-rematerialization policy benchmarks;
- training loop/data pipeline + indexer-stage scheduling;
- sparse-aware GPU continuation only if retriever quality justifies it;
- notebook compilation and Kaggle profiling.

See `docs/implementation_scope.md` for the detailed fidelity matrix.

## References

Primary references are the DeepSeek-V4.1-Flash technical report/released configuration and inference implementation, vLLM's V4.1 implementation for deployment details, and DeepSeek's released DeepSpec DSpark code.

This project is unaffiliated with DeepSeek.
