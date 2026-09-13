# Implementation scope: DeepSeek-V4.1-Flash vs nano-dsv4.1f

This document is the contract that keeps the project educational rather than cosplay.

We distinguish three classes of behavior:

1. **Reported/released DeepSeek behavior** — present in the V4.1-Flash report, released configuration, or reference inference implementation.
2. **Reimplemented idea** — the nano model preserves the architectural mechanism but scales it down.
3. **Intentional approximation/omission** — we change or omit behavior for TPU practicality, observability, or educational clarity.

## Reimplemented architectural ideas

### Causal encoder-decoder structure

We will preserve the asymmetric causal encoder/decoder idea and cross-layer reuse of compressed state, but with far fewer layers and much smaller dimensions.

### CSA2-style local + compressed/global attention

We preserve:

- a raw sliding/local attention branch;
- a compressed/global KV branch;
- cross-layer reuse of compressed KV;
- Full / Reindex / Reuse as model-level concepts;
- a sparse indexer trained to approximate useful global retrieval.

The TPU training reference initially executes the global branch densely. Sparse Top-K is **not** on the main training forward path in the first implementation.

### Compression ratios

The reference attention algebra supports `r in {1, 2}`, matching the two ratios that matter for the released V4.1 architecture.

Packed segments must be padded to a multiple of `r`. With aligned Q/K cropping, each segment satisfies

```
Q_len = r * K_len
```

which lets the packed global causal predicate be evaluated from physical packed ids:

```
same_segment && (q_id >= r * k_id)
```

without a materialized QxK mask or per-segment local-position table.

### Shared softmax denominator

Local and global attention are allowed to run as separate kernels, but they are merged exactly using each branch's log-sum-exp. This is mathematically equivalent to concatenating both attention domains before softmax.

### Single-Pass mHC

We preserve multiple residual streams, a learned pre-mix, a doubly-stochastic cross-stream mixing matrix, and a learned post distribution for branch outputs. The first reference implementation keeps these mechanics explicit and inspectable rather than hiding them in a fused kernel.

### Engram-like hashed memory and MoE

The project will include small hashed-memory and MoE modules so the sharding problem remains visible. Their tables/expert counts are intentionally tiny relative to the released model.

### FP4 compressed-KV experiment

We plan to reproduce the *idea* of QAT for compressed/global KV plus software dequantization on TPU. This is an optional kernel milestone, not a blocker for the first model training run.

## Intentional changes

### Disjoint local/global coverage for dense TPU training

The released model's fixed SWA branch can overlap with compressed representations of the same recent token region. With `r=2`, a fixed token boundary cannot always partition raw and compressed history without either overlap or a gap.

For the first dense TPU implementation we may use an alternating 127/128-token local window so the local/global boundary aligns with 2-token compression groups. This gives a clean disjoint partition and a 128-token global-Q crop.

This is **not exact V4.1 attention semantics** and must remain an ablation rather than be silently described as faithful reproduction. A fixed-128 overlapping mode should also exist for comparison.

### Sparse-aware training

We do not initially place hard Top-K sparse attention in the TPU training critical path. Dynamic Top-K, gather/scatter, and sparse backward are poor fits for the educational Kaggle TPU target.

Instead:

- train the backbone with dense compressed/global attention;
- activate indexer distillation only during a configurable late pre/mid-training interval;
- measure retrieval quality separately;
- leave sparse-aware continuation training for a later GPU experiment.

### Cheap indexer teacher queries

DeepSeek reports staged sparse-attention/indexer training, but the public report does not specify our query-subsampling rule.

Our default approximation is:

- local window = 128;
- target retrieval size = 512;
- only queries with enough causal history for `128 + 512 = 640` positions are eligible;
- select only the **latest eligible query in each packed segment** for dense teacher scoring;
- optionally use only the Full layer and the last layer served by that indexer as teachers.

This keeps teacher work fixed-shape and bounded. It is an experiment, not a claim about DeepSeek's private recipe.

### Indexer teacher implementation

SplashAttention does not expose a full token-token attention matrix. We therefore do not require it to.

For selected teacher queries, the auxiliary path recomputes only dense QK scores against the shared compressed K, using the already available full-attention LSE to recover teacher attention mass. The distillation objective can be accumulated tile-wise, so a full teacher QxK matrix never needs to survive in memory.

### Activation rematerialization

Activation checkpointing/rematerialization is an engineering choice for the nano training run, not presented as a DeepSeek architectural contribution. mHC coefficient generation is cheap enough to recompute; dense indexer teacher passes should remain outside rematerialized backward paths.

## Not currently reimplemented

- Full 552B/763B-class parameter scale or original expert count.
- DeepSeek's production training cluster topology and proprietary training stack.
- Exact tokenizer/data mixture/training corpus.
- Exact optimizer schedule unless publicly specified and useful at nano scale.
- Production sparse-attention kernels and sparse-aware backward on TPU.
- Serving-only optimizations whose purpose is unrelated to understanding the architecture.
- Exact model-quality reproduction or benchmark parity.

## Parallelism terminology

We intentionally avoid one overloaded global `TP` setting. The configuration names tensor semantics instead:

- `vocab_shard`
- `engram_table_shard`
- `expert_shard`
- `attention_context_shard`
- `attention_head_shard`
- `indexer_context_shard`

Several may reuse the same physical 8-chip TPU mesh axis at different points in the model.

## Source policy

Normal Python modules are canonical. Generated notebooks are artifacts. Any notebook-specific workaround should be pushed down into scripts/configuration rather than becoming a second implementation of the model.
