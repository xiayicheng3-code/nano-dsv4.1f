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

- a fixed 128-token raw sliding/local attention branch;
- a compressed/global KV branch;
- cross-layer reuse of compressed KV;
- Full / Reindex / Reuse as model-level concepts;
- a sparse indexer trained to approximate useful global retrieval.

The TPU training reference initially executes the global branch densely. Sparse Top-K is **not** on the main training forward path in the first implementation.

### Fixed SWA with representational overlap

The released model keeps a fixed 128-token SWA branch. With `r=2` compressed global KV, the compression lattice and the raw-token SWA boundary do not always form a perfectly disjoint partition. A compressed group can contain a raw token that is also visible through SWA.

We intentionally preserve that overlap. We do **not** alternate between 127- and 128-token local windows merely to force a disjoint partition. The local and compressed representations are different KV entries, so "repeated attention" here means overlapping source-token coverage, not literally duplicating an identical key/value vector.

### Compression ratios

The reference attention algebra supports `r in {1, 2}`, matching the two ratios that matter for the released V4.1 architecture.

Packed segments must be padded to a multiple of `r`. With the 128-token global-Q crop and the corresponding `128 / r` compressed-KV tail crop, each segment satisfies

```text
Q_len = r * K_len
```

which lets the packed global causal predicate be evaluated from physical packed ids:

```text
same_segment && (q_id >= r * k_id)
```

without a materialized QxK mask or per-segment local-position table.

### Shared softmax denominator

Local and global attention are allowed to run as separate kernels, but they are merged exactly using each branch's log-sum-exp. This is mathematically equivalent to concatenating both attention domains before softmax, including when local and compressed branches cover some of the same source-token region.

### Single-Pass mHC

We preserve multiple residual streams, a learned pre-mix, a doubly-stochastic cross-stream mixing matrix, and a learned post distribution for branch outputs. The first reference implementation keeps these mechanics explicit and inspectable rather than hiding them in a fused kernel.

### Engram-like hashed memory and MoE

The project will include small hashed-memory and MoE modules so the sharding problem remains visible. Their tables/expert counts are intentionally tiny relative to the released model.

### FP4 compressed-KV experiment

We plan to reproduce the *idea* of QAT for compressed/global KV plus software dequantization on TPU. This is an optional kernel milestone, not a blocker for the first model training run.

## Intentional changes

### Nano-scale retriever reuse groups

The released model has much longer Full/Reuse spans than we need to demonstrate cross-layer index reuse. The nano model will use small retriever groups, typically **2-3 served layers per retriever**.

Because these groups are small, indexer distillation can use **all served layers** by default without turning the teacher into a six-layer replay problem. For a 2-layer group, `full_last` and `all_served` are equivalent teacher sets.

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
- distill from **all layers served by that retriever** in the default nano configuration.

The query subsampling keeps teacher work fixed-shape and bounded. Using all served layers remains affordable because nano retriever groups are deliberately only 2-3 layers. This is an experiment, not a claim about DeepSeek's private recipe.

### Indexer teacher implementation

SplashAttention does not expose a full token-token attention matrix. We therefore do not require it to.

For selected teacher queries, the auxiliary path recomputes only dense QK scores against the shared compressed K, using the already available full-attention LSE to recover teacher attention mass. The distillation objective can be accumulated tile-wise, so a full teacher QxK matrix never needs to survive in memory.

### Activation rematerialization

Activation checkpointing/rematerialization is an engineering choice for the nano training run, not presented as a DeepSeek architectural contribution. mHC coefficient generation is cheap enough to recompute; dense indexer teacher passes should remain outside rematerialized backward paths.

## Not currently reimplemented

- Full 552B-class parameter scale or original expert count.
- The released model's original number of layers per retriever reuse group.
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
