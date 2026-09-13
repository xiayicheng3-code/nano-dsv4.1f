# Implementation scope: DeepSeek-V4.1-Flash vs nano-dsv4.1f

This document is the contract that keeps the project educational rather than cosplay.

We distinguish three classes of behavior:

1. **Reported/released DeepSeek behavior** — present in the V4.1-Flash report, released configuration, or reference inference implementation.
2. **Reimplemented mechanism** — the nano model preserves the dataflow/architectural idea but scales it down.
3. **Intentional approximation/omission** — we change or omit behavior for TPU practicality, observability, or educational clarity.

The public model calls its two halves a **Causal Encoder-Decoder (CED)**. In our code/docs we usually say **context side** and **generation side** because both halves are causal and the usual bidirectional-encoder intuition is misleading.

## Implemented reference model

### Small CED composition

The released model has 20 context-side + 20 generation-side Transformer layers. The default nano model uses **4 + 4** layers.

The important asymmetry is preserved: the generation-side global KV bank is projected from the final context-side representation rather than regenerated independently by every generation-side layer.

Default layer state machine:

```text
context side (r=2):
  L0 Full  -> L1 Reuse -> L2 Full    -> L3 Reuse

generation side (r=1):
  L4 Full  -> L5 Reuse -> L6 Reindex -> L7 Reuse
```

The first generation-side `Full` overwrites the previous shared main-KV state using the representation handed off by the context side. `Reindex` owns a new retrieval decision but **not** a new main-KV bank.

This 4+4 schedule is intentionally much smaller than the released layer spans, while still making Full / Reindex / Reuse and CED handoff observable in one model.

### CSA2-style local + compressed/global attention

Implemented now:

- fixed-width raw sliding/local attention;
- one latent KV vector shared across query heads (MLA-like latent attention);
- learned compressed/global KV;
- cross-layer main-KV reuse;
- `Full`, `Reindex`, and `Reuse` layer metadata/state ownership;
- exact shared softmax normalization between the local and global branches using per-branch log-sum-exp.

The readable backbone executes the global branch **densely**. Hard sparse Top-K retrieval is not yet on the backbone forward path.

`Reindex` therefore has the correct main-KV ownership semantics today but is numerically equivalent to `Reuse` until the auxiliary indexer is connected to sparse execution. Keeping the distinction explicit prevents us from accidentally designing the later indexer around the wrong state lifetime.

### Fixed SWA with compression-boundary overlap

The local branch uses a fixed 128-token window by default. We deliberately removed the earlier experimental 127/128 alternating window.

For `r=2`, the dense reference puts a compressed group into the older/global domain when the **first raw token** represented by that group is at least 128 tokens behind the query. A group can therefore straddle the SWA boundary: one source token may be represented both by raw SWA KV and by the compressed latent.

Those are distinct representations and both participate in the shared softmax denominator. We preserve the overlap instead of forcing a mathematically disjoint partition.

For the future packed Splash/Pallas kernel, the same boundary can be expressed with aligned rectangular Q/K coordinates (`q >= r*k`) after the 128-token crop. The current dense reference uses explicit segment-local positions because readability matters more than avoiding a tiny test mask.

### Learned r=1 / r=2 compression

The released compressor projects each token into the latent KV dimension and, for `r>1`, produces a **per-latent-channel pooling score**. Softmax is taken across the tokens in each compression group independently for every channel.

The nano compressor now supports that channel-wise rule. `r=1` is a plain projection; `r=2` performs learned two-token pooling. Packed segment boundaries must be aligned to `r` so a compression group never spans two examples.

### MLA-like projections

The reference attention keeps two MLA ideas that matter to the systems discussion:

- a low-rank query bottleneck (`q_a -> RMSNorm -> q_b`);
- one latent KV vector per position shared by all query heads.

The output also uses a small low-rank bottleneck. We **do not yet** reproduce the released grouped/block-diagonal `wo_a` layout, partial RoPE/YaRN details, or its exact checkpoint dimensions.

### Single-Pass mHC

Implemented:

- `hc_mult` parallel residual streams (default 4);
- one state-conditioned projection that generates pre-mix, post-distribution, and stream-to-stream combination coefficients;
- Sinkhorn normalization for the combination matrix (default 20 iterations, matching the released setting);
- initial one-hot pre-mix reading residual stream zero;
- the released **cross-sublayer timing**:
  - attention consumes the previous FFN's pre-mix;
  - attention generates the pre-mix consumed by the current FFN;
  - the FFN generates the pre-mix consumed by the next block's attention.

We do not reproduce the fused Mega-mHC kernel or claim checkpoint-compatible numerical initialization for `hc_split_sinkhorn`.

### MoE

The nano MoE preserves:

- routed experts plus **one shared expert** every token passes through;
- `sqrt(softplus(router_logits))` scoring;
- correction bias used for **selection only**;
- routing weights taken from unbiased scores and normalized over selected experts;
- routed scaling factor;
- SwiGLU experts with the same asymmetric clipping idea.

Default nano scale is 8 routed experts with top-2 activation instead of the released 384 routed experts / top-6. Expert-parallel dispatch is not implemented in this reference path yet; parameter gathers keep the semantics readable.

The released no-aux bias update/training controller is not reproduced yet. The learnable/reference bias exists so its selection-only role is explicit.

### Engram conditional memory

The reference Engram now implements the mechanism rather than a generic hash embedding:

1. build several causal n-gram hashes per token;
2. look up one row per `(n-gram size, hash head)`;
3. concatenate lookup rows and project them into:
   - one key per mHC residual stream;
   - one shared value;
4. compute a normalized stream-key dot product;
5. apply the released signed-square-root transform and sigmoid gate;
6. add the gated shared value to each residual stream.

Packed examples cannot form n-grams across segment boundaries.

Intentional Engram simplifications:

- raw token ids instead of the tokenizer-derived normalization-aware compressed vocabulary;
- equal disjoint hash bucket ranges instead of the released prime-sized ranges/multipliers;
- small BF/FP reference tables instead of hundreds of millions of FP8 rows;
- two small default Engram insertion layers (`L1`, `L3`) instead of the released physical layer ids.

These preserve the conditional-memory dataflow while keeping the table inspectable and trainable on Kaggle.

## Indexer training strategy

### Nano-scale reuse groups

The released model has longer reuse spans. The nano architecture uses **2 served layers per retriever by default** (configurable to 2-3).

For a 2-layer group, `all_served` and `full_last` are the same teacher set. We therefore use `all_served` as the default rather than optimizing around the industrial six-layer case.

### Cheap late-stage teacher queries

Our educational policy, not a claimed DeepSeek recipe:

- local window = 128;
- target retrieval size = 512;
- only queries with at least `128 + 512 = 640` causal positions are eligible;
- use only the **latest eligible query per packed segment**;
- distill from all layers served by that retriever;
- activate this objective only during a configurable late pre/mid-training interval.

This makes teacher score recomputation tiny at nano layer counts while preserving the question we care about: can the indexer learn the global attention preference shared across its reuse group?

### SplashAttention teacher interface

SplashAttention does not need to expose a full attention matrix. For selected teacher queries, the auxiliary path can recompute only dense QK rows against the shared compressed K and use the normal attention LSE to recover attention mass.

The full teacher QxK matrix is never required to survive in memory.

### Sparse-aware training

Hard Top-K sparse attention is deliberately **not** in the TPU training critical path yet. Dynamic Top-K, gather/scatter, and sparse backward are deferred to a later GPU continuation experiment if retrieval quality justifies it.

## FP4 compressed-KV experiment

Planned, not implemented in the reference model yet:

- QAT/fake quantization for compressed/global KV;
- MXFP4-style storage experiment;
- software dequantization inside a TPU attention tile so expanded KV is never materialized back to HBM.

This is an optional kernel milestone, not a blocker for the first end-to-end nano training run.

## Activation rematerialization

Activation checkpointing/rematerialization is an engineering choice for the nano run, not presented as a DeepSeek architectural contribution.

The config already separates `none`, `attention`, and `block` policies, but model-level remat wrappers are not wired yet. When added, mHC coefficient generation is cheap to replay; dense indexer-teacher passes should remain outside the AD/remat replay path.

## Not currently reimplemented

- Full 552B backbone / 196B Engram scale or original expert count.
- Original 20+20 layer depth and long CSA2 reuse spans.
- Production sparse-attention kernels and sparse-aware backward on TPU.
- Hierarchical candidate-block restriction in the generation-side indexer forward.
- Exact low-rank/grouped MLA output projection and full RoPE/YaRN behavior.
- FP8/FP4 expert weights and FP4 main-KV storage in the main reference path.
- Exact Engram tokenizer normalization map, prime bucket layout, or FP8 row format.
- DeepSeek's no-aux expert-bias update controller.
- DSpark speculative decoding.
- Vision encoder / multimodal projector.
- Exact tokenizer, training corpus, optimizer schedule, or checkpoint compatibility.
- Production distributed topology and proprietary training stack.
- Exact model-quality/benchmark reproduction.

## Parallelism terminology

We intentionally avoid one overloaded global `TP` setting. The configuration names tensor semantics instead:

- `vocab_shard`
- `engram_table_shard`
- `expert_shard`
- `attention_context_shard`
- `attention_head_shard`
- `indexer_context_shard`

Several may reuse the same physical 8-chip TPU mesh axis at different points in the model. These settings currently document the intended sharding contract; actual `NamedSharding`/Pallas distributed execution is a later milestone.

## Source policy

Normal Python modules are canonical. Generated notebooks are artifacts. Any Kaggle/IPython workaround should live in scripts/configuration rather than becoming a second implementation of the model.
