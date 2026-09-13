# Implementation scope: DeepSeek-V4.1-Flash vs nano-dsv4.1f

This file is the project's fidelity contract. We distinguish:

1. **Released/report behavior**: directly represented in DeepSeek's V4.1 report/config/reference code or the released DeepSpec DSpark code.
2. **Nano semantic reference**: the same mechanism/dataflow at deliberately smaller dimensions/depth.
3. **Systems approximation**: numerically meaningful software reference whose production packed kernel/distributed implementation is still pending.
4. **Omitted**: intentionally outside the educational project.

The canonical source is normal JAX/Python. Generated Kaggle notebooks are artifacts.

## Default nano topology

DeepSeek's released inference code contains a useful five-layer tiny topology anchor:

```text
L0 SWA | L1 Full(r=2) -> L2 Reuse | L3 Full(r=1) -> L4 Reuse
```

We deliberately extend that anchor by one decoder retrieval group so the nano model demonstrates `Reindex` as well as `Full` and `Reuse`:

```text
context / causal encoder
  L0  SWA only
  L1  Full, r=2, KV + index source
  L2  Reuse, r=2

CED handoff
  final context representation -> generation compressed-KV source

generation / causal decoder
  L3  Full, r=1, KV + index source + candidate-block source
  L4  Reuse, r=1
  L5  Reindex, r=1, same L3 KV/index-K
  L6  Reuse, r=1

speculation
  DSpark stage 0: one block-parallel Transformer layer + own MoE + Markov + confidence
```

The extra two layers are not another KV source. L5 projects a fresh index Q / head weighting from its own representation and rescans L3's shared index-K, restricted by the candidate blocks published by L3. L6 then reuses L5's Top-K. This is the smallest default we found that makes the decoder hierarchy visible without introducing long industrial reuse spans.

All counts/ranks/windows remain config values. If topology counts are changed, related explicit layer-id hyperparameters such as `candidate_source_layer` and DSpark `target_layer_ids` should be changed with them; this is intentional so notebook experiments never hide topology choices behind implicit magic.

## Attention / CSA2 fidelity

### Implemented semantic reference

- fixed-width SWA (default 128);
- intentional raw-SWA / compressed-boundary representational overlap;
- one latent K/V vector shared by all query heads;
- Q-LoRA-like `q_a -> RMSNorm -> q_b` bottleneck;
- compressed-KV source ownership and cross-layer reuse;
- CED handoff: first generation `Full` projects its global bank from the saved final context representation while querying from its ordinary block input;
- operational `SWA`, `Full`, `Reindex`, and `Reuse` states;
- r=1 and r=2 compressed states;
- per-channel learned softmax compression for r=2;
- exact shared softmax denominator via local/global log-sum-exp merge;
- learned denominator-only attention sink per Q head;
- grouped low-rank output projection `wo_a` followed by `wo_b` (nano default `o_groups=2`);
- partial RoPE on the **last** rotary channels;
- inverse partial RoPE on the attention output before grouped `wo_a`;
- different ordinary/compressed RoPE theta settings;
- optional DeepSeek-style YaRN frequency interpolation;
- compressed K positions use raw group-start positions (`0,r,2r,...`) rather than renumbered compressed coordinates.

The dense JAX attention path is the semantic ground truth. SplashAttention/Pallas later replaces execution, not these equations.

### Intentional scale changes

Released V4.1 uses a 512-dimensional attention latent with 64 rotary channels, 64 Q heads and 8 output groups. Nano defaults use `head_dim=64`, `rope_head_dim=8`, 8 Q heads and 2 output groups. These are hyperparameters, not hard-coded ratios.

### Sparse execution status

The backbone still computes compressed/global attention densely during ordinary training. Hard Top-K gather/scatter is not on the TPU critical path yet. This is deliberate: we first train/evaluate the retriever without making XLA sparse backward the project blocker.

## Sparse indexer

### Implemented parameterization

The reference indexer follows the released V4.1 dataflow:

- index Q is projected from the main attention Q-LoRA latent `qr`;
- index K is projected from the **pre-RoPE compressed latent**, not from the main rotated cache;
- one shared index K vector is scored by multiple index Q heads;
- Q/K receive the same partial positional convention as their layer;
- per-query head mixture weights are projected from the layer hidden state;
- score is a weighted sum of `ReLU(q_h dot k)` across index heads;
- optional FP4 fake-QAT applies independently to index Q/K;
- Full/Reindex/Reuse state lifetimes are explicit.

### Cheap teacher policy (our experiment)

This part is **not claimed as DeepSeek's private training recipe**:

- indexer distillation is active only during a configurable late pre/mid-training window;
- only queries with at least `local_window + retrieve_top_k` history are eligible;
- default selects the latest eligible query per packed segment;
- all layers served by a nano retriever are teachers (normally two layers);
- teacher probability mass is reconstructed from selected main-attention QK rows plus the complete attention LSE, so local attention, compressed/global attention and the sink all compete in the denominator.

Because K is shared across main Q heads, the teacher recomputation is `Q[H,D] x K[K,D]`, not a separate K bank per head.

The new decoder Reindex group does not increase the teacher span of an individual retriever: `L3/L4` are one two-layer teacher group and `L5/L6` are another.

### Hierarchical decoder reference

The hierarchy is active in the seven-layer default:

- L3 computes ordinary index scores and max-pools them inside fixed-size candidate blocks;
- the newest reachable/partially filled block is forced to survive;
- Top-K candidate blocks are expanded back to token candidates;
- L5 computes a fresh Reindex score only inside that retained candidate set;
- L6 reuses L5's new token Top-K;
- the main compressed KV and index-K continue to be owned by L3 throughout L3-L6.

This keeps the conceptual distinction very explicit: **KV ownership is slower-changing than retrieval ownership**.

## RoPE details

`rope.py` deliberately keeps the transformations readable:

- adjacent channel pairs (`is_neox_style=False` equivalent);
- only the last `rope_head_dim` channels rotate;
- SWA-only layers use ordinary theta and no scaling;
- compressed layers use `compress_rope_theta` and optionally YaRN;
- inverse rotation is exactly the negative angle.

This is a numerical reference, not a fused kernel or checkpoint loader.

## Quantization / QAT

Implemented as differentiable software references:

- E2M1 FP4 value emulation;
- E4M3 shared-scale emulation;
- E8M0/power-of-two shared-scale emulation;
- compressed main-KV FP4 fake-QAT **after RoPE** (default 16-channel groups);
- indexer Q/K MXFP4 fake-QAT with its own configurable block/scale convention (default 32-value blocks + UE8M0-style scales);
- optional SWA FP8 fake-QAT;
- straight-through estimator for training.

Important: these functions emulate quantized values; they do **not** reproduce vLLM's specialized serving byte layouts, padding/alignment or cache packing. In particular, the production SWA cache can use mixed/specialized NoPE/RoPE layouts. The planned Pallas milestone is packed low-precision cache storage plus on-chip software dequantization inside attention tiles.

## Single-Pass mHC

Implemented:

- configurable residual-stream multiplicity;
- one state-conditioned generator for pre/post/stream-mixing coefficients;
- doubly-stochastic Sinkhorn stream matrix;
- cross-sublayer Single-Pass timing (previous FFN pre-mix -> attention, current attention pre-mix -> FFN, current FFN pre-mix -> next attention);
- separate `mhc_eps` from language RMSNorm epsilon.

Not implemented: production Mega-mHC fusion or checkpoint-identical initialization details that are not needed to understand the mechanism.

## MoE

Backbone reference preserves:

- routed experts + one shared expert;
- `sqrt(softplus(logit))` routing score;
- correction bias used for selection only;
- normalized unbiased selected weights;
- route scale;
- clipped SwiGLU expert function.

DSpark has its **own routed-expert count and Top-K**, independently configured from the backbone.

Not implemented yet: expert-parallel all-to-all/dispatch kernels and the distributed no-aux correction-bias controller.

## Engram

Implemented mechanism:

- causal n-gram hashing that never crosses packed-example boundaries;
- multiple n-gram sizes/hash heads;
- table lookup -> per-mHC-stream keys + shared value;
- normalized stream/key score -> signed square root -> sigmoid gate;
- gated shared-value injection.

Intentional simplifications:

- raw tokenizer ids instead of DeepSeek's released normalization-aware vocabulary map;
- equal disjoint hash buckets instead of exact enormous prime-sized table layout;
- small trainable tables instead of production-scale FP8 storage.

These are documented simplifications because exact huge hash tables add capacity/storage cost, not much educational architecture value.

## DSpark

The nano repo includes one released-style DSpark stage rather than generic MTP:

- concatenates configurable target-layer backbone features and projects/norms them into DSpark context;
- default targets are the final three backbone layers (`L4/L5/L6`), mirroring the full V4.1 pattern of using its final three target layers;
- constructs each draft block as `[anchor token, noise, noise, ...]`;
- draft attention sees the recent SWA target context **through the anchor** plus every draft position in the same proposal block;
- one Transformer layer with the project's V4.1 partial-RoPE / grouped-output conventions;
- a separate DSpark MoE and expert count;
- vanilla low-rank Markov head `token -> rank -> vocab`;
- teacher-forced Markov correction for training/reference forward;
- sequential Markov block sampling helper for inference experiments;
- confidence/accept-rate scalar head using draft hidden + Markov embedding;
- target embedding / vocabulary spaces are shared with the backbone rather than maintained as a second canonical vocabulary.

Default `n_layers=1` is intentional. Production V4.1 has more speculative stages; the public tiny inference topology also demonstrates a one-stage draft setup.

Not yet implemented: full speculative verification/scheduler integration, DSpark training anchor sampler/loss weighting, or cache-efficient decode kernels.

## Hybrid optimizer

`optimizer.py` is an inspectable report-faithful reference rather than an opaque optimizer wrapper.

Parameter rules:

- token embedding, prediction head and Engram table -> Sinkhorn-balanced Nesterov update;
- vector/scalar/norm/bias parameters -> AdamW;
- ordinary matrix/batched-matrix weights -> Muon;
- head-concatenated Q projections -> head-wise Muon.

The Muon implementation exposes hybrid Newton-Schulz iteration counts and coefficients, Nesterov momentum, decoupled weight decay and update-RMS scaling. Sinkhorn uses the disclosed Nesterov form, alternating row/column normalizations, final dimension scaling and gamma multiplier; the near-zero-row mask threshold remains an explicit experiment hyperparameter because the public material does not pin down every low-level constant. AdamW and LR schedule are likewise explicit config fields.

## Activation rematerialization

Policy is configured (`none`, `attention`, `block`) but the model-level `jax.remat` wrapping and TPU memory/throughput benchmark are still pending. Dense indexer-teacher work must remain outside AD/remat replay.

## Parallelism contract

We avoid one overloaded global `TP`. Config names tensor semantics:

- `vocab_shard`
- `engram_table_shard`
- `expert_shard`
- `dspark_expert_shard`
- `attention_context_shard`
- `attention_head_shard`
- `indexer_context_shard`

The same physical TPU mesh axis may represent different logical sharding dimensions in different modules. Actual `NamedSharding`/collectives are still a TPU milestone.

## Still intentionally omitted / deferred

- production 40-layer / hundreds-of-billions parameter scale;
- exact tokenizer and pretraining corpus/data mixture;
- vision encoder and multimodal projector;
- exact enormous Engram normalized-vocabulary/prime-bucket assets;
- production sparse-attention backward kernels on TPU;
- packed MXFP4/Pallas cache kernel (software-QAT reference exists);
- real context/expert/table distributed sharding and collectives;
- no-aux routing-bias distributed controller;
- production DSpark verification/server integration;
- full checkpoint compatibility and benchmark parity;
- proprietary cluster/training infrastructure.

Those omissions are deliberate: the project aims to make V4.1's architectural and training ideas small enough to read, modify, train and profile on a Kaggle TPU, not to impersonate the production stack.
