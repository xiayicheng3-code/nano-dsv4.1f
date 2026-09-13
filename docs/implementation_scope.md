# Implementation scope: DeepSeek-V4.1-Flash vs nano-dsv4.1f

This file is the project's fidelity contract. We distinguish:

1. **Released/report behavior**: directly represented in DeepSeek's V4.1 report/config/reference code or the released DeepSpec DSpark code.
2. **Nano semantic reference**: the same mechanism/dataflow at deliberately smaller dimensions/depth.
3. **Systems approximation**: numerically meaningful software reference whose production packed kernel/distributed implementation is still pending.
4. **Omitted**: intentionally outside the educational project.

The canonical source is normal JAX/Python. Generated Kaggle notebooks are artifacts.

## Default nano topology

The default follows the *shape of the released small runnable V4.1 reference* rather than scaling the 40-layer production topology proportionally:

```text
context / causal encoder
  L0  SWA only
  L1  Full, r=2, KV + index source
  L2  Reuse, r=2

CED handoff
  final context representation -> generation compressed-KV source

generation / causal decoder
  L3  Full, r=1, KV + index source
  L4  Reuse, r=1

speculation
  DSpark stage 0: one block-parallel Transformer layer + own MoE + Markov + confidence
```

Every count/rank/window is a config value. Increasing generation depth automatically creates `Reindex` groups (`Full -> Reuse -> Reindex -> Reuse ...`) without changing block code.

## Attention / CSA2 fidelity

### Implemented semantic reference

- fixed-width SWA (default 128);
- intentional raw-SWA / compressed-boundary representational overlap;
- one latent K/V vector shared by all query heads;
- Q-LoRA-like `q_a -> RMSNorm -> q_b` bottleneck;
- compressed-KV source ownership and cross-layer reuse;
- CED handoff: first generation `Full` projects its global bank from the saved final context representation while querying from its ordinary block input;
- `SWA`, `Full`, `Reuse`, and configurable `Reindex` states;
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

The reference indexer now follows the released V4.1 dataflow:

- index Q is projected from the main attention Q-LoRA latent `qr`;
- index K is projected from the **pre-RoPE compressed latent**, not from the main rotated cache;
- one shared index K vector is scored by multiple index Q heads;
- Q/K receive the same partial positional convention as their layer;
- per-query head mixture weights are projected from the layer hidden state;
- score is a weighted sum of `ReLU(q_h dot k)` across index heads;
- optional FP4 fake-QAT applies independently to index Q/K;
- source/reuse state lifetimes are explicit.

### Cheap teacher policy (our experiment)

This part is **not claimed as DeepSeek's private training recipe**:

- indexer distillation is active only during a configurable late pre/mid-training window;
- only queries with at least `local_window + retrieve_top_k` history are eligible;
- default selects the latest eligible query per packed segment;
- all layers served by a nano retriever are teachers (normally only 2-3 layers);
- teacher probability mass is reconstructed from selected main-attention QK rows plus the complete attention LSE, so local attention and the sink still compete in the denominator.

Because K is shared across main Q heads, the teacher recomputation is `Q[H,D] x K[K,D]`, not a separate K bank per head.

### Hierarchical decoder reference

`select_candidate_blocks` implements the released hierarchy semantics for deeper decoder experiments:

- max-pool token scores inside fixed-size candidate blocks;
- force the newest reachable/partially filled block to survive;
- Top-K blocks;
- expand the selected blocks back to token candidates;
- later Reindex scores can be masked to this candidate set.

The default 5-layer nano model has no second decoder Reindex group, so this code is present but inactive unless generation depth is increased.

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
- compressed main-KV FP4 fake-QAT **after RoPE** (default block 16);
- indexer Q/K FP4 fake-QAT on their own projections;
- optional block-scaled E4M3 SWA fake-QAT;
- straight-through estimator for training.

Important: these functions emulate quantized values; they do **not** pack nibbles/bytes or provide the final TPU memory-bandwidth saving. The planned Pallas milestone is packed MXFP4 cache storage plus on-chip software dequantization inside attention tiles.

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

The nano repo now includes one released-style DSpark stage rather than generic MTP:

- concatenates configurable target-layer backbone features and projects/norms them into DSpark context;
- constructs each draft block as `[anchor token, noise, noise, ...]`;
- Transformer draft queries see target context strictly before their anchor **plus every draft position in their own block**, matching DeepSpec's block-parallel mask;
- one Transformer layer with the project's V4.1 partial-RoPE / grouped-output conventions;
- a separate DSpark MoE and expert count;
- vanilla low-rank Markov head `token -> rank -> vocab`;
- teacher-forced Markov correction for training/reference forward;
- sequential Markov block sampling helper for inference experiments;
- confidence/accept-rate scalar head using draft hidden + Markov embedding;
- target embedding / vocabulary spaces are shared with the backbone rather than maintained as a second canonical vocabulary.

Default `n_layers=1` is intentional. Production V4.1 has more speculative stages; the public small runnable reference also demonstrates a one-stage setup.

Not yet implemented: full speculative verification/scheduler integration, DSpark training anchor sampler/loss weighting, or cache-efficient decode kernels.

## Hybrid optimizer

`optimizer.py` is an inspectable report-faithful reference rather than an opaque optimizer wrapper.

Parameter rules:

- token embedding, prediction head and Engram table -> Sinkhorn-balanced Nesterov update;
- vector/scalar/norm/bias parameters -> AdamW;
- ordinary matrix/batched-matrix weights -> Muon;
- head-concatenated Q projections -> head-wise Muon.

The Muon implementation exposes hybrid Newton-Schulz iteration counts and coefficients, Nesterov momentum, decoupled weight decay and update-RMS scaling. Sinkhorn exposes its iteration count, momentum, update scaling and near-zero-row mask threshold. AdamW and LR schedule are likewise explicit config fields.

**Caveat:** where the public report/release does not fully specify a low-level training constant (for example a masking threshold or cluster-specific scheduling choice), this repo exposes the chosen value as a hyperparameter rather than claiming checkpoint-identical optimization.

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
