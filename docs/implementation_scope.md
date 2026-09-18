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

We extend that anchor by one decoder retrieval group so the nano model demonstrates `Reindex` as well as `Full` and `Reuse`:

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

L5 is not another KV source. It projects a fresh index Q / head weighting from its own representation and rescans L3's shared index-K, restricted by candidate blocks published by L3. L6 then reuses L5's retrieval decision.

All counts/ranks/windows remain config values. Explicit layer-id settings such as `candidate_source_layer` and DSpark `target_layer_ids` are also config values so notebook experiments never hide topology changes behind implicit magic.

## Attention / CSA2 fidelity

Implemented semantic reference:

- fixed-width SWA (default 128) with intentional raw/compressed boundary overlap;
- one latent K/V vector shared by all query heads;
- Q-LoRA-like `q_a -> RMSNorm -> q_b` bottleneck;
- CED handoff and cross-layer compressed-KV ownership/reuse;
- r=1/r=2 compressed states with per-channel learned pooling for r=2;
- exact shared local/global softmax denominator via log-sum-exp merge;
- denominator-only learned attention sink per Q head;
- partial RoPE on the **last** rotary channels and inverse RoPE on attention output;
- ordinary/compressed theta regimes and optional DeepSeek-style YaRN;
- compressed K positions use raw group-start positions (`0,r,2r,...`);
- grouped low-rank `wo_a -> wo_b` output projection (nano default `o_groups=2`).

The dense JAX main-attention path is the semantic ground truth. SplashAttention/Pallas later replaces execution, not these equations.

Released V4.1 uses much larger dimensions (512 latent, 64 rotary dims, 64 Q heads, 8 output groups). Nano dimensions are explicit hyperparameters, not hard-coded ratios.

## Sparse indexer

Implemented parameterization follows the released V4.1 dataflow:

- index Q from main attention Q-LoRA latent `qr`;
- index K from the **pre-RoPE compressed latent**;
- one shared index K scored by multiple index Q heads;
- matching partial positional convention;
- query-dependent head mixture weights from layer hidden state;
- weighted sum of `ReLU(q_h dot k)`;
- independent optional indexer low-precision fake-QAT;
- shared index-K ownership across decoder Reindex layers;
- hierarchical candidate-block restriction for later Reindex.

### Two indexer execution modes

Ordinary dense LM pretraining calls `apply_model(..., compute_indexer=False)`. It deliberately skips index-K construction and the full `T x K` scorer. This keeps the retriever off the critical path until its late training stage.

`compute_indexer=True` is a diagnostic/evaluation mode. It materializes the full scorer and makes the `Full -> Reuse -> Reindex -> Reuse` retrieval state machine observable, including L3 candidate blocks and L5 rescoring.

### Selective late-stage distillation (our experiment)

This is **not claimed as DeepSeek's private training recipe**. `training.py` implements:

- a configurable global-batch query budget (default 128 per retrieval group), sampled
  without replacement from all eligible real positions and refreshed each optimizer step;
- index-K built once from the owning compressed source latent;
- student scoring in fixed buffers `[B,min(query_budget,T),K]`, including masked padding slots;
- teacher mass reconstructed from selected main-attention Q/K rows and the **complete local + global + sink LSE**;
- all served layers as teachers by default (two layers per nano retrieval lifetime);
- L5 distillation over full legal history by default; L3 candidate restriction is an
  opt-in training ablation (`apply_candidate_mask=True`), with evaluation hierarchy preserved;
- teacher branch always stop-gradient;
- student backbone inputs detached by default while indexer-specific `wk/k_norm/wq_b/weights_proj` remain trainable. This gradient-isolation choice is configurable and is ours, not a report claim.

Eligibility is also explicit:

- `local_plus_topk` preserves the proposed `128 + 512 = 640` raw-position rule;
- `local_plus_ratio_topk` waits until retrieval is actually selective after compression, i.e. `128 + r*512`, which is `1152` for the r=2 encoder group.

The full indexer diagnostics and the selective training path deliberately share the same scorer equations but not the same compute schedule.

## Hierarchical decoder reference

In the seven-layer default:

- L3 scores the full legal compressed history and max-pools it into candidate blocks;
- the newest reachable/partially filled block is pinned into the candidate set;
- L5 computes its new Reindex score only inside those blocks;
- L6 reuses L5's new token selection;
- compressed main KV and index-K remain owned by L3 across L3-L6.

This makes the architectural distinction explicit: **KV ownership changes more slowly than retrieval ownership**.

## RoPE details

`rope.py` keeps the math readable:

- GPT-J/interleaved adjacent channel pairs (`is_neox_style=False` equivalent);
- only the last `rope_head_dim` channels rotate;
- SWA-only layers use ordinary theta and no extension scaling;
- compressed layers use `compress_rope_theta` and optionally YaRN;
- inverse rotation is the negative angle;
- RoPE magnitude scaling (`mscale`) is intentionally absent, matching current V4.1 serving code where it is disabled.

## Quantization / QAT

Implemented differentiable software references:

- E2M1 FP4 values;
- E4M3 shared scales;
- UE8M0 power-of-two scales with **ceil(log2)** exponent selection, matching MXFP4 kernels;
- compressed main-KV fake-QAT after RoPE (default 16-channel E4M3-scaled groups);
- indexer Q/K MXFP4-style fake-QAT (default 32-value blocks + UE8M0 scales);
- optional SWA FP8 fake-QAT;
- straight-through estimator for training.

These emulate numerical distortion; they do **not** reproduce vLLM's packed bytes, padding/alignment or specialized SWA cache layouts. Packed TPU cache storage plus on-chip Pallas dequantization remains a systems milestone.

## Single-Pass mHC

Implemented:

- configurable residual-stream multiplicity;
- state-conditioned pre/post/stream-mixing coefficients;
- doubly-stochastic Sinkhorn stream matrix;
- released cross-sublayer Single-Pass timing;
- separate `mhc_eps` from language RMSNorm epsilon.

Production Mega-mHC fusion/checkpoint-identical initialization are not goals of the readable reference.

## MoE

Backbone reference preserves routed experts + one shared expert, `sqrt(softplus(logit))` routing, selection-only correction bias, normalized unbiased selected weights, route scaling, and clipped SwiGLU. DSpark has its own routed-expert count and Top-K.

The native backend now uses dropless, static-tile expert dispatch with resident matrices, fused gate/up projection and reduce-scatter. A globally aggregated text-only no-aux correction-bias controller updates by 0.001 per step (padding excluded). The report's separate image controller and sequence-level auxiliary balance loss are not implemented. Ragged all-to-all dispatch remains deferred.

## Engram

Implemented mechanism: segment-safe causal n-gram hashing, multiple n-gram/hash heads, table lookup -> per-mHC-stream keys + shared value, normalized signed-square-root gate, and gated shared-value injection.

Intentional simplifications: raw tokenizer ids, small readable tables, and simplified hash bucket layout rather than production-scale normalized-vocabulary/prime-bucket assets.

## DSpark

The nano repo includes one released-style draft stage rather than generic MTP:

- configurable selected backbone features, defaulting to final `L4/L5/L6`;
- `[anchor token, noise, noise, ...]` draft blocks;
- recent SWA target context **through the anchor** plus all draft slots in the same proposal block;
- one draft Transformer layer with V4.1-style mHC/MoE mechanics and SWA attention;
- separate DSpark expert counts;
- vanilla low-rank Markov head `token -> rank -> vocab`;
- teacher-forced Markov correction plus sequential sampling helper;
- confidence head using draft hidden + Markov embedding.

Production speculative verification/scheduler integration and cache-efficient decode kernels remain outside this stage.

## Hybrid optimizer

`optimizer.py` exposes the hybrid parameter-family rules rather than hiding them:

- token embedding, prediction head and Engram table -> Sinkhorn-balanced Nesterov update;
- vectors/scalars/norms/biases -> AdamW;
- ordinary matrix/batched-matrix weights -> Muon;
- head-concatenated Q projections -> head-wise Muon.

Muon exposes hybrid Newton-Schulz iterations, Nesterov momentum, decay and update-RMS scaling. Sinkhorn uses the disclosed Nesterov form, 11 alternating row/column normalizations, final dimension scaling and gamma multiplier. The report section 4.2.2 pins gamma=0.18, K=11, tau=1e-3 and epsilon=1e-20. Engram learning rates are multiplied by five, and biases/scales have no AdamW decay. Other experiment constants remain explicit hyperparameters.

## Activation rematerialization

Rematerialization is now wired:

- `none`: ordinary autodiff activation retention;
- `attention`: `jax.checkpoint` around CSA2 attention while retaining MoE activations;
- `block`: checkpoint the attention+MoE block from its mHC block boundary;
- Engram lookup stays outside the block boundary;
- selective indexer teacher work is a separate auxiliary computation and is not deliberately nested inside the LM remat region.

The remaining work here is empirical TPU measurement of peak HBM and step time, not basic wiring.

## Parallelism contract

We avoid one overloaded global `TP`. Config names tensor semantics:

- `vocab_shard`
- `engram_table_shard`
- `expert_shard`
- `dspark_expert_shard`
- `attention_context_shard`
- `attention_head_shard`
- `indexer_context_shard`

The same physical TPU mesh axis may represent different logical sharding dimensions in different modules. The native backend implements NamedSharding plus context/expert collectives; real TPU performance validation remains a milestone.

## Still intentionally omitted / deferred

- production 40-layer / hundreds-of-billions parameter scale;
- exact tokenizer and pretraining corpus/data mixture;
- vision encoder and multimodal projector;
- exact enormous Engram normalized-vocabulary/prime-bucket assets;
- production sparse-attention backward kernels on TPU;
- packed MXFP4/Pallas cache kernel;
- production DSpark verification/server integration;
- full checkpoint compatibility and benchmark parity;
- proprietary cluster/training infrastructure.

Those omissions are deliberate: the project aims to make V4.1's architectural and training ideas small enough to read, modify, train and profile on a Kaggle TPU, not to impersonate the production stack.

## September 17 operator/fidelity audit

See [tpu_operator_pass.md](tpu_operator_pass.md) for pinned primary sources, the mHC formula correction, validation and explicit remaining recipe gaps.
