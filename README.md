# nano-dsv4.1f

An educational, TPU-first reimplementation of the *ideas* behind DeepSeek-V4.1-Flash, scaled down so the architecture can be inspected, trained, profiled, and eventually exported as a Kaggle notebook.

> This is not a drop-in reproduction of the 552B model. The goal is to preserve the interesting architectural constraints while making deliberate simplifications explicit and measurable.

## Design goals

- Keep the canonical implementation as normal Python modules, **not notebook cells**.
- Compile the maintained source tree into a Kaggle-friendly notebook later.
- Reimplement the architectural ideas that teach us something about long-context ML systems: CED, CSA2-style shared compressed KV, fixed-128 SWA + global attention, Single-Pass mHC, Engram conditional memory, MoE, FP4-KV QAT experiments, and explicit sharding.
- Keep expensive deployment/training machinery optional when it obscures the educational core.
- Record every deviation from DeepSeek-V4.1-Flash in `docs/implementation_scope.md`.

## Reference architecture

The current default model is deliberately small but structurally nontrivial:

```text
context side, r=2:
  L0 Full  -> L1 Reuse -> L2 Full    -> L3 Reuse

generation side, r=1:
  L4 Full  -> L5 Reuse -> L6 Reindex -> L7 Reuse
```

The first generation-side `Full` layer replaces the shared compressed/global KV using the final context-side representation. Later `Reuse` and `Reindex` layers consume that same main-KV bank; `Reindex` represents a new retrieval decision without owning a new main-KV source.

The readable reference path uses dense global attention so we can validate the architecture before introducing sparse TPU kernels. Local and global branches run separately but are merged with their log-sum-exp values, preserving one shared softmax denominator.

The local branch stays at a fixed **128-token SWA window**. For `r=2`, a compressed group can overlap the raw local window on the compression boundary. We preserve that representational overlap instead of changing SWA to an alternating 127/128-token window.

## Indexer training philosophy

Sparse retrieval is learned as an auxiliary objective late in pre/mid-training rather than placed on the critical path from step zero.

For an indexer configured with `retrieve_top_k = 512` and a local window of `128`, the default educational policy only distills queries whose causal history is at least `640` tokens. To keep the teacher cheap, the initial implementation selects the **latest eligible query per packed segment**. This is our approximation; it is not claimed to be DeepSeek's training recipe.

The nano architecture uses only **2 served layers per retriever by default** (configurable to 2–3). The default therefore distills the indexer from **all served layers**. With a 2-layer group, this is identical to a Full+last teacher policy; a 3-layer group adds only one more dense teacher QK row for the selected query.

The indexer teacher uses dense main-attention logits only for selected queries. The design keeps the dense teacher outside the ordinary SplashAttention forward path so we do not require SplashAttention to materialize a full attention-score matrix.

## Implemented reference pieces

- **CED / CSA2 composition**: explicit context-side and generation-side state ownership, Full / Reindex / Reuse modes, cross-layer compressed-KV reuse.
- **MLA-like attention**: low-rank Q projection, one latent KV vector per position shared across query heads, fixed SWA + compressed/global branch, exact LSE merge.
- **Learned KV compression**: `r=1/2`, including per-channel learned pooling for `r=2`.
- **Single-Pass mHC**: multiple residual streams, state-conditioned pre/post/combination coefficients, Sinkhorn-normalized mixing, cross-sublayer pre-mix timing.
- **MoE**: routed experts plus one shared expert, sqrt-softplus router scores, selection-only correction bias, top-k normalized routing weights, clipped SwiGLU experts.
- **Engram**: segment-safe causal n-gram hashing, conditional table lookup, one key per mHC stream plus shared value, normalized signed-sqrt sigmoid gate.
- **Packed-sequence utilities** and the fixed-shape latest-query indexer teacher selector.
- **CPU/JIT correctness tests** for the reference composition.

## Repository layout

```text
src/nano_dsv41f/
  config.py          Nano architecture, training-stage and sharding configuration
  layers.py          Functional JAX linear / RMSNorm / embedding primitives
  packing.py         Packed-sequence metadata and teacher-query selection
  attention.py       Mask algebra + shared-softmax/LSE reference helpers
  compression.py     Learned ratio-1/2 KV compression
  csa2.py            Dense CSA2 local/global attention + shared-KV state machine
  mhc.py             Single-Pass mHC residual mechanics and coefficient generator
  moe.py             Sparse routed + shared-expert reference MoE
  engram.py          Educational conditional n-gram memory
  indexer.py         Late-stage sparse-indexer distillation utilities
  indexer_scorer.py  Dense reference scorer + retrieval metrics
  model.py           End-to-end CED/CSA2 + mHC + MoE + Engram composition
scripts/
  build_notebook.py  Compile maintained Python sources into a Kaggle notebook
notebooks/
  README.md          Notebook policy; generated notebooks are artifacts
docs/
  implementation_scope.md
  indexer_training.md
tests/
  test_packing_attention.py
  test_reference_model.py
```

## Deliberately deferred

The dense reference is the semantic ground truth, not the final TPU implementation. Still pending:

- SplashAttention/Pallas replacement for long-context local/global attention;
- context/expert/table sharding on an 8-chip TPU mesh;
- sparse Top-K on the backbone forward path and sparse-aware continuation training;
- hierarchical generation-side candidate restriction;
- MXFP4 compressed-KV QAT and on-chip software dequantization;
- activation-rematerialization wrappers and throughput/memory policy benchmarks;
- checkpoint-compatible dimensions, tokenizer/data recipe, and production serving kernels.

See `docs/implementation_scope.md` for the detailed faithful-vs-simplified contract.

## Notebook policy

Normal Python modules are canonical. `scripts/build_notebook.py` packages the maintained source into a Kaggle-friendly notebook so notebook ergonomics do not fork the model implementation.

## References

Primary reference: DeepSeek-AI, **DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression** (2026), plus the released configuration and reference inference implementation.

This project is unaffiliated with DeepSeek.
