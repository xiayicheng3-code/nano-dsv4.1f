# nano-dsv4.1f

An educational, TPU-first reimplementation of the *ideas* behind DeepSeek-V4.1-Flash, scaled down so the architecture can be inspected, trained, profiled, and eventually exported as a Kaggle notebook.

> This is not a drop-in reproduction of the 552B model. The goal is to preserve the interesting architectural constraints while making deliberate simplifications explicit and measurable.

## Design goals

- Keep the canonical implementation as normal Python modules, **not notebook cells**.
- Compile the maintained source tree into a Kaggle-friendly notebook later.
- Reimplement the architectural ideas that teach us something about long-context ML systems: CED, CSA2-style shared compressed KV, SWA + global attention, Single-Pass mHC, Engram-like hashed memory, MoE, FP4-KV QAT experiments, and explicit sharding.
- Keep expensive deployment/training machinery optional when it obscures the educational core.
- Record every deviation from the DeepSeek-V4.1-Flash report in `docs/implementation_scope.md`.

## Current training philosophy

The first runnable target is a dense-training approximation of CSA2 on TPU. Sparse retrieval is learned as an auxiliary objective late in pre/mid-training rather than placed on the critical path from step zero.

For an indexer configured with `retrieve_top_k = 512` and a local window of `128`, the default educational policy only distills queries whose causal history is at least `640` tokens. To keep the teacher cheap, the initial implementation selects the **latest eligible query per packed segment**. This is our approximation; it is not claimed to be DeepSeek's training recipe.

The indexer teacher uses dense main-attention logits only for those selected queries. The design keeps the dense teacher outside the ordinary SplashAttention forward path so we do not require SplashAttention to materialize a full attention-score matrix.

## Repository layout

```text
src/nano_dsv41f/
  config.py          Small-model and training-stage configuration
  packing.py         Packed-sequence metadata and query selection
  attention.py       Mask algebra + streaming softmax merge helpers
  indexer.py         Late-stage sparse-indexer distillation
  mhc.py             Educational Single-Pass mHC implementation
  model.py           Small composable model skeleton
scripts/
  build_notebook.py  Compile maintained Python sources into a notebook
notebooks/
  README.md          Notebook policy; generated notebooks are artifacts
docs/
  implementation_scope.md
  indexer_training.md
tests/
```

## Status

The repository is being built bottom-up. The first commit establishes the architecture contracts and the pieces that are easiest to test without a TPU: packing, causal-compression mask algebra, indexer query selection, streaming softmax merge, and mHC math. TPU Splash/Pallas kernels and full distributed sharding come after the reference path is numerically stable.

## References

Primary reference: DeepSeek-AI, **DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression** (2026), plus the released configuration and reference inference implementation.

This project is unaffiliated with DeepSeek.
