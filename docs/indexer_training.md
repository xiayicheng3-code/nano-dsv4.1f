# Cheap late-stage indexer distillation

## Goal

Teach a sparse retrieval indexer to approximate the useful mass of dense compressed/global attention without forcing dense teacher scoring for every query at every training step.

The default configuration uses:

- local window: 128
- target retrieved positions: 512
- minimum eligible local query position: 640
- query sampling: latest eligible query per packed segment
- training interval: configurable late pre/mid-training window
- default cross-layer teachers: Full layer + last served layer

The 640-query rule is **our approximation**. It is not described as DeepSeek's exact recipe.

## Static-shape query selection

A JIT-friendly batch has a fixed maximum number of packed segments. Each segment gets exactly one teacher slot:

```text
teacher_indices: [num_segments]
teacher_valid:   [num_segments]
```

A short segment maps to a dummy index plus `teacher_valid=False`; a long segment maps to its final token. This avoids dynamic `nonzero()`/variable-sized gathers.

## Teacher scores

For a selected query and compressed candidate `j`, main attention provides per-head logits `z_hj`. The normal attention forward already gives the complete local+global log-sum-exp `LSE_h`.

The unnormalized teacher mass for candidate `j` is

```text
u_j = sum_h exp(z_hj - LSE_h)
```

Using the complete LSE matters because it preserves competition with the local/SWA branch.

## Distillation objective

Let `I_j` be the indexer score and let `p_j = u_j / sum(u)` be the normalized teacher. The indexer cross entropy is

```text
CE(p, softmax(I))
  = logsumexp(I) - sum_j p_j I_j
  = logsumexp(I) - sum_j u_j I_j / sum_j u_j
```

Therefore the TPU implementation only needs tile-wise accumulators for:

- `logsumexp(I)`
- `sum(u)`
- `sum(u * I)`

No normalized teacher matrix has to be retained.

## Cross-layer teacher policy

An index set can serve several downstream layers. Recomputing dense teacher QK for every served layer is expensive. We expose three policies:

- `full_only`: cheapest sanity baseline;
- `full_last`: default; tests whether the retrieval set remains useful at the end of its reuse span;
- `all_served`: most expensive reference ablation.

The goal is to measure whether `full_last` captures nearly all of the benefit before paying for every intermediate teacher.

## Interaction with rematerialization

Teacher QK scoring is an auxiliary forward computation and should be detached from the LM-gradient rematerialization path. Normal attention projections may be recomputed during backward; dense teacher QK should not accidentally replay because an entire model block was wrapped in `jax.remat`.

## Later GPU experiment

Hard Top-K sparse-aware continuation training is intentionally deferred. Once retrieval quality is established, a GPU path can test whether sparse-aware training materially improves quality enough to justify dynamic gather/scatter and sparse backward complexity.
