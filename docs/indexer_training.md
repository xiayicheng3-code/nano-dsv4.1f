# Cheap late-stage indexer distillation

## Goal

Train the sparse retrieval indexer without paying for a full `T x K` scorer on every LM step.

The ordinary dense backbone runs with `compute_indexer=False`: compressed/main attention still trains normally, but index-K construction and full retriever scoring are skipped. A separate late-stage auxiliary path scores only selected query rows.

This training policy is **our educational experiment**, not a claim about DeepSeek's private recipe.

## Default policy

- local window: 128
- target retrieval size: 512
- query sampling: latest eligible query per packed segment
- cross-layer teachers: **all layers served by that retriever**
- teacher branch: always stop-gradient
- student backbone inputs: detached by default; indexer-specific projections remain trainable
- training interval: configurable late pre/mid-training window

Two eligibility rules are exposed:

```text
local_plus_topk:
  min query position = local_window + top_k

local_plus_ratio_topk:
  min query position = local_window + compression_ratio * top_k
```

The first preserves the proposed `128 + 512 = 640` rule. The second distinguishes the decoder (`r=1`, still 640) from the r=2 encoder: 512 compressed entries represent roughly 1024 raw tokens, so actual Top-512 selection pressure starts around `128 + 2*512 = 1152` raw positions.

This is deliberately an ablation knob rather than pretending one threshold is universally correct.

## Static-shape query selection

A compiled batch has a fixed maximum number of packed segments. Each segment gets exactly one query slot:

```text
query_indices: [B, num_segments]
query_valid:   [B, num_segments]
```

Short/missing segments map to a dummy physical token with `query_valid=False`. Long segments map to their latest token. No dynamic `nonzero()` or variable-sized gather is required.

Therefore student scores have shape

```text
[B, num_segments, compressed_K]
```

rather than

```text
[B, T, compressed_K].
```

## Index-K ownership

For each retrieval-sharing group:

- build index K **once** from the owning KV source's pre-RoPE compressed latent;
- a Full source uses its own index-K;
- a later Reindex (L5 in the default model) reuses L3's index-K and only has its own Q/head-weight projections.

When `detach_backbone_inputs=True`, `stop_gradient` is applied to the source latent **before** `wk`, and to selected `qr`/hidden inputs **before** the student Q/head-weight projections. Thus `wk`, `k_norm`, `wq_b`, and `weights_proj` still train while the auxiliary loss does not perturb the dense backbone.

Turning detachment off is an explicit ablation.

## Teacher scores

For one selected query and compressed candidate `j`, main attention provides per-head logits `z_hj`. The reference backbone exposes the complete shared denominator; the native path reconstructs it only for the selected queries:

```text
LSE_h = log(local + compressed/global + sink mass)
```

The unnormalized teacher mass is

```text
u_j = sum_h exp(z_hj - LSE_h)
```

The teacher uses the main attention's shared latent K, not the indexer K. Because the complete LSE is used, raw SWA, compressed/global attention, their fixed-window overlap, and the attention sink all remain competitors in the denominator.

Teacher mass is always stop-gradient. The native path gathers only the recent local window, reuses selected global logits for the denominator and mass, and constructs only `[B,Q,K]` validity. It no longer runs a second full-sequence Splash pass or retains a dense `[B,T,K]` validity mask.

## Cross-layer teacher groups

The seven-layer default derives these groups from the actual layer schedule:

```text
L1 index source -> teachers L1, L2
L3 index source -> teachers L3, L4
L5 Reindex      -> teachers L5, L6
```

Policies:

- `all_served`: default;
- `full_last`: first and last served layer;
- `full_only`: only the index-source layer (name retained for compatibility even when the source is Reindex).

For every default two-layer group, `all_served == full_last` in teacher count.

## Hierarchical Reindex

L3 is the default decoder candidate source. For the selected L3 query rows:

1. compute student index scores over legal compressed history;
2. max-pool scores inside candidate blocks;
3. force the newest reachable block to survive;
4. keep the configured top candidate blocks.

L5's selective student loss is then computed only inside that L3 candidate pool, while L5 still uses its own index-Q/head-weight parameters. No new KV or index-K bank is created.

## Distillation objective

Let `I_j` be the student indexer score and `p_j = u_j / sum(u)` the normalized teacher over the current legal/candidate pool. The cross entropy is

```text
CE(p, softmax(I))
  = logsumexp(I) - sum_j p_j I_j
  = logsumexp(I) - sum_j u_j I_j / sum_j u_j
```

The implementation keeps invalid fixed-shape query slots numerically finite before `logsumexp`, avoiding NaN gradients from all-`-inf` dummy rows.

A future TPU kernel can stream K tiles and retain only the accumulators needed for the same equation; no normalized teacher matrix has to survive.

## Interaction with rematerialization

The backbone now supports `none`, `attention`, and `block` remat policies. Selective teacher work is invoked as a separate auxiliary computation rather than being deliberately nested inside the LM remat region. This prevents a coarse checkpoint policy from accidentally replaying dense teacher rows during backward.

## Sparse-aware continuation

Hard Top-K sparse attention is still intentionally outside the TPU LM-training critical path. Once retrieval quality is measured, a GPU continuation experiment can test whether sparse-aware backward provides enough quality improvement to justify dynamic gather/scatter and custom sparse kernels.
