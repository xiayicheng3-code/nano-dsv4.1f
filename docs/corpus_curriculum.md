# Corpus curriculum for nano-dsv4.1f

This document defines the first reproducible training-data baseline for the 4K nano model.
It assumes the 32,768-token nano tokenizer is already frozen. The objective is not to copy a
private DeepSeek data recipe; it is to build a small-model curriculum that exposes the
architecture to broad language, code, math, long-enough retrieval contexts, and a separate
reasoning-SFT view.

## Why three LM phases

The current default training schedule has two architecture-level landmarks:

- selective indexer distillation starts at progress `0.55`;
- cosine decay starts at progress `0.90`, while the default indexer auxiliary window ends
  there as well.

The corpus baseline uses three data regimes:

| Phase | Progress | FineWeb-Edu | Cosmopedia v2 | CodeParrot clean | FineMath | Packing |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| early | 0.00–0.55 | 72% | 13% | 12% | 3% 3+ | ordinary best fit |
| middle | 0.55–0.75 | 60% | 10% | 15% | 15% 3+ | ordinary best fit |
| late_mid | 0.75–1.00 | 50% | 5% | 20% | 25% 4+ | **Q-position aware** |

These are token targets, not document-count targets. The main trend is deliberate: keep web
and synthetic educational text dominant early, then increase code/math after basic language
modeling is stable. FineMath-4+ is reserved for late-mid because it is the smaller,
higher-quality slice. The numbers are a baseline for ablation, not claimed optima.

The CodeParrot source has per-file licenses. The default adapter accepts only a conservative
permissive allow-list (`MIT`, `Apache-2.0`, BSD, ISC, CC0, Unlicense) and records how many
rows were rejected. Change that list only after reviewing the terms for the corpus you plan
to redistribute/use.

With the checked-in `TrainConfig(total_steps=10_000, seq_len=4096)` and the builder's 5%
headroom, the three outputs contain 10,500 packed rows / 43,008,000 physical tokens:

```text
early      5,775 rows
middle     2,100 rows
late_mid   2,625 rows
```

The headroom is intentional: training can consume exactly its scheduled number of rows while
leaving some shuffle/restart slack.

## Document handling

Every source document is tokenized with the frozen nano tokenizer. Long documents are split
into contiguous chunks of at most `seq_len - 2` content tokens; every chunk gets nano BOS
and EOS. Chunks remain independent packed segments, so the LM never receives a next-token
target or attention history across unrelated documents/chunks.

Final packed shards store:

```text
input_ids               uint16 [rows, 4096]
segment_ids              uint16 [rows, 4096]
token_mask                uint8 [rows, 4096]
source_ids                uint8 [rows, 4096]
eligible_q               uint16 [rows]
selected_q               uint16 [rows]
q_budget_utilization     float32 [rows]
q_eligible_coverage      float32 [rows]
q_band_counts             uint16 [rows, bands]
q_expected_selected      float32 [rows, bands]
```

The existing `pack_token_sequences` implementation remains the source of truth for ratio
alignment and masked in-segment padding.

## Q-position-aware late-mid packing

For the default warmup rule, a real token at segment-local position `p` is an eligible
retriever query when

```text
p >= local_window + top_k = 128 + 512 = 640.
```

Therefore a segment with real length `L` contributes

```text
eligible_Q(L) = max(L - 640, 0).
```

A row's eligible pool is the sum across packed segments. With query budget `B=128`:

```text
selected_Q = min(B, eligible_Q)
budget_utilization = selected_Q / B
eligible_coverage = selected_Q / eligible_Q
```

Packing only for maximum LM fill can produce rows full of short documents with zero useful
Q positions. Packing only for exactly 128 eligible positions has the opposite failure mode:
it concentrates supervision around positions 640–767 and rarely teaches later retrieval.

Late-mid therefore tracks local-position bands:

```text
[640,768), [768,1024), [1024,1536), [1536,2048),
[2048,3072), [3072,4096)
```

For each candidate row, the packer computes the expected sampled Q count in each band under
the actual uniform-without-replacement query sampler. It then normalizes by band width and
prefers anchors that reduce imbalance in **expected samples per local position**, while also
penalizing under-filled query budgets. Fillers preferentially use short/non-Q-bearing
segments so an anchor's long-context profile is not accidentally swamped.

This is intentionally observable rather than magical: every shard stores the raw Q metrics,
and the phase manifest reports aggregate budget utilization, eligible coverage, expected
band samples, and per-position densities. If the heuristic is bad on the real corpus, we can
change it based on those measurements.

For the ratio-aware r=2 encoder ablation, use a 1152 threshold and matching band edges. The
current default model still uses the 640 rule for all retrievers.

## Reasoning SFT sidecar

Reasoning SFT is prepared separately from the causal-LM shards. The current LM loss masks
padding/document boundaries, but does not yet expose the assistant-only loss mask needed for
correct SFT. Keeping canonical SFT separate prevents user/tool context from accidentally
becoming prediction targets.

The initial SFT source is `open-r1/Mixture-of-Thoughts`, sampled by token budget from its
math/science/code configs:

```text
math     45%
science  30%
code     25%
```

Default target: 4,000,000 rendered tokens.

The adapter is deliberately strict for the 4K nano model:

1. require a two-turn user/assistant record;
2. require a complete explicit `<think>...</think>` span;
3. split it into `reasoning_content` and final assistant `content`;
4. use the frozen nano tokenizer for all length checks;
5. drop examples whose DeepSeek-style two-turn rendering exceeds 4096 tokens;
6. assign integer `reasoning_effort` **1..100** from tokenizer-measured reasoning-length
   percentiles, with the existing small deterministic jitter;
7. re-check length after inserting the numeric effort prefix;
8. save structured JSONL, not permanently rendered prompt strings.

The result preserves source metadata and the exact integer effort assignment provenance.
With at least 100 accepted reasoning records the assignment logic preserves coverage anchors
for every integer 1..100.

## Build commands

Install the data dependencies once:

```bash
pip install -e '.[data]'
```

Build all three LM phases from a frozen tokenizer:

```bash
python scripts/prepare_corpus.py \
  --tokenizer /path/to/tokenizer.json \
  --output-dir /kaggle/working/nano-dsv41f-corpus
```

Build only late-mid while experimenting with Q packing:

```bash
python scripts/prepare_corpus.py \
  --tokenizer /path/to/tokenizer.json \
  --output-dir /kaggle/working/nano-dsv41f-corpus \
  --phase late_mid \
  --query-budget 128 \
  --q-threshold 640
```

Build the SFT sidecar:

```bash
python scripts/prepare_reasoning_sft.py \
  --tokenizer /path/to/tokenizer.json \
  --output-dir /kaggle/working/nano-dsv41f-reasoning-sft \
  --target-tokens 4000000 \
  --max-tokens 4096
```

Both builders emit manifests containing source choices, token/row counts and the settings
needed to reproduce the data selection. The LM shards additionally contain SHA-256 hashes.

## Next measurement before training

Before treating the baseline as final, inspect the generated manifests rather than changing
weights by intuition. The useful first diagnostics are:

- actual source token fractions after packing/trimming;
- real-token packing utilization;
- late-mid mean Q-budget utilization;
- late-mid eligible-Q coverage;
- expected sampled-Q density across local-position bands;
- length and integer-effort histograms for SFT.

Those measurements tell us whether the next iteration should change source weights, document
length sampling, the Q-balance objective, or the query budget itself.
