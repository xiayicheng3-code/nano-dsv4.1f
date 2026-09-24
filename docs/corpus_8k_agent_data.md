# 8K mid-training and SFT data

This is the CPU/Kaggle preparation path for stages **2 and 3** of the canonical lifecycle:

```text
pretrain  ->  midtrain  ->  sft
```

General 3B-token pretraining is prepared separately. This pipeline produces three physical pools:

```text
document/   curated causal-LM text/code/math for mid-training
reasoning/  structured reasoning traces
agent/      reasoning -> tool -> observation -> ... trajectories
```

The trace pools are deliberately reusable. Mid-training consumes their ordinary causal-LM view; SFT consumes the assistant-only `sft_loss_mask` view.

All packed rows are 8192 tokens. Token IDs remain `uint16`; masks/source IDs/effort IDs use compact integer types. Shards are uncompressed NPZ by default because DEFLATE is usually wasted CPU for this Kaggle preprocessing job.

## Frozen tokenizer

The Kaggle notebook resolves the already-built tokenizer from:

```text
xiayicheng3gmailcom/nano-dsv41f-tokenizer-fineweb
```

The builders verify every reserved nano/DeepSeek protocol token ID before writing shards.

## Mid-training document pool

The old `middle` and `late_mid` document phases are replaced by one explicit mid-training corpus.

Starting token mix:

| source | share | packing |
| --- | ---: | --- |
| FineWeb-Edu | 50% | Q-aware |
| Cosmopedia v2 | 5% | Q-aware |
| permissively filtered CodeParrot clean | 20% | Q-aware |
| FineMath 4+ | 25% | Q-aware |

This keeps the higher-quality endpoint of the previous curriculum. Broad scale now belongs to the separate pretraining stage.

CPU tokenization uses batched Hugging Face `tokenizers.Tokenizer.encode_batch`, bounded by both document count and total characters. Long documents are split into 8K-compatible chunks with BOS/EOS, and the packer preserves ratio-2 alignment and document boundaries.

## Reasoning pool

Default target: 4M accepted nano-tokenizer tokens. The reasoning pool uses sources with explicit permissive top-level licenses:

| source | weight | license | acceptance policy |
| --- | ---: | --- | --- |
| OpenR1-Math-220k `default` | 45% | Apache-2.0 | complete generation; reject a generation explicitly marked incorrect by Math Verify |
| CHIMERA `Qwen3-235B-2507` | 30% | Apache-2.0 | `correctness=True`; Physics, Chemistry, or Biology only |
| X-Coder-SFT-376k `hybrid` | 25% | MIT | require a complete explicit `<think>...</think>` response |

OpenR1-Math is built from Apache-2.0 NuminaMath-1.5 problems and upstream-generated reasoning traces. CHIMERA describes its examples as fully synthetic; the science adapter excludes its math, computer-science, humanities, and linguistics rows. X-Coder describes its competitive-programming collection as fully synthetic.

Canonical records keep reasoning separate from final assistant content. The frozen nano tokenizer measures reasoning length and the deterministic percentile assignment maps it to integer `reasoning_effort` 1..100. Final V4.1 rendering is checked against the 8192-token row limit. Every manifest records dataset/config/split, declared license, source-specific quality filtering, and provenance notes.

This replaces the earlier `open-r1/Mixture-of-Thoughts` dependency. It is not a fallback source: if a current source becomes unavailable or changes terms, update the catalog explicitly rather than silently substituting another aggregate mixture.

## Agent pool

Default target: 8M accepted nano-tokenizer tokens.

| source | weight | acceptance policy | role |
| --- | ---: | --- | --- |
| Nebius SWE-agent trajectories | 40% | `target=True` only | repository/SWE actions |
| NVIDIA Nemotron Agentic v2 interactive | 25% | curated source rows | multi-turn tools/customer workflows |
| NVIDIA Nemotron Agentic v2 search | 15% | curated source rows | repeated web-search decisions |
| OpenSeeker v1 cleaned | 20% | `trajectory_correctness=Correct` only | long-horizon search/visit research |

Successful SWE-style traces are converted to the canonical `swe_environment` tool interface rather than training a second fenced-command protocol. Search/visit trajectories are converted to canonical tool-call IDs/results. Oversized observations are explicitly truncated with a visible marker rather than silently rewritten.

Before any source is included in the final run, verify its license and training/redistribution terms separately.

## DeepSeek V4.1 rendering

Tool schemas, DSML calls, tool-result folding, role markers, thinking markers and EOS placement are produced by the maintained `deepseek-recipe` V4.1 renderer at tokenization time. Store structured records rather than permanently rendered prompt strings.

## Two training views from one trace shard

Trace NPZ shards contain fields such as:

```text
input_ids
segment_ids
token_mask
sft_loss_mask
source_ids
reasoning_effort_ids
tool_calls
... Q diagnostics ...
```

### Mid-training view

Use ordinary causal-LM targets from `token_mask` / `segment_ids`. Do **not** apply `sft_loss_mask`.

Starting pool sampler:

```text
document  80%
reasoning  5%
agent     15%
```

This gives the ~122M model some tool/reasoning exposure without allowing those formats to dominate the continuation stage.

### SFT view

Exclude ordinary document rows by default. Supervise assistant targets only:

```text
valid_sft_target = ordinary_valid_target AND sft_loss_mask
```

User/system/tool-result tokens remain context. A starting trace-only ratio matching the 4M reasoning / 8M agent target sizes is 1/3 reasoning and 2/3 agent, but that ratio should be revisited after final filtering.

## 8K Q-aware statistics

The current retriever eligibility starts at segment-local position 640 with query budget 128. Reporting bands are:

```text
[640,768), [768,1024), [1024,1536), [1536,2048),
[2048,3072), [3072,4096), [4096,6144), [6144,8192)
```

Mid-training documents and both trace pools retain eligible-Q count, selected-Q count, budget utilization, eligible coverage and expected sampled-Q density by local-position band.

## Stage-aware entry points

Prepare the single mid-training document corpus:

```text
scripts/prepare_midtrain_corpus.py
```

Prepare reasoning/agent traces and stamp explicit stage views:

```text
scripts/prepare_stage_traces.py
```

The lower-level `prepare_document_corpus_8k.py` and `prepare_trace_corpus.py` implementations remain underneath those wrappers for reusable packing/adapter code. New runs should use the stage-aware entry points.

## Kaggle notebook

Run:

```text
notebooks/nano_dsv41f_prepare_8k_data.ipynb
```

The notebook prepares **mid-training + SFT data only**. It does not rebuild the 3B-token pretraining corpus.
