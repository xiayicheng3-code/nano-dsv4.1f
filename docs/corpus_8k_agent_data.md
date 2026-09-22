# 8K document, reasoning, and agent data

This is the CPU/Kaggle preparation path for the first 8K nano-dsv4.1f training run. It
keeps three physical pools separate so training can change mixture weights without rebuilding
data:

```text
document/   ordinary causal-LM text/code/math
reasoning/  explicit problem -> reasoning -> answer traces
agent/      reasoning -> tool -> observation -> ... trajectories
```

All packed rows are 8192 tokens. Token IDs remain `uint16`; masks/source IDs/effort IDs use
bytes. Shards are uncompressed NPZ by default because DEFLATE is usually wasted CPU for this
Kaggle preprocessing job. Pass `--compress-shards` only when storage is the limiting resource.

## Frozen tokenizer

The Kaggle notebook resolves the already-built tokenizer directly from:

```text
xiayicheng3gmailcom/nano-dsv41f-tokenizer-fineweb
```

It uses `kagglehub.dataset_download(...)` because the tokenizer is a file artifact rather
than tabular data. The builders verify every reserved nano/DeepSeek protocol token ID before
writing corpus shards.

## Document pool

The source progression remains a small-model-oriented baseline:

| phase | FineWeb-Edu | Cosmopedia v2 | code | math | packing |
|---|---:|---:|---:|---:|---|
| early | 72% | 13% | 12% | 3% FineMath-3+ | best fit |
| middle | 60% | 10% | 15% | 15% FineMath-3+ | best fit |
| late-mid | 50% | 5% | 20% | 25% FineMath-4+ | Q-aware |

With 10,000 training steps and 5% data headroom, this is still 10,500 rows, but at 8K it is
86,016,000 physical tokens rather than the earlier 43,008,000-token 4K build.

CPU tokenization is batched with Hugging Face `tokenizers.Tokenizer.encode_batch`, so the Rust
backend and Rayon can use the Kaggle CPU cores. Batches are bounded by both document count
(default 256) and total characters (default 4M) to avoid one collection of huge webpages
creating a memory spike.

## Reasoning pool

Default target: 4M accepted nano-tokenizer tokens. The reasoning pool deliberately uses
sources with explicit permissive top-level licenses:

| source | weight | license | acceptance policy |
|---|---:|---|---|
| OpenR1-Math-220k `default` | 45% | Apache-2.0 | complete generation; reject a generation explicitly marked incorrect by Math Verify |
| CHIMERA `Qwen3-235B-2507` | 30% | Apache-2.0 | `correctness=True`; Physics, Chemistry, or Biology only |
| X-Coder-SFT-376k `hybrid` | 25% | MIT | require a complete explicit `<think>...</think>` response |

OpenR1-Math is built from Apache-2.0 NuminaMath-1.5 problems and upstream-generated reasoning
traces. CHIMERA describes its examples as fully synthetic; the science adapter deliberately
excludes its math, computer-science, humanities, and linguistics rows so this bucket stays a
science complement rather than duplicating the other two buckets. X-Coder describes its
competitive-programming collection as fully synthetic.

Canonical records keep reasoning separate from final assistant content. The frozen nano
tokenizer measures reasoning length and the existing percentile assignment maps it to integer
`reasoning_effort` 1..100. The final V4.1 rendering is checked again against the 8192-token row
limit. Every manifest records dataset/config/split, declared license, source-specific quality
filtering, and provenance notes.

This replaces the earlier `open-r1/Mixture-of-Thoughts` dependency. It is intentionally not a
fallback source: if one of these datasets becomes unavailable or changes terms, update the
source catalog explicitly rather than silently substituting another aggregate mixture.

## Agent pool

Default target: 8M accepted nano-tokenizer tokens.

| source | weight | acceptance policy | role |
|---|---:|---|---|
| Nebius SWE-agent trajectories | 40% | `target=True` only | repository/SWE actions |
| NVIDIA Nemotron Agentic v2 interactive | 25% | curated source rows | multi-turn tools/customer workflows |
| NVIDIA Nemotron Agentic v2 search | 15% | curated source rows | repeated web-search decisions |
| OpenSeeker v1 cleaned | 20% | `trajectory_correctness=Correct` only | long-horizon search/visit research |

### SWE conversion

The current Nebius release stores a trajectory as rows with `role`, `text`, `mask`, and
`system_prompt`. An AI turn contains natural-language reasoning followed by its environment
command in the final fenced code block. The adapter:

1. keeps only solved trajectories (`target=True`);
2. preserves the issue/user text and SWE environment instructions;
3. converts the final fenced command into a `swe_environment(cmd=...)` tool call;
4. maps the following user/environment turn to the matching tool result;
5. keeps the preceding natural language as explicit assistant reasoning.

This avoids training the original fenced-command syntax as a second competing tool protocol.

### Search conversion

OpenSeeker's validated FSM is converted from

```text
system -> user -> reasoning -> tool_call -> tool_output -> ... -> answer
```

into canonical OpenAI-style messages with real tool-call IDs. Only Correct trajectories are
used. Search/visit observations are truncated explicitly before rendering so a single search
page cannot crowd the whole 8K context; the truncation marker remains visible to the model.

NVIDIA rows are already message + tool-schema structured, so they go through the existing
canonical cleaner with only explicit tool-output length caps.

## DeepSeek V4.1 rendering

Tool schemas, DSML calls, tool-result folding, role markers, thinking markers and EOS placement
are produced by the maintained `deepseek-recipe` V4.1 renderer.

The released V4.1 renderer itself uses the numeric prompt form

```text
Reasoning Effort: N (range 1-100, the higher the value, the more thorough the reasoning)
```

but the public Python API currently exposes named effort presets. The nano data path therefore
renders with the official 75/default form and replaces exactly that one numeric prefix with the
canonical integer assigned to the example. No tool/role serialization is hand-reimplemented.

## Two training views from one trace shard

Trace NPZ shards contain:

```text
input_ids               uint16 [rows,8192]
segment_ids              uint16 [rows,8192]
token_mask                uint8 [rows,8192]
sft_loss_mask             uint8 [rows,8192]
source_ids                uint8 [rows,8192]
reasoning_effort_ids      uint8 [rows,8192]
tool_calls               uint16 [rows]
... Q diagnostics ...
```

For mixed/continued pretraining, ignore `sft_loss_mask` and use ordinary packed causal LM. For
SFT, additionally require the target token's `sft_loss_mask`: user/system/tool observations
remain context while assistant reasoning, DSML calls, final content and EOS are supervised.

## 8K Q-aware statistics

The current retriever warmup still starts at segment-local position 640 with budget 128. The
8K reporting bands are:

```text
[640,768), [768,1024), [1024,1536), [1536,2048),
[2048,3072), [3072,4096), [4096,6144), [6144,8192)
```

Late-mid documents and both trace pools are packed with the expected sampled-Q-density
objective. Every manifest reports eligible Q count, selected Q count, budget utilization,
eligible coverage and expected sample density by local-position band.

## Initial sampler schedule

Keep physical pools independent and sample them at training time:

```text
early:     100% document
middle:     90% document + 5% reasoning +  5% agent
late-mid:   80% document + 5% reasoning + 15% agent
```

These are starting points for ablations, not claims about an optimal DeepSeek training recipe.
In particular, agent data is intentionally withheld early so the ~122M model first acquires
basic language/code/math capacity instead of overfitting tool syntax.

## Kaggle notebook

Run:

```text
notebooks/nano_dsv41f_prepare_8k_data.ipynb
```

It downloads the frozen tokenizer by Kaggle handle, clones the corpus branch, enables tokenizer
CPU parallelism, builds all three pools, and prints source fractions, retention/drop counts,
SFT supervision fractions, reasoning-effort coverage and Q-position statistics.
