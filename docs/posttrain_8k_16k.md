# 8K midtrain and 16K SFT

Run `notebooks/nano_dsv41f_prepare_midtrain8k_sft16k_cpu.ipynb` on Kaggle CPU,
then `notebooks/nano_dsv41f_midtrain8k_sft16k_tpu.ipynb` on TPU v5e-8.
Pretraining preparation and the 2.4B base run remain separate notebooks.
The corpus format is `nano-dsv41f-posttrain-v2`; regenerate any older v1 output.

## Corpus and selection

The midtrain budget is 600M nonpadding tokens: 480M documents, 30M reasoning,
90M agents. CPU preparation requests 10% headroom. Documents retain the current
50/5/20/25 FineWeb-Edu/Cosmopedia/permissive-code/FineMath proportions. Supply the
pretrain corpus and completed checkpoint together to reuse the exact unconsumed
FineWeb shuffle tail; otherwise fresh FineWeb streaming may overlap pretraining.
The checkpoint/corpus/tokenizer identities must agree before tail extraction.

The reasoning and agent source catalog stays unchanged. Each source is streamed
once, with no observation character truncation in this path. Canonical normalized
histories are retained as gzip JSONL. Batched Rust tokenization builds both
possible views, then an independent selector assigns each source-local task to
either midtrain or SFT. The selected record is written once, so it cannot be
repacked into both stages. SFT has separate 30M reasoning and 60M agent
collection targets before the same 10% preparation headroom, with a configurable
share of genuinely long (>8K) individual traces. xLAM is explicitly short-only.

A view keeps the longest complete original prefix ending at an assistant EOS that
fits; it never removes earlier context or clips a tool result. A task with no
complete 8K prefix is SFT-only. When a complete 16K view is genuinely longer than
8K, it is reserved for SFT while that source's long quota has capacity. Otherwise
a stable task hash assigns dual-eligible tasks to one stage. Once a task is owned,
later variants cannot cross stages. Short rows never backfill the long SFT bucket.
Long single-answer reasoning examples are rejected for the short view. Pivot
examples are rejected if the expected action cannot fit, and only that last
expected action receives SFT loss. Other traces supervise all retained assistant
spans. Integer effort labels use the existing length-percentile policy per CPU
batch and are stored in canonical records.

This conservative prefix policy can underrepresent late actions in very long
trajectories; the original canonical histories remain available for a future,
explicit context-selection policy. OpenResearcher requires a conservative normalized Exact Answer/reference match
and no reported generation error. The hosted Harmony channels and browser calls
are parsed explicitly. All 16 seed configs are available, keeping at most one
accepted trajectory per task. Correct final answers do not verify every action. Source provenance is not a semantic correctness
certificate. No additional LLM judge is introduced.

A stable source + task-ID (or initial-user-text) hash selects a 2% trace validation holdout
before stage selection, preventing the same exact task's different generations
from crossing the split within a source. This is not fuzzy cross-source benchmark
decontamination: task IDs are source-local and the builder does not claim semantic
deduplication between datasets. Documents reserve about 2% of shards for validation;
chunks from a document can occur on both sides, so document validation is only a
monitoring signal. Train manifests report genuine >8K trace counts separately from
16K row occupancy, plus per-source selection targets and shortfalls. Packing remains compression-pair-aligned and Q-aware with segment
boundaries. CPU memory is bounded per trace buffer; the document packer retains
its document-token arrays in RAM. At 480M tokens that component requires several
GB including intermediate arrays; use a normal Kaggle CPU RAM allocation.

Completed source units can be reused after copying saved outputs into the CPU
output directory. A changed tokenizer/budget/seed/builder requires a new directory.
The v2 TPU reader verifies shard checksums before training and rejects a v1 corpus,
because v1 duplicated selected traces across stages. Interrupted source units are
rebuilt rather than trusted.

## Training

Midtrain continues the supported narrow48 CP2/DP4 pretrained parameters, optimizer,
global steps, and original full-3B LR/indexer schedule. Candidate masking stays
off. The requested 600M allocation must match the remaining base+midtrain budget,
within one base batch. Pool scheduling tracks actual nonpadding tokens, not row
counts. Pool exhaustion fails explicitly; there is no hidden repetition or fallback
source. Source proportions reflect actual accepted corpus capacity.

SFT uses four 16K rows/update with an independent 1:2 reasoning/agent sampler.
The default one-pass raw-token budget is limited by the smaller pool's capacity
at this ratio, with a small batch headroom margin. It does not require using every
materialized agent row. Only assistant targets contribute LM loss; user/system/tool
inputs remain fully visible to attention. The loss checks the **target** token's
mask after the causal shift and still forbids cross-segment predictions.

SFT resets optimizer state, starts an independent 2.6e-5 -> 2.6e-6 LR schedule
(defaults, not a tuned optimum), activates the planned candidate mask/indexer
auxiliary, and enables compressed-layer YaRN with original_seq_len=8192 and
rope_factor=2. Pure local RoPE keeps its existing settings. The saved checkpoint
recipe records these changes. Midtrain and SFT checkpoints are stored separately.

Both stages have checksummed full-state checkpoints, deterministic per-pool
cursors, held-out loss monitoring, finite-loss/zero-drop checks, and cooperative
wall-time/signal pauses. Keep code, corpus, budgets, seeds and LR identical for
resume. A compilation can overrun the cooperative deadline; committed periodic
checkpoints remain the recovery point. The notebook allows eight hours including
setup before Kaggle's nine-hour cap. Completing both stages in one session is not
guaranteed. 16K native compilation, memory use and throughput require a real TPU
run; CPU tests cannot establish that performance.
