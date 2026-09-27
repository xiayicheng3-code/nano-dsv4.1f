# Kaggle pretraining run

Import `notebooks/nano_dsv41f_pretrain.ipynb`, choose TPU v5e-8, enable Internet,
attach `xiayicheng3gmailcom/nanodsv4-1f-pretrain-tokenized` and
`xiayicheng3gmailcom/nano-dsv41f-tokenizer-fineweb`, then Save & Run All.
No corpus rebuild is needed. The existing prepared corpus is FineWeb-Edu only.

## Current experiment allocation

This run uses **2.4B non-padding tokens for initial pretraining**, reserving
**600M of a 3B combined budget for mid-training**. SFT remains separate.
This explicitly supersedes the earlier *experiment budget* of 3B base tokens
plus independent mid-training, while preserving the three distinct data/loss
stages. Corpus preparation still targets 3B tokens; unused rows remain available.
This allocation is our engineering starting point, not a published DeepSeek ratio.

The runner counts actual non-padding tokens (including BOS/EOS), LM targets and
physical slots separately. It ends after the first complete four-row update
meeting the base token target, overshooting by fewer than 32,768 real tokens.
It never repeats the dataset implicitly and never starts mid-training or SFT.

The full-3B optimizer horizon is `ceil(3B / mean_nonpadding_tokens_per_update)`
using the frozen corpus manifest. This is a step schedule estimated from packing
density, not an exact token-coordinate schedule. Its value is recorded and must
be carried into the second run; a different mid-training packing density may
require an explicit schedule decision then. Warmup stays at 500 updates; LR is
2.6e-4, cosine decay begins at 90%, and the floor is 2.6e-5. The first notebook
stops at roughly 80% with the LR still on its plateau. No optimizer reset occurs.

The existing indexer auxiliary window is 55–90% of that full horizon. It is
independent of the data-stage name: some indexer training occurs in this initial
FineWeb-Edu run. Hierarchical candidate masking is off. DSpark is allocated and
frozen, and QAT is off. The later mid-training runner must explicitly apply its
stage policy and introduce its own data cursor while preserving learned state;
using this pretrain notebook with a different corpus is intentionally rejected.

## Validated hardware preset

- Four global rows of 8192, attention CP2/DP4, MoE EP8.
- Seven-layer d=512 backbone, 48 routed experts of width128, top-4 plus shared.
- Sequential Splash, local backward-Q tile128, compressed/global tile1024.
- `moe_buffer_divisor=4`, with per-chip full-buffer fallback and no token dropping.
- BF16 payloads/parameters, FP32 optimizer and controls, block rematerialization.
- Existing Muon/Sinkhorn/AdamW optimizer, sampled indexer query budget128/group.

The prior short TPU experiment supports about 5.9 hours of training steps for
this allocation. Real data loading, compilation, validation and checkpoints add
overhead. Trained routing may exercise buffer fallback more often; throughput
and fallback counts are logged during training.

## Data, validation and restart

All corpus shard checksums and tokenizer identity are verified before TPU
initialization. Host input prefetch is bounded to one future batch and memory
mapped shards. A fixed seed shuffles shard order and rows within each shard.
Resume skips preceding row positions without loading/decoding their token arrays.
Only completed updates advance the saved data cursor.

Validation uses a fixed held-out 32-batch subset, weighted by valid next-token
target counts. It runs at session start, every 10k global steps and completion
when wall-time permits. It uses the ordinary LM attention path, not inference
sparse-retrieval evaluation. Validation never mutates model/optimizer/router state.
There is a separate compiled forward executable and no validation gradient step.

The eight-hour deadline starts in the first code cell, before dependency setup.
The worker stops updates two minutes before that deadline, leaving roughly an
additional hour below the nine-hour Kaggle cap. It avoids starting a new compile
with less than 15 minutes remaining. This is cooperative: an in-flight compilation
cannot be interrupted by this loop, so periodic checkpoints remain essential.

Checkpoints are saved at initialization/resume, after the first ten updates,
every 2,000 updates, before new compiles/evaluation and at normal exit. Atomic
commit and a latest pointer prevent partially written checkpoints from being
selected. Two committed checkpoints are retained. Every checkpoint contains:

- All parameters (including frozen DSpark and router biases) and optimizer leaves.
- Exact BF16 bits, FP32 states, shapes, paths and per-leaf SHA-256 checksums.
- Global completed-step count, consumed-batch cursor and all three token counts.
- Initialization/data/query seeds, recipe, full schedule, corpus and tokenizer identities.

There is no mutable dropout RNG in this path; query sampling uses the saved seed
and global step. Restore rebuilds the tree against current sharding templates,
checks identity/shape/dtype/hash and restores each leaf to its sharding. Nonfinite
checkpoint leaves are rejected before updating the latest pointer. A training
failure retains the previous committed checkpoint instead of saving suspect state.

Save the completed Kaggle outputs. For a paused base run, attach the previous
outputs and set `RESUME_CHECKPOINT` to the checkpoint directory or its parent
containing `latest.json`. The notebook exports a ZIP with the latest checkpoint,
tokenizer, summary, recipe, logs and source archive. Extract the ZIP before resume.
Use the same notebook/source version. A new output directory is required; previous
outputs are never overwritten. `--max-steps` is available for a short debug launch,
but the notebook defaults to the full base allocation.
