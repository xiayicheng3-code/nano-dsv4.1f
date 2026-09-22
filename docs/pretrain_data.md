# Pretraining-only, 3B-token data run

Import `notebooks/nano_dsv41f_prepare_pretrain_3b.ipynb` into Kaggle, enable
Internet, and select CPU. It resolves the frozen tokenizer from
`xiayicheng3gmailcom/nano-dsv41f-tokenizer-fineweb`, runs a small smoke build,
and then prepares only FineWeb-Edu `sample-10BT`.

This notebook is independent of the staged corpus and trace notebooks. It uses
an isolated virtual environment and minimal CPU dependencies, so data preparation
does not install JAX, Tokamax, or the DeepSeek trace renderer.

## Budget and data policy

- Training target: **3,000,000,000 nonpadding tokens measured with our tokenizer**.
- Separate validation target: **10,000,000 tokens**. The published dataset's 10B
  label uses GPT-2 token counts and is not used as our accounting unit.
- BOS/EOS are included in the budget. Each split may overshoot by less than
  8192 tokens because the final document chunk is retained. Manifests separately
  record content tokens, nonpadding tokens, LM targets, and physical token slots.
- Documents are assigned to validation by BLAKE2b-64(exact text) modulo 100 = 0.
  Identical text therefore cannot cross train/validation, regardless of source
  file or chunking. This does not promise near-duplicate or benchmark decontamination.
- Retain upstream curated text without new scoring/deduplication passes; empty
  documents and documents above the explicit 1,000,000-character memory bound
  are skipped and counted. The bound is configurable.
- Shuffle source files with a fixed seed. Use bounded best-fit document packing
  across 32 open rows; no Q-position-aware packing or SFT masks. Long documents
  are split into chunks of at most 8190 content tokens, with BOS/EOS on each.
- Every segment is aligned to two positions for CSA2. Cross-document attention
  and LM targets remain masked, matching the existing packed training interface.

## Storage and bounded memory

The IDs require roughly **6 GB** for 3B tokens with this uint16-compatible
vocabulary, plus padding and small segment-length/offset arrays. The builder
writes at most 2048 rows per output shard; it never accumulates the 3B-token
corpus in RAM. Text encoding uses Rust `Tokenizer.encode_batch`, 256-document /
4M-character batch bounds, and the allocated CPU threads. Input Parquet is streamed.

Each output shard has three uncompressed NPY arrays:

| Array | Dtype | Meaning |
| --- | --- | --- |
| `tokens` | uint16 | Fixed `[rows, 8192]` token IDs |
| `lengths` | uint16 | Flattened real segment lengths, including BOS/EOS |
| `offsets` | uint32 | CSR offsets into `lengths`, one per row plus sentinel |

The loader reconstructs segment IDs and token masks from these small lengths.
It reads token arrays through NumPy memory maps and shuffles shards and rows.
Full-length segment/mask arrays exist only for the current host training batch.

## Reproducibility and interruption

`build_plan.json` freezes the source revision, file order, tokenizer SHA-256,
builder SHA-256, dependency versions, budgets, and packing settings. Retry with
the same command; incompatible builds are refused. Each input Parquet file is
committed atomically after its output shards and manifest are written. Existing
completed outputs are checksum-verified on resume. Only the current uncommitted
source file is rebuilt; large source files can mean substantial replay time.

Progress appears during encoding and after every committed source file. Source
exhaustion is an error with partial data preserved, never a silently shorter run.
The top-level manifest has `complete: true` only when both budgets are reached.

Kaggle session-local persistence does not survive session deletion. Save partial
outputs before ending a session, then copy them to a writable output directory
to resume elsewhere. The notebook never deletes an existing output directory.

## CLI and training interface

```bash
python -m venv /tmp/nano-pretrain-env
/tmp/nano-pretrain-env/bin/pip install -r requirements-pretrain-data.txt
/tmp/nano-pretrain-env/bin/python scripts/prepare_pretrain_corpus.py \
  --tokenizer /path/to/tokenizer.json \
  --output-dir /path/to/pretrain-3b \
  --train-tokens 3000000000 --validation-tokens 10000000
```

Run this loader from the repository root, or add the repository root to Python's
module search path:

```python
from scripts.prepare_pretrain_corpus import iter_pretrain_batches

for batch in iter_pretrain_batches(corpus_dir, batch_rows=global_microbatch_rows, seed=1701):
    # Pass these NumPy arrays to put_training_batch(...), then the compiled step.
    ids = batch['input_ids']        # int32
    segments = batch['segment_ids'] # int32
    token_mask = batch['token_mask'] # bool
```

Save the data seed and number of consumed batches alongside model checkpoints;
recreate the iterator and skip that many batches to resume the same input order.
There is no implicit repetition. `drop_last=True` drops at most one incomplete
batch per dataset pass; `drop_last=False` retains it for evaluation or inspection.

3B / 8192 is about 366,211 rows before packing overhead. For one pass, optimizer
steps are approximately `packed_rows / (global_microbatch_rows * accumulation)`.
This is independent of the old 10,000-step data-generation default. The TPU stress
test still determines the feasible microbatch; this change does not alter model
architecture, indexer-stage schedules, or launch model training.

Sources: [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
[Tokenizers API](https://huggingface.co/docs/tokenizers/api/tokenizer),
[Datasets streaming](https://huggingface.co/docs/datasets/stream).
