# Export a completed SFT run and run inference

The completed run in `posttrain-20261001-183833` finished both stages at global
step 85,663. Its final SFT step directory is `step-00085663-af2e759ac4dc`.
The run processed 71,949,953 nonpadding SFT tokens over 578 updates, with
31,208,397 shifted assistant targets. The small overshoot of the requested
71,874,006 raw tokens comes from completing the last batch.

## 1. Save the TPU output

Save the finished TPU notebook outputs as a Kaggle dataset. Retain the complete
`training/` folder: `sft/checkpoints/`, `tokenizer.json`, and `corpus_manifest.json`
are needed for export. The full training checkpoint remains the resumable copy.

## 2. Export on CPU

Import `notebooks/nano_dsv41f_export_sft_safetensors_cpu.ipynb`, select CPU,
enable Internet, and attach the saved TPU output. Leave `CHECKPOINT=''` to locate
exactly one SFT checkpoint set, or specify the attached path to the final step,
its `checkpoints/` parent, or the training output. Multiple runs require an explicit
path; the notebook never selects the first arbitrary match. `OUTPUT` must be fresh.

The exporter validates completion, the checkpoint/corpus identity, tokenizer hash,
vocabulary and special-token IDs. It reconstructs shapes using an abstract JAX
model tree, then reads and checks only the parameter leaves. It does not initialize
real replacement parameters or load optimizer arrays. BF16 uint16 checkpoint storage
is interpreted as bits, preserving the original values. FP32 parameters remain FP32.
The safetensors round trip is verified for every tensor, including DSpark weights.

The export uses `metadata.recipe.model` and `metadata.recipe.train.seq_len`, not
the base pretraining recipe. This preserves the 32K context, compressed-layer
YaRN factor 4, and SFT candidate-mask configuration. The result includes:

| File | Purpose |
| --- | --- |
| `model.safetensors` | All model parameters in the existing nano portable layout |
| `config.json` | Model architecture and effective context/YaRN settings |
| `tokenizer.json` | The tokenizer verified against the training corpus identity |
| `training_recipe.json` | Effective model, training, and native settings |
| `training_metadata.json` | Source checkpoint metadata and training provenance |
| `export_manifest.json` | Completion marker, tensor counts, and bundle file hashes |

Additional portable metadata files describe parameter shapes, tokenizer IDs and
generation defaults. Save the whole output folder as a Kaggle dataset. The bundle omits optimizer state and cannot resume
training. No pretrain dataset or tokenized corpus shards are needed for export.

The export checkout and pip/Hugging Face caches are placed in the marked
`/kaggle/temp/nano-dsv41f-export` workspace, outside Kaggle's saved output directory.
The final cell removes that workspace only after verifying the preserved bundle.
Only the export folder is written under `/kaggle/working`; Kaggle can additionally
create its standard notebook, HTML, and log files. Attached checkpoint inputs remain
under `/kaggle/input`.

To upload the export to Hugging Face, set `HF_REPO_ID='your-account/your-dataset'`
in the configuration cell and enable your write token as the Kaggle secret
`HF_TOKEN` (or change `HF_SECRET_NAME`). `HF_PRIVATE=True` creates a private dataset
repository; existing repository visibility is not changed. The upload cell passes
the token directly to `HfApi`, without printing it or writing a saved login. It
uploads only the files listed in the verified export manifest plus the manifest
itself, using `repo_type='dataset'`. The token stays out of the output bundle.
When the repo ID is blank, upload is skipped. Cleanup runs even if upload fails,
and the verified bundle is retained for Kaggle saving or retrying the transfer.

CLI equivalent from the repo with CPU dependencies installed:

```bash
JAX_PLATFORMS=cpu python scripts/export_posttrain_checkpoint.py \
  --checkpoint /path/to/training/sft/checkpoints \
  --output /path/to/new-portable-directory
```

If the accompanying training files were moved, `--tokenizer` and
`--corpus-manifest` accept their explicit paths while retaining hash checks.

## 3. Run the exported model

Import `notebooks/nano_dsv41f_sft_inference_cpu.ipynb`, select CPU, enable Internet,
and attach only the exported bundle. `MODEL_DIR=''` locates exactly one bundle;
otherwise set its directory explicitly. Both notebooks fetch `SOURCE_REF` from
GitHub and print the resolved commit. A commit SHA also works for reproducibility.

The inference notebook verifies the bundle before loading it with the existing
Torch CPU runtime. It uses FP32 arithmetic and incremental cached generation.
`chat()` and its widgets apply the DeepSeek V4.1 protocol via `deepseek-recipe`;
`complete()` uses a raw prompt. Prompts plus requested output must fit the stored
context length. Start with short inputs/outputs on CPU. The 32K training context
does not establish CPU latency or long-context task quality.

The chat widget runs generation in a worker thread so notebook controls stay
responsive. It disables message input, Send, Reset, and generation settings during
each request, ignores additional clicks while busy, and restores controls after a
reply or error. An unsuccessful request restores the message for retrying.

Thinking controls match the [official V4.1 open-weight encoder](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/encoding/README.md):
Low = 50, High = 75 (default), Max = 100, or a custom integer from 1 through 100.
The helper also accepts these three preset names. Effort is disabled in chat mode;
`max_tokens` separately caps all generated tokens, including reasoning. Numeric
effort uses the same initial-prefix adjustment as the canonical SFT renderer because
the pinned `deepseek-recipe` bindings only accept named presets. Context checks and
generation tokenize this exact adjusted prompt.

Tool calls are returned for inspection and are not executed. Optional local HTTP
serving uses the existing API backend. DSpark weights are preserved for future
use; this runtime still uses ordinary autoregressive decoding. This custom model
is not registered as a stock Transformers AutoModel or vLLM engine model.

## Validation

Tests export real tiny JAX checkpoints with mixed BF16/FP32 parameters and DSpark,
without optimizer files, and compare every saved tensor bitwise. They verify the
effective SFT recipe, compare CPU logits and cached greedy decoding before/after
export, reject mismatched/incomplete inputs, and check bundle corruption. Both
generated notebooks and the shared chat generator are compiled as Python cells.
The user's full checkpoint export and CPU throughput must be measured in Kaggle.
