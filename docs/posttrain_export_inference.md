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
generation defaults. Save the whole output folder as a Kaggle dataset; no automatic
upload or publication occurs. The bundle omits optimizer state and cannot resume
training. No pretrain dataset or tokenized corpus shards are needed for export.

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
