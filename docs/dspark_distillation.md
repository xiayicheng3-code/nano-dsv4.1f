# Train DSpark after SFT

Run [the DSpark distillation notebook](../notebooks/nano_dsv41f_dspark_distillation.ipynb)
with the exported SFT bundle and the existing format-v3 SFT corpus attached.
Use a Kaggle GPU for the intended run, or CPU for a short smoke test. This is a
single-device eager Torch trainer; it does not use the earlier eight-device TPU
training path. The backbone, shared token embeddings and LM head remain frozen.
No additional assistant dataset is required for this stage.

## Objective and teacher

For each selected anchor, the target predicts a full-vocabulary distribution for
each of the next `block_size` positions. The drafter sees only the target features
up to that anchor, through its sliding window. Its Markov head is teacher-forced
with the corpus tokens at the preceding positions. Supervision is restricted to
complete blocks of assistant tokens within one packed conversation.

The objective follows equations 8–12 of the
[DSpark paper](https://arxiv.org/html/2607.05147v1#S3.SS3):

- Token CE, coefficient 0.1.
- Full probability L1 distance to the frozen target, coefficient 0.9. The paper
  calls this `L_tv`; it is **twice** mathematical total variation distance.
- Confidence binary cross-entropy, coefficient 1.0, with detached soft label
  `1 - L1/2`. The confidence head reads the raw collapsed hidden plus the Markov
  embedding, matching the existing nano forward/reference implementation.

All terms use weights `exp(-position / block_size)` for zero-based positions,
normalized by their sum across the sampled blocks. Target logits and features
are detached. Confidence labels are also detached, so the drafter cannot improve
that loss by moving its own supervision target. The backbone forward runs in
inference mode. Only DSpark parameters enter AdamW; the DSpark router bias uses
the existing auxiliary-loss-free load-balancing update instead.

The teacher uses the **Torch sparse serving path**, not the dense JAX training
attention approximation. Target logits are materialized only for the sampled
positions. Several anchors share one backbone forward, and each draft anchor gets
an independent bounded SWA window. Training and inference share the draft block
operators. This trains the existing one-stage nano architecture without changing
checkpoint shapes. Quantized cache/QAT kernels remain outside this Torch runtime.

This first version uses **existing SFT responses**, rather than regenerating them
with the target. It is offline, corpus-conditioned distillation. The DSpark paper's
open experiments instead regenerate responses from dataset prompts; target-generated
training contexts would be a later way to reduce the mismatch with serving.

## Notebook inputs and defaults

Attach these directories:

1. A verified portable bundle from the SFT export notebook, including
   `model.safetensors`, `tokenizer.json`, `config.json`, and `export_manifest.json`.
2. The prepared SFT corpus containing `posttrain_manifest.json`, `tokenizer.json`,
   and the listed SFT train/validation `.npz` shards. Midtrain shards are not read.

Tokenizer checksums must match. Every shard is checksum-checked before use, and
explicit `sft_loss_mask` arrays are required. Train and held-out validation remain
separate. A shard present in both splits is rejected.

Defaults: FP32, 512-token maximum crops, eight anchors, learning rate `1e-4`,
1,000 total updates, validation/checkpoint every 100 updates, and an eight-hour
budget. **These are starting settings, not measured convergence or T4 throughput
claims.** Start with ten updates to inspect runtime and GPU memory. The CPU smoke
and regression tests exercise tiny models; no full trained-model GPU run has been
performed here.

Each sampled crop contains one contiguous conversation segment, with only valid
nonpadding tokens. Cropping long conversations resets positions and loses earlier
context; the target and draft are both conditioned on the same crop. The exported
model keeps its original context/RoPE settings. Sampling is deterministic with
replacement: the step number, seed and corpus determine each crop and anchor set.
SFT reasoning/agent pool proportions come from the manifest.

The source checkout and pip cache live under `/kaggle/temp`. Saved output contains
metrics, the two latest draft checkpoints, and the finished inference bundle.

CLI equivalent:

```bash
JAX_PLATFORMS=cpu python scripts/train_dspark.py \
  --model-dir /path/to/sft-bundle --corpus /path/to/sft-corpus \
  --output /path/to/new-dspark-run --device cuda \
  --steps 1000 --seq-len 512 --anchors 8
```

For CPU smoke testing add `--device cpu --steps 2 --eval-batches 1 --rollout-tokens 3`.
The runner reads no authentication key and does not download a new dataset.

## Validation and acceptance

Validation uses a fixed held-out set of crops/anchors at initialization and after
updates, reporting:

- CE, full L1, confidence BCE and confidence calibration error.
- Teacher-forced top-1 agreement and distribution overlap, overall and by position.
- Draft expert utilization.
- Actual greedy rollout proposals, verified tokens and accepted tokens, using up
  to two held-out assistant prefixes and short target-verified continuations.

Teacher-forced overlap is an acceptance proxy at **corpus prefixes**. It is not
measured speculative acceptance on generated continuations. The separate
`rollout_greedy_acceptance` metric does measure those continuations, but its short
sample is only a diagnostic, not a broad evaluation. The verifier remains
sequential, so this stage establishes a trainable drafter rather than claiming
speculative acceleration. A useful drafter and batched verification are both
needed before expecting a speed gain.

Training logs step time, crop size, supervised-token count and gradient norm.
The runner uses finite-loss/gradient checks and hashes the complete runtime
backbone before and after training. A changed backbone refuses export.

## Resume and export

Set `RESUME` to a previous run or its `checkpoints/` directory, use a fresh output
directory, and keep the original model bundle and corpus attached. Checkpoints
contain only DSpark and its AdamW state. They verify model/data/code/settings
identity on resume. `STEPS` is the total target, including completed updates; it
can be increased while the learning rate, losses, seed and crop settings remain
unchanged. Validation calls do not advance training sampling.

After training (or a time-budget pause with completed updates), `bundle/` is a
complete portable inference bundle. Non-DSpark tensors are copied from the original
safetensors file, preserving their BF16/FP32 dtypes and bytes. Updated DSpark leaves
are FP32. Every output tensor is roundtrip-checked, file checksums are regenerated,
and the standard bundle verifier runs before the output is committed. The original
bundle is not modified. `dspark_training.json` records provenance and validation.

Point the existing inference notebook at the new bundle. Ordinary chat continues
to run with MTP off. Explicit diagnostic use is:

```python
from nano_dsv41f.vllm_v41_cpu import NanoDeepseekV41CPU
from nano_dsv41f.vllm_v41_cpu.session import InferenceSession
model = NanoDeepseekV41CPU.from_pretrained('/path/to/run/bundle')
session = InferenceSession(model, mtp=True, draft_trained=True)
output = session.generate(input_ids, max_new_tokens=32, temperature=0)
print(session.last_stats)
```

`draft_trained=True` asserts that updates were performed; it is not an acceptance
quality gate. Inspect held-out results before choosing a drafter for serving.
Additional chat SFT would change the backbone and should be a separate experiment,
followed by distilling the drafter again against that updated model.
