# Chat SFT and the MTP-in-SFT ablation

Status: **measured**. One seed per arm; held-out sets are fixed and shared.

## Question

The first SFT (`codex/midtrain8k-sft16k`) used 67% agent/tool traces and 33% reasoning traces,
and no plain conversation. At ~11M active parameters the model reproduced the dominant
format: tool-call markup in chat replies. This experiment replaces that stage with chat data and
asks a second question on the side: **does adding a multi-token-prediction (MTP) loss during SFT
help a model this small?** (Gloeckle et al., 2024 report MTP helps at scale but hurts small models;
DeepSeek-V3 uses MTP in pretraining.)

## Setup

Both arms are identical except for the MTP weight.

| | |
|---|---|
| Start | completed mid-train checkpoint, step 85,085 (2.4B pretrain + 0.36B mid-train tokens) |
| Data | 95% [smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) (rev `f73fe857`), 5% reasoning traces from the v3 SFT pool; 441M tokens, one pass |
| Format | DeepSeek V4.1 non-thinking chat, assistant-only loss, packed 32K rows |
| Recipe | unchanged from the earlier SFT: fresh optimizer, LR 2.6e-5 → 2.6e-6 cosine, YaRN ×4, candidate-masked indexer auxiliary |
| Steps | 3,421 updates × 4 × 32K rows, ~2.6 s/update on TPU v5e-8 for **both** arms |
| MTP arm | `--mtp-weight 0.1`: the DSpark head predicts the next 5 assistant tokens at 256 sampled anchors per row (`nano_dsv41f/mtp.py`), position-weighted CE, gradients into the backbone |

Code: `scripts/prepare_chat_sft_corpus.py`, `scripts/run_chat_sft.py`,
`src/nano_dsv41f/mtp.py`; notebooks `nano_dsv41f_prepare_chat_sft_cpu`,
`nano_dsv41f_chat_sft_tpu`, `nano_dsv41f_chat_sft_ablation_eval`.
Raw numbers: [JSON](2026-11-09-chat-sft-mtp-ablation.json); all T=0.7 samples:
[side by side](2026-11-09-chat-sft-samples-t07.md).

## Results

### Held-out next-token loss (assistant tokens, nats)

| step | chat: baseline | chat: MTP | reasoning: baseline | reasoning: MTP |
|---:|---:|---:|---:|---:|
| 0 | 2.8809 | 2.8809 | 2.2418 | 2.2418 |
| 500 | 2.7455 | 2.7671 | 2.1903 | 2.2081 |
| 1,000 | 2.7211 | 2.7427 | 2.1859 | 2.2019 |
| 2,000 | 2.7097 | 2.7287 | 2.1796 | 2.1940 |
| 3,421 | **2.7067** | 2.7250 | **2.1772** | 2.1907 |

The MTP arm is worse at every evaluation on both pools: +0.018 nats on chat and +0.014 on
reasoning at the end, on 353K and 442K fixed held-out tokens. The MTP head itself learned
little: its loss fell from 10.1 to 8.0, with 3–4% top-1 accuracy at every draft depth.

### Benchmarks (lm-evaluation-harness, zero-shot, full sets)

| model | LAMBADA | HellaSwag | PIQA | ARC-E | ARC-C | WinoGrande | SciQ | OBQA | WikiText bpb ↓ | avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| earlier SFT (agent/reasoning) | 9.3 | 27.0 | 56.7 | 37.8 | 22.8 | 52.4 | 57.8 | 26.2 | 1.358 | 36.3 |
| **chat SFT baseline** | 10.2 | 27.1 | 57.4 | 37.8 | 22.4 | 53.0 | 56.9 | 26.8 | 1.352 | **36.5** |
| chat SFT + MTP | 10.1 | 26.9 | 57.3 | 38.1 | 21.9 | 51.4 | 56.5 | 25.8 | 1.356 | 36.0 |

All differences are within about one standard error. Chat SFT did not cost raw-LM ability.

### Chat behaviour (32 fixed prompts, no tools offered, 512-token limit)

| | earlier SFT | chat baseline | chat + MTP |
|---|---:|---:|---:|
| replies with tool-call markup, T=0.7 | **56%** | 6% | 3% |
| replies with tool-call markup, greedy | 47% | 3% | 12.5% |
| replies ending with EOS within 512 tokens, T=0.7 | 31%¹ | 6% | 3% |
| distinct word bigrams per reply, T=0.7 | 0.45 | 0.35 | 0.32 |

¹ Mostly right after an unrequested tool call.

## Conclusions

1. **Chat SFT fixes the format problem.** Unrequested tool-call markup drops from about half of
   replies to a few percent, and replies are on topic in conversational form. Content is still
   unreliable and repetitive, as expected at this size and pretraining budget.
2. **MTP during SFT hurts at this scale**, consistently if slightly, which is what Gloeckle et al.
   predict for small models. It cost almost nothing in step time.
3. **The chat baseline ships.**

## Caveats and follow-ups

- One seed per arm; benchmark differences are within noise, while the held-out loss gap is
  consistent across all 14 comparisons.
- The MTP head started untrained and learned at the SFT learning rate (2.6e-5), so the backbone
  received gradients from a weak head. A head warm-up or a higher head LR could change the result.
- The chat models seldom emit EOS within 512 tokens. They tend to continue or loop, so the chat
  notebook should keep a token limit and a repetition guard.
