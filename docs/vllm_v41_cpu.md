# DeepSeek V4.1-derived CPU inference

This directory adds a correctness-first Torch CPU runtime for `nano-dsv4.1f`.

## Provenance rule

The CPU port is derived from the DeepSeek **V4.1** architecture and the project's JAX V4.1 reference. It does not import or copy the separate `vllm.models.deepseek_v4` model implementation. Current upstream vLLM has a dedicated `vllm/models/deepseek_v41/` tree for V4.1, but no corresponding V4.1 CPU model package, so the nano project supplies generic eager Torch CPU operators for the missing platform path.

The port preserves the nano V4.1 mechanisms that matter for inference:

- partial/inverse RoPE and compressed-RoPE regime;
- learned ratio-1/ratio-2 compressed global KV;
- Full / Reindex / Reuse retrieval state;
- dynamic multi-head indexer scoring and hierarchical candidate blocks;
- Single-Pass mHC residual streams;
- Engram n-gram memory;
- sqrt-softplus routed MoE plus the shared expert;
- the context/generation layer schedule exported by `build_layer_specs`.

## Install

CPU model only:

```bash
pip install -e '.[cpu]'
```

CPU model plus DeepSeek-compatible HTTP/protocol support:

```bash
pip install -e '.[api]'
```

## Load an exported checkpoint

The runtime consumes the existing portable checkpoint directly. No DeepSeek production-name conversion is required.

```python
import torch
from nano_dsv41f.vllm_v41_cpu import NanoDeepseekV41CPU

model = NanoDeepseekV41CPU.from_pretrained(
    "/path/to/exported-checkpoint",
    dtype=torch.float32,
)

input_ids = torch.tensor([[1, 42, 17, 9]], dtype=torch.long)
logits, aux = model.forward(input_ids, sparse_retrieval=True)
```

For a numerical comparison with the JAX training backbone, disable sparse selection:

```python
logits, aux = model.forward(
    input_ids,
    compute_indexer=False,
    sparse_retrieval=False,
)
```

`tests/test_vllm_v41_cpu.py` compares that dense CPU path directly with JAX logits from the same randomly initialized parameter tree.

## Incremental cache and generation

The CPU reference now has a real autoregressive cache rather than recomputing the prefix. It keeps independent persistent compressed-KV/indexer states for the context source layer (L1) and generation source layer (L3), per-layer fixed local/SWA KV rings, and a causal pending token for ratio-2 compression.

```python
prefill_logits, cache = model.prefill_cache(
    input_ids,
    sparse_retrieval=True,
)

next_logits, cache, aux = model.forward_step(
    torch.tensor([[23]]),
    cache,
    sparse_retrieval=True,
)
```

Normal generation uses that cache automatically:

```python
output_ids = model.generate(
    input_ids,
    max_new_tokens=32,
    temperature=0.0,
)
```

The cache uses stable backing allocations up to a configured capacity. Compressed
KV, index K, token IDs and positions grow only as logical views; each local layer
stores at most `local_window` latents. Engram hashes only its n-gram tail, RoPE
inverse frequencies are reused, and sparse main attention gathers Top-K latents
before attention (the indexer still scans its keys). Prefill can return only the
last logits, avoiding a prompt-length-by-vocabulary result during generation.
Tests compare cached dense and sparse prefill against full-prefix execution and
cached greedy generation against full-prefix recomputation.

This takes the fixed-allocation approach requested for serving state, but is not
a Mamba recurrence or vLLM integration. Global attention still needs O(capacity)
storage. Exceeding the capacity raises an error; history is never silently evicted.

For prefix reuse across calls, use `InferenceSession` (the API backend and both
chat notebooks use it automatically):

```python
from nano_dsv41f.vllm_v41_cpu.session import InferenceSession, format_inference_stats
session = InferenceSession(model, capacity=32768)  # must fit exported context
output_ids = session.generate(input_ids, max_new_tokens=32)
print(format_inference_stats(session.last_stats))
# Submit the complete next prompt to this same session.
# Exact cached prefixes are skipped; changed histories are checked before reuse.
session.reset()
```

Caching persists in session RAM, not across kernel restarts. A checkpoint of the
last prompt preserves small rolling buffers plus logical global lengths, allowing
reuse if protocol formatting rewrites the previous assistant reply. Unrelated
prompts, changed attention mode, failures, or reset invalidate incompatible state.
A lock serializes generation on each session. Use separate sessions for independent
users. The last emitted token is ingested on the next request if needed.

Each turn reports new/reused/total prompt tokens, prefill seconds and tokens/s,
decode tokens/seconds/tokens/s, first-token and total latency, individual decode
step times, and cache arena bytes. Throughput excludes tokenization and protocol
parsing. Decode includes reasoning, protocol markers and generated EOS, sampling,
and optional drafting; its first token uses the final prefill logits. GPU clocks
synchronize if the runtime is manually loaded on CUDA. Cache arena bytes exclude
weights, working tensors, and the rolling-state prompt checkpoint. Prefill processes
32 tokens per causal target pass by default; configure `prefill_chunk_size` on the
session or `chunk_size` on `model.prefill_cache`. Setting it to 1 provides the scalar
baseline. Prompt loading projects only the final position's vocabulary logits
in each chunk. This remains an eager Torch runtime.

Measure the real exported model with:

```bash
python scripts/benchmark_cpu_inference.py --model-dir /path/to/bundle \
  --threads 2 --max-new-tokens 32 --repeats 3 --output /tmp/inference-stats.json
```

This measures cold KV and raw-token continuation separately after operator warmup.
The synthetic development measurement is recorded in
[the inference experiment](experiments/2026-10-04-inference-cache.md).

Packed ratio-2 decode requires segment boundaries to fall on completed compression groups, matching the packed training invariant; the runtime raises instead of compressing across a segment boundary.

## DeepSeek V4.1 API protocols

DeepSeek V4.1 does not use a simple local Jinja chat template as its authoritative protocol definition. The CPU serving layer therefore uses the maintained `deepseek-recipe` package to normalize requests, render/encode V4.1 prompts, and parse generated token IDs plus thinking/tool syntax back into the requested response format.

`NanoDeepSeekProtocolBackend` exposes four text-generation endpoint families:

- `POST /v1/completions` — classic raw-prompt completions; **no chat template is applied**;
- `POST /v1/chat/completions` — OpenAI-style Chat Completions encoded as DeepSeek V4.1;
- `POST /v1/responses` — OpenAI Responses requests encoded as DeepSeek V4.1;
- `POST /v1/messages` — Anthropic Messages requests encoded as DeepSeek V4.1.

The HTTP app also exposes `GET /v1/models` and `GET /health`.

```python
from nano_dsv41f.vllm_v41_cpu.api import (
    NanoDeepSeekProtocolBackend,
    create_app,
)

backend = NanoDeepSeekProtocolBackend.from_pretrained(
    "/path/to/exported-checkpoint",
    tokenizer_path="/path/to/tokenizer.json",
)
app = create_app(backend)
```

The correctness-first HTTP adapter currently returns complete responses. Token-by-token HTTP streaming is deliberately left for the vLLM scheduler integration, while the protocol rendering/encoding/parsing itself is already the DeepSeek V4.1 implementation rather than an approximation.

## Public Kaggle chat notebook

`notebooks/nano_dsv41f_cpu_chat.ipynb` is the visitor-facing demo, separate from the training notebook. Before publishing it on Kaggle, attach an input containing the exported checkpoint and frozen tokenizer. A visitor can then use Kaggle's normal **Copy & Edit** flow, start a CPU session, choose **Run All**, and chat in an in-notebook `ipywidgets` message box with **Send** and **Reset** controls.

The widget keeps the same multi-turn `chat()` history used by direct Python calls and exposes thinking mode, DeepSeek V4.1 effort presets (`low`=50, `high`=75, `max`=100) and custom integers 1–100, and max output tokens. No public tunnel or separate web service is needed for the visitor chat surface. The notebook also includes an optional local FastAPI launch cell for endpoint testing inside the Kaggle runtime.

Regenerate it with:

```bash
python scripts/build_cpu_chat_notebook.py
```

For a fully offline public demo, publish a Kaggle input containing both the checkpoint/tokenizer and a wheel/source bundle plus dependency wheelhouse, then replace the notebook's Git/PyPI install cell with installation from `/kaggle/input`. Until that bundle exists, the notebook expects Kaggle Internet access for package installation.

## Current execution boundary

The CPU runtime is now cache-correct and protocol-aware, but it is not yet registered as a vLLM engine model. The next vLLM-specific step is to map the validated local/compressed/indexer cache state onto vLLM's scheduler and paged cache allocation. The DeepSeek request/response protocol layer does not need to wait for that work.

DSpark now has a Torch block-proposal path matching the JAX reference: selected
block-input features, mHC, bidirectional draft-block attention, and sequential
Markov correction. Projected context KV is inserted once per target token into a
fixed SWA ring. Experimental greedy verification accepts matching proposals and
replaces the first mismatch with the target token, preserving target-model output.
It reports proposed/verified/accepted counts, acceptance among verified tokens,
and draft time. The verifier now processes a proposal block in **one causal target
pass**. Correct proposals share target computation across token positions. On
rejection, a transaction retains the accepted prefix's local rings, compressed KV,
pending compression token, and draft features without replaying the transformer.
Only the replacement token needs a new target step. A first-position mismatch is
known from cached logits and skips the speculative target pass entirely.

Sparse retrieval breaks exact score ties by earliest key. This prevents masked
future columns from changing earlier queries' selected set, as arbitrary `topk`
ties could previously do. Tied cases can therefore differ from the old runtime's
arbitrary selection; scalar, chunked, and full-prefix paths now share the rule.

This batches token positions within **one conversation**. It does not add a
multi-request scheduler. Greedy verification is supported; distribution-preserving
sampled speculative decoding is not yet implemented. Tensor kernels can introduce
normal floating-point differences across chunk sizes/devices; tests check logits
within tolerance and greedy output equality on fixtures.

The repository's pretrain/midtrain/SFT path freezes DSpark (`train_dspark=False`).
Existing exports include its parameters but do not establish a trained drafter.
MTP is disabled by default and rejects unverified draft weights unless explicitly
requested for a diagnostic run:

```python
session = InferenceSession(model, mtp=True, allow_untrained_draft=True)
output_ids = session.generate(input_ids, max_new_tokens=32, temperature=0)
print(format_inference_stats(session.last_stats))
```

Use the [DSpark distillation notebook](dspark_distillation.md) for a dedicated frozen-backbone training stage on existing SFT data. After training a draft head, use `draft_trained=True` instead. This is a
caller assertion, not an inferred property of checkpoint tensor names. Nonzero
temperature is currently rejected in MTP mode. Default sampling is unchanged.
Quantization/QAT configuration remains checkpoint metadata; this runtime executes
loaded tensors in the selected Torch dtype rather than reproducing packed FP4/FP8
cache layouts. CUDA device plumbing exists, but T4 execution and performance have
not been validated here.

The SFT inference notebook exposes `USE_MTP` and `PREFILL_CHUNK_SIZE`. With MTP on,
it checks the verified bundle's recorded draft updates and switches chat to greedy
generation. Ordinary prefill batching is enabled with MTP off too.

For measurements on your trained model:

```bash
python scripts/benchmark_mtp_inference.py --model-dir /path/to/dspark-bundle \
    --device cpu --threads 2 --repeats 3 --output /path/to/mtp-results.json
```

Use `--device cuda` on T4. The command compares scalar/chunked prefill and real
draft proposals under sequential/batched verification, and fails on greedy output
drift. It reports end-to-end phase times, accepted counts, target calls, and CUDA
peak allocated memory. MTP is useful only when acceptance pays for drafting and
verification. `mtp_verifier="sequential"` retains the old verifier for comparisons.

Stats include `decode_target_calls`, `decode_target_input_tokens`,
`decode_target_seconds`, `mtp_draft_seconds`, and `mtp_rollback_seconds`.
`decode_batch_seconds` and `decode_batch_tokens` describe actual block rounds;
`decode_step_seconds` apportions those rounds across emitted tokens and is marked
`amortized_within_batch`. These are timings, not token arrival timestamps. Verified
tokens count only the accepted prefix and first mismatch; target input tokens also
include the speculative suffix that was computed then discarded.
