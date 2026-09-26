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

The CPU reference now has a real autoregressive cache rather than recomputing the prefix. It keeps independent persistent compressed-KV/indexer states for the context source layer (L1) and generation source layer (L3), per-layer local/SWA KV histories, and a causal pending token for ratio-2 compression.

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

The cache is intentionally an ordinary Torch correctness structure rather than a page manager. Tests compare cached dense and sparse prefill against full-prefix execution and cached greedy generation against full-prefix recomputation.

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

The widget keeps the same multi-turn `chat()` history used by direct Python calls and exposes thinking mode, DeepSeek's named reasoning-effort levels (`minimal`, `low`, `medium`, `high`, `xhigh`, `max`), and max output tokens. No public tunnel or separate web service is needed for the visitor chat surface. The notebook also includes an optional local FastAPI launch cell for endpoint testing inside the Kaggle runtime.

Regenerate it with:

```bash
python scripts/build_cpu_chat_notebook.py
```

For a fully offline public demo, publish a Kaggle input containing both the checkpoint/tokenizer and a wheel/source bundle plus dependency wheelhouse, then replace the notebook's Git/PyPI install cell with installation from `/kaggle/input`. Until that bundle exists, the notebook expects Kaggle Internet access for package installation.

## Current execution boundary

The CPU runtime is now cache-correct and protocol-aware, but it is not yet registered as a vLLM engine model. The next vLLM-specific step is to map the validated local/compressed/indexer cache state onto vLLM's scheduler and paged cache allocation. The DeepSeek request/response protocol layer does not need to wait for that work.

DSpark speculative decoding remains deferred until the ordinary vLLM cached path is integrated. Quantization/QAT configuration remains checkpoint metadata, while this CPU reference executes the loaded tensors in the selected Torch dtype rather than reproducing production packed FP4/FP8 cache layouts.
