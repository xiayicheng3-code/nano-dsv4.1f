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

```bash
pip install -e '.[cpu]'
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

## Generation

```python
output_ids = model.generate(
    input_ids,
    max_new_tokens=32,
    temperature=0.0,
)
```

The initial generation path recomputes the full prefix for each token. That is intentional: it establishes an executable CPU reference before adding vLLM's paged-cache/scheduler integration.

## Current execution boundary

This first CPU milestone is a semantic runtime, not yet a registered vLLM engine model. The next serving step is to wrap the validated operators in an out-of-tree `vllm.general_plugins` model and replace prefix recomputation with V4.1-aware paged SWA/compressed-KV/indexer caches.

DSpark speculative decoding is also deferred until ordinary autoregressive decode is cache-correct. Quantization/QAT configuration remains checkpoint metadata, while this CPU reference executes the loaded tensors in the selected Torch dtype rather than reproducing production packed FP4/FP8 cache layouts.
