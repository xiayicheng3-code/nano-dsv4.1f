# Interoperability, tokenizer protocol, and agent-trace data

This project keeps three concerns separate:

1. **Model semantics** — the JAX nano DeepSeek-V4.1-style architecture.
2. **Checkpoint interchange** — lossless portable safetensors + config metadata.
3. **Serving kernels** — vLLM/SGLang adapters, which may need custom kernels for nano shapes.

That separation is intentional. A config file should never make a checkpoint *look*
compatible with a stock production kernel when the tensor semantics differ.

## Tokenizer contract

The nano tokenizer stays at 32,768 entries. We copy the **DeepSeek V4.1 conversation
protocol spellings**, not the 129,280-entry production vocabulary.

The frozen contract starts with:

- `0` — `<｜begin▁of▁sentence｜>`
- `1` — `<｜end▁of▁sentence｜>`
- `2` — nano-only `<｜pad｜>`
- reserved atomic role/thinking/DSML/task tokens
- one nano-only DSpark noise token

Every other special-token ID is nano-specific. Once pretraining starts, the tokenizer must
never be changed because Engram hashes token IDs.

Before initializing the first trainable checkpoint, bind the token-sensitive config fields:

```python
from nano_dsv41f import ModelConfig, apply_tokenizer_contract

config = apply_tokenizer_contract(ModelConfig())
```

This moves Engram padding to ID 2 and gives DSpark a dedicated reserved noise token instead
of the legacy smoke-test default of ID 0 (BOS).

Train it with:

```bash
pip install -e '.[data]'
python scripts/train_tokenizer.py corpus.jsonl artifacts/tokenizer
```

The tokenizer does not add BOS/EOS automatically. DeepSeek's protocol renderer owns those
markers.

## DeepSeek prompt rendering

Do **not** store rendered `<｜User｜>` / DSML prompt strings in the dataset.

Store OpenAI-style structured examples:

```json
{
  "messages": [
    {"role": "user", "content": "Inspect the repository."},
    {
      "role": "assistant",
      "reasoning_content": "I should list the files.",
      "content": "",
      "tool_calls": [{
        "id": "call_1",
        "type": "function",
        "function": {"name": "shell", "arguments": "{\"cmd\":\"ls\"}"}
      }]
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "README.md\nsrc/"}
  ],
  "tools": [{
    "type": "function",
    "function": {
      "name": "shell",
      "description": "Run a shell command",
      "parameters": {"type": "object"}
    }
  }],
  "thinking_mode": "thinking",
  "reasoning_effort": 75
}
```

Render only at the tokenization boundary with DeepSeek's maintained `deepseek-recipe`
package (or the release's reference `encoding.py`). DeepSeek V4.1 has no Jinja chat
template. Its renderer also performs protocol-specific operations such as folding standalone
tool-result messages into user-side `<tool_result>...</tool_result>` blocks and ordering
parallel tool results by tool-call order.

Keeping the dataset structured means the same examples can later feed JAX, Transformers,
vLLM, SGLang, or a changed prompt renderer without rewriting the source corpus.

## Cleaning public agent traces

`scripts/clean_agent_traces.py` accepts JSON/JSONL with either:

- an OpenAI-style `messages` list, or
- a simple `trajectory` / `steps` list whose steps contain role messages or explicit
  action/observation pairs.

Example:

```bash
python scripts/clean_agent_traces.py raw.jsonl clean.jsonl \
  --thinking-mode thinking \
  --require-success \
  --max-tool-result-chars 65536
```

The cleaner:

- normalizes tool definitions and function arguments;
- preserves source reasoning only when it is explicitly present;
- assigns deterministic tool-call IDs when a simple trajectory omitted them;
- rejects ambiguous/orphan tool results rather than guessing;
- retains tool results as structured `role="tool"` messages;
- removes NUL bytes and optionally truncates huge environment results with an explicit marker;
- keeps reward/success/source IDs in metadata;
- emits message-level supervision metadata outside the OpenAI messages.

Before using a public trace dataset, separately verify its license and whether the source
permits training/redistribution. Do not copy credentials, private paths, API tokens, or
personally identifying logs into the training corpus.

### Recommended quality filtering

Prefer completed/successful trajectories, but keep a smaller labelled failure slice if you
want explicit recovery training. Deduplicate repeated retries and framework boilerplate.
Drop binary/base64 dumps, huge HTML pages, heartbeat logs, and repeated environment states.
Keep tool schemas and exact tool arguments/results whenever practical; if a result must be
truncated, mark the truncation instead of silently rewriting it.

Do not synthesize hidden reasoning for datasets that do not contain it. An ordinary
assistant action remains valid SFT data without an invented chain of thought.

## Canonical pretrain → mid-train → SFT use

The training lifecycle is now a strict three-stage **optimization** plan. Data can be reused
between stages, but the stage objectives are not interleaved into one fractional curriculum.
See [`training_stages.md`](training_stages.md) for the repository-level contract.

### Pretrain

Use the separate large general-text corpus and ordinary causal-LM loss. Agent/reasoning
traces are not part of the default pretraining mixture.

### Mid-train

Use the curated Q-aware document corpus plus a minority of cleaned reasoning/agent traces.
The trace view is still ordinary causal LM over the complete structured trajectory. The
starting sampler is 80% documents / 5% reasoning / 15% agent. This is also the stage where
the selective indexer-distillation objective is enabled while candidate masking remains off.

### SFT

Use cleaned structured reasoning/agent examples with assistant-only supervision. User,
system, and tool-result tokens remain context rather than prediction targets. Ordinary
document rows are excluded from the default SFT pool, and the hierarchical candidate mask
is enabled explicitly for the final retrieval behavior.

The same canonical trace record therefore supports both later stages without duplicate
cleaning: mid-training uses `token_mask`, while SFT additionally applies `sft_loss_mask`.
Evaluation/checkpoint probes can happen between stages without turning the training plan back
into the old early/middle/late-mid curriculum.

## Portable checkpoint export

The portable format uses stable keys that mirror the JAX tree:

```text
nano.embed
nano.blocks.0.attn.q_a.weight
nano.blocks.0.moe.experts.w1
...
```

Export from Python:

```python
from nano_dsv41f import export_portable_checkpoint

export_portable_checkpoint(params, "artifacts/hf", config)
```

It writes:

- `model.safetensors`
- `config.json` for the future `NanoDeepseekV41ForCausalLM` adapter
- `deepseek_v41_probe_config.json`
- `runtime_compatibility.json`
- `nano_parameter_manifest.json`
- `nano_tokenizer_contract.json`
- `generation_config.json`

`deepseek_v41_probe_config.json` is intentionally a **probe**, not a compatibility claim.
Current stock vLLM DeepSeek-V4.1 compressor kernels are specialized to the production
512-dimensional latent head with 64 rotary dimensions, while the nano model uses 64/8.
The eventual vLLM/SGLang path should therefore be an out-of-tree NanoDeepseekV41 adapter
that consumes the portable checkpoint and reuses runtime infrastructure where its tensor
semantics actually match.
