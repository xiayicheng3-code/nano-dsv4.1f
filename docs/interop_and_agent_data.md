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
- removes NUL bytes and optionally truncates huge environment results with an explicit
  marker;
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

## Mid-training and SFT from the same canonical data

One structured dataset can create several training views:

- **continued/mid-training:** causal LM over long mixed documents/trajectories;
- **agent mid-training:** keep the full environment context but weight assistant reasoning,
  actions, and answers more heavily;
- **SFT:** loss only on assistant reasoning/content/tool-call spans; user/system/tool-result
  tokens are context, not prediction targets.

For this ~122M model, avoid switching most of the token budget to agent traces immediately.
A practical first curriculum is:

1. short general/code/education warmup until loss and syntax are stable;
2. introduce a small agent-trace mixture early;
3. gradually increase high-quality agent data while keeping general/code data dominant;
4. run an early small SFT pass as a behavior check, continue mixed training, then do the
   final SFT later.

This is deliberately a curriculum, not a rigid "pretrain completely, then mid-train, then
SFT" wall. At nano scale, early agent data is useful, but too much too soon tends to teach
tool syntax and dataset quirks before the model has enough general language/code capacity.

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
