# Length-guided reasoning effort for agent traces

DeepSeek V4.1 exposes numeric reasoning effort as an integer in **[1, 100]**. The nano
pipeline keeps that integer in canonical structured records and defers prompt rendering to
the final tokenizer/DeepSeek protocol stage.

## Recommended preprocessing order

```text
raw public agent traces
        ↓
dataset-specific adapter / structural cleaning
        ↓
canonical OpenAI-style messages + tool calls
        ↓
train/freeze the nano 32K tokenizer
        ↓
measure explicit reasoning in tokenizer tokens
        ↓
assign length-guided reasoning effort 1..100
        ↓
render DeepSeek V4.1 protocol only during tokenization/packing
```

The tokenizer should be frozen before assigning token-length labels because Engram hashes
raw token IDs and because effort should track the sequence length the model actually sees.
Character length is available only as an early fallback.

## Assignment rule

`assign_length_guided_reasoning_effort`:

1. considers only `thinking_mode="thinking"` records with non-empty explicit
   `reasoning_content`;
2. measures the sum of assistant reasoning spans;
3. sorts examples by measured reasoning length;
4. maps empirical rank monotonically across integer efforts 1..100;
5. keeps one untouched **coverage anchor** for every base effort;
6. applies a small seeded jitter (default +/-2) only to duplicate percentile buckets.

With at least 100 eligible examples, the base rank mapping necessarily touches all 100
integer efforts. Protecting one anchor per bucket means the random deviation cannot remove
that coverage.

This is deliberately a *soft control label*, not a claim that effort 73 should always mean
an exact number of tokens. The jitter prevents the model from learning a brittle lookup from
prompt number to one exact target length.

## CLI

After training the tokenizer:

```bash
python scripts/assign_reasoning_effort.py \
    data/agent.clean.jsonl \
    data/agent.effort.jsonl \
    --tokenizer artifacts/tokenizer/tokenizer.json \
    --jitter 2 \
    --seed 17
```

The command prints the number of covered effort integers and any missing values. For a
small diagnostic corpus with fewer than 100 reasoning examples, full coverage is impossible
without duplicating data; the tool reports the missing values instead of silently
oversampling.

If the tokenizer has not been trained yet, omit `--tokenizer` to use character counts as a
rough temporary metric. Re-run the assignment with token lengths before actual training.

## Exact prompt form

For inspection/tests, `render_v41_reasoning_effort_prompt(37)` returns:

```text
<｜System｜>Reasoning Effort: 37 (range 1-100, the higher the value, the more thorough the reasoning)

```

In the real data pipeline, prefer storing:

```json
{"thinking_mode": "thinking", "reasoning_effort": 37}
```

and let the final DeepSeek V4.1 renderer add the prefix. That keeps the dataset semantic and
allows the same records to be reused by JAX, vLLM/SGLang adapters, or a different renderer.

## Supervision policy

Length-guided effort is derived only from reasoning text that already exists in the source.
The cleaner never invents CoT for traces that contain only actions or final answers.

A useful SFT mask remains:

```text
system/user/tool-result tokens       loss 0
assistant reasoning                  loss 1
assistant tool calls                 loss 1
assistant final content              loss 1
```

Tool outputs stay in context but are not targets produced by the model.

## Public-source hygiene

The core cleaner intentionally does not run a generic credential/secret regex pass. For the
public datasets selected for this project, source vetting and any dataset-specific cleanup
belong in the corresponding adapter. This avoids corrupting legitimate code, hashes, paths,
or tool outputs with overly broad redaction rules.
