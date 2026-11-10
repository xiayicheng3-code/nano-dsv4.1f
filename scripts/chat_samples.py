#!/usr/bin/env python3
"""Side-by-side chat samples (greedy or seeded sampling) and behaviour metrics for portable nano bundles.

Each prompt is rendered with the DeepSeek V4.1 encoder in non-thinking mode (the same
format as chat SFT), decoded greedily or with seeded temperature/top-p sampling, and summarised with simple, model-agnostic
behaviour metrics. They are no substitute for reading the samples, but they make
regressions such as runaway generation, repetition or stray tool-call markup visible:

- stop_rate: replies ending with EOS before the token limit;
- mean_tokens: reply length;
- distinct2: unique / total word bigrams inside a reply (low = repetitive);
- tool_markup_rate: replies containing DSML/tool-call markup although no tools exist;
- repeat4: share of reply tokens that complete a token 4-gram already present in the reply;
- loop_rate / median_first_loop: replies that enter a loop (a run of >= 8 consecutive
  repeat4 tokens) and the reply token where the first loop starts (censored at the limit).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

PROMPTS = [
    "Hi! How are you today?",
    "What is photosynthesis? Explain it simply.",
    "Give me three tips for studying for an exam.",
    "Write a short poem about the ocean.",
    "What is the capital of France?",
    "Explain what a neural network is to a 10-year-old.",
    "How do I make a cup of tea?",
    "What are the main causes of climate change?",
    "Can you help me write a polite email asking my professor for an extension?",
    "What is the difference between weather and climate?",
    "Summarize the story of Romeo and Juliet in two sentences.",
    "Write a Python function that returns the sum of a list of numbers.",
    "Why is the sky blue?",
    "Suggest a name for a small coffee shop.",
    "What should I pack for a weekend camping trip?",
    "What is 15 + 27?",
    "Explain the water cycle.",
    "Give me a simple recipe for pancakes.",
    "What does a computer's CPU do?",
    "How can I improve my sleep?",
    "Translate 'good morning' into Spanish.",
    "What is gravity?",
    "Write a two-sentence horror story.",
    "List the planets of the solar system in order.",
    "What are some benefits of regular exercise?",
    "Describe a cat to someone who has never seen one.",
    "What is a prime number? Give an example.",
    "How do vaccines work?",
    "Tell me a fun fact about octopuses.",
    "What is the difference between a virus and a bacterium?",
    "Rewrite this sentence to sound more formal: 'gonna be late, sorry'.",
    "What is the meaning of the word 'ephemeral'?",
]
TOOL_MARKERS = ('｜DSML｜', '<tool_result>', '</tool_result>', '<｜action｜>', '<｜read_url｜>',
                '<｜query｜>')


def render_prompt(text: str) -> str:
    from deepseek_recipe import Conversation, DeepseekV41Encoding, UserMessage
    conversation = Conversation([UserMessage(text)], thinking_mode=False, tools=None,
                                reasoning_effort=None)
    return DeepseekV41Encoding().render_conversation(conversation).prompt


def repeat_fraction(tokens, n: int = 4) -> float:
    seen, repeats = set(), 0
    for t in range(n - 1, len(tokens)):
        gram = tuple(tokens[t + 1 - n:t + 1])
        repeats += gram in seen
        seen.add(gram)
    return repeats / len(tokens) if tokens else 0.0


def repeat_labels(tokens, n: int = 4) -> list[bool]:
    """labels[t] is True when tokens[t] completes an n-gram that occurred earlier."""
    seen, labels = set(), []
    for t in range(len(tokens)):
        gram = tuple(tokens[max(0, t + 1 - n):t + 1])
        labels.append(t + 1 >= n and gram in seen)
        if t + 1 >= n:
            seen.add(gram)
    return labels


def loop_runs(labels, min_run: int = 8) -> list[tuple[int, int]]:
    """[start, end) spans of >= min_run consecutive repeat labels (a loop)."""
    runs, start = [], None
    for t, flag in enumerate(list(labels) + [False]):
        if flag and start is None:
            start = t
        elif not flag and start is not None:
            if t - start >= min_run:
                runs.append((start, t))
            start = None
    return runs


def distinct2(text: str) -> float:
    words = text.split()
    pairs = list(zip(words, words[1:]))
    return len(set(pairs)) / len(pairs) if pairs else 1.0


def sample_bundle(bundle: Path, prompts, *, device: str, max_new_tokens: int,
                  temperature: float = 0.0, top_p: float = 1.0, seed: int = 0, model=None):
    from tokenizers import Tokenizer
    from nano_dsv41f.chat_protocol import EOS_TOKEN_ID
    from nano_dsv41f.vllm_v41_cpu import NanoDeepseekV41CPU
    from nano_dsv41f.vllm_v41_cpu.session import InferenceSession

    if model is None:
        model = NanoDeepseekV41CPU.from_pretrained(bundle, device=device, dtype=torch.float32)
    tok = Tokenizer.from_file(str(bundle / 'tokenizer.json'))
    session = InferenceSession(model)
    rows = []
    for i, prompt in enumerate(prompts):
        ids = tok.encode(render_prompt(prompt), add_special_tokens=False).ids
        session.reset()
        torch.manual_seed(seed + i)  # same draws per prompt for every bundle
        started = time.time()
        out = session.generate(torch.tensor([ids], device=model.device), max_new_tokens=max_new_tokens,
                               eos_token_id=EOS_TOKEN_ID, temperature=temperature, top_p=top_p)
        generated = out[0, len(ids):].tolist()
        stopped = bool(generated) and generated[-1] == EOS_TOKEN_ID
        text = tok.decode(generated[:-1] if stopped else generated, skip_special_tokens=False)
        runs = loop_runs(repeat_labels(generated))
        rows.append({'prompt': prompt, 'reply': text, 'tokens': len(generated), 'stopped': stopped,
                     'distinct2': round(distinct2(text), 3),
                     'repeat4': round(repeat_fraction(generated), 3),
                     'tool_markup': any(m in text for m in TOOL_MARKERS),
                     'first_loop': runs[0][0] if runs else len(generated), 'loops': len(runs),
                     'seconds': round(time.time() - started, 2)})
    n = len(rows)
    summary = {'stop_rate': sum(r['stopped'] for r in rows) / n,
               'mean_tokens': sum(r['tokens'] for r in rows) / n,
               'distinct2': sum(r['distinct2'] for r in rows) / n,
               'repeat4': sum(r['repeat4'] for r in rows) / n,
               'tool_markup_rate': sum(r['tool_markup'] for r in rows) / n,
               'loop_rate': sum(r['loops'] > 0 for r in rows) / n,
               'median_first_loop': float(sorted(r['first_loop'] for r in rows)[n // 2])}
    return {'bundle': str(bundle), 'summary': summary, 'samples': rows}


def to_markdown(results: dict) -> str:
    names = list(results)
    lines = ['| metric | ' + ' | '.join(names) + ' |', '|---|' + '---:|' * len(names)]
    for key in ('stop_rate', 'mean_tokens', 'distinct2', 'repeat4', 'tool_markup_rate', 'loop_rate',
                'median_first_loop'):
        if all(key in results[n]['summary'] for n in names):
            lines.append(f'| {key} | ' + ' | '.join(f"{results[n]['summary'][key]:.3f}" for n in names) + ' |')
    out = ['\n'.join(lines), '']
    for i, prompt in enumerate(results[names[0]]['samples']):
        out.append(f"### {i + 1}. {prompt['prompt']}\n")
        for n in names:
            reply = results[n]['samples'][i]['reply'].strip().replace('\n', '\n> ')
            out.append(f"**{n}**\n\n> {reply}\n")
    return '\n'.join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--bundle', action='append', required=True, help='NAME=PATH, repeatable')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--max-new-tokens', type=int, default=192)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--temperature', type=float, default=0.0, help='0 = greedy')
    p.add_argument('--top-p', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output', type=Path, required=True, help='writes NAME.json and .md')
    a = p.parse_args()
    prompts = PROMPTS[:a.limit] if a.limit else PROMPTS
    results = {}
    for spec in a.bundle:
        name, path = spec.split('=', 1)
        results[name] = sample_bundle(Path(path), prompts, device=a.device, max_new_tokens=a.max_new_tokens,
                                      temperature=a.temperature, top_p=a.top_p, seed=a.seed)
        print(name, json.dumps(results[name]['summary']), flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.with_suffix('.json').write_text(json.dumps(results, indent=2, ensure_ascii=False))
    a.output.with_suffix('.md').write_text(to_markdown(results))
    print('Wrote', a.output.with_suffix('.md'))


if __name__ == '__main__':
    main()
