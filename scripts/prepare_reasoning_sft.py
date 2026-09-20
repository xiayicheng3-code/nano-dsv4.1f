#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from nano_dsv41f.agent_data import normalize_agent_trace
from nano_dsv41f.chat_protocol import (
    ASSISTANT_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    THINK_END_TOKEN,
    THINK_START_TOKEN,
    USER_TOKEN,
)
from nano_dsv41f.reasoning_effort import (
    assign_length_guided_reasoning_effort,
    missing_reasoning_efforts,
    reasoning_effort_histogram,
    render_v41_reasoning_effort_prompt,
    tokenizer_reasoning_length,
)


DEFAULT_MIX = {"math": 0.45, "science": 0.30, "code": 0.25}


def parse_mix(value: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for piece in value.split(","):
        name, sep, weight = piece.partition(":")
        if not sep:
            raise ValueError("mix entries must be config:weight")
        out[name.strip()] = float(weight)
    total = sum(out.values())
    if not out or any(weight <= 0 for weight in out.values()):
        raise ValueError("mix weights must be positive")
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"mix weights must sum to 1.0, got {total}")
    return out


def split_think_content(text: str) -> tuple[str, str] | None:
    start = text.find("<think>")
    if start < 0:
        return None
    end = text.find("</think>", start + len("<think>"))
    if end < 0:
        return None
    reasoning = text[start + len("<think>") : end].strip()
    prefix = text[:start].strip()
    suffix = text[end + len("</think>") :].strip()
    if not reasoning or not suffix:
        return None
    final = "\n".join(part for part in (prefix, suffix) if part)
    return reasoning, final


def canonicalize_row(row: dict[str, Any], *, config_name: str) -> dict[str, Any] | None:
    messages = row.get("messages")
    # Mixture-of-Thoughts is a two-turn SFT corpus. Keeping this strict makes the
    # length check below exact for the nano V4.1 text-only renderer.
    if not isinstance(messages, list) or len(messages) != 2:
        return None
    if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
        return None
    user_content = messages[0].get("content")
    assistant_content = messages[1].get("content")
    if not isinstance(user_content, str) or not isinstance(assistant_content, str):
        return None
    split = split_think_content(assistant_content)
    if split is None:
        return None
    reasoning, final = split
    record = {
        "messages": [
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "reasoning_content": reasoning,
                "content": final,
            },
        ],
        "thinking_mode": "thinking",
        "metadata": {
            "dataset": "open-r1/Mixture-of-Thoughts",
            "config": config_name,
            "source": row.get("source"),
            "upstream_num_tokens": row.get("num_tokens"),
        },
    }
    return normalize_agent_trace(
        record,
        default_thinking_mode="thinking",
        default_reasoning_effort=50,
    )


def render_simple_v41_sft(case: dict[str, Any]) -> str:
    """Render the text-only two-turn subset used for the 4K length gate.

    Canonical JSONL remains structured; this string is only for exact token counting
    against the frozen nano tokenizer.
    """
    messages = case["messages"]
    if (
        len(messages) != 2
        or messages[0]["role"] != "user"
        or messages[1]["role"] != "assistant"
    ):
        raise ValueError("simple renderer expects one user turn and one assistant turn")
    assistant = messages[1]
    reasoning = assistant.get("reasoning_content")
    if not isinstance(reasoning, str) or not reasoning:
        raise ValueError("thinking SFT record needs explicit reasoning_content")
    effort = int(case["reasoning_effort"])
    return (
        BOS_TOKEN
        + render_v41_reasoning_effort_prompt(effort)
        + USER_TOKEN
        + messages[0]["content"]
        + ASSISTANT_TOKEN
        + THINK_START_TOKEN
        + reasoning
        + THINK_END_TOKEN
        + assistant.get("content", "")
        + EOS_TOKEN
    )


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


def core_token_count(tokenizer, case: dict[str, Any]) -> int:
    # Conservative pre-effort gate. Exact rendered length is checked after effort
    # assignment; 48 tokens covers the role/effort/special-token wrapper comfortably.
    messages = case["messages"]
    return (
        token_count(tokenizer, messages[0]["content"])
        + token_count(tokenizer, messages[1]["reasoning_content"])
        + token_count(tokenizer, messages[1].get("content", ""))
        + 48
    )


def stable_case_digest(case: dict[str, Any]) -> bytes:
    payload = json.dumps(case["messages"], ensure_ascii=False, sort_keys=True)
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).digest()


def collect_config(
    config_name: str,
    *,
    target_tokens: int,
    tokenizer,
    max_tokens: int,
    seed: int,
    shuffle_buffer: int,
    seen: set[bytes],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Reasoning SFT preparation needs datasets: pip install -e '.[data]'"
        ) from exc

    stream = load_dataset(
        "open-r1/Mixture-of-Thoughts",
        name=config_name,
        split="train",
        streaming=True,
    ).shuffle(seed=seed, buffer_size=shuffle_buffer)

    cases: list[dict[str, Any]] = []
    accepted_tokens = 0
    seen_rows = 0
    dropped_structure = 0
    dropped_too_long = 0
    dropped_duplicate = 0
    for row in stream:
        seen_rows += 1
        case = canonicalize_row(row, config_name=config_name)
        if case is None:
            dropped_structure += 1
            continue
        digest = stable_case_digest(case)
        if digest in seen:
            dropped_duplicate += 1
            continue
        approx = core_token_count(tokenizer, case)
        if approx > max_tokens:
            dropped_too_long += 1
            continue
        seen.add(digest)
        cases.append(case)
        accepted_tokens += approx
        if accepted_tokens >= target_tokens:
            break

    if accepted_tokens < target_tokens:
        raise RuntimeError(
            f"{config_name} exhausted at {accepted_tokens:,} accepted tokens; "
            f"target was {target_tokens:,}"
        )
    return cases, {
        "rows_seen": seen_rows,
        "accepted": len(cases),
        "approx_tokens": accepted_tokens,
        "dropped_structure_or_incomplete_think": dropped_structure,
        "dropped_too_long": dropped_too_long,
        "dropped_duplicate": dropped_duplicate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare DeepSeek-V4.1-style reasoning SFT JSONL with integer "
            "reasoning_effort labels measured using the frozen nano tokenizer."
        )
    )
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-tokens", type=int, default=4_000_000)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--mix",
        default=",".join(f"{k}:{v}" for k, v in DEFAULT_MIX.items()),
        help="Mixture-of-Thoughts configs and token weights",
    )
    parser.add_argument("--jitter", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--shuffle-buffer", type=int, default=5_000)
    args = parser.parse_args()

    if args.target_tokens <= 0 or args.max_tokens <= 0:
        raise SystemExit("token targets must be positive")
    mix = parse_mix(args.mix)

    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise SystemExit(
            "Reasoning SFT preparation needs tokenizers: pip install -e '.[data]'"
        ) from exc
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cases: list[dict[str, Any]] = []
    collection: dict[str, object] = {}
    seen: set[bytes] = set()
    for offset, (config_name, weight) in enumerate(mix.items()):
        target = round(args.target_tokens * weight)
        subset, stats = collect_config(
            config_name,
            target_tokens=target,
            tokenizer=tokenizer,
            max_tokens=args.max_tokens,
            seed=args.seed + 1009 * offset,
            shuffle_buffer=args.shuffle_buffer,
            seen=seen,
        )
        cases.extend(subset)
        collection[config_name] = {"target_tokens": target, **stats}
        print(config_name, stats)

    # Effort is assigned only after the frozen tokenizer has measured the actual
    # reasoning spans. This is the integer 1..100 signal, not a low/high proxy.
    cases = assign_length_guided_reasoning_effort(
        cases,
        length_fn=tokenizer_reasoning_length(tokenizer),
        length_unit="tokens",
        jitter=args.jitter,
        seed=args.seed,
        preserve_existing=False,
    )

    # Now that the effort prefix is known, apply the exact simple V4.1 two-turn
    # rendered length gate. Reassign efforts after filtering so percentiles remain
    # calibrated on the final SFT population.
    filtered: list[dict[str, Any]] = []
    dropped_exact_length = 0
    for case in cases:
        rendered_tokens = token_count(tokenizer, render_simple_v41_sft(case))
        if rendered_tokens > args.max_tokens:
            dropped_exact_length += 1
            continue
        case["metadata"]["rendered_tokens"] = rendered_tokens
        filtered.append(case)
    cases = assign_length_guided_reasoning_effort(
        filtered,
        length_fn=tokenizer_reasoning_length(tokenizer),
        length_unit="tokens",
        jitter=args.jitter,
        seed=args.seed,
        preserve_existing=False,
    )

    # A second exact count is cheap and catches any unusual tokenizer split of a
    # changed integer effort label.
    final_cases: list[dict[str, Any]] = []
    for case in cases:
        rendered_tokens = token_count(tokenizer, render_simple_v41_sft(case))
        if rendered_tokens <= args.max_tokens:
            case["metadata"]["rendered_tokens"] = rendered_tokens
            final_cases.append(case)
        else:
            dropped_exact_length += 1
    cases = final_cases

    rng = random.Random(args.seed + 41)
    rng.shuffle(cases)
    output = args.output_dir / "reasoning_sft.jsonl"
    with output.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    histogram = reasoning_effort_histogram(cases)
    total_rendered_tokens = sum(
        int(case["metadata"]["rendered_tokens"]) for case in cases
    )
    source_counts = Counter(
        str(case["metadata"].get("source")) for case in cases
    )
    manifest = {
        "format": "nano-dsv41f-reasoning-sft-v1",
        "dataset": "open-r1/Mixture-of-Thoughts",
        "configs": mix,
        "selection": {
            "target_tokens": args.target_tokens,
            "max_rendered_tokens": args.max_tokens,
            "seed": args.seed,
            "collection": collection,
            "dropped_after_exact_render_length": dropped_exact_length,
        },
        "reasoning_effort": {
            "type": "integer",
            "min": 1,
            "max": 100,
            "method": "reasoning_length_percentile_v1",
            "jitter": args.jitter,
            "covered_values": sum(1 for count in histogram.values() if count),
            "missing_values": list(missing_reasoning_efforts(cases)),
            "histogram": histogram,
        },
        "output": {
            "records": len(cases),
            "rendered_tokens": total_rendered_tokens,
            "source_counts": dict(source_counts),
            "jsonl": output.name,
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        {
            "records": len(cases),
            "rendered_tokens": total_rendered_tokens,
            "effort_coverage": manifest["reasoning_effort"]["covered_values"],
            "output": str(output),
        }
    )


if __name__ == "__main__":
    main()
