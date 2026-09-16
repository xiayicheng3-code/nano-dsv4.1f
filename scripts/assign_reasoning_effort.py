#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from nano_dsv41f.reasoning_effort import (
    assign_length_guided_reasoning_effort,
    missing_reasoning_efforts,
    reasoning_effort_histogram,
    tokenizer_reasoning_length,
)


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_no}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row {line_no} must be an object")
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Assign DeepSeek-V4.1 integer reasoning effort (1..100) from empirical "
            "reasoning-length percentiles. Run after the tokenizer is frozen."
        )
    )
    parser.add_argument("input", type=Path, help="canonical normalized agent JSONL")
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        help=(
            "tokenizer.json produced by train_tokenizer.py. If omitted, character length "
            "is used as a rough pre-tokenizer fallback."
        ),
    )
    parser.add_argument(
        "--jitter",
        type=int,
        default=2,
        help="seeded +/- effort-point jitter for duplicate percentile buckets (default 2)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--preserve-existing",
        action="store_true",
        help="leave records that already contain reasoning_effort untouched",
    )
    args = parser.parse_args()

    cases = read_jsonl(args.input)
    if args.tokenizer is not None:
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise SystemExit(
                "token-length assignment needs the data extra: pip install -e '.[data]'"
            ) from exc
        tokenizer = Tokenizer.from_file(str(args.tokenizer))
        length_fn = tokenizer_reasoning_length(tokenizer)
        length_unit = "tokens"
    else:
        length_fn = None
        length_unit = "characters"

    kwargs = {
        "length_unit": length_unit,
        "jitter": args.jitter,
        "seed": args.seed,
        "preserve_existing": args.preserve_existing,
    }
    if length_fn is not None:
        kwargs["length_fn"] = length_fn

    assigned = assign_length_guided_reasoning_effort(cases, **kwargs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in assigned:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    histogram = reasoning_effort_histogram(assigned)
    missing = missing_reasoning_efforts(assigned)
    print(
        {
            "records": len(assigned),
            "length_unit": length_unit,
            "effort_min": min((k for k, v in histogram.items() if v), default=None),
            "effort_max": max((k for k, v in histogram.items() if v), default=None),
            "missing_efforts": list(missing),
            "covered_efforts": 100 - len(missing),
            "output": str(args.output),
        }
    )


if __name__ == "__main__":
    main()
