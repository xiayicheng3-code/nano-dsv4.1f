#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from nano_dsv41f.agent_data import (
    CleanPolicy,
    TraceNormalizationError,
    keep_trace,
    normalize_agent_trace,
)


def iter_records(path: Path) -> Iterable[dict]:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if not stripped:
        return
    if stripped.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("JSON input must be an array of records")
        for record in data:
            yield record
        return
    if stripped.startswith("{") and "\n" not in stripped.rstrip():
        yield json.loads(text)
        return
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at line {line_no}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize public agent traces into DeepSeek-V4.1-ready structured JSONL."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--thinking-mode", choices=("thinking", "chat"), default="thinking"
    )
    parser.add_argument("--reasoning-effort", type=int, default=75)
    parser.add_argument("--max-tool-result-chars", type=int, default=65_536)
    parser.add_argument("--require-success", action="store_true")
    parser.add_argument("--min-reward", type=float)
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Drop malformed traces instead of stopping on the first one.",
    )
    args = parser.parse_args()

    policy = CleanPolicy(max_tool_result_chars=args.max_tool_result_chars)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    dropped_filter = 0
    dropped_invalid = 0
    with args.output.open("w", encoding="utf-8") as out:
        for index, record in enumerate(iter_records(args.input)):
            try:
                case = normalize_agent_trace(
                    record,
                    default_thinking_mode=args.thinking_mode,
                    default_reasoning_effort=args.reasoning_effort,
                    policy=policy,
                )
            except TraceNormalizationError as exc:
                if not args.skip_invalid:
                    raise
                dropped_invalid += 1
                print(f"skip invalid record {index}: {exc}")
                continue

            if not keep_trace(
                case,
                require_success=args.require_success,
                min_reward=args.min_reward,
            ):
                dropped_filter += 1
                continue
            out.write(json.dumps(case, ensure_ascii=False) + "\n")
            kept += 1

    print(
        {
            "kept": kept,
            "dropped_filter": dropped_filter,
            "dropped_invalid": dropped_invalid,
            "output": str(args.output),
        }
    )


if __name__ == "__main__":
    main()
