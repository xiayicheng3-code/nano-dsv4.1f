#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from nano_dsv41f.chat_protocol import nano_v41_tokenizer_contract
from nano_dsv41f.tokenizer_training import train_byte_bpe_tokenizer


def iter_text(path: Path, field: str):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            value = row.get(field)
            if isinstance(value, str) and value:
                yield value
                continue
            if field == "text" and isinstance(row.get("messages"), list):
                for msg in row["messages"]:
                    content = msg.get("content")
                    if isinstance(content, str) and content:
                        yield content
                    reasoning = msg.get("reasoning_content")
                    if isinstance(reasoning, str) and reasoning:
                        yield reasoning


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the frozen 32K byte-BPE tokenizer used by nano-dsv4.1f."
    )
    parser.add_argument("input", type=Path, help="JSONL text/canonical-agent corpus")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--field", default="text")
    parser.add_argument("--vocab-size", type=int, default=32_768)
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args()

    contract = nano_v41_tokenizer_contract(args.vocab_size)
    path = train_byte_bpe_tokenizer(
        iter_text(args.input, args.field),
        args.output_dir,
        contract=contract,
        min_frequency=args.min_frequency,
    )
    print(f"saved tokenizer: {path}")


if __name__ == "__main__":
    main()
