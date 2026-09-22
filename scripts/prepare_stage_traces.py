#!/usr/bin/env python3
"""Prepare trace shards and stamp the canonical mid-training/SFT stage views.

The underlying trace adapters and packer remain in prepare_trace_corpus.py. This
stage-aware entry point removes the legacy early/middle/late-mid sampling advice
from the top-level manifest after a successful build.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import prepare_trace_corpus as legacy
from nano_dsv41f.training_stages import (
    DEFAULT_MIDTRAIN_POOL_MIX,
    DEFAULT_SFT_POOL_MIX,
)


def _output_dir(argv: list[str]) -> Path:
    try:
        index = argv.index("--output-dir")
        return Path(argv[index + 1])
    except (ValueError, IndexError) as exc:
        raise SystemExit("--output-dir is required") from exc


def main() -> None:
    output_dir = _output_dir(sys.argv[1:])
    legacy.main()
    path = output_dir / "trace_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["recommended_training_mix"] = {
        "midtrain": dict(DEFAULT_MIDTRAIN_POOL_MIX),
        "sft": {"document": 0.0, **DEFAULT_SFT_POOL_MIX},
    }
    manifest["stage_views"] = {
        "pretrain": "not produced here; use the dedicated general pretraining corpus",
        "midtrain": "ordinary causal LM view using token_mask/segment_ids",
        "sft": "assistant-only view using token_mask AND sft_loss_mask",
    }
    manifest["stage_order"] = ["pretrain", "midtrain", "sft"]
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("updated stage-aware trace manifest:", path)


if __name__ == "__main__":
    main()
