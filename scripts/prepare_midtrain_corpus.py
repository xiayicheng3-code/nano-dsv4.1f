#!/usr/bin/env python3
"""Build the single 8K document corpus used by the mid-training stage.

This is the canonical stage-aware entry point. It reuses the maintained document
packer while replacing the legacy early/middle/late-mid phase tuple with the one
Q-aware mid-training phase declared in nano_dsv41f.training_stages.
"""
from __future__ import annotations

import prepare_document_corpus_8k as builder
from nano_dsv41f.training_stages import MIDTRAIN_DOCUMENT_PHASE


def _phase_by_name(name: str):
    if name != MIDTRAIN_DOCUMENT_PHASE.name:
        raise KeyError(name)
    return MIDTRAIN_DOCUMENT_PHASE


def main() -> None:
    builder.DEFAULT_PHASES = (MIDTRAIN_DOCUMENT_PHASE,)
    builder.phase_by_name = _phase_by_name
    builder.main()


if __name__ == "__main__":
    main()
