from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_trace_corpus.py"
SPEC = importlib.util.spec_from_file_location("prepare_trace_corpus_source_tests", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
trace_corpus = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = trace_corpus
SPEC.loader.exec_module(trace_corpus)


def _source(sources, key: str):
    return next(source for source in sources if source.key == key)


def test_trace_catalog_uses_verified_xcoder_and_official_openseeker() -> None:
    xcoder = _source(trace_corpus.REASONING_SOURCES, "xcoder")
    assert xcoder.dataset == "IIGroup/X-Coder-SFT-376k"
    assert xcoder.split == "verified_90k"

    openseeker = _source(trace_corpus.AGENT_SOURCES, "openseeker_correct")
    assert openseeker.dataset == "PolarSeeker/OpenSeeker-v1-Data"
    assert "AmanPriyanshu" not in openseeker.dataset


def test_trace_openr1_requires_positive_verifier() -> None:
    source = _source(trace_corpus.REASONING_SOURCES, "openr1_math")
    base = {
        "problem": "Compute 2 + 2.",
        "generations": ["<think>reason</think>\n4"],
        "is_reasoning_complete": [True],
    }
    assert trace_corpus.adapt_openr1_math(
        source,
        {**base, "correctness_math_verify": [False], "correctness_llama": [False]},
        0,
    ) is None
    case = trace_corpus.adapt_openr1_math(
        source,
        {**base, "correctness_math_verify": [False], "correctness_llama": [True]},
        0,
    )
    assert case is not None
    assert case["metadata"]["llama_verify"] is True


def test_official_openseeker_trajectory_is_converted_directly() -> None:
    source = _source(trace_corpus.AGENT_SOURCES, "openseeker_correct")
    row = {
        "question": "Who wrote the example book?",
        "answer": "Ada Example",
        "number of tool calls": 1,
        "trajectory correctness": "Correct",
        "trajectory": [
            {"role": "system", "content": "tool schema here"},
            {"role": "user", "content": "Who wrote the example book?"},
            {
                "role": "assistant",
                "content": (
                    '<think>I should search.</think>'
                    '<tool_calls_begin><tool_call>{"name":"search","arguments":{"query":"example book author"}}</tool_call></tool_calls_end>'
                ),
            },
            {"role": "user", "content": "<tool_response>Ada Example wrote it.</tool_response>"},
            {
                "role": "assistant",
                "content": "<think>The source identifies the author.</think><answer>Ada Example</answer>",
            },
        ],
    }
    case = trace_corpus.adapt_openseeker(source, row, 7)
    assert case is not None
    assert case["metadata"]["dataset"] == "PolarSeeker/OpenSeeker-v1-Data"
    assert case["metadata"]["trajectory_correctness"] == "Correct"
    assert any(message.get("role") == "tool" for message in case["messages"])
    assert case["messages"][-1]["content"] == "Ada Example"

    assert trace_corpus.adapt_openseeker(
        source, {**row, "trajectory correctness": "Incorrect"}, 7
    ) is None
