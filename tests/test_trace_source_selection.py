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



def test_agent_catalog_replaces_legacy_nemotron_sft_splits() -> None:
    catalog = {source.key: source for source in trace_corpus.AGENT_SOURCES}
    assert set(catalog) == {
        "swe_success",
        "openthoughts_execution",
        "openseeker_correct",
        "openresearcher",
        "xlam_verified",
        "nemotron_conversational_pivot",
    }
    assert all(source.dataset != "nvidia/Nemotron-SFT-Agentic-v2" for source in catalog.values())
    assert catalog["openthoughts_execution"].license == "apache-2.0"
    assert catalog["openresearcher"].license == "mit"
    assert catalog["xlam_verified"].license.startswith("cc-by-4.0")
    assert catalog["nemotron_conversational_pivot"].dataset.endswith("Conversational-Tool-Use-Pivot-v1")
    assert abs(sum(source.weight for source in catalog.values()) - 1.0) < 1e-9


def test_openthoughts_terminal_batch_window() -> None:
    source = _source(trace_corpus.AGENT_SOURCES, "openthoughts_execution")
    row = {
        "task": "fix the failing test",
        "model": "teacher",
        "conversations": [
            {"role": "system", "content": "terminal agent"},
            {"role": "user", "content": "Fix it."},
            {"role": "assistant", "content": '{"analysis":"inspect","plan":"run tests","commands":[{"keystrokes":"pytest -q\\n","duration":0.1}],"task_complete":false}'},
            {"role": "user", "content": "1 failed"},
            {"role": "assistant", "content": '{"analysis":"patch","plan":"edit file","commands":[{"keystrokes":"sed -i s/a/b/ x.py\\n","duration":0.1}],"task_complete":false}'},
        ],
    }
    case = trace_corpus.adapt_openthoughts(source, row, 1)
    assert case is not None
    assert case["messages"][-1]["tool_calls"][0]["function"]["name"] == "terminal_batch"
    assert case["metadata"]["oracle_verified_release"] is True


def test_xlam_verified_function_call_conversion() -> None:
    source = _source(trace_corpus.AGENT_SOURCES, "xlam_verified")
    row = {
        "query": "Weather in Toronto?",
        "tools": '[{"name":"weather","description":"Get weather","parameters":{"city":{"type":"string","description":"city","required":true}}}]',
        "answers": '[{"name":"weather","arguments":{"city":"Toronto"}}]',
    }
    case = trace_corpus.adapt_xlam(source, row, 3)
    assert case is not None
    assert case["messages"][-1]["tool_calls"][0]["function"]["name"] == "weather"
    assert case["metadata"]["apigen_verified"] is True


def test_nemotron_pivot_uses_expected_action_not_whole_trajectory() -> None:
    source = _source(trace_corpus.AGENT_SOURCES, "nemotron_conversational_pivot")
    row = {
        "trajectory_id": 17,
        "responses_create_params": {
            "input": [
                {"role": "system", "content": "customer-service policy"},
                {"role": "user", "content": "Check project status."},
            ],
            "tools": [{"type":"function","name":"get_project_status","description":"status","parameters":{"type":"object"}}],
        },
        "expected_action": {"type": "function_call", "name": "get_project_status", "arguments": {"project_id": "CER-1122"}},
    }
    case = trace_corpus.adapt_nemotron_pivot(source, row, 2)
    assert case is not None
    assert case["metadata"]["expected_action_type"] == "function_call"
    assert case["messages"][-1]["tool_calls"][0]["function"]["name"] == "get_project_status"


def test_openresearcher_harmony_window_conversion() -> None:
    source = _source(trace_corpus.AGENT_SOURCES, "openresearcher")
    row = {
        "qid": 9,
        "question": "Who wrote X?",
        "answer": "Ada",
        "messages": [
            {"role": "user", "content": "Who wrote X?"},
            {"role": "assistant", "content": [
                {"channel":"analysis","text":"Need a source."},
                {"channel": "analysis", "recipient": "browser.search", "text": "X author"},
            ]},
            {"role": "browser.search", "call_id": "harmony_9_1_1", "content": "Ada wrote X."},
            {"role": "assistant", "content": [
                {"channel":"analysis","text":"Found the author."},
                {"channel":"final","text":"Ada"},
            ]},
        ],
    }
    case = trace_corpus.adapt_openresearcher(source, row, 1)
    assert case is not None
    assert case["metadata"]["reference_answer"] == "Ada"
