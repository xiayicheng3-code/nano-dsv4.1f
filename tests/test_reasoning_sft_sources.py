from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_reasoning_sft.py"
SPEC = importlib.util.spec_from_file_location("prepare_reasoning_sft", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
reasoning_sft = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reasoning_sft)


def test_reasoning_sources_are_explicit_and_permissive() -> None:
    catalog = reasoning_sft.SOURCE_CATALOG
    assert set(catalog) == {"math", "science", "code"}
    assert catalog["math"].dataset == "open-r1/OpenR1-Math-220k"
    assert catalog["science"].dataset == "TianHongZXY/CHIMERA"
    assert catalog["code"].dataset == "IIGroup/X-Coder-SFT-376k"
    assert {source.license for source in catalog.values()} <= {"apache-2.0", "mit"}
    assert all("Mixture-of-Thoughts" not in source.dataset for source in catalog.values())


def test_openr1_math_prefers_complete_verified_generation() -> None:
    source = reasoning_sft.SOURCE_CATALOG["math"]
    row = {
        "problem": "Compute 1 + 1.",
        "uuid": "math-1",
        "source": "unit-test",
        "problem_type": "Algebra",
        "generations": [
            "<think>bad reasoning</think>\n3",
            "<think>good reasoning</think>\n2",
        ],
        "is_reasoning_complete": [True, True],
        "correctness_math_verify": [False, True],
    }
    case = reasoning_sft.adapt_openr1_math(source, row)
    assert case is not None
    assert case["messages"][1]["reasoning_content"] == "good reasoning"
    assert case["messages"][1]["content"] == "2"
    assert case["metadata"]["generation_index"] == 1


def test_chimera_science_requires_verified_science_subject() -> None:
    source = reasoning_sft.SOURCE_CATALOG["science"]
    base = {
        "question": "Why does the orbit remain stable?",
        "solution": "Balance gravity and orbital motion.",
        "answer": "The centripetal acceleration is supplied by gravity.",
        "topic": "Orbital mechanics",
        "index": 7,
        "correctness": True,
    }
    physics = reasoning_sft.adapt_chimera_science(source, {**base, "subject": "Physics"})
    assert physics is not None
    assert physics["metadata"]["subject"] == "Physics"
    assert reasoning_sft.adapt_chimera_science(
        source, {**base, "subject": "Mathematics"}
    ) is None
    assert reasoning_sft.adapt_chimera_science(
        source, {**base, "subject": "Chemistry", "correctness": False}
    ) is None


def test_xcoder_requires_explicit_complete_think_span() -> None:
    source = reasoning_sft.SOURCE_CATALOG["code"]
    case = reasoning_sft.adapt_xcoder(
        source,
        {
            "query": "Return the maximum element.",
            "response": "<think>Scan once and track the best value.</think>\nUse a linear scan.",
        },
    )
    assert case is not None
    assert case["messages"][1]["reasoning_content"] == "Scan once and track the best value."
    assert case["messages"][1]["content"] == "Use a linear scan."
    assert reasoning_sft.adapt_xcoder(
        source,
        {"query": "Return the maximum element.", "response": "Use a linear scan."},
    ) is None
