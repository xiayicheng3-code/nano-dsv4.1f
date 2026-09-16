from __future__ import annotations

import pytest

from nano_dsv41f.reasoning_effort import (
    assign_length_guided_reasoning_effort,
    missing_reasoning_efforts,
    render_v41_reasoning_effort_prompt,
    tokenizer_reasoning_length,
)


def _case(index: int, length: int, *, mode: str = "thinking") -> dict:
    return {
        "thinking_mode": mode,
        "messages": [
            {
                "role": "assistant",
                "content": "answer",
                "reasoning_content": "x" * length,
            }
        ],
        "metadata": {"id": f"case-{index}"},
    }


def test_v41_effort_prompt_matches_numeric_contract():
    assert render_v41_reasoning_effort_prompt(37) == (
        "<｜System｜>Reasoning Effort: 37 "
        "(range 1-100, the higher the value, the more thorough the reasoning)\n\n"
    )
    with pytest.raises(ValueError):
        render_v41_reasoning_effort_prompt(0)
    with pytest.raises(ValueError):
        render_v41_reasoning_effort_prompt(101)


def test_length_percentiles_cover_every_integer_effort_when_dataset_is_large_enough():
    cases = [_case(i, i + 1) for i in range(250)]
    assigned = assign_length_guided_reasoning_effort(cases, jitter=2, seed=17)

    assert missing_reasoning_efforts(assigned) == ()
    assert assigned[0]["reasoning_effort"] == 1
    assert assigned[-1]["reasoning_effort"] == 100
    assert assigned[0]["metadata"]["reasoning_effort_assignment"]["jitter"] == 0
    assert assigned[-1]["metadata"]["reasoning_effort_assignment"]["jitter"] == 0
    assert all(1 <= case["reasoning_effort"] <= 100 for case in assigned)
    assert any(
        case["metadata"]["reasoning_effort_assignment"]["jitter"] != 0
        for case in assigned
    )

    # Seeded assignment is reproducible and does not mutate the source records.
    again = assign_length_guided_reasoning_effort(cases, jitter=2, seed=17)
    assert [x["reasoning_effort"] for x in assigned] == [
        x["reasoning_effort"] for x in again
    ]
    assert all("reasoning_effort" not in case for case in cases)


def test_nonthinking_and_empty_reasoning_are_not_given_length_labels():
    cases = [
        _case(0, 20, mode="chat"),
        {"thinking_mode": "thinking", "messages": [{"role": "assistant", "content": "x"}]},
        _case(2, 30),
    ]
    assigned = assign_length_guided_reasoning_effort(cases)
    assert "reasoning_effort" not in assigned[0]
    assert "reasoning_effort" not in assigned[1]
    assert assigned[2]["reasoning_effort"] == 50


def test_preserve_existing_effort_excludes_record_from_percentile_fit():
    first = _case(0, 10)
    first["reasoning_effort"] = 88
    second = _case(1, 100)
    assigned = assign_length_guided_reasoning_effort(
        [first, second], preserve_existing=True
    )
    assert assigned[0]["reasoning_effort"] == 88
    assert assigned[1]["reasoning_effort"] == 50


def test_tokenizer_length_adapter_counts_encoded_ids():
    class Encoded:
        def __init__(self, ids):
            self.ids = ids

    class FakeTokenizer:
        def encode(self, text):
            return Encoded(text.split())

    case = {
        "messages": [
            {"role": "assistant", "reasoning_content": "one two three", "content": ""}
        ]
    }
    assert tokenizer_reasoning_length(FakeTokenizer())(case) == 3
