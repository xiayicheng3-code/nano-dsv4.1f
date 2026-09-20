import ast
import json
from pathlib import Path

import numpy as np

from nano_dsv41f.chat_protocol import ASSISTANT_TOKEN, EOS_TOKEN_ID, SPECIAL_TOKENS
from nano_dsv41f.trace_corpus import (
    DEFAULT_TRACE_Q_BANDS,
    DEFAULT_TRACE_SEQ_LEN,
    assistant_sft_loss_mask,
    parse_tagged_json,
    strip_xml_tag,
)


ROOT = Path(__file__).resolve().parents[1]


def test_8k_trace_defaults_cover_full_q_range():
    assert DEFAULT_TRACE_SEQ_LEN == 8192
    assert DEFAULT_TRACE_Q_BANDS[0] == 640
    assert DEFAULT_TRACE_Q_BANDS[-1] == 8192
    assert all(b > a for a, b in zip(DEFAULT_TRACE_Q_BANDS, DEFAULT_TRACE_Q_BANDS[1:]))


def test_assistant_loss_mask_supervises_only_assistant_span():
    assistant = SPECIAL_TOKENS.index(ASSISTANT_TOKEN)
    ids = [3, 17, 4, 91, assistant, 8, 92, EOS_TOKEN_ID, 4, 11]
    mask = assistant_sft_loss_mask(ids)
    np.testing.assert_array_equal(mask, [0, 0, 0, 0, 0, 1, 1, 1, 0, 0])


def test_tag_helpers_parse_search_fsm_payloads():
    assert strip_xml_tag("<think> inspect sources </think>", "think") == "inspect sources"
    payload = parse_tagged_json(
        '<tool_call>{"name":"search","arguments":{"query":["x"]}}</tool_call>',
        "tool_call",
    )
    assert payload["name"] == "search"
    assert payload["arguments"]["query"] == ["x"]


def test_new_data_scripts_and_notebook_are_syntactically_valid():
    for relative in (
        "scripts/prepare_document_corpus_8k.py",
        "scripts/prepare_trace_corpus.py",
    ):
        ast.parse((ROOT / relative).read_text(encoding="utf-8"), filename=relative)
    notebook = json.loads(
        (ROOT / "notebooks/nano_dsv41f_prepare_8k_data.ipynb").read_text(encoding="utf-8")
    )
    assert notebook["nbformat"] == 4
