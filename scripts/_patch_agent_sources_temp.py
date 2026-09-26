from __future__ import annotations

from pathlib import Path


def replace_between(path: str, start: str, end: str, replacement: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    a = text.index(start)
    b = text.index(end, a)
    p.write_text(text[:a] + replacement.rstrip() + "\n\n" + text[b:], encoding="utf-8")


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise RuntimeError(f"{path}: expected one match for {old[:80]!r}, got {text.count(old)}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


AGENT_SOURCES = r'''AGENT_SOURCES = (
    TraceSource(
        "swe_success", "agent", "nebius/SWE-agent-trajectories", "train", 0.20,
        "swe_agent", "cc-by-4.0 + source-repository terms + upstream model-output notice",
        max_observation_chars=5000,
        provenance="Successful SWE-agent trajectories only (target=True).",
    ),
    TraceSource(
        "openthoughts_execution", "agent",
        "open-thoughts/OpenThoughts-Agent-SFT-ColdStartForRL-10K", "train", 0.20,
        "openthoughts", "apache-2.0", max_observation_chars=2400,
        provenance=(
            "OpenThoughts cold-start SFT trajectories on SWE-Smith tasks with sandbox tests; "
            "the release is oracle-verified by construction."
        ),
    ),
    TraceSource(
        "openseeker_correct", "agent",
        "PolarSeeker/OpenSeeker-v1-Data", "train", 0.15,
        "openseeker", "mit", max_observation_chars=1400,
        provenance=(
            "Official OpenSeeker v1 trajectories; keep trajectory correctness=Correct and "
            "convert the original compound tool format locally."
        ),
    ),
    TraceSource(
        "openresearcher", "agent", "OpenResearcher/OpenResearcher-Dataset", "train", 0.15,
        "openresearcher", "mit", max_observation_chars=1800,
        provenance=(
            "Long-horizon GPT-OSS-120B deep-research trajectories with native browser tools; "
            "convert a deterministic next-action window instead of forcing 100+ turns into 8K."
        ),
    ),
    TraceSource(
        "xlam_verified", "agent", "Salesforce/xlam-function-calling-60k", "train", 0.15,
        "xlam", "cc-by-4.0 + Hugging Face access conditions", max_observation_chars=1600,
        provenance=(
            "APIGen function-calling data verified by format checks, real function execution and "
            "semantic verification; official repository requires accepting its access conditions."
        ),
    ),
    TraceSource(
        "nemotron_conversational_pivot", "agent",
        "nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1", "train", 0.15,
        "nemotron_pivot", "cc-by-4.0", max_observation_chars=1800,
        provenance=(
            "NVIDIA conversational tool-use pivot: each row is a behavior-cloning context with "
            "an expected expert action rather than an unfiltered whole generated trajectory."
        ),
    ),
)'''


NEW_ADAPTERS = r'''
def _terminal_batch_tool() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "terminal_batch",
                "description": "Execute an ordered batch of terminal keystroke commands.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "commands": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "keystrokes": {"type": "string"},
                                    "duration": {"type": "number"},
                                },
                                "required": ["keystrokes"],
                            },
                        }
                    },
                    "required": ["commands"],
                },
            },
        }
    ]


def _terminal_action(content: str, *, call_id: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    analysis = str(payload.get("analysis") or "").strip()
    plan = str(payload.get("plan") or "").strip()
    reasoning = "\n\n".join(part for part in (analysis, plan) if part)
    commands = payload.get("commands")
    if isinstance(commands, list) and commands:
        clean_commands = []
        for command in commands:
            if not isinstance(command, dict) or not str(command.get("keystrokes") or "").strip():
                return None
            item = {"keystrokes": str(command["keystrokes"])}
            if command.get("duration") is not None:
                item["duration"] = command["duration"]
            clean_commands.append(item)
        return {
            "role": "assistant",
            "content": "",
            "reasoning_content": reasoning or None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "terminal_batch",
                        "arguments": json.dumps({"commands": clean_commands}, ensure_ascii=False),
                    },
                }
            ],
        }
    if payload.get("task_complete") is True:
        final = plan or analysis or "Task complete."
        return {"role": "assistant", "content": final, "reasoning_content": analysis or None}
    return None


def adapt_openthoughts(
    source: TraceSource, row: dict[str, Any], row_index: int
) -> dict[str, Any] | None:
    raw = row.get("conversations")
    if not isinstance(raw, list) or len(raw) < 3:
        return None
    prefix: list[dict[str, Any]] = []
    steps: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    pending: dict[str, Any] | None = None
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            return None
        role = str(item.get("role", "")).lower()
        content = str(item.get("content", ""))
        if role in ("system", "user") and not steps and pending is None:
            prefix.append({"role": role, "content": content})
            continue
        if role == "assistant":
            if pending is not None:
                steps.append((pending, None))
            pending = _terminal_action(content, call_id=f"terminal_{row_index}_{i}")
            if pending is None:
                return None
            continue
        if role in ("user", "tool", "environment", "observation") and pending is not None:
            calls = pending.get("tool_calls", [])
            if calls:
                result = {
                    "role": "tool",
                    "tool_call_id": calls[0]["id"],
                    "content": truncate_text(content, source.max_observation_chars),
                }
                steps.append((pending, result))
            else:
                steps.append((pending, None))
                prefix.append({"role": "user", "content": truncate_text(content, 1200)})
            pending = None
            continue
    if pending is not None:
        steps.append((pending, None))
    if not steps:
        return None

    target_index = row_index % len(steps)
    context_steps = steps[max(0, target_index - 3) : target_index]
    messages = prefix[:2]
    for action, result in context_steps:
        messages.append(action)
        if result is not None:
            messages.append(result)
    messages.append(steps[target_index][0])
    try:
        return normalize_agent_trace(
            {
                "messages": messages,
                "tools": _terminal_batch_tool(),
                "thinking_mode": "thinking",
                "metadata": {
                    "id": _identity(source, row, row_index),
                    "dataset": source.dataset,
                    "source": source.key,
                    "task": row.get("task"),
                    "trace_source": row.get("trace_source"),
                    "teacher": row.get("model"),
                    "oracle_verified_release": True,
                    "selected_step": target_index,
                    "trajectory_steps": len(steps),
                },
            },
            default_reasoning_effort=70,
            policy=CleanPolicy(max_tool_result_chars=source.max_observation_chars),
        )
    except Exception:
        return None


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return "" if value is None else str(value)
    parts: list[str] = []
    for item in value:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            text = item.get("text", item.get("content", item.get("output_text", "")))
            if isinstance(text, str) and text:
                parts.append(text)
    return "\n".join(parts)


def _responses_context(raw: Any, *, row_index: int, max_observation_chars: int) -> list[dict[str, Any]] | None:
    if not isinstance(raw, list):
        return None
    messages: list[dict[str, Any]] = []
    pending_ids: set[str] = set()
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        kind = str(item.get("type", ""))
        role = str(item.get("role", ""))
        if kind == "function_call":
            name = item.get("name")
            if not isinstance(name, str) or not name:
                continue
            call_id = str(item.get("call_id") or item.get("id") or f"resp_{row_index}_{i}")
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": _json_arguments(item.get("arguments", {})),
                            },
                        }
                    ],
                }
            )
            pending_ids.add(call_id)
            continue
        if kind in ("function_call_output", "tool_result"):
            call_id = str(item.get("call_id") or item.get("tool_call_id") or "")
            if call_id and call_id in pending_ids:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": truncate_text(
                            item.get("output", item.get("content", "")), max_observation_chars
                        ),
                    }
                )
                pending_ids.discard(call_id)
            continue
        if role == "developer":
            role = "system"
        if role in ("system", "user"):
            text = _content_text(item.get("content"))
            if text:
                messages.append({"role": role, "content": text})
            continue
        if role == "assistant":
            content = item.get("content")
            if isinstance(content, list):
                reasoning_parts, final_parts, calls = [], [], []
                for j, part in enumerate(content):
                    if not isinstance(part, dict):
                        continue
                    channel = str(part.get("channel", ""))
                    text = _content_text([part])
                    recipient = part.get("recipient", part.get("to"))
                    if recipient and recipient != "assistant":
                        call_id = f"harmony_{row_index}_{i}_{j}"
                        calls.append(
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": str(recipient),
                                    "arguments": _json_arguments(text or {}),
                                },
                            }
                        )
                        pending_ids.add(call_id)
                    elif channel == "analysis":
                        if text:
                            reasoning_parts.append(text)
                    elif text:
                        final_parts.append(text)
                if calls or final_parts or reasoning_parts:
                    msg: dict[str, Any] = {
                        "role": "assistant",
                        "content": "\n".join(final_parts),
                    }
                    if reasoning_parts:
                        msg["reasoning_content"] = "\n".join(reasoning_parts)
                    if calls:
                        msg["tool_calls"] = calls
                    messages.append(msg)
            else:
                text = _content_text(content)
                if text:
                    messages.append({"role": "assistant", "content": text})
            continue
        if role.startswith("browser.") or role == "tool":
            call_id = str(item.get("tool_call_id") or item.get("call_id") or "")
            if not call_id and len(pending_ids) == 1:
                call_id = next(iter(pending_ids))
            if call_id and call_id in pending_ids:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": truncate_text(_content_text(item.get("content")), max_observation_chars),
                    }
                )
                pending_ids.discard(call_id)
    return messages or None


def _browser_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {"type": "object", "additionalProperties": True},
            },
        }
        for name, description in (
            ("browser.search", "Search the research corpus or web index."),
            ("browser.open", "Open a result or page and inspect its content."),
            ("browser.find", "Find a pattern in the currently opened page."),
        )
    ]


def _window_to_target(messages: list[dict[str, Any]], row_index: int) -> list[dict[str, Any]] | None:
    targets = [i for i, msg in enumerate(messages) if msg.get("role") == "assistant"]
    if not targets:
        return None
    target = targets[row_index % len(targets)]
    first_context = [msg for msg in messages[:target] if msg.get("role") in ("system", "user")][:2]
    tail = messages[max(0, target - 8) : target]
    call_ids = {
        call["id"]
        for msg in tail
        for call in msg.get("tool_calls", [])
        if isinstance(call, dict) and call.get("id")
    }
    tail = [
        msg
        for msg in tail
        if msg.get("role") != "tool" or msg.get("tool_call_id") in call_ids
    ]
    out: list[dict[str, Any]] = []
    for msg in first_context + tail:
        if msg not in out:
            out.append(msg)
    out.append(messages[target])
    return out


def adapt_openresearcher(
    source: TraceSource, row: dict[str, Any], row_index: int
) -> dict[str, Any] | None:
    messages = _responses_context(
        row.get("messages"), row_index=row_index, max_observation_chars=source.max_observation_chars
    )
    if messages is None:
        return None
    window = _window_to_target(messages, row_index)
    if window is None:
        return None
    try:
        return normalize_agent_trace(
            {
                "messages": window,
                "tools": _browser_tools(),
                "thinking_mode": "thinking",
                "metadata": {
                    "id": _identity(source, row, row_index),
                    "dataset": source.dataset,
                    "source": source.key,
                    "qid": row.get("qid"),
                    "question": row.get("question"),
                    "reference_answer": row.get("answer"),
                    "selected_action_window": True,
                },
            },
            default_reasoning_effort=80,
            policy=CleanPolicy(max_tool_result_chars=source.max_observation_chars),
        )
    except Exception:
        return None


def _xlam_tools(raw: Any) -> list[dict[str, Any]] | None:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if not isinstance(raw, list) or not raw:
        return None
    tools = []
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            return None
        properties: dict[str, Any] = {}
        required: list[str] = []
        params = item.get("parameters", {})
        if isinstance(params, dict):
            for name, spec in params.items():
                if not isinstance(spec, dict):
                    continue
                properties[name] = {
                    key: value
                    for key, value in spec.items()
                    if key in ("type", "description", "enum", "items")
                }
                if spec.get("required") is True:
                    required.append(name)
        schema: dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": item["name"],
                    "description": item.get("description", ""),
                    "parameters": schema,
                },
            }
        )
    return tools


def adapt_xlam(
    source: TraceSource, row: dict[str, Any], row_index: int
) -> dict[str, Any] | None:
    query = row.get("query")
    tools = _xlam_tools(row.get("tools"))
    answers = row.get("answers")
    if isinstance(answers, str):
        try:
            answers = json.loads(answers)
        except json.JSONDecodeError:
            return None
    if not isinstance(query, str) or not query.strip() or not tools or not isinstance(answers, list):
        return None
    calls = []
    valid_names = {tool["function"]["name"] for tool in tools}
    for i, answer in enumerate(answers):
        if not isinstance(answer, dict) or answer.get("name") not in valid_names:
            return None
        calls.append(
            {
                "id": f"xlam_{row_index}_{i}",
                "type": "function",
                "function": {
                    "name": answer["name"],
                    "arguments": _json_arguments(answer.get("arguments", {})),
                },
            }
        )
    if not calls:
        return None
    try:
        return normalize_agent_trace(
            {
                "messages": [
                    {"role": "user", "content": query.strip()},
                    {"role": "assistant", "content": "", "tool_calls": calls},
                ],
                "tools": tools,
                "thinking_mode": "chat",
                "metadata": {
                    "id": _identity(source, row, row_index),
                    "dataset": source.dataset,
                    "source": source.key,
                    "apigen_verified": True,
                },
            },
            default_thinking_mode="chat",
            default_reasoning_effort=25,
        )
    except Exception:
        return None


def adapt_nemotron_pivot(
    source: TraceSource, row: dict[str, Any], row_index: int
) -> dict[str, Any] | None:
    params = row.get("responses_create_params")
    expected = row.get("expected_action")
    if not isinstance(params, dict) or not isinstance(expected, dict):
        return None
    messages = _responses_context(
        params.get("input"), row_index=row_index, max_observation_chars=source.max_observation_chars
    )
    if messages is None:
        return None
    action_type = expected.get("type")
    if action_type == "function_call":
        name = expected.get("name")
        if not isinstance(name, str) or not name:
            return None
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"pivot_{row_index}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": _json_arguments(expected.get("arguments", {})),
                        },
                    }
                ],
            }
        )
    elif action_type == "message":
        content = _content_text(expected.get("content"))
        if not content:
            return None
        messages.append({"role": "assistant", "content": content})
    else:
        return None
    tools = params.get("tools", [])
    try:
        return normalize_agent_trace(
            {
                "messages": messages,
                "tools": tools if isinstance(tools, list) else [],
                "thinking_mode": "chat",
                "metadata": {
                    "id": _identity(source, row, row_index),
                    "dataset": source.dataset,
                    "source": source.key,
                    "trajectory_id": row.get("trajectory_id"),
                    "expected_action_type": action_type,
                    "num_unique_actions": row.get("num_unique_actions"),
                    "pass_rate": row.get("pass_rate"),
                },
            },
            default_thinking_mode="chat",
            default_reasoning_effort=25,
            policy=CleanPolicy(max_tool_result_chars=source.max_observation_chars),
        )
    except Exception:
        return None
'''


def main() -> None:
    path = "scripts/prepare_trace_corpus.py"
    replace_between(path, "AGENT_SOURCES = (", "\n\n\ndef _validate_source_weights", AGENT_SOURCES)
    replace_between(path, "def adapt_nemotron(", "\ndef _openseeker_tools(", NEW_ADAPTERS)
    replace_once(
        path,
        '    "swe_agent": adapt_swe_agent,\n    "nemotron": adapt_nemotron,\n    "openseeker": adapt_openseeker,',
        '    "swe_agent": adapt_swe_agent,\n    "openthoughts": adapt_openthoughts,\n    "openresearcher": adapt_openresearcher,\n    "xlam": adapt_xlam,\n    "nemotron_pivot": adapt_nemotron_pivot,\n    "openseeker": adapt_openseeker,',
    )
    replace_once(
        path,
        'parser.add_argument("--agent-target-tokens", type=int, default=8_000_000)',
        'parser.add_argument("--agent-target-tokens", type=int, default=16_000_000)',
    )

    docs = "docs/corpus_8k_agent_data.md"
    replace_between(
        docs,
        "## Agent pool",
        "\n\n## DeepSeek V4.1 rendering",
        '''## Agent pool

Default materialization target: **16M accepted nano-tokenizer tokens**. This is deliberately larger than the earlier 8M construction target; it is a data-preparation allowance, not the SFT sampling ratio or a requirement to consume every token.

| source | weight | acceptance policy | role |
| --- | ---: | --- | --- |
| Nebius SWE-agent trajectories | 20% | `target=True` only | repository/SWE actions |
| OpenThoughts Agent ColdStartForRL | 20% | oracle/test-verified release; deterministic next-action windows | terminal execution |
| Official OpenSeeker v1 | 15% | `trajectory correctness=Correct` only | search/visit research |
| OpenResearcher | 15% | official long-horizon trajectories; deterministic next-action windows | native browser research |
| Salesforce xLAM Function Calling 60K | 15% | APIGen execution + semantic verification | generic API/function calling |
| NVIDIA Conversational Tool-Use Pivot v1 | 15% | expected expert action per behavior-cloning context | stateful conversational tools |

The old `Nemotron-SFT-Agentic-v2` `interactive_agent` and `search` splits are no longer used. The conversational capability is supplied by NVIDIA's newer pivot dataset, which turns expert trajectories into explicit context -> expected-action examples. Search diversity comes from official OpenSeeker plus OpenResearcher instead of the old Nemotron search split.

Long OpenThoughts/OpenResearcher trajectories are not naively squeezed into one 8K row. Their adapters deterministically choose a local next-action training window with bounded tool observations, preserving agent behavior while respecting the nano model's row length.

xLAM's official Hugging Face repository is CC-BY-4.0 but gated behind acknowledgement of its access conditions. Kaggle/data-prep runs therefore need an authenticated Hugging Face token for which those conditions have already been accepted; do not substitute an unofficial mirror.

**Target semantics.** The 16M agent value is only a corpus-construction default. The stage-aware SFT sampler remains separately configured at 1/3 reasoning and 2/3 agent, so increasing the materialized agent pool does not silently change the training mixture. Only assistant targets contribute SFT loss, and manifests should be used to inspect actual supervised-token counts before the final run.
''',
    )

    curriculum = "docs/corpus_curriculum.md"
    replace_once(curriculum, "`Nemotron-SFT-Agentic-v2`", "the maintained agent source catalog") if Path(curriculum).read_text().count("`Nemotron-SFT-Agentic-v2`") == 1 else None

    tests = Path("tests/test_trace_source_selection.py")
    text = tests.read_text(encoding="utf-8")
    text += r'''


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
        "expected_action": {"type":"function_call","name":"get_project_status","arguments":"{\\"project_id\\":\\"CER-1122\\"}"},
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
                {"channel":"analysis","recipient":"browser.search","text":"{\\"query\\":\\"X author\\"}"},
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
'''
    tests.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
