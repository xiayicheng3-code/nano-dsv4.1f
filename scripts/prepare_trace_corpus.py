#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Callable

import numpy as np

from nano_dsv41f.agent_data import CleanPolicy, normalize_agent_trace
from nano_dsv41f.chat_protocol import PAD_TOKEN_ID, nano_v41_tokenizer_contract
from nano_dsv41f.corpus_curriculum import aggregate_q_metrics, pack_length_indices, q_row_metrics
from nano_dsv41f.reasoning_effort import (
    assign_length_guided_reasoning_effort,
    missing_reasoning_efforts,
    reasoning_effort_histogram,
    reasoning_text,
)
from nano_dsv41f.sequence_packing import pack_token_sequences
from nano_dsv41f.trace_corpus import (
    DEFAULT_TRACE_Q_BANDS,
    DEFAULT_TRACE_SEQ_LEN,
    TokenizedTrace,
    assistant_sft_loss_mask,
    parse_tagged_json,
    render_case_v41,
    strip_xml_tag,
    truncate_text,
)


SCIENCE_SUBJECTS = frozenset({"Physics", "Chemistry", "Biology"})


@dataclass(frozen=True)
class TraceSource:
    key: str
    pool: str
    dataset: str
    split: str
    weight: float
    adapter: str
    license: str
    config: str | None = None
    max_observation_chars: int = 4000
    provenance: str = ""


REASONING_SOURCES = (
    TraceSource(
        "openr1_math",
        "reasoning",
        "open-r1/OpenR1-Math-220k",
        "train",
        0.45,
        "openr1_math",
        "apache-2.0",
        config="default",
        provenance=(
            "Apache-2.0 NuminaMath-1.5 problems with DeepSeek-R1 reasoning traces; "
            "adapter keeps a complete upstream-verified generation."
        ),
    ),
    TraceSource(
        "chimera_science",
        "reasoning",
        "TianHongZXY/CHIMERA",
        "train",
        0.30,
        "chimera_science",
        "apache-2.0",
        config="Qwen3-235B-2507",
        provenance=(
            "Fully synthetic CHIMERA; adapter keeps correctness=True rows in Physics, "
            "Chemistry and Biology only."
        ),
    ),
    TraceSource(
        "xcoder",
        "reasoning",
        "IIGroup/X-Coder-SFT-376k",
        "verified_90k",
        0.25,
        "xcoder",
        "mit",
        provenance=(
            "Fully synthetic competitive-programming reasoning SFT data; use verified_90k, "
            "which upstream describes as all verified high-quality solutions."
        ),
    ),
)

AGENT_SOURCES = (
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
)




def _validate_source_weights(sources: tuple[TraceSource, ...]) -> None:
    total = sum(source.weight for source in sources)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"trace source weights must sum to 1.0, got {total}")


_validate_source_weights(REASONING_SOURCES)
_validate_source_weights(AGENT_SOURCES)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _identity(source: TraceSource, row: dict[str, Any], fallback: int) -> str:
    for key in ("id", "uuid", "instance_id", "qid", "task_id", "index"):
        value = row.get(key)
        if value not in (None, ""):
            return f"{source.key}:{key}:{value}"
    return f"{source.key}:stream:{fallback}"


def _split_think(text: str) -> tuple[str, str] | None:
    start = text.find("<think>")
    end = text.find("</think>", start + 7) if start >= 0 else -1
    if start < 0 or end < 0:
        return None
    reasoning = text[start + 7 : end].strip()
    final = (text[:start] + "\n" + text[end + 8 :]).strip()
    return (reasoning, final) if reasoning and final else None


def _reasoning_case(
    source: TraceSource,
    row: dict[str, Any],
    row_index: int,
    *,
    user: Any,
    reasoning: Any,
    final: Any,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not all(isinstance(x, str) and x.strip() for x in (user, reasoning, final)):
        return None
    return normalize_agent_trace(
        {
            "messages": [
                {"role": "user", "content": user.strip()},
                {
                    "role": "assistant",
                    "reasoning_content": reasoning.strip(),
                    "content": final.strip(),
                },
            ],
            "thinking_mode": "thinking",
            "metadata": {
                "id": _identity(source, row, row_index),
                "dataset": source.dataset,
                "source": source.key,
                "upstream_license": source.license,
                **(metadata or {}),
            },
        },
        default_reasoning_effort=75,
    )


def adapt_openr1_math(
    source: TraceSource, row: dict[str, Any], row_index: int
) -> dict[str, Any] | None:
    user = row.get("problem")
    if not isinstance(user, str) or not user.strip():
        messages = row.get("messages")
        if isinstance(messages, list) and messages and isinstance(messages[0], dict):
            user = messages[0].get("content")
    generations = row.get("generations")
    complete = row.get("is_reasoning_complete")
    math_verified = row.get("correctness_math_verify")
    llama_verified = row.get("correctness_llama")
    if (
        not isinstance(user, str)
        or not isinstance(generations, list)
        or not isinstance(complete, list)
    ):
        return None
    for i, generation in enumerate(generations):
        if i >= len(complete) or complete[i] is not True:
            continue
        math_ok = (
            isinstance(math_verified, list)
            and i < len(math_verified)
            and math_verified[i] is True
        )
        llama_ok = (
            isinstance(llama_verified, list)
            and i < len(llama_verified)
            and llama_verified[i] is True
        )
        if not (math_ok or llama_ok):
            continue
        if not isinstance(generation, str):
            continue
        split = _split_think(generation)
        if split is None:
            continue
        reasoning, final = split
        return _reasoning_case(
            source,
            row,
            row_index,
            user=user,
            reasoning=reasoning,
            final=final,
            metadata={
                "upstream_source": row.get("source"),
                "problem_type": row.get("problem_type"),
                "generation_index": i,
                "reasoning_complete": True,
                "math_verify": math_ok,
                "llama_verify": llama_ok,
            },
        )
    return None


def adapt_chimera_science(
    source: TraceSource, row: dict[str, Any], row_index: int
) -> dict[str, Any] | None:
    if row.get("correctness") is not True or row.get("subject") not in SCIENCE_SUBJECTS:
        return None
    return _reasoning_case(
        source,
        row,
        row_index,
        user=row.get("question"),
        reasoning=row.get("solution"),
        final=row.get("answer"),
        metadata={
            "subject": row.get("subject"),
            "topic": row.get("topic"),
            "correctness": True,
        },
    )


def adapt_xcoder(
    source: TraceSource, row: dict[str, Any], row_index: int
) -> dict[str, Any] | None:
    query, response = row.get("query"), row.get("response")
    if not isinstance(query, str) or not isinstance(response, str):
        return None
    split = _split_think(response)
    if split is None:
        return None
    reasoning, final = split
    return _reasoning_case(
        source,
        row,
        row_index,
        user=query,
        reasoning=reasoning,
        final=final,
        metadata={"upstream_split": source.split},
    )


def _swe_tool() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "swe_environment",
                "description": (
                    "Execute one command in the SWE-agent repository environment. The command may "
                    "be bash or one of the environment's documented navigation/editing commands."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}},
                    "required": ["cmd"],
                },
            },
        }
    ]


_FENCE_RE = re.compile(r"```(?:[^\n]*\n)?(.*?)```", re.DOTALL)


def _split_swe_ai_turn(text: str) -> tuple[str, str | None]:
    matches = list(_FENCE_RE.finditer(text))
    if not matches:
        return text.strip(), None
    last = matches[-1]
    action = last.group(1).strip()
    reasoning = (text[: last.start()] + text[last.end() :]).strip()
    return reasoning, action or None


def adapt_swe_agent(
    source: TraceSource, row: dict[str, Any], row_index: int
) -> dict[str, Any] | None:
    if row.get("target") is not True:
        return None
    raw = row.get("trajectory")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if not isinstance(raw, list) or not raw:
        return None

    messages: list[dict[str, Any]] = []
    pending_call: str | None = None
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            return None
        role = str(item.get("role", "")).lower()
        if role == "system":
            system = item.get("system_prompt") or item.get("text") or ""
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "This successful SWE-agent demonstration originally wrote environment commands "
                        "in fenced blocks. In this DeepSeek training view, invoke the same commands "
                        "through the swe_environment tool instead.\n\n" + str(system)
                    ),
                }
            )
            continue
        if role == "user":
            text = truncate_text(item.get("text") or "", source.max_observation_chars)
            if pending_call is not None:
                messages.append({"role": "tool", "tool_call_id": pending_call, "content": text})
                pending_call = None
            elif text:
                messages.append({"role": "user", "content": text})
            continue
        if role in ("ai", "assistant"):
            if pending_call is not None:
                return None
            text = str(item.get("text") or "")
            reasoning, action = _split_swe_ai_turn(text)
            if action is None:
                if text.strip():
                    messages.append({"role": "assistant", "content": text.strip()})
                continue
            call_id = f"swe_{row_index}_{i}"
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": reasoning or None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "swe_environment",
                                "arguments": json.dumps({"cmd": action}, ensure_ascii=False),
                            },
                        }
                    ],
                }
            )
            pending_call = call_id
            continue
        return None

    if pending_call is not None or len(messages) < 3:
        return None
    try:
        return normalize_agent_trace(
            {
                "messages": messages,
                "tools": _swe_tool(),
                "thinking_mode": "thinking",
                "metadata": {
                    "id": _identity(source, row, row_index),
                    "dataset": source.dataset,
                    "source": source.key,
                    "instance_id": row.get("instance_id"),
                    "model_name": row.get("model_name"),
                    "target": True,
                    "exit_status": row.get("exit_status"),
                },
            },
            default_reasoning_effort=75,
            policy=CleanPolicy(max_tool_result_chars=source.max_observation_chars),
        )
    except Exception:
        return None



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


def _openseeker_tools() -> list[dict[str, Any]]:
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
            ("search", "Search the web for relevant sources."),
            ("visit", "Visit and read a webpage for a research goal."),
        )
    ]


def _json_arguments(value: Any) -> str:
    if isinstance(value, str):
        try:
            json.loads(value)
            return value
        except json.JSONDecodeError:
            return json.dumps({"input": value}, ensure_ascii=False)
    return json.dumps(value if value is not None else {}, ensure_ascii=False, separators=(",", ":"))


def adapt_openseeker(
    source: TraceSource, row: dict[str, Any], row_index: int
) -> dict[str, Any] | None:
    # Parse PolarSeeker/OpenSeeker-v1-Data directly. Upstream stores calls inside
    # assistant XML and returns the corresponding tool outputs in the next user turn.
    correctness = row.get("trajectory correctness", row.get("trajectory_correctness", ""))
    if str(correctness).lower() != "correct":
        return None
    raw = row.get("trajectory")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if not isinstance(raw, list) or len(raw) < 3:
        return None

    def blocks(text: str, tag: str) -> list[str]:
        return [
            value.strip()
            for value in re.findall(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.DOTALL)
        ]

    messages: list[dict[str, Any]] = []
    pending_call_ids: list[str] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            return None
        role = str(item.get("role", "")).lower()
        content = str(item.get("content", ""))

        if role == "system":
            if not messages:
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "You are a deep-research assistant. Use the provided search and "
                            "visit tools to gather evidence and synthesize an accurate answer."
                        ),
                    }
                )
            continue

        if role == "user":
            tool_responses = blocks(content, "tool_response")
            if pending_call_ids:
                if len(tool_responses) != len(pending_call_ids):
                    return None
                for call_id, response in zip(pending_call_ids, tool_responses):
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": truncate_text(response, source.max_observation_chars),
                        }
                    )
                pending_call_ids = []
            elif tool_responses:
                return None
            elif content.strip():
                messages.append({"role": "user", "content": content.strip()})
            continue

        if role != "assistant" or pending_call_ids:
            return None

        reasoning_blocks = blocks(content, "think")
        if len(reasoning_blocks) != 1 or not reasoning_blocks[0]:
            return None
        reasoning = reasoning_blocks[0]
        call_blocks = blocks(content, "tool_call")
        answer_blocks = blocks(content, "answer")
        if call_blocks and answer_blocks:
            return None

        if call_blocks:
            tool_calls: list[dict[str, Any]] = []
            for j, raw_call in enumerate(call_blocks):
                try:
                    payload = json.loads(raw_call)
                except json.JSONDecodeError:
                    return None
                name = payload.get("name")
                if name not in {"search", "visit"}:
                    return None
                call_id = f"search_{row_index}_{i}_{j}"
                tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": _json_arguments(
                                payload.get("arguments", payload.get("parameters", {}))
                            ),
                        },
                    }
                )
                pending_call_ids.append(call_id)
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": reasoning,
                    "tool_calls": tool_calls,
                }
            )
            continue

        if len(answer_blocks) != 1 or not answer_blocks[0]:
            return None
        messages.append(
            {
                "role": "assistant",
                "reasoning_content": reasoning,
                "content": answer_blocks[0],
            }
        )

    if pending_call_ids:
        return None
    if not messages or messages[-1].get("role") != "assistant":
        return None
    if not str(messages[-1].get("content", "")).strip():
        return None
    try:
        return normalize_agent_trace(
            {
                "messages": messages,
                "tools": _openseeker_tools(),
                "thinking_mode": "thinking",
                "metadata": {
                    "id": _identity(source, row, row_index),
                    "dataset": source.dataset,
                    "source": source.key,
                    "trajectory_correctness": "Correct",
                    "number_of_tool_calls": row.get("number of tool calls"),
                    "expected_answer": row.get("answer"),
                },
            },
            default_reasoning_effort=75,
            policy=CleanPolicy(max_tool_result_chars=source.max_observation_chars),
        )
    except Exception:
        return None


ADAPTERS: dict[str, Callable[[TraceSource, dict[str, Any], int], dict[str, Any] | None]] = {
    "openr1_math": adapt_openr1_math,
    "chimera_science": adapt_chimera_science,
    "xcoder": adapt_xcoder,
    "swe_agent": adapt_swe_agent,
    "openthoughts": adapt_openthoughts,
    "openresearcher": adapt_openresearcher,
    "xlam": adapt_xlam,
    "nemotron_pivot": adapt_nemotron_pivot,
    "openseeker": adapt_openseeker,
}


def load_stream(source: TraceSource, *, seed: int, shuffle_buffer: int):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Trace preparation requires pip install -e '.[data]'") from exc
    kwargs: dict[str, Any] = {"path": source.dataset, "split": source.split, "streaming": True}
    if source.config is not None:
        kwargs["name"] = source.config
    return load_dataset(**kwargs).shuffle(seed=seed, buffer_size=shuffle_buffer)


def _batch_encode(tokenizer, texts: list[str]):
    return tokenizer.encode_batch(texts, add_special_tokens=False)


def _prepare_collection_batch(
    cases: list[dict[str, Any]], *, tokenizer, seq_len: int, seen_prompts: set[bytes]
) -> tuple[list[tuple[dict[str, Any], int]], Counter[str]]:
    stats: Counter[str] = Counter()
    rendered, rendered_cases = [], []
    for case in cases:
        try:
            rendered.append(render_case_v41(case))
            rendered_cases.append(case)
        except Exception:
            stats["render_error"] += 1
    if not rendered:
        return [], stats
    encodings = _batch_encode(tokenizer, rendered)
    reasoning_encodings = _batch_encode(tokenizer, [reasoning_text(case) for case in rendered_cases])
    accepted: list[tuple[dict[str, Any], int]] = []
    for case, prompt, enc, renc in zip(rendered_cases, rendered, encodings, reasoning_encodings):
        digest = hashlib.blake2b(prompt.encode("utf-8"), digest_size=16).digest()
        if digest in seen_prompts:
            stats["duplicate"] += 1
            continue
        if len(enc.ids) > seq_len:
            stats["too_long"] += 1
            continue
        if not assistant_sft_loss_mask(enc.ids).any():
            stats["no_supervision"] += 1
            continue
        seen_prompts.add(digest)
        case.setdefault("metadata", {})["_reasoning_tokens"] = len(renc.ids)
        accepted.append((case, len(enc.ids)))
    return accepted, stats


def collect_source_cases(
    source: TraceSource,
    *,
    target_tokens: int,
    tokenizer,
    seq_len: int,
    seed: int,
    shuffle_buffer: int,
    tokenize_batch_size: int,
    seen_prompts: set[bytes],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    stream = load_stream(source, seed=seed, shuffle_buffer=shuffle_buffer)
    adapter = ADAPTERS[source.adapter]
    cases: list[dict[str, Any]] = []
    batch: list[dict[str, Any]] = []
    accepted_tokens = rows_seen = rows_rejected = 0
    drops: Counter[str] = Counter()

    def flush() -> bool:
        nonlocal batch, accepted_tokens
        prepared, stats = _prepare_collection_batch(
            batch, tokenizer=tokenizer, seq_len=seq_len, seen_prompts=seen_prompts
        )
        drops.update(stats)
        batch = []
        for case, n_tokens in prepared:
            cases.append(case)
            accepted_tokens += n_tokens
            if accepted_tokens >= target_tokens:
                return True
        return False

    for row_index, row in enumerate(stream):
        rows_seen += 1
        case = adapter(source, row, row_index)
        if case is None:
            rows_rejected += 1
            continue
        batch.append(case)
        if len(batch) >= tokenize_batch_size and flush():
            break
    else:
        if batch:
            flush()

    exhausted_before_target = accepted_tokens < target_tokens
    if exhausted_before_target and source.pool != "agent":
        raise RuntimeError(
            f"{source.key} exhausted at {accepted_tokens:,} tokens; target={target_tokens:,}; "
            f"drops={dict(drops)}"
        )
    return cases, {
        "rows_seen": rows_seen,
        "rows_adapter_or_quality_rejected": rows_rejected,
        "accepted_cases": len(cases),
        "accepted_tokens_before_effort_relabel": accepted_tokens,
        "exhausted_before_target": exhausted_before_target,
        **{f"dropped_{k}": int(v) for k, v in drops.items()},
    }


def assign_efforts_and_tokenize(
    cases: list[dict[str, Any]], *, tokenizer, seq_len: int, batch_size: int, seed: int, jitter: int
) -> tuple[list[TokenizedTrace], dict[str, int]]:
    cases = assign_length_guided_reasoning_effort(
        cases,
        length_fn=lambda case: int(case.get("metadata", {}).get("_reasoning_tokens", 0)),
        length_unit="tokens",
        jitter=jitter,
        seed=seed,
        preserve_existing=False,
    )
    traces: list[TokenizedTrace] = []
    dropped_after_effort = render_errors = 0
    for start in range(0, len(cases), batch_size):
        group = cases[start : start + batch_size]
        prompts, valid_cases = [], []
        for case in group:
            try:
                prompts.append(render_case_v41(case))
                valid_cases.append(case)
            except Exception:
                render_errors += 1
        encodings = _batch_encode(tokenizer, prompts) if prompts else []
        for case, enc in zip(valid_cases, encodings):
            if len(enc.ids) > seq_len:
                dropped_after_effort += 1
                continue
            loss_mask = assistant_sft_loss_mask(enc.ids)
            if not loss_mask.any():
                continue
            metadata = dict(case.get("metadata", {}))
            metadata.pop("_reasoning_tokens", None)
            tool_calls = sum(
                len(msg.get("tool_calls", ()))
                for msg in case.get("messages", ())
                if msg.get("role") == "assistant"
            )
            thinking = case.get("thinking_mode") == "thinking"
            traces.append(
                TokenizedTrace(
                    tokens=np.asarray(enc.ids, dtype=np.uint16),
                    sft_loss_mask=loss_mask,
                    source=str(metadata.get("source", "unknown")),
                    reasoning_effort=int(case.get("reasoning_effort", 75)) if thinking else 0,
                    tool_calls=tool_calls,
                    metadata=metadata,
                )
            )
    return traces, {
        "dropped_after_effort_relabel": dropped_after_effort,
        "render_errors_after_effort_relabel": render_errors,
    }


def write_trace_shards(
    out_dir: Path,
    *,
    traces: list[TokenizedTrace],
    seq_len: int,
    shard_rows: int,
    query_budget: int,
    q_threshold: int,
    q_band_edges: tuple[int, ...],
    seed: int,
    compress: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lengths = [int(trace.tokens.size) for trace in traces]
    rows = pack_length_indices(
        lengths,
        seq_len=seq_len,
        alignment=2,
        q_aware=True,
        query_budget=query_budget,
        q_threshold=q_threshold,
        band_edges=q_band_edges,
        candidate_window=256,
        seed=seed,
    )
    random.Random(seed + 91).shuffle(rows)
    source_names = sorted({trace.source for trace in traces})
    source_id = {name: i + 1 for i, name in enumerate(source_names)}
    q_summary = aggregate_q_metrics(
        rows, lengths, query_budget=query_budget, q_threshold=q_threshold, band_edges=q_band_edges
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    save = np.savez_compressed if compress else np.savez
    shard_meta: list[dict[str, Any]] = []

    for shard_index, start in enumerate(range(0, len(rows), shard_rows)):
        row_group = rows[start : start + shard_rows]
        n = len(row_group)
        input_ids = np.full((n, seq_len), PAD_TOKEN_ID, dtype=np.uint16)
        segment_ids = np.zeros((n, seq_len), dtype=np.uint16)
        token_mask = np.zeros((n, seq_len), dtype=np.uint8)
        sft_loss_mask = np.zeros((n, seq_len), dtype=np.uint8)
        source_ids = np.zeros((n, seq_len), dtype=np.uint8)
        effort_ids = np.zeros((n, seq_len), dtype=np.uint8)
        tool_calls = np.zeros((n,), dtype=np.uint16)
        eligible_q = np.zeros((n,), dtype=np.uint16)
        selected_q = np.zeros((n,), dtype=np.uint16)
        q_budget_utilization = np.zeros((n,), dtype=np.float32)
        q_eligible_coverage = np.zeros((n,), dtype=np.float32)
        q_band_counts = np.zeros((n, len(q_band_edges) - 1), dtype=np.uint16)
        q_expected_selected = np.zeros((n, len(q_band_edges) - 1), dtype=np.float32)

        for local_row, row in enumerate(row_group):
            packed = pack_token_sequences(
                [traces[i].tokens for i in row],
                seq_len=seq_len,
                pad_token_id=PAD_TOKEN_ID,
                compression_ratio=2,
            )
            input_ids[local_row] = packed.input_ids[0].astype(np.uint16)
            segment_ids[local_row] = packed.segment_ids[0].astype(np.uint16)
            token_mask[local_row] = packed.token_mask[0].astype(np.uint8)
            cursor = 0
            for physical_len, trace_index in zip(packed.physical_lengths, row):
                trace = traces[trace_index]
                stop = cursor + trace.tokens.size
                sft_loss_mask[local_row, cursor:stop] = trace.sft_loss_mask
                source_ids[local_row, cursor:stop] = source_id[trace.source]
                if trace.reasoning_effort:
                    effort_ids[local_row, cursor:stop] = trace.reasoning_effort
                tool_calls[local_row] += trace.tool_calls
                cursor += physical_len
            metrics = q_row_metrics(
                packed.real_lengths,
                query_budget=query_budget,
                q_threshold=q_threshold,
                band_edges=q_band_edges,
            )
            eligible_q[local_row], selected_q[local_row] = metrics.eligible_q, metrics.selected_q
            q_budget_utilization[local_row] = metrics.budget_utilization
            q_eligible_coverage[local_row] = metrics.eligible_coverage
            q_band_counts[local_row] = metrics.band_counts
            q_expected_selected[local_row] = metrics.expected_selected_by_band

        path = out_dir / f"shard-{shard_index:05d}.npz"
        save(
            path,
            input_ids=input_ids,
            segment_ids=segment_ids,
            token_mask=token_mask,
            sft_loss_mask=sft_loss_mask,
            source_ids=source_ids,
            reasoning_effort_ids=effort_ids,
            tool_calls=tool_calls,
            eligible_q=eligible_q,
            selected_q=selected_q,
            q_budget_utilization=q_budget_utilization,
            q_eligible_coverage=q_eligible_coverage,
            q_band_counts=q_band_counts,
            q_expected_selected=q_expected_selected,
        )
        shard_meta.append(
            {"file": path.name, "rows": n, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    return shard_meta, {"rows": len(rows), "source_ids": source_id, "query": q_summary}


def prepare_pool(
    pool_name: str,
    sources: tuple[TraceSource, ...],
    *,
    target_tokens: int,
    tokenizer,
    tokenizer_path: Path,
    output_dir: Path,
    seq_len: int,
    shard_rows: int,
    tokenize_batch_size: int,
    shuffle_buffer: int,
    query_budget: int,
    q_threshold: int,
    q_band_edges: tuple[int, ...],
    seed: int,
    jitter: int,
    compress_shards: bool,
) -> dict[str, Any]:
    seen_prompts: set[bytes] = set()
    cases: list[dict[str, Any]] = []
    collection: dict[str, Any] = {}
    for offset, source in enumerate(sources):
        source_target = round(target_tokens * source.weight)
        source_cases, stats = collect_source_cases(
            source,
            target_tokens=source_target,
            tokenizer=tokenizer,
            seq_len=seq_len,
            seed=seed + offset * 1009,
            shuffle_buffer=shuffle_buffer,
            tokenize_batch_size=tokenize_batch_size,
            seen_prompts=seen_prompts,
        )
        cases.extend(source_cases)
        collection[source.key] = {"target_tokens": source_target, **stats}
        print(pool_name, source.key, stats)

    traces, relabel_stats = assign_efforts_and_tokenize(
        cases,
        tokenizer=tokenizer,
        seq_len=seq_len,
        batch_size=tokenize_batch_size,
        seed=seed,
        jitter=jitter,
    )
    if not traces:
        raise RuntimeError(f"{pool_name}: no traces survived final tokenization")
    pool_dir = output_dir / pool_name
    shards, packed = write_trace_shards(
        pool_dir,
        traces=traces,
        seq_len=seq_len,
        shard_rows=shard_rows,
        query_budget=query_budget,
        q_threshold=q_threshold,
        q_band_edges=q_band_edges,
        seed=seed,
        compress=compress_shards,
    )
    source_counts, source_tokens = Counter(), Counter()
    for trace in traces:
        source_counts[trace.source] += 1
        source_tokens[trace.source] += int(trace.tokens.size)
    effort_rows = [
        {"reasoning_effort": trace.reasoning_effort}
        for trace in traces
        if trace.reasoning_effort
    ]
    histogram = reasoning_effort_histogram(effort_rows)
    total_tokens = sum(int(trace.tokens.size) for trace in traces)
    supervised_tokens = sum(int(trace.sft_loss_mask.sum()) for trace in traces)
    manifest = {
        "format": "nano-dsv41f-packed-traces-v3",
        "pool": pool_name,
        "seq_len": seq_len,
        "target_tokens": target_tokens,
        "actual_trace_tokens": total_tokens,
        "trace_records": len(traces),
        "sft_supervised_tokens": supervised_tokens,
        "sft_supervised_fraction": supervised_tokens / total_tokens,
        "tokenizer": {
            "file": tokenizer_path.name,
            "sha256": sha256_file(tokenizer_path),
            "vocab_size": tokenizer.get_vocab_size(with_added_tokens=True),
        },
        "reasoning_effort": {
            "integer_range": [1, 100],
            "method": "reasoning_length_percentile_v1",
            "jitter": jitter,
            "covered_values": sum(1 for count in histogram.values() if count),
            "missing_values": list(missing_reasoning_efforts(effort_rows)),
            "histogram": histogram,
        },
        "sources": {
            source.key: {
                "dataset": source.dataset,
                "config": source.config,
                "split": source.split,
                "weight": source.weight,
                "adapter": source.adapter,
                "license": source.license,
                "provenance": source.provenance,
                "max_observation_chars": source.max_observation_chars,
                "collection": collection[source.key],
                "final_records": source_counts[source.key],
                "final_tokens": source_tokens[source.key],
            }
            for source in sources
        },
        "final_filter": relabel_stats,
        "packing": {
            **packed,
            "q_threshold": q_threshold,
            "q_band_edges": list(q_band_edges),
            "query_budget": query_budget,
            "compressed_npz": compress_shards,
            "shards": shards,
        },
    }
    (pool_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        {
            "pool": pool_name,
            "records": len(traces),
            "tokens": total_tokens,
            "packed_rows": packed["rows"],
            "sft_fraction": round(manifest["sft_supervised_fraction"], 4),
            "q_budget_utilization": round(packed["query"]["mean_budget_utilization"], 4),
        }
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build 8K V4.1-rendered reasoning and agent pools with integer effort, "
            "assistant SFT masks and Q-aware packing."
        )
    )
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pool", choices=("all", "reasoning", "agent"), default="all")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_TRACE_SEQ_LEN)
    parser.add_argument("--reasoning-target-tokens", type=int, default=4_000_000)
    parser.add_argument("--agent-target-tokens", type=int, default=16_000_000)
    parser.add_argument("--tokenize-batch-size", type=int, default=64)
    parser.add_argument("--shuffle-buffer", type=int, default=2_000)
    parser.add_argument("--shard-rows", type=int, default=128)
    parser.add_argument("--query-budget", type=int, default=128)
    parser.add_argument("--q-threshold", type=int, default=640)
    parser.add_argument("--q-band-edges", default=",".join(map(str, DEFAULT_TRACE_Q_BANDS)))
    parser.add_argument("--jitter", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--compress-shards", action="store_true")
    args = parser.parse_args()

    if args.seq_len <= 0 or args.seq_len % 2:
        raise SystemExit("--seq-len must be a positive even integer")
    if args.tokenize_batch_size <= 0 or args.shard_rows <= 0:
        raise SystemExit("batch/shard sizes must be positive")
    q_band_edges = tuple(int(x) for x in args.q_band_edges.split(",") if x)
    if q_band_edges[0] != args.q_threshold or q_band_edges[-1] < args.seq_len:
        raise SystemExit("Q band edges must start at threshold and cover seq-len")

    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise SystemExit("Trace preparation requires tokenizers") from exc
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    contract = nano_v41_tokenizer_contract(tokenizer.get_vocab_size(with_added_tokens=True))
    for token, expected_id in contract.token_to_id.items():
        if tokenizer.token_to_id(token) != expected_id:
            raise SystemExit(f"tokenizer contract mismatch for {token!r}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifests = []
    if args.pool in ("all", "reasoning"):
        manifests.append(
            prepare_pool(
                "reasoning",
                REASONING_SOURCES,
                target_tokens=args.reasoning_target_tokens,
                tokenizer=tokenizer,
                tokenizer_path=args.tokenizer,
                output_dir=args.output_dir,
                seq_len=args.seq_len,
                shard_rows=args.shard_rows,
                tokenize_batch_size=args.tokenize_batch_size,
                shuffle_buffer=args.shuffle_buffer,
                query_budget=args.query_budget,
                q_threshold=args.q_threshold,
                q_band_edges=q_band_edges,
                seed=args.seed,
                jitter=args.jitter,
                compress_shards=args.compress_shards,
            )
        )
    if args.pool in ("all", "agent"):
        manifests.append(
            prepare_pool(
                "agent",
                AGENT_SOURCES,
                target_tokens=args.agent_target_tokens,
                tokenizer=tokenizer,
                tokenizer_path=args.tokenizer,
                output_dir=args.output_dir,
                seq_len=args.seq_len,
                shard_rows=args.shard_rows,
                tokenize_batch_size=args.tokenize_batch_size,
                shuffle_buffer=args.shuffle_buffer,
                query_budget=args.query_budget,
                q_threshold=args.q_threshold,
                q_band_edges=q_band_edges,
                seed=args.seed + 41,
                jitter=args.jitter,
                compress_shards=args.compress_shards,
            )
        )

    summary = {
        "format": "nano-dsv41f-trace-curriculum-v3",
        "seq_len": args.seq_len,
        "pools": [
            {
                "name": m["pool"],
                "records": m["trace_records"],
                "tokens": m["actual_trace_tokens"],
                "rows": m["packing"]["rows"],
                "manifest": f"{m['pool']}/manifest.json",
            }
            for m in manifests
        ],
        "recommended_training_mix": {
            "early": {"document": 1.00, "reasoning": 0.00, "agent": 0.00},
            "middle": {"document": 0.90, "reasoning": 0.05, "agent": 0.05},
            "late_mid": {"document": 0.80, "reasoning": 0.05, "agent": 0.15},
        },
        "views": {
            "midtrain": "use token_mask/segment_ids for ordinary causal LM",
            "sft": "also apply sft_loss_mask so only assistant reasoning/tool-call/content/EOS targets train",
        },
    }
    (args.output_dir / "trace_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
