from __future__ import annotations

from dataclasses import dataclass
import copy
import json
from typing import Any, Iterable, Literal

ThinkingMode = Literal["chat", "thinking"]

_ROLE_ALIASES = {
    "human": "user",
    "model": "assistant",
    "agent": "assistant",
    "observation": "tool",
    "environment": "tool",
}
_ALLOWED_ROLES = {"system", "user", "assistant", "tool", "latest_reminder"}


class TraceNormalizationError(ValueError):
    pass


@dataclass(frozen=True)
class CleanPolicy:
    max_tool_result_chars: int | None = 65_536
    max_message_chars: int | None = None
    truncate_marker: str = "\n...[truncated by nano-dsv4.1f cleaner: {n} chars omitted]"

    def __post_init__(self) -> None:
        if self.max_tool_result_chars is not None and self.max_tool_result_chars <= 0:
            raise ValueError("max_tool_result_chars must be positive or None")
        if self.max_message_chars is not None and self.max_message_chars <= 0:
            raise ValueError("max_message_chars must be positive or None")


def _clean_text(value: Any, limit: int | None, marker: str) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = text.replace("\x00", "")
    if limit is not None and len(text) > limit:
        omitted = len(text) - limit
        text = text[:limit] + marker.format(n=omitted)
    return text


def _json_arguments(value: Any) -> str:
    if value is None:
        return "{}"
    if isinstance(value, str):
        try:
            json.loads(value)
        except json.JSONDecodeError as exc:
            raise TraceNormalizationError(
                f"tool arguments must be valid JSON; got {value[:120]!r}"
            ) from exc
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _tool_name_and_namespace(raw: dict[str, Any]) -> tuple[str, Any | None]:
    function = raw.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        namespace = raw.get("namespace", function.get("namespace"))
    else:
        name = raw.get("name") or raw.get("tool") or raw.get("tool_name")
        namespace = raw.get("namespace")
    if not isinstance(name, str) or not name:
        raise TraceNormalizationError(f"tool call is missing a name: {raw!r}")
    return name, namespace


def normalize_tool_call(raw: dict[str, Any], *, fallback_id: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TraceNormalizationError("tool call must be a mapping")
    name, namespace = _tool_name_and_namespace(raw)
    function = raw.get("function") if isinstance(raw.get("function"), dict) else raw
    arguments = (
        function.get("arguments")
        if "arguments" in function
        else function.get("args", function.get("input", {}))
    )
    call: dict[str, Any] = {
        "id": str(raw.get("id") or fallback_id),
        "type": "function",
        "function": {
            "name": name,
            "arguments": _json_arguments(arguments),
        },
    }
    if namespace is not None:
        call["namespace"] = copy.deepcopy(namespace)
    return call


def normalize_tool_definition(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize common function-tool schemas into OpenAI Chat Completions format."""
    if not isinstance(raw, dict):
        raise TraceNormalizationError("tool definition must be a mapping")
    if raw.get("type") == "function" and isinstance(raw.get("function"), dict):
        out = copy.deepcopy(raw)
        out.setdefault("type", "function")
        return out

    name = raw.get("name") or raw.get("tool") or raw.get("tool_name")
    if not isinstance(name, str) or not name:
        raise TraceNormalizationError(f"tool definition is missing a name: {raw!r}")
    parameters = raw.get("parameters", raw.get("input_schema", {"type": "object"}))
    function = {
        "name": name,
        "description": raw.get("description", ""),
        "parameters": copy.deepcopy(parameters),
    }
    out = {"type": "function", "function": function}
    if raw.get("namespace") is not None:
        out["namespace"] = copy.deepcopy(raw["namespace"])
    return out


def _reasoning_content(raw: dict[str, Any]) -> str | None:
    for key in ("reasoning_content", "reasoning", "analysis", "thought"):
        value = raw.get(key)
        if value not in (None, ""):
            return _clean_text(value, None, "")
    return None


def _normalize_message(
    raw: dict[str, Any],
    *,
    index: int,
    pending: dict[str, int],
    policy: CleanPolicy,
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TraceNormalizationError("message must be a mapping")
    role = raw.get("role")
    role = _ROLE_ALIASES.get(role, role)
    if role not in _ALLOWED_ROLES:
        raise TraceNormalizationError(
            f"unsupported role {raw.get('role')!r}; normalize it explicitly before training"
        )

    if role == "tool":
        call_id = raw.get("tool_call_id") or raw.get("tool_use_id")
        if call_id is None:
            unresolved = [key for key, count in pending.items() if count > 0]
            if len(unresolved) == 1:
                call_id = unresolved[0]
            else:
                raise TraceNormalizationError(
                    "tool result has no tool_call_id and cannot be matched unambiguously"
                )
        call_id = str(call_id)
        if pending.get(call_id, 0) <= 0:
            raise TraceNormalizationError(
                f"orphan tool result for {call_id!r}; no pending assistant tool call"
            )
        pending[call_id] -= 1
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": _clean_text(
                raw.get("content", raw.get("observation", "")),
                policy.max_tool_result_chars,
                policy.truncate_marker,
            ),
        }

    msg: dict[str, Any] = {
        "role": role,
        "content": _clean_text(
            raw.get("content", raw.get("text", "")),
            policy.max_message_chars,
            policy.truncate_marker,
        ),
    }

    if role == "assistant":
        reasoning = _reasoning_content(raw)
        if reasoning is not None:
            msg["reasoning_content"] = reasoning
        raw_calls = raw.get("tool_calls")
        if raw_calls is None and raw.get("action") is not None:
            raw_calls = [raw["action"]]
        if raw_calls:
            calls = [
                normalize_tool_call(call, fallback_id=f"call_{index}_{j}")
                for j, call in enumerate(raw_calls)
            ]
            msg["tool_calls"] = calls
            for call in calls:
                pending[call["id"]] = pending.get(call["id"], 0) + 1
        if raw.get("wo_eos"):
            msg["wo_eos"] = True

    for key in ("task", "response_format"):
        if key in raw:
            msg[key] = copy.deepcopy(raw[key])
    return msg


def _step_to_messages(
    step: dict[str, Any],
    *,
    step_index: int,
    policy: CleanPolicy,
) -> list[dict[str, Any]]:
    """Convert a simple action/observation trajectory step into message-shaped records."""
    if "role" in step:
        return [copy.deepcopy(step)]

    action = step.get("action")
    observation = step.get("observation", step.get("result"))
    if action is None:
        for key in ("user", "prompt", "instruction"):
            if key in step:
                return [{"role": "user", "content": step[key]}]
        raise TraceNormalizationError(
            f"trajectory step {step_index} has neither role nor action: {step!r}"
        )

    if isinstance(action, str):
        tool_name = step.get("tool") or step.get("tool_name")
        if tool_name:
            action = {"name": tool_name, "arguments": {"input": action}}
        else:
            assistant = {
                "role": "assistant",
                "content": action,
            }
            reasoning = _reasoning_content(step)
            if reasoning is not None:
                assistant["reasoning_content"] = reasoning
            return [assistant]

    if not isinstance(action, dict):
        raise TraceNormalizationError(f"unsupported action at step {step_index}: {action!r}")

    call_id = str(
        action.get("id")
        or step.get("tool_call_id")
        or f"call_step_{step_index}"
    )
    assistant: dict[str, Any] = {
        "role": "assistant",
        "content": _clean_text(
            step.get("content", ""), policy.max_message_chars, policy.truncate_marker
        ),
        "tool_calls": [dict(action, id=call_id)],
    }
    reasoning = _reasoning_content(step)
    if reasoning is not None:
        assistant["reasoning_content"] = reasoning

    out = [assistant]
    if observation is not None:
        out.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": observation,
            }
        )
    return out


def _extract_raw_messages(record: dict[str, Any], policy: CleanPolicy) -> list[dict[str, Any]]:
    messages = record.get("messages")
    if messages is not None:
        if not isinstance(messages, list):
            raise TraceNormalizationError("messages must be a list")
        return copy.deepcopy(messages)

    steps = record.get("trajectory", record.get("steps"))
    if steps is None:
        raise TraceNormalizationError(
            "record must contain OpenAI-style messages or a trajectory/steps list"
        )
    if not isinstance(steps, list):
        raise TraceNormalizationError("trajectory/steps must be a list")
    out: list[dict[str, Any]] = []
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            raise TraceNormalizationError(f"trajectory step {i} must be a mapping")
        out.extend(_step_to_messages(step, step_index=i, policy=policy))
    return out


def normalize_agent_trace(
    record: dict[str, Any],
    *,
    default_thinking_mode: ThinkingMode = "thinking",
    default_reasoning_effort: int = 75,
    policy: CleanPolicy | None = None,
) -> dict[str, Any]:
    """Normalize a public agent trace without pre-rendering the DeepSeek prompt.

    The output intentionally stays in OpenAI-style structured messages. DeepSeek V4.1
    rendering (including folding tool results into user ``<tool_result>`` blocks) should
    happen only during tokenization.
    """
    if not isinstance(record, dict):
        raise TraceNormalizationError("trace record must be a mapping")
    if default_thinking_mode not in ("chat", "thinking"):
        raise ValueError("default_thinking_mode must be 'chat' or 'thinking'")
    if not 1 <= int(default_reasoning_effort) <= 100:
        raise ValueError("reasoning effort must be in [1, 100]")
    policy = policy or CleanPolicy()

    raw_messages = _extract_raw_messages(record, policy)
    pending: dict[str, int] = {}
    messages = [
        _normalize_message(msg, index=i, pending=pending, policy=policy)
        for i, msg in enumerate(raw_messages)
    ]
    unresolved = [key for key, count in pending.items() if count > 0]

    tools = record.get("tools", record.get("tool_definitions", []))
    if isinstance(tools, dict):
        tools = list(tools.values())
    if not isinstance(tools, list):
        raise TraceNormalizationError("tools/tool_definitions must be a list or mapping")
    normalized_tools = [normalize_tool_definition(t) for t in tools]

    thinking_mode = record.get("thinking_mode", default_thinking_mode)
    if thinking_mode not in ("chat", "thinking"):
        raise TraceNormalizationError(f"invalid thinking_mode={thinking_mode!r}")
    reasoning_effort = int(record.get("reasoning_effort", default_reasoning_effort))
    if not 1 <= reasoning_effort <= 100:
        raise TraceNormalizationError("reasoning_effort must be in [1, 100]")

    metadata = copy.deepcopy(record.get("metadata", {}))
    for key in ("id", "trace_id", "task_id", "dataset", "source", "reward", "success"):
        if key in record and key not in metadata:
            metadata[key] = copy.deepcopy(record[key])
    if unresolved:
        metadata["incomplete_tool_calls"] = unresolved

    supervision = [
        {
            "message_index": i,
            "role": msg["role"],
            "assistant_only": msg["role"] == "assistant",
            "contains_reasoning": bool(msg.get("reasoning_content")),
            "contains_tool_call": bool(msg.get("tool_calls")),
        }
        for i, msg in enumerate(messages)
    ]

    return {
        "messages": messages,
        "tools": normalized_tools,
        "thinking_mode": thinking_mode,
        "reasoning_effort": reasoning_effort,
        "supervision": supervision,
        "metadata": metadata,
    }


def keep_trace(
    case: dict[str, Any],
    *,
    require_success: bool = False,
    min_reward: float | None = None,
) -> bool:
    metadata = case.get("metadata", {})
    if require_success and metadata.get("success") is not True:
        return False
    if min_reward is not None:
        reward = metadata.get("reward")
        if reward is None:
            return False
        try:
            if float(reward) < min_reward:
                return False
        except (TypeError, ValueError):
            return False
    return bool(case.get("messages"))


def normalize_records(
    records: Iterable[dict[str, Any]],
    **kwargs: Any,
) -> Iterable[dict[str, Any]]:
    for record in records:
        yield normalize_agent_trace(record, **kwargs)
