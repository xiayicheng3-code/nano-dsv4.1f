from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Sequence

import numpy as np

from .chat_protocol import ASSISTANT_TOKEN, EOS_TOKEN_ID, SPECIAL_TOKENS


DEFAULT_TRACE_SEQ_LEN = 8192
DEFAULT_TRACE_Q_BANDS = (640, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192)
ASSISTANT_TOKEN_ID = SPECIAL_TOKENS.index(ASSISTANT_TOKEN)
_REASONING_EFFORT_75 = (
    "Reasoning Effort: 75 (range 1-100, the higher the value, "
    "the more thorough the reasoning)\n\n"
)


@dataclass(frozen=True)
class TokenizedTrace:
    tokens: np.ndarray
    sft_loss_mask: np.ndarray
    source: str
    reasoning_effort: int
    tool_calls: int
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if self.tokens.ndim != 1 or self.sft_loss_mask.shape != self.tokens.shape:
            raise ValueError("trace tokens/loss mask must be aligned rank-1 arrays")
        if self.tokens.size == 0:
            raise ValueError("trace must contain at least one token")
        if not 0 <= int(self.reasoning_effort) <= 100:
            raise ValueError("reasoning_effort must be 0 or an integer in [1,100]")


def _native_tool_call(call: dict[str, Any]):
    from deepseek_recipe import ToolCall

    function = call.get("function", {})
    arguments = function.get("arguments", "{}")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    return ToolCall(
        str(call.get("id", "call")),
        str(function.get("name", "tool")),
        arguments,
    )


def _native_tool_definition(tool: dict[str, Any]):
    from deepseek_recipe import ToolDefinition

    function = tool.get("function", tool)
    return ToolDefinition(
        str(function.get("name", "tool")),
        str(function.get("description", "")),
        function.get("parameters", {"type": "object"}),
        function.get("strict"),
    )


def render_case_v41(case: dict[str, Any]) -> str:
    """Render canonical structured data with DeepSeek's maintained V4.1 renderer.

    ``deepseek-recipe`` currently exposes named effort presets to Python, while the
    released V4.1 renderer itself emits the numeric ``Reasoning Effort: N`` prompt.
    We render with its default/high value (75), then replace only that numeric prefix
    with the canonical nano integer 1..100. Tool definitions, DSML calls, tool results,
    reasoning tags, role tokens and EOS placement remain owned by the official renderer.

    ``render_conversation`` appends an inference-time assistant prefix. Training records
    represent completed history, so that final generation prefix is removed.
    """
    try:
        from deepseek_recipe import (
            AssistantMessage,
            Conversation,
            DeepseekV41Encoding,
            LatestReminderMessage,
            SystemMessage,
            ToolMessage,
            UserMessage,
        )
        from deepseek_recipe import ASSISTANT_SP_TOKEN, THINKING_END_TOKEN, THINKING_START_TOKEN
    except ImportError as exc:  # pragma: no cover - data extra only
        raise RuntimeError("trace rendering requires pip install -e '.[data]'") from exc

    native_messages = []
    for msg in case.get("messages", ()):
        role = msg.get("role")
        content = str(msg.get("content", ""))
        if role == "system":
            native_messages.append(SystemMessage(content))
        elif role == "user":
            native_messages.append(UserMessage(content))
        elif role == "latest_reminder":
            native_messages.append(LatestReminderMessage(content))
        elif role == "assistant":
            calls = [_native_tool_call(call) for call in msg.get("tool_calls", ())]
            native_messages.append(
                AssistantMessage(
                    content,
                    reasoning_content=msg.get("reasoning_content"),
                    tool_calls=calls or None,
                )
            )
        elif role == "tool":
            native_messages.append(
                ToolMessage(content, str(msg.get("tool_call_id", "call")))
            )
        else:
            raise ValueError(f"unsupported canonical trace role: {role!r}")

    tools = [_native_tool_definition(tool) for tool in case.get("tools", ())]
    thinking = case.get("thinking_mode", "thinking") == "thinking"
    conversation = Conversation(
        native_messages,
        thinking_mode=thinking,
        tools=tools or None,
        reasoning_effort="high" if thinking else None,
    )
    prompt = DeepseekV41Encoding().render_conversation(conversation).prompt

    if thinking:
        effort = int(case.get("reasoning_effort", 75))
        if not 1 <= effort <= 100:
            raise ValueError("thinking trace reasoning_effort must be in [1,100]")
        replacement = (
            f"Reasoning Effort: {effort} (range 1-100, the higher the value, "
            "the more thorough the reasoning)\n\n"
        )
        if _REASONING_EFFORT_75 not in prompt:
            raise ValueError("V4.1 renderer did not emit the expected numeric effort prefix")
        prompt = prompt.replace(_REASONING_EFFORT_75, replacement, 1)

    generation_suffix = ASSISTANT_SP_TOKEN + (
        THINKING_START_TOKEN if thinking else THINKING_END_TOKEN
    )
    if not prompt.endswith(generation_suffix):
        raise ValueError("unexpected V4.1 generation suffix")
    return prompt[: -len(generation_suffix)]


def assistant_sft_loss_mask(
    token_ids: Sequence[int],
    *,
    assistant_token_id: int = ASSISTANT_TOKEN_ID,
    eos_token_id: int = EOS_TOKEN_ID,
) -> np.ndarray:
    """Mark completed assistant spans in a rendered V4.1 training history.

    The atomic assistant marker itself is context. Reasoning, DSML tool calls, assistant
    content and the terminating EOS are supervised. User/system/tool-result spans are not.
    """
    ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
    mask = np.zeros(ids.shape, dtype=np.uint8)
    in_assistant = False
    for i, token in enumerate(ids.tolist()):
        if token == assistant_token_id:
            in_assistant = True
            continue
        if in_assistant:
            mask[i] = 1
            if token == eos_token_id:
                in_assistant = False
    return mask


def strip_xml_tag(text: str, tag: str) -> str:
    text = str(text).strip()
    match = re.fullmatch(
        rf"\s*<{re.escape(tag)}>(.*)</{re.escape(tag)}>\s*", text, re.DOTALL
    )
    return match.group(1).strip() if match else text


def parse_tagged_json(text: str, tag: str) -> dict[str, Any]:
    body = strip_xml_tag(text, tag)
    try:
        value = json.loads(body)
    except json.JSONDecodeError:
        left, right = body.find("{"), body.rfind("}")
        if left < 0 or right <= left:
            raise
        value = json.loads(body[left : right + 1])
    if not isinstance(value, dict):
        raise ValueError(f"<{tag}> payload must decode to a JSON object")
    return value


def truncate_text(
    value: Any,
    limit: int,
    *,
    marker: str = "\n...[observation truncated]",
) -> str:
    text = (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, sort_keys=True)
    )
    text = text.replace("\x00", "")
    if limit > 0 and len(text) > limit:
        return text[:limit] + marker
    return text
