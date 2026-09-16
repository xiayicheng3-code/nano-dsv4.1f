from __future__ import annotations

from collections import Counter
import copy
import hashlib
import math
import random
from typing import Any, Callable, Iterable, Sequence

from .chat_protocol import SYSTEM_TOKEN

MIN_REASONING_EFFORT = 1
MAX_REASONING_EFFORT = 100


def render_v41_reasoning_effort_prompt(effort: int) -> str:
    """Render the released V4.1 numeric reasoning-effort prefix.

    DeepSeek V4.1 accepts every integer from 1 through 100. The official encoder
    inserts this prefix only for the first message in thinking mode; canonical
    training records should normally store the integer and let the final prompt
    renderer add the text.
    """
    effort = int(effort)
    if not MIN_REASONING_EFFORT <= effort <= MAX_REASONING_EFFORT:
        raise ValueError("reasoning effort must be in [1, 100]")
    return (
        f"{SYSTEM_TOKEN}Reasoning Effort: {effort} "
        "(range 1-100, the higher the value, the more thorough the reasoning)\n\n"
    )


def reasoning_text(case: dict[str, Any]) -> str:
    """Concatenate explicit assistant reasoning spans without inventing missing CoT."""
    spans: list[str] = []
    for message in case.get("messages", ()):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        value = message.get("reasoning_content")
        if isinstance(value, str) and value:
            spans.append(value)
    return "\n".join(spans)


def character_reasoning_length(case: dict[str, Any]) -> int:
    return len(reasoning_text(case))


def tokenizer_reasoning_length(tokenizer: Any) -> Callable[[dict[str, Any]], int]:
    """Build a reasoning-length function for HF ``tokenizers``-style tokenizers.

    This intentionally avoids importing ``tokenizers`` in the core package. Any object
    exposing ``encode(text)`` and returning either an object with ``.ids`` or a sequence
    of IDs works.
    """

    def length(case: dict[str, Any]) -> int:
        text = reasoning_text(case)
        if not text:
            return 0
        encoded = tokenizer.encode(text)
        ids = getattr(encoded, "ids", encoded)
        try:
            return len(ids)
        except TypeError as exc:  # pragma: no cover - defensive adapter error
            raise TypeError("tokenizer.encode(text) must return a sized ID sequence") from exc

    return length


def _stable_identity(case: dict[str, Any], fallback_index: int) -> str:
    metadata = case.get("metadata")
    if isinstance(metadata, dict):
        for key in ("id", "trace_id", "task_id", "source"):
            value = metadata.get(key)
            if value not in (None, ""):
                return f"{key}:{value}"
    return f"index:{fallback_index}"


def _stable_tiebreak(identity: str, seed: int) -> int:
    digest = hashlib.blake2b(
        f"{seed}:{identity}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big")


def _percentile_effort(rank: int, count: int) -> int:
    if count <= 0:
        raise ValueError("count must be positive")
    if count == 1:
        return (MIN_REASONING_EFFORT + MAX_REASONING_EFFORT) // 2
    value = MIN_REASONING_EFFORT + round(
        rank
        * (MAX_REASONING_EFFORT - MIN_REASONING_EFFORT)
        / (count - 1)
    )
    return min(MAX_REASONING_EFFORT, max(MIN_REASONING_EFFORT, value))


def assign_length_guided_reasoning_effort(
    cases: Sequence[dict[str, Any]],
    *,
    length_fn: Callable[[dict[str, Any]], int] = character_reasoning_length,
    length_unit: str = "characters",
    jitter: int = 2,
    seed: int = 0,
    preserve_existing: bool = False,
) -> list[dict[str, Any]]:
    """Assign V4.1 reasoning effort from empirical CoT-length percentile.

    Eligible examples are thinking-mode records with non-empty explicit reasoning.
    They are sorted by measured reasoning length and mapped monotonically across the
    complete integer range 1..100. With at least 100 eligible examples this base mapping
    necessarily contains every effort integer.

    A small deterministic jitter is then applied only to *duplicate* percentile buckets.
    The shortest and longest eligible examples remain fixed at the percentile endpoints,
    and one anchor example for every base effort remains untouched. Thus full 1..100
    coverage cannot be destroyed by jitter, and the empirical extremes keep labels 1 and
    100 whenever there is more than one eligible example. This exposes nearby effort values
    without turning the label into a brittle exact function of target length.

    The original messages/reasoning are never modified. Assignment provenance is stored
    under ``metadata.reasoning_effort_assignment``.
    """
    if jitter < 0:
        raise ValueError("jitter must be non-negative")
    if not length_unit:
        raise ValueError("length_unit must be non-empty")

    out = [copy.deepcopy(case) for case in cases]
    eligible: list[tuple[int, int, str, int]] = []
    for index, case in enumerate(out):
        if case.get("thinking_mode") != "thinking":
            continue
        if preserve_existing and "reasoning_effort" in case:
            continue
        measured = int(length_fn(case))
        if measured <= 0:
            continue
        identity = _stable_identity(case, index)
        eligible.append((index, measured, identity, _stable_tiebreak(identity, seed)))

    eligible.sort(key=lambda row: (row[1], row[3], row[0]))
    count = len(eligible)
    base_efforts = [_percentile_effort(rank, count) for rank in range(count)]

    # Protect both empirical endpoints plus one example per base effort as coverage
    # anchors. The final-rank protection matters when several ranks map to effort 100:
    # otherwise the first 100-valued rank is anchored while the actual longest example
    # can still be jittered down to 99. When count >= 100, the remaining anchors preserve
    # complete 1..100 coverage.
    anchor_ranks: set[int] = {0, count - 1} if count > 1 else set()
    seen_efforts: set[int] = set()
    for rank, effort in enumerate(base_efforts):
        if effort not in seen_efforts:
            anchor_ranks.add(rank)
            seen_efforts.add(effort)

    for rank, (index, measured, identity, _) in enumerate(eligible):
        base = base_efforts[rank]
        delta = 0
        if jitter and rank not in anchor_ranks:
            rng = random.Random(f"nano-dsv41f:{seed}:{identity}:{rank}")
            delta = rng.randint(-jitter, jitter)
        effort = min(MAX_REASONING_EFFORT, max(MIN_REASONING_EFFORT, base + delta))

        case = out[index]
        case["reasoning_effort"] = effort
        metadata = case.setdefault("metadata", {})
        metadata["reasoning_effort_assignment"] = {
            "method": "reasoning_length_percentile_v1",
            "length": measured,
            "length_unit": length_unit,
            "rank": rank,
            "eligible_count": count,
            "base_effort": base,
            "jitter": delta,
            "seed": seed,
        }

    return out


def reasoning_effort_histogram(cases: Iterable[dict[str, Any]]) -> dict[int, int]:
    counts: Counter[int] = Counter()
    for case in cases:
        value = case.get("reasoning_effort")
        if type(value) is int and MIN_REASONING_EFFORT <= value <= MAX_REASONING_EFFORT:
            counts[value] += 1
    return {effort: counts.get(effort, 0) for effort in range(1, 101)}


def missing_reasoning_efforts(cases: Iterable[dict[str, Any]]) -> tuple[int, ...]:
    histogram = reasoning_effort_histogram(cases)
    return tuple(effort for effort, count in histogram.items() if count == 0)
