"""Decoding-time repetition controls applied to next-token logits.

All controls see only the tokens generated in the current reply (never the prompt) and never
penalize protected tokens such as EOS. They are deterministic functions of that history, so
greedy speculative decoding stays exact when the verifier applies the same control at every
verified position.

- ``no_repeat_ngram``: ban any token that would complete an n-gram already in the reply.
- ``repetition_penalty``: CTRL-style (Keskar et al., 2019); divide positive logits of tokens
  already in the reply by the penalty and multiply negative ones by it.
- DRY ("don't repeat yourself", as in llama.cpp / text-generation-webui): if the last ``m``
  reply tokens also occurred earlier followed by token ``c``, subtract
  ``dry_multiplier * dry_base ** (m - dry_allowed_length)`` from ``c``'s logit for
  ``m >= dry_allowed_length``. Short repeats cost little; extending a long copy costs a lot.
  Matches do not extend across ``dry_sequence_breakers`` (e.g. newline tokens).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class RepetitionControl:
    no_repeat_ngram: int = 0
    repetition_penalty: float = 1.0
    dry_multiplier: float = 0.0
    dry_base: float = 1.75
    dry_allowed_length: int = 2
    dry_max_match: int = 64
    dry_sequence_breakers: frozenset[int] = field(default_factory=frozenset)
    protected_tokens: frozenset[int] = field(default_factory=frozenset)

    def __post_init__(self):
        for name in ("dry_sequence_breakers", "protected_tokens"):
            object.__setattr__(self, name, frozenset(int(t) for t in getattr(self, name)))
        if self.no_repeat_ngram < 0 or self.no_repeat_ngram == 1:
            raise ValueError("no_repeat_ngram must be 0 (off) or >= 2")
        if self.repetition_penalty < 1.0 or self.dry_multiplier < 0 or self.dry_base < 1.0:
            raise ValueError("repetition_penalty and dry_base must be >= 1, dry_multiplier >= 0")
        if self.dry_allowed_length < 1 or self.dry_max_match < self.dry_allowed_length:
            raise ValueError("need 1 <= dry_allowed_length <= dry_max_match")

    @property
    def active(self) -> bool:
        return bool(self.no_repeat_ngram or self.repetition_penalty != 1.0 or self.dry_multiplier)

    def with_protected(self, *tokens: int | None) -> "RepetitionControl":
        extra = frozenset(t for t in tokens if t is not None)
        if extra <= self.protected_tokens:
            return self
        from dataclasses import replace
        return replace(self, protected_tokens=self.protected_tokens | extra)

    def banned_ngram_tokens(self, history: list[int]) -> set[int]:
        n = self.no_repeat_ngram
        if not n or len(history) < n - 1:
            return set()
        prefix = history[len(history) - (n - 1):] if n > 1 else []
        return {history[i + n - 1] for i in range(len(history) - n + 1)
                if history[i:i + n - 1] == prefix}

    def dry_penalties(self, history: list[int]) -> dict[int, float]:
        """token -> logit penalty for extending a repeated suffix."""
        if not self.dry_multiplier or len(history) < 2:
            return {}
        last, breakers = history[-1], self.dry_sequence_breakers
        if last in breakers:
            return {}
        best: dict[int, int] = {}
        end = len(history) - 1
        for i in range(end - 1, -1, -1):  # i: earlier position matching the current last token
            if history[i] != last:
                continue
            nxt = history[i + 1]
            if nxt in breakers:
                continue
            m = 1
            while (m < self.dry_max_match and i - m >= 0 and history[i - m] == history[end - m]
                   and history[i - m] not in breakers):
                m += 1
            if m > best.get(nxt, 0):
                best[nxt] = m
        return {tok: self.dry_multiplier * self.dry_base ** (m - self.dry_allowed_length)
                for tok, m in best.items() if m >= self.dry_allowed_length}

    def apply(self, logits: torch.Tensor, history: list[int]) -> torch.Tensor:
        """Return adjusted logits for a [1, vocab] (or [vocab]) row; input is not modified."""
        if not self.active or not history:
            return logits
        out = logits.clone()
        row = out.view(-1)
        protected = self.protected_tokens
        if self.repetition_penalty != 1.0:
            seen = torch.tensor(sorted(set(history) - protected), device=row.device, dtype=torch.long)
            if seen.numel():
                values = row[seen]
                row[seen] = torch.where(values > 0, values / self.repetition_penalty,
                                        values * self.repetition_penalty)
        for tok, penalty in self.dry_penalties(history).items():
            if tok not in protected:
                row[tok] -= penalty
        banned = self.banned_ngram_tokens(history) - protected
        if banned:
            row[torch.tensor(sorted(banned), device=row.device, dtype=torch.long)] = float("-inf")
        return out


def as_repetition_control(value) -> RepetitionControl | None:
    if value is None or isinstance(value, RepetitionControl):
        return value if value is None or value.active else None
    control = RepetitionControl(**value)
    return control if control.active else None
