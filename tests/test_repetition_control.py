import pytest
torch = pytest.importorskip("torch")

from nano_dsv41f.vllm_v41_cpu.repetition import RepetitionControl, as_repetition_control

A, B, C, D, X, NL, EOS = 1, 2, 3, 4, 5, 6, 7


def test_no_repeat_ngram_bans_completions_of_seen_ngrams():
    ctl = RepetitionControl(no_repeat_ngram=3)
    assert ctl.banned_ngram_tokens([A, B, C, X, A, B]) == {C}
    assert ctl.banned_ngram_tokens([A, B, C, A, B, D, A, B]) == {C, D}
    assert ctl.banned_ngram_tokens([A]) == set()
    logits = torch.zeros(1, 8)
    out = ctl.apply(logits, [A, B, C, X, A, B])
    assert out[0, C] == float("-inf") and torch.isfinite(out[0, [A, B, D, X]]).all()
    assert torch.equal(logits, torch.zeros(1, 8))  # input untouched


def test_repetition_penalty_scales_seen_logits_by_sign():
    ctl = RepetitionControl(repetition_penalty=2.0)
    logits = torch.tensor([[0.0, 4.0, -4.0, 4.0, 0.0, 0.0, 0.0, 0.0]])
    out = ctl.apply(logits, [A, B])
    assert out[0, A] == 2.0 and out[0, B] == -8.0 and out[0, C] == 4.0


def test_dry_penalty_grows_with_match_length_and_respects_breakers():
    ctl = RepetitionControl(dry_multiplier=1.0, dry_base=2.0, dry_allowed_length=2)
    # suffix "A B" occurred before, followed by C: match length 2 -> 1 * 2**0
    assert ctl.dry_penalties([A, B, C, X, A, B]) == {C: 1.0}
    # suffix "X A B" (length 3) occurred before, followed by C -> 2**1
    assert ctl.dry_penalties([X, A, B, C, D, X, A, B]) == {C: 2.0}
    # a single matching token is below the allowed length
    assert ctl.dry_penalties([A, C, X, A]) == {}
    # breakers stop the match from extending: "NL A B" counts only "A B"
    brk = RepetitionControl(dry_multiplier=1.0, dry_base=2.0, dry_sequence_breakers=[NL])
    assert brk.dry_penalties([NL, A, B, C, NL, A, B]) == {C: 1.0}
    assert brk.dry_penalties([A, B, C, A, NL]) == {}


def test_protected_tokens_are_never_penalized_and_inactive_is_none():
    ctl = RepetitionControl(no_repeat_ngram=2, repetition_penalty=3.0).with_protected(EOS)
    out = ctl.apply(torch.ones(1, 8), [EOS, X, EOS, X])
    assert out[0, EOS] == 1.0
    assert as_repetition_control(None) is None
    assert as_repetition_control({}) is None
    assert as_repetition_control({"dry_multiplier": 0.8}).dry_multiplier == 0.8
    with pytest.raises(ValueError):
        RepetitionControl(no_repeat_ngram=1)
