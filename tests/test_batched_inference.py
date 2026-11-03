"""Causality and transactional cache checks for chunked target verification."""
from dataclasses import replace

import jax
import pytest
torch = pytest.importorskip('torch')

from test_vllm_v41_cpu import tiny_config
from nano_dsv41f.config import DSparkConfig
from nano_dsv41f.hf_export import flatten_parameter_tree
from nano_dsv41f.model import init_model
from nano_dsv41f.vllm_v41_cpu import NanoDeepseekV41CPU
from nano_dsv41f.vllm_v41_cpu.decode_cache import CacheTransaction
from nano_dsv41f.vllm_v41_cpu.session import InferenceSession
from nano_dsv41f.vllm_v41_cpu.sparse_attention import causal_topk


def test_topk_ties_do_not_depend_on_masked_future_columns():
    scores = torch.tensor([[1., 0., 0., 0., -torch.inf]])
    for extra in (0, 1, 5, 31):
        padded = torch.cat((scores, torch.full((1, extra), -torch.inf)), -1)
        values, indices = causal_topk(padded, 3)
        assert indices.tolist() == [[0, 1, 2]]
        assert values.tolist() == [[1., 0., 0.]]


@pytest.fixture(scope='module')
def model():
    torch.set_num_threads(1)
    cfg = replace(tiny_config(), dspark=DSparkConfig(enabled=True, block_size=3,
        target_layer_ids=(4, 5, 6), markov_rank=4, n_routed_experts=4, experts_per_token=1))
    return NanoDeepseekV41CPU(cfg, flatten_parameter_tree(init_model(jax.random.PRNGKey(41), cfg)))


def close_cache(a, b):
    assert a.length == b.length
    torch.testing.assert_close(a.input_ids, b.input_ids)
    torch.testing.assert_close(a.next_logits, b.next_logits, atol=5e-4, rtol=5e-4)
    for layer in a.local_kv:
        # All tests fill the local window before comparing physical rings.
        torch.testing.assert_close(a.local_kv[layer], b.local_kv[layer], atol=5e-4, rtol=5e-4)
        torch.testing.assert_close(a.local_positions[layer], b.local_positions[layer])
    for layer, owner in a.owners.items():
        other = b.owners[layer]
        for name in ('kv', 'latent', 'index_k', 'segment_ids', 'group_start_positions'):
            x, y = getattr(owner.state, name), getattr(other.state, name)
            if x is not None:
                torch.testing.assert_close(x, y, atol=5e-4, rtol=5e-4)
        for name in ('pending_latent', 'pending_gate', 'pending_segment', 'pending_position'):
            x, y = getattr(owner, name), getattr(other, name)
            assert (x is None) == (y is None)
            if x is not None:
                torch.testing.assert_close(x, y, atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(a.draft_kv, b.draft_kv, atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(a.draft_positions, b.draft_positions)


@pytest.mark.parametrize('chunk', [2, 3, 7, 32])
@pytest.mark.parametrize('sparse', [False, True])
def test_chunk_logits_and_cache_match_scalar_with_packing(model, chunk, sparse):
    ids = (torch.arange(38)[None] * 17 + 3) % model.config.vocab_size
    segments = torch.tensor([[0]*10+[1]*16+[2]*12])
    a, b = model.new_cache(64), model.new_cache(64)
    a.collect_draft = b.collect_draft = True
    expected, _ = model.prefill_cache(ids, segment_ids=segments, cache=a,
        sparse_retrieval=sparse, chunk_size=1)
    actual, _ = model.prefill_cache(ids, segment_ids=segments, cache=b,
        sparse_retrieval=sparse, chunk_size=chunk)
    torch.testing.assert_close(actual, expected, atol=5e-4, rtol=5e-4)
    close_cache(a, b)
    # No future token may change any earlier position's logits.
    changed = ids.clone(); changed[:, -4:] = 1
    other, _ = model.prefill_cache(changed, segment_ids=segments, sparse_retrieval=sparse, chunk_size=chunk)
    torch.testing.assert_close(actual[:, :-4], other[:, :-4], atol=5e-4, rtol=5e-4)


@pytest.mark.parametrize('start', [5, 6])
@pytest.mark.parametrize('keep', [0, 1, 2, 3, 4, 5, 6])
def test_commit_every_prefix_across_compression_and_ring_wrap(model, start, keep):
    ids = (torch.arange(start+6)[None]*7 + 2) % model.config.vocab_size
    a, b = model.new_cache(32), model.new_cache(32)
    a.collect_draft = b.collect_draft = True
    model.prefill_cache(ids[:, :start], cache=a)
    model.prefill_cache(ids[:, :start], cache=b)
    pointers = [x.data_ptr() for x in a.local_kv.values()]
    transaction = CacheTransaction(a)
    logits, _, _ = model.forward_chunk(ids[:, start:], a)
    transaction.commit(keep, logits)
    if keep:
        model.prefill_cache(ids[:, start:start+keep], cache=b, chunk_size=1)
    close_cache(a, b)
    # Continue with a different suffix to detect leaked rejected states.
    la, _, _ = model.forward_chunk(torch.tensor([[11, 29, 3]]), a)
    lb, _ = model.prefill_cache(torch.tensor([[11, 29, 3]]), cache=b, chunk_size=1)
    torch.testing.assert_close(la, lb, atol=5e-4, rtol=5e-4)
    close_cache(a, b)
    assert pointers == [x.data_ptr() for x in a.local_kv.values()]
    assert a.transaction is None


@pytest.mark.parametrize('reject_at', [0, 1, 2, 3])
@pytest.mark.parametrize('eos_offset', [None, 0, 1, 2, 4])
def test_batched_mtp_greedy_rejection_eos_budget_and_continuation(model, monkeypatch, reject_at, eos_offset):
    prompt = torch.tensor([[1, 8, 4, 7, 12]])
    expected = model.generate(prompt, max_new_tokens=11)
    eos = None if eos_offset is None else int(expected[0, prompt.shape[1]+eos_offset])
    if eos is not None:
        end = expected[0, prompt.shape[1]:].tolist().index(eos)+1
        expected = expected[:, :prompt.shape[1]+end]
    def proposal(model, cache):
        # Test-only controlled proposals. Compute the full greedy reference so
        # the fixture also works on a later independent conversation.
        continuation = model.generate(cache.input_ids.clone(), max_new_tokens=3)[:, -3:]
        if reject_at < 3:
            continuation[:, reject_at] = (continuation[:, reject_at] + 1) % model.config.vocab_size
        return continuation
    monkeypatch.setattr('nano_dsv41f.vllm_v41_cpu.dspark.draft_block', proposal)
    session = InferenceSession(model, mtp=True, allow_untrained_draft=True, capacity=32)
    actual = session.generate(prompt, max_new_tokens=11, eos_token_id=eos)
    torch.testing.assert_close(actual, expected)
    assert session.cache.length == actual.shape[1]-1
    assert session.last_stats['mtp_verifier'] == 'batched_greedy'
    if reject_at == 3 and eos is None:
        assert session.last_stats['decode_target_calls'] < 10
    follow = torch.cat((actual, torch.tensor([[9, 2]])), 1)
    torch.testing.assert_close(session.generate(follow, max_new_tokens=4), model.generate(follow, max_new_tokens=4))


def test_prefill_calls_are_chunks_and_speculative_failure_invalidates(model, monkeypatch):
    session = InferenceSession(model, prefill_chunk_size=7)
    prompt = (torch.arange(24)[None]+1) % model.config.vocab_size
    calls = []
    original = model.forward_chunk
    def track(ids, *args, **kwargs):
        calls.append(ids.shape[1])
        return original(ids, *args, **kwargs)
    monkeypatch.setattr(model, 'forward_chunk', track)
    session.generate(prompt, max_new_tokens=0)
    assert calls == [7, 7, 7, 3]
    with pytest.raises(ValueError, match='positive integer'):
        InferenceSession(model, prefill_chunk_size=0)
    mtp = InferenceSession(model, mtp=True, allow_untrained_draft=True)
    def propose(model, cache):
        return cache.next_logits.argmax(-1, keepdim=True).expand(1, 3)
    monkeypatch.setattr('nano_dsv41f.vllm_v41_cpu.dspark.draft_block', propose)
    def fail(ids, cache=None, **kwargs):
        result = original(ids, cache, **kwargs)
        if cache is not None and cache.transaction is not None:
            raise RuntimeError('interrupted verification')
        return result
    monkeypatch.setattr(model, 'forward_chunk', fail)
    with pytest.raises(RuntimeError, match='interrupted verification'):
        mtp.generate(prompt, max_new_tokens=4)
    assert mtp.cache is None and not mtp.last_stats


@pytest.mark.parametrize('start', [1, 2])
@pytest.mark.parametrize('keep', [0, 1, 2, 3])
def test_short_prefix_rollback_with_previously_empty_owner_buffers(model, start, keep):
    ids = torch.tensor([[1, 7, 12, 4, 18]])
    a, b = model.new_cache(12), model.new_cache(12)
    a.collect_draft = b.collect_draft = True
    model.prefill_cache(ids[:, :start], cache=a)
    model.prefill_cache(ids[:, :start], cache=b, chunk_size=1)
    transaction = CacheTransaction(a)
    logits, _, _ = model.forward_chunk(ids[:, start:start+3], a)
    transaction.commit(keep, logits)
    if keep:
        model.prefill_cache(ids[:, start:start+keep], cache=b, chunk_size=1)
    tail = torch.tensor([[9, 6, 2, 15]])
    la, _, _ = model.forward_chunk(tail, a)
    lb, _ = model.prefill_cache(tail, cache=b, chunk_size=1)
    torch.testing.assert_close(la, lb, atol=5e-4, rtol=5e-4)
    close_cache(a, b)
