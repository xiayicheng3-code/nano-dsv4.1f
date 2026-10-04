from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
torch = pytest.importorskip("torch")

from test_vllm_v41_cpu import make_cpu, tiny_config
from nano_dsv41f.config import DSparkConfig
from nano_dsv41f.hf_export import flatten_parameter_tree
from nano_dsv41f.model import apply_model, init_model
from nano_dsv41f.dspark import apply_dspark
from nano_dsv41f.vllm_v41_cpu import NanoDeepseekV41CPU
from nano_dsv41f.vllm_v41_cpu.session import InferenceSession, format_inference_stats
from nano_dsv41f.vllm_v41_cpu.dspark import draft_block


@pytest.fixture(scope='module')
def model():
    torch.set_num_threads(1)
    return make_cpu(19)[2]


def test_fixed_storage_rings_and_capacity_match_reference(model):
    ids = torch.tensor([[1, 7, 3, 9, 2, 5, 8, 4, 11, 12, 6, 10]])
    cache = model.new_cache(capacity=12)
    pointers = None
    for i in range(ids.shape[1]):
        logits, _, _ = model.forward_step(ids[:, i:i+1], cache)
        if i >= 1:
            current = [cache.input_ids.untyped_storage().data_ptr()]
            current += [v.data_ptr() for v in cache.local_kv.values()]
            current += [v.data_ptr() for o in cache.owners.values() for v in o.buffers.values()]
            if pointers is not None:
                assert current == pointers
            pointers = current
        assert all(v.shape[1] == model.config.attention.local_window for v in cache.local_kv.values())
    reference, _ = model.forward(ids)
    torch.testing.assert_close(logits[:, -1], reference[:, -1], rtol=3e-4, atol=3e-4)
    with pytest.raises(ValueError, match='capacity'):
        model.forward_step(torch.tensor([[1]]), cache)
    assert cache.length == 12


def test_persistent_prefix_checkpoint_reset_and_statistics(model):
    session = InferenceSession(model, capacity=32)
    prompt = torch.tensor([[1, 3, 5, 7, 9]])
    first = session.generate(prompt, max_new_tokens=5)
    assert session.last_stats['prefill_tokens'] == 5
    assert session.last_stats['cached_prompt_tokens'] == 0
    assert session.last_stats['decode_tokens'] == 5
    assert len(session.last_stats['decode_step_seconds']) == 5
    second_prompt = torch.cat((first, torch.tensor([[11, 13]])), 1)
    second = session.generate(second_prompt, max_new_tokens=3)
    assert session.last_stats['cached_prompt_tokens'] == first.shape[1] - 1
    assert session.last_stats['prefill_tokens'] == 3
    assert session.last_stats['prefill_seconds'] > 0
    assert session.last_stats['decode_tokens_per_second'] > 0
    assert session.last_stats['time_to_first_token_seconds'] > 0
    assert 'Prefill:' in format_inference_stats(session.last_stats)
    torch.testing.assert_close(second, model.generate(second_prompt, max_new_tokens=3))
    # Protocol rendering changed the prior assistant text but kept its prompt.
    changed = torch.cat((second_prompt, torch.tensor([[17, 19]])), 1)
    result = session.generate(changed, max_new_tokens=2)
    assert session.last_stats['cached_prompt_tokens'] == second_prompt.shape[1]
    torch.testing.assert_close(result, model.generate(changed, max_new_tokens=2))
    session.reset()
    assert session.cache is None and not session.last_stats
    session.generate(prompt, max_new_tokens=0)
    assert session.last_stats['decode_tokens'] == 0
    assert session.last_stats['time_to_first_token_seconds'] is None
    session.generate(prompt, max_new_tokens=1)
    assert session.last_stats['prefill_tokens'] == 0
    assert session.last_stats['cached_prompt_tokens'] == 5
    session.generate(torch.tensor([[2, 4]]), max_new_tokens=1)
    assert session.last_stats['cached_prompt_tokens'] == 0


def test_session_invalidates_after_error_and_checks_budget(model, monkeypatch):
    session = InferenceSession(model, capacity=8)
    prompt = torch.tensor([[1, 2]])
    with pytest.raises(ValueError, match='capacity'):
        session.generate(prompt, max_new_tokens=7)
    original = model.forward_step
    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError('interrupted')
    monkeypatch.setattr(model, 'forward_step', fail)
    with pytest.raises(RuntimeError, match='interrupted'):
        session.generate(prompt, max_new_tokens=2)
    assert session.cache is None and not session.last_stats
    monkeypatch.setattr(model, 'forward_step', original)
    session.generate(prompt, max_new_tokens=2)
    assert session.last_stats['cached_prompt_tokens'] == 0


@pytest.fixture(scope='module')
def draft_model():
    config = replace(tiny_config(), dspark=DSparkConfig(enabled=True, block_size=3,
        target_layer_ids=(4, 5, 6), markov_rank=4, n_routed_experts=4, experts_per_token=1))
    params = init_model(jax.random.PRNGKey(31), config)
    model = NanoDeepseekV41CPU(config, flatten_parameter_tree(params))
    return config, params, model


def test_dspark_torch_block_matches_jax_after_ring_wrap(draft_model):
    config, params, model = draft_model
    ids = torch.tensor([[1, 3, 5, 7, 9, 11, 13, 15]])
    cache = model.new_cache(32)
    cache.collect_draft = True
    model.prefill_cache(ids, cache=cache, return_all_logits=False, sparse_retrieval=False)
    proposals, base = draft_block(model, cache, return_base_logits=True)
    jids = jnp.asarray(ids.numpy(), dtype=jnp.int32)
    _, aux = apply_model(params, config, jids, compute_indexer=True)
    expected = apply_dspark(params['dspark'], config, embed=params['embed'],
        lm_head=params['lm_head'], input_ids=jids,
        context_features=aux['dspark_context_features'],
        anchor_positions=jnp.asarray([[ids.shape[1]-1]], dtype=jnp.int32))
    np.testing.assert_allclose(base.numpy(), np.asarray(expected['base_logits'])[:, 0], atol=3e-4, rtol=3e-4)
    assert proposals.shape == (1, config.dspark.block_size)


@pytest.mark.parametrize('accept', [True, False])
def test_mtp_verification_preserves_target_and_counts_acceptance(draft_model, monkeypatch, accept):
    _, _, model = draft_model
    prompt = torch.tensor([[1, 3, 5, 7]])
    expected = model.generate(prompt, max_new_tokens=6)
    def propose(model, cache):
        offset = cache.length
        proposal = expected[:, offset:offset+3]
        if not accept:
            proposal = (proposal + 1) % model.config.vocab_size
        return proposal
    monkeypatch.setattr('nano_dsv41f.vllm_v41_cpu.dspark.draft_block', propose)
    session = InferenceSession(model, capacity=32, mtp=True, allow_untrained_draft=True)
    output = session.generate(prompt, max_new_tokens=6)
    torch.testing.assert_close(output, expected)
    assert session.last_stats['mtp_verified_tokens'] == 6
    assert session.last_stats['mtp_accepted_tokens'] == (6 if accept else 0)
    assert session.last_stats['mtp_draft_seconds'] > 0


def test_mtp_guard_real_draft_eos_and_sampling(draft_model):
    _, _, model = draft_model
    with pytest.raises(ValueError, match='training is unverified'):
        InferenceSession(model, mtp=True)
    session = InferenceSession(model, capacity=32, mtp=True, allow_untrained_draft=True)
    prompt = torch.tensor([[1, 3, 5, 7]])
    expected = model.generate(prompt, max_new_tokens=5)
    actual = session.generate(prompt, max_new_tokens=5)
    torch.testing.assert_close(actual, expected)
    # Reuse MTP feature rings across a real continuation and checkpoint rollback.
    continued = torch.cat((actual, torch.tensor([[17]])), 1)
    torch.testing.assert_close(session.generate(continued, max_new_tokens=3),
                               model.generate(continued, max_new_tokens=3))
    changed = torch.cat((continued, torch.tensor([[19]])), 1)
    torch.testing.assert_close(session.generate(changed, max_new_tokens=2),
                               model.generate(changed, max_new_tokens=2))
    session.reset()
    result = session.generate(prompt, max_new_tokens=5, eos_token_id=int(expected[0, prompt.shape[1]]))
    assert result.shape[1] == prompt.shape[1]+1
    assert session.last_stats['decode_tokens'] == 1
    with pytest.raises(ValueError, match='greedy'):
        session.generate(prompt, max_new_tokens=2, temperature=1.0)
