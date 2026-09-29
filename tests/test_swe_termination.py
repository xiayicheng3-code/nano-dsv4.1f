"""SWE success labels and observed terminal actions survive source normalization."""
from dataclasses import replace
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prepare_trace_corpus as traces
import prepare_posttrain_corpus as prep
from nano_dsv41f.trace_corpus import render_case_v41


def source():
    return replace(next(s for s in traces.AGENT_SOURCES if s.key == 'swe_success'),
                   max_observation_chars=None, preserve_history=True)


def row(action='submit', status='submitted', target=True):
    return {'target': target, 'exit_status': status, 'instance_id': 'example-1',
            'trajectory': [{'role': 'system', 'text': 'Use the environment.'},
                           {'role': 'user', 'text': 'Fix the issue.'},
                           {'role': 'ai', 'text': 'Inspect.\n```\nls\n```'},
                           {'role': 'user', 'text': 'file.py'},
                           {'role': 'ai', 'text': f'Done.\n```\n{action}\n```'}]}


def test_successful_terminal_submit_is_preserved_without_fake_observation():
    case = traces.adapt_swe_agent(source(), row(), 0)
    assert case is not None and len(case['messages']) == 5
    call = case['messages'][-1]['tool_calls'][0]
    assert json.loads(call['function']['arguments']) == {'cmd': 'submit'}
    assert case['messages'][-1]['role'] == 'assistant'
    assert case['metadata']['terminal_tool_call_ids'] == [call['id']]
    assert 'incomplete_tool_calls' not in case['metadata']
    assert case['metadata']['swe_end_policy'] == 'terminal_submit'
    assert 'submit' in render_case_v41(case)
    # Verify that the final action survives the real renderer -> prefix path.
    from tokenizers import Tokenizer, models, pre_tokenizers, AddedToken
    from nano_dsv41f.chat_protocol import SPECIAL_TOKENS
    vocab = {token: i for i, token in enumerate(SPECIAL_TOKENS)}
    vocab['[UNK]'] = len(vocab)
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens([AddedToken(t, special=True) for t in SPECIAL_TOKENS])
    ids = tokenizer.encode(render_case_v41(case), add_special_tokens=False).ids
    view = prep.prefix_view(ids, case, 32768)
    assert view is not None and len(view.tokens) == len(ids)
    assert view.sft_loss_mask[-1] == 1


def test_successful_context_exit_keeps_observed_history_only():
    case = traces.adapt_swe_agent(source(), row('python unfinished.py', 'submitted (exit_context)'), 0)
    assert case is not None and len(case['messages']) == 4
    assert case['messages'][-1]['role'] == 'tool'
    assert case['messages'][-1]['content'] == 'file.py'
    assert case['metadata']['dropped_unobserved_final_action']
    assert case['metadata']['swe_end_policy'] == 'observed_prefix_after_context_exit'
    assert 'incomplete_tool_calls' not in case['metadata']
    assert 'unfinished.py' not in render_case_v41(case)


def test_failed_episodes_and_unexplained_dangling_calls_still_rejected():
    assert traces.adapt_swe_agent(source(), row(target=False), 0) is None
    assert traces.adapt_swe_agent(source(), row('edit file.py'), 0) is None
    assert traces.adapt_swe_agent(source(), row(status='unknown'), 0) is None
    broken = row()
    del broken['trajectory'][3]  # Missing observation inside the conversation.
    assert traces.adapt_swe_agent(source(), broken, 0) is None


def test_context_exit_with_no_observed_assistant_prefix_rejected():
    example = row('python unfinished.py', 'submitted (exit_context)')
    example['trajectory'] = example['trajectory'][:2] + example['trajectory'][-1:]
    assert traces.adapt_swe_agent(source(), example, 0) is None
