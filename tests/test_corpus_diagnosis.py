"""Diagnostic observations must not change adapter decisions or claim false 32K gains."""
from collections import Counter
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import diagnose_posttrain_corpus as diag
from nano_dsv41f.chat_protocol import EOS_TOKEN_ID, SPECIAL_TOKENS
from nano_dsv41f.trace_corpus import ASSISTANT_TOKEN_ID


def case(source='xcoder'):
    return {'metadata': {'source': source}, 'messages': [], 'thinking_mode': 'chat'}


def test_32k_rescues_whole_answer_but_never_clips_a_longer_answer():
    ids = [0, ASSISTANT_TOKEN_ID] + [40] * 20_000 + [EOS_TOKEN_ID]
    views, metrics, end = diag.measure_views(ids, case())
    assert views[16384] is None
    assert len(views[32768].tokens) == len(ids) == end
    assert metrics[32768]['full_through_final_assistant_records'] == 1
    assert metrics[32768]['over_16k_records'] == 1
    longer = [0, ASSISTANT_TOKEN_ID] + [40] * 40_000 + [EOS_TOKEN_ID]
    assert all(v is None for v in diag.measure_views(longer, case())[0].values())


def test_extended_prefix_is_not_a_rescued_record_and_pivot_keeps_final_target():
    ids = [0, ASSISTANT_TOKEN_ID, 40, EOS_TOKEN_ID] + [41] * 20_000 + [ASSISTANT_TOKEN_ID, 42, EOS_TOKEN_ID]
    views, metrics, _ = diag.measure_views(ids, case('swe_success'))
    assert len(views[16384].tokens) == 4
    assert metrics[16384]['partial_prefix_records'] == 1
    assert metrics[32768]['full_through_final_assistant_records'] == 1
    pivot, _, _ = diag.measure_views(ids, case('nemotron_conversational_pivot'))
    assert pivot[16384] is None
    assert pivot[32768].sft_loss_mask.sum() == 2
    assert not pivot[32768].sft_loss_mask[:-2].any()


def test_trace_preserves_rejection_and_reports_caught_exception(monkeypatch):
    source = next(s for s in diag.traces.AGENT_SOURCES if s.key == 'swe_success')
    adapter = diag.traces.adapt_swe_agent
    old_trace = sys.gettrace()
    # An integer success flag fails the exact production identity check.
    rejected, status, events = diag.audited_adapter(adapter, source, {'target': 1}, 0)
    assert rejected is None and status == 'rejected'
    assert any('row.get("target") is not True' in e['context'] for e in events)
    assert sys.gettrace() is old_trace
    row = {'target': True, 'trajectory': [{'role': 'system', 'text': 'System'},
           {'role': 'user', 'text': 'Question'}, {'role': 'ai', 'text': 'Answer'}]}
    original = adapter(source, row, 0)
    observed, status, events = diag.audited_adapter(adapter, source, row, 0)
    assert observed == original and status == 'accepted' and not events
    def failing_normalizer(*args, **kwargs):
        raise ValueError('synthetic schema failure')
    monkeypatch.setattr(diag.traces, 'normalize_agent_trace', failing_normalizer)
    observed, status, events = diag.audited_adapter(adapter, source, row, 0)
    assert observed is None and status == 'rejected'
    assert any(e.get('exception_type') == 'ValueError' for e in events)
    assert sys.gettrace() is old_trace


def toy_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers, AddedToken
    vocab = {token: i for i, token in enumerate(SPECIAL_TOKENS)}
    vocab['[UNK]'] = len(vocab)
    tok = Tokenizer(models.WordLevel(vocab, unk_token='[UNK]'))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens([AddedToken(x, special=True) for x in SPECIAL_TOKENS])
    return tok


def test_end_to_end_probe_reports_32k_gain_without_extrapolation(tmp_path, monkeypatch):
    rows = [{'question': 'Long question', 'solution': 'Reason ' * 20_000,
             'answer': '42', 'subject': 'Physics', 'correctness': True},
            {'question': 'Rejected', 'correctness': False}]
    monkeypatch.setattr(diag.traces, 'load_stream', lambda *a, **k: iter(rows))
    report = diag.run(toy_tokenizer(), tmp_path, source_keys=['chimera_science'], max_rows=10)
    result = report['sources'][0]
    assert report['complete']
    assert result['counts']['rows_seen'] == 2
    assert result['counts']['adapter_rejected'] == 1
    assert result['counts']['rescued_at_32k_records'] == 1
    assert result['length_inventory_before_task_ownership']['16384']['eligible_records'] == 0
    assert result['selection_simulation_on_scanned_rows']['32768']['cross_stage_task_overlap'] == 0
    assert json.loads((tmp_path / 'diagnosis.json').read_text())['complete']
    assert 'not full-dataset capacity estimates' in (tmp_path / 'summary.md').read_text()


def test_access_error_is_not_exhaustion(tmp_path, monkeypatch):
    def blocked(*args, **kwargs):
        raise PermissionError('blocked')
    monkeypatch.setattr(diag.traces, 'load_stream', blocked)
    report = diag.run(toy_tokenizer(), tmp_path, source_keys=['swe_success'])
    assert not report['complete']
    assert report['sources'][0]['stop_reason'] == 'source_error'
    assert report['sources'][0]['rejection_evidence']['source_error']['exception_type'] == 'PermissionError'


def test_openresearcher_sample_covers_all_configs(monkeypatch):
    source = next(s for s in diag.traces.AGENT_SOURCES if s.key == 'openresearcher')
    monkeypatch.setattr(diag.traces, 'load_stream', lambda *a, **k: iter([{}] * 100))
    configs = Counter()
    assert len(list(diag.sampled_rows(source, 1701, 32, configs))) == 32
    assert len(configs) == 16 and set(configs.values()) == {2}


def test_notebook_embeds_current_engine_and_compiles():
    import nbformat
    root = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(root / 'notebooks/nano_dsv41f_diagnose_corpus_16k_32k_cpu.ipynb', as_version=4)
    nbformat.validate(notebook)
    sources = [cell.source for cell in notebook.cells if cell.cell_type == 'code']
    assert (root / 'scripts/diagnose_posttrain_corpus.py').read_text() in sources
    for source in sources:
        compile(source, '<diagnosis notebook>', 'exec')
