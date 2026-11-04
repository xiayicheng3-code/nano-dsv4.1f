"""Chat SFT corpus adapter, budget planning and the SFT-only runner control flow."""
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import posttrain_input as inputs
import prepare_chat_sft_corpus as chat_data
import run_chat_sft as runner
from nano_dsv41f.chat_protocol import EOS_TOKEN_ID
from nano_dsv41f.pretrain_recipe import pretrain_recipe, recipe_manifest
from nano_dsv41f.trace_corpus import ASSISTANT_TOKEN_ID


def test_chat_case_keeps_plain_conversations_and_rejects_unusable_rows():
    row = {'source': 'smol-magpie-ultra-short', 'messages': [
        {'role': 'system', 'content': 'Be brief.'}, {'role': 'user', 'content': 'Hi'},
        {'role': 'assistant', 'content': 'Hello!'}]}
    case = chat_data.chat_case(row, 7)
    assert case['thinking_mode'] == 'chat' and case['metadata']['subset'] == 'smol-magpie-ultra-short'
    assert [m['role'] for m in case['messages']] == ['system', 'user', 'assistant']
    bad = [
        {'messages': [{'role': 'user', 'content': 'no answer'}]},
        {'messages': [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': 'a'},
                      {'role': 'user', 'content': 'dangling'}]},
        {'messages': [{'role': 'user', 'content': 'q'}, {'role': 'tool', 'content': 'x'},
                      {'role': 'assistant', 'content': 'a'}]},
        {'messages': [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': '  '}]},
    ]
    assert all(chat_data.chat_case(r, 0) is None for r in bad)


def test_chat_render_supervises_only_non_thinking_assistant_turns():
    pytest.importorskip('deepseek_recipe')
    from nano_dsv41f.trace_corpus import render_case_v41
    case = chat_data.chat_case({'messages': [
        {'role': 'user', 'content': 'Hi'}, {'role': 'assistant', 'content': 'Hello!'},
        {'role': 'user', 'content': 'Bye'}, {'role': 'assistant', 'content': 'See you.'}]}, 0)
    text = render_case_v41(case)
    assert text.count('<｜Assistant｜></think>') == 2 and '<think>' not in text
    assert 'DSML' not in text


def _plan_manifest(real_tokens, rows):
    return {'splits': {'train': {'real_tokens': real_tokens, 'rows': rows}}}


def test_plan_uses_one_pass_capacity_and_reasoning_share():
    margin = runner.BATCH_ROWS * runner.SEQ_LEN
    tokens, weights, steps = runner.plan(_plan_manifest(100 * 32768, 100), 0, 0,
                                         fraction=0.0, requested_tokens=0)
    assert weights == {'chat': 1.0}
    assert tokens == int((100 * 32768 - margin) * .98) and steps == 24  # ceil(23.5)
    tokens, weights, _ = runner.plan(_plan_manifest(100 * 32768, 100), 50 * 32768, 50,
                                     fraction=0.1, requested_tokens=0)
    assert weights == {'chat': 0.9, 'reasoning': 0.1}
    assert tokens == int((100 * 32768 - margin) / 0.9 * .98)  # chat is the binding pool
    with pytest.raises(ValueError, match='one pass'):
        runner.plan(_plan_manifest(10 * 32768, 10), 0, 0, fraction=0.0, requested_tokens=10**9)


def _shard(directory, rows=24, length=32768):
    directory.mkdir(parents=True)
    ids = np.full((rows, length), 40, np.uint16)
    ids[:, 0] = 0; ids[:, 1] = ASSISTANT_TOKEN_ID; ids[:, -1] = EOS_TOKEN_ID
    target = np.ones_like(ids, np.uint8); target[:, :2] = 0
    file = directory / 'shard.npz'
    np.savez(file, input_ids=ids, segment_ids=np.zeros_like(ids),
             token_mask=np.ones_like(ids, np.uint8), sft_loss_mask=target)
    return {'file': file.name, 'rows': rows, 'bytes': file.stat().st_size,
            'sha256': hashlib.sha256(file.read_bytes()).hexdigest()}


def test_runner_trains_chat_sft_with_mtp_flag_and_exportable_metadata(tmp_path, monkeypatch):
    import nano_dsv41f as nano
    import nano_dsv41f.tpu as tpu
    import nano_dsv41f.tpu_native as native
    from pretrain_checkpoint import digest_file, load_checkpoint, read_metadata, save_checkpoint
    from jax.sharding import SingleDeviceSharding

    token_hash = hashlib.sha256(b'{}').hexdigest()
    chat = tmp_path / 'chat'; chat.mkdir(); (chat / 'tokenizer.json').write_text('{}')
    splits = {}
    for split, rows in (('train', 24), ('validation', 8)):
        meta = _shard(chat / 'sft' / 'chat' / split, rows)
        splits[split] = {'real_tokens': rows * 32768, 'rows': rows, 'shards': [meta]}
    (chat / 'chat_sft_manifest.json').write_text(json.dumps({
        'format': runner.CHAT_FORMAT, 'complete': True, 'lengths': {'sft': 32768},
        'identity': {'tokenizer_sha256': token_hash}, 'splits': splits}))
    # Minimal v3 corpus providing only the SFT reasoning pool.
    v3 = tmp_path / 'v3'; v3.mkdir(); (v3 / 'tokenizer.json').write_text('{}')
    views = {}
    for stage, length in inputs.LENGTHS.items():
        views[stage] = {'train': {'real_tokens': 24 * length, 'rows': 24,
                                  'shards': [_shard(v3 / stage / 'reasoning' / 'train', 24, length)]},
                        'validation': {'shards': [], 'rows': 0}}
    d = _shard(v3 / 'documents/midtrain', 8, 8192)
    (v3 / 'posttrain_manifest.json').write_text(json.dumps({
        'format': inputs.FORMAT, 'complete': True, 'lengths': inputs.LENGTHS,
        'identity': {'tokenizer_sha256': token_hash}, 'documents': {
            'path': 'documents/midtrain', 'real_tokens': 8 * 8192, 'shards': [d]},
        'sources': [{'source': {'key': 'reasoning'}, 'pool': 'reasoning', 'views': views}],
        'pool_mix': {'midtrain': {'document': 1.0}, 'sft': {'reasoning': 1.0}}}))

    def trees():
        return {'w': jnp.ones((2,), jnp.bfloat16)}, {'w': jnp.zeros((2,), jnp.float32)}
    c, t, n = pretrain_recipe(profile='narrow48', cp=2, dp=4, benchmark_total_steps=100000)
    base_recipe = json.loads(json.dumps(recipe_manifest(c, t, n)))
    midtrain = {'stage': 'midtrain', 'stage_complete': True, 'completed_steps': 120, 'base_steps': 100,
                'base_recipe': base_recipe, 'identity': {}}
    params, opt = trees()
    opt = {'w': jnp.full((2,), 9.0, jnp.float32)}  # must be discarded: SFT resets optimizer
    midpath = save_checkpoint(tmp_path / 'mid', (params, opt), midtrain)

    sharding = SingleDeviceSharding(jax.devices()[0])
    monkeypatch.setattr(nano, 'validate_v5e_runtime', lambda: None)
    monkeypatch.setattr(nano, 'make_v5e_mesh', lambda: None)
    monkeypatch.setattr(nano, 'init_model_sharded_mixed_precision', lambda *a, **kw: (trees()[0], None, None))
    monkeypatch.setattr(nano, 'init_optimizer_state_sharded', lambda *a: (trees()[1], None))
    monkeypatch.setattr(nano, 'put_training_batch', lambda ids, seg, mask, *a: tuple(jnp.array(v) for v in (ids, seg, mask)))
    monkeypatch.setattr(tpu, 'batch_named_sharding', lambda *a: sharding)
    monkeypatch.setattr(tpu, 'named_shardings', lambda *a: {'w': sharding})
    monkeypatch.setattr(native, 'NativeCompiledStep', lambda fn, *a: fn)
    import nano_dsv41f.training as training
    monkeypatch.setattr(training, 'pretrain_loss', lambda p, config, ids, **kw: (
        jnp.array(2.0), {'lm_loss': jnp.array(2.0), 'lm_tokens': (
            kw['token_mask'][:, 1:] & kw['token_mask'][:, :-1] & (kw['target_mask'][:, 1:] > 0)).sum()}))
    compiled = []

    def compile_step(p, o, spec, config, train, mesh, **kw):
        compiled.append((train.seq_len, kw['assistant_only'], kw['include_indexer'], kw['mtp_weight']))

        def step(p, o, ids, seg, number, mask, targets):
            valid = mask[:, 1:] & mask[:, :-1] & (seg[:, 1:] == seg[:, :-1]) & (targets[:, 1:] > 0)
            m = {'loss': jnp.array(1.), 'lm_loss': jnp.array(1.), 'indexer_loss': jnp.array(0.),
                 'mtp_loss': jnp.array(0.5), 'mtp_accuracy': jnp.zeros((5,)), 'mtp_blocks': jnp.array(64),
                 'lm_tokens': valid.sum(), 'expert_dropped': jnp.array(0),
                 'moe_mosaic_layers': jnp.array(config.n_layers), 'learning_rate': jnp.array(1e-5)}
            return {'w': p['w'] + jnp.array(.125, jnp.bfloat16)}, {'w': o['w'] + 1}, m
        return step
    monkeypatch.setattr(nano, 'compile_pretrain_step', compile_step)
    monkeypatch.setattr(nano, 'compile_diagnostics', lambda fn, *args: (fn, {'test': True}))

    def args(out, **kw):
        base = dict(chat_corpus=chat, output=out, midtrain=midpath, resume=None,
                    reasoning_corpus=v3, reasoning_fraction=0.25, sft_tokens=0, sft_lr=2.6e-5,
                    mtp_weight=0.1, mtp_anchors=256, seed=1701, deadline_unix=time.time() + 7200,
                    checkpoint_every=2, eval_every=100, eval_batches=1, log_every=100, max_steps=0)
        base.update(kw)
        return SimpleNamespace(**base)

    runner.run(args(tmp_path / 'first', max_steps=3))
    first = json.loads((tmp_path / 'first/summary.json').read_text())
    assert first['status'] == 'paused' and first['plan']['pool_mix'] == {'chat': 0.75, 'reasoning': 0.25}
    assert compiled == [(32768, True, True, 0.1)]
    with pytest.raises(ValueError, match='MTP setting'):
        runner.run(args(tmp_path / 'changed', midtrain=None, resume=Path(first['checkpoint']), mtp_weight=0.0))
    runner.run(args(tmp_path / 'resumed', midtrain=None, resume=Path(first['checkpoint'])))
    done = json.loads((tmp_path / 'resumed/summary.json').read_text())
    assert done['status'] == 'completed'
    _, meta = read_metadata(done['checkpoint'])
    md = meta['metadata']
    assert md['stage'] == 'sft' and md['stage_complete'] and md['recipe']['train']['seq_len'] == 32768
    assert md['completed_steps'] == 120 + md['stage_steps']
    assert md['pool_tokens']['reasoning'] > 0 and md['pool_tokens']['chat'] > md['pool_tokens']['reasoning']
    # export_posttrain_checkpoint.py verifies this pairing.
    assert md['identity']['corpus_sha256'] == digest_file(tmp_path / 'resumed/corpus_manifest.json')
    restored, _ = load_checkpoint(done['checkpoint'], trees())
    # Fresh optimizer (0) + one increment per SFT update, not the midtrain value 9.
    np.testing.assert_array_equal(np.asarray(restored[1]['w']), [md['stage_steps']] * 2)
    assert any(e['step'] == 0 for e in done['evaluations']) or any(
        e['step'] == 0 for e in first['evaluations'])


def test_chat_sft_notebooks_compile_and_call_their_entrypoints(tmp_path):
    import build_chat_sft_notebooks as builder
    import nbformat
    cpu, tpu = builder.build(tmp_path, ref='some-branch', mtp_weight=0.1)
    for path, entry in ((cpu, 'prepare_chat_sft_corpus.py'), (tpu, 'run_chat_sft.py')):
        nb = nbformat.read(path, as_version=4)
        source = '\n'.join(c.source for c in nb.cells if c.cell_type == 'code')
        assert entry in source and "'some-branch'" in source
    assert 'MTP_WEIGHT = 0.1' in '\n'.join(c.source for c in nbformat.read(tpu, as_version=4).cells)


def test_dspark_distillation_data_reads_chat_corpus(tmp_path):
    from dspark_data import SFTDraftData
    chat = tmp_path / 'chat'; chat.mkdir(); (chat / 'tokenizer.json').write_text('{}')
    token_hash = hashlib.sha256(b'{}').hexdigest()
    splits = {}
    for split, rows in (('train', 3), ('validation', 2)):
        ids_dir = chat / 'sft' / 'chat' / split
        meta = _shard(ids_dir, rows, 1024)
        if split == 'validation':  # distinct bytes: train/validation must not share a shard
            data = dict(np.load(ids_dir / meta['file']))
            data['input_ids'][:, 5] = 41
            np.savez(ids_dir / meta['file'], **data)
            meta.update(bytes=(ids_dir / meta['file']).stat().st_size,
                        sha256=hashlib.sha256((ids_dir / meta['file']).read_bytes()).hexdigest())
        splits[split] = {'real_tokens': rows * 1024, 'rows': rows, 'shards': [meta]}
    (chat / 'chat_sft_manifest.json').write_text(json.dumps({
        'format': 'nano-dsv41f-chat-sft-v1', 'complete': True, 'lengths': {'sft': 32768},
        'identity': {'tokenizer_sha256': token_hash}, 'splits': splits}))
    data = SFTDraftData(chat, tokenizer_sha256=token_hash, block_size=5, seq_len=64, anchors=4)
    sample = data.sample(0)
    assert sample['pool'] == 'chat' and len(sample['anchors']) == 4
    assert data.sample(0, split='validation')['source'].startswith('sft/chat/validation/')
