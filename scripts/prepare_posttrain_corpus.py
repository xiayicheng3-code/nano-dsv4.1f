#!/usr/bin/env python3
"""CPU build: canonical JSONL -> independent 8K midtrain / 16K SFT shards."""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict, replace
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
from tokenizers import Tokenizer

import prepare_trace_corpus as traces
import prepare_document_corpus_8k as docs
from pretrain_checkpoint import atomic_json, read_metadata
from pretrain_input import inspect_corpus, iter_pretrain_batches
from nano_dsv41f.chat_protocol import EOS_TOKEN_ID, nano_v41_tokenizer_contract
from nano_dsv41f.reasoning_effort import assign_length_guided_reasoning_effort, reasoning_text
from nano_dsv41f.trace_corpus import TokenizedTrace, assistant_sft_loss_mask, render_case_v41
from nano_dsv41f.training_stages import MIDTRAIN_DOCUMENT_PHASE

FORMAT = 'nano-dsv41f-posttrain-v1'
LENGTHS = {'midtrain': 8192, 'sft': 16384}


def prefix_view(ids, case, length):
    """Keep complete original history through a complete assistant turn; never crop observations.

    Pivot labels supervise ONLY the final expected action. Dropping that action to
    fit a row would turn context into a false training label, so reject instead.
    """
    ids = np.asarray(ids, dtype=np.uint16)
    mask = assistant_sft_loss_mask(ids)
    pivot = case['metadata']['source'] == 'nemotron_conversational_pivot'
    ends = np.flatnonzero((ids == EOS_TOKEN_ID) & mask.astype(bool)) + 1
    if not len(ends):
        return None
    if pivot:
        if ends[-1] > length:
            return None
        # Every assistant marker begins a new span; only the supplied action is a label.
        from nano_dsv41f.trace_corpus import ASSISTANT_TOKEN_ID
        starts = np.flatnonzero(ids == ASSISTANT_TOKEN_ID)
        mask[:starts[-1] + 1] = 0
        end = int(ends[-1])
    else:
        fits = ends[ends <= length]
        if not len(fits):
            return None
        end = int(fits[-1])
    return TokenizedTrace(ids[:end].copy(), mask[:end].copy(), case['metadata']['source'],
        int(case.get('reasoning_effort', 0)) if case.get('thinking_mode') == 'thinking' else 0,
        sum(len(m.get('tool_calls', [])) for m in
            [m for m in case['messages'] if m.get('role') == 'assistant'][:int((ends <= end).sum())]),
        {**case['metadata'], 'full_rendered_tokens': len(ids), 'prefix_tokens': end})


class Writer:
    """Bound memory to a few million token IDs; Q-pack each buffer independently."""
    def __init__(self, root, length, seed, shard_rows):
        self.root, self.length, self.seed, self.shard_rows = Path(root), length, seed, shard_rows
        self.buffer, self.buffer_tokens, self.shards = [], 0, []
        self.counts = Counter()

    def add(self, trace):
        self.buffer.append(trace)
        n = len(trace.tokens)
        self.buffer_tokens += n
        self.counts.update(records=1, real_tokens=n, supervised_tokens=int(trace.sft_loss_mask.sum()),
                           genuine_over_8k_records=int(n > 8192),
                           genuine_over_8k_tokens=n if n > 8192 else 0)
        if self.buffer_tokens >= 2_000_000:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        block = f'block-{len(self.shards):06d}'
        shards, _ = traces.write_trace_shards(self.root / block, traces=self.buffer,
            seq_len=self.length, shard_rows=self.shard_rows, query_budget=128, q_threshold=640,
            q_band_edges=(640, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384),
            seed=self.seed + len(self.shards), compress=False)
        self.shards.extend({**s, 'file': f'{block}/{s["file"]}'} for s in shards)
        self.buffer.clear()
        self.buffer_tokens = 0

    def finish(self):
        self.flush()
        return {**self.counts, 'rows': sum(s['rows'] for s in self.shards), 'shards': self.shards}


def task_key(case):
    metadata = case['metadata']
    for name in ('trajectory_id', 'qid', 'instance_id', 'task'):
        if metadata.get(name) is not None:
            return json.dumps([name, metadata[name]], sort_keys=True, ensure_ascii=False)
    return next((m['content'] for m in case['messages'] if m['role'] == 'user'), '')


def source_stream(source, seed):
    # Official OpenResearcher has 16 separate generation-seed configs. Select
    # one correct trajectory per task across seeds; never count seeds as new tasks.
    configs = [f'seed_{i}' for i in range(42, 58)] if source.key == 'openresearcher' else [source.config]
    for offset, config in enumerate(configs):
        yield from traces.load_stream(replace(source, config=config), seed=seed + offset, shuffle_buffer=128)


def build_trace_source(source, args, tokenizer):
    source = replace(source, max_observation_chars=None, preserve_history=True)
    key = source.key
    canonical = args.output / 'canonical' / f'{key}.jsonl.gz'
    canonical.parent.mkdir(parents=True, exist_ok=True)
    target = math.ceil(args.midtrain_tokens * (0.05 if source.pool == 'reasoning' else 0.15)
                       * source.weight * args.headroom)
    writers = {(stage, split): Writer(args.output / stage / key / split, length, args.seed, args.shard_rows)
               for stage, length in LENGTHS.items() for split in ('train', 'validation')}
    stats = Counter()
    seen, seen_tasks = set(), set()
    batch = []

    with gzip.open(canonical, 'wt', encoding='utf-8') as out:
        def flush():
            if not batch:
                return
            reason_enc = tokenizer.encode_batch([reasoning_text(c) for c in batch], add_special_tokens=False)
            for c, enc in zip(batch, reason_enc):
                c['metadata']['_reasoning_tokens'] = len(enc.ids)
            labelled = assign_length_guided_reasoning_effort(batch,
                length_fn=lambda c: c['metadata']['_reasoning_tokens'], length_unit='tokens',
                jitter=2, seed=args.seed, preserve_existing=False)
            rendered, cases = [], []
            for case in labelled:
                try:
                    rendered.append(render_case_v41(case))
                    cases.append(case)
                except (ValueError, RuntimeError):
                    stats['render_rejected'] += 1
            encodings = tokenizer.encode_batch(rendered, add_special_tokens=False)
            for case, text, enc in zip(cases, rendered, encodings):
                digest = hashlib.sha256(text.encode()).hexdigest()
                task = task_key(case)
                if key == "openresearcher" and task in seen_tasks:
                    stats["duplicate_task"] += 1
                    continue
                if digest in seen:
                    stats['duplicate'] += 1
                    continue
                seen.add(digest)
                views = {s: prefix_view(enc.ids, case, length) for s, length in LENGTHS.items()}
                if not any(v is not None for v in views.values()):
                    stats['no_complete_prefix'] += 1
                    continue
                seen_tasks.add(task)
                case['metadata'].update(canonical_sha256=digest,
                    supervision='final_expected_action' if key == 'nemotron_conversational_pivot' else 'assistant_spans',
                    effort_assignment='batch_reasoning_token_percentile_v1')
                out.write(json.dumps(case, ensure_ascii=False) + '\n')
                stats['canonical_records'] += 1
                task_hash = hashlib.sha256((key + '\n' + task).encode()).hexdigest()
                split = 'validation' if int(task_hash[:8], 16) % 50 == 0 else 'train'
                for stage, view in views.items():
                    if view is not None:
                        writers[stage, split].add(view)
                    else:
                        stats[f'{stage}_too_long'] += 1
            batch.clear()

        for index, row in enumerate(source_stream(source, args.seed)):
            stats['rows_seen'] += 1
            if stats['rows_seen'] % 250 == 0:
                print(key, 'rows', stats['rows_seen'], 'accepted midtrain tokens',
                      writers['midtrain', 'train'].counts['real_tokens'], flush=True)
            case = traces.ADAPTERS[source.adapter](source, row, index)
            if case is None:
                stats['adapter_rejected'] += 1
                continue
            batch.append(case)
            if len(batch) >= args.tokenize_batch_size:
                flush()
                if writers['midtrain', 'train'].counts['real_tokens'] >= target:
                    break
        flush()
    result = {'source': asdict(source), 'pool': source.pool, 'target_midtrain_tokens': target,
              'configs': [f'seed_{i}' for i in range(42,58)] if key == 'openresearcher' else [source.config],
              'canonical': str(canonical.relative_to(args.output)), 'collection': dict(stats),
              'views': {stage: {split: writers[stage, split].finish() for split in ('train', 'validation')}
                        for stage in LENGTHS}}
    result['exhausted_before_target'] = result['views']['midtrain']['train'].get('real_tokens', 0) < target
    if any(v['train']['rows'] == 0 for v in result['views'].values()):
        raise RuntimeError(f'{key}: empty stage view; inspect adapter/length rejection statistics: {stats}')
    atomic_json(args.output / 'canonical' / f'{key}.manifest.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'views'}), flush=True)
    return result


def unused_fineweb(args, tokenizer):
    root, manifest, identity = inspect_corpus(args.pretrain_corpus, args.tokenizer)
    _, checkpoint = read_metadata(args.pretrain_checkpoint)
    state = checkpoint['metadata']
    if (state.get('stage') != 'pretrain' or state['identity']['corpus'] != identity or
        state['real_tokens'] < state['identity']['base_tokens']):
        raise ValueError('Need completed base checkpoint matching this pretrain corpus/tokenizer')
    target = math.ceil(args.midtrain_tokens * .8 * .5 * args.headroom)
    segments, total = [], 0
    for batch in iter_pretrain_batches(root, batch_rows=4, seed=state['identity']['data_seed'],
                                      start_batch=state['consumed_batches']):
        for ids, seg, mask in zip(batch['input_ids'], batch['segment_ids'], batch['token_mask']):
            for sid in np.unique(seg[mask]):
                tokens = ids[(seg == sid) & mask].astype(np.uint16)
                segments.append(docs.Segment(tokens, 'fineweb_edu', f'unused:{len(segments)}'))
                total += len(tokens)
        if total >= target:
            break
    if total < target:
        raise ValueError(f'Unused pretrain tail has {total:,} tokens; need {target:,}')
    return segments, {'real_tokens': total, 'target_tokens': target,
        'base_consumed_batches': state['consumed_batches'], 'pretrain_identity': identity,
        'origin': 'unconsumed training rows in exact pretrain shuffle order'}


def build_documents(args, tokenizer):
    # prepare_phase's steps mean document rows, NOT optimizer updates or total-stage tokens.
    original = docs.collect_source_segments
    if args.pretrain_corpus:
        tail, tail_stats = unused_fineweb(args, tokenizer)
        def collect(key, **kwargs):
            return (tail, tail_stats) if key == 'fineweb_edu' else original(key, **kwargs)
        docs.collect_source_segments = collect
    try:
        return docs.prepare_phase(MIDTRAIN_DOCUMENT_PHASE, tokenizer=tokenizer,
            tokenizer_path=args.tokenizer, output_dir=args.output / 'documents',
            total_steps=math.ceil(args.midtrain_tokens * .8 / 8192), seq_len=8192,
            headroom=args.headroom, shard_rows=args.shard_rows, query_budget=128, q_threshold=640,
            q_band_edges=(640,768,1024,1536,2048,3072,4096,6144,8192), candidate_window=256,
            seed=args.seed, shuffle_buffer=2000, tokenize_batch_size=256,
            tokenize_batch_chars=2_000_000, seen_text=set(), compress_shards=False)
    finally:
        docs.collect_source_segments = original


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    if tokenizer.get_vocab_size(with_added_tokens=True) != 32768:
        raise ValueError('Production tokenizer must have 32768 tokens')
    for token, ident in nano_v41_tokenizer_contract().token_to_id.items():
        if tokenizer.token_to_id(token) != ident:
            raise ValueError(f'Tokenizer contract mismatch: {token}')
    identity = {'tokenizer_sha256': traces.sha256_file(args.tokenizer),
                'midtrain_tokens': args.midtrain_tokens, 'headroom': args.headroom, 'seed': args.seed,
                'tokenize_batch_size': args.tokenize_batch_size, 'shard_rows': args.shard_rows,
                'pretrain_corpus': str(args.pretrain_corpus), 'pretrain_checkpoint': str(args.pretrain_checkpoint),
                'builder_sha256': hashlib.sha256(Path(__file__).read_bytes() + Path(traces.__file__).read_bytes()).hexdigest()}
    progress = args.output / 'build_identity.json'
    if progress.exists() and json.loads(progress.read_text()) != identity:
        raise ValueError('Build settings changed; choose a new output directory')
    atomic_json(progress, identity)
    shutil.copy2(args.tokenizer, args.output / 'tokenizer.json')
    docpath = args.output / 'documents/midtrain/manifest.json'
    doc = json.loads(docpath.read_text()) if docpath.exists() else build_documents(args, tokenizer)
    sources = []
    for source in traces.REASONING_SOURCES + traces.AGENT_SOURCES:
        path = args.output / 'canonical' / f'{source.key}.manifest.json'
        sources.append(json.loads(path.read_text()) if path.exists() else build_trace_source(source, args, tokenizer))
    manifest = {'format': FORMAT, 'complete': True, 'identity': identity,
        'lengths': LENGTHS, 'midtrain_target_tokens': args.midtrain_tokens,
        'documents': {'path': 'documents/midtrain', **doc['packed']}, 'sources': sources,
        'pool_mix': {'midtrain': {'document': .8, 'reasoning': .05, 'agent': .15},
                     'sft': {'reasoning': 1/3, 'agent': 2/3}},
        'sft_policy': 'one-pass maximum at 1:2 pool ratio; limit set from materialized capacity; no implicit repeats',
        'context_policy': 'complete original prefix through assistant EOS; no observation truncation or disconnected tails'}
    atomic_json(args.output / 'posttrain_manifest.json', manifest)
    print('Completed:', args.output / 'posttrain_manifest.json', flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--midtrain-tokens', type=int, default=600_000_000)
    p.add_argument('--headroom', type=float, default=1.10)
    p.add_argument('--seed', type=int, default=1701)
    p.add_argument('--shard-rows', type=int, default=128)
    p.add_argument('--tokenize-batch-size', type=int, default=16)
    p.add_argument('--pretrain-corpus', type=Path)
    p.add_argument('--pretrain-checkpoint', type=Path)
    a = p.parse_args()
    if bool(a.pretrain_corpus) != bool(a.pretrain_checkpoint):
        p.error('Pass both pretrain corpus and completed checkpoint to reuse unused rows')
    if a.midtrain_tokens <= 0 or a.headroom < 1 or min(a.shard_rows, a.tokenize_batch_size) <= 0:
        p.error('Require positive budgets/batches and headroom >= 1')
    return a


if __name__ == '__main__':
    run(parse_args())
