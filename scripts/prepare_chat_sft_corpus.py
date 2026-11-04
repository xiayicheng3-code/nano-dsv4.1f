#!/usr/bin/env python3
"""CPU build: HuggingFaceTB/smol-smoltalk -> packed 32K assistant-only chat SFT shards.

Conversations are rendered with the maintained DeepSeek V4.1 encoder in non-thinking
mode (no tools), tokenized with the frozen nano tokenizer, and Q-aware packed into
32K rows with the same writer as the v3 post-training corpus. The dataset's own test
split becomes the validation split, so every SFT arm is scored on identical held-out
conversations. Conversations longer than 32K keep their longest complete prefix that
ends with an assistant turn.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

from tokenizers import Tokenizer

from pretrain_checkpoint import atomic_json, digest_file
from nano_dsv41f.chat_protocol import nano_v41_tokenizer_contract
from nano_dsv41f.trace_corpus import render_case_v41

FORMAT = 'nano-dsv41f-chat-sft-v1'
SEQ_LEN = 32768
DATASET = 'HuggingFaceTB/smol-smoltalk'
SPLITS = {'train': 'train', 'validation': 'test'}
ROLES = {'system', 'user', 'assistant'}


def builder_digest():
    h = hashlib.sha256()
    for path in (Path(__file__), Path(__file__).with_name('prepare_posttrain_corpus.py')):
        h.update(path.name.encode()); h.update(path.read_bytes())
    return h.hexdigest()


def chat_case(row, index):
    """smol-smoltalk row -> canonical non-thinking case, or None if unusable."""
    messages = []
    for message in row.get('messages') or ():
        role, content = message.get('role'), message.get('content')
        if role not in ROLES or not isinstance(content, str) or not content.strip():
            return None
        messages.append({'role': role, 'content': content.replace('\x00', '')})
    if not any(m['role'] == 'assistant' for m in messages) or messages[-1]['role'] != 'assistant':
        return None
    if any(a['role'] == b['role'] == 'assistant' for a, b in zip(messages, messages[1:])):
        return None
    return {'messages': messages, 'thinking_mode': 'chat',
            'metadata': {'source': 'smoltalk', 'subset': str(row.get('source') or 'unknown'),
                         'dataset': DATASET, 'row': index}}


def build_split(split, args, tokenizer, writer_cls, prefix_view):
    import datasets
    stream = datasets.load_dataset(DATASET, split=SPLITS[split], revision=args.revision,
                                   streaming=True)
    writer = writer_cls(args.output / 'sft' / 'chat' / split, SEQ_LEN, args.seed, args.shard_rows)
    limit = args.max_train_records if split == 'train' else args.max_validation_records
    stats, subsets, seen, batch = Counter(), Counter(), set(), []

    def flush():
        rendered, cases = [], []
        for case in batch:
            try:
                rendered.append(render_case_v41(case)); cases.append(case)
            except (ValueError, RuntimeError):
                stats['render_rejected'] += 1
        for case, text, enc in zip(cases, rendered,
                                   tokenizer.encode_batch(rendered, add_special_tokens=False)):
            digest = hashlib.sha256(text.encode()).hexdigest()
            if digest in seen:
                stats['duplicate'] += 1
                continue
            seen.add(digest)
            view = prefix_view(enc.ids, case, SEQ_LEN)
            if view is None:
                stats['no_complete_prefix'] += 1
                continue
            stats['truncated_to_prefix'] += int(len(view.tokens) < len(enc.ids))
            stats['records'] += 1
            subsets[case['metadata']['subset']] += len(view.tokens)
            writer.add(view)
        batch.clear()

    for index, row in enumerate(stream):
        if limit and stats['records'] + len(batch) >= limit:
            break
        stats['rows_seen'] += 1
        case = chat_case(row, index)
        if case is None:
            stats['adapter_rejected'] += 1
            continue
        batch.append(case)
        if len(batch) >= args.tokenize_batch_size:
            flush()
            if stats['rows_seen'] % 20000 < args.tokenize_batch_size:
                print(split, 'rows', stats['rows_seen'], 'records', stats['records'], flush=True)
    flush()
    result = writer.finish()
    result.update(stats=dict(stats), subset_tokens=dict(subsets))
    print(split, json.dumps({k: v for k, v in result.items() if k != 'shards'}), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True, help='frozen nano tokenizer.json')
    p.add_argument('--revision', default='main', help='dataset revision; the resolved SHA is recorded')
    p.add_argument('--seed', type=int, default=1701)
    p.add_argument('--shard-rows', type=int, default=256)
    p.add_argument('--tokenize-batch-size', type=int, default=512)
    p.add_argument('--max-train-records', type=int, default=0, help='0 = whole split')
    p.add_argument('--max-validation-records', type=int, default=4000)
    args = p.parse_args()
    import prepare_posttrain_corpus as posttrain

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'Use a fresh output directory: {args.output}')
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    contract = nano_v41_tokenizer_contract(32768)
    if tokenizer.get_vocab_size() != 32768 or any(
            tokenizer.token_to_id(t) != i for t, i in contract.token_to_id.items()):
        raise ValueError('Tokenizer does not match the frozen nano V4.1 contract')
    try:
        from huggingface_hub import HfApi
        resolved = HfApi().dataset_info(DATASET, revision=args.revision).sha
    except Exception as exc:  # offline builds still record the requested revision
        print('Could not resolve dataset revision:', exc)
        resolved = None
    splits = {split: build_split(split, args, tokenizer, posttrain.Writer, posttrain.prefix_view)
              for split in SPLITS}
    shutil.copy2(args.tokenizer, args.output / 'tokenizer.json')
    manifest = {
        'format': FORMAT, 'complete': True, 'lengths': {'sft': SEQ_LEN},
        'dataset': {'repo': DATASET, 'requested_revision': args.revision,
                    'resolved_revision': resolved, 'splits': SPLITS,
                    'render': 'DeepSeek V4.1 non-thinking, no tools, assistant-span loss'},
        'identity': {'tokenizer_sha256': digest_file(args.output / 'tokenizer.json'),
                     'builder_sha256': builder_digest(), 'seed': args.seed,
                     'max_train_records': args.max_train_records,
                     'max_validation_records': args.max_validation_records},
        'splits': splits,
    }
    atomic_json(args.output / 'chat_sft_manifest.json', manifest)
    print('Chat SFT corpus complete:', args.output, flush=True)


if __name__ == '__main__':
    main()
