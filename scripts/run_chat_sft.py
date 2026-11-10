#!/usr/bin/env python3
"""32K assistant-only chat SFT from the completed mid-train checkpoint, optionally with MTP.

This is the chat-focused replacement for the reasoning/agent SFT stage of
`run_posttrain.py`. It reuses that stage's recipe unchanged (fresh optimizer,
2.6e-5 -> 2.6e-6 cosine LR, YaRN x4 to 32K, candidate-masked indexer auxiliary), but
draws from the smol-smoltalk chat corpus (`prepare_chat_sft_corpus.py`) plus an optional
slice of the v3 corpus' SFT reasoning pool so thinking mode keeps its format.

`--mtp-weight > 0` adds the DSpark multi-token-prediction loss (`nano_dsv41f.mtp`) with
gradients into the backbone. Running once with 0 and once with e.g. 0.1, on the same
data, seed and schedule, is the MTP-in-SFT ablation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import signal
import time
import traceback

import numpy as np

from pretrain_checkpoint import atomic_json, digest_file, load_checkpoint, read_metadata, save_checkpoint
from posttrain_input import Pool, batch_counts, choose_pool

ROOT = Path(__file__).resolve().parents[1]
CHAT_FORMAT = 'nano-dsv41f-chat-sft-v1'
SEQ_LEN = 32768
BATCH_ROWS = 4


def code_digest():
    h = hashlib.sha256()
    files = sorted((ROOT / 'src/nano_dsv41f').glob('*.py')) + [
        Path(__file__), ROOT / 'scripts/posttrain_input.py', ROOT / 'scripts/run_posttrain.py']
    for p in files:
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()


def inspect_chat(root):
    root = Path(root)
    matches = list(root.rglob('chat_sft_manifest.json'))
    if len(matches) != 1:
        raise ValueError('Attach exactly one completed chat SFT corpus')
    root = matches[0].parent
    manifest = json.loads(matches[0].read_text())
    if manifest.get('format') != CHAT_FORMAT or not manifest.get('complete'):
        raise ValueError('Incomplete/unsupported chat SFT corpus')
    if manifest['lengths'] != {'sft': SEQ_LEN}:
        raise ValueError('Chat corpus must use 32K rows')
    if digest_file(root / 'tokenizer.json') != manifest['identity']['tokenizer_sha256']:
        raise ValueError('Tokenizer checksum mismatch')
    for split in ('train', 'validation'):
        for path, info in chat_shards(root, manifest, split):
            if path.stat().st_size != info['bytes'] or digest_file(path) != info['sha256']:
                raise ValueError(f'Shard checksum mismatch: {path}')
    return root, manifest, digest_file(matches[0])


def chat_shards(root, manifest, split):
    return [(root / 'sft' / 'chat' / split / s['file'], s) for s in manifest['splits'][split]['shards']]


def plan(chat_manifest, reasoning_capacity, reasoning_rows, *, fraction, requested_tokens):
    """Token budget, pool weights and step count for one pass without repeats."""
    train = chat_manifest['splits']['train']
    weights = {'chat': 1.0 - fraction}
    capacity = {'chat': train['real_tokens']}
    rows = {'chat': train['rows']}
    if fraction > 0:
        weights['reasoning'] = fraction
        capacity['reasoning'] = reasoning_capacity
        rows['reasoning'] = reasoning_rows
    margin = BATCH_ROWS * SEQ_LEN
    limit = int(min((capacity[k] - margin) / w for k, w in weights.items()) * .98)
    tokens = requested_tokens or limit
    if tokens <= 0 or tokens > limit:
        raise ValueError(f'SFT budget {tokens:,} must be in (0, {limit:,}] for one pass at this mix')
    steps = max(1, math.ceil(sum(tokens * w * rows[k] / (BATCH_ROWS * capacity[k])
                                 for k, w in weights.items())))
    return tokens, weights, steps


def run(args):
    import jax
    import jax.numpy as jnp
    from nano_dsv41f import (compile_diagnostics, compile_pretrain_step,
        init_model_sharded_mixed_precision, init_optimizer_state_sharded,
        make_v5e_mesh, put_training_batch, validate_v5e_runtime)
    from nano_dsv41f.pretrain_recipe import recipe_manifest
    from nano_dsv41f.training import pretrain_loss
    from nano_dsv41f.tpu import batch_named_sharding, named_shardings
    from nano_dsv41f.tpu_native import NativeCompiledStep
    from run_posttrain import stage_recipe

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(output.iterdir()):
        raise ValueError('Use a fresh output directory, attaching older state with --resume')
    report = {'status': 'initializing', 'started_unix': time.time(), 'evaluations': []}
    atomic_json(output / 'summary.json', report)
    stopped = {'signal': False}
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *unused: stopped.update(signal=True))
    log = (output / 'metrics.jsonl').open('a', buffering=1)
    try:
        chat_root, chat, chat_hash = inspect_chat(args.chat_corpus)
        train_shards = {'chat': chat_shards(chat_root, chat, 'train')}
        val_shards = {'chat': chat_shards(chat_root, chat, 'validation')}
        reasoning_capacity = reasoning_rows = 0
        reasoning_hash = None
        if args.reasoning_fraction > 0:
            if args.reasoning_corpus is None:
                raise ValueError('--reasoning-fraction needs --reasoning-corpus (the v3 posttrain corpus)')
            from posttrain_input import catalog, inspect
            r_root, r_manifest, reasoning_hash = inspect(args.reasoning_corpus)
            if r_manifest['identity']['tokenizer_sha256'] != chat['identity']['tokenizer_sha256']:
                raise ValueError('Chat and reasoning corpora use different tokenizers')
            train_shards['reasoning'] = catalog(r_root, r_manifest, 'sft')['reasoning']
            val_shards['reasoning'] = catalog(r_root, r_manifest, 'sft', 'validation')['reasoning']
            reasoning_capacity = sum(s['views']['sft']['train'].get('real_tokens', 0)
                                     for s in r_manifest['sources'] if s['pool'] == 'reasoning')
            reasoning_rows = sum(s['views']['sft']['train']['rows']
                                 for s in r_manifest['sources'] if s['pool'] == 'reasoning')
        sft_tokens, weights, sft_steps = plan(chat, reasoning_capacity, reasoning_rows,
                                              fraction=args.reasoning_fraction,
                                              requested_tokens=args.sft_tokens)
        identity = {'stage': 'chat_sft', 'corpus_sha256': chat_hash, 'reasoning_corpus_sha256': reasoning_hash,
                    'pool_mix': weights, 'sft_tokens': sft_tokens, 'sft_lr': args.sft_lr,
                    'mtp_weight': args.mtp_weight, 'mtp_anchors_per_row': args.mtp_anchors,
                    'seed': args.seed, 'code_sha256': code_digest()}
        initial = args.resume or args.midtrain
        _, saved = read_metadata(initial)
        old = saved['metadata']
        if args.resume:
            if old.get('identity') != identity or old.get('stage') != 'sft':
                raise ValueError('Resume corpus, code, budget, LR, MTP setting or seed differ')
            state = old
        else:
            if old.get('stage') != 'midtrain' or not old.get('stage_complete'):
                raise ValueError('--midtrain must be the completed mid-training checkpoint')
            state = {'stage': 'sft', 'completed_steps': old['completed_steps'],
                     'base_steps': old['base_steps'], 'midtrain_steps': old['completed_steps'],
                     'stage_steps': 0, 'real_tokens': 0, 'lm_tokens': 0, 'physical_tokens': 0,
                     'cursors': {k: 0 for k in weights}, 'pool_tokens': {k: 0 for k in weights},
                     'stage_complete': False, 'identity': identity, 'base_recipe': old['base_recipe'],
                     'midtrain_manifest_sha256': hashlib.sha256(
                         json.dumps(saved, sort_keys=True).encode()).hexdigest()}
        base_recipe = state['base_recipe']
        report['plan'] = {'sft_tokens': sft_tokens, 'pool_mix': weights, 'steps': sft_steps,
                          'mtp_weight': args.mtp_weight, 'mtp_anchors_per_row': args.mtp_anchors}
        atomic_json(output / 'summary.json', report)
        print('Plan:', json.dumps(report['plan']), flush=True)

        config, train, native = stage_recipe(base_recipe, 'sft', sft_tokens, args.sft_lr, sft_steps)
        validate_v5e_runtime()
        mesh = make_v5e_mesh()
        params, specs, _ = init_model_sharded_mixed_precision(
            jax.random.PRNGKey(7), config, mesh, payload_dtype=jnp.bfloat16)
        opt, _ = init_optimizer_state_sharded(params, specs, config, mesh)
        (params, opt), _ = load_checkpoint(initial, (params, opt),
                                            expected_identity=identity if args.resume else None)
        if not args.resume:
            # SFT starts a new optimizer and LR schedule; mid-training kept the pretrain state.
            opt, _ = init_optimizer_state_sharded(params, specs, config, mesh)
        shutil.copy2(chat_root / 'tokenizer.json', output / 'tokenizer.json')
        # Named like run_posttrain's copy so export_posttrain_checkpoint.py can verify it.
        shutil.copy2(chat_root / 'chat_sft_manifest.json', output / 'corpus_manifest.json')
        atomic_json(output / 'launch.json', {k: str(v) if isinstance(v, Path) else v
                                             for k, v in vars(args).items()})

        last_saved = None
        def checkpoint(reason):
            nonlocal last_saved
            key = (state['completed_steps'], state['stage_complete'])
            if key == last_saved:
                return
            state['recipe'] = recipe_manifest(config, train, native)
            state['reason'] = reason
            path = save_checkpoint(output / 'sft' / 'checkpoints', (params, opt), state)
            last_saved = key
            report.update(checkpoint=str(path), progress={k: state[k] for k in (
                'completed_steps', 'stage_steps', 'real_tokens', 'lm_tokens', 'pool_tokens')})
            atomic_json(output / 'summary.json', report)
            print('Checkpoint:', path, flush=True)

        pools = {k: Pool(v, seed=args.seed + i, length=SEQ_LEN)
                 for i, (k, v) in enumerate(sorted(train_shards.items()))}
        val = {k: Pool(v, seed=args.seed + 500 + i, length=SEQ_LEN)
               for i, (k, v) in enumerate(sorted(val_shards.items()))}
        evaluator = None

        def evaluate():
            nonlocal evaluator
            if args.deadline_unix - time.time() < 900 or stopped['signal']:
                return
            if evaluator is None:
                def loss(p, ids, seg, mask, targets):
                    # Held-out next-token loss only, so both ablation arms are comparable.
                    _, m = pretrain_loss(p, config, ids, segment_ids=seg, token_mask=mask,
                                         target_mask=targets)
                    return m['lm_loss'], m['lm_tokens']
                bs = batch_named_sharding(config, mesh)
                fn = jax.jit(loss, in_shardings=(named_shardings(specs, mesh), bs, bs, bs, bs))
                evaluator = NativeCompiledStep(fn, mesh, native)
            for name, pool in val.items():
                losses, targets = 0., 0
                for i in range(min(args.eval_batches, pool.rows // BATCH_ROWS)):
                    if args.deadline_unix - time.time() < 180:
                        break
                    b = pool.batch(i)
                    counts = batch_counts(b, SEQ_LEN, sft=True)
                    ids, seg, mask = put_training_batch(b['input_ids'], b['segment_ids'], b['token_mask'], config, mesh)
                    tm = jax.device_put(b['sft_loss_mask'], batch_named_sharding(config, mesh))
                    value, n = jax.device_get(evaluator(params, ids, seg, mask, tm))
                    if not math.isfinite(float(value)) or int(n) != counts['lm_tokens']:
                        raise FloatingPointError('Invalid validation loss/count')
                    losses += float(value) * int(n); targets += int(n)
                if targets:
                    row = {'event': 'validation', 'pool': name, 'step': state['stage_steps'],
                           'loss': losses / targets, 'lm_tokens': targets}
                    report['evaluations'].append(row); log.write(json.dumps(row) + '\n')
                    print(json.dumps(row), flush=True)

        checkpoint('stage_start')
        if state['stage_steps'] == 0:
            evaluate()  # step-0 held-out loss: the shared starting point of both arms
        executable = None
        stop_reason = 'stage_budget_complete'
        while not state['stage_complete'] and state['real_tokens'] < sft_tokens:
            remaining = args.deadline_unix - time.time()
            if stopped['signal'] or remaining < 180:
                stop_reason = 'signal' if stopped['signal'] else 'wall_time'; break
            if args.max_steps and state['stage_steps'] >= args.max_steps:
                stop_reason = 'debug_step_limit'; break
            if executable is None and remaining < 1200:
                stop_reason = 'wall_time_before_compile'; break
            pool = choose_pool(state['pool_tokens'], weights)
            try:
                batch = pools[pool].batch(state['cursors'][pool])
            except StopIteration:
                raise RuntimeError(f'sft/{pool} exhausted: no silent repeats or mix substitution')
            counts = batch_counts(batch, SEQ_LEN, sft=True)
            ids, seg, mask = put_training_batch(batch['input_ids'], batch['segment_ids'], batch['token_mask'], config, mesh)
            inputs = (ids, seg, jnp.asarray(state['stage_steps'], jnp.int32), mask,
                      jax.device_put(batch['sft_loss_mask'], batch_named_sharding(config, mesh)))
            if executable is None:
                checkpoint('before_compile')
                print(f'Compiling chat SFT, mtp_weight={args.mtp_weight}', flush=True)
                fn = compile_pretrain_step(params, opt, specs, config, train, mesh,
                    include_indexer=True, assistant_only=True, native_config=native,
                    mtp_weight=args.mtp_weight, mtp_anchors_per_row=args.mtp_anchors)
                executable, diagnostics = compile_diagnostics(fn, params, opt, *inputs)
                atomic_json(output / f'sft-compile-mtp{args.mtp_weight:g}.json', diagnostics)
                if args.deadline_unix - time.time() < 180 or stopped['signal']:
                    stop_reason = 'wall_time_after_compile'; break
            start = time.monotonic()
            params, opt, metrics = executable(params, opt, *inputs)
            m = jax.device_get({k: metrics[k] for k in (
                'loss', 'lm_loss', 'indexer_loss', 'mtp_loss', 'mtp_accuracy', 'mtp_blocks',
                'lm_tokens', 'expert_dropped', 'moe_mosaic_layers', 'learning_rate')})
            jax.block_until_ready((params, opt))
            if not all(math.isfinite(float(m[k])) for k in ('loss', 'lm_loss', 'indexer_loss', 'mtp_loss')):
                raise FloatingPointError('Nonfinite update; last committed checkpoint retained')
            if int(np.asarray(m['expert_dropped']).sum()) or int(m['moe_mosaic_layers']) != config.n_layers:
                raise RuntimeError('Native MoE validation failed')
            if int(m['lm_tokens']) != counts['lm_tokens']:
                raise RuntimeError('Shifted supervision count mismatch')
            state['completed_steps'] += 1; state['stage_steps'] += 1
            state['cursors'][pool] += 1; state['pool_tokens'][pool] += counts['real_tokens']
            for k, v in counts.items():
                state[k] += v
            if state['stage_steps'] % args.log_every == 0 or state['stage_steps'] == 1:
                row = {'event': 'train', 'step': state['stage_steps'], 'pool': pool,
                       'real_tokens': state['real_tokens'], 'lm_tokens': state['lm_tokens'],
                       'seconds': time.monotonic() - start,
                       **{k: float(m[k]) for k in ('loss', 'lm_loss', 'indexer_loss', 'mtp_loss', 'learning_rate')},
                       'mtp_blocks': int(m['mtp_blocks']),
                       'mtp_accuracy': [round(float(x), 4) for x in np.asarray(m['mtp_accuracy'])]}
                print(json.dumps(row), flush=True); log.write(json.dumps(row) + '\n')
            if state['stage_steps'] % args.checkpoint_every == 0:
                checkpoint('periodic')
            if state['stage_steps'] % args.eval_every == 0:
                checkpoint('before_validation'); evaluate()
        if state['real_tokens'] >= sft_tokens:
            state['stage_complete'] = True
        checkpoint(stop_reason)
        if state['stage_complete']:
            evaluate()
            report.update(status='completed', stop_reason=stop_reason)
        else:
            report.update(status='paused', stop_reason=stop_reason)
        report['finished_unix'] = time.time()
        atomic_json(output / 'summary.json', report)
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        atomic_json(output / 'summary.json', report)
        raise
    finally:
        log.close()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--chat-corpus', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--midtrain', type=Path, help='completed midtrain checkpoint (step dir or checkpoints dir)')
    g.add_argument('--resume', type=Path, help='this run\'s sft/checkpoints directory')
    p.add_argument('--reasoning-corpus', type=Path, default=None, help='v3 posttrain corpus for the reasoning slice')
    p.add_argument('--reasoning-fraction', type=float, default=0.0)
    p.add_argument('--sft-tokens', type=int, default=0, help='0: largest one-pass budget at this mix')
    p.add_argument('--sft-lr', type=float, default=2.6e-5)
    p.add_argument('--mtp-weight', type=float, default=0.0)
    p.add_argument('--mtp-anchors', type=int, default=256, help='sampled MTP anchors per 32K row')
    p.add_argument('--seed', type=int, default=1701)
    p.add_argument('--deadline-unix', type=float, default=None)
    p.add_argument('--checkpoint-every', type=int, default=500)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--eval-batches', type=int, default=4)
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--max-steps', type=int, default=0)
    a = p.parse_args(argv)
    if a.deadline_unix is None:
        a.deadline_unix = time.time() + 8 * 3600
    if not 0 <= a.reasoning_fraction < 1 or a.mtp_weight < 0 or a.mtp_anchors <= 0 or a.sft_lr <= 0 \
            or a.sft_tokens < 0 or a.max_steps < 0 \
            or min(a.checkpoint_every, a.eval_every, a.eval_batches, a.log_every) <= 0:
        p.error('Invalid fraction, MTP setting, budget, LR or intervals')
    return a


if __name__ == '__main__':
    run(parse_args())
