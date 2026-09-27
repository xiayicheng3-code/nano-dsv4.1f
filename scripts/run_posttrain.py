#!/usr/bin/env python3
"""Resume pretrained weights+optimizer, train 8K midtrain, then 16K assistant-only SFT."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import shutil
import signal
import time
import traceback
import numpy as np

from pretrain_checkpoint import atomic_json, load_checkpoint, read_metadata, save_checkpoint
from posttrain_input import Pool, batch_counts, capacities, catalog, choose_pool, inspect, sft_budget

ROOT = Path(__file__).resolve().parents[1]


def code_digest():
    h = hashlib.sha256()
    for p in sorted((ROOT / 'src/nano_dsv41f').glob('*.py')) + [Path(__file__), ROOT / 'scripts/posttrain_input.py']:
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()


def stage_recipe(base_recipe, stage, sft_tokens, sft_lr, sft_steps=None):
    from nano_dsv41f.pretrain_recipe import pretrain_recipe
    from nano_dsv41f.config import TrainConfig
    from nano_dsv41f.tpu_native import TPUNativeConfig
    config, _, _ = pretrain_recipe(profile='narrow48', cp=2, dp=4)
    if asdict(config) != base_recipe['model']:
        # JSON turns schedule tuples into lists; compare canonical JSON values.
        if json.loads(json.dumps(asdict(config))) != base_recipe['model']:
            raise ValueError('Pretrain architecture differs from the supported narrow48 CP2/DP4 recipe')
    train = TrainConfig(**base_recipe['train'])
    native = TPUNativeConfig(**base_recipe['native'])
    if stage == 'sft':
        config = replace(config,
            attention=replace(config.attention, rope=replace(config.attention.rope,
                original_seq_len=8192, rope_factor=2.0)),
            indexer_training=replace(config.indexer_training, apply_candidate_mask=True,
                start_fraction=0.0, end_fraction=1.0))
        train = TrainConfig(total_steps=sft_steps or max(1, math.ceil(sft_tokens / (4 * 16384))),
            seq_len=16384, learning_rate=sft_lr, min_learning_rate=sft_lr * .1,
            warmup_steps=20, cosine_decay_start_fraction=0.0)
    return config, train, native


def validate_base(state, corpus):
    if state.get('stage') != 'pretrain':
        raise ValueError('--pretrained must be a pretrain checkpoint; use --resume for this runner')
    if state['real_tokens'] < state['identity']['base_tokens']:
        raise ValueError('Complete the base pretraining allocation before midtrain')
    if state['identity']['corpus']['tokenizer_sha256'] != corpus['identity']['tokenizer_sha256']:
        raise ValueError('Pretrain and posttrain tokenizers differ')
    remaining = state['identity']['total_tokens'] - state['real_tokens']
    if abs(remaining - corpus['midtrain_target_tokens']) > 4 * 8192:
        raise ValueError('Midtrain budget must match the remaining full-training allocation (within one base batch)')


def run(args):
    import jax
    import jax.numpy as jnp
    from nano_dsv41f import (compile_diagnostics, compile_pretrain_step,
        init_model_sharded_mixed_precision, init_optimizer_state_sharded,
        make_v5e_mesh, put_training_batch, validate_v5e_runtime)
    from nano_dsv41f.pretrain_recipe import recipe_manifest
    from nano_dsv41f.training import indexer_phase_enabled, pretrain_loss
    from nano_dsv41f.tpu import batch_named_sharding, named_shardings
    from nano_dsv41f.tpu_native import NativeCompiledStep

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Use a fresh output directory, attaching older state with --resume')
    report = {'status': 'initializing', 'started_unix': time.time(), 'evaluations': []}
    atomic_json(output / 'summary.json', report)
    stopped = {'signal': False}
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *unused: stopped.update(signal=True))
    log = (output / 'metrics.jsonl').open('a', buffering=1)
    try:
        root, corpus, corpus_hash = inspect(args.corpus)
        sft_tokens = args.sft_tokens or sft_budget(corpus)
        if sft_tokens <= 0 or sft_tokens > sft_budget(corpus):
            raise ValueError('SFT target exceeds one-pass capacity at the 1:2 pool ratio')
        for pool, weight in corpus['pool_mix']['midtrain'].items():
            available = capacities(corpus, 'midtrain')[pool]
            if available < corpus['midtrain_target_tokens'] * weight + 4 * 8192:
                raise ValueError(f'{pool}: only {available:,} tokens; rebuild/increase clean-source capacity before 600M run')
        sft_rows = {pool: sum(source['views']['sft']['train']['rows']
                    for source in corpus['sources'] if source['pool'] == pool)
                    for pool in corpus['pool_mix']['sft']}
        sft_capacity = capacities(corpus, 'sft')
        sft_steps = max(1, math.ceil(sum(sft_tokens * w * sft_rows[k] / (4 * sft_capacity[k])
                                      for k, w in corpus['pool_mix']['sft'].items())))
        identity = {'corpus_sha256': corpus_hash, 'sft_tokens': sft_tokens,
                    'sft_lr': args.sft_lr, 'seed': args.seed, 'code_sha256': code_digest()}
        initial_path = args.resume or args.pretrained
        _, saved = read_metadata(initial_path)
        old = saved['metadata']
        if args.resume:
            if old.get('identity') != identity or old.get('stage') not in ('midtrain', 'sft'):
                raise ValueError('Resume corpus, code, budget, LR or seeds differ')
            base_recipe = old['base_recipe']
            state = old
        else:
            validate_base(old, corpus)
            base_recipe = old['recipe']
            state = {'stage': 'midtrain', 'completed_steps': old['completed_steps'],
                'base_steps': old['completed_steps'], 'base_real_tokens': old['real_tokens'],
                'stage_steps': 0, 'real_tokens': 0, 'lm_tokens': 0, 'physical_tokens': 0,
                'cursors': {k: 0 for k in corpus['pool_mix']['midtrain']},
                'pool_tokens': {k: 0 for k in corpus['pool_mix']['midtrain']},
                'stage_complete': False, 'identity': identity, 'base_recipe': base_recipe,
                'pretrained_manifest_sha256': hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()}
        config, train, native = stage_recipe(base_recipe, state['stage'], sft_tokens, args.sft_lr, sft_steps)
        validate_v5e_runtime()
        mesh = make_v5e_mesh()
        params, specs, _ = init_model_sharded_mixed_precision(
            jax.random.PRNGKey(7), config, mesh, payload_dtype=jnp.bfloat16)
        opt, _ = init_optimizer_state_sharded(params, specs, config, mesh)
        (params, opt), _ = load_checkpoint(initial_path, (params, opt),
                                            expected_identity=identity if args.resume else None)
        shutil.copy2(root / 'tokenizer.json', output / 'tokenizer.json')
        shutil.copy2(root / 'posttrain_manifest.json', output / 'corpus_manifest.json')
        atomic_json(output / 'launch.json', {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()})
        last_saved = None
        def checkpoint(reason):
            nonlocal last_saved
            key = (state['stage'], state['completed_steps'], state['stage_complete'])
            if key == last_saved:
                return
            state['recipe'] = recipe_manifest(config, train, native)
            state['reason'] = reason
            path = save_checkpoint(output / state['stage'] / 'checkpoints', (params, opt), state)
            last_saved = key
            report.update(checkpoint=str(path), stage=state['stage'],
                          progress={k: state[k] for k in ('completed_steps','stage_steps','real_tokens','lm_tokens','pool_tokens')})
            atomic_json(output / 'summary.json', report)
            print('Checkpoint:', path, flush=True)

        while True:
            stage = state['stage']
            length = 8192 if stage == 'midtrain' else 16384
            weights = corpus['pool_mix'][stage]
            target = corpus['midtrain_target_tokens'] if stage == 'midtrain' else sft_tokens
            pools = {k: Pool(v, seed=args.seed + i, length=length)
                     for i,(k,v) in enumerate(sorted(catalog(root, corpus, stage).items()))}
            val = {k: Pool(v, seed=args.seed + 500 + i, length=length)
                   for i,(k,v) in enumerate(sorted(catalog(root, corpus, stage, 'validation').items()))}
            executables = {}
            evaluator = None
            def evaluate():
                nonlocal evaluator
                if args.deadline_unix - time.time() < 900 or stopped['signal']:
                    return
                if evaluator is None:
                    def loss(p, ids, seg, mask, targets):
                        _, m = pretrain_loss(p, config, ids, segment_ids=seg, token_mask=mask,
                                            target_mask=targets if stage == 'sft' else None)
                        return m['lm_loss'], m['lm_tokens']
                    bs = batch_named_sharding(config, mesh)
                    fn = jax.jit(loss, in_shardings=(named_shardings(specs, mesh), bs, bs, bs, bs))
                    evaluator = NativeCompiledStep(fn, mesh, replace(native, need_teacher_lse=False))
                for name, pool in val.items():
                    losses, targets = 0., 0
                    for i in range(min(args.eval_batches, pool.rows // 4)):
                        if args.deadline_unix - time.time() < 180:
                            break
                        b = pool.batch(i)
                        counts = batch_counts(b, length, sft=stage == 'sft')
                        ids, seg, mask = put_training_batch(b['input_ids'], b['segment_ids'], b['token_mask'], config, mesh)
                        tm = jax.device_put(b['sft_loss_mask'], batch_named_sharding(config, mesh))
                        value, n = jax.device_get(evaluator(params, ids, seg, mask, tm))
                        if not math.isfinite(float(value)) or int(n) != counts['lm_tokens']:
                            raise FloatingPointError('Invalid validation loss/count')
                        losses += float(value) * int(n); targets += int(n)
                    if targets:
                        row = {'event':'validation', 'stage':stage, 'pool':name,
                               'step':state['stage_steps'], 'loss':losses/targets, 'lm_tokens':targets}
                        report['evaluations'].append(row); log.write(json.dumps(row)+'\n')
                        print(json.dumps(row), flush=True)

            checkpoint('stage_start')
            stop_reason = 'stage_budget_complete'
            while not state['stage_complete'] and state['real_tokens'] < target:
                remaining = args.deadline_unix - time.time()
                if stopped['signal'] or remaining < 180:
                    stop_reason = 'signal' if stopped['signal'] else 'wall_time'; break
                if args.max_steps and state['completed_steps'] - state['base_steps'] >= args.max_steps:
                    stop_reason = 'debug_step_limit'; break
                indexer = stage == 'sft' or indexer_phase_enabled(state['completed_steps'], train, config)
                if indexer not in executables and remaining < 1200:
                    stop_reason = 'wall_time_before_compile'; break
                pool = choose_pool(state['pool_tokens'], weights)
                try:
                    batch = pools[pool].batch(state['cursors'][pool])
                except StopIteration:
                    raise RuntimeError(f'{stage}/{pool} exhausted: no silent repeats or mix substitution')
                counts = batch_counts(batch, length, sft=stage == 'sft')
                ids, seg, mask = put_training_batch(batch['input_ids'], batch['segment_ids'], batch['token_mask'], config, mesh)
                step = state['completed_steps'] if stage == 'midtrain' else state['stage_steps']
                inputs = (ids, seg, jnp.asarray(step, jnp.int32), mask)
                if stage == 'sft':
                    inputs += (jax.device_put(batch['sft_loss_mask'], batch_named_sharding(config, mesh)),)
                if indexer not in executables:
                    checkpoint('before_compile')
                    print(f'Compiling {stage}, {length} tokens, indexer={indexer}', flush=True)
                    fn = compile_pretrain_step(params, opt, specs, config, train, mesh,
                        include_indexer=indexer, assistant_only=stage == 'sft', native_config=native)
                    executable, diagnostics = compile_diagnostics(fn, params, opt, *inputs)
                    executables[indexer] = executable
                    atomic_json(output / f'{stage}-compile-{indexer}.json', diagnostics)
                    if args.deadline_unix - time.time() < 180 or stopped['signal']:
                        stop_reason = 'wall_time_after_compile'; break
                start = time.monotonic()
                params, opt, metrics = executables[indexer](params, opt, *inputs)
                m = jax.device_get({k: metrics[k] for k in ('loss','lm_loss','indexer_loss','lm_tokens',
                                                          'expert_dropped','moe_mosaic_layers','learning_rate')})
                jax.block_until_ready((params,opt))
                if not all(math.isfinite(float(m[k])) for k in ('loss','lm_loss','indexer_loss')):
                    raise FloatingPointError('Nonfinite update; last committed checkpoint retained')
                if int(np.asarray(m['expert_dropped']).sum()) or int(m['moe_mosaic_layers']) != config.n_layers:
                    raise RuntimeError('Native MoE validation failed')
                if int(m['lm_tokens']) != counts['lm_tokens']:
                    raise RuntimeError('Shifted supervision count mismatch')
                state['completed_steps'] += 1; state['stage_steps'] += 1
                state['cursors'][pool] += 1; state['pool_tokens'][pool] += counts['real_tokens']
                for k, v in counts.items(): state[k] += v
                if state['stage_steps'] % args.log_every == 0 or state['stage_steps'] == 1:
                    row = {'event':'train','stage':stage,'step':state['stage_steps'],'pool':pool,
                        'real_tokens':state['real_tokens'],'lm_tokens':state['lm_tokens'],
                        'seconds':time.monotonic()-start,
                        **{k:float(m[k]) for k in ('loss','lm_loss','indexer_loss','learning_rate')}}
                    print(json.dumps(row), flush=True); log.write(json.dumps(row)+'\n')
                if state['stage_steps'] % args.checkpoint_every == 0 or state['stage_steps'] == 1:
                    checkpoint('periodic')
                if state['stage_steps'] % args.eval_every == 0:
                    checkpoint('before_validation'); evaluate()
            if state['real_tokens'] >= target:
                state['stage_complete'] = True
            checkpoint(stop_reason)
            if not state['stage_complete']:
                report.update(status='paused', stop_reason=stop_reason); break
            evaluate()
            if stage == 'sft':
                report.update(status='completed', stop_reason='midtrain_and_sft_complete'); break
            # SFT is a new optimizer/LR schedule. Midtrain kept the pretraining state.
            config, train, native = stage_recipe(base_recipe, 'sft', sft_tokens, args.sft_lr, sft_steps)
            opt, _ = init_optimizer_state_sharded(params, specs, config, mesh)
            state.update(stage='sft', stage_steps=0, real_tokens=0, lm_tokens=0, physical_tokens=0,
                         cursors={k:0 for k in corpus['pool_mix']['sft']},
                         pool_tokens={k:0 for k in corpus['pool_mix']['sft']}, stage_complete=False)
            executables.clear(); evaluator = None; jax.clear_caches()
        report['finished_unix'] = time.time()
        atomic_json(output / 'summary.json', report)
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        atomic_json(output / 'summary.json', report)
        raise
    finally:
        log.close()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--pretrained', type=Path)
    g.add_argument('--resume', type=Path)
    p.add_argument('--sft-tokens', type=int, default=0, help='0: largest safe one-pass budget at 1:2 reasoning/agent')
    p.add_argument('--sft-lr', type=float, default=2.6e-5)
    p.add_argument('--seed', type=int, default=1701)
    p.add_argument('--deadline-unix', type=float, default=None)
    p.add_argument('--checkpoint-every', type=int, default=500)
    p.add_argument('--eval-every', type=int, default=2000)
    p.add_argument('--eval-batches', type=int, default=2)
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--max-steps', type=int, default=0)
    a = p.parse_args()
    if a.deadline_unix is None: a.deadline_unix = time.time() + 8*3600
    if a.sft_tokens < 0 or a.max_steps < 0 or a.sft_lr <= 0 or min(a.checkpoint_every,a.eval_every,a.eval_batches,a.log_every) <= 0:
        p.error('Invalid budget, LR or intervals')
    return a


if __name__ == '__main__':
    run(parse_args())
