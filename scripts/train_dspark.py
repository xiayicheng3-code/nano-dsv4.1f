#!/usr/bin/env python3
"""Distill only DSpark from a frozen portable SFT model on existing SFT responses."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import torch
from nano_dsv41f.portable_bundle import sha256_file, verify_portable_bundle
from nano_dsv41f.vllm_v41_cpu import NanoDeepseekV41CPU
from nano_dsv41f.vllm_v41_cpu.dspark_training import DraftLossConfig, DraftTrainer, backbone_digest
from nano_dsv41f.vllm_v41_cpu.session import InferenceSession
from dspark_data import SFTDraftData
from dspark_artifacts import export_trained_draft, load_draft_checkpoint, save_draft_checkpoint
from pretrain_checkpoint import atomic_json


def code_digest():
    h = hashlib.sha256()
    paths = sorted((ROOT/'src/nano_dsv41f/vllm_v41_cpu').glob('*.py'))
    paths += [Path(__file__), ROOT/'scripts/dspark_data.py', ROOT/'scripts/dspark_artifacts.py']
    for path in paths:
        h.update(path.name.encode()); h.update(path.read_bytes())
    return h.hexdigest()


def evaluate(trainer, data, batches, rollout_tokens):
    metrics, token_count = {}, 0
    for i in range(batches):
        sample = data.sample(i, split='validation')
        row = trainer.evaluate(sample['input_ids'], sample['anchors'])
        n = row.pop('tokens')
        token_count += n
        for key, value in row.items():
            metrics[key] = metrics.get(key, 0) + value*n
    result = {key:value/token_count for key,value in metrics.items()}
    result['tokens'] = token_count
    proposed = verified = accepted = rounds = 0
    # This diagnostic is genuinely autoregressive; it does not compare draft
    # prefixes against a teacher-forced continuation after a rejected token.
    for i in range(min(batches, 2)):
        sample = data.sample(i, split='validation')
        prefix = sample['input_ids'][:int(sample['anchors'][0])+1]
        session = InferenceSession(trainer.model, mtp=True, draft_trained=True)
        session.generate(torch.tensor([prefix.tolist()], device=trainer.model.device),
            max_new_tokens=rollout_tokens, temperature=0)
        stats = session.last_stats
        proposed += stats['mtp_proposed_tokens']
        verified += stats['mtp_verified_tokens']
        accepted += stats['mtp_accepted_tokens']
        # Every rejection ends one round; each fully accepted block also ends one.
        rounds += 1
    result.update(rollout_examples=rounds, rollout_proposed_tokens=proposed,
        rollout_verified_tokens=verified, rollout_accepted_tokens=accepted,
        rollout_greedy_acceptance=accepted/verified if verified else None)
    return result


def run(args):
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Select a Kaggle GPU accelerator or set DEVICE=cpu for a smoke test')
    if args.output.exists():
        raise FileExistsError('Use a fresh OUTPUT directory; load earlier state with RESUME')
    args.output.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    verify_portable_bundle(args.model_dir)
    model = NanoDeepseekV41CPU.from_pretrained(args.model_dir, device=args.device, dtype=torch.float32)
    if args.seq_len + args.rollout_tokens > model.max_context:
        raise ValueError('Crop plus rollout budget exceeds exported model context')
    data = SFTDraftData(args.corpus, tokenizer_sha256=sha256_file(args.model_dir/'tokenizer.json'),
        block_size=model.config.dspark.block_size, seq_len=args.seq_len, anchors=args.anchors, seed=args.seed)
    loss_config = DraftLossConfig(args.ce_weight, args.distribution_weight, args.confidence_weight)
    trainer = DraftTrainer(model, learning_rate=args.learning_rate, loss_config=loss_config)
    identity = dict(source_bundle=sha256_file(args.model_dir/'export_manifest.json'),
        corpus_manifest=data.identity, code_sha256=code_digest(), learning_rate=args.learning_rate,
        seq_len=args.seq_len, anchors=args.anchors, seed=args.seed, dtype='float32',
        loss=asdict(loss_config), teacher='torch_sparse_serving', response_source='existing_sft_teacher_forced')
    if args.resume:
        load_draft_checkpoint(args.resume, trainer, identity)
    if trainer.steps >= args.steps:
        raise ValueError('STEPS is the total target; increase it beyond the resumed step')
    frozen_before = backbone_digest(model)
    stopped = {'value':False}
    def stop(*_):
        stopped['value'] = True
    previous_handlers = {sig:signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    report = dict(status='running', identity=identity, start_step=trainer.steps,
        completed_steps=trainer.steps, backbone_sha256_before=frozen_before,
        started_unix=time.time(), evaluations=[], trainable_parameters=sum(p.numel() for p in trainer.parameters))
    atomic_json(args.output/'summary.json', report)
    def emit(row):
        print(json.dumps(row, allow_nan=False), flush=True)
        with (args.output/'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(row, allow_nan=False)+'\n')
    def validate():
        metrics = evaluate(trainer, data, args.eval_batches, args.rollout_tokens)
        if any(v is not None and not math.isfinite(v) for v in metrics.values()):
            raise FloatingPointError('Nonfinite validation metrics')
        row = dict(event='validation', step=trainer.steps, **metrics)
        report['evaluations'].append(row)
        emit(row)
        atomic_json(args.output/'summary.json', report)
    try:
        validate()
        while trainer.steps < args.steps:
            if stopped['value'] or time.time() > args.deadline_unix-180:
                break
            sample = data.sample(trainer.steps)
            if sample['input_ids'].min() < 0 or sample['input_ids'].max() >= model.config.vocab_size:
                raise ValueError('Corpus contains token IDs outside the frozen model vocabulary')
            if model.device.type == 'cuda':
                torch.cuda.synchronize()
            started = time.perf_counter()
            row = trainer.train_step(sample['input_ids'], sample['anchors'])
            elapsed = time.perf_counter()-started
            report['completed_steps'] = trainer.steps
            if trainer.steps == 1 or trainer.steps % args.log_every == 0:
                emit(dict(event='train', step=trainer.steps, seconds=elapsed,
                    crop_tokens=len(sample['input_ids']), pool=sample['pool'], **row))
            if trainer.steps % args.checkpoint_every == 0:
                save_draft_checkpoint(args.output/'checkpoints', trainer, identity)
            if trainer.steps % args.eval_every == 0 and time.time() < args.deadline_unix-300:
                validate()
        save_draft_checkpoint(args.output/'checkpoints', trainer, identity)
        if report['evaluations'][-1]['step'] != trainer.steps and time.time() < args.deadline_unix-180:
            validate()
        frozen_after = backbone_digest(model)
        if frozen_after != frozen_before:
            raise RuntimeError('Frozen backbone changed; export refused')
        report.update(backbone_sha256_after=frozen_after, backbone_unchanged=True,
            completed_steps=trainer.steps, status='completed' if trainer.steps >= args.steps else 'paused',
            finished_unix=time.time())
        if trainer.steps:
            export_trained_draft(args.model_dir, args.output/'bundle', trainer, report)
            report['bundle'] = str(args.output/'bundle')
        atomic_json(args.output/'summary.json', report)
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        atomic_json(args.output/'summary.json', report)
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    return report


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', type=Path)
    p.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--seq-len', type=int, default=512)
    p.add_argument('--anchors', type=int, default=8)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--ce-weight', type=float, default=0.1)
    p.add_argument('--distribution-weight', type=float, default=0.9)
    p.add_argument('--confidence-weight', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=1701)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--log-every', type=int, default=10)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--checkpoint-every', type=int, default=100)
    p.add_argument('--eval-batches', type=int, default=4)
    p.add_argument('--rollout-tokens', type=int, default=16)
    p.add_argument('--deadline-unix', type=float, default=None)
    a = p.parse_args(argv)
    if min(a.steps,a.seq_len,a.anchors,a.threads,a.log_every,a.eval_every,a.checkpoint_every,a.eval_batches,a.rollout_tokens) < 1 or not 0 < a.learning_rate < float('inf') or a.seed < 0:
        p.error('Counts must be positive, seed nonnegative, and learning rate finite and positive')
    if a.deadline_unix is None:
        a.deadline_unix = time.time()+8*3600
    if not math.isfinite(a.deadline_unix):
        p.error('Invalid deadline')
    return a


if __name__ == '__main__':
    run(parse_args())
