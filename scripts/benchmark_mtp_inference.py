"""Compare scalar/chunked prefill and sequential/batched greedy MTP verification.

Real bundles use actual draft proposals. --synthetic --controlled additionally
measures a perfect-proposal ceiling while still paying for the real draft pass.
That controlled case is not a trained-model acceptance or throughput claim.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import torch
from nano_dsv41f.vllm_v41_cpu import NanoDeepseekV41CPU
from nano_dsv41f.vllm_v41_cpu.session import InferenceSession
import nano_dsv41f.vllm_v41_cpu.dspark as draft_module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--model-dir', type=Path)
    source.add_argument('--synthetic', action='store_true')
    parser.add_argument('--controlled', action='store_true')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--chunk-size', type=int, default=32)
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--prompt', default='Explain how speculative decoding accelerates inference.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.threads, args.repeats, args.chunk_size, args.max_new_tokens) < 1:
        parser.error('threads, repeats, chunk-size and max-new-tokens must be positive')
    if args.controlled and not args.synthetic:
        parser.error('--controlled is restricted to the explicitly synthetic benchmark')
    torch.set_num_threads(args.threads)
    if args.synthetic:
        sys.path.insert(0, str(ROOT/'tests'))
        import jax
        from test_vllm_v41_cpu import tiny_config
        from nano_dsv41f.config import DSparkConfig
        from nano_dsv41f.model import init_model
        from nano_dsv41f.hf_export import flatten_parameter_tree
        cfg = replace(tiny_config(), dspark=DSparkConfig(enabled=True, block_size=5,
            target_layer_ids=(4, 5, 6), markov_rank=4, n_routed_experts=4, experts_per_token=1))
        model = NanoDeepseekV41CPU(cfg, flatten_parameter_tree(init_model(jax.random.PRNGKey(41), cfg)), device=args.device)
        ids = (torch.arange(128, device=model.device)[None] % 60) + 1
    else:
        from nano_dsv41f.portable_bundle import verify_portable_bundle
        from tokenizers import Tokenizer
        report = verify_portable_bundle(args.model_dir)
        if report.get('dspark_steps', 0) <= 0:
            parser.error('real-model MTP benchmarks require a distilled bundle with recorded draft updates')
        model = NanoDeepseekV41CPU.from_pretrained(args.model_dir, device=args.device)
        tokenizer = Tokenizer.from_file(str(args.model_dir/'tokenizer.json'))
        ids = torch.tensor([tokenizer.encode(args.prompt).ids], device=model.device)
    capacity = ids.shape[1] + args.max_new_tokens + model.config.dspark.block_size
    if capacity > model.max_context:
        parser.error('prompt and output exceed model context')
    # All comparisons share the scalar target's continuation. Fail on drift.
    reference = InferenceSession(model, capacity=capacity, prefill_chunk_size=1).generate(
        ids, max_new_tokens=args.max_new_tokens + model.config.dspark.block_size)
    original_draft = draft_module.draft_block
    def perfect_with_draft_cost(model, cache):
        original_draft(model, cache)
        return reference[:, cache.length:cache.length+model.config.dspark.block_size]
    modes = [('scalar', False, 1, 'sequential', False),
             ('chunked', False, args.chunk_size, 'batched', False),
             ('mtp_real_sequential', True, args.chunk_size, 'sequential', False),
             ('mtp_real_batched', True, args.chunk_size, 'batched', False)]
    if args.controlled:
        modes += [('mtp_perfect_sequential', True, args.chunk_size, 'sequential', True),
                  ('mtp_perfect_batched', True, args.chunk_size, 'batched', True)]
    results = {name: [] for name, *_ in modes}
    try:
        # Round-robin modes to reduce warmup/order bias; one unrecorded round.
        for repeat in range(-1, args.repeats):
            for name, mtp, chunk, verifier, controlled in modes:
                draft_module.draft_block = perfect_with_draft_cost if controlled else original_draft
                session = InferenceSession(model, capacity=capacity, mtp=mtp,
                    allow_untrained_draft=args.synthetic, draft_trained=not args.synthetic,
                    prefill_chunk_size=chunk, mtp_verifier=verifier)
                if model.device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(model.device)
                actual = session.generate(ids, max_new_tokens=args.max_new_tokens)
                torch.testing.assert_close(actual, reference[:, :actual.shape[1]], rtol=0, atol=0)
                if repeat >= 0:
                    row = dict(session.last_stats)
                    row['peak_cuda_allocated_bytes'] = (torch.cuda.max_memory_allocated(model.device)
                        if model.device.type == 'cuda' else None)
                    results[name].append(row)
    finally:
        draft_module.draft_block = original_draft
    summary = {name: {key: statistics.median(r[key] for r in rows) for key in
        ('prefill_seconds', 'prefill_tokens_per_second', 'decode_seconds',
         'decode_tokens_per_second', 'total_seconds', 'decode_target_calls',
         'mtp_proposed_tokens', 'mtp_verified_tokens', 'mtp_accepted_tokens')}
        for name, rows in results.items()}
    report = dict(torch_version=torch.__version__, device=str(model.device), dtype=str(model.dtype),
        threads=args.threads, synthetic=args.synthetic, controlled_perfect_proposals=args.controlled,
        model_dir=str(args.model_dir) if args.model_dir else None,
        parameter_count=sum(t.numel() for t in model.weights.values()),
        prompt_tokens=ids.shape[1], output_tokens=args.max_new_tokens,
        block_size=model.config.dspark.block_size, repeats=args.repeats,
        notes='Greedy raw completion; no EOS stop. All outputs equal scalar target. Model loading and reference preparation excluded. Perfect proposals are a synthetic ceiling, including real drafting cost.',
        summary=summary, results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
