"""Measure cold generation and persistent-prefix continuation on a real bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from nano_dsv41f.vllm_v41_cpu.api import NanoDeepSeekProtocolBackend
from nano_dsv41f.vllm_v41_cpu.session import format_inference_stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--prompt', default='Explain why caching speeds up language model inference.')
    parser.add_argument('--continuation', default='\nNow summarize that in one sentence.')
    parser.add_argument('--max-new-tokens', type=int, default=32)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.threads < 1 or args.max_new_tokens < 1:
        parser.error('repeats, threads, and max-new-tokens must be positive')
    torch.set_num_threads(args.threads)
    backend = NanoDeepSeekProtocolBackend.from_pretrained(args.model_dir)
    ids = torch.tensor([backend.tokenizer.encode(args.prompt)], dtype=torch.long)
    continuation = torch.tensor([backend.tokenizer.encode(args.continuation)], dtype=torch.long)
    # Warm operators separately; cold below means a cold KV cache, not cold model I/O.
    backend.session.generate(ids[:, :min(ids.shape[1], 8)], max_new_tokens=1)
    rows = []
    for repeat in range(args.repeats):
        backend.reset_cache()
        first = backend.session.generate(ids, max_new_tokens=args.max_new_tokens)
        cold = dict(backend.last_stats)
        next_prompt = torch.cat((first, continuation), dim=1)
        backend.session.generate(next_prompt, max_new_tokens=args.max_new_tokens)
        warm = dict(backend.last_stats)
        rows.append(dict(repeat=repeat, cold=cold, continuation=warm))
        print(f'Run {repeat+1}, cold cache\n{format_inference_stats(cold)}')
        print(f'Run {repeat+1}, continuation\n{format_inference_stats(warm)}')
    report = dict(torch_version=torch.__version__, device='cpu', dtype='float32',
                  threads=args.threads, model_dir=str(args.model_dir), results=rows,
                  notes='Greedy raw token continuation; no EOS stop; model loading excluded.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
