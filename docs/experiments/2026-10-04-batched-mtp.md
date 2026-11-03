# Chunked prefill and batched MTP — 2026-10-04

Local CPU microbenchmark, **synthetic random weights** (217,164
parameters), FP32, one Torch thread, 128 prompt tokens,
64 output tokens, draft block size 5, prefill chunk 32.
Torch 2.14.1+cpu. Median of 3 recorded runs after an unrecorded
warmup round. Modes run round-robin. Raw trials are in
[the JSON report](2026-10-04-batched-mtp.json).

| Mode | Prefill s | Decode s | Decode tok/s | Target decode calls | Accepted/verified |
|---|---:|---:|---:|---:|---:|
| Scalar prefill, ordinary decode | 1.797 | 0.927 | 69.1 | 63 | 0/0 |
| Chunked prefill, ordinary decode | 0.089 | 0.908 | 70.5 | 63 | 0/0 |
| Random draft, sequential verifier | 0.088 | 1.019 | 62.8 | 63 | 2/64 |
| Random draft, batched verifier | 0.088 | 1.106 | 57.9 | 63 | 2/64 |
| Controlled perfect draft, sequential | 0.101 | 0.971 | 65.9 | 63 | 64/64 |
| Controlled perfect draft, batched | 0.095 | 0.233 | 274.4 | 13 | 64/64 |

Prefill batching is 20.25×
faster than the same runtime with chunk size 1. For the controlled perfect drafter,
batched verification is 4.16×
faster than sequential verification, and
3.89× faster
than ordinary decode. Target decode calls drop from 63 to 13 for 64 output tokens.

**The perfect-draft rows are an artificial acceptance ceiling.** The benchmark
executes the real DSpark forward (including its cost) and then substitutes the
precomputed target continuation. These rows do not measure learned acceptance.
The random draft accepts only 2/64 verified proposals and is slower than ordinary
decode. No trained nano model, convergence result, T4 speedup, or full-scale memory
claim follows from this microbenchmark.

All recorded generated token sequences exactly equal the scalar greedy target.
Model loading, warmup, and preparing the controlled continuation are excluded.
Prompt throughput counts new prompt tokens. Decode time includes drafting,
verification, rollback, sampling and loop overhead; decode throughput uses emitted
tokens, not all speculative tokens computed. Speculative work and per-batch timings
are available in the JSON. Timings depend on host load and tiny models emphasize
Python/operator overhead more than full model weight bandwidth.

Reproduce (development dependencies including JAX required for the fixture):

```bash
python scripts/benchmark_mtp_inference.py --synthetic --controlled \
    --output /tmp/batched-mtp.json
```

Measure a distilled real bundle with its actual draft proposals:

```bash
python scripts/benchmark_mtp_inference.py --model-dir /path/to/dspark-bundle \
    --device cpu --threads 2 --repeats 3 --output /path/to/results.json
```

Use `--device cuda` on T4. The script records CUDA peak allocated memory and checks
greedy output equality; it does not replace actual draft proposals for real bundles.

Validation: 119 tests passed in the combined CPU inference, DSpark training,
notebook, protocol, export and upload suite. Eight additional short-prefix rollback
cases passed, for 127 tests total. Coverage includes dense/sparse chunk equivalence,
packed segment boundaries, causal retrieval ties, every speculative commit prefix,
odd compression groups, ring wraparound, rejection at every draft position,
EOS/budget termination, continuation and failed-generation cache invalidation.
