# CPU cache development measurement — 2026-10-04

This uses random **tiny test weights**, not the trained SFT checkpoint. Torch
2.14.1+cpu, CPU, one Torch thread, 128 prompt tokens, 16 generated tokens,
then a continuation containing that output plus four new tokens and another 16
output tokens. Operators are warmed with a short request; model loading is excluded.
Three trials, median wall time, greedy output, no EOS stop. Baseline is remote
`ad8c8e59` (local equivalent `192b712`). This is a microbenchmark, not a Kaggle,
Colab, T4, or full-model throughput claim.

| Request | Before | After | Before / after |
|---|---:|---:|---:|
| Cold KV cache | 1.350 s | 1.225 s | 1.10× |
| Continuation | 1.546 s | 0.172 s | 9.01× |

After-change medians (phase rates are medians of individual trial rates):

| Request | New prefill tokens | Reused tokens | Prefill seconds | Prefill tok/s | Decode tokens | Decode seconds | Decode tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cold | 128 | 0 | 1.093 | 117.13 | 16 | 0.131 | 122.11 |
| Continuation | 5 | 143 | 0.042 | 117.85 | 16 | 0.130 | 123.17 |

The last output token is not cached until needed by a subsequent request, so the
continuation processes five tokens and reuses 143. The test cache capacity is 256;
its arena remains fixed across the two turns. Most of the continuation benefit is
avoided prefill. Cold-run timing differences are small enough to depend on workload
and CPU overhead; longer prompts and the trained model need their own measurement.

MTP is disabled in this performance measurement. Its guarded path is tested for
Torch/JAX draft-logit parity, greedy output preservation, rejection, acceptance,
EOS and continued/rewritten prompts. It has no accelerated target verifier yet.

Reproduce with the same Python environment and separate baseline/current checkouts:

```bash
python scripts/benchmark_cache_synthetic.py /path/to/baseline > before.json
python scripts/benchmark_cache_synthetic.py /path/to/current > after.json
```

Use `scripts/benchmark_cpu_inference.py --help` to measure a real exported bundle.
Full trial data and decode-step timings are in
[the JSON report](2026-10-04-inference-cache.json).
