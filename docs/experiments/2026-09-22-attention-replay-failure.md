# Attention replay: mixed-precision failure before timing

Status: **physical TPU run attempted; all 36 pairs failed before timing**.
This is an implementation failure, not evidence for or against the performance
hypotheses. The correction is locally validated; the corrected TPU run is pending.

## Recorded result

- Public notebook: [nano-dsv4-1f output](https://www.kaggle.com/code/xiayicheng3gmailcom/nano-dsv4-1f/output).
- Output version: `351680820`; directory `attention-hypothesis-20260922-010806`.
- Code: `2c455587fd44b32d29fcc7fb4382eb112b2d766c`.
- TPU v5e-8; JAX/jaxlib 0.10.2, libtpu 0.0.42.1, Tokamax 0.0.12.
- Default experiment: 3 attention families × repeated/distinct inputs ×
  4/8 global rows × 3 repeats. Tile and full-model extensions were disabled.
- Real-corpus capture succeeded. Captured Q/K/V and cotangents were **BF16**;
  sinks were **FP32**. All 36 cases stopped at `compile_and_check:sequential`.
- The baseline `vmap` compiled and executed in 24 pairs where it came first.
  The other 12 pairs tried sequential first and stopped immediately.
- **Zero timing samples and zero TPU traces** were collected. The worker
  intentionally checks both variants before timing. Compile seconds are not
  attention execution time and cannot be used to calculate S2.

[Compact evidence](2026-09-22-attention-replay-failure.json) preserves the original
report hashes, source version, environment, captured array metadata, all 36
case outcomes and a representative traceback.

## Cause

Every case reported:

```text
AssertionError: (ShapedArray(float32[8]), ShapedArray(bfloat16[8]))
```

The same assertion reproduces locally on eight simulated CPU devices with
Splash interpretation, BF16 Q/K/V and FP32 sinks. The failing path is
`lax.map` → scan transpose → gradient accumulation → dtype mismatch.

In the pinned [JAX 0.10.2 Splash backward implementation](https://github.com/jax-ml/jax/blob/jax-v0.10.2/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py),
`dsinks` is reduced from expressions in the attention output dtype, BF16 here,
and returned without converting to the FP32 sink input dtype. The scan backward
must accumulate into an FP32 sink-gradient carry and rejects the BF16 value.
This failure occurs during autodiff/lowering, before compiling the sequential
TPU executable. It is not a VMEM/HBM allocation error or a numerical gate failure.

Earlier dense-oracle and full-step tests used FP32 differentiable attention
inputs. The BF16 snapshot test only ran forward. This gap allowed the mixed
payload/control backward failure to pass the earlier CPU checks.

## Correction and rerun

A small custom-VJP wrapper converts Splash's returned sink cotangent to the
original sink dtype **before** the outer map accumulates gradients. Both vmap
and sequential schedules use the same correction. Forward sink values, Q/K/V
precision, masks, tiles and the underlying Splash kernels remain unchanged.
This does not recompute the per-row BF16 sink derivative at higher precision;
it fixes its return type and enables FP32 accumulation across rows. The corrected
baseline must be measured anew; do not splice in timings from earlier code.

New regression checks cover all three attention families, 4/8 global rows,
BF16 activations and FP32 sinks. The actual replay worker is also tested with
that dtype pairing, including materialized backward residuals and preflight-only
execution. Existing numerical thresholds are unchanged.

The supervisor now runs an untimed 8-row TPU preflight for each family before
starting the 36-pair matrix. It compiles and checks combined, forward-only and
backward-only paths. Any preflight failure saves the error and stops the sweep.
The notebook displays the failure and still allows exporting the compact ZIP.

Use the updated [attention hypothesis notebook](../../notebooks/nano_dsv41f_attention_hypothesis.ipynb)
with `NANO_DSV41F_REF=codex/attention-replay-mixed-precision`. This follow-up branch
is based on the merged PR #5 target, `codex/sft-candidate-pool`; at analysis time
`main` did not yet contain the profiling notebooks. Dataset paths and the
registered experiment controls remain the same.

## What this says about batch size

H1/H2 remain inconclusive; H4/VMEM remains unresolved. There is no new timing or
capacity result and no reason from this failed experiment to restrict training
to four rows. Earlier full-model runs still establish that 24 rows can execute,
while four rows was the best measured throughput point in that comparison.
