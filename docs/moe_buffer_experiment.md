# Reduced MoE assignment buffers

Run [the preset notebook](../notebooks/nano_dsv41f_moe_buffers.ipynb) in a fresh
Kaggle TPU v5e-8 session, using the same public tokenized corpus and tokenizer.

## Fixed settings

- Four global 8192-token rows; attention CP2/DP4 and MoE EP8.
- Sequential Splash; local `block_q_dkv=128`, compressed/global `1024`.
- Seven layers, d_model512, 48 routed experts of width128, top-4, one shared expert.
- BF16 payloads, FP32 controls, pinned JAX/Tokamax stack, existing optimizer,
  block rematerialization, QAT off, DSpark inactive, existing indexer policy.

The tile choice has physical TPU evidence. Architecture, precision and remat are
held constant to isolate the buffer change; this does not claim they are optimal.

## Intervention and safety semantics

`TPUNativeConfig.moe_buffer_divisor` accepts 1 (default), 2 or 4. At four rows,
the per-chip fast buffer contains 131072, 65536 or 32768 assignments respectively.
The original bound is `N * min(top_k, local_experts)`, independent of balancing.
A chip with more assignments than its fast buffer executes the full-buffer branch.
The condition is evaluated before slicing or calling ragged_dot. Both branches
produce the same full token-output shape. All collectives stay outside the local
conditional, so chips may choose different branches without collective divergence.

The full branch remains available for arbitrary routing skew; no token is dropped
and no average-load assumption changes training semantics. Both branches compile,
so compiler-reserved memory need not fall in proportion to fast-buffer capacity.
Autodiff residual allocation and compiler lowering may also limit speedups.

The smaller buffer reduces the logical input size to gather, masking, weighting
and combine/scatter. All-gather, full assignment sorting, token-output size and
reduce-scatter payload remain unchanged. Profile actual savings; do not infer a
4x speedup from a 4x smaller fast buffer.

## Protocol

Three fresh preflight workers compare the full buffer with each configuration,
using production expert dimensions (512/128, 48 experts, top-4) at a reduced token
count. Normal and extreme skew cases check outputs and all input/parameter gradients
with the existing 0.01 normalized RMS gate. Skew forces overflow for reduced buffers
and leaves other chips empty. Passing preflights enable full-model runs.

Nine full-model workers rotate capacity order across three process repeats. Each
measures base and late phases, with three warmups and twelve unprofiled samples.
Three additional trace steps follow timing in every phase/repeat, preserving equal
update counts across comparisons. The input is the same nested four-row prefix
sampled with seed1701 as the tile experiment; model initialization uses seed7.
No replay tile sweep, batch-size sweep or capacity search runs.

Reports preserve physical chip loads, nominal fast-buffer utilization and fallback
chip-layer counts. The nominal utilization can exceed 1 on a fallback; it does not
mean dropped assignments. Counts describe logical layer calls in recorded steps,
not extra executions caused by rematerialization. Worker failures skip that
configuration's remaining repeats; baseline failure blocks remaining comparisons.

Timing screening requires >=5% lower step time in every pair and both phases and
<=5% process spread in both arms. Source, input and non-buffer settings must match;
matched-step losses and routing histograms use the existing short-trajectory gates.
All raw results are retained. The initialized fixed-batch experiment does not
establish long-training routing behavior or full optimizer-state equivalence.

Download `moe-buffer-reports.zip` for compact reports. Full traces and HLO remain
in the public Kaggle outputs for combine/scatter attribution. No production
default is automatically changed.
