# TPU v5e operator pass — 2026-09-17

The tiled MoE described below is the historical first pass. It has since been
replaced by [Tokamax 0.0.12 ragged dots](tokamax_moe.md); that page documents the
current dispatch, backward kernels and diagnostics.

## Regression diagnosis

The supplied `nano-dsv4-1f (6).log` runs commit `8a89bf6`, JAX 0.10.2, and a TPU
client built on **June 12, 2025**. It fails in Pallas lowering before the first step.
The combined notebook installed the package with `--no-deps` but omitted the
`jax[tpu]` alignment already present in the other notebook. The old notebook builder
also still advertised keeping Kaggle's preinstalled runtime unchanged.

Both notebooks now come from one standard-library bootstrap and pin JAX 0.10.2,
jaxlib 0.10.2 and libtpu 0.0.42.1. That libtpu series is specified by JAX 0.10.2's
TPU extra, and 0.0.42.1 previously reached model compilation in log (4).
A fresh process prints the **loaded platform build**, then compiles and executes
Splash forward/backward with segments and a sink before model initialization.
No version gate is disabled. Existing tracked checkout edits are preserved.

## Operator changes

| Area | Change | Preserved behavior / limitation |
| --- | --- | --- |
| Expert dispatch | Prefix-sum compaction and static tiled scan replace boolean Top-K truncation | Every routed assignment executes, including overload; capacity is a tile size |
| Expert payload | Gate/up share one wide GEMM; one flat FP32 accumulation buffer | Same clipping, SwiGLU and shared/routed experts; floating-point operation order changes |
| Expert communication | Reduce-scatter replaces all-reduce followed by slicing | Expert weights remain resident; compact inputs still use all-gather |
| Late teacher | Reconstruct complete LSE only for selected query rows | Local window, compressed history and sink all remain in denominator; teacher stays detached |
| Indexer | Build K once per KV owner; no native `[B,T,K]` validity tensor | Full/Reindex ownership and candidate hierarchy retained |
| Muon | Factor polynomial into three GEMMs per iteration instead of four | Same coefficients, iteration counts, momentum and scaling; numerical tolerance tested |
| LM loss | `logsumexp(logits) - target_logit` | Same packed-token objective and gradients; avoids full log-softmax intermediate |

`expert_overflow` now means assignments beyond the **first tile**, all of which are
processed. `expert_dropped` must always be zero. The static scan bound is sufficient
for every token to choose the same expert. Empty later tiles skip their expert
GEMMs; there are no collectives inside divergent per-chip branches. Scan-body
rematerialization avoids retaining every tile's expert activations for backward.
This is a correctness-first improvement over truncation, not a claim that all-gather
is optimal for every batch/expert shape. Severe imbalance still costs extra tiles.

## Fidelity corrections from primary sources

Primary snapshot: DeepSeek's Hugging Face repository revision
`dba1be0a40aa45a94ad051997016db3960a90277`:

- [Technical report, sections 2.1, 2.5 and 4.2.2](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/DeepSeek_V41_Tech_Report.pdf)
- [Released mHC kernel](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/kernel.py)
- [Released model, including Gate, Expert and Block](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/model.py)
- [JAX TPU installation guidance](https://docs.jax.dev/en/latest/installation.html#google-cloud-tpu)

The existing mHC code used softmax for both pre/post gates, scaled the additive base,
and transposed the released residual-mixing convention. It now uses sigmoid + eps
for pre, twice sigmoid for post, scale-before-base, the released Sinkhorn epsilon
placement, and the correct input-to-output stream direction. Stream mixing
accumulates in FP32 and casts back at the residual boundary. Coefficient RMS uses
`norm_eps`, separate from Sinkhorn's `mhc_eps`.

The text routing correction bias now updates from each layer's globally aggregated
real-token loads, with speed 0.001. It is excluded from gradient/decay updates.
AdamW no longer decays biases/scales (normalization weights still decay). Engram
uses the reported 5x learning rate; Sinkhorn epsilon is corrected to 1e-20.

**Checkpoint behavior changes:** existing parameter shapes still load, but corrected
mHC equations and optimizer/controller defaults change outputs and continuation
behavior. Start a fresh training run for an apples-to-apples corrected experiment.
This patch is not numerically equivalent to the old, incorrect mHC model.

## Reproduction boundaries

The report explicitly trains sparse attention from scratch. This repository still
uses dense compressed attention for LM pretraining plus a selective late-indexer
objective. That is the existing nano experiment, **not the official training recipe**.
A production sparse attention backward kernel and its training schedule remain a
separate milestone. The selected teacher optimization preserves this nano objective.

Other remaining differences include the sequence-level balancing auxiliary loss,
vision/image routing, full corpus/tokenizer, 64K-to-1M context curriculum, token-budget
learning-rate schedule, scale and precision/master-weight policy. The existing
nano warmup/decay settings are not relabeled as the official 45T-token schedule.
DSpark remains frozen in backbone pretraining; its separate training stage is not
implemented by this smoke. Fake QAT is not a packed FP4 TPU kernel.

## Validation and how to reproduce

Local validation uses JAX 0.10.2 on CPU, including an eight-device CPU mesh. The
reference suite, extreme-load MoE forward/input/parameter gradients, selected teacher
versus full softmax, mHC equations/orientation, controller/decay rules, notebook
bootstrap regression, and Splash **interpreter** forward/backward are covered.
The interpreter checks include both compression ratios, segment boundaries and sink
gradients. A full tiny native backbone + late-indexer loss/gradient comparison is
included in `tests/test_operator_pass.py`.

```bash
JAX_PLATFORMS=cpu pytest -q
XLA_FLAGS=--xla_force_host_platform_device_count=8 JAX_PLATFORMS=cpu \
  pytest -q tests/test_operator_pass.py tests/test_tpu_multi_expert.py
python scripts/build_notebook.py
```

No TPU device is available in the editing environment. CPU tests are **not** evidence
of TPU compilation, HBM usage or speedup. The Kaggle notebooks provide those gates:

1. Fresh TPU v5e-8 session with Internet, import the notebook from this branch.
2. Before bootstrap, set the requested branch when testing an unmerged PR:
   `import os; os.environ['NANO_DSV41F_REF'] = 'codex/tpu-v5e-operator-pass'`.
3. Run the combined notebook for one preflight + 10 base + 10 late steps, or the main
   notebook to also run small FP32/BF16 operator parity and timing first.
4. Download `/kaggle/working/combined-smoke.json` and, for the main notebook,
   `/kaggle/working/operator-benchmark.json`.

Reports record exact commit, runtime package/build metadata, model/native settings,
compiler diagnostics, synchronized step times, real LM tokens/s, losses, zero-drop
checks and packing utilization. Compilation and the first two steps per phase are
excluded from steady-state medians. Packing utilization is not static Splash block
utilization. Compare runs at identical dimensions, precision, documents and phases;
no measured TPU speedup is claimed in this change.
