# Tokamax 0.0.12 routed MoE

The native backend replaces its custom prefix-sum/per-expert tile loop with
`tokamax.ragged_dot`. Tokamax supplies the grouped matrix products and their input
and weight VJPs. Router equations, clipping, shared expert, parameter shapes,
FP32 mixture accumulation, and the retriever query-budget policy are preserved.

## Runtime and upstream API

The Kaggle stack pins Tokamax 0.0.12, JAX/jaxlib 0.10.2 and libtpu 0.0.42.1.
Tokamax requires Python 3.11+, reflected in project metadata. CI pins the same JAX
version on CPU. Install the TPU requirements in a fresh session before JAX imports.
Both generated notebooks record the Tokamax version and run Mosaic forward/backward
parity before initializing the model.

Sources inspected for this version:

- [PyPI release and provenance](https://pypi.org/project/tokamax/0.0.12/)
- [Release-source ragged-dot API](https://github.com/openxla/tokamax/blob/964354016004720905931f1249706a1772706752/tokamax/_src/ops/ragged_dot/api.py)
- [TPU implementation and VJP selection](https://github.com/openxla/tokamax/blob/964354016004720905931f1249706a1772706752/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py)

Version 0.0.12's public API rejects `group_offset`. The adapter therefore packs
each chip's local groups starting at row zero and passes local group sizes. It
does not call private upstream kernels in production. Tests alone use the private
interpreter entry point to execute the actual TPU forward/VJP kernels on CPU.
Tokamax's lazy Abseil flag parsing is initialized with `known_only=True` so our
argparse and notebook command-line options remain valid.

## Dispatch and compute

1. Route the local token shard using the existing FP32 router; aggregate real-token
   router-load counts inside the flat expert mesh, preserving the sharding fix.
2. All-gather activations, expert IDs and mixture weights. Expert matrices stay resident.
3. Sort assignments by local expert ID, with non-owned assignments last. A histogram
   provides dynamic group sizes; there is no `[local_experts, assignments]` match matrix.
4. Apply one grouped gate/up projection, the existing clipped SwiGLU, and one grouped
   down projection. Tokamax's Mosaic VJPs perform grouped input/weight gradients.
5. Weight and scatter into one FP32 token accumulator; reduce-scatter to the original
   token shards and add the local dense shared-expert result.

With `N=B*T`, top-K `k`, and `E_local` resident experts, each chip reserves
`N * min(k, E_local)` assignment rows. Top-K expert IDs are distinct, so this buffer
holds every possible local assignment even when routing is maximally imbalanced.
There is no token-dropping capacity factor. Empty chips skip their expert GEMMs;
collectives stay outside the data-dependent branch.

Mosaic 0.0.12 requires non-expert dimensions of at least 128. The adapter pads
dimensions to multiples of 128 and slices the outputs, preserving small nano
dimensions. The model's ordinary 512/768 dimensions need no channel padding.
Unassigned rows are explicitly zeroed on both sides of every ragged dot: the
upstream kernel can leave unused output rows unwritten, which otherwise risks
NaNs during nonlinear operations or backward.

## Controls and diagnostics

`TPUNativeConfig.moe_ragged_implementation` accepts:

- `auto` (default): explicit Mosaic on TPU; Tokamax XLA on CPU.
- `mosaic`: require the TPU kernel, including its backward kernels.
- `xla`: Tokamax's JAX ragged-dot implementation for reference/ablation.

The Kaggle smoke explicitly requires `mosaic` and verifies every MoE layer reports
it. Unsupported expert-mesh shapes raise an error. The readable per-token-weight
MoE remains an independent numerical oracle and explicit reference backend.

`moe_capacity_factor` and `moe_capacity_multiple` are accepted only for configuration
compatibility and no longer affect dispatch. `expert_overflow` and `expert_dropped`
are zero. The legacy `expert_capacity` diagnostic is the physical per-expert upper
bound `B*T`; `expert_packed_rows` reports the per-chip assignment-buffer size.
`moe_mosaic_layers` and the runtime report identify the selected kernel.

## Validation and measurement

Tests cover output/input/parameter gradients, FP32 and BF16, clipping, empty groups,
unused rows, channel padding, one/multiple experts per chip, extreme imbalance,
two batch rows, padding-excluded router loads, and the outer 2x4 to flat expert mesh.
The Pallas interpreter tests exercise Tokamax's actual Mosaic forward and both VJPs.
Interpreter execution supplies simulated v5e geometry and is not hardware timing.

```bash
JAX_PLATFORMS=cpu pytest -q tests/test_tokamax_moe.py tests/test_operator_pass.py
XLA_FLAGS=--xla_force_host_platform_device_count=8 JAX_PLATFORMS=cpu \
  pytest -q tests/test_tokamax_moe.py tests/test_operator_pass.py \
  -k 'ragged_ep or invalid_expert_mesh or dropless_expert or router_loads_cross_outer'
```

On Kaggle, use a fresh session and the regenerated main notebook to run operator
parity/timing followed by the combined training smoke. Compare to commit `202e935`
at the same model dimensions, precision, query budget, packed data and runtime.
The JSON reports include synchronized timing, compilation, memory, collectives,
kernel identity, parameter configuration and zero-drop checks.

No physical TPU is available in the editing environment. The expected compute
benefit is avoiding padding every expert to the same tile size; it is not a
measured throughput improvement. This path still all-gathers tokens, sorts IDs,
allocates worst-case buffers and runs pointwise work over those buffers. Activation
memory can exceed the former rematerialized tile loop; the full model retains
block rematerialization. Measure end-to-end time and HBM, not only grouped GEMMs.
Ragged all-to-all communication and automatic tuning are separate work.
