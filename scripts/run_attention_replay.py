"""One isolated TPU replay pair: fixed inputs, numerical gate, synchronized timing."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
from attention_replay_utils import error_metrics, select_rows
from run_pretrain_stress import device_memory, save_report


def make_function(mesh, meta, mode, *, tile=None, interpret=False, independent_values=False):
    from nano_dsv41f.tpu_native import (TPUNativeConfig, TPUNativeState,
                                       _splash_runner, combined_csa2_mask)
    length = meta["arrays"]["q"]["shape"][1]
    _, kvlen, dim = meta["arrays"]["k"]["shape"]
    heads = meta["arrays"]["q"]["shape"][2]
    state = TPUNativeState(mesh, TPUNativeConfig(attention_data_shards=4,
        splash_batch_mode=mode, splash_block_q_dkv=tile, splash_interpret=interpret))
    mask = combined_csa2_mask(length, local_window=meta["local_window"],
        global_kv_len=kvlen - length, compression_ratio=meta["ratio"])
    runner = _splash_runner(state, seq_len=length, kv_len=kvlen,
        n_heads=heads, head_dim=dim, mask_array=mask, save_residuals=False)
    if independent_values:
        return lambda q, k, v, sinks, qseg, kvseg: runner(q, k, qseg, kvseg, sinks, value=v)
    # Match production aliasing and shared-KV gradient addition in primary timings.
    return lambda q, kv, sinks, qseg, kvseg: runner(q, kv, qseg, kvseg, sinks)


def operations(fn):
    import jax
    def prepare(*args):
        values, segments = args[:-2], args[-2:]
        return jax.vjp(lambda *xs: fn(*xs, *segments), *values)
    def combined(*args):
        y, backward = prepare(*args[:-1])
        cotangent = args[-1]
        return y, backward(cotangent)
    return jax.jit(fn), jax.jit(prepare), jax.jit(lambda backward, ct: backward(ct)), jax.jit(combined)


def benchmark(executable, args, warmup, steps):
    import jax
    for _ in range(warmup):
        jax.block_until_ready(executable(*args))
    samples = []
    for _ in range(steps):
        start = time.perf_counter()
        result = executable(*args)
        jax.block_until_ready(result)
        samples.append(time.perf_counter() - start)
    return {"seconds": samples, "median_seconds": float(np.median(samples)),
            "p95_seconds": float(np.percentile(samples, 95)), "warmup": warmup}


def run(a, report, checkpoint):
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from nano_dsv41f import make_v5e_mesh, validate_v5e_runtime
    from nano_dsv41f.tpu import axes_for_shard_count
    from nano_dsv41f.runtime import package_versions
    from nano_dsv41f.profiling import compiled_memory_report
    validate_v5e_runtime()
    meta = json.loads((a.bank / "manifest.json").read_text())["families"][a.family]
    bank = a.bank / f"{a.family}.npz"
    if hashlib.sha256(bank.read_bytes()).hexdigest() != meta["file_sha256"]:
        raise ValueError("Frozen bank checksum mismatch")
    mesh = make_v5e_mesh()
    cp, dp = axes_for_shard_count(mesh, 2), axes_for_shard_count(mesh, 4)
    names = ("q", "k", "sinks", "q_segments", "kv_segments", "cotangent")
    specs = (P(dp, cp, None, None), P(dp, None, None),
             P(), P(dp, cp), P(dp, None), P(dp, cp, None, None))
    values, hashes = [], {}
    with np.load(bank, allow_pickle=False) as data:
        for name, spec in zip(names, specs):
            host = select_rows(data[name], a.rows, a.composition, shared=name == "sinks")
            dtype = jnp.dtype(meta["arrays"][name]["dtype"])
            host = host.astype(dtype)
            hashes[name] = hashlib.sha256(host.tobytes()).hexdigest()
            values.append(jax.device_put(host, NamedSharding(mesh, spec)))
    jax.block_until_ready(values)
    inputs, ct = tuple(values[:-1]), values[-1]
    report.update(packages=package_versions(), input_hashes=hashes, metadata=meta,
                  devices=[str(d) for d in jax.devices()], variants={},
                  comparison="production tied K=V, Q/shared-KV/sink VJPs; independent K/V covered by oracle",
                  memory_note="compiler HBM estimates and device counters are not VMEM usage",
                  mechanism="VMEM/DMA unresolved pending compiler or usable hardware evidence")
    sequential_tiles = a.intervention == "sequential_tile"
    reference_variant = "sequential" if sequential_tiles else "vmap"
    tile = getattr(a, "tile", 256) if sequential_tiles else 256
    order = (["sequential", f"tile{tile}"] if sequential_tiles else
             ["vmap", "sequential"] if a.intervention == "schedule" else ["vmap", "tile256"])
    if a.repeat % 2:
        order.reverse()
    report["order"] = order
    # Build/check both combined executables before any timed samples.
    compiled, reference = {}, None
    outputs = {}
    for variant in order:
        checkpoint(f"compile_and_check:{variant}")
        mode = "sequential" if sequential_tiles or variant == "sequential" else "vmap"
        block = tile if variant.startswith("tile") else 128 if sequential_tiles else None
        fn = make_function(mesh, meta, mode, tile=block)
        fwd, prepare, back, combined = operations(fn)
        start = time.perf_counter()
        executable = combined.lower(*inputs, ct).compile()
        seconds = time.perf_counter() - start
        outputs[variant] = jax.device_get(jax.block_until_ready(executable(*inputs, ct)))
        compiled[variant] = (executable, fwd, prepare, back)
        result = {"compile_seconds": seconds, "memory": compiled_memory_report(executable),
                  "schedule": mode, "block_q_dkv": block or 128}
        report["variants"][variant] = result
        with gzip.open(a.output.parent / f"{a.output.stem}-{variant}-combined.hlo.txt.gz", "wt") as f:
            f.write(executable.as_text())
    reference = jax.tree.leaves(outputs[reference_variant])
    for variant in order:
        errors = {name: error_metrics(x, y) for name, x, y in zip(
            ("output", "dq", "dkv", "dsinks"), jax.tree.leaves(outputs[variant]), reference)}
        report["variants"][variant]["errors"] = errors
        if not all(e["passed"] for e in errors.values()):
            checkpoint("numerical_gate_failed")
            raise FloatingPointError(f"{variant} failed the preregistered 0.01 error gate")
    checkpoint("numerical_gate_passed")
    components = {}
    # Compile/check materialized pullbacks too: preflight exercises every path
    # that the timed run will use, with the captured mixed-precision dtypes.
    for variant in order:
        executable, fwd, prepare, back = compiled[variant]
        result = report["variants"][variant]
        checkpoint(f"compile_and_check_components:{variant}")
        # Materialized VJP residuals are runtime arguments to the backward executable.
        # This is backward-only, not a subtraction of two noisy timing measurements.
        forward_exe = fwd.lower(*inputs).compile()
        _, pullback = prepare(*inputs)
        jax.block_until_ready(pullback)
        backward_exe = back.lower(pullback, ct).compile()
        separate = (forward_exe(*inputs), backward_exe(pullback, ct))
        separate = jax.device_get(jax.block_until_ready(separate))
        errors = [error_metrics(x, y) for x, y in zip(jax.tree.leaves(separate),
                                                    jax.tree.leaves(outputs[variant]))]
        result["component_errors"] = errors
        if not all(e["passed"] for e in errors):
            raise FloatingPointError(f"{variant} separate forward/backward failed numerical gate")
        components[variant] = (forward_exe, backward_exe, pullback)
    del outputs, reference, separate
    if getattr(a, "preflight_only", False):
        report["status"] = "passed"
        checkpoint("preflight_complete")
        return
    for variant in order:
        executable = compiled[variant][0]
        forward_exe, backward_exe, pullback = components[variant]
        result = report["variants"][variant]
        checkpoint(f"timing:{variant}:combined")
        result["combined"] = benchmark(executable, (*inputs, ct), a.warmup, a.steps)
        result["combined"]["microseconds_per_token"] = result["combined"]["median_seconds"] * 1e6 / (a.rows * inputs[0].shape[1])
        checkpoint(f"timing:{variant}:forward_backward_separate")
        result["forward"] = benchmark(forward_exe, inputs, a.warmup, a.steps)
        result["backward"] = benchmark(backward_exe, (pullback, ct), a.warmup, a.steps)
        result["device_memory"] = device_memory(jax)
        del forward_exe, backward_exe, pullback
        checkpoint(f"complete:{variant}")
    # Finish every unprofiled comparison before enabling the profiler.
    for variant in order:
        executable = compiled[variant][0]
        result = report["variants"][variant]
        if a.trace_steps:
            checkpoint(f"trace:{variant}")
            trace = a.output.parent / "traces" / a.output.stem / variant
            with jax.profiler.trace(str(trace), create_perfetto_link=False):
                for step in range(a.trace_steps):
                    with jax.profiler.StepTraceAnnotation("attention_forward_vjp", step_num=step):
                        jax.block_until_ready(executable(*inputs, ct))
            files = [str(p) for p in trace.rglob("*.xplane.pb")]
            if not files:
                raise RuntimeError("Profiler exported no XPlane")
            result["trace_files"] = files
    report["status"] = "passed"
    checkpoint("complete")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--family", choices=("local", "compressed", "global"), required=True)
    p.add_argument("--rows", type=int, choices=(4, 8, 24), required=True)
    p.add_argument("--composition", choices=("repeated", "distinct"), default="repeated")
    p.add_argument("--intervention", choices=("schedule", "tile", "sequential_tile"), default="schedule")
    p.add_argument("--tile", type=int, choices=(256, 512, 1024, 2048), default=256,
                   help="Candidate block_q_dkv for sequential_tile; baseline is 128")
    p.add_argument("--repeat", type=int, default=0)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--trace-steps", type=int, default=3)
    p.add_argument("--preflight-only", action="store_true",
                   help="Compile and check all paths with actual dtypes, without timing/tracing")
    a = p.parse_args()
    if a.warmup < 3 or a.steps < 12 or a.trace_steps < 0:
        p.error("require >=3 warmups, >=12 samples and nonnegative trace steps")
    report = {"status": "running", "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    def checkpoint(stage):
        report["stage"] = stage
        save_report(a.output, report)
        print(stage, flush=True)
    checkpoint("start")
    try:
        run(a, report, checkpoint)
    except Exception as exc:
        report.update(status="failed", exception=str(exc), traceback=traceback.format_exc())
        checkpoint(report["stage"])
        raise


if __name__ == "__main__":
    main()
