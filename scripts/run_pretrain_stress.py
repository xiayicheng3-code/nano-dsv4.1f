"""Full-model TPU stress worker. Launch each case in a fresh Python process."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import subprocess
import time
import traceback

import numpy as np


def save_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def make_batch(config, *, batch_rows, seq_len=8192, layout="long", seed=7, data=None):
    """Long rows stress global history; odd packed segments exercise alignment/masking."""
    from nano_dsv41f import pack_token_sequences
    if data is not None:
        with np.load(data, allow_pickle=False) as shard:
            ids = np.asarray(shard["input_ids"][:batch_rows], dtype=np.int32)
            seg = np.asarray(shard["segment_ids"][:batch_rows], dtype=np.int32)
            mask = np.asarray(shard["token_mask"][:batch_rows], dtype=bool)
    else:
        rng = np.random.default_rng(seed)
        lengths = (seq_len,) if layout == "long" else (641, 1407, 2049, 4091)
        rows = [pack_token_sequences(
            [rng.integers(23, config.vocab_size, length, dtype=np.int32) for length in lengths],
            seq_len=seq_len, pad_token_id=config.engram.pad_token_id,
            compression_ratio=config.csa2.context_compression_ratio,
        ) for _ in range(batch_rows)]
        ids, seg, mask = (np.concatenate([getattr(row, key) for row in rows], axis=0)
                          for key in ("input_ids", "segment_ids", "token_mask"))
    shape = (batch_rows, seq_len)
    if ids.shape != shape or seg.shape != shape or mask.shape != shape:
        raise ValueError(f"input_ids/segment_ids/token_mask must each have shape {shape}")
    if ids.min() < 0 or ids.max() >= config.vocab_size:
        raise ValueError("token IDs exceed the recipe vocabulary")
    if not np.array_equal(seg[:, 0::2], seg[:, 1::2]):
        raise ValueError("segments must align to r=2 compression groups")
    lm_tokens = int((mask[:, 1:] & mask[:, :-1] & (seg[:, 1:] == seg[:, :-1])).sum())
    if lm_tokens == 0:
        raise ValueError("batch has no valid next-token targets")
    digest = hashlib.sha256()
    for array in (ids, seg, mask):
        digest.update(array.tobytes())
    return (ids, seg, mask), {"lm_tokens": lm_tokens, "real_tokens": int(mask.sum()),
                             "physical_tokens": int(ids.size), "sha256": digest.hexdigest()}


def device_memory(jax):
    out = []
    for device in jax.local_devices():
        try:
            raw = device.memory_stats()
            stats = None if raw is None else {str(k): int(v) for k, v in raw.items()
                                              if isinstance(v, (int, np.integer))}
            out.append({"device": str(device), "stats": stats})
        except Exception as exc:
            out.append({"device": str(device), "unavailable": str(exc)})
    return out


def run(args, report, checkpoint):
    import jax
    import jax.numpy as jnp
    from nano_dsv41f import (compile_diagnostics, compile_pretrain_step,
        init_model_sharded_mixed_precision, init_optimizer_state_sharded,
        make_v5e_mesh, put_training_batch, runtime_report, semantic_axes, validate_v5e_runtime)
    from nano_dsv41f.pretrain_recipe import pretrain_recipe, recipe_manifest
    from nano_dsv41f.runtime import package_versions
    config, train, native = pretrain_recipe(
        profile=args.profile, cp=args.cp, dp=args.dp,
        experts=args.experts, width=args.width, top_k=args.top_k)
    if args.batch_rows <= 0 or args.batch_rows % args.dp:
        raise ValueError("batch rows must be positive and divisible by DP")
    if args.steps < 1 or args.warmup < 1:
        raise ValueError("at least one warmup and one measured step are required")
    if args.late_step + args.warmup + args.steps > int(train.total_steps * config.indexer_training.end_fraction):
        raise ValueError("late stress steps must remain inside the indexer training phase")
    if args.late_step < math.ceil(train.total_steps * config.indexer_training.start_fraction):
        raise ValueError("late step must be inside the indexer training phase")
    host_batch, batch_info = make_batch(config, batch_rows=args.batch_rows, layout=args.layout,
                                      seed=args.seed, data=args.data)
    report.update(recipe=recipe_manifest(config, train, native), batch=batch_info,
                  packages=package_versions(), phase_results={},
                  model_note="Seven backbone layers; DSpark allocated/frozen, no draft objective. QAT off.",
                  benchmark_note="Synthetic/random-init unless --data supplied; not a learning-quality test.")
    checkpoint("runtime")
    validate_v5e_runtime()
    report["runtime"] = runtime_report()
    mesh = make_v5e_mesh()
    report["semantic_axes"] = semantic_axes(config, mesh)
    ids, segments, mask = put_training_batch(*host_batch, config, mesh)
    report["input_sharding"] = str(ids.sharding)
    report["local_input_shape"] = list(ids.addressable_shards[0].data.shape)
    checkpoint("initialize_parameters")
    params, specs, _ = init_model_sharded_mixed_precision(
        jax.random.PRNGKey(args.seed), config, mesh, payload_dtype=jnp.bfloat16)
    jax.block_until_ready(params)
    report["parameter_count"] = sum(int(x.size) for x in jax.tree.leaves(params))
    checkpoint("initialize_optimizer")
    opt, _ = init_optimizer_state_sharded(params, specs, config, mesh)
    jax.block_until_ready(opt)
    report["memory_after_init"] = device_memory(jax)
    if args.routing == "skewed":
        # Maximum dispatch imbalance: every token chooses the final K experts.
        # This changes only the initial selection bias, not architecture or kernels.
        params = {**params, "blocks": tuple(
            {**b, "moe": {**b["moe"], "router_bias": jax.device_put(
                np.arange(config.n_experts, dtype=np.float32) * 1000,
                b["moe"]["router_bias"].sharding)}} for b in params["blocks"])}
        jax.block_until_ready(params)
    # Keep both executables alive: late-transition HBM includes the base executable.
    executables = []
    phases = ("base", "late") if args.phase == "both" else (args.phase,)
    for phase in phases:
        include_indexer = phase == "late"
        first_step = args.late_step if include_indexer else train.warmup_steps
        result = {"steps": []}
        report["phase_results"][phase] = result
        checkpoint(f"{phase}:compile")
        step = compile_pretrain_step(params, opt, specs, config, train, mesh,
            include_indexer=include_indexer, n_segments=None, native_config=native)
        start = time.perf_counter()
        executable, diagnostics = compile_diagnostics(
            step, params, opt, ids, segments, jnp.asarray(first_step, jnp.int32), mask)
        result.update(compile_seconds=time.perf_counter() - start, diagnostics=diagnostics)
        executables.append(executable)
        result["memory_after_compile"] = device_memory(jax)
        checkpoint(f"{phase}:compiled")
        for i in range(args.warmup + args.steps):
            checkpoint(f"{phase}:step:{i}")
            step_number = jnp.asarray(first_step + i, jnp.int32)
            jax.block_until_ready(step_number)
            start = time.perf_counter()
            params, opt, metrics = executable(params, opt, ids, segments, step_number, mask)
            jax.block_until_ready((params, opt, metrics))
            seconds = time.perf_counter() - start
            m = jax.device_get(metrics)
            losses = {key: float(m[key]) for key in ("loss", "lm_loss", "indexer_loss")}
            if not all(math.isfinite(v) for v in losses.values()):
                raise FloatingPointError(f"nonfinite losses: {losses}")
            if int(np.asarray(m["expert_dropped"]).sum()) != 0:
                raise AssertionError("dropless MoE dropped assignments")
            if int(m["moe_mosaic_layers"]) != config.n_layers:
                raise AssertionError("not every backbone layer used Mosaic MoE")
            if int(m["lm_tokens"]) != batch_info["lm_tokens"]:
                raise AssertionError("LM masking disagrees with packed targets")
            if include_indexer and int(m["indexer"]["active_queries"]) <= 0:
                raise AssertionError("late test did not exercise eligible indexer queries")
            row = {"step": first_step + i, "warmup": i < args.warmup,
                   "seconds": seconds, **losses}
            result["steps"].append(row)
            if i == 0:
                loads = np.asarray(m["expert_loads"])
                result["routing"] = {"loads": loads.tolist(),
                    "max_over_mean": float(loads.max() / max(float(loads.mean()), 1)),
                    "packed_rows_per_chip": int(m["expert_packed_rows"]),
                    "experts_per_chip": int(m["experts_per_chip"])}
                if include_indexer:
                    result["query_counts"] = {k: int(v) for k, v in m["indexer"]["query_counts"].items()}
            print(phase, json.dumps(row), flush=True)
        times = [row["seconds"] for row in result["steps"] if not row["warmup"]]
        median = float(np.median(times))
        result.update(median_seconds=median, p95_seconds=float(np.percentile(times, 95)),
                      lm_tokens_per_second=batch_info["lm_tokens"] / median,
                      physical_tokens_per_second=batch_info["physical_tokens"] / median,
                      memory_after_steps=device_memory(jax))
        estimate = (diagnostics.get("memory") or {}).get("estimated_total_gib")
        result["compiler_headroom_gib_vs_16GB"] = None if estimate is None else 16e9 / 2**30 - estimate
        checkpoint(f"{phase}:complete")
    report["status"] = "passed"
    checkpoint("complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("baseline", "narrow24", "narrow48"), default="baseline")
    parser.add_argument("--experts", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--cp", type=int, default=8)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--batch-rows", type=int, default=4)
    parser.add_argument("--phase", choices=("both", "base", "late"), default="both")
    parser.add_argument("--layout", choices=("long", "packed"), default="long")
    parser.add_argument("--routing", choices=("normal", "skewed"), default="normal")
    parser.add_argument("--data", type=Path, help="NPZ with input_ids, segment_ids, token_mask [B,8192]")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--late-step", type=int, default=6000)
    args = parser.parse_args()
    report = {"status": "running", "arguments": {k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()}, "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()}
    def checkpoint(stage):
        report.update(stage=stage, host_max_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
        save_report(args.output, report)
        print("stage:", stage, flush=True)
    checkpoint("start")
    try:
        run(args, report, checkpoint)
    except Exception as exc:
        report.update(status="failed", exception_type=type(exc).__name__,
                      exception=str(exc), traceback=traceback.format_exc())
        checkpoint(report["stage"])
        raise


if __name__ == "__main__":
    main()
