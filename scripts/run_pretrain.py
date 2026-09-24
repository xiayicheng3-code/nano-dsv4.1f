#!/usr/bin/env python3
"""Train the 2.4B-token base allocation on TPU v5e-8; never starts mid-training."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback

import numpy as np

from pretrain_checkpoint import atomic_json, load_checkpoint, read_metadata, save_checkpoint
from pretrain_input import Prefetch, batch_counts, inspect_corpus, iter_pretrain_batches

ROOT = Path(__file__).resolve().parents[1]


def implementation_digest():
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src/nano_dsv41f").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def production_recipe(manifest, total_tokens):
    from nano_dsv41f.pretrain_recipe import pretrain_recipe
    # Freeze a step horizon using the corpus's actual non-padding density. The run
    # stop uses counted non-padding tokens; the LR/indexer horizon stays fixed on resume.
    counts = manifest["splits"]["train"]
    average_tokens_per_step = 4 * counts["real_tokens"] / counts["rows"]
    total_steps = math.ceil(total_tokens / average_tokens_per_step)
    config, train, native = pretrain_recipe(
        profile="narrow48", cp=2, dp=4, benchmark_total_steps=total_steps)
    native = replace(native, moe_buffer_divisor=4, splash_batch_mode="sequential",
                     splash_block_q_dkv=128, splash_compressed_block_q_dkv=1024,
                     splash_global_block_q_dkv=1024)
    return config, train, native


def make_evaluator(specs, config, mesh, native):
    import jax
    from nano_dsv41f.tpu import batch_named_sharding, named_shardings
    from nano_dsv41f.tpu_native import NativeCompiledStep, install_model_dispatch
    from nano_dsv41f.training import pretrain_loss
    install_model_dispatch()
    def evaluate(p, ids, seg, mask):
        _, metrics = pretrain_loss(p, config, ids, segment_ids=seg, token_mask=mask,
                                  include_indexer=False)
        return metrics["lm_loss"], metrics["lm_tokens"]
    batch_sharding = batch_named_sharding(config, mesh)
    fn = jax.jit(evaluate, in_shardings=(named_shardings(specs, mesh),
                                       batch_sharding, batch_sharding, batch_sharding))
    return NativeCompiledStep(fn, mesh, replace(native, need_teacher_lse=False))


def run(args):
    import jax
    import jax.numpy as jnp
    from nano_dsv41f import (compile_diagnostics, compile_pretrain_step,
        init_model_sharded_mixed_precision, init_optimizer_state_sharded,
        make_v5e_mesh, put_training_batch, runtime_report, validate_v5e_runtime)
    from nano_dsv41f.pretrain_recipe import recipe_manifest
    from nano_dsv41f.runtime import package_versions
    from nano_dsv41f.training import indexer_phase_enabled

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use a new output directory; attach the old checkpoint via --resume")
    report = {"status": "initializing", "stage": "pretrain", "output": str(output),
              "started_unix": time.time(), "deadline_unix": args.deadline_unix,
              "target_nonpadding_tokens": args.base_tokens,
              "total_pretrain_midtrain_tokens": args.total_tokens}
    atomic_json(output / "summary.json", report)
    stopped = {"signal": None}
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda number, frame: stopped.update(signal=number))
    input_stream = None
    log = (output / "metrics.jsonl").open("a", buffering=1)
    try:
        print("Checking corpus files and tokenizer before initializing TPU", flush=True)
        corpus, manifest, corpus_identity = inspect_corpus(args.corpus, args.tokenizer)
        if manifest["splits"]["train"]["real_tokens"] < args.base_tokens + 4 * 8192:
            raise ValueError("Corpus needs base target plus one batch of non-padding token headroom")
        config, train, native = production_recipe(manifest, args.total_tokens)
        recipe = recipe_manifest(config, train, native)
        identity = {"recipe_sha256": recipe["sha256"], "corpus": corpus_identity,
                    "data_seed": args.data_seed, "init_seed": args.init_seed,
                    "base_tokens": args.base_tokens, "total_tokens": args.total_tokens,
                    "implementation_sha256": implementation_digest(), "batch_rows": 4,
                    "data_order_version": 1}
        state = {"completed_steps": 0, "consumed_batches": 0, "real_tokens": 0,
                 "physical_tokens": 0, "lm_tokens": 0, "identity": identity,
                 "updates_with_fallback": 0, "fallback_chip_layers_total": 0,
                 "max_chip_load_seen": 0,
                 "stage": "pretrain", "rng": {"init_seed": args.init_seed,
                 "data_seed": args.data_seed, "query_seed": config.indexer_training.query_seed,
                 "note": "No dropout RNG; indexer sampling derives from query seed and global step."}}
        if args.resume:
            _, old = read_metadata(args.resume)
            if old["metadata"]["identity"] != identity:
                raise ValueError("Resume identity differs: recipe/corpus/budget/seeds/code must match")
        report.update(recipe=recipe, identity=identity, corpus=str(corpus),
                      planned_total_steps=train.total_steps, packages=package_versions(),
                      source_commit=subprocess.check_output(
                          ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                      compile=[], evaluations=[])
        shutil.copy2(corpus / "manifest.json", output / "corpus_manifest.json")
        tokenizer_path = Path(args.tokenizer)
        if tokenizer_path.is_dir():
            tokenizer_path = next(tokenizer_path.rglob("tokenizer.json"))
        shutil.copy2(tokenizer_path, output / "tokenizer.json")
        atomic_json(output / "recipe.json", recipe)
        atomic_json(output / "launch.json", {k: str(v) if isinstance(v, Path) else v
                                              for k, v in vars(args).items()})
        validate_v5e_runtime()
        report["runtime"] = runtime_report()
        mesh = make_v5e_mesh()
        print("Initializing sharded BF16 parameters and FP32 optimizer state", flush=True)
        params, specs, _ = init_model_sharded_mixed_precision(
            jax.random.PRNGKey(args.init_seed), config, mesh, payload_dtype=jnp.bfloat16)
        opt, _ = init_optimizer_state_sharded(params, specs, config, mesh)
        jax.block_until_ready((params, opt))
        if args.resume:
            (params, opt), state = load_checkpoint(args.resume, (params, opt), expected_identity=identity)
            print(f"Resumed step {state['completed_steps']}, {state['real_tokens']:,} tokens", flush=True)
        report["parameter_count"] = sum(x.size for x in jax.tree.leaves(params))
        report["resume_from"] = str(args.resume) if args.resume else None
        checkpoint_root = output / "checkpoints"
        last_saved = -1

        def checkpoint(reason):
            nonlocal last_saved
            if last_saved == state["completed_steps"]:
                return
            start = time.monotonic()
            metadata = {**state, "recipe": recipe, "source_commit": report["source_commit"],
                        "reason": reason}
            path = save_checkpoint(checkpoint_root, (params, opt), metadata)
            last_saved = state["completed_steps"]
            report.update(checkpoint=str(path), checkpoint_seconds=time.monotonic() - start,
                          progress={k: state[k] for k in ("completed_steps", "real_tokens",
                                                        "lm_tokens", "physical_tokens", "consumed_batches")})
            atomic_json(output / "summary.json", report)
            print(f"Checkpoint committed: {path.name} ({report['checkpoint_seconds']:.1f}s)", flush=True)

        def remaining():
            return args.deadline_unix - time.time()

        checkpoint("session_start")
        evaluator = make_evaluator(specs, config, mesh, native)
        eval_compiled = False
        def evaluate():
            nonlocal eval_compiled
            if remaining() < (300 if eval_compiled else 900) or stopped["signal"]:
                return
            checkpoint("before_validation")
            start = time.monotonic()
            total_loss = 0.0
            total_targets = 0
            count = 0
            iterator = iter_pretrain_batches(corpus, split="validation", batch_rows=4,
                                             seed=args.data_seed + 1)
            for batch in iterator:
                if count >= args.eval_batches or remaining() < 120 or stopped["signal"]:
                    break
                batch_counts(batch)
                ids, seg, mask = put_training_batch(*[batch[k] for k in
                    ("input_ids", "segment_ids", "token_mask")], config, mesh)
                loss, targets = jax.device_get(evaluator(params, ids, seg, mask))
                eval_compiled = True
                if not math.isfinite(float(loss)):
                    raise FloatingPointError("nonfinite validation loss")
                total_loss += float(loss) * int(targets)
                total_targets += int(targets)
                count += 1
            if total_targets:
                loss = total_loss / total_targets
                row = {"event": "validation", "step": state["completed_steps"],
                       "real_tokens": state["real_tokens"], "batches": count,
                       "lm_targets": total_targets, "lm_loss": loss,
                       "perplexity": math.exp(min(loss, 80)), "seconds": time.monotonic() - start}
                report["evaluations"].append(row)
                log.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                atomic_json(output / "summary.json", report)

        evaluate()
        executables = {}
        input_stream = Prefetch(iter_pretrain_batches(
            corpus, batch_rows=4, seed=args.data_seed, start_batch=state["consumed_batches"]))
        report["status"] = "training"
        start_step = state["completed_steps"]
        train_seconds = 0.0
        train_tokens = 0
        recent_seconds = recent_tokens = 0
        stop_reason = "base_budget_complete"
        while state["real_tokens"] < args.base_tokens:
            if stopped["signal"] or remaining() <= 120:
                stop_reason = "signal" if stopped["signal"] else "wall_time"
                break
            if args.max_steps and state["completed_steps"] - start_step >= args.max_steps:
                stop_reason = "step_limit"
                break
            include_indexer = indexer_phase_enabled(state["completed_steps"], train, config)
            if include_indexer not in executables and remaining() < 900:
                stop_reason = "wall_time_before_compile"
                break
            start = time.monotonic()
            try:
                batch = next(input_stream)
            except StopIteration:
                stop_reason = "corpus_exhausted"
                break
            counts = batch_counts(batch)
            ids, seg, mask = put_training_batch(*[batch[k] for k in
                ("input_ids", "segment_ids", "token_mask")], config, mesh)
            step_number = jnp.asarray(state["completed_steps"], dtype=jnp.int32)
            if include_indexer not in executables:
                checkpoint("before_compile")
                print(f"Compiling {'indexer' if include_indexer else 'LM'} update at step {state['completed_steps']}", flush=True)
                compile_start = time.monotonic()
                fn = compile_pretrain_step(params, opt, specs, config, train, mesh,
                    include_indexer=include_indexer, n_segments=None, native_config=native)
                executable, diagnostics = compile_diagnostics(fn, params, opt, ids, seg, step_number, mask)
                executables[include_indexer] = executable
                report["compile"].append({"indexer": include_indexer,
                    "seconds": time.monotonic() - compile_start, "diagnostics": diagnostics})
                atomic_json(output / "summary.json", report)
                # Compilation can exceed the remaining budget. This batch has not
                # been consumed and will be replayed from the committed cursor.
                if remaining() <= 120 or stopped["signal"]:
                    stop_reason = "wall_time_after_compile"
                    break
                start = time.monotonic()
            params, opt, metrics = executables[include_indexer](params, opt, ids, seg, step_number, mask)
            # Same synchronization as the validated worker; transfer compact metrics only.
            selected = {k: metrics[k] for k in ("loss", "lm_loss", "indexer_loss",
                         "learning_rate", "lm_tokens", "expert_dropped", "expert_loads_by_layer",
                         "experts_per_chip", "expert_packed_rows", "moe_mosaic_layers")}
            selected["active_queries"] = metrics["indexer"]["active_queries"]
            m = jax.device_get(selected)
            jax.block_until_ready((params, opt))
            if not all(math.isfinite(float(m[k])) for k in ("loss", "lm_loss", "indexer_loss")):
                raise FloatingPointError("nonfinite training loss; retaining previous checkpoint")
            if int(np.asarray(m["expert_dropped"]).sum()) != 0:
                raise RuntimeError("MoE dropped assignments")
            if int(m["lm_tokens"]) != counts["lm_tokens"] or int(m["moe_mosaic_layers"]) != config.n_layers:
                raise RuntimeError("LM masking/native MoE validation failed")
            seconds = time.monotonic() - start
            state["completed_steps"] += 1
            state["consumed_batches"] += 1
            for key, count in counts.items():
                state[key] += count
            train_seconds += seconds
            train_tokens += counts["real_tokens"]
            recent_seconds += seconds
            recent_tokens += counts["real_tokens"]
            loads = np.asarray(m["expert_loads_by_layer"])
            chips = loads.reshape(config.n_layers, -1, int(m["experts_per_chip"])).sum(-1)
            fallback = int((chips > int(m["expert_packed_rows"])).sum())
            state["updates_with_fallback"] += int(fallback > 0)
            state["fallback_chip_layers_total"] += fallback
            state["max_chip_load_seen"] = max(state["max_chip_load_seen"], int(chips.max()))
            if state["completed_steps"] % args.log_every == 0 or state["completed_steps"] == start_step + 1:
                row = {"event": "train", "step": state["completed_steps"],
                       "real_tokens": state["real_tokens"], "physical_tokens": state["physical_tokens"],
                       "indexer_enabled": include_indexer, "active_queries": int(m["active_queries"]),
                       "tokens_per_second": recent_tokens / recent_seconds,
                       "step_seconds": seconds, "remaining_wall_seconds": remaining(),
                       "fallback_chip_layers": fallback, "max_chip_load": int(chips.max()),
                       "updates_with_fallback": state["updates_with_fallback"],
                       "max_chip_load_seen": state["max_chip_load_seen"],
                       **{k: float(m[k]) for k in ("loss", "lm_loss", "indexer_loss", "learning_rate")}}
                log.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                recent_seconds = recent_tokens = 0
            if state["completed_steps"] % args.checkpoint_every == 0 or state["completed_steps"] == start_step + 10:
                checkpoint("periodic")
            if state["completed_steps"] % args.eval_every == 0:
                evaluate()
        checkpoint(stop_reason)
        if stop_reason == "base_budget_complete":
            evaluate()
        report.update(status="completed" if state["real_tokens"] >= args.base_tokens else "paused",
                      stop_reason=stop_reason, finished_unix=time.time(),
                      training_seconds_this_session=train_seconds,
                      training_tokens_per_second=train_tokens / train_seconds if train_seconds else None,
                      remaining_base_tokens=max(0, args.base_tokens - state["real_tokens"]),
                      reserved_midtrain_tokens=args.total_tokens - args.base_tokens)
        atomic_json(output / "summary.json", report)
        print(json.dumps({k: report[k] for k in ("status", "stop_reason", "checkpoint",
              "remaining_base_tokens", "reserved_midtrain_tokens")}, indent=2), flush=True)
        if stop_reason == "corpus_exhausted":
            raise RuntimeError("Corpus exhausted before base budget; no implicit repetition")
    except BaseException:
        report.update(status="failed", error=traceback.format_exc(), finished_unix=time.time())
        atomic_json(output / "summary.json", report)
        raise
    finally:
        if input_stream is not None:
            input_stream.close()
        log.close()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--base-tokens", type=int, default=2_400_000_000)
    p.add_argument("--total-tokens", type=int, default=3_000_000_000)
    p.add_argument("--deadline-unix", type=float, default=None)
    p.add_argument("--data-seed", type=int, default=1701)
    p.add_argument("--init-seed", type=int, default=7)
    p.add_argument("--checkpoint-every", type=int, default=2000)
    p.add_argument("--eval-every", type=int, default=10000)
    p.add_argument("--eval-batches", type=int, default=32)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=0,
                   help="Optional debug limit on additional updates; zero disables")
    args = p.parse_args()
    if args.deadline_unix is None:
        args.deadline_unix = time.time() + 8 * 3600
    if (not 0 < args.base_tokens < args.total_tokens or args.max_steps < 0 or
            any(getattr(args, k) <= 0 for k in ("checkpoint_every", "eval_every", "eval_batches", "log_every"))):
        p.error("Require 0 < base_tokens < total_tokens and positive intervals")
    return args


if __name__ == "__main__":
    run(parse_args())
