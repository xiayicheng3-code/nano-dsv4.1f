"""Production control-plane tests with real JAX CPU arrays and tiny update kernels."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import pretrain_checkpoint as ckpt
import prepare_pretrain_corpus as data
import run_pretrain as runner
from nano_dsv41f.optimizer import OptimizerLeafState


def tree_fixture():
    return ({"weight": jnp.asarray([[1.125, -2.5], [0.00012, 33]], jnp.bfloat16),
             "router_bias": jnp.asarray([0.01, -0.01], jnp.float32)},
            {"weight": OptimizerLeafState(jnp.ones((2, 2), jnp.float32), jnp.empty((0,), jnp.float32))})


def assert_tree_equal(a, b):
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        assert x.dtype == y.dtype
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_checkpoint_roundtrip_corruption_and_atomic_failure(tmp_path):
    tree = tree_fixture()
    identity = {"schedule": 123, "seed": 1701}
    state = {"completed_steps": 17, "consumed_batches": 17, "identity": identity}
    path = ckpt.save_checkpoint(tmp_path, tree, state)
    restored, meta = ckpt.load_checkpoint(tmp_path, tree_fixture(), expected_identity=identity)
    assert meta == state
    assert_tree_equal(tree, restored)
    with pytest.raises(ValueError, match="differ"):
        ckpt.load_checkpoint(tmp_path, tree_fixture(), expected_identity={"seed": 0})
    before = (tmp_path / "latest.json").read_bytes()
    with pytest.raises(FloatingPointError):
        ckpt.save_checkpoint(tmp_path, {"bad": jnp.asarray(float("nan"))}, state | {"completed_steps": 18})
    assert (tmp_path / "latest.json").read_bytes() == before
    assert not list(tmp_path.glob("*.pending"))
    leaf = next(path.glob("leaf-*.npy"))
    with leaf.open("r+b") as f:
        f.seek(-1, 2)
        f.write(b"x")
    with pytest.raises(ValueError, match="checksum"):
        ckpt.load_checkpoint(path, tree_fixture())


def test_checkpoint_retention(tmp_path):
    for step in range(4):
        ckpt.save_checkpoint(tmp_path, tree_fixture(), {"completed_steps": step})
    assert len(list(tmp_path.glob("step-*"))) == 2
    _, manifest = ckpt.read_metadata(tmp_path)
    assert manifest["metadata"]["completed_steps"] == 3


def compact_fixture(root, *, seq_len=8192, rows=40, shard_rows=7):
    directory = root / "parts/part-00000"
    splits = {}
    for split, n in (("train", rows), ("validation", 8)):
        writer = data.CompactWriter(directory / split, seq_len=seq_len, shard_rows=shard_rows)
        for i in range(n):
            # Different batches affect the simulated gradient, exposing cursor mistakes.
            row = np.full(seq_len, 3 + i % 23, dtype=np.uint16)
            row[0], row[-1] = data.BOS, data.EOS
            writer.add(row)
        splits[split] = writer.finish()
    data.write_json(directory / "manifest.json", {"splits": splits})
    manifest = {"format": data.FORMAT, "complete": True, "parts": ["parts/part-00000"],
                "config": {"seq_len": seq_len, "alignment": 2},
                "splits": {s: v["counts"] for s, v in splits.items()}}
    data.write_json(root / "manifest.json", manifest)
    return manifest


def test_resume_order_across_shards_without_reading_skipped_arrays(tmp_path, monkeypatch):
    compact_fixture(tmp_path, seq_len=16, rows=45, shard_rows=7)
    complete = list(data.iter_pretrain_batches(tmp_path, batch_rows=4, seed=9, drop_last=False))
    calls = []
    original = data.np.load
    def load(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(data.np, "load", load)
    resumed = list(data.iter_pretrain_batches(tmp_path, batch_rows=4, seed=9,
                                              start_batch=9, drop_last=False))
    for a, b in zip(complete[9:], resumed):
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
    assert len(resumed) == len(complete) - 9
    assert len(calls) <= 9  # opens only the final few shards, not all seven


def test_full_driver_resume_matches_uninterrupted_and_crosses_indexer_phase(tmp_path, monkeypatch):
    import nano_dsv41f as package
    import nano_dsv41f.runtime as runtime
    from nano_dsv41f.pretrain_recipe import pretrain_recipe
    corpus = tmp_path / "corpus"
    manifest = compact_fixture(corpus)
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}")
    monkeypatch.setattr(runner, "inspect_corpus", lambda *a, **k: (corpus, manifest, {"fixture": "v1"}))
    monkeypatch.setattr(package, "validate_v5e_runtime", lambda: None)
    monkeypatch.setattr(package, "runtime_report", lambda: {"platform": "cpu-test"})
    monkeypatch.setattr(runtime, "package_versions", lambda: {})
    monkeypatch.setattr(package, "make_v5e_mesh", lambda: None)
    monkeypatch.setattr(package, "init_model_sharded_mixed_precision",
                        lambda *a, **k: ({"w": jnp.asarray([1., 2.], jnp.bfloat16),
                                         "router_bias": jnp.zeros(2)}, None, None))
    monkeypatch.setattr(package, "init_optimizer_state_sharded",
                        lambda *a, **k: ({"momentum": jnp.zeros(2)}, None))
    monkeypatch.setattr(package, "put_training_batch", lambda ids, seg, mask, *a: tuple(map(jnp.asarray, (ids, seg, mask))))
    cfg, train, native = pretrain_recipe(profile="narrow48", cp=2, dp=4, benchmark_total_steps=10)
    monkeypatch.setattr(runner, "production_recipe", lambda *a: (cfg, train, native))
    phases = []
    def compile_step(*a, include_indexer, **kw):
        phases.append(include_indexer)
        @jax.jit
        def update(p, opt, ids, seg, step, mask):
            grad = ids[:, 1].mean() * .01 + step * .001 + (0.002 if include_indexer else 0)
            momentum = opt["momentum"] * .9 + grad
            new = {"w": (p["w"].astype(jnp.float32) - .01 * momentum).astype(jnp.bfloat16),
                   "router_bias": p["router_bias"] + .001}
            metrics = {"loss": grad, "lm_loss": grad, "indexer_loss": jnp.asarray(0.),
                       "learning_rate": jnp.asarray(.01), "lm_tokens": jnp.asarray(4 * 8191),
                       "expert_dropped": jnp.asarray(0), "expert_loads_by_layer": jnp.ones((7, 48), jnp.int32),
                       "experts_per_chip": jnp.asarray(6), "expert_packed_rows": jnp.asarray(32768),
                       "moe_mosaic_layers": jnp.asarray(7),
                       "indexer": {"active_queries": jnp.asarray(128 if include_indexer else 0)}}
            return new, {"momentum": momentum}, metrics
        return update
    monkeypatch.setattr(package, "compile_pretrain_step", compile_step)
    monkeypatch.setattr(package, "compile_diagnostics", lambda fn, *a: (fn, {"cpu_test": True}))
    monkeypatch.setattr(runner, "make_evaluator", lambda *a: jax.jit(
        lambda p, ids, seg, mask: (p["w"].astype(jnp.float32).mean(), jnp.asarray(4 * 8191))))
    # Avoid replacing pytest's signal handlers during these three runs.
    monkeypatch.setattr(runner.signal, "signal", lambda *a: None)
    def args(name, max_steps=0, resume=None):
        return SimpleNamespace(output=tmp_path / name, corpus=corpus, tokenizer=tokenizer,
            base_tokens=9 * 32768 - 1, total_tokens=12 * 32768, deadline_unix=runner.time.time() + 3600,
            data_seed=1701, init_seed=7, resume=resume, max_steps=max_steps,
            checkpoint_every=2, eval_every=3, eval_batches=2, log_every=1)
    runner.run(args("full"))
    runner.run(args("first", max_steps=7))
    first = json.loads((tmp_path / "first/summary.json").read_text())
    assert first["status"] == "paused" and first["progress"]["completed_steps"] == 7
    runner.run(args("resume", resume=tmp_path / "first/checkpoints"))
    template = ({"w": jnp.zeros(2, jnp.bfloat16), "router_bias": jnp.zeros(2)},
                {"momentum": jnp.zeros(2)})
    full, full_meta = ckpt.load_checkpoint(tmp_path / "full/checkpoints", template)
    resumed, resume_meta = ckpt.load_checkpoint(tmp_path / "resume/checkpoints", template)
    assert_tree_equal(full, resumed)
    for key in ("completed_steps", "consumed_batches", "real_tokens", "physical_tokens", "lm_tokens"):
        assert full_meta[key] == resume_meta[key]
    assert full_meta["completed_steps"] == 9
    assert True in phases and False in phases
    assert json.loads((tmp_path / "resume/summary.json").read_text())["status"] == "completed"
    # A nearly exhausted session checkpoints without consuming a prefetched batch.
    deadline_args = args("deadline")
    deadline_args.deadline_unix = runner.time.time() + 119
    runner.run(deadline_args)
    deadline = json.loads((tmp_path / "deadline/summary.json").read_text())
    assert deadline["status"] == "paused" and deadline["stop_reason"] == "wall_time"
    assert deadline["progress"]["consumed_batches"] == 0
    # A bad update preserves the last committed finite state, never the failed update.
    def broken_step(*a, **kw):
        update = compile_step(*a, **kw)
        def broken(*inputs):
            p, opt, metrics = update(*inputs)
            return p, opt, {**metrics, "loss": jnp.asarray(float("nan"))}
        return broken
    monkeypatch.setattr(package, "compile_pretrain_step", broken_step)
    with pytest.raises(FloatingPointError):
        runner.run(args("nonfinite"))
    _, preserved = ckpt.read_metadata(tmp_path / "nonfinite/checkpoints")
    assert preserved["metadata"]["completed_steps"] == 0
    assert json.loads((tmp_path / "nonfinite/summary.json").read_text())["status"] == "failed"


def test_production_preset_and_notebook():
    from build_pretrain_notebook import build
    manifest = {"splits": {"train": {"real_tokens": 3_000_000_000, "rows": 366_500}}}
    config, train, native = runner.production_recipe(manifest, 3_000_000_000)
    assert train.total_steps == 91625
    assert (config.n_experts, config.d_ff, config.experts_per_token) == (48, 128, 4)
    assert (config.parallelism.attention_context_shard, native.attention_data_shards) == (2, 4)
    assert native.moe_buffer_divisor == 4
    assert native.splash_batch_mode == "sequential"
    assert native.splash_block_q_dkv == 128
    assert native.splash_compressed_block_q_dkv == native.splash_global_block_q_dkv == 1024
    assert not config.indexer_training.apply_candidate_mask
    # Builder compiles every Python cell and validates notebook schema.
    nb = build()
    code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert "BASE_TOKENS = 2_400_000_000" in code
    assert "scripts/run_pretrain.py" in code
    assert "run_moe_buffer_experiment" not in code
