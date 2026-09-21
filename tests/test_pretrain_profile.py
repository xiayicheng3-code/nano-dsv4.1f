import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


data = script("prepare_profile_data")
worker = script("run_pretrain_stress")


@pytest.fixture
def corpus(tmp_path):
    from nano_dsv41f.chat_protocol import nano_v41_tokenizer_contract
    root = tmp_path / "mount" / "nested-corpus"
    root.mkdir(parents=True)
    vocab = nano_v41_tokenizer_contract().token_to_id
    vocab.update({f"token-{i}": i for i in range(len(vocab), 32768)})
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text(json.dumps({"model": {"vocab": vocab}}))
    config = {"vocab_size": 32768, "seq_len": 8192, "alignment": 2,
              "tokenizer_sha256": hashlib.sha256(tokenizer.read_bytes()).hexdigest()}
    manifest = {"format": data.FORMAT, "complete": True, "config": config,
                "parts": ["parts/part-00000", "parts/part-00001"]}
    for part_id, part in enumerate(manifest["parts"]):
        directory = root / part / "train"
        directory.mkdir(parents=True)
        tokens = np.full((16, 8192), 2, np.uint16)
        for i in range(16):
            tokens[i, :5] = [0, 23 + 16 * part_id + i, 24, 25, 1]
            tokens[i, 6:9] = [0, 26, 1]
        arrays = {"tokens": tokens, "lengths": np.tile(np.array([5, 3], np.uint16), 16),
                  "offsets": np.arange(0, 33, 2, dtype=np.uint32)}
        files = {}
        for key, array in arrays.items():
            path = directory / f"shard-00000-{key}.npy"
            np.save(path, array)
            files[key] = {"file": path.name}
        (directory.parent / "manifest.json").write_text(json.dumps({
            "splits": {"train": {"shards": [{"rows": 16, "files": files}]}}}))
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root, tokenizer, manifest


def test_compact_sampling_preserves_masks_and_nested_batches(corpus, tmp_path):
    from nano_dsv41f.pretrain_recipe import pretrain_recipe
    root, tokenizer, manifest = corpus
    output = tmp_path / "samples"
    report = data.prepare(root.parent, tokenizer, output, rows=(4, 8), batches=3, seed=19)
    assert len({s["global_row"] for s in report["sources"]}) == 24
    assert len({s["file"] for s in report["sources"]}) == 2
    with np.load(output / "rows-4.npz") as small, np.load(output / "rows-8.npz") as large:
        for key in data.KEYS:
            np.testing.assert_array_equal(small[key].reshape(3, 4, 8192),
                                          large[key].reshape(3, 8, 8192)[:, :4])
        np.testing.assert_array_equal(small["segment_ids"][:, ::2], small["segment_ids"][:, 1::2])
        assert small["token_mask"].sum() == 12 * 8
        assert not small["token_mask"][:, 5].any()
        assert (small["segment_ids"][:, 6:] == 1).all()
    config = pretrain_recipe()[0]
    batch, info = worker.make_batch(config, batch_rows=4, data=output / "rows-4.npz", row_offset=4)
    assert info["lm_tokens"] == 4 * 6
    assert info["real_tokens"] == 4 * 8
    with pytest.raises(ValueError, match="shape"):
        worker.make_batch(config, batch_rows=4, data=output / "rows-4.npz", row_offset=12)
    again, sources = data.sample_rows(root, manifest, 24, 19)
    assert sources == report["sources"]


def test_rejects_wrong_tokenizer_incomplete_and_short_corpus(corpus, tmp_path):
    root, tokenizer, manifest = corpus
    with pytest.raises(ValueError, match="enough"):
        data.sample_rows(root, manifest, 33, 1)
    tokenizer.write_text(tokenizer.read_text() + " ")
    with pytest.raises(ValueError, match="SHA-256"):
        data.prepare(root, tokenizer, tmp_path / "bad", rows=(4,))
    manifest["complete"] = False
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="incomplete"):
        data.find_corpus(root)


def test_routing_does_not_hide_layer_skew_or_count_padding_as_real():
    # Opposite layer skews cancel when summed; preserve the individual layers.
    physical = np.array([[8, 0, 0, 0], [0, 0, 0, 8]])
    m = {"expert_packed_rows": 8, "experts_per_chip": 2,
         "router_loads": physical // 2, "expert_loads_by_layer": physical}
    result = worker.routing_summary(m)
    assert result["physical_dispatch"]["max_over_mean_by_layer"] == [4, 4]
    assert result["physical_dispatch"]["chip_max_over_mean_by_layer"] == [2, 2]
    assert result["physical_dispatch"]["buffer_utilization_by_layer_chip"] == [[1, 0], [0, 1]]
    assert result["real_tokens"]["chip_loads_by_layer"] == [[4, 0], [0, 4]]


def test_profile_notebook_valid_and_bootstrap_canonical():
    import nbformat
    nb = nbformat.read(ROOT / "notebooks/nano_dsv41f_pretrain_profile.ipynb", as_version=4)
    nbformat.validate(nb)
    code = [c.source for c in nb.cells if c.cell_type == "code"]
    assert (ROOT / "scripts/kaggle_bootstrap.py").read_text() in code
    for source in code:
        ast.parse(source)


def test_worker_cycles_bank_and_exports_separate_traces_on_cpu(monkeypatch, tmp_path):
    """Exercise actual JAX compile/trace/report plumbing with a tiny substitute step."""
    import jax
    import jax.numpy as jnp
    import nano_dsv41f as nano
    monkeypatch.setattr(nano, "validate_v5e_runtime", lambda: None)
    monkeypatch.setattr(nano, "runtime_report", lambda: {"test_backend": "cpu"})
    monkeypatch.setattr(nano, "make_v5e_mesh", lambda: None)
    monkeypatch.setattr(nano, "semantic_axes", lambda *a: {})
    monkeypatch.setattr(nano, "put_training_batch", lambda ids, seg, mask, *a:
                        tuple(map(jnp.asarray, (ids, seg, mask))))
    monkeypatch.setattr(nano, "init_model_sharded_mixed_precision",
                        lambda *a, **kw: (jnp.zeros(()), None, None))
    monkeypatch.setattr(nano, "init_optimizer_state_sharded", lambda *a: (jnp.zeros(()), None))

    def compile_step(*args, **kwargs):
        @jax.jit
        def step(p, opt, ids, seg, number, mask):
            physical = jnp.full((7, 48), ids.size * 4 // 48, jnp.int32)
            lm = (mask[:, 1:] & mask[:, :-1] & (seg[:, 1:] == seg[:, :-1])).sum()
            metrics = {"loss": p + 1, "lm_loss": p + 1, "indexer_loss": jnp.zeros(()),
                "expert_dropped": jnp.zeros((48,), jnp.int32), "moe_mosaic_layers": jnp.array(7),
                "lm_tokens": lm, "expert_loads": physical.sum(axis=0),
                "expert_loads_by_layer": physical, "router_loads": physical,
                "expert_packed_rows": jnp.array(ids.size * 4), "experts_per_chip": jnp.array(6),
                "indexer": {"active_queries": jnp.array(128), "query_counts": {"group": jnp.array(128)}}}
            return p + 1, opt + 1, metrics
        return step
    monkeypatch.setattr(nano, "compile_pretrain_step", compile_step)
    args = SimpleNamespace(profile="narrow48", cp=2, dp=4, experts=None, width=None, top_k=4,
        batch_rows=4, steps=2, warmup=1, trace_steps=1, data_batches=2, late_step=6000,
        layout="long", seed=7, data=None, routing="normal", phase="both",
        output=tmp_path / "case.json", trace_dir=tmp_path / "traces")
    report = {}
    worker.run(args, report, lambda stage: None)
    assert report["status"] == "passed"
    assert len(report["batch_bank"]) == 2
    for phase in ("base", "late"):
        values = report["phase_results"][phase]
        assert len(values["steps"]) == 3  # warmup + measured; no traced steps mixed in
        assert [r["batch_index"] for r in values["steps"]] == [0, 1, 0]
        assert values["trace"]["steps"][0]["batch_index"] == 1
        assert values["trace"]["status"] == "exported"
        assert any(p.endswith(".xplane.pb") for p in values["trace"]["files"])
        assert Path(values["hlo_path"]).is_file()
        assert values["microseconds_per_physical_token"] > 0
