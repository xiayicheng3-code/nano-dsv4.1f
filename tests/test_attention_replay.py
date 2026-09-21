"""Causal-replay controls: dense oracle, gradients, and conservative decisions."""
from pathlib import Path
import sys
import numpy as np
import pytest
import jax
import jax.numpy as jnp

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from attention_replay_utils import decide_h1, error_metrics, select_rows
from run_attention_replay import make_function, operations


@pytest.mark.parametrize("ratio", [0, 1, 2])
def test_scheduling_and_tiles_match_dense_oracle(ratio):
    if jax.device_count() < 8:
        pytest.skip("requires eight CPU devices")
    from nano_dsv41f import make_v5e_mesh
    from nano_dsv41f.tpu_native import combined_csa2_mask
    # Two rows per DP replica exercises the proposed sequential path.
    b, t, h, d = 8, 512, 1, 64
    klen = t + (t // ratio if ratio else 0)
    meta = {"arrays": {"q": {"shape": [b, t, h, d]}, "k": {"shape": [b, klen, d]}},
            "local_window": 8, "ratio": max(1, ratio)}
    rng = np.random.default_rng(19)
    q = jnp.asarray(rng.normal(size=(b,t,h,d)).astype("float32") * .1)
    k = jnp.asarray(rng.normal(size=(b,klen,d)).astype("float32") * .1)
    v = jnp.asarray(rng.normal(size=(b,klen,d)).astype("float32") * .1)
    sinks = jnp.array([.2])
    seg = jnp.tile(jnp.repeat(jnp.arange(4), t // 4)[None], (b, 1))
    kvseg = jnp.concatenate((seg, seg[:, ::ratio]), 1) if ratio else seg
    ct = jnp.asarray(rng.normal(size=q.shape).astype("float32"))
    static = jnp.asarray(combined_csa2_mask(t, local_window=8, global_kv_len=klen-t,
                                          compression_ratio=max(1,ratio)))
    def dense(q, k, v, sinks, qseg, kseg):
        logits = jnp.einsum("bthd,bkd->bthk", q * d**-.5, k)
        mask = static[None] & (qseg[:, :, None] == kseg[:, None, :])
        logits = jnp.where(mask[:, :, None], logits, -jnp.inf)
        logits = jnp.concatenate((logits, jnp.broadcast_to(sinks[None,None,:,None], (b,t,h,1))), -1)
        return jnp.einsum("bthk,bkd->bthd", jax.nn.softmax(logits, -1)[...,:-1], v)
    inputs = (q,k,v,sinks,seg,kvseg)
    expected = operations(dense)[3](*inputs, ct)
    mesh = make_v5e_mesh()
    for mode, tile in (("vmap", None), ("sequential", None), ("vmap", 256)):
        fn = make_function(mesh, meta, mode, tile=tile, interpret=True, independent_values=True)
        fwd, prepare, back, combined = operations(fn)
        actual = combined(*inputs, ct)
        for x,y in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
            np.testing.assert_allclose(x,y,atol=2e-5,rtol=2e-4)
        # Verify that the separately timed pullback has the same gradients and
        # dynamic residuals, including the replica-reduced sink gradient.
        _, pb = prepare(*inputs)
        for x,y in zip(back(pb,ct), actual[1]):
            np.testing.assert_allclose(x,y,atol=2e-5,rtol=2e-4)
        tied = make_function(mesh, meta, mode, tile=tile, interpret=True)
        actual_tied = operations(tied)[3](q,k,sinks,seg,kvseg,ct)
        independent = operations(fn)[3](q,k,k,sinks,seg,kvseg,ct)
        expected_tied = (independent[0], (independent[1][0],
                         independent[1][1] + independent[1][2], independent[1][3]))
        for x,y in zip(jax.tree.leaves(actual_tied), jax.tree.leaves(expected_tied)):
            np.testing.assert_allclose(x,y,atol=2e-5,rtol=2e-4)


def test_row_control_and_decision_require_complete_stable_repeats():
    bank = np.arange(12).reshape(6,2)
    np.testing.assert_array_equal(select_rows(bank,8,"repeated"), np.tile(bank[:1],(8,1)))
    with pytest.raises(ValueError):
        select_rows(bank,8,"distinct")
    good = dict(vmap4=1., vmap8=3., sequential4=1., sequential8=2.1, passed=True)
    assert decide_h1([good]*3)["decision"] == "supports_H1_scheduling_remedy"
    assert decide_h1([good]*2)["decision"] == "inconclusive"
    assert decide_h1([good,good,{**good,"vmap8":3.3}])["decision"] == "inconclusive"
    assert decide_h1([{**good,"vmap8":2.1}]*3)["decision"] == "evidence_against_standalone_H1"
    assert not error_metrics([np.nan],[1])["passed"]


def test_replay_worker_uses_runtime_residuals_and_writes_report(tmp_path, monkeypatch):
    if jax.device_count() < 8:
        pytest.skip("requires eight CPU devices")
    import hashlib
    import json
    from types import SimpleNamespace
    import nano_dsv41f
    import run_attention_replay as replay
    monkeypatch.setattr(nano_dsv41f, "validate_v5e_runtime", lambda: None)
    original = replay.make_function
    monkeypatch.setattr(replay, "make_function", lambda *a, **kw: original(*a, **kw, interpret=True))
    rng = np.random.default_rng(101)
    q = rng.normal(size=(8,256,1,64)).astype("float32") * .1
    kv = rng.normal(size=(8,256,64)).astype("float32") * .1
    seg = np.tile(np.repeat(np.arange(2, dtype="int32"),128)[None], (8,1))
    arrays = dict(q=q,k=kv,v=kv,sinks=np.array([.2],dtype="float32"),
                  q_segments=seg,kv_segments=seg,cotangent=q)
    meta = {"local_window":8,"ratio":1,"arrays":{k:{"shape":list(v.shape),"dtype":str(v.dtype)} for k,v in arrays.items()}}
    bank = tmp_path / "local.npz"
    np.savez(bank, **arrays)
    meta["file_sha256"] = hashlib.sha256(bank.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps({"families":{"local":meta}}))
    args = SimpleNamespace(bank=tmp_path,family="local",rows=8,composition="repeated",
                           intervention="schedule",repeat=1,output=tmp_path/"case.json",
                           warmup=3,steps=12,trace_steps=0)
    report = {}
    replay.run(args, report, lambda stage: None)
    assert report["status"] == "passed"
    assert report["order"] == ["sequential","vmap"]
    for v in report["variants"].values():
        assert len(v["combined"]["seconds"]) == 12
        assert all(e["passed"] for e in v["errors"].values())
        assert v["backward"]["median_seconds"] > 0
    assert len(list(tmp_path.glob("*.hlo.txt.gz"))) == 2


def test_capture_round_trip_preserves_model_dtypes_and_families(tmp_path, monkeypatch):
    if jax.device_count() < 8:
        pytest.skip("requires eight CPU devices")
    import json
    from dataclasses import replace
    import nano_dsv41f
    import nano_dsv41f.pretrain_recipe as recipe
    import capture_attention_replay as capture
    from nano_dsv41f import TrainConfig
    from nano_dsv41f.tpu_native import TPUNativeConfig
    from test_reference_model import tiny_config
    cfg = tiny_config(dspark=False)
    cfg = replace(cfg, n_experts=8, remat=replace(cfg.remat, policy="block"),
                  parallelism=replace(cfg.parallelism, vocab_shard=8,engram_table_shard=8,
                      expert_shard=8,attention_context_shard=2,attention_data_shard=4,indexer_context_shard=2))
    monkeypatch.setattr(nano_dsv41f,"validate_v5e_runtime",lambda: None)
    monkeypatch.setattr(recipe,"pretrain_recipe",lambda **kw: (cfg,TrainConfig(seq_len=256),
        TPUNativeConfig(attention_data_shards=4,splash_interpret=True,moe_ragged_implementation="xla")))
    make_batch = capture.make_batch
    monkeypatch.setattr(capture,"make_batch",lambda *a,**kw: make_batch(*a,**kw,seq_len=256))
    data = tmp_path / "rows.npz"
    ids = np.random.default_rng(1).integers(3,64,size=(4,256),dtype="int32")
    np.savez(data,input_ids=ids,segment_ids=np.zeros_like(ids),token_mask=np.ones_like(ids,dtype=bool))
    capture.capture(data,tmp_path/"bank",rows=4)
    meta = json.loads((tmp_path/"bank"/"manifest.json").read_text())
    assert set(meta["families"]) == {"local","compressed","global"}
    for family,length in (("local",256),("compressed",384),("global",512)):
        spec = meta["families"][family]
        assert spec["arrays"]["k"]["shape"][1] == length
        with np.load(tmp_path/"bank"/f"{family}.npz") as stored:
            import hashlib
            for name,array in stored.items():
                restored = array.astype(jnp.dtype(spec["arrays"][name]["dtype"]))
                assert hashlib.sha256(restored.tobytes()).hexdigest() == spec["arrays"][name]["sha256"]
