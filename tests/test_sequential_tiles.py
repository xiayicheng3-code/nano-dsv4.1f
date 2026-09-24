"""The focused preset remains reproducible after its notebook is archived."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_sequential_tiles as sweep


@pytest.mark.parametrize("fail512", [False, True])
def test_focused_plan_and_candidate_failure_isolation(tmp_path, monkeypatch, fail512):
    calls = []
    def launch(command, path, timeout):
        calls.append(command)
        tile = int(command[command.index("--tile") + 1])
        assert command[command.index("--intervention") + 1] == "sequential_tile"
        assert command[command.index("--rows") + 1] == "8"
        assert command[command.index("--composition") + 1] == "distinct"
        assert command[command.index("--trace-steps") + 1] == "0"
        if fail512 and tile == 512:
            assert "--preflight-only" in command
            return {"status": "failed", "exception": "simulated candidate compilation failure"}
        return {"status": "passed", "variants": {
            "sequential": {"combined": {"median_seconds": 1.0}},
            f"tile{tile}": {"combined": {"median_seconds": .8}}}}
    monkeypatch.setattr(sweep, "launch", launch)
    monkeypatch.setattr(sys, "argv", ["sweep", "--bank", str(tmp_path / "bank"),
                                     "--output", str(tmp_path / "run")])
    if fail512:
        with pytest.raises(SystemExit, match="1"):
            sweep.main()
    else:
        sweep.main()
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert len(summary["preflights"]) == 4
    assert len(summary["cases"]) == (6 if fail512 else 12)
    assert len(calls) == (10 if fail512 else 16)
    for row in summary["comparisons"]:
        assert row["complete"] == (not fail512 or row["candidate_tile"] == 256)
        if row["complete"]:
            assert row["median_time_reduction_fraction"] == pytest.approx(.2)


def test_archived_notebook_preserves_focused_preset_and_active_path_is_notice(monkeypatch):
    import os
    import nbformat

    notebook = nbformat.read(ROOT / "notebooks/archive/nano_dsv41f_sequential_tiles.ipynb",
                             as_version=4)
    nbformat.validate(notebook)
    monkeypatch.setenv("NANO_ATTN_ROWS", "4,8,24")
    monkeypatch.setenv("NANO_ATTN_FULL_MODEL", "1")
    monkeypatch.setenv("NANO_ATTN_TILE_TEST", "1")
    monkeypatch.setenv("NANO_DSV41F_REF", "old-commit")
    # Isolate the historical settings-cell environment assignment from the test process.
    with monkeypatch.context() as patch:
        patch.setattr(os, "environ", dict(os.environ))
        namespace = {}
        exec(notebook.cells[1].source, namespace)
        settings = namespace["SETTINGS"]
        assert os.environ["NANO_ATTN_ROWS"] == "8"
        assert os.environ["NANO_ATTN_TILES"] == "128,256,512"
        assert os.environ["NANO_ATTN_FULL_MODEL"] == "0"
        assert os.environ["NANO_ATTN_TILE_TEST"] == "0"
        assert os.environ["NANO_DSV41F_REF"] != "old-commit"
        assert settings["NANO_ATTN_FAMILIES"] == "compressed,global"
    code = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    assert "scripts/run_sequential_tiles.py" in code
    assert "scripts/run_attention_experiment.py" not in code
    assert "scripts/run_stress_suite.py" not in code

    notice = nbformat.read(ROOT / "notebooks/nano_dsv41f_sequential_tiles.ipynb", as_version=4)
    nbformat.validate(notice)
    assert not [cell for cell in notice.cells if cell.cell_type == "code"]
    text = "\n".join(cell.source for cell in notice.cells if cell.cell_type == "markdown")
    assert "Archived" in text and "final_attention_tuning" in text


@pytest.mark.parametrize("ratio", [1, 2])
def test_bf16_global_families_sequential_tiles_match(ratio):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from nano_dsv41f import make_v5e_mesh
    from run_attention_replay import make_function, operations
    from attention_replay_utils import error_metrics
    if jax.device_count() < 8:
        pytest.skip("requires eight CPU devices")
    # CP2 leaves 512 query positions per shard, enough for the largest tile.
    batch, length, dim = 8, 1024, 64
    kvlen = length + length // ratio
    meta = {"arrays": {"q": {"shape": [batch, length, 1, dim]},
                       "k": {"shape": [batch, kvlen, dim]}},
            "local_window": 128, "ratio": ratio}
    rng = np.random.default_rng(17)
    q = jnp.asarray(rng.normal(size=(batch, length, 1, dim)) * .1, jnp.bfloat16)
    kv = jnp.asarray(rng.normal(size=(batch, kvlen, dim)) * .1, jnp.bfloat16)
    ct = jnp.asarray(rng.normal(size=q.shape), jnp.bfloat16)
    segments = jnp.tile(jnp.repeat(jnp.arange(2), length // 2)[None], (batch, 1))
    kv_segments = jnp.concatenate((segments, segments[:, ::ratio]), axis=1)
    args = (q, kv, jnp.array([.234567], jnp.float32), segments, kv_segments, ct)
    mesh = make_v5e_mesh()
    reference = operations(make_function(mesh, meta, "sequential", tile=128, interpret=True))[3](*args)
    for tile in (256, 512):
        actual = operations(make_function(mesh, meta, "sequential", tile=tile, interpret=True))[3](*args)
        assert actual[1][-1].dtype == jnp.float32
        for x, y in zip(jax.tree.leaves(actual), jax.tree.leaves(reference)):
            assert error_metrics(x, y)["passed"]
