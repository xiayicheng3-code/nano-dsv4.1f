import ast
from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from nano_dsv41f.pretrain_recipe import pretrain_recipe, recipe_manifest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("stress_worker", ROOT / "scripts/run_pretrain_stress.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def test_recipe_keeps_full_architecture_and_training_schedule():
    from nano_dsv41f import ModelConfig, TrainConfig
    from nano_dsv41f.chat_protocol import validate_model_token_ids
    from nano_dsv41f.model import build_layer_specs
    baseline, train, native = pretrain_recipe()
    assert baseline.n_layers == 7 and baseline.d_model == 512 and baseline.vocab_size == 32768
    assert build_layer_specs(baseline) == build_layer_specs(ModelConfig())
    assert baseline.optimizer == ModelConfig().optimizer
    assert replace(train, seq_len=4096) == TrainConfig()
    assert not validate_model_token_ids(baseline)
    assert baseline.remat.policy == "block" and native.moe_ragged_implementation == "mosaic"
    assert not baseline.indexer_training.apply_candidate_mask
    assert baseline.indexer.candidate_topk_blocks * baseline.indexer.candidate_block_size > 512
    for name in ("narrow24", "narrow48"):
        candidate, _, _ = pretrain_recipe(profile=name)
        assert candidate.n_experts * candidate.d_ff == baseline.n_experts * baseline.d_ff
        assert candidate.experts_per_token == (4 if name == "narrow48" else 2)
    assert pretrain_recipe(profile="narrow48", top_k=2)[0].experts_per_token == 2
    assert recipe_manifest(baseline, train, native)["sha256"] == recipe_manifest(*pretrain_recipe())["sha256"]


@pytest.mark.parametrize("layout,real,lm", [("long", 8192, 8191), ("packed", 8188, 8184)])
def test_batch_masks_and_reproducible_8k_rows(layout, real, lm):
    config, _, _ = pretrain_recipe()
    (ids, seg, mask), info = worker.make_batch(config, batch_rows=4, layout=layout)
    assert ids.shape == seg.shape == mask.shape == (4, 8192)
    assert info["real_tokens"] == 4 * real and info["lm_tokens"] == 4 * lm
    np.testing.assert_array_equal(seg[:, ::2], seg[:, 1::2])
    assert info == worker.make_batch(config, batch_rows=4, layout=layout)[1]
    assert not np.array_equal(ids[0], ids[1])


def test_invalid_four_device_recipe_rejected():
    with pytest.raises(ValueError, match="supported"):
        pretrain_recipe(cp=2, dp=2)


def test_archived_stress_notebook_is_valid_and_preserves_canonical_bootstrap():
    import nbformat
    archived = ROOT / "notebooks/archive/nano_dsv41f_pretrain_stress.ipynb"
    nb = nbformat.read(archived, as_version=4)
    nbformat.validate(nb)
    code = [cell.source for cell in nb.cells if cell.cell_type == "code"]
    assert (ROOT / "scripts/kaggle_bootstrap.py").read_text() in code
    for source in code:
        ast.parse(source)

    notice = nbformat.read(ROOT / "notebooks/nano_dsv41f_pretrain_stress.ipynb", as_version=4)
    nbformat.validate(notice)
    assert not [cell for cell in notice.cells if cell.cell_type == "code"]
    text = "\n".join(cell.source for cell in notice.cells if cell.cell_type == "markdown")
    assert "Archived" in text and "final_attention_tuning" in text


@pytest.mark.parametrize("profile,top_k", [("baseline", 2), ("narrow48", 4)])
def test_failure_report_is_saved_before_tpu_required(tmp_path, profile, top_k):
    import os
    import subprocess
    import sys
    path = tmp_path / "failed.json"
    result = subprocess.run([sys.executable, "scripts/run_pretrain_stress.py", "--output", str(path),
                             "--profile", profile],
                            cwd=ROOT, env=dict(os.environ, JAX_PLATFORMS="cpu",
                                               PYTHONPATH=str(ROOT / "src")), capture_output=True)
    assert result.returncode != 0
    report = json.loads(path.read_text())
    assert report["status"] == "failed" and report["stage"] == "runtime"
    assert "expected a TPU runtime" in report["exception"]
    assert report["recipe"]["train"]["seq_len"] == 8192
    assert report["recipe"]["model"]["experts_per_token"] == top_k
