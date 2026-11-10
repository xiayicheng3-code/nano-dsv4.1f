from pathlib import Path
import sys

import numpy as np
import pytest
torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from chat_samples import loop_runs, repeat_labels as list_repeat_labels
from repetition_direction import FFNRecorder, FFNSteer, onset_labels, repeat_labels, steer_configs
from test_vllm_v41_cpu import make_cpu


def test_repeat_labels_agree_and_loop_runs():
    tokens = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 9, 8, 7]
    assert list(repeat_labels(tokens, 4)) == list_repeat_labels(tokens, 4)
    # repeats start at index 7 (completes 1 2 3 4 again) and last through index 11
    assert loop_runs(list_repeat_labels(tokens, 4), min_run=5) == [(7, 12)]
    assert loop_runs(list_repeat_labels(tokens, 4), min_run=6) == []


def test_onset_labels_mark_first_copied_ngram_and_distant_fresh_tokens():
    fresh = list(range(100, 130))
    loop = [1, 2, 3, 4, 5] * 4
    tokens = fresh + loop
    labels = onset_labels(tokens, n=4, min_run=8, margin=8)
    first_repeat = len(fresh) + 5 + 3  # the second "1 2 3 4" completes at this index
    assert labels[first_repeat - 3:first_repeat + 1].tolist() == [1, 1, 1, 1]
    assert (labels[:len(fresh) - 8] == 0).all()
    assert (labels[first_repeat + 1:] == -1).all()
    assert set(np.unique(labels)) <= {-1, 0, 1}


def test_ffn_steer_ablation_removes_direction_and_restores_runtime():
    from nano_dsv41f.vllm_v41_cpu import model as runtime_model
    torch.set_num_threads(1)
    model = make_cpu(23)[2]
    original = runtime_model.apply_moe
    ids = torch.tensor([[1, 5, 9, 2, 7, 3]])
    d = model.config.d_model
    r = torch.zeros(d); r[0] = 1.0
    dirs = {i: r for i in range(model.config.n_layers)}
    with FFNSteer(dirs, mode="ablate", strength=1.0), FFNRecorder() as rec:
        model.forward(ids)
    assert all(float((v @ r).abs().max()) < 1e-6 for v in rec.outputs.values())
    # Subtraction at layer 0 only: later layers see changed inputs, so compare layer 0.
    with FFNSteer({0: r}, mode="subtract", strength=2.0, scales={0: 0.5}), FFNRecorder() as rec_sub:
        model.forward(ids)
    with FFNRecorder() as rec_base:
        model.forward(ids)
    torch.testing.assert_close(rec_sub.outputs[0], rec_base.outputs[0] - 1.0 * r)
    assert not torch.allclose(rec_sub.outputs[1], rec_base.outputs[1])
    assert runtime_model.apply_moe is original


def test_steer_configs_include_baseline_and_random_controls():
    report = {"layers": {str(i): {} for i in range(3)}}
    names = [c["name"] for c in steer_configs(report, [2, 0])]
    assert names[0] == "none" and "ablate_L2-0" in names and "ablate_all" in names
    assert {"random_subtract_x1", "random_subtract_x2", "random_subtract_x4", "random_ablate_all"} <= set(names)


def test_repetition_notebook_builds_and_covers_every_config(tmp_path):
    import json
    import build_repetition_notebook as B
    path = B.build(tmp_path, ref="some-branch", prompts=12)
    nb = json.loads(path.read_text())
    source = "".join("".join(c["source"]) for c in nb["cells"])
    assert "'some-branch'" in source and "PROMPTS = 12" in source
    report = {"layers": {str(i): {} for i in range(7)}}
    names = {c["name"] for c in steer_configs(report, [4, 6])}
    split = {("ablate_L4-6" if n == "ablate_top2" else n) for group in B.GPU_SPLIT for n in group}
    assert split == names
