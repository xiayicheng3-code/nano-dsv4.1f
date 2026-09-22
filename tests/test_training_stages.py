import math

from nano_dsv41f.training_stages import (
    DEFAULT_MIDTRAIN_POOL_MIX,
    DEFAULT_SFT_POOL_MIX,
    DEFAULT_TRAINING_STAGES,
    MIDTRAIN_DOCUMENT_PHASE,
    stage_by_name,
)


def test_training_stages_are_explicit_and_ordered():
    assert [stage.name for stage in DEFAULT_TRAINING_STAGES] == [
        "pretrain",
        "midtrain",
        "sft",
    ]
    assert not stage_by_name("pretrain").q_aware_packing
    assert stage_by_name("midtrain").q_aware_packing
    assert stage_by_name("sft").assistant_only_loss
    assert not stage_by_name("midtrain").candidate_mask
    assert stage_by_name("sft").candidate_mask


def test_single_midtrain_document_phase_replaces_legacy_subphases():
    assert MIDTRAIN_DOCUMENT_PHASE.name == "midtrain"
    assert MIDTRAIN_DOCUMENT_PHASE.start_fraction == 0.0
    assert MIDTRAIN_DOCUMENT_PHASE.end_fraction == 1.0
    assert MIDTRAIN_DOCUMENT_PHASE.q_aware_packing
    assert math.isclose(sum(MIDTRAIN_DOCUMENT_PHASE.weights.values()), 1.0)
    assert MIDTRAIN_DOCUMENT_PHASE.weights["finemath_4plus"] == 0.25


def test_stage_pool_mixes_are_normalized():
    assert math.isclose(sum(DEFAULT_MIDTRAIN_POOL_MIX.values()), 1.0)
    assert math.isclose(sum(DEFAULT_SFT_POOL_MIX.values()), 1.0)
    assert "document" in DEFAULT_MIDTRAIN_POOL_MIX
    assert "document" not in DEFAULT_SFT_POOL_MIX
