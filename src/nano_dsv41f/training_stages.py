from __future__ import annotations

from dataclasses import dataclass

from .corpus_curriculum import PhaseSpec


@dataclass(frozen=True)
class TrainingStageSpec:
    """Repository-level training stage semantics.

    Stage budgets are intentionally independent. They are not fractions of one shared
    optimizer-step schedule: pretraining is token-budgeted, while mid-training and SFT
    can choose their own step/token budgets after the preceding checkpoint exists.
    """

    name: str
    objective: str
    data_view: str
    q_aware_packing: bool
    indexer_distillation: bool
    candidate_mask: bool
    assistant_only_loss: bool


PRETRAIN_STAGE = TrainingStageSpec(
    name="pretrain",
    objective="causal_lm",
    data_view="general_pretraining_text",
    q_aware_packing=False,
    indexer_distillation=False,
    candidate_mask=False,
    assistant_only_loss=False,
)

MIDTRAIN_STAGE = TrainingStageSpec(
    name="midtrain",
    objective="causal_lm_plus_indexer_distillation",
    data_view="curated_documents_plus_reasoning_and_agent_traces",
    q_aware_packing=True,
    indexer_distillation=True,
    candidate_mask=False,
    assistant_only_loss=False,
)

SFT_STAGE = TrainingStageSpec(
    name="sft",
    objective="assistant_only_supervised_finetuning",
    data_view="structured_reasoning_and_agent_traces",
    q_aware_packing=True,
    indexer_distillation=True,
    candidate_mask=True,
    assistant_only_loss=True,
)

DEFAULT_TRAINING_STAGES = (PRETRAIN_STAGE, MIDTRAIN_STAGE, SFT_STAGE)

# The old middle/late-mid document curricula are replaced by one explicit mid-training
# corpus. The weights intentionally preserve the higher-quality endpoint of the old
# schedule: general/educational text remains present, while code/math are enriched.
MIDTRAIN_DOCUMENT_PHASE = PhaseSpec(
    name="midtrain",
    start_fraction=0.0,
    end_fraction=1.0,
    source_weights=(
        ("fineweb_edu", 0.50),
        ("cosmopedia_v2", 0.05),
        ("codeparrot_clean", 0.20),
        ("finemath_4plus", 0.25),
    ),
    q_aware_packing=True,
)

# Starting points for stage-level pool sampling. These are ablation baselines, not
# claims about an optimal recipe. SFT excludes ordinary document rows by default.
DEFAULT_MIDTRAIN_POOL_MIX = {
    "document": 0.80,
    "reasoning": 0.05,
    "agent": 0.15,
}
DEFAULT_SFT_POOL_MIX = {
    "reasoning": 1.0 / 3.0,
    "agent": 2.0 / 3.0,
}


def stage_by_name(name: str) -> TrainingStageSpec:
    for stage in DEFAULT_TRAINING_STAGES:
        if stage.name == name:
            return stage
    raise KeyError(name)
