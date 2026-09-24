# Archived pretraining stress experiment

This document described an earlier synthetic TPU stress/profiling schedule. It is **not** the current production pretraining plan.

The exact historical document is preserved at [`docs/experiments/archive/pretrain_stress.md`](experiments/archive/pretrain_stress.md), together with the dated experiment evidence under `docs/experiments/`.

Current training semantics are defined in [`docs/training_stages.md`](training_stages.md): pretraining is a separate token-budgeted stage targeting 3B non-padding training tokens, followed by independently budgeted mid-training and SFT stages.

The old 10,000-step coordinate system and step-6000 late-indexer probe remain meaningful only as historical/synthetic benchmark settings. They must not be interpreted as the production 3B-token schedule.

For current TPU experiments, use `notebooks/nano_dsv41f_final_attention_tuning.ipynb` or `notebooks/nano_dsv41f_moe_buffers.ipynb`.
