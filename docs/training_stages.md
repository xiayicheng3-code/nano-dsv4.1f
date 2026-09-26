# Training stages for nano-dsv4.1f

The canonical training plan is now three explicit stages:

1. **Pretrain** — broad causal language modelling on the large general corpus.
2. **Mid-train** — one curated, long-context stage that enriches code/math and introduces reasoning/agent trajectories while training the retriever/indexer.
3. **SFT** — assistant-only supervised fine-tuning on cleaned structured conversations and trajectories.

These are separate **data and supervision regimes**; their budgets are configured
by the experiment. The current Kaggle experiment allocates **2.4B tokens to
pretraining and 600M to mid-training**, sharing a continuous full-3B optimizer
schedule. SFT remains separate. This replaces the earlier experiment allocation
of 3B base tokens plus independently budgeted mid-training; it does not restore
the old early/middle/late-mid data stages. See [the run guide](pretrain_run.md).

The machine-readable policy is `nano_dsv41f.training_stages.DEFAULT_TRAINING_STAGES`.

## Stage contract

| Stage | Primary data | Loss view | Q-aware packing | Indexer distillation | Hierarchical candidate mask |
| --- | --- | --- | --- | --- | --- |
| pretrain | broad general text | ordinary causal LM | no | late/configurable auxiliary | off |
| midtrain | curated documents + a minority of reasoning/agent traces | ordinary causal LM + indexer auxiliary loss | yes | on | off |
| sft | cleaned reasoning/assistant/tool trajectories | assistant-only supervised loss | retained | on | on |

The candidate-mask column is the intended stage policy. The current training runner does not infer the stage from a checkpoint name, so callers must still set the relevant indexer configuration explicitly. The existing late selective indexer-distillation window may still run during pretraining; the three-stage redesign does not remove that auxiliary objective. It separates **data and supervision regimes** while keeping unrestricted legal history for indexer training until SFT.

## 1. Pretrain

Use the dedicated pretraining-only pipeline rather than the old staged corpus builder. The prepared corpus target is:

- **3,000,000,000** non-padding training tokens measured with the frozen nano tokenizer;
- **10,000,000** validation tokens;
- 8192-token packed rows;
- FineWeb-Edu as the general pretraining source;
- ordinary best-fit packing with ratio-2 segment alignment;
- no reasoning/agent mixture and no assistant-only loss mask;
- the existing late selective indexer auxiliary may remain enabled, with candidate masking off.

The data preparation implementation is integrated into main. The production
notebook consumes **2.4B** non-padding tokens from this 3B-token corpus, then saves
a full continuation checkpoint. Preparing 3B corpus tokens does not require
training on every row in the first stage.

Pretraining is deliberately simple at the data level. Its job is to acquire broad language/statistical capacity before we spend substantial mixture budget on tool syntax and reasoning formats. The configured late indexer auxiliary can still begin training retrieval before the dedicated mid-training stage.

## 2. Mid-train

The previous `middle` and `late_mid` document phases are replaced by **one** mid-training stage. `scripts/prepare_midtrain_corpus.py` is the canonical document entry point.

The starting document mix is:

| source | token share |
| --- | ---: |
| FineWeb-Edu | 50% |
| Cosmopedia v2 | 5% |
| permissively filtered CodeParrot clean | 20% |
| FineMath 4+ | 25% |

This intentionally keeps the higher-quality endpoint of the old curriculum instead of averaging the old middle and late-mid mixtures. The model has already seen the large general pretraining corpus; mid-training is where enrichment is useful.

The starting **pool-level** sampler is:

```text
document  80%
reasoning  5%
agent     15%
```

These are ablation baselines, not claimed optimal weights. Reasoning and agent traces use the ordinary causal-LM view during this stage: `token_mask` and `segment_ids` determine valid targets; `sft_loss_mask` is not applied.

All mid-training document rows are Q-aware. We keep the existing query-budget diagnostics and expected sampled-Q-density objective so long-context positions are not starved merely because ordinary best-fit packing happens to favor short segments.

Mid-training continues the selective indexer-distillation auxiliary objective. The hierarchical L3→L5 candidate restriction remains **off** here so the student can learn against all legal compressed history.

The data-preparation wrapper has its own `--total-steps` value. That argument
sizes the **mid-training corpus**; it does not reset the optimizer schedule.
For the current 80/20 experiment the mid-training training budget is 600M tokens,
and its launcher must preserve the full-3B optimizer schedule and learned state.
The source mixtures below remain preparation defaults; the proposed 50% broad /
25% extra code-technical / 15% math-solutions / 10% agent mixture will be selected
when preparing the second notebook, not by this pretraining launcher.

## 3. SFT

SFT is a separate optimization stage, not another document-curriculum phase.

The trace shards already contain two compatible views:

- context/causal metadata: `input_ids`, `segment_ids`, `token_mask`;
- assistant supervision: `sft_loss_mask`.

For SFT, supervise only targets satisfying both the ordinary token mask and the assistant mask. User/system/tool-result spans remain context; assistant reasoning, tool calls, final content and EOS are supervised according to the renderer/cleaner policy.

Ordinary document rows are excluded from the default SFT pool. A starting trace-only mix matching the existing 4M reasoning / 8M agent token targets is:

```text
reasoning  1/3
agent      2/3
```

This ratio should be revisited when the final SFT sources are selected; it is not a reason to preserve any particular upstream dataset. The current reasoning-source catalog uses explicit permissive sources (OpenR1-Math, CHIMERA, and X-Coder); changing those sources does not change the stage contract.

SFT is also where the current architecture intends to turn on the hierarchical candidate mask. This makes the final retrieval/indexer behavior match the inference hierarchy after unrestricted pretraining/mid-training distillation.

## Stage-aware preparation

Prepare the single mid-training document corpus:

```bash
python scripts/prepare_midtrain_corpus.py \
  --tokenizer /path/to/tokenizer.json \
  --output-dir /path/to/midtrain-8k \
  --total-steps 10000 \
  --seq-len 8192 \
  --query-budget 128 \
  --q-threshold 640 \
  --q-band-edges 640,768,1024,1536,2048,3072,4096,6144,8192
```

Prepare reasoning/agent traces and stamp explicit stage views:

```bash
python scripts/prepare_stage_traces.py \
  --tokenizer /path/to/tokenizer.json \
  --output-dir /path/to/traces \
  --pool all \
  --seq-len 8192
```

The stage-aware trace wrapper delegates to the maintained source adapters, then replaces the legacy `early` / `middle` / `late_mid` training advice in the top-level manifest with `midtrain` and `sft` views.

The underlying document packer still writes `curriculum_manifest.json` for compatibility with existing tooling, but the canonical wrapper produces exactly one document phase named `midtrain`.

## Migration from the old curriculum

The old design divided one document run into `early`, `middle`, and `late_mid` fractions tied to optimizer progress. That made pretraining, mid-training, and behavior tuning blur together and made later design changes easy to misread as independent stages.

The new rule is simpler:

- general-text scale belongs to **pretrain**;
- the old higher-quality/Q-aware continuation belongs to **mid-train**;
- assistant-only behavior learning belongs to **SFT**.

Legacy helpers remain temporarily because the Q-aware packer and source adapters are reused internally. New notebooks and manifests should use the explicit stage-aware entry points above.
