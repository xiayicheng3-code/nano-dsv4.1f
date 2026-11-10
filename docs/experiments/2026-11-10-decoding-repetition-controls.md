# Decoding-time repetition controls

Status: **measured, shipped**. The chat notebook (`notebooks/nano_dsv41f_sft_inference_cpu.ipynb`)
now uses DRY with multiplier 2.0 by default.

## Question

Neither weight edits nor inference-time steering of FFN directions stopped the chat model from
looping ([write-up](2026-11-10-repetition-direction-edit.md)). Looping looks like attention-driven
copying. The direct fix is to stop the decoder from choosing the copy. Which decoding-time
control removes loops at the lowest cost?

## Controls (`src/nano_dsv41f/vllm_v41_cpu/repetition.py`)

Every control looks only at the tokens of the current reply and never penalizes EOS.

- **no-repeat n-gram**: a token that would complete a 4-gram already in the reply is banned.
- **repetition penalty** (CTRL): the logits of tokens already in the reply are divided by
  1.2 (or multiplied by 1.2 when negative).
- **DRY** ("don't repeat yourself"): suppose the last `m` tokens already occurred earlier and
  were followed by token `c`. Then `c`'s logit is lowered by `mult · 1.75^(m−2)` for `m ≥ 2`.
  A match never extends across a newline, `:`, `"` or `*` token. Short echoes cost little, and
  each extra token of a long copy costs 1.75× more.

Each control is a deterministic function of the reply so far, so DSpark speculative decoding
stays exact. The verifier applies the same control at every verified position, and the tests
check that greedy MTP output matches plain greedy decoding under the controls.

## Method (`scripts/decoding_sweep.py`)

- The shipped `chat-dspark` checkpoint generated replies to the 32 fixed chat prompts of
  `scripts/chat_samples.py`, up to 256 new tokens each.
- Decoding was greedy, plus T=0.7 / top-p 0.9 with seeds 0, 1 and 2.
- A reply *loops* when it contains a run of ≥ 8 consecutive tokens that each complete an
  earlier 4-gram.
- *First loop* is the token index where the first loop starts; it is 256 when a reply never
  loops.
- *Self-NLL* is the loss of the reply's tokens under the unconstrained model. Loops score very
  low and word salad very high, so it is a sanity bound, not a quality score.
- The CIs are paired per-prompt bootstrap intervals against no control, with the seeds averaged
  for each prompt first.

## Results

| control | mode | replies that loop | median first loop | distinct-3 | stop rate | self-NLL | Δ first loop (95% CI) |
|---|---|---:|---:|---:|---:|---:|---:|
| none | greedy | 97% | 27 | 0.16 | 3% | 0.38 | – |
| no-repeat 4-gram | greedy | 0% | 256 | 0.91 | 28% | 1.32 | +202 [+182, +219] |
| rep. penalty 1.2 | greedy | 84% | 75 | 0.59 | 16% | 0.72 | +77 [+54, +101] |
| DRY 0.8 | greedy | 62% | 197 | 0.66 | 3% | 0.87 | +144 [+114, +173] |
| **DRY 2.0** | greedy | 34% | 256 | 0.79 | 3% | 1.01 | +180 [+151, +205] |
| DRY 0.8 + rep. 1.1 | greedy | 38% | 256 | 0.77 | 16% | 0.91 | +162 [+134, +189] |
| none | T=0.7 | 94% | 84 | 0.52 | 4% | 0.87 | – |
| no-repeat 4-gram | T=0.7 | 0% | 256 | 0.94 | 26% | 1.62 | +140 [+120, +159] |
| rep. penalty 1.2 | T=0.7 | 30% | 203 | 0.92 | 43% | 1.66 | +92 [+74, +110] |
| DRY 0.8 | T=0.7 | 25% | 256 | 0.84 | 8% | 1.30 | +125 [+108, +141] |
| **DRY 2.0** | T=0.7 | **6%** | 256 | 0.93 | 4% | 1.45 | +150 [+133, +167] |
| DRY 0.8 + rep. 1.1 | T=0.7 | 5% | 256 | 0.94 | 28% | 1.58 | +136 [+117, +153] |

Every control delays the first loop significantly. The deciding factor is what each one breaks:

- **The no-repeat 4-gram ban** removes loops entirely, but it also bans legitimate repeats.
  On the tool-call prompt the greedy reply's `</｜DSML｜ invoke>` closing markup was corrupted
  (`</congress: …`), because protocol markup must repeat by design.
- **The repetition penalty** ends replies early. It penalizes every token already used, so
  EOS, which is exempt, wins sooner: the stop rate rises to 43% at T=0.7 and replies are
  31 tokens shorter (CI [−42, −21]). Combining it with DRY carries the same early-stop cost.
- **DRY 2.0** keeps the length (Δ tokens +0.1, CI [−3.8, +3.8]), the stop rate and the
  tool-call output unchanged. It reaches the loop rate of the strongest settings at T=0.7. It
  also has the lowest self-NLL among the settings that nearly eliminate loops.

**Default:** the chat notebook samples at T=0.7 / top-p 0.9 with DRY 2.0. The breaker token ids
are computed from the tokenizer. Set `REPETITION = None` in the notebook's first cell to see the
raw model.

## Caveats

- Breaking loops does not make this 7-layer model coherent; replies become varied, often
  off-topic text instead of a repeated phrase. Self-NLL rises from 0.87 to 1.45 at T=0.7. Part
  of that rise is expected (fresh text is less predictable than a copy), but the samples should
  be read, not just the metrics.
- Under greedy decoding DRY still loops in 34% of replies. Most of these loops repeat across a
  sequence breaker (e.g. `$$` separated by blank lines), and DRY does not match across
  breakers by design. Sampling closes most of the remaining gap.
- 32 prompts, 256 tokens. Longer replies would give a long copy more chances to recur.

Raw numbers: [JSON](2026-11-10-decoding-repetition-controls.json).
