# Training-free repetition-direction edit

Status: **measured, negative result**. The chat baseline stays the shipped checkpoint.

## Question

The chat model loops: 79% of greedy reply tokens complete a 4-gram already present in the
same reply. Following the weight-space loop-removal idea in *Sparks of Weight-Space
Programming* (Takehi et al.), can removing one FFN-written "repetition" direction from a few
neurons reduce looping **without any training**?

## Method (`scripts/repetition_direction.py`)

1. **collect**: greedy-decode 96 smol-smoltalk test prompts (192 tokens), re-run them
   teacher-forced and record every layer's MoE output. A position is *repeat* if the token
   it predicts completes an earlier 4-gram of the same reply, *fresh* otherwise (both classes
   are model text). Direction r = difference of class means; separation is measured on every
   4th prompt held out.
2. **edit**: in chosen layers, take the k FFN output neurons (rows of routed and shared `w2`,
   6,272 per layer) with the largest |w·r| and set `w <- w - alpha (w·r) r`. The bundle is
   re-hashed and records the edit.
3. Evaluate on the 32 fixed chat prompts (`scripts/chat_samples.py`, greedy and T=0.7/top-p 0.9,
   same seeds per prompt) and teacher-forced NLL of 200 held-out reference replies (`nll`).

Held-out separation (Cohen's d) by layer: 0.48, 0.32, 0.44, 0.70, **0.84**, 0.47, **0.79**.
Layers 4 and 6 were edited.

## Results

| edit (layers 4+6) | weights changed | greedy repeat4 | greedy distinct2 | T=0.7 repeat4 | T=0.7 distinct2 | reply NLL |
|---|---:|---:|---:|---:|---:|---:|
| none | – | 0.797 | 0.181 | 0.420 | 0.506 | 2.6259 |
| k=64, α=0.5 | 0.06% | 0.835 | 0.142 | 0.361 | 0.550 | 2.6271 |
| k=64, α=1.0 | 0.06% | 0.813 | 0.182 | 0.388 | 0.499 | 2.6317 |
| k=512, α=1.0 | 0.47% | 0.794 | 0.201 | 0.350 | 0.592 | 2.6362 |

Paired per-prompt bootstrap (32 prompts): every 95% interval for the change in repeat4 or
distinct2 includes zero, except k=64/α=0.5 under greedy decoding, which is slightly *more*
repetitive (+0.038, CI [0.003, 0.087]). The T=0.7 improvements of the k=512 edit
(repeat4 −0.07, CI [−0.18, 0.04]; distinct2 +0.09, CI [−0.01, 0.19]) are suggestive but not
significant. Stop rate stays at 0–9% for all variants. The NLL cost is small (+0.01 nats at
k=512).

**The edit did remove the direction.** Teacher-forcing the baseline's greedy replies, the
mean projection of the FFN output on r at repeat minus fresh positions shrinks:

| | layer 4 gap | layer 6 gap |
|---|---:|---:|
| none | 0.200 | 0.319 |
| k=64, α=1.0 | 0.130 | 0.170 |
| k=512, α=1.0 | 0.065 | 0.081 |

## Conclusions

1. The FFN repetition direction is real and separable (d ≈ 0.8), and a 0.5% weight edit
   removes most of it, but looping barely changes. At this scale the direction is a
   **readout** of "already in a loop", not its cause. Looping is the model's default mode
   (79% of greedy tokens), most plausibly driven by attention copying earlier context,
   which an FFN-output edit does not touch.
2. No edited checkpoint is published. Sampling (T=0.7, top-p 0.9) remains the practical fix.

## Caveats and follow-ups

- 32 prompts, one sampling seed; a larger prompt set could resolve the T=0.7 trend.
- Only difference-of-means directions on FFN outputs were tried. The natural next targets
  are attention heads that attend to the previous occurrence of the current token (induction
  heads), or a direction in the residual stream rather than one FFN output.

Raw numbers: [JSON](2026-11-10-repetition-direction-edit.json).
