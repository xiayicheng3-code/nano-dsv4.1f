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
2. No edited checkpoint is published. The practical fix is at decoding time: sampling plus a DRY
   penalty ([decoding-control sweep](2026-11-10-decoding-repetition-controls.md)).

## Caveats and follow-ups

- 32 prompts, one sampling seed; a larger prompt set could resolve the T=0.7 trend.
- Only difference-of-means directions on FFN outputs were tried. The natural next targets
  are attention heads that attend to the previous occurrence of the current token (induction
  heads), or a direction in the residual stream rather than one FFN output.

Raw numbers: [JSON](2026-11-10-repetition-direction-edit.json).

## Follow-up: a loop-onset direction, tested by inference-time steering

Status: **measured, negative result**. Kaggle T4 x2 notebook
`notebooks/nano_dsv41f_repetition_steering_t4.ipynb` (`scripts/build_repetition_notebook.py`);
raw numbers: [JSON](2026-11-10-loop-onset-steering.json).

If the first direction only reads out "already in a loop", the entry into a loop might still be
steerable. `collect --label onset` marks the 4 tokens of the first repeated 4-gram of every loop
(a run of >= 8 repeat tokens) against fresh tokens at least 8 tokens from any loop, on 200
held-out prompts (1,804 onset vs 4,155 fresh positions). `steer` then edits the FFN outputs at
inference time, with no weight change, and compares against norm-matched random directions.

The model is fairly confident when it enters a loop, between fresh text and loop interiors:

| positions | top-1 probability | top-1 minus top-2 |
|---|---:|---:|
| fresh text | 0.49 | 0.39 |
| loop onset | 0.63 | 0.55 |
| inside a loop | 0.78 | 0.72 |

Held-out separation by layer (Cohen's d): 0.53, 0.67, 0.56, 0.77, **0.97**, 0.57, 0.50; layers 4
and 3 were steered. The scale of `subtract_xβ` is β times the onset-vs-fresh class-mean gap.

| intervention | Δ reply NLL | greedy loops | greedy median tokens to 1st loop | Δ first loop, greedy (95% CI) | T=0.7 median to 1st loop |
|---|---:|---:|---:|---:|---:|
| none | 0 (2.695) | 31/32 | 27 | – | 75 |
| remove r, layers 4+3 | +0.008 | 31/32 | 28 | +3.1 [−4.2, +10.5] | 104 |
| remove r, all layers | +0.068 | 31/32 | 22 | −3.1 [−9.9, +2.9] | 78 |
| subtract ×1 | +0.018 | 31/32 | 31 | +7.3 [+0.2, +14.5] | 62 |
| subtract ×2 | +0.078 | 32/32 | 33 | +4.4 [−2.3, +10.4] | 94 |
| subtract ×4 | +0.332 | 32/32 | 34 | +5.0 [−2.8, +11.8] | 70 |
| random ×1 | +0.016 | 32/32 | 30 | +3.9 [−4.6, +12.4] | 91 |
| random ×2 | +0.068 | 32/32 | 32 | +4.9 [−2.9, +13.3] | 70 |
| random ×4 | +0.312 | 32/32 | 37 | +3.8 [−2.8, +10.3] | 137 |
| random removal, all layers | +0.004 | 31/32 | 30 | +0.5 [−5.3, +6.7] | 72 |

No intervention keeps replies out of loops: greedy replies loop 31–32/32 under every setting. The
only interval excluding zero (subtract ×1, +7 tokens before the first greedy loop) is about the
same size as random-direction perturbations of the same norm (+4 to +5), so it is not specific to
the direction. Sampled (T=0.7) results swing in both directions for real and random edits alike:
once any perturbation changes one sampled token the trajectories diverge, so a single sampling
seed per prompt is too noisy to detect effects of this size.

**Conclusion.** Neither the loop-interior direction nor the loop-onset direction in the FFN
outputs is a causal lever for looping in this model. Entry into a loop is a fairly confident
decision (top-1 0.63), consistent with copying driven by attention: once the current token matches
an earlier one, the model predicts what followed it before. The next experiment should
intervene on attention, e.g. find heads whose attention at onset positions targets the token
after the previous occurrence of the current token (induction heads) and ablate them, with the
same random-head controls and several sampling seeds per prompt.
