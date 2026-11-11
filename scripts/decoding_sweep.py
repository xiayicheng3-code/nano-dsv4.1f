#!/usr/bin/env python3
"""Compare decoding-time repetition controls on the fixed chat prompts of ``chat_samples.py``.

For each control the model decodes the 32 prompts greedily and with T=0.7 / top-p 0.9 under
several seeds. Besides the ``chat_samples`` metrics it reports ``self_nll``: the reply's
teacher-forced loss under the *unconstrained* model. Loops score very low and word salad very
high, so it is a sanity bound rather than a quality score; read the samples too.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

CONFIGS = {
    "none": None,
    "no_repeat_4gram": {"no_repeat_ngram": 4},
    "rep_penalty_1.2": {"repetition_penalty": 1.2},
    "dry_0.8": {"dry_multiplier": 0.8, "dry_base": 1.75, "dry_allowed_length": 2},
    "dry_2.0": {"dry_multiplier": 2.0, "dry_base": 1.75, "dry_allowed_length": 2},
    "dry_0.8+rep_1.1": {"dry_multiplier": 0.8, "dry_base": 1.75, "dry_allowed_length": 2,
                        "repetition_penalty": 1.1},
}
BREAKERS = ("\n", "\n\n", ":", '"', "*")


def breaker_ids(tok) -> list[int]:
    """Single-token encodings of the DRY sequence breakers (multi-token strings are skipped)."""
    ids = []
    for text in BREAKERS:
        enc = tok.encode(text, add_special_tokens=False).ids
        if len(enc) == 1:
            ids.append(enc[0])
    return sorted(set(ids))


@torch.inference_mode()
def self_nll(model, tok, rows) -> float:
    from chat_samples import render_prompt
    total = count = 0.0
    for r in rows:
        prompt = tok.encode(render_prompt(r["prompt"]), add_special_tokens=False).ids
        reply = tok.encode(r["reply"], add_special_tokens=False).ids
        if not reply:
            continue
        ids = torch.tensor([prompt + reply], device=model.device)
        logits, _ = model.forward(ids)
        logp = torch.log_softmax(logits[0, len(prompt) - 1:-1].float(), -1)
        total -= float(logp.gather(-1, ids[0, len(prompt):, None]).sum())
        count += len(reply)
    return total / count if count else float("nan")


def paired(values: dict, name: str, metric: str, mode: str) -> dict:
    """Mean per-prompt difference vs ``none`` (seeds averaged first) with a bootstrap 95% CI."""
    def per_prompt(cfg):
        runs = values[cfg][mode]
        return np.mean([[s[metric] for s in run["samples"]] for run in runs], 0)
    d = per_prompt(name) - per_prompt("none")
    rng = np.random.default_rng(0)
    boot = [rng.choice(d, d.size).mean() for _ in range(5000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean": float(d.mean()), "ci95": [float(lo), float(hi)]}


def main():
    from tokenizers import Tokenizer
    from chat_samples import PROMPTS, sample_bundle
    from nano_dsv41f.vllm_v41_cpu import NanoDeepseekV41CPU

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    model = NanoDeepseekV41CPU.from_pretrained(a.bundle, device=a.device, dtype=torch.float32)
    tok = Tokenizer.from_file(str(a.bundle / "tokenizer.json"))
    breakers = breaker_ids(tok)
    prompts = PROMPTS[:a.limit] if a.limit else PROMPTS
    modes = {"greedy": [(0.0, 1.0, 0)], "t07": [(0.7, 0.9, s) for s in a.seeds]}
    results = {"breakers": breakers, "max_new_tokens": a.max_new_tokens, "configs": {}}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    for name, cfg in CONFIGS.items():
        if a.only and name not in a.only:
            continue
        control = None if cfg is None else dict(cfg, dry_sequence_breakers=breakers if "dry_multiplier" in cfg else [])
        entry = {"control": control}
        for mode, settings in modes.items():
            runs = []
            for temp, top_p, seed in settings:
                res = sample_bundle(a.bundle, prompts, device=a.device, max_new_tokens=a.max_new_tokens,
                                    temperature=temp, top_p=top_p, seed=seed, model=model, repetition=control)
                res["summary"]["self_nll"] = self_nll(model, tok, res["samples"])
                runs.append(res)
                print(name, mode, seed, json.dumps({k: round(v, 3) for k, v in res["summary"].items()}), flush=True)
            entry[mode] = runs
        results["configs"][name] = entry
        a.output.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    summary = {}
    for name, entry in results["configs"].items():
        row = {}
        for mode in modes:
            keys = entry[mode][0]["summary"]
            row[mode] = {k: float(np.mean([r["summary"][k] for r in entry[mode]])) for k in keys}
            if name != "none" and "none" in results["configs"]:
                row[mode]["paired_vs_none"] = {m: paired(results["configs"], name, m, mode)
                                               for m in ("first_loop", "distinct3", "tokens")}
        summary[name] = row
    results["summary"] = summary
    a.output.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
