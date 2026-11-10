#!/usr/bin/env python3
"""Find a "repetition" direction in FFN outputs and remove it from a few neurons, training-free.

Inspired by the Qwen3.5-0.8B loop-removal experiment in *Sparks of Weight-Space Programming*
(Takehi et al.), which subtracted one MLP-written direction from 64 output columns in each of
two layers. This script implements a standard, inspectable version of that idea:

``collect``
    Greedy-decode chat prompts (which loop on this model), then re-run each prompt + reply
    teacher-forced and record every layer's FFN (MoE) output. A reply position is *repeat* when
    the token it predicts completes an n-gram already present earlier in the same reply, and
    *fresh* otherwise. Both classes come from the model's own replies, so the contrast is
    "about to repeat" rather than "model text vs human text". Per layer, the direction is the
    difference of class means; its separation (Cohen's d) is measured on held-out prompts.

``edit``
    For chosen layers, take the ``k`` FFN output neurons (rows of the routed experts' and the
    shared expert's ``w2``) whose write vectors have the largest component along the unit
    direction r, and remove a fraction alpha of that component: ``w <- w - alpha (w . r) r``.
    Everything else is byte-identical; the bundle manifest is re-hashed and records the edit.

``collect --label onset``
    The default labels mark every repeat-completing token, so 3/4 of positive positions sit deep
    inside loops. ``onset`` instead marks the tokens of the first repeated n-gram of each loop (a
    run of >= ``min_run`` repeat tokens) against fresh tokens far from any loop, and reports the
    model's top-1 probability and margin at those positions.

``steer``
    Causality test before any weight edit: apply ablation or subtraction of the direction to the
    FFN outputs at inference time (plus a norm-matched random-direction control), then sample the
    fixed chat prompts and score held-out replies under each intervention.

``nll``
    Teacher-forced loss of reference smol-smoltalk replies for several bundles on the same
    held-out pairs (disjoint from the collect prompts), to check what an edit costs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch


def repeat_labels(tokens: list[int], n: int) -> np.ndarray:
    """labels[t] = 1 if tokens[t] completes an n-gram that occurred earlier in the sequence."""
    seen, labels = set(), np.zeros(len(tokens), dtype=bool)
    for t in range(len(tokens)):
        if t + 1 >= n:
            gram = tuple(tokens[t + 1 - n:t + 1])
            labels[t] = gram in seen
            seen.add(gram)
    return labels


def load_prompts(count: int, skip: int, seed: int) -> list[str]:
    """First user turns from smol-smoltalk's test split (disjoint from chat_samples prompts)."""
    import datasets
    ds = datasets.load_dataset("HuggingFaceTB/smol-smoltalk", split="test", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=4096)
    prompts = []
    for i, row in enumerate(ds):
        if i < skip:
            continue
        user = next((m["content"] for m in row["messages"] if m["role"] == "user"), "")
        if 10 <= len(user) <= 400:
            prompts.append(user)
        if len(prompts) >= count:
            break
    return prompts


class FFNRecorder:
    """Wrap the runtime's apply_moe to keep each layer's FFN output for the last forward."""

    def __init__(self):
        from nano_dsv41f.vllm_v41_cpu import model as runtime_model
        self.module, self.original, self.outputs = runtime_model, runtime_model.apply_moe, {}

    def __enter__(self):
        def wrapped(model, x, layer_id=None, **kw):
            out = self.original(model, x, layer_id, **kw)
            if layer_id is not None:
                self.outputs[layer_id] = out.detach().float()
            return out
        self.module.apply_moe = wrapped
        return self

    def __exit__(self, *exc):
        self.module.apply_moe = self.original


def onset_labels(tokens: list[int], n: int, min_run: int, margin: int = 8) -> np.ndarray:
    """1 = token belongs to the first repeated n-gram of a loop (the copy is starting),
    0 = fresh token at least ``margin`` tokens away from any loop, -1 = excluded."""
    from chat_samples import loop_runs
    rep = repeat_labels(tokens, n)
    runs = loop_runs(rep, min_run)
    labels = np.full(len(tokens), -1, dtype=np.int8)
    near = np.zeros(len(tokens), dtype=bool)
    for start, stop in runs:
        near[max(0, start - n + 1 - margin):min(len(tokens), stop + margin)] = True
    labels[~rep & ~near] = 0
    for start, _ in runs:
        labels[max(0, start - n + 1):start + 1] = 1
    return labels


class FFNSteer:
    """Edit every layer's FFN output at inference time along unit directions r_l.

    ``ablate``: out <- out - alpha (out . r) r.  ``subtract``: out <- out - beta * scale_l * r,
    with scale_l the class-mean gap measured by ``collect``.
    """

    def __init__(self, directions: dict[int, torch.Tensor], *, mode: str, strength: float,
                 scales: dict[int, float] | None = None):
        from nano_dsv41f.vllm_v41_cpu import model as runtime_model
        self.module, self.original = runtime_model, runtime_model.apply_moe
        self.directions, self.mode, self.strength, self.scales = directions, mode, strength, scales or {}

    def __enter__(self):
        def wrapped(model, x, layer_id=None, **kw):
            out = self.original(model, x, layer_id, **kw)
            r = self.directions.get(layer_id)
            if r is None:
                return out
            r = r.to(out.device, out.dtype)
            if self.mode == "ablate":
                return out - self.strength * (out @ r)[..., None] * r
            return out - self.strength * self.scales[layer_id] * r
        self.module.apply_moe = wrapped
        return self

    def __exit__(self, *exc):
        self.module.apply_moe = self.original


@torch.inference_mode()
def collect(args):
    from tokenizers import Tokenizer
    from chat_samples import render_prompt
    from nano_dsv41f.chat_protocol import EOS_TOKEN_ID
    from nano_dsv41f.vllm_v41_cpu import NanoDeepseekV41CPU
    from nano_dsv41f.vllm_v41_cpu.session import InferenceSession

    model = NanoDeepseekV41CPU.from_pretrained(args.bundle, device=args.device, dtype=torch.float32)
    tok = Tokenizer.from_file(str(args.bundle / "tokenizer.json"))
    prompts = load_prompts(args.prompts, args.skip, args.seed)
    session = InferenceSession(model)
    feats = {i: [] for i in range(model.config.n_layers)}
    labels, groups, reps, conf = [], [], [], []
    for p_index, prompt in enumerate(prompts):
        ids = tok.encode(render_prompt(prompt), add_special_tokens=False).ids
        session.reset()
        out = session.generate(torch.tensor([ids], device=model.device), max_new_tokens=args.max_new_tokens,
                               eos_token_id=EOS_TOKEN_ID, temperature=0.0)
        reply = out[0, len(ids):].tolist()
        if len(reply) < args.ngram + 2:
            continue
        rep = repeat_labels(reply, args.ngram)
        lab = rep.astype(np.int8) if args.label == "repeat" else onset_labels(reply, args.ngram, args.min_run)
        reps.append(float(rep.mean()))
        full = torch.tensor([ids + reply], device=model.device)
        with FFNRecorder() as rec:
            logits, _ = model.forward(full, compute_indexer=False, sparse_retrieval=False)
        # The activation at absolute position j predicts token j+1. Reply token t sits at
        # len(ids)+t, so it is predicted from position len(ids)+t-1.
        pos = len(ids) - 1 + np.arange(len(reply))
        top2 = torch.softmax(logits[0, pos].float(), -1).topk(2, -1).values.cpu().numpy()
        conf.append(np.stack([top2[:, 0], top2[:, 0] - top2[:, 1], rep], 1))
        for layer, value in rec.outputs.items():
            feats[layer].append(value[0, pos].cpu().numpy())
        labels.append(lab)
        groups.append(np.full(len(reply), p_index))
        if (p_index + 1) % 16 == 0:
            print(f"{p_index + 1}/{len(prompts)} prompts, mean repeat rate {np.mean(reps):.3f}", flush=True)
    y_all = np.concatenate(labels)
    g_all = np.concatenate(groups)
    c_all = np.concatenate(conf)
    keep = y_all >= 0
    y, g = y_all[keep].astype(bool), g_all[keep]
    held = (g % 4) == 3  # every 4th prompt is held out for measuring separation
    report = {"label": args.label, "prompts": len(labels), "prompts_requested": args.prompts, "positions": int(y.size),
              "positive_fraction": float(y.mean()), "positive_positions": int(y.sum()),
              "repeat_fraction": float(np.mean(reps)), "ngram": args.ngram, "min_run": args.min_run,
              "layers": {}}
    # How confident is the model where it enters, continues or avoids a loop? (top-1 prob, top1-top2)
    classes = {"positive": y_all == 1, "negative": y_all == 0, "repeat_any": c_all[:, 2] == 1}
    report["confidence"] = {k: {"top1_prob": float(c_all[m, 0].mean()), "top1_margin": float(c_all[m, 1].mean()),
                                "positions": int(m.sum())} for k, m in classes.items() if m.any()}
    directions = {}
    for layer, chunks in feats.items():
        x = np.concatenate(chunks)[keep]
        mu_pos, mu_neg = x[~held & y].mean(0), x[~held & ~y].mean(0)
        r = mu_pos - mu_neg
        r_unit = r / np.linalg.norm(r)
        proj = x[held] @ r_unit
        a, b = proj[y[held]], proj[~y[held]]
        pooled = np.sqrt((a.var() + b.var()) / 2)
        d = float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0
        directions[f"layer{layer}"] = r_unit.astype(np.float32)
        report["layers"][layer] = {"cohens_d_heldout": round(d, 4), "raw_norm": float(np.linalg.norm(r)),
                                   "mean_ffn_norm": float(np.linalg.norm(x, axis=1).mean())}
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(args.output / "directions.npz", **directions)
    (args.output / "collect_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def load_pairs(count: int, seed: int, exclude: set[str]) -> list[tuple[str, str]]:
    """(first user turn, first assistant reply) pairs from smol-smoltalk test, excluding prompts."""
    import datasets
    ds = datasets.load_dataset("HuggingFaceTB/smol-smoltalk", split="test", streaming=True)
    pairs = []
    for row in ds.shuffle(seed=seed, buffer_size=4096):
        msgs = row["messages"]
        if len(msgs) >= 2 and msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant":
            user, reply = msgs[0]["content"], msgs[1]["content"]
            if user not in exclude and 10 <= len(user) <= 400 and 20 <= len(reply) <= 2000:
                pairs.append((user, reply))
        if len(pairs) >= count:
            return pairs
    return pairs


@torch.inference_mode()
def reply_nll(model, tok, pairs) -> dict:
    from chat_samples import render_prompt
    from nano_dsv41f.chat_protocol import EOS_TOKEN_ID
    total, count = 0.0, 0
    for user, reply in pairs:
        prompt = tok.encode(render_prompt(user), add_special_tokens=False).ids
        target = tok.encode(reply, add_special_tokens=False).ids + [EOS_TOKEN_ID]
        ids = torch.tensor([prompt + target], device=model.device)
        logits, _ = model.forward(ids)
        logp = torch.log_softmax(logits[0, len(prompt) - 1:-1].float(), -1)
        total -= float(logp.gather(-1, ids[0, len(prompt):, None]).sum())
        count += len(target)
    return {"nll": total / count, "tokens": count, "pairs": len(pairs)}


@torch.inference_mode()
def nll(args):
    """Teacher-forced NLL of reference replies (+EOS) for each bundle on identical held-out pairs."""
    from tokenizers import Tokenizer
    from nano_dsv41f.vllm_v41_cpu import NanoDeepseekV41CPU

    exclude = set(load_prompts(args.exclude_prompts, 0, 0))
    pairs = load_pairs(args.pairs, args.seed, exclude)
    results = {}
    for spec in args.bundle:
        name, path = spec.split("=", 1)
        model = NanoDeepseekV41CPU.from_pretrained(Path(path), device=args.device, dtype=torch.float32)
        tok = Tokenizer.from_file(str(Path(path) / "tokenizer.json"))
        results[name] = reply_nll(model, tok, pairs)
        print(name, json.dumps(results[name]), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))


def steer_configs(report: dict, layers: list[int]) -> list[dict]:
    """Inference-time interventions compared by ``steer``; ``random_*`` are matched random-direction controls."""
    return ([{"name": "none"}]
            + [{"name": f"ablate_L{'-'.join(map(str, layers))}", "mode": "ablate", "strength": 1.0, "layers": layers},
               {"name": "ablate_all", "mode": "ablate", "strength": 1.0, "layers": sorted(map(int, report["layers"]))}]
            + [{"name": f"subtract_x{b:g}", "mode": "subtract", "strength": b, "layers": layers} for b in (1, 2, 4)]
            + [{"name": f"random_subtract_x{b:g}", "mode": "subtract", "strength": b, "layers": layers, "random": True}
               for b in (1, 2, 4)]
            + [{"name": "random_ablate_all", "mode": "ablate", "strength": 1.0, "random": True,
                "layers": sorted(map(int, report["layers"]))}])


@torch.inference_mode()
def steer(args):
    """Causality test: apply each intervention at inference time, then sample and score."""
    from tokenizers import Tokenizer
    from chat_samples import PROMPTS, sample_bundle
    from nano_dsv41f.vllm_v41_cpu import NanoDeepseekV41CPU

    report = json.loads((args.directions / "collect_report.json").read_text())
    raw = np.load(args.directions / "directions.npz")
    ranked = sorted(report["layers"], key=lambda l: -report["layers"][l]["cohens_d_heldout"])
    layers = args.layers or [int(l) for l in ranked[:2]]
    configs = [c for c in steer_configs(report, layers) if not args.only or c["name"] in args.only]
    model = NanoDeepseekV41CPU.from_pretrained(args.bundle, device=args.device, dtype=torch.float32)
    tok = Tokenizer.from_file(str(args.bundle / "tokenizer.json"))
    pairs = load_pairs(args.pairs, args.nll_seed, set(load_prompts(report["prompts_requested"], 0, 0))) \
        if args.pairs else []
    prompts = PROMPTS[:args.limit] if args.limit else PROMPTS
    args.output.mkdir(parents=True, exist_ok=True)
    results = {"layers": layers, "directions_sha256": sha256(args.directions / "directions.npz"), "configs": {}}
    for cfg in configs:
        dirs, scales = {}, {}
        for l in cfg.get("layers", []):
            r = raw[f"layer{l}"]
            if cfg.get("random"):
                r = np.random.default_rng(1000 + l).standard_normal(r.shape).astype(np.float32)
                r /= np.linalg.norm(r)
            dirs[l] = torch.from_numpy(r)
            scales[l] = report["layers"][str(l)]["raw_norm"]
        steer_ctx = FFNSteer(dirs, mode=cfg.get("mode", "ablate"), strength=cfg.get("strength", 0.0), scales=scales)
        row = {"config": cfg}
        with steer_ctx:
            for mode, temp, top_p in (("greedy", 0.0, 1.0), ("t07", 0.7, 0.9)):
                res = sample_bundle(args.bundle, prompts, device=args.device, max_new_tokens=args.max_new_tokens,
                                    temperature=temp, top_p=top_p, seed=0, model=model)
                row[mode] = res
                print(cfg["name"], mode, json.dumps(res["summary"]), flush=True)
            if pairs:
                row["nll"] = reply_nll(model, tok, pairs)
                print(cfg["name"], "nll", json.dumps(row["nll"]), flush=True)
        results["configs"][cfg["name"]] = row
        (args.output / "steer_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def edit(args):
    from safetensors.torch import load_file, save_file
    from nano_dsv41f.portable_bundle import verify_portable_bundle

    verify_portable_bundle(args.bundle)
    if args.output.exists():
        raise FileExistsError(f"Use a fresh output directory: {args.output}")
    directions = np.load(args.directions)
    tensors = load_file(str(args.bundle / "model.safetensors"))
    record = {"method": "remove fraction alpha of a repetition direction from top-k FFN output neurons",
              "alpha": args.alpha, "k": args.k, "layers": {}, "directions_sha256": sha256(args.directions)}
    changed = 0
    for layer in args.layers:
        r = torch.from_numpy(directions[f"layer{layer}"]).float()
        names = [f"nano.blocks.{layer}.moe.experts.w2", f"nano.blocks.{layer}.moe.shared.w2"]
        rows = [tensors[names[0]].float().reshape(-1, r.numel()), tensors[names[1]].float()]
        stacked = torch.cat(rows)  # [48*128 + 128, d_model]: one write vector per neuron
        comp = stacked @ r
        top = torch.topk(comp.abs(), args.k).indices
        edited = stacked.clone()
        edited[top] -= args.alpha * comp[top, None] * r[None]
        n_expert = rows[0].shape[0]
        for name, block, shape in ((names[0], edited[:n_expert], tensors[names[0]].shape),
                                   (names[1], edited[n_expert:], tensors[names[1]].shape)):
            tensors[name] = block.reshape(shape).to(tensors[name].dtype).contiguous()
        changed += args.k * r.numel()
        record["layers"][layer] = {"neurons": [int(i) for i in top], "shared_neurons": int((top >= n_expert).sum()),
                                   "component_before_mean_abs": float(comp[top].abs().mean()),
                                   "component_after_mean_abs": float((edited[top] @ r).abs().mean())}
    record["changed_weights"] = changed
    record["changed_fraction"] = changed / sum(t.numel() for t in tensors.values())
    pending = args.output.with_name(args.output.name + ".pending")
    shutil.rmtree(pending, ignore_errors=True)
    shutil.copytree(args.bundle, pending)
    save_file(tensors, str(pending / "model.safetensors"))
    (pending / "weight_edit.json").write_text(json.dumps(record, indent=2))
    manifest = json.loads((pending / "export_manifest.json").read_text())
    manifest["weight_edit"] = {k: v for k, v in record.items() if k != "layers"} | {"layers": list(args.layers)}
    manifest["source_bundle_manifest_sha256"] = sha256(args.bundle / "export_manifest.json")
    for name in list(manifest["files"]) + ["weight_edit.json"]:
        path = pending / name
        manifest["files"][name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    (pending / "export_manifest.json").write_text(json.dumps(manifest, indent=2))
    verify_portable_bundle(pending)
    pending.rename(args.output)
    print(json.dumps({k: v for k, v in record.items() if k != "layers"}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--bundle", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--prompts", type=int, default=96)
    c.add_argument("--skip", type=int, default=0)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--max-new-tokens", type=int, default=192)
    c.add_argument("--ngram", type=int, default=4)
    c.add_argument("--label", choices=("repeat", "onset"), default="repeat",
                   help="repeat: any repeat-completing token vs fresh; onset: loop entry vs fresh text")
    c.add_argument("--min-run", type=int, default=8, help="onset: repeat run length that counts as a loop")
    c.add_argument("--device", default="cpu")
    e = sub.add_parser("edit")
    e.add_argument("--bundle", type=Path, required=True)
    e.add_argument("--directions", type=Path, required=True)
    e.add_argument("--layers", type=int, nargs="+", required=True)
    e.add_argument("--k", type=int, default=64)
    e.add_argument("--alpha", type=float, default=0.5)
    e.add_argument("--output", type=Path, required=True)
    n = sub.add_parser("nll")
    n.add_argument("--bundle", action="append", required=True, help="NAME=PATH, repeatable")
    n.add_argument("--pairs", type=int, default=200)
    n.add_argument("--seed", type=int, default=1)
    n.add_argument("--exclude-prompts", type=int, default=96, help="collect prompts to exclude")
    n.add_argument("--device", default="cpu")
    n.add_argument("--output", type=Path)
    s = sub.add_parser("steer")
    s.add_argument("--bundle", type=Path, required=True)
    s.add_argument("--directions", type=Path, required=True, help="collect output directory")
    s.add_argument("--layers", type=int, nargs="*", default=None, help="default: top-2 by held-out d")
    s.add_argument("--only", nargs="*", default=None, help="subset of config names")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--max-new-tokens", type=int, default=192)
    s.add_argument("--pairs", type=int, default=200)
    s.add_argument("--nll-seed", type=int, default=1)
    s.add_argument("--device", default="cpu")
    s.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    {"collect": collect, "edit": edit, "nll": nll, "steer": steer}[a.cmd](a)


if __name__ == "__main__":
    main()
