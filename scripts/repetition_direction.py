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
    labels, groups, reps = [], [], []
    for p_index, prompt in enumerate(prompts):
        ids = tok.encode(render_prompt(prompt), add_special_tokens=False).ids
        session.reset()
        out = session.generate(torch.tensor([ids], device=model.device), max_new_tokens=args.max_new_tokens,
                               eos_token_id=EOS_TOKEN_ID, temperature=0.0)
        reply = out[0, len(ids):].tolist()
        if len(reply) < args.ngram + 2:
            continue
        lab = repeat_labels(reply, args.ngram)
        reps.append(float(lab.mean()))
        full = torch.tensor([ids + reply], device=model.device)
        with FFNRecorder() as rec:
            model.forward(full, compute_indexer=False, sparse_retrieval=False)
        # The activation at absolute position j predicts token j+1. Reply token t sits at
        # len(ids)+t, so it is predicted from position len(ids)+t-1.
        pos = len(ids) - 1 + np.arange(len(reply))
        for layer, value in rec.outputs.items():
            feats[layer].append(value[0, pos].cpu().numpy())
        labels.append(lab)
        groups.append(np.full(len(reply), p_index))
        if (p_index + 1) % 16 == 0:
            print(f"{p_index + 1}/{len(prompts)} prompts, mean repeat rate {np.mean(reps):.3f}", flush=True)
    y = np.concatenate(labels)
    g = np.concatenate(groups)
    held = (g % 4) == 3  # every 4th prompt is held out for measuring separation
    report = {"prompts": len(labels), "positions": int(y.size), "repeat_fraction": float(y.mean()),
              "ngram": args.ngram, "layers": {}}
    directions = {}
    for layer, chunks in feats.items():
        x = np.concatenate(chunks)
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
    c.add_argument("--device", default="cpu")
    e = sub.add_parser("edit")
    e.add_argument("--bundle", type=Path, required=True)
    e.add_argument("--directions", type=Path, required=True)
    e.add_argument("--layers", type=int, nargs="+", required=True)
    e.add_argument("--k", type=int, default=64)
    e.add_argument("--alpha", type=float, default=0.5)
    e.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    (collect if a.cmd == "collect" else edit)(a)


if __name__ == "__main__":
    main()
