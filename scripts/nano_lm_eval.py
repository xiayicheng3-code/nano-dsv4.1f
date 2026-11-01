#!/usr/bin/env python3
"""Standard zero-shot benchmarks for nano-dsv4.1f through EleutherAI lm-evaluation-harness.

The module adapts the portable Torch runtime (``nano_dsv41f.vllm_v41_cpu``) to the
harness ``TemplateLM`` interface, so nano checkpoints are scored by the exact same task
definitions, prompts and metrics as the Hugging Face baselines.

Design notes
------------
* Every pretraining document was tokenized as ``[BOS, text..., EOS]``. Contexts are
  therefore prefixed with BOS (id 0), and BOS is the prefix for empty contexts and
  rolling-window perplexity.
* Requests are right-padded into batches. The model is causal everywhere (local SWA,
  compressed global KV, indexer selection, Engram n-grams, per-token MoE routing), so
  padding after a sequence never changes its logits. ``check_padding_invariance``
  verifies this on the loaded weights before any task runs.
* The reference runtime's MoE gathers whole ``[D, F]`` expert matrices per token,
  which needs ~3 MB of temporary memory per token in FP32. ``install_grouped_moe``
  swaps in a grouped dispatch (one GEMM per selected expert) with identical routing
  math; ``check_grouped_moe`` compares it with the reference implementation.
* Benchmarks are short, so the dense backbone path is used (``sparse_retrieval=False``).
  With a 512-entry Top-K and ``<= 2048``-token windows the indexer selects every legal
  compressed entry, so sparse and dense retrieval coincide for these tasks.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Iterable

import torch
import torch.nn.functional as F

BOS_TOKEN_ID = 0
EOS_TOKEN_ID = 1
PAD_TOKEN_ID = 2

DEFAULT_TASKS = (
    "lambada_openai",
    "hellaswag",
    "piqa",
    "arc_easy",
    "arc_challenge",
    "winogrande",
    "sciq",
    "openbookqa",
    "wikitext",
)


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------

def grouped_apply_moe(model: Any, x: torch.Tensor, layer_id: int | None = None, *,
                      prefix: str | None = None, top_k: int | None = None) -> torch.Tensor:
    """Drop-in replacement for ``model_ops.apply_moe`` with grouped expert GEMMs.

    Router scoring, selection bias, normalization, SwiGLU clipping and the shared
    expert follow the reference function line for line; only the routed products
    are evaluated per expert instead of per token.
    """
    prefix = prefix or f"blocks.{layer_id}.moe"
    top_k = model.config.experts_per_token if top_k is None else top_k
    logits = torch.matmul(x.float(), model._w(f"{prefix}.router_weight").float())
    raw = torch.sqrt(F.softplus(logits))
    selection = raw + model._w(f"{prefix}.router_bias").float()
    indices = torch.topk(selection, k=top_k, dim=-1).indices
    weights = raw.gather(-1, indices)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(model.config.route_eps)
    weights = weights * model.config.route_scale

    w1_all = model._w(f"{prefix}.experts.w1")
    w2_all = model._w(f"{prefix}.experts.w2")
    w3_all = model._w(f"{prefix}.experts.w3")
    limit = model.config.swiglu_limit
    lead_shape = x.shape[:-1]
    x_flat = x.reshape(-1, x.shape[-1]).to(w1_all.dtype)
    idx_flat = indices.reshape(-1, top_k)
    w_flat = weights.reshape(-1, top_k)
    routed = torch.zeros(x_flat.shape[0], w2_all.shape[-1], device=x.device, dtype=w2_all.dtype)

    order = torch.argsort(idx_flat.reshape(-1), stable=True)
    counts = torch.bincount(idx_flat.reshape(-1), minlength=w1_all.shape[0]).tolist()
    token_of = order // top_k
    slot_weight = w_flat.reshape(-1)[order]
    start = 0
    for expert, count in enumerate(counts):
        if count == 0:
            continue
        rows = token_of[start:start + count]
        xe = x_flat.index_select(0, rows)
        gate = xe @ w1_all[expert]
        up = xe @ w3_all[expert]
        if limit > 0:
            gate = torch.clamp(gate, max=limit)
            up = up.clamp(-limit, limit)
        out = (F.silu(gate) * up) @ w2_all[expert]
        routed.index_add_(0, rows, slot_weight[start:start + count].to(out.dtype).unsqueeze(-1) * out)
        start += count
    routed = routed.reshape(*lead_shape, -1)

    shared_w1 = model._w(f"{prefix}.shared.w1")
    shared_w2 = model._w(f"{prefix}.shared.w2")
    shared_w3 = model._w(f"{prefix}.shared.w3")
    shared_x = x.to(shared_w1.dtype)
    shared_gate = torch.matmul(shared_x, shared_w1)
    shared_up = torch.matmul(shared_x, shared_w3)
    if limit > 0:
        shared_gate = torch.clamp(shared_gate, max=limit)
        shared_up = shared_up.clamp(-limit, limit)
    shared = torch.matmul(F.silu(shared_gate) * shared_up, shared_w2)
    return (routed.to(x.dtype) + shared.to(x.dtype)).to(x.dtype)


def install_grouped_moe() -> Any:
    """Route backbone MoE calls through ``grouped_apply_moe``; return the original."""
    from nano_dsv41f.vllm_v41_cpu import model as runtime_model

    original = runtime_model.apply_moe
    if original is not grouped_apply_moe:
        runtime_model.apply_moe = grouped_apply_moe
    return original


def check_grouped_moe(model: Any, reference_fn: Any, *, tokens: int = 96, seed: int = 0) -> float:
    """Max abs difference between grouped and reference MoE on random activations."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    dim = model.config.d_model
    worst = 0.0
    for spec in model.specs:
        x = torch.randn(2, tokens // 2, dim, generator=gen).to(model.device, model.dtype)
        ref = reference_fn(model, x, spec.layer_id)
        new = grouped_apply_moe(model, x, spec.layer_id)
        worst = max(worst, float((ref - new).abs().max()))
    return worst


def check_padding_invariance(model: Any, *, seed: int = 0) -> float:
    """Logit difference between a sequence run alone and right-padded in a batch."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    vocab = model.config.vocab_size
    lengths = (37, 300, 515)
    seqs = [torch.cat([torch.tensor([BOS_TOKEN_ID]),
                       torch.randint(10, vocab, (n - 1,), generator=gen)]) for n in lengths]
    width = max(lengths) + 21
    batch = torch.full((len(seqs), width), PAD_TOKEN_ID, dtype=torch.long)
    for row, seq in enumerate(seqs):
        batch[row, :len(seq)] = seq
    batched, _ = model.forward(batch, compute_indexer=False, sparse_retrieval=False)
    worst = 0.0
    for row, seq in enumerate(seqs):
        alone, _ = model.forward(seq[None], compute_indexer=False, sparse_retrieval=False)
        diff = (alone[0].float() - batched[row, :len(seq)].float()).abs().max()
        worst = max(worst, float(diff))
    return worst


def count_parameters(model: Any) -> dict[str, int]:
    """Total, DSpark-only and per-token active parameter counts of a loaded nano model."""
    cfg = model.config
    total = dspark = expert = 0
    for key, value in model.weights.items():
        n = int(value.numel())
        total += n
        if ".dspark." in f".{key}." or key.startswith("nano.dspark"):
            dspark += n
        elif ".moe.experts." in key:
            expert += n
    backbone = total - dspark
    active_expert = expert * cfg.experts_per_token // cfg.n_experts
    engram = sum(int(v.numel()) for k, v in model.weights.items() if ".engram.table" in k)
    embed = int(model.weights["nano.embed"].numel()) if "nano.embed" in model.weights else 0
    head = int(model.weights["nano.lm_head"].numel()) if "nano.lm_head" in model.weights else 0
    return {
        "total_including_dspark": total,
        "backbone_total": backbone,
        "backbone_active_per_token": backbone - expert + active_expert,
        # Pythia convention: exclude input embedding and unembedding (plus Engram lookup table).
        "backbone_non_embedding": backbone - embed - head - engram,
        "backbone_active_non_embedding": backbone - expert + active_expert - embed - head - engram,
        "embedding": embed,
        "lm_head": head,
        "engram_table": engram,
    }


# ---------------------------------------------------------------------------
# lm-evaluation-harness adapter
# ---------------------------------------------------------------------------

def _make_lm_class():
    from lm_eval.api.model import TemplateLM
    from lm_eval import utils as lm_utils
    from tqdm.auto import tqdm

    class NanoDSV41FLM(TemplateLM):
        """``TemplateLM`` wrapper around ``NanoDeepseekV41CPU`` (any Torch device)."""

        def __init__(self, model: Any, tokenizer: Any, *, max_length: int = 2048,
                     token_budget: int = 16384, max_batch: int = 64) -> None:
            super().__init__()
            self.model = model
            self.tok = tokenizer
            self._max_length = max_length
            self.token_budget = token_budget
            self.max_batch = max_batch
            self.forward_calls = 0
            self.scored_tokens = 0

        # -- tokenization -------------------------------------------------
        @property
        def eot_token_id(self) -> int:
            return EOS_TOKEN_ID

        @property
        def prefix_token_id(self) -> int:
            return BOS_TOKEN_ID

        @property
        def max_length(self) -> int:
            return self._max_length

        def tok_encode(self, string: str, add_special_tokens: bool | None = None, **kwargs) -> list[int]:
            ids = self.tok.encode(string, add_special_tokens=False).ids
            if add_special_tokens is False:
                return ids
            return [BOS_TOKEN_ID] + ids

        def tok_decode(self, tokens: Iterable[int], **kwargs) -> str:
            return self.tok.decode(list(tokens))

        # -- scoring -------------------------------------------------------
        def _logits(self, batch: torch.Tensor) -> torch.Tensor:
            logits, _ = self.model.forward(batch, compute_indexer=False, sparse_retrieval=False)
            self.forward_calls += 1
            return logits

        def _score_windows(self, windows: list[tuple[list[int], int]]) -> list[tuple[float, bool]]:
            """Score ``(tokens, n_target)``: log-prob of the final ``n_target`` tokens."""
            results: list[tuple[float, bool] | None] = [None] * len(windows)
            order = sorted(range(len(windows)), key=lambda i: -len(windows[i][0]))
            pos = 0
            while pos < len(order):
                width = len(windows[order[pos]][0]) - 1
                rows = max(1, min(self.max_batch, self.token_budget // max(width, 1)))
                chunk = order[pos:pos + rows]
                pos += len(chunk)
                batch = torch.full((len(chunk), width), PAD_TOKEN_ID, dtype=torch.long)
                for r, i in enumerate(chunk):
                    tokens = windows[i][0][:-1]
                    batch[r, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
                logits = self._logits(batch)
                for r, i in enumerate(chunk):
                    tokens, n_target = windows[i]
                    inp_len = len(tokens) - 1
                    span = logits[r, inp_len - n_target:inp_len].float()
                    logp = F.log_softmax(span, dim=-1)
                    target = torch.tensor(tokens[-n_target:], device=logp.device)
                    score = float(logp.gather(-1, target[:, None]).sum())
                    greedy = bool((logp.argmax(-1) == target).all())
                    results[i] = (score, greedy)
                    self.scored_tokens += n_target
            return results  # type: ignore[return-value]

        def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False, **kwargs):
            windows = []
            for _, context_enc, continuation_enc in requests:
                whole = (context_enc + continuation_enc)[-(self.max_length + 1):]
                windows.append((whole, len(continuation_enc)))
            out = []
            step = 4096
            for start in tqdm(range(0, len(windows), step), disable=disable_tqdm,
                              desc="nano loglikelihood (x4096)"):
                out.extend(self._score_windows(windows[start:start + step]))
            return out

        def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> list[float]:
            totals = []
            for req in tqdm(requests, disable=disable_tqdm, desc="nano rolling loglikelihood"):
                (string,) = req.args
                token_list = self.tok_encode(string, add_special_tokens=False)
                pieces = list(map(lm_utils.make_disjoint_window, lm_utils.get_rolling_token_windows(
                    token_list=token_list, prefix_token=self.prefix_token_id,
                    max_seq_len=self.max_length, context_len=1)))
                windows = [(ctx + cont, len(cont)) for ctx, cont in pieces]
                totals.append(sum(score for score, _ in self._score_windows(windows)))
            return totals

        def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
            raise NotImplementedError("Only log-likelihood tasks are configured for nano-dsv4.1f")

    return NanoDSV41FLM


def load_nano(bundle: str | Path, *, device: str = "cuda", dtype: torch.dtype = torch.float32):
    from nano_dsv41f.vllm_v41_cpu import NanoDeepseekV41CPU
    from tokenizers import Tokenizer

    bundle = Path(bundle)
    model = NanoDeepseekV41CPU.from_pretrained(bundle, device=device, dtype=dtype)
    tokenizer = Tokenizer.from_file(str(bundle / "tokenizer.json"))
    return model, tokenizer


def make_nano_lm(model: Any, tokenizer: Any, **kwargs: Any):
    return _make_lm_class()(model, tokenizer, **kwargs)


# ---------------------------------------------------------------------------
# Base (pretraining) checkpoint -> portable bundle
# ---------------------------------------------------------------------------

def export_training_checkpoint(checkpoint: str | Path, output: str | Path, *,
                               tokenizer: str | Path | None = None) -> dict[str, Any]:
    """Write a minimal portable bundle (config, weights, tokenizer) from any stage.

    ``export_posttrain_checkpoint.py`` intentionally accepts only completed SFT runs.
    This helper reuses its checksum-verified parameter loader for the 2.4B-token base
    checkpoint so the base model can be evaluated with the same Torch runtime.
    Optimizer leaves are never read. Requires CPU JAX (``JAX_PLATFORMS=cpu``).
    """
    import shutil
    from export_posttrain_checkpoint import find_asset, load_parameters
    from pretrain_checkpoint import checkpoint_path, read_metadata
    from nano_dsv41f.hf_export import export_portable_checkpoint
    from nano_dsv41f.vllm_v41_cpu.config_io import model_config_from_export

    checkpoint, manifest = read_metadata(checkpoint_path(Path(checkpoint)))
    metadata = manifest["metadata"]
    recipe = metadata["recipe"]
    config = model_config_from_export(recipe["model"])
    length = int(recipe["train"]["seq_len"])
    params = load_parameters(checkpoint, manifest, config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    export_portable_checkpoint(params, output, config, max_position_embeddings=length,
                               metadata={"format": "nano-dsv41f-portable-v1",
                                         "stage": str(metadata.get("stage", "pretrain")),
                                         "checkpoint": checkpoint.name})
    shutil.copy2(find_asset(checkpoint, "tokenizer.json", tokenizer), output / "tokenizer.json")
    info = {"checkpoint": checkpoint.name, "stage": metadata.get("stage"),
            "completed_steps": metadata.get("completed_steps"),
            "real_tokens": metadata.get("real_tokens"), "seq_len": length,
            "parameter_count": sum(int(v.size) for v in params.values())}
    (output / "eval_export.json").write_text(json.dumps(info, indent=2))
    return info


# ---------------------------------------------------------------------------
# Running and summarizing
# ---------------------------------------------------------------------------

PRIMARY_METRIC = {
    "lambada_openai": ("acc,none", "acc"),
    "hellaswag": ("acc_norm,none", "acc_norm"),
    "piqa": ("acc_norm,none", "acc_norm"),
    "arc_easy": ("acc_norm,none", "acc_norm"),
    "arc_challenge": ("acc_norm,none", "acc_norm"),
    "winogrande": ("acc,none", "acc"),
    "sciq": ("acc_norm,none", "acc_norm"),
    "openbookqa": ("acc_norm,none", "acc_norm"),
    "wikitext": ("bits_per_byte,none", "bits_per_byte"),
}

# Expected accuracy of uniform guessing, used to report lift over chance.
CHANCE = {"lambada_openai": 0.0, "hellaswag": 0.25, "piqa": 0.5, "arc_easy": 0.25,
          "arc_challenge": 0.25, "winogrande": 0.5, "sciq": 0.25, "openbookqa": 0.25}


def run_tasks(lm: Any, tasks: Iterable[str], *, limit: int | float | None = None,
              batch_size: Any = None, seed: int = 1234) -> dict[str, Any]:
    """Run each task separately so a dataset download failure skips only that task."""
    import lm_eval

    summary: dict[str, Any] = {"tasks": {}, "errors": {}}
    for task in tasks:
        started = time.time()
        try:
            kwargs = dict(model=lm, tasks=[task], num_fewshot=0, limit=limit,
                          random_seed=seed, numpy_random_seed=seed, torch_random_seed=seed,
                          fewshot_random_seed=seed, log_samples=False)
            if batch_size is not None:
                kwargs["batch_size"] = batch_size
            out = lm_eval.simple_evaluate(**kwargs)
        except Exception as exc:  # keep the remaining benchmarks running
            summary["errors"][task] = f"{type(exc).__name__}: {exc}"
            print(f"[{task}] FAILED: {summary['errors'][task]}", flush=True)
            continue
        metrics = out["results"][task]
        summary["tasks"][task] = {
            "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
            "n_samples": out.get("n-samples", {}).get(task),
            "seconds": round(time.time() - started, 1),
            "version": out.get("versions", {}).get(task),
        }
        key, name = PRIMARY_METRIC.get(task, (None, None))
        value = metrics.get(key) if key else None
        print(f"[{task}] {name}={value} ({time.time() - started:.0f}s)", flush=True)
    return summary


def primary_table(results: dict[str, dict[str, Any]], tasks: Iterable[str]) -> list[dict[str, Any]]:
    """Flatten {model: summary} into rows of primary metrics plus stderr."""
    rows = []
    for name, summary in results.items():
        row: dict[str, Any] = {"model": name}
        accs = []
        for task in tasks:
            key, _ = PRIMARY_METRIC[task]
            metrics = summary.get("tasks", {}).get(task, {}).get("metrics", {})
            value = metrics.get(key)
            row[task] = value
            stderr_key = key.replace(",", "_stderr,")
            row[task + "_stderr"] = metrics.get(stderr_key)
            if task != "wikitext" and value is not None:
                accs.append(value)
        row["avg_acc"] = sum(accs) / len(accs) if accs else None
        row["n_acc_tasks"] = len(accs)
        rows.append(row)
    return rows


def evaluate_nano(bundle: Path, *, tasks: list[str], limit: Any, device: str,
                  max_length: int, token_budget: int, max_batch: int) -> dict[str, Any]:
    model, tokenizer = load_nano(bundle, device=device)
    original = install_grouped_moe()
    moe_err = check_grouped_moe(model, original)
    pad_err = check_padding_invariance(model)
    print(f"grouped MoE max|diff|={moe_err:.2e}; padding invariance max|diff|={pad_err:.2e}", flush=True)
    if moe_err > 1e-3 or pad_err > 1e-2:
        raise RuntimeError("nano runtime self-checks failed; refusing to report scores")
    lm = make_nano_lm(model, tokenizer, max_length=max_length, token_budget=token_budget,
                      max_batch=max_batch)
    summary = run_tasks(lm, tasks, limit=limit)
    info_file = bundle / "eval_export.json"
    export_info = json.loads(info_file.read_text()) if info_file.is_file() else {}
    meta_file = bundle / "training_metadata.json"
    if meta_file.is_file():
        meta = json.loads(meta_file.read_text())
        export_info.setdefault("stage", meta.get("stage"))
        export_info.setdefault("completed_steps", meta.get("completed_steps"))
    summary.update(kind="nano", bundle=str(bundle), max_length=max_length,
                   params=count_parameters(model), export=export_info,
                   checks={"grouped_moe_max_abs_diff": moe_err, "padding_max_abs_diff": pad_err},
                   forward_calls=lm.forward_calls, scored_tokens=lm.scored_tokens)
    return summary


def evaluate_hf(pretrained: str, *, revision: str | None, tasks: list[str], limit: Any,
                device: str, batch_size: int, dtype: str, max_length: int) -> dict[str, Any]:
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=pretrained, revision=revision or "main", device=device,
              batch_size=batch_size, dtype=dtype)
    # Cap rolling-perplexity windows at the nano window (models with shorter limits keep theirs).
    if lm.max_length > max_length:
        lm._max_length = max_length
    model = lm.model
    total = sum(p.numel() for p in model.parameters())
    embed = sum(p.numel() for p in model.get_input_embeddings().parameters())
    out_emb = model.get_output_embeddings()
    head = 0
    if out_emb is not None and out_emb.weight.data_ptr() != model.get_input_embeddings().weight.data_ptr():
        head = out_emb.weight.numel()
    summary = run_tasks(lm, tasks, limit=limit)
    summary.update(kind="hf", pretrained=pretrained, revision=revision or "main",
                   max_length=lm.max_length, dtype=dtype,
                   params={"total": total, "embedding": embed, "lm_head_untied": head,
                           "non_embedding": total - embed - head})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bundle", type=Path, help="portable nano-dsv4.1f bundle directory")
    source.add_argument("--hf", help="Hugging Face model id for a baseline")
    parser.add_argument("--revision", default=None, help="HF revision, e.g. Pythia 'step1000'")
    parser.add_argument("--name", default=None)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--token-budget", type=int, default=16384)
    parser.add_argument("--max-batch", type=int, default=128)
    parser.add_argument("--hf-batch-size", type=int, default=8)
    parser.add_argument("--hf-dtype", default="float32")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tasks = [t for t in args.tasks.split(",") if t]
    limit = None if args.limit is None else (int(args.limit) if args.limit >= 1 else args.limit)
    started = time.time()
    if args.bundle is not None:
        summary = evaluate_nano(args.bundle, tasks=tasks, limit=limit, device=args.device,
                                max_length=args.max_length, token_budget=args.token_budget,
                                max_batch=args.max_batch)
        default_name = args.bundle.name
    else:
        summary = evaluate_hf(args.hf, revision=args.revision, tasks=tasks, limit=limit,
                              device=args.device, batch_size=args.hf_batch_size, dtype=args.hf_dtype,
                              max_length=args.max_length)
        default_name = args.hf + (f"@{args.revision}" if args.revision else "")
    summary.update(name=args.name or default_name, limit=limit,
                   total_seconds=round(time.time() - started, 1),
                   device=torch.cuda.get_device_name(0) if args.device.startswith("cuda") else args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print("Wrote", args.output, flush=True)


if __name__ == "__main__":
    main()
