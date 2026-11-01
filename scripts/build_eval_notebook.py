#!/usr/bin/env python3
"""Generate the public Kaggle GPU notebook for standard zero-shot benchmarks."""
from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent

import nbformat as nbf

# Exported SFT bundles and the Torch runtime they load with live on this commit.
RUNTIME_COMMIT = "05c9613f1e86f8596120c0525b6394425e06c9ed"
EVAL_REF = "claude/trusting-pascal-9ln3ic"


def _md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(source).strip() + "\n")


def _code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(source).strip() + "\n")


def build_notebook(limit: int | None = None) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["metadata"].update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    })
    nb.cells = [
        _md(
            '''
            # nano-dsv4.1f — standard zero-shot benchmarks vs. open nano models

            [nano-dsv4.1f](https://github.com/xiayicheng3-code/nano-dsv4.1f) is a from-scratch
            JAX reimplementation of DeepSeek-V4.1-Flash ideas (CSA2 sparse attention, mHC,
            routed MoE, Engram), pretrained on **2.4B FineWeb-Edu tokens** on a free Kaggle TPU v5e-8,
            then mid-trained (0.6B tokens) and SFT'd.

            This notebook scores its checkpoints with **EleutherAI lm-evaluation-harness**, using
            the same task definitions, prompts and metrics as for the Hugging Face baselines:

            | Task | Metric | Chance |
            |---|---|---|
            | LAMBADA (OpenAI) | last-word accuracy | 0 |
            | HellaSwag | acc_norm | 25% |
            | PIQA | acc_norm | 50% |
            | ARC-Easy / ARC-Challenge | acc_norm | 25% |
            | WinoGrande | acc | 50% |
            | SciQ | acc_norm | 25% |
            | OpenBookQA | acc_norm | 25% |
            | WikiText-2 | bits per byte (tokenizer-independent, lower is better) | — |

            All tasks are zero-shot and use full test/validation sets. `acc_norm` normalizes by
            continuation byte length, so different tokenizers are compared fairly.

            **Baselines.** GPT-2 124M, Pythia-160M at the **token-matched** checkpoints `step1000`
            (≈2.1B tokens) and `step2000` (≈4.2B tokens) plus its final 300B-token weights,
            and SmolLM-135M / SmolLM2-135M (600B / 2T tokens). nano-dsv4.1f is evaluated at three
            stages: base (2.4B tokens), mid-trained (3.0B tokens) and SFT, all as raw
            language models without a chat template.

            **How nano is run.** The weights load into the project's Torch port of the model
            (`nano_dsv41f.vllm_v41_cpu`, here on CUDA), using the dense backbone path. Before scoring,
            the notebook checks that (1) the grouped-GEMM MoE dispatch used here matches the
            reference MoE, and (2) right-padding in a batch leaves logits unchanged. Torch logits
            were also compared with the JAX training forward on the real base weights during
            development.

            Settings: **GPU T4 ×2** and Internet on. The two GPUs evaluate models in parallel.
            '''
        ),
        _code(
            f'''
            SOURCE_REPO = "https://github.com/xiayicheng3-code/nano-dsv4.1f.git"
            RUNTIME_COMMIT = "{RUNTIME_COMMIT}"  # runtime that loads the exported bundles
            EVAL_REF = "{EVAL_REF}"  # branch providing scripts/nano_lm_eval.py
            TASKS = ["lambada_openai", "hellaswag", "piqa", "arc_easy", "arc_challenge",
                     "winogrande", "sciq", "openbookqa", "wikitext"]
            LIMIT = __LIMIT__  # e.g. 50 for a quick smoke run; None = full benchmarks

            # (display name, HF id, revision, training tokens)
            BASELINES = [
                ("GPT-2 124M", "openai-community/gpt2", None, "WebText (undisclosed)"),
                ("Pythia-160M @2.1B tok", "EleutherAI/pythia-160m", "step1000", "2.1B (Pile)"),
                ("Pythia-160M @4.2B tok", "EleutherAI/pythia-160m", "step2000", "4.2B (Pile)"),
                ("Pythia-160M final", "EleutherAI/pythia-160m", None, "300B (Pile)"),
                ("SmolLM-135M", "HuggingFaceTB/SmolLM-135M", None, "600B (SmolLM-Corpus)"),
                ("SmolLM2-135M", "HuggingFaceTB/SmolLM2-135M", None, "2T (SmolLM2 mix)"),
            ]
            '''.replace("__LIMIT__", repr(limit))
        ),
        _code(
            '''
            import os, subprocess, sys, json, shutil, time
            from pathlib import Path

            WORK = Path("/kaggle/working")
            REPO = Path("/kaggle/temp/nano-dsv4.1f")
            if not REPO.exists():
                subprocess.run(["git", "clone", "-q", SOURCE_REPO, str(REPO)], check=True)
            subprocess.run(["git", "-C", str(REPO), "fetch", "-q", "origin", RUNTIME_COMMIT, EVAL_REF], check=True)
            subprocess.run(["git", "-C", str(REPO), "checkout", "-q", "--detach", RUNTIME_COMMIT], check=True)
            eval_sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", f"origin/{EVAL_REF}"], text=True).strip()
            script = subprocess.check_output(["git", "-C", str(REPO), "show", f"{eval_sha}:scripts/nano_lm_eval.py"], text=True)
            EVAL_SCRIPT = REPO / "scripts" / "nano_lm_eval.py"
            EVAL_SCRIPT.write_text(script)
            print("runtime commit:", RUNTIME_COMMIT)
            print("eval script commit:", eval_sha)

            # JAX is only used on CPU (config classes + reading the JAX base checkpoint).
            subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q",
                            "jax-cuda12-plugin", "jax-cuda12-pjrt"], check=False)
            subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                            "lm_eval[hf]==0.4.13", "jax==0.10.2", "jaxlib==0.10.2", "tokamax==0.0.12",
                            "ml_dtypes", "tabulate", "safetensors>=0.6", "tokenizers>=0.21"], check=True)
            ENV = dict(os.environ, JAX_PLATFORMS="cpu", TOKENIZERS_PARALLELISM="false",
                       HF_HUB_DISABLE_PROGRESS_BARS="1",
                       PYTHONPATH=os.pathsep.join([str(REPO / "src"), str(REPO / "scripts")]))
            print(subprocess.check_output([sys.executable, "-c",
                "import torch, lm_eval, transformers; print('torch', torch.__version__, '| lm_eval', lm_eval.__version__, '| transformers', transformers.__version__); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"],
                env=ENV, text=True))
            '''
        ),
        _md(
            '''
            ## Locate checkpoints

            * **SFT**: the exported safetensors bundle (`export_manifest.json`).
            * **Base**: the JAX pretraining checkpoint at 2.4B tokens. It is converted on CPU to the
              same portable layout (parameters only, checksums verified, optimizer state never read).
            * **Mid-trained** (optional): exported the same way if a `midtrain/checkpoints` pointer is attached.
            '''
        ),
        _code(
            '''
            INPUT = Path("/kaggle/input")
            BUNDLES = WORK / "bundles"
            BUNDLES.mkdir(exist_ok=True)
            nano_models = []  # (display name, bundle dir, training tokens)

            def export(checkpoints_dir, name):
                out = BUNDLES / name
                if not (out / "model.safetensors").exists():
                    shutil.rmtree(out, ignore_errors=True)
                    code = ("import json, nano_lm_eval as E; "
                            f"print(json.dumps(E.export_training_checkpoint({str(checkpoints_dir)!r}, {str(out)!r})))")
                    print(subprocess.check_output([sys.executable, "-c", code], env=ENV, text=True).strip().splitlines()[-1])
                return out

            base = sorted((p.parent for p in INPUT.rglob("checkpoints/latest.json")
                           if "midtrain" not in p.parts and "sft" not in p.parts),
                          key=lambda p: "pretrain" not in str(p))
            if base:
                nano_models.append(("nano-dsv4.1f base", export(base[0], "nano-base"), "2.4B (FineWeb-Edu)"))
            mid = [p.parent for p in INPUT.rglob("midtrain/checkpoints/latest.json")]
            if mid:
                nano_models.append(("nano-dsv4.1f mid-trained", export(mid[0], "nano-midtrain"), "3.0B (+code/math/traces)"))
            sft = [p.parent for p in INPUT.rglob("export_manifest.json")]
            if sft:
                nano_models.append(("nano-dsv4.1f SFT", sft[0], "3.0B + SFT"))
            for row in nano_models:
                print(row)
            assert nano_models, "attach the nano-dsv4.1f checkpoints as inputs"
            '''
        ),
        _md(
            '''
            ## Run all evaluations

            Each model runs in its own process (`scripts/nano_lm_eval.py`), pinned to one T4. Logs are
            written to `eval-logs/`, results to `eval-results/<model>.json`. Finished results
            are not recomputed if the cell is run again.
            '''
        ),
        _code(
            '''
            import re, threading, queue, torch

            RESULTS = WORK / "eval-results"; RESULTS.mkdir(exist_ok=True)
            LOGS = WORK / "eval-logs"; LOGS.mkdir(exist_ok=True)
            slug = lambda s: re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").lower()

            jobs = []
            for name, bundle, tokens in nano_models:
                jobs.append((name, ["--bundle", str(bundle)], tokens))
            for name, hf_id, rev, tokens in BASELINES:
                jobs.append((name, ["--hf", hf_id] + (["--revision", rev] if rev else []), tokens))
            META = {name: tokens for name, _, tokens in jobs}

            pending = queue.Queue()
            for job in jobs:
                if not (RESULTS / f"{slug(job[0])}.json").exists():
                    pending.put(job)
            print(pending.qsize(), "of", len(jobs), "models to evaluate")

            def worker(gpu):
                while True:
                    try:
                        name, args, _ = pending.get_nowait()
                    except queue.Empty:
                        return
                    out = RESULTS / f"{slug(name)}.json"
                    cmd = [sys.executable, "-u", str(EVAL_SCRIPT), *args, "--name", name,
                           "--tasks", ",".join(TASKS), "--output", str(out)]
                    if LIMIT:
                        cmd += ["--limit", str(LIMIT)]
                    started = time.time()
                    with open(LOGS / f"{slug(name)}.log", "w") as log:
                        rc = subprocess.run(cmd, env=dict(ENV, CUDA_VISIBLE_DEVICES=str(gpu)),
                                            stdout=log, stderr=subprocess.STDOUT).returncode
                    print(f"[gpu{gpu}] {name}: exit {rc} after {(time.time() - started) / 60:.1f} min", flush=True)

            n_gpus = max(1, torch.cuda.device_count())
            threads = [threading.Thread(target=worker, args=(g,)) for g in range(n_gpus)]
            t0 = time.time()
            for t in threads: t.start()
            for t in threads: t.join()
            print(f"all done in {(time.time() - t0) / 60:.1f} min")
            '''
        ),
        _code(
            '''
            # Per-task progress lines and any failures, from the logs.
            for log in sorted(LOGS.glob("*.log")):
                lines = [l.rstrip() for l in log.read_text(errors="replace").splitlines()
                         if l.startswith("[") and ("=" in l or "FAILED" in l) and not l.startswith("[Task")]
                lines += [l for l in log.read_text(errors="replace").splitlines() if "grouped MoE" in l or "Traceback" in l]
                print(f"== {log.stem}"); print("\\n".join(lines[-14:]))
            '''
        ),
        _md("## Results"),
        _code(
            '''
            import pandas as pd
            sys.path.insert(0, str(REPO / "scripts"))
            import importlib, nano_lm_eval as E
            importlib.reload(E)

            results = {}
            for name, _, _ in jobs:
                f = RESULTS / f"{slug(name)}.json"
                if f.exists():
                    results[name] = json.loads(f.read_text())
            rows = E.primary_table(results, TASKS)
            df = pd.DataFrame(rows).set_index("model")
            df.insert(0, "train tokens", [META[m] for m in df.index])
            def nonemb(m):
                p = results[m]["params"]
                if results[m]["kind"] == "nano":
                    return f'{p["backbone_non_embedding"] / 1e6:.0f}M ({p["backbone_active_non_embedding"] / 1e6:.0f}M active)'
                return f'{p["non_embedding"] / 1e6:.0f}M'
            df.insert(1, "non-emb params", [nonemb(m) for m in df.index])
            show = ["train tokens", "non-emb params"] + TASKS + ["avg_acc"]
            pretty = df[show].copy()
            for t in TASKS + ["avg_acc"]:
                pretty[t] = pretty[t].map(lambda v: "—" if pd.isna(v) else (f"{v:.3f}" if t == "wikitext" else f"{100 * v:.1f}"))
            pretty = pretty.rename(columns={"wikitext": "wikitext bpb ↓", "avg_acc": "avg acc (8 tasks)"})
            df.to_csv(WORK / "standard_evals.csv")
            (WORK / "standard_evals.md").write_text(pretty.to_markdown())
            json.dump(results, open(WORK / "standard_evals.json", "w"), indent=2)
            print("accuracies in %, zero-shot, full sets; wikitext in bits/byte")
            pretty
            '''
        ),
        _code(
            '''
            # Standard errors (percentage points) reported by the harness.
            se = df[[t + "_stderr" for t in TASKS if t != "wikitext"]] * 100
            se.columns = [c.replace("_stderr", "") for c in se.columns]
            se.round(2)
            '''
        ),
        _code(
            '''
            import matplotlib.pyplot as plt
            import numpy as np

            acc_tasks = [t for t in TASKS if t != "wikitext"]
            models = list(df.index)
            fig, ax = plt.subplots(figsize=(13, 5))
            width = 0.8 / len(models)
            x = np.arange(len(acc_tasks))
            for i, m in enumerate(models):
                vals = [100 * (df.loc[m, t] if pd.notna(df.loc[m, t]) else 0) for t in acc_tasks]
                ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=m,
                       edgecolor="black" if m.startswith("nano") else "none", linewidth=0.8)
            chance = [100 * E.CHANCE[t] for t in acc_tasks]
            ax.scatter(x, chance, marker="_", s=900, color="black", zorder=5, label="chance")
            ax.set_xticks(x, acc_tasks)
            ax.set_ylabel("zero-shot accuracy (%)")
            ax.set_title("nano-dsv4.1f vs. open ~100M-parameter models (lm-evaluation-harness)")
            ax.legend(ncol=3, fontsize=8)
            fig.tight_layout()
            fig.savefig(WORK / "standard_evals.png", dpi=150)
            plt.show()
            '''
        ),
        _md(
            '''
            ## Reading the results

            * **Most relevant comparison:** Pythia-160M at `step1000`/`step2000` saw a similar number of
              tokens (2.1B / 4.2B), but from the Pile instead of FineWeb-Edu. The fully trained models
              saw 100–1000× more data and show how far ~100M-parameter models can go.
            * nano-dsv4.1f has fewer *active* parameters than every baseline: top-4 of 48 small experts
              plus a shared expert, d_model 512, 7 layers.
            * FineWeb-Edu is educational web text. It typically helps knowledge QA (ARC, SciQ,
              OpenBookQA) and hurts narrative cloze tasks (LAMBADA, WikiText).
            * The SFT checkpoint is scored as a raw LM on raw prompts, without its chat template.
            * Differences of less than about two standard errors (table above) are noise.
            '''
        ),
    ]
    return nb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("notebooks/nano_dsv41f_standard_evals.ipynb"))
    parser.add_argument("--limit", type=int, default=None, help="per-task sample cap for smoke runs")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build_notebook(args.limit), args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
