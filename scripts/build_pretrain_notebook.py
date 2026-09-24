"""Build the Kaggle 2.4B-token production pretraining notebook."""
import argparse
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_REF = "codex/pretrain-session"


def build(source_ref=DEFAULT_SOURCE_REF):
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.cells = [md('''# nano-dsv4.1f — pretrain 2.4B tokens

Select **TPU v5e-8**, enable **Internet**, attach the two datasets below, and use
**Save Version → Save & Run All** in a fresh session.

- `xiayicheng3gmailcom/nanodsv4-1f-pretrain-tokenized`
- `xiayicheng3gmailcom/nano-dsv41f-tokenizer-fineweb`

This run trains only on the prepared **FineWeb-Edu** corpus. It stops at **2.4B
non-padding tokens**, reserving **600M** of the 3B experiment budget for a later
mid-training run. The final batch may exceed the target by fewer than 32,768 tokens.
The existing corpus can contain 3B tokens; it does not need rebuilding.

Fixed settings: **4 × 8192 tokens/update**, CP2/DP4 attention, EP8 MoE, **48 ×128
experts, top-4 + shared**, sequential Splash, local backward-Q tile128,
compressed/global backward-Q tile1024, **quarter MoE buffers with dropless fallback**,
BF16 parameters/payloads, FP32 optimizer/control paths, block rematerialization.
QAT and DSpark training remain off; hierarchical candidate masking remains off.

The LR/indexer schedule uses a frozen full-3B step horizon derived from corpus
packing density. Indexer distillation is active at 55–90% of that horizon. LR
warmup is 500 updates; peak 2.6e-4, final 2.6e-5, cosine decay starts at 90%.
Reaching 2.4B does **not** decay the LR to its floor or reset optimizer state.

Expected training steps alone: about **six hours**. Compilation, validation,
checkpointing and input I/O are additional. The deadline is **eight hours from
the first code cell**, with two minutes reserved for the final checkpoint and
roughly one further hour before Kaggle's nine-hour cap. Periodic checkpoints are
also saved every 2,000 steps and before new compilations.

Held-out validation uses the same 32 shuffled validation batches (~1M tokens)
at session start, every 10,000 global updates and completion when time permits.
No benchmark workers or training-data preparation jobs are launched.'''),
    code('''import os
import time
# Timer includes source setup, package installation, corpus verification and training.
SESSION_STARTED = time.time()
BASE_TOKENS = 2_400_000_000
TOTAL_TOKENS = 3_000_000_000
SESSION_HOURS = 8.0
# For a resumed base run, attach the previous outputs and point to the checkpoint
# directory containing manifest.json, or its parent containing latest.json.
RESUME_CHECKPOINT = ""
CORPUS = "/kaggle/input/datasets/xiayicheng3gmailcom/nanodsv4-1f-pretrain-tokenized"
TOKENIZER = "/kaggle/input/datasets/xiayicheng3gmailcom/nano-dsv41f-tokenizer-fineweb"
os.environ["NANO_DSV41F_REF"] = ''' + repr(source_ref) + '''
print({"base_tokens": BASE_TOKENS, "midtrain_reserved": TOTAL_TOKENS - BASE_TOKENS,
       "session_hours": SESSION_HOURS, "resume": RESUME_CHECKPOINT or "fresh initialization"})'''),
    code((ROOT / "scripts/kaggle_bootstrap.py").read_text()),
    code('''import json
from datetime import datetime, timezone

def attached_path(configured, slug):
    path = Path(configured)
    if path.exists():
        return path
    # Kaggle exposes both legacy /kaggle/input/<slug> and newer owner-qualified paths.
    matches = sorted(p for p in Path("/kaggle/input").rglob(slug) if p.is_dir())
    if len(matches) != 1:
        raise FileNotFoundError(f"Attach dataset {slug}; found {len(matches)} matching directories")
    return matches[0]

CORPUS = attached_path(CORPUS, "nanodsv4-1f-pretrain-tokenized")
TOKENIZER = attached_path(TOKENIZER, "nano-dsv41f-tokenizer-fineweb")
if RESUME_CHECKPOINT and not Path(RESUME_CHECKPOINT).is_dir():
    raise FileNotFoundError(RESUME_CHECKPOINT)
RUN_ROOT = Path("/kaggle/working") / ("pretrain-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
RUN_ROOT.mkdir(exist_ok=False)
OUTPUT = RUN_ROOT / "training"
# Preserve the exact runnable source alongside the model, without dependency caches.
subprocess.run(["git", "archive", "--format=tar.gz", "-o", str(RUN_ROOT / "source.tar.gz"), "HEAD"], check=True)
print("Output:", RUN_ROOT)
print("Corpus:", CORPUS)
print("Tokenizer:", TOKENIZER)'''),
    code('''command = [sys.executable, "-u", "scripts/run_pretrain.py",
    "--corpus", str(CORPUS), "--tokenizer", str(TOKENIZER), "--output", str(OUTPUT),
    "--base-tokens", str(BASE_TOKENS), "--total-tokens", str(TOTAL_TOKENS),
    "--deadline-unix", str(SESSION_STARTED + SESSION_HOURS * 3600),
    "--checkpoint-every", "2000", "--eval-every", "10000", "--eval-batches", "32",
    "--log-every", "100", "--data-seed", "1701", "--init-seed", "7"]
if RESUME_CHECKPOINT:
    command += ["--resume", RESUME_CHECKPOINT]
# A fresh subprocess owns the TPU runtime. Tee progress to a persistent log.
with (RUN_ROOT / "train.log").open("w", buffering=1) as log:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        returncode = process.wait()
    except KeyboardInterrupt:
        process.terminate()  # runner handles SIGTERM and checkpoints between updates
        print("Stop requested; waiting for the training process to save its checkpoint.")
        returncode = process.wait()
print("Training process return code:", returncode)
summary_path = OUTPUT / "summary.json"
summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
print(json.dumps({k: summary.get(k) for k in
    ("status", "stop_reason", "progress", "remaining_base_tokens", "checkpoint")}, indent=2))
if returncode:
    print("Training failed. Inspect train.log; the latest committed checkpoint is retained.")'''),
    md('''## Download and resume

The final bundle contains the **latest full training checkpoint**, tokenizer,
recipe, metrics, summary and source archive. The two most recent checkpoints are
also retained separately in Kaggle outputs. `completed` means the 2.4B base budget
was reached; `paused` means resume this base run before switching data regimes.

To resume: save these outputs as a Kaggle dataset, attach it to a fresh copy of
this notebook, and set `RESUME_CHECKPOINT` above. If using the ZIP, extract it
first. Keep the same source version, corpus, tokenizer, budgets and seeds.
The data cursor resumes without reading all preceding token rows. The first
2.4B allocation does not automatically launch mid-training or SFT.

Checkpoints are finite-value checked and SHA-256 verified on restore. A failed
step does not replace the last committed checkpoint. The wall-time stop is
cooperative; the periodically saved checkpoints remain the recovery point if
Kaggle terminates a process while it is compiling or writing an output.'''),
    code('''import zipfile
from IPython.display import FileLink, display

latest_path = OUTPUT / "checkpoints/latest.json"
if latest_path.exists():
    latest = json.loads(latest_path.read_text())["checkpoint"]
    bundle = RUN_ROOT / "pretrain-resume.zip"
    checkpoint_dir = OUTPUT / "checkpoints" / latest
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path in sorted(RUN_ROOT.rglob("*")):
            if not path.is_file() or path == bundle:
                continue
            if "checkpoints" in path.relative_to(RUN_ROOT).parts:
                if path != latest_path and checkpoint_dir not in path.parents:
                    continue
            archive.write(path, path.relative_to(RUN_ROOT))
    display(FileLink(str(bundle)))
    print("Resume from the extracted training/checkpoints directory.")
else:
    print("No committed checkpoint yet. Inspect train.log and training/summary.json.")
display(FileLink(str(RUN_ROOT / "train.log")))''')]
    path = ROOT / "notebooks/nano_dsv41f_pretrain.ipynb"
    for i, cell in enumerate(nb.cells):
        cell.id = f"pretrain-{i:02d}"
        if cell.cell_type == "code":
            compile(cell.source, str(path), "exec")
    nbf.validate(nb)
    nbf.write(nb, path)
    print(path)
    return nb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-ref", default=DEFAULT_SOURCE_REF)
    build(parser.parse_args().source_ref)
