#!/usr/bin/env python3
"""CPU-only, resumable FineWeb-Edu preparation; no JAX/package installation required.

The same file exposes iter_pretrain_batches() for the training input pipeline.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import runpy
import shutil
import time

import numpy as np

# Read the canonical contract without importing nano_dsv41f.__init__ (JAX/TPU).
_CONTRACT = runpy.run_path(str(Path(__file__).resolve().parents[1] / "src/nano_dsv41f/chat_protocol.py"))
BOS = _CONTRACT["BOS_TOKEN_ID"]
EOS = _CONTRACT["EOS_TOKEN_ID"]
PAD = _CONTRACT["PAD_TOKEN_ID"]
FORMAT = "nano-dsv41f-pretrain-compact-v1"


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def split_for_text(text, validation_modulus=100):
    # Identical text always stays in the same split, including across source files.
    h = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return "validation" if int.from_bytes(h, "little") % validation_modulus == 0 else "train"


def token_chunks(ids, seq_len):
    for start in range(0, len(ids), seq_len - 2):
        content = ids[start:start + seq_len - 2]
        yield np.asarray([BOS, *content, EOS], dtype=np.uint16)


class CompactWriter:
    """Bounded best-fit packing, with document-aligned r=2 compression groups."""

    def __init__(self, root, *, seq_len=8192, alignment=2, shard_rows=2048, open_rows=32):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.seq_len, self.alignment = seq_len, alignment
        self.shard_rows, self.max_open_rows = shard_rows, open_rows
        self.open_rows = []
        self.rows, self.lengths = [], []
        self.shards = []
        self.counts = Counter()

    def add(self, tokens):
        size = len(tokens)
        physical = math.ceil(size / self.alignment) * self.alignment
        if not 3 <= size <= self.seq_len or physical > self.seq_len:
            raise ValueError("segment must contain BOS/content/EOS and fit in a row")
        candidates = [i for i, row in enumerate(self.open_rows) if row[0] + physical <= self.seq_len]
        if candidates:
            i = max(candidates, key=lambda i: self.open_rows[i][0])
        else:
            if len(self.open_rows) == self.max_open_rows:
                i = max(range(len(self.open_rows)), key=lambda i: self.open_rows[i][0])
                self._emit(self.open_rows.pop(i))
            self.open_rows.append([0, []])
            i = len(self.open_rows) - 1
        row = self.open_rows[i]
        row[0] += physical
        row[1].append(tokens)
        if row[0] == self.seq_len:
            self._emit(self.open_rows.pop(i))

    def _emit(self, row):
        tokens = np.full(self.seq_len, PAD, dtype=np.uint16)
        lengths = []
        cursor = 0
        for segment in row[1]:
            length = len(segment)
            tokens[cursor:cursor + length] = segment
            lengths.append(length)
            cursor += math.ceil(length / self.alignment) * self.alignment
        self.rows.append(tokens)
        self.lengths.append(lengths)
        self.counts.update(rows=1, real_tokens=sum(lengths), lm_tokens=sum(n - 1 for n in lengths),
                           content_tokens=sum(n - 2 for n in lengths), segments=len(lengths),
                           physical_tokens=self.seq_len)
        if len(self.rows) >= self.shard_rows:
            self._flush()

    def _flush(self):
        if not self.rows:
            return
        index = len(self.shards)
        prefix = f"shard-{index:05d}"
        offsets = np.concatenate(([0], np.cumsum([len(x) for x in self.lengths]))).astype(np.uint32)
        arrays = {
            "tokens": np.stack(self.rows),
            "lengths": np.asarray([n for row in self.lengths for n in row], dtype=np.uint16),
            "offsets": offsets,
        }
        files = {}
        for key, array in arrays.items():
            path = self.root / f"{prefix}-{key}.npy"
            np.save(path, array, allow_pickle=False)
            files[key] = {"file": path.name, "bytes": path.stat().st_size, "sha256": digest_file(path)}
        self.shards.append({"rows": len(self.rows), "files": files})
        self.rows.clear()
        self.lengths.clear()

    def finish(self):
        for row in self.open_rows:
            self._emit(row)
        self.open_rows.clear()
        self._flush()
        return {"counts": dict(self.counts), "shards": self.shards}


def restore_rows(tokens, lengths, offsets, *, alignment=2):
    """Reconstruct the three arrays accepted by put_training_batch()."""
    ids = np.asarray(tokens, dtype=np.int32)
    segments = np.empty(ids.shape, dtype=np.int32)
    mask = np.zeros(ids.shape, dtype=bool)
    for row in range(len(ids)):
        real = np.asarray(lengths[int(offsets[row]):int(offsets[row + 1])], dtype=np.int32)
        if not len(real):
            raise ValueError("empty packed row")
        physical = ((real + alignment - 1) // alignment) * alignment
        if np.any(real < 3) or physical.sum() > ids.shape[1]:
            raise ValueError("invalid segment lengths")
        physical[-1] += ids.shape[1] - int(physical.sum())
        segments[row] = np.repeat(np.arange(len(real), dtype=np.int32), physical)
        starts = np.cumsum(physical) - physical
        for start, length in zip(starts, real):
            mask[row, start:start + length] = True
    return {"input_ids": ids, "segment_ids": segments, "token_mask": mask}


def iter_pretrain_batches(root, *, split="train", batch_rows=1, seed=1701, drop_last=True):
    """One pass, shuffled shards/rows, bounded memory; no implicit epoch repetition.

    Concatenates the tail of one shard with the next, so drop_last loses at most
    batch_rows-1 rows per pass, rather than per shard. Save seed and consumed batch
    index with the model checkpoint to reproduce/resume the training input order.
    """
    if split not in ("train", "validation") or batch_rows <= 0:
        raise ValueError("invalid split or batch size")
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("format") != FORMAT or not manifest.get("complete"):
        raise ValueError("training requires a completed pretraining manifest")
    shards = []
    for part in manifest["parts"]:
        meta = json.loads((root / part / "manifest.json").read_text())
        for shard in meta["splits"][split]["shards"]:
            shards.append((root / part / split, shard))
    rng = np.random.default_rng(seed)
    rng.shuffle(shards)
    pending = []
    for directory, shard in shards:
        arrays = {key: np.load(directory / info["file"], mmap_mode="r", allow_pickle=False)
                  for key, info in shard["files"].items()}
        for index in rng.permutation(shard["rows"]):
            lo, hi = arrays["offsets"][index:index + 2]
            pending.append((arrays["tokens"][index].copy(), arrays["lengths"][lo:hi].copy()))
            if len(pending) == batch_rows:
                yield _restore_pending(pending, manifest["config"]["alignment"])
                pending.clear()
    if pending and not drop_last:
        yield _restore_pending(pending, manifest["config"]["alignment"])


def _restore_pending(pending, alignment):
    lengths = np.concatenate([item[1] for item in pending])
    offsets = np.concatenate(([0], np.cumsum([len(item[1]) for item in pending])))
    return restore_rows(np.stack([item[0] for item in pending]), lengths, offsets, alignment=alignment)


def prepare_part(rows, tokenizer, directory, config, remaining):
    """One source file is an atomic restart unit. Upstream curated text is retained."""
    writers = {s: CompactWriter(Path(directory) / s, seq_len=config["seq_len"],
                               alignment=config["alignment"], shard_rows=config["shard_rows"],
                               open_rows=config["open_rows"]) for s in remaining}
    accepted = Counter()
    stats = Counter()
    pending, pending_chars = [], 0
    started = time.monotonic()

    def done():
        return all(accepted[s] >= target for s, target in remaining.items())

    def flush():
        if not pending:
            return
        encodings = tokenizer.encode_batch([text for text, _ in pending], add_special_tokens=False)
        stats["encode_batch_calls"] += 1
        for encoding, (_, split) in zip(encodings, pending):
            if accepted[split] >= remaining[split]:
                continue
            for chunk in token_chunks(encoding.ids, config["seq_len"]):
                if accepted[split] >= remaining[split]:
                    break
                writers[split].add(chunk)
                accepted[split] += len(chunk)
        pending.clear()

    for row in rows:
        stats["documents_seen"] += 1
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            stats["empty_documents"] += 1
            continue
        if len(text) > config["max_document_chars"]:
            stats["overlong_documents"] += 1
            continue
        split = split_for_text(text, config["validation_modulus"])
        if accepted[split] >= remaining[split]:
            continue
        if pending and (len(pending) >= config["batch_size"] or
                        pending_chars + len(text) > config["batch_chars"]):
            flush()
            pending_chars = 0
            if done():
                break
        pending.append((text, split))
        pending_chars += len(text)
        if stats["documents_seen"] % 10000 == 0:
            elapsed = max(time.monotonic() - started, 1e-6)
            print(json.dumps({"documents_seen": stats["documents_seen"], "accepted": dict(accepted),
                              "tokens_per_second": round(sum(accepted.values()) / elapsed)}), flush=True)
    flush()
    return {"splits": {s: writer.finish() for s, writer in writers.items()}, "source_stats": dict(stats)}


def verify_part(directory, meta):
    for split in ("train", "validation"):
        for shard in meta["splits"][split]["shards"]:
            for info in shard["files"].values():
                path = Path(directory) / split / info["file"]
                if not path.is_file() or path.stat().st_size != info["bytes"] or digest_file(path) != info["sha256"]:
                    raise ValueError(f"missing/corrupt completed shard: {path}")


def build(root, tokenizer, config, source_files, row_loader):
    """Resume only a matching recipe; never silently accept short or corrupt output."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "build_plan.json"
    plan = {"format": FORMAT, "config": config, "source_files": source_files}
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError("output belongs to a different recipe/tokenizer/revision; use a new output directory")
    else:
        if any(root.iterdir()):
            raise ValueError("output directory is nonempty without a build plan")
        write_json(plan_path, plan)
    targets = {"train": config["train_tokens"], "validation": config["validation_tokens"]}
    totals = {s: Counter() for s in targets}
    parts = []
    manifest = {}
    for index, filename in enumerate(source_files):
        if all(totals[s]["real_tokens"] >= targets[s] for s in targets):
            break
        relative = f"parts/part-{index:05d}"
        directory = root / relative
        if (directory / "manifest.json").exists():
            meta = json.loads((directory / "manifest.json").read_text())
            if meta["source_file"] != filename:
                raise ValueError("source order changed")
            verify_part(directory, meta)
            print(f"resume: {relative}", flush=True)
        else:
            temporary = root / (relative + ".pending")
            # Only the uncommitted work of this builder is discarded on retry.
            if temporary.exists():
                shutil.rmtree(temporary)
            print(f"prepare: {index + 1}/{len(source_files)} {filename}", flush=True)
            remaining = {s: max(0, targets[s] - totals[s]["real_tokens"]) for s in targets}
            meta = prepare_part(row_loader(filename), tokenizer, temporary, config, remaining)
            meta["source_file"] = filename
            write_json(temporary / "manifest.json", meta)
            os.replace(temporary, directory)
        parts.append(relative)
        for s in targets:
            totals[s].update(meta["splits"][s]["counts"])
        manifest = {"format": FORMAT, "config": config, "parts": parts,
                    "splits": {s: dict(counts) for s, counts in totals.items()},
                    "complete": all(totals[s]["real_tokens"] >= targets[s] for s in targets)}
        write_json(root / "manifest.json", manifest)
        print(json.dumps({"complete": manifest["complete"], "counts": manifest["splits"]}), flush=True)
    if not manifest.get("complete"):
        raise RuntimeError(f"source exhausted before token targets; partial output preserved at {root}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-tokens", type=int, default=3_000_000_000)
    parser.add_argument("--validation-tokens", type=int, default=10_000_000)
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--shard-rows", type=int, default=2048)
    parser.add_argument("--open-rows", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--batch-chars", type=int, default=4_000_000)
    parser.add_argument("--max-document-chars", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()
    if args.seq_len < 4 or args.seq_len > 65534 or args.seq_len % 2:
        parser.error("seq-len must be even and in [4, 65534]")
    if any(getattr(args, name) <= 0 for name in ("train_tokens", "validation_tokens", "shard_rows", "open_rows", "batch_size", "batch_chars", "max_document_chars")):
        parser.error("token budgets and memory bounds must be positive")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("RAYON_NUM_THREADS", str(max(1, os.cpu_count() or 1)))
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from tokenizers import Tokenizer
    import datasets
    import tokenizers

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    vocabulary = tokenizer.get_vocab()
    if max(vocabulary.values()) > 65535:
        raise ValueError("token IDs exceed uint16 storage")
    for token, token_id in _CONTRACT["nano_v41_tokenizer_contract"](len(vocabulary)).token_to_id.items():
        if tokenizer.token_to_id(token) != token_id:
            raise ValueError(f"tokenizer contract mismatch: {token!r}")
    dataset = "HuggingFaceFW/fineweb-edu"
    plan_path = args.output_dir / "build_plan.json"
    previous = json.loads(plan_path.read_text()) if plan_path.exists() else None
    api = HfApi()
    revision = (previous["config"]["dataset_revision"] if previous and args.dataset_revision == "main"
                else api.dataset_info(dataset, revision=args.dataset_revision).sha)
    source_files = (previous["source_files"] if previous else sorted(
        f for f in api.list_repo_files(dataset, repo_type="dataset", revision=revision)
        if f.startswith("sample/10BT/") and f.endswith(".parquet")))
    if not source_files:
        raise ValueError("FineWeb-Edu sample/10BT parquet files not found")
    if not previous:
        random.Random(args.seed).shuffle(source_files)
    config = {"dataset": dataset, "dataset_config": "sample-10BT", "dataset_revision": revision,
              "builder_sha256": digest_file(Path(__file__)),
              "tokenizer_sha256": digest_file(args.tokenizer), "vocab_size": len(vocabulary),
              "seq_len": args.seq_len, "alignment": 2, "train_tokens": args.train_tokens,
              "validation_tokens": args.validation_tokens, "validation_modulus": 100,
              "seed": args.seed, "shard_rows": args.shard_rows, "open_rows": args.open_rows,
              "batch_size": args.batch_size, "batch_chars": args.batch_chars,
              "max_document_chars": args.max_document_chars,
              "tokenizers_version": tokenizers.__version__, "datasets_version": datasets.__version__,
              "budget_unit": "nonpadding tokens including BOS/EOS; overshoot < seq_len per split",
              "packing": "bounded best-fit, BOS/EOS per document chunk, no Q-aware reordering",
              "split_policy": "BLAKE2b-64 of exact text modulo 100; zero = validation"}

    def row_loader(filename):
        return load_dataset("parquet", data_files=[f"hf://datasets/{dataset}@{revision}/{filename}"],
                            split="train", streaming=True)

    manifest = build(args.output_dir, tokenizer, config, source_files, row_loader)
    destination = args.output_dir / "tokenizer.json"
    if not destination.exists():
        shutil.copyfile(args.tokenizer, destination)
    elif digest_file(destination) != config["tokenizer_sha256"]:
        raise ValueError("output tokenizer.json does not match the manifest")
    print(json.dumps({"output": str(args.output_dir), "train": manifest["splits"]["train"],
                      "validation": manifest["splits"]["validation"]}, indent=2))


if __name__ == "__main__":
    main()
