"""Sample compact pretraining shards for profiling; no JAX or tokenizer dependency."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np

FORMAT = "nano-dsv41f-pretrain-compact-v1"
KEYS = ("input_ids", "segment_ids", "token_mask")


def find_corpus(root):
    root = Path(root)
    candidates = []
    for path in ([root] if root.is_file() else sorted(root.rglob("manifest.json"))):
        meta = json.loads(path.read_text())
        if meta.get("format") == FORMAT and "parts" in meta:
            candidates.append((path.parent, meta))
    if len(candidates) != 1:
        raise ValueError(f"Expected one {FORMAT} corpus under {root}; found {len(candidates)}")
    directory, meta = candidates[0]
    if not meta.get("complete"):
        raise ValueError("Corpus manifest is incomplete")
    if meta["config"]["seq_len"] != 8192 or meta["config"]["alignment"] != 2:
        raise ValueError("Profiling requires 8192-token rows with alignment=2")
    return directory, meta


def validate_tokenizer(root, config):
    path = Path(root)
    path = path / "tokenizer.json" if path.is_dir() else path
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != config["tokenizer_sha256"]:
        raise ValueError("Tokenizer SHA-256 differs from the corpus tokenizer")
    data = json.loads(path.read_text())
    vocab = dict(data["model"]["vocab"])
    vocab.update({item["content"]: item["id"] for item in data.get("added_tokens", [])})
    contract = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                                 "src/nano_dsv41f/chat_protocol.py"))
    if len(vocab) != 32768 or config["vocab_size"] != 32768:
        raise ValueError("Tokenizer must have the recipe's 32768-token vocabulary")
    for token, expected in contract["nano_v41_tokenizer_contract"]().token_to_id.items():
        if vocab.get(token) != expected:
            raise ValueError(f"Special token mismatch: {token}")
    return {"path": str(path), "sha256": digest, "vocab_size": len(vocab)}


def sample_rows(root, manifest, count, seed):
    """Uniform global-row sample without replacement, reading only selected NPY pages."""
    shards = []
    for part in manifest["parts"]:
        directory = root / part
        meta = json.loads((directory / "manifest.json").read_text())
        shards.extend((directory / "train", s) for s in meta["splits"]["train"]["shards"])
    ends = np.cumsum([s["rows"] for _, s in shards])
    if not len(ends) or count > ends[-1] or count <= 0:
        raise ValueError("Not enough training rows for the requested sample")
    selected = np.random.default_rng(seed).choice(int(ends[-1]), count, replace=False)
    shard_ids = np.searchsorted(ends, selected, side="right")
    ids = np.empty((count, 8192), dtype=np.int32)
    seg = np.empty_like(ids)
    mask = np.zeros_like(ids, dtype=bool)
    provenance = [None] * count
    for shard_id in np.unique(shard_ids):
        directory, shard = shards[shard_id]
        arrays = {k: np.load(directory / v["file"], mmap_mode="r", allow_pickle=False)
                  for k, v in shard["files"].items()}
        if arrays["tokens"].shape != (shard["rows"], 8192):
            raise ValueError("Invalid token shard shape")
        for dest in np.flatnonzero(shard_ids == shard_id):
            row = int(selected[dest] - (ends[shard_id - 1] if shard_id else 0))
            lo, hi = map(int, arrays["offsets"][row:row + 2])
            lengths = np.asarray(arrays["lengths"][lo:hi], dtype=np.int32)
            physical = ((lengths + 1) // 2) * 2
            if not len(lengths) or np.any(lengths < 3) or physical.sum() > 8192:
                raise ValueError("Invalid compact segment lengths")
            physical[-1] += 8192 - int(physical.sum())
            ids[dest] = arrays["tokens"][row]
            seg[dest] = np.repeat(np.arange(len(lengths)), physical)
            for start, length in zip(np.cumsum(physical) - physical, lengths):
                mask[dest, start:start + length] = True
                if ids[dest, start] != 0 or ids[dest, start + length - 1] != 1:
                    raise ValueError("Packed segment lacks expected BOS/EOS")
            provenance[dest] = {"file": str(directory / shard["files"]["tokens"]["file"]),
                                "row": row, "global_row": int(selected[dest])}
        del arrays
    if ids.min() < 0 or ids.max() >= manifest["config"]["vocab_size"]:
        raise ValueError("Corpus contains out-of-vocabulary IDs")
    if np.any(ids[~mask] != 2):
        raise ValueError("Corpus padding differs from tokenizer PAD=2")
    return dict(zip(KEYS, (ids, seg, mask))), provenance


def prepare(corpus, tokenizer, output, *, rows=(4, 8, 24), batches=3, seed=1701):
    if batches < 1 or not rows or any(r <= 0 or r % 4 for r in rows):
        raise ValueError("Use positive batches and positive row counts divisible by DP4")
    root, manifest = find_corpus(corpus)
    tokenizer_info = validate_tokenizer(tokenizer, manifest["config"])
    arrays, sources = sample_rows(root, manifest, max(rows) * batches, seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = {"corpus": str(root), "tokenizer": tokenizer_info, "seed": seed,
              "manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
              "batches": batches, "max_rows": max(rows), "sources": sources, "files": {}}
    for n in rows:
        # Each batch size takes a nested prefix from the same sampled batch bank.
        subset = {k: v.reshape(batches, max(rows), 8192)[:, :n].reshape(batches * n, 8192)
                  for k, v in arrays.items()}
        path = output / f"rows-{n}.npz"
        np.savez(path, **subset)
        report["files"][str(n)] = {"path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (output / "data_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--rows", default="4,8,24")
    p.add_argument("--batches", type=int, default=3)
    p.add_argument("--seed", type=int, default=1701)
    a = p.parse_args()
    result = prepare(a.corpus, a.tokenizer, a.output, rows=tuple(map(int, a.rows.split(","))),
                     batches=a.batches, seed=a.seed)
    print(json.dumps({k: v for k, v in result.items() if k != "sources"}, indent=2))
