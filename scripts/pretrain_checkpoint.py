"""Atomic, pickle-free checkpoints for a single-host sharded JAX training run.

Store one host leaf at a time; restore against a caller-created tree/sharding template.
BF16 is stored as uint16 bits to avoid NumPy's lossy void-dtype serialization.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np

FORMAT = "nano-dsv41f-training-state-v1"


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def checkpoint_path(path):
    path = Path(path)
    if (path / "manifest.json").is_file():
        return path
    if (path / "latest.json").is_file():
        name = json.loads((path / "latest.json").read_text())["checkpoint"]
        if Path(name).name != name:
            raise ValueError("invalid checkpoint pointer")
        return path / name
    raise ValueError(f"No committed checkpoint in {path}")


def read_metadata(path):
    path = checkpoint_path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("format") != FORMAT:
        raise ValueError("unsupported training checkpoint")
    return path, manifest


def save_checkpoint(root, tree, metadata, *, keep=2):
    import jax
    if keep < 1:
        raise ValueError("keep must be positive")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    serial = uuid.uuid4().hex[:12]
    name = f"step-{metadata['completed_steps']:08d}-{serial}"
    pending = root / (name + ".pending")
    pending.mkdir()
    try:
        entries = []
        for i, (path, leaf) in enumerate(jax.tree_util.tree_flatten_with_path(tree)[0]):
            host = np.asarray(jax.device_get(leaf))
            if not np.isfinite(host).all():
                raise FloatingPointError(f"nonfinite checkpoint leaf: {path}")
            dtype = str(host.dtype)
            storage = host.view(np.uint16) if dtype == "bfloat16" else host
            file = pending / f"leaf-{i:05d}.npy"
            np.save(file, storage, allow_pickle=False)
            entries.append({"path": jax.tree_util.keystr(path), "file": file.name,
                            "shape": list(host.shape), "dtype": dtype,
                            "sha256": digest_file(file)})
        atomic_json(pending / "manifest.json", {
            "format": FORMAT, "metadata": metadata, "leaves": entries})
        pending.rename(root / name)
        atomic_json(root / "latest.json", {"checkpoint": name})
    except BaseException:
        shutil.rmtree(pending, ignore_errors=True)
        raise
    # Only committed checkpoints in this writer's own directory are eligible.
    committed = sorted((p for p in root.glob("step-*") if (p / "manifest.json").is_file()),
                       key=lambda p: p.stat().st_mtime_ns, reverse=True)
    for old in committed[keep:]:
        if old.name != name:
            shutil.rmtree(old)
    return root / name


def load_checkpoint(path, template, *, expected_identity=None):
    import jax
    import ml_dtypes
    path, manifest = read_metadata(path)
    metadata = manifest["metadata"]
    if expected_identity is not None and metadata["identity"] != expected_identity:
        raise ValueError("Checkpoint recipe, corpus, schedule or seeds differ from this run")
    pairs, treedef = jax.tree_util.tree_flatten_with_path(template)
    if len(pairs) != len(manifest["leaves"]):
        raise ValueError("Checkpoint tree leaf count differs")
    leaves = []
    for (key, target), info in zip(pairs, manifest["leaves"]):
        if (info["path"] != jax.tree_util.keystr(key) or
                info["shape"] != list(target.shape) or info["dtype"] != str(target.dtype)):
            raise ValueError(f"Checkpoint leaf contract differs: {key}")
        if Path(info["file"]).name != info["file"]:
            raise ValueError("invalid checkpoint filename")
        file = path / info["file"]
        if digest_file(file) != info["sha256"]:
            raise ValueError(f"Checkpoint checksum failed: {file.name}")
        host = np.load(file, allow_pickle=False)
        if info["dtype"] == "bfloat16":
            if host.dtype != np.uint16:
                raise ValueError("invalid BF16 checkpoint storage")
            host = host.view(ml_dtypes.bfloat16)
        if list(host.shape) != info["shape"] or str(host.dtype) != info["dtype"]:
            raise ValueError("stored leaf shape/dtype differs")
        if not np.isfinite(host).all():
            raise FloatingPointError("nonfinite checkpoint value")
        leaves.append(jax.device_put(host, target.sharding))
    tree = treedef.unflatten(leaves)
    jax.block_until_ready(tree)
    return tree, metadata
