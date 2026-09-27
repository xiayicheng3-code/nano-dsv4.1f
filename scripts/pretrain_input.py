"""Bounded CPU prefetch and corpus integrity checks for production pretraining."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json

import numpy as np

from prepare_pretrain_corpus import digest_file, iter_pretrain_batches
from prepare_profile_data import find_corpus, validate_tokenizer


def inspect_corpus(corpus, tokenizer, *, verify_checksums=True):
    root, manifest = find_corpus(corpus)
    token_root = Path(tokenizer)
    matches = sorted(token_root.rglob("tokenizer.json")) if token_root.is_dir() else [token_root]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one tokenizer.json under {tokenizer}")
    tokenizer_info = validate_tokenizer(matches[0], manifest["config"])
    digest = hashlib.sha256((root / "manifest.json").read_bytes())
    shard_count = 0
    counts = {s: 0 for s in ("train", "validation")}
    for part in manifest["parts"]:
        directory = root / part
        raw = (directory / "manifest.json").read_bytes()
        digest.update(raw)
        meta = json.loads(raw)
        for split in counts:
            for shard in meta["splits"][split]["shards"]:
                counts[split] += shard["rows"]
                arrays = {}
                for key, info in shard["files"].items():
                    file = directory / split / info["file"]
                    if file.stat().st_size != info["bytes"]:
                        raise ValueError(f"Corpus size mismatch: {file}")
                    if verify_checksums and digest_file(file) != info["sha256"]:
                        raise ValueError(f"Corpus checksum mismatch: {file}")
                    arrays[key] = np.load(file, mmap_mode="r", allow_pickle=False)
                if arrays["tokens"].shape != (shard["rows"], 8192):
                    raise ValueError("Corpus token shape mismatch")
                offsets = arrays["offsets"]
                if (offsets.shape != (shard["rows"] + 1,) or offsets[0] != 0 or
                        offsets[-1] != len(arrays["lengths"]) or np.any(np.diff(offsets.astype(np.int64)) <= 0)):
                    raise ValueError("Corpus offsets mismatch")
                shard_count += 1
                if shard_count % 100 == 0:
                    print(f"Verified {shard_count} corpus shards", flush=True)
    for split, rows in counts.items():
        if rows != manifest["splits"][split]["rows"] or rows < 4:
            raise ValueError(f"Corpus row count mismatch/insufficient rows: {split}")
    return root, manifest, {"manifest_sha256": digest.hexdigest(),
                            "tokenizer_sha256": tokenizer_info["sha256"]}


def batch_counts(batch, *, vocab_size=32768):
    ids, seg, mask = (batch[k] for k in ("input_ids", "segment_ids", "token_mask"))
    if ids.shape != (4, 8192) or seg.shape != ids.shape or mask.shape != ids.shape:
        raise ValueError("Production batches must be four 8192-token rows")
    if ids.min() < 0 or ids.max() >= vocab_size or np.any(ids[~mask] != 2):
        raise ValueError("Invalid token IDs or padding")
    if not np.array_equal(seg[:, ::2], seg[:, 1::2]):
        raise ValueError("Segment boundaries must align to compression pairs")
    lm = int((mask[:, 1:] & mask[:, :-1] & (seg[:, 1:] == seg[:, :-1])).sum())
    if lm <= 0:
        raise ValueError("Batch has no valid LM targets")
    return {"real_tokens": int(mask.sum()), "lm_tokens": lm, "physical_tokens": int(ids.size)}


class Prefetch:
    """One future batch in a single CPU thread; checkpoints track consumed batches only."""
    def __init__(self, iterator):
        self.iterator = iter(iterator)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="corpus")
        self.future = self.pool.submit(next, self.iterator, None)

    def __iter__(self):
        return self

    def __next__(self):
        result = self.future.result()
        if result is None:
            raise StopIteration
        self.future = self.pool.submit(next, self.iterator, None)
        return result

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
