"""CPU-only data-path checks: packing, budget, restart, and holdout invariants."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("pretrain_data", Path(__file__).parents[1] / "scripts/prepare_pretrain_corpus.py")
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)


class FakeTokenizer:
    def __init__(self):
        self.calls = 0

    def encode_batch(self, texts, **kwargs):
        self.calls += 1
        return [SimpleNamespace(ids=[3 + ord(c) % 20 for c in text]) for text in texts]


def config(**changes):
    value = dict(seq_len=16, alignment=2, shard_rows=2, open_rows=2, batch_size=3,
                 batch_chars=100, max_document_chars=1000, validation_modulus=4,
                 train_tokens=130, validation_tokens=30)
    return value | changes


def records(count=100, offset=0):
    return [{"text": f"document {i}"} for i in range(offset, count + offset)]


def assert_batch(batch):
    ids, segments, mask = [batch[k] for k in ("input_ids", "segment_ids", "token_mask")]
    assert ids.dtype == np.int32 and mask.dtype == bool
    # No compression pair spans a document boundary, even with odd document lengths.
    assert np.array_equal(segments[:, ::2], segments[:, 1::2])
    for i in range(len(ids)):
        for segment in np.unique(segments[i]):
            actual = ids[i][(segments[i] == segment) & mask[i]]
            assert actual[0] == data.BOS and actual[-1] == data.EOS
        assert np.all(ids[i][~mask[i]] == data.PAD)


def test_pack_restore_long_documents_and_masks(tmp_path):
    writer = data.CompactWriter(tmp_path, seq_len=16, shard_rows=2, open_rows=3)
    documents = [[7] * size for size in (1, 7, 28, 3, 4, 10, 1)]
    expected_lm = expected_real = 0
    for doc in documents:
        for chunk in data.token_chunks(doc, 16):
            expected_real += len(chunk)
            expected_lm += len(chunk) - 1
            writer.add(chunk)
    meta = writer.finish()
    real = lm = 0
    for shard in meta["shards"]:
        arrays = {k: np.load(tmp_path / v["file"]) for k, v in shard["files"].items()}
        batch = data.restore_rows(**arrays)
        assert_batch(batch)
        mask, segments = batch["token_mask"], batch["segment_ids"]
        real += int(mask.sum())
        lm += int((mask[:, :-1] & mask[:, 1:] & (segments[:, :-1] == segments[:, 1:])).sum())
    assert real == expected_real == meta["counts"]["real_tokens"]
    assert lm == expected_lm == meta["counts"]["lm_tokens"]


def test_budget_resume_and_training_reader(tmp_path):
    cfg = config()
    tokenizer = FakeTokenizer()
    manifest = data.build(tmp_path, tokenizer, cfg, ["a", "b"], lambda _: iter(records()))
    assert manifest["complete"]
    for split in ("train", "validation"):
        n = manifest["splits"][split]["real_tokens"]
        assert cfg[f"{split}_tokens"] <= n < cfg[f"{split}_tokens"] + cfg["seq_len"]
        batches = list(data.iter_pretrain_batches(tmp_path, split=split, batch_rows=3, drop_last=False))
        for batch in batches:
            assert_batch(batch)
        assert sum(int(b["token_mask"].sum()) for b in batches) == n
        assert sum(len(b["input_ids"]) for b in batches) == manifest["splits"][split]["rows"]
    calls = tokenizer.calls
    def must_not_read(_):
        raise AssertionError("completed source should not be downloaded or tokenized again")
    assert data.build(tmp_path, tokenizer, cfg, ["a", "b"], must_not_read) == manifest
    assert tokenizer.calls == calls
    with pytest.raises(ValueError, match="different recipe"):
        data.build(tmp_path, tokenizer, cfg | {"train_tokens": 131}, ["a", "b"], must_not_read)


def test_interrupted_part_resumes_without_double_counting(tmp_path):
    cfg = config(train_tokens=400, validation_tokens=80)
    def interrupted(name):
        if name == "a":
            return iter(records(8))
        def rows():
            yield from records(3, offset=100)
            raise RuntimeError("network interrupted")
        return rows()
    with pytest.raises(RuntimeError, match="network interrupted"):
        data.build(tmp_path, FakeTokenizer(), cfg, ["a", "b"], interrupted)
    first = json.loads((tmp_path / "parts/part-00000/manifest.json").read_text())
    loaded = []
    def resume(name):
        loaded.append(name)
        return iter(records(100, offset=100))
    manifest = data.build(tmp_path, FakeTokenizer(), cfg, ["a", "b"], resume)
    assert loaded == ["b"]
    assert first == json.loads((tmp_path / "parts/part-00000/manifest.json").read_text())
    assert manifest["complete"]
    assert manifest["splits"]["train"]["real_tokens"] < 416


def test_corruption_and_exhaustion_are_errors(tmp_path):
    cfg = config()
    with pytest.raises(RuntimeError, match="exhausted"):
        data.build(tmp_path, FakeTokenizer(), cfg, ["a"], lambda _: iter(records(2)))
    with pytest.raises(ValueError, match="completed"):
        next(data.iter_pretrain_batches(tmp_path))
    file = next(tmp_path.rglob("*-tokens.npy"))
    with file.open("r+b") as f:
        f.seek(-1, 2)
        f.write(b"x")
    with pytest.raises(ValueError, match="corrupt"):
        data.build(tmp_path, FakeTokenizer(), cfg, ["a"], lambda _: iter(records()))


def test_duplicate_documents_and_long_chunks_cannot_cross_holdout(tmp_path):
    class IdentityTokenizer:
        def encode_batch(self, texts, **kwargs):
            # Each document has a unique content ID and spans multiple packed rows.
            return [SimpleNamespace(ids=[100 + int(text)] * 40) for text in texts]
    text_rows = [{"text": str(i)} for i in range(100)]
    cfg = config(train_tokens=1_000_000, validation_tokens=1_000_000)
    present = {"train": set(), "validation": set()}
    for name, rows in (("a", text_rows), ("b", list(reversed(text_rows)))):
        meta = data.prepare_part(iter(rows), IdentityTokenizer(), tmp_path / name, cfg,
                                 {s: 1_000_000 for s in present})
        for split in present:
            for shard in meta["splits"][split]["shards"]:
                ids = np.load(tmp_path / name / split / shard["files"]["tokens"]["file"])
                present[split].update(int(n) for n in ids.flat if n >= 100)
    assert present["train"] and present["validation"]
    assert not present["train"].intersection(present["validation"])
    assert present["train"] | present["validation"] == set(range(100, 200))


def test_real_tokenizer_contract_and_batched_encoding(tmp_path):
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    vocab = {token: i for i, token in enumerate(data._CONTRACT["SPECIAL_TOKENS"])}
    vocab.update({"[UNK]": len(vocab), "word": len(vocab) + 1})
    tokenizer = tokenizers.Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    texts = [{"text": "word " * 9 + str(i)} for i in range(200)]
    manifest = data.build(tmp_path, tokenizer, config(), ["fixture.parquet"], lambda _: iter(texts))
    assert manifest["complete"]
    for batch in data.iter_pretrain_batches(tmp_path, batch_rows=2, drop_last=False):
        assert_batch(batch)


def test_parquet_stream_with_actual_datasets(tmp_path):
    datasets = pytest.importorskip("datasets")
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = tmp_path / "fixture.parquet"
    pq.write_table(pa.Table.from_pylist(records(100)), path)
    def load(_):
        return datasets.load_dataset("parquet", data_files=[str(path)], split="train", streaming=True)
    manifest = data.build(tmp_path / "output", FakeTokenizer(), config(), [str(path)], load)
    assert manifest["complete"]
