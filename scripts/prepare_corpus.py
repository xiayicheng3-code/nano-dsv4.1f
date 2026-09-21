#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Iterable

import numpy as np

from nano_dsv41f.chat_protocol import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    nano_v41_tokenizer_contract,
)
from nano_dsv41f.corpus_curriculum import (
    DEFAULT_PHASES,
    DEFAULT_Q_BAND_EDGES,
    DEFAULT_Q_THRESHOLD,
    DEFAULT_QUERY_BUDGET,
    SOURCE_CATALOG,
    aggregate_q_metrics,
    pack_length_indices,
    phase_by_name,
    phase_target_rows,
    q_row_metrics,
    source_token_targets,
)
from nano_dsv41f.sequence_packing import pack_token_sequences


@dataclass
class Segment:
    tokens: np.ndarray
    source: str
    doc_key: str


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_license(value: object) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def row_is_allowed(source, row: dict) -> bool:
    if not source.license_field or not source.allowed_licenses:
        return True
    value = normalize_license(row.get(source.license_field))
    return value in source.allowed_licenses


def row_identity(source_key: str, row: dict, text: str) -> str:
    for key in ("id", "url", "hash"):
        value = row.get(key)
        if value not in (None, ""):
            return f"{source_key}:{key}:{value}"
    if row.get("repo_name") and row.get("path"):
        return f"{source_key}:repo:{row['repo_name']}:{row['path']}"
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=12).hexdigest()
    return f"{source_key}:text:{digest}"


def text_digest(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


def token_chunks(tokenizer, text: str, *, seq_len: int) -> Iterable[np.ndarray]:
    if seq_len < 4:
        raise ValueError("seq_len must leave room for BOS/content/EOS")
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    if not ids:
        return
    max_content = seq_len - 2
    for start in range(0, len(ids), max_content):
        content = ids[start : start + max_content]
        if not content:
            continue
        out = np.empty(len(content) + 2, dtype=np.uint16)
        out[0] = BOS_TOKEN_ID
        out[1:-1] = np.asarray(content, dtype=np.uint16)
        out[-1] = EOS_TOKEN_ID
        yield out


def load_stream(source, *, seed: int, shuffle_buffer: int):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Corpus preparation needs datasets + tokenizers: "
            "pip install -e '.[data]'"
        ) from exc
    kwargs = {
        "path": source.dataset,
        "split": source.split,
        "streaming": True,
    }
    if source.config is not None:
        kwargs["name"] = source.config
    stream = load_dataset(**kwargs)
    return stream.shuffle(seed=seed, buffer_size=shuffle_buffer)


def collect_source_segments(
    source_key: str,
    *,
    target_tokens: int,
    tokenizer,
    seq_len: int,
    seed: int,
    shuffle_buffer: int,
    seen_text: set[bytes],
) -> tuple[list[Segment], dict[str, int]]:
    source = SOURCE_CATALOG[source_key]
    stream = load_stream(source, seed=seed, shuffle_buffer=shuffle_buffer)
    segments: list[Segment] = []
    real_tokens = 0
    rows_seen = 0
    rows_license_dropped = 0
    rows_empty = 0
    rows_duplicate = 0

    for row in stream:
        rows_seen += 1
        if not row_is_allowed(source, row):
            rows_license_dropped += 1
            continue
        text = row.get(source.text_field)
        if not isinstance(text, str) or not text.strip():
            rows_empty += 1
            continue
        digest = text_digest(text)
        if digest in seen_text:
            rows_duplicate += 1
            continue
        seen_text.add(digest)
        doc_key = row_identity(source_key, row, text)
        produced = False
        for chunk_index, ids in enumerate(
            token_chunks(tokenizer, text, seq_len=seq_len)
        ):
            segments.append(
                Segment(
                    tokens=ids,
                    source=source_key,
                    doc_key=f"{doc_key}:chunk:{chunk_index}",
                )
            )
            real_tokens += int(ids.size)
            produced = True
            if real_tokens >= target_tokens:
                break
        if real_tokens >= target_tokens:
            break
        if not produced:
            rows_empty += 1

    if real_tokens < target_tokens:
        raise RuntimeError(
            f"{source_key} exhausted after {real_tokens:,} tokens; "
            f"target was {target_tokens:,}"
        )
    return segments, {
        "rows_seen": rows_seen,
        "rows_license_dropped": rows_license_dropped,
        "rows_empty": rows_empty,
        "rows_duplicate": rows_duplicate,
        "segments": len(segments),
        "real_tokens": real_tokens,
        "target_tokens": target_tokens,
    }


def write_shards(
    phase_dir: Path,
    *,
    rows: list[list[int]],
    segments: list[Segment],
    seq_len: int,
    shard_rows: int,
    compression_ratio: int,
    query_budget: int,
    q_threshold: int,
    q_band_edges: tuple[int, ...],
    source_id: dict[str, int],
) -> list[dict[str, object]]:
    phase_dir.mkdir(parents=True, exist_ok=True)
    shard_meta: list[dict[str, object]] = []
    for shard_index, start in enumerate(range(0, len(rows), shard_rows)):
        row_group = rows[start : start + shard_rows]
        n = len(row_group)
        input_ids = np.empty((n, seq_len), dtype=np.uint16)
        segment_ids = np.empty((n, seq_len), dtype=np.uint16)
        token_mask = np.empty((n, seq_len), dtype=np.uint8)
        source_ids = np.empty((n, seq_len), dtype=np.uint8)
        eligible_q = np.empty((n,), dtype=np.uint16)
        selected_q = np.empty((n,), dtype=np.uint16)
        q_budget_utilization = np.empty((n,), dtype=np.float32)
        q_eligible_coverage = np.empty((n,), dtype=np.float32)
        q_band_counts = np.empty((n, len(q_band_edges) - 1), dtype=np.uint16)
        q_expected_selected = np.empty(
            (n, len(q_band_edges) - 1), dtype=np.float32
        )

        for local_row, row in enumerate(row_group):
            seqs = [segments[i].tokens for i in row]
            packed = pack_token_sequences(
                seqs,
                seq_len=seq_len,
                pad_token_id=PAD_TOKEN_ID,
                compression_ratio=compression_ratio,
            )
            input_ids[local_row] = packed.input_ids[0].astype(np.uint16)
            segment_ids[local_row] = packed.segment_ids[0].astype(np.uint16)
            token_mask[local_row] = packed.token_mask[0].astype(np.uint8)
            src_row = np.zeros((seq_len,), dtype=np.uint8)
            for seg_id, segment_index in enumerate(row):
                src_row[packed.segment_ids[0] == seg_id] = source_id[
                    segments[segment_index].source
                ]
            source_ids[local_row] = src_row

            metrics = q_row_metrics(
                packed.real_lengths,
                query_budget=query_budget,
                q_threshold=q_threshold,
                band_edges=q_band_edges,
            )
            eligible_q[local_row] = metrics.eligible_q
            selected_q[local_row] = metrics.selected_q
            q_budget_utilization[local_row] = metrics.budget_utilization
            q_eligible_coverage[local_row] = metrics.eligible_coverage
            q_band_counts[local_row] = metrics.band_counts
            q_expected_selected[local_row] = metrics.expected_selected_by_band

        path = phase_dir / f"shard-{shard_index:05d}.npz"
        np.savez_compressed(
            path,
            input_ids=input_ids,
            segment_ids=segment_ids,
            token_mask=token_mask,
            source_ids=source_ids,
            eligible_q=eligible_q,
            selected_q=selected_q,
            q_budget_utilization=q_budget_utilization,
            q_eligible_coverage=q_eligible_coverage,
            q_band_counts=q_band_counts,
            q_expected_selected=q_expected_selected,
        )
        shard_meta.append(
            {
                "file": path.name,
                "rows": n,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return shard_meta


def prepare_phase(
    phase,
    *,
    tokenizer,
    tokenizer_path: Path,
    output_dir: Path,
    total_steps: int,
    seq_len: int,
    headroom: float,
    shard_rows: int,
    compression_ratio: int,
    query_budget: int,
    q_threshold: int,
    q_band_edges: tuple[int, ...],
    candidate_window: int,
    seed: int,
    shuffle_buffer: int,
    seen_text: set[bytes],
) -> dict[str, object]:
    target_rows = phase_target_rows(
        phase, total_steps=total_steps, headroom=headroom
    )
    targets = source_token_targets(
        phase,
        total_steps=total_steps,
        seq_len=seq_len,
        headroom=headroom,
    )
    segments: list[Segment] = []
    collection: dict[str, object] = {}
    for source_offset, (source_key, _) in enumerate(phase.source_weights):
        source_segments, stats = collect_source_segments(
            source_key,
            target_tokens=targets[source_key],
            tokenizer=tokenizer,
            seq_len=seq_len,
            seed=(
                seed
                + 1009 * source_offset
                + 7919 * int(phase.start_fraction * 100)
            ),
            shuffle_buffer=shuffle_buffer,
            seen_text=seen_text,
        )
        segments.extend(source_segments)
        collection[source_key] = stats
        print(
            phase.name,
            source_key,
            f"{stats['real_tokens']:,} tokens",
            f"{stats['segments']:,} segments",
        )

    lengths = [int(segment.tokens.size) for segment in segments]
    rows = pack_length_indices(
        lengths,
        seq_len=seq_len,
        alignment=compression_ratio,
        q_aware=phase.q_aware_packing,
        query_budget=query_budget,
        q_threshold=q_threshold,
        band_edges=q_band_edges,
        candidate_window=candidate_window,
        seed=seed + 17,
    )
    if len(rows) < target_rows:
        raise RuntimeError(
            f"{phase.name}: only {len(rows):,} packed rows for target "
            f"{target_rows:,}; increase --headroom"
        )

    # Random trimming makes source mix drift much less than taking the first N packing rows.
    rng = random.Random(seed + 29)
    rng.shuffle(rows)
    rows = rows[:target_rows]

    actual_source_tokens: Counter[str] = Counter()
    for row in rows:
        for index in row:
            actual_source_tokens[segments[index].source] += int(
                segments[index].tokens.size
            )
    actual_real = sum(actual_source_tokens.values())
    actual_weights = {
        source: actual_source_tokens[source] / actual_real
        for source, _ in phase.source_weights
    }
    q_summary = aggregate_q_metrics(
        rows,
        lengths,
        query_budget=query_budget,
        q_threshold=q_threshold,
        band_edges=q_band_edges,
    )

    phase_dir = output_dir / phase.name
    source_id = {
        source: i for i, (source, _) in enumerate(phase.source_weights)
    }
    shards = write_shards(
        phase_dir,
        rows=rows,
        segments=segments,
        seq_len=seq_len,
        shard_rows=shard_rows,
        compression_ratio=compression_ratio,
        query_budget=query_budget,
        q_threshold=q_threshold,
        q_band_edges=q_band_edges,
        source_id=source_id,
    )
    manifest = {
        "format": "nano-dsv41f-packed-corpus-v1",
        "phase": {
            "name": phase.name,
            "start_fraction": phase.start_fraction,
            "end_fraction": phase.end_fraction,
            "q_aware_packing": phase.q_aware_packing,
            "source_weights_target": dict(phase.source_weights),
            "source_weights_actual_real_tokens": actual_weights,
        },
        "training": {
            "total_steps": total_steps,
            "seq_len": seq_len,
            "target_rows_with_headroom": target_rows,
            "headroom": headroom,
            "compression_alignment": compression_ratio,
        },
        "tokenizer": {
            "path_name": tokenizer_path.name,
            "sha256": sha256_file(tokenizer_path),
            "vocab_size": tokenizer.get_vocab_size(with_added_tokens=True),
            "bos_token_id": BOS_TOKEN_ID,
            "eos_token_id": EOS_TOKEN_ID,
            "pad_token_id": PAD_TOKEN_ID,
        },
        "query_packing": {
            "query_budget": query_budget,
            "q_threshold": q_threshold,
            "q_band_edges": list(q_band_edges),
            **q_summary,
        },
        "sources": {
            key: {
                **asdict(SOURCE_CATALOG[key]),
                "target_tokens": targets[key],
                "collection": collection[key],
                "source_id": source_id[key],
            }
            for key, _ in phase.source_weights
        },
        "packed": {
            "rows": len(rows),
            "real_tokens": actual_real,
            "physical_tokens": len(rows) * seq_len,
            "real_token_utilization": actual_real / (len(rows) * seq_len),
            "shards": shards,
        },
    }
    manifest_path = phase_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        {
            "phase": phase.name,
            "rows": len(rows),
            "real_token_utilization": round(
                manifest["packed"]["real_token_utilization"], 4
            ),
            "q_budget_utilization": round(
                q_summary["mean_budget_utilization"], 4
            ),
            "manifest": str(manifest_path),
        }
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build early/middle/late-mid nano-dsv4.1f packed LM corpora with "
            "late-mid Q-position-aware packing."
        )
    )
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=("all",) + tuple(phase.name for phase in DEFAULT_PHASES),
        default="all",
    )
    parser.add_argument("--total-steps", type=int, default=10_000)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--headroom", type=float, default=1.05)
    parser.add_argument("--shard-rows", type=int, default=256)
    parser.add_argument("--compression-ratio", type=int, default=2)
    parser.add_argument("--query-budget", type=int, default=DEFAULT_QUERY_BUDGET)
    parser.add_argument("--q-threshold", type=int, default=DEFAULT_Q_THRESHOLD)
    parser.add_argument(
        "--q-band-edges",
        default=",".join(map(str, DEFAULT_Q_BAND_EDGES)),
        help="comma-separated local-position edges; first must equal q-threshold",
    )
    parser.add_argument("--candidate-window", type=int, default=256)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()

    if args.shard_rows <= 0:
        raise SystemExit("--shard-rows must be positive")
    if args.compression_ratio <= 0 or args.seq_len % args.compression_ratio:
        raise SystemExit("--seq-len must be divisible by --compression-ratio")
    q_band_edges = tuple(int(x) for x in args.q_band_edges.split(",") if x)
    if not q_band_edges or q_band_edges[0] != args.q_threshold:
        raise SystemExit("--q-band-edges must start at --q-threshold")
    if q_band_edges[-1] < args.seq_len:
        raise SystemExit("--q-band-edges must cover --seq-len")

    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise SystemExit(
            "Corpus preparation needs tokenizers: pip install -e '.[data]'"
        ) from exc
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    contract = nano_v41_tokenizer_contract(
        tokenizer.get_vocab_size(with_added_tokens=True)
    )
    for token, expected_id in contract.token_to_id.items():
        actual = tokenizer.token_to_id(token)
        if actual != expected_id:
            raise SystemExit(
                f"tokenizer contract mismatch for {token!r}: "
                f"got {actual}, expected {expected_id}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    phases = (
        DEFAULT_PHASES
        if args.phase == "all"
        else (phase_by_name(args.phase),)
    )
    seen_text: set[bytes] = set()
    manifests = []
    for phase in phases:
        manifests.append(
            prepare_phase(
                phase,
                tokenizer=tokenizer,
                tokenizer_path=args.tokenizer,
                output_dir=args.output_dir,
                total_steps=args.total_steps,
                seq_len=args.seq_len,
                headroom=args.headroom,
                shard_rows=args.shard_rows,
                compression_ratio=args.compression_ratio,
                query_budget=args.query_budget,
                q_threshold=args.q_threshold,
                q_band_edges=q_band_edges,
                candidate_window=args.candidate_window,
                seed=args.seed,
                shuffle_buffer=args.shuffle_buffer,
                seen_text=seen_text,
            )
        )

    summary = {
        "format": "nano-dsv41f-corpus-curriculum-v1",
        "phases": [
            {
                "name": manifest["phase"]["name"],
                "rows": manifest["packed"]["rows"],
                "manifest": f"{manifest['phase']['name']}/manifest.json",
            }
            for manifest in manifests
        ],
    }
    (args.output_dir / "curriculum_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
