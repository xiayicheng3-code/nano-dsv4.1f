from __future__ import annotations

from dataclasses import dataclass
from math import ceil
import random
from typing import Iterable, Sequence


DEFAULT_SEQ_LEN = 4096
DEFAULT_QUERY_BUDGET = 128
DEFAULT_Q_THRESHOLD = 640
DEFAULT_Q_BAND_EDGES = (640, 768, 1024, 1536, 2048, 3072, 4096)


@dataclass(frozen=True)
class CorpusSource:
    key: str
    dataset: str
    config: str | None
    split: str
    text_field: str
    license: str
    license_field: str | None = None
    allowed_licenses: tuple[str, ...] = ()


SOURCE_CATALOG: dict[str, CorpusSource] = {
    "fineweb_edu": CorpusSource(
        key="fineweb_edu",
        dataset="HuggingFaceFW/fineweb-edu",
        config="sample-10BT",
        split="train",
        text_field="text",
        license="odc-by",
    ),
    "cosmopedia_v2": CorpusSource(
        key="cosmopedia_v2",
        dataset="HuggingFaceTB/smollm-corpus",
        config="cosmopedia-v2",
        split="train",
        text_field="text",
        license="odc-by",
    ),
    "codeparrot_clean": CorpusSource(
        key="codeparrot_clean",
        dataset="codeparrot/codeparrot-clean",
        config=None,
        split="train",
        text_field="content",
        license="per-file",
        license_field="license",
        # Keep the default redistribution story simple. Users can broaden this in a
        # local manifest after reviewing the source license terms they are comfortable with.
        allowed_licenses=(
            "apache-2.0",
            "bsd-2-clause",
            "bsd-3-clause",
            "cc0-1.0",
            "isc",
            "mit",
            "unlicense",
        ),
    ),
    "finemath_3plus": CorpusSource(
        key="finemath_3plus",
        dataset="HuggingFaceTB/finemath",
        config="finemath-3plus",
        split="train",
        text_field="text",
        license="odc-by",
    ),
    "finemath_4plus": CorpusSource(
        key="finemath_4plus",
        dataset="HuggingFaceTB/finemath",
        config="finemath-4plus",
        split="train",
        text_field="text",
        license="odc-by",
    ),
}


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    start_fraction: float
    end_fraction: float
    source_weights: tuple[tuple[str, float], ...]
    q_aware_packing: bool = False

    def __post_init__(self) -> None:
        if not (0.0 <= self.start_fraction < self.end_fraction <= 1.0):
            raise ValueError("phase fractions must satisfy 0 <= start < end <= 1")
        if not self.source_weights:
            raise ValueError("phase needs at least one source")
        unknown = [key for key, _ in self.source_weights if key not in SOURCE_CATALOG]
        if unknown:
            raise ValueError(f"unknown corpus sources: {unknown}")
        if any(weight <= 0 for _, weight in self.source_weights):
            raise ValueError("all source weights must be positive")
        total = sum(weight for _, weight in self.source_weights)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"source weights must sum to 1.0, got {total}")

    @property
    def span(self) -> float:
        return self.end_fraction - self.start_fraction

    @property
    def weights(self) -> dict[str, float]:
        return dict(self.source_weights)


# The boundaries intentionally line up with the current retriever schedule: selective
# indexer training starts at 0.55, and late-mid becomes explicitly Q-aware before the
# 0.90 cosine-decay boundary. The same late-mid corpus can continue through decay.
DEFAULT_PHASES: tuple[PhaseSpec, ...] = (
    PhaseSpec(
        name="early",
        start_fraction=0.00,
        end_fraction=0.55,
        source_weights=(
            ("fineweb_edu", 0.72),
            ("cosmopedia_v2", 0.13),
            ("codeparrot_clean", 0.12),
            ("finemath_3plus", 0.03),
        ),
    ),
    PhaseSpec(
        name="middle",
        start_fraction=0.55,
        end_fraction=0.75,
        source_weights=(
            ("fineweb_edu", 0.60),
            ("cosmopedia_v2", 0.10),
            ("codeparrot_clean", 0.15),
            ("finemath_3plus", 0.15),
        ),
    ),
    PhaseSpec(
        name="late_mid",
        start_fraction=0.75,
        end_fraction=1.00,
        source_weights=(
            ("fineweb_edu", 0.50),
            ("cosmopedia_v2", 0.05),
            ("codeparrot_clean", 0.20),
            ("finemath_4plus", 0.25),
        ),
        q_aware_packing=True,
    ),
)


def phase_by_name(name: str) -> PhaseSpec:
    for phase in DEFAULT_PHASES:
        if phase.name == name:
            return phase
    raise KeyError(name)


def phase_target_rows(
    phase: PhaseSpec,
    *,
    total_steps: int,
    headroom: float = 1.05,
) -> int:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if headroom < 1.0:
        raise ValueError("headroom must be >= 1")
    return ceil(total_steps * phase.span * headroom)


def source_token_targets(
    phase: PhaseSpec,
    *,
    total_steps: int,
    seq_len: int = DEFAULT_SEQ_LEN,
    headroom: float = 1.05,
) -> dict[str, int]:
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    physical = phase_target_rows(
        phase, total_steps=total_steps, headroom=headroom
    ) * seq_len
    return {
        source: ceil(physical * weight)
        for source, weight in phase.source_weights
    }


def aligned_length(length: int, alignment: int) -> int:
    if length <= 0:
        raise ValueError("length must be positive")
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    return ((length + alignment - 1) // alignment) * alignment


@dataclass(frozen=True)
class QRowMetrics:
    eligible_q: int
    selected_q: int
    budget_utilization: float
    eligible_coverage: float
    band_counts: tuple[int, ...]
    expected_selected_by_band: tuple[float, ...]


def _validate_q_bands(q_threshold: int, band_edges: Sequence[int]) -> tuple[int, ...]:
    edges = tuple(int(x) for x in band_edges)
    if q_threshold < 0:
        raise ValueError("q_threshold must be non-negative")
    if len(edges) < 2 or edges[0] != q_threshold:
        raise ValueError("q band edges must start at q_threshold and contain >=2 edges")
    if any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError("q band edges must be strictly increasing")
    return edges


def q_band_counts_for_lengths(
    real_lengths: Sequence[int],
    *,
    q_threshold: int = DEFAULT_Q_THRESHOLD,
    band_edges: Sequence[int] = DEFAULT_Q_BAND_EDGES,
) -> tuple[int, ...]:
    edges = _validate_q_bands(q_threshold, band_edges)
    counts = [0] * (len(edges) - 1)
    for raw_length in real_lengths:
        length = int(raw_length)
        if length <= 0:
            raise ValueError("real lengths must be positive")
        # Eligible local query positions are q_threshold .. length-1.
        for i, (lo, hi) in enumerate(zip(edges, edges[1:])):
            counts[i] += max(min(length, hi) - lo, 0)
    return tuple(counts)


def q_row_metrics(
    real_lengths: Sequence[int],
    *,
    query_budget: int = DEFAULT_QUERY_BUDGET,
    q_threshold: int = DEFAULT_Q_THRESHOLD,
    band_edges: Sequence[int] = DEFAULT_Q_BAND_EDGES,
) -> QRowMetrics:
    if query_budget <= 0:
        raise ValueError("query_budget must be positive")
    counts = q_band_counts_for_lengths(
        real_lengths, q_threshold=q_threshold, band_edges=band_edges
    )
    # Include eligible positions above the final reporting edge in budget accounting.
    # With the default 4K segments and final edge 4096 this is normally zero.
    eligible = sum(max(int(length) - q_threshold, 0) for length in real_lengths)
    selected = min(query_budget, eligible)
    utilization = selected / query_budget
    coverage = selected / eligible if eligible else 0.0
    sample_fraction = selected / eligible if eligible else 0.0
    expected = tuple(count * sample_fraction for count in counts)
    return QRowMetrics(
        eligible_q=eligible,
        selected_q=selected,
        budget_utilization=utilization,
        eligible_coverage=coverage,
        band_counts=counts,
        expected_selected_by_band=expected,
    )


def expected_q_density(
    expected_selected_by_band: Sequence[float],
    *,
    band_edges: Sequence[int] = DEFAULT_Q_BAND_EDGES,
) -> tuple[float, ...]:
    edges = tuple(band_edges)
    if len(expected_selected_by_band) != len(edges) - 1:
        raise ValueError("band count does not match band edges")
    return tuple(
        float(value) / (hi - lo)
        for value, lo, hi in zip(expected_selected_by_band, edges, edges[1:])
    )


def q_balance_score(
    cumulative_expected: Sequence[float],
    row_metrics: QRowMetrics,
    *,
    band_edges: Sequence[int] = DEFAULT_Q_BAND_EDGES,
    utilization_weight: float = 2.0,
    coverage_weight: float = 0.15,
) -> float:
    """Lower is better: uniform per-position Q exposure plus full budget use.

    We normalize by band width, so a 1024-position band is not expected to receive the
    same total number of sampled Qs as a 128-position band. The objective is roughly
    uniform expected samples *per local position* across late-mid rows.
    """
    if utilization_weight < 0 or coverage_weight < 0:
        raise ValueError("score weights must be non-negative")
    if len(cumulative_expected) != len(row_metrics.expected_selected_by_band):
        raise ValueError("cumulative band shape mismatch")
    combined = tuple(
        float(a) + float(b)
        for a, b in zip(cumulative_expected, row_metrics.expected_selected_by_band)
    )
    density = expected_q_density(combined, band_edges=band_edges)
    mean = sum(density) / len(density)
    if mean:
        imbalance = (
            sum((x - mean) ** 2 for x in density) / len(density)
        ) ** 0.5 / mean
    else:
        imbalance = 1.0
    utilization_penalty = 1.0 - row_metrics.budget_utilization
    # Some oversupply is necessary to reach later Q positions. Keep this deliberately
    # light: it mainly breaks ties in favor of supervising more of the eligible Q pool.
    coverage_penalty = (
        1.0 - row_metrics.eligible_coverage if row_metrics.eligible_q else 1.0
    )
    return (
        imbalance
        + utilization_weight * utilization_penalty
        + coverage_weight * coverage_penalty
    )


def pack_length_indices(
    lengths: Sequence[int],
    *,
    seq_len: int = DEFAULT_SEQ_LEN,
    alignment: int = 2,
    q_aware: bool = False,
    query_budget: int = DEFAULT_QUERY_BUDGET,
    q_threshold: int = DEFAULT_Q_THRESHOLD,
    band_edges: Sequence[int] = DEFAULT_Q_BAND_EDGES,
    candidate_window: int = 256,
    seed: int = 0,
) -> list[list[int]]:
    """Windowed best-fit packing over complete tokenized segments.

    In Q-aware mode, each row first chooses an anchor from the current candidate window
    using expected sampled-Q position balance. Remaining space is filled by best fit,
    preferring non-Q-bearing segments so the anchor's Q profile is not accidentally
    swamped by unrelated long documents.
    """
    if seq_len <= 0 or alignment <= 0 or seq_len % alignment:
        raise ValueError("seq_len must be positive and divisible by alignment")
    if candidate_window <= 0:
        raise ValueError("candidate_window must be positive")
    if not lengths:
        return []
    physical = [aligned_length(int(length), alignment) for length in lengths]
    if any(length > seq_len for length in physical):
        raise ValueError("all segments must fit in seq_len before packing")

    rng = random.Random(seed)
    pending = list(range(len(lengths)))
    rng.shuffle(pending)
    rows: list[list[int]] = []
    cumulative_expected = [0.0] * (len(tuple(band_edges)) - 1)

    while pending:
        window_n = min(candidate_window, len(pending))
        if q_aware:
            best_pos = 0
            best_key: tuple[float, int, int] | None = None
            for pos in range(window_n):
                idx = pending[pos]
                metrics = q_row_metrics(
                    [lengths[idx]],
                    query_budget=query_budget,
                    q_threshold=q_threshold,
                    band_edges=band_edges,
                )
                score = q_balance_score(
                    cumulative_expected, metrics, band_edges=band_edges
                )
                key = (
                    score,
                    -min(metrics.eligible_q, query_budget),
                    -physical[idx],
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best_pos = pos
            anchor = pending.pop(best_pos)
        else:
            # Largest item in the window tends to reduce fragmentation.
            best_pos = max(
                range(window_n), key=lambda pos: physical[pending[pos]]
            )
            anchor = pending.pop(best_pos)

        row = [anchor]
        remaining = seq_len - physical[anchor]

        while pending and remaining >= alignment:
            window_n = min(candidate_window, len(pending))
            fitting = [
                pos
                for pos in range(window_n)
                if physical[pending[pos]] <= remaining
            ]
            if not fitting:
                break
            if q_aware:
                non_q = [
                    pos
                    for pos in fitting
                    if int(lengths[pending[pos]]) <= q_threshold
                ]
                pool = non_q or fitting
            else:
                pool = fitting
            best_pos = max(pool, key=lambda pos: physical[pending[pos]])
            idx = pending.pop(best_pos)
            row.append(idx)
            remaining -= physical[idx]

        rows.append(row)
        if q_aware:
            metrics = q_row_metrics(
                [lengths[i] for i in row],
                query_budget=query_budget,
                q_threshold=q_threshold,
                band_edges=band_edges,
            )
            for j, value in enumerate(metrics.expected_selected_by_band):
                cumulative_expected[j] += value

    return rows


def aggregate_q_metrics(
    rows: Iterable[Sequence[int]],
    lengths: Sequence[int],
    *,
    query_budget: int = DEFAULT_QUERY_BUDGET,
    q_threshold: int = DEFAULT_Q_THRESHOLD,
    band_edges: Sequence[int] = DEFAULT_Q_BAND_EDGES,
) -> dict[str, object]:
    row_metrics = [
        q_row_metrics(
            [lengths[i] for i in row],
            query_budget=query_budget,
            q_threshold=q_threshold,
            band_edges=band_edges,
        )
        for row in rows
    ]
    n_bands = len(tuple(band_edges)) - 1
    if not row_metrics:
        return {
            "rows": 0,
            "eligible_q": 0,
            "selected_q": 0,
            "mean_budget_utilization": 0.0,
            "mean_eligible_coverage": 0.0,
            "expected_selected_by_band": [0.0] * n_bands,
            "expected_q_density": [0.0] * n_bands,
        }
    expected = [
        sum(m.expected_selected_by_band[i] for m in row_metrics)
        for i in range(n_bands)
    ]
    return {
        "rows": len(row_metrics),
        "eligible_q": sum(m.eligible_q for m in row_metrics),
        "selected_q": sum(m.selected_q for m in row_metrics),
        "mean_budget_utilization": (
            sum(m.budget_utilization for m in row_metrics) / len(row_metrics)
        ),
        "mean_eligible_coverage": (
            sum(m.eligible_coverage for m in row_metrics) / len(row_metrics)
        ),
        "expected_selected_by_band": expected,
        "expected_q_density": list(
            expected_q_density(expected, band_edges=band_edges)
        ),
    }
