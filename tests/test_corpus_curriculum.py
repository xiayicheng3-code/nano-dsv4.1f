import math

from nano_dsv41f.corpus_curriculum import (
    DEFAULT_PHASES,
    DEFAULT_Q_BAND_EDGES,
    aggregate_q_metrics,
    aligned_length,
    expected_q_density,
    pack_length_indices,
    phase_target_rows,
    q_row_metrics,
    source_token_targets,
)


def test_default_phase_curriculum_covers_training_once():
    assert [p.name for p in DEFAULT_PHASES] == ["early", "middle", "late_mid"]
    assert DEFAULT_PHASES[0].start_fraction == 0.0
    assert DEFAULT_PHASES[-1].end_fraction == 1.0
    for left, right in zip(DEFAULT_PHASES, DEFAULT_PHASES[1:]):
        assert left.end_fraction == right.start_fraction
    assert math.isclose(sum(p.span for p in DEFAULT_PHASES), 1.0)
    for phase in DEFAULT_PHASES:
        assert math.isclose(sum(phase.weights.values()), 1.0)
    assert not DEFAULT_PHASES[0].q_aware_packing
    assert not DEFAULT_PHASES[1].q_aware_packing
    assert DEFAULT_PHASES[2].q_aware_packing


def test_default_10k_step_corpus_has_five_percent_headroom():
    rows = [
        phase_target_rows(phase, total_steps=10_000, headroom=1.05)
        for phase in DEFAULT_PHASES
    ]
    assert rows == [5775, 2100, 2625]
    assert sum(rows) == 10_500
    early_targets = source_token_targets(
        DEFAULT_PHASES[0], total_steps=10_000, seq_len=4096, headroom=1.05
    )
    assert set(early_targets) == set(DEFAULT_PHASES[0].weights)
    assert sum(early_targets.values()) >= rows[0] * 4096


def test_q_metrics_match_indexer_eligibility_and_budget():
    # Segment-local positions 640..767 give exactly 128 eligible Qs.
    exact = q_row_metrics([768])
    assert exact.eligible_q == 128
    assert exact.selected_q == 128
    assert exact.budget_utilization == 1.0
    assert exact.eligible_coverage == 1.0
    assert exact.band_counts == (128, 0, 0, 0, 0, 0)

    long = q_row_metrics([4096])
    assert long.eligible_q == 4096 - 640
    assert long.selected_q == 128
    assert long.budget_utilization == 1.0
    assert math.isclose(long.eligible_coverage, 128 / (4096 - 640))


def test_full_length_segment_has_uniform_expected_q_density_per_position():
    metrics = q_row_metrics([4096])
    density = expected_q_density(metrics.expected_selected_by_band)
    assert max(density) - min(density) < 1e-12
    assert metrics.band_counts == tuple(
        hi - lo for lo, hi in zip(DEFAULT_Q_BAND_EDGES, DEFAULT_Q_BAND_EDGES[1:])
    )


def test_q_aware_packer_preserves_every_segment_and_alignment():
    lengths = [95, 127, 300, 641, 768, 900, 1200, 1600, 2200, 4096]
    rows = pack_length_indices(
        lengths,
        seq_len=4096,
        alignment=2,
        q_aware=True,
        candidate_window=32,
        seed=17,
    )
    flat = [index for row in rows for index in row]
    assert sorted(flat) == list(range(len(lengths)))
    assert len(flat) == len(set(flat))
    for row in rows:
        assert sum(aligned_length(lengths[i], 2) for i in row) <= 4096

    summary = aggregate_q_metrics(rows, lengths)
    assert summary["rows"] == len(rows)
    assert 0.0 <= summary["mean_budget_utilization"] <= 1.0
    assert 0.0 <= summary["mean_eligible_coverage"] <= 1.0


def test_q_aware_anchor_prefers_position_coverage_over_only_threshold_queries():
    # With both candidates visible, a full-context row covers every Q-position band,
    # while a 768-token row only covers the first 128 eligible positions.
    lengths = [768, 4096, 100, 100]
    rows = pack_length_indices(
        lengths,
        q_aware=True,
        candidate_window=16,
        seed=0,
    )
    assert rows[0] == [1]
