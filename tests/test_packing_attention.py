import jax.numpy as jnp

from nano_dsv41f.attention import (
    compressed_global_mask,
    fixed_local_window_mask,
    merge_attention_outputs,
)
from nano_dsv41f.indexer import latest_teacher_indices
from nano_dsv41f.packing import PackedSegments, segment_local_positions


def test_segment_local_positions_reset_at_pack_boundaries():
    ids = jnp.array([0, 0, 0, 1, 1, 2], dtype=jnp.int32)
    got = segment_local_positions(ids)
    assert jnp.array_equal(got, jnp.array([0, 1, 2, 0, 1, 0]))


def test_latest_teacher_query_is_fixed_shape_and_requires_640_history():
    # Segment 0 has local positions 0..639 and is not eligible.
    # Segment 1 has local positions 0..641 and its final query is eligible.
    ids = jnp.concatenate(
        [jnp.zeros((640,), dtype=jnp.int32), jnp.ones((642,), dtype=jnp.int32)]
    )
    indices, valid = latest_teacher_indices(
        ids,
        n_segments=2,
        local_window=128,
        retrieve_top_k=512,
    )
    assert indices.shape == (2,)
    assert jnp.array_equal(valid, jnp.array([False, True]))
    assert int(indices[1]) == 640 + 641


def test_global_packed_mask_needs_only_segment_id_plus_q_ge_rk():
    # After the 128-token crop, each segment satisfies Q_len = 2 * K_len.
    original_lengths = (256, 320)
    r = 2
    crop = 128
    q_lengths = tuple(length - crop for length in original_lengths)
    k_lengths = tuple(length // r - crop // r for length in original_lengths)
    assert q_lengths == tuple(r * length for length in k_lengths)

    q_segments = jnp.concatenate(
        [jnp.full((length,), i) for i, length in enumerate(q_lengths)]
    )
    k_segments = jnp.concatenate(
        [jnp.full((length,), i) for i, length in enumerate(k_lengths)]
    )
    q_ids = jnp.arange(sum(q_lengths))
    k_ids = jnp.arange(sum(k_lengths))

    packed = compressed_global_mask(
        q_ids,
        k_ids,
        q_segments,
        k_segments,
        compression_ratio=r,
    )

    # Compare against the explicit segment-local predicate.
    expected = jnp.zeros_like(packed)
    q0 = k0 = 0
    for ql, kl in zip(q_lengths, k_lengths):
        q_local = jnp.arange(ql)
        k_local = jnp.arange(kl)
        local = q_local[:, None] >= r * k_local[None, :]
        expected = expected.at[q0 : q0 + ql, k0 : k0 + kl].set(local)
        q0 += ql
        k0 += kl
    assert jnp.array_equal(packed, expected)


def test_fixed_128_swa_intentionally_overlaps_r2_global_representation():
    # Original token query t=128 has fixed SWA coverage [1, 128].
    q = jnp.array([128], dtype=jnp.int32)
    raw_k = jnp.arange(129, dtype=jnp.int32)
    seg_q = jnp.array([0], dtype=jnp.int32)
    seg_raw = jnp.zeros((129,), dtype=jnp.int32)
    local = fixed_local_window_mask(q, raw_k, seg_q, seg_raw, window=128)
    assert bool(local[0, 1])
    assert not bool(local[0, 0])

    # The corresponding cropped global query has q'=0. For r=2 it can attend
    # compressed group k=0, representing raw tokens {0, 1}. Token 1 is therefore
    # represented in both the local and compressed/global branches.
    global_visible = compressed_global_mask(
        jnp.array([0], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
        compression_ratio=2,
    )
    assert bool(global_visible[0, 0])

    # One token later, fixed SWA is [2, 129] while the same compressed group still
    # covers {0, 1}; the overlap alternates with the r=2 compression lattice.
    q_next = jnp.array([129], dtype=jnp.int32)
    raw_k_next = jnp.arange(130, dtype=jnp.int32)
    local_next = fixed_local_window_mask(
        q_next,
        raw_k_next,
        seg_q,
        jnp.zeros((130,), dtype=jnp.int32),
        window=128,
    )
    assert not bool(local_next[0, 1])
    assert bool(local_next[0, 2])


def test_lse_merge_matches_shared_denominator_weights():
    local_out = jnp.array([[2.0, 4.0]])
    global_out = jnp.array([[10.0, 20.0]])
    local_lse = jnp.log(jnp.array([2.0]))
    global_lse = jnp.log(jnp.array([3.0]))
    out, total = merge_attention_outputs(
        local_out, local_lse, global_out, global_lse
    )
    expected = (2.0 / 5.0) * local_out + (3.0 / 5.0) * global_out
    assert jnp.allclose(out, expected)
    assert jnp.allclose(total, jnp.log(jnp.array([5.0])))
