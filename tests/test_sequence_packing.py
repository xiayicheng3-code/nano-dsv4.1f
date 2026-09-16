import jax.numpy as jnp
import numpy as np

from nano_dsv41f.indexer import latest_teacher_indices_batched
from nano_dsv41f.sequence_packing import pack_token_sequences
from nano_dsv41f.training import causal_lm_loss


def test_odd_documents_pack_to_ratio_aligned_segments_without_fake_teachers():
    docs = tuple(np.arange(length, dtype=np.int32) + 1 for length in (641, 703, 701))
    packed = pack_token_sequences(
        docs,
        seq_len=2048,
        pad_token_id=0,
        compression_ratio=2,
    )

    assert packed.real_lengths == (641, 703, 701)
    assert packed.physical_lengths == (642, 704, 702)
    assert packed.real_tokens == 2045
    assert packed.lm_tokens == 2042
    assert packed.input_ids.shape == (1, 2048)
    assert packed.segment_ids.shape == (1, 2048)
    assert packed.token_mask.shape == (1, 2048)
    assert int(packed.token_mask.sum()) == 2045

    logits = jnp.zeros((1, 2048, 2048), dtype=jnp.float32)
    _, lm_tokens = causal_lm_loss(
        logits,
        jnp.asarray(packed.input_ids),
        jnp.asarray(packed.segment_ids),
        token_mask=jnp.asarray(packed.token_mask),
    )
    assert int(lm_tokens) == 2042

    indices, valid = latest_teacher_indices_batched(
        jnp.asarray(packed.segment_ids),
        n_segments=3,
        min_local_position=640,
        token_mask=jnp.asarray(packed.token_mask),
    )
    np.testing.assert_array_equal(np.asarray(indices), np.asarray([[640, 1344, 2046]]))
    assert np.asarray(valid).all()


def test_tail_padding_stays_inside_last_physical_segment_but_is_masked():
    docs = (np.arange(5, dtype=np.int32) + 1, np.arange(6, dtype=np.int32) + 11)
    packed = pack_token_sequences(
        docs,
        seq_len=16,
        pad_token_id=0,
        compression_ratio=2,
    )
    assert packed.physical_lengths == (6, 10)
    assert packed.real_lengths == (5, 6)
    assert packed.segment_ids[0, -1] == 1
    assert not packed.token_mask[0, -1]
