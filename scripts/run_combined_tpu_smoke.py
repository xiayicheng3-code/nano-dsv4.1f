from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from nano_dsv41f import (
    ModelConfig,
    TPUNativeConfig,
    TrainConfig,
    compile_diagnostics,
    compile_pretrain_step,
    init_model_sharded_mixed_precision,
    init_optimizer_state_sharded,
    make_v5e_mesh,
    pack_token_sequences,
    put_training_batch,
    runtime_report,
    semantic_axes,
    validate_sequence_length,
    validate_v5e_runtime,
)


def _synthetic_documents() -> tuple[np.ndarray, ...]:
    # Deliberately tiny entropy: repeated next-token transitions should visibly overfit in
    # only a few optimizer steps while odd document lengths exercise per-segment padding.
    pattern = np.asarray([17, 23, 42, 9, 31, 5, 77, 12], dtype=np.int32)
    return tuple(np.resize(pattern, length).astype(np.int32) for length in (641, 703, 701))


def _scalar(x) -> float:
    return float(np.asarray(jax.device_get(x)))


def main() -> None:
    print("runtime:", runtime_report())
    for warning in validate_v5e_runtime():
        print("WARNING:", warning)
    mesh = make_v5e_mesh()

    config = replace(ModelConfig(), n_experts=16)
    train_config = TrainConfig(
        total_steps=20,
        seq_len=2048,
        learning_rate=1.0e-3,
        warmup_steps=1,
        min_learning_rate=1.0e-3,
        cosine_decay_start_fraction=1.0,
    )
    native_config = TPUNativeConfig(
        moe_capacity_factor=2.0,
        moe_capacity_multiple=128,
        force_block_remat=True,
    )
    print("semantic axes:", semantic_axes(config, mesh))
    for warning in validate_sequence_length(train_config.seq_len, config, mesh):
        print("WARNING:", warning)

    packed = pack_token_sequences(
        _synthetic_documents(),
        seq_len=train_config.seq_len,
        pad_token_id=config.engram.pad_token_id,
        compression_ratio=config.csa2.context_compression_ratio,
    )
    assert packed.real_lengths == (641, 703, 701)
    assert packed.physical_lengths == (642, 704, 702)
    assert packed.real_tokens == 2045
    assert packed.lm_tokens == 2042
    print(
        "packed batch:",
        {
            "real_lengths": packed.real_lengths,
            "physical_lengths": packed.physical_lengths,
            "real_tokens": packed.real_tokens,
            "lm_tokens": packed.lm_tokens,
        },
    )

    ids, segments, token_mask = put_training_batch(
        packed.input_ids,
        packed.segment_ids,
        packed.token_mask,
        config,
        mesh,
    )
    print("input sharding:", ids.sharding)
    print("local shard:", ids.addressable_shards[0].data.shape)

    key = jax.random.PRNGKey(0)
    params, param_specs, _ = init_model_sharded_mixed_precision(
        key, config, mesh, payload_dtype=jnp.bfloat16
    )
    jax.block_until_ready(jax.tree_util.tree_leaves(params)[0])
    opt_state, _ = init_optimizer_state_sharded(params, param_specs, config, mesh)
    jax.block_until_ready(jax.tree_util.tree_leaves(opt_state)[0])

    base_step = compile_pretrain_step(
        params,
        opt_state,
        param_specs,
        config,
        train_config,
        mesh,
        include_indexer=False,
        n_segments=None,
        native_config=native_config,
    )
    late_step = compile_pretrain_step(
        params,
        opt_state,
        param_specs,
        config,
        train_config,
        mesh,
        include_indexer=True,
        n_segments=packed.n_segments,
        native_config=native_config,
    )

    zero_step = jnp.asarray(0, dtype=jnp.int32)
    _, diagnostics = compile_diagnostics(
        late_step, params, opt_state, ids, segments, zero_step, token_mask
    )
    print("late collectives:", diagnostics["collectives"])
    print("late compiler memory:", diagnostics["memory"])

    base_losses: list[float] = []
    print("\n=== 10 base/no-indexer overfit steps ===")
    for step_i in range(10):
        params, opt_state, metrics = base_step(
            params,
            opt_state,
            ids,
            segments,
            jnp.asarray(step_i, dtype=jnp.int32),
            token_mask,
        )
        jax.block_until_ready(metrics["loss"])
        loss = _scalar(metrics["lm_loss"])
        base_losses.append(loss)
        if step_i == 0:
            assert int(jax.device_get(metrics["lm_tokens"])) == packed.lm_tokens
            assert int(jax.device_get(metrics["native_moe_layers"])) == config.n_layers
            assert int(jax.device_get(metrics["experts_per_chip"])) == 2
            loads = np.asarray(jax.device_get(metrics["expert_loads"]))
            overflow = np.asarray(jax.device_get(metrics["expert_overflow"]))
            assert loads.shape == (config.n_experts,)
            expected_assignments = (
                config.n_layers
                * train_config.seq_len
                * config.experts_per_token
            )
            assert int(loads.sum()) == expected_assignments
            print("expert loads across all backbone layers:", loads.tolist())
            print("expert overflow across all backbone layers:", overflow.tolist())
            print("per-expert capacity per layer:", int(jax.device_get(metrics["expert_capacity"])))
        print(f"base step {step_i:02d}: lm_loss={loss:.6f}")

    expected_teacher = np.asarray([[640, 1344, 2046]], dtype=np.int32)
    late_lm_losses: list[float] = []
    late_indexer_losses: list[float] = []
    print("\n=== 10 late-indexer overfit steps ===")
    for offset in range(10):
        step_i = 10 + offset
        params, opt_state, metrics = late_step(
            params,
            opt_state,
            ids,
            segments,
            jnp.asarray(step_i, dtype=jnp.int32),
            token_mask,
        )
        jax.block_until_ready(metrics["loss"])
        lm_loss = _scalar(metrics["lm_loss"])
        index_loss = _scalar(metrics["indexer_loss"])
        late_lm_losses.append(lm_loss)
        late_indexer_losses.append(index_loss)

        index_aux = metrics["indexer"]
        if offset == 0:
            assert np.isfinite(index_loss) and index_loss > 0.0
            assert int(jax.device_get(index_aux["active_queries"])) == 9
            for key, count in index_aux["query_counts"].items():
                assert int(jax.device_get(count)) == 3, (key, count)
            for key, query_indices in index_aux["teacher_query_indices"].items():
                got = np.asarray(jax.device_get(query_indices), dtype=np.int32)
                np.testing.assert_array_equal(got, expected_teacher, err_msg=key)
            for key, valid in index_aux["teacher_query_valid"].items():
                assert np.asarray(jax.device_get(valid)).all(), key
            print(
                "late-indexer query counts:",
                {k: int(jax.device_get(v)) for k, v in index_aux["query_counts"].items()},
            )
            print(
                "teacher query indices:",
                {
                    k: np.asarray(jax.device_get(v)).tolist()
                    for k, v in index_aux["teacher_query_indices"].items()
                },
            )
        print(
            f"late step {offset:02d}: lm_loss={lm_loss:.6f} "
            f"indexer_loss={index_loss:.6f}"
        )

    print("\nbase LM trajectory:", [round(x, 6) for x in base_losses])
    print("late LM trajectory:", [round(x, 6) for x in late_lm_losses])
    print("late indexer trajectory:", [round(x, 6) for x in late_indexer_losses])

    if base_losses[-1] >= base_losses[0]:
        raise AssertionError(
            "synthetic LM loss did not fall across the 10 base steps; inspect optimizer/runtime"
        )
    if late_lm_losses[-1] >= base_losses[0]:
        raise AssertionError("LM did not overfit the repeated packed synthetic batch")
    if min(late_indexer_losses[1:]) >= late_indexer_losses[0]:
        raise AssertionError(
            "late-indexer loss never improved after its first supervised step"
        )
    print("combined packed + multi-expert + late-indexer overfit smoke: PASS")


if __name__ == "__main__":
    main()
