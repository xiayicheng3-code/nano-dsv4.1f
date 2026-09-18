from __future__ import annotations

from dataclasses import replace
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import time

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('combined-smoke.json'))
    parser.add_argument('--query-budget', type=int, default=128,
                        help='Eligible query positions per global batch and retriever group')
    parser.add_argument('--query-seed', type=int, default=0)
    parser.add_argument('--apply-candidate-mask', action='store_true',
                        help='Restrict the late L5 objective to L3 candidates (default: off)')
    args = parser.parse_args()
    config = replace(ModelConfig(), n_experts=16)
    config = replace(config, indexer_training=replace(
        config.indexer_training, query_budget=args.query_budget, query_seed=args.query_seed,
        apply_candidate_mask=args.apply_candidate_mask,
    ))
    from nano_dsv41f.runtime import pallas_preflight
    runtime = pallas_preflight()
    for warning in validate_v5e_runtime():
        print("WARNING:", warning)
    mesh = make_v5e_mesh()

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
    print("indexer training:", asdict(config.indexer_training))
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
    compile_start = time.perf_counter()
    _, diagnostics = compile_diagnostics(
        late_step, params, opt_state, ids, segments, zero_step, token_mask
    )
    print("late collectives:", diagnostics["collectives"])
    print("late compiler memory:", diagnostics["memory"])
    compile_seconds = time.perf_counter() - compile_start

    base_losses: list[float] = []
    base_seconds = []
    late_seconds = []
    print("\n=== 10 base/no-indexer overfit steps ===")
    for step_i in range(10):
        start = time.perf_counter()
        params, opt_state, metrics = base_step(
            params,
            opt_state,
            ids,
            segments,
            jnp.asarray(step_i, dtype=jnp.int32),
            token_mask,
        )
        jax.block_until_ready((params, opt_state, metrics))
        base_seconds.append(time.perf_counter() - start)
        assert int(jax.device_get(metrics['expert_dropped'].sum())) == 0
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
        assert np.isfinite(loss), (step_i, loss)
        print(f"base step {step_i:02d}: lm_loss={loss:.6f} seconds={base_seconds[-1]:.4f}")

    # All eligible real positions, not just each document's final token.
    eligible_positions = set([640]) | set(range(1282, 1345)) | set(range(1986, 2047))
    expected_count = min(config.indexer_training.query_budget, len(eligible_positions))
    sampled_positions = set()
    late_lm_losses: list[float] = []
    late_indexer_losses: list[float] = []
    print("\n=== 10 late-indexer overfit steps ===")
    for offset in range(10):
        step_i = 10 + offset
        start = time.perf_counter()
        params, opt_state, metrics = late_step(
            params,
            opt_state,
            ids,
            segments,
            jnp.asarray(step_i, dtype=jnp.int32),
            token_mask,
        )
        jax.block_until_ready((params, opt_state, metrics))
        late_seconds.append(time.perf_counter() - start)
        assert int(jax.device_get(metrics['expert_dropped'].sum())) == 0
        lm_loss = _scalar(metrics["lm_loss"])
        index_loss = _scalar(metrics["indexer_loss"])
        late_lm_losses.append(lm_loss)
        late_indexer_losses.append(index_loss)
        assert np.isfinite(lm_loss) and np.isfinite(index_loss), (offset, lm_loss, index_loss)

        index_aux = metrics["indexer"]
        assert int(jax.device_get(index_aux["active_queries"])) == 3 * expected_count
        for key, count in index_aux["query_counts"].items():
            assert int(jax.device_get(count)) == expected_count, (key, count)
            assert int(jax.device_get(index_aux["eligible_query_counts"][key])) == len(eligible_positions)
            got = np.asarray(jax.device_get(index_aux["teacher_query_indices"][key]))
            valid = np.asarray(jax.device_get(index_aux["teacher_query_valid"][key]))
            selected = set(got[valid].tolist())
            assert len(selected) == expected_count and selected <= eligible_positions, key
            sampled_positions.update(selected)
        if offset == 0:
            assert np.isfinite(index_loss) and index_loss > 0.0
            print(
                "late-indexer query counts:",
                {k: int(jax.device_get(v)) for k, v in index_aux["query_counts"].items()},
            )
            print(
                "valid teacher query indices:",
                {
                    k: np.asarray(jax.device_get(v))[
                        np.asarray(jax.device_get(index_aux["teacher_query_valid"][k]))
                    ].tolist()
                    for k, v in index_aux["teacher_query_indices"].items()
                },
            )
        print(
            f"late step {offset:02d}: lm_loss={lm_loss:.6f} "
            f"indexer_loss={index_loss:.6f} seconds={late_seconds[-1]:.4f}"
        )

    print("\nbase LM trajectory:", [round(x, 6) for x in base_losses])
    print("late LM trajectory:", [round(x, 6) for x in late_lm_losses])
    print("late indexer trajectory:", [round(x, 6) for x in late_indexer_losses])

    report = {
        'runtime': runtime,
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'model_config': asdict(config), 'train_config': asdict(train_config),
        'native_config': asdict(native_config), 'late_compile_seconds': compile_seconds,
        'late_diagnostics': diagnostics,
        'real_token_utilization': packed.real_tokens / train_config.seq_len,
        'lm_token_utilization': packed.lm_tokens / (train_config.seq_len - 1),
        'base_losses': base_losses, 'late_lm_losses': late_lm_losses,
        'late_indexer_losses': late_indexer_losses,
        'eligible_query_positions': len(eligible_positions),
        'sampled_positions_over_late_steps': len(sampled_positions),
        'queries_per_group_per_step': expected_count,
        'base_seconds': base_seconds, 'late_seconds': late_seconds,
        'base_median_seconds': float(np.median(base_seconds[2:])),
        'late_median_seconds': float(np.median(late_seconds[2:])),
        'base_lm_tokens_per_second': packed.lm_tokens / float(np.median(base_seconds[2:])),
        'late_lm_tokens_per_second': packed.lm_tokens / float(np.median(late_seconds[2:])),
        'expert_dropped': int(jax.device_get(metrics['expert_dropped'].sum())),
        'timing_note': 'Synchronized complete steps; first two steps per phase excluded from medians. Physical packing utilization is not Splash block utilization.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print('saved report:', args.output)
    if base_losses[-1] >= base_losses[0]:
        raise AssertionError(
            "synthetic LM loss did not fall across the 10 base steps; inspect optimizer/runtime"
        )
    if late_lm_losses[-1] >= base_losses[0]:
        raise AssertionError("LM did not overfit the repeated packed synthetic batch")
    if expected_count == len(eligible_positions) and min(late_indexer_losses[1:]) >= late_indexer_losses[0]:
        raise AssertionError(
            "late-indexer loss never improved after its first supervised step"
        )
    if expected_count < len(eligible_positions):
        print("Query subsets change each step; sampled indexer losses are diagnostic, not an overfit gate.")
    print("combined packed + multi-expert + late-indexer overfit smoke: PASS")


if __name__ == "__main__":
    main()
