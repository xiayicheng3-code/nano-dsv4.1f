from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nano_dsv41f.config import (
    AttentionConfig,
    CSA2Config,
    DSparkConfig,
    EngramConfig,
    IndexerConfig,
    ModelConfig,
    ParallelismConfig,
    RopeConfig,
)
from nano_dsv41f.hf_export import export_portable_checkpoint, flatten_parameter_tree
from nano_dsv41f.model import apply_model, init_model


torch = pytest.importorskip("torch")
from nano_dsv41f.vllm_v41_cpu import NanoDeepseekV41CPU  # noqa: E402


def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        d_model=32,
        d_ff=48,
        n_experts=4,
        experts_per_token=2,
        mhc_streams=2,
        mhc_sinkhorn_iters=4,
        attention=AttentionConfig(
            d_model=32,
            n_heads=4,
            head_dim=8,
            q_rank=8,
            o_rank=8,
            o_groups=2,
            local_window=4,
            rope=RopeConfig(rope_head_dim=4),
        ),
        csa2=CSA2Config(
            context_layers=3,
            generation_layers=4,
            context_swa_only_layers=1,
            context_retriever_group_size=2,
            generation_retriever_group_size=2,
            context_compression_ratio=2,
            generation_compression_ratio=1,
        ),
        indexer=IndexerConfig(
            n_heads=2,
            head_dim=4,
            top_k=2,
            candidate_source_layer=3,
            candidate_topk_blocks=1,
            candidate_block_size=4,
        ),
        engram=EngramConfig(
            enabled=True,
            layer_ids=(1,),
            table_size=128,
            max_ngram_size=3,
            n_hash_heads=1,
            head_dim=4,
        ),
        dspark=DSparkConfig(enabled=False),
        parallelism=ParallelismConfig(expert_shard=4),
    )


def make_cpu(seed: int) -> tuple[ModelConfig, object, NanoDeepseekV41CPU]:
    config = tiny_config()
    params = init_model(jax.random.PRNGKey(seed), config)
    cpu = NanoDeepseekV41CPU(
        config,
        flatten_parameter_tree(params),
        dtype=torch.float32,
    )
    return config, params, cpu


def test_dense_cpu_logits_match_jax_reference_with_packing():
    config, params, cpu = make_cpu(7)
    input_ids = jnp.asarray(
        [[1, 5, 3, 8, 13, 21, 34, 2]], dtype=jnp.int32
    )
    # Keep the boundary aligned with ratio-2 compression, matching the packed-data
    # invariant used by the JAX reference while exercising segment-local masks/RoPE/Engram.
    segment_ids = jnp.asarray(
        [[0, 0, 0, 0, 1, 1, 1, 1]], dtype=jnp.int32
    )
    expected, _ = apply_model(
        params,
        config,
        input_ids,
        segment_ids=segment_ids,
        compute_indexer=False,
    )
    actual, _ = cpu.forward(
        torch.tensor(np.asarray(input_ids), dtype=torch.long),
        segment_ids=torch.tensor(np.asarray(segment_ids), dtype=torch.long),
        compute_indexer=False,
        sparse_retrieval=False,
    )
    np.testing.assert_allclose(
        actual.cpu().numpy(),
        np.asarray(expected),
        rtol=3e-4,
        atol=3e-4,
    )


def test_exported_checkpoint_loads_without_weight_renaming(tmp_path):
    config = tiny_config()
    params = init_model(jax.random.PRNGKey(9), config)
    export_portable_checkpoint(params, tmp_path, config, max_position_embeddings=128)
    cpu = NanoDeepseekV41CPU.from_pretrained(tmp_path, dtype=torch.float32)
    ids = torch.tensor([[0, 4, 7, 11, 15, 3, 2, 1]], dtype=torch.long)
    loaded_logits, _ = cpu.forward(
        ids, compute_indexer=False, sparse_retrieval=False
    )
    direct = NanoDeepseekV41CPU(
        config,
        flatten_parameter_tree(params),
        dtype=torch.float32,
    )
    direct_logits, _ = direct.forward(
        ids, compute_indexer=False, sparse_retrieval=False
    )
    torch.testing.assert_close(loaded_logits, direct_logits, rtol=0.0, atol=0.0)


def test_sparse_cpu_path_accepts_odd_prefix_and_reindexes():
    config, _, cpu = make_cpu(11)
    ids = torch.tensor(
        [[1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.long
    )
    logits, aux = cpu.forward(ids, sparse_retrieval=True)
    assert logits.shape == (1, 9, config.vocab_size)
    assert torch.isfinite(logits).all()
    assert aux["final_global_source_layer"] == 3
    assert aux["layers"][5]["retrieval_mask"] is not None


def test_cached_dense_prefill_matches_full_prefix_with_packing():
    _, _, cpu = make_cpu(13)
    ids = torch.tensor([[1, 5, 3, 8, 13, 21, 34, 2]], dtype=torch.long)
    segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=torch.long)
    full, _ = cpu.forward(
        ids,
        segment_ids=segments,
        compute_indexer=False,
        sparse_retrieval=False,
    )
    cached, cache = cpu.prefill_cache(
        ids,
        segment_ids=segments,
        compute_indexer=False,
        sparse_retrieval=False,
    )
    torch.testing.assert_close(cached, full, rtol=4e-4, atol=4e-4)
    assert cache.length == ids.shape[1]
    assert set(cache.owners) == {1, 3}


def test_cached_sparse_prefill_matches_full_prefix():
    _, _, cpu = make_cpu(17)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.long)
    full, _ = cpu.forward(ids, sparse_retrieval=True)
    cached, cache = cpu.prefill_cache(ids, sparse_retrieval=True)
    torch.testing.assert_close(cached, full, rtol=5e-4, atol=5e-4)
    assert cache.owners[1].pending_latent is not None
    assert cache.owners[3].state.index_k is not None


def test_cached_greedy_generation_matches_full_prefix_recompute():
    _, _, cpu = make_cpu(19)
    prompt = torch.tensor([[1, 7, 3, 9, 4, 12, 5, 2]], dtype=torch.long)
    expected = prompt.clone()
    for _ in range(3):
        logits, _ = cpu.forward(expected, sparse_retrieval=True)
        next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
        expected = torch.cat((expected, next_id), dim=-1)

    actual = cpu.generate(
        prompt,
        max_new_tokens=3,
        temperature=0.0,
        sparse_retrieval=True,
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_cpu_port_has_no_deepseek_v4_imports():
    import nano_dsv41f.vllm_v41_cpu as cpu_package

    root = Path(cpu_package.__file__).parent
    forbidden = (
        "from vllm.models.deepseek_v4",
        "import vllm.models.deepseek_v4",
    )
    for source in root.glob("*.py"):
        text = source.read_text(encoding="utf-8")
        assert not any(pattern in text for pattern in forbidden), source
