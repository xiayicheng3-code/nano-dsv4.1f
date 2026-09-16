from __future__ import annotations

import jax.numpy as jnp

from nano_dsv41f.agent_data import CleanPolicy, normalize_agent_trace
from nano_dsv41f.chat_protocol import (
    BOS_TOKEN_ID,
    DSPARK_NOISE_TOKEN_ID,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    apply_tokenizer_contract,
    nano_v41_tokenizer_contract,
    validate_model_token_ids,
)
from nano_dsv41f.config import ModelConfig
from nano_dsv41f.hf_export import (
    build_deepseek_v41_probe_config,
    build_nano_hf_config,
    flatten_parameter_tree,
    runtime_compatibility_report,
)


def test_tokenizer_contract_and_model_ids_are_frozen_before_training():
    contract = nano_v41_tokenizer_contract()
    assert contract.token_to_id[contract.special_tokens[0]] == BOS_TOKEN_ID == 0
    assert contract.token_to_id[contract.special_tokens[1]] == EOS_TOKEN_ID == 1
    assert contract.token_to_id[contract.special_tokens[2]] == PAD_TOKEN_ID == 2

    legacy = ModelConfig()
    assert validate_model_token_ids(legacy, contract)

    config = apply_tokenizer_contract(legacy, contract)
    assert config.engram.pad_token_id == PAD_TOKEN_ID
    assert config.dspark.noise_token_id == DSPARK_NOISE_TOKEN_ID
    assert validate_model_token_ids(config, contract) == ()


def test_agent_trace_normalizes_tool_calls_without_rendering_dsml():
    record = {
        "id": "demo",
        "success": True,
        "thinking_mode": "thinking",
        "tools": [
            {
                "name": "shell",
                "description": "Run a command",
                "parameters": {
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}},
                },
            }
        ],
        "messages": [
            {"role": "user", "content": "List files"},
            {
                "role": "assistant",
                "reasoning": "I should inspect the directory.",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "name": "shell",
                        "arguments": {"cmd": "ls"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "a.py\nb.py"},
            {"role": "assistant", "content": "There are two files."},
        ],
    }

    case = normalize_agent_trace(record, policy=CleanPolicy(max_tool_result_chars=100))
    assert case["messages"][1]["reasoning_content"] == "I should inspect the directory."
    assert case["messages"][1]["tool_calls"][0]["function"]["arguments"] == '{"cmd":"ls"}'
    assert case["messages"][2] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "a.py\nb.py",
    }
    assert "<｜DSML｜" not in str(case)
    assert case["supervision"][1]["assistant_only"] is True
    assert case["supervision"][2]["assistant_only"] is False


def test_agent_step_trace_assigns_stable_tool_call_id_and_truncates_observation():
    case = normalize_agent_trace(
        {
            "steps": [
                {"prompt": "Inspect"},
                {
                    "thought": "Use the shell.",
                    "action": {"tool": "shell", "args": {"cmd": "pwd"}},
                    "observation": "x" * 30,
                },
            ]
        },
        policy=CleanPolicy(max_tool_result_chars=10),
    )
    call = case["messages"][1]["tool_calls"][0]
    assert call["id"] == "call_step_1"
    assert case["messages"][2]["tool_call_id"] == "call_step_1"
    assert "truncated" in case["messages"][2]["content"]


def test_probe_config_captures_nano_csa2_schedule():
    config = apply_tokenizer_contract(ModelConfig())
    probe = build_deepseek_v41_probe_config(config)
    text = probe["text_config"]
    assert text["compress_ratios"] == [0, 2, 2, 1, 1, 1, 1]
    assert text["kv_source_layer_ids"] == [1, 3]
    assert text["index_source_layer_ids"] == [1, 3, 5]
    assert text["candidate_source_layer_id"] == 3
    assert probe["_nano_probe_only"] is True

    nano = build_nano_hf_config(config)
    assert nano["architectures"] == ["NanoDeepseekV41ForCausalLM"]
    assert nano["nano_tokenizer_contract"]["pad_token_id"] == 2


def test_portable_parameter_names_are_lossless_and_stable():
    tree = {
        "embed": jnp.zeros((4, 3)),
        "blocks": (
            {"attn": {"q_a": {"weight": jnp.ones((3, 2))}}},
            {"moe": {"router_bias": jnp.zeros((8,))}},
        ),
    }
    flat = flatten_parameter_tree(tree)
    assert set(flat) == {
        "nano.embed",
        "nano.blocks.0.attn.q_a.weight",
        "nano.blocks.1.moe.router_bias",
    }


def test_stock_vllm_report_is_explicitly_not_claimed_compatible():
    report = runtime_compatibility_report(apply_tokenizer_contract(ModelConfig()))
    assert report["stock_deepseek_v41_runtime_compatible"] is False
    assert any(
        item["component"] == "vllm.deepseek_v41.compressor"
        for item in report["blockers"]
    )
