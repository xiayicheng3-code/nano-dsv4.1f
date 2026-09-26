import pytest

# These tests belong to the optional CPU/API surface. The ordinary JAX-only job should
# collect the repository without pulling in Torch or deepseek-recipe; the dedicated
# cpu-inference job installs both extras and executes this module fully.
torch = pytest.importorskip("torch")
pytest.importorskip("deepseek_recipe")

from nano_dsv41f.chat_protocol import (  # noqa: E402
    ASSISTANT_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN_ID,
    USER_TOKEN,
)


def test_deepseek_recipe_renders_chat_completions():
    from nano_dsv41f.vllm_v41_cpu.api import prepare_protocol_request

    prepared = prepare_protocol_request(
        "chat_completions",
        {
            "model": "nano-dsv4.1f",
            "messages": [{"role": "user", "content": "Hello"}],
            "thinking": {"type": "enabled"},
            "reasoning_effort": "high",
            "stream": False,
            "max_tokens": 32,
        },
    )
    assert prepared.prompt.startswith(BOS_TOKEN)
    assert USER_TOKEN in prepared.prompt
    assert ASSISTANT_TOKEN in prepared.prompt
    assert "Hello" in prepared.prompt
    assert prepared.inference_options.max_tokens == 32
    assert prepared.conversation_request.conversation.thinking_mode is True


def test_deepseek_recipe_renders_responses_api():
    from nano_dsv41f.vllm_v41_cpu.api import prepare_protocol_request

    prepared = prepare_protocol_request(
        "responses",
        {
            "model": "nano-dsv4.1f",
            "input": "Explain 2+2 briefly.",
            "max_output_tokens": 24,
        },
    )
    assert prepared.prompt.startswith(BOS_TOKEN)
    assert USER_TOKEN in prepared.prompt
    assert ASSISTANT_TOKEN in prepared.prompt
    assert "Explain 2+2 briefly." in prepared.prompt
    assert prepared.inference_options.max_tokens == 24


def test_deepseek_recipe_renders_anthropic_messages():
    from nano_dsv41f.vllm_v41_cpu.api import prepare_protocol_request

    prepared = prepare_protocol_request(
        "messages",
        {
            "model": "nano-dsv4.1f",
            "max_tokens": 20,
            "messages": [{"role": "user", "content": "Hello from Messages"}],
        },
    )
    assert prepared.prompt.startswith(BOS_TOKEN)
    assert USER_TOKEN in prepared.prompt
    assert ASSISTANT_TOKEN in prepared.prompt
    assert "Hello from Messages" in prepared.prompt
    assert prepared.inference_options.max_tokens == 20


def test_generation_eos_is_transport_stop_not_parser_text():
    from nano_dsv41f.vllm_v41_cpu.api import NanoDeepSeekProtocolBackend

    class StubModel:
        device = torch.device("cpu")

        def generate(self, input_ids, **_kwargs):
            suffix = torch.tensor([[17, 23, EOS_TOKEN_ID]], dtype=torch.long)
            return torch.cat((input_ids, suffix), dim=-1)

    backend = NanoDeepSeekProtocolBackend(StubModel(), object())
    generated, prompt_tokens, finish = backend._generate_ids(
        [4, 5, 6], max_tokens=8, temperature=0.0, top_p=1.0
    )
    assert generated == [17, 23]
    assert prompt_tokens == 3
    assert finish == "stop"


def test_cpu_http_app_exposes_major_endpoint_families():
    from nano_dsv41f.vllm_v41_cpu.api import create_app

    class StubBackend:
        model_name = "nano-dsv4.1f"

    app = create_app(StubBackend())
    paths = {route.path for route in app.routes}
    assert "/v1/completions" in paths
    assert "/v1/chat/completions" in paths
    assert "/v1/responses" in paths
    assert "/v1/messages" in paths
    assert "/v1/models" in paths
    assert "/health" in paths
