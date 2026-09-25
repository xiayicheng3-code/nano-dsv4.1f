from nano_dsv41f.chat_protocol import (
    ASSISTANT_TOKEN,
    BOS_TOKEN,
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
