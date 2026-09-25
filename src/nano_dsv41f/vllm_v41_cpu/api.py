from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any, Literal
from uuid import uuid4

import torch

from ..chat_protocol import EOS_TOKEN, EOS_TOKEN_ID
from .model import NanoDeepseekV41CPU

Protocol = Literal["chat_completions", "responses", "messages"]


@dataclass(frozen=True)
class PreparedProtocolRequest:
    protocol: Protocol
    request: Any
    conversation_request: Any
    prompt: str
    include_usage: bool = False
    custom_tool_names: frozenset[str] = frozenset()

    @property
    def model_name(self) -> str | None:
        return self.conversation_request.model

    @property
    def stream(self) -> bool:
        return bool(self.conversation_request.stream)

    @property
    def inference_options(self) -> Any:
        return self.conversation_request.inference_options


def _recipe_types() -> dict[str, Any]:
    try:
        from deepseek_recipe import (
            ChatCompletionRequest,
            ChatCompletionResponse,
            ConversionOptions,
            DeepseekV41Encoding,
            InferenceChunk,
            InferenceFinishReason,
            MessagesRequest,
            MessagesResponse,
            PromptUsage,
            ResponsesRequest,
            ResponsesResponse,
            StreamProcessor,
            Tokenizer,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "DeepSeek protocol serving needs `pip install 'nano-dsv41f[api]'`"
        ) from exc
    return {
        "ChatCompletionRequest": ChatCompletionRequest,
        "ChatCompletionResponse": ChatCompletionResponse,
        "MessagesRequest": MessagesRequest,
        "MessagesResponse": MessagesResponse,
        "ResponsesRequest": ResponsesRequest,
        "ResponsesResponse": ResponsesResponse,
        "ConversionOptions": ConversionOptions,
        "DeepseekV41Encoding": DeepseekV41Encoding,
        "InferenceChunk": InferenceChunk,
        "InferenceFinishReason": InferenceFinishReason,
        "PromptUsage": PromptUsage,
        "StreamProcessor": StreamProcessor,
        "Tokenizer": Tokenizer,
    }


def prepare_protocol_request(
    protocol: Protocol,
    body: bytes | str | dict[str, Any],
) -> PreparedProtocolRequest:
    """Normalize and render one request using DeepSeek's maintained V4.1 protocol.

    V4.1 is deliberately not represented by a local Jinja template. Chat Completions,
    Responses, and Anthropic Messages are normalized by `deepseek-recipe`, which owns
    tool-result folding, reasoning effort, DSML syntax, and output parsing rules.
    """
    recipe = _recipe_types()
    request_types = {
        "chat_completions": recipe["ChatCompletionRequest"],
        "responses": recipe["ResponsesRequest"],
        "messages": recipe["MessagesRequest"],
    }
    if protocol not in request_types:
        raise ValueError(f"unsupported protocol: {protocol}")
    if isinstance(body, dict):
        body = json.dumps(body, ensure_ascii=False).encode("utf-8")
    elif isinstance(body, str):
        body = body.encode("utf-8")

    request = request_types[protocol](body)
    converted = request.convert(recipe["ConversionOptions"]())
    rendered = recipe["DeepseekV41Encoding"]().render_conversation(
        converted.conversation
    )
    if rendered.image_sources:
        raise ValueError(
            "nano-dsv4.1f CPU serving is text-only; V4.1 image prompt rendering is "
            "recognized but no vision encoder is implemented"
        )
    include_usage = (
        bool(request.include_usage()) if protocol == "chat_completions" else False
    )
    custom_tool_names = (
        frozenset(request.custom_tool_names()) if protocol == "responses" else frozenset()
    )
    return PreparedProtocolRequest(
        protocol=protocol,
        request=request,
        conversation_request=converted,
        prompt=rendered.prompt,
        include_usage=include_usage,
        custom_tool_names=custom_tool_names,
    )


class NanoTokenizer:
    """Frozen nano tokenizer with both raw and DeepSeek-protocol views."""

    def __init__(self, tokenizer_json: str | Path) -> None:
        try:
            from tokenizers import Tokenizer as HFTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "CPU API serving needs `pip install 'nano-dsv41f[api]'`"
            ) from exc
        self.path = Path(tokenizer_json)
        self.tokenizer = HFTokenizer.from_file(str(self.path))
        recipe = _recipe_types()
        self.recipe_tokenizer = recipe["Tokenizer"].from_file(str(self.path))

    def encode(self, text: str) -> list[int]:
        """Encode a raw completion prompt without adding protocol markers."""
        return list(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def encode_conversation(self, conversation: Any) -> list[int]:
        """Encode a structured V4.1 conversation through DeepSeek's own encoder."""
        recipe = _recipe_types()
        encoding = recipe["DeepseekV41Encoding"]().with_tokenizer(
            self.recipe_tokenizer
        )
        return list(encoding.encode(conversation))

    def decode(self, token_ids: list[int] | torch.Tensor) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)


class NanoDeepSeekProtocolBackend:
    """DeepSeek-V4.1 protocol frontend backed by the validated Torch CPU runtime."""

    def __init__(
        self,
        model: NanoDeepseekV41CPU,
        tokenizer: NanoTokenizer,
        *,
        model_name: str = "nano-dsv4.1f",
        default_max_tokens: int = 256,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.default_max_tokens = default_max_tokens

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str | Path,
        *,
        tokenizer_path: str | Path | None = None,
        model_name: str = "nano-dsv4.1f",
        dtype: torch.dtype = torch.float32,
    ) -> "NanoDeepSeekProtocolBackend":
        root = Path(checkpoint_dir)
        tokenizer_path = Path(tokenizer_path) if tokenizer_path else root / "tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"tokenizer.json not found at {tokenizer_path}; publish it beside the "
                "portable checkpoint or pass tokenizer_path explicitly"
            )
        return cls(
            NanoDeepseekV41CPU.from_pretrained(root, dtype=dtype),
            NanoTokenizer(tokenizer_path),
            model_name=model_name,
        )

    def _generate_ids(
        self,
        prompt_ids: list[int],
        *,
        max_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
    ) -> tuple[list[int], int, str]:
        if not prompt_ids:
            raise ValueError("prompt tokenized to an empty sequence")
        input_ids = torch.tensor(
            [prompt_ids], dtype=torch.long, device=self.model.device
        )
        output = self.model.generate(
            input_ids,
            max_new_tokens=max_tokens or self.default_max_tokens,
            eos_token_id=EOS_TOKEN_ID,
            temperature=1.0 if temperature is None else float(temperature),
            top_p=0.95 if top_p is None else float(top_p),
            sparse_retrieval=True,
        )
        generated = output[0, input_ids.shape[1] :].detach().cpu().tolist()
        finish = "stop" if generated and generated[-1] == EOS_TOKEN_ID else "length"
        return generated, len(prompt_ids), finish

    def complete_raw(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> dict[str, Any]:
        """Classic Completion semantics: caller text is used as the raw model prompt."""
        prompt_ids = self.tokenizer.encode(prompt)
        generated, prompt_tokens, finish = self._generate_ids(
            prompt_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        text = self.tokenizer.decode(generated)
        if text.endswith(EOS_TOKEN):
            text = text[: -len(EOS_TOKEN)]
        completion_tokens = len(generated)
        return {
            "id": f"cmpl-{uuid4().hex}",
            "object": "text_completion",
            "created": int(time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "text": text,
                    "logprobs": None,
                    "finish_reason": finish,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def complete_protocol(
        self,
        protocol: Protocol,
        body: bytes | str | dict[str, Any],
    ) -> tuple[PreparedProtocolRequest, str]:
        """Return a complete JSON response for Chat/Responses/Messages.

        Both input tokenization and output token parsing stay inside `deepseek-recipe`.
        That is important for V4.1 special tokens, thinking blocks, and DSML tool calls.
        """
        prepared = prepare_protocol_request(protocol, body)
        opts = prepared.inference_options
        prompt_ids = self.tokenizer.encode_conversation(
            prepared.conversation_request.conversation
        )
        generated, prompt_tokens, finish = self._generate_ids(
            prompt_ids,
            max_tokens=opts.max_tokens,
            temperature=opts.temperature,
            top_p=opts.top_p,
        )
        recipe = _recipe_types()
        request_type = type(prepared.request)
        response_type = {
            "chat_completions": recipe["ChatCompletionResponse"],
            "responses": recipe["ResponsesResponse"],
            "messages": recipe["MessagesResponse"],
        }[protocol]
        response_id = f"nano-{uuid4().hex}"
        model_name = prepared.model_name or self.model_name
        generator = request_type.chunk_generator(
            prepared.conversation_request, response_id, model_name
        )
        if protocol == "chat_completions":
            generator = generator.with_include_usage(prepared.include_usage)
        elif protocol == "responses":
            generator = generator.with_custom_tool_names(prepared.custom_tool_names)
        processor = recipe["StreamProcessor"](
            generator,
            prepared.conversation_request.parsing_options,
            self.tokenizer.recipe_tokenizer,
        )
        response = response_type(response_id, model_name, int(time()), 0, 0)
        reason = (
            recipe["InferenceFinishReason"].Stop
            if finish == "stop"
            else recipe["InferenceFinishReason"].Length
        )
        chunks = [
            recipe["InferenceChunk"].ready(
                prompt_usage=recipe["PromptUsage"](prompt_tokens=prompt_tokens)
            ),
            *(recipe["InferenceChunk"].token(token_id) for token_id in generated),
            recipe["InferenceChunk"].finish(finish_reason=reason),
        ]
        try:
            for chunk in chunks:
                for output in processor.push(chunk):
                    response.append(output)
            if not processor.finished:
                for output in processor.finish():
                    response.append(output)
            return prepared, response.to_json()
        finally:
            processor.close()


def create_app(backend: NanoDeepSeekProtocolBackend):
    """Create a FastAPI app for the major text-generation endpoint families.

    Protocol-aware endpoints use DeepSeek's official V4.1 renderer/parser. The classic
    Completions endpoint intentionally does not apply any chat template.
    """
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, Response
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "HTTP serving needs `pip install 'nano-dsv41f[api]'`"
        ) from exc

    app = FastAPI(title="nano-dsv4.1f CPU API", version="0.1")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "model": backend.model_name}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": backend.model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "nano-dsv4.1f",
                }
            ],
        }

    @app.post("/v1/completions")
    async def completions(request: Request):
        try:
            body = await request.json()
            prompt = body.get("prompt")
            if not isinstance(prompt, str):
                raise ValueError("CPU reference /v1/completions currently requires string prompt")
            if body.get("stream"):
                raise ValueError(
                    "stream=true is not yet exposed by the CPU HTTP adapter; use non-streaming "
                    "requests or the direct backend while vLLM scheduler integration is pending"
                )
            payload = backend.complete_raw(
                prompt,
                max_tokens=body.get("max_tokens"),
                temperature=body.get("temperature"),
                top_p=body.get("top_p"),
            )
            return JSONResponse(payload)
        except (ValueError, RuntimeError) as exc:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": str(exc),
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "invalid_request_error",
                    }
                },
            )

    async def protocol_response(protocol: Protocol, request: Request):
        try:
            body = await request.body()
            prepared = prepare_protocol_request(protocol, body)
            if prepared.stream:
                raise ValueError(
                    "stream=true protocol responses are deferred to the vLLM serving adapter; "
                    "the correctness-first CPU HTTP server currently returns complete responses"
                )
            _prepared, payload = backend.complete_protocol(protocol, body)
            return Response(payload, media_type="application/json")
        except Exception as exc:
            status = int(getattr(exc, "status_code", 400))
            error_body = getattr(exc, "body", None)
            if error_body is not None:
                return Response(error_body, status_code=status, media_type="application/json")
            return JSONResponse(
                status_code=status,
                content={
                    "error": {
                        "message": str(exc),
                        "type": "invalid_request_error" if status < 500 else "internal_error",
                        "param": None,
                        "code": "invalid_request_error" if status < 500 else "internal_error",
                    }
                },
            )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await protocol_response("chat_completions", request)

    @app.post("/v1/responses")
    async def responses(request: Request):
        return await protocol_response("responses", request)

    @app.post("/v1/messages")
    async def messages(request: Request):
        return await protocol_response("messages", request)

    return app
