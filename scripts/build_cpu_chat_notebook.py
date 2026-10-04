#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent

import nbformat as nbf


def _md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(source).strip() + "\n")


def _code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(source).strip() + "\n")


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["metadata"].update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        }
    )

    nb.cells = [
        _md(
            '''
            # nano-dsv4.1f — CPU chat demo

            This notebook is intended to be published as a **public Kaggle notebook**.
            A visitor opens the published version, clicks **Copy & Edit**, starts a CPU
            session, enables Internet if the source package is not attached as a Kaggle
            input, and then chooses **Run All**.

            The notebook uses the official DeepSeek V4.1 protocol renderer/parser through
            `deepseek-recipe`. It does not invent a Jinja chat template. The demo supports
            normal multi-turn Chat Completions semantics and exposes the same backend used
            by `/v1/chat/completions`, `/v1/responses`, `/v1/messages`, and raw
            `/v1/completions`.

            Before publishing, attach a Kaggle dataset containing the exported checkpoint
            (`model.safetensors` + `config.json`) and the frozen `tokenizer.json`.
            '''
        ),
        _code(
            r'''
            # Install the public repo + CPU/API dependencies.
            # If you attach a source wheel as a Kaggle input later, replace this cell with
            # an offline `pip install /kaggle/input/.../*.whl` for a zero-network demo.
            import os
            import subprocess
            import sys

            REPO_REF = os.environ.get("NANO_DSV41F_REF", "main")
            repo = "/kaggle/working/nano-dsv4.1f"
            if not os.path.exists(repo):
                subprocess.check_call(
                    [
                        "git",
                        "clone",
                        "--depth",
                        "1",
                        "--branch",
                        REPO_REF,
                        "https://github.com/xiayicheng3-code/nano-dsv4.1f.git",
                        repo,
                    ]
                )
            if subprocess.check_output(["git", "-C", repo, "status", "--porcelain",
                                        "--untracked-files=no"], text=True).strip():
                raise RuntimeError("Preserve tracked edits before updating the source checkout")
            subprocess.check_call(["git", "-C", repo, "fetch", "--depth", "1", "origin", REPO_REF])
            subprocess.check_call(["git", "-C", repo, "checkout", "--detach", "FETCH_HEAD"])
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "-e", f"{repo}[api]"]
            )
            print("Installed nano-dsv4.1f from", REPO_REF)
            '''
        ),
        _code(
            r'''
            # Locate an attached portable checkpoint and frozen tokenizer.
            from pathlib import Path

            input_root = Path("/kaggle/input")
            model_files = list(input_root.rglob("model.safetensors"))
            model_files = [p for p in model_files if (p.parent / "config.json").exists()]
            if not model_files:
                raise FileNotFoundError(
                    "Attach the exported nano checkpoint as a Kaggle input. Expected "
                    "model.safetensors and config.json in the same directory."
                )
            CHECKPOINT_DIR = model_files[0].parent

            same_dir_tokenizer = CHECKPOINT_DIR / "tokenizer.json"
            if same_dir_tokenizer.exists():
                TOKENIZER_PATH = same_dir_tokenizer
            else:
                tokenizer_files = list(input_root.rglob("tokenizer.json"))
                if not tokenizer_files:
                    raise FileNotFoundError(
                        "Attach the frozen nano tokenizer dataset containing tokenizer.json."
                    )
                TOKENIZER_PATH = tokenizer_files[0]

            print("Checkpoint:", CHECKPOINT_DIR)
            print("Tokenizer :", TOKENIZER_PATH)
            '''
        ),
        _code(
            r'''
            # Batch prompt tokens and optionally verify trained DSpark blocks.
            DEVICE = "cpu"  # "cuda" for an available GPU; GPU performance needs measuring
            PREFILL_CHUNK_SIZE = 32  # 1 restores scalar prefill for comparison
            USE_MTP = False  # enable after DSpark distillation and acceptance evaluation
            DRAFT_TRAINED = False  # set True only for your distilled bundle
            import torch
            from nano_dsv41f.vllm_v41_cpu.api import NanoDeepSeekProtocolBackend

            torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
            backend = NanoDeepSeekProtocolBackend.from_pretrained(
                CHECKPOINT_DIR,
                tokenizer_path=TOKENIZER_PATH,
                dtype=torch.float32, device=DEVICE,
            )
            from nano_dsv41f.vllm_v41_cpu.session import InferenceSession
            backend.session = InferenceSession(backend.model, mtp=USE_MTP,
                draft_trained=DRAFT_TRAINED, prefill_chunk_size=PREFILL_CHUNK_SIZE)
            print("Loaded", backend.model_name, "on", backend.model.device)
            '''
        ),
        _code(
            r'''
            # Multi-turn DeepSeek V4.1 Chat Completions helper.
            import json

            history = []
            REASONING_EFFORT_PRESETS = {"low": 50, "high": 75, "max": 100}

            def reset_chat():
                history.clear()
                backend.reset_cache()

            def chat(message, *, thinking=False, reasoning_effort="high", max_tokens=128, temperature=None):
                if isinstance(reasoning_effort, str):
                    if reasoning_effort not in REASONING_EFFORT_PRESETS:
                        raise ValueError("reasoning_effort must be low, high, max, or an integer in [1, 100]")
                    reasoning_effort = REASONING_EFFORT_PRESETS[reasoning_effort]
                if type(reasoning_effort) is not int or not 1 <= reasoning_effort <= 100:
                    raise ValueError("reasoning_effort must be low, high, max, or an integer in [1, 100]")
                if temperature is None:
                    temperature = 0.0 if getattr(getattr(backend, "session", None), "mtp", False) else 1.0
                request_messages = history + [{"role": "user", "content": message}]
                payload = {
                    "model": backend.model_name,
                    "messages": request_messages,
                    "thinking": {"type": "enabled" if thinking else "disabled"},
                    "reasoning_effort": reasoning_effort,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.95,
                    "stream": False,
                }
                _, raw = backend.complete_protocol("chat_completions", payload)
                response = json.loads(raw)
                assistant = response["choices"][0]["message"]
                history.append({"role": "user", "content": message})
                saved = {"role": "assistant", "content": assistant.get("content") or ""}
                if assistant.get("reasoning_content"):
                    saved["reasoning_content"] = assistant["reasoning_content"]
                if assistant.get("tool_calls"):
                    saved["tool_calls"] = assistant["tool_calls"]
                history.append(saved)
                assistant["_inference_stats"] = dict(getattr(backend, "last_stats", {}))
                return assistant

            def print_reply(reply):
                reasoning = reply.get("reasoning_content")
                if reasoning:
                    print("Reasoning:\n" + reasoning + "\n")
                print("Assistant:\n" + (reply.get("content") or ""))
                from nano_dsv41f.vllm_v41_cpu.session import format_inference_stats
                print(format_inference_stats(reply.get("_inference_stats", {})))
            '''
        ),
        _code(
            r'''
            # Run All reaches this cell and proves the chat path is live.
            reply = chat("Hello! In one short sentence, introduce yourself.", max_tokens=64)
            print_reply(reply)
            '''
        ),
        _md(
            '''
            ## Chat here

            The next cell creates a small in-notebook chat control. Type a message and press
            **Send**; the conversation history is preserved until you press **Reset**.
            Input, Send, Reset, and generation settings are locked while a response is
            being generated. They unlock after completion or an error.
            Enable Thinking to choose Low (50), High (75), Max (100), or a custom
            integer effort from 1 to 100. Max tokens caps the entire generated response,
            including reasoning; effort is a prompt setting rather than a token limit.
            `USE_MTP=True` selects batched greedy verification and makes chat use
            temperature zero. Prompt prefill is chunked even when MTP is off.
            '''
        ),
        _code(
            r'''
            # Kaggle/Jupyter-native chat controls: no public tunnel or separate web app needed.
            import asyncio
            import ipywidgets as widgets
            from IPython.display import display

            if globals().get("_chat_task") is not None and not _chat_task.done():
                raise RuntimeError("Wait for the current response before rebuilding chat controls")
            message_box = widgets.Textarea(
                placeholder="Type a message…",
                layout=widgets.Layout(width="100%", height="90px"),
            )
            thinking_box = widgets.Checkbox(value=False, description="Thinking")
            effort_box = widgets.Dropdown(
                options=[("Low (50)", 50), ("High (75)", 75), ("Max (100)", 100),
                         ("Custom (1–100)", "custom")],
                value=75,
                description="Effort:",
                disabled=True,
            )
            custom_effort_box = widgets.BoundedIntText(
                value=75, min=1, max=100, step=1, description="Value:", disabled=True,
            )
            max_tokens_box = widgets.BoundedIntText(
                value=128,
                min=1,
                max=1024,
                step=1,
                description="Max tokens:",
            )
            send_button = widgets.Button(description="Send", button_style="primary")
            reset_button = widgets.Button(description="Reset")
            chat_output = widgets.Output(layout=widgets.Layout(border="1px solid #ddd"))
            status_label = widgets.Label(value="Ready")
            _chat_busy = False
            _chat_task = None

            def _update_effort_controls(_=None):
                if effort_box.value != "custom":
                    custom_effort_box.value = effort_box.value
                effort_box.disabled = _chat_busy or not thinking_box.value
                custom_effort_box.disabled = (_chat_busy or not thinking_box.value
                                              or effort_box.value != "custom")

            def _set_busy(busy):
                global _chat_busy
                _chat_busy = busy
                for control in (message_box, thinking_box, max_tokens_box, send_button, reset_button):
                    control.disabled = busy
                _update_effort_controls()
                send_button.description = "Generating…" if busy else "Send"
                status_label.value = "Generating response…" if busy else "Ready"

            async def _generate_reply(message, settings):
                try:
                    # Keep the kernel event loop available for widget updates and
                    # rejecting extra clicks. All widget writes stay on this loop.
                    reply = await asyncio.to_thread(chat, message, **settings)
                    reasoning = reply.get("reasoning_content")
                    if reasoning:
                        chat_output.append_stdout(f"Reasoning: {reasoning}\n")
                    chat_output.append_stdout(f"Assistant: {reply.get('content') or ''}\n")
                    from nano_dsv41f.vllm_v41_cpu.session import format_inference_stats
                    chat_output.append_stdout(format_inference_stats(reply.get("_inference_stats", {})) + "\n\n")
                except Exception as exc:
                    message_box.value = message
                    chat_output.append_stdout(f"Error: {exc}\n\n")
                finally:
                    _set_busy(False)

            def _send(_):
                global _chat_task
                if _chat_busy:
                    return
                message = message_box.value.strip()
                if not message:
                    return
                settings = dict(thinking=thinking_box.value,
                                reasoning_effort=custom_effort_box.value,
                                max_tokens=max_tokens_box.value)
                _set_busy(True)
                message_box.value = ""
                chat_output.append_stdout(f"You: {message}\n")
                try:
                    _chat_task = asyncio.get_running_loop().create_task(_generate_reply(message, settings))
                except Exception as exc:
                    message_box.value = message
                    _set_busy(False)
                    chat_output.append_stdout(f"Error: {exc}\n\n")

            def _reset(_):
                if _chat_busy:
                    return
                reset_chat()
                chat_output.clear_output()
                chat_output.append_stdout("Conversation reset.\n\n")

            thinking_box.observe(_update_effort_controls, names="value")
            effort_box.observe(_update_effort_controls, names="value")
            send_button.on_click(_send)
            reset_button.on_click(_reset)
            display(
                widgets.VBox(
                    [
                        message_box,
                        widgets.HBox([thinking_box, effort_box, custom_effort_box]),
                        widgets.HBox([max_tokens_box, send_button, reset_button]),
                        status_label,
                        chat_output,
                    ]
                )
            )
            '''
        ),
        _md(
            '''
            ## Or call the helper directly

            ```python
            print_reply(chat("Why is sparse attention useful?"))
            print_reply(chat("Now explain that more simply."))  # keeps the same history
            ```

            For DeepSeek V4.1 reasoning mode, use `low` (50), `high` (75, the default),
            `max` (100), or any integer in `[1, 100]`, matching the open-weight encoder:

            ```python
            print_reply(
                chat(
                    "Solve 37*43 carefully.",
                    thinking=True,
                    reasoning_effort="high",
                    max_tokens=256,
                )
            )
            print_reply(chat("Try a shorter explanation.", thinking=True, reasoning_effort=30))
            ```

            Use `reset_chat()` to start a fresh conversation.
            '''
        ),
        _code(
            r'''
            # Optional: expose the same model through local OpenAI/Anthropic-compatible HTTP routes.
            # Run this cell only when you want an API server inside the Kaggle session.
            # from nano_dsv41f.vllm_v41_cpu.api import create_app
            # import uvicorn
            # app = create_app(backend)
            # uvicorn.run(app, host="127.0.0.1", port=8000)
            '''
        ),
    ]
    for i, cell in enumerate(nb.cells):
        cell.id = f"cpu-chat-{i:02d}"
    return nb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("notebooks/nano_dsv41f_cpu_chat.ipynb"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build_notebook(), args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
