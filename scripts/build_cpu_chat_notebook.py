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
            '''
            # Install the public repo + CPU/API dependencies.
            # If you attach a source wheel as a Kaggle input later, replace this cell with
            # an offline `pip install /kaggle/input/.../*.whl` for a zero-network demo.
            import os
            import subprocess
            import sys

            REPO_REF = os.environ.get("NANO_DSV41F_REF", "codex/v41-cpu-inference")
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
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "-e", f"{repo}[api]"]
            )
            print("Installed nano-dsv4.1f from", REPO_REF)
            '''
        ),
        _code(
            '''
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
            '''
            # Load the correctness-first incremental CPU runtime.
            import torch
            from nano_dsv41f.vllm_v41_cpu.api import NanoDeepSeekProtocolBackend

            torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
            backend = NanoDeepSeekProtocolBackend.from_pretrained(
                CHECKPOINT_DIR,
                tokenizer_path=TOKENIZER_PATH,
                dtype=torch.float32,
            )
            print("Loaded", backend.model_name, "on", backend.model.device)
            '''
        ),
        _code(
            '''
            # Multi-turn DeepSeek V4.1 Chat Completions helper.
            import json

            history = []
            VALID_REASONING_EFFORTS = {"minimal", "low", "medium", "high", "xhigh", "max"}

            def reset_chat():
                history.clear()

            def chat(message, *, thinking=False, reasoning_effort="high", max_tokens=128):
                if reasoning_effort not in VALID_REASONING_EFFORTS:
                    raise ValueError(
                        f"reasoning_effort must be one of {sorted(VALID_REASONING_EFFORTS)}"
                    )
                request_messages = history + [{"role": "user", "content": message}]
                payload = {
                    "model": backend.model_name,
                    "messages": request_messages,
                    "thinking": {"type": "enabled" if thinking else "disabled"},
                    "reasoning_effort": reasoning_effort,
                    "max_tokens": max_tokens,
                    "temperature": 1.0,
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
                return assistant

            def print_reply(reply):
                reasoning = reply.get("reasoning_content")
                if reasoning:
                    print("Reasoning:\n" + reasoning + "\n")
                print("Assistant:\n" + (reply.get("content") or ""))
            '''
        ),
        _code(
            '''
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
            Thinking mode and DeepSeek reasoning effort can be changed from the controls.
            '''
        ),
        _code(
            '''
            # Kaggle/Jupyter-native chat controls: no public tunnel or separate web app needed.
            import ipywidgets as widgets
            from IPython.display import display

            message_box = widgets.Textarea(
                placeholder="Type a message…",
                layout=widgets.Layout(width="100%", height="90px"),
            )
            thinking_box = widgets.Checkbox(value=False, description="Thinking")
            effort_box = widgets.Dropdown(
                options=["minimal", "low", "medium", "high", "xhigh", "max"],
                value="high",
                description="Effort:",
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

            def _send(_):
                message = message_box.value.strip()
                if not message:
                    return
                message_box.value = ""
                with chat_output:
                    print(f"You: {message}")
                    try:
                        reply = chat(
                            message,
                            thinking=thinking_box.value,
                            reasoning_effort=effort_box.value,
                            max_tokens=max_tokens_box.value,
                        )
                        reasoning = reply.get("reasoning_content")
                        if reasoning:
                            print(f"Reasoning: {reasoning}")
                        print(f"Assistant: {reply.get('content') or ''}\n")
                    except Exception as exc:
                        print(f"Error: {exc}\n")

            def _reset(_):
                reset_chat()
                chat_output.clear_output()
                with chat_output:
                    print("Conversation reset.\n")

            send_button.on_click(_send)
            reset_button.on_click(_reset)
            display(
                widgets.VBox(
                    [
                        message_box,
                        widgets.HBox([thinking_box, effort_box, max_tokens_box]),
                        widgets.HBox([send_button, reset_button]),
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

            For DeepSeek-style reasoning mode, use the API's named reasoning levels
            (`minimal`, `low`, `medium`, `high`, `xhigh`, or `max`):

            ```python
            print_reply(
                chat(
                    "Solve 37*43 carefully.",
                    thinking=True,
                    reasoning_effort="high",
                    max_tokens=256,
                )
            )
            ```

            Use `reset_chat()` to start a fresh conversation.
            '''
        ),
        _code(
            '''
            # Optional: expose the same model through local OpenAI/Anthropic-compatible HTTP routes.
            # Run this cell only when you want an API server inside the Kaggle session.
            # from nano_dsv41f.vllm_v41_cpu.api import create_app
            # import uvicorn
            # app = create_app(backend)
            # uvicorn.run(app, host="127.0.0.1", port=8000)
            '''
        ),
    ]
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
