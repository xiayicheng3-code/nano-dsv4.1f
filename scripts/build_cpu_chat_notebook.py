#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf


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
        nbf.v4.new_markdown_cell(
            """# nano-dsv4.1f — CPU chat demo\n\n"
            "This notebook is intended to be published as a **public Kaggle notebook**. "
            "A visitor opens the published version, clicks **Copy & Edit**, starts a CPU "
            "session, enables Internet if the source package is not attached as a Kaggle "
            "input, and then chooses **Run All**.\n\n"
            "The notebook uses the official DeepSeek V4.1 protocol renderer/parser through "
            "`deepseek-recipe`. It does not invent a Jinja chat template. The demo supports "
            "normal multi-turn Chat Completions semantics and exposes the same backend used "
            "by `/v1/chat/completions`, `/v1/responses`, `/v1/messages`, and raw "
            "`/v1/completions`.\n\n"
            "Before publishing, attach a Kaggle dataset containing the exported checkpoint "
            "(`model.safetensors` + `config.json`) and the frozen `tokenizer.json`."
            """
        ),
        nbf.v4.new_code_cell(
            """# Install the public repo + CPU/API dependencies.\n"
            "# If you attach a source wheel as a Kaggle input later, replace this cell with\n"
            "# an offline `pip install /kaggle/input/.../*.whl` for a zero-network demo.\n"
            "import os, subprocess, sys\n\n"
            "REPO_REF = os.environ.get('NANO_DSV41F_REF', 'codex/v41-cpu-inference')\n"
            "repo = '/kaggle/working/nano-dsv4.1f'\n"
            "if not os.path.exists(repo):\n"
            "    subprocess.check_call([\n"
            "        'git', 'clone', '--depth', '1', '--branch', REPO_REF,\n"
            "        'https://github.com/xiayicheng3-code/nano-dsv4.1f.git', repo,\n"
            "    ])\n"
            "subprocess.check_call([\n"
            "    sys.executable, '-m', 'pip', 'install', '-q', '-e', f'{repo}[api]'\n"
            "])\n"
            "print('Installed nano-dsv4.1f from', REPO_REF)"
        ),
        nbf.v4.new_code_cell(
            """# Locate an attached portable checkpoint and frozen tokenizer.\n"
            "from pathlib import Path\n\n"
            "input_root = Path('/kaggle/input')\n"
            "model_files = list(input_root.rglob('model.safetensors'))\n"
            "model_files = [p for p in model_files if (p.parent / 'config.json').exists()]\n"
            "if not model_files:\n"
            "    raise FileNotFoundError(\n"
            "        'Attach the exported nano checkpoint as a Kaggle input. Expected '\n"
            "        'model.safetensors and config.json in the same directory.'\n"
            "    )\n"
            "CHECKPOINT_DIR = model_files[0].parent\n\n"
            "same_dir_tokenizer = CHECKPOINT_DIR / 'tokenizer.json'\n"
            "if same_dir_tokenizer.exists():\n"
            "    TOKENIZER_PATH = same_dir_tokenizer\n"
            "else:\n"
            "    tokenizer_files = list(input_root.rglob('tokenizer.json'))\n"
            "    if not tokenizer_files:\n"
            "        raise FileNotFoundError(\n"
            "            'Attach the frozen nano tokenizer dataset containing tokenizer.json.'\n"
            "        )\n"
            "    TOKENIZER_PATH = tokenizer_files[0]\n\n"
            "print('Checkpoint:', CHECKPOINT_DIR)\n"
            "print('Tokenizer :', TOKENIZER_PATH)"
        ),
        nbf.v4.new_code_cell(
            """# Load the correctness-first incremental CPU runtime.\n"
            "import torch\n"
            "from nano_dsv41f.vllm_v41_cpu.api import NanoDeepSeekProtocolBackend\n\n"
            "torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))\n"
            "backend = NanoDeepSeekProtocolBackend.from_pretrained(\n"
            "    CHECKPOINT_DIR,\n"
            "    tokenizer_path=TOKENIZER_PATH,\n"
            "    dtype=torch.float32,\n"
            ")\n"
            "print('Loaded', backend.model_name, 'on', backend.model.device)"
        ),
        nbf.v4.new_code_cell(
            """# Multi-turn DeepSeek V4.1 Chat Completions helper.\n"
            "import json\n\n"
            "history = []\n"
            "VALID_REASONING_EFFORTS = {'minimal', 'low', 'medium', 'high', 'xhigh', 'max'}\n\n"
            "def reset_chat():\n"
            "    history.clear()\n\n"
            "def chat(message, *, thinking=False, reasoning_effort='high', max_tokens=128):\n"
            "    if reasoning_effort not in VALID_REASONING_EFFORTS:\n"
            "        raise ValueError(f'reasoning_effort must be one of {sorted(VALID_REASONING_EFFORTS)}')\n"
            "    request_messages = history + [{'role': 'user', 'content': message}]\n"
            "    payload = {\n"
            "        'model': backend.model_name,\n"
            "        'messages': request_messages,\n"
            "        'thinking': {'type': 'enabled' if thinking else 'disabled'},\n"
            "        'reasoning_effort': reasoning_effort,\n"
            "        'max_tokens': max_tokens,\n"
            "        'temperature': 1.0,\n"
            "        'top_p': 0.95,\n"
            "        'stream': False,\n"
            "    }\n"
            "    _, raw = backend.complete_protocol('chat_completions', payload)\n"
            "    response = json.loads(raw)\n"
            "    assistant = response['choices'][0]['message']\n"
            "    history.append({'role': 'user', 'content': message})\n"
            "    saved = {'role': 'assistant', 'content': assistant.get('content') or ''}\n"
            "    if assistant.get('reasoning_content'):\n"
            "        saved['reasoning_content'] = assistant['reasoning_content']\n"
            "    if assistant.get('tool_calls'):\n"
            "        saved['tool_calls'] = assistant['tool_calls']\n"
            "    history.append(saved)\n"
            "    return assistant\n\n"
            "def print_reply(reply):\n"
            "    reasoning = reply.get('reasoning_content')\n"
            "    if reasoning:\n"
            "        print('Reasoning:\\n' + reasoning + '\\n')\n"
            "    print('Assistant:\\n' + (reply.get('content') or ''))"
        ),
        nbf.v4.new_code_cell(
            """# Run All reaches this cell and proves the chat path is live.\n"
            "reply = chat('Hello! In one short sentence, introduce yourself.', max_tokens=64)\n"
            "print_reply(reply)"
        ),
        nbf.v4.new_markdown_cell(
            """## Chat here\n\n"
            "The next cell creates a small in-notebook chat control. Type a message and press "
            "**Send**; the conversation history is preserved until you press **Reset**. "
            "Thinking mode and DeepSeek reasoning effort can be changed from the controls."
        ),
        nbf.v4.new_code_cell(
            """# Kaggle/Jupyter-native chat controls: no public tunnel or separate web app needed.\n"
            "import ipywidgets as widgets\n"
            "from IPython.display import display\n\n"
            "message_box = widgets.Textarea(\n"
            "    placeholder='Type a message…',\n"
            "    layout=widgets.Layout(width='100%', height='90px'),\n"
            ")\n"
            "thinking_box = widgets.Checkbox(value=False, description='Thinking')\n"
            "effort_box = widgets.Dropdown(\n"
            "    options=['minimal', 'low', 'medium', 'high', 'xhigh', 'max'],\n"
            "    value='high',\n"
            "    description='Effort:',\n"
            ")\n"
            "max_tokens_box = widgets.BoundedIntText(\n"
            "    value=128, min=1, max=1024, step=1, description='Max tokens:'\n"
            ")\n"
            "send_button = widgets.Button(description='Send', button_style='primary')\n"
            "reset_button = widgets.Button(description='Reset')\n"
            "chat_output = widgets.Output(layout=widgets.Layout(border='1px solid #ddd'))\n\n"
            "def _send(_):\n"
            "    message = message_box.value.strip()\n"
            "    if not message:\n"
            "        return\n"
            "    message_box.value = ''\n"
            "    with chat_output:\n"
            "        print(f'You: {message}')\n"
            "        try:\n"
            "            reply = chat(\n"
            "                message,\n"
            "                thinking=thinking_box.value,\n"
            "                reasoning_effort=effort_box.value,\n"
            "                max_tokens=max_tokens_box.value,\n"
            "            )\n"
            "            reasoning = reply.get('reasoning_content')\n"
            "            if reasoning:\n"
            "                print(f'Reasoning: {reasoning}')\n"
            "            print(f\"Assistant: {reply.get('content') or ''}\\n\")\n"
            "        except Exception as exc:\n"
            "            print(f'Error: {exc}\\n')\n\n"
            "def _reset(_):\n"
            "    reset_chat()\n"
            "    chat_output.clear_output()\n"
            "    with chat_output:\n"
            "        print('Conversation reset.\\n')\n\n"
            "send_button.on_click(_send)\n"
            "reset_button.on_click(_reset)\n"
            "display(widgets.VBox([\n"
            "    message_box,\n"
            "    widgets.HBox([thinking_box, effort_box, max_tokens_box]),\n"
            "    widgets.HBox([send_button, reset_button]),\n"
            "    chat_output,\n"
            "]))"
        ),
        nbf.v4.new_markdown_cell(
            """## Or call the helper directly\n\n"
            "```python\n"
            "print_reply(chat('Why is sparse attention useful?'))\n"
            "print_reply(chat('Now explain that more simply.'))  # keeps the same history\n"
            "```\n\n"
            "For DeepSeek-style reasoning mode, use the API's named reasoning levels "
            "(`minimal`, `low`, `medium`, `high`, `xhigh`, or `max`):\n\n"
            "```python\n"
            "print_reply(chat('Solve 37*43 carefully.', thinking=True, reasoning_effort='high', max_tokens=256))\n"
            "```\n\n"
            "Use `reset_chat()` to start a fresh conversation."
        ),
        nbf.v4.new_code_cell(
            """# Optional: expose the same model through local OpenAI/Anthropic-compatible HTTP routes.\n"
            "# Run this cell only when you want an API server inside the Kaggle session.\n"
            "# from nano_dsv41f.vllm_v41_cpu.api import create_app\n"
            "# import uvicorn\n"
            "# app = create_app(backend)\n"
            "# uvicorn.run(app, host='127.0.0.1', port=8000)"
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
