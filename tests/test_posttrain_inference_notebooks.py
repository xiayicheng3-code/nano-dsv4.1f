import json
from pathlib import Path
import sys
import nbformat
import pytest
import asyncio
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from build_posttrain_inference_notebooks import build
from build_cpu_chat_notebook import build_notebook


def test_generated_export_and_inference_notebooks_compile(tmp_path):
    for path in build(tmp_path):
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        for cell in nb.cells:
            if cell.cell_type == 'code':
                compile(cell.source, str(path), 'exec')
        committed = json.loads((ROOT / 'notebooks' / path.name).read_text())
        assert json.loads(path.read_text()) == committed
    for cell in build_notebook().cells:
        if cell.cell_type == 'code':
            compile(cell.source, 'cpu_chat', 'exec')


@pytest.mark.parametrize('notebook_name', ['nano_dsv41f_cpu_chat.ipynb', 'nano_dsv41f_sft_inference_cpu.ipynb'])
@pytest.mark.parametrize('fails', [False, True])
def test_widget_locks_inputs_rejects_extra_clicks_and_recovers(notebook_name, fails, monkeypatch):
    pytest.importorskip('ipywidgets')
    import IPython.display
    monkeypatch.setattr(IPython.display, 'display', lambda *_: None)
    nb = nbformat.read(ROOT / 'notebooks' / notebook_name, as_version=4)
    cell = next(c.source for c in nb.cells if c.cell_type == 'code' and 'async def _generate_reply' in c.source)
    started, release = threading.Event(), threading.Event()
    calls, resets = [], []
    def chat(message, **settings):
        calls.append((message, settings))
        started.set()
        assert release.wait(5), 'test did not release generation'
        if fails:
            raise ValueError('Test generation failed')
        return {'content': 'Reply', 'reasoning_content': 'Reasoning'}
    namespace = {'chat': chat, 'reset_chat': lambda: resets.append(True)}
    exec(compile(cell, notebook_name, 'exec'), namespace)

    async def scenario():
        namespace['thinking_box'].value = True
        namespace['effort_box'].value = 'custom'
        namespace['custom_effort_box'].value = 37
        namespace['message_box'].value = 'Hello'
        namespace['_send'](None)
        assert namespace['_chat_busy']
        controls = ['message_box', 'thinking_box', 'effort_box', 'custom_effort_box',
                    'max_tokens_box', 'send_button', 'reset_button']
        assert all(namespace[name].disabled for name in controls)
        try:
            assert await asyncio.to_thread(started.wait, 2)
            # The kernel loop remains responsive while CPU inference is blocked.
            namespace['_send'](None)
            namespace['_reset'](None)
            assert not resets
            namespace['custom_effort_box'].value = 100  # simulated queued settings update
        finally:
            release.set()
        await namespace['_chat_task']
        assert calls == [('Hello', {'thinking': True, 'reasoning_effort': 37, 'max_tokens': 128})]
        assert not namespace['_chat_busy']
        assert all(not namespace[name].disabled for name in controls)
        assert namespace['status_label'].value == 'Ready'
        output = ''.join(item.get('text', '') for item in namespace['chat_output'].outputs)
        if fails:
            assert 'Error: Test generation failed' in output
            assert namespace['message_box'].value == 'Hello'
        else:
            assert 'Assistant: Reply' in output and 'Reasoning: Reasoning' in output
            assert namespace['message_box'].value == ''
        namespace['_reset'](None)
        assert resets == [True]
        namespace['thinking_box'].value = False
        assert namespace['effort_box'].disabled and namespace['custom_effort_box'].disabled
    asyncio.run(scenario())


@pytest.mark.parametrize('effort,expected', [('low', 50), ('high', 75), ('max', 100), (37, 37)])
def test_chat_helper_passes_numeric_effort(effort, expected):
    cell = build_notebook().cells[4].source
    payloads = []
    class Backend:
        model_name = 'nano'
        def complete_protocol(self, protocol, payload):
            payloads.append(payload)
            return None, json.dumps({'choices': [{'message': {'content': 'Hello'}}]})
    namespace = {'backend': Backend()}
    exec(compile(cell, 'chat-helper', 'exec'), namespace)
    namespace['chat']('Hi', thinking=True, reasoning_effort=effort)
    assert payloads[0]['reasoning_effort'] == expected
    assert payloads[0]['thinking'] == {'type': 'enabled'}
    assert len(namespace['history']) == 2
