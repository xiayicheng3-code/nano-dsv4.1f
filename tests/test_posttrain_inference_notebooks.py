import json
from pathlib import Path
import sys
import nbformat

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
