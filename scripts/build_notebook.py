from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
SOURCE_GLOBS = [
    "pyproject.toml",
    "src/nano_dsv41f/*.py",
]


def iter_sources():
    for pattern in SOURCE_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            if path.is_file():
                yield path


def writefile_cell(path: Path) -> str:
    rel = path.relative_to(ROOT).as_posix()
    content = path.read_text(encoding="utf-8")
    return (
        "from pathlib import Path\n"
        f"p = Path({rel!r})\n"
        "p.parent.mkdir(parents=True, exist_ok=True)\n"
        f"p.write_text({content!r}, encoding='utf-8')\n"
        f"print('wrote {rel}')"
    )


def build(output: Path) -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            "# nano-dsv4.1f — generated Kaggle notebook\n\n"
            "This notebook is generated from the maintained repository sources. "
            "Edit the Python package, not this notebook."
        ),
        nbf.v4.new_code_cell(
            "# Kaggle usually provides JAX/TPU support already; install only project extras.\n"
            "%pip install -q nbformat pytest"
        ),
    ]

    nb["cells"].extend(nbf.v4.new_code_cell(writefile_cell(path)) for path in iter_sources())
    nb["cells"].append(
        nbf.v4.new_code_cell(
            "%pip install -q -e .\n"
            "import jax\n"
            "print(jax.devices())"
        )
    )
    nb["cells"].append(
        nbf.v4.new_markdown_cell(
            "## Training entry point\n\n"
            "The full TPU training cell is intentionally added only after the reference "
            "model and sharding tests are stable."
        )
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, output)
    print(f"wrote {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "notebooks" / "nano_dsv41f_kaggle.ipynb",
    )
    args = parser.parse_args()
    build(args.output)
