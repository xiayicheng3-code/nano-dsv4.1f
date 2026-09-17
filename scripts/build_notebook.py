"""Generate both Kaggle entry points from one maintained runtime bootstrap."""
from __future__ import annotations
import argparse
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def build(output: Path, *, combined=False):
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    nb.cells = [
        nbf.v4.new_markdown_cell(
            '# nano-dsv4.1f — Kaggle TPU v5e-8 operator validation\n\n'
            'Select TPU v5e-8 and Internet. Start a fresh session and run all cells. '
            'The bootstrap pins JAX/jaxlib/libtpu before any JAX import. '
            'Set `NANO_DSV41F_REF` to a branch or commit before bootstrap to test it; default is `main`. '
            'Training runs in a fresh Python process. Results are written under `/kaggle/working`. '
            'This synthetic test is not the official pretraining data/recipe.'),
        nbf.v4.new_code_cell((ROOT / 'scripts/kaggle_bootstrap.py').read_text()),
    ]
    if not combined:
        nb.cells += [
            nbf.v4.new_markdown_cell('## Operator parity and timing\n\nForward/backward parity precedes timing. Compilation and warmup are excluded.'),
            nbf.v4.new_code_cell("subprocess.run([sys.executable, '-u', 'scripts/benchmark_tpu_operators.py', '--output', '/kaggle/working/operator-benchmark.json'], check=True)"),
        ]
    nb.cells += [
        nbf.v4.new_markdown_cell('## Packed, multi-expert and late-indexer smoke\n\nOne Pallas preflight, then 10 base + 10 late steps. Report includes synchronized timing, compiler memory, collectives, utilization and loss trajectories.'),
        nbf.v4.new_code_cell("subprocess.run([sys.executable, '-u', 'scripts/run_combined_tpu_smoke.py', '--output', '/kaggle/working/combined-smoke.json'], check=True)"),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, output)
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output:
        build(args.output)
    else:
        build(ROOT / 'notebooks/nano_dsv41f_kaggle.ipynb')
        build(ROOT / 'notebooks/nano_dsv41f_combined_smoke.ipynb', combined=True)
