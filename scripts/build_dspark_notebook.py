"""Generate the standalone Kaggle DSpark distillation notebook."""
from pathlib import Path
from textwrap import dedent
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def build(output=ROOT/'notebooks/nano_dsv41f_dspark_distillation.ipynb'):
    cells = []
    def md(text):
        cells.append(nbf.v4.new_markdown_cell(dedent(text).strip()+'\n'))
    def code(text):
        source = dedent(text).strip()+'\n'
        compile(source, 'dspark-notebook', 'exec')
        cells.append(nbf.v4.new_code_cell(source))
    md('''
    # Distill DSpark with the SFT backbone frozen

    Select a **GPU accelerator** (T4 is the intended first target), enable Internet,
    and attach (1) the exported SFT safetensors bundle and (2) the prepared format-v3
    midtrain/SFT corpus with its tokenizer and SFT shards. Optimizer files from
    pretraining are not needed. `DEVICE='cpu'` supports a short smoke test.

    This stage trains only the existing DSpark drafter. It matches the finished
    model's logits on existing SFT responses, with assistant-only supervision.
    Responses are **teacher-forced corpus text**, not newly generated target rollouts.
    The dataset split is preserved; crops stay inside individual packed conversations.
    Long conversations are cropped to `SEQ_LEN`, resetting positions within each crop.
    The saved model's original context configuration remains intact.

    Loss: `0.1 * token CE + 0.9 * full probability L1 + 1.0 * confidence BCE`, with
    early-position weighting. The confidence target is detached `1 - L1/2`.
    These are defaults from the DSpark paper, not a claim to reproduce an unpublished
    V4.1 training recipe. Full probability distributions are used at selected anchors.

    Training uses the eager Torch serving operators, sparse retrieval, and FP32.
    Start with a small step budget to measure your GPU's memory use and speed.
    This notebook has CPU correctness coverage; T4 throughput is not yet measured.
    ''')
    code('''
    SOURCE_REF = 'main'  # durable default for the maintained notebook
    MODEL_DIR = ''  # blank: locate exactly one export_manifest.json under /kaggle/input
    CORPUS_DIR = ''  # blank: locate exactly one posttrain_manifest.json under /kaggle/input
    RESUME = ''  # optional prior run directory, attached under /kaggle/input
    DEVICE = 'cuda'  # one GPU; use cpu for a smoke test
    STEPS = 1000  # total target, including any resumed steps; start with 10 to smoke-test
    SEQ_LEN = 512
    ANCHORS = 8
    LEARNING_RATE = 1e-4
    CE_WEIGHT, DISTRIBUTION_WEIGHT, CONFIDENCE_WEIGHT = 0.1, 0.9, 1.0
    EVAL_EVERY = 100
    EVAL_BATCHES = 4
    CHECKPOINT_EVERY = 100
    ROLLOUT_TOKENS = 16
    HOURS = 8.0  # runner reserves time to checkpoint/export before Kaggle's hard limit
    ''')
    code('''
    import os
    import sys
    import subprocess
    from pathlib import Path
    import time

    if any(name in sys.modules for name in ('jax', 'nano_dsv41f')):
        raise RuntimeError('Restart the kernel before updating source and dependencies')
    os.environ['JAX_PLATFORMS'] = 'cpu'  # JAX is only an import dependency here; Torch owns the GPU
    os.environ['PIP_CACHE_DIR'] = '/kaggle/temp/nano-dspark-pip-cache'
    REPO = Path('/kaggle/temp/nano-dspark-source')
    if not REPO.exists():
        subprocess.run(['git', 'clone', '--depth', '1',
            'https://github.com/xiayicheng3-code/nano-dsv4.1f.git', str(REPO)], check=True)
    if subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain',
        '--untracked-files=no'], text=True).strip():
        raise RuntimeError('Preserve tracked edits before updating the temporary checkout')
    subprocess.run(['git', '-C', str(REPO), 'fetch', '--depth', '1', 'origin', SOURCE_REF], check=True)
    subprocess.run(['git', '-C', str(REPO), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
        '-r', str(REPO/'requirements-posttrain-inference.txt')], check=True)
    import importlib.util
    if importlib.util.find_spec('torch') is None:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'torch'], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', str(REPO), '--no-deps'], check=True)
    sys.path.insert(0, str(REPO/'src'))
    sys.path.insert(0, str(REPO/'scripts'))
    print('Source:', subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip())
    ''')
    code('''
    import torch
    import json
    from nano_dsv41f.portable_bundle import verify_portable_bundle

    def locate(directory, filename):
        if directory:
            root = Path(directory)
            if not (root/filename).is_file():
                raise FileNotFoundError(f'{filename} is missing from {root}')
            return root
        found = list(Path('/kaggle/input').rglob(filename))
        if len(found) != 1:
            raise ValueError(f'Found {len(found)} {filename} files; set its directory explicitly')
        return found[0].parent

    model_root = locate(MODEL_DIR, 'export_manifest.json')
    corpus_root = locate(CORPUS_DIR, 'posttrain_manifest.json')
    verified = verify_portable_bundle(model_root)
    if DEVICE == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Enable a Kaggle GPU accelerator or set DEVICE=cpu')
    OUTPUT = Path('/kaggle/working') / time.strftime('dspark-%Y%m%d-%H%M%S', time.gmtime())
    print('Frozen model:', model_root)
    print('SFT corpus:', corpus_root)
    print('Output:', OUTPUT)
    print('Device:', torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU')
    print('Original context limit:', verified['max_position_embeddings'])
    ''')
    md('''
    ## Train and measure

    The first evaluation measures the initial drafter. Later evaluations report
    held-out CE, probability L1, overlap, confidence error and agreement per block
    position. Teacher-forced overlap/agreement are diagnostic proxies. The separate
    `rollout_greedy_acceptance` comes from actual target-verified autoregressive
    continuations of held-out assistant prefixes (up to two examples per evaluation).
    Evaluation prompts and anchors are deterministic for before/after comparisons.

    Checkpoints contain only DSpark weights and its optimizer; the original bundle
    remains the frozen source on resume. Attach a saved run, set `RESUME` to its
    directory, keep data/model/training settings unchanged, and raise `STEPS` if
    necessary. Each launch writes to a fresh output directory and retains two recent
    checkpoints. The runner exits before its deadline and exports completed updates.
    ''')
    code('''
    deadline = time.time() + HOURS*3600
    command = [sys.executable, '-u', str(REPO/'scripts/train_dspark.py'),
        '--model-dir', str(model_root), '--corpus', str(corpus_root),
        '--output', str(OUTPUT), '--device', DEVICE, '--steps', str(STEPS),
        '--seq-len', str(SEQ_LEN), '--anchors', str(ANCHORS),
        '--learning-rate', str(LEARNING_RATE), '--ce-weight', str(CE_WEIGHT),
        '--distribution-weight', str(DISTRIBUTION_WEIGHT), '--confidence-weight', str(CONFIDENCE_WEIGHT),
        '--eval-every', str(EVAL_EVERY), '--eval-batches', str(EVAL_BATCHES),
        '--checkpoint-every', str(CHECKPOINT_EVERY), '--rollout-tokens', str(ROLLOUT_TOKENS),
        '--deadline-unix', str(deadline)]
    if RESUME:
        command += ['--resume', RESUME]
    subprocess.run(command, check=True)
    ''')
    code('''
    from IPython.display import display, FileLink
    summary = json.loads((OUTPUT/'summary.json').read_text())
    print('Status:', summary['status'], 'completed DSpark updates:', summary['completed_steps'])
    print('Backbone unchanged:', summary.get('backbone_unchanged'))
    for evaluation in summary['evaluations']:
        print(json.dumps(evaluation, indent=2))
    bundle = OUTPUT/'bundle'
    if bundle.exists():
        result = verify_portable_bundle(bundle)
        assert result['backbone_bitwise_preserved']
        print('Verified inference bundle:', bundle)
        display(FileLink(str(bundle/'dspark_training.json')))
    display(FileLink(str(OUTPUT/'summary.json')))
    display(FileLink(str(OUTPUT/'metrics.jsonl')))
    ''')
    md('''
    ## Save and use the trained drafter

    Save this notebook's outputs. `bundle/` is a complete inference bundle with the
    original backbone tensors and updated DSpark weights; attach it to the SFT chat
    notebook. Preserve `checkpoints/` as well to resume this stage.

    MTP remains off in ordinary chat. For an explicit greedy diagnostic with this
    trained bundle, load its model, then use
    `InferenceSession(model, mtp=True, draft_trained=True)` and `temperature=0`.
    Training a head does not guarantee useful acceptance. Compare held-out metrics
    before choosing it for serving. Target verification now batches proposal tokens
    and rolls back rejected suffixes; measure throughput against MTP-off decoding.

    This stage leaves backbone chat quality unchanged. Additional assistant SFT
    would be a separate experiment, followed by refreshing the drafter against the
    updated backbone. No new dataset or backbone update is needed for this run.
    ''')
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.kernelspec = {'display_name':'Python 3', 'language':'python', 'name':'python3'}
    for i, cell in enumerate(nb.cells):
        cell.id = f'dspark-distill-{i:02d}'
    nbf.validate(nb)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, output)
    return output


if __name__ == '__main__':
    print(build())
