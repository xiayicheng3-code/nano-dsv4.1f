"""Generate CPU safetensors export and chat notebooks for completed SFT runs."""
from pathlib import Path
from textwrap import dedent
import nbformat as nbf

from build_cpu_chat_notebook import build_notebook as chat_notebook

ROOT = Path(__file__).resolve().parents[1]
REF = 'codex/midtrain8k-sft16k'


def md(text):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + '\n')


def code(text):
    return nbf.v4.new_code_cell(dedent(text).strip() + '\n')


def bootstrap(*, temporary_export=False):
    cell = code('''
        import importlib.util
        import os
        from pathlib import Path
        import subprocess
        import sys

        if any(name in sys.modules for name in ('jax', 'nano_dsv41f')):
            raise RuntimeError('Restart the session before updating the source and dependencies')
        os.environ['JAX_PLATFORMS'] = 'cpu'
        REPO = Path('/kaggle/working/nano-dsv4.1f')
        if not REPO.exists():
            subprocess.run(['git', 'clone', '--depth', '1',
                            'https://github.com/xiayicheng3-code/nano-dsv4.1f.git', str(REPO)], check=True)
        if subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain',
                                    '--untracked-files=no'], text=True).strip():
            raise RuntimeError('Preserve tracked edits before updating this checkout')
        subprocess.run(['git', '-C', str(REPO), 'fetch', '--depth', '1', 'origin', SOURCE_REF], check=True)
        subprocess.run(['git', '-C', str(REPO), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
        if importlib.util.find_spec('torch') is None:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'torch',
                            '--index-url', 'https://download.pytorch.org/whl/cpu'], check=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                        '-r', str(REPO / 'requirements-posttrain-inference.txt')], check=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', str(REPO), '--no-deps'], check=True)
        sys.path.insert(0, str(REPO / 'src'))
        sys.path.insert(0, str(REPO / 'scripts'))
        print('Source commit:', subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip())
    ''')
    if temporary_export:
        cell.source = cell.source.replace("REPO = Path('/kaggle/working/nano-dsv4.1f')", '''EXPORT_WORK = Path('/kaggle/temp/nano-dsv41f-export')
if EXPORT_WORK.is_symlink():
    raise RuntimeError('The export temporary directory must not be a symlink')
marker = EXPORT_WORK / '.export-workspace'
if EXPORT_WORK.exists() and (not marker.is_file() or marker.read_text().strip() != 'nano-dsv41f-export-workspace-v1'):
    raise RuntimeError('The export temporary directory belongs to another task')
EXPORT_WORK.mkdir(parents=True, exist_ok=True)
marker.write_text('nano-dsv41f-export-workspace-v1')
os.environ['PIP_CACHE_DIR'] = str(EXPORT_WORK / 'pip-cache')
os.environ['HF_HOME'] = str(EXPORT_WORK / 'hf-cache')
os.environ['XDG_CACHE_HOME'] = str(EXPORT_WORK / 'cache')
REPO = EXPORT_WORK / 'source'
'''.rstrip())
    return cell


def save(name, cells, output_dir):
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.kernelspec = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    for i, cell in enumerate(nb.cells):
        cell.id = f'{name}-{i:02d}'
        if cell.cell_type == 'code':
            compile(cell.source, name, 'exec')
    nbf.validate(nb)
    path = Path(output_dir) / f'nano_dsv41f_{name}.ipynb'
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)
    return path


def build(output_dir=ROOT / 'notebooks'):
    export = save('export_sft_safetensors_cpu', [
        md('''
        # Export the completed SFT model to safetensors (CPU)

        Choose **CPU**, enable **Internet**, and attach the saved TPU posttraining
        output dataset. Keep its `training/` folder intact, including `tokenizer.json`,
        `corpus_manifest.json`, and `sft/checkpoints/`. No corpus shards or pretrain
        dataset are required. The finished run's final step is
        `step-00085663-af2e759ac4dc` under `posttrain-20261001-183833/training/sft/checkpoints/`.

        This exports only the completed SFT model parameters, including DSpark weights,
        with their original BF16/FP32 dtypes. It reads the effective SFT recipe so the
        32K context and compressed-layer YaRN factor 4 survive export. Optimizer files
        are never loaded. Every model tensor is checksum-checked and compared bit for
        bit after safetensors export. The bundled tokenizer must match the training hash.

        The output uses the project's portable layout for its custom CPU runtime.
        Save the original training checkpoint too if you want to resume training.
        '''),
        code(f'''SOURCE_REF = {REF!r}
CHECKPOINT = ''  # blank: locate exactly one SFT checkpoint set under /kaggle/input
OUTPUT = '/kaggle/working/nano-dsv41f-sft32k-safetensors'  # use a fresh directory
HF_REPO_ID = ''  # set account/repository to upload; blank skips Hugging Face upload
HF_PRIVATE = True  # visibility when creating a new dataset repository
HF_SECRET_NAME = 'HF_TOKEN'  # Kaggle Secrets: enable your Hugging Face write token
'''),
        bootstrap(temporary_export=True),
        code('''
        import json
        from export_posttrain_checkpoint import resolve_checkpoint
        from pretrain_checkpoint import read_metadata

        selected = resolve_checkpoint(CHECKPOINT or '/kaggle/input')
        _, checkpoint_manifest = read_metadata(selected)
        state = checkpoint_manifest['metadata']
        print('Selected checkpoint:', selected)
        print('Stage:', state.get('stage'), 'complete:', state.get('stage_complete'))
        print('Global step:', state['completed_steps'])
        print('SFT context:', state['recipe']['train']['seq_len'])
        print('RoPE:', json.dumps(state['recipe']['model']['attention']['rope'], indent=2))
        command = [sys.executable, '-u', str(REPO / 'scripts/export_posttrain_checkpoint.py'),
                   '--checkpoint', str(selected), '--output', OUTPUT]
        subprocess.run(command, check=True)
        '''),
        code('''
        from nano_dsv41f.portable_bundle import verify_portable_bundle
        from IPython.display import FileLink, display
        report = verify_portable_bundle(OUTPUT)
        print('Verified tensors:', report['tensor_count'], 'parameters:', report['parameter_count'])
        print('Context length:', report['max_position_embeddings'])
        for path in sorted(Path(OUTPUT).iterdir()):
            display(FileLink(str(path)))
        '''),
        md('''
        ## Upload to Hugging Face and clean temporary files

        Set `HF_REPO_ID` above to `your-account/your-dataset`, add your Hugging Face
        write token as the Kaggle secret `HF_TOKEN`, and enable it for this notebook.
        This cell creates a **dataset** repo (private by default) and uploads the
        verified export files at its root. It runs automatically when the repo ID
        is set; leaving it blank saves only the Kaggle output. The key is passed
        directly to the API and is not printed or saved as a login file.

        The source checkout and caches live outside `/kaggle/working`. After this
        cell, the marked temporary workspace is removed, even if an upload fails.
        The exported bundle remains available for Kaggle saving and retrying upload.
        '''),
        code('''
        from export_bundle_upload import cleanup_export_workspace, upload_export_bundle

        def upload_with_kaggle_secret():
            from kaggle_secrets import UserSecretsClient
            try:
                token = UserSecretsClient().get_secret(HF_SECRET_NAME)
            except Exception:
                raise RuntimeError(f'Enable the Kaggle secret {HF_SECRET_NAME} for Hugging Face upload') from None
            try:
                return upload_export_bundle(OUTPUT, HF_REPO_ID, token, private=HF_PRIVATE)
            finally:
                del token

        try:
            if HF_REPO_ID:
                uploaded = upload_with_kaggle_secret()
                print('Hugging Face dataset:', f'https://huggingface.co/datasets/{HF_REPO_ID}')
                print('Upload commit:', uploaded.commit_url)
            else:
                print('Hugging Face upload skipped: HF_REPO_ID is blank')
        finally:
            cleanup_export_workspace(OUTPUT, EXPORT_WORK)
            print('Temporary export checkout and caches removed')
            print('Saved bundle:', OUTPUT)
        '''),
        md('''
        ## Save and use

        Save this notebook version with its outputs, then save the entire
        `nano-dsv41f-sft32k-safetensors` folder as a Kaggle dataset. Keep all files
        together, especially `model.safetensors`, `config.json`, `tokenizer.json`,
        `training_recipe.json`, and `export_manifest.json`. Attach this exported
        dataset to `nano_dsv41f_sft_inference_cpu.ipynb`. GitHub supplies the runtime
        code; the weights remain in your chosen dataset. Only the bundle is written
        under `/kaggle/working`; Kaggle may also generate its normal notebook/log files.
        The optional upload cell publishes the same bundle to your Hugging Face dataset.
        ''')], output_dir)

    demo = chat_notebook()
    helpers = demo.cells[4].source
    helpers = helpers.replace('_, raw = backend.complete_protocol("chat_completions", payload)',
        '''from nano_dsv41f.vllm_v41_cpu.api import prepare_protocol_request
    prepared = prepare_protocol_request("chat_completions", payload)
    prompt_ids = backend.tokenizer.encode_request(prepared)
    check_context(prompt_ids, max_tokens)
    _, raw = backend.complete_protocol("chat_completions", payload)''')
    inference = save('sft_inference_cpu', [
        md('''
        # Run the trained 32K SFT model on CPU

        Choose **CPU**, enable **Internet**, and attach the safetensors dataset
        produced by `nano_dsv41f_export_sft_safetensors_cpu.ipynb`. No optimizer,
        training corpus, or separate tokenizer input is needed.

        The runtime comes from GitHub and uses the project's Torch CPU implementation
        with bounded caches, prefix reuse between turns, and DeepSeek V4.1 chat encoding. CPU execution uses
        FP32; the saved weights keep their original precision. Start with short prompts
        and outputs: the stored 32K context limit does not promise fast CPU inference.
        This notebook loads your trained weights; it never initializes a replacement model.
        '''),
        code(f'''SOURCE_REF = {REF!r}
MODEL_DIR = ''  # blank: locate exactly one exported SFT bundle under /kaggle/input
DEVICE = 'cpu'  # 'cuda' on a GPU session; benchmark throughput on your hardware
PREFILL_CHUNK_SIZE = 32
USE_MTP = False  # requires a bundle from the DSpark distillation notebook
'''),
        bootstrap(),
        code('''
        import json
        import torch
        from nano_dsv41f.portable_bundle import verify_portable_bundle
        from nano_dsv41f.vllm_v41_cpu.api import NanoDeepSeekProtocolBackend

        if MODEL_DIR:
            model_root = Path(MODEL_DIR)
        else:
            bundles = list(Path('/kaggle/input').rglob('export_manifest.json'))
            if len(bundles) != 1:
                raise ValueError('Attach one exported bundle or set MODEL_DIR to its directory')
            model_root = bundles[0].parent
        report = verify_portable_bundle(model_root)
        config = json.loads((model_root / 'config.json').read_text())
        MAX_CONTEXT = int(config['max_position_embeddings'])
        torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
        backend = NanoDeepSeekProtocolBackend.from_pretrained(model_root, dtype=torch.float32, device=DEVICE)
        from nano_dsv41f.vllm_v41_cpu.session import InferenceSession
        draft_trained = report.get("dspark_steps", 0) > 0
        backend.session = InferenceSession(backend.model, mtp=USE_MTP,
            draft_trained=draft_trained, prefill_chunk_size=PREFILL_CHUNK_SIZE)
        print('Loaded SFT checkpoint:', report['checkpoint'])
        print('Context:', MAX_CONTEXT, 'RoPE:', config['nano_config']['attention']['rope'])

        def check_context(prompt_ids, max_tokens):
            if max_tokens <= 0 or len(prompt_ids) + max_tokens > MAX_CONTEXT:
                raise ValueError(f'Prompt + output must fit {MAX_CONTEXT} tokens; shorten the request or reset_chat()')

        def complete(prompt, *, max_tokens=64, temperature=0.0):
            check_context(backend.tokenizer.encode(prompt), max_tokens)
            return backend.complete_raw(prompt, max_tokens=max_tokens, temperature=temperature)
        '''),
        code(helpers),
        code('''
        # A short first response checks model loading, chat encoding, and cached generation.
        print_reply(chat('Hello! Introduce yourself in one sentence.', max_tokens=32))
        reset_chat()
        '''),
        demo.cells[6], demo.cells[7], demo.cells[8],
        md('''
        ## Raw completion and tool output

        `complete("The capital of France is", max_tokens=32)` uses the prompt directly.
        `chat(...)` applies the V4.1 conversation protocol and maintains history.
        Both functions check prompt plus requested output against the saved context limit.
        For tool-call inspection, examine the returned reply dictionary; this notebook
        does not execute generated tool calls. Use `reset_chat()` for a new conversation.
        Every reply reports prefill/decode token counts, elapsed seconds, tokens/second,
        reused prompt tokens, first-token latency, and reserved cache arena size.
        Prefill throughput counts only newly processed tokens. Decode counts include
        reasoning, protocol tokens, and EOS when generated; its time includes sampling
        and any drafting. The cache arena figure excludes weights, temporary tensors,
        and the small prompt checkpoint. Detailed step times are in `backend.last_stats`.

        Prefix caches persist in session RAM until reset/restart. When the protocol
        rewrites a previous reply, the runtime reuses the last matching prompt checkpoint
        or rebuilds safely. Prompt prefill batches up to `PREFILL_CHUNK_SIZE` tokens
        per target pass; set it to 1 for the scalar comparison. No fused GPU kernels
        or independent-request batching are used.

        To accelerate greedy decoding, attach the bundle from the DSpark distillation
        notebook and set `USE_MTP=True`. The verified export must record completed
        draft updates. Chat automatically uses temperature zero in this mode.
        One causal target pass verifies a proposal block. The first mismatch is
        replaced with the target token, and rejected cache entries are rolled back
        without recomputing the accepted prefix. Output matches greedy target decoding
        within normal floating-point limits. Speed depends on acceptance and hardware;
        an inaccurate drafter can still make decoding slower.

        `backend.last_stats` includes target calls/input tokens, drafting time,
        verification time, rollback time, and accepted/verified/proposed counts.
        Batch times and emitted counts are recorded separately; per-token step times
        are amortized within each batch. Set `USE_MTP=False` to compare normal decoding.
        For a real-bundle benchmark, run `scripts/benchmark_mtp_inference.py` with
        `--model-dir`, `--device cpu` or `cuda`, and `--output /path/to/results.json`.
        The optional local API cell below serves the same loaded model.
        '''),
        demo.cells[9]], output_dir)
    return export, inference


if __name__ == '__main__':
    for path in build():
        print(path)
