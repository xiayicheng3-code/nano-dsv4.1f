#!/usr/bin/env python3
"""Export completed SFT parameters to portable safetensors on CPU, without optimizer state."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

from pretrain_checkpoint import atomic_json, checkpoint_path, digest_file, read_metadata


def resolve_checkpoint(path):
    """Accept a step, checkpoint parent, training output, or attached dataset root."""
    path = Path(path)
    if (path / 'manifest.json').is_file() or (path / 'latest.json').is_file():
        return checkpoint_path(path)
    matches = list(path.rglob('sft/checkpoints/latest.json'))
    if len(matches) != 1:
        raise ValueError('Set CHECKPOINT to one SFT step or its checkpoints directory; '
                         f'found {len(matches)} SFT checkpoint pointers under {path}')
    return checkpoint_path(matches[0].parent)


def find_asset(checkpoint, name, explicit=None):
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    for parent in checkpoint.parents:
        candidate = parent / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f'Keep training/{name} with the checkpoint or supply its path explicitly')


def load_parameters(checkpoint, manifest, config):
    """Use an abstract shape tree; read only parameter leaves from (params, optimizer)."""
    import jax
    import jax.numpy as jnp
    import ml_dtypes
    import numpy as np
    from nano_dsv41f.model import init_model

    shapes = jax.eval_shape(lambda: init_model(jax.random.PRNGKey(0), config))
    expected = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(shapes)[0]:
        parts = [str(key.key) if isinstance(key, jax.tree_util.DictKey) else str(key.idx)
                 for key in path]
        expected['[0]' + jax.tree_util.keystr(path)] = ('.'.join(parts), list(leaf.shape))
    params, seen = {}, set()
    for info in manifest['leaves']:
        key = info['path']
        if key in seen:
            raise ValueError(f'Duplicate checkpoint leaf: {key}')
        seen.add(key)
        if key not in expected:
            if key.startswith('[1]'):
                continue  # Optimizer arrays are never opened or materialized.
            raise ValueError(f'Unexpected checkpoint parameter: {key}')
        name, shape = expected[key]
        if info['shape'] != shape or Path(info['file']).name != info['file']:
            raise ValueError(f'Invalid parameter shape or filename: {key}')
        file = checkpoint / info['file']
        if digest_file(file) != info['sha256']:
            raise ValueError(f'Checkpoint checksum failed: {file.name}')
        host = np.load(file, allow_pickle=False)
        if info['dtype'] == 'bfloat16':
            if host.dtype != np.uint16:
                raise ValueError('Invalid BF16 checkpoint storage')
            host = host.view(ml_dtypes.bfloat16)
        if list(host.shape) != shape or str(host.dtype) != info['dtype']:
            raise ValueError(f'Stored leaf shape/dtype differs: {key}')
        if info['dtype'] not in ('bfloat16', 'float32') or not np.isfinite(host).all():
            raise ValueError(f'Unsupported or nonfinite parameter: {key}')
        params[name] = jnp.asarray(host)
    if set(expected) - seen:
        raise ValueError('Checkpoint is missing model parameters')
    return params


def export_checkpoint(checkpoint, output, *, tokenizer=None, corpus_manifest=None):
    import jax
    import numpy as np
    from safetensors import safe_open
    from tokenizers import Tokenizer
    from nano_dsv41f.chat_protocol import nano_v41_tokenizer_contract
    from nano_dsv41f.hf_export import export_portable_checkpoint, flatten_parameter_tree
    from nano_dsv41f.portable_bundle import FORMAT, verify_portable_bundle
    from nano_dsv41f.vllm_v41_cpu.config_io import model_config_from_export

    if jax.default_backend() != 'cpu':
        raise ValueError('Run export on CPU with JAX_PLATFORMS=cpu')
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Use a fresh output directory: {output}')
    checkpoint, manifest = read_metadata(resolve_checkpoint(checkpoint))
    metadata = manifest['metadata']
    if metadata.get('stage') != 'sft' or not metadata.get('stage_complete'):
        raise ValueError('Export requires a completed SFT checkpoint')
    # The effective SFT recipe carries YaRN and candidate-mask settings. base_recipe does not.
    recipe = metadata['recipe']
    config = model_config_from_export(recipe['model'])
    length = int(recipe['train']['seq_len'])
    tokenizer = find_asset(checkpoint, 'tokenizer.json', tokenizer)
    corpus_manifest = find_asset(checkpoint, 'corpus_manifest.json', corpus_manifest)
    if digest_file(corpus_manifest) != metadata['identity']['corpus_sha256']:
        raise ValueError('Corpus manifest does not match the SFT checkpoint')
    corpus = json.loads(corpus_manifest.read_text())
    if digest_file(tokenizer) != corpus['identity']['tokenizer_sha256']:
        raise ValueError('Tokenizer checksum does not match training')
    if length != corpus['lengths']['sft']:
        raise ValueError('SFT recipe and corpus context lengths differ')
    tok = Tokenizer.from_file(str(tokenizer))
    contract = nano_v41_tokenizer_contract(config.vocab_size)
    if tok.get_vocab_size() != config.vocab_size or any(
        tok.token_to_id(token) != index for token, index in contract.token_to_id.items()
    ):
        raise ValueError('Tokenizer vocabulary or special-token IDs differ from the model')

    params = load_parameters(checkpoint, manifest, config)
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = Path(tempfile.mkdtemp(prefix=output.name + '.pending-', dir=output.parent))
    try:
        weights = export_portable_checkpoint(params, pending, config,
            max_position_embeddings=length,
            metadata={'format': 'nano-dsv41f-portable-v1', 'stage': 'sft',
                      'checkpoint_manifest_sha256': digest_file(checkpoint / 'manifest.json')})
        # Round-trip every tensor, including BF16 bit patterns and FP32 master parameters.
        flat = flatten_parameter_tree(params)
        with safe_open(weights, framework='flax') as handle:
            if set(handle.keys()) != set(flat):
                raise ValueError('Exported parameter names differ')
            for name, expected in flat.items():
                actual, expected = np.asarray(handle.get_tensor(name)), np.asarray(expected)
                if (actual.dtype != expected.dtype or actual.shape != expected.shape
                        or actual.tobytes() != expected.tobytes()):
                    raise ValueError(f'Safetensors round-trip differs: {name}')
        shutil.copy2(tokenizer, pending / 'tokenizer.json')
        atomic_json(pending / 'training_recipe.json', recipe)
        atomic_json(pending / 'training_metadata.json', metadata)
        (pending / 'README.md').write_text(
            '# nano-dsv4.1f SFT inference bundle\n\n'
            f'Completed SFT step: {metadata["completed_steps"]}. Context: {length} tokens.\n'
            'Load with nano_dsv41f.vllm_v41_cpu.NanoDeepseekV41CPU or the SFT inference notebook.\n'
            'Weights preserve training BF16/FP32 dtypes. No optimizer state is included.\n'
            'This custom architecture requires the nano runtime; stock Transformers/vLLM '
            'AutoModel loading is not implemented. Keep the original training checkpoint for resume.\n')
        report = {'format': FORMAT, 'complete': True, 'stage': 'sft',
                  'checkpoint': checkpoint.name, 'completed_steps': metadata['completed_steps'],
                  'checkpoint_manifest_sha256': digest_file(checkpoint / 'manifest.json'),
                  'training_code_sha256': metadata['identity'].get('code_sha256'),
                  'max_position_embeddings': length,
                  'parameter_count': sum(int(value.size) for value in params.values()),
                  'tensor_count': len(params), 'bitwise_roundtrip_verified': True,
                  'files': {p.name: {'bytes': p.stat().st_size, 'sha256': digest_file(p)}
                            for p in sorted(pending.iterdir()) if p.is_file()}}
        atomic_json(pending / 'export_manifest.json', report)
        verify_portable_bundle(pending)
        pending.rename(output)
    except BaseException:
        shutil.rmtree(pending, ignore_errors=True)
        raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path)
    parser.add_argument('--corpus-manifest', type=Path)
    args = parser.parse_args()
    os.environ['JAX_PLATFORMS'] = 'cpu'
    report = export_checkpoint(args.checkpoint, args.output,
                               tokenizer=args.tokenizer, corpus_manifest=args.corpus_manifest)
    print(json.dumps(report, indent=2))
    print('Export complete:', args.output)


if __name__ == '__main__':
    main()
