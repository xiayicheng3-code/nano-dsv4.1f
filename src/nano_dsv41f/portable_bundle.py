"""Integrity checks for a portable model exported from a completed training run."""
import hashlib
import json
from pathlib import Path

FORMAT = 'nano-dsv41f-inference-bundle-v1'
REQUIRED = {'model.safetensors', 'config.json', 'tokenizer.json', 'training_recipe.json',
            'nano_parameter_manifest.json', 'nano_tokenizer_contract.json'}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_portable_bundle(root):
    root = Path(root)
    manifest = json.loads((root / 'export_manifest.json').read_text())
    if manifest.get('format') != FORMAT or not manifest.get('complete'):
        raise ValueError('Incomplete or unsupported inference bundle')
    files = manifest['files']
    if not REQUIRED.issubset(files):
        raise ValueError('Inference bundle is missing required files')
    for name, info in files.items():
        if Path(name).name != name:
            raise ValueError('Invalid bundle filename')
        path = root / name
        if path.stat().st_size != info['bytes'] or sha256_file(path) != info['sha256']:
            raise ValueError(f'Inference bundle checksum mismatch: {name}')
    return manifest
