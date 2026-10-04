"""Draft-only resumable checkpoints and full bundles preserving backbone bytes."""
import json
from pathlib import Path
import shutil
import tempfile

import torch
from safetensors.torch import load_file, save_file
from nano_dsv41f.portable_bundle import sha256_file, verify_portable_bundle
from pretrain_checkpoint import atomic_json


def save_draft_checkpoint(root, trainer, identity):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    name = f'step-{trainer.steps:08d}.pt'
    target = root / name
    pending = root / (name + '.tmp')
    try:
        torch.save(dict(format='nano-dspark-training-v1', step=trainer.steps,
            draft=trainer.draft_state(), optimizer=trainer.optimizer.state_dict(), identity=identity), pending)
        pending.replace(target)
        atomic_json(root / 'latest.json', dict(file=name, sha256=sha256_file(target)))
    finally:
        pending.unlink(missing_ok=True)
    # Only this fresh run's managed checkpoint directory is pruned.
    for old in sorted(root.glob('step-????????.pt'))[:-2]:
        old.unlink()
    return target


def load_draft_checkpoint(path, trainer, identity):
    path = Path(path)
    if path.is_dir() and (path/'checkpoints/latest.json').is_file():
        path = path/'checkpoints'
    if not path.is_dir():
        raise ValueError('RESUME must point to a run or its checkpoints directory')
    pointer = json.loads((path/'latest.json').read_text())
    if Path(pointer['file']).name != pointer['file']:
        raise ValueError('Invalid checkpoint filename')
    checkpoint = path/pointer['file']
    if sha256_file(checkpoint) != pointer['sha256']:
        raise ValueError('Draft checkpoint checksum mismatch')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved.get('format') != 'nano-dspark-training-v1' or saved['identity'] != identity:
        raise ValueError('Resume model, data, code or training settings differ')
    trainer.load_draft(saved['draft'])
    trainer.optimizer.load_state_dict(saved['optimizer'])
    trainer.steps = int(saved['step'])
    return trainer.steps


def export_trained_draft(base, output, trainer, training_report):
    base, output = Path(base), Path(output)
    original = verify_portable_bundle(base)
    if output.exists():
        raise FileExistsError('Use a new bundle output directory')
    if trainer.steps <= 0:
        raise ValueError('No DSpark updates have completed')
    tensors = load_file(str(base/'model.safetensors'), device='cpu')
    draft = trainer.draft_state()
    expected = {name for name in tensors if name.startswith('nano.dspark.')}
    if set(draft) != expected:
        raise ValueError('Draft names do not match the source bundle')
    for name, value in draft.items():
        if tensors[name].shape != value.shape or not torch.isfinite(value).all():
            raise ValueError(f'Invalid trained draft tensor: {name}')
        tensors[name] = value.contiguous()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name+'.pending-', dir=output.parent))
    try:
        for name in original['files']:
            if name != 'model.safetensors':
                shutil.copy2(base/name, staging/name)
        save_file(tensors, str(staging/'model.safetensors'), metadata={'format':'nano-dsv41f-portable-v1', 'stage':'sft_dspark'})
        # Bitwise roundtrip all output tensors. Non-draft leaves came directly from
        # the original file, preserving BF16/FP32 rather than the runtime FP32 casts.
        reloaded = load_file(str(staging/'model.safetensors'))
        if set(reloaded) != set(tensors):
            raise ValueError('Export tensor names changed')
        for name, value in tensors.items():
            actual = reloaded[name]
            if actual.dtype != value.dtype or not torch.equal(actual.reshape(-1).view(torch.uint8), value.reshape(-1).view(torch.uint8)):
                raise ValueError(f'Bitwise roundtrip failed: {name}')
        atomic_json(staging/'nano_parameter_manifest.json', {
            name: dict(shape=list(t.shape), dtype=str(t.dtype).removeprefix('torch.'))
            for name, t in tensors.items()})
        atomic_json(staging/'dspark_training.json', training_report)
        with (staging/'README.md').open('a') as handle:
            handle.write('\nDSpark was subsequently distilled with a frozen backbone. '
                'See dspark_training.json for steps, data identity and held-out metrics. '
                'All non-DSpark tensors preserve source dtype and bytes. '
                'Training does not imply an acceptance or speed threshold was met.\n')
        report = {**original, 'stage':'sft_dspark', 'dspark_steps':trainer.steps,
            'source_bundle_manifest_sha256':sha256_file(base/'export_manifest.json'),
            'backbone_bitwise_preserved':True, 'bitwise_roundtrip_verified':True,
            'files': {p.name:dict(bytes=p.stat().st_size, sha256=sha256_file(p))
                      for p in sorted(staging.iterdir()) if p.is_file()}}
        atomic_json(staging/'export_manifest.json', report)
        verify_portable_bundle(staging)
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report
