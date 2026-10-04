"""Real training-checkpoint -> safetensors -> CPU inference regression coverage."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('safetensors')
import jax
import jax.numpy as jnp
import numpy as np
from safetensors.torch import load_file
from tokenizers import Tokenizer, models

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from export_posttrain_checkpoint import export_checkpoint, resolve_checkpoint
from pretrain_checkpoint import atomic_json, digest_file, save_checkpoint
from nano_dsv41f.chat_protocol import SPECIAL_TOKENS
from nano_dsv41f.hf_export import flatten_parameter_tree
from nano_dsv41f.model import init_model
from nano_dsv41f.portable_bundle import verify_portable_bundle
from nano_dsv41f.vllm_v41_cpu import NanoDeepseekV41CPU
from test_vllm_v41_cpu import tiny_config


@pytest.fixture
def completed(tmp_path):
    config = tiny_config()
    config = replace(config, attention=replace(config.attention,
        rope=replace(config.attention.rope, original_seq_len=8192, rope_factor=4)),
        dspark=replace(config.dspark, enabled=True, markov_rank=8, target_layer_ids=(1, 3, 6)),
        indexer_training=replace(config.indexer_training, apply_candidate_mask=True))
    params = init_model(jax.random.PRNGKey(19), config)
    params = jax.tree.map(lambda x: x.astype(jnp.bfloat16) if x.ndim >= 2 else x, params)
    training = tmp_path / 'run' / 'training'
    training.mkdir(parents=True)
    vocab = {token: i for i, token in enumerate(SPECIAL_TOKENS)}
    vocab.update({f'word{i}': i for i in range(len(vocab), config.vocab_size)})
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token='word23'))
    tokenizer.add_special_tokens(list(SPECIAL_TOKENS))
    tokenizer.save(str(training / 'tokenizer.json'))
    atomic_json(training / 'corpus_manifest.json', {
        'identity': {'tokenizer_sha256': digest_file(training / 'tokenizer.json')},
        'lengths': {'midtrain': 8192, 'sft': 32768}})
    metadata = {'stage': 'sft', 'stage_complete': True, 'completed_steps': 85663,
        'identity': {'corpus_sha256': digest_file(training / 'corpus_manifest.json'),
                     'code_sha256': 'training-code'},
        'recipe': {'model': asdict(config), 'train': {'seq_len': 32768}},
        'base_recipe': {'model': {'attention': {'rope': {'rope_factor': 1}}}}}
    checkpoint = save_checkpoint(training / 'sft/checkpoints', (params, {'moment': jnp.ones(4)}), metadata)
    # Inference export must not need any optimizer files.
    manifest = json.loads((checkpoint / 'manifest.json').read_text())
    for entry in manifest['leaves']:
        if entry['path'].startswith('[1]'):
            (checkpoint / entry['file']).unlink()
    return training, checkpoint, config, params


def test_export_preserves_bf16_and_effective_sft_recipe_and_cpu_logits(completed, tmp_path):
    training, checkpoint, config, params = completed
    output = tmp_path / 'portable'
    report = export_checkpoint(training.parent, output)
    assert verify_portable_bundle(output) == report
    assert report['bitwise_roundtrip_verified'] and report['max_position_embeddings'] == 32768
    payload = json.loads((output / 'config.json').read_text())
    assert payload['nano_config'] == json.loads(json.dumps(asdict(config)))
    assert payload['nano_config']['attention']['rope']['rope_factor'] == 4
    assert payload['nano_config']['indexer_training']['apply_candidate_mask']
    tensors = load_file(str(output / 'model.safetensors'))
    flat = flatten_parameter_tree(params)
    assert set(tensors) == set(flat)
    assert 'nano.dspark.markov_head.weight' in tensors
    assert {x.dtype for x in tensors.values()} == {torch.float32, torch.bfloat16}
    for name, value in flat.items():
        actual = tensors[name]
        actual_bytes = (actual.view(torch.uint16).numpy().tobytes()
                        if actual.dtype == torch.bfloat16 else actual.numpy().tobytes())
        assert actual_bytes == np.asarray(value).tobytes()
    direct = NanoDeepseekV41CPU(config, {k: np.asarray(v).astype(np.float32) for k,v in flat.items()})
    loaded = NanoDeepseekV41CPU.from_pretrained(output)
    ids = torch.tensor([[0, 4, 25, 5, 27, 1]])
    expected, _ = direct.forward(ids, sparse_retrieval=True)
    actual, _ = loaded.forward(ids, sparse_retrieval=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(loaded.generate(ids, max_new_tokens=2, temperature=0),
                               direct.generate(ids, max_new_tokens=2, temperature=0))
    assert not list(output.glob('leaf-*.npy'))
    assert report['checkpoint'] == checkpoint.name
    with pytest.raises(FileExistsError):
        export_checkpoint(checkpoint, output)


@pytest.mark.parametrize('mutation,match',[
    ('incomplete', 'completed SFT'), ('tokenizer', 'Tokenizer checksum'),
    ('corpus', 'Corpus manifest'), ('weight', 'Checkpoint checksum'),
    ('missing', 'missing model'), ('duplicate', 'Duplicate checkpoint'),
])
def test_export_rejects_incomplete_or_mismatched_inputs(completed, tmp_path, mutation, match):
    training, checkpoint, _, _ = completed
    file = checkpoint / 'manifest.json'
    manifest = json.loads(file.read_text())
    if mutation == 'incomplete':
        manifest['metadata']['stage_complete'] = False
    elif mutation in ('tokenizer', 'corpus'):
        target = training / ('tokenizer.json' if mutation == 'tokenizer' else 'corpus_manifest.json')
        target.write_text(target.read_text() + '\n')
    elif mutation == 'weight':
        target = checkpoint / manifest['leaves'][0]['file']
        target.write_bytes(target.read_bytes() + b'corrupt')
    elif mutation == 'missing':
        manifest['leaves'].pop(0)
    else:
        manifest['leaves'].append(manifest['leaves'][0])
    atomic_json(file, manifest)
    output = tmp_path / 'portable'
    with pytest.raises(ValueError, match=match):
        export_checkpoint(training, output)
    assert not output.exists()


def test_inference_verification_detects_changed_weights(completed, tmp_path):
    output = tmp_path / 'portable'
    export_checkpoint(completed[0], output)
    weights = output / 'model.safetensors'
    weights.write_bytes(weights.read_bytes() + b'corrupt')
    with pytest.raises(ValueError, match='checksum mismatch: model.safetensors'):
        verify_portable_bundle(output)


def test_checkpoint_discovery_refuses_multiple_runs(completed, tmp_path):
    import shutil
    training, checkpoint, _, _ = completed
    assert resolve_checkpoint(training) == checkpoint
    assert resolve_checkpoint(checkpoint.parent) == checkpoint
    shutil.copytree(training, tmp_path / 'second' / 'training')
    with pytest.raises(ValueError, match='found 2'):
        resolve_checkpoint(tmp_path)
