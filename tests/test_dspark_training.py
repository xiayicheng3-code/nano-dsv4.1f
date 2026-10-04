from dataclasses import replace
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('safetensors')
from safetensors.torch import load_file
from nano_dsv41f.config import DSparkConfig
from nano_dsv41f.hf_export import flatten_parameter_tree
from nano_dsv41f.model import init_model, apply_model
from nano_dsv41f.dspark import apply_dspark
from nano_dsv41f.portable_bundle import sha256_file, verify_portable_bundle
from nano_dsv41f.vllm_v41_cpu import NanoDeepseekV41CPU
from nano_dsv41f.vllm_v41_cpu.dspark import draft_from_features
from nano_dsv41f.vllm_v41_cpu.dspark_training import DraftLossConfig, DraftTrainer, backbone_digest, draft_loss
from test_vllm_v41_cpu import tiny_config
from test_posttrain_export import completed  # real BF16/FP32 checkpoint fixture

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from export_posttrain_checkpoint import export_checkpoint
from dspark_artifacts import export_trained_draft, load_draft_checkpoint, save_draft_checkpoint
from dspark_data import SFTDraftData
from train_dspark import parse_args, run
from build_dspark_notebook import build


@pytest.fixture(scope='module')
def weights():
    torch.set_num_threads(1)
    config = replace(tiny_config(), dspark=DSparkConfig(enabled=True, block_size=3,
        target_layer_ids=(4,5,6), markov_rank=4, n_routed_experts=4, experts_per_token=1))
    params = init_model(jax.random.PRNGKey(37), config)
    return config, params


def make_model(weights):
    config, params = weights
    return NanoDeepseekV41CPU(config, flatten_parameter_tree(params))


def test_objective_coefficients_position_weights_and_gradient_boundaries():
    q_logits = torch.tensor([[[1., 2.], [3., 1.]]], requires_grad=True)
    p_logits = torch.tensor([[[2., 1.], [1., 3.]]], requires_grad=True)
    conf = torch.tensor([[0.1, -0.2]], requires_grad=True)
    labels = torch.tensor([[0, 1]])
    loss, metrics = draft_loss(q_logits, p_logits, labels, conf)
    q, p = q_logits.softmax(-1), p_logits.softmax(-1)
    l1 = (q-p).abs().sum(-1)
    accept = 1-l1/2
    weights = torch.exp(-torch.arange(2)/2)
    ce = -q.log().gather(-1, labels[..., None]).squeeze(-1)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(conf, accept, reduction='none')
    expected = ((0.1*ce + 0.9*l1 + bce)*weights).sum()/weights.sum()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert p_logits.grad is None and q_logits.grad.abs().sum() > 0
    assert conf.grad.abs().sum() > 0
    q2 = q_logits.detach().clone().requires_grad_()
    only_conf, _ = draft_loss(q2, p_logits, labels, conf, config=DraftLossConfig(0, 0, 1))
    only_conf.backward()
    assert torch.equal(q2.grad, torch.zeros_like(q2))
    assert 0 <= metrics['teacher_forced_overlap'] <= 1


def test_training_forward_parity_alignment_and_no_future_feature_leak(weights):
    config, params = weights
    model = make_model(weights)
    trainer = DraftTrainer(model)
    ids = np.array([1,3,5,7,9,11,13,15,17,19,21,23])
    anchors = np.array([2, 5, 8])
    batch = trainer.prepare(ids, anchors)
    all_logits, _ = model.forward(torch.tensor([ids.tolist()]))
    positions = torch.tensor(anchors)[:, None] + torch.arange(config.dspark.block_size)[None]
    torch.testing.assert_close(batch['target_logits'], all_logits[0, positions])
    torch.testing.assert_close(batch['labels'], torch.tensor(ids)[positions+1])
    logits, confidence = draft_from_features(model, batch['features'], batch['context_positions'],
        batch['context_mask'], batch['anchor_ids'], batch['anchor_positions'], batch['previous_ids'])
    # Compare to the independent JAX drafter under dense target features to avoid
    # confusing sparse-vs-dense target differences with draft implementation errors.
    jids = jnp.asarray([ids.tolist()], dtype=jnp.int32)
    _, aux = apply_model(params, config, jids)
    pos = batch['context_positions'].cpu().numpy()
    features = torch.tensor(np.asarray(aux['dspark_context_features'])[0, pos])
    actual, conf = draft_from_features(model, features, batch['context_positions'], batch['context_mask'],
        batch['anchor_ids'], batch['anchor_positions'], batch['previous_ids'])
    expected = apply_dspark(params['dspark'], config, embed=params['embed'], lm_head=params['lm_head'],
        input_ids=jids, context_features=aux['dspark_context_features'],
        anchor_positions=jnp.asarray([anchors.tolist()], dtype=jnp.int32))
    np.testing.assert_allclose(actual.detach().numpy(), np.asarray(expected['draft_logits'])[0], rtol=4e-4, atol=4e-4)
    np.testing.assert_allclose(conf.detach().numpy(), np.asarray(expected['confidence'])[0], rtol=4e-4, atol=4e-4)
    altered = ids.copy(); altered[3:] = 31
    changed = trainer.prepare(altered, np.array([2]))
    torch.testing.assert_close(batch['features'][:1], changed['features'], rtol=0, atol=0)
    assert not batch['target_logits'].requires_grad and not batch['features'].requires_grad
    assert logits.shape == (3, 3, config.vocab_size) and confidence.shape == (3, 3)


def test_only_draft_changes_and_loss_falls(weights):
    model = make_model(weights)
    trainer = DraftTrainer(model, learning_rate=2e-3)
    ids, anchors = np.array([1,3,5,7,9,11,13,15]), np.array([2,4])
    frozen = backbone_digest(model)
    original = trainer.draft_state()
    before = trainer.evaluate(ids, anchors)['loss']
    for _ in range(15):
        result = trainer.train_step(ids, anchors)
        assert all(np.isfinite(v) for v in result.values())
    after = trainer.evaluate(ids, anchors)['loss']
    assert after < before
    assert backbone_digest(model) == frozen
    assert any(not torch.equal(original[k], v) for k,v in trainer.draft_state().items())
    assert all(not value.requires_grad and value.grad is None for name,value in model.weights.items()
               if not name.startswith('nano.dspark.'))


def test_resume_restores_optimizer_and_exact_next_update(weights, tmp_path):
    trainer = DraftTrainer(make_model(weights))
    ids, anchors = np.array([1,3,5,7,9,11,13,15]), np.array([2,4])
    identity = {'model':'test', 'seed':17}
    trainer.train_step(ids, anchors)
    save_draft_checkpoint(tmp_path/'checkpoints', trainer, identity)
    restored = DraftTrainer(make_model(weights))
    load_draft_checkpoint(tmp_path, restored, identity)
    trainer.train_step(ids, anchors)
    restored.train_step(ids, anchors)
    assert restored.steps == trainer.steps == 2
    for name, value in trainer.draft_state().items():
        torch.testing.assert_close(value, restored.draft_state()[name], rtol=0, atol=0)
    with pytest.raises(ValueError, match='settings differ'):
        load_draft_checkpoint(tmp_path, restored, {'model':'other'})


def make_corpus(root, tokenizer):
    root.mkdir()
    (root/'tokenizer.json').write_bytes(tokenizer.read_bytes())
    source = dict(source={'key':'test'}, pool='reasoning', views={'sft':{}})
    for split in ('train', 'validation'):
        directory = root/'sft/test'/split; directory.mkdir(parents=True)
        ids = np.full((2, 32768), 2, dtype=np.int32)
        seg = np.full_like(ids, -1)
        valid = np.zeros_like(ids, dtype=bool)
        target = np.zeros_like(ids, dtype=bool)
        for row in range(2):
            ids[row, :32] = 25 + row + int(split=='validation')
            ids[row, 32:64] = 35 + row + int(split=='validation')
            seg[row, :32] = 0; seg[row, 32:64] = 1
            valid[row, :64] = True
            target[row, 6:32] = True; target[row, 38:64] = True
        shard = directory/'part.npz'
        np.savez_compressed(shard, input_ids=ids, segment_ids=seg, token_mask=valid, sft_loss_mask=target)
        source['views']['sft'][split] = dict(shards=[dict(file=shard.name, rows=2,
            bytes=shard.stat().st_size, sha256=sha256_file(shard))])
    manifest = dict(format='nano-dsv41f-posttrain-v3', complete=True,
        identity=dict(tokenizer_sha256=sha256_file(tokenizer)),
        pool_mix={'sft':{'reasoning':1.0}}, sources=[source], lengths={'sft':32768})
    (root/'posttrain_manifest.json').write_text(json.dumps(manifest))
    return root


def test_data_crops_are_assistant_only_single_segment_and_deterministic(tmp_path):
    tokenizer = tmp_path/'tokenizer.json'; tokenizer.write_text('{}')
    root = make_corpus(tmp_path/'corpus', tokenizer)
    data = SFTDraftData(root, tokenizer_sha256=sha256_file(tokenizer), block_size=3, seq_len=16)
    for i in range(15):
        sample = data.sample(i)
        assert len(set(sample['input_ids'])) == 1
        assert len(sample['input_ids']) <= 16
        assert (sample['start'] < 32 and sample['end'] <= 32) or sample['start'] >= 32
        absolute = sample['anchors'] + sample['start']
        for offset in (1,2,3):
            assert np.all(((absolute+offset)%32) >= 6)
        data.sample(4, split='validation')
        np.testing.assert_array_equal(data.sample(i)['anchors'], sample['anchors'])
    fresh = SFTDraftData(root, tokenizer_sha256=sha256_file(tokenizer), block_size=3)
    next((root/'sft/test/train').glob('*.npz')).write_bytes(b'broken')
    with pytest.raises(ValueError, match='checksum'):
        fresh.sample(0)


def test_full_runner_resume_and_export_preserve_backbone_bytes(completed, tmp_path):
    training, _, _, _ = completed
    bundle = tmp_path/'source'
    export_checkpoint(training, bundle)
    corpus = make_corpus(tmp_path/'corpus', bundle/'tokenizer.json')
    output = tmp_path/'draft_run'
    common = ['--model-dir', str(bundle), '--corpus', str(corpus), '--device','cpu',
        '--seq-len','16','--anchors','2','--eval-batches','1','--rollout-tokens','3',
        '--checkpoint-every','1','--eval-every','1','--log-every','1','--threads','1']
    report = run(parse_args(common+['--output',str(output),'--steps','2']))
    assert report['status'] == 'completed' and report['backbone_unchanged']
    assert len(report['evaluations']) == 3
    assert all('rollout_greedy_acceptance' in x for x in report['evaluations'])
    exported = verify_portable_bundle(output/'bundle')
    assert exported['dspark_steps'] == 2 and exported['backbone_bitwise_preserved']
    old = load_file(str(bundle/'model.safetensors'))
    new = load_file(str(output/'bundle/model.safetensors'))
    assert old.keys() == new.keys()
    for name in old:
        if not name.startswith('nano.dspark.'):
            assert old[name].dtype == new[name].dtype
            assert torch.equal(old[name].reshape(-1).view(torch.uint8), new[name].reshape(-1).view(torch.uint8))
    assert any(not torch.equal(old[k].float(), new[k]) for k in old if k.startswith('nano.dspark.'))
    continued = run(parse_args(common+['--output',str(tmp_path/'resumed'),'--steps','3','--resume',str(output)]))
    assert continued['start_step'] == 2 and continued['completed_steps'] == 3
    assert continued['backbone_unchanged']
    # The new bundle loads through the unchanged inference format.
    loaded = NanoDeepseekV41CPU.from_pretrained(output/'bundle')
    baseline = NanoDeepseekV41CPU.from_pretrained(bundle)
    ids = torch.tensor([[1,25,26,27,28,29]])
    torch.testing.assert_close(loaded.forward(ids)[0], baseline.forward(ids)[0], rtol=0, atol=0)


def test_distillation_notebook_regenerates_and_compiles(tmp_path):
    import nbformat
    generated = build(tmp_path/'draft.ipynb')
    nb = nbformat.read(generated, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == 'code':
            compile(cell.source, 'dspark-notebook', 'exec')
    assert json.loads(generated.read_text()) == json.loads((ROOT/'notebooks/nano_dsv41f_dspark_distillation.ipynb').read_text())
