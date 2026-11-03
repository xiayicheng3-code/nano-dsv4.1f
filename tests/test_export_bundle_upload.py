"""Verify cleanup boundaries and the exact payload sent to the Hub."""
import json
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from export_bundle_upload import WORKSPACE_MARKER, cleanup_export_workspace, upload_export_bundle
from nano_dsv41f.portable_bundle import FORMAT, REQUIRED, sha256_file


@pytest.fixture
def bundle(tmp_path):
    output = tmp_path / 'working' / 'bundle'
    output.mkdir(parents=True)
    for name in REQUIRED:
        (output / name).write_text('test content')
    files = {p.name: {'bytes': p.stat().st_size, 'sha256': sha256_file(p)}
             for p in output.iterdir()}
    (output / 'export_manifest.json').write_text(json.dumps({
        'format': FORMAT, 'complete': True, 'files': files}))
    return output


def marked_workspace(tmp_path):
    workspace = tmp_path / 'temp' / 'nano-dsv41f-export'
    workspace.mkdir(parents=True)
    (workspace / '.export-workspace').write_text(WORKSPACE_MARKER)
    (workspace / 'source').mkdir()
    (workspace / 'pip-cache').mkdir()
    return workspace


def test_cleanup_preserves_bundle_and_unrelated_files(bundle, tmp_path):
    workspace = marked_workspace(tmp_path)
    unrelated = tmp_path / 'working' / 'other-notebook.txt'
    unrelated.write_text('keep')
    before = {p.name: p.read_bytes() for p in bundle.iterdir()}
    cleanup_export_workspace(bundle, workspace)
    assert not workspace.exists()
    assert {p.name: p.read_bytes() for p in bundle.iterdir()} == before
    assert unrelated.read_text() == 'keep'
    cleanup_export_workspace(bundle, workspace)  # Already cleaned is harmless.


def test_cleanup_refuses_unmarked_or_overlapping_workspace(bundle, tmp_path):
    workspace = tmp_path / 'foreign'
    workspace.mkdir()
    with pytest.raises(ValueError, match='unmarked'):
        cleanup_export_workspace(bundle, workspace)
    assert workspace.exists()
    with pytest.raises(ValueError, match='separate'):
        cleanup_export_workspace(bundle, bundle.parent)
    with pytest.raises(ValueError, match='separate'):
        cleanup_export_workspace(bundle, bundle / 'temp')


def test_corrupt_export_blocks_cleanup(bundle, tmp_path):
    workspace = marked_workspace(tmp_path)
    (bundle / 'model.safetensors').write_text('changed')
    with pytest.raises(ValueError, match='checksum'):
        cleanup_export_workspace(bundle, workspace)
    assert workspace.exists()


def test_cleanup_refuses_symlink(bundle, tmp_path):
    workspace = marked_workspace(tmp_path)
    link = tmp_path / 'linked-workspace'
    link.symlink_to(workspace, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        cleanup_export_workspace(bundle, link)
    assert workspace.exists()


def test_upload_uses_dataset_type_and_only_verified_files(bundle, monkeypatch):
    hub = pytest.importorskip('huggingface_hub')
    calls = []
    class Client:
        def __init__(self, token):
            assert token == 'test-write-token'
        def create_repo(self, **kwargs):
            calls.append(('create', kwargs))
        def upload_folder(self, **kwargs):
            calls.append(('upload', kwargs))
            return 'upload result'
    monkeypatch.setattr(hub, 'HfApi', Client)
    (bundle / 'unrelated-secret.txt').write_text('do not upload')
    assert upload_export_bundle(bundle, 'test-account/test-export', 'test-write-token') == 'upload result'
    assert calls[0][1] == {'repo_id': 'test-account/test-export', 'repo_type': 'dataset',
                          'private': True, 'exist_ok': True}
    uploaded = calls[1][1]
    assert uploaded['repo_type'] == 'dataset'
    assert set(uploaded['allow_patterns']) == REQUIRED | {'export_manifest.json'}
    assert 'unrelated-secret.txt' not in uploaded['allow_patterns']
    assert 'delete_patterns' not in uploaded


def test_invalid_bundle_never_creates_or_uploads_repo(bundle, monkeypatch):
    hub = pytest.importorskip('huggingface_hub')
    def unexpected_client(**kwargs):
        pytest.fail('An invalid bundle must not make Hub requests')
    monkeypatch.setattr(hub, 'HfApi', unexpected_client)
    (bundle / 'tokenizer.json').write_text('changed')
    with pytest.raises(ValueError, match='checksum'):
        upload_export_bundle(bundle, 'test-account/test-export', 'test-write-token')


def test_notebook_cleans_workspace_after_failed_upload_without_exposing_secret(bundle, tmp_path, monkeypatch, capsys):
    from types import ModuleType
    workspace = marked_workspace(tmp_path)
    secrets = ModuleType('kaggle_secrets')
    class Secrets:
        def get_secret(self, name):
            assert name == 'HF_TOKEN'
            return 'test-write-token'
    secrets.UserSecretsClient = Secrets
    monkeypatch.setitem(sys.modules, 'kaggle_secrets', secrets)
    import export_bundle_upload
    def failed_transfer(output, repo_id, token, *, private):
        assert token == 'test-write-token'
        raise RuntimeError('Transfer interrupted')
    monkeypatch.setattr(export_bundle_upload, 'upload_export_bundle', failed_transfer)
    notebook = json.loads((ROOT / 'notebooks/nano_dsv41f_export_sft_safetensors_cpu.ipynb').read_text())
    cell = next(''.join(c['source']) for c in notebook['cells']
                if 'def upload_with_kaggle_secret' in ''.join(c['source']))
    namespace = {'OUTPUT': str(bundle), 'EXPORT_WORK': workspace, 'HF_REPO_ID': 'account/repo',
                 'HF_PRIVATE': True, 'HF_SECRET_NAME': 'HF_TOKEN'}
    with pytest.raises(RuntimeError, match='Transfer interrupted'):
        exec(compile(cell, 'upload-cell', 'exec'), namespace)
    assert not workspace.exists()
    assert bundle.exists()
    assert 'token' not in namespace
    captured = capsys.readouterr()
    assert 'test-write-token' not in captured.out + captured.err
