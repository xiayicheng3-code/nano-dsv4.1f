"""Upload a verified export and remove the notebook's temporary workspace."""
from pathlib import Path
import shutil

from nano_dsv41f.portable_bundle import verify_portable_bundle

WORKSPACE_MARKER = 'nano-dsv41f-export-workspace-v1'


def cleanup_export_workspace(output, workspace):
    """Delete only a marked temporary tree, after the preserved output verifies."""
    verify_portable_bundle(output)
    output, workspace = Path(output).resolve(), Path(workspace)
    if workspace.is_symlink():
        raise ValueError('Export workspace must not be a symlink')
    workspace = workspace.resolve()
    if output == workspace or output.is_relative_to(workspace) or workspace.is_relative_to(output):
        raise ValueError('Export output and temporary workspace must be separate')
    if not workspace.exists():
        return
    marker = workspace / '.export-workspace'
    if not marker.is_file() or marker.read_text().strip() != WORKSPACE_MARKER:
        raise ValueError('Refusing to remove an unmarked export workspace')
    shutil.rmtree(workspace)


def upload_export_bundle(output, repo_id, token, *, private=True):
    """Use an explicit token without persisting login; upload only manifest files."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import validate_repo_id

    validate_repo_id(repo_id)
    if '/' not in repo_id:
        raise ValueError('Set HF_REPO_ID to your account/repository')
    if not token or not token.strip():
        raise ValueError('The Hugging Face secret is empty')
    report = verify_portable_bundle(output)
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type='dataset', private=private, exist_ok=True)
    return api.upload_folder(repo_id=repo_id, repo_type='dataset',
        folder_path=str(Path(output)), path_in_repo='',
        allow_patterns=sorted([*report['files'], 'export_manifest.json']),
        commit_message='Upload verified nano-dsv4.1f SFT inference bundle')
