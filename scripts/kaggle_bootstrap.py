"""Canonical notebook bootstrap. Standard library only: runs before importing JAX."""
from pathlib import Path
from importlib.metadata import version
import os
import subprocess
import sys

if any(name in sys.modules for name in ("jax", "jaxlib", "libtpu", "tokamax", "nano_dsv41f")):
    raise RuntimeError("Restart the Kaggle session and run this cell first: JAX/TPU was already imported.")

ROOT = Path('/kaggle/working/nano-dsv4.1f')
REPO_REF = os.environ.get('NANO_DSV41F_REF', 'main')
if not ROOT.exists():
    subprocess.run(['git', 'clone', '--depth', '1',
                    'https://github.com/xiayicheng3-code/nano-dsv4.1f.git', str(ROOT)], check=True)
if not (ROOT / '.git').exists():
    raise RuntimeError(f'{ROOT} exists but is not a git checkout.')
status = subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain',
                                  '--untracked-files=no'], text=True)
if status.strip():
    raise RuntimeError('Checkout has tracked edits; preserve them before updating:\n' + status)
subprocess.run(['git', '-C', str(ROOT), 'fetch', '--depth', '1', 'origin', REPO_REF], check=True)
subprocess.run(['git', '-C', str(ROOT), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
# Pin the entire validated package combination, not just the Python jax package.
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade',
                '-r', str(ROOT / 'requirements-tpu.txt')], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', str(ROOT), '--no-deps'], check=True)
os.chdir(ROOT)
os.environ['PYTHONPATH'] = str(ROOT / 'src') + os.pathsep + os.environ.get('PYTHONPATH', '')
os.environ['JAX_PLATFORMS'] = 'tpu'
print('repo:', ROOT)
print('requested ref:', REPO_REF)
print('commit:', subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
print('packages:', {p: version(p) for p in ('jax', 'jaxlib', 'libtpu', 'tokamax')})
# Execution cells run fresh Python processes. Never retain an old TPU client in IPython.
