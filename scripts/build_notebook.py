from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
SOURCE_GLOBS = [
    "pyproject.toml",
    "src/nano_dsv41f/*.py",
]


def iter_sources():
    for pattern in SOURCE_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            if path.is_file():
                yield path


def writefile_cell(path: Path) -> str:
    rel = path.relative_to(ROOT).as_posix()
    content = path.read_text(encoding="utf-8")
    return (
        "from pathlib import Path\n"
        f"p = Path({rel!r})\n"
        "p.parent.mkdir(parents=True, exist_ok=True)\n"
        f"p.write_text({content!r}, encoding='utf-8')\n"
        f"print('wrote {rel}')"
    )


def build(output: Path) -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            "# nano-dsv4.1f — Kaggle TPU v5e-8 notebook\n\n"
            "Generated from the maintained repository sources. Edit the Python package, "
            "not this notebook. The notebook deliberately uses Kaggle's preinstalled "
            "JAX/libtpu stack; **do not upgrade JAX inside the running TPU kernel**."
        ),
        nbf.v4.new_code_cell(
            "# Runtime preflight before touching the Python environment.\n"
            "import jax\n"
            "print('JAX:', jax.__version__)\n"
            "print('devices:', jax.devices())\n"
            "print('process_count:', jax.process_count(), 'local_device_count:', jax.local_device_count())\n"
            "if not jax.devices() or jax.devices()[0].platform != 'tpu':\n"
            "    raise RuntimeError('Select TPU in Kaggle Settings > Accelerator before continuing.')"
        ),
        nbf.v4.new_markdown_cell(
            "## Materialize the maintained package\n\n"
            "The following generated cells write the repository source files into the "
            "notebook working directory. `%pip install -e . --no-deps` then exposes the "
            "package without replacing Kaggle's TPU-matched JAX/libtpu wheels."
        ),
    ]

    nb["cells"].extend(
        nbf.v4.new_code_cell(writefile_cell(path)) for path in iter_sources()
    )
    nb["cells"].append(
        nbf.v4.new_code_cell(
            "%pip install -q -e . --no-deps\n"
            "print('installed nano-dsv41f without modifying JAX dependencies')"
        )
    )
    nb["cells"].extend(
        [
            nbf.v4.new_markdown_cell(
                "## v5e-8 topology and semantic sharding\n\n"
                "The target is one 8-chip v5e host with a 2×4 ICI topology. `jax.make_mesh` "
                "chooses a topology-aware device ordering. Logical roles reuse that same "
                "physical mesh: sequence/context, routed experts, vocabulary rows and "
                "Engram rows are not conflated into one global TP setting."
            ),
            nbf.v4.new_code_cell(
                "from nano_dsv41f import (\n"
                "    ModelConfig, TrainConfig, V5E, make_v5e_mesh, runtime_report,\n"
                "    semantic_axes, validate_sequence_length, validate_v5e_runtime,\n"
                ")\n\n"
                "print(runtime_report())\n"
                "warnings = validate_v5e_runtime()\n"
                "for warning in warnings:\n"
                "    print('WARNING:', warning)\n"
                "mesh = make_v5e_mesh()\n"
                "print('mesh:', mesh)\n"
                "print('v5e profile:', V5E)"
            ),
            nbf.v4.new_code_cell(
                "config = ModelConfig()\n"
                "train_config = TrainConfig(seq_len=4096)\n"
                "print('semantic axes:', semantic_axes(config, mesh))\n"
                "for warning in validate_sequence_length(train_config.seq_len, config, mesh):\n"
                "    print('WARNING:', warning)"
            ),
            nbf.v4.new_markdown_cell(
                "## Direct-to-shard initialization\n\n"
                "We first `eval_shape` the model, derive parameter `PartitionSpec`s, and "
                "JIT initialization with explicit output shardings. This avoids creating a "
                "full temporary copy of the model in one TPU's HBM before resharding it."
            ),
            nbf.v4.new_code_cell(
                "from nano_dsv41f import (\n"
                "    init_model_sharded, init_optimizer_state_sharded, memory_report,\n"
                ")\n\n"
                "key = jax.random.PRNGKey(0)\n"
                "params, param_specs, param_shardings = init_model_sharded(key, config, mesh)\n"
                "jax.block_until_ready(jax.tree_util.tree_leaves(params)[0])\n"
                "print(memory_report(params, param_specs, mesh))\n"
                "opt_state, opt_state_shardings = init_optimizer_state_sharded(\n"
                "    params, param_specs, config, mesh\n"
                ")\n"
                "jax.block_until_ready(jax.tree_util.tree_leaves(opt_state)[0])\n"
                "print('model + optimizer state initialized on mesh')"
            ),
            nbf.v4.new_markdown_cell(
                "## Static base and late-indexer executables\n\n"
                "Compile two separate functions. The base graph never contains selective "
                "teacher work; the late graph adds the selected-row retriever objective. "
                "The Python training driver switches executables according to the configured "
                "training phase rather than placing a large dynamic branch inside XLA."
            ),
            nbf.v4.new_code_cell(
                "from nano_dsv41f import compile_pretrain_step\n\n"
                "base_step = compile_pretrain_step(\n"
                "    params, opt_state, param_specs, config, train_config, mesh,\n"
                "    include_indexer=False, n_segments=None,\n"
                ")\n"
                "late_indexer_step = compile_pretrain_step(\n"
                "    params, opt_state, param_specs, config, train_config, mesh,\n"
                "    include_indexer=True, n_segments=1,\n"
                ")\n"
                "print('created base_step and late_indexer_step JIT executables')"
            ),
            nbf.v4.new_markdown_cell(
                "## v5e-friendly smoke batch\n\n"
                "`T=1024` gives 128 query tokens per chip under 8-way context sharding. "
                "It is long enough to exercise the decoder's 128+512 retriever eligibility "
                "later, while keeping the dense semantic attention reference small enough "
                "for a notebook smoke test."
            ),
            nbf.v4.new_code_cell(
                "import numpy as np\n"
                "from nano_dsv41f import put_training_batch\n\n"
                "smoke_t = 1024\n"
                "host_ids = (np.arange(smoke_t, dtype=np.int32)[None, :] % config.vocab_size)\n"
                "host_segments = np.zeros_like(host_ids, dtype=np.int32)\n"
                "host_mask = np.ones_like(host_ids, dtype=bool)\n"
                "ids, segments, token_mask = put_training_batch(\n"
                "    host_ids, host_segments, host_mask, config, mesh\n"
                ")\n"
                "print('input sharding:', ids.sharding)\n"
                "print('local shard shape:', ids.addressable_shards[0].data.shape)"
            ),
            nbf.v4.new_markdown_cell(
                "## First compiled training step\n\n"
                "The first call includes XLA compilation. Parameters and optimizer state are "
                "donated to the executable, so always assign the returned trees back to the "
                "same variables. Time steady-state steps only after this warmup call."
            ),
            nbf.v4.new_code_cell(
                "step = jnp.asarray(0, dtype=jnp.int32)\n"
                "params, opt_state, metrics = base_step(\n"
                "    params, opt_state, ids, segments, step, token_mask\n"
                ")\n"
                "jax.block_until_ready(metrics['loss'])\n"
                "print({k: float(v) for k, v in metrics.items() if getattr(v, 'ndim', 1) == 0})"
            ),
            nbf.v4.new_markdown_cell(
                "## Next systems milestones\n\n"
                "This notebook now establishes real v5e placement and a GSPMD baseline. "
                "The routed expert gather is still a dense semantic implementation, so its "
                "8-way parameter sharding is **not yet an efficient all-to-all EP kernel**. "
                "Likewise, dense CSA2 establishes correctness before replacing local/global "
                "attention with Splash/Pallas. The next profiling pass should inspect HLO/" 
                "collectives, then replace the two dominant communication patterns rather "
                "than guessing."
            ),
        ]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, output)
    print(f"wrote {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "notebooks" / "nano_dsv41f_kaggle.ipynb",
    )
    args = parser.parse_args()
    build(args.output)
