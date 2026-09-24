"""Freeze actual recipe attention inputs from a real-corpus forward pass."""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from run_pretrain_stress import make_batch, save_report

FAMILIES = {"local": 0, "compressed": 1, "global": 3}


def capture(data, output, rows=8, seed=7):
    import jax
    import jax.numpy as jnp
    from nano_dsv41f import (init_model_sharded_mixed_precision, make_v5e_mesh,
                            put_training_batch, validate_v5e_runtime)
    from nano_dsv41f.model import apply_model, build_layer_specs
    from nano_dsv41f.pretrain_recipe import pretrain_recipe, recipe_manifest
    from nano_dsv41f.tpu_native import install_model_dispatch, tpu_native_context
    from nano_dsv41f.runtime import package_versions
    validate_v5e_runtime()
    config, train, native = pretrain_recipe(profile="narrow48", cp=2, dp=4)
    native = replace(native, need_teacher_lse=True)
    mesh = make_v5e_mesh()
    batch, batch_info = make_batch(config, batch_rows=rows, data=data)
    ids, segments, mask = put_training_batch(*batch, config, mesh)
    params, _, _ = init_model_sharded_mixed_precision(
        jax.random.PRNGKey(seed), config, mesh, payload_dtype=jnp.bfloat16)
    install_model_dispatch()

    def forward(p, ids, seg, valid):
        _, aux = apply_model(p, config, ids, segment_ids=seg, token_mask=valid)
        chosen = tuple({key: aux["layers"][layer][key] for key in
                        ("q", "local_kv", "main_kv", "global_segment_ids", "attn_sink")}
                       for layer in FAMILIES.values())
        # Retain all seven backbone layers, without materializing vocabulary logits.
        return chosen, jnp.sum(aux["final_hidden"].astype(jnp.float32))

    print("Capturing seven-layer forward pass", flush=True)
    with tpu_native_context(mesh, native):
        values, checksum = jax.jit(forward)(params, ids, segments, mask)
        jax.block_until_ready((values, checksum))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "seed": seed, "rows": rows, "batch": batch_info,
                "recipe": recipe_manifest(config, train, native), "packages": package_versions(),
                "snapshot": "initialized BF16 recipe parameters; no optimizer updates",
                "final_hidden_sum": float(checksum), "families": {}}
    for (family, layer), aux in zip(FAMILIES.items(), values):
        q, kv = aux["q"], aux["local_kv"]
        kvseg = segments
        if aux["main_kv"] is not None:
            kv = jnp.concatenate((kv, aux["main_kv"]), axis=1)
            kvseg = jnp.concatenate((segments, aux["global_segment_ids"]), axis=1)
        arrays = dict(q=q, k=kv, v=kv, q_segments=segments, kv_segments=kvseg,
                      sinks=aux["attn_sink"])
        arrays["cotangent"] = jax.random.normal(jax.random.PRNGKey(900 + layer), q.shape, dtype=q.dtype)
        metadata = {"layer": layer, "save_residuals": False,
                    "local_window": config.attention.local_window,
                    "ratio": max(1, build_layer_specs(config)[layer].compression_ratio),
                    "arrays": {}}
        stored = {}
        for name, value in arrays.items():
            host = np.asarray(jax.device_get(value))
            metadata["arrays"][name] = {"dtype": str(host.dtype), "shape": list(host.shape),
                                       "sha256": hashlib.sha256(host.tobytes()).hexdigest()}
            # BF16 values are losslessly representable in FP32; avoid NPZ void dtype.
            stored[name] = host.astype(np.float32) if str(host.dtype) == "bfloat16" else host
        path = output / f"{family}.npz"
        np.savez(path, **stored)
        metadata["file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["families"][family] = metadata
        save_report(output / "manifest.json", manifest)
    print("Frozen replay bank:", output, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rows", type=int, default=8)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    capture(a.data, a.output, a.rows, a.seed)
