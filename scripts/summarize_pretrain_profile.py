"""Extract compact, auditable measurements from stress JSON and XProf HLO exports.

This performs offline analysis only. XProf exports are optional and must be named
case-NN-PHASE-hlo_stats.json in --xprof-dir. Missing exports stay unmeasured.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

SCOPES = ("splash", "ragged_expert_forward", "moe_combine_scatter",
          "moe_all_gather", "moe_dispatch_sort")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def summarize(run_dir, xprof_dir):
    summary_path = run_dir / "summary.json"
    reports = json.loads(summary_path.read_text())
    out = {"schema": 1, "run": run_dir.name,
           "summary_sha256": digest(summary_path),
           "scope_precedence": list(SCOPES) + ["other"],
           "trace_method": "Sum HLO total_self_time (us) by first matching tf_op_name scope; "
                           "divide by device_count * captured_steps * 1000 for mean ms/device/step. "
                           "Source attribution can be affected by fusion; other is not an MoE category.",
           "cases": []}
    for case in reports:
        row = {"name": case["name"], "status": case["status"], "commit": case["commit"],
               "case": case["case"], "packages": case["packages"], "runtime": case["runtime"],
               "recipe_sha256": case["recipe"]["sha256"], "batch_bank": case["batch_bank"],
               "phases": {}}
        for phase, result in case["phase_results"].items():
            item = {k: result[k] for k in ("median_seconds", "p95_seconds",
                    "physical_tokens_per_second", "lm_tokens_per_second",
                    "microseconds_per_physical_token")}
            item["compiler_estimated_total_gib"] = result["diagnostics"]["memory"]["estimated_total_gib"]
            steps = [s for s in result["steps"] if not s["warmup"]]
            item["measured_steps"] = len(steps)
            item["routing_maxima_over_measured_steps_and_layers"] = {
                kind: {key: max(max(s["routing"][kind][key]) for s in steps)
                       for key in ("max_over_mean_by_layer", "chip_max_over_mean_by_layer")}
                for kind in ("real_tokens", "physical_dispatch")}
            path = xprof_dir / f"{case['name']}-{phase}-hlo_stats.json"
            item["hlo_profile"] = None
            if path.exists():
                table = json.loads(path.read_text())
                columns = [c["id"] for c in table["cols"]]
                groups = Counter()
                for entry in table["rows"]:
                    values = dict(zip(columns, (c.get("v") for c in entry["c"])))
                    name = values.get("tf_op_name") or ""
                    group = next((s for s in SCOPES if s in name), "other")
                    groups[group] += values["total_self_time"]
                count = len(result["trace"]["steps"])
                devices = case["runtime"]["device_count"]
                if count <= 0 or devices <= 0:
                    raise ValueError("Trace normalization needs positive device and step counts")
                item["hlo_profile"] = {
                    "export": path.name, "export_sha256": digest(path),
                    "captured_steps": count, "devices": devices,
                    "ms_per_device_step_by_scope": {k: v / (count * devices * 1000)
                                                    for k, v in groups.items()}}
            row["phases"][phase] = item
        out["cases"].append(row)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--xprof-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = summarize(args.run_dir, args.xprof_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(args.output)
