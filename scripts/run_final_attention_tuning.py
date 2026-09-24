"""Bounded tile search followed by full-model validation, with no production changes."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from statistics import median
import subprocess
import sys

from run_pretrain_stress import save_report
from run_stress_suite import run_case

FAMILIES = ("compressed", "global")


def select_tiles(replay, repeats=3):
    """Rank complete, stable, numerically passed comparisons; prefer near-tied smaller tiles."""
    selected, eligible, details = {}, {}, {}
    for family in FAMILIES:
        candidates = {}
        for row in replay.get("comparisons", []):
            if row["family"] != family or not row.get("complete") or row.get("passed_repeats") != repeats:
                continue
            tile = row["candidate_tile"]
            reports = [r for r in replay.get("cases", []) if r.get("arguments", {}).get("family") == family
                       and r.get("arguments", {}).get("tile") == tile]
            if len(reports) != repeats or {r["arguments"]["repeat"] for r in reports} != set(range(repeats)):
                continue
            ratios, candidate_times, baseline_times = [], [], []
            valid = True
            for report in reports:
                if report.get("status") != "passed":
                    valid = False
                    break
                for name in ("sequential", f"tile{tile}"):
                    variant = report.get("variants", {}).get(name, {})
                    errors = list(variant.get("errors", {}).values()) + variant.get("component_errors", [])
                    if (set(variant.get("errors", {})) != {"output", "dq", "dkv", "dsinks"}
                            or len(variant.get("component_errors", [])) != 4
                            or not all(e.get("passed") and e.get("finite") for e in errors)):
                        valid = False
                if not valid:
                    break
                baseline = report["variants"]["sequential"]["combined"]["median_seconds"]
                candidate = report["variants"][f"tile{tile}"]["combined"]["median_seconds"]
                if not all(math.isfinite(v) and v > 0 for v in (baseline, candidate)):
                    valid = False
                    break
                ratios.append(candidate / baseline)
                baseline_times.append(baseline)
                candidate_times.append(candidate)
            if valid:
                spread = max(max(xs) / min(xs) - 1 for xs in (baseline_times, candidate_times))
                candidates[tile] = {"ratio": median(ratios), "worst_ratio": max(ratios),
                                    "process_spread": spread, "stable": spread <= .05}
        eligible[family] = {tile for tile, value in candidates.items() if value["stable"]}
        improved = {tile: value for tile, value in candidates.items()
                    if value["stable"] and value["worst_ratio"] <= .95}
        if improved:
            best = min(v["ratio"] for v in improved.values())
            selected[family] = min(t for t, v in improved.items() if v["ratio"] <= best * 1.02)
        else:
            selected[family] = 128
        details[family] = candidates
    profiles = {"baseline128": {"compressed": 128, "global": 128}}
    # A freshly validated 512 fallback is useful if a large tile fails in the full graph.
    if all(512 in eligible[family] for family in FAMILIES):
        profiles["fallback512"] = {"compressed": 512, "global": 512}
    winner = {"compressed": selected["compressed"], "global": selected["global"]}
    if winner not in profiles.values():
        profiles["selected"] = winner
    return {"selected_by_family": selected, "profiles": profiles, "details": details,
            "rules": "Three complete numerically valid repeats; <=5% process spread for both arms; "
                     ">=5% improvement in every paired repeat. Prefer smaller tiles within 2% of best median ratio. "
                     "A validated 512 fallback is retained even if it is not the winner."}


def full_plan(profiles, inputs, repeats, warmup, steps, seed):
    plan = []
    for repeat in range(repeats):
        rows_order = (4, 8) if repeat % 2 == 0 else (8, 4)
        names = list(profiles) if repeat % 2 == 0 else list(profiles)[::-1]
        for rows in rows_order:
            for name in names:
                tiles = profiles[name]
                case = dict(profile="narrow48", top_k=4, cp=2, dp=4, batch_rows=rows,
                            splash_batch_mode="sequential", splash_block_q_dkv=128,
                            splash_compressed_block_q_dkv=tiles["compressed"],
                            splash_global_block_q_dkv=tiles["global"],
                            data=str(inputs / f"rows-{rows}.npz"), data_batches=1,
                            seed=seed, phase="both", warmup=warmup, steps=steps, trace_steps=0)
                plan.append(dict(label=name, repeat=repeat, rows=rows, case=case))
    return plan


def compare_pair(baseline, candidate, phase):
    """Loss/routing trajectory screen; deliberately not parameter-equivalence certification."""
    if baseline.get("status") != "passed" or candidate.get("status") != "passed":
        return {"passed": False, "reason": "worker failed or timed out"}
    if baseline.get("batch_bank") != candidate.get("batch_bank"):
        return {"passed": False, "reason": "input hashes or token counts differ"}
    if baseline.get("commit") != candidate.get("commit"):
        return {"passed": False, "reason": "source commits differ"}
    ignored = {"output", "trace_dir", "splash_compressed_block_q_dkv", "splash_global_block_q_dkv"}
    settings = [{k: v for k, v in r.get("arguments", {}).items() if k not in ignored}
                for r in (baseline, candidate)]
    if settings[0] != settings[1]:
        return {"passed": False, "reason": "non-tile worker settings differ"}
    a, b = baseline["phase_results"][phase], candidate["phase_results"][phase]
    pairs = list(zip(a["steps"], b["steps"]))
    if not pairs or len(a["steps"]) != len(b["steps"]) or any(x["step"] != y["step"] for x, y in pairs):
        return {"passed": False, "reason": "step alignment differs"}
    deltas = {}
    for key in ("loss", "lm_loss", "indexer_loss"):
        values = [(x[key], y[key]) for x, y in pairs]
        if not all(math.isfinite(v) for pair in values for v in pair):
            return {"passed": False, "reason": "nonfinite loss"}
        deltas[key] = max(abs(x - y) for x, y in values)
    routing_l1 = 0.0
    for x, y in pairs:
        xa, ya = (s["routing"]["physical_dispatch"]["loads_by_layer"] for s in (x, y))
        if len(xa) != len(ya) or any(len(i) != len(j) or sum(i) != sum(j) for i, j in zip(xa, ya)):
            return {"passed": False, "reason": "routing shape or total assignment count differs"}
        routing_l1 = max(routing_l1, max(sum(abs(i-j) for i, j in zip(u, v)) / max(sum(u), 1)
                                      for u, v in zip(xa, ya)))
    return {"passed": max(deltas.values()) <= .01 and routing_l1 <= .10,
            "max_absolute_loss_difference": deltas, "max_layer_load_histogram_l1_fraction": routing_l1,
            "baseline_seconds": a["median_seconds"], "candidate_seconds": b["median_seconds"],
            "time_reduction_fraction": 1 - b["median_seconds"] / a["median_seconds"]}


def summarize_full(reports, profiles, repeats):
    groups, decisions = [], []
    for rows in (4, 8):
        eligible = []
        for name in profiles:
            matched = [r for r in reports if r["label"] == name and r["rows"] == rows]
            group = {"rows": rows, "profile": name, "phases": {}, "complete": False}
            complete = (len(matched) == repeats and {r["repeat"] for r in matched} == set(range(repeats))
                        and all(r.get("status") == "passed" for r in matched))
            if complete:
                group["complete"] = True
                for phase in ("base", "late"):
                    phase_reports = [r["phase_results"][phase] for r in matched]
                    times = [r["median_seconds"] for r in phase_reports]
                    memory = [(r.get("diagnostics", {}).get("memory") or {}).get("estimated_total_gib")
                              for r in phase_reports]
                    memory = [v for v in memory if v is not None]
                    item = {"median_seconds": median(times), "process_spread": max(times)/min(times)-1,
                            "physical_tokens_per_second": rows * 8192 / median(times),
                            "max_compiler_estimate_gib": max(memory) if memory else None}
                    if name != "baseline128":
                        comparisons = []
                        for r in matched:
                            base = next((b for b in reports if b["label"] == "baseline128" and
                                         b["rows"] == rows and b["repeat"] == r["repeat"]), None)
                            comparisons.append(compare_pair(base, r, phase) if base else
                                               {"passed": False, "reason": "missing baseline"})
                        item["paired_checks"] = comparisons
                        base_times = [c["baseline_seconds"] for c in comparisons if "baseline_seconds" in c]
                        base_stable = (len(base_times) == repeats and max(base_times)/min(base_times)-1 <= .05)
                        item["meets_screen"] = (item["process_spread"] <= .05 and
                            base_stable and
                            all(c["passed"] and c.get("time_reduction_fraction", -1) >= .05 for c in comparisons))
                    group["phases"][phase] = item
                if name != "baseline128" and all(p["meets_screen"] for p in group["phases"].values()):
                    eligible.append(group)
            groups.append(group)
        base_group = next(g for g in groups if g["rows"] == rows and g["profile"] == "baseline128")
        best = min(eligible, key=lambda g: sum(v["median_seconds"] for v in g["phases"].values())) if eligible else None
        decisions.append({"rows": rows, "profile_passing_screen": best["profile"] if best else None,
                          "baseline_complete": base_group["complete"],
                          "note": "No production setting was changed. Compare physical throughput across row counts; "
                                  "this is a short fixed-batch test, not a capacity or learning-quality result."})
    return {"groups": groups, "decisions": decisions,
            "screen": ">=5% paired full-step improvement in both phases for all repeats; <=5% process spread; "
                      "max absolute matched-step loss difference <=0.01; layer load-histogram L1 fraction <=0.10. "
                      "These are short-trajectory checks, not full parameter/optimizer equivalence."}


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    replay_dir, full_dir = a.output / "replay", a.output / "full-model"
    command = [sys.executable, "-u", "scripts/run_sequential_tiles.py", "--bank", str(a.bank),
               "--output", str(replay_dir), "--rows", "8", "--families", "compressed,global",
               "--tiles", "128,512,1024,2048", "--repeats", str(a.repeats), "--warmup", str(a.warmup),
               "--steps", str(a.steps), "--trace-steps", "0", "--timeout", str(a.replay_timeout)]
    result = subprocess.run(command, check=False)
    summary = {"stage": "replay_complete", "replay_returncode": result.returncode, "full_reports": []}
    summary_path = a.output / "summary.json"
    if not (replay_dir / "summary.json").exists():
        summary.update(stage="replay_failed", status="failed", reason="No replay report was produced")
        save_report(summary_path, summary)
        return summary
    replay = json.loads((replay_dir / "summary.json").read_text())
    selection = select_tiles(replay, a.repeats)
    summary["selection"] = selection
    save_report(a.output / "selection.json", selection)
    save_report(summary_path, summary)
    profiles = selection["profiles"]
    print("Full-model configurations:", json.dumps(profiles), flush=True)
    if len(profiles) == 1:
        summary.update(stage="complete", status="inconclusive", reason="No eligible candidate; full-model sweep skipped")
        save_report(summary_path, summary)
        return summary
    full_dir.mkdir()
    plan = full_plan(profiles, a.inputs, a.repeats, a.warmup, a.steps, a.seed)
    save_report(full_dir / "plan.json", plan)
    reports, failed, blocked_rows = [], set(), set()
    environment = dict(os.environ, JAX_PLATFORMS="tpu")
    summary["skipped_full_cases"] = []
    for i, entry in enumerate(plan):
        if (entry["label"], entry["rows"]) in failed or entry["rows"] in blocked_rows:
            summary["skipped_full_cases"].append({**entry, "reason": "earlier worker failure at this configuration/row count"})
            continue
        summary.update(stage=f"full:{entry['label']}:rows{entry['rows']}:repeat{entry['repeat']}")
        save_report(summary_path, summary)
        report = run_case(entry["case"], index=i, output_dir=full_dir,
                          timeout_seconds=a.full_timeout, environment=environment)
        report.update(label=entry["label"], repeat=entry["repeat"], rows=entry["rows"])
        reports.append(report)
        if report["status"] != "passed":
            failed.add((entry["label"], entry["rows"]))
            if entry["label"] == "baseline128":
                blocked_rows.add(entry["rows"])
        summary.update(full_reports=reports, full_comparison=summarize_full(reports, profiles, a.repeats))
        save_report(full_dir / "summary.json", reports)
        save_report(summary_path, summary)
    summary.update(stage="complete", status="complete" if not failed else "completed_with_failures")
    save_report(summary_path, summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--replay-timeout", type=int, default=1800)
    p.add_argument("--full-timeout", type=int, default=3600)
    a = p.parse_args()
    if a.repeats < 3 or a.warmup < 3 or a.steps < 12 or min(a.replay_timeout, a.full_timeout) <= 0:
        p.error("require >=3 repeats/warmups, >=12 samples and positive timeouts")
    for rows in (4, 8):
        if not (a.inputs / f"rows-{rows}.npz").exists():
            p.error(f"missing rows-{rows}.npz")
    summary = run(a)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("full_reports", "skipped_full_cases")}, indent=2))


if __name__ == "__main__":
    main()
