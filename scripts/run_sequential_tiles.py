"""Focused sequential Splash tile comparisons; no batching or full-model sweep."""
from __future__ import annotations

import argparse
from pathlib import Path
from statistics import median
import sys

from run_attention_experiment import launch
from run_pretrain_stress import save_report


def summarize(cases, preflights, families, tiles, repeats):
    comparisons = []
    for family in families:
        for tile in tiles:
            selected = [r for r in cases if r.get("arguments", {}).get("family") == family
                        and r.get("arguments", {}).get("tile") == tile]
            passed = [r for r in selected if r.get("status") == "passed"]
            row = {"family": family, "candidate_tile": tile, "baseline_tile": 128,
                   "complete": len(passed) == repeats, "passed_repeats": len(passed)}
            if passed:
                pairs = [{"repeat": r["arguments"]["repeat"],
                          "baseline_seconds": r["variants"]["sequential"]["combined"]["median_seconds"],
                          "candidate_seconds": r["variants"][f"tile{tile}"]["combined"]["median_seconds"]}
                         for r in passed]
                for pair in pairs:
                    pair["time_reduction_fraction"] = 1 - pair["candidate_seconds"] / pair["baseline_seconds"]
                row.update(pairs=pairs,
                    median_time_reduction_fraction=median(p["time_reduction_fraction"] for p in pairs),
                    baseline_process_spread=max(p["baseline_seconds"] for p in pairs) /
                                           min(p["baseline_seconds"] for p in pairs) - 1)
            comparisons.append(row)
    complete = all(r["complete"] for r in comparisons)
    return {"experiment": "sequential_tiles", "status": "passed" if complete else "incomplete",
            "schedule": "sequential", "comparisons": comparisons,
            "preflights": preflights, "cases": cases,
            "note": "Isolated forward+VJP timing; no full-model claim, automatic adoption, or VMEM diagnosis."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, choices=(4, 8), default=8)
    parser.add_argument("--families", default="compressed,global")
    parser.add_argument("--tiles", default="128,256,512")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--trace-steps", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=1800)
    a = parser.parse_args()
    families = a.families.split(",")
    tiles = [int(t) for t in a.tiles.split(",")]
    if not families or len(set(families)) != len(families) or not set(families) <= {"compressed", "global"}:
        parser.error("families must be compressed and/or global, without duplicates")
    if 128 not in tiles or len(set(tiles)) != len(tiles) or not set(tiles) <= {128, 256, 512} or len(tiles) < 2:
        parser.error("tiles must include baseline 128 and candidate 256 and/or 512, without duplicates")
    tiles.remove(128)
    if a.repeats < 3 or a.warmup < 3 or a.steps < 12 or a.trace_steps < 0 or a.timeout <= 0:
        parser.error("require >=3 repeats/warmups, >=12 samples, nonnegative traces and positive timeout")
    a.output.mkdir(parents=True, exist_ok=False)
    cases, preflights, eligible = [], [], set()
    plan = [{"family": family, "tile": tile, "repeat": repeat}
            for repeat in range(a.repeats) for family in families
            for tile in (tiles if repeat % 2 == 0 else tiles[::-1])]
    save_report(a.output / "plan.json", plan)

    def execute(case, name, preflight=False):
        command = [sys.executable, "-u", "scripts/run_attention_replay.py",
                   "--bank", str(a.bank), "--output", str(a.output / f"{name}.json"),
                   "--rows", str(a.rows), "--composition", "distinct",
                   "--intervention", "sequential_tile", "--warmup", str(a.warmup),
                   "--steps", str(a.steps), "--trace-steps",
                   str(a.trace_steps if not preflight and case["repeat"] == 0 else 0)]
        for key, value in case.items():
            command.extend(["--" + key, str(value)])
        if preflight:
            command.append("--preflight-only")
        result = launch(command, a.output / f"{name}.json", a.timeout)
        result.setdefault("arguments", {}).update(case, rows=a.rows, composition="distinct",
                                                  intervention="sequential_tile")
        return result

    # Check each candidate separately. A 512 compiler/numerical failure cannot
    # suppress a valid 256 comparison, and failed pairs never enter timed repeats.
    for family in families:
        for tile in tiles:
            print(f"Preflight: {family}, sequential 128 versus {tile}", flush=True)
            result = execute(dict(family=family, tile=tile, repeat=0), f"preflight-{family}-{tile}", True)
            preflights.append(result)
            if result["status"] == "passed":
                eligible.add((family, tile))
            save_report(a.output / "summary.json", summarize(cases, preflights, families, tiles, a.repeats))
    for i, case in enumerate(plan):
        if (case["family"], case["tile"]) not in eligible:
            continue
        print(f"Timing {case}", flush=True)
        cases.append(execute(case, f"case-{i:03d}"))
        save_report(a.output / "summary.json", summarize(cases, preflights, families, tiles, a.repeats))
    summary = summarize(cases, preflights, families, tiles, a.repeats)
    save_report(a.output / "summary.json", summary)
    print("Focused experiment:", summary["status"], flush=True)
    if summary["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
