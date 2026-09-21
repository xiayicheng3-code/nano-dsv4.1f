"""CPU-only supervisor for independent attention replay pairs and decision reports."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from attention_replay_utils import decide_h1
from run_pretrain_stress import save_report


def launch(command, report_path, timeout):
    """Preserve partial reports after compiler errors, host kills, or timeouts."""
    log = report_path.with_suffix(".log")
    if report_path.exists():
        raise FileExistsError(report_path)
    start, timed_out, last_stage = time.monotonic(), False, None
    with log.open("w") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                if report_path.exists():
                    stage = json.loads(report_path.read_text()).get("stage")
                    if stage != last_stage:
                        print(report_path.stem, stage, flush=True)
                        last_stage = stage
                if time.monotonic() - start > timeout:
                    timed_out = True
                    process.kill()
                    break
                time.sleep(1)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
    result = json.loads(report_path.read_text()) if report_path.exists() else {}
    result.update(returncode=process.returncode, timed_out=timed_out, log=str(log))
    if timed_out or process.returncode or result.get("status") != "passed":
        result["status"] = "timeout" if timed_out else "failed"
        result["failure_note"] = "Failure alone does not identify HBM or VMEM OOM; inspect the log."
    save_report(report_path, result)
    return result


def summarize(reports, families, repeats):
    summary = {"H1": {}, "H2": {}, "H4": "unresolved: inspect compiler/DMA evidence", "cases": reports}
    def lookup(family, repeat, rows, composition, variant="vmap"):
        matches = [r for r in reports if r.get("status") == "passed" and
                   all(r.get("arguments", {}).get(k) == value for k, value in
                       (("family",family),("repeat",repeat),("rows",rows),
                        ("composition",composition),("intervention","schedule")))]
        return matches[0]["variants"][variant]["combined"]["median_seconds"] if len(matches) == 1 else None
    for family in families:
        pairs, h2 = [], []
        for repeat in range(repeats):
            p = {f"{mode}{rows}": lookup(family,repeat,rows,"repeated",mode)
                 for mode in ("vmap","sequential") for rows in (4,8)}
            p["passed"] = all(v is not None for v in p.values())
            pairs.append(p)
            r4, r8 = (lookup(family,repeat,b,"repeated") for b in (4,8))
            d4, d8 = (lookup(family,repeat,b,"distinct") for b in (4,8))
            if None not in (r4,r8,d4,d8):
                h2.append({"repeated_s2": r8/(2*r4), "distinct_s2": d8/(2*d4),
                           "b8_composition_ratio": d8/r8})
        summary["H1"][family] = decide_h1(pairs)
        summary["H2"][family] = {"pairs": h2, "decision": "inconclusive"}
        if len(h2) >= 3 and all(x["distinct_s2"] >= 1.20 and x["repeated_s2"] <= 1.10
                               and x["b8_composition_ratio"] >= 1.10 for x in h2):
            summary["H2"][family]["decision"] = "supports_input_composition_effect"
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rows", default="4,8")
    p.add_argument("--families", default="local,compressed,global")
    p.add_argument("--compositions", default="repeated,distinct")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--trace-steps", type=int, default=3)
    p.add_argument("--tile-test", type=int, choices=(0,1), default=0)
    p.add_argument("--timeout", type=int, default=1800)
    a = p.parse_args()
    rows = [int(x) for x in a.rows.split(",")]
    families, compositions = a.families.split(","), a.compositions.split(",")
    if not {4,8}.issubset(rows) or any(b not in (4,8,24) for b in rows):
        p.error("rows must contain 4,8 and optionally 24")
    if a.repeats < 3 or a.warmup < 3 or a.steps < 12:
        p.error("registered protocol requires >=3 repeats, >=3 warmups, >=12 samples")
    if not set(families) <= {"local","compressed","global"} or not set(compositions) <= {"repeated","distinct"}:
        p.error("unknown family or composition")
    a.output.mkdir(parents=True, exist_ok=False)
    reports, plan = [], []
    # Alternate both shape order and A/B execution order across fresh processes.
    for repeat in range(a.repeats):
        for family in families:
            for composition in compositions:
                for batch in rows if repeat % 2 == 0 else rows[::-1]:
                    plan.append(dict(family=family, composition=composition, rows=batch,
                                     repeat=repeat, intervention="schedule"))
            if a.tile_test:
                for batch in (4,8):
                    plan.append(dict(family=family, composition="repeated", rows=batch,
                                     repeat=repeat, intervention="tile"))
    save_report(a.output / "plan.json", plan)
    for i, case in enumerate(plan):
        report_path = a.output / f"case-{i:03d}.json"
        command = [sys.executable,"-u","scripts/run_attention_replay.py", "--bank",str(a.bank),
                   "--output",str(report_path), "--warmup",str(a.warmup),"--steps",str(a.steps),
                   "--trace-steps",str(a.trace_steps if case["repeat"] == 0 else 0)]
        for key,value in case.items():
            command.extend(["--" + key.replace("_","-"), str(value)])
        print(f"Starting {i+1}/{len(plan)}: {case}", flush=True)
        result = launch(command, report_path, a.timeout)
        # Retain identifying metadata even when a worker dies before its first write.
        result.setdefault("arguments", {}).update(case)
        reports.append(result)
        save_report(a.output / "summary.json", summarize(reports, families, a.repeats))
    print(json.dumps({k:v for k,v in summarize(reports,families,a.repeats).items() if k != "cases"}, indent=2))
    if any(r["status"] != "passed" for r in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
