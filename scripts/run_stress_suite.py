"""Supervisor without a TPU client: isolate cases and preserve logs after hard failures."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_pretrain_stress import save_report
from capacity_search import capacity_outcome, search_capacity


def run_case(case, *, index, output_dir, timeout_seconds, environment):
    name = f"case-{index:02d}"
    output = output_dir / f"{name}.json"
    if output.exists():
        raise FileExistsError(f"Use a new output directory; result already exists: {output}")
    log = output_dir / f"{name}.log"
    command = [sys.executable, "-u", "scripts/run_pretrain_stress.py", "--output", str(output)]
    for key, value in case.items():
        if key == "output":
            raise ValueError("the supervisor owns output paths")
        command.extend(["--" + key.replace("_", "-"), str(value)])
    print(f"Starting {name}: {case}; log: {log}", flush=True)
    start = time.monotonic()
    timed_out = False
    with log.open("w") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=environment)
        try:
            last_stage = None
            while process.poll() is None:
                if output.exists():
                    stage = json.loads(output.read_text()).get("stage")
                    if stage != last_stage:
                        print(f"  {name}: {stage}", flush=True)
                        last_stage = stage
                if time.monotonic() - start > timeout_seconds:
                    timed_out = True
                    process.kill()
                    break
                time.sleep(1)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
    report = json.loads(output.read_text()) if output.exists() else {"stage": "before_report"}
    report.update(returncode=process.returncode, timed_out=timed_out,
                  worker_log=str(log), wall_seconds=time.monotonic() - start)
    if timed_out:
        report["status"] = "timeout"
    elif process.returncode != 0 or report.get("status") != "passed":
        report["status"] = "failed" if process.returncode >= 0 else "terminated"
    report["failure_note"] = (
        "A signal/timeout alone does not prove device OOM. Inspect exception, last stage, and log."
        if report["status"] != "passed" else None)
    save_report(output, report)
    print(f"Finished {name}: {report['status']} ({report['wall_seconds']:.1f}s)", flush=True)
    return {"name": name, "case": case, "report": str(output), **report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--capacity-max-rows", type=int,
                        help="Search one plan entry up to this global row count, in DP-sized increments")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if not isinstance(plan, list) or not plan or not all(isinstance(c, dict) for c in plan):
        raise ValueError("plan must be a nonempty list of worker option dictionaries")
    if args.capacity_max_rows is not None:
        if len(plan) != 1 or plan[0].get("phase", "both") != "both":
            raise ValueError("capacity search requires one fixed recipe with phase=both")
        case = plan[0]
        start, multiple = int(case.get("batch_rows", 4)), int(case.get("dp", 1))
        if (multiple <= 0 or start <= 0 or args.capacity_max_rows < start
                or start % multiple or args.capacity_max_rows % multiple):
            raise ValueError("capacity bounds must be positive DP multiples with max >= start")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, JAX_PLATFORMS="tpu")
    # Preflight also lives in its own process; the supervisor never owns a TPU client.
    subprocess.run([sys.executable, "-u", "-c",
                    "from nano_dsv41f.runtime import pallas_preflight; pallas_preflight()"],
                   check=True, env=environment, timeout=args.timeout_seconds)
    summary = []

    def run_and_save(case):
        report = run_case(case, index=len(summary), output_dir=args.output_dir,
                          timeout_seconds=args.timeout_seconds, environment=environment)
        summary.append(report)
        save_report(args.output_dir / "summary.json", summary)
        return report

    if args.capacity_max_rows is not None:
        def probe(rows):
            report = run_and_save({**case, "batch_rows": rows})
            with Path(report["worker_log"]).open("rb") as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell() - 65536))
                tail = stream.read().decode(errors="replace")
            return capacity_outcome(report, tail)

        def update(state):
            save_report(args.output_dir / "capacity.json", {"case": case, **state})

        capacity = search_capacity(probe, start_rows=start, max_rows=args.capacity_max_rows,
                                   row_multiple=multiple, on_update=update)
        print("Capacity search:", json.dumps(capacity, indent=2), flush=True)
        if capacity["status"] in ("inconclusive", "no_fit_observed"):
            raise SystemExit(1)
    else:
        for case in plan:
            run_and_save(case)
        if any(case["status"] != "passed" for case in summary):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
