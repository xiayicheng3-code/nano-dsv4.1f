"""Supervisor without a TPU client: isolate cases and preserve logs after hard failures."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_pretrain_stress import save_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if not isinstance(plan, list) or not plan:
        raise ValueError("plan must be a nonempty list of worker option dictionaries")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, JAX_PLATFORMS="tpu")
    # Preflight also lives in its own process; the supervisor never owns a TPU client.
    subprocess.run([sys.executable, "-u", "-c",
                    "from nano_dsv41f.runtime import pallas_preflight; pallas_preflight()"],
                   check=True, env=environment, timeout=args.timeout_seconds)
    summary = []
    for index, case in enumerate(plan):
        name = f"case-{index:02d}"
        output = args.output_dir / f"{name}.json"
        if output.exists():
            raise FileExistsError(f"Use a new output directory; result already exists: {output}")
        log = args.output_dir / f"{name}.log"
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
                    if time.monotonic() - start > args.timeout_seconds:
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
        summary.append({"name": name, "case": case, "report": str(output), **report})
        save_report(args.output_dir / "summary.json", summary)
        print(f"Finished {name}: {report['status']} ({report['wall_seconds']:.1f}s)", flush=True)
    if any(case["status"] != "passed" for case in summary):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
