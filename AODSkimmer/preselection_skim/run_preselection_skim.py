#!/usr/bin/env python3
"""Produce and independently validate the 72 default 09-28 signal-MC skims, one file per sample.

Run in the CMSSW environment on LPC AFTER the one-file timing test:
  python3 AODSkimmer/preselection_skim/run_preselection_skim.py --run-dir <new-rundir> --workers <1..6>
Uses the existing 72 lists (four files each), rejects campaign mixing and existing outputs, records
actual commands/stdout/stderr per sample in <rundir>/logs/, and exits nonzero for any failed sample.
RUN.md and environment.log are maintained by the caller per the project study-run conventions.
"""
import argparse
import concurrent.futures
import datetime
from pathlib import Path
import shlex
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--workers", required=True, type=int, choices=range(1, 7))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    lists = sorted((repo / "AODSkimmer/fileLists/signal/2022EE/iDMmu").glob("*_flist.txt"))
    if len(lists) != 72:
        raise RuntimeError("Expected 72 default signal sample lists, found %d" % len(lists))
    all_inputs = []
    for path in lists:
        files = [line.strip() for line in path.read_text().splitlines()
                 if line.strip() and not line.lstrip().startswith("#")]
        if len(files) != 4 or any("_20260928_" not in f or "/260928_" not in f for f in files):
            raise RuntimeError("Unexpected input coverage/campaign in " + str(path))
        all_inputs.extend(files)
    if len(set(all_inputs)) != 288:
        raise RuntimeError("Expected 288 distinct input files")
    run = args.run_dir.resolve()
    logs = run / "logs"
    output = run / "outputs"
    logs.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=True)
    for path in lists:
        sample = path.name[:-10]
        if (output / (sample + ".root")).exists() or (logs / (sample + ".log")).exists():
            raise RuntimeError("Refusing to overwrite existing sample artifacts: " + sample)

    def process(path):
        sample = path.name[:-10]  # strip _flist.txt
        target = output / (sample + ".root")
        commands = [
            ["cmsRun", str(repo / "AODSkimmer/scripts/displacedMuonPreselectionSkim_cfg.py"),
             "flist=" + str(path), "outFile=" + str(target)],
            ["python3", str(repo / "AODSkimmer/preselection_skim/check_preselection_skim.py"),
             "--input", str(path), "--output", str(target)],
        ]
        start = time.monotonic()
        with (logs / (sample + ".log")).open("x", buffering=1) as log:
            log.write("Started UTC: %s\nWorking directory: %s\n" %
                      (datetime.datetime.now(datetime.timezone.utc).isoformat(), repo))
            for command in commands:
                log.write("COMMAND: " + shlex.join(command) + "\n")
                result = subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
                log.write("EXIT: %d\n" % result.returncode)
                if result.returncode:
                    return sample, result.returncode, time.monotonic() - start
            log.write("Finished UTC: %s\n" % datetime.datetime.now(datetime.timezone.utc).isoformat())
        return sample, 0, time.monotonic() - start

    failures = []
    with (logs / "production_commands.log").open("x", buffering=1) as summary:
        summary.write("Inputs: 72 samples, 288 unique 2026-09-28 signal MC files; workers=%d\n" % args.workers)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            jobs = {pool.submit(process, path): path for path in lists}
            for job in concurrent.futures.as_completed(jobs):
                try:
                    sample, code, seconds = job.result()
                except Exception as error:
                    sample, code, seconds = jobs[job].name, 1, 0
                    summary.write("EXCEPTION %s: %s\n" % (sample, error))
                line = "%s exit=%d elapsed_seconds=%.1f" % (sample, code, seconds)
                print(line, flush=True)
                summary.write(line + "\n")
                if code:
                    failures.append(sample)
        summary.write("Completed: %d passed, %d failed\n" % (len(lists) - len(failures), len(failures)))
    if failures:
        raise RuntimeError("Failed samples: " + ", ".join(failures))


if __name__ == "__main__":
    main()
