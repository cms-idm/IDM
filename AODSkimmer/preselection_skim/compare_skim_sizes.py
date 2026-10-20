#!/usr/bin/env python3
"""Compare each completed sample skim with its four original MINIAOD file sizes (metadata only).

In the LPC CMSSW environment:
  python3 AODSkimmer/preselection_skim/compare_skim_sizes.py --run-dir <completed-rundir>
Requires all 72 skim/check jobs to have passed. Saves size_comparison.md, exact input stat responses
and output sizes in logs/file_sizes.log, and event-count validation totals in logs/validation_summary.log.
Refuses to overwrite those records. MiB/GiB use powers of 1024; size reduction is not event efficiency.
"""
import argparse
import concurrent.futures
from pathlib import Path
import re
import shlex
import subprocess
from urllib.parse import urlsplit


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    repo = Path(__file__).resolve().parents[2]
    if "Completed: 72 passed, 0 failed" not in (run / "logs/production_commands.log").read_text():
        raise RuntimeError("Full production/validation has not passed")
    lists = sorted((repo / "AODSkimmer/fileLists/signal/2022EE/iDMmu").glob("*_flist.txt"))
    if len(lists) != 72:
        raise RuntimeError("Expected 72 sample lists")
    samples = []
    all_files = []
    processed = retained = 0
    for path in lists:
        sample = path.name[:-10]
        files = [line.strip() for line in path.read_text().splitlines()
                 if line.strip() and not line.lstrip().startswith("#")]
        if len(files) != 4 or any("_20260928_" not in f or "/260928_" not in f for f in files):
            raise RuntimeError("Unexpected original input coverage/campaign: " + sample)
        checks = re.findall(r"PASS: input_events=(\d+) retained_events=(\d+);",
                            (run / "logs" / (sample + ".log")).read_text())
        if len(checks) != 1:
            raise RuntimeError("Expected exactly one independent validation PASS: " + sample)
        count, kept = map(int, checks[0])
        processed += count
        retained += kept
        samples.append((sample, files, (run / "outputs" / (sample + ".root")).stat().st_size))
        all_files.extend(files)
    if len(set(all_files)) != 288 or len(list((run / "outputs").glob("*.root"))) != 72:
        raise RuntimeError("Unexpected original/output file coverage")
    targets = [run / "size_comparison.md", run / "logs/file_sizes.log", run / "logs/validation_summary.log"]
    if any(path.exists() for path in targets):
        raise RuntimeError("Refusing to overwrite comparison records")

    def original_size(url):
        parts = urlsplit(url)
        if parts.scheme != "root" or parts.netloc != "cmseos.fnal.gov":
            raise RuntimeError("Unexpected input location: " + url)
        command = ["xrdfs", "root://" + parts.netloc, "stat", parts.path]
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        match = re.search(r"^Size:\s+(\d+)\s*$", result.stdout, re.MULTILINE)
        if result.returncode or not match:
            raise RuntimeError("File-size query failed: %s\n%s" % (shlex.join(command), result.stdout))
        return url, int(match.group(1)), shlex.join(command), result.stdout

    sizes = {}
    with targets[1].open("x") as log:
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            for url, size, command, response in pool.map(original_size, all_files):
                sizes[url] = size
                log.write("COMMAND: %s\n%s\n" % (command, response))
        for sample, _, size in samples:
            log.write("OUTPUT_BYTES %s %d\n" % (sample, size))
    original_total = sum(sizes.values())
    skim_total = sum(sample[2] for sample in samples)
    mib, gib = 1024 ** 2, 1024 ** 3
    with targets[0].open("x") as report:
        report.write("# MINIAOD preselection: per-sample file sizes\n\n")
        report.write("Each row compares one completed sample skim with its four original 2026-09-28 "
                     "MINIAOD inputs. Individual input sizes follow the tracked file-list order. "
                     "MiB = 2^20 bytes; GiB = 2^30 bytes. Totals and reductions use exact bytes before rounding. "
                     "Size reduction is a storage comparison, not event efficiency.\n\n")
        report.write("Coverage: 72 samples, 288 original files, 72 skim files; all independent checks passed.\n\n")
        report.write("| Sample | Four original files (MiB) | Original total (MiB) | Skim (MiB) | Size reduction |\n")
        report.write("|---|---:|---:|---:|---:|\n")
        for sample, files, size in samples:
            individual = [sizes[url] for url in files]
            total = sum(individual)
            report.write("| %s | %s | %.1f | %.1f | %.1f%% |\n" %
                         (sample, ", ".join("%.1f" % (n / mib) for n in individual), total / mib,
                          size / mib, 100 * (1 - size / total)))
        report.write("\n**Total:** %.2f GiB original → %.2f GiB skim; %.1f%% size reduction.\n" %
                     (original_total / gib, skim_total / gib, 100 * (1 - skim_total / original_total)))
        report.write("\nExact original stat responses and skim byte sizes: `logs/file_sizes.log`. "
                     "Input lists and processing revision: `RUN.md`.\n")
    summary = ("PASS: 72 samples; 288 distinct original files; 72 output ROOT files; "
               "input_events=%d retained_events=%d (raw, unweighted event counts; cleaning + component ID; "
               "no eta/pT/gen/charge/matching gate).\nOriginal bytes=%d; skim bytes=%d; "
               "original_GiB=%.3f; skim_GiB=%.3f; size_reduction_percent=%.3f\n" %
               (processed, retained, original_total, skim_total, original_total / gib, skim_total / gib,
                100 * (1 - skim_total / original_total)))
    targets[2].write_text(summary)
    print(summary, end="")


if __name__ == "__main__":
    main()
