#!/usr/bin/env python3
"""Chunk per-sample filelists into condor jobs.

Input: a directory of filelists, one ``<sample>.txt`` per sample, each listing ntuple ROOT
files (one per line; xrootd URLs recommended). Output: per-chunk filelists under ``--outdir``
and a ``job_args.txt`` with one ``<sample> <chunk> <filelist>`` line per job (consumed by
condor/submit.sub).

    python condor/make_job_args.py --filelists-dir condor/filelists_in --files-per-job 5
"""

import argparse
import os
from pathlib import Path


def read_filelist(path):
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filelists-dir", required=True,
                    help="dir of <sample>.txt filelists (one ROOT file per line)")
    ap.add_argument("--files-per-job", type=int, default=5)
    ap.add_argument("--outdir", default="condor/filelists",
                    help="where per-chunk filelists are written")
    ap.add_argument("--job-args", default="condor/job_args.txt")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    n_jobs = 0
    with open(args.job_args, "w") as job_args:
        for src in sorted(Path(args.filelists_dir).glob("*.txt")):
            sample = src.stem
            files = read_filelist(src)
            for i in range(0, len(files), args.files_per_job):
                chunk = i // args.files_per_job
                chunk_files = files[i:i + args.files_per_job]
                fl = Path(args.outdir) / f"{sample}_{chunk}.txt"
                with open(fl, "w") as f:
                    f.write("\n".join(chunk_files) + "\n")
                # paths are written relative to condor/ (condor_submit runs from there)
                job_args.write(f"{sample} {chunk} {os.path.relpath(fl, 'condor')}\n")
                n_jobs += 1
    print(f"Made {n_jobs} jobs; wrote {args.job_args}")


if __name__ == "__main__":
    main()
