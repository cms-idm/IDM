#!/usr/bin/env python3
"""Run the IdmProcessor over one chunk (a filelist) and save a .coffea + .meta.yaml sidecar.

Used by condor/run_job.sh on the batch workers, and runnable locally for testing:

    python condor/run_idm_chunk.py --sample sig --filelist files.txt --output out.coffea \
        --cuts has_muon --hists muon_pt
"""

import argparse
import os

import coffea.util
from coffea import processor

from idm.tools.processor import IdmProcessor
from idm.tools import metadata
from idm.schema import MySchema


def read_filelist(path):
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--filelist", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--treename", default="ntuples/outT")
    ap.add_argument("--chunksize", type=int, default=50_000)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--cuts", default="", help="comma-separated cut names (idm.definitions.cuts)")
    ap.add_argument("--hists", default="", help="comma-separated hist names (empty = all)")
    args = ap.parse_args()

    files = read_filelist(args.filelist)
    fileset = {args.sample: {"files": files, "metadata": {"sample": args.sample}}}
    cuts = [c.strip() for c in args.cuts.split(",") if c.strip()]
    hists = [h.strip() for h in args.hists.split(",") if h.strip()] or None

    print(f"sample={args.sample}  files={len(files)}  cuts={cuts}  hists={hists or 'ALL'}")

    runner = processor.Runner(
        executor=processor.FuturesExecutor(workers=args.workers),
        schema=MySchema,
        skipbadfiles=True,
        chunksize=args.chunksize,
    )
    out = runner.run(
        fileset, treename=args.treename, processor_instance=IdmProcessor(cuts=cuts, hists=hists)
    )
    # coffea-2025's Runner wraps the accumulator as {"out":..., "processed":..., "exception":...};
    # unwrap so the .coffea holds the processor output ({"hists":..., "cutflow":...}) directly.
    result = out["out"] if isinstance(out, dict) and {"out", "processed"} <= set(out) else out

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    coffea.util.save(result, args.output)
    metadata.write_run_metadata(
        args.output, fileset=fileset, schema="MySchema", chunksize=args.chunksize,
        selections=cuts, hist_collections=hists or [],
    )
    print("Saved:", args.output, "(+ .meta.yaml sidecar)")


if __name__ == "__main__":
    main()
