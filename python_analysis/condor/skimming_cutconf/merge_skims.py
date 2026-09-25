#!/usr/bin/env python3
"""
Check a skim job set and merge split samples' per-job outputs.

Samples submitted as a single job already have their one output file in <out dir>/
and only get checked. Samples split into several jobs (-n N) have their outputs in
<out dir>/parts/; once all of them exist they are hadd-ed into files of up to
--target-gb each in <out dir>/ (ntuples_<sample>_skimmed.root, or _0, _1, ... when
more than one), the entry counts are verified, and nFiles in the skimmed sample json
is updated. Parts are only deleted with --delete-parts.

Needs ROOT (hadd) and xrootd, e.g.:
  source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc13-opt/setup.sh
  python3 merge_skims.py submissions_skim_cutconf/<jobset> [--check] [--target-gb 4] [-p NAME ...] [--delete-parts]
"""
import os
import sys
import json
import argparse
import subprocess

import uproot
from XRootD import client

EOS = "root://cmseos.fnal.gov"
xrd = client.FileSystem(EOS)


def eos_sizes(path):
    """{name: size in bytes} for the files in an EOS directory ({} if it does not exist)."""
    status, listing = xrd.dirlist(path, flags=client.flags.DirListFlags.STAT)
    return {e.name: e.statinfo.size for e in listing} if status.ok else {}


def n_entries(url):
    with uproot.open(url, timeout=180) as f:
        return f["ntuples/outT"].num_entries


def group_by_size(names, sizes, target):
    groups, current, current_size = [], [], 0
    for n in names:
        if current and current_size + sizes[n] > target:
            groups.append(current)
            current, current_size = [], 0
        current.append(n)
        current_size += sizes[n]
    if current:
        groups.append(current)
    return groups


def merge_sample(name, samp, args):
    """Returns the number of merged files written to <out dir>/, or None if not merged."""
    out_dir, parts_dir = samp["out_dir"], samp["out_dir"] + "/parts"
    expected = [j["out_file"] for j in samp["jobs"]]
    part_sizes = eos_sizes(parts_dir)
    missing = [p for p in expected if p not in part_sizes]
    if missing:
        print(f"INCOMPLETE {name}: {len(missing)}/{len(expected)} part(s) missing -- not merging")
        return None
    top = [n for n in eos_sizes(out_dir) if n.endswith(".root")]
    if top:
        print(f"SKIP {name}: {out_dir} already holds merged file(s) {top}")
        return None
    groups = group_by_size(expected, part_sizes, args.target_gb * 1e9)
    print(f"MERGE {name}: {len(expected)} part(s), {sum(part_sizes.values()) / 1e9:.2f} GB -> {len(groups)} file(s)")
    if args.check:
        return None

    for k, group in enumerate(groups):
        out_file = f"{samp['out_stem']}_skimmed" + (f"_{k}" if len(groups) > 1 else "") + ".root"
        urls = [f"{EOS}/{parts_dir}/{p}" for p in group]
        subprocess.run(["hadd", "-f", out_file] + urls, check=True)
        n_in, n_out = sum(n_entries(u) for u in urls), n_entries(out_file)
        if n_in != n_out:
            os.remove(out_file)
            sys.exit(f"{name}: merged file has {n_out} entries, parts have {n_in}")
        subprocess.run(["xrdcp", "-f", "-s", out_file, f"{EOS}/{out_dir}/{out_file}"], check=True)
        os.remove(out_file)
        print(f"  wrote {out_dir}/{out_file} ({n_out} events)")

    if args.delete_parts:
        for p in expected:
            xrd.rm(f"{parts_dir}/{p}")
        xrd.rmdir(parts_dir)
        print(f"  deleted {len(expected)} part(s)")
    return len(groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("jobset_dir", help="submissions_skim_cutconf/<jobset>")
    parser.add_argument("--check", action="store_true", help="only report status, don't merge")
    parser.add_argument("--target-gb", type=float, default=4, help="max size of each merged file")
    parser.add_argument("-p", "--particular", nargs="+", default=None, help="only these sample names")
    parser.add_argument("--delete-parts", action="store_true", help="delete parts/ after a verified merge")
    args = parser.parse_args()

    with open(os.path.join(args.jobset_dir, "manifest.json")) as f:
        manifest = json.load(f)

    n_merged = {}
    incomplete = 0
    for name, samp in manifest["samples"].items():
        if args.particular and name not in args.particular:
            continue
        if samp["merged"]:
            job = samp["jobs"][0]
            ok = job["out_file"] in eos_sizes(job["out_dir"])
            incomplete += not ok
            print(f"{'OK' if ok else 'MISSING'} {name}: {job['out_dir']}/{job['out_file']}")
            continue
        n = merge_sample(name, samp, args)
        if n is not None:
            n_merged[name] = n
        elif not all(j["out_file"] in eos_sizes(samp["out_dir"] + "/parts") for j in samp["jobs"]):
            incomplete += 1

    if incomplete:
        print(f"\n{incomplete} sample(s) with missing output; resubmit with\n"
              f"  python3 submit_condor_skim_cutconf.py --resubmit {args.jobset_dir}")

    # keep nFiles in the skimmed sample json in sync with what the merge produced
    if n_merged and os.path.exists(manifest["skimmed_json"]):
        with open(manifest["skimmed_json"]) as f:
            entries = json.load(f)
        for e in entries:
            if e["name"] in n_merged:
                e["nFiles"] = n_merged[e["name"]]
        with open(manifest["skimmed_json"], "w") as f:
            json.dump(entries, f, indent=4)
        print(f"updated nFiles in {manifest['skimmed_json']}")


if __name__ == "__main__":
    main()
