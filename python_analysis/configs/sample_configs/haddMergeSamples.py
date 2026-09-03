#!/usr/bin/env python3
"""
Merge (hadd) the many small per-sample ROOT ntuple files listed in a sample
config JSON into fewer, larger files under a new EOS directory, mirroring
each sample's existing <year>/<group>/<subgroup>/ subdirectory structure,
so histmaker_background.py's coffea Runner doesn't pay per-file
open/task-dispatch overhead on files with only a handful of events each.

Run this from the normal coffea/uproot environment (needed for XRootD file
listing and reading TTree entry counts). `hadd` itself lives in CMSSW, not
the coffea env, so this script sources a CMSSW ('cmsenv') runtime once via
a subshell and reuses its captured environment for every hadd subprocess --
it never touches this process's own os.environ, so uproot/XRootD imports
keep working normally.

After merging, point a fresh sample config at the merged directory the same
way the existing configs were built, e.g.:
    python3 makeSignalConfigs.py -m bkg -y 2024 -a aEM \
        -p /store/group/lpcmetx/iDMe/Samples/Ntuples/background_Aug2026noIDSep_merged/ \
        -n Aug2026-merged
    python3 getSignalXsec.py bkg_2024_Aug2026-merged.json
    python3 sumGenWgts.py bkg_2024_Aug2026-merged.json
(that regenerates 'location'/'blacklist' from scratch against the new files,
rather than trying to hand-patch the old config.)

Usage:
    python3 haddMergeSamples.py \
        -c bkg_2024_Aug2026.json \
        -o /store/group/lpcmetx/iDMe/Samples/Ntuples/background_Aug2026noIDSep_merged \
        --target-events 300000 --dry-run

    python3 haddMergeSamples.py \
        -c bkg_2024_Aug2026-slim.json \
        -o /store/group/lpcmetx/iDMe/Samples/Ntuples/background_Aug2026noIDSep_slim_merged \
        --target-events 300000
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import uproot

CMSSW_SRC = "/uscms_data/d3/acrobert/run3idme/CMSSW_14_0_21/src"
CMSSET_DEFAULT = "/cvmfs/cms.cern.ch/cmsset_default.sh"
EOS_LOCAL_PREFIX = "/eos/uscms"
TREEPATH = "ntuples/outT"


def eos_local(path):
    """Map an EOS xrootd-style path (e.g. /store/group/...) to its local FUSE mount."""
    path = path.rstrip("/")
    if path.startswith(EOS_LOCAL_PREFIX):
        return path
    return EOS_LOCAL_PREFIX + path


def get_cmsenv():
    """Source the CMSSW runtime in a throwaway subshell and return its
    environment as a dict, so `hadd` can be run without polluting this
    process's own env (which needs the coffea env's uproot/XRootD)."""
    cmd = (
        f"source {CMSSET_DEFAULT} && cd {CMSSW_SRC} && "
        f"eval `scramv1 runtime -sh` >/dev/null && env -0"
    )
    out = subprocess.run(["bash", "-lc", cmd], capture_output=True, check=True)
    env = {}
    for entry in out.stdout.decode().split("\0"):
        if "=" in entry:
            k, v = entry.split("=", 1)
            env[k] = v
    if not shutil.which("hadd", path=env.get("PATH", "")):
        sys.exit("Could not find `hadd` after sourcing cmsenv -- check CMSSW_SRC in this script.")
    return env


def list_sample_files(location, blacklist, treepath=TREEPATH):
    local_dir = eos_local(location)
    blacklist = set(blacklist or [])
    names = sorted(
        f for f in os.listdir(local_dir)
        if f.endswith(".root") and f not in blacklist
    )
    return [os.path.join(local_dir, f) for f in names]


def group_by_events(files, target_events, treepath=TREEPATH):
    """Greedily bucket files so each group has >= target_events entries
    (last group may be smaller). Skips files whose TTree can't be opened."""
    groups, cur, cur_n = [], [], 0
    for f in files:
        try:
            n = uproot.open(f"{f}:{treepath}").num_entries
        except Exception as e:
            print(f"  [warn] skipping unreadable file {f}: {e}")
            continue
        cur.append(f)
        cur_n += n
        if cur_n >= target_events:
            groups.append((cur, cur_n))
            cur, cur_n = [], 0
    if cur:
        groups.append((cur, cur_n))
    return groups


def run_hadd(task):
    out_file, in_files, env = task
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    cmd = ["hadd", "-f", "-j", "1", out_file] + in_files
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    ok = result.returncode == 0
    if not ok:
        print(f"[FAIL] {out_file}\n{result.stderr[-2000:]}")
    return out_file, len(in_files), ok


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-c", "--config",
                     default="bkg_2024_Aug2026.json",
                     help="sample config JSON, a list of {'name', 'location', 'blacklist', ...} "
                          "entries as produced by makeSignalConfigs.py (default: "
                          "bkg_2024_Aug2026.json)")
    ap.add_argument("-o", "--out-prefix",
                     default="/store/group/lpcmetx/iDMe/Samples/Ntuples/background_Aug2026noIDSep_merged",
                     help="new EOS base dir for merged files, xrootd-style (default: "
                          "/store/group/lpcmetx/iDMe/Samples/Ntuples/background_Aug2026noIDSep_merged)")
    ap.add_argument("--target-events", type=int, default=300_000,
                     help="approx events per merged output file (default: 300000)")
    ap.add_argument("--treepath", default=TREEPATH, help=f"tree path (default: {TREEPATH})")
    ap.add_argument("--samples", nargs="*", default=None,
                     help="restrict to these exact sample 'name' entries (default: all)")
    ap.add_argument("-g", "--group", default=None,
                     help="restrict to samples whose 'name' contains this substring "
                          "(e.g. 'WtoLNu' selects WtoLNu_1J, WtoLNu_2J, ...); "
                          "alternative to --samples for grabbing a whole family at once")
    ap.add_argument("--workers", type=int, default=4, help="parallel hadd jobs (default: 4)")
    ap.add_argument("--dry-run", action="store_true", help="print the merge plan and exit")
    args = ap.parse_args()

    if args.samples and args.group:
        sys.exit("Use either --samples or --group, not both.")

    with open(args.config) as f:
        samples = json.load(f)
    if not isinstance(samples, list) or (samples and not isinstance(samples[0], dict)):
        sys.exit(f"{args.config} is not a sample config (expected a JSON list of "
                 f"{{'name', 'location', ...}} entries, e.g. bkg_2024_Aug2026.json)")
    if args.samples:
        wanted = set(args.samples)
        samples = [s for s in samples if s["name"] in wanted]
        missing = wanted - {s["name"] for s in samples}
        if missing:
            sys.exit(f"Unknown sample name(s): {sorted(missing)}")
    elif args.group:
        samples = [s for s in samples if args.group in s["name"]]
        if not samples:
            sys.exit(f"No sample names contain '{args.group}'")

    out_prefix = args.out_prefix.rstrip("/")

    # Locations look like ".../background_Aug2026noIDSep/2024/QCD/HT2000toInf/";
    # find the common base dir shared by all samples so we can mirror the
    # <year>/<group>/<subgroup>/... structure under out_prefix instead of
    # flattening everything into out_prefix/<sample_name>/.
    in_base = os.path.commonpath([s["location"].rstrip("/") for s in samples])

    plan = []  # (out_file_xrootd_style, out_file_local, in_files, n_events)
    total_in_files = 0
    for s in samples:
        print(f"[{s['name']}] listing files at {s['location']}...")
        files = list_sample_files(s["location"], s.get("blacklist"), args.treepath)
        if not files:
            print(f"  [warn] no files found for {s['name']}, skipping")
            continue
        groups = group_by_events(files, args.target_events, args.treepath)
        if not groups:
            print(f"  [warn] no readable files for {s['name']}, skipping")
            continue
        total_in_files += len(files)
        rel_dir = os.path.relpath(s["location"].rstrip("/"), in_base)
        for i, (group_files, n) in enumerate(groups):
            out_xrootd = f"{out_prefix}/{rel_dir}/{s['name']}_merged_{i:03d}.root"
            plan.append((out_xrootd, eos_local(out_xrootd), group_files, n))
        print(f"  {len(files)} files -> {len(groups)} merged files (target ~{args.target_events} evts each)")

    if not plan:
        print("Nothing to do.")
        return

    print(f"\nTotal: {total_in_files} input files -> {len(plan)} merged output files")

    if args.dry_run:
        print("\n--- DRY RUN: merge plan (nothing written) ---")
        for out_xrootd, _, group_files, n in plan:
            print(f"{out_xrootd}  <- {len(group_files)} files, ~{n} events")
        return

    print("\nSourcing CMSSW runtime for `hadd`...")
    env = get_cmsenv()

    print(f"Running {len(plan)} hadd jobs with {args.workers} workers...")
    tasks = [(out_local, group_files, env) for _, out_local, group_files, _ in plan]
    n_ok = n_fail = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_hadd, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            out_file, n_in, ok = fut.result()
            print(f"  [{'OK' if ok else 'FAIL'}] {out_file} ({n_in} files merged)")
            n_ok += ok
            n_fail += not ok

    print(f"\nDone: {n_ok} succeeded, {n_fail} failed.")
    print(f"Merged files are under (xrootd path): {out_prefix}/<year>/<group>/<subgroup>/")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
