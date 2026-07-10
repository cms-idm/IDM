#!/usr/bin/env python3
"""
Check output ntuples on EOS for a failed/missing tree and resubmit those jobs.

Usage:
    python3 resubmit_failed.py [--dry-run]

Must be run from the ElectronNtuplizer condor directory (where ElectronNtuplizer_config.jdl lives).
"""

import os
import sys
import subprocess
import argparse

try:
    import uproot
except ImportError:
    sys.exit("uproot not found — run inside a CMSSW environment or pip install uproot")

VERS     = 'Jun2026noID'
YEAR     = '2024'
NTHREADS = 4
IS_DATA  = 'False'
IS_SIG   = 'True'
CMSSW    = 'CMSSW_14_0_21'
ENV_TAR  = 'ntuplizer_CMSSW_14_0_21_acrobert.tar.gz'

XRD      = 'root://cmseos.fnal.gov/'
EOS_BASE = f'/store/group/lpcmetx/iDMe/Samples/Ntuples/signal_{VERS}/{YEAR}'
TREE          = 'ntuples/outT'
REQUIRED_BRANCH = 'GenEle_matchedAllLowPt'
SPLIT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'split_fileLists')

def xrdfs_ls(path):
    r = subprocess.run(['xrdfs', 'root://cmseos.fnal.gov/', 'ls', path],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return []
    return [l.strip() for l in r.stdout.strip().splitlines() if l.strip()]


def tree_ok(xrd_url):
    try:
        with uproot.open(xrd_url) as f:
            if TREE not in f:
                return False
            t = f[TREE]
            if t.num_entries == 0:
                return False
            return REQUIRED_BRANCH in t.keys()
    except Exception:
        return False


def collect_output_files():
    """Return list of full EOS paths for every .root file under EOS_BASE."""
    files = []
    for mass_dir in xrdfs_ls(EOS_BASE):
        for ctau_dir in xrdfs_ls(mass_dir):
            for fpath in xrdfs_ls(ctau_dir):
                if fpath.endswith('.root'):
                    files.append(fpath)
    return files


def resubmit(sublist_name, out_path, dry_run):
    split_file = os.path.join(SPLIT_DIR, sublist_name + '.txt')
    if not os.path.exists(split_file):
        print(f"  [SKIP] split file list not found: {split_file}")
        return

    split_full = os.path.realpath(split_file)
    cmd = (
        f'condor_submit ElectronNtuplizer_config.jdl '
        f'-append "Arguments = {sublist_name} {YEAR} {NTHREADS} {IS_DATA} {IS_SIG} {out_path} {CMSSW} {ENV_TAR}" '
        f'-append "transfer_input_files = {split_full}" '
        f'-append "request_cpus = {NTHREADS}"'
    )
    if dry_run:
        print(f"  [DRY RUN] {cmd}")
    else:
        print(f"  Submitting {sublist_name}")
        os.system(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be submitted without actually submitting')
    args = parser.parse_args()

    # must run from the directory containing the JDL
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print(f"Scanning {EOS_BASE} ...")
    all_files = collect_output_files()
    print(f"Found {len(all_files)} output files\n")

    failed = []
    for fpath in all_files:
        url = XRD + fpath
        ok = tree_ok(url)
        status = 'OK  ' if ok else 'FAIL'
        print(f"  [{status}] {os.path.basename(fpath)}")
        if not ok:
            failed.append(fpath)

    print(f"\n{len(failed)} / {len(all_files)} files failed")

    if not failed:
        print("Nothing to resubmit.")
        return

    print("\nResubmitting failed jobs:")
    for fpath in failed:
        fname = os.path.basename(fpath)                   # ntuples_..._flist_NN.root
        sublist_name = fname.removeprefix('ntuples_').removesuffix('.root')  # ..._flist_NN
        out_path = os.path.dirname(fpath) + '/'           # /store/.../M1-.../ctau-N/
        resubmit(sublist_name, out_path, args.dry_run)


if __name__ == '__main__':
    main()
