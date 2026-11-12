#!/usr/bin/env python3
"""
Check output ntuples on EOS for a failed/missing tree and resubmit those jobs.

Usage:
    python3 resubmit_failed.py <key> [<key> ...] [--dry-run]       # background
    python3 resubmit_failed.py --signal [<key> ...] [--dry-run]    # signal

    <key> is a top-level directory to check (EOS_BASE/<key>/<subsample>/*.root):
      background: a group, e.g. Multiboson (required)
      signal (--signal): a mass point, e.g. M1-0p5_dM-0p05_mZD-1p5 (subsamples are ctau-N);
                         with no key, the whole signal grid is checked

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

VERS     = 'Nov2026noIDdR'
YEAR     = '2024'
NTHREADS = 4
IS_DATA  = 'False'
CMSSW    = os.environ['CMSSW_VERSION']
ENV_TAR  = f'ntuplizer_{CMSSW}_acrobert.tar.gz'

XRD      = 'root://cmseos.fnal.gov/'
EOS_NTUPLES   = '/store/group/lpcmetx/iDMe/Samples/Ntuples'
TREE          = 'ntuples/outT'
REQUIRED_BRANCH = 'PFJet_pt'
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


def eos_bases(is_sig):
    """Return (EOS_BASE, EOS_BASE_SLIM). Signal jobs write no slim file, so its slim base is None
    (mirrors submit_ElectronNtuplizer_condor.sh)."""
    category = 'signal' if is_sig else 'background'
    base = f'{EOS_NTUPLES}/{category}_{VERS}/{YEAR}'
    base_slim = None if is_sig else f'{EOS_NTUPLES}/{category}_{VERS}_slim/{YEAR}'
    return base, base_slim


def collect_output_files(eos_base, key):
    """Return list of full EOS paths for every .root file under <eos_base>/<key>."""
    files = []
    key_dir = f'{eos_base}/{key}'
    for subsample_dir in xrdfs_ls(key_dir):
        for fpath in xrdfs_ls(subsample_dir):
            if fpath.endswith('.root'):
                files.append(fpath)
    return files


def resubmit(sublist_name, is_sig, out_path, out_path_slim, dry_run):
    split_file = os.path.join(SPLIT_DIR, sublist_name + '.txt')
    if not os.path.exists(split_file):
        print(f"  [SKIP] split file list not found: {split_file}")
        return

    split_full = os.path.realpath(split_file)
    cmd = (
        f'condor_submit ElectronNtuplizer_config.jdl '
        f'-append "Arguments = {sublist_name} {YEAR} {NTHREADS} {IS_DATA} {is_sig} {out_path} {CMSSW} {ENV_TAR} {out_path_slim}" '
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
    parser.add_argument('keys', nargs='*',
                        help='Top-level key(s) to check: background group (e.g. Multiboson) or, '
                             'with --signal, mass point (e.g. M1-0p5_dM-0p05_mZD-1p5). '
                             'Required for background; with --signal, omit to check the whole grid')
    parser.add_argument('--signal', action='store_true',
                        help='Check signal ntuples (signal_<VERS>) instead of background')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be submitted without actually submitting')
    args = parser.parse_args()

    if not args.keys and not args.signal:
        parser.error('give at least one background key (only --signal defaults to the full grid)')

    is_sig = 'True' if args.signal else 'False'
    eos_base, eos_base_slim = eos_bases(args.signal)

    # must run from the directory containing the JDL
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    if args.keys:
        keys = args.keys
    else:
        keys = sorted(os.path.basename(p) for p in xrdfs_ls(eos_base))
        if not keys:
            sys.exit(f"No signal points found under {eos_base}")
        print(f"Checking full signal grid: {len(keys)} points under {eos_base}")

    all_files = []
    failed = []
    summary = []   # (key, n_files, n_failed)
    for key in keys:
        print(f"\nScanning {eos_base}/{key} ...")
        key_files = collect_output_files(eos_base, key)
        print(f"Found {len(key_files)} output files")

        key_failed = []
        for fpath in key_files:
            ok = tree_ok(XRD + fpath)
            status = 'OK  ' if ok else 'FAIL'
            print(f"  [{status}] {os.path.basename(fpath)}")
            if not ok:
                key_failed.append(fpath)

        all_files += key_files
        failed += key_failed
        summary.append((key, len(key_files), len(key_failed)))

    if len(keys) > 1:
        width = max(len(k) for k in keys)
        print("\nSummary:")
        for key, n_files, n_failed in summary:
            note = '  (no output files)' if n_files == 0 else ''
            print(f"  {key:<{width}}  {n_failed:>4} / {n_files:<4} failed{note}")

    print(f"\n{len(failed)} / {len(all_files)} files failed")

    if not failed:
        print("Nothing to resubmit.")
        return

    print("\nResubmitting failed jobs:")
    for fpath in failed:
        fname = os.path.basename(fpath)                   # ntuples_..._flist_NN.root
        sublist_name = fname.removeprefix('ntuples_').removesuffix('.root')  # ..._flist_NN
        out_dir = os.path.dirname(fpath)                  # /store/.../<group>/<subsample>
        out_path = out_dir + '/'
        if eos_base_slim is None:
            out_path_slim = ''                            # signal: no slim output
        else:
            rel_dir = os.path.relpath(out_dir, eos_base)  # <group>/<subsample>
            out_path_slim = f'{eos_base_slim}/{rel_dir}/'
        resubmit(sublist_name, is_sig, out_path, out_path_slim, args.dry_run)


if __name__ == '__main__':
    main()
