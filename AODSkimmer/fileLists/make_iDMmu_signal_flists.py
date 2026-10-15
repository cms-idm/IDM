#!/usr/bin/env python3

import argparse
import os
import re
import subprocess
from pathlib import Path
from collections import defaultdict


EOS_HOST = "root://cmseos.fnal.gov"

# Use the actual MiniAOD directory, not just the top production directory
EOS_DIR = "/store/group/lpcmetx/iDMe/iDMmu_Run3_Production/2022EE/MINIAOD"

OUT_DIR = "signal/2022EE/iDMmu"

# Set to 15 if you only want to write lists for points with >= 15 ROOT files
MIN_FILES = 1

# Write entries as root://cmseos.fnal.gov//store/...
WRITE_XROOTD_PATHS = True


def run_cmd(cmd):
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(cmd)
            + "\n\nstderr:\n"
            + result.stderr
        )

    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def clean_eos_find_line(line):
    """
    eos find usually returns plain paths, but this also handles path=... style output.
    """

    line = line.strip()

    if line.startswith("path="):
        line = line.split("path=", 1)[1]

    return line


def list_root_files(eos_dir):
    """
    Let EOS do the recursive search.

    This avoids the recursion bug from manually descending through directories.
    """

    print("Finding ROOT files with eos find...")

    lines = run_cmd([
        "eos",
        EOS_HOST,
        "find",
        "-f",
        eos_dir,
    ])

    files = []

    for line in lines:
        path = clean_eos_find_line(line)

        if path.endswith(".root"):
            files.append(path)

    return files


def extract_signal_point(path):
    """
    Extract iDMmu signal point from paths like:

    .../Mchi-10p5_dMchi-1p0/...ctau-100.../*.root

    Returns a key like:

    Mchi-10p5_dMchi-1p0_ctau-100

    If no ctau token is found, returns:

    Mchi-10p5_dMchi-1p0_ctau-UNKNOWN
    """

    mass_match = re.search(
        r"Mchi-(?P<mchi>[^/_]+)_dMchi-(?P<dmchi>[^/_]+)",
        path,
    )

    if not mass_match:
        return None

    mchi = mass_match.group("mchi")
    dmchi = mass_match.group("dmchi")

    ctau_match = re.search(
        r"ctau-(?P<ctau>[^/_]+)",
        path,
    )

    if ctau_match:
        ctau = ctau_match.group("ctau")
    else:
        ctau = "UNKNOWN"

    return f"Mchi-{mchi}_dMchi-{dmchi}_ctau-{ctau}"


def condition(path):
    """
    Keep only relevant ROOT files.
    Add tighter filters here if needed.
    """

    if not path.endswith(".root"):
        return False

    # Since EOS_DIR already points to MINIAOD, this is optional.
    # Keep it commented unless your production directory has multiple tiers.
    #
    # if "/MINIAOD/" not in path:
    #     return False

    return True


def output_path_for_key(key):
    return os.path.join(OUT_DIR, f"{key}_flist.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default="",
                    help="keep only one production, e.g. 20260928 (matches _<campaign>_ in the path); "
                         "EOS holds several productions per signal point")
    args = ap.parse_args()

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    print("Scanning EOS directory:")
    print(f"  {EOS_HOST}/{EOS_DIR}")

    files = list_root_files(EOS_DIR)
    files = [f for f in files if condition(f)]
    if args.campaign:
        files = [f for f in files if f"_{args.campaign}_" in f]

    print(f"Found {len(files)} ROOT files.")

    grouped_files = defaultdict(list)

    for f in files:
        key = extract_signal_point(f)

        if key is None:
            print("[warning] Could not parse signal point from:")
            print(f"  {f}")
            continue

        grouped_files[key].append(f)

    print(f"Found {len(grouped_files)} signal points.")

    for key in sorted(grouped_files):
        point_files = sorted(grouped_files[key])

        if len(point_files) < MIN_FILES:
            print(f"[skip] {key}: only {len(point_files)} files")
            continue

        outpath = output_path_for_key(key)

        with open(outpath, "w") as fout:
            for f in point_files:
                if WRITE_XROOTD_PATHS:
                    fout.write(f"{EOS_HOST}/{f}\n")
                else:
                    fout.write(f"{f}\n")

        print(f"[write] {outpath}: {len(point_files)} files")


if __name__ == "__main__":
    main()
