from XRootD import client
import json
import sys
import subprocess
import numpy as np
import re
import datetime as dt
import os
import glob
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("-m", "--mode", required=True)
parser.add_argument("-y", "--year", required=True)
parser.add_argument("-a", "--aEM", required=True)
parser.add_argument("-p", "--prefix", required=True)
parser.add_argument("-n", "--name", required=True)
parser.add_argument("-s", "--skimmed", default=False, type=bool)
parser.add_argument("-r", "--ref_file", default="")
args = parser.parse_args()

mode = args.mode
year = args.year
alpha = args.aEM
prefix = args.prefix.rstrip("/")
name = args.name
skimmed = args.skimmed
ref_file = args.ref_file


def year_value(year_string):
    """
    Convert year to int when possible.

    Examples:
        "2022"   -> 2022
        "2022EE" -> "2022EE"
    """
    try:
        return int(year_string)
    except ValueError:
        return year_string


def xrd_dir_names(xrdClient, path):
    """
    Return directory/file names from an EOS path.
    """
    status, entries = xrdClient.dirlist(path)
    print ("path1=",path)

    if not status.ok:
        print ("Not working!")
        print ("status=", status)
        print(f"[warning] Could not list: {path}")
        return []

    return [entry.name for entry in entries]


def xrd_root_files(xrdClient, path):
    """
    Return ROOT files directly inside an EOS path.
    """
    entries = xrd_dir_names(xrdClient, path)
    print ("entries=", entries)
    return [entry for entry in entries if entry.endswith(".root")]
    


if skimmed:
    ref_info = {}
    with open(ref_file, "r") as fin:
        ref_json = json.load(fin)
    for f in ref_json:
        ref_info[f["name"]] = f

if skimmed and ref_file == "":
    sys.exit("Need to specify a reference json to get info for skimmed config!")

if mode != "sig" and mode != "bkg" and mode != "data":
    sys.exit("Invalid mode: use sig/bkg/data")

xrdClient = client.FileSystem("root://cmseos.fnal.gov")


if mode == "sig":

    if skimmed:
        sig_base_dir = f"{prefix}/"
    else:
        sig_base_dir = f"{prefix}/"
        print ("sig_base_dir=", sig_base_dir)

        # Handle optional extra signal-name layer, e.g.
        #
        # /store/.../signal_iDMmu/2022EE/iDMmu/Mchi-.../ctau-.../
        #
        # Original code expected:
        #
        # /store/.../signal_iDMmu/2022EE/Mchi-.../ctau-.../
        entries_tmp = xrd_dir_names(xrdClient, sig_base_dir)
        print ("entries_tmp=", entries_tmp)

        if name in entries_tmp:
            sig_base_dir = f"{sig_base_dir}/{name}/"

    points = xrd_dir_names(xrdClient, sig_base_dir)
    

    output = []

    for p in points:
        print ("p=", p)

        if skimmed:
            mchi = float(p.split("_")[2].split("-")[1].replace("p", "."))
            dmchi = float(p.split("_")[3].split("-")[1].replace("p", "."))
            ctau = int(float(p.split("_")[4].split("-")[1].replace("p", ".")))

            for ref_pt in ref_json:
                if (
                    ref_pt["Mchi"] == mchi
                    and ref_pt["dMchi"] == dmchi
                    and ref_pt["ctau"] == ctau
                ):
                    entry = ref_pt.copy()
                    entry["location"] = f"{prefix}/{p}/"
                    output.append(entry)

        else:
            # Expected p:
            # Mchi-10p5_dMchi-1p0
            #
            # Skip things like:
            # iDMmu
            # README
            # other non-mass directories
            if not p.startswith("Mchi-"):
                print(f"[skip] Not a signal mass-point directory: {p}")
                continue

            try:
                mchi = float(p.split("_")[0].split("-")[1].replace("p", "."))
                dmchi = float(p.split("_")[1].split("-")[1].replace("p", "."))
            except Exception as exc:
                print(f"[skip] Could not parse signal point: {p}")
                print(f"       {exc}")
                continue

            if "mZD" in p:
                mzd = p.split("_")[2]
            else:
                mzd = ""

            point_dir = f"{sig_base_dir}/{p}"
            print ("point_dir=",point_dir)
            lifetimes = xrd_dir_names(xrdClient, point_dir)

            for l in lifetimes:
                # Expected l:
                # ctau-1
                # ctau-10
                # ctau-100
                # ctau-1000
                if not l.startswith("ctau-"):
                    print(f"[skip] Not a ctau directory: {point_dir}/{l}")
                    continue

                try:
                    ct = int(float(l.split("-")[1].replace("p", ".")))
                except Exception as exc:
                    print(f"[skip] Could not parse ctau directory: {point_dir}/{l}")
                    print(f"       {exc}")
                    continue

                info = {}
                info["location"] = f"{sig_base_dir}/{p}/{l}/"
                info["Mchi"] = mchi
                info["dMchi"] = dmchi
                info["ctau"] = ct

                if mzd != "":
                    info["name"] = "sig_Mchi-{0}_dMchi-{1}_ct-{2}_{3}".format(
                        info["Mchi"],
                        info["dMchi"],
                        info["ctau"],
                        mzd,
                    )
                else:
                    info["name"] = "sig_Mchi-{0}_dMchi-{1}_ct-{2}".format(
                        info["Mchi"],
                        info["dMchi"],
                        info["ctau"],
                    )

                info["sum_wgt"] = 0.0
                info["type"] = "signal"
                info["year"] = year_value(year)
                info["alphaD"] = alpha
                info["xsec"] = 0.0

                rootFiles = xrd_root_files(xrdClient, info["location"])
                info["nFiles"] = len(rootFiles)

                output.append(info)

    if skimmed:
        out_json = "skimmed_signal_{0}_{1}.json".format(year, name)
    else:
        out_json = "signal_{0}_{1}_{2}.json".format(year, name, alpha)

    with open(out_json, "w") as outfile:
        json.dump(output, outfile, indent=4)


elif mode == "bkg":

    if skimmed:
        status, bkgs = xrdClient.dirlist(f"{prefix}/")
    else:
        status, bkgs = xrdClient.dirlist(f"{prefix}/{year}/")

    bkgs = [bkg.name for bkg in bkgs]

    output = []

    for bkg in bkgs:

        if skimmed:
            base_dir = f"{prefix}/{bkg}"
            subsamples = [bkg]
        else:
            base_dir = f"{prefix}/{year}/{bkg}"
            subsamples = [d.name for d in xrdClient.dirlist(base_dir)[1]]

        for subsample in subsamples:

            if skimmed:
                target_dir = base_dir
            else:
                target_dir = f"{base_dir}/{subsample}/"

            rootFiles = [
                f
                for f in glob.glob(f"/eos/uscms/{target_dir}/**/*.root", recursive=True)
            ]

            fileDirs = ["/".join(f.split("/")[:-1]) + "/" for f in rootFiles]
            fileDirs = [d.split("/eos/uscms/")[-1] for d in fileDirs]
            fileDirs = list(set(fileDirs))

            info = {}

            if skimmed:
                info["name"] = subsample.replace("output_", "")
            else:
                info["name"] = f"{bkg}_{subsample}"

            info["location"] = fileDirs[0] if len(fileDirs) == 1 else fileDirs
            info["type"] = "bkg"
            info["year"] = year_value(year)

            nFiles = 0
            for fdir in fileDirs:
                nFiles += len(
                    [rf.name for rf in xrdClient.dirlist(fdir)[1] if ".root" in rf.name]
                )

            info["nFiles"] = nFiles

            if skimmed:
                samp_name = subsample.replace("output_", "")
                info["sum_wgt"] = ref_info[samp_name]["sum_wgt"]
                info["xsec"] = ref_info[samp_name]["xsec"]
                info["blacklist"] = ref_info[samp_name]["blacklist"]
            else:
                info["sum_wgt"] = 0.0
                info["xsec"] = 0.0

            output.append(info)

    if skimmed:
        out_json = "skimmed_bkg_{0}_{1}.json".format(year, name)
    else:
        out_json = "bkg_{0}_{1}.json".format(year, name)

    with open(out_json, "w") as outfile:
        json.dump(output, outfile, indent=4)


elif mode == "data":

    if skimmed:
        status, samples = xrdClient.dirlist(f"{prefix}/")
    else:
        status, samples = xrdClient.dirlist(f"{prefix}/{year}/")

    samples = [samp.name for samp in samples]

    output = []

    for samp in samples:

        if skimmed:
            base_dir = f"{prefix}/{samp}"
            subsamples = [samp]
        else:
            base_dir = f"{prefix}/{year}/{samp}"
            subsamples = [d.name for d in xrdClient.dirlist(base_dir)[1]]

        for subsample in subsamples:

            if skimmed:
                target_dir = base_dir
            else:
                target_dir = f"{base_dir}/{subsample}/"

            rootFiles = [
                f
                for f in glob.glob(f"/eos/uscms/{target_dir}/**/*.root", recursive=True)
            ]

            fileDirs = ["/".join(f.split("/")[:-1]) + "/" for f in rootFiles]
            fileDirs = [d.split("/eos/uscms/")[-1] for d in fileDirs]
            fileDirs = list(set(fileDirs))

            info = {}

            if skimmed:
                info["name"] = subsample.replace("output_", "")
            else:
                info["name"] = f"{samp}_{subsample}"

            info["location"] = fileDirs[0] if len(fileDirs) == 1 else fileDirs
            info["sum_wgt"] = 0.0
            info["type"] = "data"
            info["year"] = year_value(year)
            info["xsec"] = 0.0

            nFiles = 0
            for fdir in fileDirs:
                nFiles += len(
                    [rf.name for rf in xrdClient.dirlist(fdir)[1] if ".root" in rf.name]
                )

            info["nFiles"] = nFiles

            if skimmed:
                samp_name = subsample.replace("output_", "")
                info["num_events"] = ref_info[samp_name]["num_events"]
                info["blacklist"] = ref_info[samp_name]["blacklist"]

            output.append(info)

    if skimmed:
        out_json = "skimmed_data_{0}_{1}.json".format(year, name)
    else:
        out_json = "data_{0}_{1}.json".format(year, name)

    with open(out_json, "w") as outfile:
        json.dump(output, outfile, indent=4)

