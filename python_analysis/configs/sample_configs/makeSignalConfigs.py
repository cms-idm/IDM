from XRootD import client
import json
import sys
import subprocess
import numpy as np
import re
import datetime as dt
import os
from argparse import ArgumentParser
import glob

parser = ArgumentParser()
parser.add_argument("-m","--mode",required=True)
parser.add_argument("-y","--year",required=True)
parser.add_argument("-a","--aEM",required=True)
parser.add_argument("-p","--prefix",required=True)
parser.add_argument("-n","--name",required=True)
parser.add_argument("-s","--skimmed",default=False,type=bool)
parser.add_argument("-r","--ref_file",default="")
args = parser.parse_args()

mode = args.mode
year = args.year
alpha = args.aEM
prefix = args.prefix
name = args.name
skimmed = args.skimmed
ref_file = args.ref_file

if skimmed:
    ref_info = {}
    with open(ref_file,"r") as fin:
        ref_json = json.load(fin)
    for f in ref_json:
        ref_info[f['name']] = f

if skimmed and ref_file == "":
    sys.exit("Need to specify a reference json to get info for skimmed config!")

if mode != "sig" and mode != "bkg" and mode != "data":
    sys.exit("Invalid mode: use sig/bkg/data")

xrdClient = client.FileSystem("root://cmseos.fnal.gov")

if mode == "sig":
    if skimmed:
        status, points = xrdClient.dirlist(f"{prefix}/")
    else:
        status, points = xrdClient.dirlist(f"{prefix}/{year}/")
    points = [item.name for item in points]
    output = []
    for p in points:
        if skimmed:
            mchi = float(p.split("_")[2].split("-")[1].replace("p","."))
            dmchi = float(p.split("_")[3].split("-")[1].replace("p","."))
            ctau = int(p.split("_")[4].split("-")[1].replace("p","."))
            for ref_pt in ref_json:
                if ref_pt['Mchi'] == mchi and ref_pt['dMchi'] == dmchi and ref_pt['ctau'] == ctau:
                    entry = ref_pt
                    entry['location'] = f"{prefix}/{p}/"
                    output.append(entry)
        else:
            mchi = float(p.split("_")[0].split("-")[1].replace("p","."))
            dmchi = float(p.split("_")[1].split("-")[1].replace("p","."))
            if 'mZD' in p:
                mzd = p.split("_")[2]
            else:
                mzd = ""
            status, lifetimes = xrdClient.dirlist(f"{prefix}/{year}/{p}")
            lifetimes = [l.name for l in lifetimes]
            
            for l in lifetimes:
                ct = int(l.split("-")[1])
                info = {}
                info["location"] = f"{prefix}/{year}/{p}/{l}/"
                info["Mchi"] = mchi
                info["dMchi"] = dmchi
                info["ctau"] = ct
                if mzd != "":
                    info["name"] = "sig_Mchi-{0}_dMchi-{1}_ct-{2}_{3}".format(info["Mchi"],info["dMchi"],info["ctau"],mzd)
                else:
                    info["name"] = "sig_Mchi-{0}_dMchi-{1}_ct-{2}".format(info["Mchi"],info["dMchi"],info["ctau"])
                info["sum_wgt"] = 0.0
                info["type"] = "signal"
                info["year"] = int(year)
                info["alphaD"] = alpha
                info["xsec"] = 0.0
                rootFiles = [rf.name for rf in xrdClient.dirlist(info["location"])[1] if '.root' in rf.name]
                info["nFiles"] = len(rootFiles)
                output.append(info)

    if skimmed:
        out_json = "skimmed_signal_{0}_{1}.json".format(year,name)
    else:
        out_json = "signal_{0}_{1}_{2}.json".format(year,name,alpha)
    with open(out_json,"w") as outfile:
        json.dump(output,outfile,indent=4)
elif mode == "bkg":
    if skimmed:
        status,bkgs = xrdClient.dirlist(f"{prefix}/")
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
            #rootFiles = subprocess.run(['eos','root://cmseos.fnal.gov/','find','-name','*.root','-f',target_dir],stdout=subprocess.PIPE).stdout.decode('utf-8').splitlines()
            #rootFiles = [r for r in rootFiles if '.root' in r]
            rootFiles = [ f for f in glob.glob(f"/eos/uscms/{target_dir}/**/*.root", recursive=True) ]
            fileDirs = ["/".join(f.split("/")[:-1])+"/" for f in rootFiles]
            fileDirs = [ d.split("/eos/uscms/")[-1] for d in fileDirs]
            fileDirs = list(set(fileDirs)) # list of unique file directories
            
            info = {}
            if skimmed:
                info["name"] = subsample.replace("output_","")
            else:
                info["name"] = f"{bkg}_{subsample}"
            info["location"] = fileDirs[0] if len(fileDirs) == 1 else fileDirs
            info["type"] = "bkg"
            info["year"] = int(year)
            nFiles=0
            for fdir in fileDirs:
                nFiles += len([rf.name for rf in xrdClient.dirlist(fdir)[1] if '.root' in rf.name])
            info["nFiles"] = nFiles

            if skimmed:
                samp_name = subsample.replace("output_","")
                info['sum_wgt'] = ref_info[samp_name]['sum_wgt']
                info['xsec'] = ref_info[samp_name]['xsec']
                info['blacklist'] = ref_info[samp_name]['blacklist']
            else:
                info["sum_wgt"] = 0.0
                info["xsec"] = 0.0

            output.append(info)

    if skimmed:
        out_json = "skimmed_bkg_{0}_{1}.json".format(year,name)
    else:
        out_json = "bkg_{0}_{1}.json".format(year,name)
    with open(out_json,"w") as outfile:
        json.dump(output,outfile,indent=4)
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
            #rootFiles = subprocess.run(['eos','root://cmseos.fnal.gov/','find','-name','*.root','-f',target_dir],stdout=subprocess.PIPE).stdout.decode('utf-8').splitlines()
            #rootFiles = [r for r in rootFiles if '.root' in r]
            rootFiles = [ f for f in glob.glob(f"/eos/uscms/{target_dir}/**/*.root", recursive=True) ]
            fileDirs = ["/".join(f.split("/")[:-1])+"/" for f in rootFiles]
            fileDirs = [ d.split("/eos/uscms/")[-1] for d in fileDirs]
            fileDirs = list(set(fileDirs)) # list of unique file directories
            info = {}
            if skimmed:
                info["name"] = subsample.replace("output_","")
            else:
                info["name"] = f"{samp}_{subsample}"
            info["location"] = fileDirs[0] if len(fileDirs) == 1 else fileDirs
            info["sum_wgt"] = 0.0
            info["type"] = "data"
            info["year"] = int(year)
            info["xsec"] = 0.0
            nFiles=0
            for fdir in fileDirs:
                nFiles += len([rf.name for rf in xrdClient.dirlist(fdir)[1] if '.root' in rf.name])
            info["nFiles"] = nFiles
    
            if skimmed:
                samp_name = subsample.replace("output_","")
                info['num_events'] = ref_info[samp_name]['num_events']
                info['blacklist'] = ref_info[samp_name]['blacklist']
            output.append(info)

    if skimmed:
        out_json = "skimmed_data_{0}_{1}.json".format(year,name)
    else:
        out_json = "data_{0}_{1}.json".format(year,name)
    with open(out_json,"w") as outfile:
        json.dump(output,outfile,indent=4)
