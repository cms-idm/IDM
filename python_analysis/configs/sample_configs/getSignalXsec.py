import pandas as pd
import sys
import json
import os
import re
import subprocess
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("inputJson")
parser.add_argument("--xsec_csv", default=None,
                     help="Path to the xsec/filter_eff table produced by AODSkimmer/calcAuxilliaryData.py "
                          "(e.g. bkg_2024_xsec_filtereff_table.csv). Only used for --mode bkg; "
                          "defaults to a file named '<bkg json basename>_xsec_filtereff_table.csv' "
                          "next to inputJson, falling back to bkg_xsecs.json if not found.")
args = parser.parse_args()

inputJson = args.inputJson
kind = 'sig' if 'signal' in inputJson else 'bkg'
if kind != "sig" and kind != "bkg":
    print("Bad kind input ",kind)
    print("Use 'sig' or 'bkg'")

with open(inputJson) as f:
    samples = json.load(f)

if kind == 'sig':
    # enable acrobert workflow
    unified_csv = False
    reporoot = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    csvpath = os.path.join(reporoot, "python_analysis/configs/sample_configs/signal_r3xsec_filtereff_table.csv")
    if os.path.isfile(csvpath):
        df = pd.read_csv(csvpath)
        unified_csv = True
    else:        
        df = pd.read_csv('/uscms_data/d3/sbrightt/iDMe/signal_xsec/condor/signal_xsec_table.csv')
        with open("filter_effs_simple.json","r") as fin:
            effs = json.load(fin)
    for samp in samples:
        mchi = samp["Mchi"]
        dmchi = samp["dMchi"]
        ct = samp["ctau"]
        aD = str(samp['alphaD'])
        delta = dmchi/(mchi-dmchi/2)
        
        if delta == 0.1 and not unified_csv: # just use scaled 100 mm xsecs for delta = 0.1
            sel_row = df[(df["Mchi"] == mchi) & (df["dMchi"] == dmchi) & (df["ct"] == 100) & (df["alphaD"] == aD) & (df['mA/m1'] == 3)]
        else:
            sel_row = df[(df["Mchi"] == mchi) & (df["dMchi"] == dmchi) & (df["ct"] == ct) & (df["alphaD"] == aD) & (df['mA/m1'] == 3)]
        filter_eff = None
        if unified_csv:
            if not sel_row.empty:
                filter_eff = sel_row['filter_eff'].iloc[0]
                if isinstance(filter_eff, str):
                    # some rows carry a trailing "# old" note in the csv cell itself
                    filter_eff = float(filter_eff.split('#')[0].strip())
                else:
                    filter_eff = float(filter_eff)
        else:
            for eff in effs:
                if eff['mchi'] == mchi and eff['dmchi'] == dmchi and eff['ct'] == ct:
                    filter_eff = eff["feff"]
        if filter_eff is None:
            print(f"Couldn't find a filter eff for {samp['name']}, setting to 1")
            filter_eff = 1
        if sel_row.empty:
            print(f"No xsec found for {samp['name']}")
            samp["xsec"] = 0.0
            samp['filter_eff'] = filter_eff
        else:
            scale = 100/ct if (delta==0.1 and not unified_csv) else 1.0
            samp["xsec"] = scale*float(sel_row["xsec(pb)"].iloc[0]) * 1000 # convert to fb
            samp['filter_eff'] = filter_eff

if kind == 'bkg':
    # AODSkimmer/calcAuxilliaryData.py writes one row per (sample, subsample) with
    # columns sample, subsample, dataset, xsec(pb), filter_eff. The bkg config jsons
    # produced by makeSignalConfigs.py name each entry "{sample}_{subsample}", so we
    # match on that.
    if args.xsec_csv:
        csvpath = args.xsec_csv
    else:
        yearMatch = re.search(r"(\d{4})", os.path.basename(inputJson))
        csvpath = None
        if yearMatch:
            guess = f"bkg_{yearMatch.group(1)}_xsec_filtereff_table.csv"
            csvpath = os.path.join(os.path.dirname(os.path.abspath(inputJson)), guess)

    unified_csv = csvpath is not None and os.path.isfile(csvpath)
    if unified_csv:
        df = pd.read_csv(csvpath)
        df["full_name"] = df["sample"] + "_" + df["subsample"]
    else:
        with open("bkg_xsecs.json","r") as fin:
            xsecs = json.load(fin)

    for samp in samples:
        name = samp['name']
        if unified_csv:
            sel_row = df[df["full_name"] == name]
            if sel_row.empty:
                print(f"No xsec found for sample {name}")
                samp["xsec"] = 0.0
                samp["filter_eff"] = 1.0
            else:
                samp["xsec"] = float(sel_row["xsec(pb)"].iloc[0]) * 1000 # convert to fb
                samp["filter_eff"] = float(sel_row["filter_eff"].iloc[0])
        else:
            if name not in list(xsecs.keys()):
                print(f"No xsec for sample {name}")
                samp["xsec"] = 0.0
            else:
                samp["xsec"] = xsecs[name]

class CustomEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, pd.Series):
            return o.tolist()
        if isinstance(o, pd.DataFrame):
            return o.to_dict(orient="list")
        return super().default(o)

with open(inputJson, "w") as f:
    json.dump(samples, f, indent=4, cls=CustomEncoder)
