import pandas as pd
import sys
import json
import os
import subprocess

inputJson = sys.argv[1]
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
    csvpath = os.path.join(reporoot, "python_analysis/configs/sample_configs/signal_xsec_filtereff_table.csv")
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
            filter_eff = sel_row['filter_eff'].item()
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
    with open("bkg_xsecs.json","r") as fin:
        xsecs = json.load(fin)
    for samp in samples:
        name = samp['name']
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
