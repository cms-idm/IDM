import pandas as pd
import sys
import os
import json
import argparse
import subprocess

XROOTD_REDIRECTOR = "root://cmsxrootd.fnal.gov/"

DEFAULT_NFILES = 5

def truncate_flist(flist, nfiles):
    with open(flist, "r") as fin:
        lines = [l for l in fin if l.strip()]

    base, ext = os.path.splitext(flist)
    trunc_flist = f"{base}_first{nfiles}{ext}"
    with open(trunc_flist, "w") as fout:
        fout.writelines(l if l.endswith("\n") else l + "\n" for l in lines[:nfiles])
    return trunc_flist

def stringfy_friendly(num):
    if isinstance(num, int):
        return str(num)
    elif isinstance(num, float):
        if int(num*1000) > 0:
            num = round(num, 3)
            return str(num).replace('.', 'p') if '.' in str(num) else str(num)
        else:
            num = '%.3e' % num
            return num.replace('.', 'p')
    else:
        raise ValueError("{0} is not a number!".format(num))

def cmsRunScript(flist, script):

    # pass the file list as a path rather than expanding it into individual
    # inputFiles= arguments, which can exceed the shell's argument length limit
    # for large samples
    command = 'cmsRun {} flist={} maxEvents=-1 2>&1'.format(script, flist)
    result = subprocess.check_output(command, shell=True, text=True).split('\n')
    return result


def calculate_xsec(flist, nfiles=DEFAULT_NFILES):

    flist = truncate_flist(flist, nfiles)
    result = cmsRunScript(flist, 'scripts/genXSecAnalyzer_cfg.py')
    for line in result:
        if 'final cross section' in line:
            elements = line.split(' ')
            return float(elements[6])

def calculate_filter_eff(flist, nfiles=DEFAULT_NFILES):

    flist = truncate_flist(flist, nfiles)
    result = cmsRunScript(flist, 'scripts/genFilterEfficiency_cfg.py')
    for line in result:
        if 'Filter efficiency' in line:
            elements = line.split(' ')
            return float(elements[3])


def query_das_files(dataset):
    query = f"dasgoclient --query='file dataset={dataset}' -json"
    entries = json.loads(os.popen(query).read())
    files = []
    for entry in entries:
        for f in entry['file']:
            files.append(XROOTD_REDIRECTOR + f['name'])
    return files

def parse_year_from_filename(json_path):
    # e.g. fileLists/background/MINIAOD/bkg_2024.json -> 2024
    base = os.path.splitext(os.path.basename(json_path))[0]
    return base.rsplit("_", 1)[-1]

def get_or_make_flist(key, subsample, dataset, year):
    flistdir = f"fileLists/background/MINIAOD/fileLists/{year}"
    os.makedirs(flistdir, exist_ok=True)
    flist = f"{flistdir}/{key}_{subsample}_flist.txt"

    if os.path.exists(flist):
        return flist

    print(f"Querying DAS for {dataset}")
    files = query_das_files(dataset)
    if not files:
        print(f"WARNING: no files found for {dataset}, skipping")
        return None

    with open(flist, "w") as fout:
        fout.write("\n".join(files) + "\n")
    print(f"Wrote {len(files)} files to {flist}")
    return flist

def run_background(json_path, year=None, key=None, nfiles=DEFAULT_NFILES):

    year = year or parse_year_from_filename(json_path)

    with open(json_path, "r") as fin:
        samples = json.load(fin)

    if key is not None and key not in samples:
        raise KeyError(f"'{key}' not found in {json_path}. Available keys: {list(samples.keys())}")

    keys = [key] if key is not None else list(samples.keys())

    col_key = []
    col_subsample = []
    col_dataset = []
    col_xs = []
    col_eff = []

    for k in keys:
        for subsample, dataset in samples[k].items():
            if subsample.startswith("_"):
                continue

            flist = get_or_make_flist(k, subsample, dataset, year)
            if flist is None:
                continue

            xsec = calculate_xsec(flist, nfiles)

            try:
                fileff = calculate_filter_eff(flist, nfiles)
            except subprocess.CalledProcessError:
                print(f"WARNING: no GenFilterInfo found for {k}/{subsample}, assuming filter_eff=1.0")
                fileff = 1.0

            col_key.append(k); col_subsample.append(subsample); col_dataset.append(dataset)
            col_xs.append(xsec); col_eff.append(fileff)

            print(k, subsample, xsec, fileff)

    data = {"sample": col_key, "subsample": col_subsample, "dataset": col_dataset,
            "xsec(pb)": col_xs, "filter_eff": col_eff}
    df = pd.DataFrame(data)

    outname = os.path.splitext(os.path.basename(json_path))[0] + "_xsec_filtereff_table.csv"
    df.to_csv(outname, index=False)
    print(f"Saved {outname}")

def run_signal(nfiles=DEFAULT_NFILES):

    vers = 'Feb2026'

    m1l = [0.05, 0.5, 1, 2, 5, 50]
    dml = [0.05, 0.1, 0.2]
    ctaul = [1, 10, 100]

    # columns
    col_m1 = []
    col_m2 = []
    col_dm = []
    col_mzd = []
    col_ctau = []
    col_mchi = []
    col_dmchi = []

    col_xs = []
    col_eff = []

    for m1 in m1l:
        for dm in dml:
            for ctau in ctaul:

                m2 = round(m1*(1+dm), 5)
                mchi = round((m1+m2)/2, 5)
                dmchi = round(m2-m1, 5)
                mzd = round(3 * m1, 5)

                col_m1.append(m1); col_m2.append(m2); col_dm.append(dm); col_mzd.append(mzd)
                col_mchi.append(mchi); col_dmchi.append(dmchi); col_ctau.append(ctau)

                stfm1 = stringfy_friendly(m1)
                stfdm = stringfy_friendly(dm)
                stfmzd = stringfy_friendly(mzd)

                flist = f'fileLists/signal/2024/M1-{stfm1}_dM-{stfdm}_mZD-{stfmzd}_ctau-{ctau}_flist.txt'

                xsec = calculate_xsec(flist, nfiles)
                col_xs.append(xsec)
                fileff = calculate_filter_eff(flist, nfiles)
                col_eff.append(fileff)

                print(m1, dm, ctau, xsec, fileff)


    #df[(df["Mchi"] == mchi) & (df["dMchi"] == dmchi) & (df["ct"] == 100) & (df["alphaD"] == aD) & (df['mA/m1'] == 3)]

    # Create a sample DataFrame
    data = {"M1": col_m1, "M2": col_m2, "dM": col_dm, "mZD": col_mzd,
            "Mchi": col_mchi, "dMchi": col_dmchi, "ct": col_ctau,
            "alphaD": ["aEM" for m in col_m1], "mA/m1": [3 for m in col_m1],
            "xsec(pb)": col_xs, "filter_eff": col_eff}
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv('signal_xsec_filtereff_table.csv', index=False)

def main():
    parser = argparse.ArgumentParser(description="Compute cross sections and filter efficiencies for signal or background MC samples")
    parser.add_argument("--mode", choices=["sig", "bkg"], default="signal",
                         help="Compute for the hardcoded signal grid, or for a background sample JSON (default: signal)")
    parser.add_argument("--json", default=None,
                         help="Path to a background sample JSON, e.g. fileLists/background/MINIAOD/bkg_2024.json (required for --mode background)")
    parser.add_argument("--year", default=None,
                         help="MC year (default: parsed from the JSON filename, e.g. bkg_2024.json -> 2024)")
    parser.add_argument("--key", default=None,
                         help="Only process this top-level key in the background JSON, e.g. QCD (default: all keys)")
    parser.add_argument("--nfiles", type=int, default=DEFAULT_NFILES,
                         help="Only run xsec/filter-eff calculations over the first N files of each "
                              "dataset's file list, instead of the full dataset (default: %(default)s)")
    args = parser.parse_args()

    if args.mode == "bkg":
        if not args.json:
            parser.error("--json is required for --mode background")
        run_background(args.json, year=args.year, key=args.key, nfiles=args.nfiles)
    else:
        run_signal(nfiles=args.nfiles)

if __name__ == "__main__":
    main()
