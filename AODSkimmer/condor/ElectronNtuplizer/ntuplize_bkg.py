import os
import json
import argparse

XROOTD_REDIRECTOR = "root://cmsxrootd.fnal.gov/"

CMSSW = 'CMSSW_14_0_21'
compiled_CMSSW_envs = 'ntuplizer_CMSSW_14_0_21_acrobert.tar.gz'

def query_das_files(dataset):
    query = f"dasgoclient --query='file dataset={dataset}' -json"
    entries = json.loads(os.popen(query).read())
    files = []
    for entry in entries:
        for f in entry['file']:
            files.append(XROOTD_REDIRECTOR + f['name'])
    return files

def parse_year_from_filename(json_path):
    # e.g. ../../fileLists/background/MINIAOD/bkg_2024.json -> 2024
    base = os.path.splitext(os.path.basename(json_path))[0]
    return base.rsplit("_", 1)[-1]

def main():
    parser = argparse.ArgumentParser(description="Query background MC datasets from DAS and submit ElectronNtuplizer condor jobs")
    parser.add_argument("key", help="Top-level key in the background JSON to process, e.g. QCD")
    parser.add_argument("--json", default="../../fileLists/background/MINIAOD/bkg_2024.json", help="Path to the background sample config JSON: {top_key: {subsample: DAS dataset}}")
    parser.add_argument("--subkey", default=None, help="Only process this subsample under 'key', e.g. --subkey WWZ")
    parser.add_argument("--year", default=None, help="MC year (default: parsed from the JSON filename, e.g. bkg_2024.json -> 2024)")
    parser.add_argument("--vers", default="Aug2026noIDSep", help="Version tag used in the output EOS directory (background_<vers>) and split file list names")
    parser.add_argument("--nsplit", type=int, default=10, help="Number of input files per condor job")
    parser.add_argument("--nthreads", type=int, default=4, help="Number of threads per condor job")
    parser.add_argument("--dry-run", action="store_true", help="Query DAS and write file lists, but don't submit condor jobs")
    args = parser.parse_args()

    year = args.year or parse_year_from_filename(args.json)

    with open(args.json, "r") as fin:
        samples = json.load(fin)

    if args.key not in samples:
        raise KeyError(f"'{args.key}' not found in {args.json}. Available keys: {list(samples.keys())}")

    subsamples = samples[args.key]

    if args.subkey is not None:
        if args.subkey not in subsamples:
            raise KeyError(f"'{args.subkey}' not found under '{args.key}' in {args.json}. Available subkeys: {list(subsamples.keys())}")
        subsamples = {args.subkey: subsamples[args.subkey]}

    flistdir = f"../../fileLists/background/MINIAOD/fileLists/{year}"
    os.makedirs(flistdir, exist_ok=True)

    for subsample, dataset in subsamples.items():
        if subsample.startswith("_comment"):
            continue
        print(f"Querying DAS for {dataset}")
        files = query_das_files(dataset)
        if not files:
            print(f"WARNING: no files found for {dataset}, skipping")
            continue

        flist = f"{flistdir}/{args.key}_{subsample}_flist.txt"
        with open(flist, "w") as fout:
            fout.write("\n".join(files) + "\n")
        print(f"Wrote {len(files)} files to {flist}")

        outDirName = f"{args.key}/{subsample}"
        cmd = (
            f'source submit_ElectronNtuplizer_condor.sh {flist} {year} {args.nthreads} '
            f'False False {args.vers} {args.nsplit} {CMSSW} {compiled_CMSSW_envs} background {outDirName}'
        )
        print(cmd)
        if not args.dry_run:
            os.system(f'bash -c "{cmd}"')

if __name__ == "__main__":
    main()
