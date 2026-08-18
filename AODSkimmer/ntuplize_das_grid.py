import os
import json
import argparse

XROOTD_REDIRECTOR = "root://cmsxrootd.fnal.gov/"

def query_das_files(dataset):
    query = f"dasgoclient --query='file dataset={dataset}' -json"
    entries = json.loads(os.popen(query).read())
    files = []
    for entry in entries:
        for f in entry['file']:
            files.append(XROOTD_REDIRECTOR + f['name'])
    return files

def parse_year_from_filename(json_path):
    # e.g. fileLists/background/AOD/bkg_2018.json -> 2018
    base = os.path.splitext(os.path.basename(json_path))[0]
    return base.rsplit("_", 1)[-1]

def main():
    parser = argparse.ArgumentParser(description="Run the AODSkimmer ntuplizer over datasets queried from DAS")
    parser.add_argument("json", help="Path to a sample config JSON: {top_key: {subsample: DAS dataset}}")
    parser.add_argument("key", help="Top-level key in the JSON to process")
    parser.add_argument("--year", default=None, help="Data/MC year (default: parsed from the JSON filename, e.g. bkg_2018.json -> 2018)")
    parser.add_argument("--data", action="store_true", help="Run on data instead of MC")
    parser.add_argument("--signal", action="store_true", help="Run in signal mode")
    parser.add_argument("--vers", default="Aug2026NoID", help="Version tag used in output filenames")
    parser.add_argument("--outdir", default=None, help="Output directory for ntuples (default: ntuples/background/<year>)")
    parser.add_argument("--flistdir", default=None, help="Directory to write DAS-queried file lists (default: fileLists/background/AOD/fileLists/<year>)")
    parser.add_argument("--n-files", type=int, default=-1, help="Limit the number of files ntuplized per dataset (default: -1, i.e. all files)")
    parser.add_argument("--dry-run", action="store_true", help="Print cmsRun commands without executing them")
    args = parser.parse_args()

    year = args.year or parse_year_from_filename(args.json)

    with open(args.json, "r") as fin:
        samples = json.load(fin)

    if args.key not in samples:
        raise KeyError(f"'{args.key}' not found in {args.json}. Available keys: {list(samples.keys())}")

    subsamples = samples[args.key]

    outdir = args.outdir or f"ntuples/background/{year}"
    flistdir = args.flistdir or f"fileLists/background/MINIAOD/fileLists/{year}"
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(flistdir, exist_ok=True)

    for subsample, dataset in subsamples.items():
        if subsample.startswith("_"):
            continue
        print(f"Querying DAS for {dataset}")
        files = query_das_files(dataset)
        if not files:
            print(f"WARNING: no files found for {dataset}, skipping")
            continue

        if args.n_files >= 0:
            files = files[:args.n_files]

        flist = f"{flistdir}/{args.key}_{subsample}.txt"
        with open(flist, "w") as fout:
            fout.write("\n".join(files) + "\n")

        outfile = f"{outdir}/iDMe_run3_ntuples_{args.vers}_{args.key}_{subsample}.root"
        cmd = (
            f'cmsRun scripts/ElectronNtuplizer_cfg.py year={year} '
            f'data={int(args.data)} signal={int(args.signal)} flist="{flist}" '
            f'outfile={outfile} selectionMode="hlt"'
        )
        print(cmd)
        if not args.dry_run:
            os.system(cmd)

if __name__ == "__main__":
    main()
