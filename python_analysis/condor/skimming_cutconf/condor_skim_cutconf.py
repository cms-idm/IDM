#!/usr/bin/env python3
"""
Worker-side skimmer: copies to a new ntuple only the events that pass a cut
config up to (and including) a given cut.

For each input file in the job (read directly over xrootd, no local copy):
  1. evaluate the cut config's cut1..cutN on the raw NanoEvents (coffea, lazily,
     in chunks -- only the branches the cuts use are read) and record the entry
     numbers of the passing events
  2. copy exactly those entries of ntuples/outT, byte-for-byte with all branches,
     into a local partial file (ROOT TEntryList + CopyTree)
If either step fails (e.g. a dropped xrootd connection) the whole file is retried.
Then the partial files are hadd-ed into one output, skim metadata is added, and
the result is xrdcp-ed to EOS.

No Analyzer pre-processing is run (no computeExtraVariables, good-vertex
definitions, event weights, ...), so the cuts must only use raw ntuple branches.

Usage: python3 condor_skim_cutconf.py job.json [--local-out DIR] [--max-files N]
  --local-out DIR : write the final file to DIR instead of EOS (for testing)
  --max-files N   : only process the first N input files (for testing)
"""
import os
import re
import sys
import json
import time
import argparse
import importlib
import subprocess
import traceback

import numpy as np

TREE = "ntuples/outT"
RETRIES = 3


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd, retries=1):
    for attempt in range(1, retries + 1):
        log(f"$ {cmd}" + (f"   (attempt {attempt}/{retries})" if retries > 1 else ""))
        if subprocess.run(cmd, shell=True).returncode == 0:
            return
        time.sleep(10 * attempt)
    raise RuntimeError(f"command failed after {retries} attempt(s): {cmd}")


def load_cuts(cut_file, last_cut):
    """Import the cut config and return its cut1..last_cut functions in order."""
    mod = importlib.import_module(os.path.splitext(os.path.basename(cut_file))[0])
    names = sorted([c for c in dir(mod) if re.fullmatch(r"cut\d+", c)], key=lambda c: int(c[3:]))
    if last_cut not in names:
        raise ValueError(f"{last_cut} not found in {cut_file} (has {names})")
    names = names[:names.index(last_cut) + 1]
    return names, [getattr(mod, n) for n in names]


def passing_entries(path, info, cut_names, cuts, chunksize):
    """Run the cuts over one file in chunks.

    Returns (n_entries, sorted passing entry numbers, per-cut stats) where stats maps
    'all'/cutN -> [raw count, sum of genWgt (event count for data)].
    """
    import awkward as ak
    import uproot
    from coffea.nanoevents import NanoEventsFactory
    from analysisTools.mySchema_newCoffea import MySchema

    with uproot.open(path, timeout=180) as f:
        n_entries = f[TREE].num_entries
    stats = {k: [0, 0.0] for k in ["all"] + cut_names}
    passed = []
    for start in range(0, n_entries, chunksize):
        stop = min(start + chunksize, n_entries)
        events = NanoEventsFactory.from_root({path: TREE}, schemaclass=MySchema, mode="virtual",
                                             entry_start=start, entry_stop=stop,
                                             uproot_options={"timeout": 180}).events()
        events["skimEntry"] = ak.Array(np.arange(start, stop, dtype=np.int64))
        has_wgt = "genWgt" in events.fields

        stats["all"][0] += len(events)
        stats["all"][1] += float(ak.sum(events.genWgt)) if has_wgt else len(events)
        for name, cut in zip(cut_names, cuts):
            events, _, _, _ = cut(events, info)
            stats[name][0] += len(events)
            stats[name][1] += float(ak.sum(events.genWgt)) if has_wgt else len(events)
        passed.append(ak.to_numpy(events.skimEntry))
    entries = np.sort(np.concatenate(passed)) if passed else np.array([], dtype=np.int64)
    return n_entries, entries, stats


def copy_entries(in_path, entries, out_path):
    """Copy the given entries of ntuples/outT into out_path, keeping the ntuples/ directory."""
    import ROOT
    fin = ROOT.TFile.Open(in_path)
    if not fin or fin.IsZombie():
        raise OSError(f"could not open {in_path}")
    tree = fin.Get(TREE)
    elist = ROOT.TEntryList("skimList", "skimList", tree)
    for e in entries:
        elist.Enter(int(e))
    tree.SetEntryList(elist)

    fout = ROOT.TFile(out_path, "RECREATE")
    fout.mkdir("ntuples").cd()
    out_tree = tree.CopyTree("")
    n_out = out_tree.GetEntries()
    out_tree.Write("", ROOT.TObject.kOverwrite)
    fout.Close()
    fin.Close()
    if n_out != len(entries):
        raise RuntimeError(f"copied {n_out} entries from {in_path}, expected {len(entries)}")
    return n_out


def skim_file(remote, part, info, cut_names, cuts, chunksize):
    """Skim one file over xrootd, retrying the whole file on failure."""
    for attempt in range(1, RETRIES + 1):
        try:
            n_in, entries, stats = passing_entries(remote, info, cut_names, cuts, chunksize)
            n_pass = copy_entries(remote, entries, part)
            return n_in, n_pass, stats
        except Exception:
            log(f"  attempt {attempt}/{RETRIES} on {remote} failed:\n{traceback.format_exc()}")
            if os.path.exists(part):
                os.remove(part)
            if attempt == RETRIES:
                raise
            time.sleep(30 * attempt)


def write_metadata(out_path, job, cut_names, stats, per_file):
    """Store skim bookkeeping in the output file, outside ntuples/ so the Analyzer ignores it.

    - skimCutflow / skimCutflowWgt (TH1D): raw / genWgt-summed counts after each cut
    - skimFiles (TTree): one row per input file (hadd concatenates these when merging)
    - skimConfig (TNamed): json with the cut config, cut name and input sample json
    """
    import ROOT
    from array import array
    f = ROOT.TFile(out_path, "UPDATE")
    bins = ["all"] + cut_names
    h_n = ROOT.TH1D("skimCutflow", "raw events after each cut", len(bins), 0, len(bins))
    h_w = ROOT.TH1D("skimCutflowWgt", "sum of genWgt after each cut (event count for data)", len(bins), 0, len(bins))
    for i, b in enumerate(bins, start=1):
        h_n.GetXaxis().SetBinLabel(i, b)
        h_w.GetXaxis().SetBinLabel(i, b)
        h_n.SetBinContent(i, stats[b][0])
        h_w.SetBinContent(i, stats[b][1])
    h_n.Write()
    h_w.Write()

    t = ROOT.TTree("skimFiles", "per-input-file skim bookkeeping")
    name = ROOT.std.string()
    n_in, n_pass = array("q", [0]), array("q", [0])
    t.Branch("file", name)
    t.Branch("n_in", n_in, "n_in/L")
    t.Branch("n_pass", n_pass, "n_pass/L")
    for fname, nin, npass in per_file:
        name.replace(0, name.size(), fname)
        n_in[0], n_pass[0] = nin, npass
        t.Fill()
    t.Write()

    cfg = {k: job[k] for k in ("cut_config", "cut", "sample_json", "sample_name")}
    ROOT.TNamed("skimConfig", json.dumps(cfg)).Write()
    f.Close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job_json")
    parser.add_argument("--local-out", default=None, help="write output here instead of EOS (testing)")
    parser.add_argument("--max-files", type=int, default=-1, help="only process the first N files (testing)")
    args = parser.parse_args()

    t0 = time.time()
    with open(args.job_json) as f:
        job = json.load(f)
    info = job["sample"]
    files = job["files"][:args.max_files] if args.max_files > 0 else job["files"]

    sys.path.insert(0, os.getcwd())  # analysisTools/ and the cut config are unpacked here
    cut_names, cuts = load_cuts(job["cut_config"], job["cut"])
    log(f"sample {job['sample_name']}: {len(files)} file(s), applying {cut_names} from {job['cut_config']}")

    stats = {k: [0, 0.0] for k in ["all"] + cut_names}
    per_file = []
    parts = []
    for i, remote in enumerate(files):
        part = f"part_{i}.root"
        n_in, n_pass, file_stats = skim_file(remote, part, info, cut_names, cuts, job["chunksize"])
        for k in stats:
            stats[k][0] += file_stats[k][0]
            stats[k][1] += file_stats[k][1]
        parts.append(part)
        per_file.append((os.path.basename(remote), n_in, n_pass))
        log(f"  {os.path.basename(remote)}: {n_pass}/{n_in} pass ({n_pass / max(n_in, 1):.3%})")

    out_name = job["out_file"]
    if len(parts) == 1:
        os.rename(parts[0], out_name)
    else:
        run(f"hadd -f {out_name} {' '.join(parts)}")
        for p in parts:
            os.remove(p)
    write_metadata(out_name, job, cut_names, stats, per_file)

    tot_in, tot_pass = stats["all"][0], stats[cut_names[-1]][0]
    log(f"total: {tot_pass}/{tot_in} pass ({tot_pass / max(tot_in, 1):.3%}), "
        f"{os.path.getsize(out_name) / 1e6:.1f} MB")

    if args.local_out:
        os.makedirs(args.local_out, exist_ok=True)
        os.replace(out_name, os.path.join(args.local_out, out_name))
        log(f"wrote {os.path.join(args.local_out, out_name)}")
    else:
        run(f"xrdcp -f -s {out_name} root://cmseos.fnal.gov/{job['out_dir']}/{out_name}", retries=3)
        os.remove(out_name)
    log(f"SKIM DONE in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
