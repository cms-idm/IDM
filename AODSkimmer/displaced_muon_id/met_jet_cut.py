#!/usr/bin/env python3
"""Event-level MET > 80 GeV and leading-jet pT > 80 GeV, |eta| < 2.4 on top of the reco-only dimuon selection, signal and
background (kb/studies/bkg-reco-dimuons.md Part 2; LPC only).

python3 met_jet_cut.py scan --flist-dir DIR --out-dir RUN/scan [--workers 6]
python3 met_jet_cut.py analyze --signal-summary recovery.root --signal-scan-dir SIG_SCAN --bkg-summary bkg_reco_dimuons.root \
    --bkg-scan-dir RUN/scan --output OUT.root

scan: FWLite pass over background MINIAOD; per event (run, lumi, event), Type-1 PF MET (slimmedMETs[0] corPt/corPhi Type1)
and the highest-pT slimmedJets jet (AK4 CHS as stored, no ID, no cleaning), identical to signal_trigger_eff.py scan_sample
(whose signal scan trees are reused). analyze: joins the summary event trees (sample index resolved through each summary's
own 'inputs' metadata), the producer outputs listed there (pfMetPt) and the scan trees one-to-one on (run, lumi, event);
cut = pfMetPt > 80 and leading jet pT > 80 and |eta| < 2.4 (no jet fails). Raw unweighted event counts.
"""
import argparse
import concurrent.futures
from collections import Counter, defaultdict
from pathlib import Path
import re
import time
import traceback

import numpy as np

MET_CUT, JET_PT_CUT, JET_ETA_CUT = 80., 80., 2.4


def scan_sample(job):
    import ROOT
    ROOT.gROOT.SetBatch(True)
    ROOT.gSystem.Load("libFWCoreFWLite")
    ROOT.FWLiteEnabler.enable()
    from DataFormats.FWLite import Events, Handle
    sample, files, outdir = job
    hm, hj = Handle("std::vector<pat::MET>"), Handle("std::vector<pat::Jet>")
    cols = {k: [] for k in ["run", "lumi", "event", "pfMet", "pfMetPhi", "nJet", "jetPt", "jetEta", "jetPhi"]}
    processed = []
    try:
        for filename in files:
            events = Events(filename)
            expected, seen = int(events.size()), 0
            for event in events:
                event.getByLabel("slimmedMETs", hm)
                event.getByLabel("slimmedJets", hj)
                if not hm.isValid() or not hj.isValid() or hm.product().size() < 1:
                    raise RuntimeError("Missing slimmedMETs/slimmedJets: " + filename)
                met, jets = hm.product()[0], hj.product()
                aux = event.object().id()
                cols["run"].append(aux.run())
                cols["lumi"].append(aux.luminosityBlock())
                cols["event"].append(aux.event())
                cols["pfMet"].append(met.corPt(ROOT.pat.MET.Type1))
                cols["pfMetPhi"].append(met.corPhi(ROOT.pat.MET.Type1))
                n = jets.size()
                cols["nJet"].append(n)
                lead = max(range(n), key=lambda i: jets[i].pt()) if n else -1
                cols["jetPt"].append(jets[lead].pt() if n else np.nan)
                cols["jetEta"].append(jets[lead].eta() if n else np.nan)
                cols["jetPhi"].append(jets[lead].phi() if n else np.nan)
                seen += 1
            if seen != expected or seen == 0:
                raise RuntimeError("Incomplete/empty input: %s: %d/%d" % (filename, seen, expected))
            processed.append((filename, seen))
        import uproot
        out = Path(outdir) / (sample + ".root")
        if out.exists():
            raise RuntimeError("Refusing to overwrite " + str(out))
        tree = {k: np.asarray(v, dtype=np.int64 if k in ("run", "lumi", "event", "nJet") else np.float64)
                for k, v in cols.items()}
        with uproot.recreate(out) as f:
            f["scan"] = tree
            f["inputs"] = "\n".join("%s\t%d" % p for p in processed)
        return sample, processed, None
    except Exception:
        return sample, processed, traceback.format_exc()


def scan(args):
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for flist in sorted(Path(args.flist_dir).glob("*.txt")):
        files = [l.strip() for l in flist.read_text().splitlines() if l.strip() and not l.lstrip().startswith("#")]
        jobs.append((flist.stem, files, str(outdir)))
    start = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        res = list(pool.map(scan_sample, jobs))
    for (sample, files, _), r in zip(jobs, res):
        ok = not r[2] and [f for f, _ in r[1]] == files
        print("SAMPLE %s files=%d/%d events=%d %s" % (sample, len(r[1]), len(files), sum(n for _, n in r[1]),
                                                       "OK" if ok else "FAIL"), flush=True)
        if r[2]:
            print(r[2], flush=True)
    failed = [r[0] for (s, files, _), r in zip(jobs, res) if r[2] or [f for f, _ in r[1]] != files]
    if failed:
        raise RuntimeError("Failed samples: %r" % failed)
    print("SCAN COMPLETE samples=%d events=%d elapsed=%.1fs" % (len(res), sum(n for r in res for _, n in r[1]),
                                                                time.monotonic() - start))


def keyed(arrays, what):
    keys = list(zip(arrays["run"].tolist(), arrays["lumi"].tolist(), arrays["event"].tolist()))
    index = {k: i for i, k in enumerate(keys)}
    if len(index) != len(keys):
        raise RuntimeError("Duplicate (run, lumi, event) in " + what)
    return keys, index


def load(summary, scan_dir, signal):
    """Per summary event row: sample name, MET, jet pass, selected (and T, recovered for signal)."""
    import uproot
    with uproot.open(summary) as f:
        inputs = [l.split() for l in str(f["inputs"].member("fTitle")).splitlines() if l.strip()]
        fields = ["sample", "run", "lumi", "event", "nPassing"] + (["T", "nCorrect"] if signal else [])
        ev = f["events"].arrays(fields, library="np")
    rows = []
    for index, (name, producer) in enumerate(inputs):
        sel = ev["sample"] == index
        s = {k: v[sel] for k, v in ev.items()}
        skeys, _ = keyed(s, "summary " + name)
        with uproot.open(producer) as f:
            p = f["preselectedVertexing/events"].arrays(["run", "lumi", "event", "pfMetPt"], library="np")
            if str(f["preselectedVertexing/sample"].member("fTitle")) != name:
                raise RuntimeError("Producer sample mismatch: " + producer)
        with uproot.open(Path(scan_dir) / (name + ".root")) as f:
            j = f["scan"].arrays(["run", "lumi", "event", "pfMet", "nJet", "jetPt", "jetEta"], library="np")
        _, pindex = keyed(p, "producer " + name)
        _, jindex = keyed(j, "scan " + name)
        if not (len(skeys) == len(pindex) == len(jindex)) or any(k not in pindex or k not in jindex for k in skeys):
            raise RuntimeError("Incomplete join for %s: summary %d, producer %d, scan %d"
                               % (name, len(skeys), len(pindex), len(jindex)))
        pi = np.array([pindex[k] for k in skeys])
        ji = np.array([jindex[k] for k in skeys])
        met = p["pfMetPt"][pi].astype(np.float64)
        dmet = np.abs(j["pfMet"][ji] - met)
        if not np.all(np.isfinite(dmet)) or dmet.max() > 1e-3:
            raise RuntimeError("PF MET mismatch scan vs producer in " + name)
        jet_pt, jet_eta = j["jetPt"][ji], j["jetEta"][ji]
        with np.errstate(invalid="ignore"):
            met_ok = met > MET_CUT
            jet_ok = (jet_pt > JET_PT_CUT) & (np.abs(jet_eta) < JET_ETA_CUT)  # NaN (no jet) fails
        out = dict(name=name, met=met_ok, jet=jet_ok, noJet=j["nJet"][ji] == 0, selected=s["nPassing"] > 0)
        if signal:
            out["T"] = s["T"].astype(bool)
            out["recovered"] = s["nCorrect"] > 0
        rows.append(out)
    if sum(len(r["met"]) for r in rows) != len(ev["sample"]):
        raise RuntimeError("Summary rows not all assigned to an inputs entry: " + str(summary))
    return rows


def counts(r):
    cut = r["met"] & r["jet"]
    c = Counter(events=len(cut), met=r["met"].sum(), jet=r["jet"].sum(), met_and_jet=cut.sum(), no_jet=r["noJet"].sum(),
                selected=r["selected"].sum(), selected_met_and_jet=(r["selected"] & cut).sum())
    if "T" in r:
        c.update(T=r["T"].sum(), T_met_and_jet=(r["T"] & cut).sum(), recovered=r["recovered"].sum(),
                 recovered_met_and_jet=(r["recovered"] & cut).sum())
    return Counter({k: int(v) for k, v in c.items()})


def ratio(n, d):
    return "undefined (0/0)" if not d else "%.4f%% (%d/%d)" % (100 * n / d, n, d)


def analyze(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    import ROOT
    groups = defaultdict(Counter)
    for kind, summary, scan_dir in (("signal", args.signal_summary, args.signal_scan_dir),
                                    ("background", args.bkg_summary, args.bkg_scan_dir)):
        for r in load(summary, scan_dir, kind == "signal"):
            c = counts(r)
            if kind == "signal":
                m = re.fullmatch(r"(Mchi-[\dp]+_dMchi-[\dp]+)_(ctau-[\dp]+)", r["name"])
                if not m:
                    raise RuntimeError("Unexpected signal sample " + r["name"])
                names = ("overall", "mass/" + m[1], "lifetime/" + m[2], "sample/" + r["name"])
            else:
                names = ("overall", "group/" + r["name"].split("_")[0], "sample/" + r["name"])
            for g in names:
                groups[kind + "/" + g].update(c)
    root = ROOT.TFile.Open(str(output), "CREATE")
    if not root or root.IsZombie():
        raise RuntimeError("Cannot create output")
    ROOT.TNamed("definitions", "Raw unweighted event counts. Cut = Type-1 PF MET (slimmedMETs[0] corPt) > 80 GeV and highest-pT "
                "slimmedJets jet (AK4 CHS as stored, no ID, no cleaning) pT > 80 GeV and |eta| < 2.4; no jet fails. "
                "selected = >=1 dimuon pair passing the reco-only chain (unchanged); recovered = nCorrect > 0 (signal T events). "
                "Inputs: %s, %s, scans %s, %s" % (args.signal_summary, args.bkg_summary, args.signal_scan_dir,
                                                  args.bkg_scan_dir)).Write()
    for g, c in sorted(groups.items()):
        directory = root
        for part in g.split("/"):
            directory = directory.GetDirectory(part) or directory.mkdir(part)
        directory.cd()
        keys = sorted(c)
        h = ROOT.TH1D("counts", "Raw event counts;Counter;Events", len(keys), 0, len(keys))
        for i, k in enumerate(keys, 1):
            h.GetXaxis().SetBinLabel(i, k)
            h.SetBinContent(i, c[k])
        h.Write()
    root.Close()
    order = lambda g: (not g[0].startswith("signal"), g[0].split("/")[1] != "overall", g[0])
    for g, c in sorted(groups.items(), key=order):
        line = "%-58s cut/all %s | selected/all %s -> with cut %s | cut on selected %s | no jet %d" % (
            g, ratio(c["met_and_jet"], c["events"]), ratio(c["selected"], c["events"]),
            ratio(c["selected_met_and_jet"], c["events"]), ratio(c["selected_met_and_jet"], c["selected"]), c["no_jet"])
        if "T" in c:
            line += " | T recovery %s -> with cut %s" % (ratio(c["recovered"], c["T"]), ratio(c["recovered_met_and_jet"], c["T"]))
        print(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--flist-dir", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--workers", type=int, default=6)
    a = sub.add_parser("analyze")
    for k in ("--signal-summary", "--signal-scan-dir", "--bkg-summary", "--bkg-scan-dir", "--output"):
        a.add_argument(k, required=True)
    args = parser.parse_args()
    (scan if args.cmd == "scan" else analyze)(args)


if __name__ == "__main__":
    main()
