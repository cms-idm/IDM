#!/usr/bin/env python3
"""Signal-MC trigger efficiency, Run-3 analogue of AN-18-124 Section 5 and Appendix C
(study note: kb/studies/signal-trigger-efficiency.md; LPC only).

python3 signal_trigger_eff.py scan --run-dir RUN [--workers 6] [--test]
python3 signal_trigger_eff.py analyze --run-dir RUN --input-dir MET_PROXY_RUN/inputs --input-revision 5a2b10e \
    --revision <sha> --trigger-root TRIGGER_RUN/trigger_counts.root [--test]
python3 signal_trigger_eff.py plot --run-dir RUN [--log]

scan: one pass over the original 09-28 MINIAOD; per event (run, lumi, event), HLT accept bits of HLT_PATHS, the MET filters
(TriggerResults::PAT: Flag_METFilters + 8 individual flags), Type-1 PF MET (slimmedMETs[0] corPt/corPhi Type1), Calo MET (caloMETPt) and the leading slimmedJets jet
(AK4 CHS as stored, no ID). All paths and filters must be present in every event.
--test also sums packedPFCandidates |pdgId| 13 directly per event (METNoMu cross-check).
analyze: joins with the PreselectedVertexing outputs on (run, lumi, event) per sample (one-to-one). Offline METNoMu =
|PF MET + sum(px, py) of all packedPFCandidates with |pdgId| 13| (PreselectedVertexing pfMu*). Populations: AN = Flag_METFilters
and |PF MET - Calo MET| / Calo MET < 0.5 (Calo MET > 0); ANjet = AN and leading jet pT > 80 GeV, |eta| < 2.4;
chainA = AN and met_proxy chain A (G > I > DSA vertex, chi2/ndof < 4, reco dR < 0.9, |dphi(PF MET, reco pT_mumu)| < 0.5).
Efficiency = N(pop and trigger) / N(pop) per bin; raw, unweighted, event level; pooling unweighted. Gen muons: the two
signal muons (FACTS.md), leading/subleading by gen pT; max gen |d0| straight line to (0,0).
"""
import argparse
from array import array
import concurrent.futures
import glob
from pathlib import Path
import re
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "met_dphi"))
import met_proxy  # noqa: E402
import metnomu_chain_a as mca  # noqa: E402

MET = {x: "HLT_PFMET%d_PFMHT%d_IDTight_v22" % (x, x) for x in (110, 120, 130, 140)}
NOMU = {120: "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v22", 130: "HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v21",
        140: "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v21"}
MU3MET = ["HLT_Mu3er1p5_PFJet100er2p5_PFMET%d_PFMHT%d_IDTight_v4" % (x, x) for x in (70, 80, 90, 100)]
DOUBLEMU = "HLT_DoubleMu3_DCA_PFMET50_PFMHT60_v12"
HLT_PATHS = sorted(set(mca.S10 + mca.MU3 + list(MET.values()) + list(NOMU.values()) + MU3MET + [DOUBLEMU]))
TRIGGERS = {"NoMu120": [NOMU[120]], "MET120": [MET[120]],
            "ORAN": [MET[120], MET[130], MET[140], NOMU[120], NOMU[130], NOMU[140]], "S14": mca.S10 + mca.MU3,
            "NoMuOR": [NOMU[120], NOMU[130], NOMU[140]]}
TRIG_LABEL = {"NoMu120": "PFMETNoMu120_PFMHTNoMu120_IDTight", "MET120": "PFMET120_PFMHT120_IDTight",
              "ORAN": "OR PFMET + PFMETNoMu 120/130/140", "S14": "OR of 14 METNoMu paths",
              "NoMuOR": "OR PFMETNoMu 120/130/140 (AN analysis trigger)"}
TABLE_PATHS = [MET[x] for x in (110, 120, 130, 140)] + [NOMU[x] for x in (120, 130, 140)] + [DOUBLEMU] + MU3MET + mca.MU3
FILTERS = ["Flag_METFilters", "Flag_goodVertices", "Flag_globalSuperTightHalo2016Filter", "Flag_EcalDeadCellTriggerPrimitiveFilter",
           "Flag_BadPFMuonFilter", "Flag_BadPFMuonDzFilter", "Flag_hfNoisyHitsFilter", "Flag_eeBadScFilter",
           "Flag_ecalBadCalibFilter"]
AN_POINTS = {"Mchi-60p0_dMchi-20p0": "(m1, D) = (50, 20) GeV", "Mchi-52p5_dMchi-5p0": "(m1, D) = (50, 5) GeV",
             "Mchi-6p0_dMchi-2p0": "(m1, D) = (5, 2) GeV", "Mchi-5p25_dMchi-0p5": "(m1, D) = (5, 0.5) GeV"}
POPS = {"AN": "MET filters, |PF-Calo|/Calo < 0.5", "ANjet": "AN + lead jet pT > 80 GeV, |eta| < 2.4",
        "chainA": "AN + chain A (vertex, chi2, dR, dphi)"}
MET_EDGES = np.concatenate([np.arange(0, 400, 20), np.arange(400, 600, 50), np.arange(600, 1001, 100)]).astype(float)
VARS = {  # name: (edges, label)
    "metnomu": (MET_EDGES, "offline METNoMu = |PF MET + PF muons| [GeV]"),
    "pfmet": (MET_EDGES, "Type-1 PF MET [GeV]"),
    "maxD0": (met_proxy.D0_EDGES, "max gen |d0| of the two signal muons [cm]"),
    "jetPt": (np.concatenate([np.arange(0, 300, 20), np.arange(300, 601, 50)]).astype(float),
              "leading jet pT [GeV] (events with >= 1 jet)"),
    "jetEta": (np.linspace(-5, 5, 41), "leading jet eta (events with >= 1 jet)"),
    "jetPhi": (np.linspace(-np.pi, np.pi, 33), "leading jet phi [rad] (events with >= 1 jet)"),
    "mu1Pt": (np.array([0, 2, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50, 70, 100, 150.]), "leading gen signal muon pT [GeV]"),
    "mu1Eta": (np.linspace(-4, 4, 33), "leading gen signal muon eta"),
    "mu1Phi": (np.linspace(-np.pi, np.pi, 33), "leading gen signal muon phi [rad]"),
    "mu2Pt": (np.array([0, 2, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50, 70, 100, 150.]), "subleading gen signal muon pT [GeV]"),
    "mu2Eta": (np.linspace(-4, 4, 33), "subleading gen signal muon eta"),
    "mu2Phi": (np.linspace(-np.pi, np.pi, 33), "subleading gen signal muon phi [rad]"),
}
SAMPLE_VARS = ("metnomu", "pfmet")  # per-sample groups keep only these
MU1_BINS = [(0, 10), (10, 20), (20, 30), (30, 40), (40, np.inf)]  # Fig-92 analogue, leading gen signal muon pT [GeV]


def file_jobs(test):
    repo = Path(__file__).resolve().parents[2]
    jobs = []
    for src in sorted(glob.glob(str(repo / "AODSkimmer/fileLists/signal/2022EE/iDMmu/*_flist.txt"))):
        files = [l.strip() for l in Path(src).read_text().splitlines() if l.strip() and not l.lstrip().startswith("#")]
        if len(files) != 4 or any("260928_" not in p for p in files):
            raise RuntimeError("Unexpected file list " + src)
        jobs.append((Path(src).name.removesuffix("_flist.txt"), files))
    if len(jobs) != 72:
        raise RuntimeError("Expected 72 samples")
    return jobs[:1] if test else jobs


def scan_sample(job):
    import ROOT
    ROOT.gROOT.SetBatch(True)
    ROOT.gSystem.Load("libFWCoreFWLite")
    ROOT.FWLiteEnabler.enable()
    from DataFormats.FWLite import Events, Handle
    sample, files, outdir, direct = job
    hh, hp = Handle("edm::TriggerResults"), Handle("edm::TriggerResults")
    hc = Handle("std::vector<pat::PackedCandidate>")
    hm, hj = Handle("std::vector<pat::MET>"), Handle("std::vector<pat::Jet>")
    cols = {k: [] for k in ["run", "lumi", "event", "hlt", "flt", "pfMet", "pfMetPhi", "caloMet", "nJet", "jetPt", "jetEta",
                            "jetPhi", "muPxDirect", "muPyDirect"]}
    processed = []
    try:
        for filename in files:
            events = Events(filename)
            expected, seen = int(events.size()), 0
            menus = {}
            for event in events:
                ids = []
                for handle, proc, names in ((hh, "HLT", HLT_PATHS), (hp, "PAT", FILTERS)):
                    event.getByLabel("TriggerResults", "", proc, handle)
                    if not handle.isValid():
                        raise RuntimeError("Missing TriggerResults::%s: %s" % (proc, filename))
                    res = handle.product()
                    if proc not in menus or menus[proc][0] != res.parameterSetID():
                        tn = event.object().triggerNames(res)
                        menu = [str(tn.triggerName(i)) for i in range(tn.size())]
                        missing = [n for n in names if n not in menu]
                        if missing:
                            raise RuntimeError("Missing %s entries in %s: %r" % (proc, filename, missing))
                        menus[proc] = (ROOT.edm.ParameterSetID(res.parameterSetID()), [menu.index(n) for n in names])
                    ids.append([bool(res.accept(i)) for i in menus[proc][1]])
                event.getByLabel("slimmedMETs", hm)
                event.getByLabel("slimmedJets", hj)
                if not hm.isValid() or not hj.isValid() or hm.product().size() < 1:
                    raise RuntimeError("Missing slimmedMETs/slimmedJets: " + filename)
                met, jets = hm.product()[0], hj.product()
                aux = event.object().id()
                cols["run"].append(aux.run())
                cols["lumi"].append(aux.luminosityBlock())
                cols["event"].append(aux.event())
                cols["hlt"].append(ids[0])
                cols["flt"].append(ids[1])
                cols["pfMet"].append(met.corPt(ROOT.pat.MET.Type1))
                cols["pfMetPhi"].append(met.corPhi(ROOT.pat.MET.Type1))
                cols["caloMet"].append(met.caloMETPt())
                n = jets.size()
                cols["nJet"].append(n)
                lead = max(range(n), key=lambda i: jets[i].pt()) if n else -1
                cols["jetPt"].append(jets[lead].pt() if n else np.nan)
                cols["jetEta"].append(jets[lead].eta() if n else np.nan)
                cols["jetPhi"].append(jets[lead].phi() if n else np.nan)
                px = py = np.nan
                if direct:
                    event.getByLabel("packedPFCandidates", hc)
                    px = py = 0.0
                    for c in hc.product():
                        if abs(c.pdgId()) == 13:
                            px += c.px()
                            py += c.py()
                cols["muPxDirect"].append(px)
                cols["muPyDirect"].append(py)
                seen += 1
            if seen != expected or seen == 0:
                raise RuntimeError("Incomplete/empty input: %s: %d/%d" % (filename, seen, expected))
            processed.append((filename, seen))
        import uproot
        out = Path(outdir) / (sample + ".root")
        if out.exists():
            raise RuntimeError("Refusing to overwrite " + str(out))
        hlt, flt = np.array(cols.pop("hlt"), dtype=bool), np.array(cols.pop("flt"), dtype=bool)
        tree = {k: np.asarray(v, dtype=np.int64 if k in ("run", "lumi", "event", "nJet") else np.float64)
                for k, v in cols.items()}
        tree.update({"hlt_%d" % i: hlt[:, i] for i in range(len(HLT_PATHS))})
        tree.update({"flt_%d" % i: flt[:, i] for i in range(len(FILTERS))})
        with uproot.recreate(out) as f:
            f["scan"] = tree
            f["hltPaths"] = "\n".join(HLT_PATHS)
            f["filters"] = "\n".join(FILTERS)
            f["inputs"] = "\n".join("%s\t%d" % p for p in processed)
        return sample, processed, None
    except Exception:
        return sample, processed, traceback.format_exc()


def scan(args):
    outdir = Path(args.run_dir) / "scan"
    outdir.mkdir(parents=True, exist_ok=True)
    jobs = [(s, f, str(outdir), args.test) for s, f in file_jobs(args.test)]
    start = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        res = list(pool.map(scan_sample, jobs))
    failed = [r[0] for r in res if r[2]]
    for r in res:
        print("SAMPLE %s files=%d events=%d %s" % (r[0], len(r[1]), sum(n for _, n in r[1]), "FAIL" if r[2] else "OK"),
              flush=True)
        if r[2]:
            print(r[2], flush=True)
    if failed:
        raise RuntimeError("Failed samples: %r" % failed)
    print("SCAN COMPLETE samples=%d events=%d elapsed=%.1fs" % (len(res), sum(n for r in res for _, n in r[1]),
                                                                time.monotonic() - start))


def load_sample(scan_path, vpath, input_revision):
    import uproot
    with uproot.open(scan_path) as f:
        if str(f["hltPaths"]).split("\n") != HLT_PATHS or str(f["filters"]).split("\n") != FILTERS:
            raise RuntimeError("Path/filter list mismatch in " + str(scan_path))
        s = f["scan"].arrays(library="np")
    sample, vkeys, base, ca, _ = mca.chain_a(vpath, input_revision)
    with uproot.open(vpath) as f:
        v = f["preselectedVertexing/events"].arrays(["pfMetPt", "pfMetPhi", "pfMuPx", "pfMuPy", "sigPt", "sigEta", "sigPhi", "sigD0"],
                                                     library="np")
    hkeys = np.stack([s[k].astype(np.int64) for k in ("run", "lumi", "event")], axis=1)
    idx = mca.join(vkeys, hkeys, sample)  # scan row for each vertexing row
    s = {k: a[idx] for k, a in s.items()}
    dmet = np.abs(s["pfMet"] - v["pfMetPt"])
    if np.nanmax(dmet) > 1e-3:
        raise RuntimeError("PF MET mismatch in %s: max %.3g GeV" % (sample, np.nanmax(dmet)))
    dphi = np.abs(met_proxy.wrap(s["pfMetPhi"] - v["pfMetPhi"]))
    if np.nanmax(dphi) > 1e-3:
        raise RuntimeError("PF MET phi mismatch in %s: max %.3g" % (sample, np.nanmax(dphi)))
    if np.isfinite(s["muPxDirect"]).all():
        dmu = np.hypot(s["muPxDirect"] - v["pfMuPx"], s["muPyDirect"] - v["pfMuPy"])
        if dmu.max() > 1e-2:
            raise RuntimeError("Direct PF-muon sum mismatch in %s: max %.3g GeV" % (sample, dmu.max()))
        print("CHECK %s: direct PF-muon sum vs pfMu* max |diff| %.2e GeV over %d events" % (sample, dmu.max(), len(dmu)))
    q = {"pfmet": s["pfMet"], "jetPt": s["jetPt"], "jetEta": s["jetEta"], "jetPhi": s["jetPhi"]}
    q["metnomu"] = np.hypot(s["pfMet"] * np.cos(s["pfMetPhi"]) + v["pfMuPx"], s["pfMet"] * np.sin(s["pfMetPhi"]) + v["pfMuPy"])
    for k in ("mu1Pt", "mu1Eta", "mu1Phi", "mu2Pt", "mu2Eta", "mu2Phi", "maxD0"):
        q[k] = np.full(len(idx), np.nan)
    for i in range(len(idx)):
        pt = np.asarray(v["sigPt"][i], float)
        if len(pt) != 2:
            continue
        o = np.argsort(-pt)
        for j, tag in zip(o, ("mu1", "mu2")):
            q[tag + "Pt"][i], q[tag + "Eta"][i], q[tag + "Phi"][i] = pt[j], v["sigEta"][i][j], v["sigPhi"][i][j]
        q["maxD0"][i] = max(v["sigD0"][i])
    flt = s["flt_0"].astype(bool)  # Flag_METFilters
    flt8 = np.all(np.stack([s["flt_%d" % i] for i in range(1, len(FILTERS))], axis=1), axis=1)
    calo_ok = (s["caloMet"] > 0) & (np.abs(s["pfMet"] - s["caloMet"]) < 0.5 * np.where(s["caloMet"] > 0, s["caloMet"], 1))
    an = flt & calo_ok
    pops = {"all": np.ones(len(idx), bool), "AN": an,
            "ANjet": an & (s["jetPt"] > 80) & (np.abs(s["jetEta"]) < 2.4), "chainA": an & ca, "chainA_noAN": ca}
    hlt = {p: s["hlt_%d" % i] for i, p in enumerate(HLT_PATHS)}
    trig = {t: np.any(np.stack([hlt[p] for p in ps], axis=1), axis=1) for t, ps in TRIGGERS.items()}
    aux = dict(flt=flt, calo_ok=calo_ok, caloZero=s["caloMet"] <= 0, fltDiff=flt != flt8, noJet=s["nJet"] == 0)
    return sample, q, pops, hlt, trig, aux


def cp(k, n, cl=0.6827):
    from scipy.stats import beta
    k, n = np.asarray(k, float), np.asarray(n, float)
    a = (1 - cl) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        lo = np.where(k > 0, beta.ppf(a, k, n - k + 1), 0.0)
        hi = np.where(k < n, beta.ppf(1 - a, k + 1, n - k), 1.0)
        eff = np.where(n > 0, k / np.where(n > 0, n, 1), np.nan)
    return eff, np.where(n > 0, lo, np.nan), np.where(n > 0, hi, np.nan)


def erf_fit(edges, num, den):
    """AN eq. 8 fit eps/2 [erf((x - mu)/sigma) + 1] to the binned efficiency (bins with den >= 20)."""
    from scipy.optimize import curve_fit
    from scipy.special import erf
    x = 0.5 * (edges[:-1] + edges[1:])
    e, lo, hi = cp(num, den)
    ok = den >= 20
    if ok.sum() < 4:
        return [np.nan] * 6
    sig = np.maximum(0.5 * (hi - lo), 1e-4)[ok]
    f = lambda xx, eps, mu, s: 0.5 * eps * (erf((xx - mu) / s) + 1)  # noqa: E731
    try:
        p, c = curve_fit(f, x[ok], e[ok], p0=[0.98, 150, 60], sigma=sig, absolute_sigma=True,
                         bounds=([0, 0, 1], [1.0, 1000, 1000]))
        return list(p) + list(np.sqrt(np.diag(c)))
    except Exception:
        return [np.nan] * 6


def analyze(args):
    import ROOT
    ROOT.gROOT.SetBatch(True)
    run = Path(args.run_dir)
    out = run / args.output
    if out.exists():
        raise RuntimeError("Refusing to overwrite " + str(out))
    vpaths = {Path(p).stem: p for p in glob.glob(str(Path(args.input_dir) / "*.root"))}
    samples = [s for s, _ in file_jobs(args.test)]
    data = []
    for s in samples:
        data.append(load_sample(run / "scan" / (s + ".root"), vpaths[s], args.input_revision))
        if data[-1][0] != s:
            raise RuntimeError("Sample mismatch " + s)
        print("LOADED %s events=%d AN=%d chainA=%d" % (s, len(data[-1][1]["pfmet"]), data[-1][2]["AN"].sum(),
                                                       data[-1][2]["chainA"].sum()), flush=True)
    total = sum(len(d[1]["pfmet"]) for d in data)
    if not args.test and total != 720000:
        raise RuntimeError("Coverage %d" % total)
    # validation against earlier studies
    if not args.test:
        import uproot
        with uproot.open(args.trigger_root) as ft:
            h = ft["overall/accepted"]
            ref = dict(zip(h.axis().labels(), h.values()))
        for p in HLT_PATHS:
            n = int(sum(d[3][p].sum() for d in data))
            if n != int(ref[p]):
                raise RuntimeError("Path count mismatch %s: %d vs %d" % (p, n, ref[p]))
        if int(sum(d[2]["chainA_noAN"].sum() for d in data)) != 96715:
            raise RuntimeError("chain-A count differs from 96715")
        if int(sum((d[2]["chainA_noAN"] & d[4]["S14"]).sum() for d in data)) != 50817:
            raise RuntimeError("chain A and S14 differs from 50817")
        print("VALIDATED: %d path counts match trigger_counts.root; chain A = 96715, chain A and S14 = 50817; "
              "PF MET and phi match per event" % len(HLT_PATHS))

    mass_of = lambda s: re.match(r"(Mchi-[^_]+_dMchi-[^_]+)_", s).group(1)  # noqa: E731
    life_of = lambda s: re.search(r"(ctau-[^_]+)$", s).group(1)  # noqa: E731
    groups = {"overall": list(range(len(data)))}
    for i, d in enumerate(data):
        groups.setdefault("mass/" + mass_of(d[0]), []).append(i)
        groups.setdefault("lifetime/" + life_of(d[0]), []).append(i)
        groups["sample/" + d[0]] = [i]

    f = ROOT.TFile.Open(str(out), "CREATE")
    ROOT.TNamed("definitions", __doc__).Write()
    ROOT.TNamed("analysisRevision", args.revision).Write()
    ROOT.TNamed("inputRevision", args.input_revision).Write()
    ROOT.TNamed("samples", "\n".join(samples)).Write()
    ROOT.TNamed("triggers", "\n".join("%s: %s" % (t, " | ".join(ps)) for t, ps in TRIGGERS.items())).Write()
    ROOT.TNamed("filters", "\n".join(FILTERS)).Write()

    def th1(name, edges, x, title):
        h = ROOT.TH1D(name, title, len(edges) - 1, array("d", [float(e) for e in edges]))
        h.Sumw2()
        x = x[np.isfinite(x)]
        cnt = np.bincount(np.searchsorted(edges, x, side="right"), minlength=len(edges) + 1)
        for b, c in enumerate(cnt):
            h.SetBinContent(b, float(c))
            h.SetBinError(b, float(np.sqrt(c)))
        h.SetEntries(float(len(x)))
        h.Write()
        return cnt[1:-1]

    def cat(idx, getter):
        return np.concatenate([getter(data[i]) for i in idx])

    fits = []
    for g, idx in groups.items():
        varlist = SAMPLE_VARS if g.startswith("sample/") else tuple(VARS)
        for pop in POPS:
            pm = cat(idx, lambda d: d[2][pop])
            dname = "%s/%s" % (pop, g)
            f.mkdir(dname)
            f.cd(dname)
            for v in varlist:
                edges, lab = VARS[v]
                x = cat(idx, lambda d: d[1][v])
                den = th1("den__%s" % v, edges, x[pm], "%s;%s;Events" % (POPS[pop], lab))
                for t in TRIGGERS:
                    tm = cat(idx, lambda d: d[4][t])
                    num = th1("num__%s__%s" % (t, v), edges, x[pm & tm], "%s and %s;%s;Events" % (POPS[pop], t, lab))
                    if v in ("metnomu", "pfmet") and (g == "overall" or g.startswith("mass/")):
                        fits.append((pop, g, t, v, erf_fit(edges, num, den)))
            f.cd()
    # Fig-92 analogue: METNoMu efficiency in leading gen signal muon pT bins, population ANjet
    for g in [k for k in groups if not k.startswith("sample/")]:
        idx = groups[g]
        pm = cat(idx, lambda d: d[2]["ANjet"])
        mu1 = cat(idx, lambda d: d[1]["mu1Pt"])
        x = cat(idx, lambda d: d[1]["metnomu"])
        f.mkdir("mu1bins/" + g)
        f.cd("mu1bins/" + g)
        for lo, hi in MU1_BINS:
            sel = pm & (mu1 >= lo) & (mu1 < hi)
            tag = "%g_%s" % (lo, "inf" if np.isinf(hi) else "%g" % hi)
            th1("den__%s" % tag, MET_EDGES, x[sel], "ANjet, lead gen mu pT %s GeV;%s;Events" % (tag, VARS["metnomu"][1]))
            for t in ("NoMu120", "S14"):
                tm = cat(idx, lambda d: d[4][t])
                th1("num__%s__%s" % (t, tag), MET_EDGES, x[sel & tm], "")
        f.cd()
    # tables (AN Tables 32-34 analogue): per sample, paths x denominators all / AN
    f.mkdir("tables")
    f.cd("tables")
    for d in data:
        for pop in ("all", "AN"):
            h = ROOT.TH1D("%s__%s" % (d[0], pop), "%s %s;;Events" % (d[0], pop), len(TABLE_PATHS) + 1, 0, len(TABLE_PATHS) + 1)
            h.GetXaxis().SetBinLabel(1, "N")
            h.SetBinContent(1, float(d[2][pop].sum()))
            for i, p in enumerate(TABLE_PATHS, 2):
                h.GetXaxis().SetBinLabel(i, p)
                h.SetBinContent(i, float((d[2][pop] & d[3][p]).sum()))
            h.Write()
    f.cd()
    # fits and offline-cut flow
    t = ROOT.TTree("fits", "erf fits eps/2 [erf((x-mu)/sigma)+1]")
    sb = {k: ROOT.std.string() for k in ("pop", "group", "trigger", "var")}
    fb = {k: np.zeros(1) for k in ("eps", "mu", "sigma", "eps_err", "mu_err", "sigma_err")}
    for k, b in sb.items():
        t.Branch(k, b)
    for k, b in fb.items():
        t.Branch(k, b, k + "/D")
    for pop, g, tr, v, p in fits:
        for k, val in zip(sb, (pop, g, tr, v)):
            sb[k].assign(val)
        for k, val in zip(fb, p):
            fb[k][0] = val
        t.Fill()
    t.Write()
    flow = [("all", total), ("MET filters", sum(d[5]["flt"].sum() for d in data)),
            ("Calo MET = 0", sum(d[5]["caloZero"].sum() for d in data)),
            ("METFilters != AND of 8 flags", sum(d[5]["fltDiff"].sum() for d in data)),
            ("no jet", sum(d[5]["noJet"].sum() for d in data)), ("AN", sum(d[2]["AN"].sum() for d in data)),
            ("ANjet", sum(d[2]["ANjet"].sum() for d in data)), ("chainA (no AN)", sum(d[2]["chainA_noAN"].sum() for d in data)),
            ("chainA and AN", sum(d[2]["chainA"].sum() for d in data))]
    h = ROOT.TH1D("counts", "event counts;;Events", len(flow), 0, len(flow))
    for i, (lab, n) in enumerate(flow, 1):
        h.GetXaxis().SetBinLabel(i, lab)
        h.SetBinContent(i, float(n))
    h.Write()
    f.Close()
    for lab, n in flow:
        print("  %-16s %8d" % (lab, n))
    for pop, g, tr, v, p in fits:
        if g == "overall":
            print("FIT %-7s %-8s %-6s eps %.4f+-%.4f mu %.1f+-%.1f sigma %.1f+-%.1f" % (pop, tr, v, p[0], p[3], p[1], p[4], p[2], p[5]))
    for pop in ("all", "AN"):
        print("TABLE denominator=%s (efficiency %%; N in first row)" % pop)
        for mp in AN_POINTS:
            for d in data:
                if mass_of(d[0]) == mp:
                    n = d[2][pop].sum()
                    print("  %-34s N=%-6d " % (d[0], n) + " ".join("%s=%.2f" % (re.sub(r"^HLT_|_v\d+$", "", p),
                          100.0 * (d[2][pop] & d[3][p]).sum() / n) for p in TABLE_PATHS))


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import uproot
    run = Path(args.run_dir)
    f = uproot.open(run / args.output)
    out = run / args.plot_dir
    samples = str(f["samples"].member("fTitle")).split("\n")
    masses = sorted({re.match(r"(Mchi-[^_]+_dMchi-[^_]+)_", s).group(1) for s in samples},
                    key=lambda s: float(s.split("_")[0][5:].replace("p", ".")))
    lifes = sorted({re.search(r"(ctau-[^_]+)$", s).group(1) for s in samples}, key=lambda s: float(s[5:]))
    foot = "Signal MC 2022EE (09-28), raw unweighted event counts, Clopper-Pearson 68%; pooling unweighted"
    tcol = {"NoMu120": "#0072B2", "MET120": "#D55E00", "ORAN": "#009E73", "S14": "#CC79A7", "NoMuOR": "#56B4E9"}

    def save(fig, path, axes=()):
        fig.text(0.01, 0.005, foot, fontsize=6.5, color="0.35")
        fig.tight_layout(rect=(0, 0.02, 1, 0.95))
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=110)
        if args.log and axes:
            try:
                for ax in axes:
                    lo, hi = ax.get_ylim()
                    ax.set_yscale("log")
                    ax.set_ylim(1e-3 if hi <= 1.1 else 0.5, hi)  # efficiency axes start at 1e-3, counts at 0.5
                fig.savefig(path.with_name(path.stem + "_log.png"), dpi=110)
            except ValueError as err:
                print("no log copy for %s: %s" % (path, err))
        plt.close(fig)

    def eff(d, t, v):
        num, den = f["%s/num__%s__%s" % (d, t, v)], f["%s/den__%s" % (d, v)]
        e, lo, hi = cp(num.values(), den.values())
        return den.axis().edges(), e, lo, hi, den.values()

    def draw(ax, edges, e, lo, hi, **kw):
        x = 0.5 * (edges[:-1] + edges[1:])
        ok = np.isfinite(e)
        ax.errorbar(x[ok], e[ok], yerr=[(e - lo)[ok], (hi - e)[ok]], xerr=(0.5 * np.diff(edges))[ok], fmt="o", ms=3,
                    lw=1, **kw)

    def axes_fmt(ax, v):
        ax.set_xlabel(VARS[v][1] if v in VARS else v)
        ax.set_ylabel("Trigger efficiency")
        ax.set_ylim(0, 1.05)
        ax.axhline(1, color="0.7", lw=0.8)

    def is_log_x(v):
        return v == "maxD0"

    groups = ["overall"] + ["mass/" + m for m in masses] + ["lifetime/" + l for l in lifes] + ["sample/" + s for s in samples]
    # 1. per group: each trigger vs each variable, AN-like vs chain A overlay (Figs 12, 93, 94, 95, 96)
    for g in groups:
        vs = SAMPLE_VARS if g.startswith("sample/") else tuple(VARS)
        for v in vs:
            fig, axes = plt.subplots(1, len(TRIGGERS), figsize=(5 * len(TRIGGERS), 4.6))
            for ax, t in zip(axes, TRIGGERS):
                for pop, col in (("AN", "k"), ("chainA", "#E69F00")):
                    edges, e, lo, hi, den = eff("%s/%s" % (pop, g), t, v)
                    draw(ax, edges, e, lo, hi, color=col, label="%s (N=%d)" % (POPS[pop], den.sum()))
                axes_fmt(ax, v)
                ax.set_title(TRIG_LABEL[t], fontsize=9)
                if is_log_x(v):
                    ax.set_xscale("symlog", linthresh=0.1)
                    ax.set_xlim(left=0)
            axes[0].legend(fontsize=7, loc="upper left")
            fig.suptitle("Trigger efficiency vs %s | %s" % (v, g.replace("/", ": ")), fontsize=11)
            save(fig, out / "per_group" / g.replace("/", "_") / ("eff_%s.png" % v), list(axes))
    # 2. trigger comparison on one plot (Fig 94 analogue), overall, AN-like and chain A
    for pop in ("AN", "chainA"):
        for v in VARS:
            fig, ax = plt.subplots(figsize=(7, 5))
            for t in TRIGGERS:
                edges, e, lo, hi, den = eff("%s/overall" % pop, t, v)
                draw(ax, edges, e, lo, hi, color=tcol[t], label=TRIG_LABEL[t])
            axes_fmt(ax, v)
            if is_log_x(v):
                ax.set_xscale("symlog", linthresh=0.1)
                ax.set_xlim(left=0)
            ax.legend(fontsize=7)
            ax.set_title("All samples | %s" % POPS[pop], fontsize=9)
            save(fig, out / "triggers" / pop / ("compare_%s.png" % v), [ax])
    # 3. overlays: masses (pooled over lifetimes), lifetimes (pooled over masses), AN four points (Fig 12)
    sets = {"mass": ["mass/" + m for m in masses], "lifetime": ["lifetime/" + l for l in lifes],
            "anpoints": ["mass/" + m for m in AN_POINTS if m in masses]}
    for pop in ("AN", "chainA"):
        for kind, gs in sets.items():
            cmap = plt.get_cmap("viridis", len(gs))
            for t in TRIGGERS:
                for v in ("metnomu", "pfmet", "maxD0"):
                    fig, ax = plt.subplots(figsize=(8, 5.5))
                    for i, g in enumerate(gs):
                        edges, e, lo, hi, den = eff("%s/%s" % (pop, g), t, v)
                        lab = g.split("/")[1] + (" " + AN_POINTS[g.split("/")[1]] if kind == "anpoints" else "")
                        draw(ax, edges, e, lo, hi, color=cmap(i), label=lab)
                    axes_fmt(ax, v)
                    if is_log_x(v):
                        ax.set_xscale("symlog", linthresh=0.1)
                        ax.set_xlim(left=0)
                    ax.legend(fontsize=6, ncol=2 if kind == "mass" else 1)
                    ax.set_title("%s | %s overlay | %s" % (TRIG_LABEL[t], kind, POPS[pop]), fontsize=9)
                    save(fig, out / "overlays" / pop / ("%s_%s_%s.png" % (kind, t, v)), [ax])
                # denominator distributions, raw and unit area
            for v in ("metnomu", "pfmet"):
                for norm in (False, True):
                    fig, ax = plt.subplots(figsize=(8, 5.5))
                    for i, g in enumerate(gs):
                        h = f["%s/%s/den__%s" % (pop, g, v)]
                        vals, tot = h.values(), h.values(flow=True).sum()
                        if tot:
                            ax.stairs(vals / tot if norm else vals, h.axis().edges(), color=cmap(i), label=g.split("/")[1])
                    ax.set_xlabel(VARS[v][1])
                    ax.set_ylabel("Fraction of events / bin (unit area incl. under/overflow)" if norm else "Events / bin")
                    ax.set_ylim(bottom=0)
                    ax.legend(fontsize=6, ncol=2 if kind == "mass" else 1)
                    ax.set_title("Denominator %s | %s overlay | %s | %s" % (v, kind, POPS[pop], "unit area" if norm else "raw"),
                                 fontsize=9)
                    save(fig, out / "overlays" / pop / ("den_%s_%s_%s.png" % (kind, v, "unit" if norm else "raw")), [ax])
    # 4. Fig-92 analogue: leading gen signal muon pT bins
    for g in [k for k in groups if not k.startswith("sample/")]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, t in zip(axes, ("NoMu120", "S14")):
            cmap = plt.get_cmap("plasma", len(MU1_BINS) + 1)
            for i, (lo, hi) in enumerate(MU1_BINS):
                tag = "%g_%s" % (lo, "inf" if np.isinf(hi) else "%g" % hi)
                num, den = f["mu1bins/%s/num__%s__%s" % (g, t, tag)], f["mu1bins/%s/den__%s" % (g, tag)]
                e, l, h = cp(num.values(), den.values())
                draw(ax, den.axis().edges(), e, l, h, color=cmap(i),
                     label="lead gen mu pT %s GeV (N=%d)" % (tag.replace("_", "-"), den.values().sum()))
            axes_fmt(ax, "metnomu")
            ax.set_title(TRIG_LABEL[t], fontsize=9)
            ax.legend(fontsize=7)
        fig.suptitle("Efficiency in leading gen signal muon pT bins | %s | %s" % (g.replace("/", ": "), POPS["ANjet"]), fontsize=10)
        save(fig, out / "mu1bins" / ("%s.png" % g.replace("/", "_")), list(axes))
    print("PLOTS written to %s" % out)


# AN-18-124 v12 figures (PDF page, crop box x0, y0, x1, y1 in pixels at 150 dpi) and our analogue layout
AN_FIGS = {
    "fig11": (35, (140, 675, 1110, 1530), "Fig 11: PFMETNoMu120 vs METNoMu (Run 2 data + bkg MC)"),
    "fig12": (37, (140, 320, 1110, 1530), "Fig 12: METNoMu120 in signal vs pTmiss and METNoMu, 4 points"),
    "fig91": (137, (140, 230, 1110, 1080), "Fig 91: METNoMu120 in leading global muon pT bins (data + bkg MC)"),
    "fig92": (138, (140, 510, 1110, 1340), "Fig 92: METNoMu120 in leading global muon pT bins, signal"),
    "fig93": (139, (140, 510, 1110, 1340), "Fig 93: PFMET120 vs pTmiss (Run 2 data + bkg MC)"),
    "fig94": (140, (140, 305, 1110, 1545), "Fig 94: OR of MET and METNoMu triggers vs METNoMu (data + bkg MC)"),
    "tables": (141, (140, 280, 1110, 1175), "Tables 32-34: signal efficiencies of other triggers"),
    "fig95": (142, (140, 420, 1110, 705), "Fig 95: signal efficiency vs leading jet pT, eta, phi"),
    "fig96": (142, (140, 1110, 1110, 1425), "Fig 96: signal efficiency vs selected muon pT, eta, phi"),
}
AN_LEG = {"Mchi-60p0_dMchi-20p0": "(50,70) GeV", "Mchi-6p0_dMchi-2p0": "(5,7) GeV", "Mchi-52p5_dMchi-5p0": "(50,55) GeV",
          "Mchi-5p25_dMchi-0p5": "(5,5.5) GeV"}  # AN legend order and (m1, m2) labels
AN_COL = ["k", "#C8B400", "#33CC33", "#00CCFF"]


def compare_an(args):
    """Composite PNGs: the AN figure (left, cropped from the private PDF render) next to our signal-MC analogue (right)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import uproot
    from scipy.special import erf
    run = Path(args.run_dir)
    f = uproot.open(run / args.output)
    fits = f["fits"].arrays(library="np")
    out = run / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    out.chmod(0o700)  # contains private CMS-internal AN content
    pages = {p: next(Path(args.an_dir).glob("page%d-*.png" % p)) for p, _, _ in AN_FIGS.values()}
    foot = ("Right: Run-3 signal MC 2022EE (09-28), offline cuts Flag_METFilters + |PF-Calo|/Calo < 0.5; raw unweighted events; "
            "Clopper-Pearson 68%%; points pooled over 4 lifetimes. Left: AN-18-124 v12 (Run 2), PRIVATE.")

    def eff(path_num, path_den):
        num, den = f[path_num], f[path_den]
        e, lo, hi = cp(num.values(), den.values())
        return den.axis().edges(), e, lo, hi, den.values().sum()

    def draw(ax, edges, e, lo, hi, **kw):
        x = 0.5 * (edges[:-1] + edges[1:])
        ok = np.isfinite(e)
        ax.errorbar(x[ok], e[ok], yerr=[(e - lo)[ok], (hi - e)[ok]], xerr=(0.5 * np.diff(edges))[ok], fmt="o", ms=2.5,
                    lw=0.9, **kw)

    def style(ax, xlab, xmax=1000):
        ax.set_xlabel(xlab, fontsize=9)
        ax.set_ylabel("Efficiency", fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.set_xlim(0, xmax)
        ax.tick_params(labelsize=8)
        ax.text(0.02, 0.97, "Run-3 signal MC (2022EE)", transform=ax.transAxes, fontsize=7, va="top")

    def fitline(ax, pop, group, trig, var, col):
        sel = (fits["pop"] == pop) & (fits["group"] == group) & (fits["trigger"] == trig) & (fits["var"] == var)
        if sel.sum() != 1:
            return
        i = np.flatnonzero(sel)[0]
        p = [fits[k][i] for k in ("eps", "mu", "sigma", "eps_err", "mu_err", "sigma_err")]
        x = np.linspace(0, 1000, 500)
        ax.plot(x, 0.5 * p[0] * (erf((x - p[1]) / p[2]) + 1), color=col, lw=1)
        ax.text(0.55, 0.35, "erf fit\neps: %.4f +/- %.4f\nmu: %.1f +/- %.1f GeV\nsigma: %.1f +/- %.1f GeV"
                % (p[0], p[3], p[1], p[4], p[2], p[5]), transform=ax.transAxes, fontsize=7, color=col)

    def four_points(ax, base, trig, var):
        for (m, lab), col in zip(AN_LEG.items(), AN_COL):
            edges, e, lo, hi, n = eff("%s/mass/%s/num__%s__%s" % (base, m, trig, var), "%s/mass/%s/den__%s" % (base, m, var))
            draw(ax, edges, e, lo, hi, color=col, label="%s (N=%d)" % (lab, n))
        ax.legend(fontsize=6, loc="lower right")

    def mu1_panel(ax, group, tag, trig, four):
        if four:
            for (m, lab), col in zip(AN_LEG.items(), AN_COL):
                edges, e, lo, hi, n = eff("mu1bins/mass/%s/num__%s__%s" % (m, trig, tag), "mu1bins/mass/%s/den__%s" % (m, tag))
                draw(ax, edges, e, lo, hi, color=col, label="%s (N=%d)" % (lab, n))
        else:
            edges, e, lo, hi, n = eff("mu1bins/%s/num__%s__%s" % (group, trig, tag), "mu1bins/%s/den__%s" % (group, tag))
            draw(ax, edges, e, lo, hi, color="r", label="all 72 samples (N=%d)" % n)
        ax.legend(fontsize=6, loc="lower right")
        style(ax, "offline METNoMu [GeV]")
        ax.set_title("lead gen signal muon pT %s GeV, %s" % (tag.replace("_inf", "+").replace("_", "-"), TRIG_LABEL[trig]),
                     fontsize=8)

    def compose(key, nrows, ncols, fill, width_ratio=1.0):
        page, box, title = AN_FIGS[key]
        img = mpimg.imread(str(pages[page]))[box[1]:box[3], box[0]:box[2]]
        h_in = 6.5 if nrows == 1 else 4.0 * nrows
        fig = plt.figure(figsize=(9 + 5.2 * ncols * width_ratio, h_in))
        gs = fig.add_gridspec(1, 2, width_ratios=[9, 5.2 * ncols * width_ratio])
        axl = fig.add_subplot(gs[0])
        axl.imshow(img)
        axl.axis("off")
        axl.set_title("AN-18-124 (Run 2): " + title, fontsize=9)
        sub = gs[1].subgridspec(nrows, ncols, hspace=0.45, wspace=0.3)
        axes = [fig.add_subplot(sub[r, c]) for r in range(nrows) for c in range(ncols)]
        fill(axes)
        fig.text(0.01, 0.005, foot, fontsize=6.5, color="0.35")
        fig.suptitle("%s | left: AN, right: this study (Run 3)" % key, fontsize=11)
        fig.tight_layout(rect=(0, 0.02, 1, 0.96))
        fig.savefig(out / ("%s.png" % key), dpi=110)
        plt.close(fig)

    def f11(axes):
        edges, e, lo, hi, n = eff("AN/overall/num__NoMu120__metnomu", "AN/overall/den__metnomu")
        draw(axes[0], edges, e, lo, hi, color="r", label="signal MC, all 72 samples (N=%d)" % n)
        fitline(axes[0], "AN", "overall", "NoMu120", "metnomu", "r")
        axes[0].legend(fontsize=7, loc="lower right")
        style(axes[0], "offline METNoMu [GeV]")
        axes[0].set_title(TRIG_LABEL["NoMu120"], fontsize=8)

    def f12(axes):
        for ax, v, lab in zip(axes, ("pfmet", "metnomu"), ("Type-1 PF MET [GeV]", "offline METNoMu [GeV]")):
            four_points(ax, "AN", "NoMu120", v)
            style(ax, lab)
            ax.set_title(TRIG_LABEL["NoMu120"], fontsize=8)

    bins_an = ["10_20", "20_30", "30_40", "40_inf"]

    def f91(axes):
        for ax, tag in zip(axes, bins_an):
            mu1_panel(ax, "overall", tag, "NoMu120", False)

    def f92(axes):
        for ax, tag in zip(axes, bins_an):
            mu1_panel(ax, "overall", tag, "NoMu120", True)

    def f93(axes):
        edges, e, lo, hi, n = eff("AN/overall/num__MET120__pfmet", "AN/overall/den__pfmet")
        draw(axes[0], edges, e, lo, hi, color="r", label="signal MC, all 72 samples (N=%d)" % n)
        fitline(axes[0], "AN", "overall", "MET120", "pfmet", "r")
        axes[0].legend(fontsize=7, loc="lower right")
        style(axes[0], "Type-1 PF MET [GeV]")
        axes[0].set_title(TRIG_LABEL["MET120"], fontsize=8)

    def f94(axes):
        edges, e, lo, hi, n = eff("AN/overall/num__ORAN__metnomu", "AN/overall/den__metnomu")
        draw(axes[0], edges, e, lo, hi, color="r", label="signal MC, all 72 samples (N=%d)" % n)
        fitline(axes[0], "AN", "overall", "ORAN", "metnomu", "r")
        axes[0].legend(fontsize=7, loc="lower right")
        style(axes[0], "offline METNoMu [GeV]")
        axes[0].set_title(TRIG_LABEL["ORAN"] + " (erf only)", fontsize=8)

    def kin(axes, vars_, labs, xr):
        for ax, v, lab, (x0, x1) in zip(axes, vars_, labs, xr):
            four_points(ax, "AN", "NoMuOR", v)
            style(ax, lab, x1)
            ax.set_xlim(x0, x1)
            ax.set_ylim(0, 1.1)
            ax.set_title(TRIG_LABEL["NoMuOR"], fontsize=8)

    def f95(axes):
        kin(axes, ("jetPt", "jetEta", "jetPhi"), ("leading jet pT [GeV]", "leading jet eta", "leading jet phi"),
            [(0, 600), (-2.5, 2.5), (-np.pi, np.pi)])

    def f96(axes):
        kin(axes, ("mu1Pt", "mu1Eta", "mu1Phi"),
            ("leading gen signal muon pT [GeV]", "leading gen signal muon eta", "leading gen signal muon phi"),
            [(0, 150), (-2.5, 2.5), (-np.pi, np.pi)])

    def ftables(axes):
        ax = axes[0]
        ax.axis("off")
        rows = [("PFMET%d" % x, MET[x]) for x in (110, 120, 130)] + [("DoubleMu3_DCA_PFMET50", DOUBLEMU)] + [
            ("Mu3er1p5_PFJet100_PFMET%d" % x, p) for x, p in zip((70, 80, 90, 100), MU3MET)]
        cols = [(m, c) for m in ("Mchi-60p0_dMchi-20p0", "Mchi-52p5_dMchi-5p0", "Mchi-6p0_dMchi-2p0", "Mchi-5p25_dMchi-0p5")
                for c in ("1", "10", "100", "1000")]
        labels = [{"Mchi-60p0_dMchi-20p0": "(50,20)", "Mchi-52p5_dMchi-5p0": "(50,5)", "Mchi-6p0_dMchi-2p0": "(5,2)",
                   "Mchi-5p25_dMchi-0p5": "(5,0.5)"}[m] + "\n%s mm" % c for m, c in cols]
        cell = []
        for lab, p in rows:
            line = []
            for m, c in cols:
                h = f["tables/%s_ctau-%s__all" % (m, c)]
                v, names = h.values(), list(h.axis().labels())
                line.append("%.2f" % (100.0 * v[names.index(p)] / v[0]))
            cell.append(line)
        ntot = ["%d" % f["tables/%s_ctau-%s__all" % (m, c)].values()[0] for m, c in cols]
        t = ax.table(cellText=[ntot] + cell, rowLabels=["N_total"] + [r[0] for r in rows], colLabels=labels,
                     loc="center", cellLoc="center")
        t.auto_set_font_size(False)
        t.set_fontsize(6.5)
        t.scale(1, 2.2)
        ax.set_title("This study (Run 3): efficiency in %% (N_trig / N_total), all events of each sample, no offline cuts;\n"
                     "(m1, Delta) GeV x ctau. AN Table 34 lists PFMET (not NoMu) for the Mu3er1p5 paths, as here.", fontsize=8)

    compose("fig11", 1, 1, f11)
    compose("fig12", 1, 2, f12)
    compose("fig91", 2, 2, f91)
    compose("fig92", 2, 2, f92)
    compose("fig93", 1, 1, f93)
    compose("fig94", 1, 1, f94)
    compose("fig95", 1, 3, f95)
    compose("fig96", 1, 3, f96)
    compose("tables", 1, 1, ftables, width_ratio=2.6)
    print("AN COMPARISON written to %s" % out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("scan")
    a.add_argument("--run-dir", required=True)
    a.add_argument("--workers", type=int, default=6)
    a.add_argument("--test", action="store_true")
    a = sub.add_parser("analyze")
    a.add_argument("--run-dir", required=True)
    a.add_argument("--input-dir", required=True)
    a.add_argument("--input-revision", required=True)
    a.add_argument("--revision", required=True)
    a.add_argument("--trigger-root", required=True)
    a.add_argument("--output", default="signal_trigger_eff.root")
    a.add_argument("--test", action="store_true")
    a = sub.add_parser("plot")
    a.add_argument("--run-dir", required=True)
    a.add_argument("--output", default="signal_trigger_eff.root")
    a.add_argument("--plot-dir", default="plots")
    a.add_argument("--log", action="store_true")
    a = sub.add_parser("compare-an")
    a.add_argument("--run-dir", required=True)
    a.add_argument("--output", default="signal_trigger_eff_v2.root")
    a.add_argument("--an-dir", required=True, help="private dir with the AN pages rendered by pdftoppm -r 150")
    a.add_argument("--out-dir", default="an_compare")
    args = ap.parse_args()
    {"scan": scan, "analyze": analyze, "plot": plot, "compare-an": compare_an}[args.cmd](args)


if __name__ == "__main__":
    main()
