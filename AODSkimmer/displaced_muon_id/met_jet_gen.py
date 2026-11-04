#!/usr/bin/env python3
"""Leading jet vs dimuon: |dphi| and dR, reco after all selections and the MET, jet cut, gen for all events with a gen
dimuon; six mass points vs all background, ZJets, WJets, QCD (kb/studies/bkg-reco-dimuons.md Part 8; LPC only).

python3 met_jet_gen.py scan --flist-dir DIR --out-dir RUN/gen_scan [--workers 6]     (DIR: <sample>.txt lists; idempotent)
python3 met_jet_gen.py fill --signal-summary recovery.root --bkg-summary bkg_reco_dimuons.root --met-jet met_jet_cut_50.root \
    --signal-scan-dir SIG_SCAN --bkg-scan-dir BKG_SCAN --gen-scan-dir RUN/gen_scan --producer-dir INPUTS --output gen.root
python3 -I met_jet_gen.py plot --input gen.root --signal-summary ... --bkg-summary ... --met-jet ... --style-dir RUNDIR --out-dir OUT

scan: per event the leading slimmedGenJets jet, plus every status-1, last-copy muon of prunedGenParticles.
fill: raw unweighted counts in ROOT, flows kept. Reco: one entry per selected-and-cut event (lowest chi2/ndof pair). Gen: signal =
the truth pair (twoAcc), one entry per event; background = all opposite-sign pairs of accepted muons, several per event.
"""
import argparse
import itertools
import math
import multiprocessing
import os
from pathlib import Path
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from met_jet_dists import JET_ETA_CUT, JET_PT_CUT, LEG_BRANCHES, MET_CUT, PAIR_BRANCHES, keyed
from met_jet_masspoints import GROUPS, MASSES, representative, sample_slices

CHUNK_FILES = 20  # as met_jet_cut.py: FWLite virtual memory grows per looped file
EV_COLS = ["run", "lumi", "event", "nGenJet", "gjPt", "gjEta", "gjPhi"]
MU_COLS = ["ev", "pt", "eta", "phi", "q", "vx", "vy", "vz"]
OBS = {"dphi": (np.linspace(0, np.pi, 33), "#pi/32"), "dr": (np.arange(0, 6.0001, 0.2), "0.2")}
TITLE = {("reco", "dphi"): "|#Delta#phi(leading jet, #mu#mu)|", ("reco", "dr"): "#DeltaR(leading jet, #mu#mu)",
         ("gen", "dphi"): "|#Delta#phi(leading gen jet, gen #mu#mu)|", ("gen", "dr"): "#DeltaR(leading gen jet, gen #mu#mu)"}
CHECK_EVENTS = 200  # bounded independent recomputations per sample


def scan_chunk(job):
    import ROOT
    ROOT.gROOT.SetBatch(True)
    ROOT.gSystem.Load("libFWCoreFWLite")
    ROOT.FWLiteEnabler.enable()
    from DataFormats.FWLite import Events, Handle
    sample, files = job
    hp, hj = Handle("std::vector<reco::GenParticle>"), Handle("std::vector<reco::GenJet>")
    ev, mu, processed = {k: [] for k in EV_COLS}, {k: [] for k in MU_COLS}, []
    try:
        row = 0
        for filename in files:
            events = Events(filename)
            expected, seen = int(events.size()), 0
            for event in events:
                event.getByLabel("prunedGenParticles", hp)
                event.getByLabel("slimmedGenJets", hj)
                if not hp.isValid() or not hj.isValid():
                    raise RuntimeError("Missing gen collections: " + filename)
                jets = hj.product()
                aux = event.object().id()
                n = jets.size()
                lead = max(range(n), key=lambda i: jets[i].pt()) if n else -1
                for k, v in zip(EV_COLS, (aux.run(), aux.luminosityBlock(), aux.event(), n, jets[lead].pt() if n else np.nan,
                                          jets[lead].eta() if n else np.nan, jets[lead].phi() if n else np.nan)):
                    ev[k].append(v)
                for p in hp.product():
                    if abs(p.pdgId()) == 13 and p.status() == 1 and p.isLastCopy():
                        for k, v in zip(MU_COLS, (row, p.pt(), p.eta(), p.phi(), p.charge(), p.vx(), p.vy(), p.vz())):
                            mu[k].append(v)
                row += 1
                seen += 1
            if seen != expected or seen == 0:
                raise RuntimeError("Incomplete/empty input: %s: %d/%d" % (filename, seen, expected))
            processed.append((filename, seen))
        ints = ("run", "lumi", "event", "nGenJet", "ev", "q")
        return (sample, {k: np.asarray(v, dtype=np.int64 if k in ints else np.float64) for k, v in ev.items()},
                {k: np.asarray(v, dtype=np.int64 if k in ints else np.float64) for k, v in mu.items()}, processed, None)
    except Exception:
        return sample, None, None, processed, traceback.format_exc()


def scan(args):
    import uproot
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    samples, jobs = {}, []
    for flist in sorted(Path(args.flist_dir).glob("*.txt")):
        if (out / (flist.stem + ".root")).exists():
            continue
        files = [l.strip() for l in flist.read_text().splitlines() if l.strip() and not l.lstrip().startswith("#")]
        samples[flist.stem] = files
        jobs += [(flist.stem, files[i:i + CHUNK_FILES]) for i in range(0, len(files), CHUNK_FILES)]
    print("PENDING %d samples, %d chunks" % (len(samples), len(jobs)), flush=True)
    with multiprocessing.Pool(args.workers, maxtasksperchild=1) as pool:
        res = pool.map(scan_chunk, jobs, chunksize=1)
    failed = []
    for sample, files in samples.items():
        chunks = [r for r in res if r[0] == sample]
        processed = [p for r in chunks for p in r[3]]
        errors = [r[4] for r in chunks if r[4]]
        ok = not errors and [f for f, _ in processed] == files
        print("SAMPLE %s files=%d/%d events=%d %s" % (sample, len(processed), len(files), sum(n for _, n in processed),
                                                       "OK" if ok else "FAIL"), flush=True)
        for e in errors:
            print(e, flush=True)
        if not ok:
            failed.append(sample)
            continue
        offsets = np.cumsum([0] + [len(r[1]["run"]) for r in chunks])
        with uproot.recreate(out / (sample + ".root")) as f:
            f["events"] = {k: np.concatenate([r[1][k] for r in chunks]) for k in EV_COLS}
            f["muons"] = {k: np.concatenate([r[2][k] + (o if k == "ev" else 0) for r, o in zip(chunks, offsets)]) for k in MU_COLS}
            f["inputs"] = "\n".join("%s\t%d" % p for p in processed)
    if failed:
        raise RuntimeError("Failed samples: %r" % failed)
    print("SCAN DONE", flush=True)


def wrap(d):
    return np.abs(d - 2 * np.pi * np.round(d / (2 * np.pi)))


def dimuon(pa, ea, fa, pb, eb, fb):
    """Direction (pT, eta, phi) of the Cartesian sum of two massless muons; eta/phi NaN where the transverse sum is 0."""
    px, py = pa * np.cos(fa) + pb * np.cos(fb), pa * np.sin(fa) + pb * np.sin(fb)
    pz = pa * np.sinh(ea) + pb * np.sinh(eb)
    pt = np.hypot(px, py)
    with np.errstate(invalid="ignore", divide="ignore"):
        good = np.isfinite(pt) & (pt > 0) & np.isfinite(pz)
        return pt, np.where(good, np.arcsinh(pz / np.where(good, pt, 1)), np.nan), np.where(good, np.arctan2(py, px), np.nan)


def angles(jpt, jeta, jphi, meta, mphi):
    """|dphi| and dR with pseudorapidity; NaN if undefined (jet pT <= 0, non-finite inputs)."""
    bad = ~(np.isfinite(jpt) & (jpt > 0) & np.isfinite(jeta) & np.isfinite(jphi) & np.isfinite(meta) & np.isfinite(mphi))
    dphi = wrap(jphi - mphi)
    dr = np.hypot(jeta - meta, dphi)
    dphi[bad] = np.nan
    dr[bad] = np.nan
    return dphi, dr


def check_angles(ROOT, what, legs, jets, dphi, dr):
    """Up to CHECK_EVENTS entries through TLorentzVector, independent of dimuon() and angles()."""
    for i in np.flatnonzero(np.isfinite(dphi))[:CHECK_EVENTS]:
        a, b, j = ROOT.TLorentzVector(), ROOT.TLorentzVector(), ROOT.TLorentzVector()
        a.SetPtEtaPhiM(float(legs[0][i]), float(legs[1][i]), float(legs[2][i]), 0)
        b.SetPtEtaPhiM(float(legs[3][i]), float(legs[4][i]), float(legs[5][i]), 0)
        j.SetPtEtaPhiM(float(jets[0][i]), float(jets[1][i]), float(jets[2][i]), 0)
        m = a + b
        if abs(abs(j.DeltaPhi(m)) - dphi[i]) > 1e-5 or abs(j.DeltaR(m) - dr[i]) > 1e-5:
            raise RuntimeError("Angle differs from TLorentzVector: %s entry %d" % (what, i))


def accepted(m):
    lxy = np.hypot(m["vx"], m["vy"])
    return (np.abs(m["eta"]) < 2.4) & (lxy >= 0) & (lxy < 740) & (np.abs(m["vz"]) < 960)


def bkg_pairs(m, acc):
    """Indices (i, j) into the muon arrays of all opposite-sign pairs of accepted muons, within each event."""
    idx = np.flatnonzero(acc)
    ev = m["ev"][idx]
    ia, ib = [], []
    for e in np.unique(ev):
        here = idx[ev == e]
        q = m["q"][here]
        pairs = itertools.product(here[q > 0], here[q < 0])
        for i, j in pairs:
            ia.append(i); ib.append(j)
    return np.array(ia, dtype=np.int64), np.array(ib, dtype=np.int64)


def check_bkg_pairs(m, acc, ia, ib):
    """For the first CHECK_EVENTS events with muons: accepted list and the unordered opposite-sign pair set, recomputed."""
    for e in np.unique(m["ev"])[:CHECK_EVENTS]:
        here = np.flatnonzero(m["ev"] == e).tolist()
        ok = [i for i in here if abs(m["eta"][i]) < 2.4 and math.hypot(m["vx"][i], m["vy"][i]) < 740 and abs(m["vz"][i]) < 960]
        if sorted(ok) != np.flatnonzero(acc & (m["ev"] == e)).tolist():
            raise RuntimeError("Accepted muons differ in event row %d" % e)
        want = {frozenset(c) for c in itertools.combinations(ok, 2) if m["q"][c[0]] * m["q"][c[1]] == -1}
        got = {frozenset((int(i), int(j))) for i, j in zip(ia[m["ev"][ia] == e], ib[m["ev"][ia] == e])}
        if want != got:
            raise RuntimeError("Opposite-sign pairs differ in event row %d" % e)


def fill(args):
    import ROOT
    import uproot
    from bkg_reco_dimuons import vector_stages
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    ROOT.TH1.AddDirectory(False)
    mj = uproot.open(args.met_jet)
    expected = {kind: {k.split("/")[2].split(";")[0] for k in mj.keys(recursive=True)
                       if k.startswith(kind + "/sample/") and k.count("/") == 2} for kind in ("signal", "background")}
    expected["signal"] = {n for n in expected["signal"] if sample_slices("signal", n)}
    hists, counts, processed = {}, {}, {"signal": set(), "background": set()}

    def put(kind, s, level, name, x):
        key = (kind, s, level, name)
        if key not in hists:
            e = OBS[name][0]
            hists[key] = ROOT.TH1D(name, "", len(e) - 1, np.asarray(e, dtype=np.float64))
        if len(x):
            hists[key].FillN(len(x), x.astype(np.float64), np.ones(len(x)))

    for kind, summary, scan_dir in (("signal", args.signal_summary, args.signal_scan_dir),
                                    ("background", args.bkg_summary, args.bkg_scan_dir)):
        sf = uproot.open(summary)
        for name, _ in (l.split() for l in str(sf["inputs"].member("fTitle")).splitlines() if l.strip()):
            if name not in expected[kind] or (args.only and not __import__("re").search(args.only, name)):
                continue
            processed[kind].add(name)
            with uproot.open(Path(args.producer_dir) / (name + ".root")) as f:
                if str(f["preselectedVertexing/sample"].member("fTitle")) != name:
                    raise RuntimeError("Producer sample mismatch: " + name)
                ev = f["preselectedVertexing/events"].arrays(
                    ["run", "lumi", "event", "pfMetPt", "pfMetPhi", "twoAcc", "sigPt", "sigEta", "sigPhi", "sigQ"], library="np")
                pr = f["preselectedVertexing/pairs"].arrays(PAIR_BRANCHES + LEG_BRANCHES + ["pRecoMuMuPhi"], library="np")
            with uproot.open(Path(scan_dir) / (name + ".root")) as f:
                sc = f["scan"].arrays(["run", "lumi", "event", "pfMet", "jetPt", "jetEta", "jetPhi"], library="np")
            with uproot.open(Path(args.gen_scan_dir) / (name + ".root")) as f:
                gs = f["events"].arrays(EV_COLS, library="np")
                mu = f["muons"].arrays(MU_COLS, library="np")
            pkeys, _ = keyed(ev["run"], ev["lumi"], ev["event"], "producer " + name)
            nev = len(pkeys)
            joined = []
            for what, t in (("scan", sc), ("gen scan", gs)):
                _, index = keyed(t["run"], t["lumi"], t["event"], what + " " + name)
                if len(index) != nev or any(k not in index for k in pkeys):
                    raise RuntimeError("Incomplete %s join for %s" % (what, name))
                joined.append(np.array([index[k] for k in pkeys], dtype=np.int64))
            si, gi = joined
            met = ev["pfMetPt"].astype(np.float64)
            dmet = np.abs(sc["pfMet"][si] - met)
            if not np.all(np.isfinite(dmet)) or dmet.max() > 1e-3:
                raise RuntimeError("PF MET mismatch in " + name)
            with np.errstate(invalid="ignore"):
                cut = (met > MET_CUT) & (sc["jetPt"][si] > JET_PT_CUT) & (np.abs(sc["jetEta"][si]) < JET_ETA_CUT)
            passing = vector_stages(pr, ev["pfMetPt"], ev["pfMetPhi"]) == 6
            row = pr["pEvent"].astype(np.int64)
            chi = pr["pChi2"].astype(np.float64) / pr["pNdof"].astype(np.float64)
            rep = representative(np.flatnonzero(passing & cut[row]), row, chi)
            m = mj["%s/sample/%s/counts" % (kind, name)]
            m = dict(zip(m.axis().labels(), m.values()))
            if (len(rep), nev) != (m["selected_met_and_jet"], m["events"]):
                raise RuntimeError("Counts differ from met_jet_cut for " + name)
            # Reco.
            legs = [pr["pReco%s%s" % (q, l)][rep].astype(np.float64) for l in "AB" for q in ("Pt", "Eta", "Phi")]
            mpt, meta, mphi = dimuon(*legs)
            stored_pt, stored_phi = pr["pRecoMuMuPt"][rep].astype(np.float64), pr["pRecoMuMuPhi"][rep].astype(np.float64)
            if len(rep) and (np.abs(mpt - stored_pt).max() > 1e-3 + 1e-5 * stored_pt.max() or wrap(mphi - stored_phi).max() > 1e-3):
                raise RuntimeError("Reco dimuon differs from the stored pT or phi in " + name)
            rj = [sc[k][si][row[rep]].astype(np.float64) for k in ("jetPt", "jetEta", "jetPhi")]
            rdphi, rdr = angles(rj[0], rj[1], rj[2], meta, mphi)
            check_angles(ROOT, name + " reco", legs, rj, rdphi, rdr)
            # Gen.
            ej = [gs[k][gi].astype(np.float64) for k in ("gjPt", "gjEta", "gjPhi")]
            # The scan's muon "ev" counts scan rows: map to producer rows.
            to_prod = np.empty(nev, dtype=np.int64)
            to_prod[gi] = np.arange(nev)
            muons = dict(mu, ev=to_prod[mu["ev"]])
            has_jet = gs["nGenJet"][gi] > 0
            if kind == "signal":
                two = np.flatnonzero(ev["twoAcc"].astype(bool))
                for i in two[:CHECK_EVENTS]:  # each truth muon must be a scan muon of that event, same charge
                    here = np.flatnonzero(muons["ev"] == i)
                    for pt, q in zip(ev["sigPt"][i], ev["sigQ"][i]):
                        if not any(abs(muons["pt"][k] - pt) < 1e-3 * max(pt, 1) and muons["q"][k] == q for k in here):
                            raise RuntimeError("Truth muon not in the gen scan: %s row %d" % (name, i))
                g = [np.array([ev[b][i][leg] for i in two], dtype=np.float64) for leg in (0, 1) for b in ("sigPt", "sigEta", "sigPhi")]
                if len(two) and np.any(np.array([ev["sigQ"][i][0] * ev["sigQ"][i][1] for i in two]) >= 0):
                    raise RuntimeError("Same-sign truth pair in " + name)
                evs = two
                n_gen_events = len(two)
            else:
                acc = accepted(muons)
                ia, ib = bkg_pairs(muons, acc)
                check_bkg_pairs(muons, acc, ia, ib)
                g = [muons[k][i] for i in (ia, ib) for k in ("pt", "eta", "phi")]
                evs = muons["ev"][ia]
                n_gen_events = len(np.unique(evs))
            keep = has_jet[evs]
            n_nojet = int((~keep).sum())
            g = [x[keep] for x in g]
            evs = evs[keep]
            gj = [x[evs] for x in ej]
            _, geta, gphi = dimuon(*g)
            gdphi, gdr = angles(gj[0], gj[1], gj[2], geta, gphi)
            check_angles(ROOT, name + " gen", g, gj, gdphi, gdr)
            print("SAMPLE %s %s events %d reco %d gen entries %d gen events %d no jet %d" % (
                kind, name, nev, len(rep), len(gdphi), n_gen_events, n_nojet), flush=True)
            found = {"events": nev, "reco_events": len(rep), "reco_undefined": int((~np.isfinite(rdphi)).sum()),
                     "gen_entries": len(gdphi) + n_nojet, "gen_events": n_gen_events, "gen_nojet": n_nojet,
                     "gen_undefined": int((~np.isfinite(gdphi)).sum()),
                     "bkg_muons_accepted": 0 if kind == "signal" else int(acc.sum())}
            for s in sample_slices(kind, name):
                c = counts.setdefault((kind, s), {})
                for k, x in found.items():
                    c[k] = c.get(k, 0) + x
                for level, d1, d2 in (("reco", rdphi, rdr), ("gen", gdphi, gdr)):
                    put(kind, s, level, "dphi", d1[np.isfinite(d1)])
                    put(kind, s, level, "dr", d2[np.isfinite(d2)])
    if not args.only:
        for kind in processed:
            if processed[kind] != expected[kind]:
                raise RuntimeError("%s samples %d != expected %d" % (kind, len(processed[kind]), len(expected[kind])))
    root = ROOT.TFile.Open(str(out), "CREATE")
    ROOT.TNamed("coverage", "signal %d background %d" % (len(processed["signal"]), len(processed["background"]))).Write()
    ROOT.TNamed("definitions", "Raw unweighted. reco/: selected and MET, jet cut, one entry per event (lowest chi2/ndof pair). gen/: "
                "signal twoAcc truth pair; background all opposite-sign pairs of accepted status-1 last-copy muons. Leading "
                "jet any eta. Inputs: %s" % " ".join(map(str, (args.producer_dir, args.gen_scan_dir, args.signal_scan_dir,
                                                                args.bkg_scan_dir, args.met_jet)))).Write()

    def cd(*parts):
        d = root
        for part in parts:
            d = d.GetDirectory(part) or d.mkdir(part)
        d.cd()

    for (kind, s, level, name), h in sorted(hists.items()):
        c = counts[(kind, s)]
        want = (c["reco_events"] - c["reco_undefined"]) if level == "reco" else (c["gen_entries"] - c["gen_nojet"] - c["gen_undefined"])
        if abs(h.GetEntries() - want) > 0.5:
            raise RuntimeError("Closure failed: %s %s %s %s" % (kind, s, level, name))
        cd(kind, *s.split("/"), level)
        h.Write()
    for (kind, s), c in sorted(counts.items()):
        cd(kind, *s.split("/"))
        keys = sorted(c)
        h = ROOT.TH1D("counts", "", len(keys), 0, len(keys))
        for i, k in enumerate(keys, 1):
            h.GetXaxis().SetBinLabel(i, k)
            h.SetBinContent(i, c[k])
        h.Write()
        print("TOTAL %s %s %s" % (kind, s, " ".join("%s=%d" % (k, c[k]) for k in keys)))
    root.Close()
    print("FILL DONE", out)


def plot(args):
    import ROOT
    sys.path.insert(0, args.style_dir)
    import cms_style as cs
    from met_jet_plots import M_HEADER, M_KEYS, load, mass_key
    cs.setup(era="2022EE (13.6 TeV)", status="Simulation Private Work", norm_label="Unit area")
    ROOT.gErrorIgnoreLevel = ROOT.kWarning
    out = Path(args.out_dir)
    if out.exists():
        raise FileExistsError(out)
    sig, bkg, n_bkg = load(args)
    f = ROOT.TFile.Open(args.input)
    if f.Get("coverage").GetTitle() != "signal 24 background %d" % n_bkg:
        raise RuntimeError("Unexpected coverage: " + f.Get("coverage").GetTitle())

    def get(k):
        h = f.Get(k)
        if not h:
            raise RuntimeError("Missing " + k)
        return h

    def stored(path):
        h = get(path + "/counts")
        return {h.GetXaxis().GetBinLabel(i): h.GetBinContent(i) for i in range(1, h.GetNbinsX() + 1)}

    masses = ["mass/" + m for m in sorted(MASSES, key=mass_key)]
    fam = sorted({mass_key(k[5:])[0] for k in masses})
    colors, styles = [cs.BLUE, cs.ORANGE], [(3, 25), (2, 24), (1, 20)]
    look = {k: (colors[fam.index(mass_key(k[5:])[0])], styles[[0.1, 0.2, 0.4].index(round(mass_key(k[5:])[1] / mass_key(k[5:])[0], 3))])
            for k in masses}
    groups = {"%g GeV" % m: colors[i] for i, m in enumerate(fam)}
    keys = dict(zip(M_KEYS, styles))
    BKGS = [("overall", "bkg_all", "all %d" % n_bkg)] + [("group/" + g, "bkg_" + g, g) for g in GROUPS]
    nall = {k: sig[k][0] for k in masses}
    nall.update({b: bkg[b][0] for b, _, _ in BKGS})
    for k in masses + [b for b, _, _ in BKGS]:
        kind = "signal" if k in masses else "background"
        c, v = stored("%s/%s" % (kind, k)), (sig if kind == "signal" else bkg)[k]
        if (c["events"], c["reco_events"]) != (v[0], v[8]):
            raise RuntimeError("Fill and cutflow counts differ for " + k)

    def band_plot(base, graphs, xr, xtitle, ytitle, lines, extra):
        c = cs.canvas("c_" + str(base).replace("/", "_"), band=True)
        top = 0.34
        c.SetTopMargin(top)
        hi = max([p[3] for p in cs._points(graphs)] or [1])
        ymax = cs._round_up(1.1 * hi)
        wide = ymax < 0.1 or ymax >= 1000
        if wide:
            c.SetLeftMargin(0.19)
        fr = cs.frame(c, xr[0], xr[1], 0, ymax, xtitle, ytitle)
        if wide:
            fr.GetYaxis().SetTitleOffset(1.75)
        fr.GetYaxis().SetTitleSize(0.045)
        c.Update()
        for g in graphs:
            g.Draw(cs._info[id(g)][1] + " same")
        t = cs.header(c, 0.15, 0.95, (0.04, 0.031), 0.96)
        t.SetTextSize(0.028); t.SetTextAlign(13)
        for i, line in enumerate(lines):
            t.DrawLatex(0.18, 0.90 - 0.035 * i, line)
        ytop = 0.90 - 0.035 * len(lines) - 0.005
        cs.legend([(cs._proxy(col, 1, 20), lab, "l") for lab, col in dict(groups, **extra).items()], y2=ytop, y1=1 - top + 0.01,
                  x1=0.50, header=M_HEADER, ncol=2, size=0.026)
        cs.legend([(cs._proxy(cs.BLACK, ls, ms), lab, "l") for lab, (ls, ms) in keys.items()], y2=ytop, y1=1 - top + 0.01,
                  x1=0.17, x2=0.47, size=0.026)
        cs.save(c, base, log=True)

    def graph(path, nev, color, style, mode):
        h = get(path)
        if mode == "frac":
            h = h.Clone(); cs._keep.append(h)
            h.Scale(1. / nev)  # the clone only; the stored histogram stays raw
        return cs.hist_graph(h, color, style[0], style[1], step=True, norm=mode == "norm")

    caps = cs.Captions("Leading jet vs dimuon (kb/studies/bkg-reco-dimuons.md Part 8)", [
        "Signal: mass points M = m_chi - dm/2 = 5 and 50 GeV with dm = 0.1, 0.2, 0.4 M, each pooled unweighted over c-tau (40,000 events each). Colour = M, line style = dm. Background: all %d 2022EE samples "
        "pooled, or one process group. Raw MC, unweighted." % n_bkg,
        "|dphi| wrapped to [0, pi]; dR = sqrt(deta^2 + dphi^2) with pseudorapidity. Dimuon direction = Cartesian sum of two massless muons. Undefined angles are not filled, are counted below and stay in the fraction denominator.",
        "reco/: selected events passing Type-1 PF MET > 80 GeV and highest-pT slimmedJets jet pT > 80 GeV, |eta| < 2.4 (no jet ID, no muon cleaning). One entry per event: among its passing pairs, lowest chi2/ndof, then lowest "
        "pair index; legs are the paired component tracks at their reference point.",
        "gen/: no reco selection, no MET or jet cut. Leading jet = highest-pT slimmedGenJets jet, any eta; these jets include muons, so the jet can contain the dimuon's muons. Acceptance: |eta| < 2.4, 0 <= vxy < 740 cm, |vz| < 960 cm, no pT "
        "minimum. Signal: only the truth pair (exactly two signal muons from chi2, opposite charge, both in acceptance), one entry per event. Background: all opposite-sign pairs of accepted status-1, last-copy muons of any origin, so an event "
        "with several pairs counts several times. Muons from decays simulated in the detector simulation are not in the gen record, so the gen background population need not be the one passing the reco selection.",
        "Versions: raw; *_norm ('Unit area': normalized within the plotted range, flows excluded); *_frac (entries / all events of the slice: a true event fraction for reco and gen signal; pairs per event for gen background). Log-y: *_log.png."])
    for level, line in (("reco", "Selected, after cut"), ("gen", "All events with a gen dimuon")):
        for obs, (edges, per) in OBS.items():
            for b, name, lab in BKGS:
                ent = "Events" if level == "reco" else "Entries"
                for mode, sfx, yt in (("raw", "", ent + " / "), ("norm", "_norm", "Fraction of entries / "),
                                      ("frac", "_frac", ent + " / all events / ")):
                    gr = [graph("signal/%s/%s/%s" % (k, level, obs), nall[k], look[k][0], look[k][1], mode) for k in masses]
                    gr.append(graph("background/%s/%s/%s" % (b, level, obs), nall[b], cs.BLACK, (1, 21), mode))
                    band_plot(out / level / obs / (name + sfx), gr, (edges[0], edges[-1]), TITLE[(level, obs)], yt + per,
                              [line + "; signal pooled over lifetimes"] + (["Unit area"] if mode == "norm" else []),
                              {"Bkg, " + lab: cs.BLACK})
    rows = []
    for k, kind, n in [(k, "signal", k[5:]) for k in masses] + [(b, "background", n) for b, _, n in BKGS]:
        c = stored("%s/%s" % (kind, k))
        rows.append([n] + ["%d" % c[x] for x in ("events", "reco_events", "reco_undefined", "gen_events", "gen_entries", "gen_nojet", "gen_undefined")])
    caps.add("<reco|gen>/<dphi|dr>/bkg_<all|ZJets|WJets|QCD>[_norm|_frac].png", "Six mass points and one background. gen_events = events with a gen dimuon; gen_entries = signal events or background pairs, before the no-jet and undefined "
             "exclusions (counted in the same unit):", ["slice", "events", "reco_events", "reco_undefined", "gen_events", "gen_entries", "gen_nojet", "gen_undefined"], rows)
    print("CAPTIONS", caps.write(out))
    print("PLOTS DONE", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--flist-dir", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--workers", type=int, default=6)
    a = sub.add_parser("fill")
    for k in ("--signal-summary", "--bkg-summary", "--met-jet", "--signal-scan-dir", "--bkg-scan-dir", "--gen-scan-dir",
              "--producer-dir", "--output"):
        a.add_argument(k, required=True)
    a.add_argument("--only", help="regex on sample names; tests only (skips the coverage check)")
    b = sub.add_parser("plot")
    for k in ("--input", "--signal-summary", "--bkg-summary", "--met-jet", "--style-dir", "--out-dir"):
        b.add_argument(k, required=True)
    args = ap.parse_args()
    {"scan": scan, "fill": fill, "plot": plot}[args.cmd](args)


if __name__ == "__main__":
    main()
