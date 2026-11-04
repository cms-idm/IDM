#!/usr/bin/env python3
"""Dimuon distributions of passing pairs, at the end of the reco-only chain and with the MET > 80 GeV and leading-jet
pT > 80 GeV, |eta| < 2.4 cut, signal and background (kb/studies/bkg-reco-dimuons.md Part 4; LPC only).

python3 met_jet_dists.py fill --signal-summary recovery.root --bkg-summary bkg_reco_dimuons.root --met-jet met_jet_cut.root \
    --signal-scan-dir SIG_SCAN --bkg-scan-dir BKG_SCAN --output dists.root
python3 -I met_jet_dists.py plot --input dists.root --style-dir RUNDIR --out-dir RUNDIR/dists

fill (uproot, awkward): one pass over the saved producer outputs listed in each summary's 'inputs' (background: the
samples present in met_jet_cut.root). Pair stages from bkg_reco_dimuons.vector_stages; passing = stage 6; cut flag per
event from the producer pfMetPt and the Part 2 scan trees joined one-to-one on (run, lumi, event). Raw unweighted pair
counts in TH1D with explicit edges, flows kept. plot (PyROOT, cms_style): outlines, unnormalized and unit-area copies.
"""
import argparse
from pathlib import Path
import re
import sys

import numpy as np

MET_CUT, JET_PT_CUT, JET_ETA_CUT = 80., 80., 2.4
VARS = {  # name: (branch, edges, x title, y title per bin)
    "pt_pca": ("pPcaMuMuPt", np.arange(0, 200.001, 5), "p_{T}^{#mu#mu} at the PCA [GeV]", "5 GeV"),
    "pt_reco": ("pRecoMuMuPt", np.arange(0, 200.001, 5), "p_{T}^{#mu#mu} from the reco legs [GeV]", "5 GeV"),
    "lxy_short": ("pLxyBS", np.arange(0, 2.0001, 0.02), "Vertex L_{xy} to the beamspot [cm]", "0.02 cm"),
    "lxy_long": ("pLxyBS", np.arange(0, 300.001, 5), "Vertex L_{xy} to the beamspot [cm]", "5 cm"),
    "dr_pca": ("pPcaDR", np.arange(0, 0.9001, 0.02), "#DeltaR(#mu#mu) at the PCA", "0.02"),
    "dr_reco": ("pRecoDR", np.arange(0, 1.5001, 0.025), "#DeltaR(#mu#mu) from the reco legs", "0.025"),
    # user 2026-11-04: Lxy to 500 cm, pT(mumu) 0-25 GeV, plotted with the cut only (--vars ... --cut-only)
    "lxy_500": ("pLxyBS", np.arange(0, 500.001, 5), "Vertex L_{xy} to the beamspot [cm]", "5 cm"),
    "pt_pca_25": ("pPcaMuMuPt", np.arange(0, 25.0001, 0.5), "p_{T}^{#mu#mu} at the PCA [GeV]", "0.5 GeV"),
    "pt_reco_25": ("pRecoMuMuPt", np.arange(0, 25.0001, 0.5), "p_{T}^{#mu#mu} from the reco legs [GeV]", "0.5 GeV"),
}
# Per-leg variables (Part 5; need the producer rerun with pPca/pReco Pt/Eta/Phi A/B, via --producer-dir). Derived per
# pair in derived(): leading/subleading = larger/smaller leg pT; |deta| = |eta_A - eta_B|; |dphi| wrapped to [0, pi].
for _v, _t in (("pca", "at the PCA"), ("reco", "of the reco legs")):
    VARS["lead_pt_%s_50" % _v] = ("lead_pt_" + _v, np.arange(0, 50.0001, 1), "Leading muon p_{T} %s [GeV]" % _t, "1 GeV")
    VARS["lead_pt_%s_25" % _v] = ("lead_pt_" + _v, np.arange(0, 25.0001, 0.5), "Leading muon p_{T} %s [GeV]" % _t, "0.5 GeV")
    VARS["sub_pt_%s_25" % _v] = ("sub_pt_" + _v, np.arange(0, 25.0001, 0.5), "Subleading muon p_{T} %s [GeV]" % _t, "0.5 GeV")
    VARS["deta_%s" % _v] = ("deta_" + _v, np.arange(0, 1.2001, 0.02), "|#Delta#eta(#mu#mu)| %s" % _t, "0.02")
    VARS["dphi_%s" % _v] = ("dphi_" + _v, np.arange(0, 1.2001, 0.02), "|#Delta#phi(#mu#mu)| %s" % _t, "0.02")
LEG_VARS = [v for v in VARS if VARS[v][0].endswith(("_pca", "_reco")) and VARS[v][0].split("_")[0] in ("lead", "sub", "deta", "dphi")]
LEG_BRANCHES = ["p%s%s%s" % (v, q, l) for v in ("Pca", "Reco") for q in ("Pt", "Eta", "Phi") for l in "AB"]
DEFAULT_VARS = ["pt_pca", "pt_reco", "lxy_short", "lxy_long", "dr_pca", "dr_reco"]
SELECTIONS = {"selected": "Selected #mu#mu", "cut": "+ MET, jet cut"}
M_HEADER, M_KEYS = "M = m_{#chi} - #Deltam/2", ("#Deltam = 0.1 M", "0.2 M", "0.4 M")
CUM_TOP = 0.37
GROUPS = ["ZJets", "WJets", "QCD", "TTbar", "SingleTop", "DYJets", "Multiboson", "WWZ", "WZZ", "ZZ", "ZZZ", "ZGamma"]
PAIR_BRANCHES = ["pEvent", "pQA", "pQB", "pStatus", "pChi2", "pNdof", "pPcaOK", "pPcaDR", "pPcaMuMuPt", "pPcaMuMuPhi",
                 "pRecoDR", "pRecoMuMuPt", "pLxyBS"]


def keyed(r, l, e, what):
    keys = list(zip(r.tolist(), l.tolist(), e.tolist()))
    index = {k: i for i, k in enumerate(keys)}
    if len(index) != len(keys):
        raise RuntimeError("Duplicate (run, lumi, event) in " + what)
    return keys, index


def slices(kind, name):
    if kind == "signal":
        m = re.fullmatch(r"(Mchi-[\dp]+_dMchi-[\dp]+)_(ctau-[\dp]+)", name)
        if not m:
            raise RuntimeError("Unexpected signal sample " + name)
        return ["overall", "mass/" + m[1], "lifetime/" + m[2]]
    if name.split("_")[0] not in GROUPS:
        raise RuntimeError("Background sample outside the 12 groups: " + name)
    return ["overall", "group/" + name.split("_")[0]]


def derived(pr):
    """Per-pair leading/subleading pT, |deta|, |dphi| from the per-leg branches; NaN where a leg is a -999 sentinel
    (PCA legs of pairs without pPcaOK), so such pairs count as invalid, never as a fake 0."""
    out = {}
    for v, V in (("pca", "Pca"), ("reco", "Reco")):
        pa, pb, ea, eb, fa, fb = (pr["p%s%s%s" % (V, q, l)].astype(np.float64) for q in ("Pt", "Eta", "Phi") for l in "AB")
        bad = (pa == -999) | (pb == -999) | (ea == -999) | (eb == -999) | (fa == -999) | (fb == -999)
        d = fa - fb
        vals = {"lead_pt": np.maximum(pa, pb), "sub_pt": np.minimum(pa, pb), "deta": np.abs(ea - eb),
                "dphi": np.abs(d - 2 * np.pi * np.round(d / (2 * np.pi)))}
        for k, x in vals.items():
            x[bad] = np.nan
            out["%s_%s" % (k, v)] = x
    return out


def fill(args):
    import ROOT
    import uproot
    from bkg_reco_dimuons import vector_stages
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    ROOT.TH1.AddDirectory(False)
    hists, totals, invalid = {}, {}, {}
    mj = uproot.open(args.met_jet)
    included = {k.split("/")[2] for k in mj.keys(recursive=True) if k.startswith("background/sample/") and k.count("/") == 2}
    included = {k.split(";")[0] for k in included}
    expected = {kind: {k.split("/")[2].split(";")[0] for k in mj.keys(recursive=True)
                       if k.startswith(kind + "/sample/") and k.count("/") == 2} for kind in ("signal", "background")}
    processed = {"signal": [], "background": []}
    for kind, summary, scan_dir in (("signal", args.signal_summary, args.signal_scan_dir),
                                    ("background", args.bkg_summary, args.bkg_scan_dir)):
        sf = uproot.open(summary)
        inputs = [l.split() for l in str(sf["inputs"].member("fTitle")).splitlines() if l.strip()]
        for name, producer in inputs:
            if kind == "background" and name not in included:
                print("SKIP %s (not in met_jet_cut.root)" % name)
                continue
            processed[kind].append(name)
            if args.producer_dir:  # e.g. the per-leg rerun; same sample, same events (verified separately)
                producer = str(Path(args.producer_dir) / (name + ".root"))
            legs = any(v in LEG_VARS for v in args.vars)
            with uproot.open(producer) as f:
                if str(f["preselectedVertexing/sample"].member("fTitle")) != name:
                    raise RuntimeError("Producer sample mismatch: " + producer)
                ev = f["preselectedVertexing/events"].arrays(["run", "lumi", "event", "pfMetPt", "pfMetPhi"], library="np")
                pr = f["preselectedVertexing/pairs"].arrays(PAIR_BRANCHES + (LEG_BRANCHES if legs else []), library="np")
            if legs:
                pr.update(derived(pr))
            with uproot.open(Path(scan_dir) / (name + ".root")) as f:
                sc = f["scan"].arrays(["run", "lumi", "event", "pfMet", "jetPt", "jetEta"], library="np")
            pkeys, _ = keyed(ev["run"], ev["lumi"], ev["event"], "producer " + name)
            _, sindex = keyed(sc["run"], sc["lumi"], sc["event"], "scan " + name)
            if len(pkeys) != len(sindex) or any(k not in sindex for k in pkeys):
                raise RuntimeError("Incomplete producer-scan join for " + name)
            si = np.array([sindex[k] for k in pkeys], dtype=np.int64)
            met = ev["pfMetPt"].astype(np.float64)
            dmet = np.abs(sc["pfMet"][si] - met)
            if not np.all(np.isfinite(dmet)) or dmet.max() > 1e-3:
                raise RuntimeError("PF MET mismatch scan vs producer in " + name)
            with np.errstate(invalid="ignore"):
                cut = (met > MET_CUT) & (sc["jetPt"][si] > JET_PT_CUT) & (np.abs(sc["jetEta"][si]) < JET_ETA_CUT)
            passing = vector_stages(pr, ev["pfMetPt"], ev["pfMetPhi"]) == 6
            row = pr["pEvent"].astype(np.int64)
            sel = {"selected": passing, "cut": passing & cut[row]}
            ref = sf["sample/%s/counts" % name]
            c = dict(zip(ref.axis().labels(), ref.values()))
            m = mj["%s/sample/%s/counts" % (kind, name)]
            m = dict(zip(m.axis().labels(), m.values()))
            n_cut_events = len(np.unique(row[sel["cut"]]))
            if passing.sum() != c["candidates_met_angle"] or n_cut_events != m["selected_met_and_jet"]:
                raise RuntimeError("%s %s: passing pairs %d vs %d, cut events %d vs %d" % (
                    kind, name, passing.sum(), c["candidates_met_angle"], n_cut_events, m["selected_met_and_jet"]))
            print("SAMPLE %s %s passing pairs %d, with cut %d (events %d)" % (kind, name, passing.sum(), sel["cut"].sum(),
                                                                              n_cut_events))
            for s in slices(kind, name):
                for tag, mask in sel.items():
                    totals[(kind, s, tag)] = totals.get((kind, s, tag), 0) + int(mask.sum())
                    for var in args.vars:
                        branch, edges = VARS[var][:2]
                        key = (kind, s, tag, var)
                        if key not in hists:
                            hists[key] = ROOT.TH1D(var, "", len(edges) - 1, np.asarray(edges, dtype=np.float64))
                        x = pr[branch][mask].astype(np.float64)
                        ok = np.isfinite(x) & (x >= 0)  # producer sentinels are -999; all variables are >= 0
                        invalid[key] = invalid.get(key, 0) + int((~ok).sum())
                        x = x[ok]
                        if len(x):
                            hists[key].FillN(len(x), x, np.ones(len(x)))
    for kind, names in processed.items():
        if len(names) != len(set(names)) or set(names) != expected[kind]:
            raise RuntimeError("%s samples processed %d (unique %d) != met_jet_cut.root %d" % (
                kind, len(names), len(set(names)), len(expected[kind])))
    print("COVERAGE signal %d, background %d samples" % (len(processed["signal"]), len(processed["background"])))
    coverage = "signal %d background %d" % (len(processed["signal"]), len(processed["background"]))
    root = ROOT.TFile.Open(str(out), "CREATE")
    ROOT.TNamed("coverage", coverage).Write()
    ROOT.TNamed("definitions", "Raw unweighted pair counts of passing pairs (reco-only chain stage 6), pair-level. "
                "selected = all passing pairs; cut = passing pairs in events with Type-1 PF MET > 80 GeV and highest-pT "
                "slimmedJets jet pT > 80 GeV, |eta| < 2.4. Non-finite or negative values not filled (counted in the fill log). Variables: " + ", ".join(
                    "%s=%s" % (k, VARS[k][0]) for k in args.vars) + ". Flows kept. Inputs: %s, %s, %s, scans %s, %s" % (
                    args.signal_summary, args.bkg_summary, args.met_jet, args.signal_scan_dir, args.bkg_scan_dir)).Write()
    for (kind, s, tag, var), h in sorted(hists.items()):
        if abs(h.Integral(0, h.GetNbinsX() + 1) + invalid[(kind, s, tag, var)] - totals[(kind, s, tag)]) > 0.5:
            raise RuntimeError("Histogram integral mismatch %s %s %s %s" % (kind, s, tag, var))
        d = root
        for part in [kind] + s.split("/") + [tag]:
            d = d.GetDirectory(part) or d.mkdir(part)
        d.cd()
        h.Write()
    root.Close()
    for (kind, s, tag), n in sorted(totals.items()):
        if s == "overall" or s.startswith("group/"):
            print("TOTAL %s %s %s pairs %d" % (kind, s, tag, n))
    bad = {k: v for k, v in invalid.items() if v}
    print("INVALID (non-finite or < 0, not filled): %s" % (", ".join("%s=%d" % ("/".join(k), v) for k, v in sorted(bad.items()))
                                                          or "none"))
    print("FILL DONE", out)


def plot(args):
    import ROOT
    sys.path.insert(0, args.style_dir)
    import cms_style as cs
    cs.setup(era="2022EE (13.6 TeV)", status="Simulation Private Work", norm_label="Unit area")
    ROOT.gErrorIgnoreLevel = ROOT.kWarning
    out = Path(args.out_dir)
    if out.exists():
        raise FileExistsError(out)
    f = ROOT.TFile.Open(args.input)
    get = lambda k: f.Get(k) or (_ for _ in ()).throw(RuntimeError("Missing " + k))
    names = lambda d: sorted(k.GetName() for k in f.Get(d).GetListOfKeys())
    taus = sorted(("lifetime/" + n for n in names("signal/lifetime")), key=lambda k: float(k.split("-")[1]))

    def mkey(k):
        mchi, dm = (float(x.replace("p", ".")) for x in re.fullmatch(r"mass/Mchi-([\dp]+)_dMchi-([\dp]+)", k).groups())
        return round(mchi - dm / 2, 6), dm
    masses = sorted(("mass/" + n for n in names("signal/mass")), key=mkey)
    num = lambda x: x.replace("p", ".")
    label = {"overall": "all 72 samples", **{k: "c#tau = %s mm" % k.split("-")[1] for k in taus},
             **{k: "m_{#chi} / #Deltam = %s / %s GeV" % tuple(num(x) for x in re.fullmatch(
                 r"mass/Mchi-([\dp]+)_dMchi-([\dp]+)", k).groups()) for k in masses}}
    B_SEL, B_CUT = cs.OPEN_DASHED, cs.FILLED_SOLID
    cov = f.Get("coverage")
    if not cov:
        raise RuntimeError("Input lacks the coverage record (refill with this version)")
    n_bkg = int(cov.GetTitle().split()[-1])
    BKGN = "Bkg (all 50)" if n_bkg == 50 else "Bkg (%d of 50)" % n_bkg
    norm_line = lambda norm: [cs.STYLE["norm_label"]] if norm else []
    upcase = lambda t: t if t.startswith("pT") else t[0].upper() + t[1:]
    nbkg_text = "all 50" if n_bkg == 50 else "%d of the 50" % n_bkg
    plain = {"pt_pca": "pT(mumu) at the PCA (pPcaMuMuPt, the vector sum of the two muon momenta propagated to the vertex)",
             "pt_reco": "pT(mumu) from the reco legs (pRecoMuMuPt: the two paired component tracks at their reference "
                        "point)",
             "lxy_short": "vertex Lxy to the beamspot (pLxyBS), 0-2 cm view", "lxy_long": "vertex Lxy, 0-300 cm view",
             "lxy_500": "vertex Lxy, 0-500 cm view", "pt_pca_25": "pT(mumu) at the PCA, 0-25 GeV view",
             "pt_reco_25": "pT(mumu) from the reco legs, 0-25 GeV view",
             "dr_pca": "dR between the two muons at the PCA (pPcaDR, the < 0.9 cut variable)",
             "dr_reco": "dR between the two reco legs (pRecoDR)"}
    for v, t in (("pca", "at the PCA (leg momenta propagated to the vertex)"),
                 ("reco", "of the reco legs (paired component tracks at their reference point)")):
        plain["lead_pt_%s_50" % v] = "leading (higher-pT) muon pT %s, 0-50 GeV view" % t
        plain["lead_pt_%s_25" % v] = "leading muon pT %s, 0-25 GeV view" % t
        plain["sub_pt_%s_25" % v] = "subleading (lower-pT) muon pT %s, 0-25 GeV view" % t
        plain["deta_%s" % v] = "|deta| between the two muons %s" % t
        plain["dphi_%s" % v] = "|dphi| between the two muons %s, wrapped to [0, pi]" % t
    caps = cs.Captions("Dimuon distributions of passing pairs (kb/studies/bkg-reco-dimuons.md Part 4)", [
        "Population: every pair passing the full reco-only chain (eligible pair, opposite sign, valid vertex, "
        "chi2/ndof < 4, PCA valid, PCA dR < 0.9, |dphi(Type-1 PF MET, mumu PCA)| < 0.5), pair-level: every passing pair "
        "of an event enters. Raw MC pair counts, unweighted.",
        "Samples: signal = the 72 iDM 2022EE samples (2026-09-28 production); background = %s 2022EE "
        "background MC samples. A c-tau slice pools the 18 masses, a mass slice pools the 4 lifetimes, all unweighted; "
        "'Bkg' is the pooled background." % nbkg_text,
        "Selections: 'selected' = all passing pairs; '+ MET, jet' = passing pairs in events with Type-1 PF MET > 80 GeV "
        "and highest-pT slimmedJets jet pT > 80 GeV, |eta| < 2.4 (no jet ID, no muon cleaning; no jet fails).",
        "Variables here: " + "; ".join("%s = %s, %g-%g in bins of %s" % (v, plain[v], VARS[v][1][0], VARS[v][1][-1],
                                                                         VARS[v][3]) for v in args.vars) + ".",
        "Unit-area copies (*_norm.png, 'Unit area'): normalized within the plotted range, flows excluded, so they "
        "compare the fraction of each curve inside the window; the in-range fractions are tabulated below where they "
        "matter. Under- and overflow are kept in the ROOT file.",
        "Mass overlays: colour = mass family M = m_chi - dm/2 (5, 10, 20, 30, 40, 50 GeV); line style = dm = 0.1, 0.2 "
        "or 0.4 x M."])
    tot = lambda h: h.Integral(0, h.GetNbinsX() + 1)

    def band_plot(base, graphs, xtitle, ytitle, lines, groups, stages, header, ymax=None, log=None, top=0.34):
        """Band layout with the keys placed below the context lines (as in met_jet_plots.py)."""
        c = cs.canvas("c_" + str(base).replace("/", "_"), band=True)
        c.SetTopMargin(top)
        hi = max([p[3] for p in cs._points(graphs)] or [1])
        cs.frame(c, edges[0], edges[-1], 0, ymax or cs._round_up(1.1 * hi), xtitle, ytitle)
        c.Update()
        for g in graphs:
            g.Draw(cs._info[id(g)][1] + " same")
        t = cs.header(c, 0.15, 0.95, (0.04, 0.031), 0.96)
        t.SetTextSize(0.028); t.SetTextAlign(13)
        for i, line in enumerate(lines):
            t.DrawLatex(0.18, 0.90 - 0.035 * i, line)
        ytop = 0.90 - 0.035 * len(lines) - 0.005
        cs.legend([(cs._proxy(col, *groups_style.get(lab, (1, 20))), lab, "l") for lab, col in groups.items()], y2=ytop,
                  y1=1 - top + 0.01, x1=0.50, header=header, ncol=2, size=0.026)
        if stages:
            # lines only: the distributions are step outlines without markers
            cs.legend([(cs._proxy(cs.BLACK, ls, ms), lab, "l") for lab, (ls, ms) in stages.items()], y2=ytop,
                      y1=1 - top + 0.01, x1=0.17, x2=0.47, size=0.026)
        cs.save(c, base, log=args.log if log is None else log)
    groups_style = {}  # legend line style per group label (default solid)

    def checked(prefix):
        """All variables stored under prefix: zero underflow and one common total (all passing pairs; no invalid
        values), so in-window fractions and cumulative fractions use the full passing-pair denominator."""
        hs = {v: f.Get(prefix + "/" + v) for v in VARS if f.Get(prefix + "/" + v)}
        totals = {v: tot(h) for v, h in hs.items()}
        if any(h.GetBinContent(0) != 0 for h in hs.values()) or len(set(totals.values())) != 1:
            raise RuntimeError("Closure failed for %s: totals %s, underflow %s" % (
                prefix, totals, {v: h.GetBinContent(0) for v, h in hs.items()}))
        return hs

    def in_range(prefix, var):
        h = checked(prefix)[var]
        return 100. * h.Integral(1, h.GetNbinsX()) / tot(h)

    def cum_graph(key, color, style):
        """Fraction of all passing pairs (incl. flows) with Lxy >= x at each lower bin edge; at the last edge the
        overflow alone. 68% Clopper-Pearson under the independent-pair assumption."""
        h = checked(key.rsplit("/", 1)[0])[key.rsplit("/", 1)[1]]
        nb, n = h.GetNbinsX(), int(tot(h))
        gr = ROOT.TGraphAsymmErrors()
        for i in range(1, nb + 2):
            k = int(h.Integral(i, nb + 1)); x = h.GetXaxis().GetBinLowEdge(i)
            lo = ROOT.TEfficiency.ClopperPearson(n, k, cs.STYLE["level"], False)
            up = ROOT.TEfficiency.ClopperPearson(n, k, cs.STYLE["level"], True)
            gr.SetPoint(i - 1, x, k / n); gr.SetPointError(i - 1, 0, 0, k / n - lo, up - k / n)
        g = cs._style(gr, color, style[0], style[1], [h.GetXaxis().GetXmin(), h.GetXaxis().GetXmax()], opt="LP",
                      kind="eff")
        g.SetMarkerSize(0.5)
        return g

    def cumulative(edges, xtitle):
        """Cumulative Lxy after the cut (user 2026-11-04): c-tau overlay + background, and overall signal vs background."""
        yt = "Fraction with L_{xy} #geq x"
        xt = "L_{xy} threshold x [cm]"
        lines = ["+ MET, jet cut; passing #mu#mu pairs", "Denominator: all passing pairs"]
        bstyle = (2, 24)
        series = [cum_graph("signal/%s/cut/lxy_500" % k, c, (1, 20 + i)) for i, (k, c) in enumerate(zip(taus, cs.COLORS))]
        series.append(cum_graph("background/overall/cut/lxy_500", cs.BLACK, bstyle))
        groups = {label[k]: c for k, c in zip(taus, cs.COLORS)}
        groups[BKGN] = cs.BLACK
        groups_style.clear(); groups_style[BKGN] = bstyle
        band_plot(out / "lxy_500_cumulative" / "overlay_ctau", series, xt, yt, lines, groups, {},
                  "Signal, pooled over masses", ymax=1.05, log=False, top=CUM_TOP)
        series = [cum_graph("signal/overall/cut/lxy_500", cs.BLUE, (1, 20)),
                  cum_graph("background/overall/cut/lxy_500", cs.BLACK, bstyle)]
        band_plot(out / "lxy_500_cumulative" / "overall_signal_bkg", series, xt, yt, lines,
                  {"Signal, all 72 samples": cs.BLUE, BKGN: cs.BLACK}, {}, None, ymax=1.05, log=False,
                  top=CUM_TOP)
        groups_style.clear()
        rows = []
        for k in ["overall"] + taus + ["background"]:
            key = "background/overall/cut/lxy_500" if k == "background" else "signal/%s/cut/lxy_500" % k
            g = cum_graph(key, cs.BLUE, (1, 20))
            n = int(tot(checked(key.rsplit("/", 1)[0])["lxy_500"]))
            rows.append([k, "%d" % n] + ["%.3f" % g.GetPointY(i) for i in (1, 2, 10, 20, 100)])
            print("cumulative Lxy >= 50 cm, %s: %.4f" % (k, g.GetPointY(10)))
        caps.add("lxy_500_cumulative/<overlay_ctau|overall_signal_bkg>.png",
                 "Fraction of all passing pairs (+ MET, jet) with Lxy >= x, per lower bin edge x of the 0-500 cm "
                 "histogram; the last point is the overflow beyond 500 cm alone. Denominator includes all flows. "
                 "Bars: CP 68% assuming independent pairs (pairs of one event are correlated; coverage uncalibrated).",
                 ["slice", "pairs", ">= 5 cm", ">= 10 cm", ">= 50 cm", ">= 100 cm", "> 500 cm"], rows)

    for var in args.vars:
        _, edges, xtitle, per = VARS[var]
        for norm in (False, True):
            yt = ("Fraction of pairs / " if norm else "Pairs / ") + per
            sfx = "_norm" if norm else ""
            hg = lambda key, color, style: cs.hist_graph(get(key), color, style[0], style[1], step=True, norm=norm)
            for s in ["overall"] + taus + masses:
                d = "overall" if s == "overall" else s
                if args.cut_only:
                    series = [hg("signal/%s/cut/%s" % (s, var), cs.BLUE, B_CUT),
                              hg("background/overall/cut/%s" % var, cs.BLACK, B_SEL)]
                    cs.plot(out / var / d / ("signal_bkg" + sfx), series, ["Signal, + MET, jet", BKGN + ", + MET, jet"],
                            xtitle, yt, lines=["Signal: " + label[s]] + norm_line(norm), x1=0.58, log=args.log)
                    continue
                series = [hg("signal/%s/selected/%s" % (s, var), cs.BLUE, B_SEL), hg("signal/%s/cut/%s" % (s, var), cs.BLUE, B_CUT),
                          hg("background/overall/selected/%s" % var, cs.BLACK, B_SEL),
                          hg("background/overall/cut/%s" % var, cs.BLACK, B_CUT)]
                cs.plot(out / var / d / ("signal_bkg" + sfx), series,
                        ["Signal, selected", "Signal, + MET, jet", "Bkg, selected", "Bkg, + MET, jet"], xtitle, yt,
                        lines=["Signal: " + label[s]] + norm_line(norm), x1=0.62, log=args.log)
            for tag, sl in SELECTIONS.items():
                if args.cut_only and tag != "cut":
                    continue
                series = [hg("background/overall/%s/%s" % (tag, var), cs.BLACK, B_CUT)]
                series += [hg("background/group/%s/%s/%s" % (g, tag, var), c, B_CUT) for g, c in
                           (("ZJets", cs.BLUE), ("WJets", cs.ORANGE), ("QCD", cs.GREEN))]
                cs.plot(out / var / "background" / (tag + sfx), series, ["All %d" % n_bkg, "ZJets", "WJets", "QCD"], xtitle, yt,
                        lines=["Background, " + sl] + norm_line(norm), x1=0.66, log=args.log)
            series = [hg("signal/%s/cut/%s" % (k, var), c, B_CUT) for k, c in zip(taus, cs.COLORS)]
            series.append(hg("background/overall/cut/%s" % var, cs.BLACK, B_SEL))
            cs.plot(out / var / "overlays" / ("ctau" + sfx), series,
                    ["Signal, " + label[k] for k in taus] + [BKGN], xtitle, yt,
                    lines=["Signal pooled over masses"] + norm_line(norm), x1=0.60, log=args.log)
            fam = sorted({mkey(k)[0] for k in masses})
            fstyle = [(3, 25), (2, 24), (1, 20)]
            series = [hg("signal/%s/cut/%s" % (k, var), cs.COLORS[fam.index(mkey(k)[0])],
                         fstyle[[0.1, 0.2, 0.4].index(round(mkey(k)[1] / mkey(k)[0], 3))]) for k in masses]
            series.append(hg("background/overall/cut/%s" % var, cs.BLACK, (1, 21)))
            groups = {"%g GeV" % m: cs.COLORS[i] for i, m in enumerate(fam)}
            groups[BKGN] = cs.BLACK
            band_plot(out / var / "overlays" / ("mass" + sfx), series, xtitle, yt,
                      ["Signal pooled over lifetimes"] + norm_line(norm),
                      groups, dict(zip(M_KEYS, fstyle)), M_HEADER)
        sels = ["cut"] if args.cut_only else ["selected", "cut"]
        rows = [[label[s].replace("#tau", "tau").replace("m_{#chi} / #Deltam = ", "")] + ["%.1f%%" % in_range("signal/%s/%s" % (s, sel), var) for sel in sels]
                for s in ["overall"] + taus + masses]
        rows.append([BKGN] + ["%.1f%%" % in_range("background/overall/" + sel, var) for sel in sels])
        caps.add("%s/<slice>/signal_bkg[_norm].png" % var,
                 "%s for one signal slice (<slice> = overall, lifetime/ctau-<1|10|100|1000> or mass/Mchi-<m>_dMchi-<dm>) "
                 "and the pooled background, %s. Fraction of each curve's passing pairs inside the plotted range "
                 "(the unit-area copies are normalized to this part):" % (upcase(plain[var]),
                                                                         "+ MET, jet only" if args.cut_only else
                                                                         "selected and + MET, jet"),
                 ["signal slice / bkg"] + ["in range, " + ("selected" if s == "selected" else "+ MET, jet") for s in sels],
                 rows)
        caps.add("%s/background/<%s>[_norm].png" % (var, "|".join(sels)),
                 "%s for the pooled background and its three largest groups (ZJets, WJets, QCD), one plot per selection."
                 % (upcase(plain[var])))
        caps.add("%s/overlays/<ctau|mass>[_norm].png" % var,
                 "%s, + MET, jet (passing pairs in events with MET > 80 GeV and jet pT > 80 GeV; the plots themselves do not "
                 "label the cut): the four lifetimes (each pooled over masses), or the 18 mass points (each pooled over "
                 "lifetimes), with the pooled background." % (upcase(plain[var])))
        if var == "lxy_500":
            cumulative(edges, xtitle)
    print("CAPTIONS", caps.write(out))
    print("PLOTS DONE", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("fill")
    for k in ("--signal-summary", "--bkg-summary", "--met-jet", "--signal-scan-dir", "--bkg-scan-dir", "--output"):
        a.add_argument(k, required=True)
    a.add_argument("--vars", nargs="+", default=DEFAULT_VARS, choices=list(VARS))
    a.add_argument("--producer-dir", help="read <dir>/<sample>.root instead of the producer paths in the summaries")
    b = sub.add_parser("plot")
    for k in ("--input", "--style-dir", "--out-dir"):
        b.add_argument(k, required=True)
    b.add_argument("--log", action="store_true", help="add log-y copies (only on the user's approval)")
    b.add_argument("--vars", nargs="+", default=DEFAULT_VARS, choices=list(VARS))
    b.add_argument("--cut-only", action="store_true", help="only the + MET, jet cut selection (signal vs background)")
    args = ap.parse_args()
    (fill if args.cmd == "fill" else plot)(args)


if __name__ == "__main__":
    main()
