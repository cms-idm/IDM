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
DEFAULT_VARS = ["pt_pca", "pt_reco", "lxy_short", "lxy_long", "dr_pca", "dr_reco"]
SELECTIONS = {"selected": "Selected #mu#mu", "cut": "+ MET, jet p_{T} > 80 GeV"}
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
            with uproot.open(producer) as f:
                if str(f["preselectedVertexing/sample"].member("fTitle")) != name:
                    raise RuntimeError("Producer sample mismatch: " + producer)
                ev = f["preselectedVertexing/events"].arrays(["run", "lumi", "event", "pfMetPt", "pfMetPhi"], library="np")
                pr = f["preselectedVertexing/pairs"].arrays(PAIR_BRANCHES, library="np")
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
    root = ROOT.TFile.Open(str(out), "CREATE")
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
    cs.setup(era="2022EE (13.6 TeV)", status="Simulation Private Work",
             norm_label="Unit area in plot range")
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
             **{k: "%s / %s GeV" % tuple(num(x) for x in re.fullmatch(
                 r"mass/Mchi-([\dp]+)_dMchi-([\dp]+)", k).groups()) for k in masses}}
    B_SEL, B_CUT = cs.OPEN_DASHED, cs.FILLED_SOLID
    pairs = "Passing #mu#mu pairs, unweighted"
    bkg = "Bkg: 47 of 50 samples"

    def band_plot(base, graphs, xtitle, ytitle, lines, groups, stages, header):
        """Band layout with the keys placed below the context lines (as in met_jet_plots.py)."""
        c = cs.canvas("c_" + str(base).replace("/", "_"), band=True)
        c.SetTopMargin(0.34)
        hi = max([p[3] for p in cs._points(graphs)] or [1])
        cs.frame(c, edges[0], edges[-1], 0, cs._round_up(1.1 * hi), xtitle, ytitle)
        c.Update()
        for g in graphs:
            g.Draw(cs._info[id(g)][1] + " same")
        t = cs.header(c, 0.15, 0.95, (0.04, 0.031), 0.96)
        t.SetTextSize(0.028); t.SetTextAlign(13)
        for i, line in enumerate(lines):
            t.DrawLatex(0.18, 0.90 - 0.035 * i, line)
        ytop = 0.90 - 0.035 * len(lines) - 0.005
        cs.legend([(cs._proxy(col, 1, 20), lab, "l") for lab, col in groups.items()], y2=ytop, y1=0.67, x1=0.50,
                  header=header, ncol=2, size=0.026)
        cs.legend([(cs._proxy(cs.BLACK, ls, ms), lab, "lp") for lab, (ls, ms) in stages.items()], y2=ytop, y1=0.67,
                  x1=0.17, x2=0.47, size=0.026)
        cs.save(c, base, log=args.log)

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
                    cs.plot(out / var / d / ("signal_bkg" + sfx), series, ["Signal", "Background"], xtitle, yt,
                            lines=["Signal: " + label[s], bkg, "+ MET, jet p_{T} > 80 GeV", pairs], x1=0.68, log=args.log)
                    continue
                series = [hg("signal/%s/selected/%s" % (s, var), cs.BLUE, B_SEL), hg("signal/%s/cut/%s" % (s, var), cs.BLUE, B_CUT),
                          hg("background/overall/selected/%s" % var, cs.BLACK, B_SEL),
                          hg("background/overall/cut/%s" % var, cs.BLACK, B_CUT)]
                cs.plot(out / var / d / ("signal_bkg" + sfx), series,
                        ["Signal, selected", "Signal, + MET, jet", "Bkg, selected", "Bkg, + MET, jet"], xtitle, yt,
                        lines=["Signal: " + label[s], bkg, pairs], x1=0.62, log=args.log)
            for tag, sl in SELECTIONS.items():
                if args.cut_only and tag != "cut":
                    continue
                series = [hg("background/overall/%s/%s" % (tag, var), cs.BLACK, B_CUT)]
                series += [hg("background/group/%s/%s/%s" % (g, tag, var), c, B_CUT) for g, c in
                           (("ZJets", cs.BLUE), ("WJets", cs.ORANGE), ("QCD", cs.GREEN))]
                cs.plot(out / var / "background" / (tag + sfx), series, ["All 47", "ZJets", "WJets", "QCD"], xtitle, yt,
                        lines=[bkg, sl, pairs], x1=0.66, log=args.log)
            series = [hg("signal/%s/cut/%s" % (k, var), c, B_CUT) for k, c in zip(taus, cs.COLORS)]
            series.append(hg("background/overall/cut/%s" % var, cs.BLACK, B_SEL))
            cs.plot(out / var / "overlays" / ("ctau" + sfx), series,
                    ["Signal, " + label[k] for k in taus] + ["Bkg (47 of 50)"], xtitle, yt,
                    lines=["+ MET, jet cut", "Signal pooled over masses", bkg, pairs], x1=0.60, log=args.log)
            fam = sorted({mkey(k)[0] for k in masses})
            fstyle = [(3, 25), (2, 24), (1, 20)]
            series = [hg("signal/%s/cut/%s" % (k, var), cs.COLORS[fam.index(mkey(k)[0])],
                         fstyle[[0.1, 0.2, 0.4].index(round(mkey(k)[1] / mkey(k)[0], 3))]) for k in masses]
            series.append(hg("background/overall/cut/%s" % var, cs.BLACK, (1, 21)))
            groups = {"%g GeV" % m: cs.COLORS[i] for i, m in enumerate(fam)}
            groups["Bkg (47 of 50)"] = cs.BLACK
            band_plot(out / var / "overlays" / ("mass" + sfx), series, xtitle, yt,
                      ["+ MET, jet cut; signal pooled over lifetimes; " + bkg,
                       pairs + ("; " + cs.STYLE["norm_label"] if norm else "")],
                      groups, {"#Deltam = 0.1 m": fstyle[0], "0.2 m": fstyle[1], "0.4 m": fstyle[2]},
                      "m_{#chi} - #Deltam/2")
    print("PLOTS DONE", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("fill")
    for k in ("--signal-summary", "--bkg-summary", "--met-jet", "--signal-scan-dir", "--bkg-scan-dir", "--output"):
        a.add_argument(k, required=True)
    a.add_argument("--vars", nargs="+", default=DEFAULT_VARS, choices=list(VARS))
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
