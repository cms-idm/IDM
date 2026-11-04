#!/usr/bin/env python3
"""Six mass points (M = 5, 50 GeV; dm = 0.1, 0.2, 0.4 M; pooled over c-tau) vs all background, ZJets, WJets, QCD: cutflow,
F, distributions, threshold scans of F and pT-Lxy maps (kb/studies/bkg-reco-dimuons.md Part 6; LPC only).

python3 met_jet_masspoints.py fill --signal-summary recovery.root --bkg-summary bkg_reco_dimuons.root \
    --met-jet met_jet_cut_50.root --signal-scan-dir SIG_SCAN --bkg-scan-dir BKG_SCAN --producer-dir INPUTS --output mp.root
python3 -I met_jet_masspoints.py plot --input mp.root --signal-summary ... --bkg-summary ... --met-jet ... \
    --style-dir RUNDIR --out-dir RUNDIR/plots [--views ...] [--scans ...] [--maps ...]

fill (uproot, awkward): one pass over the producer outputs. Raw unweighted counts in ROOT, flows kept: pair histograms
per view and TH2 pT vs Lxy, one entry per event from its representative pair (lowest chi2/ndof, then lowest pair index;
Part 7; the Part 6 pair-level version is commit d6802a3); event histograms of the per-event max and min over passing pairs
(threshold scans); |dphi(leading jet, MET)| per event. Every threshold pass from max/min is compared event by event with
the direct "any passing pair passes" predicate.
plot (PyROOT, cms_style): linear and log copies; no change to the stored histograms.
"""
import argparse
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sibling scripts, also under python3 -I
import numpy as np

from met_jet_dists import JET_ETA_CUT, JET_PT_CUT, LEG_BRANCHES, MET_CUT, PAIR_BRANCHES, derived, keyed

MASSES = ["Mchi-5p25_dMchi-0p5", "Mchi-5p5_dMchi-1p0", "Mchi-6p0_dMchi-2p0",
          "Mchi-52p5_dMchi-5p0", "Mchi-55p0_dMchi-10p0", "Mchi-60p0_dMchi-20p0"]
GROUPS = ["ZJets", "WJets", "QCD"]
TAGS = ("selected", "cut")
STAGE = {"selected": "Selected", "cut": "Selected, after cut"}
PT_T, LEAD_T, SUB_T, LXY_T = ("p_{T}^{#mu#mu} from the reco legs [GeV]", "Leading muon p_{T} of the reco legs [GeV]",
                              "Subleading muon p_{T} of the reco legs [GeV]", "Vertex L_{xy} to the beamspot [cm]")
VIEWS = {  # name: (variable, edges, x title, per-bin text)
    "pt_200": ("pt", np.arange(0, 200.001, 5), PT_T, "5 GeV"),
    "pt_25": ("pt", np.arange(0, 25.0001, 0.5), PT_T, "0.5 GeV"),
    "lead_50": ("lead", np.arange(0, 50.0001, 1), LEAD_T, "1 GeV"),
    "sub_25": ("sub", np.arange(0, 25.0001, 0.5), SUB_T, "0.5 GeV"),
    "lxy_500": ("lxy", np.arange(0, 500.001, 5), LXY_T, "5 cm"),
    "lxy_2": ("lxy", np.arange(0, 2.0001, 0.02), LXY_T, "0.02 cm"),
}
SCANS = {  # name: (variable, edges = thresholds, x title)
    "pt": ("pt", np.arange(0, 100.001, 1), "Threshold on " + PT_T),
    "lead": ("lead", np.arange(0, 100.001, 1), "Threshold on l" + LEAD_T[1:]),
    "sub": ("sub", np.arange(0, 50.0001, 0.5), "Threshold on s" + SUB_T[1:]),
    "lxy_500": ("lxy", np.arange(0, 500.001, 5), "Threshold on v" + LXY_T[1:]),
    "lxy_2": ("lxy", np.arange(0, 2.0001, 0.02), "Threshold on v" + LXY_T[1:]),
}
MAPS = {"map_500": np.arange(0, 500.001, 10), "map_2": np.arange(0, 2.0001, 0.1)}  # Lxy edges
MAP_PT = np.arange(0, 100.001, 2.5)
VARS = ("pt", "lead", "sub", "lxy")
DPHI_EDGES = np.linspace(0, np.pi, 33)
DPHI_T = "|#Delta#phi(leading jet, MET)|"
MIN_BKG_RATIO = 10  # scan points with fewer pooled-background events are left out of the F(x)/F(reference) table


def sample_slices(kind, name):
    if kind == "signal":
        mass = re.fullmatch(r"(Mchi-[\dp]+_dMchi-[\dp]+)_ctau-[\dp]+", name)[1]
        return ["mass/" + mass] if mass in MASSES else []
    g = name.split("_")[0]
    return ["overall"] + (["group/" + g] if g in GROUPS else [])


def representative(idx, row, chi):
    """Pair indices, one per event: among pairs idx, lowest chi (chi2/ndof), then lowest pair index. Verified against an
    independent per-event loop."""
    order = idx[np.lexsort((idx, chi[idx], row[idx]))]
    rep = order[np.r_[True, row[order][1:] != row[order][:-1]]] if len(order) else order
    best = {}
    for i in idx.tolist():
        k = (chi[i], i)
        if row[i] not in best or k < best[row[i]]:
            best[row[i]] = k
    if sorted(rep.tolist()) != sorted(v[1] for v in best.values()):
        raise RuntimeError("Representative pair differs from the independent minimum")
    return rep


def threshold_check(name, nev, rows, x, emax, emin, thresholds):
    """Direct predicate vs the max/min pass, per event, including equality at actual values."""
    for t in thresholds:
        up, dn = np.zeros(nev, bool), np.zeros(nev, bool)
        up[rows[x >= t]] = True
        dn[rows[x < t]] = True
        if not np.array_equal(up, emax >= t) or not np.array_equal(dn, emin < t):
            raise RuntimeError("Threshold pass differs from the direct predicate: %s at %r" % (name, t))


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

    def hist(key, edges, edges2=None):
        if key not in hists:
            e = np.asarray(edges, dtype=np.float64)
            hists[key] = ROOT.TH1D(key[-1], "", len(e) - 1, e) if edges2 is None else ROOT.TH2D(
                key[-1], "", len(e) - 1, e, len(edges2) - 1, np.asarray(edges2, dtype=np.float64))
        return hists[key]

    for kind, summary, scan_dir in (("signal", args.signal_summary, args.signal_scan_dir),
                                    ("background", args.bkg_summary, args.bkg_scan_dir)):
        sf = uproot.open(summary)
        for name, _ in (l.split() for l in str(sf["inputs"].member("fTitle")).splitlines() if l.strip()):
            if name not in expected[kind] or (args.only and not re.search(args.only, name)):
                continue
            processed[kind].add(name)
            with uproot.open(Path(args.producer_dir) / (name + ".root")) as f:
                if str(f["preselectedVertexing/sample"].member("fTitle")) != name:
                    raise RuntimeError("Producer sample mismatch: " + name)
                ev = f["preselectedVertexing/events"].arrays(["run", "lumi", "event", "pfMetPt", "pfMetPhi"], library="np")
                pr = f["preselectedVertexing/pairs"].arrays(PAIR_BRANCHES + LEG_BRANCHES, library="np")
            pr.update(derived(pr))
            with uproot.open(Path(scan_dir) / (name + ".root")) as f:
                sc = f["scan"].arrays(["run", "lumi", "event", "pfMet", "pfMetPhi", "nJet", "jetPt", "jetEta", "jetPhi"], library="np")
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
            ref = dict(zip(ref.axis().labels(), ref.values()))
            m = mj["%s/sample/%s/counts" % (kind, name)]
            m = dict(zip(m.axis().labels(), m.values()))
            nev = len(pkeys)
            n_ev = {t: len(np.unique(row[mask])) for t, mask in sel.items()}
            if (passing.sum(), n_ev["selected"], n_ev["cut"], nev) != (
                    ref["candidates_met_angle"], m["selected"], m["selected_met_and_jet"], m["events"]):
                raise RuntimeError("%s %s: counts differ from the summaries" % (kind, name))
            print("SAMPLE %s %s events %d selected %d cut %d pairs %d / %d" % (
                kind, name, nev, n_ev["selected"], n_ev["cut"], sel["selected"].sum(), sel["cut"].sum()))
            values = {"pt": pr["pRecoMuMuPt"].astype(np.float64), "lead": pr["lead_pt_reco"], "sub": pr["sub_pt_reco"],
                      "lxy": pr["pLxyBS"].astype(np.float64)}
            per_slice = [("events", nev)]  # (name, x) with counts, once per sample
            chi = pr["pChi2"].astype(np.float64) / pr["pNdof"].astype(np.float64)
            if not np.all(np.isfinite(chi[passing])):
                raise RuntimeError("Non-finite chi2/ndof on a passing pair in " + name)
            rep_sel = representative(np.flatnonzero(passing), row, chi)
            if not set(representative(np.flatnonzero(sel["cut"]), row, chi).tolist()) <= set(rep_sel.tolist()):
                raise RuntimeError("Cut representative differs from the selected one in " + name)
            jphi, mphi = sc["jetPhi"][si].astype(np.float64), ev["pfMetPhi"].astype(np.float64)
            if np.abs(np.remainder(sc["pfMetPhi"][si] - mphi + np.pi, 2 * np.pi) - np.pi).max() > 1e-3:
                raise RuntimeError("PF MET phi mismatch scan vs producer in " + name)
            has_jet = sc["nJet"][si] > 0
            defined = has_jet & np.isfinite(jphi) & np.isfinite(mphi) & (met > 0)
            d = jphi - mphi
            dphi = np.abs(d - 2 * np.pi * np.round(d / (2 * np.pi)))
            cut_ev = np.zeros(nev, bool)
            cut_ev[row[sel["cut"]]] = True
            if not defined[cut_ev].all():
                raise RuntimeError("Undefined dphi in a selected event after cuts: " + name)
            per_slice += [("no_jet", int((~has_jet).sum())), ("dphi_undefined", int((has_jet & ~defined).sum()))]
            for s in sample_slices(kind, name):
                for tag, x in (("all", dphi[defined]), ("cut", dphi[cut_ev])):
                    h = hist((kind, s, tag, "dphi_jet_met"), DPHI_EDGES)
                    if len(x):
                        h.FillN(len(x), x, np.ones(len(x)))
            for tag, mask in sel.items():
                rows = row[mask]
                rep = rep_sel[mask[rep_sel]]
                if len(rep) != n_ev[tag]:
                    raise RuntimeError("Not one pair per event: %s %s" % (name, tag))
                per_slice += [("%s_events" % tag, n_ev[tag]), ("%s_pairs" % tag, len(rows))]
                extremes = {}
                for var in VARS:
                    x = values[var][mask]
                    ok = np.isfinite(x) & (x >= 0)  # producer sentinels are -999; all variables are >= 0
                    per_slice.append(("%s_invalid_%s" % (tag, var), int((~ok).sum())))
                    emax, emin = np.full(nev, -np.inf), np.full(nev, np.inf)
                    np.maximum.at(emax, rows[ok], x[ok])
                    np.minimum.at(emin, rows[ok], x[ok])
                    extremes[var] = (x, ok, emax[np.isfinite(emax)], emin[np.isfinite(emin)])
                    per_slice.append(("%s_novalid_%s" % (tag, var), n_ev[tag] - int(np.isfinite(emax).sum())))
                    probes = np.unique(x[ok])[:200]
                    for scan in SCANS.values():
                        if scan[0] == var:
                            probes = np.concatenate([probes, scan[1]])
                    threshold_check("%s %s %s" % (name, tag, var), nev, rows[ok], x[ok], emax, emin, probes)
                    x = values[var][rep]
                    per_slice.append(("%s_repinvalid_%s" % (tag, var), int((~(np.isfinite(x) & (x >= 0))).sum())))
                x_p, y_l = values["pt"][rep], values["lxy"][rep]
                both = np.isfinite(x_p) & (x_p >= 0) & np.isfinite(y_l) & (y_l >= 0)
                for s in sample_slices(kind, name):
                    for view, (var, edges, *_) in VIEWS.items():
                        x = values[var][rep]
                        ok = np.isfinite(x) & (x >= 0)
                        if ok.any():
                            hist((kind, s, tag, view), edges).FillN(int(ok.sum()), x[ok], np.ones(int(ok.sum())))
                        else:
                            hist((kind, s, tag, view), edges)
                    for scan, (var, edges, _) in SCANS.items():
                        emax, emin = extremes[var][2:]
                        for which, e in (("max_", emax), ("min_", emin)):
                            h = hist((kind, s, tag, which + scan), edges)
                            if len(e):
                                h.FillN(len(e), e, np.ones(len(e)))
                    for mp, ly in MAPS.items():
                        h = hist((kind, s, tag, mp), ly, MAP_PT)
                        if both.any():
                            h.FillN(int(both.sum()), y_l[both], x_p[both], np.ones(int(both.sum())))
            for s in sample_slices(kind, name):
                c = counts.setdefault((kind, s), {})
                for k, x in per_slice:
                    c[k] = c.get(k, 0) + x
    if not args.only:
        for kind in processed:
            if processed[kind] != expected[kind]:
                raise RuntimeError("%s samples processed %d != expected %d" % (kind, len(processed[kind]), len(expected[kind])))
    print("COVERAGE signal %d, background %d samples" % (len(processed["signal"]), len(processed["background"])))
    root = ROOT.TFile.Open(str(out), "CREATE")
    ROOT.TNamed("coverage", "signal %d background %d" % (len(processed["signal"]), len(processed["background"]))).Write()
    ROOT.TNamed("definitions", "Raw unweighted counts, one entry per event. VIEWS and map_* (x Lxy, y reco pT(mumu)) from the representative pair "
                "(lowest chi2/ndof among passing pairs, then lowest pair index); dphi_jet_met per event (all = before anything); "
                "max_<scan>/min_<scan> = per-event max/min over passing pairs with a valid value. selected = passing pairs; "
                "cut = passing pairs in events with the MET, jet cut. Inputs: %s, %s, %s, %s, %s, %s" % (
                    args.signal_summary, args.bkg_summary, args.met_jet, args.signal_scan_dir, args.bkg_scan_dir,
                    args.producer_dir)).Write()

    def cd(*parts):
        d = root
        for part in parts:
            d = d.GetDirectory(part) or d.mkdir(part)
        d.cd()

    for (kind, s, tag, name), h in sorted(hists.items()):
        c = counts[(kind, s)]
        if name.startswith("map_"):
            pass  # closure in plot
        elif name == "dphi_jet_met":
            want = c["cut_events"] if tag == "cut" else c["events"] - c["no_jet"] - c["dphi_undefined"]
            if abs(h.GetEntries() - want) > 0.5:
                raise RuntimeError("dphi closure failed: %s %s %s" % (kind, s, tag))
        elif name.startswith(("max_", "min_")):
            var = SCANS[name[4:]][0]
            if abs(h.GetEntries() + c["%s_novalid_%s" % (tag, var)] - c[tag + "_events"]) > 0.5:
                raise RuntimeError("Event histogram closure failed: %s %s %s %s" % (kind, s, tag, name))
        else:
            var = VIEWS[name][0]
            if abs(h.GetEntries() + c["%s_repinvalid_%s" % (tag, var)] - c[tag + "_events"]) > 0.5:
                raise RuntimeError("Event histogram closure failed: %s %s %s %s" % (kind, s, tag, name))
        cd(kind, *s.split("/"), tag)
        h.Write()
    for (kind, s), c in sorted(counts.items()):
        cd(kind, *s.split("/"))
        keys = sorted(c)
        h = ROOT.TH1D("counts", "", len(keys), 0, len(keys))
        for i, k in enumerate(keys, 1):
            h.GetXaxis().SetBinLabel(i, k)
            h.SetBinContent(i, c[k])
        h.Write()
        print("TOTAL %s %s %s" % (kind, s, " ".join("%s=%d" % (k, c[k]) for k in keys if c[k] or "invalid" not in k)))
    root.Close()
    print("FILL DONE", out)


def plot(args):
    import ROOT
    sys.path.insert(0, args.style_dir)
    import cms_style as cs
    from met_jet_plots import CHAIN, FOM_LINE, M_HEADER, M_KEYS, STAGES, Plotter, fom, load, mass_key, mass_label
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
    sl = {"selected": 7, "cut": 8}
    nall = {("signal", k): sig[k][0] for k in masses}
    nall.update({("background", b): bkg[b][0] for b, _, _ in BKGS})
    for k in masses + [b for b, _, _ in BKGS]:  # the fill and the cutflow counts must agree
        kind = "signal" if k in masses else "background"
        v, c = (sig if kind == "signal" else bkg)[k], stored("%s/%s" % (kind, k))
        if (c["events"], c["selected_events"], c["cut_events"]) != (v[0], v[7], v[8]):
            raise RuntimeError("Fill and cutflow counts differ for " + k)
    caps = cs.Captions("Six mass points vs all background, ZJets, WJets, QCD (kb/studies/bkg-reco-dimuons.md Part 6)", [
        "Signal: mass points M = m_chi - dm/2 = 5 and 50 GeV with dm = 0.1, 0.2, 0.4 M, each pooled unweighted over c-tau = "
        "1, 10, 100, 1000 mm (40,000 events each). Colour = M, line style = dm. Background: all %d 2022EE samples pooled, or "
        "one process group. Raw MC, unweighted: no cross sections, generator weights or luminosity." % n_bkg,
        "Selections: 'selected' = pairs passing the reco-only chain (eligible, opposite sign, valid vertex, chi2/ndof < 4, "
        "PCA valid, PCA dR < 0.9, |dphi(Type-1 PF MET, mumu PCA)| < 0.5). 'cut' adds, at event level, Type-1 PF MET > 80 GeV "
        "and highest-pT slimmedJets jet pT > 80 GeV, |eta| < 2.4. Directory names give the selection.",
        "Variables, reco legs (paired component tracks at their reference point): pT(mumu) = pRecoMuMuPt; leading, "
        "subleading = larger, smaller leg pT; Lxy = pLxyBS, vertex to beamspot at the vertex z.",
        "Every event counts once. Distributions and maps use one representative pair per event: among its passing pairs, "
        "lowest chi2/ndof, then lowest pair index. The same pair enters all variables, with and without the cut.",
        "Three versions of each distribution: raw (events/bin; heights reflect the generated event counts, not a physical "
        "rate); *_norm ('Unit area': shape, normalized within the plotted range, flows excluded; efficiency removed); "
        "*_frac (events in the bin / all events of the slice; efficiency kept). Bins plus flows sum to selected / all "
        "or selected and cut / all. Log-y copies: *_log.png.",
        "F = eps_s / sqrt(eps_b), relative and unweighted, not an expected significance; the pooled eps_b weights every MC "
        "event equally. Bars: first-order propagation of Clopper-Pearson 68% intervals (approximate)."])
    P = Plotter(cs, out, True)

    def band_plot(base, graphs, xr, xtitle, ytitle, lines, top=0.34, extra=None):
        """Band layout, keys below the context lines (met_jet_dists.py); line-only keys."""
        c = cs.canvas("c_" + str(base).replace("/", "_"), band=True)
        c.SetTopMargin(top)
        hi = max([p[3] for p in cs._points(graphs)] or [1])
        ymax = cs._round_up(1.1 * hi)
        wide = ymax < 0.1 or ymax >= 1000  # long tick labels: room for the y title (as met_jet_plots.py)
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
        g = dict(groups, **(extra or {}))
        cs.legend([(cs._proxy(col, 1, 20), lab, "l") for lab, col in g.items()], y2=ytop, y1=1 - top + 0.01, x1=0.50,
                  header=M_HEADER, ncol=2, size=0.026)
        cs.legend([(cs._proxy(cs.BLACK, ls, ms), lab, "l") for lab, (ls, ms) in keys.items()], y2=ytop, y1=1 - top + 0.01,
                  x1=0.17, x2=0.47, size=0.026)
        cs.save(c, base, log=True)

    # Cutflow, fixed F.
    stage_labels = [lab for _, lab in STAGES]
    for b, name, lab in BKGS:
        gr = [P.eff(sig[k], sig[k][0], look[k][0], look[k][1], "cf%s%s" % (name, k)) for k in masses]
        gr.append(P.eff(bkg[b], bkg[b][0], cs.BLACK, (1, 21), "cf" + name))
        P.draw("cutflow/" + name, gr, None, stage_labels, "Fraction of all events",
               [CHAIN, "Signal pooled over lifetimes"], band=(dict(groups, **{"Bkg, " + lab: cs.BLACK}), keys, M_HEADER), log=True)
    caps.add("cutflow/bkg_<all|ZJets|WJets|QCD>.png", "Fraction of all events passing each cumulative stage (>= 1 pair "
             "reaching it), ending with the MET, jet cut, for the six mass points and one background. Event counts:",
             ["slice"] + [k.replace("events_", "") for k, _ in STAGES],
             [[k] + ["%d" % x for x in sig[k]] for k in masses] + [[n] + ["%d" % x for x in bkg[b]] for b, _, n in BKGS])
    xl = [mass_label(k[5:]) for k in masses]
    rows = []
    for b, name, lab in BKGS:
        gr = []
        for tag, sty, leg in (("selected", cs.OPEN_DASHED, "Selected #mu#mu"), ("cut", cs.FILLED_SOLID, "After cut")):
            pts = []
            for i, k in enumerate(masses):
                r = fom(sig[k][sl[tag]], sig[k][0], bkg[b][sl[tag]], bkg[b][0])
                if r:
                    pts.append((i, r))
                    rows.append([lab, tag, xl[i], "%.3f +%.3f -%.3f" % r])
            gr.append(P.graph(pts, cs.BLUE, sty))
        P.draw("fom/fixed_" + name, gr, ["Selected #mu#mu", "After cut"], xl, "#varepsilon_{s} / #sqrt{#varepsilon_{b}}",
               ["Background: " + lab, FOM_LINE], xtitle="Signal m_{#chi} / #Deltam [GeV] (pooled over lifetimes)", log=True)
    caps.add("fom/fixed_bkg_<all|ZJets|WJets|QCD>.png", "F at the end of the chain (open) and with the MET, jet cut "
             "(filled), per mass point, against one background.", ["background", "selection", "m_chi / dm", "F"], rows)

    # Distributions.
    def graph(path, nev, color, style, mode):
        h = get(path)
        if mode == "frac":
            h = h.Clone(); cs._keep.append(h)
            h.Scale(1. / nev)  # the clone only; the stored histogram stays raw
        return cs.hist_graph(h, color, style[0], style[1], step=True, norm=mode == "norm")

    tot = lambda h: h.Integral(0, h.GetNbinsX() + 1)
    for view in args.views:
        var, edges, xt, per = VIEWS[view]
        for tag in TAGS:
            for b, name, lab in BKGS:
                for mode, sfx, yt in (("raw", "", "Events / "), ("norm", "_norm", "Fraction of events / "),
                                      ("frac", "_frac", "Fraction of all events / ")):
                    gr = [graph("signal/%s/%s/%s" % (k, tag, view), nall[("signal", k)], look[k][0], look[k][1], mode)
                          for k in masses]
                    gr.append(graph("background/%s/%s/%s" % (b, tag, view), nall[("background", b)], cs.BLACK, (1, 21), mode))
                    lines = [STAGE[tag] + "; signal pooled over lifetimes"] + (["Unit area"] if mode == "norm" else [])
                    band_plot(out / "dists" / view / tag / (name + sfx), gr, (edges[0], edges[-1]), xt, yt + per, lines,
                              extra={"Bkg, " + lab: cs.BLACK})
        rows = []
        for k, kind, n in [(k, "signal", k[5:]) for k in masses] + [(b, "background", n) for b, _, n in BKGS]:
            row = [n]
            for tag in TAGS:
                h = get("%s/%s/%s/%s" % (kind, k, tag, view))
                if tot(h) != (sig if kind == "signal" else bkg)[k][sl[tag]]:
                    raise RuntimeError("Event closure failed: %s %s %s" % (k, tag, view))
                row.append("%.1f%%" % (100. * h.Integral(1, h.GetNbinsX()) / tot(h)))
            rows.append(row)
        caps.add("dists/%s/<selected|cut>/bkg_<all|ZJets|WJets|QCD>[_norm|_frac].png" % view,
                 "Selected events, one entry per event: six mass points and one background, %g-%g in bins of %s. Fraction of each "
                 "curve's events inside the plotted range (what the unit-area copy is normalized to):" % (edges[0], edges[-1], per),
                 ["slice", "in range, selected", "in range, cut"], rows)

    # Threshold scans.
    def scan_points(path, nev, nedges, direction):
        """(edge index, count, n) per threshold: 'up' = v >= edge, 'down' = v < edge."""
        h = get(path + "/" + ("max_" if direction == "up" else "min_") + scan)
        nb = h.GetNbinsX()
        return [h.Integral(i + 1, nb + 1) if direction == "up" else h.Integral(0, i) for i in range(nedges)], h, nev

    ratio_rows = []
    for scan in args.scans:
        var, edges, xt = SCANS[scan]
        for tag in TAGS:
            for direction, sign in (("up", "#geq"), ("down", "<")):
                curves = {}
                for b, name, lab in BKGS:
                    cb, hb, nb_ = scan_points("background/%s/%s" % (b, tag), nall[("background", b)], len(edges), direction)
                    if (cb[0] if direction == "up" else tot(hb)) != bkg[b][sl[tag]]:
                        raise RuntimeError("Scan end point differs from the selected count: %s %s" % (b, tag))
                    gr = []
                    for k in masses:
                        cs_, h, ns = scan_points("signal/%s/%s" % (k, tag), nall[("signal", k)], len(edges), direction)
                        d = np.diff(cs_)
                        if (direction == "up" and (d > 0).any()) or (direction == "down" and (d < 0).any()):
                            raise RuntimeError("Scan not monotonic: " + k)
                        if (cs_[0] if direction == "up" else tot(h)) != sig[k][sl[tag]]:  # no no-valid events allowed
                            raise RuntimeError("Scan end point differs from the selected count: %s %s" % (k, tag))
                        g = ROOT.TGraphAsymmErrors()
                        for i, x in enumerate(edges):
                            r = fom(cs_[i], ns, cb[i], nb_) if cs_[i] > 0 else None
                            if r:
                                j = g.GetN()
                                g.SetPoint(j, x, r[0]); g.SetPointError(j, 0, 0, 0, 0)  # lines only; legible with six curves
                        cs._style(g, look[k][0], look[k][1][0], look[k][1][1], [edges[0], edges[-1]], opt="L")
                        gr.append(g)
                        curves[(b, k)] = (cs_, cb, ns, nb_)
                    band_plot(out / "fom_scan" / scan / tag / ("%s_%s" % (name, direction)), gr, (edges[0], edges[-1]), xt,
                              "#varepsilon_{s} / #sqrt{#varepsilon_{b}}",
                              ["Background: %s; pass: variable %s threshold" % (lab, sign), FOM_LINE])
                for k in masses:  # F(x) / F(reference): best x against the pooled background, then the groups there
                    best, bestr = None, 0
                    cs0, cb0, ns, nb0 = curves[("overall", k)]
                    fref = lambda b: fom(sig[k][sl[tag]], sig[k][0], bkg[b][sl[tag]], bkg[b][0])
                    for i in range(len(edges)):
                        if cs0[i] > 0 and cb0[i] >= MIN_BKG_RATIO:
                            r = fom(cs0[i], ns, cb0[i], nb0)[0] / fref("overall")[0]
                            if r > bestr:
                                best, bestr = i, r
                    if best is None:
                        continue
                    rs = []
                    for b, _, _ in BKGS:
                        cs1, cb1, ns1, nb1 = curves[(b, k)]
                        r = fom(cs1[best], ns1, cb1[best], nb1) if cb1[best] > 0 and cs1[best] > 0 else None
                        rs.append(r[0] / fref(b)[0] if r and fref(b) else None)
                    row = [scan, tag, "%s %g" % (">=" if direction == "up" else "<", edges[best]), mass_label(k[5:])] + [
                        "-" if r is None else "%.2f" % r for r in rs]
                    print("RATIO", " | ".join(row))
                    if all(r is not None and r > 1 for r in rs):
                        ratio_rows.append(row)
        caps.add("fom_scan/%s/<selected|cut>/bkg_<all|ZJets|WJets|QCD>_<up|down>.png" % scan,
                 "F vs threshold x, event level, %g-%g. up: an event passes if >= 1 passing pair has v >= x; down: if >= 1 has "
                 "v < x. eps(x) = passing events / all events of the slice. Reference (the fixed F): up at x = 0; down above "
                 "all values, overflow included. Scans use any passing pair, not only the representative pair, so a scan need not equal the integral of a distribution. Thresholds with no surviving background or signal event are not drawn. Lines only, "
                 "no error bars (six curves); the statistical precision worsens where few background events remain."
                 % (edges[0], edges[-1]))
    caps.add("fom_scan (F(x) / F(reference))", "Thresholds passing the Part 6 pointer: at the x that maximizes the ratio against "
             "the pooled background (with >= %d pooled-background events), the ratio is > 1 against the pooled background and "
             "each of ZJets, WJets, QCD. Exploratory: F(x) and the reference share events and x is scanned, so this carries "
             "no significance. All rows are in the plot log." % MIN_BKG_RATIO,
             ["scan", "selection", "x", "m_chi / dm"] + [n for _, _, n in BKGS], ratio_rows)

    # pT-Lxy maps.
    for mp in args.maps:
        rows = []
        for k, kind, name, lab in [(k, "signal", k[5:], "m_{#chi} / #Deltam = %s GeV" % mass_label(k[5:])) for k in masses] + [
                (b, "background", n, "Bkg, " + lab) for b, n, lab in BKGS]:
            row = [lab.replace("m_{#chi} / #Deltam = ", "")]
            for tag in TAGS:
                h = get("%s/%s/%s/%s" % (kind, k, tag, mp))
                base = out / "maps" / mp / tag / name
                total = h.Integral(0, h.GetNbinsX() + 1, 0, h.GetNbinsY() + 1)
                if abs(total - stored("%s/%s" % (kind, k))[tag + "_events"]) > 0.5:
                    raise RuntimeError("Map closure failed: %s %s %s" % (k, tag, mp))
                row.append("%.1f%%" % (100. * h.Integral(1, h.GetNbinsX(), 1, h.GetNbinsY()) / total))
                for logz, sfx in ((False, ""), (True, "_log")):
                    d = h.Clone(); cs._keep.append(d)
                    base.parent.mkdir(parents=True, exist_ok=True)
                    cs.map2d(str(base) + sfx, d, LXY_T, PT_T, "Events / bin", lines=[lab, STAGE[tag]], logz=logz)
            rows.append(row)
        ly = MAPS[mp]
        caps.add("maps/%s/<selected|cut>/<slice>[_log].png" % mp,
                 "Reco pT(mumu) vs vertex Lxy of the representative pair, raw event counts, one map per mass point or background. "
                 "Lxy %g-%g cm in %g cm bins, pT 0-100 GeV in 2.5 GeV bins; empty bins are white. _log = log colour scale. "
                 "Fraction of events inside the map:"
                 % (ly[0], ly[-1], ly[1] - ly[0]), ["slice", "in map, selected", "in map, cut"], rows)
    # |dphi(leading jet, MET)|, all background only.
    per = "#pi/32"
    for tag, line in (("all", "All events"), ("cut", "Selected, after cut")):
        for mode, sfx, yt in (("raw", "", "Events / "), ("norm", "_norm", "Fraction of events / "),
                              ("frac", "_frac", "Fraction of all events / ")):
            gr = [graph("signal/%s/%s/dphi_jet_met" % (k, tag), nall[("signal", k)], look[k][0], look[k][1], mode) for k in masses]
            gr.append(graph("background/overall/%s/dphi_jet_met" % tag, nall[("background", "overall")], cs.BLACK, (1, 21), mode))
            band_plot(out / "dphi_jet_met" / tag / ("bkg_all" + sfx), gr, (0, np.pi), DPHI_T, yt + per,
                      [line] + (["Unit area"] if mode == "norm" else []), extra={"Bkg, all %d" % n_bkg: cs.BLACK})
    rows = []
    for k, kind, n in [(k, "signal", k[5:]) for k in masses] + [("overall", "background", "all %d" % n_bkg)]:
        c = stored("%s/%s" % (kind, k))
        rows.append([n, "%d" % c["events"], "%d" % c["no_jet"], "%d" % c["dphi_undefined"], "%d" % c["cut_events"]])
    caps.add("dphi_jet_met/<all|cut>/bkg_all[_norm|_frac].png",
             "|dphi| between the highest-pT slimmedJets jet (any eta; no ID, no muon cleaning) and Type-1 PF MET, wrapped to "
             "[0, pi], 32 bins, one entry per event. all = before anything; cut = selected events passing the MET, jet cut. In "
             "the 'all' plots, events with no jet or no defined angle are not filled but stay in the *_frac denominator.",
             ["slice", "all events", "no jet", "undefined angle", "selected after cut"], rows)
    print("CAPTIONS", caps.write(out))
    print("PLOTS DONE", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("fill")
    for k in ("--signal-summary", "--bkg-summary", "--met-jet", "--signal-scan-dir", "--bkg-scan-dir", "--producer-dir",
              "--output"):
        a.add_argument(k, required=True)
    a.add_argument("--only", help="regex on sample names; timing tests only (skips the coverage check)")
    b = sub.add_parser("plot")
    for k in ("--input", "--signal-summary", "--bkg-summary", "--met-jet", "--style-dir", "--out-dir"):
        b.add_argument(k, required=True)
    b.add_argument("--views", nargs="*", default=list(VIEWS), choices=list(VIEWS))
    b.add_argument("--scans", nargs="*", default=list(SCANS), choices=list(SCANS))
    b.add_argument("--maps", nargs="*", default=list(MAPS), choices=list(MAPS))
    args = ap.parse_args()
    (fill if args.cmd == "fill" else plot)(args)


if __name__ == "__main__":
    main()
