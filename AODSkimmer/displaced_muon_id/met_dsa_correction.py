#!/usr/bin/env python3
"""PF MET corrected for selected DSA legs without a PF muon (Run 2 style) vs PF MET and chi1chi1 pT, Delta-phi to mumu.

python3 met_dsa_correction.py analyze --input-dir RUN/inputs --report-dir RUN/logs --revision <sha> --output met_dsa.root
python3 met_dsa_correction.py plot --output met_dsa.root --plot-dir plots [--log]
Study note: kb/studies/met-dimuon-dphi.md (Part 4). Inputs: PreselectedVertexing with vertexDirections, pfStation2 and
pfHitMatch (pf_muon_overlap_hit_20261027). Population: met-dimuon-dphi Part 3 vertices (pf_muon_overlap.vertex_legs).
PF association per leg (displaced-pf-muon-overlap Part 3): or01 = innermost-hit dR < 0.1 or segments >= 2/3,
prox01 = innermost-hit dR < 0.1; no pT/charge gate. Both DSA legs on one PF muon: the leg with the smaller hit dR keeps
it (no hit dR last, then higher segment fraction, then lower gen index). MET' = Type-1 PF MET - sum of the pT (at the
3D PCA to the fitted vertex) of DSA legs without a PF muon; global/inner legs never change MET. mumu = sum of the two
PCA leg pT. chi1chi1 = sum of the two chiOK gen chi1. Raw unweighted, one entry per vertex; analyze writes a per-event
tree, plot fills every histogram from it. Zero-length vectors have no direction: excluded from Delta-phi and residuals
(counted), kept in |X|. Fractions: Clopper-Pearson 68.3%. Paired PF vs MET' test on |dphi| < 0.5: (n10-n01)/sqrt(n10+n01).
"""
import argparse
from array import array
from pathlib import Path
import re
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preselected_vertexing_analysis import check_report, flowcounts
from pf_muon_overlap import HF_LEG, HF_SCALAR, HIT_LEG, SCALAR, vertex_legs

DSA = 1
VARIANTS = ("or01", "prox01")
XS = ("pf", "or01", "prox01", "chi")
XLAB = {"pf": "PF p_{T}^{miss}", "or01": "PF p_{T}^{miss} #minus DSA w/o PF (hit/seg.)",
        "prox01": "PF p_{T}^{miss} #minus DSA w/o PF (hit)", "chi": "p_{T}(#chi_{1}#chi_{1})"}
EXTRA = ("hfVertexPt", "hfLegSegPfDR", "pfMetPt", "pfMetPhi", "chiOK", "chiPx", "chiPy", "sigD0")
DPHI_EDGES = np.round(np.arange(0, 3.1501, 0.05), 2)
RES_EDGES = np.round(np.arange(-3.15, 3.1501, 0.05), 2)
MET_EDGES = np.arange(0, 400.1, 5.0)
D0_EDGES = np.arange(0, 60.1, 2.0)
CUT = 0.5
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#000000"]
TOP, BOTTOM = 0.92, 0.13


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def sample_events(a, ak, name, si):
    """Per selected vertex: PF MET, MET' per variant, chi1chi1 and mumu vectors, plus bookkeeping."""
    v = {var: vertex_legs(a, ak, name, si, var) for var in VARIANTS}
    rows = v["or01"]["evrow"][::2]
    if not np.array_equal(v["prox01"]["evrow"], v["or01"]["evrow"]):
        raise RuntimeError("Variant selections differ in " + name)
    for k in ("hfVertexPt", "hfLegSegPfDR", "sigD0", "hfLegSegFrac", "hfLegQ", "hfLegHitPfPt", "hfLegSegPfPt", "hfLegHitPfQ",
              "hfLegSegPfQ", "sigPt"):
        if np.any(np.asarray(ak.num(a[k]))[rows] != 2):
            raise RuntimeError("Non-pair %s in a selected event of %s" % (k, name))
    pick = lambda k: ak.to_numpy(a[k][rows])
    pt, phi = pick("hfVertexPt").astype(float), pick("hfVertexPhi").astype(float)
    if np.any(~(pt > 0) | ~np.isfinite(phi) | (np.abs(phi) > 4)):
        raise RuntimeError("Invalid PCA leg momentum (sentinel or non-positive) in " + name)
    typ = v["or01"]["matchType"].reshape(-1, 2)
    lpx, lpy = pt * np.cos(phi), pt * np.sin(phi)
    met_ok = (np.asarray(a["pfMetPt"])[rows] != -999) & (np.asarray(a["pfMetPhi"])[rows] != -999)
    mpt, mphi = np.asarray(a["pfMetPt"], dtype=float)[rows], np.asarray(a["pfMetPhi"], dtype=float)[rows]
    chi_ok = np.asarray(a["chiOK"])[rows].astype(bool)
    e = {"sample": np.full(len(rows), si, dtype=np.int32), "run": np.asarray(a["run"])[rows],
         "lumi": np.asarray(a["lumi"])[rows], "event": np.asarray(a["event"])[rows],
         "typeA": typ[:, 0].astype(np.int32), "typeB": typ[:, 1].astype(np.int32),
         "nDSA": (typ == DSA).sum(1).astype(np.int32), "d0max": np.max(np.abs(pick("sigD0")), axis=1),
         "pfOK": met_ok.astype(np.int32), "chiOK": chi_ok.astype(np.int32),
         "mumuPx": lpx.sum(1), "mumuPy": lpy.sum(1),
         "pfPx": np.where(met_ok, mpt * np.cos(mphi), np.nan), "pfPy": np.where(met_ok, mpt * np.sin(mphi), np.nan),
         "chiPx": np.where(chi_ok, np.asarray(a["chiPx"], dtype=float)[rows], np.nan),
         "chiPy": np.where(chi_ok, np.asarray(a["chiPy"], dtype=float)[rows], np.nan)}
    segfrac = pick("hfLegSegFrac")
    legq, genpt, refpt = pick("hfLegQ"), pick("sigPt").astype(float), pick("hfLegPt").astype(float)
    stats = {}
    for var in VARIANTS:
        assoc = v[var]["pfMatched"].reshape(-1, 2).copy()
        idx = v[var]["legPfIdx"].reshape(-1, 2)
        hitdr = v[var]["hitDR"].reshape(-1, 2)
        dsa = typ == DSA
        shared = dsa.all(1) & assoc.all(1) & (idx[:, 0] == idx[:, 1])
        use_seg = (var == "or01") & ~((hitdr >= 0) & (hitdr < 0.1))  # as vertex_legs: proximity PF first
        adr = np.where(use_seg, pick("hfLegSegPfDR"), hitdr)  # hit dR of the associated PF muon itself
        hd = np.where(adr >= 0, adr, np.inf)
        # loser = leg 1 unless leg 1 is strictly better (smaller hit dR, then higher segment fraction)
        b1 = (hd[:, 1] < hd[:, 0]) | ((hd[:, 1] == hd[:, 0]) & (segfrac[:, 1] > segfrac[:, 0]))
        assoc[shared & ~b1, 1] = False
        assoc[shared & b1, 0] = False
        other = dsa & assoc & ~dsa[:, ::-1] & v[var]["pfMatched"].reshape(-1, 2)[:, ::-1] & (idx == idx[:, ::-1])
        corr = dsa & ~assoc
        e["nCorr_" + var] = corr.sum(1).astype(np.int32)
        e[var + "Px"] = e["pfPx"] - np.where(corr, lpx, 0).sum(1)
        e[var + "Py"] = e["pfPy"] - np.where(corr, lpy, 0).sum(1)
        pfpt = np.where(use_seg, pick("hfLegSegPfPt"), pick("hfLegHitPfPt")).astype(float)
        pfq = np.where(use_seg, pick("hfLegSegPfQ"), pick("hfLegHitPfQ"))
        if np.any(~np.isclose(pfpt[assoc], v[var]["legPfPt"].reshape(-1, 2)[assoc])):
            raise RuntimeError("Associated PF pT disagrees with vertex_legs in " + name)
        a_dsa = dsa & assoc
        rleg = np.abs(pfpt - refpt) / refpt
        stats[var] = {"DSA legs": int(dsa.sum()), "associated DSA legs": int(a_dsa.sum()),
                      "corrected DSA legs": int(corr.sum()), "shared PF muon, DSA-DSA (one leg released)": int(shared.sum()),
                      "DSA leg sharing its PF muon with a global/inner leg (kept associated)": int(other.sum()),
                      "associated DSA: PF charge = leg charge": int((a_dsa & (pfq == legq)).sum()),
                      "associated DSA: |pT_PF - pT_leg|/pT_leg < 0.2": int((a_dsa & (rleg < 0.2)).sum()),
                      "associated DSA: same charge and |pT_gen - pT_PF|/pT_gen < 0.1":
                          int((a_dsa & (pfq == legq) & (np.abs(genpt - pfpt) / genpt < 0.1)).sum())}
    return e, stats


def analyze(args):
    import awkward as ak
    import ROOT
    import uproot
    if Path(args.output).exists():
        raise RuntimeError("Refusing to overwrite numerical ROOT output")
    paths = sorted(Path(args.input_dir).glob("*.root"))
    if len(paths) != args.expected_samples:
        raise RuntimeError("Input sample count mismatch: %d" % len(paths))
    names, revisions, evs, n_events, stats = [], set(), [], 0, {}
    for si, path in enumerate(paths):
        with uproot.open(path) as f:
            pre = "preselectedVertexing/"
            name = str(f[pre + "sample"].member("fTitle"))
            if name in names or not re.fullmatch(r"Mchi-[\dp]+_dMchi-[\dp]+_ctau-[\dp]+", name):
                raise RuntimeError("Unexpected or duplicate sample: " + name)
            names.append(name)
            revisions.add(str(f[pre + "revision"].member("fTitle")))
            if str(f[pre + "hierarchyOrder"].member("fTitle")) != "global>inner>outer(DSA)":
                raise RuntimeError("Unexpected hierarchy order: " + name)
            for flag in ("vertexDirections", "pfStation2", "pfHitMatch"):
                if str(f[pre + flag].member("fTitle")) == "disabled":
                    raise RuntimeError("%s was disabled: %s" % (flag, name))
            t = f[pre + "events"]
            check_report(Path(args.report_dir), name, t.num_entries)
            n_events += t.num_entries
            a = t.arrays(list(SCALAR + HF_SCALAR + HF_LEG + HIT_LEG + EXTRA[:-1] +
                              ("sigQ", "sigPt", "sigLxy", "sigD0", "pfMuPt", "pfMuSt2Ok", "pfMuN", "pfSlimN",
                               "pfSlimLinked", "segDropped")), library="ak")
        e, s = sample_events(a, ak, name, si)
        evs.append(e)
        for var in s:
            for k, n in s[var].items():
                stats.setdefault(var, {}).setdefault(k, 0)
                stats[var][k] += n
    if len(revisions) != 1 or "unspecified" in revisions:
        raise RuntimeError("Mixed/unspecified input revisions")
    if n_events != args.expected_events:
        raise RuntimeError("Event total mismatch: %d" % n_events)
    ev = {k: np.concatenate([x[k] for x in evs]) for k in evs[0]}
    print("SELECTED VERTICES %d (>= 1 DSA leg %d); events read %d" % (len(ev["sample"]), int((ev["nDSA"] > 0).sum()), n_events))
    for var in VARIANTS:
        for k, n in stats[var].items():
            print("ASSOC %s %s: %d" % (var, k, n))
    with uproot.recreate(args.output) as f:
        f["events"] = ev
        f["samples"] = ",".join(names)
        f["inputRevision"] = revisions.pop()
        f["revision"] = args.revision
        f["assocStats"] = "\n".join("%s %s: %d" % (var, k, n) for var in VARIANTS for k, n in stats[var].items())
    print("WROTE", args.output)


# ---------------------------------------------------------------------------------------------- plotting
def derived(ev):
    """|X|, |dphi(X, mumu)|, residual vs chi1chi1 per X, with validity masks."""
    mm = np.hypot(ev["mumuPx"], ev["mumuPy"])
    mphi = np.arctan2(ev["mumuPy"], ev["mumuPx"])
    d = {}
    for x in XS:
        px, py = ev[x + "Px"], ev[x + "Py"]
        mag = np.hypot(px, py)
        okm = np.isfinite(mag)
        okd = okm & (mag > 0) & (mm > 0)
        dphi = np.where(okd, wrap(np.arctan2(py, px) - mphi), np.nan)
        d[x] = {"mag": mag, "okm": okm, "okd": okd, "dphi": dphi, "adphi": np.abs(dphi)}
    for x in XS[:-1]:
        ok = d[x]["okd"] & d["chi"]["okd"]
        d[x]["res"] = np.where(ok, wrap(d[x]["dphi"] - d["chi"]["dphi"]), np.nan)
        d[x]["okr"] = ok
    return d


def plot(args):
    import ROOT
    import uproot
    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0); ROOT.gStyle.SetOptTitle(0); ROOT.gStyle.SetTextFont(42)
    ROOT.gStyle.SetLabelFont(42, "XYZ"); ROOT.gStyle.SetTitleFont(42, "XYZ"); ROOT.gStyle.SetPadTickX(1); ROOT.gStyle.SetPadTickY(1)
    ROOT.gStyle.SetEndErrorSize(0); ROOT.TH1.AddDirectory(False); ROOT.TGaxis.SetMaxDigits(6)
    with uproot.open(args.output) as f:
        ev = {k: np.asarray(x) for k, x in f["events"].arrays(library="np").items()}
        names = str(f["samples"]).split(",")
    d = derived(ev)
    out = Path(args.plot_dir); out.mkdir(parents=True, exist_ok=True)
    color = lambda i: ROOT.TColor.GetColor(PALETTE[i % len(PALETTE)])
    keep, count = [], [0, 0]

    # groups: overall, mass (pooled over lifetimes), lifetime (pooled over masses), sample; each for all and >= 1 DSA
    groups = {"overall": np.ones(len(ev["sample"]), bool)}
    for si, n in enumerate(names):
        mass, tau = n.rsplit("_ctau-", 1)
        groups.setdefault("mass_" + mass, np.zeros(len(ev["sample"]), bool))
        groups.setdefault("ctau_" + tau, np.zeros(len(ev["sample"]), bool))
        groups["mass_" + mass] |= ev["sample"] == si
        groups["ctau_" + tau] |= ev["sample"] == si
    for si, n in enumerate(names):
        groups["sample_" + n] = ev["sample"] == si
    subsets = {"all": np.ones(len(ev["sample"]), bool), "dsa": ev["nDSA"] > 0}
    sublab = {"all": "All selected vertices", "dsa": "Vertices with #geq 1 DSA leg"}

    def glabel(g):
        if g == "overall":
            return "All samples"
        if g.startswith("ctau_"):
            return "c#tau = %s mm, all masses" % g.split("_", 1)[1].replace("p", ".")
        a, b = (x.replace("p", ".") for x in re.findall(r"Mchi-([\dp]+)", g))
        tail = (", c#tau = %s mm" % g.rsplit("_ctau-", 1)[1].replace("p", ".")) if g.startswith("sample_") else ", all c#tau"
        return "m_{#chi} = %s GeV, #Deltam = %s GeV%s" % (a, b, tail)

    def numkey(g):
        return [float(x.replace("p", ".")) for x in re.findall(r"(?:Mchi|ctau)-([\dp]+)", g)] or [float(g.split("_")[-1].replace("p", "."))]

    def canvas(name):
        c = ROOT.TCanvas(name, "", 800, 700)
        c.SetLeftMargin(0.17); c.SetRightMargin(0.04); c.SetTopMargin(1 - TOP); c.SetBottomMargin(BOTTOM)
        keep.append(c)
        return c

    def frame(h, xt, yt):
        h.GetXaxis().SetTitle(xt); h.GetYaxis().SetTitle(yt); h.GetXaxis().SetNdivisions(505)
        for ax in (h.GetXaxis(), h.GetYaxis()):
            ax.SetTitleSize(0.05); ax.SetLabelSize(0.042)
        h.GetXaxis().SetTitleOffset(1.15); h.GetYaxis().SetTitleOffset(1.65)

    def labels(c, lines):
        c.cd()
        t = ROOT.TLatex(); t.SetNDC(); keep.append(t)
        t.SetTextFont(61); t.SetTextSize(0.05); t.SetTextAlign(11); t.DrawLatex(0.17, 0.94, "CMS")
        t.SetTextFont(52); t.SetTextSize(0.038); t.DrawLatex(0.275, 0.94, "Simulation Private Work")
        t.SetTextFont(42); t.SetTextAlign(31); t.DrawLatex(0.96, 0.94, "2022EE (13.6 TeV)")
        t.SetTextSize(0.032); t.SetTextAlign(13)
        for i, line in enumerate(lines): t.DrawLatex(0.21, 0.895 - 0.042 * i, line)

    def legend(n, nlines, cols=1, size=0.030):
        top = 0.895 - 0.042 * nlines - 0.01
        bottom = top - ((n + cols - 1) // cols) * size * 1.25
        leg = ROOT.TLegend(0.20, bottom, 0.95, top); keep.append(leg)
        leg.SetBorderSize(0); leg.SetFillStyle(0); leg.SetTextSize(size); leg.SetNColumns(cols)
        return leg, bottom

    headroom = lambda lo, hi, bottom: lo + (hi - lo) * (TOP - BOTTOM) / max(bottom - BOTTOM - 0.02, 0.1)

    def hist(name, edges, values):
        h = ROOT.TH1D(name, "", len(edges) - 1, array("d", edges)); h.Sumw2()
        for b, n in enumerate(flowcounts(values, edges)):
            h.SetBinContent(b, float(n)); h.SetBinError(b, float(np.sqrt(n)))
        keep.append(h)
        return h

    def save(c, fr, name, hs, bottom):
        c.SaveAs(str(out / (name + ".png"))); count[0] += 1
        vals = [h.GetBinContent(b) for h in hs for b in range(1, h.GetNbinsX() + 1) if h.GetBinContent(b) > 0]
        if args.log and vals:
            top, low = max(vals), min(vals) * 0.5
            lmax = np.log10(low) + (np.log10(top) - np.log10(low)) * (TOP - BOTTOM) / max(bottom - BOTTOM - 0.02, 0.1)
            fr.SetMinimum(low); fr.SetMaximum(10 ** lmax)
            c.SetLogy(1); c.Modified(); c.SaveAs(str(out / (name + "_log.png"))); count[1] += 1

    KINDS = {"adphi": (DPHI_EDGES, "|#Delta#phi(X, p_{T}^{#mu#mu})|", "okd", "Vertices / 0.05"),
             "res": (RES_EDGES, "#Delta#phi(X, p_{T}^{#mu#mu}) #minus #Delta#phi(#chi_{1}#chi_{1}, p_{T}^{#mu#mu})", "okr",
                     "Vertices / 0.05"),
             "mag": (MET_EDGES, "|X| [GeV]", "okm", "Vertices / 5 GeV")}

    def series_label(x, kind, m):
        if kind == "adphi":
            ok = m & d[x]["okd"]
            n = int(ok.sum())
            p = int((ok & (d[x]["adphi"] < CUT)).sum())
            return "%s: %.1f%% < %.1f" % (XLAB[x], 100.0 * p / n if n else 0, CUT)
        return XLAB[x]

    def distribution(tag, kind, xs, m, lines, norm):
        edges, xt, okk, yt = KINDS[kind]
        c = canvas("c_%s_%s_%s_%d" % (tag, kind, "_".join(xs), norm))
        hs = []
        for i, x in enumerate(xs):
            ok = m & d[x][okk]
            h = hist("h_%s_%s_%s_%d" % (tag, kind, x, norm), edges, d[x][kind][ok])
            if norm and ok.sum(): h.Scale(1.0 / ok.sum())
            ci = XS.index(x)
            h.SetLineColor(color(ci)); h.SetLineWidth(2); h.SetLineStyle(1 + (ci % 3))
            hs.append((h, x, int(ok.sum())))
        leg, bottom = legend(len(hs), len(lines))
        top = max(h.GetBinContent(h.GetMaximumBin()) for h, _, _ in hs) or 1.0
        fr = hs[0][0]
        frame(fr, xt, ("Fraction of vertices / %s" % yt.split("/ ")[1]) if norm else yt)
        fr.SetMinimum(0); fr.SetMaximum(headroom(0, top, bottom)); fr.Draw("hist")
        for h, _, _ in hs[1:]: h.Draw("hist same")
        for h, x, n in hs: leg.AddEntry(h, series_label(x, kind, m) + " (%d)" % n, "l")
        leg.Draw(); labels(c, lines)
        save(c, fr, "%s_%s_%s%s" % (tag, kind, "overlay" if len(xs) > 1 else xs[0], "_unitarea" if norm else ""),
             [h for h, _, _ in hs], bottom)

    def pass_vs_d0(tag, m, lines):
        c = canvas("c_d0_" + tag)
        leg, bottom = legend(len(XS), len(lines))
        fr = ROOT.TH1D("fr_d0_" + tag, "", 1, D0_EDGES[0], D0_EDGES[-1]); keep.append(fr)
        fr.SetMinimum(0); fr.SetMaximum(headroom(0, 1.0, bottom))
        frame(fr, "max gen |d_{0}| of the two #mu [cm]", "Fraction with |#Delta#phi| < %.1f" % CUT); fr.Draw("axis")
        for i, x in enumerate(XS):
            ok = m & d[x]["okd"]
            den = hist("den_%s_%s" % (tag, x), D0_EDGES, ev["d0max"][ok])
            num = hist("num_%s_%s" % (tag, x), D0_EDGES, ev["d0max"][ok & (d[x]["adphi"] < CUT)])
            eff = ROOT.TEfficiency(num, den); keep.append(eff)
            gr = eff.CreateGraph(); keep.append(gr)
            gr.SetLineColor(color(i)); gr.SetMarkerColor(color(i)); gr.SetMarkerStyle(20 + i); gr.SetMarkerSize(0.8)
            gr.SetLineWidth(2); gr.Draw("PZ same")
            leg.AddEntry(gr, XLAB[x], "lp")
        leg.Draw(); labels(c, lines + ["Vertices beyond 60 cm not shown"])
        c.SaveAs(str(out / ("%s_pass_vs_d0.png" % tag))); count[0] += 1

    for g, gm in groups.items():
        for s, sm in subsets.items():
            m = gm & sm
            if not m.any():
                continue
            tag = "%s_%s" % (g, s)
            lines = [glabel(g), sublab[s]]
            for norm in (False, True):
                distribution(tag, "adphi", XS, m, lines, norm)
                distribution(tag, "res", XS[:-1], m, lines, norm)
                distribution(tag, "mag", XS, m, lines, norm)
            pass_vs_d0(tag, m, lines)
            if g.startswith("sample_"):
                continue
            for x in XS:
                distribution(tag, "adphi", [x], m, lines, False)
                distribution(tag, "mag", [x], m, lines, False)
            for x in XS[:-1]:
                distribution(tag, "res", [x], m, lines, False)

    # mass and lifetime overlays of |dphi(X, mumu)|, one plot per X
    for pre, what in (("mass_", "masses"), ("ctau_", "lifetimes")):
        gs = sorted([g for g in groups if g.startswith(pre)], key=numkey)
        for s, sm in subsets.items():
            for x in XS:
                for norm in (False, True):
                    c = canvas("c_ov_%s_%s_%s_%d" % (pre, s, x, norm))
                    lines = ["%s, all %s" % (XLAB[x], what), sublab[s]]
                    leg, bottom = legend(len(gs), len(lines), cols=2 if len(gs) > 6 else 1, size=0.024 if len(gs) > 6 else 0.030)
                    hs = []
                    for i, g in enumerate(gs):
                        ok = groups[g] & sm & d[x]["okd"]
                        h = hist("ov_%s_%s_%s_%s_%d" % (pre, s, x, g, norm), DPHI_EDGES, d[x]["adphi"][ok])
                        if norm and ok.sum(): h.Scale(1.0 / ok.sum())
                        col = ROOT.TColor.GetColorPalette(int(i * (ROOT.TColor.GetNumberOfColors() - 1) / max(len(gs) - 1, 1)))
                        h.SetLineColor(col); h.SetLineWidth(2)
                        hs.append(h)
                        leg.AddEntry(h, glabel(g).replace(", all c#tau", "").replace(", all masses", ""), "l")
                    top = max(h.GetBinContent(h.GetMaximumBin()) for h in hs) or 1.0
                    frame(hs[0], KINDS["adphi"][1], "Fraction of vertices / 0.05" if norm else "Vertices / 0.05")
                    hs[0].SetMinimum(0); hs[0].SetMaximum(headroom(0, top, bottom)); hs[0].Draw("hist")
                    for h in hs[1:]: h.Draw("hist same")
                    leg.Draw(); labels(c, lines)
                    save(c, hs[0], "overlay_%s%s_%s%s" % (pre.rstrip("_"), "" if s == "all" else "_dsa", x, "_unitarea" if norm else ""),
                         hs, bottom)

    # numbers: fractions, paired test, residual summaries, exclusions
    for s, sm in list(subsets.items()) + [("pair %d-%d" % t, ((ev["typeA"] == t[0]) & (ev["typeB"] == t[1])) |
                                           ((ev["typeA"] == t[1]) & (ev["typeB"] == t[0])))
                                          for t in ((2, 2), (2, 0), (2, 1), (0, 0), (0, 1), (1, 1))]:
        print("SUBSET %s: %d vertices" % (s, int(sm.sum())))
        for x in XS:
            ok = sm & d[x]["okd"]
            p = int((ok & (d[x]["adphi"] < CUT)).sum())
            print("  %-7s |dphi|<%.1f %d / %d = %.4f; excluded: no X %d, zero-length X or mumu %d" %
                  (x, CUT, p, int(ok.sum()), p / max(int(ok.sum()), 1), int((sm & ~d[x]["okm"]).sum()),
                   int((sm & d[x]["okm"] & ~d[x]["okd"]).sum())))
        for x in XS[:-1]:
            ok = sm & d[x]["okr"]
            if ok.any():
                q = np.quantile(d[x]["res"][ok], [0.025, 0.16, 0.5, 0.84, 0.975])
                print("  res %-7s N %d median %+.4f 16-84 width %.4f 2.5-97.5 width %.4f |res|>2.5 %d" %
                      (x, int(ok.sum()), q[2], q[3] - q[1], q[4] - q[0], int((ok & (np.abs(d[x]["res"]) > 2.5)).sum())))
        for var in VARIANTS:
            ok = sm & d["pf"]["okd"] & d[var]["okd"]
            pp, pv = d["pf"]["adphi"] < CUT, d[var]["adphi"] < CUT
            n10, n01 = int((ok & ~pp & pv).sum()), int((ok & pp & ~pv).sum())
            z = (n10 - n01) / np.sqrt(n10 + n01) if n10 + n01 else 0.0
            print("  paired PF vs %s on %d: PF fail/MET' pass %d, PF pass/MET' fail %d, z = %.2f; changed MET %d" %
                  (var, int(ok.sum()), n10, n01, z, int((sm & (ev["nCorr_" + var] > 0)).sum())))
    print("PLOTTED %d linear PNGs and %d log copies" % tuple(count))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--input-dir", required=True); a.add_argument("--report-dir", required=True)
    a.add_argument("--output", required=True); a.add_argument("--revision", required=True)
    a.add_argument("--expected-samples", type=int, default=72); a.add_argument("--expected-events", type=int, default=720000)
    b = sub.add_parser("plot")
    b.add_argument("--output", required=True); b.add_argument("--plot-dir", required=True)
    b.add_argument("--log", action="store_true", help="log copies of the distributions")
    args = p.parse_args()
    {"analyze": analyze, "plot": plot}[args.cmd](args)


if __name__ == "__main__":
    main()
