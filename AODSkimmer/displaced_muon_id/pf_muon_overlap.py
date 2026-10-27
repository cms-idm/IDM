#!/usr/bin/env python3
"""Nearest PF muon to each hierarchy-matched signal leg: PF-match and same-sign fractions, relative pT difference.

python3 pf_muon_overlap.py analyze --input-dir RUN/inputs --report-dir RUN/logs --revision <sha> --output RUN/pf_overlap.root
python3 pf_muon_overlap.py plot --output RUN/pf_overlap.root --plot-dir RUN/plots [--log]
Study note: kb/studies/displaced-pf-muon-overlap.md. Inputs: PreselectedVertexing with pfNearest=True.
Legs: twoAcc events, signal gen muons matched by the staged G>I>DSA hierarchy (matchComp), object level, each leg on
its own. Leg = matched component's track at its reference point. PF muons: all packedPFCandidates |pdgId|==13.
PF-matched: nearest PF muon with 0 <= dR < 0.2 (strict). Same sign: q_PF == q_leg among PF-matched legs.
r = (pT_leg - pT_PF)/pT_leg among PF-matched legs. x-variables: gen |d0| (straight line), gen Lxy, gen pT.
Raw unweighted counts, unweighted pooling; Sumw2, flow bins kept. Fractions: TEfficiency Clopper-Pearson 68.3%,
approximate (the two legs of an event can be correlated). plot reads only the ROOT output; --log adds log copies.
"""
import argparse
from array import array
from pathlib import Path
import re
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preselected_vertexing_analysis import check_report, flowcounts

TYPES = (("all", "All types", None), ("global", "Global", 2), ("inner", "Inner", 0), ("dsa", "DSA", 1))
X_EDGES = {"d0": np.arange(0, 60.1, 2.0), "lxy": np.arange(0, 120.1, 4.0), "pt": np.arange(0, 60.1, 2.0)}
X_TITLES = {"d0": "Gen |d_{0}| [cm]", "lxy": "Gen L_{xy} [cm]", "pt": "Gen p_{T} [GeV]"}
DIST_EDGES = {"dr": np.round(np.arange(0, 1.001, 0.01), 2), "r": np.round(np.arange(-1, 1.001, 0.02), 2),
              "n02": np.arange(0, 6.0) - 0.5}
DIST_TITLES = {"dr": "#DeltaR(leg, nearest PF muon)", "r": "(p_{T}^{leg} #minus p_{T}^{PF}) / p_{T}^{leg}",
               "n02": "PF muons with #DeltaR < 0.2"}
SCALAR = ("run", "lumi", "event", "twoAcc")
VECTOR = ("sigQ", "sigPt", "sigLxy", "sigD0", "matchComp", "matchType", "legPt", "legQ", "legPfN02", "legPfIdx",
          "legPfDR", "legPfPt", "legPfQ")
MIN_MEDIAN = 10  # legs per bin needed to draw a median point


def load(args):
    import awkward as ak
    import uproot
    paths = sorted(Path(args.input_dir).glob("*.root"))
    if len(paths) != args.expected_samples:
        raise RuntimeError("Input sample count mismatch: %d" % len(paths))
    names, revisions, legs, n_events = [], set(), [], 0
    for si, path in enumerate(paths):
        with uproot.open(path) as f:
            prefix = "preselectedVertexing/"
            name = str(f[prefix + "sample"].member("fTitle"))
            if name in names or not re.fullmatch(r"Mchi-[\dp]+_dMchi-[\dp]+_ctau-[\dp]+", name):
                raise RuntimeError("Unexpected or duplicate sample: " + name)
            names.append(name)
            revisions.add(str(f[prefix + "revision"].member("fTitle")))
            if str(f[prefix + "hierarchyOrder"].member("fTitle")) != "global>inner>outer(DSA)":
                raise RuntimeError("Unexpected hierarchy order: " + name)
            if str(f[prefix + "pfNearest"].member("fTitle")) == "disabled":
                raise RuntimeError("pfNearest was disabled: " + name)
            t = f[prefix + "events"]
            check_report(Path(args.report_dir), name, t.num_entries)
            n_events += t.num_entries
            a = t.arrays(list(SCALAR + VECTOR), library="ak")
        two = np.asarray(a["twoAcc"]).astype(bool)
        for k in VECTOR:
            num = np.asarray(ak.num(a[k]))
            if np.any(num[two] != 2):
                raise RuntimeError("Non-pair vector %s in %s" % (k, name))
            if k.startswith("leg") or k.startswith("match"):
                if np.any(num[~two] != 0):
                    raise RuntimeError("%s filled outside twoAcc in %s" % (k, name))
        v = {k: ak.to_numpy(a[k][two]).ravel() for k in VECTOR}  # two legs per twoAcc event, gen-index order
        v["sample"] = np.full(len(v["sigQ"]), si, dtype=np.int32)
        v["evrow"] = np.repeat(np.flatnonzero(two), 2)
        legs.append(v)
    if len(revisions) != 1 or "unspecified" in revisions:
        raise RuntimeError("Mixed/unspecified input revisions")
    if n_events != args.expected_events:
        raise RuntimeError("Event total mismatch: %d" % n_events)
    e = {k: np.concatenate([v[k] for v in legs]) for k in legs[0]}
    return names, e, revisions.pop(), n_events


def analyze(args):
    import ROOT
    ROOT.gROOT.SetBatch(True)
    if Path(args.output).exists():
        raise RuntimeError("Refusing to overwrite numerical ROOT output")
    names, e, revision, n_events = load(args)
    matched = e["matchComp"] >= 0
    if np.any(matched != (e["matchType"] >= 0)) or np.any(matched != (e["legPfN02"] >= 0)):
        raise RuntimeError("Leg match flags disagree")
    if np.any(matched & (e["legQ"] != e["sigQ"])):
        raise RuntimeError("Matched leg charge differs from gen (same-charge matching required)")
    count = {"matched legs": int(matched.sum())}
    legs = matched
    has_pf = legs & (e["legPfIdx"] >= 0)
    count["no PF muon in event"] = int((legs & ~has_pf).sum())
    if np.any(has_pf & ~(np.isfinite(e["legPfDR"]) & (e["legPfDR"] >= 0))):
        raise RuntimeError("Valid PF index with invalid dR")
    pfm = has_pf & (e["legPfDR"] < 0.2)
    if np.any(pfm != (legs & (e["legPfN02"] > 0))):
        raise RuntimeError("legPfN02 disagrees with the nearest dR")
    bad_pfpt = pfm & ~np.isfinite(e["legPfPt"])
    if bad_pfpt.any():
        raise RuntimeError("PF-matched leg with non-finite PF pT")
    same = pfm & (e["legPfQ"] == e["legQ"])
    # r only where the leg pT is finite and > 0 (excluded PF-matched legs are counted).
    rok = pfm & np.isfinite(e["legPt"]) & (e["legPt"] > 0)
    count["PF-matched, r excluded (leg pT not finite or <= 0)"] = int((pfm & ~rok).sum())
    r = np.where(rok, (e["legPt"] - e["legPfPt"]) / np.where(rok, e["legPt"], 1), np.nan)
    # Sharing: both legs of the event PF-matched to the same event-local packed index.
    pair = lambda x: x.reshape(-1, 2)
    shared_ev = pair(pfm).all(axis=1) & (pair(e["legPfIdx"])[:, 0] == pair(e["legPfIdx"])[:, 1])
    shared = np.repeat(shared_ev, 2)
    count.update({"PF-matched (dR<0.2)": int(pfm.sum()), "same sign": int(same.sum()),
                  "opposite sign": int((pfm & ~same).sum()), ">= 2 PF muons within 0.2": int((legs & (e["legPfN02"] >= 2)).sum()),
                  "shared PF muon (legs)": int(shared.sum())})
    x = {"d0": e["sigD0"], "lxy": e["sigLxy"], "pt": e["sigPt"]}

    groups = {"overall": np.ones(len(legs), bool)}
    for kind, fn in (("mass", lambda s: s.rsplit("_ctau-", 1)[0]),
                     ("ctau", lambda s: "ctau-" + s.rsplit("_ctau-", 1)[1])):
        for key in sorted({fn(s) for s in names}):
            groups[kind + "/" + key] = np.isin(e["sample"], [i for i, s in enumerate(names) if fn(s) == key])
    for si, name in enumerate(names):
        groups["sample/" + name] = e["sample"] == si

    root = ROOT.TFile.Open(str(args.output), "CREATE")
    ROOT.TNamed("definitions", __doc__).Write()
    ROOT.TNamed("inputRevision", revision).Write()
    ROOT.TNamed("analysisRevision", args.revision).Write()
    ROOT.TNamed("samples", "\n".join(names)).Write()

    def hist(name, title, edges, values):
        h = ROOT.TH1D(name, title, len(edges) - 1, array("d", edges))
        h.Sumw2()
        for b, c in enumerate(flowcounts(values, edges)):
            h.SetBinContent(b, float(c)); h.SetBinError(b, float(np.sqrt(c)))
        h.Write()

    for group, gm in groups.items():
        root.mkdir(group); root.cd(group)
        for tag, label, t in TYPES:
            sel = gm & (e["matchType"] == t if t is not None else True)
            for var, edges in X_EDGES.items():
                xt = X_TITLES[var]
                hist("den_%s_%s" % (var, tag), "%s matched legs;%s;Legs" % (label, xt), edges, x[var][sel & legs])
                hist("pf_%s_%s" % (var, tag), "%s PF-matched legs;%s;Legs" % (label, xt), edges, x[var][sel & pfm])
                hist("ss_%s_%s" % (var, tag), "%s PF-matched same-sign legs;%s;Legs" % (label, xt), edges, x[var][sel & same])
            hist("dr_%s" % tag, "%s legs with a PF muon;%s;Legs" % (label, DIST_TITLES["dr"]), DIST_EDGES["dr"],
                 e["legPfDR"][sel & has_pf])
            hist("r_%s" % tag, "%s PF-matched legs;%s;Legs" % (label, DIST_TITLES["r"]), DIST_EDGES["r"], r[sel & rok])
            hist("n02_%s" % tag, "%s matched legs;%s;Legs" % (label, DIST_TITLES["n02"]), DIST_EDGES["n02"],
                 e["legPfN02"][sel & legs])
        root.cd()

    # Exact per-leg values: medians and any rebinning without rereading the inputs.
    t = ROOT.TTree("legs", "Matched legs, one row per leg; r is NaN unless PF-matched with finite leg pT > 0")
    ints = {k: array("i", [0]) for k in ("sample", "type", "hasPF", "pfMatched", "sameSign", "n02", "shared", "pfIdx")}
    floats = {k: array("d", [0]) for k in ("d0", "lxy", "pt", "legPt", "pfPt", "dr", "r")}
    for k, b in ints.items(): t.Branch(k, b, k + "/I")
    for k, b in floats.items(): t.Branch(k, b, k + "/D")
    src_i = {"sample": e["sample"], "type": e["matchType"], "hasPF": has_pf, "pfMatched": pfm, "sameSign": same,
             "n02": e["legPfN02"], "shared": shared, "pfIdx": e["legPfIdx"]}
    src_f = {"d0": x["d0"], "lxy": x["lxy"], "pt": x["pt"], "legPt": e["legPt"], "pfPt": e["legPfPt"],
             "dr": e["legPfDR"], "r": r}
    for i in np.flatnonzero(legs):
        for k in ints: ints[k][0] = int(src_i[k][i])
        for k in floats: floats[k][0] = float(src_f[k][i])
        t.Fill()
    t.Write()
    acc = ROOT.TTree("accounting", "Raw leg counts, overall")
    an, av = ROOT.std.string(), array("i", [0])
    acc.Branch("condition", an); acc.Branch("n", av, "n/I")
    for k, v in count.items():
        an.assign(k); av[0] = v; acc.Fill()
        print("ACCOUNT %-40s %d" % (k, v))
    acc.Write()
    for tag, label, tt in TYPES:
        sel = legs & (e["matchType"] == tt if tt is not None else True)
        n, npf, nss = int(sel.sum()), int((sel & pfm).sum()), int((sel & same).sum())
        print("TYPE %-6s legs %d PF-matched %d (%.4f) same-sign %d (%.4f of matched) r_median %.4g" %
              (tag, n, npf, npf / n if n else np.nan, nss, nss / npf if npf else np.nan,
               np.median(r[sel & rok]) if (sel & rok).any() else np.nan))
    root.Close()
    print("VALIDATED %d samples, %d events, %d legs" % (len(names), n_events, int(legs.sum())), flush=True)


def plot(args):
    import ROOT
    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0); ROOT.gStyle.SetOptTitle(0)
    ROOT.gStyle.SetTextFont(42); ROOT.gStyle.SetLabelFont(42, "XYZ"); ROOT.gStyle.SetTitleFont(42, "XYZ")
    ROOT.gStyle.SetPadTickX(1); ROOT.gStyle.SetPadTickY(1)
    ROOT.TH1.AddDirectory(False)
    ROOT.TGaxis.SetMaxDigits(6)
    out = Path(args.plot_dir)
    out.mkdir(parents=True, exist_ok=True)
    f = ROOT.TFile.Open(str(args.output))
    names = str(f.Get("samples").GetTitle()).split("\n")
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#F0E442"]
    color = lambda i: ROOT.TColor.GetColor(palette[i % len(palette)])
    style = lambda i: 1 + (i // len(palette))
    keep = []
    count = [0]

    def canvas(name):
        c = ROOT.TCanvas(name, "", 800, 700)
        c.SetLeftMargin(0.15); c.SetRightMargin(0.04); c.SetTopMargin(0.08); c.SetBottomMargin(0.13)
        keep.append(c)
        return c

    def frame(h, xt, yt):
        h.GetXaxis().SetTitle(xt); h.GetYaxis().SetTitle(yt)
        for ax in (h.GetXaxis(), h.GetYaxis()):
            ax.SetTitleSize(0.05); ax.SetLabelSize(0.042)
        h.GetXaxis().SetTitleOffset(1.15); h.GetYaxis().SetTitleOffset(1.45)

    def labels(c, lines):
        c.cd()
        t = ROOT.TLatex(); t.SetNDC(); keep.append(t)
        t.SetTextFont(61); t.SetTextSize(0.05); t.SetTextAlign(11); t.DrawLatex(0.15, 0.94, "CMS")
        t.SetTextFont(52); t.SetTextSize(0.038); t.DrawLatex(0.255, 0.94, "Simulation Private Work")
        t.SetTextFont(42); t.SetTextAlign(31); t.DrawLatex(0.96, 0.94, "2022EE (13.6 TeV)")
        t.SetTextSize(0.032); t.SetTextAlign(13)
        for i, line in enumerate(lines): t.DrawLatex(0.19, 0.88 - 0.042 * i, line)

    def legend(n):
        leg = ROOT.TLegend(0.55, max(0.30, 0.88 - 0.038 * n), 0.94, 0.88); keep.append(leg)
        leg.SetBorderSize(0); leg.SetFillStyle(0); leg.SetTextSize(0.028 if n > 8 else 0.033)
        if n > 9: leg.SetNColumns(2)
        return leg

    def save(c, fr, name, log_bottom=None):
        c.Modified(); c.SaveAs(str(out / (name + ".png"))); count[0] += 1
        if args.log and log_bottom is not None:
            top = fr.GetMaximum()
            c.SetLogy(1); fr.SetMinimum(log_bottom); fr.SetMaximum(top * 30)
            c.Modified(); c.SaveAs(str(out / (name + "_log.png"))); c.SetLogy(0)
            fr.SetMinimum(0); fr.SetMaximum(top)

    def get(group, name):
        h = f.Get("%s/%s" % (group, name))
        if not h:
            raise RuntimeError("Missing %s/%s" % (group, name))
        return h

    # Series sets: "overall" compares types; mass and ctau overlays compare points, one figure per type.
    masses = sorted({s.rsplit("_ctau-", 1)[0] for s in names}, key=lambda s: float(s.split("_")[0][5:].replace("p", ".")))
    ctaus = sorted({"ctau-" + s.rsplit("_ctau-", 1)[1] for s in names}, key=lambda s: float(s[5:].replace("p", ".")))
    pretty = lambda g: (g.split("/")[1].replace("Mchi-", "m_{#chi} = ").replace("_dMchi-", ", #Deltam = ").replace("p", ".") + " GeV"
                        if g.startswith("mass/") else "c#tau = " + g.split("-")[1].replace("p", ".") + " mm")
    figures = [("overall", "overall", [("overall", tag, label) for tag, label, _ in TYPES], "Matched legs, all samples")]
    for tag, label, _ in TYPES:
        figures.append(("ctau_" + tag, "c#tau overlay", [("ctau/" + c, tag, pretty("ctau/" + c)) for c in ctaus],
                        label + " legs, pooled over masses"))
        figures.append(("mass_" + tag, "mass overlay", [("mass/" + m, tag, pretty("mass/" + m)) for m in masses],
                        label + " legs, pooled over c#tau"))

    tree = f.Get("legs")
    sample_of = {}
    for g in [s for fig in figures for (s, _, _) in fig[2]]:
        if g == "overall":
            sample_of[g] = None
        else:
            key = g.split("/", 1)[1]
            sample_of[g] = [i for i, s in enumerate(names)
                            if (s.rsplit("_ctau-", 1)[0] == key if g.startswith("mass/") else "ctau-" + s.rsplit("_ctau-", 1)[1] == key)]
    cols = {k: np.asarray(v) for k, v in
            ROOT.RDataFrame(tree).AsNumpy(["sample", "type", "pfMatched", "d0", "lxy", "pt", "r"]).items()}
    type_code = {tag: t for tag, _, t in TYPES}

    for fname, kind, series, head in figures:
        # Fractions vs x: PF-match among matched legs; same sign among PF-matched legs.
        for var in X_EDGES:
            for metric, num, den, ytitle in (("pfmatch", "pf", "den", "Fraction with a PF muon within #DeltaR < 0.2"),
                                             ("samesign", "ss", "pf", "Same-sign fraction of PF-matched legs")):
                c = canvas("c_%s_%s_%s" % (fname, metric, var))
                fr = ROOT.TH1D("fr_%s_%s_%s" % (fname, metric, var), "", 1, X_EDGES[var][0], X_EDGES[var][-1]); keep.append(fr)
                fr.SetMinimum(0); fr.SetMaximum(1.45 if len(series) <= 4 else 1.9)
                frame(fr, X_TITLES[var], ytitle); fr.Draw("axis")
                leg = legend(len(series))
                for i, (g, tag, label) in enumerate(series):
                    p, d = get(g, "%s_%s_%s" % (num, var, tag)), get(g, "%s_%s_%s" % (den, var, tag))
                    if d.Integral(0, d.GetNbinsX() + 1) == 0:
                        continue
                    eff = ROOT.TEfficiency(p, d); keep.append(eff)
                    gr = eff.CreateGraph(); keep.append(gr)
                    gr.SetLineColor(color(i)); gr.SetMarkerColor(color(i)); gr.SetLineStyle(style(i))
                    gr.SetMarkerStyle(20 + i % 4); gr.SetMarkerSize(0.8); gr.SetLineWidth(2)
                    gr.Draw("PZ same")
                    over = (int(d.GetBinContent(d.GetNbinsX() + 1)))
                    leg.AddEntry(gr, "%s (%d legs%s)" % (label, d.Integral(0, d.GetNbinsX() + 1),
                                                       ", %d overflow" % over if over and len(series) <= 4 else ""), "lp")
                leg.Draw()
                labels(c, [head])
                save(c, fr, "%s_%s_vs_%s" % (fname, metric, var))
        # Median r vs x, q16-q84 band as error bars, bins with >= MIN_MEDIAN PF-matched legs.
        for var, edges in X_EDGES.items():
            allpts = []
            for i, (g, tag, label) in enumerate(series):
                m = (cols["pfMatched"] == 1) & np.isfinite(cols["r"])
                if sample_of[g] is not None: m &= np.isin(cols["sample"], sample_of[g])
                if type_code[tag] is not None: m &= cols["type"] == type_code[tag]
                pts = []
                for lo, hi in zip(edges[:-1], edges[1:]):
                    v = cols["r"][m & (cols[var] >= lo) & (cols[var] < hi)]
                    if len(v) >= MIN_MEDIAN:
                        q16, q50, q84 = np.percentile(v, [16, 50, 84])
                        pts.append(((lo + hi) / 2, q50, q50 - q16, q84 - q50))
                allpts.append((i, label, int(m.sum()), pts))
            lows = [p[1] - p[2] for _, _, _, pts in allpts for p in pts] or [-0.5]
            highs = [p[1] + p[3] for _, _, _, pts in allpts for p in pts] or [0.5]
            ylo, yhi = min(min(lows), 0.0), max(max(highs), 0.0)
            span = max(yhi - ylo, 0.1)
            c = canvas("c_%s_rmed_%s" % (fname, var))
            fr = ROOT.TH1D("frm_%s_%s" % (fname, var), "", 1, edges[0], edges[-1]); keep.append(fr)
            # Room above the data for the labels and legend (taller legend for many series).
            fr.SetMinimum(ylo - 0.08 * span); fr.SetMaximum(yhi + span * (0.8 if len(series) <= 4 else 1.4))
            frame(fr, X_TITLES[var], "Median (p_{T}^{leg} #minus p_{T}^{PF}) / p_{T}^{leg}"); fr.Draw("axis")
            zero = ROOT.TLine(edges[0], 0, edges[-1], 0); zero.SetLineStyle(3); zero.SetLineColor(ROOT.kGray + 2); zero.Draw(); keep.append(zero)
            leg = legend(len(series))
            for i, label, nlegs, pts in allpts:
                if not pts:
                    continue
                gr = ROOT.TGraphAsymmErrors(len(pts)); keep.append(gr)
                for j, (xc, q50, dlo, dhi) in enumerate(pts):
                    gr.SetPoint(j, xc, q50); gr.SetPointError(j, 0, 0, dlo, dhi)
                gr.SetLineColor(color(i)); gr.SetMarkerColor(color(i)); gr.SetLineStyle(style(i))
                gr.SetMarkerStyle(20 + i % 4); gr.SetMarkerSize(0.8); gr.SetLineWidth(2)
                gr.Draw("PZ same")
                leg.AddEntry(gr, "%s (%d legs)" % (label, nlegs), "lp")
            leg.Draw()
            labels(c, [head, "Bars: 16th#minus84th percentile"])
            save(c, fr, "%s_rmedian_vs_%s" % (fname, var))
        # Distributions: nearest-PF dR (legs with a PF muon) and r (PF-matched legs); unnormalized and unit area.
        for dist in ("dr", "r"):
            for norm in (False, True):
                c = canvas("c_%s_%s_%d" % (fname, dist, norm))
                hs = []
                for i, (g, tag, label) in enumerate(series):
                    h = get(g, "%s_%s" % (dist, tag)).Clone("h_%s_%s_%s_%d_%d" % (fname, dist, tag, i, norm)); keep.append(h)
                    tot = h.Integral(0, h.GetNbinsX() + 1)
                    if tot == 0:
                        continue
                    if norm: h.Scale(1.0 / tot)  # unit area including flow bins
                    h.SetLineColor(color(i)); h.SetLineStyle(style(i)); h.SetLineWidth(2)
                    hs.append((h, label, tot))
                if not hs:
                    continue
                top = max(h.GetBinContent(h.GetMaximumBin()) for h, _, _ in hs)
                fr = hs[0][0]
                frame(fr, DIST_TITLES[dist], ("Fraction of legs / %g" if norm else "Legs / %g") % (fr.GetBinWidth(1)))
                fr.SetMinimum(0); fr.SetMaximum(top * (1.5 if len(hs) <= 4 else 2.0))
                fr.Draw("hist")
                for h, _, _ in hs[1:]: h.Draw("hist same")
                leg = legend(len(hs))
                for h, label, tot in hs:
                    leg.AddEntry(h, "%s (%d)" % (label, tot), "l")
                leg.Draw()
                if dist == "dr":
                    ln = ROOT.TLine(0.2, 0, 0.2, fr.GetMaximum()); ln.SetLineStyle(3); ln.SetLineColor(ROOT.kGray + 2); ln.Draw(); keep.append(ln)
                flow = "overflow #geq 1" if dist == "dr" else "under/overflow outside [#minus1, 1]"
                labels(c, [head, ("Unit area incl. %s" if norm else "Not drawn: %s") % flow])
                save(c, fr, "%s_%s%s" % (fname, dist, "_unitarea" if norm else ""),
                     log_bottom=(0.5 / min(t for _, _, t in hs)) if norm else 0.5)
    print("PLOTTED %d linear PNGs" % count[0] + (" plus log copies of the distributions" if args.log else ""))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--input-dir", required=True); a.add_argument("--report-dir", required=True)
    a.add_argument("--output", required=True); a.add_argument("--revision", required=True)
    a.add_argument("--expected-samples", type=int, default=72); a.add_argument("--expected-events", type=int, default=720000)
    b = sub.add_parser("plot")
    b.add_argument("--output", required=True); b.add_argument("--plot-dir", required=True)
    b.add_argument("--log", action="store_true", help="log copies of the distributions (fractions stay linear)")
    args = p.parse_args()
    {"analyze": analyze, "plot": plot}[args.cmd](args)


if __name__ == "__main__":
    main()
