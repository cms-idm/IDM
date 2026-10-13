#!/usr/bin/env python3
"""Charge-flip plots from charge_flip_check.C output (Level 1). Unweighted counts.

Per x-variable (gen pT, eta, Lxy, |d0|) and per selection (overall, per lifetime pooled over
mass points, per mass point pooled over lifetimes, per sample), one canvas with
  top   : charge-disagreement fraction = N(sign-off St2 match, reco q != gen q) / N(sign-off match)
          (filled) and same-sign-pair fraction (open): disagreeing matches whose partner reco
          (the other gen muon's match) has the same charge as this reco
  bottom: sign-off St2 matching efficiency = N(sign-off match) / N(accepted signal gen)
for displacedMuons inner tracks, global tracks and DSA. Intervals: 68.27% Clopper-Pearson.
Linear axes; entries outside the x range (pT 5-30 GeV, Lxy 0-75 cm, |d0| 0-30 cm) are not shown.
Usage: python3 plot_charge_flip.py out.root plots_dir
"""
import os
import sys
import numpy as np
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)
ROOT.gErrorIgnoreLevel = ROOT.kWarning

infile, outdir = sys.argv[1], sys.argv[2]
cols = ["genQ", "genPt", "genEta", "genVxy", "genD0", "mchi", "dmchi", "ctau"]
for c in ("in", "gl", "dsa"):
    cols += [f"{c}_offIdx", f"{c}_offQ", f"{c}_offOtherQ"]
f = ROOT.TFile.Open(infile)
d = ROOT.RDataFrame(f.Get("genMuons")).AsNumpy(cols)
d["absD0"] = np.abs(d["genD0"])

COLL = [("in", "inner track (dR<0.06)", ROOT.kAzure + 1),
        ("gl", "global track (dR<0.06)", ROOT.kRed + 1),
        ("dsa", "DSA (dR<0.10)", ROOT.kGreen + 2)]
VARS = [
    ("pt", "genPt", "gen p_{T} [GeV]", np.linspace(5, 30, 11), False),
    ("eta", "genEta", "gen #eta", np.linspace(-2.4, 2.4, 17), False),
    ("lxy", "genVxy", "gen L_{xy} [cm]", np.linspace(0, 75, 16), False),
    ("d0", "absD0", "gen |d_{0}| [cm]  (|v_{x}sin#phi - v_{y}cos#phi|)", np.linspace(0, 30, 16), False),
]
CL = 0.682689492


def counts(x, sel, edges):
    return np.histogram(x[sel], bins=edges)[0]


def graph(k, n, edges, logx):
    g = ROOT.TGraphAsymmErrors()
    for i in range(len(edges) - 1):
        if n[i] == 0:
            continue
        p = k[i] / n[i]
        lo = ROOT.TEfficiency.ClopperPearson(int(n[i]), int(k[i]), CL, False)
        hi = ROOT.TEfficiency.ClopperPearson(int(n[i]), int(k[i]), CL, True)
        x = np.sqrt(edges[i] * edges[i + 1]) if logx else 0.5 * (edges[i] + edges[i + 1])
        j = g.GetN()
        g.SetPoint(j, x, p)
        g.SetPointError(j, x - edges[i], edges[i + 1] - x, p - lo, hi - p)
    return g


def make(sel, label, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keep = []
    for vname, branch, xtitle, edges, logx in VARS:
        x = d[branch]
        c = ROOT.TCanvas("c", "", 800, 800)
        top = ROOT.TPad("top", "", 0, 0.34, 1, 1)
        bot = ROOT.TPad("bot", "", 0, 0, 1, 0.34)
        top.SetBottomMargin(0.02); top.SetLeftMargin(0.12); top.SetRightMargin(0.04); top.SetTopMargin(0.13)
        bot.SetTopMargin(0.03); bot.SetBottomMargin(0.30); bot.SetLeftMargin(0.12); bot.SetRightMargin(0.04)
        for p in (top, bot):
            p.SetLogx(logx); p.SetGridy(); p.Draw()
        top.cd()
        tops = []  # this variable's top-panel graphs
        leg = ROOT.TLegend(0.14, 0.66, 0.95, 0.86); leg.SetNColumns(2); leg.SetBorderSize(0)
        leg.SetFillStyle(0); leg.SetTextSize(0.032)
        effs = []
        for c_, name, color in COLL:
            m = sel & (d[f"{c_}_offIdx"] >= 0)
            flip = m & (d[f"{c_}_offQ"] != d["genQ"])
            ss = flip & (d[f"{c_}_offOtherQ"] == d[f"{c_}_offQ"])
            n = counts(x, m, edges)
            g1 = graph(counts(x, flip, edges), n, edges, logx)
            g2 = graph(counts(x, ss, edges), n, edges, logx)
            for g, ms in ((g1, 20), (g2, 24)):
                g.SetMarkerStyle(ms); g.SetMarkerColor(color); g.SetLineColor(color); g.SetMarkerSize(1.0)
                keep.append(g); tops.append(g)
            leg.AddEntry(g1, f"{name}: q_{{reco}} #neq q_{{gen}}", "p")
            leg.AddEntry(g2, "  same-sign reco pair", "p")
            effs.append((graph(n, counts(x, sel, edges), edges, logx), color))
        # linear y: headroom above the largest central value for the legend
        ymax = max([g.GetY()[i] for g in tops for i in range(g.GetN())] + [0.0])
        ymax = min(1.05, max(0.01, 1.7 * ymax))
        fr = top.DrawFrame(edges[0], 0, edges[-1], ymax)
        fr.GetYaxis().SetTitle("fraction of sign-off matches"); fr.GetYaxis().SetTitleSize(0.05)
        fr.GetYaxis().SetTitleOffset(1.1); fr.GetXaxis().SetLabelSize(0)
        for g in tops:
            g.Draw("P same")
        leg.Draw()
        t = ROOT.TLatex(); t.SetNDC(); t.SetTextSize(0.033)
        t.DrawLatex(0.12, 0.95, f"#bf{{{label}}}   N_{{acc gen}} = {int(sel.sum())}")
        t.SetTextSize(0.026)
        t.DrawLatex(0.12, 0.90, "signal gen #mu, station 2, greedy 1-to-1, charge NOT required, unweighted, gen p_{T}>5 GeV")
        bot.cd()
        fb = bot.DrawFrame(edges[0], 0, edges[-1], 1.05)
        fb.GetXaxis().SetTitle(xtitle); fb.GetXaxis().SetTitleSize(0.10); fb.GetXaxis().SetLabelSize(0.09)
        fb.GetXaxis().SetTitleOffset(1.2); fb.GetYaxis().SetTitle("match eff.")
        fb.GetYaxis().SetTitleSize(0.09); fb.GetYaxis().SetLabelSize(0.08); fb.GetYaxis().SetTitleOffset(0.6)
        fb.GetYaxis().SetNdivisions(505)
        for g, color in effs:
            g.SetMarkerStyle(20); g.SetMarkerSize(0.8); g.SetMarkerColor(color); g.SetLineColor(color)
            g.Draw("P same"); keep.append(g)
        c.SaveAs(f"{path}_vs_{vname}.png")
        c.Close()


def fmt(v):
    return (f"{v:g}").replace(".", "p")


allsel = np.ones(len(d["genPt"]), dtype=bool)
make(allsel, "All samples", f"{outdir}/overall/charge_flip")
for ct in sorted(set(d["ctau"])):
    make(d["ctau"] == ct, f"c#tau = {ct:g} mm, all mass points", f"{outdir}/lifetime/ctau-{fmt(ct)}/charge_flip")
masses = sorted(set(zip(d["mchi"], d["dmchi"])))
for mc, dm in masses:
    selm = (d["mchi"] == mc) & (d["dmchi"] == dm)
    make(selm, f"M_{{#chi}} = {mc:g}, #DeltaM = {dm:g} GeV, all c#tau",
         f"{outdir}/mass/Mchi-{fmt(mc)}_dMchi-{fmt(dm)}/charge_flip")
    for ct in sorted(set(d["ctau"][selm])):
        make(selm & (d["ctau"] == ct), f"M_{{#chi}} = {mc:g}, #DeltaM = {dm:g} GeV, c#tau = {ct:g} mm",
             f"{outdir}/sample/Mchi-{fmt(mc)}_dMchi-{fmt(dm)}_ctau-{fmt(ct)}/charge_flip")
print("done", len(masses), "mass points")
