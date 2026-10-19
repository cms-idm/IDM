#!/usr/bin/env python3
"""Clean conditional ID-efficiency plots of the five whole-object ID policies vs gen pT, |d0| and eta.

Usage: python3 plot_id_efficiency_clean.py RUN/id_all.json OUTDIR [--log]
Presentation-only replot of the counts stored by plot_displaced_muon_id.py (no input reprocessing).
ID efficiency = baseline_passed_id / baseline_matched: pre-ID same-charge station-2 greedy 1-to-1
matches held fixed (no rematching after ID). Unweighted; 68.27% Clopper-Pearson intervals.
Stored under/overflow is already folded into the first/last bins; display limits cut no counts.
--log adds log-x copies (pT and |d0|) next to the linear plots.
"""
import argparse
import json
import os
import re
import ROOT

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("input")
parser.add_argument("outdir")
parser.add_argument("--log", action="store_true", help="also write log-x copies for pT and |d0|")
args = parser.parse_args()
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)
ROOT.gErrorIgnoreLevel = ROOT.kWarning

with open(args.input) as handle:
    summary = json.load(handle)
groups = summary["groups"]

# (set, curve key, legend label); all five share the pre-ID matched denominator.
POLICIES = [("all_components", "all", "All present components pass"),
            ("hierarchy", "both", "Hierarchy; I+O: both IDs"),
            ("hierarchy", "tracker", "Hierarchy; I+O: tracker ID"),
            ("hierarchy", "dsa", "Hierarchy; I+O: DSA ID"),
            ("hierarchy", "either", "Hierarchy; I+O: either ID")]
POLICY_FILES = ["all_components", "hierarchy_both", "hierarchy_tracker", "hierarchy_dsa", "hierarchy_either"]
# (name, x title, linear view max, log-x view min or None)
VARS = [("pt", "gen muon p_{T} [GeV]", 30., .25),
        ("d0", "gen muon |d_{0}| [cm] (straight line to (0,0))", 30., .05),
        ("eta", "gen muon #eta", 2.4, None)]
XMIN = {"pt": 0., "d0": 0., "eta": -2.4}
COLORS = [ROOT.kBlack, ROOT.kAzure+1, ROOT.kGreen+2, ROOT.kOrange+7, ROOT.kRed+1, ROOT.kMagenta+1,
          ROOT.kCyan+2, ROOT.kViolet+1, ROOT.kYellow+3, ROOT.kPink+9, ROOT.kSpring-6, ROOT.kTeal+3,
          ROOT.kBlue+2, ROOT.kGray+2, ROOT.kOrange+3, ROOT.kRed-7, ROOT.kAzure-6, ROOT.kGreen-6]
MARKERS = [20, 21, 22, 23, 33, 34, 24, 25, 26, 32, 27, 28, 29, 30, 39, 41, 43, 45]
NOTE = ["ID eff. = passing ID / pre-ID",
        "signal-matched displacedMuons",
        "(match held fixed, no rematch)",
        "Same-charge St2 greedy 1-to-1,",
        "#DeltaR<0.06 (I, G), <0.10 (O)",
        "|#eta|<2.4, L_{xy}<740, |v_{z}|<960 cm",
        "No gen p_{T} cut; unweighted",
        "68% Clopper-Pearson intervals",
        "2022EE iDM#mu, 09-23 production",
        "End bins include under/overflow"]


def graph(k, n, edges, logx):
    g = ROOT.TGraphAsymmErrors()
    for i in range(len(n)):
        if n[i] == 0:
            continue
        p = k[i] / n[i]
        lo = ROOT.TEfficiency.ClopperPearson(int(n[i]), int(k[i]), .682689492, False)
        hi = ROOT.TEfficiency.ClopperPearson(int(n[i]), int(k[i]), .682689492, True)
        x = (edges[i] * edges[i+1]) ** .5 if logx and edges[i] > 0 else (edges[i] + edges[i+1]) / 2
        if logx and edges[i] == 0:
            x = edges[i+1] / 2
        j = g.GetN()
        g.SetPoint(j, x, p)
        g.SetPointError(j, x - edges[i], edges[i+1] - x, p - lo, hi - p)
    return g


def draw(curves, var, header, path, logx, legend_title=""):
    """curves: list of (label, k, n, edges). Frame on the left; legend and definitions on the right."""
    name, xtitle, xmax, logmin = var
    xmin = logmin if logx else XMIN[name]
    graphs, ymin = [], 1.
    for j, (label, k, n, edges) in enumerate(curves):
        g = graph(k, n, edges, logx)
        for i in range(g.GetN()):
            if xmin <= g.GetPointX(i) <= xmax:
                ymin = min(ymin, g.GetPointY(i) - g.GetErrorYlow(i))
        g.SetMarkerStyle(MARKERS[j % len(MARKERS)]); g.SetMarkerSize(1.1)
        g.SetMarkerColor(COLORS[j % len(COLORS)]); g.SetLineColor(COLORS[j % len(COLORS)])
        graphs.append((g, label, int(sum(k)), int(sum(n))))
    ylo = max(0., int((ymin - .05) * 10) / 10.)
    c = ROOT.TCanvas("c", "", 1400, 850)
    c.SetLeftMargin(.08); c.SetRightMargin(.31); c.SetBottomMargin(.12); c.SetTopMargin(.08)
    c.SetGridy(); c.SetLogx(logx)
    frame = c.DrawFrame(xmin, ylo, xmax, 1.01)
    frame.GetXaxis().SetTitle(xtitle); frame.GetYaxis().SetTitle("ID efficiency | matched")
    for axis in (frame.GetXaxis(), frame.GetYaxis()):
        axis.SetTitleSize(.045); axis.SetLabelSize(.04)
    frame.GetXaxis().SetTitleOffset(1.1); frame.GetYaxis().SetTitleOffset(.85)
    if logx:
        frame.GetXaxis().SetMoreLogLabels()
    for g, _, _, _ in graphs:
        g.Draw("P same")
    text = ROOT.TLatex(); text.SetNDC()
    text.SetTextSize(.040); text.DrawLatex(.08, .94, "#bf{" + header + "}")
    many = len(graphs) > 5
    step = .034 if many else .085
    top = .92
    legend = ROOT.TLegend(.70, top - step * len(graphs) - (.04 if legend_title else 0), .99, top)
    legend.SetBorderSize(0); legend.SetFillStyle(0); legend.SetTextSize(.026 if many else .030)
    if legend_title:
        legend.SetHeader(legend_title)
    for g, label, k, n in graphs:
        eff = f"{100. * k / n:.1f}%" if n else "n/a"
        legend.AddEntry(g, f"{label}  ({eff})" if many else f"#splitline{{{label}}}{{{eff} overall, N_{{match}} = {n}}}", "p")
    legend.Draw()
    text.SetTextSize(.022)
    y0 = min(legend.GetY1NDC() - .04, .45)
    for i, line in enumerate(NOTE):
        text.DrawLatex(.71, y0 - .032 * i, line)
    c.SaveAs(path)
    c.Close()


def label_of(group):
    kind, _, tag = group.partition("/")
    if kind == "overall":
        return "All samples (pooled, unweighted)"
    if kind == "lifetime":
        return f"c#tau = {tag[5:].replace('p', '.')} mm, all masses"
    mchi, dm, ctau = re.fullmatch(r"Mchi-([0-9p]+)_dMchi-([0-9p]+)(?:_ctau-([0-9p]+))?", tag).groups()
    text = f"M_{{#chi}} = {mchi.replace('p', '.')}, #DeltaM = {dm.replace('p', '.')} GeV"
    return text + (f", c#tau = {ctau.replace('p', '.')} mm" if ctau else ", all lifetimes")


def curve(group, setname, key, vname):
    v = groups[group]["sets"][setname][vname]
    c = v["curves"][key]
    return c["baseline_passed_id"], c["baseline_matched"], v["edges"]


def outputs(path_base, var, curves, header, legend_title=""):
    draw(curves, var, header, path_base + ".png", False, legend_title)
    if args.log and var[3] is not None:
        draw(curves, var, header, path_base + "_logx.png", True, legend_title)


count = 0
# 1) Five policies overlaid, per group (overall, per lifetime, per mass, per sample).
for group in groups:
    target = os.path.join(args.outdir, "policies", group)
    os.makedirs(target, exist_ok=True)
    for var in VARS:
        curves = [(label, *curve(group, s, key, var[0])) for s, key, label in POLICIES]
        outputs(os.path.join(target, f"id_eff_vs_{var[0]}"), var, curves, label_of(group))
        count += 1
# 2) Per policy: all lifetimes (pooled over masses) and all masses (pooled over lifetimes) overlaid.
for kind in ("lifetime", "mass"):
    members = sorted((g for g in groups if g.startswith(kind + "/")),
                     key=lambda g: tuple(float(x.replace("p", ".")) for x in re.findall(r"[0-9p]+", g.split("/")[1])))
    for (s, key, plabel), pfile in zip(POLICIES, POLICY_FILES):
        target = os.path.join(args.outdir, "overlay_" + kind, pfile)
        os.makedirs(target, exist_ok=True)
        for var in VARS:
            # Short legend labels: the numbers only, units in the legend title.
            curves = [(" / ".join(x.replace("p", ".") for x in re.findall(r"[0-9p]+", g.split("/")[1])),
                       *curve(g, s, key, var[0])) for g in members]
            title = "c#tau [mm]" if kind == "lifetime" else "M_{#chi} / #DeltaM [GeV]"
            header = f"{plabel}: all {'lifetimes (each pooled over masses)' if kind == 'lifetime' else 'masses (each pooled over lifetimes)'}"
            outputs(os.path.join(target, f"id_eff_vs_{var[0]}"), var, curves, header, title)
            count += 1
for s, key, label in POLICIES:
    k, n, _ = curve("overall", s, key, "pt")
    print(f"overall {label}: passed={sum(k)} matched={sum(n)} eff={sum(k)/sum(n):.4f}")
print(f"wrote {count} linear plot sets to {args.outdir}")
