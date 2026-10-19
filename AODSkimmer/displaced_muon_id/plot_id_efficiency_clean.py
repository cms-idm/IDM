#!/usr/bin/env python3
"""Clean conditional ID-efficiency plots vs gen pT, |d0| and eta: six curves on one canvas.

Usage: python3 plot_id_efficiency_clean.py RUN/id_all.json OUTDIR [--log] [--root RUN/id_all.root]
Presentation-level replot of the outputs of plot_displaced_muon_id.py / displaced_muon_id.C (no input
ntuple reprocessing). ID efficiency = passing / pre-ID same-charge station-2 one-to-one matches, held fixed
(no rematching after ID). Curves:
  all pass        all_components "all" (JSON): every present component passes its ID
  inner/outer/global alone   "components" (JSON): each component type matched independently; its own denominator
  any passes      computed here from the genMuons rows: matched object with at least one passing component
                  (pass bits are only set for present components, displaced_muon_id.C Component::get)
  priority        hierarchy "dsa" (JSON): global if present must pass; else outer/DSA (DSA ID); else inner (isTracker)
The row-based code is cross-checked against the JSON "all pass" counts for every group and variable.
Unweighted; 68.27% Clopper-Pearson intervals. Overflow was folded into the stored end bins (pT 100 GeV,
|d0| 740 cm); the view stops at 30 GeV / 30 cm, so bins beyond are hidden but still counted in the totals.
--log adds log-x copies (pT and |d0|) next to the linear plots.
"""
import argparse
import json
import os
import re
import numpy as np
import ROOT

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("input")
parser.add_argument("outdir")
parser.add_argument("--log", action="store_true", help="also write log-x copies for pT and |d0|")
parser.add_argument("--root", help="per-gen rows (default: the input path with .root)")
args = parser.parse_args()
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)
ROOT.gErrorIgnoreLevel = ROOT.kWarning

with open(args.input) as handle:
    summary = json.load(handle)
groups = summary["groups"]

# (source, key, legend label, file stem). Source "rows" = computed here from the genMuons tree.
POLICIES = [("all_components", "all", "All present tracks pass their ID", "all_pass"),
            ("components", "inner", "Inner track alone: tracker-muon flag", "inner_alone"),
            ("components", "outer", "Outer/DSA track alone: DSA ID", "outer_dsa_alone"),
            ("components", "global", "Global track alone: global-muon flag", "global_alone"),
            ("rows", "any", "Any present track passes its ID", "any_pass"),
            ("hierarchy", "dsa", "Priority: global, else outer/DSA, else inner", "priority_global_outer_inner")]
# (name, x title, linear view max, log-x view min or None)
VARS = [("pt", "gen muon p_{T} [GeV]", 30., .25),
        ("d0", "gen muon |d_{0}| [cm] (straight line to (0,0))", 30., .05),
        ("eta", "gen muon #eta", 2.4, None)]
XMIN = {"pt": 0., "d0": 0., "eta": -2.4}
COLORS = [ROOT.kBlack, ROOT.kAzure+1, ROOT.kGreen+2, ROOT.kOrange+7, ROOT.kRed+1, ROOT.kMagenta+1,
          ROOT.kCyan+2, ROOT.kViolet+1, ROOT.kYellow+3, ROOT.kPink+9, ROOT.kSpring-6, ROOT.kTeal+3,
          ROOT.kBlue+2, ROOT.kGray+2, ROOT.kOrange+3, ROOT.kRed-7, ROOT.kAzure-6, ROOT.kGreen-6]
MARKERS = [20, 21, 22, 23, 33, 34, 24, 25, 26, 32, 27, 28, 29, 30, 39, 41, 43, 45]
NOTE = ["#bf{What is plotted}",
        "Fraction of reco muons matched to a signal",
        "gen muon (match made before any ID)",
        "that pass the ID.",
        "#bf{Track IDs}",
        "inner: tracker-muon flag",
        "outer/DSA: DSA quality ID",
        "global: global-muon flag",
        "#bf{\"alone\" curves}",
        "each track type matched on its own,",
        "ignoring the muon's other tracks.",
        "#bf{Matching}",
        "same charge, smallest #DeltaR at muon station 2",
        "(<0.06 inner/global, <0.10 outer/DSA),",
        "each gen and reco muon used once.",
        "#bf{Sample}",
        "2022EE iDM#mu signal MC (09-23 prod.),",
        "|#eta|<2.4, L_{xy}<740 cm, any p_{T}; unweighted.",
        "Bars: 68% CL. Plots stop at 30 GeV / 30 cm;",
        "points beyond count in the overall %."]


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
    frame.GetXaxis().SetTitle(xtitle); frame.GetYaxis().SetTitle("ID efficiency of matched muons")
    for axis in (frame.GetXaxis(), frame.GetYaxis()):
        axis.SetTitleSize(.045); axis.SetLabelSize(.04)
    frame.GetXaxis().SetTitleOffset(1.1); frame.GetYaxis().SetTitleOffset(.85)
    if logx:
        frame.GetXaxis().SetMoreLogLabels()
    for g, _, _, _ in graphs:
        g.Draw("P same")
    text = ROOT.TLatex(); text.SetNDC(); text.SetTextFont(42)
    text.SetTextSize(.038); text.DrawLatex(.08, .94, "#bf{" + header + "}")
    # Side panel: legend on top (two columns when there are many curves), definitions below.
    many = len(graphs) > 6
    ncol = 2 if many else 1
    rows = (len(graphs) + ncol - 1) // ncol
    step = .030 if many else .062
    top = .97
    bottom = top - step * rows - (.035 if legend_title else 0)
    legend = ROOT.TLegend(.695, bottom, .995, top)
    legend.SetTextFont(42); legend.SetNColumns(ncol); legend.SetBorderSize(0); legend.SetFillStyle(0)
    legend.SetTextSize(.021 if many else .025)
    if legend_title:
        legend.SetHeader(legend_title)
    for g, label, k, n in graphs:
        eff = f"{100. * k / n:.1f}%" if n else "n/a"
        legend.AddEntry(g, f"{label} ({eff})" if many else
                        f"#splitline{{{label}}}{{{eff} overall, {n} matched}}", "p")
    legend.Draw()
    text.SetTextSize(.020)
    spacing = .0235
    y = bottom - .03
    for line in NOTE:
        text.DrawLatex(.70, y, line)
        y -= spacing
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


def fmt(v):
    return f"{v:g}".replace(".", "p")


# Per-gen rows for the "any passes" curve, binned exactly like counts() in plot_displaced_muon_id.py.
rows = ROOT.RDataFrame("genMuons", args.root or os.path.splitext(args.input)[0] + ".root").AsNumpy(
    ["genPt", "genEta", "genD0", "mchi", "dmchi", "ctau", "objIdx", "objMask", "objIdBits"])
rows["absD0"] = np.abs(rows["genD0"])
ROW_X = {"pt": "genPt", "eta": "genEta", "d0": "absD0"}
matched = rows["objIdx"] >= 0
ROW_CURVES = {"any": (matched & (rows["objIdBits"] != 0), matched),
              "all": (matched & (rows["objMask"] != 0) & (rows["objIdBits"] == rows["objMask"]), matched)}
selection = {"overall": np.ones(len(matched), dtype=bool)}
for ct in set(rows["ctau"]):
    selection["lifetime/ctau-" + fmt(ct)] = rows["ctau"] == ct
for mc, dm in set(zip(rows["mchi"], rows["dmchi"])):
    selm = (rows["mchi"] == mc) & (rows["dmchi"] == dm)
    selection[f"mass/Mchi-{fmt(mc)}_dMchi-{fmt(dm)}"] = selm
    for ct in set(rows["ctau"][selm]):
        selection[f"sample/Mchi-{fmt(mc)}_dMchi-{fmt(dm)}_ctau-{fmt(ct)}"] = selm & (rows["ctau"] == ct)
if set(selection) != set(groups):
    raise RuntimeError("Row groups differ from JSON groups")


def row_counts(group, key, vname):
    edges = np.array(groups[group]["sets"]["all_components"][vname]["edges"])
    x = rows[ROW_X[vname]]
    folded = lambda sel: np.histogram(np.clip(x[sel], edges[0], np.nextafter(edges[-1], -np.inf)), bins=edges)[0]
    passed, base = ROW_CURVES[key]
    sel = selection[group]
    return folded(sel & passed).tolist(), folded(sel & base).tolist(), edges.tolist()


def curve(group, source, key, vname):
    if source == "rows":
        return row_counts(group, key, vname)
    v = groups[group]["sets"][source][vname]
    c = v["curves"][key]
    return c["baseline_passed_id"], c["baseline_matched"], v["edges"]


# The row-based binning must reproduce the stored all-pass counts everywhere.
for group in groups:
    for vname in ROW_X:
        if row_counts(group, "all", vname)[:2] != list(curve(group, "all_components", "all", vname)[:2]):
            raise RuntimeError(f"Row counts disagree with JSON for {group} {vname}")
print(f"cross-check: row-based all-pass counts match JSON for {len(groups)} groups x {len(ROW_X)} variables")


def outputs(path_base, var, curves, header, legend_title=""):
    draw(curves, var, header, path_base + ".png", False, legend_title)
    if args.log and var[3] is not None:
        draw(curves, var, header, path_base + "_logx.png", True, legend_title)


count = 0
# 1) Six curves overlaid, per group (overall, per lifetime, per mass, per sample).
for group in groups:
    target = os.path.join(args.outdir, "policies", group)
    os.makedirs(target, exist_ok=True)
    for var in VARS:
        curves = [(label, *curve(group, s, key, var[0])) for s, key, label, _ in POLICIES]
        outputs(os.path.join(target, f"id_eff_vs_{var[0]}"), var, curves, label_of(group))
        count += 1
# 2) Per policy: all lifetimes (pooled over masses) and all masses (pooled over lifetimes) overlaid.
for kind in ("lifetime", "mass"):
    members = sorted((g for g in groups if g.startswith(kind + "/")),
                     key=lambda g: tuple(float(x.replace("p", ".")) for x in re.findall(r"[0-9p]+", g.split("/")[1])))
    for s, key, plabel, pfile in POLICIES:
        target = os.path.join(args.outdir, "overlay_" + kind, pfile)
        os.makedirs(target, exist_ok=True)
        for var in VARS:
            # Short legend labels: the numbers only, units in the legend title.
            curves = [(" / ".join(x.replace("p", ".") for x in re.findall(r"[0-9p]+", g.split("/")[1])),
                       *curve(g, s, key, var[0])) for g in members]
            title = "c#tau [mm]" if kind == "lifetime" else "M_{#chi} / #DeltaM [GeV]"
            header = f"{plabel}: per {'lifetime (masses pooled)' if kind == 'lifetime' else 'mass (lifetimes pooled)'}"
            outputs(os.path.join(target, f"id_eff_vs_{var[0]}"), var, curves, header, title)
            count += 1
for s, key, label, _ in POLICIES:
    k, n, _ = curve("overall", s, key, "pt")
    print(f"overall {label}: passed={sum(k)} matched={sum(n)} eff={sum(k)/sum(n):.4f}")
print(f"wrote {count} linear plot sets to {args.outdir}")
