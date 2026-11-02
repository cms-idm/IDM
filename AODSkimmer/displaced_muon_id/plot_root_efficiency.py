#!/usr/bin/env python3
"""Render saved raw-count efficiency histograms in the clean CMS talk style.

This is the portable rendering tool for a plotting skill. It depends only on
Python >=3.9 and PyROOT (run after cmsenv); it does not import the muon study,
read event trees, select physics objects, normalize, rebin, or modify ROOT files.
The study-specific analysis remains in mumu_talk_plots.py.

Each --series supplies seven shell-quoted fields:
    NUMERATOR_KEY DENOMINATOR_KEY GROUP_LABEL STAGE_LABEL COLOR LINE MARKER
Keys are full paths inside --input. Labels use ROOT TLatex syntax, colors use
#RRGGBB, and LINE/MARKER are ROOT style integers. Color identifies a group;
line/marker style identifies a stage. Repeated group/stage labels must have
consistent styles. Use short labels in the keys; place context in --annotation.

Example (substitute actual histogram keys and an UNUSED output basename):
    python3 -I plot_root_efficiency.py --input saved.root \\
      --series group/num group/den 'Signal' 'Matched' '#0072B2' 2 24 \\
      --series group/valid group/den 'Signal' 'Matched and vertexed' '#0072B2' 1 20 \\
      --xlabel 'Gen L_{xy} [cm]' --ylabel 'Event efficiency' \\
      --status 'Simulation Private Work' --era '2022EE (13.6 TeV)' \\
      --annotation 'Both signal muons accepted' --xlim 0 100 --ylim 0 0.8 \\
      --output-base /path/to/new/efficiency

Input contract:
* One-dimensional TH1 numerator/denominator pairs, with identical bin edges
  across the overlay. Contents must be finite integer counts, 0 <= num <= den.
  Stored Sumw2, when present, must agree with Poisson count variances. These
  checks catch many weighted inputs; the caller must still establish that the
  numerator is a subset of the denominator and both are raw, unweighted counts.
* Points use num/den and exact central Clopper-Pearson intervals at --confidence.
  The minimum denominator is a display mask, not a change of population. Empty
  denominator bins are omitted even when --min-den is zero. Flow bins are checked
  but not drawn. They remain in the untouched input ROOT file.
* --xlim crops the display, including error bars at its boundary; it does not
  rebin or recount. Labels must describe the actual population, matching,
  exclusions and weighting; the tool cannot infer these from histogram names.

Outputs are PNG + vector PDF; existing outputs are refused. --log adds a log-y
PNG alongside the linear exports (zero efficiencies are invisible on log-y).
After every new configuration, inspect the PNG AND a rendered PDF for readable
labels, correct ranges, unobscured points/errors, and no clipping or overlap:
    pdftoppm -cropbox -scale-to 900 -singlefile -png plot.pdf preview
Use a working ROOT font environment (e.g. LPC); missing local Symbol fonts can
make Greek letters disappear in previews. A successful export is not visual or
physics validation. Deployment, input permissions and run records follow the
owning project's instructions; this tool grants no permission to read data.
"""

import argparse
import math
from pathlib import Path
import re


def parser():
    """Keep --help usable without importing ROOT or opening any files."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Saved ROOT file, opened read-only")
    p.add_argument("--series", nargs=7, action="append", required=True,
                   metavar=("NUM", "DEN", "GROUP", "STAGE", "COLOR", "LINE", "MARKER"))
    p.add_argument("--output-base", required=True, type=Path, help="Unused basename, without extension; writes PNG+PDF")
    p.add_argument("--xlabel", required=True, help="ROOT TLatex axis title, including units when applicable")
    p.add_argument("--ylabel", required=True, help="Metric and counting unit, e.g. Event efficiency")
    p.add_argument("--status", required=True, help="Accurate CMS qualifier, e.g. Simulation Private Work")
    p.add_argument("--era", default="", help="Right-hand header; no campaign or energy is assumed")
    p.add_argument("--annotation", action="append", default=[], help="Repeat for context lines; maximum three")
    p.add_argument("--group-title", default="", help="Optional title of the color key, including units")
    p.add_argument("--stage-title", default="", help="Optional title of the line/marker key")
    p.add_argument("--xlim", nargs=2, type=float, metavar=("MIN", "MAX"), help="Display crop within saved axis coverage")
    p.add_argument("--ylim", nargs=2, type=float, default=(0.0, 1.05), metavar=("MIN", "MAX"))
    p.add_argument("--min-den", type=int, default=20, help="Draw bins with at least this many trials (default: 20)")
    p.add_argument("--confidence", type=float, default=0.6827, help="Clopper-Pearson confidence level (default: 0.6827)")
    p.add_argument("--log", action="store_true", help="Also write a log-y PNG; retain linear PNG+PDF")
    return p


def read_counts(root_file, key):
    """Validate visible AND flow counts before computing any binomial interval."""
    h = root_file.Get(key)
    if not h or not h.InheritsFrom("TH1") or h.GetDimension() != 1 or h.InheritsFrom("TProfile"):
        raise ValueError("Expected a one-dimensional count TH1: " + key)
    edges = [h.GetXaxis().GetBinLowEdge(i) for i in range(1, h.GetNbinsX() + 2)]
    if not all(math.isfinite(x) for x in edges) or any(a >= b for a, b in zip(edges, edges[1:])):
        raise ValueError("Invalid histogram bin edges: " + key)
    counts = []
    for i in range(h.GetNbinsX() + 2):
        value = h.GetBinContent(i)
        if not math.isfinite(value) or value < 0 or not math.isclose(value, round(value), abs_tol=1e-8, rel_tol=0):
            raise ValueError("Non-integer/negative/nonfinite count in %s bin %d" % (key, i))
        if h.GetSumw2N() and not math.isclose(h.GetBinError(i)**2, value, rel_tol=1e-7, abs_tol=1e-8):
            raise ValueError("Non-Poisson Sumw2 in %s bin %d; weighted/normalized inputs are unsupported" % (key, i))
        counts.append(int(round(value)))
    return edges, counts


def build_graph(ROOT, root_file, series, min_den, confidence):
    """Pure read of stored counts; match the study's ratio, interval and bin-width bars."""
    num_key, den_key, group, stage, color, line, marker = series
    edges, num = read_counts(root_file, num_key)
    den_edges, den = read_counts(root_file, den_key)
    if edges != den_edges or any(k > n for k, n in zip(num, den)):
        raise ValueError("Numerator/denominator binning mismatch or numerator exceeds denominator: " + num_key)
    graph = ROOT.TGraphAsymmErrors()
    for i, (k, n) in enumerate(zip(num[1:-1], den[1:-1])):
        if n == 0 or n < min_den:
            continue
        y = k / n
        lo = ROOT.TEfficiency.ClopperPearson(n, k, confidence, False)
        hi = ROOT.TEfficiency.ClopperPearson(n, k, confidence, True)
        j = graph.GetN()
        graph.SetPoint(j, (edges[i] + edges[i + 1]) / 2, y)
        half_width = (edges[i + 1] - edges[i]) / 2
        graph.SetPointError(j, half_width, half_width, y - lo, hi - y)
    graph.SetLineColor(ROOT.TColor.GetColor(color)); graph.SetMarkerColor(graph.GetLineColor())
    graph.SetLineStyle(int(line)); graph.SetMarkerStyle(int(marker))
    graph.SetLineWidth(2); graph.SetMarkerSize(1.0)
    return graph, edges


def render(args):
    """Render one explicit overlay; keep ROOT objects alive until all exports finish."""
    import ROOT
    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0); ROOT.gStyle.SetOptTitle(0)
    ROOT.gStyle.SetTextFont(42); ROOT.gStyle.SetLabelFont(42, "XYZ"); ROOT.gStyle.SetTitleFont(42, "XYZ")
    ROOT.gStyle.SetPadTickX(1); ROOT.gStyle.SetPadTickY(1)
    ROOT.TGaxis.SetMaxDigits(6)
    outputs = [Path(str(args.output_base) + ext) for ext in (".png", ".pdf")]
    if args.log:
        outputs.append(Path(str(args.output_base) + "_log.png"))
    if any(p.exists() for p in outputs):
        raise ValueError("Output exists; choose a new --output-base to preserve reviewed artifacts")

    groups, stages, combinations = {}, {}, set()
    for num, den, group, stage, color, line, marker in args.series:
        if not group or not stage or not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
            raise ValueError("Each series needs nonempty group/stage labels and a #RRGGBB color")
        if int(line) < 1 or int(marker) < 1:
            raise ValueError("Line and marker style integers must be positive")
        if group in groups and groups[group] != color.lower():
            raise ValueError("Inconsistent color for group " + group)
        if stage in stages and stages[stage] != (int(line), int(marker)):
            raise ValueError("Inconsistent line/marker style for stage " + stage)
        if group not in groups and color.lower() in groups.values():
            raise ValueError("Distinct groups need distinct colors")
        if stage not in stages and (int(line), int(marker)) in stages.values():
            raise ValueError("Distinct stages need distinct line/marker styles")
        if (group, stage) in combinations:
            raise ValueError("Each group/stage combination must identify exactly one series")
        combinations.add((group, stage))
        groups[group] = color.lower(); stages[stage] = (int(line), int(marker))

    # The legend band grows with the explicit keys. Refuse a crowded layout
    # rather than shrinking text until it becomes unreadable on a talk slide.
    meta_rows = max(len(args.annotation), len(groups) + bool(args.group_title), 1)
    meta_bottom = 0.915 - 0.04 * meta_rows
    stage_top = meta_bottom - 0.025
    stage_bottom = stage_top - 0.04 * (math.ceil(len(stages) / 2) + bool(args.stage_title))
    frame_top = stage_bottom - 0.025
    if frame_top < 0.55:
        raise ValueError("Too many legend entries for this talk template; split the overlay into focused plots")

    root_file = ROOT.TFile.Open(args.input, "READ")
    if not root_file or root_file.IsZombie():
        raise ValueError("Cannot open ROOT input: " + args.input)
    try:
        graphs, edges = [], None
        for series in args.series:
            graph, axis = build_graph(ROOT, root_file, series, args.min_den, args.confidence)
            if edges is not None and axis != edges:
                raise ValueError("All overlay series must have identical bin edges; no automatic rebinning")
            graphs.append(graph); edges = axis
        xmin, xmax = args.xlim or (edges[0], edges[-1])
        if xmin < edges[0] or xmax > edges[-1]:
            raise ValueError("--xlim exceeds saved histogram coverage")
        if not any(xmin <= g.GetPointX(i) <= xmax for g in graphs for i in range(g.GetN())):
            raise ValueError("No bins pass the denominator mask in the displayed x range")

        canvas = ROOT.TCanvas("efficiency", "", 900, 800)
        canvas.SetCanvasSize(900, 800)
        canvas.SetLeftMargin(0.15); canvas.SetRightMargin(0.04)
        canvas.SetBottomMargin(0.13); canvas.SetTopMargin(1 - frame_top)
        frame = canvas.DrawFrame(xmin, args.ylim[0], xmax, args.ylim[1])
        frame.GetXaxis().SetTitle(args.xlabel); frame.GetYaxis().SetTitle(args.ylabel)
        for axis in (frame.GetXaxis(), frame.GetYaxis()):
            axis.SetTitleSize(0.05); axis.SetLabelSize(0.042)
        frame.GetXaxis().SetTitleOffset(1.1); frame.GetYaxis().SetTitleOffset(1.4)
        canvas.Update()  # settle the frame before placing NDC text; prevents clipped headers
        for graph in graphs:
            graph.Draw("P same")
        text = ROOT.TLatex(); text.SetNDC(); text.SetTextAlign(11)
        text.SetTextFont(61); text.SetTextSize(0.04); text.DrawLatex(0.15, 0.95, "CMS")
        text.SetTextFont(52); text.SetTextSize(0.031); text.DrawLatex(0.255, 0.95, args.status)
        text.SetTextFont(42); text.SetTextAlign(31); text.DrawLatex(0.96, 0.95, args.era)
        text.SetTextSize(0.028); text.SetTextAlign(13)
        for i, line in enumerate(args.annotation):
            text.DrawLatex(0.19, 0.90 - 0.04 * i, line)

        group_key = ROOT.TLegend(0.60, meta_bottom, 0.94, 0.915)
        stage_key = ROOT.TLegend(0.19, stage_bottom, 0.94, stage_top)
        stage_key.SetNColumns(2)
        proxies = []
        for key, title in ((group_key, args.group_title), (stage_key, args.stage_title)):
            key.SetBorderSize(0); key.SetFillStyle(0); key.SetTextSize(0.028)
            if title:
                key.SetHeader(title)
        for group, color in groups.items():
            proxy = ROOT.TGraph(); proxies.append(proxy)
            proxy.SetLineColor(ROOT.TColor.GetColor(color)); proxy.SetLineWidth(2)
            group_key.AddEntry(proxy, group, "l")
        for stage, (line, marker) in stages.items():
            proxy = ROOT.TGraph(); proxies.append(proxy)
            proxy.SetLineColor(ROOT.kBlack); proxy.SetMarkerColor(ROOT.kBlack)
            proxy.SetLineStyle(line); proxy.SetMarkerStyle(marker); proxy.SetLineWidth(2)
            stage_key.AddEntry(proxy, stage, "lp")
        group_key.Draw(); stage_key.Draw()
        args.output_base.parent.mkdir(parents=True, exist_ok=True)
        canvas.Modified(); canvas.SaveAs(str(outputs[0])); canvas.SaveAs(str(outputs[1]))
        if args.log:
            frame.SetMinimum(max(1e-3, args.ylim[0])); canvas.SetLogy(1); canvas.Modified()
            canvas.SaveAs(str(outputs[2]))
        # ROOT can log a SaveAs error without raising a Python exception.
        if any(not output.is_file() or output.stat().st_size == 0 for output in outputs):
            raise OSError("ROOT export failed; inspect diagnostics and any partial output files")
        for output in outputs:
            print("Wrote", output)
    finally:
        root_file.Close()


def main():
    p = parser()
    args = p.parse_args()
    for name in ("xlim", "ylim"):
        limits = getattr(args, name)
        if limits and (not all(math.isfinite(v) for v in limits) or limits[0] >= limits[1]):
            p.error("--%s needs finite MIN < MAX" % name)
    if args.min_den < 0 or not 0 < args.confidence < 1 or len(args.annotation) > 3:
        p.error("Require --min-den >= 0, 0 < --confidence < 1 and at most three annotations")
    if args.log and args.ylim[1] <= max(1e-3, args.ylim[0]):
        p.error("Log-y display needs an upper limit above its positive lower limit")
    try:
        render(args)
    except (ValueError, OSError) as error:
        p.error(str(error))


if __name__ == "__main__":
    main()
