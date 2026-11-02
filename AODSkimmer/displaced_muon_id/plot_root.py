#!/usr/bin/env python3
"""Render ROOT distributions, efficiency ratios and saved graphs in the CMS talk style.

This is the portable rendering tool for a plotting skill. It depends only on
Python >=3.9 and PyROOT (run after cmsenv); it does not import the muon study,
read event trees, select physics objects, normalize, rebin, or modify ROOT files.
The study-specific analysis remains in mumu_talk_plots.py.

Choose one input form (repeat that option to overlay curves):
  --hist KEY GROUP_LABEL STAGE_LABEL COLOR LINE MARKER
      TH1 distribution: preserve its contents/normalization and stored Sumw2.
      With no stored Sumw2, draw without error bars; do not invent uncertainties.
  --graph KEY GROUP_LABEL STAGE_LABEL COLOR LINE MARKER
      TGraph, TGraphErrors or TGraphAsymmErrors: preserve points and saved errors.
  --series NUMERATOR_KEY DENOMINATOR_KEY GROUP_LABEL STAGE_LABEL COLOR LINE MARKER
      Ratio of TH1 contents. Default: no error bars. --errors binomial explicitly
      requests raw unweighted counts and central Clopper-Pearson intervals.

Each --series supplies seven shell-quoted fields:
    NUMERATOR_KEY DENOMINATOR_KEY GROUP_LABEL STAGE_LABEL COLOR LINE MARKER
Keys are full paths inside --input. Labels use ROOT TLatex syntax, colors use
#RRGGBB, and LINE/MARKER are ROOT style integers. Color identifies a group;
line/marker style identifies a stage. Repeated group/stage labels must have
consistent styles. Use short labels in the keys; place context in --annotation.
Without --connect, distinct stages need distinct markers so line-only style
differences cannot disappear on graphs without error bars.

Example (substitute actual histogram keys and an UNUSED output basename):
    python3 -I plot_root.py --input saved.root \\
      --series group/num group/den 'Signal' 'Matched' '#0072B2' 2 24 \\
      --series group/valid group/den 'Signal' 'Matched and vertexed' '#0072B2' 1 20 \\
      --xlabel 'Gen L_{xy} [cm]' --ylabel 'Event efficiency' \\
      --status 'Simulation Private Work' --era '2022EE (13.6 TeV)' \\
      --annotation 'Both signal muons accepted' --xlim 0 100 --ylim 0 0.8 \\
      --errors binomial --min-den 20 --output-base /path/to/new/efficiency

Distribution example (weighted, signed or already normalized contents accepted):
    python3 -I plot_root.py --input saved.root \\
      --hist sample/pt 'Signal' 'Selected' '#0072B2' 1 20 \\
      --xlabel 'Muon p_{T} [GeV]' --ylabel 'Weighted events / bin' \\
      --status 'Simulation Private Work' --output-base /path/to/new/pt
No normalization is performed. Use the y title to state the stored convention
(counts, weighted yield, unit area, density, etc.), and annotations for its scope.

Input contract:
* TH1 pairs must have identical bin edges within each pair. Weighted/normalized
  ratios use --errors none (the default); saved weighted-efficiency uncertainties
  can instead be supplied through --graph. The tool never assumes independent
  numerator/denominator errors: their covariance is generally needed.
* --errors binomial requires finite integer counts, 0 <= num <= den, and Poisson
  Sumw2 when present. Checks cannot prove event-level subset membership or lack
  of weighting; the caller must establish both. Confidence defaults to 68.27%.
* --min-den is an optional display mask for ratio inputs, in the units of the
  stored denominator (default 0). Zero-denominator bins are always omitted.
  TH1 flow bins are checked but not drawn; they remain in the untouched input.
* --xlim crops the display, including error bars at its boundary; it does not
  rebin or recount. Labels must describe the actual population, matching,
  exclusions and weighting; the tool cannot infer these from histogram names.

Outputs are PNG + vector PDF; existing outputs are refused. --log adds a log-y
PNG alongside linear exports (nonpositive points are omitted from log-y only).
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
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--series", nargs=7, action="append",
                   metavar=("NUM", "DEN", "GROUP", "STAGE", "COLOR", "LINE", "MARKER"))
    for option in ("hist", "graph"):
        inputs.add_argument("--" + option, nargs=6, action="append",
                            metavar=("KEY", "GROUP", "STAGE", "COLOR", "LINE", "MARKER"))
    p.add_argument("--output-base", required=True, type=Path, help="Unused basename, without extension; writes PNG+PDF")
    p.add_argument("--xlabel", required=True, help="ROOT TLatex axis title, including units when applicable")
    p.add_argument("--ylabel", required=True, help="Metric and counting unit, e.g. Event efficiency")
    p.add_argument("--status", required=True, help="Accurate CMS qualifier, e.g. Simulation Private Work")
    p.add_argument("--era", default="", help="Right-hand header; no campaign or energy is assumed")
    p.add_argument("--annotation", action="append", default=[], help="Repeat for context lines; maximum three")
    p.add_argument("--group-title", default="", help="Optional title of the color key, including units")
    p.add_argument("--stage-title", default="", help="Optional title of the line/marker key")
    p.add_argument("--xlim", nargs=2, type=float, metavar=("MIN", "MAX"), help="Display range; no rebinning or recounting")
    p.add_argument("--ylim", nargs=2, type=float, metavar=("MIN", "MAX"),
                   help="Default: 0..1.05 for ratios; fit visible points/errors for distributions/graphs")
    p.add_argument("--errors", choices=("stored", "none", "binomial"),
                   help="Default: stored for hist/graph, none for ratios; binomial is explicit raw-count opt-in")
    p.add_argument("--min-den", type=float, default=0, help="Ratio-only display threshold on denominator contents (default: 0)")
    p.add_argument("--confidence", type=float, default=0.6827, help="Clopper-Pearson confidence level (default: 0.6827)")
    p.add_argument("--log", action="store_true", help="Also write a log-y PNG; retain linear PNG+PDF")
    p.add_argument("--connect", action="store_true", help="Connect points with styled lines; default is points/error bars")
    return p


def read_hist(root_file, key, raw_counts=False):
    """Read finite TH1 contents and stored errors; raw-count checks are opt-in."""
    h = root_file.Get(key)
    if not h or not h.InheritsFrom("TH1") or h.GetDimension() != 1 or h.InheritsFrom("TProfile"):
        raise ValueError("Expected a one-dimensional TH1: " + key)
    edges = [h.GetXaxis().GetBinLowEdge(i) for i in range(1, h.GetNbinsX() + 2)]
    if not all(math.isfinite(x) for x in edges) or any(a >= b for a, b in zip(edges, edges[1:])):
        raise ValueError("Invalid histogram bin edges: " + key)
    values, errors = [], []
    for i in range(h.GetNbinsX() + 2):
        value = h.GetBinContent(i)
        if not math.isfinite(value):
            raise ValueError("Nonfinite content in %s bin %d" % (key, i))
        if raw_counts and (value < 0 or not math.isclose(value, round(value), abs_tol=1e-8, rel_tol=0)):
            raise ValueError("Non-integer/negative/nonfinite count in %s bin %d" % (key, i))
        error = h.GetBinError(i) if h.GetSumw2N() else 0.0
        if not math.isfinite(error) or error < 0:
            raise ValueError("Invalid stored uncertainty in %s bin %d" % (key, i))
        if raw_counts and h.GetSumw2N() and not math.isclose(error**2, value, rel_tol=1e-7, abs_tol=1e-8):
            raise ValueError("Non-Poisson Sumw2 in %s bin %d; cannot use --errors binomial" % (key, i))
        values.append(int(round(value)) if raw_counts else value)
        errors.append(error)
    if not raw_counts and not h.GetSumw2N():
        print("No stored Sumw2 in %s: no histogram error bars supplied" % key)
    return edges, values, errors


def build_ratio(ROOT, root_file, series, min_den, confidence, errors):
    """Compute central ratios; binomial errors require an explicit raw-count choice."""
    num_key, den_key = series[:2]
    binomial = errors == "binomial"
    edges, num, _ = read_hist(root_file, num_key, binomial)
    den_edges, den, _ = read_hist(root_file, den_key, binomial)
    if edges != den_edges or (binomial and any(k > n for k, n in zip(num, den))):
        raise ValueError("Numerator/denominator binning mismatch or numerator exceeds denominator: " + num_key)
    graph = ROOT.TGraphAsymmErrors()
    for i, (k, n) in enumerate(zip(num[1:-1], den[1:-1])):
        if n == 0 or (min_den > 0 and n < min_den):
            continue
        y = k / n
        if not math.isfinite(y):
            raise ValueError("Nonfinite ratio in " + num_key)
        lo = ROOT.TEfficiency.ClopperPearson(n, k, confidence, False) if binomial else y
        hi = ROOT.TEfficiency.ClopperPearson(n, k, confidence, True) if binomial else y
        j = graph.GetN()
        graph.SetPoint(j, (edges[i] + edges[i + 1]) / 2, y)
        half_width = (edges[i + 1] - edges[i]) / 2
        graph.SetPointError(j, half_width, half_width, y - lo, hi - y)
    return graph, edges


def build_distribution(ROOT, root_file, series, errors):
    """Represent TH1 bins as points without changing weighted/normalized contents."""
    edges, values, stored_errors = read_hist(root_file, series[0])
    graph = ROOT.TGraphAsymmErrors()
    for i, (value, error) in enumerate(zip(values[1:-1], stored_errors[1:-1])):
        graph.SetPoint(i, (edges[i] + edges[i + 1]) / 2, value)
        half_width = (edges[i + 1] - edges[i]) / 2
        error = error if errors == "stored" else 0.0
        graph.SetPointError(i, half_width, half_width, error, error)
    return graph, edges


def read_graph(ROOT, root_file, series, errors):
    """Preserve saved graph points and asymmetric uncertainties (e.g. weighted efficiencies)."""
    source = root_file.Get(series[0])
    if not source or not source.InheritsFrom("TGraph"):
        raise ValueError("Expected TGraph/TGraphErrors/TGraphAsymmErrors: " + series[0])
    graph = ROOT.TGraphAsymmErrors()
    asym, sym = source.InheritsFrom("TGraphAsymmErrors"), source.InheritsFrom("TGraphErrors")
    extent = []
    for i in range(source.GetN()):
        x, y = source.GetPointX(i), source.GetPointY(i)
        if errors == "stored" and asym:
            exl, exh = source.GetErrorXlow(i), source.GetErrorXhigh(i)
            eyl, eyh = source.GetErrorYlow(i), source.GetErrorYhigh(i)
        elif errors == "stored" and sym:
            exl = exh = source.GetErrorX(i); eyl = eyh = source.GetErrorY(i)
        else:
            exl = exh = eyl = eyh = 0.0
        if not all(math.isfinite(v) for v in (x, y, exl, exh, eyl, eyh)) or min(exl, exh, eyl, eyh) < 0:
            raise ValueError("Nonfinite point or invalid graph uncertainty: " + series[0])
        graph.SetPoint(i, x, y); graph.SetPointError(i, exl, exh, eyl, eyh)
        extent.extend((x - exl, x + exh))
    return graph, extent


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

    records = args.series or args.hist or args.graph
    groups, stages, combinations = {}, {}, set()
    for record in records:
        group, stage, color, line, marker = record[-5:]
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
        if not args.connect and stage not in stages and any(int(marker) == value[1] for value in stages.values()):
            raise ValueError("Distinct stages need distinct markers, or --connect to display their line styles")
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
        graphs, coverage = [], []
        for record in records:
            if args.series:
                graph, extent = build_ratio(ROOT, root_file, record, args.min_den, args.confidence, args.errors)
            elif args.hist:
                graph, extent = build_distribution(ROOT, root_file, record, args.errors)
            else:
                graph, extent = read_graph(ROOT, root_file, record, args.errors)
            group, stage, color, line, marker = record[-5:]
            graph.SetLineColor(ROOT.TColor.GetColor(color)); graph.SetMarkerColor(graph.GetLineColor())
            graph.SetLineStyle(int(line)); graph.SetMarkerStyle(int(marker))
            graph.SetLineWidth(2); graph.SetMarkerSize(1.0)
            graphs.append(graph); coverage.extend(extent)
        if not coverage:
            raise ValueError("No input points to plot")
        xmin, xmax = args.xlim or (min(coverage), max(coverage))
        if xmin == xmax:
            xmin -= 0.5; xmax += 0.5
        if not any(xmin <= g.GetPointX(i) <= xmax for g in graphs for i in range(g.GetN())):
            raise ValueError("No points in the displayed x range (check the ratio denominator mask)")
        if args.ylim is None:
            if args.series:
                # Keep standard efficiencies readable, but do not silently clip signed/weighted ratios.
                values = [g.GetPointY(i) for g in graphs for i in range(g.GetN()) if xmin <= g.GetPointX(i) <= xmax]
                if min(values) >= 0 and max(values) <= 1:
                    args.ylim = (0.0, 1.05)
            if args.ylim is None:
                lows, highs = [], []
                for g in graphs:
                    for i in range(g.GetN()):
                        if xmin <= g.GetPointX(i) <= xmax:
                            lows.append(g.GetPointY(i) - g.GetErrorYlow(i))
                            highs.append(g.GetPointY(i) + g.GetErrorYhigh(i))
                lo, hi = min(0.0, min(lows)), max(0.0, max(highs))
                pad = 0.10 * (hi - lo or 1.0)
                args.ylim = (lo - pad if lo < 0 else 0.0, hi + pad)
        if args.log and args.ylim[1] <= max(1e-3, args.ylim[0]):
            raise ValueError("Log-y display needs a positive upper limit above its lower limit")
        print("Input form: %s; error policy: %s; contents and normalization preserved" % (
            "ratio" if args.series else "TH1" if args.hist else "graph", args.errors))

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
            graph.Draw("LP same" if args.connect else "P same")
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
            # ROOT can pin zero-valued symbols/error bars to the log floor.
            # Filter copies for the extra log export; preserve the linear graphs.
            log_graphs = []
            for graph in graphs:
                canvas.GetListOfPrimitives().Remove(graph)
                positive = graph.Clone(); log_graphs.append(positive)
                for i in range(positive.GetN() - 1, -1, -1):
                    if positive.GetPointY(i) <= 0:
                        positive.RemovePoint(i)
                positive.Draw("LP same" if args.connect else "P same")
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
    if not math.isfinite(args.min_den) or args.min_den < 0 or not 0 < args.confidence < 1 or len(args.annotation) > 3:
        p.error("Require --min-den >= 0, 0 < --confidence < 1 and at most three annotations")
    if args.errors is None:
        args.errors = "none" if args.series else "stored"
    if args.series and args.errors == "stored":
        p.error("Ratio inputs have no saved ratio uncertainty; use --graph or choose --errors none/binomial")
    if not args.series and args.errors == "binomial":
        p.error("--errors binomial requires explicit numerator/denominator --series")
    if not args.series and args.min_den:
        p.error("--min-den applies only to numerator/denominator --series")
    try:
        render(args)
    except (ValueError, OSError) as error:
        p.error(str(error))


if __name__ == "__main__":
    main()
