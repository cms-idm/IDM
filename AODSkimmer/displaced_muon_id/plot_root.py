#!/usr/bin/env python3
"""Render ROOT distributions, efficiency ratios and saved graphs in the CMS talk style.

This is the portable rendering tool for a plotting skill. It depends on
Python >=3.9, NumPy and PyROOT (run after cmsenv); it does not import the muon study,
read event trees, select physics objects, normalize, rebin, or modify ROOT files.
The study-specific analysis remains in mumu_talk_plots.py.

UHI interoperability (PlottableHistogram protocol 1.2):
https://uhi.readthedocs.io/en/latest/plotting.html
The histogram consumer uses only kind, axes, values() and variances(). It accepts
one continuous, non-categorical COUNT axis. Profiles/MEAN, discrete axes and
multiple dimensions are rejected explicitly. Flow bins are outside UHI's base
plotting protocol; the ROOT-file adapter checks them but plots only normal bins.
UHI counts() may be effective counts for weighted storage: NEVER substitute it
for bin contents or binomial trials. Variances=None means no error bars, not zero
uncertainty. Histograms/axes can be sliced with their provider's UHI indexing
before passing them in; this plotter is a consumer, not an indexing implementation.

Python API (no temporary ROOT file required, UHI need not be installed at runtime):
    from plot_root import plot_histograms
    plot_histograms(
        {"signal": h},  # h: hist.Hist, boost_histogram.Histogram, uproot TH1, ...
        [("signal", "Signal", "Selected", "#0072B2", 1, 20)],
        output_base="/path/to/new/distribution", xlabel="p_{T} [GeV]",
        ylabel="Weighted events / bin", status="Simulation Private Work")
For ratios, use ratio=True and seven-field records (NUM, DEN, GROUP, STAGE,
COLOR, LINE, MARKER), where NUM/DEN address the supplied histogram mapping.
Keyword options match CLI destinations, e.g. xlim=(0, 100), errors="binomial",
annotation=["..."]. No normalization, conversion of kind, or rebinning is done.
from_root_histogram(th1) also exposes a ROOT TH1 as a UHI-compatible object.

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
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from typing import Mapping, Sequence
    from uhi.typing.plottable import PlottableHistogram


class _Traits(NamedTuple):
    circular: bool = False
    discrete: bool = False


class _ContinuousAxis(tuple):
    """UHI axes are sequences of bin intervals, not backend-specific edge arrays."""
    traits = _Traits()


class RootHistogram:
    """Read-only UHI view of a COUNT TH1; keeps unknown variances as None.

    A missing Sumw2 cannot establish unweighted fills, so counts() is unknown
    then. With Sumw2 it returns effective counts, as required by UHI, not sumw.
    The snapshots include flow for ROOT-side checks; public UHI methods do not.
    """
    kind = "COUNT"

    def __init__(self, histogram):
        import numpy as np
        if (not histogram or not histogram.InheritsFrom("TH1") or
                histogram.GetDimension() != 1 or histogram.InheritsFrom("TProfile")):
            raise ValueError("Expected a one-dimensional COUNT TH1")
        n = histogram.GetNbinsX()
        axis = histogram.GetXaxis()
        labels = axis.GetLabels()
        if labels and labels.GetSize():
            raise ValueError("Labeled/categorical ROOT axes are unsupported by the continuous-axis template")
        self.axes = (_ContinuousAxis((axis.GetBinLowEdge(i), axis.GetBinUpEdge(i)) for i in range(1, n + 1)),)
        self._flow_values = np.array([histogram.GetBinContent(i) for i in range(n + 2)], dtype=float)
        self._flow_variances = (np.array([histogram.GetSumw2().At(i) for i in range(n + 2)], dtype=float)
                                if histogram.GetSumw2N() else None)
        if not np.all(np.isfinite(self._flow_values)):
            raise ValueError("Nonfinite TH1 content, including flow bins")
        if self._flow_variances is not None and (not np.all(np.isfinite(self._flow_variances)) or
                                                  np.any(self._flow_variances < 0)):
            raise ValueError("Invalid TH1 variance, including flow bins")

    def values(self):
        return self._flow_values[1:-1].copy()

    def variances(self):
        return None if self._flow_variances is None else self._flow_variances[1:-1].copy()

    def counts(self):
        import numpy as np
        values, variances = self.values(), self.variances()
        if variances is None:
            return None
        return np.divide(values**2, variances, out=np.zeros_like(values), where=variances != 0)


def from_root_histogram(histogram) -> "PlottableHistogram":
    """Adapt TH1 without writing files; usable by other UHI-aware plotters too."""
    return RootHistogram(histogram)


def histogram_data(histogram: "PlottableHistogram", raw_counts=False):
    """Read UHI protocol members only; counts() is intentionally not a numerator.

    Only one continuous COUNT axis is supported. Variances are variances of the
    values; do not divide by effective counts (that would be a MEAN operation).
    """
    import numpy as np
    if histogram.kind != "COUNT":
        raise ValueError("Only UHI kind COUNT is supported; profiles/MEAN need separate error semantics")
    if len(histogram.axes) != 1 or histogram.axes[0].traits.discrete:
        raise ValueError("Expected one continuous UHI axis; discrete and multidimensional inputs are unsupported")
    bins = list(histogram.axes[0])
    if not bins or any(len(interval) != 2 for interval in bins):
        raise ValueError("UHI continuous axes must provide (lower, upper) intervals")
    edges = np.array([bins[0][0]] + [interval[1] for interval in bins], dtype=float)
    if (not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0) or
            any(not math.isclose(a[1], b[0], rel_tol=1e-12, abs_tol=1e-12) for a, b in zip(bins, bins[1:]))):
        raise ValueError("Expected finite, increasing, contiguous UHI bin intervals")
    values = np.asarray(histogram.values(), dtype=float)
    variance = histogram.variances()
    variance = None if variance is None else np.asarray(variance, dtype=float)
    if values.shape != (len(bins),) or not np.all(np.isfinite(values)):
        raise ValueError("Invalid UHI values: need one finite value per bin")
    if variance is not None and (variance.shape != values.shape or not np.all(np.isfinite(variance)) or np.any(variance < 0)):
        raise ValueError("Invalid UHI variances: need nonnegative finite variance per bin, or None")
    if raw_counts:
        _check_raw_counts(values, variance)
    errors = np.zeros_like(values) if variance is None else np.sqrt(variance)
    return edges.tolist(), values.tolist(), errors.tolist()


def _check_raw_counts(values, variances):
    """Explicit binomial opt-in only; effective counts never prove raw trials."""
    for i, value in enumerate(values):
        if value < 0 or not math.isclose(value, round(value), abs_tol=1e-8, rel_tol=0):
            raise ValueError("Binomial errors require raw nonnegative integer bin contents")
        if variances is not None and not math.isclose(variances[i], value, rel_tol=1e-7, abs_tol=1e-8):
            raise ValueError("Non-Poisson variances; cannot use --errors binomial")


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
    """ROOT I/O adapter; retain legacy flow checks, consume visible bins via UHI."""
    h = from_root_histogram(root_file.Get(key))
    edges, _, _ = histogram_data(h, raw_counts)
    values = h._flow_values.tolist()
    if raw_counts:
        _check_raw_counts(values, h._flow_variances)
        values = [int(round(value)) for value in values]
    errors = ([0.0] * len(values) if h._flow_variances is None else
              [math.sqrt(value) for value in h._flow_variances])
    return edges, values, errors


def build_ratio(ROOT, root_file, series, min_den, confidence, errors):
    """Compute central ratios; binomial errors require an explicit raw-count choice."""
    num_key, den_key = series[:2]
    binomial = errors == "binomial"
    edges, num, _ = read_hist(root_file, num_key, binomial)
    den_edges, den, _ = read_hist(root_file, den_key, binomial)
    if edges != den_edges or (binomial and any(k > n for k, n in zip(num, den))):
        raise ValueError("Numerator/denominator binning mismatch or numerator exceeds denominator: " + num_key)
    return _ratio_graph(ROOT, edges, num[1:-1], den[1:-1], min_den, confidence, errors)


def ratio_graph(ROOT, numerator: "PlottableHistogram", denominator: "PlottableHistogram",
                min_den=0, confidence=0.6827, errors="none"):
    """Public UHI ratio adapter, using values(), not effective counts()."""
    if errors not in ("none", "binomial"):
        raise ValueError("Ratios require errors='none' or explicit raw-count errors='binomial'")
    edges, num, _ = histogram_data(numerator, errors == "binomial")
    den_edges, den, _ = histogram_data(denominator, errors == "binomial")
    if edges != den_edges or (errors == "binomial" and any(k > n for k, n in zip(num, den))):
        raise ValueError("Ratio binning mismatch or raw numerator exceeds denominator")
    return _ratio_graph(ROOT, edges, num, den, min_den, confidence, errors)


def _ratio_graph(ROOT, edges, num, den, min_den, confidence, errors):
    if not math.isfinite(min_den) or min_den < 0 or not 0 < confidence < 1:
        raise ValueError("Require finite min_den >= 0 and 0 < confidence < 1")
    binomial = errors == "binomial"
    graph = ROOT.TGraphAsymmErrors()
    for i, (k, n) in enumerate(zip(num, den)):
        if n == 0 or (min_den > 0 and n < min_den):
            continue
        y = k / n
        if not math.isfinite(y):
            raise ValueError("Nonfinite ratio")
        lo = ROOT.TEfficiency.ClopperPearson(int(round(n)), int(round(k)), confidence, False) if binomial else y
        hi = ROOT.TEfficiency.ClopperPearson(int(round(n)), int(round(k)), confidence, True) if binomial else y
        j = graph.GetN()
        graph.SetPoint(j, (edges[i] + edges[i + 1]) / 2, y)
        half_width = (edges[i + 1] - edges[i]) / 2
        graph.SetPointError(j, half_width, half_width, y - lo, hi - y)
    return graph, edges


def build_distribution(ROOT, root_file, series, errors):
    """Represent TH1 bins as points without changing weighted/normalized contents."""
    return histogram_graph(ROOT, from_root_histogram(root_file.Get(series[0])), errors)


def histogram_graph(ROOT, histogram: "PlottableHistogram", errors="stored"):
    """Public UHI distribution adapter; variances=None produces no error bars."""
    if errors not in ("stored", "none"):
        raise ValueError("Distributions require errors='stored' or 'none'")
    edges, values, stored_errors = histogram_data(histogram)
    graph = ROOT.TGraphAsymmErrors()
    for i, (value, error) in enumerate(zip(values, stored_errors)):
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


def render(args, histograms=None):
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

    root_file = ROOT.TFile.Open(args.input, "READ") if histograms is None else None
    if histograms is None and (not root_file or root_file.IsZombie()):
        raise ValueError("Cannot open ROOT input: " + args.input)
    try:
        graphs, coverage = [], []
        for record in records:
            if histograms is not None and args.series:
                graph, extent = ratio_graph(ROOT, histograms[record[0]], histograms[record[1]],
                                            args.min_den, args.confidence, args.errors)
            elif histograms is not None:
                graph, extent = histogram_graph(ROOT, histograms[record[0]], args.errors)
            elif args.series:
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
        if root_file is not None:
            root_file.Close()


def _validate_args(args):
    """Share CLI/API validation and explicit uncertainty defaults."""
    for name in ("xlim", "ylim"):
        limits = getattr(args, name)
        if limits and (not all(math.isfinite(v) for v in limits) or limits[0] >= limits[1]):
            raise ValueError("--%s needs finite MIN < MAX" % name)
    if not math.isfinite(args.min_den) or args.min_den < 0 or not 0 < args.confidence < 1 or len(args.annotation) > 3:
        raise ValueError("Require --min-den >= 0, 0 < --confidence < 1 and at most three annotations")
    if args.errors is None:
        args.errors = "none" if args.series else "stored"
    if args.series and args.errors == "stored":
        raise ValueError("Ratio inputs have no saved ratio uncertainty; use --graph or choose --errors none/binomial")
    if not args.series and args.errors == "binomial":
        raise ValueError("--errors binomial requires explicit numerator/denominator --series")
    if not args.series and args.min_den:
        raise ValueError("--min-den applies only to numerator/denominator --series")


def plot_histograms(histograms: "Mapping[str, PlottableHistogram]", series: "Sequence[Sequence]", *,
                    output_base, xlabel, ylabel, status, ratio=False, **options):
    """Plot UHI objects directly with the same configuration and style as the CLI.

    series contains six-field distribution or seven-field ratio records (see
    module examples). A provider's own indexing/slicing may be applied before
    this call; no backend-specific access is made to those histogram objects.
    Files are never opened by this API. Missing mapping keys raise KeyError.
    """
    allowed = {"era", "annotation", "group_title", "stage_title", "xlim", "ylim", "errors", "min_den",
               "confidence", "log", "connect"}
    if set(options) - allowed:
        raise ValueError("Unknown plotting options: " + ", ".join(sorted(set(options) - allowed)))
    records = list(series)
    if not records or any(len(record) != (7 if ratio else 6) for record in records):
        raise ValueError("Need nonempty seven-field ratio or six-field distribution records")
    p = parser()
    settings = {name: p.get_default(name) for name in allowed}
    settings.update(options)
    if settings["errors"] not in (None, "stored", "none", "binomial"):
        raise ValueError("Unknown uncertainty policy")
    args = argparse.Namespace(**settings, input=None, output_base=Path(output_base), xlabel=xlabel,
                              ylabel=ylabel, status=status, hist=None if ratio else records,
                              series=records if ratio else None, graph=None)
    _validate_args(args)
    render(args, histograms=histograms)


def main():
    p = parser()
    args = p.parse_args()
    try:
        _validate_args(args)
        render(args)
    except (ValueError, OSError) as error:
        p.error(str(error))


if __name__ == "__main__":
    main()
