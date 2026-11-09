# idm_plot_utils.py

import os
import re
import subprocess
from collections import defaultdict

import ROOT


def setup_root_style():
    ROOT.gROOT.SetBatch(True)
    ROOT.TH1.SetDefaultSumw2(True)
    ROOT.gStyle.SetOptStat(0)


# =============================================================================
# EOS / file helpers
# =============================================================================

def strip_xrootd_path(path):
    if path.startswith("root://"):
        m = re.match(r"^root://[^/]+/(.*)$", path)
        if m:
            return "/" + m.group(1).lstrip("/")
    return path


def xrootd_prefix_from_path(path):
    if path.startswith("root://"):
        m = re.match(r"^(root://[^/]+)", path)
        if m:
            return m.group(1)
    return "root://cmseos.fnal.gov"


def list_root_files(input_path):
    """
    Accepts:
      - one ROOT file
      - root://... EOS directory
      - /store/... EOS directory
      - local directory
    """
    if input_path.endswith(".root"):
        return [input_path]

    if input_path.startswith("root://") or input_path.startswith("/store/"):
        xrd_prefix = xrootd_prefix_from_path(input_path)
        eos_path = strip_xrootd_path(input_path)

        cmd = ["xrdfs", xrd_prefix, "ls", "-R", eos_path]
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)

        files = []
        for line in output.splitlines():
            line = line.strip()
            if line.endswith(".root"):
                if line.startswith("/store/"):
                    files.append(f"{xrd_prefix}/{line}")
                else:
                    files.append(line)

        return sorted(set(files))

    if os.path.isdir(input_path):
        files = []
        for root, _, filenames in os.walk(input_path):
            for fname in filenames:
                if fname.endswith(".root"):
                    files.append(os.path.join(root, fname))
        return sorted(files)

    raise RuntimeError(f"Cannot interpret input path: {input_path}")


# =============================================================================
# Sample parsing
# =============================================================================

def sample_name_from_path(path):
    basename = os.path.basename(path)

    m = re.search(r"(Mchi-[^_/]+_dMchi-[^_/]+_ctau-[^_/\.]+)", basename)
    if m:
        return m.group(1)

    for piece in path.split("/"):
        m = re.search(r"(Mchi-[^_/]+_dMchi-[^_/]+_ctau-[^_/\.]+)", piece)
        if m:
            return m.group(1)

    return "unknown_sample"


def parse_sample(sample):
    """
    Example:
      Mchi-10p5_dMchi-1p0_ctau-100

    Returns:
      ("Mchi-10p5_dMchi-1p0", "100")
    """
    m = re.match(r"(Mchi-[^_]+_dMchi-[^_]+)_ctau-([^_]+)", sample)
    if not m:
        return sample, "unknown"

    return m.group(1), m.group(2)


def sample_sort_key(sample):
    m = re.search(r"Mchi-([^_]+)_dMchi-([^_]+)_ctau-([^_]+)", sample)
    if not m:
        return (9999, 9999, 9999)

    def conv(x):
        return float(x.replace("p", "."))

    return conv(m.group(1)), conv(m.group(2)), conv(m.group(3))


def group_files_by_sample(files):
    sample_to_files = defaultdict(list)

    for f in files:
        sample = sample_name_from_path(f)
        sample_to_files[sample].append(f)

    samples = sorted(sample_to_files.keys(), key=sample_sort_key)

    return sample_to_files, samples


def load_samples(input_path, verbose=True):
    files = list_root_files(input_path)
    sample_to_files, samples = group_files_by_sample(files)

    if verbose:
        print(f"Found {len(files)} ROOT files")
        print(f"Found {len(samples)} grouped samples")
        for s in samples:
            print(f"{s:35s}  nfiles = {len(sample_to_files[s])}")

    return files, sample_to_files, samples


# =============================================================================
# ROOT helpers
# =============================================================================

def make_chain(files, tree_name="ntuples/outT"):
    ch = ROOT.TChain(tree_name)

    for f in files:
        ch.Add(f)

    return ch


# =============================================================================
# Label helpers
# =============================================================================

def pretty_sample_label(sample):
    """
    Example:
      Mchi-10p5_dMchi-1p0_ctau-100

    Returns ROOT TLatex-safe label:
      m_{#chi}=10.5, #Delta m_{#chi}=1.0, c#tau=100 mm
    """
    m = re.search(r"Mchi-([^_]+)_dMchi-([^_]+)_ctau-([^_]+)", sample)
    if not m:
        return sample

    mchi = m.group(1).replace("p", ".")
    dmchi = m.group(2).replace("p", ".")
    ctau = m.group(3).replace("p", ".")

    return f"m_{{#chi}}={mchi}, #Delta m_{{#chi}}={dmchi}, c#tau={ctau} mm"


def pretty_mass_label(sample):
    """
    Example:
      Mchi-10p5_dMchi-1p0_ctau-100

    Returns:
      m_{#chi}=10.5, #Delta m_{#chi}=1.0
    """
    m = re.search(r"Mchi-([^_]+)_dMchi-([^_]+)_ctau-([^_]+)", sample)
    if not m:
        return sample

    mchi = m.group(1).replace("p", ".")
    dmchi = m.group(2).replace("p", ".")

    return f"m_{{#chi}}={mchi}, #Delta m_{{#chi}}={dmchi}"


def pretty_ctau_label(sample):
    """
    Example:
      Mchi-10p5_dMchi-1p0_ctau-100

    Returns:
      c#tau=100 mm
    """
    m = re.search(r"_ctau-([^_]+)", sample)
    if not m:
        return sample

    ctau = m.group(1).replace("p", ".")

    return f"c#tau={ctau} mm"


# =============================================================================
# Histogram creation
# =============================================================================

DEFAULT_COLORS = [
    ROOT.kBlack,
    ROOT.kRed + 1,
    ROOT.kBlue + 1,
    ROOT.kGreen + 2,
    ROOT.kMagenta + 1,
    ROOT.kCyan + 2,
    ROOT.kOrange + 7,
    ROOT.kViolet + 1,
    ROOT.kTeal + 2,
    ROOT.kPink + 7,
]

DEFAULT_LINE_STYLES = [1, 2, 3, 4, 5]


def make_histograms_by_sample(
    sample_to_files,
    samples,
    branch,
    tree_name="ntuples/outT",
    nbins=15,
    xmin=0.0,
    xmax=1.5,
    cut="GenSigDimuon_isValid",
    normalize=True,
    colors=None,
    line_styles=None,
):
    """
    Returns:
      hists, summary

    hists:
      dict sample -> TH1F

    summary:
      list of dicts with entries/drawn/mean/rms/overflow
    """
    if colors is None:
        colors = DEFAULT_COLORS

    if line_styles is None:
        line_styles = DEFAULT_LINE_STYLES

    hists = {}
    summary = []

    for idx, sample in enumerate(samples):
        chain = make_chain(sample_to_files[sample], tree_name=tree_name)

        hist_name = "h_" + re.sub(r"[^a-zA-Z0-9_]", "_", f"{branch}_{sample}")

        h = ROOT.TH1F(
            hist_name,
            sample,
            nbins,
            xmin,
            xmax,
        )

        h.Sumw2()

        expr = f"{branch}>>{hist_name}"
        n_drawn = chain.Draw(expr, cut, "goff")

        color = colors[idx % len(colors)]
        style = line_styles[(idx // len(colors)) % len(line_styles)]

        h.SetStats(False)
        h.SetLineColor(color)
        h.SetMarkerColor(color)
        h.SetLineStyle(style)
        h.SetLineWidth(2)

        if normalize and h.Integral() > 0:
            h.Scale(1.0 / h.Integral())

        hists[sample] = h

        summary.append(
            {
                "sample": sample,
                "entries": int(chain.GetEntries()),
                "drawn": int(n_drawn),
                "mean": h.GetMean(),
                "rms": h.GetRMS(),
                "underflow": h.GetBinContent(0),
                "overflow": h.GetBinContent(h.GetNbinsX() + 1),
            }
        )

    return hists, summary


def print_summary(summary):
    for row in summary:
        print(
            f"{row['sample']:35s} "
            f"entries={row['entries']:8d} "
            f"drawn={row['drawn']:8d} "
            f"mean={row['mean']:.4f} "
            f"rms={row['rms']:.4f} "
            f"overflow={row['overflow']:.4f}"
        )


# =============================================================================
# Drawing
# =============================================================================

def draw_overlay(
    hist_items,
    title,
    outname,
    outdir="plots",
    label_func=pretty_sample_label,
    legend_title=None,
    normalize=True,
    xaxis_title="#DeltaR(#mu^{+}, #mu^{-})",
    yaxis_title=None,
    xmin=0.0,
    xmax=1.5,
    ymin=0.0,
    ymax=1.0,
    logy=False,
    legend_x1=0.68,
    legend_y1=0.12,
    legend_x2=0.99,
    legend_y2=0.90,
    legend_text_size=0.022,
    canvas_width=1200,
    canvas_height=850,
    right_margin=0.34,
    left_margin=0.12,
    bottom_margin=0.12,
    top_margin=0.08,
    save_pdf=True,
    save_png=True,
):
    os.makedirs(outdir, exist_ok=True)

    if yaxis_title is None:
        yaxis_title = "Fraction of events / 0.1" if normalize else "Events / 0.1"

    c = ROOT.TCanvas("c_" + re.sub(r"[^a-zA-Z0-9_]", "_", outname), "", canvas_width, canvas_height)

    c.SetRightMargin(right_margin)
    c.SetLeftMargin(left_margin)
    c.SetBottomMargin(bottom_margin)
    c.SetTopMargin(top_margin)

    if logy:
        c.SetLogy()

    leg = ROOT.TLegend(legend_x1, legend_y1, legend_x2, legend_y2)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(legend_text_size)

    if legend_title is not None:
        leg.SetHeader(legend_title, "C")

    first = True

    for label, h in hist_items:
        h.SetStats(False)
        h.SetTitle(title)

        h.GetXaxis().SetTitle(xaxis_title)
        h.GetYaxis().SetTitle(yaxis_title)

        h.GetXaxis().SetRangeUser(xmin, xmax)

        h.GetXaxis().SetTitleSize(0.045)
        h.GetYaxis().SetTitleSize(0.045)
        h.GetXaxis().SetLabelSize(0.040)
        h.GetYaxis().SetLabelSize(0.040)

        h.GetYaxis().SetTitleOffset(1.25)

        if logy:
            h.SetMinimum(max(ymin, 1e-8))
            h.SetMaximum(ymax)
        else:
            h.SetMinimum(ymin)
            h.SetMaximum(ymax)

        h.Draw("hist" if first else "hist same")
        leg.AddEntry(h, label_func(label), "l")

        first = False

    leg.Draw()

    if save_png:
        c.SaveAs(os.path.join(outdir, outname + ".png"))

    if save_pdf:
        c.SaveAs(os.path.join(outdir, outname + ".pdf"))

    return c


def make_fixed_ctau_canvases(
    samples,
    hists,
    outdir,
    normalize=True,
    logy=False,
):
    ctau_to_samples = defaultdict(list)

    for sample in samples:
        _, ctau = parse_sample(sample)
        ctau_to_samples[ctau].append(sample)

    canvases = {}

    for ctau in sorted(ctau_to_samples.keys(), key=lambda x: float(x.replace("p", "."))):
        items = [(s, hists[s]) for s in sorted(ctau_to_samples[ctau], key=sample_sort_key)]

        safe_ctau = re.sub(r"[^a-zA-Z0-9_]", "_", ctau)

        canvases[ctau] = draw_overlay(
            items,
            f"genSigDimuon #DeltaR, c#tau = {ctau} mm",
            f"genSigDimuon_dr_by_ctau_{safe_ctau}",
            outdir=outdir,
            label_func=pretty_mass_label,
            legend_title="Mass points",
            normalize=normalize,
            logy=logy,
            legend_x1=0.70,
            legend_y1=0.58,
            legend_x2=0.99,
            legend_y2=0.88,
            legend_text_size=0.027,
        )

    return canvases


def make_fixed_mass_canvases(
    samples,
    hists,
    outdir,
    normalize=True,
    logy=False,
):
    mass_to_samples = defaultdict(list)

    for sample in samples:
        mass_key, _ = parse_sample(sample)
        mass_to_samples[mass_key].append(sample)

    canvases = {}

    for mass_key in sorted(mass_to_samples.keys()):
        items = [(s, hists[s]) for s in sorted(mass_to_samples[mass_key], key=sample_sort_key)]

        safe_mass = re.sub(r"[^a-zA-Z0-9_]", "_", mass_key)

        canvases[mass_key] = draw_overlay(
            items,
            f"genSigDimuon #DeltaR, {mass_key}",
            f"genSigDimuon_dr_by_mass_{safe_mass}",
            outdir=outdir,
            label_func=pretty_ctau_label,
            legend_title="Lifetimes",
            normalize=normalize,
            logy=logy,
            legend_x1=0.72,
            legend_y1=0.60,
            legend_x2=0.99,
            legend_y2=0.88,
            legend_text_size=0.030,
        )

    return canvases
