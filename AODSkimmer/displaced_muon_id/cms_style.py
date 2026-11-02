#!/usr/bin/env python3
"""CMS-style PyROOT plotting helpers for any saved 1D/2D histogram: distributions, efficiencies/fractions, 2D maps.
The defaults are the user's choices of 2026-11-02 (mumu_talk_plots.py plot-cms, plot-cms-pt, plot-2d-all); every
year/campaign/label-specific piece is a parameter.

Style only: no event loops, selections or rebinning; nothing is written back to the inputs. Inputs: anything with UHI
axes/values()/variances() (hist, boost-histogram, uproot TH1) or a ROOT TH1; TH1 variances are GetBinError()**2
(ROOT's convention). Run with PyROOT (cmsenv on LPC).

Defaults (change per plot, or globally with setup(era=..., status=...)):
* Header: "CMS" bold + STYLE["status"] italic at left, STYLE["era"] at right; context lines are short and top left.
* Colors: Okabe-Ito (COLORS, or BLUE, ORANGE, ...). Two-stage overlays: OPEN_DASHED (open 24, dashed) for the looser
  stage, FILLED_SOLID (filled 20, solid) for the tighter one (MATCHED / VERTEXED are the same styles).
* Efficiencies/fractions (eff_graph): raw counts, central Clopper-Pearson 68.3% bars, bins with den < 20 not drawn.
* Distributions (hist_graph): points with stored errors, or step=True for an outline; norm=True draws a copy whose
  bin contents sum to 1 (over all normal bins, flow excluded, not divided by bin width: "Fraction of events / bin");
  the stored histogram is never changed.
* 1D layouts: "frame" = legend inside the frame (top or bottom right), y max fitted so there is no empty space above
  the data; "band" = color key (groups) and style key (stages) above the frame, for group x stage overlays.
* Linear PNG + PDF; log=True adds a _log.png copy (points <= 0 left out), never replacing the linear plot.
* 2D (map2d): COLZ, viridis, optional red outer border around flagged (e.g. low-count) bins.

Examples:
    import cms_style as cs
    cs.setup(era="2022EE (13.6 TeV)", status="Simulation Private Work")
    # efficiency, frame layout
    s = [cs.eff_graph(f["matched"], f["den"], cs.BLUE, 1, 21), cs.eff_graph(f["valid"], f["den"], cs.ORANGE, 1, 20)]
    cs.plot("out/eff_lxy", s, ["Both muons matched", "Matched and vertexed"], "Gen L_{xy} [cm]", "Event efficiency",
            lines=["Both signal muons accepted"])
    # distributions, unnormalized and unit-area copies
    for norm, yt in ((False, "Muons / 1 GeV"), (True, "Fraction of muons / 1 GeV")):
        s = [cs.hist_graph(h, c, step=True, norm=norm) for h, c in zip(hists, cs.COLORS)]
        cs.plot("out/pt" + ("_norm" if norm else ""), s, labels, "Muon p_{T} [GeV]", yt)
    # groups x stages, band layout
    cs.plot(..., labels=None, layout="band", groups={"5.25 / 0.5": cs.BLUE, "55.0 / 10.0": cs.ORANGE},
            stages={"Matched": cs.OPEN_DASHED, "Matched and vertexed": cs.FILLED_SOLID}, group_header="m_{#chi} / #Deltam [GeV]")
"""
from pathlib import Path
import math
import numpy as np
import ROOT

BLUE, ORANGE, GREEN, YELLOW, SKY, PURPLE, BLACK = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9", "#CC79A7", "#000000"
COLORS = (BLUE, ORANGE, GREEN, YELLOW, SKY, PURPLE, BLACK)
OPEN_DASHED, FILLED_SOLID = (2, 24), (1, 20)  # (line style, marker style)
MATCHED, VERTEXED = OPEN_DASHED, FILLED_SOLID
STYLE = {"era": "2022EE (13.6 TeV)", "status": "Simulation Private Work", "min_den": 20, "level": 0.6827}
LEG_DY = 0.05
_keep = []    # PyROOT objects must outlive SaveAs
_info = {}    # id(graph) -> (x range of its histogram, draw option, legend option, kind "eff" or "hist")


def setup(**style):
    """Global ROOT style; keyword arguments override STYLE (era, status, min_den, level)."""
    STYLE.update(style)
    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0); ROOT.gStyle.SetOptTitle(0)
    ROOT.gStyle.SetTextFont(42); ROOT.gStyle.SetLabelFont(42, "XYZ"); ROOT.gStyle.SetTitleFont(42, "XYZ")
    ROOT.gStyle.SetPadTickX(1); ROOT.gStyle.SetPadTickY(1); ROOT.gStyle.SetPalette(ROOT.kViridis)
    ROOT.TGaxis.SetMaxDigits(6)


def arrays(h):
    """(edges, values, variances) of the normal bins of a UHI histogram or a ROOT TH1."""
    if hasattr(h, "GetNbinsX"):
        n, ax = h.GetNbinsX(), h.GetXaxis()
        edges = np.array([ax.GetBinLowEdge(i) for i in range(1, n + 2)])
        return (edges, np.array([h.GetBinContent(i) for i in range(1, n + 1)]),
                np.array([h.GetBinError(i) ** 2 for i in range(1, n + 1)]))
    bins = list(h.axes[0])
    edges = np.array([bins[0][0]] + [b[1] for b in bins], dtype=float)
    var = h.variances()
    return edges, np.asarray(h.values(), float), None if var is None else np.asarray(var, float)


def _style(gr, color, line, marker, edges, opt="P", legopt="lp", kind="hist"):
    _info[id(gr)] = ((edges[0], edges[-1]), opt, legopt, kind)
    c = ROOT.TColor.GetColor(color)
    gr.SetLineColor(c); gr.SetMarkerColor(c); gr.SetLineStyle(line); gr.SetMarkerStyle(marker)
    gr.SetLineWidth(2); gr.SetMarkerSize(1.0)
    _keep.append(gr)
    return gr


def eff_graph(num, den, color, line=1, marker=20, min_den=None, level=None):
    """num/den per bin (efficiency, acceptance, fraction) with central Clopper-Pearson bars; raw unweighted counts
    (Poisson variances) required; bins with den < min_den are not drawn."""
    min_den = STYLE["min_den"] if min_den is None else min_den
    level = STYLE["level"] if level is None else level
    e, k, vk = arrays(num)
    e2, n, vn = arrays(den)
    poisson = all(v is None or np.allclose(v, x) for v, x in ((vk, k), (vn, n)))  # weighted fills fail this
    if not poisson or not np.allclose(e, e2) or np.any(k > n) or np.any(k < 0) or np.any(k != np.round(k)) or np.any(n != np.round(n)):
        raise ValueError("eff_graph needs equal binning and raw (Poisson-variance) counts with 0 <= num <= den")
    gr = ROOT.TGraphAsymmErrors()
    for i in np.flatnonzero((n > 0) & (n >= min_den)):
        r, w, x = k[i] / n[i], 0.5 * (e[i + 1] - e[i]), 0.5 * (e[i] + e[i + 1])
        lo = ROOT.TEfficiency.ClopperPearson(int(n[i]), int(k[i]), level, False)
        hi = ROOT.TEfficiency.ClopperPearson(int(n[i]), int(k[i]), level, True)
        j = gr.GetN(); gr.SetPoint(j, x, r); gr.SetPointError(j, w, w, r - lo, hi - r)
    return _style(gr, color, line, marker, e, kind="eff")


def hist_graph(h, color, line=1, marker=20, step=False, norm=False, errors=True):
    """Distribution: points with stored errors (none if the provider reports variances=None), or step=True for an
    outline without errors. norm=True draws contents and errors divided by the sum of all normal bins (flow excluded,
    also bins later cropped by xlim; no division by bin width); say so in the y title. The input is not modified."""
    e, v, var = arrays(h)
    err = np.zeros_like(v) if var is None or not errors else np.sqrt(var)
    if norm:
        s = v.sum()
        if s <= 0:
            raise ValueError("norm=True needs a positive sum of the normal bins")
        v, err = v / s, err / s
    if step:
        gr = ROOT.TGraphAsymmErrors()  # same interface as the other series (zero errors)
        for i in range(len(v)):
            gr.SetPoint(gr.GetN(), e[i], v[i]); gr.SetPoint(gr.GetN(), e[i + 1], v[i])
        return _style(gr, color, line, marker, e, opt="L", legopt="l")
    gr = ROOT.TGraphAsymmErrors()
    for i in range(len(v)):
        w = 0.5 * (e[i + 1] - e[i])
        gr.SetPoint(i, e[i] + w, v[i]); gr.SetPointError(i, w, w, err[i], err[i])
    return _style(gr, color, line, marker, e)


def canvas(name, band=False):
    c = ROOT.TCanvas(name, "", 900, 800) if band else ROOT.TCanvas(name, "", 800, 700)
    c.SetLeftMargin(0.15); c.SetRightMargin(0.04); c.SetBottomMargin(0.13); c.SetTopMargin(0.30 if band else 0.08)
    _keep.append(c)
    return c


def frame(c, xlo, xhi, ylo, yhi, xtitle, ytitle):
    h = c.DrawFrame(xlo, ylo, xhi, yhi); _keep.append(h)
    h.GetXaxis().SetTitle(xtitle); h.GetYaxis().SetTitle(ytitle)
    for ax in (h.GetXaxis(), h.GetYaxis()):
        ax.SetTitleSize(0.05); ax.SetLabelSize(0.042)
    h.GetXaxis().SetTitleOffset(1.1); h.GetYaxis().SetTitleOffset(1.4)
    return h


def header(c, x0, y, sizes, xr):
    """CMS + status at left, era at right (NDC)."""
    c.cd()
    t = ROOT.TLatex(); t.SetNDC(); _keep.append(t)
    t.SetTextFont(61); t.SetTextSize(sizes[0]); t.SetTextAlign(11); t.DrawLatex(x0, y, "CMS")
    t.SetTextFont(52); t.SetTextSize(sizes[1]); t.DrawLatex(x0 + 0.105, y, STYLE["status"])
    t.SetTextFont(42); t.SetTextAlign(31); t.DrawLatex(xr, y, STYLE["era"])
    return t


def context(c, lines=(), band=False):
    """Header plus context lines (top left inside the frame; in the band above it for the band layout)."""
    y, big, small, txt, y0, dy = (0.95, 0.04, 0.031, 0.028, 0.90, 0.04) if band else (0.94, 0.05, 0.038, 0.035, 0.88, 0.045)
    t = header(c, 0.15, y, (big, small), 0.96)
    t.SetTextSize(txt); t.SetTextAlign(13)
    for i, line in enumerate(lines):
        t.DrawLatex(0.19, y0 - dy * i, line)


def legend(entries, y2=0.89, x1=0.52, header=None, x2=0.94, ncol=1, size=0.033, y1=None):
    """entries: (object, label, option); borderless, transparent; height LEG_DY per row unless y1 is given."""
    rows = -(-len(entries) // ncol) + bool(header)
    leg = ROOT.TLegend(x1, y2 - LEG_DY * rows if y1 is None else y1, x2, y2); _keep.append(leg)
    leg.SetBorderSize(0); leg.SetFillStyle(0); leg.SetTextSize(size); leg.SetNColumns(ncol)
    if header:
        leg.SetHeader(header)
    for o, lab, opt in entries:
        leg.AddEntry(o, lab, opt)
    leg.Draw()
    return leg


def _points(graphs):
    return [(gr.GetPointX(i) - gr.GetErrorXlow(i), gr.GetPointX(i) + gr.GetErrorXhigh(i),
             gr.GetPointY(i) - gr.GetErrorYlow(i), gr.GetPointY(i) + gr.GetErrorYhigh(i))
            for gr in graphs for i in range(gr.GetN())]


def _round_up(y):
    """0.05 steps on the efficiency scale (0.1-2, as on 2026-11-02); otherwise 5% of the leading decade."""
    if 0.1 <= y <= 2:
        return math.ceil(y * 20 - 1e-9) / 20
    m = 10 ** math.floor(math.log10(y))
    return math.ceil(y / m * 20 - 1e-9) / 20 * m


def fit_ymax(graphs, xlo, xhi, n_leg, x1, lines):
    """frame layout (plot-cms): smallest y max keeping every point and its bars clear of the context lines and of the
    legend, with the legend top right or bottom right; y min is 0. Returns (ymax, legend y2)."""
    pts = [p for p in _points(graphs) if p[1] >= xlo and p[0] <= xhi]
    if not pts or max(p[3] for p in pts) <= 0:
        return 1.05, 0.89
    fy = lambda ndc: (ndc - 0.13) / 0.79
    xd = lambda ndc: xlo + (ndc - 0.15) / 0.81 * (xhi - xlo)
    text_x = (xd(0.15), xd(0.19 + 0.0135 * max([len(s.replace("#", "")) for s in lines] or [0])))
    text_f = fy(0.88 - 0.045 * len(lines) - 0.01) if lines else 0.97
    leg_x = (xd(x1), xd(0.94))
    over = lambda p, r: p[0] < r[1] and p[1] > r[0]
    top = lambda p: text_f if lines and over(p, text_x) else 0.97
    lb = fy(0.89 - LEG_DY * n_leg - 0.01)
    best = (max(p[3] / (min(top(p), lb) if over(p, leg_x) else top(p)) for p in pts), 0.89)
    y2b = 0.16 + LEG_DY * n_leg
    ym = max(p[3] / top(p) for p in pts)
    if all(p[2] > fy(y2b + 0.01) * ym for p in pts if over(p, leg_x)) and ym < best[0]:
        best = (ym, y2b)
    return _round_up(best[0]), best[1]


def _proxy(color, line, marker):
    p = ROOT.TGraph(); _keep.append(p)
    c = ROOT.TColor.GetColor(color)
    p.SetLineColor(c); p.SetMarkerColor(c); p.SetLineStyle(line); p.SetMarkerStyle(marker); p.SetLineWidth(2)
    return p


def split_keys(groups, stages, group_header=None):
    """Band layout keys: groups {label: color} as colored lines (top right), stages {label: (line, marker)} in black,
    two columns, just above the frame."""
    if groups:
        legend([(_proxy(col, 1, 20), lab, "l") for lab, col in groups.items()], y2=0.915, y1=0.79, x1=0.60,
               header=group_header, size=0.028)
    if stages:
        legend([(_proxy(BLACK, ls, ms), lab, "lp") for lab, (ls, ms) in stages.items()], y2=0.775, y1=0.72, x1=0.19,
               ncol=2, size=0.028)


def save(c, base, pdf=True, log=False, logy=None):
    """PNG (+ PDF); log=True adds base_log.png with points <= 0 left out, then restores the linear canvas.
    logy = (min, max) of the log copy; default 1e-3..5 when every series is an eff_graph, else from the data."""
    base = Path(base); base.parent.mkdir(parents=True, exist_ok=True)
    c.Modified(); c.SaveAs(str(base) + ".png")
    if pdf:
        c.SaveAs(str(base) + ".pdf")
    if not log:
        return
    prims = list(c.GetListOfPrimitives())
    fr = [p for p in prims if p.InheritsFrom("TH1")][0]
    lin = (fr.GetMinimum(), fr.GetMaximum())
    graphs, clones = [p for p in prims if p.InheritsFrom("TGraph")], []
    for g in graphs:
        c.GetListOfPrimitives().Remove(g)
        pos = g.Clone(); _keep.append(pos); clones.append(pos)
        step = _info.get(id(g), (0, "P"))[1] == "L"
        for i in range(pos.GetN() - 1, -1, -1):
            if pos.GetPointY(i) <= 0:
                if step:   # outline drops below the frame instead of bridging empty bins
                    pos.SetPoint(i, pos.GetPointX(i), 1e-300)
                else:
                    pos.RemovePoint(i)
        pos.Draw(_info.get(id(g), (None, "P"))[1] + " same")
    if logy is None:
        ys = [p for p in _points(clones) if p[3] > 1e-299]
        hi = max([p[3] for p in ys] or [1])
        if all(_info.get(id(g), (0, 0, 0, "hist"))[3] == "eff" for g in graphs):
            logy = (1e-3, 5)
        else:
            pos_lo = [p[2] if p[2] > 0 else (p[2] + p[3]) / 2 for p in ys if p[3] > 0]
            logy = (max(min(pos_lo) / 2, hi * 1e-7), hi * 50) if pos_lo else (1e-3, 1)  # nothing positive: empty copy
    fr.SetMinimum(logy[0]); fr.SetMaximum(logy[1]); c.SetLogy(1); c.Modified()
    c.SaveAs(str(base) + "_log.png")
    for pos, g in zip(clones, graphs):  # restore the linear canvas
        c.GetListOfPrimitives().Remove(pos); g.Draw(_info.get(id(g), (None, "P"))[1] + " same")
    fr.SetMinimum(lin[0]); fr.SetMaximum(lin[1]); c.SetLogy(0); c.Modified()


def plot(base, graphs, labels, xtitle, ytitle, lines=(), layout="frame", xlim=None, ylim=None, x1=0.52, header=None,
         groups=None, stages=None, group_header=None, pdf=True, log=False, logy=None):
    """One 1D overlay of graphs from eff_graph / hist_graph.
    frame: labels are per-series legend labels (header optional); y range fitted unless ylim=(min, max) is given.
    band: groups/stages keys (labels unused); y max 1.05 for values in [0, 1.05], else fitted with 10% headroom.
    x range: the full binning unless xlim (display crop only). y min: 0, unless points go negative (then fitted)."""
    band = layout == "band"
    c = canvas("c_" + Path(base).name, band)
    if xlim is None:
        xlim = (min(_info[id(g)][0][0] for g in graphs), max(_info[id(g)][0][1] for g in graphs))
    pts = [p for p in _points(graphs) if p[1] >= xlim[0] and p[0] <= xlim[1]]
    lo = min([p[2] for p in pts] or [0])
    y2 = 0.89
    if ylim is None:
        hi = max([p[3] for p in pts] or [1])
        if lo < 0:      # signed contents: simple symmetric headroom, legend top right
            pad = 0.1 * ((hi - lo) or abs(lo))
            ylim = (lo - pad, hi + 3 * pad)
        elif band:
            ylim = (0, 1.05 if hi <= 1.05 else _round_up(1.1 * hi))
        else:
            ymax, y2 = fit_ymax(graphs, xlim[0], xlim[1], len(graphs) + bool(header), x1, lines)
            ylim = (0, ymax)
    frame(c, xlim[0], xlim[1], ylim[0], ylim[1], xtitle, ytitle)
    c.Update()
    for g in graphs:
        g.Draw(_info.get(id(g), (None, "P"))[1] + " same")
    context(c, lines, band)
    if band:
        split_keys(groups or {}, stages or {}, group_header)
    else:
        legend([(g, lab, _info.get(id(g), (None, None, "lp"))[2]) for g, lab in zip(graphs, labels)],
               y2=y2, x1=x1, header=header)
    save(c, base, pdf, log, logy)
    return c



def map2d(base, h, xtitle, ytitle, ztitle, lines=(), zrange=None, low=None, pdf=True, logz=False):
    """TH2 as COLZ (bins with content 0 are not drawn; set a real 0 to 1e-9 to draw it). low: optional bool array
    [nx+2, ny+2] of bins to outline in red, outer border only (edges not shared with another flagged bin)."""
    c = ROOT.TCanvas("c_" + Path(base).name, "", 800, 700); _keep.append(c)
    c.SetLeftMargin(0.13); c.SetRightMargin(0.17); c.SetTopMargin(0.08); c.SetBottomMargin(0.13)
    c.SetLogz(int(logz))
    h.GetXaxis().SetTitle(xtitle); h.GetYaxis().SetTitle(ytitle); h.GetZaxis().SetTitle(ztitle)
    for ax in (h.GetXaxis(), h.GetYaxis(), h.GetZaxis()):
        ax.SetTitleSize(0.045); ax.SetLabelSize(0.038)
    h.GetYaxis().SetTitleOffset(1.25); h.GetZaxis().SetTitleOffset(1.35)
    if zrange:
        h.SetMinimum(zrange[0]); h.SetMaximum(zrange[1])
    h.Draw("COLZ")
    if low is not None:
        xa, ya = h.GetXaxis(), h.GetYaxis()
        for i, j in zip(*map(lambda a: a.tolist(), np.nonzero(low))):
            x0, x1, y0, y1 = xa.GetBinLowEdge(i), xa.GetBinUpEdge(i), ya.GetBinLowEdge(j), ya.GetBinUpEdge(j)
            for nb, seg in (((i - 1, j), (x0, y0, x0, y1)), ((i + 1, j), (x1, y0, x1, y1)),
                            ((i, j - 1), (x0, y0, x1, y0)), ((i, j + 1), (x0, y1, x1, y1))):
                if not low[nb]:
                    ln = ROOT.TLine(*seg); ln.SetLineColor(ROOT.kRed); ln.SetLineWidth(2); ln.Draw(); _keep.append(ln)
    t = header(c, 0.13, 0.94, (0.05, 0.038), 0.83)
    t.SetTextSize(0.035); t.SetTextAlign(33)
    for k, line in enumerate(lines):
        t.DrawLatex(0.80, 0.89 - 0.045 * k, line)
    c.RedrawAxis()
    save(c, base, pdf)
    return c
