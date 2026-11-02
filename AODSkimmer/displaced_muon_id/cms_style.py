#!/usr/bin/env python3
"""CMS-style PyROOT plotting helpers: the plot choices used for the 2026-11-02 mumu talk plots (mumu_talk_plots.py
plot-cms, plot-cms-pt, plot-2d-all), kept in one place so any agent reproduces them.

Style only: no event loops, selections, normalization or rebinning. Inputs are saved histograms: anything with UHI
axes/values()/variances() (hist, boost-histogram, uproot TH1) or a ROOT TH1; bin variances of a TH1 are
GetBinError()**2 (ROOT's convention, Poisson when no Sumw2 is stored). Run with PyROOT (cmsenv on LPC).

Choices kept:
* Header "CMS" (bold) + "Simulation Private Work" (italic) at left, "2022EE (13.6 TeV)" at right; no other text
  beyond short context lines (population, c#tau, mass point).
* Okabe-Ito colors; for efficiency stages "matched" = open marker 24 dashed, "matched and vertexed" = filled 20 solid.
* Efficiencies: raw counts, central Clopper-Pearson 68.3% bars, bins with den < min_den (default 20) not drawn.
* Two 1D layouts:
  "frame" (plot-cms): legend inside the frame, top or bottom right, y max fitted to the points and text
    (no empty space above the data);
  "band" (plot-cms-pt): wider canvas, color key (groups) and stage key above the frame, fixed y max (1.05).
* Linear PNG + PDF; log=True adds a log-y PNG copy (1e-3 .. 5), never replacing the linear plot.
* 2D maps: COLZ, viridis, efficiency z in [0, 1], optional red outer border around low-count bins.

Example (efficiency, frame layout):
    import cms_style as cs
    cs.setup()
    s = [cs.eff_graph(f["matched"], f["den"], cs.BLUE, *cs.MATCHED), cs.eff_graph(f["valid"], f["den"], cs.ORANGE, *cs.VERTEXED)]
    cs.plot("out/overall_lxy", s, ["Both muons matched", "Matched and vertexed"], "Gen L_{xy} [cm]", "Event efficiency",
            lines=["Both signal muons accepted"])
Band layout: cs.plot(..., layout="band", groups={"5.25 / 0.5": cs.BLUE, ...}, stages={"Matched": cs.MATCHED, ...},
group_header="m_{#chi} / #Deltam [GeV]") draws the two keys instead of per-series labels.
"""
from pathlib import Path
import numpy as np
import ROOT

BLUE, ORANGE, GREEN, YELLOW, BLACK = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#000000"
MATCHED, VERTEXED = (2, 24), (1, 20)  # (line style, marker style)
MIN_DEN = 20
LEG_DY = 0.05
_keep = []  # PyROOT objects must outlive SaveAs
_xrange = {}  # id(graph) -> (first, last) bin edge of its histogram, the default x range


def setup():
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


def _style(gr, color, line, marker, edges):
    _xrange[id(gr)] = (edges[0], edges[-1])
    c = ROOT.TColor.GetColor(color)
    gr.SetLineColor(c); gr.SetMarkerColor(c); gr.SetLineStyle(line); gr.SetMarkerStyle(marker)
    gr.SetLineWidth(2); gr.SetMarkerSize(1.0)
    _keep.append(gr)
    return gr


def eff_graph(num, den, color, line=1, marker=20, min_den=MIN_DEN, level=0.6827):
    """num/den per bin with central Clopper-Pearson bars; raw unweighted counts required."""
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
    return _style(gr, color, line, marker, e)


def hist_graph(h, color, line=1, marker=20):
    """Distribution as points with stored errors (none if the provider reports variances=None); contents unchanged."""
    e, v, var = arrays(h)
    err = np.zeros_like(v) if var is None else np.sqrt(var)
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


def labels(c, lines=(), band=False, x0=0.15, era="2022EE (13.6 TeV)", status="Simulation Private Work", xr=0.96):
    """Header plus context lines (top left inside the frame; in the band above it for the band layout)."""
    c.cd()
    t = ROOT.TLatex(); t.SetNDC(); _keep.append(t)
    y, big, small, txt, y0, dy = (0.95, 0.04, 0.031, 0.028, 0.90, 0.04) if band else (0.94, 0.05, 0.038, 0.035, 0.88, 0.045)
    t.SetTextFont(61); t.SetTextSize(big); t.SetTextAlign(11); t.DrawLatex(x0, y, "CMS")
    t.SetTextFont(52); t.SetTextSize(small); t.DrawLatex(x0 + 0.105, y, status)
    t.SetTextFont(42); t.SetTextAlign(31); t.DrawLatex(xr, y, era)
    t.SetTextSize(txt); t.SetTextAlign(13)
    for i, line in enumerate(lines):
        t.DrawLatex(x0 + 0.04, y0 - dy * i, line)


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


def fit_ymax(graphs, xlo, xhi, n_leg, x1, lines):
    """plot-cms placement: smallest y max (in 0.05 steps) keeping every point and its bars clear of the context lines
    and of the legend, with the legend top right or bottom right. Returns (ymax, legend y2)."""
    pts = [(gr.GetPointX(i) - gr.GetErrorXlow(i), gr.GetPointX(i) + gr.GetErrorXhigh(i),
            gr.GetPointY(i) - gr.GetErrorYlow(i), gr.GetPointY(i) + gr.GetErrorYhigh(i))
           for gr in graphs for i in range(gr.GetN())]
    if not pts:
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
    return max(0.1, np.ceil(best[0] * 20) / 20), best[1]


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
    legend([(_proxy(BLACK, ls, ms), lab, "lp") for lab, (ls, ms) in stages.items()], y2=0.775, y1=0.72, x1=0.19,
           ncol=2, size=0.028)


def save(c, base, pdf=True, log=False):
    base = Path(base); base.parent.mkdir(parents=True, exist_ok=True)
    c.Modified(); c.SaveAs(str(base) + ".png")
    if pdf:
        c.SaveAs(str(base) + ".pdf")
    if log:  # extra copy; points with y <= 0 are left out (ROOT would pin them to the axis floor)
        prims = list(c.GetListOfPrimitives())
        fr = [p for p in prims if p.InheritsFrom("TH1")][0]
        lin = (fr.GetMinimum(), fr.GetMaximum())
        graphs, clones = [p for p in prims if p.InheritsFrom("TGraph")], []
        for g in graphs:
            c.GetListOfPrimitives().Remove(g)
            pos = g.Clone(); _keep.append(pos); clones.append(pos)
            for i in range(pos.GetN() - 1, -1, -1):
                if pos.GetPointY(i) <= 0:
                    pos.RemovePoint(i)
            pos.Draw("P same")
        fr.SetMinimum(1e-3); fr.SetMaximum(5); c.SetLogy(1); c.Modified()
        c.SaveAs(str(base) + "_log.png")
        for pos, g in zip(clones, graphs):  # restore the linear canvas
            c.GetListOfPrimitives().Remove(pos); g.Draw("P same")
        fr.SetMinimum(lin[0]); fr.SetMaximum(lin[1]); c.SetLogy(0); c.Modified()


def plot(base, graphs, labels_, xtitle, ytitle, lines=(), layout="frame", xlim=None, ymax=None, x1=0.52, header=None,
         groups=None, stages=None, group_header=None, pdf=True, log=False):
    """One 1D overlay. frame: labels_ are per-series legend labels, y max fitted unless given. band: groups/stages keys,
    y max 1.05 unless given. x range: the full binning unless xlim (display crop only) is given."""
    band = layout == "band"
    c = canvas("c_" + Path(base).name, band)
    if xlim is None:  # full stored binning
        xlim = (min(_xrange[id(g)][0] for g in graphs), max(_xrange[id(g)][1] for g in graphs))
    y2 = 0.89
    if ymax is None:
        ymax, y2 = (1.05, None) if band else fit_ymax(graphs, xlim[0], xlim[1], len(graphs) + bool(header), x1, lines)
    frame(c, xlim[0], xlim[1], 0, ymax, xtitle, ytitle)
    c.Update()
    for g in graphs:
        g.Draw("P same")
    labels(c, lines, band)
    if band:
        split_keys(groups or {}, stages or {}, group_header)
    else:
        legend([(g, lab, "lp") for g, lab in zip(graphs, labels_)], y2=y2, x1=x1, header=header)
    save(c, base, pdf, log)
    return c


def map2d(base, h, xtitle, ytitle, ztitle, lines=(), zrange=None, low=None, pdf=True):
    """TH2 as COLZ (bins with content 0 are not drawn; set efficiency 0 to 1e-9 to draw it). low: optional 2D bool
    array [nx+2, ny+2] of bins to outline in red; only their outer border (edges not shared with another low bin)."""
    c = ROOT.TCanvas("c_" + Path(base).name, "", 800, 700); _keep.append(c)
    c.SetLeftMargin(0.13); c.SetRightMargin(0.17); c.SetTopMargin(0.08); c.SetBottomMargin(0.13)
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
    t = ROOT.TLatex(); t.SetNDC(); _keep.append(t)
    t.SetTextFont(61); t.SetTextSize(0.05); t.SetTextAlign(11); t.DrawLatex(0.13, 0.94, "CMS")
    t.SetTextFont(52); t.SetTextSize(0.038); t.DrawLatex(0.235, 0.94, "Simulation Private Work")
    t.SetTextFont(42); t.SetTextAlign(31); t.DrawLatex(0.83, 0.94, "2022EE (13.6 TeV)")
    t.SetTextSize(0.035); t.SetTextAlign(33)
    for k, line in enumerate(lines):
        t.DrawLatex(0.80, 0.89 - 0.045 * k, line)
    c.RedrawAxis()
    save(c, base, pdf)
    return c
