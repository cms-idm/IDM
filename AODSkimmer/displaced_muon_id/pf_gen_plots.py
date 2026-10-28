#!/usr/bin/env python3
"""DSA legs of the selected vertices: PF muon and DSA pT vs gen, from saved pf_muon_overlap.py legs trees (no rerun).

python3 pf_gen_plots.py --input prox01=pf_overlap_prox01.root --input st2=pf_overlap_st2.root --plot-dir plots/gen [--log]
r_gen(X) = (pT_gen - pT_X)/pT_gen, X = the variant's PF muon or the DSA leg; gen pT at production, reco pT at the
reference point. Plots per variant: r_gen of the PF muon and of the DSA leg for the PF-associated DSA legs, overlaid,
unnormalized and unit area (incl. flow). One overlay of all variants: fraction of all selected DSA legs whose PF muon has
the leg's charge (= gen charge) and |r_gen| < 0.1, vs gen |d0|; Clopper-Pearson (approximate, correlated legs).
Raw unweighted. Study note: kb/studies/displaced-pf-muon-overlap.md (Part 3, pT vs gen).
"""
import argparse
from array import array
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preselected_vertexing_analysis import flowcounts

DSA = 1
R_EDGES = np.round(np.arange(-1, 1.001, 0.02), 2)
D0_EDGES = np.arange(0, 60.1, 2.0)
LABELS = {"st2": "Station 2, #DeltaR < 0.2", "prox01": "Innermost hit, #DeltaR < 0.1", "prox02": "Innermost hit, #DeltaR < 0.2",
          "seg": "Segments #geq 2/3", "or01": "Hit < 0.1 or segments", "or02": "Hit < 0.2 or segments"}
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
TOP, BOTTOM = 0.92, 0.13


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", action="append", required=True, help="variant=pf_overlap_<variant>.root")
    p.add_argument("--plot-dir", required=True)
    p.add_argument("--log", action="store_true", help="log copies of the r_gen distributions")
    args = p.parse_args()
    import ROOT
    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0); ROOT.gStyle.SetOptTitle(0); ROOT.gStyle.SetTextFont(42)
    ROOT.gStyle.SetLabelFont(42, "XYZ"); ROOT.gStyle.SetTitleFont(42, "XYZ"); ROOT.gStyle.SetPadTickX(1); ROOT.gStyle.SetPadTickY(1)
    ROOT.gStyle.SetEndErrorSize(0); ROOT.TH1.AddDirectory(False); ROOT.TGaxis.SetMaxDigits(6)
    out = Path(args.plot_dir); out.mkdir(parents=True, exist_ok=True)
    color = lambda i: ROOT.TColor.GetColor(PALETTE[i % len(PALETTE)])
    keep, count = [], [0, 0]

    def canvas(name):
        c = ROOT.TCanvas(name, "", 800, 700)
        c.SetLeftMargin(0.17); c.SetRightMargin(0.04); c.SetTopMargin(1 - TOP); c.SetBottomMargin(BOTTOM)
        keep.append(c)
        return c

    def frame(h, xt, yt):
        h.GetXaxis().SetTitle(xt); h.GetYaxis().SetTitle(yt); h.GetXaxis().SetNdivisions(505)
        for ax in (h.GetXaxis(), h.GetYaxis()):
            ax.SetTitleSize(0.05); ax.SetLabelSize(0.042)
        h.GetXaxis().SetTitleOffset(1.15); h.GetYaxis().SetTitleOffset(1.65)

    def labels(c, lines):
        c.cd()
        t = ROOT.TLatex(); t.SetNDC(); keep.append(t)
        t.SetTextFont(61); t.SetTextSize(0.05); t.SetTextAlign(11); t.DrawLatex(0.17, 0.94, "CMS")
        t.SetTextFont(52); t.SetTextSize(0.038); t.DrawLatex(0.275, 0.94, "Simulation Private Work")
        t.SetTextFont(42); t.SetTextAlign(31); t.DrawLatex(0.96, 0.94, "2022EE (13.6 TeV)")
        t.SetTextSize(0.032); t.SetTextAlign(13)
        for i, line in enumerate(lines): t.DrawLatex(0.21, 0.895 - 0.042 * i, line)

    def legend(n, nlines, cols=2):
        size = 0.030
        top = 0.895 - 0.042 * nlines - 0.01
        bottom = top - ((n + cols - 1) // cols) * size * 1.25
        leg = ROOT.TLegend(0.20, bottom, 0.95, top); keep.append(leg)
        leg.SetBorderSize(0); leg.SetFillStyle(0); leg.SetTextSize(size); leg.SetNColumns(cols)
        return leg, bottom

    headroom = lambda lo, hi, bottom: lo + (hi - lo) * (TOP - BOTTOM) / max(bottom - BOTTOM - 0.02, 0.1)

    def hist(name, edges, values):
        h = ROOT.TH1D(name, "", len(edges) - 1, array("d", edges)); h.Sumw2()
        for b, n in enumerate(flowcounts(values, edges)):
            h.SetBinContent(b, float(n)); h.SetBinError(b, float(np.sqrt(n)))
        keep.append(h)
        return h

    data = {}
    for item in args.input:
        v, path = item.split("=", 1)
        f = ROOT.TFile.Open(path)
        if f.Get("variant") and f.Get("variant").GetTitle() != v:
            raise RuntimeError("Variant mismatch: %s vs %s" % (v, f.Get("variant").GetTitle()))
        t = {k: np.asarray(x) for k, x in
             ROOT.RDataFrame(f.Get("legs")).AsNumpy(["type", "pfMatched", "sameSign", "d0", "pt", "legPt", "pfPt"]).items()}
        f.Close()
        m = t["type"] == DSA
        t = {k: x[m] for k, x in t.items()}
        if np.any(~(t["pt"] > 0)):
            raise RuntimeError("Non-positive gen pT")
        t["rpf"] = np.where(t["pfMatched"] == 1, (t["pt"] - t["pfPt"]) / t["pt"], np.nan)
        t["rleg"] = (t["pt"] - t["legPt"]) / t["pt"]
        data[v] = t
        good = (t["pfMatched"] == 1) & (t["sameSign"] == 1) & (np.abs(t["rpf"]) < 0.1)
        print("VARIANT %s DSA legs %d associated %d charge+|r_gen|<0.1 %d" % (v, int(m.sum()), int((t["pfMatched"] == 1).sum()), int(good.sum())))

    # r_gen of the PF muon and of the DSA leg, PF-associated DSA legs, per variant.
    for v, t in data.items():
        a = t["pfMatched"] == 1
        for norm in (False, True):
            c = canvas("c_r_%s_%d" % (v, norm))
            hs = [(hist("hp_%s_%d" % (v, norm), R_EDGES, t["rpf"][a]), "PF muon"),
                  (hist("hl_%s_%d" % (v, norm), R_EDGES, t["rleg"][a]), "DSA leg")]
            n = int(a.sum())
            for i, (h, _) in enumerate(hs):
                if norm and n: h.Scale(1.0 / n)
                h.SetLineColor(color(i)); h.SetLineWidth(2); h.SetLineStyle(1 + i)
            lines = ["PF-associated DSA legs (%s)" % LABELS.get(v, v),
                     ("Unit area incl. outside [#minus1, 1]" if norm else "Not drawn: outside [#minus1, 1]")]
            leg, bottom = legend(2, len(lines))
            top = max(h.GetBinContent(h.GetMaximumBin()) for h, _ in hs) or 1.0
            fr = hs[0][0]
            frame(fr, "(p_{T}^{gen} #minus p_{T}) / p_{T}^{gen}", ("Fraction of legs / %g" if norm else "Legs / %g") % 0.02)
            fr.SetMinimum(0); fr.SetMaximum(headroom(0, top, bottom)); fr.Draw("hist"); hs[1][0].Draw("hist same")
            for h, lab in hs:
                leg.AddEntry(h, "%s (%d)" % (lab, n), "l")
            leg.Draw(); labels(c, lines)
            name = "dsa_rgen_%s%s" % (v, "_unitarea" if norm else "")
            c.SaveAs(str(out / (name + ".png"))); count[0] += 1
            if args.log:
                low = min([h.GetBinContent(b) for h, _ in hs for b in range(1, h.GetNbinsX() + 1) if h.GetBinContent(b) > 0] or [top * 1e-3]) * 0.5
                lmax = np.log10(low) + (np.log10(top) - np.log10(low)) * (TOP - BOTTOM) / max(bottom - BOTTOM - 0.02, 0.1)
                c.SetLogy(1); fr.SetMinimum(low); fr.SetMaximum(10 ** lmax)
                c.Modified(); c.SaveAs(str(out / (name + "_log.png"))); count[1] += 1

    # Fraction of all selected DSA legs with a right-charge PF muon within 10% of gen pT, vs gen |d0|.
    c = canvas("c_good_d0")
    lines = ["Selected-vertex DSA legs, all samples",
             "PF muon with the gen charge and |r_{gen}| < 0.1", "Legs beyond the x range not shown"]
    leg, bottom = legend(len(data), len(lines), cols=1)
    fr = ROOT.TH1D("fr_good", "", 1, D0_EDGES[0], D0_EDGES[-1]); keep.append(fr)
    fr.SetMinimum(0); fr.SetMaximum(headroom(0, 0.6, bottom))
    frame(fr, "Gen |d_{0}| [cm]", "Fraction of DSA legs"); fr.Draw("axis")
    for i, (v, t) in enumerate(data.items()):
        good = (t["pfMatched"] == 1) & (t["sameSign"] == 1) & (np.abs(t["rpf"]) < 0.1)
        den, num = hist("den_%s" % v, D0_EDGES, t["d0"]), hist("num_%s" % v, D0_EDGES, t["d0"][good])
        eff = ROOT.TEfficiency(num, den); keep.append(eff)
        gr = eff.CreateGraph(); keep.append(gr)
        gr.SetLineColor(color(i)); gr.SetMarkerColor(color(i)); gr.SetMarkerStyle(20 + i); gr.SetMarkerSize(0.8); gr.SetLineWidth(2)
        gr.Draw("PZ same")
        leg.AddEntry(gr, "%s (%d of %d)" % (LABELS.get(v, v), int(good.sum()), len(t["d0"])), "lp")
    leg.Draw(); labels(c, lines)
    c.SaveAs(str(out / "dsa_good_pf_vs_d0.png")); count[0] += 1
    print("PLOTTED %d linear PNGs and %d log copies" % tuple(count))


if __name__ == "__main__":
    main()
