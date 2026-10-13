#!/usr/bin/env python3

import os
import math
import argparse
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)


def branch_exists(tree, name):
    return tree.GetBranch(name) is not None


def finite_dr(x, invalid=900.0):
    try:
        x = float(x)
    except Exception:
        return False
    return math.isfinite(x) and x >= 0.0 and x < invalid


def make_hist(name, nbins, xmin, xmax):
    h = ROOT.TH1F(name, "", nbins, xmin, xmax)
    h.Sumw2()
    return h


def normalize(h):
    integral = h.Integral()
    if integral > 0:
        h.Scale(1.0 / integral)


def draw_overlay(hists, labels, outbase, title, xtitle, normalize_to_unit=True, logy=False):
    canvas = ROOT.TCanvas("c_" + os.path.basename(outbase), "", 1000, 800)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.04)
    canvas.SetTopMargin(0.08)
    canvas.SetBottomMargin(0.12)

    if logy:
        canvas.SetLogy()

    colors = [
        ROOT.kBlack,
        ROOT.kAzure + 1,
        ROOT.kRed + 1,
        ROOT.kGreen + 2,
        ROOT.kMagenta + 1,
    ]

    max_y = 0.0
    for h in hists:
        if normalize_to_unit:
            normalize(h)
        max_y = max(max_y, h.GetMaximum())

    for i, h in enumerate(hists):
        h.SetLineColor(colors[i % len(colors)])
        h.SetMarkerColor(colors[i % len(colors)])
        h.SetLineWidth(3)

        h.GetXaxis().SetTitle(xtitle)
        h.GetYaxis().SetTitle("Normalized entries" if normalize_to_unit else "Entries")
        h.GetXaxis().SetTitleSize(0.045)
        h.GetYaxis().SetTitleSize(0.045)
        h.GetXaxis().SetLabelSize(0.04)
        h.GetYaxis().SetLabelSize(0.04)
        h.GetYaxis().SetTitleOffset(1.35)

        if logy:
            h.SetMinimum(1e-5 if normalize_to_unit else 0.5)
            h.SetMaximum(max_y * 20.0 if max_y > 0 else 1.0)
        else:
            h.SetMinimum(0.0)
            h.SetMaximum(max_y * 1.35 if max_y > 0 else 1.0)

        h.Draw("hist" if i == 0 else "hist same")

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextFont(42)
    latex.SetTextSize(0.043)
    latex.DrawLatex(0.16, 0.94, title)

    leg = ROOT.TLegend(0.47, 0.68, 0.93, 0.90)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.032)

    for h, label in zip(hists, labels):
        leg.AddEntry(h, label, "l")

    leg.Draw()

    canvas.SaveAs(outbase + ".pdf")
    canvas.SaveAs(outbase + ".png")


def book_hist_set(prefix, nbins, xmax, zoom_xmax):
    return {
        "gen": make_hist(prefix + "_gen", nbins, 0.0, xmax),
        "before": make_hist(prefix + "_before", nbins, 0.0, xmax),
        "after": make_hist(prefix + "_after", nbins, 0.0, xmax),
        "gen_zoom": make_hist(prefix + "_gen_zoom", nbins, 0.0, zoom_xmax),
        "before_zoom": make_hist(prefix + "_before_zoom", nbins, 0.0, zoom_xmax),
        "after_zoom": make_hist(prefix + "_after_zoom", nbins, 0.0, zoom_xmax),
    }


def fill_hist_pair(h_full, h_zoom, value, invalid):
    if finite_dr(value, invalid):
        h_full.Fill(value)
        h_zoom.Fill(value)
        return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Overlay gen dimuon dR with closest gen-to-DSA dR before/after propagation."
    )
    parser.add_argument("root_file", help="Input ntuple ROOT file")
    parser.add_argument("-t", "--tree", default="ntuples/outT", help="Tree path")
    parser.add_argument("-o", "--outdir", default="dr_overlays", help="Output directory")
    parser.add_argument("--station", choices=["1", "2"], default="1", help="Propagation station to use")
    parser.add_argument("--nbins", type=int, default=80)
    parser.add_argument("--xmax", type=float, default=5.0)
    parser.add_argument("--zoom-xmax", type=float, default=0.5)
    parser.add_argument("--raw", action="store_true", help="Do not normalize histograms")
    parser.add_argument("--logy", action="store_true", help="Use log y-axis")
    parser.add_argument("--invalid", type=float, default=900.0, help="Invalid dR threshold; excludes 999 by default")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    f = ROOT.TFile.Open(args.root_file)
    if not f or f.IsZombie():
        raise RuntimeError(f"Could not open {args.root_file}")

    t = f.Get(args.tree)
    if not t:
        raise RuntimeError(f"Could not find tree {args.tree} in {args.root_file}")

    st = args.station

    before_branches = [
        "GenSigMuon_minDrToDSAMuon",
        "GenSigAntiMuon_minDrToDSAMuon",
    ]

    after_branches = [
        f"GenSigMuon_minDrToDSAMuonPropSt{st}",
        f"GenSigAntiMuon_minDrToDSAMuonPropSt{st}",
    ]

    required = [
        "genSigDimuon_dr",
        *before_branches,
        *after_branches,
    ]

    missing = [b for b in required if not branch_exists(t, b)]
    if missing:
        print("Missing required branches:")
        for b in missing:
            print("  " + b)
        raise RuntimeError("Missing required branches.")

    h_all = book_hist_set("all_events", args.nbins, args.xmax, args.zoom_xmax)
    h_prop_success = book_hist_set("prop_success_events", args.nbins, args.xmax, args.zoom_xmax)

    n_events = 0

    n_all_gen = 0
    n_all_before = 0
    n_all_after = 0

    n_prop_success_events = 0
    n_prop_success_gen = 0
    n_prop_success_before = 0
    n_prop_success_after = 0

    for ev in t:
        n_events += 1

        gen_dr = getattr(ev, "genSigDimuon_dr")

        before_vals = [getattr(ev, br) for br in before_branches]
        after_vals = [getattr(ev, br) for br in after_branches]

        # ------------------------------------------------------------
        # Inclusive/all-events histograms.
        # ------------------------------------------------------------
        if fill_hist_pair(h_all["gen"], h_all["gen_zoom"], gen_dr, args.invalid):
            n_all_gen += 1

        for dr in before_vals:
            if fill_hist_pair(h_all["before"], h_all["before_zoom"], dr, args.invalid):
                n_all_before += 1

        for dr in after_vals:
            if fill_hist_pair(h_all["after"], h_all["after_zoom"], dr, args.invalid):
                n_all_after += 1

        # ------------------------------------------------------------
        # Propagation-success event selection.
        #
        # This requires both signal gen muons to have finite propagated
        # closest-DSA dR at the requested station.
        # The gen dimuon dR is also filled only for these events.
        # ------------------------------------------------------------
        prop_success_event = all(finite_dr(dr, args.invalid) for dr in after_vals)

        if not prop_success_event:
            continue

        n_prop_success_events += 1

        if fill_hist_pair(
            h_prop_success["gen"],
            h_prop_success["gen_zoom"],
            gen_dr,
            args.invalid,
        ):
            n_prop_success_gen += 1

        for dr in before_vals:
            if fill_hist_pair(
                h_prop_success["before"],
                h_prop_success["before_zoom"],
                dr,
                args.invalid,
            ):
                n_prop_success_before += 1

        for dr in after_vals:
            if fill_hist_pair(
                h_prop_success["after"],
                h_prop_success["after_zoom"],
                dr,
                args.invalid,
            ):
                n_prop_success_after += 1

    normalize_to_unit = not args.raw

    labels_all = [
        f"gen sig #mu#mu #DeltaR, N={n_all_gen}",
        f"closest same-sign DSA before prop., N={n_all_before}",
        f"closest same-sign DSA after prop. St{st}, N={n_all_after}",
    ]

    labels_prop_success = [
        f"gen sig #mu#mu #DeltaR, N={n_prop_success_gen}",
        f"closest same-sign DSA before prop., N={n_prop_success_before}",
        f"closest same-sign DSA after prop. St{st}, N={n_prop_success_after}",
    ]

    # Inclusive/all-events plots.
    draw_overlay(
        [h_all["gen"], h_all["before"], h_all["after"]],
        labels_all,
        os.path.join(args.outdir, f"overlay_genDimuon_vs_DSA_before_after_propSt{st}_allEvents"),
        f"All events: gen #mu#mu vs closest DSA #DeltaR, station {st}",
        "#DeltaR",
        normalize_to_unit=normalize_to_unit,
        logy=args.logy,
    )

    draw_overlay(
        [h_all["gen_zoom"], h_all["before_zoom"], h_all["after_zoom"]],
        labels_all,
        os.path.join(args.outdir, f"overlay_genDimuon_vs_DSA_before_after_propSt{st}_allEvents_zoom"),
        f"All events: gen #mu#mu vs closest DSA #DeltaR, station {st} zoom",
        "#DeltaR",
        normalize_to_unit=normalize_to_unit,
        logy=args.logy,
    )

    # Propagation-success selected plots.
    draw_overlay(
        [h_prop_success["gen"], h_prop_success["before"], h_prop_success["after"]],
        labels_prop_success,
        os.path.join(args.outdir, f"overlay_genDimuon_vs_DSA_before_after_propSt{st}_propSuccessEvents"),
        f"Prop-success events: gen #mu#mu vs closest DSA #DeltaR, station {st}",
        "#DeltaR",
        normalize_to_unit=normalize_to_unit,
        logy=args.logy,
    )

    draw_overlay(
        [h_prop_success["gen_zoom"], h_prop_success["before_zoom"], h_prop_success["after_zoom"]],
        labels_prop_success,
        os.path.join(args.outdir, f"overlay_genDimuon_vs_DSA_before_after_propSt{st}_propSuccessEvents_zoom"),
        f"Prop-success events: gen #mu#mu vs closest DSA #DeltaR, station {st} zoom",
        "#DeltaR",
        normalize_to_unit=normalize_to_unit,
        logy=args.logy,
    )

    summary_path = os.path.join(args.outdir, f"summary_propSt{st}.txt")
    with open(summary_path, "w") as s:
        s.write(f"Input file: {args.root_file}\n")
        s.write(f"Tree: {args.tree}\n")
        s.write(f"Entries: {n_events}\n")
        s.write(f"Station: {st}\n\n")

        s.write("[All events]\n")
        s.write(f"genSigDimuon_dr finite entries: {n_all_gen}\n")
        s.write(f"closest DSA before propagation finite entries: {n_all_before}\n")
        s.write(f"closest DSA after propagation finite entries: {n_all_after}\n\n")

        s.write("[Propagation-success event selection]\n")
        s.write(
            "Selection: both "
            f"GenSigMuon_minDrToDSAMuonPropSt{st} and "
            f"GenSigAntiMuon_minDrToDSAMuonPropSt{st} are finite.\n"
        )
        s.write(f"selected events: {n_prop_success_events} / {n_events}\n")
        s.write(f"genSigDimuon_dr finite entries after selection: {n_prop_success_gen}\n")
        s.write(f"closest DSA before propagation finite entries after selection: {n_prop_success_before}\n")
        s.write(f"closest DSA after propagation finite entries after selection: {n_prop_success_after}\n")

    print(f"Done. Output directory: {args.outdir}")
    print(f"Summary: {summary_path}")
    print(f"Propagation-success selected events: {n_prop_success_events} / {n_events}")


if __name__ == "__main__":
    main()