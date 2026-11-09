#!/usr/bin/env python3
import os
import argparse
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

def get_tree(fname, tpath="ntuples/outT"):
    f = ROOT.TFile.Open(fname)
    if not f or f.IsZombie():
        raise RuntimeError(f"Could not open {fname}")

    t = f.Get(tpath)
    if not t:
        f.ls()
        raise RuntimeError(f"Could not find tree {tpath}")

    return f, t

def has_branch(t, br):
    return bool(t.GetBranch(br))

def save(c, outdir, name):
    c.SaveAs(os.path.join(outdir, name + ".png"))
    c.SaveAs(os.path.join(outdir, name + ".pdf"))

def draw_overlay(t, var, valid_cut, invalid_cut, outdir, name, title, x_title,
                 nbins, xmin, xmax, normalize=True):
    c = ROOT.TCanvas("c_" + name, title, 900, 700)
    c.SetLeftMargin(0.13)
    c.SetBottomMargin(0.12)

    h_valid = ROOT.TH1F("h_valid_" + name, "", nbins, xmin, xmax)
    h_invalid = ROOT.TH1F("h_invalid_" + name, "", nbins, xmin, xmax)

    t.Draw(f"{var}>>h_valid_{name}", valid_cut, "goff")
    t.Draw(f"{var}>>h_invalid_{name}", invalid_cut, "goff")

    if normalize:
        if h_valid.Integral() > 0:
            h_valid.Scale(1.0 / h_valid.Integral())
        if h_invalid.Integral() > 0:
            h_invalid.Scale(1.0 / h_invalid.Integral())

    h_valid.SetLineColor(ROOT.kBlue + 1)
    h_invalid.SetLineColor(ROOT.kRed + 1)
    h_valid.SetLineWidth(3)
    h_invalid.SetLineWidth(3)

    ymax = max(h_valid.GetMaximum(), h_invalid.GetMaximum())
    h_valid.SetMaximum(1.25 * ymax if ymax > 0 else 1.0)

    h_valid.SetTitle(title)
    h_valid.GetXaxis().SetTitle(x_title)
    h_valid.GetYaxis().SetTitle("Normalized events" if normalize else "Events")
    h_valid.GetXaxis().SetTitleSize(0.045)
    h_valid.GetYaxis().SetTitleSize(0.045)
    h_valid.GetXaxis().SetLabelSize(0.040)
    h_valid.GetYaxis().SetLabelSize(0.040)

    h_valid.Draw("HIST")
    h_invalid.Draw("HIST SAME")

    leg = ROOT.TLegend(0.62, 0.72, 0.88, 0.88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.038)
    leg.AddEntry(h_valid, "Propagation valid", "l")
    leg.AddEntry(h_invalid, "Propagation invalid", "l")
    leg.Draw()

    save(c, outdir, name)

def draw_efficiency(t, var, valid_cut, total_cut, outdir, name, title, x_title,
                    nbins, xmin, xmax):
    c = ROOT.TCanvas("c_" + name, title, 900, 700)
    c.SetLeftMargin(0.13)
    c.SetBottomMargin(0.12)

    h_total = ROOT.TH1F("h_total_" + name, "", nbins, xmin, xmax)
    h_pass = ROOT.TH1F("h_pass_" + name, "", nbins, xmin, xmax)

    t.Draw(f"{var}>>h_total_{name}", total_cut, "goff")
    t.Draw(f"{var}>>h_pass_{name}", valid_cut, "goff")

    eff = ROOT.TEfficiency(h_pass, h_total)
    eff.SetTitle(title + ";" + x_title + ";Propagation valid fraction")
    eff.SetLineWidth(3)
    eff.SetMarkerStyle(20)
    eff.Draw("AP")

    c.Update()
    gr = eff.GetPaintedGraph()
    if gr:
        gr.SetMinimum(0.0)
        gr.SetMaximum(1.05)
        gr.GetXaxis().SetTitleSize(0.045)
        gr.GetYaxis().SetTitleSize(0.045)
        gr.GetXaxis().SetLabelSize(0.040)
        gr.GetYaxis().SetLabelSize(0.040)

    save(c, outdir, name)

def draw_2d(t, xvar, yvar, cut, outdir, name, title, x_title, y_title,
            xbins, xmin, xmax, ybins, ymin, ymax):
    c = ROOT.TCanvas("c_" + name, title, 950, 750)
    c.SetRightMargin(0.15)
    c.SetLeftMargin(0.12)
    c.SetBottomMargin(0.12)

    hname = "h_" + name
    t.Draw(f"{yvar}:{xvar}>>{hname}({xbins},{xmin},{xmax},{ybins},{ymin},{ymax})", cut, "COLZ")
    h = ROOT.gDirectory.Get(hname)

    if h:
        h.SetTitle(title)
        h.GetXaxis().SetTitle(x_title)
        h.GetYaxis().SetTitle(y_title)
        h.GetZaxis().SetTitle("Events")
        h.GetXaxis().SetTitleSize(0.045)
        h.GetYaxis().SetTitleSize(0.045)
        h.GetZaxis().SetTitleSize(0.040)
        h.Draw("COLZ")

    save(c, outdir, name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root_file", nargs="?", default="test_genMuonPropagation_10000Events.root")
    parser.add_argument("-o", "--outdir", default="propagation_validity_plots")
    parser.add_argument("-t", "--tree", default="ntuples/outT")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    f, t = get_tree(args.root_file, args.tree)

    particles = [
        ("GenSigMuon", "signal muon"),
        ("GenSigAntiMuon", "signal anti-muon"),
    ]

    stations = [
        ("propSt1Valid", "St1"),
        ("propSt2Valid", "St2"),
    ]

    for tag, label in particles:
        pt = f"{tag}_pt"
        eta = f"{tag}_eta"

        if not has_branch(t, pt):
            print(f"Missing {pt}, skipping {tag}")
            continue

        for valid_branch, station in stations:
            valid = f"{tag}_{valid_branch}"

            if not has_branch(t, valid):
                print(f"Missing {valid}, skipping")
                continue

            valid_cut = f"{valid}==1"
            invalid_cut = f"{valid}==0"
            total_cut = "1"

            print(f"\n{tag} {station}")
            print("  valid:  ", int(t.GetEntries(valid_cut)))
            print("  invalid:", int(t.GetEntries(invalid_cut)))

            draw_overlay(
                t,
                pt,
                valid_cut,
                invalid_cut,
                args.outdir,
                f"{tag}_{station}_valid_vs_invalid_pt",
                f"{label}: {station} propagation validity vs p_{{T}}",
                f"{tag} p_{{T}} [GeV]",
                80,
                0,
                40,
                normalize=True
            )

            draw_efficiency(
                t,
                pt,
                valid_cut,
                total_cut,
                args.outdir,
                f"{tag}_{station}_valid_fraction_vs_pt",
                f"{label}: {station} valid fraction vs p_{{T}}",
                f"{tag} p_{{T}} [GeV]",
                40,
                0,
                40
            )

            if has_branch(t, eta):
                draw_overlay(
                    t,
                    eta,
                    valid_cut,
                    invalid_cut,
                    args.outdir,
                    f"{tag}_{station}_valid_vs_invalid_eta",
                    f"{label}: {station} propagation validity vs #eta",
                    f"{tag} #eta",
                    60,
                    -3,
                    3,
                    normalize=True
                )

                draw_efficiency(
                    t,
                    eta,
                    valid_cut,
                    total_cut,
                    args.outdir,
                    f"{tag}_{station}_valid_fraction_vs_eta",
                    f"{label}: {station} valid fraction vs #eta",
                    f"{tag} #eta",
                    48,
                    -2.4,
                    2.4
                )

                draw_2d(
                    t,
                    pt,
                    eta,
                    valid_cut,
                    args.outdir,
                    f"{tag}_{station}_valid_eta_vs_pt",
                    f"{label}: valid {station} propagation, #eta vs p_{{T}}",
                    f"{tag} p_{{T}} [GeV]",
                    f"{tag} #eta",
                    60,
                    0,
                    40,
                    60,
                    -3,
                    3
                )

                draw_2d(
                    t,
                    pt,
                    eta,
                    invalid_cut,
                    args.outdir,
                    f"{tag}_{station}_invalid_eta_vs_pt",
                    f"{label}: invalid {station} propagation, #eta vs p_{{T}}",
                    f"{tag} p_{{T}} [GeV]",
                    f"{tag} #eta",
                    60,
                    0,
                    40,
                    60,
                    -3,
                    3
                )

            # Optional displaced variables if present in your ntuple
            for var, axis, xmin, xmax, nbins in [
                (f"{tag}_vxy", f"{tag} v_{{xy}} [cm]", 0, 800, 80),
                (f"{tag}_vz", f"{tag} v_{{z}} [cm]", -1000, 1000, 80),
            ]:
                if has_branch(t, var):
                    draw_overlay(
                        t,
                        var,
                        valid_cut,
                        invalid_cut,
                        args.outdir,
                        f"{tag}_{station}_valid_vs_invalid_{var.split('_')[-1]}",
                        f"{label}: {station} propagation validity vs {axis}",
                        axis,
                        nbins,
                        xmin,
                        xmax,
                        normalize=True
                    )

                    draw_efficiency(
                        t,
                        var,
                        valid_cut,
                        total_cut,
                        args.outdir,
                        f"{tag}_{station}_valid_fraction_vs_{var.split('_')[-1]}",
                        f"{label}: {station} valid fraction vs {axis}",
                        axis,
                        nbins,
                        xmin,
                        xmax
                    )

    print(f"\nSaved plots in: {args.outdir}")
    f.Close()

if __name__ == "__main__":
    main()
