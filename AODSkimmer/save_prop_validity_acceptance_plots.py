#!/usr/bin/env python3
import os
import argparse
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetPalette(ROOT.kBird)
ROOT.gStyle.SetTitleFontSize(0.035)


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


def cut_and(*parts):
    parts = [p for p in parts if p and p != "1"]
    return " && ".join(f"({p})" for p in parts) if parts else "1"


def save(c, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    c.SaveAs(os.path.join(outdir, name + ".png"))
    c.SaveAs(os.path.join(outdir, name + ".pdf"))


def draw_overlay_counts(t, var, valid_cut, invalid_cut, outdir, name, title, x_title, nbins, xmin, xmax):
    c = ROOT.TCanvas("c_" + name, title, 900, 700)
    c.SetLeftMargin(0.13)
    c.SetBottomMargin(0.12)

    h_valid = ROOT.TH1F("h_valid_" + name, "", nbins, xmin, xmax)
    h_invalid = ROOT.TH1F("h_invalid_" + name, "", nbins, xmin, xmax)
    h_valid.SetDirectory(0)
    h_invalid.SetDirectory(0)

    t.Draw(f"{var}>>h_valid_{name}", valid_cut, "goff")
    t.Draw(f"{var}>>h_invalid_{name}", invalid_cut, "goff")

    h_valid.SetLineColor(ROOT.kBlue + 1)
    h_invalid.SetLineColor(ROOT.kRed + 1)
    h_valid.SetLineWidth(3)
    h_invalid.SetLineWidth(3)

    ymax = max(h_valid.GetMaximum(), h_invalid.GetMaximum())
    h_valid.SetMaximum(1.25 * ymax if ymax > 0 else 1.0)

    h_valid.SetTitle(title)
    h_valid.GetXaxis().SetTitle(x_title)
    h_valid.GetYaxis().SetTitle("Events")
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


def draw_efficiency(t, var, valid_cut, total_cut, outdir, name, title, x_title, nbins, xmin, xmax):
    c = ROOT.TCanvas("c_" + name, title, 900, 700)
    c.SetLeftMargin(0.13)
    c.SetBottomMargin(0.12)

    h_total = ROOT.TH1F("h_total_" + name, "", nbins, xmin, xmax)
    h_pass = ROOT.TH1F("h_pass_" + name, "", nbins, xmin, xmax)
    h_total.SetDirectory(0)
    h_pass.SetDirectory(0)

    t.Draw(f"{var}>>h_total_{name}", total_cut, "goff")
    t.Draw(f"{var}>>h_pass_{name}", valid_cut, "goff")

    if h_total.Integral() == 0:
        print(f"Skipping {name}: zero total entries")
        return

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


def draw_2d(t, xvar, yvar, cut, outdir, name, title, x_title, y_title, xbins, xmin, xmax, ybins, ymin, ymax):
    c = ROOT.TCanvas("c_" + name, title, 950, 750)
    c.SetRightMargin(0.15)
    c.SetLeftMargin(0.12)
    c.SetBottomMargin(0.12)

    hname = "h_" + name
    expr = f"{yvar}:{xvar}>>{hname}({xbins},{xmin},{xmax},{ybins},{ymin},{ymax})"
    t.Draw(expr, cut, "COLZ")
    h = ROOT.gDirectory.Get(hname)
    if h:
        h.SetDirectory(0)
        h.SetTitle(title)
        h.GetXaxis().SetTitle(x_title)
        h.GetYaxis().SetTitle(y_title)
        h.GetZaxis().SetTitle("Events")
        h.GetXaxis().SetTitleSize(0.045)
        h.GetYaxis().SetTitleSize(0.045)
        h.GetZaxis().SetTitleSize(0.040)
        h.GetXaxis().SetLabelSize(0.040)
        h.GetYaxis().SetLabelSize(0.040)
        h.GetZaxis().SetLabelSize(0.035)
        h.Draw("COLZ")
        c.Update()

    save(c, outdir, name)


def make_plots_for_region(t, root_outdir, region_label, region_title, cut_fn, pt_min, pt_max):
    outdir = os.path.join(root_outdir, region_label)
    os.makedirs(outdir, exist_ok=True)

    particles = [
        ("GenSigMuon", "signal muon"),
        ("GenSigAntiMuon", "signal anti-muon"),
    ]
    stations = [
        ("propSt1Valid", "St1"),
        ("propSt2Valid", "St2"),
    ]

    summary_path = os.path.join(outdir, "summary_counts.txt")
    with open(summary_path, "w") as sf:
        sf.write(f"Propagation validity summary: {region_label}\n")
        sf.write("=" * 70 + "\n\n")

        for tag, label in particles:
            pt = f"{tag}_pt"
            eta = f"{tag}_eta"
            if not has_branch(t, pt) or not has_branch(t, eta):
                print(f"Missing {pt} or {eta}, skipping {tag}")
                continue

            region_cut = cut_fn(tag)
            n_region = int(t.GetEntries(region_cut))
            sf.write(f"{tag} ({label})\n")
            sf.write("-" * 70 + "\n")
            sf.write(f"Region cut: {region_cut}\n")
            sf.write(f"Total in region: {n_region}\n\n")

            for valid_branch, station in stations:
                valid = f"{tag}_{valid_branch}"
                if not has_branch(t, valid):
                    print(f"Missing {valid}, skipping")
                    continue

                total_cut = region_cut
                valid_cut = cut_and(region_cut, f"{valid}==1")
                invalid_cut = cut_and(region_cut, f"{valid}==0")

                n_total = int(t.GetEntries(total_cut))
                n_valid = int(t.GetEntries(valid_cut))
                n_invalid = int(t.GetEntries(invalid_cut))
                frac = 100.0 * n_valid / n_total if n_total else 0.0

                print(f"\n{region_label} | {tag} {station}")
                print(f"  total:   {n_total}")
                print(f"  valid:   {n_valid}")
                print(f"  invalid: {n_invalid}")
                print(f"  valid fraction: {frac:.3f}%")

                sf.write(f"{station:>4s}: total={n_total:8d}  valid={n_valid:8d}  invalid={n_invalid:8d}  valid fraction={frac:7.3f}%\n")

                # 1D raw-count overlays.
                draw_overlay_counts(
                    t, pt, valid_cut, invalid_cut, outdir,
                    f"{tag}_{station}_{region_label}_valid_vs_invalid_pt_counts",
                    f"{label}: {station} propagation validity vs p_{{T}}; {region_title}",
                    f"{tag} p_{{T}} [GeV]", 70, pt_min, pt_max
                )
                draw_overlay_counts(
                    t, eta, valid_cut, invalid_cut, outdir,
                    f"{tag}_{station}_{region_label}_valid_vs_invalid_eta_counts",
                    f"{label}: {station} propagation validity vs #eta; {region_title}",
                    f"{tag} #eta", 48, -2.4, 2.4
                )

                # Valid fractions.
                draw_efficiency(
                    t, pt, valid_cut, total_cut, outdir,
                    f"{tag}_{station}_{region_label}_valid_fraction_vs_pt",
                    f"{label}: {station} valid fraction vs p_{{T}}; {region_title}",
                    f"{tag} p_{{T}} [GeV]", 35, pt_min, pt_max
                )
                draw_efficiency(
                    t, eta, valid_cut, total_cut, outdir,
                    f"{tag}_{station}_{region_label}_valid_fraction_vs_eta",
                    f"{label}: {station} valid fraction vs #eta; {region_title}",
                    f"{tag} #eta", 48, -2.4, 2.4
                )

                # 2D eta-vs-pt valid/invalid maps.
                draw_2d(
                    t, pt, eta, valid_cut, outdir,
                    f"{tag}_{station}_{region_label}_valid_eta_vs_pt",
                    f"{label}: valid {station} propagation, #eta vs p_{{T}}; {region_title}",
                    f"{tag} p_{{T}} [GeV]", f"{tag} #eta",
                    60, pt_min, pt_max, 48, -2.4, 2.4
                )
                draw_2d(
                    t, pt, eta, invalid_cut, outdir,
                    f"{tag}_{station}_{region_label}_invalid_eta_vs_pt",
                    f"{label}: invalid {station} propagation, #eta vs p_{{T}}; {region_title}",
                    f"{tag} p_{{T}} [GeV]", f"{tag} #eta",
                    60, pt_min, pt_max, 48, -2.4, 2.4
                )

                # Optional displaced variables, raw-count overlays + valid fraction.
                for var_suffix, axis, xmin, xmax, nbins in [
                    ("vxy", f"{tag} v_{{xy}} [cm]", 0, 800, 80),
                    ("vz", f"{tag} v_{{z}} [cm]", -1000, 1000, 80),
                ]:
                    var = f"{tag}_{var_suffix}"
                    if not has_branch(t, var):
                        continue
                    draw_overlay_counts(
                        t, var, valid_cut, invalid_cut, outdir,
                        f"{tag}_{station}_{region_label}_valid_vs_invalid_{var_suffix}_counts",
                        f"{label}: {station} propagation validity vs {axis}; {region_title}",
                        axis, nbins, xmin, xmax
                    )
                    draw_efficiency(
                        t, var, valid_cut, total_cut, outdir,
                        f"{tag}_{station}_{region_label}_valid_fraction_vs_{var_suffix}",
                        f"{label}: {station} valid fraction vs {axis}; {region_title}",
                        axis, nbins, xmin, xmax
                    )

            sf.write("\n")

    print("Wrote:", summary_path)


def main():
    parser = argparse.ArgumentParser(
        description="Save propagation-validity plots inside acceptance regions with raw event counts and no histogram normalization."
    )
    parser.add_argument("root_file", nargs="?", default="test_genMuonPropagation_10000Events.root")
    parser.add_argument("-o", "--outdir", default="propagation_validity_acceptance_plots")
    parser.add_argument("-t", "--tree", default="ntuples/outT")
    parser.add_argument("--ptmax", type=float, default=40.0)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    f, t = get_tree(args.root_file, args.tree)

    print(f"Opened:  {args.root_file}")
    print(f"Tree:    {args.tree}")
    print(f"Entries: {int(t.GetEntries())}")
    print(f"Output:  {args.outdir}")

    regions = [
        (
            "eta2p4",
            "|#eta| < 2.4",
            lambda tag: f"abs({tag}_eta)<2.4",
            0.0,
            args.ptmax,
        ),
        (
            "eta2p4_pt5",
            "|#eta| < 2.4, p_{T} > 5 GeV",
            lambda tag: f"abs({tag}_eta)<2.4 && {tag}_pt>5",
            5.0,
            args.ptmax,
        ),
    ]

    for region_label, region_title, cut_fn, pt_min, pt_max in regions:
        make_plots_for_region(t, args.outdir, region_label, region_title, cut_fn, pt_min, pt_max)

    f.Close()
    print("\nSaved raw-count PNG/PDF plots in:")
    print("  " + args.outdir)


if __name__ == "__main__":
    main()
