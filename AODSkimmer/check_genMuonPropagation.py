#!/usr/bin/env python3
import argparse
import json
import os
from collections import OrderedDict

import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetPalette(ROOT.kBird)
ROOT.gStyle.SetTitleFontSize(0.035)


def get_tree(root_file, tree_path):
    f = ROOT.TFile.Open(root_file)
    if not f or f.IsZombie():
        raise RuntimeError(f"Could not open ROOT file: {root_file}")

    t = f.Get(tree_path)
    if not t:
        t = f.Get("ntuples/outT")

    if not t:
        print("Available top-level keys:")
        f.ls()
        raise RuntimeError(f"Could not find TTree at '{tree_path}' or 'ntuples/outT'.")

    return f, t


def branch_exists(t, name):
    return bool(t.GetBranch(name))


def count(t, cut):
    return int(t.GetEntries(cut))


def pct(num, den):
    return 100.0 * float(num) / float(den) if den else 0.0


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_canvas(canvas, outdir, name):
    canvas.SaveAs(os.path.join(outdir, name + ".png"))


def draw_2d(t, x, y, cut, outdir, name, title, x_title, y_title,
            x_bins=60, x_min=0.0, x_max=60.0,
            y_bins=80, y_min=0.0, y_max=5.0):
    c = ROOT.TCanvas("c_" + name, title, 950, 750)
    c.SetRightMargin(0.15)
    c.SetLeftMargin(0.12)
    c.SetBottomMargin(0.12)

    hname = "h_" + name
    expr = f"{y}:{x}>>{hname}({x_bins},{x_min},{x_max},{y_bins},{y_min},{y_max})"
    n = t.Draw(expr, cut, "COLZ")
    h = ROOT.gDirectory.Get(hname)

    if h:
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

    save_canvas(c, outdir, name)
    return n


def draw_1d_overlay(t, branches, labels, cuts, outdir, name, title,
                    x_title, x_bins=80, x_min=0.0, x_max=1.0, normalize=True):
    c = ROOT.TCanvas("c_" + name, title, 950, 750)
    c.SetLeftMargin(0.12)
    c.SetBottomMargin(0.12)

    colors = [ROOT.kBlack, ROOT.kBlue + 1, ROOT.kRed + 1, ROOT.kGreen + 2]
    hists = []
    ymax = 0.0

    for i, (br, label, cut) in enumerate(zip(branches, labels, cuts)):
        hname = f"h_{name}_{i}"
        expr = f"{br}>>{hname}({x_bins},{x_min},{x_max})"
        t.Draw(expr, cut, "goff")
        h = ROOT.gDirectory.Get(hname)
        if not h:
            continue

        h.SetDirectory(0)
        h.SetLineColor(colors[i % len(colors)])
        h.SetLineWidth(3)
        h.SetMarkerColor(colors[i % len(colors)])

        if normalize and h.Integral() > 0:
            h.Scale(1.0 / h.Integral())

        ymax = max(ymax, h.GetMaximum())
        hists.append((h, label))

    if not hists:
        return

    leg = ROOT.TLegend(0.58, 0.68, 0.88, 0.88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.035)

    for i, (h, label) in enumerate(hists):
        h.SetTitle(title)
        h.GetXaxis().SetTitle(x_title)
        h.GetYaxis().SetTitle("Normalized events" if normalize else "Events")
        h.GetXaxis().SetTitleSize(0.045)
        h.GetYaxis().SetTitleSize(0.045)
        h.GetXaxis().SetLabelSize(0.040)
        h.GetYaxis().SetLabelSize(0.040)
        h.SetMaximum(1.25 * ymax if ymax > 0 else 1.0)
        h.Draw("HIST" if i == 0 else "HIST SAME")
        leg.AddEntry(h, label, "l")

    leg.Draw()
    c.Update()
    save_canvas(c, outdir, name)


def draw_bar_chart(values, outdir, name, title, y_title):
    labels = list(values.keys())
    nums = list(values.values())
    n = len(nums)

    c = ROOT.TCanvas("c_" + name, title, 900, 700)
    c.SetLeftMargin(0.12)
    c.SetBottomMargin(0.18)

    h = ROOT.TH1F("h_" + name, title, n, 0, n)
    h.SetDirectory(0)
    for i, (lab, val) in enumerate(zip(labels, nums), start=1):
        h.SetBinContent(i, val)
        h.GetXaxis().SetBinLabel(i, lab)

    h.GetYaxis().SetTitle(y_title)
    h.GetYaxis().SetTitleSize(0.045)
    h.GetYaxis().SetLabelSize(0.040)
    h.GetXaxis().SetLabelSize(0.032)
    h.LabelsOption("v", "X")
    h.SetMaximum(1.25 * max(nums) if nums else 1.0)
    h.Draw("BAR")
    c.Update()
    save_canvas(c, outdir, name)


def make_summary_for_particle(t, tag, threshold):
    old = f"{tag}_minDrToDSAMuon"
    st1 = f"{tag}_minDrToDSAMuonPropSt1"
    st2 = f"{tag}_minDrToDSAMuonPropSt2"
    v1 = f"{tag}_propSt1Valid"
    v2 = f"{tag}_propSt2Valid"

    total = int(t.GetEntries())

    finite_old = count(t, f"{old}<999")
    valid_st1 = count(t, f"{v1}==1")
    valid_st2 = count(t, f"{v2}==1")
    finite_st1 = count(t, f"{st1}<999")
    finite_st2 = count(t, f"{st2}<999")

    pass_old_total = count(t, f"{old}<{threshold}")
    pass_st1_total = count(t, f"{st1}<{threshold}")
    pass_st2_total = count(t, f"{st2}<{threshold}")

    paired_st1 = count(t, f"{old}<999 && {st1}<999")
    paired_st2 = count(t, f"{old}<999 && {st2}<999")

    pass_old_paired_st1 = count(t, f"{old}<{threshold} && {st1}<999")
    pass_st1_paired = count(t, f"{st1}<{threshold} && {old}<999")

    pass_old_paired_st2 = count(t, f"{old}<{threshold} && {st2}<999")
    pass_st2_paired = count(t, f"{st2}<{threshold} && {old}<999")

    improve_st1 = count(t, f"{st1} < {old} && {st1}<999 && {old}<999")
    improve_st2 = count(t, f"{st2} < {old} && {st2}<999 && {old}<999")

    worse_st1 = count(t, f"{st1} > {old} && {st1}<999 && {old}<999")
    worse_st2 = count(t, f"{st2} > {old} && {st2}<999 && {old}<999")

    equal_st1 = paired_st1 - improve_st1 - worse_st1
    equal_st2 = paired_st2 - improve_st2 - worse_st2

    no_dsa = count(t, "nDSAMuon==0")
    dsa_but_no_st1_match = count(t, f"nDSAMuon>0 && {st1}>=999")
    dsa_but_no_st2_match = count(t, f"nDSAMuon>0 && {st2}>=999")

    out = OrderedDict()
    out["total_entries"] = total
    out["finite_old"] = finite_old
    out["valid_st1"] = valid_st1
    out["valid_st2"] = valid_st2
    out["finite_st1"] = finite_st1
    out["finite_st2"] = finite_st2
    out[f"pass_old_dr_lt_{threshold}_total"] = pass_old_total
    out[f"pass_st1_dr_lt_{threshold}_total"] = pass_st1_total
    out[f"pass_st2_dr_lt_{threshold}_total"] = pass_st2_total
    out["paired_finite_st1"] = paired_st1
    out["paired_finite_st2"] = paired_st2
    out[f"pass_old_dr_lt_{threshold}_paired_st1"] = pass_old_paired_st1
    out[f"pass_st1_dr_lt_{threshold}_paired"] = pass_st1_paired
    out[f"pass_old_dr_lt_{threshold}_paired_st2"] = pass_old_paired_st2
    out[f"pass_st2_dr_lt_{threshold}_paired"] = pass_st2_paired
    out["improved_st1_vs_old"] = improve_st1
    out["improved_st2_vs_old"] = improve_st2
    out["worse_st1_vs_old"] = worse_st1
    out["worse_st2_vs_old"] = worse_st2
    out["equal_st1_vs_old"] = equal_st1
    out["equal_st2_vs_old"] = equal_st2
    out["no_dsa_events"] = no_dsa
    out["nDSAMuon_gt0_but_no_st1_match"] = dsa_but_no_st1_match
    out["nDSAMuon_gt0_but_no_st2_match"] = dsa_but_no_st2_match

    rates = OrderedDict()
    rates["finite_old_over_total_percent"] = pct(finite_old, total)
    rates["valid_st1_over_total_percent"] = pct(valid_st1, total)
    rates["valid_st2_over_total_percent"] = pct(valid_st2, total)
    rates["finite_st1_over_total_percent"] = pct(finite_st1, total)
    rates["finite_st2_over_total_percent"] = pct(finite_st2, total)
    rates["old_pass_over_finite_old_percent"] = pct(pass_old_total, finite_old)
    rates["st1_pass_over_finite_st1_percent"] = pct(pass_st1_total, finite_st1)
    rates["st2_pass_over_finite_st2_percent"] = pct(pass_st2_total, finite_st2)
    rates["old_pass_over_paired_st1_percent"] = pct(pass_old_paired_st1, paired_st1)
    rates["st1_pass_over_paired_st1_percent"] = pct(pass_st1_paired, paired_st1)
    rates["old_pass_over_paired_st2_percent"] = pct(pass_old_paired_st2, paired_st2)
    rates["st2_pass_over_paired_st2_percent"] = pct(pass_st2_paired, paired_st2)
    rates["improved_st1_over_paired_percent"] = pct(improve_st1, paired_st1)
    rates["improved_st2_over_paired_percent"] = pct(improve_st2, paired_st2)

    return out, rates


def write_summary_txt(path, summaries, rates, threshold):
    with open(path, "w") as f:
        f.write("Gen-muon propagation validation summary\n")
        f.write("=" * 45 + "\n\n")
        f.write(f"Threshold: dR < {threshold}\n\n")
        for tag in summaries:
            f.write(f"{tag}\n")
            f.write("-" * len(tag) + "\n")
            for k, v in summaries[tag].items():
                f.write(f"{k:45s} {v}\n")
            f.write("\nRates\n")
            for k, v in rates[tag].items():
                f.write(f"{k:45s} {v:.3f}%\n")
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Validate propagated gen-to-DSA matching branches and save PNG plots."
    )
    parser.add_argument(
        "root_file",
        nargs="?",
        default="test_genMuonPropagation_10000Events.root",
        help="Input ROOT file."
    )
    parser.add_argument("-t", "--tree", default="ntuples/outT", help="Tree path.")
    parser.add_argument("-o", "--outdir", default="genMuonPropagation_checks", help="Output directory.")
    parser.add_argument("--threshold", type=float, default=0.15, help="DSA matching dR threshold.")
    parser.add_argument("--ptmax", type=float, default=60.0, help="Maximum pT shown in 2D plots.")
    parser.add_argument("--drmax", type=float, default=5.0, help="Maximum dR shown in broad 2D plots.")
    args = parser.parse_args()

    ensure_dir(args.outdir)
    f, t = get_tree(args.root_file, args.tree)

    required = [
        "nDSAMuon",
        "GenSigMuon_pt",
        "GenSigAntiMuon_pt",
        "GenSigMuon_propSt1Valid",
        "GenSigMuon_propSt2Valid",
        "GenSigAntiMuon_propSt1Valid",
        "GenSigAntiMuon_propSt2Valid",
        "GenSigMuon_minDrToDSAMuon",
        "GenSigMuon_minDrToDSAMuonPropSt1",
        "GenSigMuon_minDrToDSAMuonPropSt2",
        "GenSigAntiMuon_minDrToDSAMuon",
        "GenSigAntiMuon_minDrToDSAMuonPropSt1",
        "GenSigAntiMuon_minDrToDSAMuonPropSt2",
        "recoDSAMuonPropSt1Valid",
        "recoDSAMuonPropSt2Valid",
    ]

    missing = [b for b in required if not branch_exists(t, b)]
    if missing:
        print("Missing required branches:")
        for b in missing:
            print("  " + b)
        raise RuntimeError("Missing required branches.")

    print(f"Opened {args.root_file}")
    print(f"Entries: {int(t.GetEntries())}")
    print(f"Output directory: {args.outdir}")

    summaries = OrderedDict()
    rates = OrderedDict()
    for tag in ["GenSigMuon", "GenSigAntiMuon"]:
        summaries[tag], rates[tag] = make_summary_for_particle(t, tag, args.threshold)

    txt_path = os.path.join(args.outdir, "summary.txt")
    json_path = os.path.join(args.outdir, "summary.json")
    write_summary_txt(txt_path, summaries, rates, args.threshold)
    with open(json_path, "w") as jf:
        json.dump({"summary": summaries, "rates_percent": rates}, jf, indent=2)

    for tag in ["GenSigMuon", "GenSigAntiMuon"]:
        pt = f"{tag}_pt"
        old = f"{tag}_minDrToDSAMuon"
        st1 = f"{tag}_minDrToDSAMuonPropSt1"
        st2 = f"{tag}_minDrToDSAMuonPropSt2"
        pretty = "signal muon" if tag == "GenSigMuon" else "signal anti-muon"

        draw_2d(t, pt, old, f"{old}<10", args.outdir, f"{tag}_old_minDr_vs_pt_broad",
                f"{pretty}: old unpropagated min #DeltaR to DSA vs p_{{T}}",
                f"{tag} p_{{T}} [GeV]", "Old min #DeltaR to DSA",
                x_max=args.ptmax, y_max=args.drmax)
        draw_2d(t, pt, st1, f"{st1}<10", args.outdir, f"{tag}_propSt1_minDr_vs_pt_broad",
                f"{pretty}: propagated St1 min #DeltaR to DSA vs p_{{T}}",
                f"{tag} p_{{T}} [GeV]", "Propagated St1 min #DeltaR to DSA",
                x_max=args.ptmax, y_max=args.drmax)
        draw_2d(t, pt, st2, f"{st2}<10", args.outdir, f"{tag}_propSt2_minDr_vs_pt_broad",
                f"{pretty}: propagated St2 min #DeltaR to DSA vs p_{{T}}",
                f"{tag} p_{{T}} [GeV]", "Propagated St2 min #DeltaR to DSA",
                x_max=args.ptmax, y_max=args.drmax)

        draw_2d(t, pt, old, f"{old}<0.5", args.outdir, f"{tag}_old_minDr_vs_pt_zoom",
                f"{pretty}: old min #DeltaR to DSA vs p_{{T}}; #DeltaR < 0.5",
                f"{tag} p_{{T}} [GeV]", "Old min #DeltaR to DSA",
                x_max=args.ptmax, y_max=0.5)
        draw_2d(t, pt, st1, f"{st1}<0.5", args.outdir, f"{tag}_propSt1_minDr_vs_pt_zoom",
                f"{pretty}: propagated St1 min #DeltaR to DSA vs p_{{T}}; #DeltaR < 0.5",
                f"{tag} p_{{T}} [GeV]", "Propagated St1 min #DeltaR to DSA",
                x_max=args.ptmax, y_max=0.5)
        draw_2d(t, pt, st2, f"{st2}<0.5", args.outdir, f"{tag}_propSt2_minDr_vs_pt_zoom",
                f"{pretty}: propagated St2 min #DeltaR to DSA vs p_{{T}}; #DeltaR < 0.5",
                f"{tag} p_{{T}} [GeV]", "Propagated St2 min #DeltaR to DSA",
                x_max=args.ptmax, y_max=0.5)

        draw_1d_overlay(t, [old, st1, st2], ["Old", "Prop St1", "Prop St2"],
                        [f"{old}<999", f"{st1}<999", f"{st2}<999"],
                        args.outdir, f"{tag}_minDr_overlay_zoom",
                        f"{pretty}: min #DeltaR to DSA, finite matches",
                        "min #DeltaR to DSA", x_bins=80, x_min=0.0, x_max=0.5, normalize=True)

        draw_1d_overlay(t, [old, st1, st2], ["Old", "Prop St1", "Prop St2"],
                        [f"{old}<10", f"{st1}<10", f"{st2}<10"],
                        args.outdir, f"{tag}_minDr_overlay_broad",
                        f"{pretty}: min #DeltaR to DSA, broad view",
                        "min #DeltaR to DSA", x_bins=100, x_min=0.0, x_max=5.0, normalize=True)

        draw_1d_overlay(t, [f"{old}-{st1}", f"{old}-{st2}"],
                        ["Old - Prop St1", "Old - Prop St2"],
                        [f"{old}<999 && {st1}<999", f"{old}<999 && {st2}<999"],
                        args.outdir, f"{tag}_old_minus_prop_deltaR",
                        f"{pretty}: event-by-event #DeltaR improvement",
                        "Old min #DeltaR - propagated min #DeltaR",
                        x_bins=100, x_min=-0.5, x_max=5.0, normalize=False)

    for tag in ["GenSigMuon", "GenSigAntiMuon"]:
        s = summaries[tag]
        values_st1 = OrderedDict([
            ("Old, paired St1", s[f"pass_old_dr_lt_{args.threshold}_paired_st1"]),
            ("Prop St1", s[f"pass_st1_dr_lt_{args.threshold}_paired"]),
        ])
        draw_bar_chart(values_st1, args.outdir, f"{tag}_paired_threshold_pass_counts_st1",
                       f"{tag}: #DeltaR < {args.threshold} on paired finite St1 events",
                       "Passing events")

        values_st2 = OrderedDict([
            ("Old, paired St2", s[f"pass_old_dr_lt_{args.threshold}_paired_st2"]),
            ("Prop St2", s[f"pass_st2_dr_lt_{args.threshold}_paired"]),
        ])
        draw_bar_chart(values_st2, args.outdir, f"{tag}_paired_threshold_pass_counts_st2",
                       f"{tag}: #DeltaR < {args.threshold} on paired finite St2 events",
                       "Passing events")

    print("\nKey results:")
    for tag in ["GenSigMuon", "GenSigAntiMuon"]:
        s = summaries[tag]
        r = rates[tag]
        print(f"\n{tag}")
        print(f"  entries: {s['total_entries']}")
        print(f"  finite old: {s['finite_old']}")
        print(f"  finite St1: {s['finite_st1']}")
        print(f"  finite St2: {s['finite_st2']}")
        print(f"  paired finite St1: {s['paired_finite_st1']}")
        print(f"  paired finite St2: {s['paired_finite_st2']}")
        print(
            f"  old vs St1 pass on paired denominator: "
            f"{s[f'pass_old_dr_lt_{args.threshold}_paired_st1']} / {s['paired_finite_st1']} "
            f"({r['old_pass_over_paired_st1_percent']:.2f}%) -> "
            f"{s[f'pass_st1_dr_lt_{args.threshold}_paired']} / {s['paired_finite_st1']} "
            f"({r['st1_pass_over_paired_st1_percent']:.2f}%)"
        )
        print(
            f"  old vs St2 pass on paired denominator: "
            f"{s[f'pass_old_dr_lt_{args.threshold}_paired_st2']} / {s['paired_finite_st2']} "
            f"({r['old_pass_over_paired_st2_percent']:.2f}%) -> "
            f"{s[f'pass_st2_dr_lt_{args.threshold}_paired']} / {s['paired_finite_st2']} "
            f"({r['st2_pass_over_paired_st2_percent']:.2f}%)"
        )
        print(
            f"  improved St1 vs old: {s['improved_st1_vs_old']} / {s['paired_finite_st1']} "
            f"({r['improved_st1_over_paired_percent']:.2f}%)"
        )
        print(
            f"  improved St2 vs old: {s['improved_st2_vs_old']} / {s['paired_finite_st2']} "
            f"({r['improved_st2_over_paired_percent']:.2f}%)"
        )

    print("\nSaved PNG plots and summaries in:")
    print("  " + args.outdir)
    f.Close()


if __name__ == "__main__":
    main()
