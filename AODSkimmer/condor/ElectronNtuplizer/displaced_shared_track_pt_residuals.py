#!/usr/bin/env python3
"""pT residuals for gen muons matched by BOTH displacedMuon inner and global tracks.

    python3 displaced_shared_track_pt_residuals.py --input displaced_shared_tracks.root

Input: schema_version=2, reco_tracks TTree from DisplacedMuonDoubleCounting.cc.
       ALL displacedMuons are exported; duplicate-track checks are independent.
Gen/track acceptance: finite pt > 0, |eta| < 2.4, 0 <= vxy < 740 cm,
                     |vz| < 960 cm, valid finite Station-2 propagation.
Shared = the SAME accepted gen muon has nearest same-charge inner AND global matches
         at Station-2 deltaR <= 0.06. The two matches may be different reco objects.
         Each accepted gen contributes once; reco reuse is allowed, as in the original.
Inner/global residuals use those two matches on the full shared-gen cohort.
Always-global vs BestTrack uses the global-matched object's globalTrack and
muonBestTrack(), holding the reco object and gen identity fixed. The common policy
cohort additionally requires accepted, same-charge BestTrack; no extra BestTrack dR cut.
BestTrack choice fractions use ALL shared-gen matches, including other/unavailable
choices, classified by TrackRef identity on the global-matched reco object.
Duplicates independently compare exact non-null inner/outer/global TrackRefs.
Output: PNGs with outside legends; ROOT/JSON histograms, counts and curves.
68/90/99% containment = quantiles of ABSOLUTE signed/relative residuals;
minimum 10 shared-gen matches per gen-pT bin. Empty displays are explicitly labelled.
"""
import argparse
import math
import json
import os
import re
import subprocess
from collections import defaultdict
from urllib.parse import urlparse

import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.TH1.AddDirectory(False)

DEFAULT_INPUT = "displaced_shared_tracks.root"
TRACKS = {
    "inner": ("Inner track", ROOT.kBlue + 1),
    "global": ("Global track", ROOT.kRed + 1),
    "global_priority": ("Always global", ROOT.kOrange + 7),
    "best": ("BestTrack", ROOT.kGreen + 2),
}
COMPARISONS = {
    "inner_global": ("Inner vs global", ("inner", "global")),
    "priority_best": ("Always global vs BestTrack", ("global_priority", "best")),
}
CHOICES = {
    0: ("Inner", ROOT.kBlue + 1),
    1: ("Global", ROOT.kRed + 1),
    2: ("Other", ROOT.kViolet + 1),
    3: ("Unavailable", ROOT.kGray + 2),
}
GEN_FIELDS = ("valid", "pt", "eta", "charge", "vxy", "vz",
              "propValid", "st2Eta", "st2Phi")
TRACK_FIELDS = ("present", "valid", "pt", "eta", "charge", "vxy", "vz",
                "propValid", "st2Eta", "st2Phi")
# metric: (nbins, low, high, y title)
METRICS = {
    "res": (200, -50., 50., "p_{T}^{reco} - p_{T}^{gen} [GeV]"),
    "rel_res": (200, -2., 3., "(p_{T}^{reco} - p_{T}^{gen}) / p_{T}^{gen}"),
}
RECO_PT_INTERVALS = ((0., 10.), (20., 30.), (40., 50.), (50., 60.))
PT_BINS = (25, 0., 50.)
INVALID_COORD = 900.


def find_files(spec):
    if spec.startswith("root://"):
        p = urlparse(spec)
        host = f"root://{p.netloc}"
        if spec.endswith(".root"):
            return [spec]
        out = subprocess.run(["xrdfs", host, "ls", "-R", p.path],
                             check=True, text=True, capture_output=True).stdout
        return sorted(f"{host}//{l.strip().lstrip('/')}"
                      for l in out.splitlines() if l.strip().endswith(".root"))
    if os.path.isdir(spec):
        return sorted(os.path.join(r, n) for r, _, ns in os.walk(spec)
                      for n in ns if n.endswith(".root"))
    return [spec]


def finite_coord(x):
    return math.isfinite(x) and abs(x) < INVALID_COORD


def delta_r(eta1, phi1, eta2, phi2):
    dphi = math.atan2(math.sin(phi1 - phi2), math.cos(phi1 - phi2))
    return math.hypot(eta1 - eta2, dphi)


def gen_muon(tree, index, args):
    """Return (pt, charge, st2_eta, st2_phi) or None."""
    g = lambda f: tree[f"gen_{f}"][index]
    if not bool(g("valid")) or not bool(g("propValid")):
        return None
    pt, eta, vxy, vz = (float(g(f)) for f in ("pt", "eta", "vxy", "vz"))
    charge = int(g("charge"))
    seta, sphi = float(g("st2Eta")), float(g("st2Phi"))
    if not (math.isfinite(pt) and pt > 0. and abs(eta) < args.eta_max
            and 0. <= vxy < args.vxy_max and abs(vz) < args.vz_max):
        return None
    if charge not in (-1, 1) or not (finite_coord(seta) and finite_coord(sphi)):
        return None
    return pt, charge, seta, sphi


def track_candidate(tree, prefix, args):
    """Return (pt, charge, st2_eta, st2_phi) for this object's track, or None."""
    v = {f: tree[f"{prefix}_{f}"] for f in TRACK_FIELDS}
    if not bool(v["valid"]) or not bool(v["propValid"]):
        return None
    pt, eta, vxy, vz = (float(v[f]) for f in ("pt", "eta", "vxy", "vz"))
    charge = int(v["charge"])
    seta, sphi = float(v["st2Eta"]), float(v["st2Phi"])
    if not (math.isfinite(pt) and pt > 0. and abs(eta) < args.eta_max
            and 0. <= vxy < args.vxy_max and abs(vz) < args.vz_max):
        return None
    if charge not in (-1, 1) or not (finite_coord(seta) and finite_coord(sphi)):
        return None
    return pt, charge, seta, sphi


def read_events(tree):
    """Group consecutive all-reco rows by event; copy ROOT proxies before GetEntry."""
    rows, previous = [], None
    for i in range(tree.GetEntries()):
        tree.GetEntry(i)
        key = (int(tree.run), int(tree.lumi), int(tree.event))
        if previous is not None and key != previous:
            yield rows
            rows = []
        row = {"best_choice": int(tree.best_choice), "muon_index": int(tree.muon_index)}
        for prefix in ("inner", "global", "best"):
            for field in TRACK_FIELDS:
                value = getattr(tree, f"{prefix}_{field}")
                cast = bool if field in ("present", "valid", "propValid") else int if field == "charge" else float
                row[f"{prefix}_{field}"] = cast(value)
        for field in GEN_FIELDS:
            cast = bool if field in ("valid", "propValid") else int if field == "charge" else float
            row[f"gen_{field}"] = [cast(getattr(tree, f"gen_{field}")[j]) for j in range(2)]
        rows.append(row)
        previous = key
    if rows:
        yield rows


def closest_match(gen, candidates, dr_max):
    """Original nearest same-charge St2 matching; equal-distance ties use the last row."""
    best, best_dr = None, dr_max
    for row, track in candidates:
        if track is None or track[1] != gen[1]:
            continue
        dr = delta_r(gen[2], gen[3], track[2], track[3])
        if dr <= best_dr:
            best, best_dr = (row, track), dr
    return best


def no_data(text):
    """Keep missing statistics visible instead of presenting an unlabelled blank plot."""
    label = ROOT.TLatex()
    label.SetNDC(True)
    label.SetTextSize(0.028)
    label.DrawLatex(0.18, 0.50, text)
    return label


def outside_legend(canvas):
    canvas.SetRightMargin(0.34)
    legend = ROOT.TLegend(0.68, 0.57, 0.99, 0.86)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.026)
    return legend


def quantile(sorted_values, q):
    pos = (len(sorted_values) - 1) * q
    lo = int(pos)
    frac = pos - lo
    if frac == 0.:
        return sorted_values[lo]
    return sorted_values[lo] * (1. - frac) + sorted_values[lo + 1] * frac


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--tree", default="reco_tracks")
    ap.add_argument("--outdir", default="displaced_shared_track_pt_residuals")
    ap.add_argument("--sample-regex", help="regex applied to file paths, e.g. 'ctau-1/'")
    ap.add_argument("--max-events", type=int, default=0, help="events with reco objects per file; 0 = all")
    ap.add_argument("--eta-max", type=float, default=2.4)
    ap.add_argument("--vxy-max", type=float, default=740.)
    ap.add_argument("--vz-max", type=float, default=960.)
    ap.add_argument("--dr-max", type=float, default=0.06)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    files = find_files(args.input)
    if args.sample_regex:
        files = [f for f in files if re.search(args.sample_regex, f)]
    if not files:
        raise SystemExit(f"No ROOT files found under {args.input}")

    hists = {}
    for name, (label, _) in TRACKS.items():
        for metric, (n, lo, hi, ytitle) in METRICS.items():
            hists[name, metric] = ROOT.TH2D(
                f"h_{metric}_vs_genpt_{name}",
                f"Shared-gen {label};gen p_{{T}} [GeV];{ytitle}",
                *PT_BINS, n, lo, hi)
    shared_hists = {}
    for name in TRACKS:
        for metric, (n, lo, hi, title) in METRICS.items():
            for low, high in RECO_PT_INTERVALS:
                shared_hists[name, metric, low, high] = ROOT.TH1D(
                    f"h_shared_{metric}_{name}_recopt_{int(low)}_{int(high)}",
                    f";{title};Fraction of shared gen matches / bin", n, lo, hi)
    values = defaultdict(list)  # (name, metric, ptbin) -> residuals, for quantile curves
    counts = defaultdict(int)
    choice_total = ROOT.TH1D("h_best_choice_counts", ";BestTrack choice;Shared gen matches", 4, -0.5, 3.5)
    choice_den, choice_num = {}, {}
    for variable in ("innerpt", "genpt"):
        xtitle = "Matched inner p_{T} [GeV]" if variable == "innerpt" else "gen p_{T} [GeV]"
        choice_den[variable] = ROOT.TH1D(f"h_best_choice_den_{variable}",
                                        f";{xtitle};Shared gen matches", *PT_BINS)
        for code, (label, _) in CHOICES.items():
            choice_num[variable, code] = ROOT.TH1D(f"h_best_choice_num_{variable}_{code}",
                                                  f"{label};{xtitle};Shared gen matches", *PT_BINS)
    for code, (label, _) in CHOICES.items():
        choice_total.GetXaxis().SetBinLabel(code + 1, label)

    branches = [f"gen_{f}" for f in GEN_FIELDS]
    branches += [f"{t}_{f}" for t in ("inner", "global", "best") for f in TRACK_FIELDS]
    branches += ["best_choice", "run", "lumi", "event", "muon_index"]

    for k, path in enumerate(files, 1):
        src = ROOT.TFile.Open(path)
        if not src or src.IsZombie():
            raise OSError(f"Cannot open {path}")
        version = src.Get("schema_version")
        if not version or int(version.GetVal()) != 2:
            raise RuntimeError(f"{path}: old duplicate-only export; rerun the corrected analyzer on MiniAOD")
        for field in ("events_processed", "duplicate_inner_pairs", "duplicate_outer_pairs",
                      "duplicate_global_pairs", "duplicate_any_pairs"):
            obj = src.Get(field)
            if not obj:
                raise RuntimeError(f"Missing {field} in {path}")
            counts[field] += int(obj.GetVal())
        tree = src.Get(args.tree)
        if not tree:
            raise RuntimeError(f"Missing tree {args.tree} in {path}")
        missing = sorted(set(branches) - {b.GetName() for b in tree.GetListOfBranches()})
        if missing:
            raise RuntimeError(f"Missing branches in {path}: {missing}")
        tree.SetBranchStatus("*", 0)
        for b in branches:
            tree.SetBranchStatus(b + "*", 1)

        events_in_file = 0
        for rows in read_events(tree):
            if args.max_events and events_in_file >= args.max_events:
                break
            events_in_file += 1
            counts["events_with_reco_processed"] += 1
            counts["reco_objects"] += len(rows)
            candidates = {prefix: [(row, track_candidate(row, prefix, args)) for row in rows]
                          for prefix in ("inner", "global")}
            for j in range(2):
                gen = gen_muon(rows[0], j, args)
                if gen is None:
                    continue
                counts["accepted_gen"] += 1
                inner_match = closest_match(gen, candidates["inner"], args.dr_max)
                global_match = closest_match(gen, candidates["global"], args.dr_max)
                if inner_match is not None:
                    counts["inner_matched_gen"] += 1
                if global_match is not None:
                    counts["global_matched_gen"] += 1
                if inner_match is None or global_match is None:
                    continue
                counts["shared_gen"] += 1
                gen_pt = gen[0]
                # Both tracks match the same gen; duplicate-reference status is irrelevant.
                global_row, global_track = global_match
                choice = global_row["best_choice"]
                if choice not in CHOICES:
                    raise RuntimeError(f"Unknown BestTrack choice {choice} in {path}")
                choice_total.Fill(choice)
                counts[f"best_choice_{choice}"] += 1
                for variable, pt in (("innerpt", inner_match[1][0]), ("genpt", gen_pt)):
                    choice_den[variable].Fill(pt)
                    choice_num[variable, choice].Fill(pt)
                best = track_candidate(global_row, "best", args)
                matched_pts = {"inner": inner_match[1][0], "global": global_track[0]}
                if best is not None and best[1] == gen[1]:
                    matched_pts.update(global_priority=global_track[0], best=best[0])
                    counts["cohort_priority_best"] += 1
                else:
                    counts["excluded_priority_best"] += 1
                counts["cohort_inner_global"] += 1
                for name, reco_pt in matched_pts.items():
                    counts[name] += 1
                    res = reco_pt - gen_pt
                    for metric, val in (("res", res), ("rel_res", res / gen_pt)):
                        h = hists[name, metric]
                        h.Fill(gen_pt, val)
                        b = h.GetXaxis().FindBin(gen_pt)
                        if 1 <= b <= h.GetNbinsX():
                            values[name, metric, b].append(val)
                        for low, high in RECO_PT_INTERVALS:
                            if low <= reco_pt < high:
                                shared_hists[name, metric, low, high].Fill(val)
        src.Close()
        print(f"[{k}/{len(files)}] {os.path.basename(path)}  "
              f"events={counts['events_with_reco_processed']}  shared_gen={counts['shared_gen']}  "
              + "  ".join(f"{n}={counts[n]}" for n in TRACKS), flush=True)

    out = ROOT.TFile.Open(os.path.join(args.outdir, "residuals.root"), "RECREATE")
    if not out or out.IsZombie():
        raise OSError(f"Cannot create residuals.root in {args.outdir}")
    out.cd()

    numerical = {"definition": {
        "shared": "Same accepted gen muon independently matched by both innerTrack and globalTrack; NOT shared TrackRefs or duplicate objects",
        "comparisons": "inner_global: all shared gen; priority_best: same global-matched reco object, accepted same-charge BestTrack; count exclusions explicitly",
        "global_priority": "Always use the globalTrack of the global-matched displacedMuon",
        "best": "muonBestTrack() of that SAME global-matched displacedMuon; no extra BestTrack dR cut; not tunePMuonBestTrack()",
        "best_choices": "Reference identity on global-matched reco object: inner=0, global=1, other=2, null/unavailable=3; inner wins if references coincide",
        "choice_denominator": "All shared-gen matches; no BestTrack acceptance or propagation requirement",
        "duplicates": "Independent exact non-null TrackRef equality (product ID and key), for inner, outer and global; pairs i<j; any_pairs is their union",
        "binning": "Each policy uses its own reco pT; lower inclusive, upper exclusive",
        "res": "reco pT - gen pT [GeV] (signed)",
        "rel_res": "(reco pT - gen pT) / gen pT",
        "normalization": "Unweighted gen matches; residual underflow and overflow included in display-fraction denominator",
        "interval_resolution": "68/90/99% quantiles of absolute residuals per gen pT bin; minimum 10 matches; unbinned linear interpolation includes residual tails",
        "matching": "Nearest same-charge track separately for inner/global at St2 deltaR <= dr_max; ties use last reco row as original script; reco reuse allowed",
        "fraction_uncertainty": "No uncertainty assigned; matched objects and gen within an event can be correlated",
        "acceptance": "Gen and tracks: finite pt > 0, |eta| < eta_max, 0 <= vxy < vxy_max, |vz| < vz_max, finite valid St2 propagation",
        "input_events_processed": "Full analyzer event count including zero-reco events; max-events limits only plotter events with reco rows; duplicate counters describe full input"},
        "config": vars(args), "inputs": files, "counts": dict(counts),
        "reco_pt_intervals_GeV": RECO_PT_INTERVALS, "histograms": {}, "graphs": {}}

    def save_hist(h):
        h.Write()
        axes = [h.GetXaxis()] + ([h.GetYaxis()] if h.GetDimension() == 2 else [])
        record = {"edges": [[a.GetBinLowEdge(b) for b in range(1, a.GetNbins() + 2)]
                            for a in axes], "entries": h.GetEntries(),
                  "axis_titles": [a.GetTitle() for a in axes]}
        if h.GetDimension() == 1:
            record["contents"] = [h.GetBinContent(b) for b in range(h.GetNbinsX() + 2)]
            record["errors"] = [h.GetBinError(b) for b in range(h.GetNbinsX() + 2)]
        else:
            record["contents"] = [[h.GetBinContent(x, y) for y in range(h.GetNbinsY() + 2)]
                                  for x in range(h.GetNbinsX() + 2)]
            record["errors"] = [[h.GetBinError(x, y) for y in range(h.GetNbinsY() + 2)]
                                for x in range(h.GetNbinsX() + 2)]
        numerical["histograms"][h.GetName()] = record

    # Residual overlays, one common gen cohort per comparison.
    for comparison, (comparison_title, names) in COMPARISONS.items():
        for metric in METRICS:
            for low, high in RECO_PT_INTERVALS:
                c = ROOT.TCanvas(f"c_shared_{comparison}_{metric}_{int(low)}_{int(high)}", "", 1200, 800)
                c.SetLeftMargin(0.15)
                c.SetBottomMargin(0.13)
                c.SetTopMargin(0.12)
                leg = outside_legend(c)
                displays = []
                for name in names:
                    label, color = TRACKS[name]
                    h = shared_hists[name, metric, low, high]
                    save_hist(h)
                    display = h.Clone(h.GetName() + "_normalized")
                    total = h.Integral(0, h.GetNbinsX() + 1)
                    if total:
                        display.Scale(1. / total)
                    display.GetXaxis().SetRangeUser(
                        -0.5 if metric == "rel_res" else -10.,
                        0.5 if metric == "rel_res" else 10.)
                    display.SetLineColor(color)
                    display.SetLineWidth(2)
                    display.GetXaxis().SetTitleSize(0.045)
                    display.GetYaxis().SetTitleSize(0.04)
                    display.GetYaxis().SetTitleOffset(1.65)
                    save_hist(display)
                    displays.append(display)
                    leg.AddEntry(display, f"{label} (N = {int(total)})", "l")
                maximum = max(h.GetMaximum() for h in displays)
                displays[0].SetMinimum(0.)
                displays[0].SetMaximum(1.3 * maximum if maximum else 1.)
                for i, h in enumerate(displays):
                    h.Draw("HIST" if i == 0 else "HIST SAME")
                line = ROOT.TLine(0., 0., 0., displays[0].GetMaximum())
                line.SetLineStyle(2)
                line.SetLineColor(ROOT.kGray + 2)
                line.Draw()
                empty = no_data("No shared-gen matches in this reco-p_{T} interval") if not any(h.GetEntries() for h in displays) else None
                leg.Draw()
                text = ROOT.TLatex()
                text.SetNDC(True)
                text.SetTextSize(0.028)
                text.DrawLatex(0.15, 0.94, f"Same gen matched by inner and global: {comparison_title}")
                text.DrawLatex(0.15, 0.89, f"{int(low)} #leq p_{{T}}^{{reco}} < {int(high)} GeV (per track)")
                c.SaveAs(os.path.join(args.outdir,
                                     f"shared_{comparison}_{metric}_recopt_{int(low)}_{int(high)}.png"))

    # 2D distributions
    for (name, metric), h in hists.items():
        save_hist(h)
        c = ROOT.TCanvas(f"c_2d_{name}_{metric}", "", 1000, 800)
        c.SetRightMargin(0.14)
        if metric == "rel_res":
            h.GetYaxis().SetRangeUser(-0.5, 0.5)
        h.Draw("COLZ")
        empty = no_data("No accepted shared-gen matches") if not h.GetEntries() else None
        c.SaveAs(os.path.join(args.outdir, f"{metric}_vs_genpt_{name}.png"))

    # Median and central 68% half-width for each paired common cohort.
    for comparison, (comparison_title, names) in COMPARISONS.items():
        for metric, (_, _, _, ytitle) in METRICS.items():
            for stat, stat_title in (("median", "Median of " + ytitle),
                                     ("width68", "(84th - 16th pct.)/2 of " + ytitle)):
                c = ROOT.TCanvas(f"c_{comparison}_{metric}_{stat}", "", 1200, 800)
                c.SetLeftMargin(0.13)
                mg = ROOT.TMultiGraph()
                leg = outside_legend(c)
                graphs = []
                for name in names:
                    label, color = TRACKS[name]
                    g = ROOT.TGraph()
                    g.SetName(f"g_{metric}_{stat}_{name}")
                    h = hists[name, metric]
                    for b in range(1, h.GetNbinsX() + 1):
                        vals = sorted(values.get((name, metric, b), []))
                        if len(vals) < 20:
                            continue
                        y = (quantile(vals, 0.5) if stat == "median"
                             else (quantile(vals, 0.84) - quantile(vals, 0.16)) / 2.)
                        g.SetPoint(g.GetN(), h.GetXaxis().GetBinCenter(b), y)
                    g.SetLineColor(color)
                    g.SetMarkerColor(color)
                    g.SetMarkerStyle(20)
                    g.SetLineWidth(2)
                    g.Write()
                    numerical["graphs"][g.GetName()] = {
                        "x": [g.GetPointX(i) for i in range(g.GetN())],
                        "y": [g.GetPointY(i) for i in range(g.GetN())]}
                    graphs.append(g)
                    if g.GetN():
                        mg.Add(g, "LP")
                        leg.AddEntry(g, label, "lp")
                if mg.GetListOfGraphs() and mg.GetListOfGraphs().GetSize():
                    mg.SetTitle(f"{comparison_title};gen p_{{T}} [GeV];{stat_title}")
                    mg.Draw("A")
                    if stat == "width68":
                        mg.SetMinimum(0.)
                    leg.Draw()
                    c.SaveAs(os.path.join(args.outdir, f"shared_{comparison}_{metric}_{stat}_vs_genpt.png"))
                else:
                    empty = no_data("No gen-p_{T} bin has at least 20 accepted matches")
                    c.SaveAs(os.path.join(args.outdir, f"shared_{comparison}_{metric}_{stat}_vs_genpt.png"))

    # Zero-centered containment radii: quantiles of |residual|, as in L1TrackNtuplePlot.
    for metric in METRICS:
        for name, (label, color) in TRACKS.items():
            c = ROOT.TCanvas(f"c_interval_{name}_{metric}", "", 1200, 800)
            c.SetLeftMargin(0.15)
            mg = ROOT.TMultiGraph()
            leg = outside_legend(c)
            graphs = []
            h = hists[name, metric]
            for fraction, marker in ((0.68, 20), (0.90, 26), (0.99, 24)):
                percent = int(round(100 * fraction))
                g = ROOT.TGraph()
                g.SetName(f"g_{metric}_interval{percent}_{name}")
                for b in range(1, h.GetNbinsX() + 1):
                    vals = sorted(abs(v) for v in values.get((name, metric, b), []))
                    if len(vals) < 10:
                        continue
                    g.SetPoint(g.GetN(), h.GetXaxis().GetBinCenter(b),
                               quantile(vals, fraction))
                g.SetLineColor(color)
                g.SetMarkerColor(color)
                g.SetMarkerStyle(marker)
                g.Write()
                numerical["graphs"][g.GetName()] = {
                    "x": [g.GetPointX(i) for i in range(g.GetN())],
                    "y": [g.GetPointY(i) for i in range(g.GetN())]}
                graphs.append(g)
                if g.GetN():
                    mg.Add(g, "P")
                    leg.AddEntry(g, f"{percent}%", "p")
            if mg.GetListOfGraphs() and mg.GetListOfGraphs().GetSize():
                ytitle = ("|p_{T}^{reco} - p_{T}^{gen}| containment radius [GeV]"
                          if metric == "res" else
                          "|(p_{T}^{reco} - p_{T}^{gen}) / p_{T}^{gen}| containment radius")
                mg.SetTitle(f"Shared-gen {label};gen p_{{T}} [GeV];{ytitle}")
                mg.SetMinimum(0.)
                mg.Draw("A")
                leg.Draw()
                c.SaveAs(os.path.join(args.outdir,
                                     f"{metric}_interval_vs_genpt_{name}.png"))
            else:
                empty = no_data("No gen-p_{T} bin has at least 10 accepted matches")
                c.SaveAs(os.path.join(args.outdir,
                                     f"{metric}_interval_vs_genpt_{name}.png"))

    # Overlay 68% and 90% absolute-residual containment radii for common cohorts.
    for comparison, (comparison_title, names) in COMPARISONS.items():
        for metric in METRICS:
            c = ROOT.TCanvas(f"c_shared_interval_{comparison}_{metric}", "", 1200, 800)
            c.SetLeftMargin(0.15)
            mg = ROOT.TMultiGraph()
            leg = outside_legend(c)
            graphs = []
            for name in names:
                label, color = TRACKS[name]
                h = hists[name, metric]
                for fraction, marker, style in ((0.68, 20, 1), (0.90, 26, 2)):
                    percent = int(round(100 * fraction))
                    g = ROOT.TGraph()
                    g.SetName(f"g_shared_{comparison}_{metric}_interval{percent}_{name}")
                    for b in range(1, h.GetNbinsX() + 1):
                        vals = sorted(abs(v) for v in values.get((name, metric, b), []))
                        if len(vals) < 10:
                            continue
                        g.SetPoint(g.GetN(), h.GetXaxis().GetBinCenter(b),
                                   quantile(vals, fraction))
                    g.SetLineColor(color)
                    g.SetMarkerColor(color)
                    g.SetMarkerStyle(marker)
                    g.SetLineStyle(style)
                    g.SetLineWidth(2)
                    g.Write()
                    numerical["graphs"][g.GetName()] = {
                        "x": [g.GetPointX(i) for i in range(g.GetN())],
                        "y": [g.GetPointY(i) for i in range(g.GetN())]}
                    graphs.append(g)
                    if g.GetN():
                        mg.Add(g, "LP")
                        leg.AddEntry(g, f"{label}, {percent}%", "lp")
            if mg.GetListOfGraphs() and mg.GetListOfGraphs().GetSize():
                ytitle = ("|p_{T}^{reco} - p_{T}^{gen}| containment radius [GeV]"
                          if metric == "res" else
                          "|(p_{T}^{reco} - p_{T}^{gen}) / p_{T}^{gen}| containment radius")
                mg.SetTitle(f"{comparison_title};gen p_{{T}} [GeV];{ytitle}")
                mg.SetMinimum(0.)
                mg.Draw("A")
                leg.Draw()
                c.SaveAs(os.path.join(args.outdir, f"shared_{comparison}_{metric}_interval_vs_genpt.png"))
            else:
                empty = no_data("No gen-p_{T} bin has at least 10 accepted matches")
                c.SaveAs(os.path.join(args.outdir, f"shared_{comparison}_{metric}_interval_vs_genpt.png"))

    # Choice numerators and denominators retain pT underflow/overflow as raw counts.
    save_hist(choice_total)
    for variable, denominator in choice_den.items():
        save_hist(denominator)
        c = ROOT.TCanvas(f"c_best_choices_{variable}", "", 1200, 800)
        c.SetLeftMargin(0.15)
        c.SetBottomMargin(0.13)
        leg = outside_legend(c)
        frame = denominator.Clone(f"h_best_choice_frame_{variable}")
        frame.Reset()
        frame.SetTitle("BestTrack choices on global-matched reco object")
        frame.GetYaxis().SetTitle("Fraction of shared gen matches")
        frame.SetMinimum(0.)
        frame.SetMaximum(1.05)
        frame.Draw("AXIS")
        graphs = []
        total = choice_total.Integral(0, choice_total.GetNbinsX() + 1)
        for code, (label, color) in CHOICES.items():
            numerator = choice_num[variable, code]
            save_hist(numerator)
            graph = ROOT.TGraph()
            graph.SetName(f"g_best_choice_fraction_{variable}_{code}")
            for b in range(1, denominator.GetNbinsX() + 1):
                n = denominator.GetBinContent(b)
                if n:
                    graph.SetPoint(graph.GetN(), denominator.GetBinCenter(b),
                                   numerator.GetBinContent(b) / n)
            graph.SetLineColor(color)
            graph.SetMarkerColor(color)
            graph.SetMarkerStyle(20 + code)
            graph.SetLineStyle(1 + code)
            graph.SetLineWidth(2)
            graph.Write()
            numerical["graphs"][graph.GetName()] = {
                "x": [graph.GetPointX(i) for i in range(graph.GetN())],
                "y": [graph.GetPointY(i) for i in range(graph.GetN())]}
            graphs.append(graph)
            if graph.GetN():
                graph.Draw("LP SAME")
            overall = 100. * choice_total.GetBinContent(code + 1) / total if total else 0.
            leg.AddEntry(graph, f"{label} (all: {overall:.1f}%)", "lp")
        empty = no_data("No shared-gen matches") if not denominator.GetEntries() else None
        leg.Draw()
        text = ROOT.TLatex()
        text.SetNDC(True)
        text.SetTextSize(0.025)
        text.DrawLatex(0.15, 0.94, "Same gen matched by both inner and global; all BestTrack choices retained")
        if variable == "genpt":
            text.DrawLatex(0.15, 0.90, "BestTrack belongs to the global-matched reco object")
        else:
            text.DrawLatex(0.15, 0.90, "p_{T} of the independently matched inner track")
        c.SaveAs(os.path.join(args.outdir, f"shared_best_choice_fractions_vs_{variable}.png"))

    ROOT.TObjString(json.dumps({k: v for k, v in numerical.items()
                               if k not in ("histograms", "graphs")})).Write("metadata")
    out.Close()
    with open(os.path.join(args.outdir, "residuals.json"), "w") as dest:
        json.dump(numerical, dest, indent=2)
    print(f"Wrote {args.outdir}/residuals.root and residuals.json and PNGs")


if __name__ == "__main__":
    main()
