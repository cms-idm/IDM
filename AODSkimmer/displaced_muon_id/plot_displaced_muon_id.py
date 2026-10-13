#!/usr/bin/env python3
"""Plot the three ID comparisons. Usage: python3 plot_displaced_muon_id.py INPUT.root OUTDIR [--overall-only].
Fixed pre-ID same-charge matches; all x axes use accepted signal gen quantities.
Reco efficiency = passing matched units / accepted gen. ID efficiency = passing / matched units.
ROOT histograms retain numerators and denominators; PNGs are unweighted exploratory plots.
"""
import argparse
import os
import json
from array import array
import numpy as np
import ROOT

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("input")
parser.add_argument("outdir")
parser.add_argument("--overall-only", action="store_true", help="bounded plotting validation")
args = parser.parse_args()
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)
ROOT.gErrorIgnoreLevel = ROOT.kWarning
f = ROOT.TFile.Open(args.input, "UPDATE")
if not f or f.IsZombie() or not f.Get("genMuons"):
    raise RuntimeError("Missing input genMuons tree")
cols = ["genPt", "genEta", "genVxy", "genVz", "genD0", "genProp", "mchi", "dmchi", "ctau",
        "objIdx", "objMask", "objIdBits", "objSelectedIdx", "objSelectedMask", "priorityIdx",
        "prioritySelectedIdx", "prioritySelectedMask"]
for name in ("inner", "outer", "global"):
    cols += [name + "_idx", name + "_pass", name + "_selectedIdx"]
d = ROOT.RDataFrame(f.Get("genMuons")).AsNumpy(cols)
if len(d["genPt"]) == 0:
    raise RuntimeError("No accepted signal gen rows")
d["absD0"] = np.abs(d["genD0"])
VARS = [
    ("pt", "genPt", "gen p_{T} [GeV]", np.array([0, .5, 1, 2, 3, 4, 5, 7.5, 10, 15, 20, 30, 50, 70, 100.]), False),
    ("eta", "genEta", "gen #eta", np.linspace(-2.4, 2.4, 17), False),
    ("lxy", "genVxy", "gen L_{xy} [cm]", np.array([.01, .03, .1, .3, 1, 3, 10, 20, 30, 45, 60, 80, 110, 200, 300, 500, 740.]), True),
    ("d0", "absD0", "gen |d_{0}| [cm] (straight line to origin)", np.array([.001, .003, .01, .03, .1, .3, 1, 3, 10, 20, 40, 70, 110, 200, 400, 740.]), True),
]
colors = [ROOT.kAzure+1, ROOT.kGreen+2, ROOT.kRed+1, ROOT.kMagenta+1, ROOT.kOrange+7,
          ROOT.kCyan+2, ROOT.kViolet+1]
allsel = np.ones(len(d["genPt"]), dtype=bool)
matched = d["objIdx"] >= 0
mask, bits = d["objMask"], d["objIdBits"]
inner_ok, outer_ok, global_ok = [(bits & b) != 0 for b in (1, 2, 4)]
categories = [((mask & 4) != 0, "Global component present", "global"),
              (mask == 1, "Inner only", "inner_only"),
              (mask == 2, "Outer only", "outer_only"),
              (mask == 3, "Inner + outer, no global", "mixed")]
all_ok = matched & (mask != 0) & (bits == mask)
fixed = (((mask & 4) != 0) & global_ok) | ((mask == 1) & inner_ok) | ((mask == 2) & outer_ok)
mixed_rules = [(inner_ok & outer_ok, "Both IDs", "both"),
               (inner_ok, "Tracker ID required", "tracker"),
               (outer_ok, "DSA ID required", "dsa"),
               (inner_ok | outer_ok, "Either ID", "either")]
# Every entry is (baseline denominator, passing numerator, label, key).
sets = {
    "components": [(d[n+"_idx"] >= 0, (d[n+"_idx"] >= 0) & (d[n+"_pass"] == 1), label, n)
                   for n, label in [("inner", "Inner: tracker-muon flag"),
                                    ("outer", "Outer: DSA ID"), ("global", "Global: global-muon flag")]],
    "all_components": [(matched, all_ok, "All displacedMuons", "all")] +
                      [(matched & cat, all_ok & cat, label, key) for cat, label, key in categories],
    "hierarchy_categories": [(matched & cat, matched & cat & passed, label, key)
                             for (cat, label, key), passed in zip(categories[:3], [global_ok, inner_ok, outer_ok])] +
                            [(matched & (mask == 3), matched & (mask == 3) & passed,
                              "Inner+outer: " + label, key) for passed, label, key in mixed_rules],
    "hierarchy": [(matched, matched & (fixed | ((mask == 3) & passed)),
                   "Mixed objects: " + label, key) for passed, label, key in mixed_rules],
}
# NumPy represents ROOT fixed-size array branches as an object array of arrays.
selected_idx = np.stack(d["objSelectedIdx"])
selected_mask = np.stack(d["objSelectedMask"])
priority_selected_idx = np.stack(d["prioritySelectedIdx"])
reco_sets = {
    "components": [d[n+"_selectedIdx"] >= 0 for n in ("inner", "outer", "global")],
    "all_components": [selected_idx[:, 0] >= 0] +
                      [(selected_idx[:, 0] >= 0) & cat for cat in
                       [(selected_mask[:, 0] & 4) != 0, selected_mask[:, 0] == 1,
                        selected_mask[:, 0] == 2, selected_mask[:, 0] == 3]],
    "hierarchy_categories": [(selected_idx[:, 1] >= 0) & cat for cat in
                             [(selected_mask[:, 1] & 4) != 0, selected_mask[:, 1] == 1, selected_mask[:, 1] == 2]] +
                            [(selected_idx[:, p] >= 0) & (selected_mask[:, p] == 3) for p in range(1, 5)],
    "hierarchy": [selected_idx[:, p] >= 0 for p in range(1, 5)],
}
for series in sets.values():
    for base, passed, _, _ in series:
        if np.any(passed & ~base):
            raise RuntimeError("ID numerator exceeds its matched denominator")
if np.any(all_ok & ~sets["hierarchy"][0][1]):
    raise RuntimeError("All-component policy is not a subset of the both-ID hierarchy")
if np.any(sets["hierarchy"][0][1] & ~sets["hierarchy"][3][1]):
    raise RuntimeError("Both-ID hierarchy is not a subset of either-ID hierarchy")


def counts(x, sel, edges):
    # Retain the charge-flip workflow's endpoint folding; report it on every canvas.
    return np.histogram(np.clip(x[sel], edges[0], np.nextafter(edges[-1], -np.inf)), bins=edges)[0]


def graph(k, n, edges, logx):
    g = ROOT.TGraphAsymmErrors()
    for i in range(len(n)):
        if n[i] == 0:
            continue
        p = k[i] / n[i]
        lo = ROOT.TEfficiency.ClopperPearson(int(n[i]), int(k[i]), .682689492, False)
        hi = ROOT.TEfficiency.ClopperPearson(int(n[i]), int(k[i]), .682689492, True)
        x = np.sqrt(edges[i] * edges[i+1]) if logx else (edges[i]+edges[i+1])/2
        j = g.GetN()
        g.SetPoint(j, x, p)
        g.SetPointError(j, x-edges[i], edges[i+1]-x, p-lo, hi-p)
    return g


os.makedirs(args.outdir, exist_ok=True)
numeric = f  # One central ROOT: original gen/reco trees plus all efficiency histograms.
summary = {"definitions": {
    "acceptance": "signal status1 lastCopy abs(firstDifferentMotherID)=1000023; abs(eta)<2.4; 0<=vxy<740 cm; abs(vz)<960 cm; no pT minimum",
    "signal_fields": "GenSigMuon/GenSigAntiMuon scalars; abs(charge)=1",
    "matching": "same-charge station2 position eta/phi; global-nearest greedy 1-to-1 with fallback; strict inner/global dR<0.06, outer dR<0.10",
    "reco_efficiency": "matched after ID selection / accepted signal gen (propagation failures retained)",
    "id_efficiency": "passing ID / baseline pre-ID signal-matched units; fixed pre-ID assignment",
    "fixed_retention_efficiency": "passing baseline matched units / accepted signal gen",
    "component_order": ["inner", "outer", "global"],
    "policy_order": ["all_components", "hierarchy_both", "hierarchy_tracker", "hierarchy_dsa", "hierarchy_either"],
    "priority_matching": "first present valid component global->inner->outer; no fallback if that component fails propagation, charge or dR",
    "d0": "abs(vx*sin(phi)-vy*cos(phi)); gen straight-line transverse IP to (0,0)",
    "pooling": "unweighted", "interval": "68.27% Clopper-Pearson",
    "endpoint_folding": "underflow/overflow included in first/last visible bin",
    "inner_id": "parent isTracker=true", "global_id": "parent isGlobal=true",
    "outer_id": "CSC+DT>12; CSC>0 or DT>18; normalizedChi2<2.5; pt>0 and ptError/pt<1",
    "mixed_single_id": "inclusive requirement; no veto on passing the other ID",
    "deduplication": "none; distinct reco objects remain distinct units",
}, "groups": {}}


def store_hist(directory, name, values, edges):
    directory.cd()
    hist = ROOT.TH1D(name, name, len(edges)-1, array("d", edges))
    for i, value in enumerate(values, 1):
        hist.SetBinContent(i, int(value))
    hist.Write()
    hist.SetDirectory(0)


def make(sel, label, group):
    report = {"accepted_gen": int(sel.sum()), "propagated_gen": int((sel & (d["genProp"] == 1)).sum()),
              "any_vs_priority": {}, "sets": {}}
    for key, a, b in [("pre_id", matched, d["priorityIdx"] >= 0)] + [
        (name, selected_idx[:, p] >= 0, priority_selected_idx[:, p] >= 0)
        for p, name in enumerate(summary["definitions"]["policy_order"])]:
        report["any_vs_priority"][key] = {"any": int((sel & a).sum()), "priority": int((sel & b).sum()),
                                             "any_only": int((sel & a & ~b).sum()), "priority_only": int((sel & b & ~a).sum())}
    for setname, series in sets.items():
        report["sets"][setname] = {}
        target = os.path.join(args.outdir, setname, group)
        os.makedirs(target, exist_ok=True)
        for vname, branch, xtitle, edges, logx in VARS:
            directory = numeric.mkdir(f"efficiencies/{setname}/{group}/{vname}", "", True)
            x = d[branch]
            accepted = counts(x, sel, edges)
            binreport = {"edges": edges.tolist(), "accepted_gen": accepted.tolist(), "curves": {}}
            store_hist(directory, "accepted_gen", accepted, edges)
            c = ROOT.TCanvas("c", "", 1100, 1000)
            titlepad = ROOT.TPad("title", "", 0, .77, 1, 1)
            top = ROOT.TPad("reco", "", 0, .40, 1, .77)
            bottom = ROOT.TPad("id", "", 0, 0, 1, .40)
            for p in (titlepad, top, bottom):
                p.Draw()
            for p in (top, bottom):
                p.SetLeftMargin(.12); p.SetRightMargin(.03); p.SetTopMargin(.035)
                p.SetGridy(); p.SetLogx(logx)
            top.SetBottomMargin(.025)
            bottom.SetBottomMargin(.24)
            keep = []
            titlepad.cd()
            text = ROOT.TLatex(); text.SetNDC(); text.SetTextSize(.085)
            text.DrawLatex(.12, .89, f"#bf{{{label}}}   N_{{acc gen}} = {int(sel.sum())}")
            text.SetTextSize(.070)
            text.DrawLatex(.12, .75, "Same-charge St2 greedy 1-to-1; reco: select ID then match; ID: fixed baseline")
            text.DrawLatex(.12, .63, "|#eta|<2.4, 0#leqL_{xy}<740 cm, |v_{z}|<960 cm; unweighted; edges folded")
            legend = ROOT.TLegend(.12, .02, .97, .55)
            legend.SetNColumns(2); legend.SetTextSize(.080); legend.SetBorderSize(0); legend.SetFillStyle(0)
            top.cd()
            fr = top.DrawFrame(edges[0], 0, edges[-1], 1.05)
            fr.GetYaxis().SetTitle("reco efficiency with ID")
            fr.GetXaxis().SetLabelSize(0)
            bottom.cd()
            fb = bottom.DrawFrame(edges[0], 0, edges[-1], 1.05)
            fb.GetYaxis().SetTitle("ID efficiency | matched")
            fb.GetXaxis().SetTitle(xtitle); fb.GetXaxis().SetTitleSize(.07); fb.GetXaxis().SetLabelSize(.06)
            fb.GetXaxis().SetTitleOffset(1.3)
            for frame in (fr, fb):
                frame.GetYaxis().SetTitleSize(.06); frame.GetYaxis().SetLabelSize(.055)
                frame.GetYaxis().SetTitleOffset(.85); frame.GetYaxis().SetNdivisions(505)
            for j, (base, passed, name, key) in enumerate(series):
                n, k = counts(x, sel & base, edges), counts(x, sel & passed, edges)
                selected = counts(x, sel & reco_sets[setname][j], edges)
                store_hist(directory, key+"_matched", n, edges)
                store_hist(directory, key+"_passed", k, edges)
                store_hist(directory, key+"_selected_matched", selected, edges)
                binreport["curves"][key] = {"baseline_matched": n.tolist(), "baseline_passed_id": k.tolist(),
                                            "selected_matched": selected.tolist()}
                for pad, numerator, denominator in ((top, selected, accepted), (bottom, k, n)):
                    pad.cd()
                    g = graph(numerator, denominator, edges, logx)
                    g.SetMarkerStyle(20+j); g.SetMarkerSize(.8)
                    g.SetMarkerColor(colors[j]); g.SetLineColor(colors[j])
                    g.Draw("P same"); keep.append(g)
                    if pad == top:
                        legend.AddEntry(g, name, "p")
            titlepad.cd(); legend.Draw()
            c.SaveAs(os.path.join(target, f"id_vs_{vname}.png"))
            c.Close()
            report["sets"][setname][vname] = binreport
    summary["groups"][group] = report


def fmt(v):
    return f"{v:g}".replace(".", "p")


print(f"accepted_gen={len(allsel)} propagated_gen={int((d['genProp']==1).sum())}")
for setname, series in sets.items():
    print(setname)
    for base, passed, label, _ in series:
        n, k = int(base.sum()), int(passed.sum())
        j = [item[3] for item in series].index(_)
        selected = int(reco_sets[setname][j].sum())
        print(f"  {label}: matched={n} baseline_pass={k} selected_matched={selected} reco_eff={selected/len(allsel):.6f} id_eff={k/n if n else float('nan'):.6f}")
make(allsel, "All samples", "overall")
if not args.overall_only:
    for ct in sorted(set(d["ctau"])):
        make(d["ctau"] == ct, f"c#tau={ct:g} mm, all masses", "lifetime/ctau-"+fmt(ct))
    for mc, dm in sorted(set(zip(d["mchi"], d["dmchi"]))):
        selm = (d["mchi"] == mc) & (d["dmchi"] == dm)
        mass = f"Mchi-{fmt(mc)}_dMchi-{fmt(dm)}"
        make(selm, f"M_{{#chi}}={mc:g}, #DeltaM={dm:g} GeV, all lifetimes", "mass/"+mass)
        for ct in sorted(set(d["ctau"][selm])):
            make(selm & (d["ctau"] == ct), f"M_{{#chi}}={mc:g}, #DeltaM={dm:g} GeV, c#tau={ct:g} mm",
                 "sample/"+mass+"_ctau-"+fmt(ct))
# Audit every reconstructed object, not just signal-matched objects.
reco_cols = ["mchi", "dmchi", "ctau", "presenceMask", "isGlobal", "isTracker"]
rd = ROOT.RDataFrame(f.Get("displacedMuons")).AsNumpy(reco_cols)
audit = {}
for mc, dm, ct in sorted(set(zip(rd["mchi"], rd["dmchi"], rd["ctau"]))):
    sel = (rd["mchi"] == mc) & (rd["dmchi"] == dm) & (rd["ctau"] == ct)
    has_global = sel & ((rd["presenceMask"] & 4) != 0)
    key = f"Mchi-{fmt(mc)}_dMchi-{fmt(dm)}_ctau-{fmt(ct)}"
    audit[key] = {"objects": int(sel.sum()), "global_present": int(has_global.sum()),
                  "global_present_flag_true": int((has_global & (rd["isGlobal"] == 1)).sum()),
                  "global_present_flag_false": int((has_global & (rd["isGlobal"] != 1)).sum()),
                  "global_flag_true_component_absent": int((sel & ((rd["presenceMask"] & 4) == 0) & (rd["isGlobal"] == 1)).sum())}
summary["global_flag_audit"] = {"per_sample": audit, "overall": {
    k: sum(v[k] for v in audit.values()) for k in ["objects", "global_present", "global_present_flag_true", "global_present_flag_false", "global_flag_true_component_absent"]}}
input_counts = ROOT.RDataFrame(f.Get("inputSummary")).AsNumpy(["events", "acceptedGen", "propagatedGen"])
summary["input_counts"] = {k: int(v.sum()) for k, v in input_counts.items()}
json_path = os.path.splitext(args.input)[0] + ".json"
with open(json_path, "w") as out:
    json.dump(summary, out, indent=2, allow_nan=False)
numeric.cd()
ROOT.TObjString(json.dumps(summary, allow_nan=False)).Write("analysis_summary_json", ROOT.TObject.kOverwrite)
numeric.Close()
print("Central outputs:", args.input, json_path)
print("Plotting complete")
