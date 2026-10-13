#!/usr/bin/env python3
"""Tables from charge_flip_check.C output (Level 1). Unweighted counts, all samples pooled.

SSflip = disagreeing matches whose partner reco (other gen muon's match, same pass) has the
same charge as this reco: the same-sign reco pair a genuine single flip produces.

flip fraction = N(sign-off match, reco charge != gen charge) / N(sign-off match)
eff_off / eff_on = N(matched) / N(accepted signal gen), sign not required / required.
Intervals: 68.27% Clopper-Pearson.
Usage: python3 summarize_charge_flip.py out.root
"""
import sys
import numpy as np
import ROOT

ROOT.gROOT.SetBatch(True)
f = ROOT.TFile.Open(sys.argv[1])
cols = ["genQ", "genProp", "genPt", "genVxy"]
for c in ("in", "gl", "dsa"):
    cols += [f"{c}_offIdx", f"{c}_offQ", f"{c}_offDr", f"{c}_offExtra", f"{c}_onMatched",
             f"{c}_looseIdx", f"{c}_looseQ", f"{c}_looseDr", f"{c}_otherOnIdx",
             f"{c}_vtxIdx", f"{c}_vtxQ", f"{c}_vtxDr", f"{c}_offOtherQ", f"{c}_vtxOtherQ"]
d = ROOT.RDataFrame(f.Get("genMuons")).AsNumpy(cols)
THR = {"in": 0.06, "gl": 0.06, "dsa": 0.10}
NAME = {"in": "inner track", "gl": "global track", "dsa": "DSA"}


def cp(k, n):
    if n == 0:
        return float("nan"), float("nan")
    return (ROOT.TEfficiency.ClopperPearson(int(n), int(k), 0.682689492, False),
            ROOT.TEfficiency.ClopperPearson(int(n), int(k), 0.682689492, True))


def frac(k, n):
    lo, hi = cp(k, n)
    if n == 0:
        return "      -          "
    p = k / n
    return f"{100*p:6.3f}% [{100*lo:.3f},{100*hi:.3f}]"


def row(c, sel, label):
    n_acc = sel.sum()
    q = d["genQ"]
    m = sel & (d[f"{c}_offIdx"] >= 0)
    flip = m & (d[f"{c}_offQ"] != q)
    # Disagreement classes from the other gen muon's match in the same pass:
    #   swap: other reco carries this gen's charge -> opposite-sign reco pair assigned crosswise
    #   SS  : other reco has the same charge as this reco -> same-sign reco pair (genuine flip)
    #   one : other gen unmatched (ambiguous)
    swap = flip & (d[f"{c}_offOtherQ"] == q)
    ss = flip & (d[f"{c}_offOtherQ"] == d[f"{c}_offQ"])
    tight = m & (d[f"{c}_offDr"] < THR[c] / 2)
    flip_t = tight & (d[f"{c}_offQ"] != q)
    on = sel & (d[f"{c}_onMatched"] == 1)
    ml = sel & (d[f"{c}_looseIdx"] >= 0)
    flip_l = ml & (d[f"{c}_looseQ"] != q)
    mv = sel & (d[f"{c}_vtxIdx"] >= 0)
    flip_v = mv & (d[f"{c}_vtxQ"] != q)
    swap_v = flip_v & (d[f"{c}_vtxOtherQ"] == q)
    ss_v = flip_v & (d[f"{c}_vtxOtherQ"] == d[f"{c}_vtxQ"])
    print(f"  {label:<20} acc={n_acc:8d} eff off/on={100*m.sum()/max(n_acc,1):5.2f}/{100*on.sum()/max(n_acc,1):5.2f}% | "
          f"St2 {flip.sum():5d}/{m.sum():7d} {frac(flip.sum(), m.sum())} (swap/SS/one {swap.sum()}/{ss.sum()}/{flip.sum()-swap.sum()-ss.sum()}) "
          f"half-dR {flip_t.sum():5d}/{tight.sum():7d} loose {flip_l.sum():5d}/{ml.sum():7d} | "
          f"SSflip {frac(ss.sum(), m.sum())} | VTX {flip_v.sum():5d}/{mv.sum():7d} {frac(flip_v.sum(), mv.sum())} "
          f"(swap/SS/one {swap_v.sum()}/{ss_v.sum()}/{flip_v.sum()-swap_v.sum()-ss_v.sum()})")


pt_bins = [5, 10, 20, 30, 50, 80, 1e9]
lxy_bins = [0, 1, 10, 30, 60, 110, 300, 740]
for c in ("in", "gl", "dsa"):
    print(f"\n=== {NAME[c]} (dR<{THR[c]} at station 2, greedy 1-to-1, signal gen) ===")
    allsel = np.ones_like(d["genPt"], dtype=bool)
    row(c, allsel, "all accepted")
    if c == "dsa":
        row(c, allsel & ((d["dsa_offIdx"] < 0) | (d["dsa_offExtra"] == 1)), "matched DSA has ID==1*")
    print(" gen pT bins:")
    for lo, hi in zip(pt_bins[:-1], pt_bins[1:]):
        row(c, (d["genPt"] >= lo) & (d["genPt"] < hi), f"pT [{lo},{hi if hi < 1e8 else 'inf'})")
    print(" gen Lxy bins (cm):")
    for lo, hi in zip(lxy_bins[:-1], lxy_bins[1:]):
        row(c, (d["genVxy"] >= lo) & (d["genVxy"] < hi), f"Lxy [{lo},{hi})")
    print(" high pT x large Lxy:")
    for ptlo in (30, 50):
        for lxylo in (30, 60, 110):
            row(c, (d["genPt"] >= ptlo) & (d["genVxy"] >= lxylo), f"pT>={ptlo}, Lxy>={lxylo}")
print("\n* DSA ID row: denominator drops gen muons whose sign-off match fails displacedId==1;"
      " eff columns there are not efficiencies.")
