#!/usr/bin/env python3
"""Test of the PreselectedVertexing per-leg pair branches (kb/studies/bkg-reco-dimuons.md Part 5; LPC only).

python3 check_leg_branches.py --old OLD.root --new NEW.root

OLD = existing recoOnly output of a sample (all its files); NEW = the new code on a subset of those files. Every event
of NEW must be in OLD; for those events, every old pairs branch must agree pair by pair (same order), and so must the
stage of each pair. Closure: dR and pT(mumu) recomputed from the new per-leg floats match the stored aggregates
(dR abs 1e-4; pT abs 1e-3 GeV + rel 1e-5). PCA leg fields are -999 exactly when pPcaOK is false.
"""
import argparse

import numpy as np
import uproot

from bkg_reco_dimuons import vector_stages

LEG = ["pPcaPtA", "pPcaEtaA", "pPcaPhiA", "pPcaPtB", "pPcaEtaB", "pPcaPhiB",
       "pRecoPtA", "pRecoEtaA", "pRecoPhiA", "pRecoPtB", "pRecoEtaB", "pRecoPhiB"]


def dphi(a, b):
    d = a - b
    return d - 2 * np.pi * np.round(d / (2 * np.pi))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    args = ap.parse_args()
    old, new = uproot.open(args.old)["preselectedVertexing"], uproot.open(args.new)["preselectedVertexing"]
    oe = old["events"].arrays(["run", "lumi", "event", "pfMetPt", "pfMetPhi"], library="np")
    ne = new["events"].arrays(["run", "lumi", "event", "pfMetPt", "pfMetPhi"], library="np")
    names = old["pairs"].keys()
    missing = [k for k in names + LEG if k not in new["pairs"].keys()]
    if missing:
        raise RuntimeError("New output lacks branches %r" % missing)
    op = old["pairs"].arrays(names, library="np")
    np_ = new["pairs"].arrays(names + LEG, library="np")
    okeys = {k: i for i, k in enumerate(zip(oe["run"].tolist(), oe["lumi"].tolist(), oe["event"].tolist()))}
    nkeys = list(zip(ne["run"].tolist(), ne["lumi"].tolist(), ne["event"].tolist()))
    if len(set(nkeys)) != len(nkeys) or any(k not in okeys for k in nkeys):
        raise RuntimeError("New events not a duplicate-free subset of the old output")
    orow = np.array([okeys[k] for k in nkeys])
    for b in ("pfMetPt", "pfMetPhi"):
        if not np.array_equal(oe[b][orow], ne[b]):
            raise RuntimeError("Event branch differs: " + b)
    # Old pairs of the selected events, in event order and within-event order (as the producer writes them).
    o_by_event = {}
    for i, e in enumerate(op["pEvent"]):
        o_by_event.setdefault(int(e), []).append(i)
    n_by_event = {}
    for i, e in enumerate(np_["pEvent"]):
        n_by_event.setdefault(int(e), []).append(i)
    oi, ni = [], []
    for nrow, orow_i in enumerate(orow):
        a, b = o_by_event.get(int(orow_i), []), n_by_event.get(nrow, [])
        if len(a) != len(b):
            raise RuntimeError("Pair count differs for event %r: old %d, new %d" % (nkeys[nrow], len(a), len(b)))
        oi += a; ni += b
    oi, ni = np.array(oi, dtype=np.int64), np.array(ni, dtype=np.int64)
    if len(ni) != len(np_["pEvent"]):
        raise RuntimeError("New pairs not all matched to events")
    for k in names:
        if k == "pEvent":
            continue
        x, y = op[k][oi], np_[k][ni]
        same = (x == y) | (np.isnan(x) & np.isnan(y)) if x.dtype.kind == "f" else (x == y)
        if not np.all(same):
            raise RuntimeError("Old branch %s differs in %d pairs" % (k, int((~same).sum())))
    old_sub = {k: op[k][oi].copy() for k in names}
    old_sub["pEvent"] = np_["pEvent"][ni]  # re-index to the new event rows, whose MET equals the old one
    so = vector_stages(old_sub, ne["pfMetPt"], ne["pfMetPhi"])
    sn = vector_stages({k: np_[k][ni] for k in names}, ne["pfMetPt"], ne["pfMetPhi"])
    if not np.array_equal(so, sn):
        raise RuntimeError("Stage masks differ")
    p = {k: np_[k].astype(np.float64) for k in names + LEG}
    ok = np_["pPcaOK"].astype(bool)
    pca = np.stack([p[k] for k in LEG[:6]])
    if not np.all((pca[:, ~ok] == -999).all(axis=0)) or np.any(pca[:, ok] == -999) or not np.all(np.isfinite(pca[:, ok])):
        raise RuntimeError("PCA leg sentinels do not follow pPcaOK")
    reco = np.stack([p[k] for k in LEG[6:]])
    if np.any(~np.isfinite(reco)) or np.any(reco == -999):
        raise RuntimeError("Reco leg fields missing")
    def closure(tag, pt_a, eta_a, phi_a, pt_b, eta_b, phi_b, dr, mumu, mask):
        rdr = np.hypot(eta_a - eta_b, dphi(phi_a, phi_b))
        px = pt_a * np.cos(phi_a) + pt_b * np.cos(phi_b)
        py = pt_a * np.sin(phi_a) + pt_b * np.sin(phi_b)
        rpt = np.hypot(px, py)
        e_dr = np.abs(rdr - dr)[mask]
        e_pt = (np.abs(rpt - mumu) - 1e-5 * np.abs(mumu))[mask]
        if not (np.all(np.isfinite(e_dr)) and np.all(np.isfinite(e_pt))):
            raise RuntimeError("Non-finite closure residual: " + tag)
        print("CLOSURE %s pairs %d: max |dDR| %.2e, max |dpT| - 1e-5 pT %.2e GeV" % (
            tag, int(mask.sum()), e_dr.max(initial=0), e_pt.max(initial=0)))
        if e_dr.max(initial=0) > 1e-4 or e_pt.max(initial=0) > 1e-3:
            raise RuntimeError("Closure failed: " + tag)
    closure("reco", *(p[k] for k in LEG[6:]), p["pRecoDR"], p["pRecoMuMuPt"], np.ones(len(ok), dtype=bool))
    closure("pca", *(p[k] for k in LEG[:6]), p["pPcaDR"], p["pPcaMuMuPt"], ok)
    print("CHECK OK events %d, pairs %d (pPcaOK %d), passing %d; all %d old pair branches identical" % (
        len(nkeys), len(ni), int(ok.sum()), int((sn == 6).sum()), len(names) - 1))


if __name__ == "__main__":
    main()
