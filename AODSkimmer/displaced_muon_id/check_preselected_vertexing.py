#!/usr/bin/env python3
"""Validate PreselectedVertexing outputs (run on LPC with uproot; see study note).

python3 check_preselected_vertexing.py --inputs DIR_OR_FILE... [--old-efficiency efficiency.root] [--oracle-events N]
Checks per sample: (a) the hierarchy match of every twoAcc event equals the scalar oracle of the component study
on the saved primitives (gated and, with eligibility=ID only, the NoEta version); (b) with --old-efficiency, the
NoEta hierarchy equals the component study's saved per-gen rows (object index and stage) for the same events
(valid when the input is a prefix of the sample's events, i.e. inputEntry = tree entry); (c) bookkeeping:
categories partition best-best, status/cat consistency, pair tree rows = sum(anyNPairs), valid <= pairs.
"""
import argparse
from importlib import util
from pathlib import Path
import numpy as np
import uproot

here = Path(__file__).resolve().parent
spec = util.spec_from_file_location("cme", here / "component_matching_efficiency.py")
cme = util.module_from_spec(spec)
spec.loader.exec_module(cme)


def check_file(path, old, oracle_events):
    with uproot.open(path) as f:
        sample = str(f["preselectedVertexing/sample"].member("fTitle"))
        t = f["preselectedVertexing/events"]
        a = t.arrays(library="np")
        npairs = f["preselectedVertexing/pairs"].num_entries
    n = t.num_entries
    two = np.flatnonzero(a["twoAcc"])
    out = dict(sample=sample, events=n, twoAcc=len(two))
    mism_oracle = mism_old = 0
    checked = 0
    old_sel = None
    if old is not None:
        si = old["samples"].index(sample)
        sel = old["sample"] == si
        old_sel = {(int(e), int(g)): (int(h), int(ht)) for e, g, h, ht in
                   zip(old["inputEntry"][sel], old["genParticleIndex"][sel], old["hierarchy"][sel], old["hierType"][sel])}
    for k, ei in enumerate(two):
        gq = np.asarray(a["sigQ"][ei]); gp = np.asarray(a["sigProp"][ei]).astype(int)
        ge = np.asarray(a["sigEta2"][ei], dtype=np.float32); gf = np.asarray(a["sigPhi2"][ei], dtype=np.float32)
        ct = np.asarray(a["cType"][ei]); co = np.asarray(a["cObj"][ei]); cq = np.asarray(a["cQ"][ei])
        idp = np.asarray(a["cIdPass"][ei]).astype(bool); eta = np.asarray(a["cInEta"][ei]).astype(bool)
        cp = np.asarray(a["cProp"][ei]).astype(int)
        ce = np.asarray(a["cEta2"][ei], dtype=np.float32); cf = np.asarray(a["cPhi2"][ei], dtype=np.float32)
        for gated, objs, types in ((True, a["matchObj"][ei], a["matchType"][ei]),
                                   (False, a["matchObjNoEta"][ei], a["matchTypeNoEta"][ei])):
            if k < oracle_events:
                elig = idp & (eta if gated else True)
                _, hier, tier = cme.oracle(gq, gp, ge, gf, ct, co, cq, (cp * elig).astype(int), ce, cf)
                if not (np.array_equal(hier, objs) and np.array_equal(tier, types)):
                    mism_oracle += 1
                if gated:
                    checked += 1
            if not gated and old_sel is not None:
                for g, gi in enumerate(a["sigIndex"][ei]):
                    ref = old_sel.get((int(ei), int(gi)))
                    if ref is None or ref != (int(objs[g]), int(types[g])):
                        mism_old += 1
    out.update(oracle_events=checked, oracle_mismatch=mism_oracle, old_mismatch=mism_old if old_sel else None)
    # bookkeeping
    st, cat, ok = a["bbStatus"], a["bbCat"], a["twoAcc"]
    matched = np.array([len(m) == 2 and m[0] >= 0 and m[1] >= 0 for m in a["matchComp"]])
    assert np.array_equal(matched, st > 0), "bbStatus>0 must equal both matched"
    assert np.all((cat >= 0) == (st > 0)) and np.all(cat < 6)
    assert ok[st > 0].all()
    assert a["anyNPairs"].sum() == npairs, "pair tree rows != sum(anyNPairs)"
    assert np.all(a["anyNValid"] <= a["anyNPairs"])
    out.update(matched=int(matched.sum()), valid=int((st == 3).sum()), failed=int((st == 2).sum()),
               shared=int((st == 1).sum()), pairs=int(npairs))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--old-efficiency")
    p.add_argument("--oracle-events", type=int, default=200)
    args = p.parse_args()
    old = None
    if args.old_efficiency:
        with uproot.open(args.old_efficiency) as f:
            old = f["matches"].arrays(library="np")
            old["samples"] = f["samples"].member("fTitle").splitlines()
    files = []
    for item in args.inputs:
        q = Path(item)
        files += sorted(q.glob("*.root")) if q.is_dir() else [q]
    for path in files:
        print(check_file(path, old, args.oracle_events), flush=True)


if __name__ == "__main__":
    main()
