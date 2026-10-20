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
        combos = f["preselectedVertexing/combos"].arrays(library="np")
    n = t.num_entries
    two = np.flatnonzero(a["twoAcc"])
    out = dict(sample=sample, events=n, twoAcc=len(two))
    mism_oracle = mism_old = 0
    checked = 0
    old_sel = None
    if old is not None:
        si = old["samples"].index(sample)
        sel = old["sample"] == si
        old_sel = {(int(e), int(g)): (int(h), int(ht), int(i0), int(i1), int(i2)) for e, g, h, ht, i0, i1, i2 in
                   zip(old["inputEntry"][sel], old["genParticleIndex"][sel], old["hierarchy"][sel], old["hierType"][sel],
                       old["inner"][sel], old["outer"][sel], old["global"][sel])}
    for k, ei in enumerate(two):
        gq = np.asarray(a["sigQ"][ei]); gp = np.asarray(a["sigProp"][ei]).astype(int)
        ge = np.asarray(a["sigEta2"][ei], dtype=np.float32); gf = np.asarray(a["sigPhi2"][ei], dtype=np.float32)
        ct = np.asarray(a["cType"][ei]); co = np.asarray(a["cObj"][ei]); cq = np.asarray(a["cQ"][ei])
        idp = np.asarray(a["cIdPass"][ei]).astype(bool); eta = np.asarray(a["cInEta"][ei]).astype(bool)
        cp = np.asarray(a["cProp"][ei]).astype(int)
        ce = np.asarray(a["cEta2"][ei], dtype=np.float32); cf = np.asarray(a["cPhi2"][ei], dtype=np.float32)
        for gated, objs, types, inds in ((True, a["matchObj"][ei], a["matchType"][ei], a["indObj"][ei]),
                                         (False, a["matchObjNoEta"][ei], a["matchTypeNoEta"][ei], a["indObjNoEta"][ei])):
            if k < oracle_events:
                elig = idp & (eta if gated else True)
                indep, hier, tier = cme.oracle(gq, gp, ge, gf, ct, co, cq, (cp * elig).astype(int), ce, cf)
                ind_flat = np.asarray(inds).reshape(3, 2)  # [type][gen]
                if not (np.array_equal(hier, objs) and np.array_equal(tier, types)
                        and np.array_equal(indep.T, ind_flat)):
                    mism_oracle += 1
                if gated:
                    checked += 1
            if not gated and old_sel is not None:
                for g, gi in enumerate(a["sigIndex"][ei]):
                    ref = old_sel.get((int(ei), int(gi)))
                    mine = (int(objs[g]), int(types[g]), *(int(inds[2 * t + g]) for t in range(3)))
                    if ref is None or ref != mine:
                        mism_old += 1
    out.update(oracle_events=checked, oracle_mismatch=mism_oracle, old_mismatch=mism_old if old_sel else None)
    # bookkeeping
    st, ok = a["bbStatus"], a["twoAcc"]
    matched = np.array([len(m) == 2 and m[0] >= 0 and m[1] >= 0 for m in a["matchComp"]])
    assert np.array_equal(matched, st > 0), "bbStatus>0 must equal both matched"
    assert ok[st > 0].all()
    assert a["anyNPairs"].sum() == npairs, "pair tree rows != sum(anyNPairs)"
    assert np.all(a["anyNValid"] <= a["anyNPairs"])
    # combos: one row per ordered (a,b) where gen0 has an independent type-a and gen1 a type-b match
    expected = []
    for ei in two:
        ind = np.asarray(a["indObj"][ei]).reshape(3, 2)
        expected += [(int(ei), ta, tb) for ta in range(3) if ind[ta, 0] >= 0 for tb in range(3) if ind[tb, 1] >= 0]
    got = list(zip(combos["kEvent"].tolist(), combos["kTypeA"].tolist(), combos["kTypeB"].tolist()))
    assert len(set(got)) == len(got), "duplicate (event, typeA, typeB) combo rows"
    assert sorted(got) == sorted(expected), "combo (event, typeA, typeB) set differs from independent matches"
    exp = len(expected)
    assert np.all(ok[combos["kEvent"]]), "combo in non-twoAcc event"
    out.update(combos=int(exp), combos_valid=int((combos["kStatus"] == 3).sum()))
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
        res = check_file(path, old, args.oracle_events)
        print(res, flush=True)
        if res["oracle_mismatch"] or res["old_mismatch"]:
            raise SystemExit("MISMATCH in " + str(path))


if __name__ == "__main__":
    main()
