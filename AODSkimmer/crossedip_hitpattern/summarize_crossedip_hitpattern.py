#!/usr/bin/env python3
"""Summarize CrossedIPHitPatternCheck output: exact crossedIP (TrackExtra) vs HitPattern proxy.

  python3 summarize_crossedip_hitpattern.py crossedip_hitpattern_*.root

Primary cohort: inner tracks with exactly two displacedMuon users (the interim policy's pairs).
Secondary: every distinct displacedMuon inner track. Values: -1 undefined (exact: no TrackExtra),
0 not crossed, 1 crossed. Policy (DECISIONS.md 2026-10-15): crossed -> drop the higher index (B);
not crossed -> drop the one without a global track if exactly one has it; otherwise unresolved.
"""
import sys
from collections import Counter

import ROOT

ROOT.gROOT.SetBatch(True)
SEQ = {0: "barrel-only", 1: "endcap-only", 2: "mixed"}
UNDEF = {0: "-", 1: "<3 valid hits", 2: "HitPattern full", 3: "endcap d not finite", 4: "layer not in map"}


def table(counts, title):
    print(f"\n{title}  (rows exact, columns proxy)")
    print(f"{'':>10}" + "".join(f"{f'proxy={p}':>10}" for p in (-1, 0, 1)) + f"{'total':>10}")
    for e in (-1, 0, 1):
        row = [counts.get((e, p), 0) for p in (-1, 0, 1)]
        print(f"{f'exact={e}':>10}" + "".join(f"{n:>10}" for n in row) + f"{sum(row):>10}")
    defined = sum(counts.get((e, p), 0) for e in (0, 1) for p in (0, 1))
    agree = counts.get((0, 0), 0) + counts.get((1, 1), 0)
    print(f"both defined: {defined}, agree {agree}, disagree {defined - agree}")


def decision(c, ga, gb):
    if c == 1:
        return "drop B"
    if c == 0 and ga != gb:
        return "drop A" if gb else "drop B"
    return "unresolved"


def failure_mode(r):
    e, p = r["exact"], r["proxy"]
    if p == -1:
        return "proxy undefined: " + UNDEF[r["undefReason"]]
    if e == 1 and p == 0:
        if not r["interiorMin"]:
            return "missed crossing: no sampled interior minimum"
        return "missed crossing: endpoint d <= 3 cm" + (" (PXB1 endpoint)" if r["firstValidPXB1"] or r["lastValidPXB1"] else "")
    if e == 0 and p == 1:
        if min(r["innerRho"], r["outerRho"]) <= 3:
            return "false crossing: exact endpoint rho <= 3 cm"
        return "false crossing: proxy minimum, no exact sign flip"
    return "other"


def main(files):
    chain = ROOT.TChain("crossedIPHitPatternCheck/tracks")
    for f in files:
        chain.Add(f)
    df = ROOT.RDataFrame(chain)
    print(f"files: {len(files)}, tracks: {df.Count().GetValue()}")

    # secondary: all inner tracks
    cols = ["nUsers", "exact", "proxy", "seqClass"]
    allv = df.AsNumpy(cols)
    allc, allseq = Counter(), {k: Counter() for k in SEQ}
    for nu, e, p, s in zip(*(allv[c] for c in cols)):
        allc[(int(e), int(p))] += 1
        allseq[int(s)][(int(e), int(p))] += 1
    print("users per track:", dict(sorted(Counter(int(n) for n in allv["nUsers"]).items())))
    table(allc, "ALL displacedMuon inner tracks (secondary)")
    for k, name in SEQ.items():
        table(allseq[k], f"ALL, {name}")

    # primary: two-user tracks
    pcols = ["run", "lumi", "event", "idxA", "idxB", "globalA", "globalB", "exact", "proxy", "undefReason",
             "interiorMin", "seqClass", "nValid", "nTrackHits", "saturated", "firstType", "lastType",
             "firstValidPXB1", "lastValidPXB1", "dFirst", "dLast", "innerRho", "outerRho", "innerZ", "outerZ",
             "pt", "eta", "hitCode", "hitD"]
    pv = df.Filter("nUsers == 2").AsNumpy(pcols)
    rows = [{c: pv[c][i] for c in pcols} for i in range(len(pv["exact"]))]
    pc, pseq = Counter(), {k: Counter() for k in SEQ}
    dec = Counter()
    for r in rows:
        key = (int(r["exact"]), int(r["proxy"]))
        pc[key] += 1
        pseq[int(r["seqClass"])][key] += 1
        dec[(decision(key[0], r["globalA"], r["globalB"]), decision(key[1], r["globalA"], r["globalB"]))] += 1
    table(pc, "TWO-USER shared inner tracks (primary)")
    for k, name in SEQ.items():
        table(pseq[k], f"TWO-USER, {name}")

    print("\nTWO-USER policy decisions (exact -> proxy): count")
    for (a, b), n in sorted(dec.items()):
        print(f"  {a:>10} -> {b:<10} {n}{'' if a == b else '   CHANGED'}")
    print(f"changed decisions: {sum(n for (a, b), n in dec.items() if a != b)} of {len(rows)}")

    bad = [r for r in rows if int(r["exact"]) != -1 and int(r["exact"]) != int(r["proxy"])]
    print(f"\nTWO-USER disagreements (exact defined): {len(bad)}")
    print("by failure mode:", dict(Counter(failure_mode(r) for r in bad)))
    print("by sequence class:", dict(Counter(SEQ[int(r["seqClass"])] for r in bad)))
    for r in bad:
        codes = " ".join(str(c) for c in r["hitCode"])
        ds = " ".join(f"{d:.1f}" for d in r["hitD"])
        print(f"- {r['run']}:{r['lumi']}:{r['event']} idx {r['idxA']},{r['idxB']} exact {r['exact']} proxy {r['proxy']} "
              f"[{failure_mode(r)}] {SEQ[int(r['seqClass'])]} pt {r['pt']:.1f} eta {r['eta']:.2f} "
              f"inner rho/z {r['innerRho']:.1f}/{r['innerZ']:.1f} outer {r['outerRho']:.1f}/{r['outerZ']:.1f} "
              f"types first/last {r['firstType']}/{r['lastType']} PXB1 {r['firstValidPXB1']}/{r['lastValidPXB1']} "
              f"sat {r['saturated']}\n    codes: {codes}\n    d: {ds}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
