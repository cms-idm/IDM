#!/usr/bin/env python3
"""Summarize CrossedIPHitPatternCheck output: exact crossedIP (TrackExtra) vs HitPattern proxy.

  python3 summarize_crossedip_hitpattern.py [--proxy proxyFix] crossedip_hitpattern_*.root

--proxy chooses the column compared with the exact flag: proxy (HitPattern only, default) or proxyFix
(proxy OR rule E). With proxyFix, a rule E section follows: recovery of the HitPattern misses, false
firings, and the margins.

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


def failure_mode(r, col):
    e, p = r["exact"], r[col]
    if p == -1:
        return "proxy undefined: " + UNDEF[r["undefReason"]]
    if e == 1 and p == 0:
        if col == "proxyFix" and r["firstEndcap"]:
            return "missed crossing: endcap first hit, rule E not met"
        if not r["interiorMin"]:
            return "missed crossing: no sampled interior minimum"
        return "missed crossing: endpoint d <= 3 cm" + (" (PXB1 endpoint)" if r["firstValidPXB1"] or r["lastValidPXB1"] else "")
    if e == 0 and p == 1 and col == "proxyFix" and r["ruleE"] and not r["proxy"]:
        return "false crossing: rule E only"
    if e == 0 and p == 1:
        if min(r["innerRho"], r["outerRho"]) <= 3:
            return "false crossing: exact endpoint rho <= 3 cm"
        return "false crossing: proxy minimum, no exact sign flip"
    return "other"


def quantiles(v):
    v = sorted(v)
    if not v:
        return "none"
    q = lambda f: v[min(len(v) - 1, int(f * len(v)))]
    return f"n {len(v)} min {v[0]:.1f} q25 {q(0.25):.1f} median {q(0.5):.1f} q75 {q(0.75):.1f} max {v[-1]:.1f}"


def main(files, col):
    print(f"compared column: {col}")
    chain = ROOT.TChain("crossedIPHitPatternCheck/tracks")
    for f in files:
        chain.Add(f)
    df = ROOT.RDataFrame(chain)
    print(f"files: {len(files)}, tracks: {df.Count().GetValue()}")

    # secondary: all inner tracks
    cols = ["nUsers", "exact", col, "seqClass"]
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
             "pt", "eta", "hitCode", "hitD", "proxy", "proxyFix", "ruleE", "firstEndcap", "lastEndcap",
             "marginFirst", "marginLast", "dzError", "rhoFirstBound", "vz", "dxy"]
    pv = df.Filter("nUsers == 2").AsNumpy(pcols)
    rows = [{c: pv[c][i] for c in pcols} for i in range(len(pv["exact"]))]
    pc, pseq = Counter(), {k: Counter() for k in SEQ}
    dec = Counter()
    for r in rows:
        key = (int(r["exact"]), int(r[col]))
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

    hit = lambda r: ("E" if r["firstEndcap"] else "B") + "->" + ("E" if r["lastEndcap"] else "B")
    split = Counter((hit(r), int(r["exact"]), int(r[col])) for r in rows)
    print("\nTWO-USER by first->last hit type (B barrel, E endcap): (type, exact, compared): count")
    for k, n in sorted(split.items()):
        print(f"  {k}: {n}")
    bad = [r for r in rows if int(r["exact"]) != -1 and int(r["exact"]) != int(r[col])]
    print(f"\nTWO-USER disagreements (exact defined): {len(bad)}")
    print("by failure mode:", dict(Counter(failure_mode(r, col) for r in bad)))
    print("by sequence class:", dict(Counter(SEQ[int(r["seqClass"])] for r in bad)))
    for r in bad:
        codes = " ".join(str(c) for c in r["hitCode"])
        ds = " ".join(f"{d:.1f}" for d in r["hitD"])
        print(f"- {r['run']}:{r['lumi']}:{r['event']} idx {r['idxA']},{r['idxB']} exact {r['exact']} {col} {r[col]} "
              f"[{failure_mode(r, col)}] {hit(r)} vz {r['vz']:.1f} dzErr {r['dzError']:.2f} dxy {r['dxy']:.2f} {SEQ[int(r['seqClass'])]} pt {r['pt']:.1f} eta {r['eta']:.2f} "
              f"inner rho/z {r['innerRho']:.1f}/{r['innerZ']:.1f} outer {r['outerRho']:.1f}/{r['outerZ']:.1f} "
              f"types first/last {r['firstType']}/{r['lastType']} PXB1 {r['firstValidPXB1']}/{r['lastValidPXB1']} "
              f"sat {r['saturated']}\n    codes: {codes}\n    d: {ds}")


    if col == "proxyFix":
        rule_e(df, rows)


def rule_e(df, rows):
    print("\nRULE E")
    miss = [r for r in rows if r["exact"] == 1 and r["proxy"] == 0]
    print(f"TWO-USER HitPattern misses: {len(miss)}; first hit endcap {sum(1 for r in miss if r['firstEndcap'])}, "
          f"barrel {sum(1 for r in miss if not r['firstEndcap'])} (not recoverable by rule E)")
    rec = [r for r in miss if r["ruleE"]]
    print(f"recovered by rule E: {len(rec)} of {sum(1 for r in miss if r['firstEndcap'])} with an endcap first hit")
    print("  recovered marginFirst [cm]:", quantiles([r["marginFirst"] for r in rec]))
    print("  recovered marginFirst / dzError:", quantiles([r["marginFirst"] / r["dzError"] for r in rec if r["dzError"] > 0]))
    left = [r for r in miss if r["firstEndcap"] and not r["ruleE"]]
    print("  endcap-first not recovered, marginFirst [cm]:", quantiles([r["marginFirst"] for r in left]))
    false2 = [r for r in rows if r["exact"] == 0 and r["ruleE"] and r["proxy"] == 0]
    print(f"TWO-USER rule E new firings (proxy 0) on exact 0: {len(false2)}")
    print("  marginFirst [cm]:", quantiles([r["marginFirst"] for r in false2]))
    # secondary, separating rule E's false firings from the HitPattern proxy's
    v = df.Filter("exact != -1").AsNumpy(["exact", "proxy", "ruleE", "marginFirst", "marginLast", "dzError",
                                         "lastEndcap", "firstEndcap"])
    n = len(v["exact"])
    fe = [i for i in range(n) if v["exact"][i] == 0 and v["ruleE"][i] and v["proxy"][i] == 0]
    tp = [i for i in range(n) if v["exact"][i] == 1 and v["ruleE"][i] and v["proxy"][i] == 0]
    print(f"ALL: rule E new firings (proxy 0): exact 1 (recovered) {len(tp)}, exact 0 (false) {len(fe)}")
    print(f"  false, by last hit endcap/barrel: {sum(1 for i in fe if v['lastEndcap'][i])}/{sum(1 for i in fe if not v['lastEndcap'][i])}")
    print("  false marginFirst [cm]:", quantiles([v["marginFirst"][i] for i in fe]))
    print("  false marginFirst / dzError:", quantiles([v["marginFirst"][i] / v["dzError"][i] for i in fe if v["dzError"][i] > 0]))
    print("  recovered marginFirst [cm]:", quantiles([v["marginFirst"][i] for i in tp]))
    print("  recovered marginFirst / dzError:", quantiles([v["marginFirst"][i] / v["dzError"][i] for i in tp if v["dzError"][i] > 0]))
    for m in (0, 1, 2, 5):
        print(f"  with marginFirst > {m} cm: recovered {sum(1 for i in tp if v['marginFirst'][i] > m)}, "
              f"false {sum(1 for i in fe if v['marginFirst'][i] > m)}; two-user recovered "
              f"{sum(1 for r in rec if r['marginFirst'] > m)}, two-user false {sum(1 for r in false2 if r['marginFirst'] > m)}")


if __name__ == "__main__":
    args = sys.argv[1:]
    col = "proxy"
    if args[:1] == ["--proxy"]:
        col, args = args[1], args[2:]
    if not args or col not in ("proxy", "proxyFix"):
        sys.exit(__doc__)
    main(args, col)
