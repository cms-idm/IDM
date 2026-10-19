#!/usr/bin/env python3
"""Summarize CrossedIPHitPatternCheck output: exact crossedIP (TrackExtra) vs HitPattern proxy.

  python3 summarize_crossedip_hitpattern.py [--proxy proxyFix] crossedip_hitpattern_*.root
  python3 summarize_crossedip_hitpattern.py --side crossedip_hitpattern_*.root

--side: muon-system side test vs the exact crossedIP on the two-user tracks (kb/studies/
shared-track-duplicates.md, "Muon-system side test vs crossedIP"). Sources M (all associated chambers),
Arb (arbitrated DT/CSC segments), S (standalone endpoints). Copy class: ahead / behind / mixed (both
signs or s = 0) / none; pair class, in order: undefined (a copy none), mixed (a copy mixed), opposite,
same. Agreement: exact 1 with opposite, exact 0 with same.

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


SOURCES = ("M", "Arb", "S")
PAIR = ("opposite", "same", "mixed", "undefined")


def copy_class(r, src, c):
    a, b, z = (int(r[f"side{src}_{c}_{n}"]) for n in ("nAhead", "nBehind", "nPlane"))
    if a + b + z == 0:
        return "none"
    if z or (a and b):
        return "mixed"
    return "ahead" if a else "behind"


def pair_class(r, src):
    ca, cb = copy_class(r, src, "A"), copy_class(r, src, "B")
    if "none" in (ca, cb):
        return "undefined"
    if "mixed" in (ca, cb):
        return "mixed"
    return "opposite" if ca != cb else "same"


def agrees(r, src):
    pc = pair_class(r, src)
    return (r["exact"] == 1 and pc == "opposite") or (r["exact"] == 0 and pc == "same")


def binned(rows, src, key, edges, label):
    print(f"  agreement vs {label} (all pairs per bin; agree / total):")
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in rows if lo <= key(r) < hi]
        print(f"    [{lo}, {hi}): {sum(agrees(r, src) for r in sel)} / {len(sel)}")


def side(files):
    chain = ROOT.TChain("crossedIPHitPatternCheck/tracks")
    for f in files:
        chain.Add(f)
    df = ROOT.RDataFrame(chain).Filter("nUsers == 2")
    cols = ["run", "lumi", "event", "idxA", "idxB", "exact", "pt", "eta", "sideIdenticalM"] + [
        f"side{s}_{c}_{n}" for s in SOURCES for c in "AB"
        for n in ("nAhead", "nBehind", "nPlane", "nDropped", "sMin", "sMax", "absSMin")]
    v = df.AsNumpy(cols)
    rows = [{c: v[c][i] for c in cols} for i in range(len(v["exact"]))]
    print(f"files: {len(files)}, two-user tracks: {len(rows)}; exact: {dict(Counter(int(r['exact']) for r in rows))}")
    print(f"source M identical chamber sets: {sum(int(r['sideIdenticalM']) == 1 for r in rows)}")
    for src in SOURCES:
        print(f"\n=== source {src} ===")
        t = Counter((int(r["exact"]), pair_class(r, src)) for r in rows)
        print(f"{'':>10}" + "".join(f"{p:>11}" for p in PAIR) + f"{'total':>8}")
        for e in (-1, 0, 1):
            n = [t.get((e, p), 0) for p in PAIR]
            if sum(n):
                print(f"{f'exact={e}':>10}" + "".join(f"{x:>11}" for x in n) + f"{sum(n):>8}")
        print(f"agree (exact 1 & opposite, exact 0 & same): {sum(agrees(r, src) for r in rows)} of {len(rows)}")
        cc = Counter(copy_class(r, src, c) for r in rows for c in "AB")
        print("copy classes (A and B together):", dict(cc))
        dropped = sum(int(r[f"side{src}_{c}_nDropped"]) for r in rows for c in "AB")
        plane = sum(int(r[f"side{src}_{c}_nPlane"]) for r in rows for c in "AB")
        print(f"items dropped (no geometry / non-finite s): {dropped}; items with s = 0: {plane}")
        binned(rows, src, lambda r: r["pt"], [0, 3, 5, 10, 20, 1e9], "shared-track pT [GeV]")
        binned(rows, src, lambda r: abs(r["eta"]), [0, 0.9, 1.2, 2.1, 2.5, 9], "shared-track |eta|")
        opp = [r for r in rows if pair_class(r, src) == "opposite"]
        aheadA = sum(copy_class(r, src, "A") == "ahead" for r in opp)
        print(f"opposite pairs: ahead copy is the lower index (A) in {aheadA} of {len(opp)} (information only)")
        for cls in PAIR[:3]:
            vals = sorted(min(float(r[f"side{src}_{c}_absSMin"]) for c in "AB") for r in rows if pair_class(r, src) == cls)
            if vals:
                print(f"  smallest |s| over both copies [cm], {cls}: min {vals[0]:.0f} median {vals[len(vals) // 2]:.0f} max {vals[-1]:.0f} (n {len(vals)})")
        bad = [r for r in rows if r["exact"] != -1 and not agrees(r, src) and pair_class(r, src) in ("opposite", "same")]
        print(f"disagreements with both copies classified: {len(bad)}")
        for r in bad:
            print(f"  - {r['run']}:{r['lumi']}:{r['event']} idx {r['idxA']},{r['idxB']} exact {r['exact']} {pair_class(r, src)} "
                  f"pt {r['pt']:.1f} eta {r['eta']:.2f} A {copy_class(r, src, 'A')} "
                  f"s [{r[f'side{src}_A_sMin']:.0f}, {r[f'side{src}_A_sMax']:.0f}] B {copy_class(r, src, 'B')} "
                  f"s [{r[f'side{src}_B_sMin']:.0f}, {r[f'side{src}_B_sMax']:.0f}] identicalM {r['sideIdenticalM']}")
    # S vs M where both are classified
    both = [r for r in rows if pair_class(r, "S") in ("opposite", "same") and pair_class(r, "M") in ("opposite", "same")]
    print(f"\nS vs M where both classified: {sum(pair_class(r, 'S') == pair_class(r, 'M') for r in both)} agree of {len(both)}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--side"]:
        if len(args) < 2:
            sys.exit(__doc__)
        side(args[1:])
        sys.exit(0)
    col = "proxy"
    if args[:1] == ["--proxy"]:
        col, args = args[1], args[2:]
    if not args or col not in ("proxy", "proxyFix"):
        sys.exit(__doc__)
    main(args, col)
