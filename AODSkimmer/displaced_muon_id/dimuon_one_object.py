#!/usr/bin/env python3
"""Separation-dependent loss of a signal dimuon leg, from saved PreselectedVertexing outputs (no MINIAOD processing).

python3 -I dimuon_one_object.py analyze --input-dir RUN/inputs --output OUT.root --revision <sha> [--samples a,b]
python3 -I dimuon_one_object.py validate --input-dir RUN/inputs --output OUT.root --samples a,b
python3 -I dimuon_one_object.py plot --output OUT.root --plot-dir DIR --style-dir DIR_WITH_cms_style [--log]

Study note: kb/studies/dimuon-one-object.md. Inputs: met_proxy_20261022 (hierarchy G > I > DSA, 5a2b10e).
Population: two-accepted events (twoAcc). Raw, unweighted. Categories: 2, 1 or 0 legs with matchObj >= 0, at the
adopted matching (station 2, reco-gen charge agreement, dR < 0.06 G/I, < 0.10 DSA, staged greedy, whole-object claim).
s2 = hypot(deta, wrapped dphi) of the gen station-2 positions; undefined if a gen muon failed propagation (stored in
the s2 underflow bin, so it stays visible and in every total).
Decisive test: close pairs (s2 < 0.2) against p = eps(leg 1) * eps(leg 2), with eps from legs of separated pairs
(s2 >= 0.3), one map pooled over all samples. D = sum(p - both matched). Closure: map from s2 >= 1.0 predicts
0.3 <= s2 < 1.0; b = |non-closure| / two-accepted. Interval: Poisson event bootstrap refitting the map.
"""
import argparse
from array import array
import math
from pathlib import Path
import re
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python3 -I drops the script directory
from preselected_vertexing_talk import flowcounts, write_hist

INPUT_REV, INPUT_ORDER = "5a2b10e", "global>inner>outer(DSA)"
S_EDGES = np.array([0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.6, 1.0, 2.0])
D0_EDGES = np.array([0, 1, 2, 3, 5, 10, 20, 50, 100, 740.])
AXES = {"s2": S_EDGES, "genDR": S_EDGES, "d0max": D0_EDGES}
PT_EDGES, LXY_EDGES = (3., 5., 10., 20.), (10., 60.)  # single-leg map cells, with the barrel/endcap split at |eta| 1.2
NCELL = 5 * 3 * 2
CLOSE, SEPARATED, CLOSURE_MAP = 0.2, 0.3, 1.0
THRESHOLD = 0.02
CLASSES = "ABCDE"
STATS = ("den", "n2", "n1", "n0", "supp", "supp2", "pred") + tuple("sec" + c for c in CLASSES)


def wrap(d):
    return (d + math.pi) % (2 * math.pi) - math.pi


def classify(leg, claimed, q, st2, comps):
    """Secondary table for the unmatched leg of a one-leg event. Returns (class, closure violated).
    comps: (obj, type, q, idPass, inEta, prop, eta2, phi2); type 1 is DSA."""
    if st2 is None:
        return "E", False
    best, closure = None, False
    for i, (obj, typ, cq, idp, ineta, prop, eta2, phi2) in enumerate(comps):
        if not (idp and ineta and prop):
            continue
        d = math.hypot(st2[0] - eta2, wrap(st2[1] - phi2))
        if d >= (0.10 if typ == 1 else 0.06):
            continue
        if obj != claimed and cq == q:
            closure = True  # must never happen: hierarchy() would have matched it
        if best is None or (d, i) < best[:2]:
            best = (d, i, obj == claimed, cq == q)
    if best is None:
        return "D", closure
    if best[2]:
        return ("A" if best[3] else "B"), closure
    return ("C" if not best[3] else "F"), closure


def load(args):
    import awkward as ak
    import uproot
    paths = sorted(Path(args.input_dir).glob("*.root"))
    if args.samples:
        keep = set(args.samples.split(","))
        paths = [p for p in paths if p.stem in keep]
        if len(paths) != len(keep):
            raise RuntimeError("Missing requested samples")
    elif len(paths) != args.expected_samples:
        raise RuntimeError("Unexpected number of inputs: %d" % len(paths))
    br = ["twoAcc", "sigQ", "sigPt", "sigEta", "sigLxy", "sigD0", "sigProp", "sigEta2", "sigPhi2", "matchObj", "genDR",
          "cObj", "cType", "cQ", "cIdPass", "cInEta", "cProp", "cEta2", "cPhi2"]
    names, parts, total = [], [], 0
    for si, path in enumerate(paths):
        with uproot.open(path) as f:
            sample = str(f["preselectedVertexing/sample"].member("fTitle"))
            if sample in names or not re.fullmatch(r"Mchi-[\dp]+_dMchi-[\dp]+_ctau-[\dp]+", sample):
                raise RuntimeError("Bad/duplicate sample " + sample)
            if str(f["preselectedVertexing/revision"].member("fTitle")) != INPUT_REV or \
                    str(f["preselectedVertexing/hierarchyOrder"].member("fTitle")) != INPUT_ORDER:
                raise RuntimeError("Unexpected input revision or order in " + sample)
            a = f["preselectedVertexing/events"].arrays(br, library="ak")
        names.append(sample)
        total += len(a)
        two = np.asarray(a["twoAcc"]).astype(bool)
        if np.any(ak.num(a["matchObj"][~two]) > 0):
            raise RuntimeError("matching filled outside twoAcc in " + sample)
        a = a[two]
        for b in ("sigQ", "sigPt", "sigEta", "sigLxy", "sigD0", "sigProp", "sigEta2", "sigPhi2", "matchObj"):
            if np.any(ak.num(a[b]) != 2):
                raise RuntimeError("%s length != 2 in %s" % (b, sample))
        mo = ak.to_numpy(a["matchObj"]).astype(int)
        prop = ak.to_numpy(a["sigProp"]).astype(bool)
        eta2, phi2 = ak.to_numpy(a["sigEta2"]).astype(float), ak.to_numpy(a["sigPhi2"]).astype(float)
        both_prop = prop.all(axis=1)
        s2 = np.full(len(a), np.nan)
        dphi = (phi2[:, 0] - phi2[:, 1] + np.pi) % (2 * np.pi) - np.pi
        s2[both_prop] = np.hypot(eta2[:, 0] - eta2[:, 1], dphi)[both_prop]
        sec = np.full(len(a), -1)
        q = ak.to_list(a["sigQ"])
        one = np.flatnonzero((mo >= 0).sum(axis=1) == 1)
        cl = [ak.to_list(a[b][one]) for b in ("cObj", "cType", "cQ", "cIdPass", "cInEta", "cProp", "cEta2", "cPhi2")]
        for k, i in enumerate(one):
            g = 0 if mo[i, 0] < 0 else 1
            st2 = (float(eta2[i, g]), float(phi2[i, g])) if prop[i, g] else None
            c, bad = classify(g, mo[i, 1 - g], q[i][g], st2, list(zip(*[x[k] for x in cl])))
            if bad or c == "F":
                raise RuntimeError("Closure check failed: unclaimed same-charge candidate in %s event %d" % (sample, i))
            sec[i] = CLASSES.index(c)
        parts.append(dict(si=np.full(len(a), si), s2=s2, genDR=np.asarray(a["genDR"], float),
                          d0max=np.abs(ak.to_numpy(a["sigD0"]).astype(float)).max(axis=1),
                          pt=ak.to_numpy(a["sigPt"]).astype(float), eta=ak.to_numpy(a["sigEta"]).astype(float),
                          lxy=ak.to_numpy(a["sigLxy"]).astype(float), m=mo >= 0, sec=sec))
    ev = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return names, ev, total


def cells(ev):
    ip = np.searchsorted(PT_EDGES, ev["pt"], side="right")
    il = np.searchsorted(LXY_EDGES, ev["lxy"], side="right")
    ir = (np.abs(ev["eta"]) >= 1.2).astype(int)
    c = ip * 6 + il * 2 + ir
    if c.max() >= NCELL or c.min() < 0:
        raise RuntimeError("Bad map cell")
    return c


def build_map(c, m, mask, w=None):
    ww = np.ones(c.shape) if w is None else np.repeat(w[:, None], 2, axis=1)
    den = np.bincount(c[mask].ravel(), weights=ww[mask].ravel(), minlength=NCELL)
    num = np.bincount(c[mask].ravel(), weights=(ww * m)[mask].ravel(), minlength=NCELL)
    return num, den


def eff(num, den, fallback=None):
    out = np.zeros(NCELL) if fallback is None else fallback.copy()
    out[den > 0] = num[den > 0] / den[den > 0]
    return out


def group_labels(names):
    delta = sorted({n.split("dMchi-")[1].split("_")[0] for n in names}, key=lambda d: float(d.replace("p", ".")))
    return delta, {n: delta.index(n.split("dMchi-")[1].split("_")[0]) + 1 for n in names}  # group 0 = overall


def group_sum(idx, vals, n):
    """Sum per group: group 0 is all entries, groups 1.. are the splittings."""
    s = np.bincount(idx, weights=vals, minlength=n)
    s[0] = vals.sum()
    return s


def analyze(args):
    import ROOT
    ROOT.gROOT.SetBatch(True)
    t0 = time.time()
    names, ev, total = load(args)
    print("loaded %d samples, %d events, %d two-accepted (%.1f s)" % (len(names), total, len(ev["si"]), time.time() - t0))
    N = len(ev["si"])
    delta, dmap = group_labels(names)
    labels = ["overall"] + ["delta_" + d for d in delta]
    ng = len(labels)
    dg = np.array([dmap[n] for n in names])[ev["si"]]
    c = cells(ev)
    m = ev["m"]
    both = m.all(axis=1)
    nleg = m.sum(axis=1)
    defined = np.isfinite(ev["s2"])
    s2 = np.where(defined, ev["s2"], -1.0)  # -1 = undefined, s2 underflow bin
    sep, close = s2 >= SEPARATED, defined & (s2 < CLOSE)

    num, den = build_map(c, m, sep)
    e = eff(num, den)
    supported = (den[c[:, 0]] > 0) & (den[c[:, 1]] > 0)
    p = e[c[:, 0]] * e[c[:, 1]]
    cn, cd = build_map(c, m, s2 >= CLOSURE_MAP)
    ce = eff(cn, cd)
    tgt = (s2 >= SEPARATED) & (s2 < CLOSURE_MAP) & (cd[c[:, 0]] > 0) & (cd[c[:, 1]] > 0)
    cp_ = ce[c[:, 0]] * ce[c[:, 1]]

    gidx_close = dg[close & supported]
    Ng = group_sum(dg, np.ones(N), ng)
    res = dict(N=Ng, undefined=group_sum(dg, (~defined).astype(float), ng),
               close=group_sum(dg, close.astype(float), ng),
               close_unsupported=group_sum(dg, (close & ~supported).astype(float), ng),
               obs=group_sum(gidx_close, both[close & supported].astype(float), ng),
               pred=group_sum(gidx_close, p[close & supported], ng),
               clo_obs=group_sum(dg[tgt], both[tgt].astype(float), ng), clo_pred=group_sum(dg[tgt], cp_[tgt], ng))
    res["D"] = res["pred"] - res["obs"]
    res["frac"] = res["D"] / Ng
    res["b"] = np.abs(res["clo_pred"] - res["clo_obs"]) / Ng

    rng = np.random.default_rng(args.seed)
    reps = np.zeros((args.bootstrap, ng))
    cc, mm, cb = c[close & supported], both[close & supported], dg[close & supported]
    sepi = np.flatnonzero(sep)
    n_fallback = 0
    t1 = time.time()
    for r in range(args.bootstrap):
        w = rng.poisson(1.0, N).astype(float)
        wc = w[close & supported]
        nr = np.bincount(c[sepi].ravel(), weights=(np.repeat(w[sepi, None], 2, axis=1) * m[sepi]).ravel(), minlength=NCELL)
        dr = np.bincount(c[sepi].ravel(), weights=np.repeat(w[sepi, None], 2, axis=1).ravel(), minlength=NCELL)
        n_fallback += int(np.any((dr == 0) & (den > 0)))
        er = eff(nr, dr, e)
        d = group_sum(cb, wc * (er[cc[:, 0]] * er[cc[:, 1]] - mm), ng)
        reps[r] = d / group_sum(dg, w, ng)
    res["L"], res["U"] = (np.percentile(reps, 2.5, axis=0), np.percentile(reps, 97.5, axis=0)) if args.bootstrap else \
        (res["frac"].copy(), res["frac"].copy())
    res["lo"], res["hi"] = res["L"] - res["b"], res["U"] + res["b"]
    print("bootstrap %d replicas, %d with an empty nominal map cell (%.1f s)" % (args.bootstrap, n_fallback, time.time() - t1))
    decision = "GO" if np.any(res["lo"] > THRESHOLD) else "STOP" if np.all(res["hi"] < THRESHOLD) else "INCONCLUSIVE"

    # Histograms per sample and axis; groups are sums of samples, so every group uses the same counts.
    axis_vals = {"s2": s2, "genDR": ev["genDR"], "d0max": ev["d0max"]}
    counts = {}
    masks = {"den": np.ones(N, bool), "n2": nleg == 2, "n1": nleg == 1, "n0": nleg == 0,
             "supp": defined & supported, "supp2": defined & supported & both}
    masks.update({"sec" + k: ev["sec"] == i for i, k in enumerate(CLASSES)})
    for ax, edges in AXES.items():
        counts[ax] = np.zeros((len(names), len(STATS), len(edges) + 1))
        bins = np.searchsorted(edges, axis_vals[ax], side="right")
        key = ev["si"] * (len(edges) + 1) + bins
        size = len(names) * (len(edges) + 1)
        for i, st in enumerate(STATS):
            if st == "pred":
                k = masks["supp"]
                counts[ax][:, i] = np.bincount(key[k], weights=p[k], minlength=size).reshape(len(names), -1)
            else:
                k = masks[st]
                counts[ax][:, i] = np.bincount(key[k], minlength=size).reshape(len(names), -1)
        c0 = counts[ax]
        if not np.all(c0[:, 1:4].sum(axis=1) == c0[:, 0]) or c0[:, 0].sum() != N:
            raise RuntimeError("Category sums fail on " + ax)
        if not np.all(c0[:, 7:].sum(axis=1) == c0[:, 2]):
            raise RuntimeError("Secondary classes do not sum to one-leg events on " + ax)
    if not args.samples and (total != 720000 or N != 494255 or both.sum() != 199931):
        raise RuntimeError("Full-run counts differ from the mumu talk study: %d %d %d" % (total, N, both.sum()))

    groups = {"overall": np.ones(len(names), bool)}
    for lab, fn in (("delta", lambda n: n.split("dMchi-")[1].split("_")[0]), ("mass", lambda n: n.split("_ctau")[0]),
                    ("ctau", lambda n: n.split("_ctau-")[1]), ("sample", lambda n: n)):
        for v in sorted({fn(n) for n in names}):
            groups[lab + "_" + v] = np.array([fn(n) == v for n in names])

    f = ROOT.TFile.Open(args.output, "RECREATE")
    def named(k, v):
        ROOT.TNamed(k, str(v)).Write()
    named("revision", args.revision); named("input_revision", INPUT_REV); named("input_order", INPUT_ORDER)
    named("decision", decision); named("samples", ",".join(names)); named("bootstrap", args.bootstrap)
    named("seed", args.seed); named("empty_cell_replicas", n_fallback); named("threshold", THRESHOLD)
    named("definition", "twoAcc events, raw unweighted; s2 underflow = undefined; pred = sum of p over supp events; "
          "supp = s2 defined and both legs in a map cell with separated legs; map pooled over all samples.")
    for name, edges, vals in (("map_num", None, num), ("map_den", None, den), ("closure_map_num", None, cn),
                              ("closure_map_den", None, cd)):
        h = ROOT.TH1D(name, name + ";cell = 6*ipT + 2*iLxy + iendcap;legs", NCELL, 0, NCELL)
        for i in range(NCELL):
            h.SetBinContent(i + 1, float(vals[i])); h.SetBinError(i + 1, math.sqrt(vals[i]))
        h.Write()
    for k, v in res.items():
        h = ROOT.TH1D("result_" + k, k, ng, 0, ng)
        for i in range(ng):
            h.SetBinContent(i + 1, float(v[i])); h.GetXaxis().SetBinLabel(i + 1, labels[i])
        h.Write()
    for lab, sel in groups.items():
        for ax, edges in AXES.items():
            sums = counts[ax][sel].sum(axis=0)
            for i, st in enumerate(STATS):
                if st == "pred":
                    h = ROOT.TH1D("%s__%s__pred" % (lab, ax), "sum of p", len(edges) - 1, array("d", edges))
                    for b, x in enumerate(sums[i]):
                        h.SetBinContent(b, float(x))
                    h.Write()
                else:
                    write_hist("%s__%s__%s" % (lab, ax, st), st, edges, sums[i])
    f.Close()
    print("%-12s %8s %8s %6s %8s %8s %8s %9s %9s %9s %9s %9s" % ("group", "N", "close", "unsup", "obs", "pred", "D", "D/N", "L", "U", "b", "U+b"))
    for i, g in enumerate(labels):
        print("%-12s %8d %8d %6d %8d %8.1f %8.1f %9.5f %9.5f %9.5f %9.5f %9.5f" % (
            g, res["N"][i], res["close"][i], res["close_unsupported"][i], res["obs"][i], res["pred"][i], res["D"][i],
            res["frac"][i], res["L"][i], res["U"][i], res["b"][i], res["hi"][i]))
    print("undefined s2: %d of %d; closure overall obs %.0f pred %.1f" % (res["undefined"][0], N, res["clo_obs"][0], res["clo_pred"][0]))
    print("DECISION", decision, "(%.1f s)" % (time.time() - t0))


def validate(args):
    """Independent scalar recount on raw branches with PyROOT, against the saved per-sample histograms."""
    import ROOT
    ROOT.gROOT.SetBatch(True)
    out = ROOT.TFile.Open(args.output)
    for sample in args.samples.split(","):
        f = ROOT.TFile.Open(str(Path(args.input_dir) / (sample + ".root")))
        t = f.Get("preselectedVertexing/events")
        den, leg, secn = np.zeros(len(S_EDGES) + 1), np.zeros((3, len(S_EDGES) + 1)), np.zeros((5, len(S_EDGES) + 1))
        for e in t:
            if not e.twoAcc:
                continue
            ok = e.sigProp[0] and e.sigProp[1]
            s = -1.0
            if ok:
                dp = abs(e.sigPhi2[0] - e.sigPhi2[1])
                dp = min(dp, 2 * math.pi - dp)
                s = math.sqrt((e.sigEta2[0] - e.sigEta2[1]) ** 2 + dp ** 2)
            b = sum(1 for x in S_EDGES if x <= s)
            den[b] += 1
            n = int(e.matchObj[0] >= 0) + int(e.matchObj[1] >= 0)
            leg[n, b] += 1
            if n != 1:
                continue
            g = 0 if e.matchObj[0] < 0 else 1
            if not e.sigProp[g]:
                secn[4, b] += 1
                continue
            cand = []
            for i in range(e.cObj.size()):
                if not (e.cIdPass[i] and e.cInEta[i] and e.cProp[i]):
                    continue
                dp = abs(e.sigPhi2[g] - e.cPhi2[i])
                dp = min(dp, 2 * math.pi - dp)
                d = math.sqrt((e.sigEta2[g] - e.cEta2[i]) ** 2 + dp ** 2)
                if d < (0.10 if e.cType[i] == 1 else 0.06):
                    cand.append((d, i))
            if not cand:
                secn[3, b] += 1
                continue
            i = min(cand)[1]
            same_obj, same_q = e.cObj[i] == e.matchObj[1 - g], e.cQ[i] == e.sigQ[g]
            secn[0 if same_obj and same_q else 1 if same_obj else 2, b] += 1
        for st, arr in [("den", den), ("n0", leg[0]), ("n1", leg[1]), ("n2", leg[2])] + \
                [("sec" + k, secn[i]) for i, k in enumerate(CLASSES)]:
            h = out.Get("sample_%s__s2__%s" % (sample, st))
            saved = np.array([h.GetBinContent(i) for i in range(len(S_EDGES) + 1)])
            if not np.array_equal(saved, arr):
                raise RuntimeError("Mismatch %s %s: %s vs %s" % (sample, st, saved, arr))
        print("validated", sample, "two-accepted", int(den.sum()), "one-leg", int(leg[1].sum()))
        f.Close()
    print("VALIDATE PASS")


def plot(args):
    sys.path.insert(0, str(Path(args.style_dir).resolve()))
    import ROOT
    import cms_style as cs
    ROOT.gROOT.SetBatch(True)
    cs.setup(era="2022EE (13.6 TeV)", status="Simulation Private Work")
    out = Path(args.plot_dir)
    f = ROOT.TFile.Open(args.output)
    keys = [k.GetName() for k in f.GetListOfKeys()]
    groups = sorted({k.split("__")[0] for k in keys if k.endswith("__s2__den")})
    XT = {"s2": "Gen #DeltaR(#mu#mu) at station 2", "genDR": "Gen #DeltaR(#mu#mu) at production",
          "d0max": "Max gen |d_{0}| of the two muons [cm]"}
    get = lambda g, ax, st: f.Get("%s__%s__%s" % (g, ax, st))
    def lab(g):
        k, _, v = g.partition("_")
        v = v.replace("Mchi-", "").replace("_dMchi-", " / ").replace("_ctau-", ", c#tau ").replace("p", ".")
        return {"overall": "All signal samples", "delta": "#Deltam = %s GeV" % v, "mass": "m_{#chi} / #Deltam = %s GeV" % v,
                "ctau": "c#tau = %s mm" % v, "sample": "m_{#chi} / #Deltam = %s mm" % v}[k]
    FR = ((("n2", "2 legs matched"), cs.BLUE, 20), (("n1", "1 leg matched"), cs.ORANGE, 21), (("n0", "0 legs matched"), cs.GREEN, 22))
    for g in groups:
        kind = g.split("_")[0]
        d = out / kind
        d.mkdir(parents=True, exist_ok=True)
        for ax in AXES:
            s = [cs.eff_graph(get(g, ax, st), get(g, ax, "den"), col, 1, mk) for (st, _), col, mk in FR]
            if s and all(x.GetN() for x in s):
                cs.plot(str(d / ("fractions_%s_%s" % (ax, g))), s, [l for (_, l), _, _ in FR], XT[ax], "Fraction of two-accepted events",
                        lines=[lab(g)], log=args.log)
    def overlay(prefix, kind, name):
        gs = [g for g in groups if g.startswith(kind + "_")]
        d = out / "overlays"
        d.mkdir(parents=True, exist_ok=True)
        styles = [(cs.COLORS[i % 7], 20 + i // 7 * 4 + (i % 2)) for i in range(len(gs))]
        short = lambda g: lab(g).split("= ")[-1]
        for ax in AXES:
            ss = [(cs.eff_graph(get(g, ax, "n2"), get(g, ax, "den"), c, 1, m), short(g)) for g, (c, m) in zip(gs, styles)]
            ss = [x for x in ss if x[0].GetN()]
            cs.plot(str(d / ("%s_f2_%s" % (prefix, ax))), [x[0] for x in ss], [x[1] for x in ss], XT[ax],
                    "Fraction with 2 legs matched", header=name, log=args.log)
            for norm, yt in ((False, "Two-accepted events / bin"), (True, "Fraction of events / bin")):
                s = [cs.hist_graph(get(g, ax, "den"), c, step=True, norm=norm) for g, (c, m) in zip(gs, styles)]
                cs.plot(str(d / ("%s_den_%s%s" % (prefix, ax, "_norm" if norm else ""))), s, [short(g) for g in gs], XT[ax], yt,
                        header=name, log=args.log)
    overlay("mass", "mass", "m_{#chi} / #Deltam [GeV]")
    overlay("ctau", "ctau", "Pooled over masses")
    overlay("delta", "delta", "Pooled over masses, lifetimes")
    d = out / "prediction"
    d.mkdir(parents=True, exist_ok=True)
    for g in [x for x in groups if x == "overall" or x.startswith("delta_")]:
        supp, supp2, pred = get(g, "s2", "supp"), get(g, "s2", "supp2"), get(g, "s2", "pred")
        ratio = pred.Clone(g + "_ratio"); ratio.Divide(supp)
        ratio.Sumw2(False)
        a = cs.eff_graph(supp2, supp, cs.BLUE, 1, 20)
        b = cs.hist_graph(ratio, cs.ORANGE, 1, 21, errors=False)
        cs.plot(str(d / ("observed_vs_predicted_" + g)), [a, b], ["Observed", "Predicted (#varepsilon_{1}#varepsilon_{2})"], XT["s2"],
                "Fraction with 2 legs matched", lines=[lab(g), "Close: < 0.2, map from #geq 0.3"], log=args.log)
    res = lambda k: f.Get("result_" + k)
    r = {k: [res(k).GetBinContent(i + 1) for i in range(res(k).GetNbinsX())] for k in
         ("N", "undefined", "close", "close_unsupported", "obs", "pred", "D", "frac", "L", "U", "b", "lo", "hi", "clo_obs", "clo_pred")}
    names = [res("frac").GetXaxis().GetBinLabel(i + 1) for i in range(res("frac").GetNbinsX())]
    xs = [float(n.split("_")[1].replace("p", ".")) for n in names[1:]]
    edges = np.array([0., 22.])
    g1, g2, g3 = ROOT.TGraphAsymmErrors(), ROOT.TGraphAsymmErrors(), ROOT.TGraphAsymmErrors()
    for i, x in enumerate(xs):
        v = 100 * r["frac"][i + 1]
        g1.SetPoint(i, x, v); g1.SetPointError(i, 0, 0, v - 100 * r["L"][i + 1], 100 * r["U"][i + 1] - v)
        g2.SetPoint(i, x, v); g2.SetPointError(i, 0, 0, v - 100 * r["lo"][i + 1], 100 * r["hi"][i + 1] - v)
    g3.SetPoint(0, 0, 100 * THRESHOLD); g3.SetPoint(1, 22, 100 * THRESHOLD)
    s = [cs._style(g2, cs.ORANGE, 1, 1, edges, kind="hist"), cs._style(g1, cs.BLUE, 1, 20, edges, kind="hist"),
         cs._style(g3, cs.BLACK, 2, 1, edges, opt="L", legopt="l", kind="hist")]
    v = f.Get("decision").GetTitle()
    cs.plot(str(d / "deficit_vs_delta"), s, ["With bias allowance b", "95% bootstrap interval", "2% threshold"], "#Deltam [GeV]",
            "Deficit D / two-accepted events [%]", lines=["Overall %.2f%%, [%.2f, %.2f]%%" % (100 * r["frac"][0], 100 * r["lo"][0], 100 * r["hi"][0])])
    rows = [[n.replace("delta_", "Δ = ").replace("p", "."), "%d" % r["N"][i], "%d" % r["close"][i], "%d" % r["close_unsupported"][i], "%d" % r["undefined"][i],
             "%d" % r["obs"][i], "%.1f" % r["pred"][i], "%.3f" % (100 * r["frac"][i]), "[%.3f, %.3f]" % (100 * r["L"][i], 100 * r["U"][i]),
             "%.3f" % (100 * r["b"][i]), "[%.3f, %.3f]" % (100 * r["lo"][i], 100 * r["hi"][i])] for i, n in enumerate(names)]
    ov = [[("undefined" if b == 0 else "overflow" if b == len(S_EDGES) else "%g–%g" % (S_EDGES[b - 1], S_EDGES[b])),
           "%d" % get("overall", "s2", "supp").GetBinContent(b), "%d" % get("overall", "s2", "supp2").GetBinContent(b),
           "%.1f" % get("overall", "s2", "pred").GetBinContent(b),
           "%.1f" % (get("overall", "s2", "pred").GetBinContent(b) - get("overall", "s2", "supp2").GetBinContent(b))]
          + ["%d" % get("overall", "s2", "sec" + k).GetBinContent(b) for k in CLASSES] for b in range(len(S_EDGES) + 1)]
    caps = cs.Captions("One-object dimuon study: separation-dependent leg loss", [
        "Signal: 2022EE iDMmu MC, 2026-09-28 production; samples: %s. Saved PreselectedVertexing trees (met_proxy_20261022, %s), no rescan." % (len(f.Get("samples").GetTitle().split(",")), INPUT_REV),
        "Denominator: two-accepted events (exactly two signal muons, opposite charge, both |η| < 2.4, 0 ≤ Lxy < 740 cm, |vz| < 960 cm, no pT minimum). Event level, raw, unweighted; pooling unweighted.",
        "Matched leg: cleanedDisplacedMuons component, staged G > I > DSA, station-2 position, reco–gen charge agreement required, ΔR < 0.06 (G, I) or < 0.10 (DSA), globally-nearest greedy one-to-one, object claimed whole. Includes reconstruction, ID, |η| < 2.4, propagation and matching. Charge relaxation not measured here.",
        "Station-2 ΔR: hypot(Δη, wrapped Δφ) of the two gen muons' station-2 positions; undefined if either failed propagation (kept in the ROOT underflow, not drawn). Max |d0|: straight-line to the origin.",
        "Bars are 68% Clopper–Pearson unless stated. Results are an association with separation, not a measurement of merged muons."])
    caps.add("*/fractions_*", "Fraction of two-accepted events with 2, 1 and 0 matched legs, per group and x-variable. Fractions sum to one in each bin.")
    caps.add("overlays/*_f2_*", "Two-leg fraction overlaid by mass, lifetime or splitting. Bins with fewer than 20 events are not drawn.")
    caps.add("overlays/*_den_*", "Two-accepted events in each bin: unnormalized, and _norm copies normalized to unity.")
    caps.add("prediction/observed_vs_predicted_*", "Observed two-leg fraction against the mean of ε(leg 1) × ε(leg 2), per station-2 ΔR bin. ε is one map from legs of pairs with station-2 ΔR ≥ 0.3 pooled over all samples, in 30 cells of leg pT, Lxy and barrel/endcap. Bins at ≥ 0.3 are in-sample. Only pairs with defined separation and both legs in occupied cells. Secondary classes for the unmatched leg of one-leg events: A claimed object, charge agrees; B claimed object, charge disagrees; C unclaimed object, charge disagrees; D no eligible component inside the threshold; E no station-2 position.",
             ["Station-2 ΔR", "Pairs", "Observed 2-leg", "Predicted", "D"] + list(CLASSES), ov)
    caps.add("prediction/deficit_vs_delta*", "D = sum over close pairs (station-2 ΔR < 0.2) of predicted minus observed two-leg, over two-accepted events. Blue: 95%% Poisson event-bootstrap interval [L, U], marginal, refitting the map. Orange: [L − b, U + b], where b is the size of the closure mismatch (map from ≥ 1.0 predicting 0.3–1.0), an assumed allowance. Dashed: 2%% threshold. Decision: %s. D covers only pairs with defined separation and supported cells, and assumes the prediction carries over to the close region." % v,
             ["Group", "Two-accepted", "Close", "Unsupported", "Undefined", "Observed", "Predicted", "D/N [%]", "95% [%]", "b [%]", "With b [%]"], rows)
    print("CAPTIONS", caps.write(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--input-dir", required=True); a.add_argument("--output", required=True)
    a.add_argument("--revision", required=True); a.add_argument("--samples", default="")
    a.add_argument("--expected-samples", type=int, default=72); a.add_argument("--bootstrap", type=int, default=1000)
    a.add_argument("--seed", type=int, default=20261106)
    v = sub.add_parser("validate")
    v.add_argument("--input-dir", required=True); v.add_argument("--output", required=True)
    v.add_argument("--samples", required=True)
    p = sub.add_parser("plot")
    p.add_argument("--output", required=True); p.add_argument("--plot-dir", required=True)
    p.add_argument("--style-dir", required=True); p.add_argument("--log", action="store_true")
    args = ap.parse_args()
    {"analyze": analyze, "validate": validate, "plot": plot}[args.cmd](args)


if __name__ == "__main__":
    main()
