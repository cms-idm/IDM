#!/usr/bin/env python3
"""Muon-corrected MET proxy vs Type-1 PF MET, against gen pT(chi1 chi1) (study note: kb/studies/met-proxy-residual.md).

python3 met_proxy.py analyze --input-dir RUN/inputs --output RUN/met_proxy.root --revision <sha>
python3 met_proxy.py plot --output RUN/met_proxy.root --plot-dir RUN/plots [--log]

Inputs: PreselectedVertexing outputs with hierarchyOrder Global > Inner > DSA (checked) and the hf*, pfMu*, chi* branches.
Vertex: the hierarchy-fallback vertex (hf*: best-best pair, else other matched components of the same two objects in pair
order GG > GI > G-DSA > II > I-DSA > DSA-DSA), one per event. Pair type = unordered leg types of that vertex.
Base: two accepted signal gen muons (twoAcc) and a valid hf vertex. Population a: base with chiOK and a finite PF MET;
population b: a with chi2/ndof < 4 and reco dR(mu, mu) < 0.9 (track reference points) on that vertex.
Vectors (GeV): PF = Type-1 PF MET; proxy = PF + sum(px, py of all packed PF candidates with |pdgId| 13) - pT(mumu), pT(mumu) =
vector sum of the two legs' reference-point track pT; G = vector sum of the two chi1. For X in {PF, proxy}:
dmag = |X| - |G|; rel = dmag / |G|; dphiG = phi(X) - phi(G) wrapped to [-pi, pi); upar = (X - G) . G/|G|;
uperp = z-component of G/|G| x (X - G) (positive = residual counter-clockwise from G); dphiMM = phi(X) - phi(mumu) wrapped.
|G| < 1 GeV: rel, dphiG, upar, uperp undefined (counted); |X| = 0: dphiG and dphiMM undefined (counted).
Chains (same vertex): base -> chi2/ndof < 4 -> dR < 0.9 -> |dphi(X, mumu)| < 0.5 for X = PF (A), proxy (A'), genMetTrue (E),
pT(chi1 chi1) (C1, truth invisible; user 2026-10-22); missing or zero-length X fails. Raw, unweighted, one entry per event; pooling unweighted. Max gen |d0| = straight line to (0,0).
"""
import argparse
from array import array
from pathlib import Path
import re

import numpy as np

CHI2_CUT, DR_CUT, DPHI_CUT, G_FLOOR = 4.0, 0.9, 0.5, 1.0
ORDER_EXPECTED = "global>inner>outer(DSA)"
RANK = {2: 0, 0: 1, 1: 2}  # Global > Inner > DSA (types: 0 inner, 1 outer/DSA, 2 global)
PAIRS = ("GG", "GI", "GDSA", "II", "IDSA", "DSADSA")
PAIR_LABEL = {"all": "all pair types", "GG": "Global-Global", "GI": "Global-Inner", "GDSA": "Global-DSA",
              "II": "Inner-Inner", "IDSA": "Inner-DSA", "DSADSA": "DSA-DSA"}
POPS = {"a": "valid vertex", "b": "valid vertex, chi2/ndof < 4, reco dR < 0.9"}
XS = ("PF", "proxy")
X_LABEL = {"PF": "Type-1 PF MET", "proxy": "proxy (PF MET + PF muons - selected dimuon)"}
EDGES = {
    "dmag": np.linspace(-100, 100, 81), "rel": np.linspace(-2, 3, 51), "dphiG": np.linspace(-np.pi, np.pi, 33),
    "absdphiG": np.linspace(0, np.pi, 33), "upar": np.linspace(-100, 100, 81), "uperp": np.linspace(-100, 100, 81),
    "absdphiMM": np.linspace(0, np.pi, 33)}
VAR_LABEL = {"dmag": "|X| - |G| [GeV]", "rel": "(|X| - |G|) / |G|", "dphiG": "dphi(X, G) [rad]",
             "absdphiG": "|dphi(X, G)| [rad]", "upar": "u_par = (X - G) along G [GeV]",
             "uperp": "u_perp = (X - G) perpendicular to G [GeV]", "absdphiMM": "|dphi(X, pT_mumu)| [rad]"}
D0_EDGES = np.array([0, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000.])
G_EDGES = np.array([0, 5, 10, 15, 20, 30, 40, 60, 80, 100, 150, 200, 300.])
BRANCHES = ["twoAcc", "hfStatus", "hfTried", "hfTypeA", "hfTypeB", "hfChi2", "hfNdof", "hfRecoDR", "hfRecoMuMuPt", "hfRecoMuMuPhi",
            "pfMetPt", "pfMetPhi", "genMetPt", "genMetPhi", "pfMuN", "pfMuPx", "pfMuPy", "chiOK", "chiPx", "chiPy", "sigD0"]
TREE_FLOATS = ["maxD0", "Gx", "Gy", "Gpt", "mmPt", "mmPhi", "chi2ndof", "recoDR", "genMetGx", "genMetGy"] + [
    "%s_%s" % (x, v) for x in XS for v in ("px", "py", "pt", "dmag", "rel", "dphiG", "upar", "uperp", "dphiMM")] + [
    "genMet_dphiMM", "chi_dphiMM"]


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def pair_index(ta, tb):
    lo, hi = sorted((RANK[int(ta)], RANK[int(tb)]))
    return {(0, 0): 0, (0, 1): 1, (0, 2): 2, (1, 1): 3, (1, 2): 4, (2, 2): 5}[(lo, hi)]


def load(args):
    import uproot
    paths = sorted(Path(args.input_dir).glob("*.root"))
    if len(paths) != args.expected_samples:
        raise RuntimeError("Unexpected number of inputs: %d" % len(paths))
    cols, names = {k: [] for k in BRANCHES + ["sample"]}, []
    for si, path in enumerate(paths):
        with uproot.open(path) as f:
            d = f["preselectedVertexing"]
            sample = str(d["sample"].member("fTitle"))
            order = str(d["hierarchyOrder"].member("fTitle"))
            rev = str(d["revision"].member("fTitle"))
            if order != ORDER_EXPECTED or rev != args.input_revision:
                raise RuntimeError("%s: hierarchyOrder %s, revision %s" % (path, order, rev))
            if sample in names or not re.fullmatch(r"Mchi-[\dp]+_dMchi-[\dp]+_ctau-[\dp]+", sample):
                raise RuntimeError("Bad/duplicate sample " + sample)
            names.append(sample)
            a = d["events"].arrays(BRANCHES, library="np")
        n = len(a["twoAcc"])
        for k in BRANCHES:
            if k == "sigD0":
                cols[k].append(np.array([max(v) if len(v) == 2 else np.nan for v in a[k]], dtype=float))
            else:
                cols[k].append(np.asarray(a[k], dtype=float))
        cols["sample"].append(np.full(n, si))
    return names, {k: np.concatenate(v) for k, v in cols.items()}


def derive(c):
    """Per-event derived quantities (all events; validity masks separate)."""
    n = len(c["twoAcc"])
    met_ok = np.isfinite(c["pfMetPt"]) & (c["pfMetPt"] > -500) & np.isfinite(c["pfMetPhi"])
    base = (c["twoAcc"] > 0) & (c["hfStatus"] == 3)
    chi_ok = c["chiOK"] > 0
    q = {"maxD0": c["sigD0"], "Gx": np.where(chi_ok, c["chiPx"], np.nan), "Gy": np.where(chi_ok, c["chiPy"], np.nan)}
    q["Gpt"] = np.hypot(q["Gx"], q["Gy"])
    q["mmPt"], q["mmPhi"] = c["hfRecoMuMuPt"], c["hfRecoMuMuPhi"]
    q["chi2ndof"] = c["hfChi2"] / np.where(c["hfNdof"] > 0, c["hfNdof"], np.nan)
    q["recoDR"] = c["hfRecoDR"]
    mmx, mmy = q["mmPt"] * np.cos(q["mmPhi"]), q["mmPt"] * np.sin(q["mmPhi"])
    pfx = np.where(met_ok, c["pfMetPt"] * np.cos(c["pfMetPhi"]), np.nan)
    pfy = np.where(met_ok, c["pfMetPt"] * np.sin(c["pfMetPhi"]), np.nan)
    vec = {"PF": (pfx, pfy), "proxy": (pfx + c["pfMuPx"] - mmx, pfy + c["pfMuPy"] - mmy)}
    gok = q["Gpt"] >= G_FLOOR
    ux, uy = q["Gx"] / np.where(gok, q["Gpt"], np.nan), q["Gy"] / np.where(gok, q["Gpt"], np.nan)
    for x, (vx, vy) in vec.items():
        pt = np.hypot(vx, vy)
        xok = pt > 0
        q[x + "_px"], q[x + "_py"], q[x + "_pt"] = vx, vy, pt
        q[x + "_dmag"] = pt - q["Gpt"]
        q[x + "_rel"] = np.where(gok, q[x + "_dmag"] / np.where(gok, q["Gpt"], 1), np.nan)
        q[x + "_dphiG"] = np.where(gok & xok, wrap(np.arctan2(vy, vx) - np.arctan2(q["Gy"], q["Gx"])), np.nan)
        rx, ry = vx - q["Gx"], vy - q["Gy"]
        q[x + "_upar"] = rx * ux + ry * uy
        q[x + "_uperp"] = ux * ry - uy * rx
        q[x + "_dphiMM"] = np.where(xok & (q["mmPt"] > 0), wrap(np.arctan2(vy, vx) - q["mmPhi"]), np.nan)
    gm_ok = np.isfinite(c["genMetPt"]) & (c["genMetPt"] > 0) & np.isfinite(c["genMetPhi"])  # 0: no direction
    q["genMet_dphiMM"] = np.where(gm_ok & (q["mmPt"] > 0), wrap(c["genMetPhi"] - q["mmPhi"]), np.nan)
    q["chi_dphiMM"] = np.where((q["Gpt"] > 0) & (q["mmPt"] > 0), wrap(np.arctan2(q["Gy"], q["Gx"]) - q["mmPhi"]), np.nan)
    q["genMetGx"] = np.where(gm_ok, c["genMetPt"] * np.cos(c["genMetPhi"]), np.nan) - q["Gx"]
    q["genMetGy"] = np.where(gm_ok, c["genMetPt"] * np.sin(c["genMetPhi"]), np.nan) - q["Gy"]
    pair = np.full(n, -1)
    for i in np.flatnonzero(base):
        pair[i] = pair_index(c["hfTypeA"][i], c["hfTypeB"][i])
    popa = base & chi_ok & met_ok
    popb = popa & (q["chi2ndof"] < CHI2_CUT) & (q["recoDR"] < DR_CUT)
    masks = dict(base=base, chi_ok=chi_ok, met_ok=met_ok, a=popa, b=popb, gok=gok)
    return q, pair, masks


def analyze(args):
    import ROOT
    ROOT.gROOT.SetBatch(True)
    if Path(args.output).exists():
        raise RuntimeError("Refusing to overwrite ROOT output")
    names, c = load(args)
    total = len(c["twoAcc"])
    if args.expected_events and total != args.expected_events:
        raise RuntimeError("Event total mismatch: %d" % total)
    q, pair, m = derive(c)
    mass_of = [re.match(r"(Mchi-[^_]+_dMchi-[^_]+)_", s).group(1) for s in names]
    life_of = [re.search(r"(ctau-[^_]+)$", s).group(1) for s in names]
    smp = c["sample"].astype(int)
    groups = {"overall": np.ones(total, bool)}
    for mname in sorted(set(mass_of)):
        groups["mass/" + mname] = np.isin(smp, [i for i, x in enumerate(mass_of) if x == mname])
    for lname in sorted(set(life_of)):
        groups["lifetime/" + lname] = np.isin(smp, [i for i, x in enumerate(life_of) if x == lname])
    for i, s in enumerate(names):
        groups["sample/" + s] = smp == i

    root = ROOT.TFile.Open(str(args.output), "CREATE")
    ROOT.TNamed("definitions", __doc__).Write()
    ROOT.TNamed("analysisRevision", args.revision).Write()
    ROOT.TNamed("inputRevision", args.input_revision).Write()
    ROOT.TNamed("samples", "\n".join(names)).Write()

    def th1(name, edges, x):
        h = ROOT.TH1D(name, name, len(edges) - 1, array("d", edges))
        h.Sumw2()
        x = x[np.isfinite(x)]
        counts = np.bincount(np.searchsorted(edges, x, side="right"), minlength=len(edges) + 1)
        for b, v in enumerate(counts):  # bin 0 = underflow, last = overflow
            h.SetBinContent(b, float(v))
            h.SetBinError(b, float(np.sqrt(v)))
        h.SetEntries(float(len(x)))
        h.Write()

    def counts_hist(name, labels, vals):
        h = ROOT.TH1D(name, name, len(vals), 0, len(vals))
        h.Sumw2()
        for b, (lab, v) in enumerate(zip(labels, vals), 1):
            h.SetBinContent(b, float(v))
            h.SetBinError(b, float(np.sqrt(v)))
            h.GetXaxis().SetBinLabel(b, lab)
        h.Write()

    # distributions: population x group x pair type x X x variable
    for pop in POPS:
        for gname, gmask in groups.items():
            for pname in ("all",) + PAIRS:
                sel = m[pop] & gmask & ((pair >= 0) if pname == "all" else (pair == PAIRS.index(pname)))
                dname = "%s/%s/%s" % (pop, gname, pname)
                root.mkdir(dname)
                root.cd(dname)
                for x in XS:
                    for v, edges in EDGES.items():
                        src = np.abs(q["%s_%s" % (x, v[3:])]) if v.startswith("abs") else q["%s_%s" % (x, v)]
                        th1("%s__%s" % (x, v), edges, src[sel])
                counts_hist("counts", ["events", "Gpt<1", "PF pt=0", "proxy pt=0"],
                            [sel.sum(), (sel & ~m["gok"]).sum(), (sel & (q["PF_pt"] == 0)).sum(),
                             (sel & (q["proxy_pt"] == 0)).sum()])
                root.cd()
    # chains and migrations (overall and per pair type), on base events
    root.mkdir("chains")
    root.cd("chains")
    dphis = {"A_PF": q["PF_dphiMM"], "Aprime_proxy": q["proxy_dphiMM"], "E_genMet": q["genMet_dphiMM"],
             "C1_chi1chi1": q["chi_dphiMM"]}
    chain_rows = {}
    for pname in ("all",) + PAIRS:
        sel = m["base"] & ((pair >= 0) if pname == "all" else (pair == PAIRS.index(pname)))
        s1 = sel & (q["chi2ndof"] < CHI2_CUT)
        s2 = s1 & (q["recoDR"] < DR_CUT)
        for cname, dp in dphis.items():
            s3 = s2 & np.isfinite(dp) & (np.abs(dp) < DPHI_CUT)
            vals = [sel.sum(), s1.sum(), s2.sum(), s3.sum()]
            chain_rows[(pname, cname)] = vals
            counts_hist("chain_%s__%s" % (cname, pname), ["base", "chi2/ndof<4", "dR<0.9", "|dphi|<0.5"], vals)
        for pop in POPS:
            ps = m[pop] & ((pair >= 0) if pname == "all" else (pair == PAIRS.index(pname)))
            pf = np.isfinite(q["PF_dphiMM"]) & (np.abs(q["PF_dphiMM"]) < DPHI_CUT)
            px = np.isfinite(q["proxy_dphiMM"]) & (np.abs(q["proxy_dphiMM"]) < DPHI_CUT)
            counts_hist("migration_%s__%s" % (pop, pname), ["PF pass, proxy pass", "PF pass, proxy fail",
                        "PF fail, proxy pass", "PF fail, proxy fail"],
                        [(ps & pf & px).sum(), (ps & pf & ~px).sum(), (ps & ~pf & px).sum(), (ps & ~pf & ~px).sum()])
    root.cd()
    flow = [("initial", total), ("twoAcc", int((c["twoAcc"] > 0).sum())), ("base (valid vertex)", int(m["base"].sum())),
            ("base, fallback used", int((m["base"] & (c["hfTried"] > 1)).sum())),
            ("base, not chiOK", int((m["base"] & ~m["chi_ok"]).sum())),
            ("base, MET missing", int((m["base"] & ~m["met_ok"]).sum())), ("population a", int(m["a"].sum())),
            ("population b", int(m["b"].sum()))]
    counts_hist("counts", [f[0] for f in flow], [f[1] for f in flow])
    # per-event tree of base events (exact quantiles and replotting)
    t = ROOT.TTree("events", "base events: twoAcc and a valid hf vertex")
    keep = np.flatnonzero(m["base"])
    buf = {}
    for k in ("sample", "pair", "popA", "popB", "chiOK"):
        buf[k] = array("i", [0])
        t.Branch(k, buf[k], k + "/I")
    for k in TREE_FLOATS:
        buf[k] = array("d", [0])
        t.Branch(k, buf[k], k + "/D")
    for i in keep:
        buf["sample"][0], buf["pair"][0] = int(smp[i]), int(pair[i])
        buf["popA"][0], buf["popB"][0], buf["chiOK"][0] = int(m["a"][i]), int(m["b"][i]), int(m["chi_ok"][i])
        for k in TREE_FLOATS:
            buf[k][0] = float(q[k][i])
        t.Fill()
    t.Write()
    root.Close()

    # report
    print("VALIDATED: samples=%d events=%d" % (len(names), total))
    for lab, v in flow:
        print("  %-24s %8d" % (lab, v))
    def qs(x):
        x = x[np.isfinite(x)]
        return "n=%d median %+.3f [16-84%%: %+.3f, %+.3f] [2.5-97.5%%: %+.3f, %+.3f]" % (
            len(x), *np.percentile(x, [50, 16, 84, 2.5, 97.5])) if len(x) else "n=0"
    for pop in POPS:
        for pname in ("all",) + PAIRS:
            sel = m[pop] & ((pair >= 0) if pname == "all" else (pair == PAIRS.index(pname)))
            print("population %s, %s:" % (pop, PAIR_LABEL[pname]))
            for v in ("dmag", "upar", "uperp", "dphiG"):
                for x in XS:
                    src = np.abs(q["%s_%s" % (x, v)]) if v == "dphiG" else q["%s_%s" % (x, v)]
                    print("   %-6s %-8s %s" % (x, ("|dphiG|" if v == "dphiG" else v), qs(src[sel])))
    for pname in ("all",) + PAIRS:
        print("chains %-8s " % pname + "; ".join("%s %s" % (cn, chain_rows[(pname, cn)]) for cn in dphis))
    gm = np.hypot(q["genMetGx"], q["genMetGy"])
    print("|genMetTrue - G| (base, chiOK):", qs(gm[m["base"] & m["chi_ok"]]))


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import uproot
    f = uproot.open(args.output)
    out = Path(args.plot_dir)
    foot = ("Signal MC 2022EE, cleaned pre-selected displacedMuons, hierarchy G > I > DSA with fallback; G = pT(chi1 chi1); "
            "raw, unweighted, one entry per event")
    colors = {"PF": "#0072B2", "proxy": "#D55E00"}

    def save(fig, path, logy_axes=()):
        fig.text(0.01, 0.005, foot, fontsize=6.5, color="0.35")
        fig.tight_layout(rect=(0, 0.02, 1, 0.95))
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=110)
        if args.log and logy_axes:
            for ax in logy_axes:
                ax.set_yscale("log")
            fig.savefig(path.with_name(path.stem + "_log.png"), dpi=110)
        plt.close(fig)

    def step(ax, h, **kw):
        v, e = h.values(flow=False), h.axis().edges()
        ax.stairs(v, e, **kw)
        return h.values(flow=True)

    samples = str(f["samples"].member("fTitle")).split("\n")
    masses = sorted({re.match(r"(Mchi-[^_]+_dMchi-[^_]+)_", s).group(1) for s in samples},
                    key=lambda s: float(s.split("_")[0][5:].replace("p", ".")))
    lifes = sorted({re.search(r"(ctau-[^_]+)$", s).group(1) for s in samples}, key=lambda s: float(s[5:]))
    order = ["dmag", "rel", "dphiG", "absdphiG", "upar", "uperp", "absdphiMM"]
    for pop in (() if args.chains_only else POPS):
        targets = [("overall", p) for p in ("all",) + PAIRS]
        targets += [("mass/" + mm, p) for mm in masses for p in ("all",) + PAIRS]
        targets += [("lifetime/" + ll, p) for ll in lifes for p in ("all",) + PAIRS]
        targets += [("sample/" + s, "all") for s in samples]
        for gname, pname in targets:
            d = f["%s/%s/%s" % (pop, gname, pname)]
            cnt = d["counts"].values()
            fig, axes = plt.subplots(2, 4, figsize=(17, 7.5))
            used = []
            for ax, v in zip(axes.flat, order):
                for x in XS:
                    fl = step(ax, d["%s__%s" % (x, v)], color=colors[x], lw=1.5, label=X_LABEL[x])
                    ax.text(0.98, 0.95 - 0.07 * XS.index(x), "%s: under %d, over %d" % (x, fl[0], fl[-1]),
                            transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color=colors[x])
                if v == "absdphiMM":
                    ax.axvline(DPHI_CUT, color="0.3", ls="--", lw=1)
                ax.set_xlabel(VAR_LABEL[v].replace("X", "MET"))
                ax.set_ylabel("Events / bin")
                ax.set_ylim(bottom=0)
                used.append(ax)
            axes.flat[-1].axis("off")
            handles, labels = axes.flat[0].get_legend_handles_labels()
            axes.flat[-1].legend(handles, labels, fontsize=8, loc="lower left")
            axes.flat[-1].text(0.0, 0.8, "events %d\n|G| < 1 GeV (no direction): %d\nPF |MET| = 0: %d, proxy |MET| = 0: %d"
                               % tuple(cnt), fontsize=9, va="top")
            fig.suptitle("%s | %s | %s" % (gname.replace("/", ": "), PAIR_LABEL[pname], POPS[pop]), fontsize=11)
            save(fig, out / pop / gname.replace("/", "_") / ("residuals_%s.png" % pname), used)
        # mass and lifetime overlays, raw and unit area
        for kind, keys in (("mass", masses), ("lifetime", lifes)):
            for pname in ("all",) + PAIRS:
                for v in ("upar", "absdphiG"):
                    for norm in (False, True):
                        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
                        cmap = plt.get_cmap("viridis", len(keys))
                        for ax, x in zip(axes, XS):
                            for i, k in enumerate(keys):
                                h = f["%s/%s/%s/%s/%s__%s" % (pop, kind, k, pname, x, v)]
                                vals, e = h.values(flow=False), h.axis().edges()
                                tot = h.values(flow=True).sum()
                                if tot == 0:
                                    continue
                                ax.stairs(vals / tot if norm else vals, e, color=cmap(i), lw=1.1, label=k)
                            ax.set_title(X_LABEL[x], fontsize=9)
                            ax.set_xlabel(VAR_LABEL[v].replace("X", "MET"))
                            ax.set_ylim(bottom=0)
                        axes[0].set_ylabel("Fraction of events / bin (unit area incl. under/overflow)" if norm
                                           else "Events / bin")
                        axes[1].legend(fontsize=6, ncol=2 if kind == "mass" else 1)
                        fig.suptitle("%s overlay | %s | %s | %s" % (kind, PAIR_LABEL[pname], POPS[pop],
                                     "unit area" if norm else "raw"), fontsize=10)
                        save(fig, out / pop / "overlays" / ("%s_%s_%s_%s.png" % (kind, v, pname,
                             "unit" if norm else "raw")), list(axes))
    # 2D: median and 16-84% band vs max gen |d0| and vs |G|, from the event tree
    ev = None if args.chains_only else f["events"].arrays(library="np")
    for pop, flag in (() if args.chains_only else (("a", "popA"), ("b", "popB"))):
        for pname in ("all",) + PAIRS:
            sel = (ev[flag] > 0) & ((ev["pair"] >= 0) if pname == "all" else (ev["pair"] == PAIRS.index(pname)))
            for xvar, edges, xl in (("maxD0", D0_EDGES, "max gen |d0| of the two signal muons [cm] (bin index axis)"),
                                    ("Gpt", G_EDGES, "|G| = pT(chi1 chi1) [GeV] (bin index axis)")):
                fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
                for ax, v in zip(axes, ("upar", "uperp", "dmag", "dphiG")):
                    for x in XS:
                        y = ev["%s_%s" % (x, v)][sel]
                        y = np.abs(y) if v == "dphiG" else y
                        xs = ev[xvar][sel]
                        ok = np.isfinite(y) & np.isfinite(xs)
                        idx = np.digitize(xs[ok], edges) - 1
                        med, lo, hi, cen = [], [], [], []
                        for b in range(len(edges) - 1):
                            yy = y[ok][idx == b]
                            if len(yy) < 50:
                                continue
                            p16, p50, p84 = np.percentile(yy, [16, 50, 84])
                            cen.append(b + 0.5); med.append(p50); lo.append(p16); hi.append(p84)
                        ax.plot(cen, med, "o-", color=colors[x], ms=3, label=X_LABEL[x] + " median")
                        ax.fill_between(cen, lo, hi, color=colors[x], alpha=0.2, label=x + " 16-84%")
                    ax.set_xticks(range(len(edges)))
                    ax.set_xticklabels(["%g" % e for e in edges], rotation=60, fontsize=7)
                    ax.set_xlabel(xl, fontsize=8)
                    ax.set_ylabel(("|dphi(MET, G)| [rad]" if v == "dphiG" else VAR_LABEL[v].replace("X", "MET")),
                                  fontsize=8)
                    ax.axhline(0, color="0.5", lw=0.8)
                axes[0].legend(fontsize=6)
                fig.suptitle("Residual vs %s | %s | %s (bins with < 50 events omitted)" % (
                    xvar, PAIR_LABEL[pname], POPS[pop]), fontsize=10)
                save(fig, out / pop / "vs" / ("%s_%s.png" % (xvar, pname)))
    # chains A, A', E per pair type
    fig, axes = plt.subplots(1, 7, figsize=(24, 4.5))
    for ax, pname in zip(axes, ("all",) + PAIRS):
        for j, (cname, lab) in enumerate((("A_PF", "A: PF MET"), ("Aprime_proxy", "A': proxy"), ("E_genMet", "E: gen MET"),
                                          ("C1_chi1chi1", "C1: pT(chi1 chi1)"))):
            v = f["chains/chain_%s__%s" % (cname, pname)].values()
            ax.bar(np.arange(4) + 0.2 * (j - 1.5), v / max(v[0], 1), width=0.2, label=lab)
            ax.text(3 + 0.2 * (j - 1.5), v[3] / max(v[0], 1) + 0.02, "%.1f%%" % (100 * v[3] / max(v[0], 1)),
                    ha="center", fontsize=6, rotation=90)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["base", "chi2/ndof<4", "dR<0.9", "|dphi|<0.5"], rotation=45, fontsize=7)
        ax.set_title("%s (base %d)" % (PAIR_LABEL[pname], v[0]), fontsize=8)
        ax.set_ylim(0, 1.15)
    axes[0].set_ylabel("Fraction of base events")
    axes[0].legend(fontsize=7)
    fig.suptitle("Sequential requirements on the hierarchy vertex; dphi = phi(MET or pT(chi1 chi1)) - phi(reco dimuon pT)",
                 fontsize=10)
    save(fig, out / "chains_A_Aprime_E.png")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--input-dir", required=True)
    a.add_argument("--output", required=True)
    a.add_argument("--revision", required=True)
    a.add_argument("--input-revision", default="5a2b10e")
    a.add_argument("--expected-samples", type=int, default=72)
    a.add_argument("--expected-events", type=int, default=720000)
    b = sub.add_parser("plot")
    b.add_argument("--output", required=True)
    b.add_argument("--plot-dir", required=True)
    b.add_argument("--log", action="store_true")
    b.add_argument("--chains-only", action="store_true", help="only the chain figure (A, A', E, C1)")
    args = p.parse_args()
    analyze(args) if args.cmd == "analyze" else plot(args)


if __name__ == "__main__":
    main()
