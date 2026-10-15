"""Plots and JSON summary for the DimuonVertexTruth outputs (vertexing step 1).

  python3 plot_dimuon_vertex.py 'RUNDIR/dimuon_vertex_*.root' OUTDIR [--groups overall,mass,lifetime,sample]
                                 [--require-id] [--log]

Groups: overall, per mass point (pooled over lifetimes), per lifetime (pooled over mass points), per
sample. Pooling is unweighted. Denominator: events with both signal muons accepted.

Track-type combinations: II, GG, OO and the mixed IG, IO, GO. A mixed combination has two orderings
(mu- leg type, mu+ leg type), e.g. IG = {mu- inner + mu+ global, mu- global + mu+ inner}.
  - efficiencies are per event: matched if either ordering has both legs matched, fit valid if
    either ordering has a valid fit;
  - residuals, pulls, chi2 are per pair: every valid fit of either ordering is one entry.
fitStatus: 0 not both matched, 1 legs share a track, 2 fit failed, 3 valid. Matches include the
station-1/ECAL fallback; its share is stored in the JSON.

Per group and x-variable, one PNG with 6 panels: pair-match efficiency (both matched / both accepted),
fit efficiency (valid / both matched), 68% (solid) and 95% (dashed) containment of |residual| along /
perpendicular to the gen transverse flight direction and in z, and 68% containment of |pull_along|
(1 for Gaussian pulls with correct errors). Containment needs >= MIN_ENTRIES entries per bin and carries
no uncertainty; efficiencies have 68.27% Clopper-Pearson intervals. Values outside the shown range are
kept in the JSON (underflow/overflow bins), not cut.

Scales: every plot is linear. <name>_trackerLegs.png repeats a figure with only II, GG, IG (inner/global
residuals, ~0.01 cm, vanish next to DSA ones on a linear axis). --log adds <name>_log.png copies with log
residual/pull axes; they never replace the linear plots.

gen_vertex.png (every group): gen signal-vertex Lxy (production vertex of the signal mu-, rho from (0,0))
over 0-740 cm and the fraction above Lxy; denominator = every event with one signal mu- and one mu+ (no
acceptance; Lxy >= 740 counted as overflow). Fractions with Lxy > 112, 273.5, 330, >= 740 cm, |vz| > 273.5
cm and outside the old fitter box (Lxy >= 112 or |vz| >= 273.5) are in the plot and the JSON (genVertex).

x-variables (gen): min pT, max |eta|, max |d0|, leading/subleading (by pT) pT, eta, |d0|, dimuon pT and
eta, Lxy (0-75 cm), LxyFull (0-740 cm), dR(mu,mu). |d0| = |vx sin(phi) - vy cos(phi)|, straight line to
(0,0); Lxy = vertex rho from (0,0). Per-sample groups use only min pT, max |eta|, max |d0|, Lxy, LxyFull, dR.

HIER (object-level hierarchy, files from 63c413d on): one representative track per kept displacedMuon
(isGlobal -> global, else isTracker -> inner, else DSA-ID'd outer -> outer; no ID'd component -> unused),
matched at station 2 with station-1 fallback; same panels with pair-match = both gen muons matched to a
representative / both accepted, fit efficiency = valid / both matched. Extra hierarchy.png (non-sample
groups): composition of valid HIER fits by representative pair, and the merging check per matched gen
muon: whether a DSA-ID'd raw standalone track matched to the same gen muon is an outer track of the
representative's objects (hierOuterCheck 1 none matched, 2 own, 3 not own but objects have an outer, 4
objects have no outer); counts in the JSON. --require-id does not change HIER (its IDs are built in).
merging.png (overall only, files with the dsa* branches): per matched gen muon with a DSA-ID'd standalone
matched to it, inner representatives split by hierOuterCheck 2 (own) / 3 / 4, global representatives with
2 as the merged reference; distributions normalized to unit area, plus the owner and other-gen fractions.

--require-id (component-level ID): a leg counts as matched only if its matched track passes the ID of
its type (matchIdPass: inner -> a parent muon isTracker, global -> a parent isGlobal, outer -> DSA ID;
OR over parent muons). Pairs with a failing leg are treated as not matched (fitStatus 0), so the first
panel becomes both matched and ID'd / both accepted and the fit efficiency valid / both matched and ID'd.
"""
import argparse
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot
from scipy.stats import beta

MIN_ENTRIES = 20
ID_MODE = False  # set by --require-id
ID_LABEL = ""
LOG = False  # set by --log: add log-scale copies
TRACKER_COMBOS = ["II", "GG", "IG"]
GEN_LXY_CUTS = [112.0, 273.5, 330.0, 740.0]
COMBOS = {"II": [0], "GG": [4], "OO": [8], "IG": [1, 3], "IO": [2, 6], "GO": [5, 7]}
COLORS = {"II": "#1f77b4", "GG": "#ff7f0e", "OO": "#2ca02c", "IG": "#9467bd", "IO": "#8c564b", "GO": "#e377c2",
          "HIER": "#000000"}
HIER_FIT = {"fitStatus": "hierFitStatus", "fitResAlong": "hierFitResAlong", "fitResPerp": "hierFitResPerp",
            "fitResZ": "hierFitResZ", "fitSigAlong": "hierFitSigAlong", "fitProb": "hierFitProb",
            "fitChi2": "hierFitChi2", "fitNdof": "hierFitNdof"}
HIER_EXTRA = ["hierSurface", "hierType", "hierMatched", "hierOuterCheck"]
MERGE_ON = [False]  # set by load: inputs carry the dsa* merging branches
MERGE = ["dsaDR", "dsaDROtherGen", "hierDsaDRSt2", "hierDROtherGen", "dsaPt", "hierPt", "dsaNStations",
         "dsaOwnerN", "dsaOwnerTracker", "dsaOwnerGlobal", "dsaOwnerInner", "dsaOfOtherRep", "hierNMatchedStations",
         "hierDR", "dsaOwnerCandType", "dsaOwnerInnerSame", "dsaOwnerInnerDRRep", "dsaOwnerInnerPtRatio",
         "dsaOwnerCandDR", "dsaSurface"]
BRANCHES = ["genAccepted", "genPt", "genEta", "genD0", "genLxy", "genDimuonPt", "genDimuonEta", "genDimuonDR",
            "matchSurface", "fitStatus", "fitResAlong", "fitResPerp", "fitResZ", "fitSigAlong", "fitProb",
            "fitChi2", "fitNdof", "matchIdPass"]

XVARS = {  # name: (label, display edges); underflow (-inf, first) and overflow [last, inf) bins are added for the JSON
    "minPt": ("min gen p_T [GeV]", np.arange(0, 30.01, 2)),
    "maxAbsEta": ("max gen |eta|", np.linspace(0, 2.4, 13)),
    "maxAbsD0": ("max gen |d0| [cm]", np.arange(0, 30.01, 1)),
    "leadPt": ("leading gen p_T [GeV]", np.arange(0, 30.01, 2)),
    "subPt": ("subleading gen p_T [GeV]", np.arange(0, 30.01, 2)),
    "leadEta": ("leading gen eta", np.linspace(-2.4, 2.4, 17)),
    "subEta": ("subleading gen eta", np.linspace(-2.4, 2.4, 17)),
    "leadD0": ("leading gen |d0| [cm]", np.arange(0, 30.01, 1)),
    "subD0": ("subleading gen |d0| [cm]", np.arange(0, 30.01, 1)),
    "dimuonPt": ("gen dimuon p_T [GeV]", np.arange(0, 60.01, 4)),
    "dimuonEta": ("gen dimuon eta", np.linspace(-4, 4, 17)),
    "Lxy": ("gen L_xy [cm]", np.arange(0, 75.01, 3)),
    "LxyFull": ("gen L_xy [cm]", np.arange(0, 740.01, 20)),
    "dR": ("gen dR(mu, mu)", np.arange(0, 1.501, 0.1)),
}
SAMPLE_XVARS = ["minPt", "maxAbsEta", "maxAbsD0", "Lxy", "LxyFull", "dR"]


def load(pattern):
    """Per sample: events with both signal muons accepted, and (GEN) the gen vertex of every event."""
    data = {}
    for fn in sorted(glob.glob(pattern)):
        f = uproot.open(fn)
        sample = str(f["dimuonVertexTruth/sample"].member("fTitle"))
        tree = f["dimuonVertexTruth/events"]
        hier = "hierFitStatus" in tree.keys()
        if data and hier != ("HIER" in COMBOS):
            raise RuntimeError(f"{fn}: inputs mix files with and without the hierarchy branches")
        names = BRANCHES + (list(HIER_FIT.values()) + HIER_EXTRA if hier else [])
        merge = "dsaDR" in tree.keys()
        if data and merge != MERGE_ON[0]:
            raise RuntimeError(f"{fn}: inputs mix files with and without the dsa* branches")
        MERGE_ON[0] = merge
        if merge:
            names += MERGE
        a = {k: np.asarray(v) for k, v in tree.arrays(names, library="np").items()}
        if hier:  # the hierarchy fit becomes column 9 of the fit arrays
            COMBOS["HIER"] = [9]
            for k, h in HIER_FIT.items():
                a[k] = np.column_stack([a[k], a.pop(h)])
        keep = a["genAccepted"].sum(1) == 2
        if sample in data:
            raise RuntimeError(f"sample {sample} appears in more than one input file ({fn})")
        data[sample] = {k: v[keep] for k, v in a.items()}
        vz = tree["genVtx"].array(library="np")[:, 2]
        GEN[sample] = {"genLxy": a["genLxy"], "genVz": np.asarray(vz)}
    return data


GEN = {}  # sample -> gen vertex of every event (filled by load)


def require_id(a):
    """Treat combinations with a leg failing its component ID as not matched (fitStatus 0)."""
    st = a["fitStatus"].copy()
    idp = a["matchIdPass"]  # [mu- I, G, O, mu+ I, G, O]
    for idx in range(9):  # column 9 (HIER) already has its IDs
        ta, tb = divmod(idx, 3)
        st[(idp[:, ta] == 0) | (idp[:, 3 + tb] == 0), idx] = 0
    a["fitStatus"] = st


def concat(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def xvalues(a):
    pt, eta, d0 = a["genPt"], a["genEta"], a["genD0"]
    lead = np.argmax(pt, axis=1)
    sub = 1 - lead
    r = np.arange(len(pt))
    return {
        "minPt": pt.min(1), "maxAbsEta": np.abs(eta).max(1), "maxAbsD0": d0.max(1),
        "leadPt": pt[r, lead], "subPt": pt[r, sub], "leadEta": eta[r, lead], "subEta": eta[r, sub],
        "leadD0": d0[r, lead], "subD0": d0[r, sub],
        "dimuonPt": a["genDimuonPt"], "dimuonEta": a["genDimuonEta"], "Lxy": a["genLxy"], "LxyFull": a["genLxy"],
        "dR": a["genDimuonDR"],
    }


def clopper_pearson(k, n, cl=0.6827):
    k, n = np.asarray(k, float), np.asarray(n, float)
    a = (1 - cl) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        lo = np.where(k > 0, beta.ppf(a, k, n - k + 1), 0.0)
        hi = np.where(k < n, beta.ppf(1 - a, k + 1, n - k), 1.0)
        eff = np.where(n > 0, k / n, np.nan)
    return eff, lo, hi


def combo_event_flags(a, c):
    st = a["fitStatus"][:, COMBOS[c]]
    surf = a["matchSurface"]
    matched = (st > 0).any(1)
    valid = (st == 3).any(1)
    fallback = np.zeros(len(st), bool)
    for idx in COMBOS[c]:
        if idx == 9:
            fallback |= (a["fitStatus"][:, 9] > 0) & (a["hierSurface"] > 1).any(1)
            continue
        ta, tb = divmod(idx, 3)
        fallback |= (a["fitStatus"][:, idx] > 0) & ((surf[:, ta] > 1) | (surf[:, 3 + tb] > 1))
    return matched, valid, fallback


def combo_pairs(a, c, x):
    """Per-pair entries (valid fits of every ordering) with the event's x value."""
    xs, ra, rp, rz, pull, prob, chi2ndof = [], [], [], [], [], [], []
    for idx in COMBOS[c]:
        v = a["fitStatus"][:, idx] == 3
        xs.append(x[v])
        ra.append(a["fitResAlong"][v, idx])
        rp.append(a["fitResPerp"][v, idx])
        rz.append(a["fitResZ"][v, idx])
        pull.append(a["fitResAlong"][v, idx] / a["fitSigAlong"][v, idx])
        prob.append(a["fitProb"][v, idx])
        chi2ndof.append(a["fitChi2"][v, idx] / a["fitNdof"][v, idx])
    cat = np.concatenate
    return dict(x=cat(xs), along=cat(ra), perp=cat(rp), z=cat(rz), pull=cat(pull), prob=cat(prob),
                chi2ndof=cat(chi2ndof))


def binned_quantile(x, y, edges, q):
    idx = np.digitize(x, edges) - 1
    out, n = [], []
    for b in range(len(edges) - 1):
        s = np.abs(y[idx == b])
        n.append(int(len(s)))
        out.append(float(np.quantile(s, q)) if len(s) >= MIN_ENTRIES else None)
    return out, n


def summarize_group(a, xvars):
    xv = xvalues(a)
    res = {"nEvents": int(len(a["genPt"])), "combos": {}}
    for c in COMBOS:
        matched, valid, fallback = combo_event_flags(a, c)
        res["combos"][c] = {"matched": int(matched.sum()), "matchedWithFallbackLeg": int(fallback.sum()),
                            "valid": int(valid.sum())}
    res["x"] = {}
    for name in xvars:
        edges = np.concatenate([[-np.inf], XVARS[name][1], [np.inf]])
        x = xv[name]
        den = np.histogram(x, edges)[0]
        assert den.sum() == len(x), name
        entry = {"edges": ["-inf"] + [float(e) for e in edges[1:-1]] + ["inf"], "bothAccepted": den.tolist(),
                 "combos": {}}
        for c in COMBOS:
            matched, valid, _ = combo_event_flags(a, c)
            p = combo_pairs(a, c, x)
            e = {"matched": np.histogram(x[matched], edges)[0].tolist(),
                 "validGivenMatched": np.histogram(x[matched & valid], edges)[0].tolist()}
            for comp in ("along", "perp", "z"):
                for q in (0.68, 0.95):
                    e[f"res_{comp}_q{int(q * 100)}_cm"], e["nPairs"] = binned_quantile(p["x"], p[comp], edges, q)
            e["pull_along_q68"], _ = binned_quantile(p["x"], p["pull"], edges, 0.68)
            entry["combos"][c] = e
        res["x"][name] = entry
    return res, xv


def plot_group(a, summary, xv, title, outdir, xvars):
    os.makedirs(outdir, exist_ok=True)
    for name in xvars:
        plot_group_figure(summary, title, outdir, name, list(COMBOS), False, "")
        plot_group_figure(summary, title, outdir, name, TRACKER_COMBOS, False, "_trackerLegs")
        if LOG:
            plot_group_figure(summary, title, outdir, name, list(COMBOS), True, "_log")


def plot_group_figure(summary, title, outdir, name, combos, log, suffix):
    label, disp = XVARS[name]
    e = summary["x"][name]
    nb = len(disp) - 1
    centers = 0.5 * (disp[:-1] + disp[1:])
    sl = slice(1, nb + 1)  # skip underflow/overflow bins
    den = np.array(e["bothAccepted"][sl])
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    for c in combos:
        ce = e["combos"][c]
        col = COLORS[c]
        m = np.array(ce["matched"][sl])
        v = np.array(ce["validGivenMatched"][sl])
        for ax, k, n in ((axs[0, 0], m, den), (axs[0, 1], v, m)):
            eff, lo, hi = clopper_pearson(k, n)
            ok = n > 0
            ax.errorbar(centers[ok], eff[ok], yerr=[eff[ok] - lo[ok], hi[ok] - eff[ok]],
                        xerr=0.5 * np.diff(disp)[ok], fmt="o", ms=3, color=col, label=c)
        for ax, comp in ((axs[0, 2], "along"), (axs[1, 0], "perp"), (axs[1, 1], "z")):
            for q, ls in ((68, "-"), (95, "--")):
                y = np.array([np.nan if t is None else t for t in ce[f"res_{comp}_q{q}_cm"][sl]], float)
                ax.plot(centers, y, ls, marker="o", ms=3, color=col, label=c if q == 68 else None)
        y = np.array([np.nan if t is None else t for t in ce["pull_along_q68"][sl]], float)
        axs[1, 2].plot(centers, y, "-", marker="o", ms=3, color=col, label=c)
    both = "both matched and ID'd" if ID_MODE else "both matched"
    axs[0, 0].set_ylabel(f"pair-match efficiency ({both} / both accepted)")
    axs[0, 1].set_ylabel(f"fit efficiency (valid / {both})")
    for ax in (axs[0, 0], axs[0, 1]):
        ax.set_ylim(0, 1.05)
    for ax, comp in ((axs[0, 2], "along flight dir."), (axs[1, 0], "perp. to flight dir."), (axs[1, 1], "z")):
        ax.set_yscale("log" if log else "linear")
        if not log:
            ax.set_ylim(bottom=0)
        ax.set_ylabel(f"|reco - gen vertex| {comp} [cm]\n68% (solid), 95% (dashed) containment")
    axs[1, 2].axhline(1.0, color="gray", lw=0.8)
    axs[1, 2].set_yscale("log" if log else "linear")
    if not log:
        axs[1, 2].set_ylim(bottom=0)
    axs[1, 2].set_ylabel("68% containment of |pull_along|")
    for ax in axs.flat:
        ax.set_xlabel(label)
        ax.set_xlim(disp[0], disp[-1])
        ax.grid(alpha=0.3)
    axs[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle(f"{title} | denominator: both signal muons accepted ({summary['nEvents']} events) | "
                 f"I=inner G=global O=DSA; HIER = object-level hierarchy; {ID_LABEL}mixed = both orderings; unweighted; "
                 f"containment needs >= {MIN_ENTRIES} pairs/bin | combos: {', '.join(combos)}; "
                 f"{'log' if log else 'linear'} residual/pull axes", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(outdir, f"{name}{suffix}.png"), dpi=90)
    plt.close(fig)


def plot_distributions(a, title, outdir):
    os.makedirs(outdir, exist_ok=True)
    plot_distributions_figure(a, title, outdir, list(COMBOS), np.linspace(0, 50, 101), "")
    plot_distributions_figure(a, title, outdir, TRACKER_COMBOS, np.linspace(0, 0.1, 101), "_trackerLegs")
    if LOG:
        plot_distributions_figure(a, title, outdir, list(COMBOS), np.logspace(-5, 3, 81), "_log")


def plot_distributions_figure(a, title, outdir, combos, resbins, suffix):
    log = suffix == "_log"
    dummy = np.zeros(len(a["genPt"]))
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    lo, hi = (1.01e-5, 9.9e2) if log else (0.0, resbins[-1] * (1 - 1e-9))
    for c in combos:
        p = combo_pairs(a, c, dummy)
        if len(p["along"]) == 0:
            continue
        w = np.full(len(p["along"]), 1.0 / len(p["along"]))  # fraction of pairs per bin
        kw = dict(histtype="step", weights=w, color=COLORS[c], label=f"{c} ({len(p['along'])})")
        for ax, comp in ((axs[0, 0], "along"), (axs[0, 1], "perp"), (axs[0, 2], "z")):
            ax.hist(np.clip(np.abs(p[comp]), lo, hi), bins=resbins, **kw)
        axs[1, 0].hist(np.clip(p["pull"], -10, 10), bins=np.linspace(-10, 10, 81), **kw)
        axs[1, 1].hist(p["prob"], bins=np.linspace(0, 1, 51), **kw)
        axs[1, 2].hist(np.clip(p["chi2ndof"], 0, 20), bins=np.linspace(0, 20, 81), **kw)
    g = np.linspace(-10, 10, 400)
    axs[1, 0].plot(g, 0.25 * np.exp(-g * g / 2) / np.sqrt(2 * np.pi), color="k", lw=0.8, label="N(0,1)")
    for ax, comp in ((axs[0, 0], "along flight dir."), (axs[0, 1], "perp. to flight dir."), (axs[0, 2], "z")):
        if log:
            ax.set_xscale("log")
        ax.set_xlabel(f"|reco - gen vertex| {comp} [cm] (clipped to [{lo:g}, {resbins[-1]:g}])")
    if log:
        for ax in (axs[1, 0], axs[1, 2]):
            ax.set_yscale("log")
    axs[1, 0].set_xlabel("pull along flight dir. (clipped to +-10)")
    axs[1, 1].set_xlabel("fit probability")
    axs[1, 2].set_xlabel("chi2 / ndof (clipped to 20)")
    for ax in axs.flat:
        ax.set_ylabel("fraction of pairs per bin")
        ax.grid(alpha=0.3)
    axs[0, 0].legend(fontsize=8)
    axs[1, 0].legend(fontsize=8)
    fig.suptitle(f"{title} | {ID_LABEL}valid fits, mixed combos include both orderings | unweighted | "
                 f"{'log' if log else 'linear'} axes", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(outdir, f"distributions{suffix}.png"), dpi=90)
    plt.close(fig)


def hierarchy_counts(a):
    """Merging check per matched gen muon by representative type; valid HIER fits by representative pair."""
    m, t, oc = a["hierMatched"] == 1, a["hierType"], a["hierOuterCheck"]
    out = {"outerCheck": {}, "validFitsByRepPair": {}}
    for k, name in enumerate("IGO"):
        sel = m & (t == k)
        out["outerCheck"][name] = {str(c): int((sel & (oc == c)).sum()) for c in (1, 2, 3, 4)}
    v = a["fitStatus"][:, 9] == 3
    for i in range(3):
        for j in range(3):
            out["validFitsByRepPair"]["IGO"[i] + "IGO"[j]] = int((v & (t[:, 0] == i) & (t[:, 1] == j)).sum())
    return out


def plot_hierarchy(a, title, outdir):
    os.makedirs(outdir, exist_ok=True)
    label, disp = XVARS["Lxy"]
    centers = 0.5 * (disp[:-1] + disp[1:])
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    x = a["genLxy"]
    v = a["fitStatus"][:, 9] == 3
    t = a["hierType"]
    nv = np.histogram(x[v], disp)[0]
    for pair, col in (("GG", "#ff7f0e"), ("IG", "#9467bd"), ("II", "#1f77b4"), ("GO", "#e377c2"),
                      ("IO", "#8c564b"), ("OO", "#2ca02c")):
        i, j = "IGO".index(pair[0]), "IGO".index(pair[1])
        sel = v & (((t[:, 0] == i) & (t[:, 1] == j)) | ((t[:, 0] == j) & (t[:, 1] == i)))
        with np.errstate(invalid="ignore", divide="ignore"):
            axs[0].plot(centers, np.histogram(x[sel], disp)[0] / nv, "-o", ms=3, color=col, label=pair)
    axs[0].set_ylabel("fraction of valid HIER fits (representative pair, either order)")
    xg = np.concatenate([x, x])
    m = np.concatenate([a["hierMatched"][:, 0], a["hierMatched"][:, 1]]) == 1
    tg = np.concatenate([t[:, 0], t[:, 1]])
    og = np.concatenate([a["hierOuterCheck"][:, 0], a["hierOuterCheck"][:, 1]])
    for ax, k in ((axs[1], 0), (axs[2], 1)):
        sel = m & (tg == k)
        den = np.histogram(xg[sel & (og > 1)], disp)[0]
        for c, col, lab in ((3, "#d62728", "not own, objects have an outer"), (4, "#7f7f7f", "objects have no outer")):
            eff, lo, hi = clopper_pearson(np.histogram(xg[sel & (og == c)], disp)[0], den)
            ok = den > 0
            ax.errorbar(centers[ok], eff[ok], yerr=[eff[ok] - lo[ok], hi[ok] - eff[ok]], fmt="o", ms=3, color=col,
                        label=lab)
        ax.set_ylabel("fraction of gen muons with a DSA-ID'd standalone matched\n(DSA not an outer of the "
                      f"{'inner' if k == 0 else 'global'}-represented objects)")
        ax.set_title(f"{'inner' if k == 0 else 'global'} representatives: {int((sel & (og > 1)).sum())} "
                     f"with an ID'd DSA matched, {int((sel & (og == 1)).sum())} without", fontsize=9)
        ax.set_ylim(0, 1.05)
    for ax in axs:
        ax.set_xlabel(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"{title} | object-level hierarchy (G > I > DSA-ID'd O), station 2 + station-1 fallback | "
                 "unweighted", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(outdir, "hierarchy.png"), dpi=90)
    plt.close(fig)


def plot_merging(a, title, outdir):
    out = plot_merging_figure(a, title, outdir, False)
    if LOG:
        plot_merging_figure(a, title, outdir, True)
    return out


def plot_merging_figure(a, title, outdir, log):
    """Is the DSA matched to a gen muon of an inner representative wrongly matched, or unmerged?"""
    def per_gen(k):
        return np.concatenate([a[k][:, 0], a[k][:, 1]])
    genpt = per_gen("genPt")
    m, t, oc = per_gen("hierMatched") == 1, per_gen("hierType"), per_gen("hierOuterCheck")
    st2 = per_gen("dsaSurface") == 1  # DSA matched at station 2, so its dR is comparable with dsaDROtherGen
    cats = [("inner rep, DSA own (2)", m & (t == 0) & (oc == 2), "#1f77b4"),
            ("inner rep, DSA of another object, own outer exists (3)", m & (t == 0) & (oc == 3), "#d62728"),
            ("inner rep, DSA of another object, no own outer (4)", m & (t == 0) & (oc == 4), "#7f7f7f"),
            ("global rep, DSA own (2), reference", m & (t == 1) & (oc == 2), "#ff7f0e")]
    with np.errstate(divide="ignore", invalid="ignore"):
        dvars = [("dsaDR", "station-2 dR(DSA, its gen muon) (DSA matched at station 2)", np.linspace(0, 0.1, 41),
                  np.where(st2, per_gen("dsaDR"), -999.0)),
                 ("dsaDROtherGen", "station-2 dR(DSA, other gen muon) (clipped to 2)", np.linspace(0, 2, 41),
                  np.clip(per_gen("dsaDROtherGen"), -1, 1.999)),
                 ("hierDsaDRSt2", "station-2 dR(representative, DSA) (clipped to 0.5)", np.linspace(0, 0.5, 51),
                  np.clip(per_gen("hierDsaDRSt2"), -1, 0.4999)),
                 ("ptRatio", "DSA p_T / gen p_T (clipped to 5)", np.linspace(0, 5, 51),
                  np.clip(per_gen("dsaPt") / genpt, 0, 4.999)),
                 ("repPtRatio", "representative p_T / gen p_T (clipped to 5)", np.linspace(0, 5, 51),
                  np.clip(per_gen("hierPt") / genpt, 0, 4.999)),
                 ("nStations", "DSA muon stations with valid hits", np.arange(-0.5, 5.5, 1), per_gen("dsaNStations"))]
    fig, axs = plt.subplots(2, 4, figsize=(20, 9))
    flat = axs.flat
    for (name, lab, edges, x), ax in zip(dvars, flat):
        for cl, sel, col in cats:
            v = x[sel & (x > -900)]
            if len(v):
                ax.hist(v, bins=edges, histtype="step", color=col, weights=np.full(len(v), 1.0 / len(v)),
                        label=f"{cl}: {len(v)}")
        ax.set_xlabel(lab)
        ax.set_ylabel("fraction per bin")
        ax.grid(alpha=0.3)
    axs[0, 0].legend(fontsize=7)
    ax = axs[1, 2]
    fr = {"owner tracker": per_gen("dsaOwnerTracker"), "owner global": per_gen("dsaOwnerGlobal"),
          "owner has inner": per_gen("dsaOwnerInner"), "DSA of other gen's rep": per_gen("dsaOfOtherRep"),
          "DSA closer to other gen (st2)": st2 & (per_gen("dsaDROtherGen") > -900)
                                           & (per_gen("dsaDROtherGen") < per_gen("dsaDR")),
          "owner inner = rep's": per_gen("dsaOwnerInnerSame") == 1,
          "owner inner dR<0.01": (per_gen("dsaOwnerInnerDRRep") > -900) & (per_gen("dsaOwnerInnerDRRep") < 0.01),
          "owner cand within 0.06": (per_gen("dsaOwnerCandDR") > -900) & (per_gen("dsaOwnerCandDR") < 0.06)}
    w = 0.2
    ax.set_title("bars per category: owner flags, other-gen association, owner duplicate", fontsize=8)
    for j, (cl, sel, col) in enumerate(cats):
        n = max(int(sel.sum()), 1)
        ax.bar(np.arange(len(fr)) + (j - 1.5) * w, [float((v[sel] > 0).sum()) / n for v in fr.values()], w,
               color=col)
    ax.set_xticks(np.arange(len(fr)))
    ax.set_xticklabels(list(fr), rotation=20, fontsize=7)
    ax.set_ylabel("fraction of gen muons in category")
    ax.grid(alpha=0.3)
    ax = axs[1, 3]
    odr = per_gen("dsaOwnerInnerDRRep")
    for cl, sel, col in cats:
        v = np.clip(odr[sel & (odr > -900)], 0, 0.4999)
        if len(v):
            ax.hist(v, bins=np.linspace(0, 0.5, 51), histtype="step", color=col, weights=np.full(len(v), 1.0 / len(v)))
    ax.set_yscale("log" if log else "linear")
    ax.set_xlabel("dR(DSA owner's inner track, representative track) at reference point (clipped to 0.5)")
    ax.set_ylabel("fraction per bin")
    ax.grid(alpha=0.3)
    fig.suptitle(f"{title} | merging check: DSA-ID'd standalone matched to the same gen muon (st2 + st1 fallback) | "
                 "unweighted, unit area", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(outdir, "merging_log.png" if log else "merging.png"), dpi=90)
    plt.close(fig)
    return {cl: {"n": int(sel.sum()), **{k: int((v[sel] > 0).sum()) for k, v in fr.items()}} for cl, sel, _ in cats}


def gen_vertex_counts(g):
    lxy, vz = g["genLxy"], np.abs(g["genVz"])
    n = len(lxy)
    out = {"nEvents": int(n), "LxyAbove": {f"{c:g}": int((lxy > c).sum()) for c in GEN_LXY_CUTS[:-1]},
           "LxyAtLeast740": int((lxy >= 740).sum()), "absVzAbove273.5": int((vz > 273.5).sum()),
           "outsideOldBox": int(((lxy >= 112) | (vz >= 273.5)).sum())}
    return out


def plot_gen_vertex(g, counts, title, outdir):
    os.makedirs(outdir, exist_ok=True)
    lxy = g["genLxy"]
    n = max(len(lxy), 1)
    edges = np.arange(0, 740.01, 10)
    for log in ([False, True] if LOG else [False]):
        fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
        h = np.histogram(np.clip(lxy, 0, None), np.append(edges, np.inf))[0]  # last bin = overflow (>= 740)
        axs[0].stairs(h[:-1] / n, edges, color="k")
        axs[0].set_ylabel("fraction of gen signal vertices per 10 cm")
        grid = np.linspace(0, 740, 741)
        above = 1.0 - np.searchsorted(np.sort(lxy), grid, side="right") / n
        axs[1].plot(grid, above, color="k")
        axs[1].set_ylabel("fraction of gen signal vertices with L_xy > x")
        for ax in axs:
            for c, col in zip(GEN_LXY_CUTS[:-1], ("#d62728", "#1f77b4", "#2ca02c")):
                ax.axvline(c, color=col, ls="--", lw=1, label=f"{c:g} cm")
            ax.set_xlabel("gen signal-vertex L_xy [cm] (rho from (0,0))")
            ax.set_xlim(0, 740)
            ax.set_yscale("log" if log else "linear")
            if not log:
                ax.set_ylim(bottom=0)
            ax.grid(alpha=0.3)
        axs[1].legend(fontsize=8)
        k = counts
        txt = "\n".join([f"N = {k['nEvents']} events (no acceptance)"]
                        + [f"L_xy > {c} cm: {v} ({v / n:.4f})" for c, v in k["LxyAbove"].items()]
                        + [f"L_xy >= 740 cm (overflow): {k['LxyAtLeast740']} ({k['LxyAtLeast740'] / n:.4f})",
                           f"|vz| > 273.5 cm: {k['absVzAbove273.5']} ({k['absVzAbove273.5'] / n:.4f})",
                           f"outside old box (L_xy>=112 or |vz|>=273.5): {k['outsideOldBox']} "
                           f"({k['outsideOldBox'] / n:.4f})"])
        axs[0].text(0.97, 0.97, txt, transform=axs[0].transAxes, ha="right", va="top", fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"))
        fig.suptitle(f"{title} | gen signal vertex (signal mu- production vertex), every event with one signal mu- "
                     f"and one mu+ | unweighted | {'log' if log else 'linear'} y", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(os.path.join(outdir, "gen_vertex_log.png" if log else "gen_vertex.png"), dpi=90)
        plt.close(fig)


def plot_symmetry(a, outdir):
    plot_symmetry_figure(a, outdir, False)
    if LOG:
        plot_symmetry_figure(a, outdir, True)


def plot_symmetry_figure(a, outdir, log):
    """Overall check that the two orderings of each mixed combination agree."""
    os.makedirs(outdir, exist_ok=True)
    x = a["genLxy"]
    label, disp = XVARS["Lxy"]
    centers = 0.5 * (disp[:-1] + disp[1:])
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    for j, (c, pair) in enumerate((("IG", (1, 3)), ("IO", (2, 6)), ("GO", (5, 7)))):
        for idx, ls in zip(pair, ("-", "--")):
            name = "IGO"[idx // 3] + "IGO"[idx % 3]
            st = a["fitStatus"][:, idx]
            den = np.histogram(x, disp)[0]
            k = np.histogram(x[st > 0], disp)[0]
            eff, lo, hi = clopper_pearson(k, den)
            axs[0, j].errorbar(centers, eff, yerr=[eff - lo, hi - eff], fmt="o" + ls, ms=3,
                               label=f"mu- {name[0]}, mu+ {name[1]}: {int((st > 0).sum())} matched")
            v = st == 3
            q, _ = binned_quantile(x[v], a["fitResAlong"][v, idx], disp, 0.68)
            axs[1, j].plot(centers, [np.nan if t is None else t for t in q], ls, marker="o", ms=3,
                           label=f"mu- {name[0]}, mu+ {name[1]}: {int(v.sum())} valid")
        axs[0, j].set_ylabel(f"{c}: pair-match efficiency per ordering")
        axs[1, j].set_ylabel(f"{c}: 68% containment |res along| [cm]")
        axs[1, j].set_yscale("log" if log else "linear")
        for ax in (axs[0, j], axs[1, j]):
            ax.set_xlabel(label)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
    fig.suptitle(f"Overall: {ID_LABEL}orderings of mixed combinations (mu+/mu- symmetry check) | unweighted", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(outdir, "ordering_symmetry_Lxy_log.png" if log else "ordering_symmetry_Lxy.png"), dpi=90)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern")
    ap.add_argument("outdir")
    ap.add_argument("--groups", default="overall,mass,lifetime,sample")
    ap.add_argument("--require-id", action="store_true", help="require the component ID on both legs")
    ap.add_argument("--log", action="store_true", help="also write log-scale copies (*_log.png)")
    args = ap.parse_args()
    global ID_MODE, ID_LABEL, LOG
    LOG = args.log
    if args.require_id:
        ID_MODE, ID_LABEL = True, "both legs pass component ID (I: tracker, G: global, O: DSA ID); "
    groups = args.groups.split(",")
    data = load(args.pattern)
    if ID_MODE:
        for a in data.values():
            require_id(a)
    rx = re.compile(r"(Mchi-[^_]+_dMchi-[^_]+)_(ctau-[^_]+)$")
    by_mass, by_life = {}, {}
    for s in data:
        match = rx.match(s)
        if not match:
            raise RuntimeError(f"unexpected sample name {s}")
        m, l = match.groups()
        by_mass.setdefault(m, []).append(s)
        by_life.setdefault(l, []).append(s)
    todo = []
    if "overall" in groups:
        todo.append(("overall", "overall", list(data), list(XVARS)))
    if "mass" in groups:
        todo += [(f"mass/{m}", f"mass {m} (pooled over lifetimes)", ss, list(XVARS)) for m, ss in sorted(by_mass.items())]
    if "lifetime" in groups:
        todo += [(f"lifetime/{l}", f"lifetime {l} (pooled over masses)", ss, list(XVARS)) for l, ss in sorted(by_life.items())]
    if "sample" in groups:
        todo += [(f"sample/{s}", f"sample {s}", [s], SAMPLE_XVARS) for s in sorted(data)]
    out = {"definitions": __doc__, "minEntries": MIN_ENTRIES, "requireId": ID_MODE, "log": LOG, "samples": sorted(data),
           "groups": {}}
    for key, title, samples, xvars in todo:
        a = concat([data[s] for s in samples])
        summary, xv = summarize_group(a, xvars)
        summary["samples"] = samples
        g = concat([GEN[s] for s in samples])
        summary["genVertex"] = gen_vertex_counts(g)
        plot_gen_vertex(g, summary["genVertex"], title, os.path.join(args.outdir, key))
        if "HIER" in COMBOS:
            summary["hierarchy"] = hierarchy_counts(a)
        out["groups"][key] = summary
        plot_group(a, summary, xv, title, os.path.join(args.outdir, key), xvars)
        if not key.startswith("sample/"):
            plot_distributions(a, title, os.path.join(args.outdir, key))
            if "HIER" in COMBOS:
                plot_hierarchy(a, title, os.path.join(args.outdir, key))
        if key == "overall":
            plot_symmetry(a, os.path.join(args.outdir, key))
            if "dsaDR" in a:
                summary["merging"] = plot_merging(a, title, os.path.join(args.outdir, key))
        print(key, summary["nEvents"], flush=True)
    with open(os.path.join(args.outdir, "dimuon_vertex_summary.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
