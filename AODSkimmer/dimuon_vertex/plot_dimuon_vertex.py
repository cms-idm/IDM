"""Plots and JSON summary for the DimuonVertexTruth outputs (vertexing step 1).

  python3 plot_dimuon_vertex.py 'RUNDIR/dimuon_vertex_*.root' OUTDIR [--groups overall,mass,lifetime,sample]
                                 [--require-id]

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
(1 for Gaussian pulls with correct errors). Residual panels use a log y axis. Containment needs
>= MIN_ENTRIES entries per bin and carries no uncertainty; efficiencies have 68.27% Clopper-Pearson
intervals. x axes are linear; values outside the shown range are kept in the JSON (underflow/overflow
bins), not cut.

x-variables (gen): min pT, max |eta|, max |d0|, leading/subleading (by pT) pT, eta, |d0|, dimuon pT and
eta, Lxy, dR(mu,mu). |d0| = |vx sin(phi) - vy cos(phi)|, straight line to (0,0); Lxy = vertex rho from
(0,0). Per-sample groups use only min pT, max |eta|, max |d0|, Lxy, dR.

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
COMBOS = {"II": [0], "GG": [4], "OO": [8], "IG": [1, 3], "IO": [2, 6], "GO": [5, 7]}
COLORS = {"II": "#1f77b4", "GG": "#ff7f0e", "OO": "#2ca02c", "IG": "#9467bd", "IO": "#8c564b", "GO": "#e377c2"}
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
    "dR": ("gen dR(mu, mu)", np.arange(0, 1.501, 0.1)),
}
SAMPLE_XVARS = ["minPt", "maxAbsEta", "maxAbsD0", "Lxy", "dR"]


def load(pattern):
    data = {}
    for fn in sorted(glob.glob(pattern)):
        f = uproot.open(fn)
        sample = str(f["dimuonVertexTruth/sample"].member("fTitle"))
        a = f["dimuonVertexTruth/events"].arrays(BRANCHES, library="np")
        a = {k: np.asarray(v) for k, v in a.items()}
        keep = a["genAccepted"].sum(1) == 2
        if sample in data:
            raise RuntimeError(f"sample {sample} appears in more than one input file ({fn})")
        data[sample] = {k: v[keep] for k, v in a.items()}
    return data


def require_id(a):
    """Treat combinations with a leg failing its component ID as not matched (fitStatus 0)."""
    st = a["fitStatus"].copy()
    idp = a["matchIdPass"]  # [mu- I, G, O, mu+ I, G, O]
    for idx in range(9):
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
        "dimuonPt": a["genDimuonPt"], "dimuonEta": a["genDimuonEta"], "Lxy": a["genLxy"], "dR": a["genDimuonDR"],
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
        label, disp = XVARS[name]
        e = summary["x"][name]
        nb = len(disp) - 1
        centers = 0.5 * (disp[:-1] + disp[1:])
        sl = slice(1, nb + 1)  # skip underflow/overflow bins
        den = np.array(e["bothAccepted"][sl])
        fig, axs = plt.subplots(2, 3, figsize=(16, 9))
        for c in COMBOS:
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
            ax.set_yscale("log")
            ax.set_ylabel(f"|reco - gen vertex| {comp} [cm]\n68% (solid), 95% (dashed) containment")
        axs[1, 2].axhline(1.0, color="gray", lw=0.8)
        axs[1, 2].set_yscale("log")
        axs[1, 2].set_ylabel("68% containment of |pull_along|")
        for ax in axs.flat:
            ax.set_xlabel(label)
            ax.set_xlim(disp[0], disp[-1])
            ax.grid(alpha=0.3)
        axs[0, 0].legend(fontsize=8, ncol=2)
        fig.suptitle(f"{title} | denominator: both signal muons accepted ({summary['nEvents']} events) | "
                     f"I=inner G=global O=DSA; {ID_LABEL}mixed = both orderings; unweighted; "
                     f"containment needs >= {MIN_ENTRIES} pairs/bin", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(os.path.join(outdir, f"{name}.png"), dpi=90)
        plt.close(fig)


def plot_distributions(a, title, outdir):
    os.makedirs(outdir, exist_ok=True)
    dummy = np.zeros(len(a["genPt"]))
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    logbins = np.logspace(-5, 3, 81)
    for c in COMBOS:
        p = combo_pairs(a, c, dummy)
        if len(p["along"]) == 0:
            continue
        w = np.full(len(p["along"]), 1.0 / len(p["along"]))  # fraction of pairs per bin
        kw = dict(histtype="step", weights=w, color=COLORS[c], label=f"{c} ({len(p['along'])})")
        for ax, comp in ((axs[0, 0], "along"), (axs[0, 1], "perp"), (axs[0, 2], "z")):
            ax.hist(np.clip(np.abs(p[comp]), 1.01e-5, 9.9e2), bins=logbins, **kw)
        axs[1, 0].hist(np.clip(p["pull"], -10, 10), bins=np.linspace(-10, 10, 81), **kw)
        axs[1, 1].hist(p["prob"], bins=np.linspace(0, 1, 51), **kw)
        axs[1, 2].hist(np.clip(p["chi2ndof"], 0, 20), bins=np.linspace(0, 20, 81), **kw)
    g = np.linspace(-10, 10, 400)
    axs[1, 0].plot(g, 0.25 * np.exp(-g * g / 2) / np.sqrt(2 * np.pi), color="k", lw=0.8, label="N(0,1)")
    for ax, comp in ((axs[0, 0], "along flight dir."), (axs[0, 1], "perp. to flight dir."), (axs[0, 2], "z")):
        ax.set_xscale("log")
        ax.set_xlabel(f"|reco - gen vertex| {comp} [cm] (clipped to [1e-5, 1e3])")
    axs[1, 0].set_xlabel("pull along flight dir. (clipped to +-10)")
    axs[1, 1].set_xlabel("fit probability")
    axs[1, 2].set_xlabel("chi2 / ndof (clipped to 20)")
    for ax in axs.flat:
        ax.set_ylabel("fraction of pairs per bin")
        ax.grid(alpha=0.3)
    axs[0, 0].legend(fontsize=8)
    axs[1, 0].legend(fontsize=8)
    fig.suptitle(f"{title} | {ID_LABEL}valid fits, mixed combos include both orderings | unweighted", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(outdir, "distributions.png"), dpi=90)
    plt.close(fig)


def plot_symmetry(a, outdir):
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
        axs[1, j].set_yscale("log")
        for ax in (axs[0, j], axs[1, j]):
            ax.set_xlabel(label)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
    fig.suptitle(f"Overall: {ID_LABEL}orderings of mixed combinations (mu+/mu- symmetry check) | unweighted", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(outdir, "ordering_symmetry_Lxy.png"), dpi=90)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern")
    ap.add_argument("outdir")
    ap.add_argument("--groups", default="overall,mass,lifetime,sample")
    ap.add_argument("--require-id", action="store_true", help="require the component ID on both legs")
    args = ap.parse_args()
    global ID_MODE, ID_LABEL
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
    out = {"definitions": __doc__, "minEntries": MIN_ENTRIES, "requireId": ID_MODE, "samples": sorted(data), "groups": {}}
    for key, title, samples, xvars in todo:
        a = concat([data[s] for s in samples])
        summary, xv = summarize_group(a, xvars)
        summary["samples"] = samples
        out["groups"][key] = summary
        plot_group(a, summary, xv, title, os.path.join(args.outdir, key), xvars)
        if not key.startswith("sample/"):
            plot_distributions(a, title, os.path.join(args.outdir, key))
        if key == "overall":
            plot_symmetry(a, os.path.join(args.outdir, key))
        print(key, summary["nEvents"], flush=True)
    with open(os.path.join(args.outdir, "dimuon_vertex_summary.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
