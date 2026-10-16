"""dR(mu, mu) and vertex chi2/ndof of the HIER dimuon fits (DimuonVertexTruth outputs, files from 63c413d on).

  python3 plot_dimuon_vertex_dr_chi2.py 'RUNDIR/dimuon_vertex_*.root' OUTDIR [--log]

Population: events with both signal muons accepted and a valid HIER fit (hierFitStatus 3: both gen muons
matched to a representative, legs share no track, fit valid inside the tracker bounds). One entry per event,
raw counts, unweighted. Matching as in DimuonVertexTruth (station 2 with station-1 fallback, same reco-gen
charge, dR < 0.06 inner/global, < 0.10 outer, greedy one-to-one).

dR(mu, mu): from hierEta/hierPhi, the momentum directions of the two representative tracks at their
reference points (not refitted at the vertex). chi2/ndof: hierFitChi2 / hierFitNdof (KalmanVertexFitter with
refit; ndof = 1 for two tracks, so chi2/ndof = chi2). max gen |d0|: larger of the two signal muons' |d0| =
|vx sin(phi) - vy cos(phi)|, straight line to (0,0).

Histograms (OUTDIR/dimuon_vertex_dr_chi2.root, one directory per group, under/overflow kept): dR 0-3.2 in
0.025, chi2/ndof 0-100 in 0.25, 2D dR x chi2/ndof (0-2 x 0-20, 0.05 x 0.5), 2D max gen |d0| x dR and
max gen |d0| x chi2/ndof (d0 0-30 cm in 1 cm). Groups: overall, per mass point (pooled over lifetimes,
unweighted), per lifetime (pooled over masses, unweighted), per sample.

PNGs (linear; displayed ranges dR 0-2, chi2/ndof 0-20 within the stored binning; overflow fractions in
the legends): per group dr_chi2.png (dR, chi2/ndof, heatmap) and vs_d0.png (heatmaps vs max gen |d0|);
overlays/ mass_*.png (all masses, each pooled over lifetimes) and lifetime_*.png (all lifetimes, each pooled
over masses), each as raw counts and unit area (_unitArea; area includes under/overflow). --log adds
*_log.png copies (log y, log colour scale); they never replace the linear plots.
"""
import argparse
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ROOT
ROOT.gROOT.SetBatch(True)
import uproot
from matplotlib.colors import LogNorm

BRANCHES = ["genAccepted", "genD0", "hierFitStatus", "hierFitChi2", "hierFitNdof", "hierEta", "hierPhi"]
H1 = {  # name: (axis label, stored edges, displayed upper edge)
    "dR": ("reco dR(mu, mu)", np.linspace(0, 3.2, 129), 2.0),
    "chi2ndof": ("vertex chi2 / ndof", np.linspace(0, 100, 401), 20.0),
}
H2 = {  # name: (x var, y var, x edges, y edges)
    "dR_chi2ndof": ("dR", "chi2ndof", np.linspace(0, 2, 41), np.linspace(0, 20, 41)),
    "maxD0_dR": ("maxD0", "dR", np.linspace(0, 30, 31), np.linspace(0, 2, 41)),
    "maxD0_chi2ndof": ("maxD0", "chi2ndof", np.linspace(0, 30, 31), np.linspace(0, 20, 41)),
}
LABELS = {"dR": H1["dR"][0], "chi2ndof": H1["chi2ndof"][0], "maxD0": "max gen |d0| [cm]"}
POP = "both accepted, valid HIER fit | 1 entry per event | unweighted"
LOG = False


def flow_edges(edges):
    return np.concatenate([[-np.inf], edges, [np.inf]])


def load(pattern):
    data = {}
    for fn in sorted(glob.glob(pattern)):
        f = uproot.open(fn)
        sample = str(f["dimuonVertexTruth/sample"].member("fTitle"))
        if sample in data:
            raise RuntimeError(f"sample {sample} appears in more than one input file ({fn})")
        a = f["dimuonVertexTruth/events"].arrays(BRANCHES, library="np")
        sel = (a["genAccepted"].sum(1) == 2) & (a["hierFitStatus"] == 3)
        eta, phi = a["hierEta"][sel], a["hierPhi"][sel]
        dphi = np.mod(phi[:, 0] - phi[:, 1] + np.pi, 2 * np.pi) - np.pi
        data[sample] = {"dR": np.hypot(eta[:, 0] - eta[:, 1], dphi),
                        "chi2ndof": a["hierFitChi2"][sel] / a["hierFitNdof"][sel],
                        "maxD0": np.abs(a["genD0"][sel]).max(1)}
        for k, v in data[sample].items():
            if not np.isfinite(v).all():
                raise RuntimeError(f"{fn}: {(~np.isfinite(v)).sum()} non-finite {k} values among valid HIER fits")
        print(f"{sample}: {sel.sum()} valid HIER fits of {len(sel)} events", flush=True)
    if not data:
        raise RuntimeError(f"no files match {pattern}")
    return data


def fill(a):
    """Counts with under/overflow (first and last bins)."""
    h = {k: np.histogram(a[k], flow_edges(e))[0] for k, (_, e, _) in H1.items()}
    for k, (x, y, ex, ey) in H2.items():
        h[k] = np.histogram2d(a[x], a[y], [flow_edges(ex), flow_edges(ey)])[0]
    h["n"] = len(a["dR"])
    return h


def add(hs):
    return {k: sum(h[k] for h in hs) for k in hs[0]}


def write_root(path, groups):
    """TH1D/TH2D per group with under/overflow; unweighted counts (no Sumw2 needed)."""
    f = ROOT.TFile(path, "RECREATE")
    ROOT.TNamed("definitions", __doc__).Write()
    for key, h in groups.items():
        f.mkdir(key)
        f.cd(key)
        for k, (label, e, _) in H1.items():
            t = ROOT.TH1D(k, f";{label};entries per bin", len(e) - 1, e)
            for i, c in enumerate(h[k]):
                t.SetBinContent(i, c)
            t.SetEntries(h["n"])
            t.Write()
        for k, (x, y, ex, ey) in H2.items():
            t = ROOT.TH2D(k, f";{LABELS[x]};{LABELS[y]}", len(ex) - 1, ex, len(ey) - 1, ey)
            for i in range(h[k].shape[0]):
                for j in range(h[k].shape[1]):
                    t.SetBinContent(i, j, h[k][i, j])
            t.SetEntries(h["n"])
            t.Write()
    f.Close()


def overflow_frac(h, k):
    hi = H1[k][2]
    e = H1[k][1]
    above = h[k][-1] + h[k][1:-1][e[:-1] >= hi].sum()
    return above / max(h["n"], 1)


def draw_1d(ax, h, k, label, color, unit_area):
    e = H1[k][1]
    hi = H1[k][2]
    m = e[1:] <= hi + 1e-9
    y = h[k][1:-1][m].astype(float)
    if unit_area:
        y = y / max(h["n"], 1)
    ax.stairs(y, e[: m.sum() + 1], color=color, label=f"{label} ({h['n']}; >{hi:g}: {overflow_frac(h, k):.1%})")


def finish_1d(ax, k, unit_area, log):
    ax.set_xlabel(LABELS[k])
    ax.set_ylabel("fraction of entries per bin" if unit_area else "entries per bin")
    if log:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)


def draw_2d(fig, ax, h, k, log):
    x, y, ex, ey = H2[k]
    z = h[k][1:-1, 1:-1].T
    norm = LogNorm(vmin=1, vmax=max(z.max(), 1)) if log else None
    pc = ax.pcolormesh(ex, ey, np.ma.masked_where(z == 0, z) if log else z, norm=norm, cmap="viridis")
    fig.colorbar(pc, ax=ax, label="entries per bin")
    ax.set_xlabel(LABELS[x])
    ax.set_ylabel(LABELS[y])


def plot_group(h, title, outdir):
    os.makedirs(outdir, exist_ok=True)
    for log in [False] + ([True] if LOG else []):
        sfx = "_log" if log else ""
        fig, axs = plt.subplots(1, 3, figsize=(17, 5))
        for ax, k in zip(axs[:2], H1):
            draw_1d(ax, h, k, "HIER", "k", False)
            finish_1d(ax, k, False, log)
            ax.legend(fontsize=8)
        draw_2d(fig, axs[2], h, "dR_chi2ndof", log)
        fig.suptitle(f"{title} | {POP} | {'log' if log else 'linear'}", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(os.path.join(outdir, f"dr_chi2{sfx}.png"), dpi=90)
        plt.close(fig)
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        for ax, k in zip(axs, ["maxD0_dR", "maxD0_chi2ndof"]):
            draw_2d(fig, ax, h, k, log)
        fig.suptitle(f"{title} | {POP} | d0 to (0,0), gen | {'log' if log else 'linear'}", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(os.path.join(outdir, f"vs_d0{sfx}.png"), dpi=90)
        plt.close(fig)


def plot_overlay(hs, kind, outdir):
    """hs: ordered {label: hist}; kind 'mass' (each pooled over lifetimes) or 'lifetime' (pooled over masses)."""
    os.makedirs(outdir, exist_ok=True)
    pooled = "pooled over lifetimes" if kind == "mass" else "pooled over masses"
    colors = plt.cm.tab20(np.linspace(0, 1, 20))[: len(hs)] if len(hs) > 10 else plt.cm.tab10(range(len(hs)))
    for k in H1:
        for unit_area in (False, True):
            for log in [False] + ([True] if LOG else []):
                fig, ax = plt.subplots(figsize=(9, 6))
                for (label, h), c in zip(hs.items(), colors):
                    draw_1d(ax, h, k, label, c, unit_area)
                finish_1d(ax, k, unit_area, log)
                ax.legend(fontsize=7, ncol=2 if len(hs) > 8 else 1)
                norm = "unit area (incl. under/overflow)" if unit_area else "raw counts"
                ax.set_title(f"all {kind}s, each {pooled} | {POP} | {norm} | {'log' if log else 'linear'}",
                             fontsize=8)
                fig.tight_layout()
                name = f"{kind}_{k}{'_unitArea' if unit_area else ''}{'_log' if log else ''}.png"
                fig.savefig(os.path.join(outdir, name), dpi=90)
                plt.close(fig)


def num(s):
    return float(re.search(r"-([0-9p]+)", s).group(1).replace("p", "."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern")
    ap.add_argument("outdir")
    ap.add_argument("--log", action="store_true", help="also write log-scale copies (*_log.png)")
    args = ap.parse_args()
    global LOG
    LOG = args.log
    hs = {s: fill(a) for s, a in load(args.pattern).items()}
    rx = re.compile(r"(Mchi-[^_]+_dMchi-[^_]+)_(ctau-[^_]+)$")
    by_mass, by_life = {}, {}
    for s in hs:
        match = rx.match(s)
        if not match:
            raise RuntimeError(f"unexpected sample name {s}")
        m, l = match.groups()
        by_mass.setdefault(m, []).append(s)
        by_life.setdefault(l, []).append(s)
    groups = {"overall": add(list(hs.values()))}
    titles = {"overall": "overall"}
    for m in sorted(by_mass, key=num):
        groups[f"mass/{m}"] = add([hs[s] for s in by_mass[m]])
        titles[f"mass/{m}"] = f"mass {m} (pooled over lifetimes)"
    for l in sorted(by_life, key=num):
        groups[f"lifetime/{l}"] = add([hs[s] for s in by_life[l]])
        titles[f"lifetime/{l}"] = f"lifetime {l} (pooled over masses)"
    for s in sorted(hs):
        groups[f"sample/{s}"] = hs[s]
        titles[f"sample/{s}"] = f"sample {s}"
    os.makedirs(args.outdir, exist_ok=True)
    write_root(os.path.join(args.outdir, "dimuon_vertex_dr_chi2.root"), groups)
    for key, h in groups.items():
        plot_group(h, titles[key], os.path.join(args.outdir, key))
        print(key, h["n"], flush=True)
    plot_overlay({m: groups[f"mass/{m}"] for m in sorted(by_mass, key=num)}, "mass",
                 os.path.join(args.outdir, "overlays"))
    plot_overlay({l: groups[f"lifetime/{l}"] for l in sorted(by_life, key=num)}, "lifetime",
                 os.path.join(args.outdir, "overlays"))


if __name__ == "__main__":
    main()
