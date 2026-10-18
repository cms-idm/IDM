"""Delta phi(Type-1 PF MET, reco dimuon pT) for gen signal muons matched to ID'd PF muons or displacedMuons
(DimuonVertexTruth outputs with pf* branches, from 6307531 on).

  python3 plot_reco_met_dphi.py 'RUNDIR/dimuon_vertex_*.root' OUTDIR [--log]
  python3 plot_reco_met_dphi.py OUTDIR/reco_met_dphi_hists.root OUTDIR --from-root [--log]   (replot only)

Matching (DimuonVertexTruth): only accepted gen signal muons (|eta| < 2.4, 0 <= vxy < 740 cm, |vz| < 960 cm, no pT
minimum) are matched, at station 2 with a station-1 fallback, same reco-gen charge, dR < 0.06 (inner, global)
or < 0.10 (DSA), strict greedy one-to-one. PF: slimmedMuons with isPFMuon and (isGlobalMuon or isTrackerMuon),
representative global track if isGlobalMuon else inner track. displacedMuons: object-level hierarchy (HIER) after
the interim dedup: isGlobalMuon -> global, else isTrackerMuon -> inner (tracker), else DSA-ID'd outer -> DSA;
each object counted once. PF and displacedMuon matching are independent.

Populations (event level, both signal muons; one entry per event, raw counts, unweighted):
  pf_all     both matched to ID'd PF muons
  dm_all     both matched to HIER displacedMuons
  pf_trk / dm_trk   both representatives tracker (inner) type
  pf_glb / dm_glb   both global type
  dm_dsa     both DSA type (displacedMuons only; PF has no DSA)
  dm_notpf   both matched to HIER displacedMuons and neither matched to an ID'd PF muon
Mixed-type events (e.g. one global, one tracker leg) are in *_all only; their counts are in `counts`.

Observable: reco pT^mumu = vector sum of the two matched representative tracks (at their reference points).
Delta phi = phi(Type-1 PF MET) - phi(reco mumu) wrapped to [-pi, pi); |Delta phi| in [0, pi] (exactly pi goes in
the last bin). max gen |d0| = larger of the two signal muons' |vx sin(phi) - vy cos(phi)| (straight line to (0,0)).

Histograms (OUTDIR/reco_met_dphi_hists.root, one directory per group, under/overflow kept): per population
dphi_<pop> (-pi..pi, 40 bins), absdphi_<pop> (0..pi, 20 bins), d0_absdphi_<pop> (max gen |d0| 0-30 cm in 1 cm
x |Delta phi|); `counts`. Groups: overall, per mass point (pooled over lifetimes, unweighted), per lifetime
(pooled over masses, unweighted), per sample.

PNGs (linear): per group all.png (pf_all and dm_all), tracker.png, global.png (PF vs displacedMuons overlaid),
dsa.png, notpf.png, each also _unitArea; vs_d0.png (heatmaps, all populations). overlays/ mass_* (all masses,
each pooled over lifetimes) and lifetime_* (all lifetimes, each pooled over masses), raw and unit area. --log
adds *_log.png copies; they never replace the linear plots.
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

BRANCHES = ["genAccepted", "genD0", "pfMatched", "pfType", "pfPt", "pfPhi", "hierMatched", "hierType", "hierPt",
            "hierPhi", "pfMetPt", "pfMetPhi"]
POPS = {  # name: (legend label, colour)
    "pf_all": ("ID'd PF muons (all types)", "tab:blue"),
    "dm_all": ("displacedMuons HIER (all types)", "tab:red"),
    "pf_trk": ("PF, both tracker (inner)", "tab:blue"),
    "dm_trk": ("displacedMuons, both tracker (inner)", "tab:red"),
    "pf_glb": ("PF, both global", "tab:blue"),
    "dm_glb": ("displacedMuons, both global", "tab:red"),
    "dm_dsa": ("displacedMuons, both DSA", "tab:green"),
    "dm_notpf": ("displacedMuons HIER, neither leg PF-matched", "tab:purple"),
}
FIGS = {  # file stem: (populations, title)
    "all": (["pf_all", "dm_all"], "all ID'd PF vs all HIER displacedMuons"),
    "tracker": (["pf_trk", "dm_trk"], "both legs tracker (inner) type"),
    "global": (["pf_glb", "dm_glb"], "both legs global type"),
    "dsa": (["dm_dsa"], "both legs DSA type (displacedMuons only)"),
    "notpf": (["dm_notpf"], "both legs HIER displacedMuons, neither PF-matched"),
}
COUNTS = ["events (2 signal muons)", "both accepted"] + list(POPS) + ["pf mixed types", "dm mixed types"]
SIGNED = np.linspace(-np.pi, np.pi, 41)
ABS = np.linspace(0, np.pi, 21)
D0 = np.linspace(0, 30, 31)
LOG = False


def flow_edges(edges):
    return np.concatenate([[-np.inf], edges, [np.inf]])


def wrap(x):
    return np.mod(x + np.pi, 2 * np.pi) - np.pi


def dimuon_phi(pt, phi):
    pt, phi = pt.astype(float), phi.astype(float)
    return np.arctan2((pt * np.sin(phi)).sum(1), (pt * np.cos(phi)).sum(1))


def load(pattern):
    data = {}
    for fn in sorted(glob.glob(pattern)):
        f = uproot.open(fn)
        sample = str(f["dimuonVertexTruth/sample"].member("fTitle"))
        if sample in data:
            raise RuntimeError(f"sample {sample} appears in more than one input file ({fn})")
        a = f["dimuonVertexTruth/events"].arrays(BRANCHES, library="np")
        acc = a["genAccepted"].sum(1) == 2
        pf = a["pfMatched"].sum(1) == 2
        dm = a["hierMatched"].sum(1) == 2
        sel = {"pf_all": pf, "dm_all": dm,
               "pf_trk": pf & (a["pfType"] == 0).all(1), "dm_trk": dm & (a["hierType"] == 0).all(1),
               "pf_glb": pf & (a["pfType"] == 1).all(1), "dm_glb": dm & (a["hierType"] == 1).all(1),
               "dm_dsa": dm & (a["hierType"] == 2).all(1),
               "dm_notpf": dm & (a["pfMatched"].sum(1) == 0)}
        if (sel["pf_all"] & ~acc).any() or (sel["dm_all"] & ~acc).any():
            raise RuntimeError(f"{fn}: matched events with a non-accepted gen muon")
        if ((a["pfMetPt"] == -999) & (pf | dm)).any():
            raise RuntimeError(f"{fn}: selected events without PF MET")
        dphi = {"pf": wrap(a["pfMetPhi"].astype(float) - dimuon_phi(a["pfPt"], a["pfPhi"])),
                "dm": wrap(a["pfMetPhi"].astype(float) - dimuon_phi(a["hierPt"], a["hierPhi"]))}
        maxd0 = np.abs(a["genD0"]).max(1).astype(float)
        d = {"counts": np.array([len(acc), acc.sum()] + [sel[p].sum() for p in POPS] +
                                [(pf & (a["pfType"][:, 0] != a["pfType"][:, 1])).sum(),
                                 (dm & (a["hierType"][:, 0] != a["hierType"][:, 1])).sum()])}
        for p in POPS:
            d[p] = (dphi[p[:2]][sel[p]], maxd0[sel[p]])
        data[sample] = d
        print(f"{sample}: " + ", ".join(f"{k} {int(v)}" for k, v in zip(COUNTS, d["counts"])), flush=True)
    if not data:
        raise RuntimeError(f"no files match {pattern}")
    return data


def fill(d):
    h = {"counts": d["counts"]}
    for p in POPS:
        x, d0 = d[p]
        ax = np.minimum(np.abs(x), np.nextafter(np.pi, 0))
        h[f"dphi_{p}"] = np.histogram(x, flow_edges(SIGNED))[0]
        h[f"absdphi_{p}"] = np.histogram(ax, flow_edges(ABS))[0]
        h[f"d0_absdphi_{p}"] = np.histogram2d(d0, ax, [flow_edges(D0), flow_edges(ABS)])[0]
    return h


def add(hs):
    return {k: sum(h[k] for h in hs) for k in hs[0]}


def n_of(h, p):
    return int(h["counts"][COUNTS.index(p)])


def write_root(path, groups):
    """TH1D/TH2D per group with under/overflow; unweighted counts (no Sumw2 needed)."""
    f = ROOT.TFile(path, "RECREATE")
    ROOT.TNamed("definitions", __doc__).Write()
    for key, h in groups.items():
        f.mkdir(key)
        f.cd(key)
        t = ROOT.TH1D("counts", ";;events", len(COUNTS), 0, len(COUNTS))
        for i, (lab, c) in enumerate(zip(COUNTS, h["counts"])):
            t.GetXaxis().SetBinLabel(i + 1, lab)
            t.SetBinContent(i + 1, c)
        t.Write()
        for p in POPS:
            for kind, e, s in (("dphi", SIGNED, "#Delta#phi"), ("absdphi", ABS, "|#Delta#phi|")):
                k = f"{kind}_{p}"
                t = ROOT.TH1D(k, f"{POPS[p][0]};{s}(PF MET, reco p_{{T}}^{{#mu#mu}}) [rad];events per bin", len(e) - 1, e)
                for i, c in enumerate(h[k]):
                    t.SetBinContent(i, c)
                t.SetEntries(n_of(h, p))
                t.Write()
            k = f"d0_absdphi_{p}"
            t = ROOT.TH2D(k, f"{POPS[p][0]};max gen |d_{{0}}| [cm];|#Delta#phi|(PF MET, reco p_{{T}}^{{#mu#mu}}) [rad]",
                          len(D0) - 1, D0, len(ABS) - 1, ABS)
            for i in range(h[k].shape[0]):
                for j in range(h[k].shape[1]):
                    t.SetBinContent(i, j, h[k][i, j])
            t.SetEntries(n_of(h, p))
            t.Write()
    f.Close()


def read_root(path):
    """Groups from a file written by write_root; refuses binnings that differ from this script."""
    f = uproot.open(path)
    groups = {}
    for key in f.keys(recursive=True, cycle=False):
        if not key.endswith("/counts"):
            continue
        g = key[: -len("/counts")]
        h = {"counts": f[key].values()}
        if len(h["counts"]) != len(COUNTS):
            raise RuntimeError(f"{path}: {g}/counts has {len(h['counts'])} bins, expected {len(COUNTS)}")
        for p in POPS:
            for kind, e in (("dphi", SIGNED), ("absdphi", ABS)):
                t = f[f"{g}/{kind}_{p}"]
                if not np.allclose(t.axis().edges(), e):
                    raise RuntimeError(f"{path}: {g}/{kind}_{p} binning differs from this script")
                h[f"{kind}_{p}"] = t.values(flow=True)
            t = f[f"{g}/d0_absdphi_{p}"]
            if not (np.allclose(t.axis(0).edges(), D0) and np.allclose(t.axis(1).edges(), ABS)):
                raise RuntimeError(f"{path}: {g}/d0_absdphi_{p} binning differs from this script")
            h[f"d0_absdphi_{p}"] = t.values(flow=True)
        groups[g] = h
    return groups


def xlabel(kind):
    s = "|Δφ|" if kind == "absdphi" else "Δφ = φ(MET) − φ(μμ)"
    return f"{s}  (Type-1 PF MET vs reco pT^μμ) [rad]"


def draw_1d(ax, h, k, n, label, color, unit_area):
    e = SIGNED if k.startswith("dphi") else ABS
    y = h[k][1:-1].astype(float)
    if unit_area:
        y = y / max(n, 1)
    ax.stairs(y, e, color=color, label=f"{label} (N = {n})")


def finish_1d(ax, kind, unit_area, log):
    ax.set_xlabel(xlabel(kind))
    ax.set_ylabel("fraction of events per bin" if unit_area else "events per bin")
    ax.set_xlim(-np.pi if kind == "dphi" else 0, np.pi)
    if log:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)


def head(h, title):
    c = dict(zip(COUNTS, h["counts"].astype(int)))
    return (f"{title} | {c['both accepted']} of {c['events (2 signal muons)']} events both accepted | mixed-type events "
            f"(in 'all' only): PF {c['pf mixed types']}, displacedMuons {c['dm mixed types']} | unweighted")


def plot_group(h, title, outdir):
    os.makedirs(outdir, exist_ok=True)
    for log in [False] + ([True] if LOG else []):
        for stem, (pops, ftitle) in FIGS.items():
            for unit_area in (False, True):
                fig, axs = plt.subplots(1, 2, figsize=(13, 5))
                for ax, kind in zip(axs, ("dphi", "absdphi")):
                    for p in pops:
                        draw_1d(ax, h, f"{kind}_{p}", n_of(h, p), POPS[p][0], POPS[p][1], unit_area)
                    finish_1d(ax, kind, unit_area, log)
                    ax.legend(fontsize=8)
                norm = "unit area" if unit_area else "raw counts"
                fig.suptitle(f"{head(h, title)}\n{ftitle} | {norm} | {'log' if log else 'linear'}", fontsize=9)
                fig.tight_layout(rect=(0, 0, 1, 0.9))
                fig.savefig(os.path.join(outdir, f"{stem}{'_unitArea' if unit_area else ''}{'_log' if log else ''}.png"),
                            dpi=90)
                plt.close(fig)
        fig, axs = plt.subplots(2, 4, figsize=(22, 9))
        for ax, p in zip(axs.flat, POPS):
            z = h[f"d0_absdphi_{p}"][1:-1, 1:-1].T
            norm = LogNorm(vmin=1, vmax=max(z.max(), 1)) if log else None
            pc = ax.pcolormesh(D0, ABS, np.ma.masked_where(z == 0, z) if log else z, norm=norm, cmap="viridis")
            fig.colorbar(pc, ax=ax, label="events per bin")
            ov = h[f"d0_absdphi_{p}"][-1, :].sum()
            ax.set_title(f"{POPS[p][0]}\nN = {n_of(h, p)}; max |d0| > 30 cm: {ov / max(n_of(h, p), 1):.1%}", fontsize=8)
            ax.set_xlabel("max gen |d0| [cm]  (|vx sinφ − vy cosφ|, to (0,0))", fontsize=8)
            ax.set_ylabel("|Δφ| (PF MET, reco pT^μμ) [rad]", fontsize=8)
        fig.suptitle(f"{head(h, title)} | {'log' if log else 'linear'}", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(os.path.join(outdir, f"vs_d0{'_log' if log else ''}.png"), dpi=80)
        plt.close(fig)


def plot_overlay(hs, kind_group, outdir):
    """hs: ordered {label: hist}; kind_group 'mass' (each pooled over lifetimes) or 'lifetime' (pooled over masses)."""
    os.makedirs(outdir, exist_ok=True)
    pooled = "pooled over lifetimes" if kind_group == "mass" else "pooled over masses"
    plural = "masses" if kind_group == "mass" else "lifetimes"
    colors = plt.cm.tab20(np.linspace(0, 1, 20))[: len(hs)] if len(hs) > 10 else plt.cm.tab10(range(len(hs)))
    for p in POPS:
        for kind in ("dphi", "absdphi"):
            for unit_area in (False, True):
                for log in [False] + ([True] if LOG else []):
                    fig, ax = plt.subplots(figsize=(9, 6))
                    for (label, h), c in zip(hs.items(), colors):
                        draw_1d(ax, h, f"{kind}_{p}", n_of(h, p), label, c, unit_area)
                    finish_1d(ax, kind, unit_area, log)
                    ax.legend(fontsize=7, ncol=2 if len(hs) > 8 else 1)
                    norm = "unit area" if unit_area else "raw counts"
                    ax.set_title(f"{POPS[p][0]} | all {plural}, each {pooled} | unweighted | {norm} | "
                                 f"{'log' if log else 'linear'}", fontsize=8)
                    fig.tight_layout()
                    name = f"{kind_group}_{kind}_{p}{'_unitArea' if unit_area else ''}{'_log' if log else ''}.png"
                    fig.savefig(os.path.join(outdir, name), dpi=90)
                    plt.close(fig)


def num(s):
    return float(re.search(r"-([0-9p]+)", s).group(1).replace("p", "."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern", help="DimuonVertexTruth output glob, or with --from-root the reco_met_dphi_hists.root")
    ap.add_argument("outdir")
    ap.add_argument("--log", action="store_true", help="also write log-scale copies (*_log.png)")
    ap.add_argument("--from-root", action="store_true", help="replot from saved histograms; no tree pass")
    args = ap.parse_args()
    global LOG
    LOG = args.log
    if args.from_root:
        groups = read_root(args.pattern)
    else:
        hs = {s: fill(d) for s, d in load(args.pattern).items()}
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
        for m in by_mass:
            groups[f"mass/{m}"] = add([hs[s] for s in by_mass[m]])
        for l in by_life:
            groups[f"lifetime/{l}"] = add([hs[s] for s in by_life[l]])
        for s in sorted(hs):
            groups[f"sample/{s}"] = hs[s]
        os.makedirs(args.outdir, exist_ok=True)
        write_root(os.path.join(args.outdir, "reco_met_dphi_hists.root"), groups)
    masses = sorted((k.split("/", 1)[1] for k in groups if k.startswith("mass/")), key=num)
    lifes = sorted((k.split("/", 1)[1] for k in groups if k.startswith("lifetime/")), key=num)
    titles = {"overall": "overall"}
    titles.update({f"mass/{m}": f"mass {m} (pooled over lifetimes)" for m in masses})
    titles.update({f"lifetime/{l}": f"lifetime {l} (pooled over masses)" for l in lifes})
    titles.update({k: f"sample {k[7:]}" for k in groups if k.startswith("sample/")})
    for key, h in groups.items():
        plot_group(h, titles[key], os.path.join(args.outdir, key))
        print(key, [int(c) for c in h["counts"]], flush=True)
    plot_overlay({m: groups[f"mass/{m}"] for m in masses}, "mass", os.path.join(args.outdir, "overlays"))
    plot_overlay({l: groups[f"lifetime/{l}"] for l in lifes}, "lifetime", os.path.join(args.outdir, "overlays"))


if __name__ == "__main__":
    main()
