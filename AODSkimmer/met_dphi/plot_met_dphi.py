"""Delta phi(MET, gen signal dimuon pT), signed and absolute, from fill_met_dphi.py trees.

  python3 plot_met_dphi.py 'RUNDIR/trees/met_dphi_*.root' OUTDIR [--log]
  python3 plot_met_dphi.py OUTDIR/met_dphi_hists.root OUTDIR --from-root [--log]   (replot only)

Population: events with exactly two gen signal muons, one mu- and one mu+ (fill_met_dphi.py `valid`); events
failing it are counted per sample (`counts` histogram), not plotted. One entry per event, raw counts,
unweighted. Selections: noCuts (no further cut) and accepted (both signal muons |eta| < 2.4,
0 <= vxy < 740 cm, |vz| < 960 cm at the gen production vertex; no pT minimum).

Observables: pT^mumu = vector sum of the two gen signal muons. Delta phi = phi(MET) - phi(mumu) wrapped to
[-pi, pi); |Delta phi| in [0, pi] (a value of exactly pi is put in the last bin). MET: pf = slimmedMETs Type-1
PF MET; gen = genMetTrue. The ntuple branch genSigDimuon_METdPhi is phi(mumu) - phi(genMET), i.e. minus the
gen signed value here. Reco MET against gen dimuon: no reco muons involved. max gen |d0|: larger of the two
signal muons' |vx sin(phi) - vy cos(phi)|, straight line to (0,0).

Histograms (OUTDIR/met_dphi_hists.root, one directory per group, under/overflow kept): for met in (pf, gen)
and sel in (noCuts, accepted): dphi_<met>_<sel> (-pi..pi, 40 bins), absdphi_<met>_<sel> (0..pi, 20 bins),
d0_absdphi_<met>_<sel> (max gen |d0| 0-30 cm in 1 cm x |Delta phi| 20 bins); `counts` (all events, valid,
accepted). Groups: overall, per mass point (pooled over lifetimes, unweighted), per lifetime (pooled over
masses, unweighted), per sample.

PNGs (linear): per group dphi.png and dphi_unitArea.png (rows pf / gen MET, columns signed / absolute, no cuts
and accepted overlaid) and vs_d0.png (heatmaps); overlays/ mass_* (all masses, each pooled over lifetimes) and
lifetime_* (all lifetimes, each pooled over masses), raw counts and unit area (_unitArea). --log adds *_log.png
copies; they never replace the linear plots.
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

METS = {"pf": "Type-1 PF MET", "gen": "gen MET"}
SELS = {"noCuts": "no cuts", "accepted": "both muons accepted"}
SEL_COLORS = {"noCuts": "k", "accepted": "tab:red"}
SIGNED = np.linspace(-np.pi, np.pi, 41)
ABS = np.linspace(0, np.pi, 21)
D0 = np.linspace(0, 30, 31)
COUNTS = ["all events", "valid signal dimuon", "both accepted"]
LOG = False


def flow_edges(edges):
    return np.concatenate([[-np.inf], edges, [np.inf]])


def wrap(x):
    return np.mod(x + np.pi, 2 * np.pi) - np.pi


def h1_label(kind, met):
    s = "|#Delta#phi|" if kind == "absdphi" else "#Delta#phi"
    return f"{s}({METS[met]}, gen p_{{T}}^{{#mu#mu}}) [rad]"


def plain_label(kind, met):
    s = "|Δφ|" if kind == "absdphi" else "Δφ = φ(MET) − φ(μμ)"
    return f"{s}  ({METS[met]} vs gen pT^μμ) [rad]"


def load(pattern):
    data = {}
    for fn in sorted(glob.glob(pattern)):
        f = uproot.open(fn)
        sample = str(f["sample"].member("fTitle"))
        if sample in data:
            raise RuntimeError(f"sample {sample} appears in more than one input file ({fn})")
        a = f["events"].arrays(library="np")
        v = a["valid"] == 1
        acc = v.copy()
        for i in (0, 1):
            acc &= (np.abs(a[f"mu{i}_eta"]) < 2.4) & (a[f"mu{i}_vxy"] >= 0) & (a[f"mu{i}_vxy"] < 740) & \
                   (np.abs(a[f"mu{i}_vz"]) < 960)
        for met in METS:
            bad = v & (a[f"{met}met_pt"] == -999)
            if bad.any():
                raise RuntimeError(f"{fn}: {bad.sum()} valid events without {met} MET")
        d = {"counts": np.array([len(v), v.sum(), acc.sum()]), "acc": acc[v],
             "maxD0": np.maximum(a["mu0_d0"], a["mu1_d0"])[v].astype(float)}
        for met in METS:
            d[f"dphi_{met}"] = wrap(a[f"{met}met_phi"][v].astype(float) - a["mm_phi"][v].astype(float))
        data[sample] = d
        print(f"{sample}: {len(v)} events, {v.sum()} valid, {acc.sum()} accepted", flush=True)
    if not data:
        raise RuntimeError(f"no files match {pattern}")
    return data


def fill(d):
    h = {"counts": d["counts"]}
    for met in METS:
        x = d[f"dphi_{met}"]
        ax = np.minimum(np.abs(x), np.nextafter(np.pi, 0))
        for sel in SELS:
            m = d["acc"] if sel == "accepted" else np.ones_like(d["acc"])
            h[f"dphi_{met}_{sel}"] = np.histogram(x[m], flow_edges(SIGNED))[0]
            h[f"absdphi_{met}_{sel}"] = np.histogram(ax[m], flow_edges(ABS))[0]
            h[f"d0_absdphi_{met}_{sel}"] = np.histogram2d(d["maxD0"][m], ax[m], [flow_edges(D0), flow_edges(ABS)])[0]
    return h


def add(hs):
    return {k: sum(h[k] for h in hs) for k in hs[0]}


def n_of(h, sel):
    return int(h["counts"][2] if sel == "accepted" else h["counts"][1])


def write_root(path, groups):
    """TH1D/TH2D per group with under/overflow; unweighted counts (no Sumw2 needed)."""
    f = ROOT.TFile(path, "RECREATE")
    ROOT.TNamed("definitions", __doc__).Write()
    for key, h in groups.items():
        f.mkdir(key)
        f.cd(key)
        t = ROOT.TH1D("counts", ";;events", 3, 0, 3)
        for i, (lab, c) in enumerate(zip(COUNTS, h["counts"])):
            t.GetXaxis().SetBinLabel(i + 1, lab)
            t.SetBinContent(i + 1, c)
        t.Write()
        for met in METS:
            for sel in SELS:
                for kind, e in (("dphi", SIGNED), ("absdphi", ABS)):
                    k = f"{kind}_{met}_{sel}"
                    t = ROOT.TH1D(k, f"{SELS[sel]};{h1_label(kind, met)};events per bin", len(e) - 1, e)
                    for i, c in enumerate(h[k]):
                        t.SetBinContent(i, c)
                    t.SetEntries(n_of(h, sel))
                    t.Write()
                k = f"d0_absdphi_{met}_{sel}"
                t = ROOT.TH2D(k, f"{SELS[sel]};max gen |d_{{0}}| [cm];{h1_label('absdphi', met)}",
                              len(D0) - 1, D0, len(ABS) - 1, ABS)
                for i in range(h[k].shape[0]):
                    for j in range(h[k].shape[1]):
                        t.SetBinContent(i, j, h[k][i, j])
                t.SetEntries(n_of(h, sel))
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
        for met in METS:
            for sel in SELS:
                for kind, e in (("dphi", SIGNED), ("absdphi", ABS)):
                    t = f[f"{g}/{kind}_{met}_{sel}"]
                    if not np.allclose(t.axis().edges(), e):
                        raise RuntimeError(f"{path}: {g}/{kind}_{met}_{sel} binning differs from this script")
                    h[f"{kind}_{met}_{sel}"] = t.values(flow=True)
                t = f[f"{g}/d0_absdphi_{met}_{sel}"]
                if not (np.allclose(t.axis(0).edges(), D0) and np.allclose(t.axis(1).edges(), ABS)):
                    raise RuntimeError(f"{path}: {g}/d0_absdphi_{met}_{sel} binning differs from this script")
                h[f"d0_absdphi_{met}_{sel}"] = t.values(flow=True)
        groups[g] = h
    return groups


def draw_1d(ax, h, k, n, label, color, unit_area):
    e = SIGNED if k.startswith("dphi") else ABS
    y = h[k][1:-1].astype(float)
    if unit_area:
        y = y / max(n, 1)
    ax.stairs(y, e, color=color, label=f"{label} (N = {n})")


def finish_1d(ax, kind, met, unit_area, log):
    ax.set_xlabel(plain_label(kind, met))
    ax.set_ylabel("fraction of events per bin" if unit_area else "events per bin")
    ax.set_xlim(-np.pi if kind == "dphi" else 0, np.pi)
    if log:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)


def plot_group(h, title, outdir):
    os.makedirs(outdir, exist_ok=True)
    c = h["counts"]
    head = f"{title} | events: {int(c[0])} total, {int(c[1])} valid signal dimuon, {int(c[2])} both accepted | unweighted"
    for log in [False] + ([True] if LOG else []):
        for unit_area in (False, True):
            fig, axs = plt.subplots(2, 2, figsize=(13, 9))
            for r, met in enumerate(METS):
                for col, kind in enumerate(("dphi", "absdphi")):
                    ax = axs[r, col]
                    for sel in SELS:
                        draw_1d(ax, h, f"{kind}_{met}_{sel}", n_of(h, sel), SELS[sel], SEL_COLORS[sel], unit_area)
                    finish_1d(ax, kind, met, unit_area, log)
                    ax.legend(fontsize=8)
            norm = "unit area" if unit_area else "raw counts"
            fig.suptitle(f"{head}\n{norm} | {'log' if log else 'linear'}", fontsize=9)
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            fig.savefig(os.path.join(outdir, f"dphi{'_unitArea' if unit_area else ''}{'_log' if log else ''}.png"), dpi=90)
            plt.close(fig)
        fig, axs = plt.subplots(2, 2, figsize=(13, 9))
        for r, met in enumerate(METS):
            for col, sel in enumerate(SELS):
                ax = axs[r, col]
                z = h[f"d0_absdphi_{met}_{sel}"][1:-1, 1:-1].T
                norm = LogNorm(vmin=1, vmax=max(z.max(), 1)) if log else None
                pc = ax.pcolormesh(D0, ABS, np.ma.masked_where(z == 0, z) if log else z, norm=norm, cmap="viridis")
                fig.colorbar(pc, ax=ax, label="events per bin")
                ov = h[f"d0_absdphi_{met}_{sel}"][-1, :].sum()
                ax.set_title(f"{SELS[sel]} (N = {n_of(h, sel)}; max |d0| > 30 cm: {ov / max(n_of(h, sel), 1):.1%})",
                             fontsize=9)
                ax.set_xlabel("max gen |d0| [cm]  (|vx sinφ − vy cosφ|, to (0,0))")
                ax.set_ylabel(plain_label("absdphi", met))
        fig.suptitle(f"{head} | {'log' if log else 'linear'}", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(os.path.join(outdir, f"vs_d0{'_log' if log else ''}.png"), dpi=90)
        plt.close(fig)


def plot_overlay(hs, kind_group, outdir):
    """hs: ordered {label: hist}; kind_group 'mass' (each pooled over lifetimes) or 'lifetime' (pooled over masses)."""
    os.makedirs(outdir, exist_ok=True)
    pooled = "pooled over lifetimes" if kind_group == "mass" else "pooled over masses"
    plural = "masses" if kind_group == "mass" else "lifetimes"
    colors = plt.cm.tab20(np.linspace(0, 1, 20))[: len(hs)] if len(hs) > 10 else plt.cm.tab10(range(len(hs)))
    for met in METS:
        for sel in SELS:
            for kind in ("dphi", "absdphi"):
                for unit_area in (False, True):
                    for log in [False] + ([True] if LOG else []):
                        fig, ax = plt.subplots(figsize=(9, 6))
                        for (label, h), c in zip(hs.items(), colors):
                            draw_1d(ax, h, f"{kind}_{met}_{sel}", n_of(h, sel), label, c, unit_area)
                        finish_1d(ax, kind, met, unit_area, log)
                        ax.legend(fontsize=7, ncol=2 if len(hs) > 8 else 1)
                        norm = "unit area" if unit_area else "raw counts"
                        ax.set_title(f"all {plural}, each {pooled} | {SELS[sel]} | unweighted | {norm} | "
                                     f"{'log' if log else 'linear'}", fontsize=8)
                        fig.tight_layout()
                        name = f"{kind_group}_{kind}_{met}_{sel}{'_unitArea' if unit_area else ''}{'_log' if log else ''}.png"
                        fig.savefig(os.path.join(outdir, name), dpi=90)
                        plt.close(fig)


def num(s):
    return float(re.search(r"-([0-9p]+)", s).group(1).replace("p", "."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern", help="tree glob, or with --from-root the met_dphi_hists.root to replot")
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
        write_root(os.path.join(args.outdir, "met_dphi_hists.root"), groups)
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
