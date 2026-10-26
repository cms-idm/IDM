#!/usr/bin/env python3
"""Best-best vertex-fit failures vs gen Lxy, max gen |d0| and Bz at the gen vertex; sigma(Lxy) of valid fits.

python3 vertex_fit_field.py analyze --input-dir RUN/inputs --report-dir RUN/logs --revision <sha> --output RUN/vertex_fit_field.root
python3 vertex_fit_field.py plot --output RUN/vertex_fit_field.root --plot-dir RUN/plots [--log]
Study note: kb/studies/vertex-fit-field.md. Inputs: PreselectedVertexing trees with sigV*, sigB*, bbSigmaLxy.
Population: two accepted gen (twoAcc) with a best-best fit attempted (bbStatus 2 failed or 3 valid); event level,
raw unweighted counts summed over samples. Failed fraction = N(2) / N(2 or 3). Event gen Lxy, Bz = muon 0's
(the two gen vertices are required to agree within 1e-4 cm, else analyze stops). ROOT is the numerical source
(Sumw2, flow bins kept); `plot` reads only that file. --log adds log-y copies of the sigma(Lxy) plots.
"""
import argparse
from array import array
from pathlib import Path
import re
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preselected_vertexing_analysis import check_report, cp, flowcounts, write_hist  # noqa: E402

EDGES = {"lxy": np.arange(0, 741, 20.0), "d0": np.arange(0, 741, 20.0), "bz": np.round(np.arange(-2.5, 4.501, 0.1), 2)}
LABELS = {"lxy": r"Gen production $L_{xy}$ [cm]", "d0": r"Max gen $|d_0|$ to origin [cm]",
          "bz": r"$B_z$ at gen vertex [T]"}
SIGMA_EDGES = np.r_[0, np.logspace(-4, 3, 141)]  # storage binning [cm]; plots choose their own range
SLICES = ((200, 300), (300, 400), (400, 600))  # gen Lxy [cm], DSA-DSA best-best only
SLICE_BZ = np.round(np.arange(-2.5, 4.501, 0.25), 2)
QUANTILES = (("q16", 0.16), ("q50", 0.50), ("q84", 0.84))
VTX_TOL = 1e-4  # cm


def load(args):
    import awkward as ak
    import uproot
    paths = sorted(Path(args.input_dir).glob("*.root"))
    if len(paths) != args.expected_samples:
        raise RuntimeError("Unexpected number of sample ROOT inputs: %d" % len(paths))
    names, revisions, cols = [], set(), []
    for si, path in enumerate(paths):
        with uproot.open(path) as f:
            sample = str(f["preselectedVertexing/sample"].member("fTitle"))
            if sample in names or not re.fullmatch(r"Mchi-[\dp]+_dMchi-[\dp]+_ctau-[\dp]+", sample):
                raise RuntimeError("Unexpected/duplicate sample name: " + sample)
            names.append(sample)
            revisions.add(str(f["preselectedVertexing/revision"].member("fTitle")))
            t = f["preselectedVertexing/events"]
            n = t.num_entries
            if args.report_dir is not None:
                check_report(Path(args.report_dir), sample, n)
            a = t.arrays(["twoAcc", "bbStatus", "bbSigmaLxy", "sigLxy", "sigD0", "sigBz", "sigVx", "sigVy", "sigVz",
                          "matchType"], library="ak")
            two = np.asarray(a["twoAcc"])
            idx = np.flatnonzero(two)

            def per2(x):
                out = np.full((n, 2), np.nan)
                out[idx] = ak.to_numpy(ak.pad_none(x[idx], 2, clip=True)).astype(float)
                return out
            v = np.stack([per2(a[k]) for k in ("sigVx", "sigVy", "sigVz")], axis=2)  # (n, muon, xyz)
            cols.append(dict(sample=np.full(n, si), two=two, status=np.asarray(a["bbStatus"]),
                             sigma=np.asarray(a["bbSigmaLxy"], dtype=float), lxy=per2(a["sigLxy"])[:, 0],
                             d0=np.nanmax(np.where(two[:, None], per2(a["sigD0"]), -1.0), axis=1),
                             bz=per2(a["sigBz"])[:, 0], vdiff=np.abs(v[:, 0] - v[:, 1]).max(axis=1),
                             mtype=per2(a["matchType"])))
    if len(revisions) != 1 or "unspecified" in revisions:
        raise RuntimeError("Mixed/unspecified processing revision")
    e = {k: np.concatenate([c[k] for c in cols]) for k in cols[0]}
    return names, e, next(iter(revisions))


def write_th2(name, title, xedges, yedges, x, y):
    import ROOT
    h = ROOT.TH2D(name, title, len(xedges) - 1, array("d", xedges), len(yedges) - 1, array("d", yedges))
    h.Sumw2()
    bx = np.searchsorted(xedges, x, side="right")  # ROOT bin numbers incl. under/overflow
    by = np.searchsorted(yedges, y, side="right")
    c = np.zeros((len(xedges) + 1, len(yedges) + 1))
    np.add.at(c, (bx, by), 1)
    for i, j in zip(*np.nonzero(c)):
        h.SetBinContent(int(i), int(j), float(c[i, j]))
        h.SetBinError(int(i), int(j), float(np.sqrt(c[i, j])))
    h.SetEntries(float(len(x)))
    h.Write()


def analyze(args):
    import ROOT
    if Path(args.output).exists():
        raise RuntimeError("Refusing to overwrite numerical ROOT source")
    names, e, revision = load(args)
    total = len(e["two"])
    if args.expected_events and total != args.expected_events:
        raise RuntimeError("Event total %d != expected %d" % (total, args.expected_events))
    two, st = e["two"], e["status"]
    if np.any((st > 0) & ~two):
        raise RuntimeError("Fit status outside twoAcc events")
    bad_vtx = int((two & ~(e["vdiff"] <= VTX_TOL)).sum())
    print("GEN-VERTEX CHECK: %d twoAcc events, %d with the two gen vertices differing by > %g cm" %
          (int(two.sum()), bad_vtx, VTX_TOL), flush=True)
    if bad_vtx:
        raise RuntimeError("Gen vertices of the two muons disagree in %d events: ask the user before using muon 0" % bad_vtx)
    att = two & ((st == 2) | (st == 3))
    fail = att & (st == 2)
    valid = att & (st == 3)
    sig_ok = valid & (e["sigma"] > 0)
    print("STATUS (twoAcc): not fitted %d, shared %d, failed %d, valid %d; valid with sigma sentinel %d" % (tuple(
        int((two & (st == s)).sum()) for s in range(4)) + (int((valid & ~sig_ok).sum()),)), flush=True)
    dsa = att & (e["mtype"][:, 0] == 1) & (e["mtype"][:, 1] == 1)

    groups = {"overall": None}
    for si, sample in enumerate(names):
        mass, tau = sample.rsplit("_ctau-", 1)
        groups["sample_" + sample] = {si}
        groups.setdefault("mass_" + mass, set()).add(si)
        groups.setdefault("ctau_" + tau, set()).add(si)
    root = ROOT.TFile.Open(str(args.output), "CREATE")
    if not root or root.IsZombie():
        raise RuntimeError("Cannot create ROOT output")
    ROOT.TNamed("definitions",
        "twoAcc = exactly two opposite-charge signal muons both accepted (|eta|<2.4,0<=Lxy<740,|vz|<960,no pT). "
        "Best-best = hierarchy G>I>DSA matched legs, KalmanVertexFitter(refit), tracker bounds 740/960 cm. "
        "att = twoAcc and bbStatus in {2 failed, 3 valid}; fail = bbStatus 2. Event gen Lxy, Bz = muon 0 "
        "(gen vertices agree within 1e-4 cm, checked); d0 = max gen |d0| to origin. sigma = bbSigmaLxy (vertex "
        "covariance only, beamspot fixed; valid fits with sigma>0). q16/q50/q84 = per-bin quantiles of unbinned sigma. "
        "dsa_* = att with both best-best legs DSA, in gen-Lxy slices. Raw unweighted event counts; flow kept.").Write()
    ROOT.TNamed("analysisRevision", args.revision).Write()
    ROOT.TNamed("inputRevision", revision).Write()
    ROOT.TNamed("samples", "\n".join(names)).Write()
    top = root.mkdir("groups")
    for gname, ids in groups.items():
        d = top.mkdir(gname)
        d.cd()
        sel = np.ones(total, bool) if ids is None else np.isin(e["sample"], list(ids))
        write_hist("status", "best-best status in twoAcc (0 not fitted,1 shared,2 failed,3 valid)",
                   np.arange(5, dtype=float) - 0.5, np.r_[0, [int((sel & two & (st == s)).sum()) for s in range(4)], 0])
        for var, edges in EDGES.items():
            write_hist("att__" + var, LABELS[var], edges, flowcounts(e[var][att & sel], edges))
            write_hist("fail__" + var, LABELS[var], edges, flowcounts(e[var][fail & sel], edges))
        for var in ("lxy", "d0"):
            m = sig_ok & sel
            write_th2("sigma2d__" + var, LABELS[var], EDGES[var], SIGMA_EDGES, e[var][m], e["sigma"][m])
            edges = EDGES[var]
            b = np.searchsorted(edges, e[var][m], side="right") - 1
            for qn, q in QUANTILES:
                h = ROOT.TH1D("%s__%s" % (qn, var), LABELS[var], len(edges) - 1, array("d", edges))
                for i in range(len(edges) - 1):
                    s_ = e["sigma"][m][b == i]
                    if s_.size:
                        h.SetBinContent(i + 1, float(np.quantile(s_, q)))
                h.Write()
            write_hist("nsigma__" + var, LABELS[var], edges, flowcounts(e[var][m], edges))
        if ids is None:
            for lo, hi in SLICES:
                m = dsa & (e["lxy"] >= lo) & (e["lxy"] < hi)
                write_hist("dsa_att__bz__lxy%d_%d" % (lo, hi), LABELS["bz"], SLICE_BZ, flowcounts(e["bz"][m], SLICE_BZ))
                write_hist("dsa_fail__bz__lxy%d_%d" % (lo, hi), LABELS["bz"], SLICE_BZ,
                           flowcounts(e["bz"][m & fail], SLICE_BZ))
    root.Close()
    chk = ROOT.TFile.Open(str(args.output))  # totals incl. flow == selected events (overall)
    for name, m in (("att__lxy", att), ("fail__bz", fail), ("nsigma__d0", sig_ok)):
        hh = chk.Get("groups/overall/" + name)
        if int(hh.Integral(0, hh.GetNbinsX() + 1)) != int(m.sum()):
            raise RuntimeError("Histogram total differs: " + name)
    chk.Close()
    print("VALIDATED: events=%d att=%d fail=%d groups=%d" % (total, int(att.sum()), int(fail.sum()), len(groups)), flush=True)
    for lo, hi in SLICES:
        m = dsa & (e["lxy"] >= lo) & (e["lxy"] < hi)
        bz = e["bz"][m]
        print("DSA-DSA slice %d-%d cm: att %d fail %d; Bz min/16%%/50%%/84%%/max = %s T" % (
            lo, hi, int(m.sum()), int((m & fail).sum()),
            " ".join("%.2f" % x for x in np.quantile(bz, (0, .16, .5, .84, 1))) if bz.size else "n/a"), flush=True)


# ---------------------------------------------------------------------------------------------- plotting
def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import uproot
    out = Path(args.plot_dir)
    for sub in ("", "mass", "lifetime", "sample"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    f = uproot.open(args.output)
    groups = sorted({k.split("/")[1] for k in f.keys() if k.startswith("groups/") and k.count("/") == 2})
    foot = ("Signal MC, best-best fit (G>I>DSA), events with two accepted signal muons and a fit attempted; "
            "raw unweighted; 68% Clopper-Pearson; bins with < 20 attempted fits omitted")
    foot_s = ("Valid best-best fits; sigma(Lxy) from the vertex covariance (beamspot fixed); "
              "band = 16-84% of valid fits per bin (spread, not an uncertainty)")

    def cmap(i, n):
        return plt.get_cmap("turbo")(0.05 + 0.9 * i / max(n - 1, 1)) if n > 10 else plt.get_cmap("tab10")(i)

    def numkey(g):
        nums = [float(x.replace("p", ".")) for x in re.findall(r"(?:Mchi|ctau)-([\dp]+)", g)]
        return nums if nums else [float(g.split("_")[-1])]

    def glabel(g):
        if g == "overall":
            return "overall"
        if g.startswith("ctau_"):
            return "cτ = %s cm" % g.split("_")[1]
        a, b = (float(x.replace("p", ".")) for x in re.findall(r"Mchi-([\dp]+)", g))
        lab = "Mχ=%g, ΔM=%g GeV" % (a, b)
        return lab + (", cτ = %s cm" % g.rsplit("ctau-", 1)[1] if g.startswith("sample_") else "")

    def h1(g, name):
        o = f["groups/%s/%s" % (g, name)]
        return o.values(flow=True)[1:-1], o.axis().edges()

    def frac(g, var, num="fail", den="att"):
        n, edges = h1(g, "%s__%s" % (num, var))
        d, _ = h1(g, "%s__%s" % (den, var))
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(d > 0, n / d, np.nan)
        lo, hi = cp(n, d)
        return edges, r, lo, hi, d

    def curve(ax, edges, r, lo, hi, d, label=None, color="k", bars=True):
        x = 0.5 * (edges[1:] + edges[:-1])
        ok = d >= 20
        if bars:
            ax.errorbar(x[ok], r[ok], yerr=[(r - lo)[ok], (hi - r)[ok]], fmt="o", ms=3, lw=1, label=label, color=color)
        else:
            ax.plot(x[ok], r[ok], "-o", ms=2.5, lw=1, label=label, color=color)

    def save(fig, path, note, log_axes=()):
        fig.text(0.01, 0.005, note, fontsize=6.5, color="0.35")
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(path, dpi=110)
        if args.log and log_axes:
            for a in log_axes:
                a.set_yscale("log")
            fig.savefig(path.with_name(path.stem + "_log.png"), dpi=110)
        plt.close(fig)

    def where(g):
        return out / ("mass" if g.startswith("mass_") else "lifetime" if g.startswith("ctau_")
                      else "sample" if g.startswith("sample_") else "")

    def fail_fig(gs, fname, title):
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
        for i, var in enumerate(EDGES):
            for j, g in enumerate(gs):
                curve(ax[i], *frac(g, var), label=glabel(g), color=cmap(j, len(gs)) if len(gs) > 1 else "k",
                      bars=len(gs) <= 10)
            ax[i].set_xlabel(LABELS[var]); ax[i].set_ylabel("Failed / attempted fits"); ax[i].set_ylim(0, 1.05)
            ax[i].grid(alpha=.3)
        if len(gs) > 1:
            hd, lb = ax[0].get_legend_handles_labels()
            fig.legend(hd, lb, loc="upper center", ncol=min(len(gs), 6), fontsize=6.5)
            fig.subplots_adjust(top=0.8)
        fig.suptitle(title, fontsize=9, y=0.995 if len(gs) <= 1 else 0.88)
        save(fig, fname, foot if len(gs) <= 10 else foot + "; lines connect bins (no error bars)")

    def sigma_fig(g, fname):
        fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))
        for i, var in enumerate(("lxy", "d0")):
            q = {qn: h1(g, "%s__%s" % (qn, var))[0] for qn, _ in QUANTILES}
            n, edges = h1(g, "nsigma__" + var)
            x = 0.5 * (edges[1:] + edges[:-1]); ok = n >= 20
            a = ax[0, i]
            a.fill_between(x[ok], q["q16"][ok], q["q84"][ok], color="C0", alpha=.3, step="mid", label="16–84%")
            a.plot(x[ok], q["q50"][ok], "o-", color="C0", ms=3, label="median")
            a.set_xlabel(LABELS[var]); a.set_ylabel(r"$\sigma(L_{xy})$ [cm]"); a.grid(alpha=.3); a.legend(fontsize=7)
            a.set_ylim(0, None)
            o = f["groups/%s/sigma2d__%s" % (g, var)]
            v = o.values()
            xe, ye = o.axes[0].edges(), o.axes[1].edges()
            b = ax[1, i]
            b.pcolormesh(xe, ye, v.T, cmap="viridis", shading="flat")
            top = ye[min(len(ye) - 1, np.searchsorted(ye, np.nanmax(q["q84"][ok]) * 3 if ok.any() else ye[-1]))]
            b.set_ylim(0, top); b.set_xlabel(LABELS[var]); b.set_ylabel(r"$\sigma(L_{xy})$ [cm]")
            b.set_title("valid fits per cell (raw); cells above the y range in ROOT", fontsize=7)
        fig.suptitle("%s: valid fits %d" % (glabel(g), int(h1(g, "nsigma__lxy")[0].sum())), fontsize=9)
        save(fig, fname, foot_s, log_axes=(ax[0, 0], ax[0, 1]))

    def sigma_overlay(gs, fname, title):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
        for i, var in enumerate(("lxy", "d0")):
            for j, g in enumerate(gs):
                med, edges = h1(g, "q50__" + var)
                n, _ = h1(g, "nsigma__" + var)
                x = 0.5 * (edges[1:] + edges[:-1]); ok = n >= 20
                ax[i].plot(x[ok], med[ok], "-o", ms=2.5, lw=1, color=cmap(j, len(gs)), label=glabel(g))
            ax[i].set_xlabel(LABELS[var]); ax[i].set_ylabel(r"Median $\sigma(L_{xy})$ [cm]"); ax[i].grid(alpha=.3)
            ax[i].set_ylim(0, None)
        hd, lb = ax[0].get_legend_handles_labels()
        fig.legend(hd, lb, loc="upper center", ncol=6, fontsize=6.5)
        fig.subplots_adjust(top=0.8)
        fig.suptitle(title, fontsize=9, y=0.88)
        save(fig, fname, "Valid best-best fits; median per bin; bins with < 20 valid fits omitted", log_axes=ax)

    mg = sorted([g for g in groups if g.startswith("mass_")], key=numkey)
    lg = sorted([g for g in groups if g.startswith("ctau_")], key=numkey)
    for g in groups:
        fail_fig([g], where(g) / ("failfrac__%s.png" % g), glabel(g))
        sigma_fig(g, where(g) / ("sigma__%s.png" % g))
    fail_fig(mg, out / "overlay_mass_failfrac.png", "Mass overlay (each pooled over lifetimes)")
    fail_fig(lg, out / "overlay_lifetime_failfrac.png", "Lifetime overlay (each pooled over masses)")
    sigma_overlay(mg, out / "overlay_mass_sigma.png", "Mass overlay (each pooled over lifetimes)")
    sigma_overlay(lg, out / "overlay_lifetime_sigma.png", "Lifetime overlay (each pooled over masses)")

    fig, ax = plt.subplots(1, len(SLICES), figsize=(5 * len(SLICES), 4.4), sharey=True)
    for a, (lo, hi) in zip(ax, SLICES):
        tag = "bz__lxy%d_%d" % (lo, hi)
        n, edges = h1("overall", "dsa_fail__" + tag)
        d, _ = h1("overall", "dsa_att__" + tag)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(d > 0, n / d, np.nan)
        l_, u_ = cp(n, d)
        curve(a, edges, r, l_, u_, d)
        a.set_title("DSA–DSA, gen $L_{xy}$ %d–%d cm: %d attempted" % (lo, hi, int(d.sum())), fontsize=8)
        a.set_xlabel(LABELS["bz"]); a.set_ylim(0, 1.05); a.grid(alpha=.3)
    ax[0].set_ylabel("Failed / attempted fits")
    save(fig, out / "dsa_slices_failfrac_vs_bz.png", foot.replace("fit attempted", "fit attempted, both legs DSA"))
    print("Plots done in", out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("analyze", "plot"))
    ap.add_argument("--input-dir")
    ap.add_argument("--report-dir")
    ap.add_argument("--revision", default="unspecified")
    ap.add_argument("--output", required=True)
    ap.add_argument("--plot-dir")
    ap.add_argument("--expected-samples", type=int, default=72)
    ap.add_argument("--expected-events", type=int, default=720000)
    ap.add_argument("--log", action="store_true")
    args = ap.parse_args()
    analyze(args) if args.mode == "analyze" else plot(args)


if __name__ == "__main__":
    main()
