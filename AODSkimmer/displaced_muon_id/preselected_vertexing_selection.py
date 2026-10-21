#!/usr/bin/env python3
"""Cost of vertex-quality and topology requirements on the "any valid typed combo" sample (no new MINIAOD processing).

python3 preselected_vertexing_selection.py analyze --input-dir RUN/inputs --output RUN/selection.root --revision <sha>
python3 preselected_vertexing_selection.py plot --output RUN/selection.root --plot-dir RUN/plots_selection [--log]

Baseline: events with two accepted gen and >= 1 valid typed combo (any of the 9 ordered leg-type combos, one vertex fit each).
An event passes a requirement if AT LEAST ONE valid combo passes it; for a chain of requirements the SAME combo must pass
all of them. Requirements: chi2/ndof < 4 (ndof = 1); dR(mu, mu) < 0.9 in four versions; |dphi(pTmiss, pT_mumu)| < 0.5 in
two versions (dphi = phi(MET) - phi(mumu), wrapped; missing MET or an unavailable position fails the requirement).
  dR versions: reco at the track reference point (point of closest approach; momentum direction of the two legs),
  reco at station 2 (propagated position of the two legs' components), gen at production (momentum directions),
  gen at station 2 (propagated final-state gen muons; propagation failure fails the requirement).
  dphi versions: Type-1 PF MET with the vector sum of the two legs' reference-point pT; genMetTrue with the vector sum
  of the two gen signal-muon pT.
Chains: A reco(PCA)+PF, B reco(station 2)+PF, C gen(production)+genMET, D gen(station 2)+genMET.
Shape plots: dphi of the P1 vertex (first valid combo under priority Global>Inner>DSA, one per event) vs the gen version.
"""
import argparse
from array import array
from pathlib import Path
import re
import numpy as np

CHI2_CUT, DR_CUT, DPHI_CUT = 4.0, 0.9, 0.5
DRV = ("drPCA", "drST2", "drGenVtx", "drGenST2")
DRV_LABEL = {"drPCA": "reco, track reference point (closest approach)", "drST2": "reco, station 2",
             "drGenVtx": "gen, production vertex", "drGenST2": "gen, station 2"}
DPV = ("dphiPF", "dphiGen")
DPV_LABEL = {"dphiPF": "PF MET vs reco dimuon pT", "dphiGen": "gen MET vs gen dimuon pT"}
CHAINS = {"A": ("drPCA", "dphiPF"), "B": ("drST2", "dphiPF"), "C": ("drGenVtx", "dphiGen"), "D": ("drGenST2", "dphiGen")}
CHAIN_LABEL = {"A": "A: reco dR at closest approach + PF MET", "B": "B: reco dR at station 2 + PF MET",
               "C": "C: gen dR at production + gen MET", "D": "D: gen dR at station 2 + gen MET"}
RANK_P1 = {2: 0, 0: 1, 1: 2}  # Global > Inner > DSA (types: 0 inner, 1 outer/DSA, 2 global)
DPHI_EDGES = {"dphi": np.linspace(-np.pi, np.pi, 33), "absdphi": np.linspace(0, np.pi, 33)}


def wrap(x):
    x = np.where(np.abs(x) > 500, np.nan, x)
    return (x + np.pi) % (2 * np.pi) - np.pi


def load(args):
    import uproot
    paths = sorted(Path(args.input_dir).glob("*.root"))
    if len(paths) != args.expected_samples:
        raise RuntimeError("Unexpected number of inputs: %d" % len(paths))
    out = {k: [] for k in ("ev", "chi2", "drPCA", "drST2", "recophi", "key")}
    ev_cols = {k: [] for k in ("drGenVtx", "drGenST2", "genphi", "genmetphi", "pfmetphi")}
    names, base = [], 0
    for path in paths:
        with uproot.open(path) as f:
            sample = str(f["preselectedVertexing/sample"].member("fTitle"))
            if sample in names or not re.fullmatch(r"Mchi-[\dp]+_dMchi-[\dp]+_ctau-[\dp]+", sample):
                raise RuntimeError("Bad/duplicate sample " + sample)
            names.append(sample)
            a = f["preselectedVertexing/events"].arrays(
                ["twoAcc", "sigProp", "sigEta2", "sigPhi2", "genDR", "genMuMuPhi", "genMetPhi", "pfMetPhi",
                 "cObj", "cType", "cProp", "cEta2", "cPhi2", "indObj"], library="np")
            k = f["preselectedVertexing/combos"].arrays(library="np")
        n = len(a["twoAcc"])
        two = np.asarray(a["twoAcc"])
        drst2 = np.full(n, np.nan)
        for ei in np.flatnonzero(two):
            sp = np.asarray(a["sigProp"][ei])
            if len(sp) == 2 and sp.all():
                e_, p_ = np.asarray(a["sigEta2"][ei]), np.asarray(a["sigPhi2"][ei])
                drst2[ei] = np.hypot(e_[0] - e_[1], (p_[0] - p_[1] + np.pi) % (2 * np.pi) - np.pi)
        ev_cols["drGenVtx"].append(np.where(two, np.asarray(a["genDR"], dtype=float), np.nan))
        ev_cols["drGenST2"].append(drst2)
        ev_cols["genphi"].append(np.asarray(a["genMuMuPhi"], dtype=float))
        ev_cols["genmetphi"].append(np.asarray(a["genMetPhi"], dtype=float))
        ev_cols["pfmetphi"].append(np.asarray(a["pfMetPhi"], dtype=float))
        ke = k["kEvent"]
        valid = np.flatnonzero(k["kStatus"] == 3)
        # reco dR at station 2 for the valid combos: legs are the matched components (obj, type) of gen0 / gen1
        cache_ev, comps, ind = -1, None, None
        dr2 = np.full(len(valid), np.nan)
        for j, r in enumerate(valid):
            e = int(ke[r])
            if e != cache_ev:
                cache_ev = e
                comps = {(int(o), int(t)): (int(p), float(et), float(ph)) for o, t, p, et, ph in
                         zip(a["cObj"][e], a["cType"][e], a["cProp"][e], a["cEta2"][e], a["cPhi2"][e])}
                ind = np.asarray(a["indObj"][e]).reshape(3, 2)
            ta, tb = int(k["kTypeA"][r]), int(k["kTypeB"][r])
            p0 = comps.get((int(ind[ta, 0]), ta)); p1 = comps.get((int(ind[tb, 1]), tb))
            if p0 and p1 and p0[0] and p1[0]:
                dr2[j] = np.hypot(p0[1] - p1[1], (p0[2] - p1[2] + np.pi) % (2 * np.pi) - np.pi)
        out["ev"].append(ke[valid] + base)
        out["chi2"].append(k["kChi2"][valid] / np.maximum(k["kNdof"][valid], 1e-9))
        out["drPCA"].append(k["kRecoDR"][valid].astype(float))
        out["drST2"].append(dr2)
        out["recophi"].append(k["kRecoMuMuPhi"][valid].astype(float))
        out["key"].append(np.array([RANK_P1[int(x)] * 3 + RANK_P1[int(y)] for x, y in
                                    zip(k["kTypeA"][valid], k["kTypeB"][valid])]))
        base += n
    c = {key: np.concatenate(v) for key, v in out.items()}
    e = {key: np.concatenate(v) for key, v in ev_cols.items()}
    return names, base, c, e


def analyze(args):
    import ROOT
    if Path(args.output).exists():
        raise RuntimeError("Refusing to overwrite ROOT output")
    names, total, c, e = load(args)
    if args.expected_events and total != args.expected_events:
        raise RuntimeError("Event total mismatch")
    ev = c["ev"]
    c["dphiPF"] = np.abs(wrap(e["pfmetphi"][ev] - c["recophi"]))
    sdphiPF = wrap(e["pfmetphi"][ev] - c["recophi"])
    e["dphiGen"] = np.abs(wrap(e["genmetphi"] - e["genphi"]))
    sdphiGen = wrap(e["genmetphi"] - e["genphi"])
    c["drGenVtx"] = e["drGenVtx"][ev]; c["drGenST2"] = e["drGenST2"][ev]; c["dphiGen"] = e["dphiGen"][ev]
    nan_ok = lambda x, cut: np.isfinite(x) & (x < cut)   # NaN (missing quantity) fails
    chi_ok = c["chi2"] < CHI2_CUT
    dr_ok = {v: nan_ok(c[v], DR_CUT) for v in DRV}
    dp_ok = {v: nan_ok(c[v], DPHI_CUT) for v in DPV}
    ev_any = lambda m: np.bincount(ev[m], minlength=total) > 0
    base = ev_any(np.ones(len(ev), bool))
    nb = int(base.sum())
    # accepted / initial context
    n_init = total
    # individual costs
    rows = [("baseline: >= 1 valid combo", nb)]
    indiv = {"chi2 < %g" % CHI2_CUT: int(ev_any(chi_ok).sum())}
    for v in DRV:
        indiv["dR < %g, %s" % (DR_CUT, DRV_LABEL[v])] = int(ev_any(dr_ok[v]).sum())
    for v in DPV:
        indiv["|dphi| < %g, %s" % (DPHI_CUT, DPV_LABEL[v])] = int(ev_any(dp_ok[v]).sum())
    # chains (sequential, same combo)
    chains = {}
    for name, (dv, pv) in CHAINS.items():
        m1 = chi_ok; m2 = m1 & dr_ok[dv]; m3 = m2 & dp_ok[pv]
        chains[name] = [nb, int(ev_any(m1).sum()), int(ev_any(m2).sum()), int(ev_any(m3).sum())]
    # chain-independent all-three-order check: requirements are commutative for a single combo
    # shape distributions: P1 combo (first valid under Global>Inner>DSA), one per event
    order = np.lexsort((c["key"], ev))
    first = order[np.r_[True, np.diff(ev[order]) > 0]]
    sel_ev = ev[first]
    shapes = {"PF": (sdphiPF[first], np.ones(len(first), bool)),
              "Gen": (sdphiGen[sel_ev], np.ones(len(first), bool))}
    root = ROOT.TFile.Open(str(args.output), "CREATE")
    ROOT.TNamed("definitions", __doc__).Write()
    ROOT.TNamed("analysisRevision", args.revision).Write()
    ROOT.TNamed("samples", "\n".join(names)).Write()
    def th(name, edges, counts):
        h = ROOT.TH1D(name, name, len(edges) - 1, array("d", edges)); h.Sumw2()
        for b, v in enumerate(counts):
            h.SetBinContent(b, float(v)); h.SetBinError(b, float(np.sqrt(v)))
        h.Write()
    th("baseline_individual", np.arange(len(indiv) + 2, dtype=float), np.r_[0, nb, list(indiv.values()), 0])
    for name, vals in chains.items():
        th("chain_" + name, np.arange(5, dtype=float), np.r_[0, vals, 0])
    for key, (vals, ok) in shapes.items():
        for tag, edges, x in (("dphi", DPHI_EDGES["dphi"], vals), ("absdphi", DPHI_EDGES["absdphi"], np.abs(vals))):
            x = x[np.isfinite(x)]
            th("%s__%s" % (key, tag), edges, np.bincount(np.searchsorted(edges, x, side="right"), minlength=len(edges) + 1))
    ROOT.TNamed("indiv_labels", "\n".join(indiv)).Write()
    ROOT.TNamed("counts", "initial=%d baseline=%d" % (n_init, nb)).Write()
    root.Close()
    # report
    print("VALIDATED: events=%d, valid combos=%d" % (total, len(ev)))
    print("baseline (>=1 valid combo): %d  = %.2f%% of initial" % (nb, 100 * nb / n_init))
    print("individual requirement (event passes if any valid combo passes it):")
    for lab, v in indiv.items():
        print("  %-62s %7d  kept %.2f%%  lost %.2f%%" % (lab, v, 100 * v / nb, 100 * (nb - v) / nb))
    for name, vals in chains.items():
        print("chain %s (same combo): baseline %d -> chi2 %d (%.2f%% kept) -> dR %d (%.2f%% of prev) -> dphi %d (%.2f%% of prev); "
              "total kept %.2f%% of baseline, %.2f%% of initial" % (
                  CHAIN_LABEL[name], vals[0], vals[1], 100 * vals[1] / vals[0], vals[2], 100 * vals[2] / vals[1],
                  vals[3], 100 * vals[3] / vals[2], 100 * vals[3] / vals[0], 100 * vals[3] / n_init))
    nanc = {v: int((~np.isfinite(c[v])).sum()) for v in DRV + DPV}
    print("valid combos with unavailable quantity (fail the requirement):", nanc)
    print("shape entries (one per event):", len(first))


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import uproot
    out = Path(args.plot_dir)
    out.mkdir(parents=True, exist_ok=True)
    f = uproot.open(args.output)
    foot = "Signal MC (2022EE), pre-selected displacedMuons; baseline = events with two accepted gen and >= 1 valid typed vertex; raw unweighted"
    initial = int(str(f["counts"].member("fTitle")).split()[0].split("=")[1])

    def finish(fig, path, bottom=0.02):
        fig.text(0.01, 0.005, foot, fontsize=7, color="0.35")
        fig.tight_layout(rect=(0, bottom, 1, 0.96))
        fig.savefig(path, dpi=120)
        if args.log and path.name.startswith("sel_dphi"):
            for a_ in fig.axes:
                a_.set_yscale("log")
            fig.savefig(path.with_name(path.stem + "_log.png"), dpi=120)
        plt.close(fig)

    # sequential cutflow per chain
    fig, axs = plt.subplots(2, 2, figsize=(15, 8))
    stage_lab = ["baseline\n(>=1 valid combo)", "+ chi2/ndof < %g" % CHI2_CUT, "+ dR < %g" % DR_CUT, "+ |dphi| < %g" % DPHI_CUT]
    for ax, name in zip(axs.ravel(), CHAINS):
        v = f["chain_" + name].values()
        y = np.arange(4)[::-1]
        ax.barh(y, v, color=["0.6", "#0072B2", "#E69F00", "#009E73"])
        for yi, i in zip(y, range(4)):
            cost = "" if i == 0 else "  lost %.1f%% of previous step" % (100 * (1 - v[i] / v[i - 1]))
            ax.text(v[i], yi, " %d (%.1f%% of baseline)%s" % (v[i], 100 * v[i] / v[0], cost), va="center", fontsize=8)
        ax.set_yticks(y); ax.set_yticklabels(stage_lab, fontsize=9); ax.set_xlim(0, v[0] * 1.9)
        ax.set_xlabel("Events (raw)"); ax.set_title(CHAIN_LABEL[name], fontsize=10)
    fig.suptitle("Cost of the requirements, sequential (an event passes if one valid combo satisfies all requirements so far); "
                 "baseline = %d events (%.1f%% of the %d initial)" % (v[0], 100 * v[0] / initial, initial), fontsize=11)
    finish(fig, out / "sel_cutflow_chains.png")

    # individual cost of each requirement
    labs = str(f["indiv_labels"].member("fTitle")).split("\n")
    vals = f["baseline_individual"].values()
    nb = vals[0]; vals = vals[1:]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    y = np.arange(len(labs))[::-1]
    cols = ["#0072B2"] + ["#E69F00"] * 4 + ["#009E73"] * 2
    ax.barh(y, vals, color=cols)
    for yi, v in zip(y, vals):
        ax.text(v, yi, " %d: kept %.1f%%, lost %.1f%%" % (v, 100 * v / nb, 100 * (1 - v / nb)), va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=8); ax.set_xlim(0, nb * 1.5)
    ax.axvline(nb, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Events (raw) of the %d baseline events (dashed line)" % nb)
    ax.set_title("Each requirement alone: events with >= 1 valid combo passing it", fontsize=10)
    finish(fig, out / "sel_cost_individual.png")

    # dphi shapes, same bins and axes, unit area
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, tag, xl in ((axs[0], "dphi", r"$\Delta\phi$ = $\phi$(MET) − $\phi$($\mu\mu$ $p_T$) [rad]"),
                        (axs[1], "absdphi", r"$|\Delta\phi|$ [rad]")):
        for key, col, lab, ls in (("PF", "#0072B2", "PF MET vs reco dimuon pT (P1 vertex)", "-"),
                                  ("Gen", "#D55E00", "gen MET vs gen dimuon pT (same events)", "--")):
            h_ = f["%s__%s" % (key, tag)]
            v = h_.values(flow=True); e_ = h_.axis().edges(); n = v.sum()
            ax.stairs(v[1:-1] / n, e_, color=col, lw=2, ls=ls, label="%s (N=%d)" % (lab, n))
        if tag == "absdphi":
            ax.axvline(DPHI_CUT, color="k", lw=1, ls=":", label="requirement |dphi| < %g" % DPHI_CUT)
        ax.set_xlabel(xl); ax.set_ylabel("Fraction of events per bin (unit area)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.suptitle("Shape of the MET / dimuon-pT azimuthal difference, reco vs gen (vertexed events, one vertex per event: first valid combo, "
                 "priority Global>Inner>DSA); same bins and axes", fontsize=10)
    finish(fig, out / "sel_dphi_shapes.png")
    print("Plots done in", out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("analyze", "plot"))
    ap.add_argument("--input-dir")
    ap.add_argument("--revision", default="unspecified")
    ap.add_argument("--output", required=True)
    ap.add_argument("--plot-dir")
    ap.add_argument("--expected-samples", type=int, default=72)
    ap.add_argument("--expected-events", type=int, default=720000)
    ap.add_argument("--log", action="store_true", help="add log-y copies of the dphi shape plot")
    args = ap.parse_args()
    analyze(args) if args.mode == "analyze" else plot(args)


if __name__ == "__main__":
    main()
