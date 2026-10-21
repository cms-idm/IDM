#!/usr/bin/env python3
"""Cutflow, efficiencies and vertex-category distributions from PreselectedVertexing trees.

python3 preselected_vertexing_analysis.py analyze --input-dir RUN/inputs --report-dir RUN/logs --revision <sha> --output RUN/vertexing.root
python3 preselected_vertexing_analysis.py plot --output RUN/vertexing.root --plot-dir RUN/plots [--log]
Study note: kb/studies/preselected-vertexing-efficiency.md. ROOT is the numerical source (raw unweighted counts,
Sumw2, flow bins kept); `plot` reads only that ROOT file. --log adds log copies, never replaces linear PNGs.
Efficiency denominators: events with two accepted gen ('den') unless stated 'matched'. Sets: bb = hierarchy
best-best (one set), any = all pairs of eligible objects, six typed sets (independent per-type matching, ordered
combos). Distribution entries: bb one per event; any one per valid pair; typed one per valid ORDERED fit.
Initial events = all processed events (10000 per sample).
"""
import argparse
from array import array
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import numpy as np

CATS = ("GG", "II", "OO", "GO", "GI", "OI")
CAT_NAMES = {"GG": "Global–Global", "II": "Inner–Inner", "OO": "DSA–DSA", "GO": "Global–DSA",
             "GI": "Global–Inner", "OI": "DSA–Inner"}
# Types: 0 inner, 1 outer (DSA), 2 global. Ordered combos (gen0 type, gen1 type) per unordered category.
ORDERINGS = {"GG": ((2, 2),), "II": ((0, 0),), "OO": ((1, 1),), "GO": ((2, 1), (1, 2)),
             "GI": ((2, 0), (0, 2)), "OI": ((1, 0), (0, 1))}
SETS = ("bb", "any") + CATS
SET_NAMES = {"bb": "Best–best (hierarchy)", "any": "Any two pre-selected objects",
             **{c: CAT_NAMES[c] + " (typed fit)" for c in CATS}}
EFF_EDGES = {
    "d0": np.arange(0, 16, 1.0), "lxy": np.arange(0, 31, 1.0), "pt": np.arange(0, 31, 1.0),
    "eta": np.linspace(0, 2.4, 13),
}
EFF_LABELS = {"pt": r"Leading gen signal-muon $p_T$ [GeV]", "eta": r"Max gen signal-muon $|\eta|$",
              "lxy": r"Gen production $L_{xy}$ [cm]", "d0": r"Max gen $|d_0|$ to origin [cm]"}
DIST_EDGES = {
    "chi2": np.r_[np.arange(0, 10.01, .25), 12, 15, 20, 30, 50, 100, 1000],
    "drGen": np.arange(0, 6.01, .1), "drReco": np.arange(0, 6.01, .1),
    "dphiGen": np.linspace(-np.pi, np.pi, 33), "dphiPF": np.linspace(-np.pi, np.pi, 33),
    "absdphiGen": np.linspace(0, np.pi, 33), "absdphiPF": np.linspace(0, np.pi, 33),
}
PLOT_DISTS = ("chi2", "drGen", "drReco", "absdphiGen", "absdphiPF")
DIST_LABELS = {"chi2": r"Vertex $\chi^2$/ndof", "drGen": r"$\Delta R$(gen signal pair)",
               "drReco": r"$\Delta R$(reco pair, track ref. point)",
               "dphiGen": r"$\Delta\phi$(gen MET, gen $\mu\mu$ $p_T$) [rad]",
               "dphiPF": r"$\Delta\phi$(PF MET, reco $\mu\mu$ $p_T$) [rad]",
               "absdphiGen": r"$|\Delta\phi|$(gen MET, gen $\mu\mu$ $p_T$) [rad]",
               "absdphiPF": r"$|\Delta\phi|$(PF MET, reco $\mu\mu$ $p_T$) [rad]"}
CUTFLOW = ("initial events", "two signal muons (mu-,mu+)", "both accepted", "both matched (hierarchy)",
           "best-best vertex valid") + tuple("typed matched " + CAT_NAMES[c] for c in CATS) + tuple("typed valid " + CAT_NAMES[c] for c in CATS) + (
          "accepted & >=2 pre-selected objects", "accepted & >=1 valid any-pair", "matched & >=1 valid any-pair")


def wrap(x):
    """Wrapped difference; NaN when an input is the -999 sentinel (missing MET or leg)."""
    x = np.where(np.abs(x) > 500, np.nan, x)
    return (x + np.pi) % (2 * np.pi) - np.pi


def load_samples(args):
    import awkward as ak
    import uproot
    paths = sorted(Path(args.input_dir).glob("*.root"))
    if len(paths) != args.expected_samples:
        raise RuntimeError("Unexpected number of sample ROOT inputs: %d" % len(paths))
    names, revisions, cols, pcols, kcols, counts = [], set(), [], [], [], []
    base = 0
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
            a = t.arrays(library="ak")
            sq = a["sigQ"]
            sig2 = (ak.num(sq) == 2) & (ak.fill_none(ak.prod(sq, axis=1, mask_identity=False), 0) < 0)
            two = np.asarray(a["twoAcc"])
            def per2(x, fn):
                out = np.full(n, np.nan)
                idx = np.flatnonzero(two)
                v = ak.to_numpy(ak.pad_none(x[idx], 2, clip=True))
                out[idx] = fn(v)
                return out
            cols.append(dict(
                sample=np.full(n, si), sig2=np.asarray(sig2), two=two,
                matched=np.asarray(a["bbStatus"]) > 0, status=np.asarray(a["bbStatus"]),
                nElig=np.asarray(a["nElig"]), anyN=np.asarray(a["anyNPairs"]), anyV=np.asarray(a["anyNValid"]),
                pt=per2(a["sigPt"], lambda v: v.max(axis=1)), eta=per2(a["sigEta"], lambda v: np.abs(v).max(axis=1)),
                d0=per2(a["sigD0"], lambda v: v.max(axis=1)), lxy=per2(a["sigLxy"], lambda v: v[:, 0]),
                genDR=np.asarray(a["genDR"]), genPhi=np.asarray(a["genMuMuPhi"]), genMetPhi=np.asarray(a["genMetPhi"]),
                pfMetPhi=np.asarray(a["pfMetPhi"]), chi2=np.asarray(a["bbChi2"]), drReco=np.asarray(a["bbRecoDR"]),
                recoPhi=np.asarray(a["bbRecoMuMuPhi"])))
            p = f["preselectedVertexing/pairs"].arrays(library="np")
            p["pEvent"] = p["pEvent"] + base
            pcols.append(p)
            k = f["preselectedVertexing/combos"].arrays(library="np")
            k["kEvent"] = k["kEvent"] + base
            kcols.append(k)
            counts.append(n)
            base += n
    cat_cols = {k: np.concatenate([c[k] for c in cols]) for k in cols[0]}
    pair_cols = {k: np.concatenate([p[k] for p in pcols]) for k in pcols[0]}
    combo_cols = {k: np.concatenate([c[k] for c in kcols]) for k in kcols[0]}
    if len(revisions) != 1 or "unspecified" in revisions:
        raise RuntimeError("Mixed/unspecified processing revision")
    return names, counts, cat_cols, pair_cols, combo_cols, next(iter(revisions))


def check_report(report_dir, sample, n_events):
    report = ET.parse(report_dir / (sample + ".xml")).getroot()
    if [e for e in report.findall(".//FrameworkError") if e.get("ExitStatus") != "0"]:
        raise RuntimeError("FrameworkJobReport reports a processing error: " + sample)
    flist = Path(__file__).resolve().parents[2] / "AODSkimmer/fileLists/signal/2022EE/iDMmu" / (sample + "_flist.txt")
    originals = [l.strip() for l in flist.read_text().splitlines() if l.strip() and not l.lstrip().startswith("#")]
    canon = lambda s: s[s.find("/store/"):]
    observed = [(canon(f.findtext("PFN", "")), int(f.findtext("EventsRead", "-1"))) for f in report.findall(".//InputFile")]
    if (len(observed) != len(originals) or set(p for p, n in observed) != set(map(canon, originals))
            or any(n <= 0 for p, n in observed) or sum(n for p, n in observed) != n_events):
        raise RuntimeError("Per-file coverage/events mismatch: " + sample)
    print("COVERAGE %s: %d original files, %d events verified" % (sample, len(originals), n_events), flush=True)


def flowcounts(values, edges):
    return np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1)


def write_hist(name, title, edges, counts):
    import ROOT
    h = ROOT.TH1D(name, title, len(edges) - 1, array("d", edges))
    h.Sumw2()
    for b, c in enumerate(counts):
        h.SetBinContent(b, float(c))
        h.SetBinError(b, float(np.sqrt(c)))
    h.SetEntries(float(np.sum(counts)))
    h.Write()


def analyze(args):
    import ROOT
    if Path(args.output).exists():
        raise RuntimeError("Refusing to overwrite numerical ROOT source")
    names, counts, e, p, k, revision = load_samples(args)
    total = sum(counts)
    if args.expected_events and total != args.expected_events:
        raise RuntimeError("Event total %d != expected %d" % (total, args.expected_events))
    st = e["status"]
    if np.any(e["matched"] & ~e["two"]) or np.any(e["two"] & ~e["sig2"]):
        raise RuntimeError("Matched/acceptance inconsistency")
    if int(p["pStatus"].size) != int(e["anyN"].sum()) or int((p["pStatus"] == 3).sum()) != int(e["anyV"].sum()):
        raise RuntimeError("Pair rows differ from per-event counts")
    if not np.all(e["two"][k["kEvent"]]):
        raise RuntimeError("Typed combo in a non-two-accepted event")
    anyvalid = e["two"] & (e["anyV"] > 0)
    masks = {"den": e["two"], "matched": e["matched"], "valid": st == 3, "anyValid": anyvalid,
             "anyMatchedValid": e["matched"] & anyvalid}
    combo_sel = {}
    for c in CATS:
        sel = np.zeros(len(k["kEvent"]), bool)
        for a_, b_ in ORDERINGS[c]:
            sel |= (k["kTypeA"] == a_) & (k["kTypeB"] == b_)
        combo_sel[c] = sel
        m_ev = np.zeros(total, bool); m_ev[k["kEvent"][sel]] = True
        v_ev = np.zeros(total, bool); v_ev[k["kEvent"][sel & (k["kStatus"] == 3)]] = True
        masks["tmatched_" + c] = m_ev
        masks["tvalid_" + c] = v_ev
    if np.any(masks["den"] < (masks["matched"])) or any(np.any(masks["tvalid_" + c] & ~masks["tmatched_" + c]) for c in CATS):
        raise RuntimeError("Numerator outside denominator")
    cutflow = [np.ones(total, bool), e["sig2"], e["two"], e["matched"], st == 3] + \
        [masks["tmatched_" + c] for c in CATS] + [masks["tvalid_" + c] for c in CATS] + \
        [e["two"] & (e["nElig"] >= 2), anyvalid, e["matched"] & anyvalid]
    for k_ in ("genMetPhi", "pfMetPhi", "genPhi", "recoPhi"):
        e[k_] = np.where(e[k_] < -900, np.nan, e[k_])
    for k_ in ("pRecoMuMuPhi",):
        p[k_] = np.where(p[k_] < -900, np.nan, p[k_])
    k["kRecoMuMuPhi"] = np.where(k["kRecoMuMuPhi"] < -900, np.nan, k["kRecoMuMuPhi"])
    def vals(chi2, drgen, drreco, genmetphi, genphi, pfmetphi, recophi):
        d = {"chi2": chi2, "drGen": drgen, "drReco": drreco, "dphiGen": wrap(genmetphi - genphi),
             "dphiPF": wrap(pfmetphi - recophi)}
        d["absdphiGen"] = np.abs(d["dphiGen"]); d["absdphiPF"] = np.abs(d["dphiPF"])
        return d
    dist = {"bb": (st == 3, vals(e["chi2"], e["genDR"], e["drReco"], e["genMetPhi"], e["genPhi"], e["pfMetPhi"], e["recoPhi"]),
                   e["sample"])}
    pe = p["pEvent"]
    dist["any"] = (p["pStatus"] == 3, vals(p["pChi2"], e["genDR"][pe], p["pRecoDR"], e["genMetPhi"][pe], e["genPhi"][pe],
                                          e["pfMetPhi"][pe], p["pRecoMuMuPhi"]), e["sample"][pe])
    ke = k["kEvent"]
    kv = vals(k["kChi2"], e["genDR"][ke], k["kRecoDR"], e["genMetPhi"][ke], e["genPhi"][ke], e["pfMetPhi"][ke],
              k["kRecoMuMuPhi"])
    for c in CATS:
        dist[c] = (combo_sel[c] & (k["kStatus"] == 3), kv, e["sample"][ke])
    groups = {"overall": None}
    for si, sample in enumerate(names):
        mass, tau = sample.rsplit("_ctau-", 1)
        groups["sample_" + sample] = {si}
        groups.setdefault("mass_" + mass, set()).add(si)
        groups.setdefault("ctau_" + tau, set()).add(si)
    root = ROOT.TFile.Open(str(args.output), "CREATE")
    if not root or root.IsZombie():
        raise RuntimeError("Cannot create ROOT output")
    ROOT.TNamed("definitions", "Events with two accepted gen = exactly two opposite-charge signal muons both accepted "
        "(|eta|<2.4,0<=Lxy<740,|vz|<960,no pT). Eligible component = own ID and own |eta|<2.4. bb = hierarchy "
        "global>outer>inner (station-2, same charge, dR<0.06 I,G <0.10 O, whole-object claim) best-best fit, one set. "
        "Typed sets GG,II,OO,GO,GI,OI = independent per-type matching and ordered combo fits; event matched/valid = "
        "any ordering; distributions one entry per valid ordered fit. any = all pairs of distinct eligible objects at "
        "best component (G>O>I), one entry per valid pair. Raw unweighted; flow kept; initial events = all processed.").Write()
    ROOT.TNamed("analysisRevision", args.revision).Write()
    ROOT.TNamed("inputRevision", revision).Write()
    ROOT.TNamed("samples", "\n".join(names)).Write()
    ROOT.TNamed("coverage", "\n".join("%s: %d events" % x for x in zip(names, counts))).Write()
    top = root.mkdir("groups")
    for gname, ids in groups.items():
        d = top.mkdir(gname)
        d.cd()
        ev_sel = np.ones(total, bool) if ids is None else np.isin(e["sample"], list(ids))
        write_hist("cutflow", "cutflow", np.arange(len(CUTFLOW) + 1, dtype=float),
                   np.r_[0, [int((m & ev_sel).sum()) for m in cutflow], 0])
        for var, edges in EFF_EDGES.items():
            for mname, m in masks.items():
                write_hist("%s__%s" % (mname, var), EFF_LABELS[var], edges, flowcounts(e[var][m & ev_sel], edges))
        for s_, (sel, v, sid) in dist.items():
            gsel = sel & (np.ones(len(sel), bool) if ids is None else np.isin(sid, list(ids)))
            for var, edges in DIST_EDGES.items():
                write_hist("%s__%s" % (s_, var), DIST_LABELS[var], edges, flowcounts(v[var][gsel & np.isfinite(v[var])], edges))
    root.Close()
    chk = ROOT.TFile.Open(str(args.output))  # histogram totals incl. flow == selected entries (overall)
    for s_, (sel, v, sid) in dist.items():
        hh = chk.Get("groups/overall/%s__chi2" % s_)
        if int(hh.Integral(0, hh.GetNbinsX() + 1)) != int((sel & np.isfinite(v["chi2"])).sum()):
            raise RuntimeError("Histogram total differs from valid entries: " + s_)
    for m_ in ("den", "matched", "valid", "anyValid", "anyMatchedValid") + tuple(
            p_ + c for p_ in ("tmatched_", "tvalid_") for c in CATS):
        hh = chk.Get("groups/overall/%s__d0" % m_)
        if int(hh.Integral(0, hh.GetNbinsX() + 1)) != int((masks[m_] & np.isfinite(e["d0"])).sum()):
            raise RuntimeError("Efficiency histogram total differs: " + m_)
    chk.Close()
    for s_, (sel, v, sid) in dist.items():
        for var in ("dphiGen", "dphiPF"):
            bad = int((sel & ~np.isfinite(v[var])).sum())
            if bad:
                print("EXCLUDED %d %s entries with missing MET/leg phi in %s" % (bad, var, s_), flush=True)
    print("VALIDATED: events=%d groups=%d" % (total, len(groups)), flush=True)
    print("CUTFLOW (overall; initial events = %d):" % total)
    allc = [int(m.sum()) for m in cutflow]
    for label, c in zip(CUTFLOW, allc):
        print("  %-38s %9d  %.4f of initial  %.4f of accepted" % (label, c, c / allc[0], c / max(allc[2], 1)), flush=True)
    print("PER-SAMPLE CUTFLOW: sample, initial, accepted, matched, vertexed, matched/initial, vertexed/initial")
    for si, name in enumerate(names):
        sel = e["sample"] == si
        n0, n2, n3, n4 = (int((m & sel).sum()) for m in (cutflow[0], cutflow[2], cutflow[3], cutflow[4]))
        print("  %-36s %6d %6d %6d %6d  %.4f %.4f" % (name, n0, n2, n3, n4, n3 / n0, n4 / n0), flush=True)


# ---------------------------------------------------------------------------------------------- plotting
def cp(k, n, level=0.6827):
    from scipy.stats import beta
    a = (1 - level) / 2
    lo = np.where(k > 0, beta.ppf(a, k, np.maximum(n - k + 1, 1e-9)), 0.0)
    hi = np.where(k < n, beta.ppf(1 - a, k + 1, np.maximum(n - k, 1e-9)), 1.0)
    return lo, hi


def plot(args):
    """Deliberately small PNG set from the ROOT file only: cutflow (overall/mass/lifetime), efficiencies vs
    d0/Lxy/pT/eta (overall + mass and lifetime overlays), distribution panels per set (overall + overlays)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import uproot
    out = Path(args.plot_dir)
    out.mkdir(parents=True, exist_ok=True)
    f = uproot.open(args.output)
    groups = sorted({k.split("/")[1] for k in f.keys() if k.startswith("groups/") and k.count("/") == 2})
    foot_eff = ("Pre-selected displacedMuons (ID, |eta|<2.4); accepted = exactly 2 signal muons, both accepted; "
                "raw unweighted; error bars: 68% Clopper-Pearson; bins with < 20 denominator events omitted")
    foot = "Pre-selected displacedMuons (ID, |eta|<2.4); events with two accepted signal muons; raw unweighted"

    def cmap(i, n):  # >10 groups: sequential colours in the (sorted) legend order; else tab10
        return plt.get_cmap("turbo")(0.05 + 0.9 * i / max(n - 1, 1)) if n > 10 else plt.get_cmap("tab10")(i)

    def numkey(g):  # sort overlay groups numerically (mass: Mchi then dMchi; lifetime: ctau)
        nums = [float(x.replace("p", ".")) for x in re.findall(r"(?:Mchi|ctau)-([\dp]+)", g)]
        return nums if nums else [float(g.split("_")[-1])]

    def glabel(g):
        if g.startswith("ctau_"):
            return "c\u03c4 = %s cm" % g.split("_")[1]
        a, b = (float(x.replace("p", ".")) for x in re.findall(r"Mchi-([\dp]+)", g))
        return "M\u03c7=%g, \u0394M=%g GeV" % (a, b)

    def h(group, name):
        o = f["groups/%s/%s" % (group, name)]
        return o.values(flow=True), o.axis().edges()

    def finish(fig, path, note=None, bottom=0.02):
        fig.text(0.01, 0.005, note or foot, fontsize=6.5, color="0.35")
        fig.tight_layout(rect=(0, bottom, 1, 1))
        fig.savefig(path, dpi=110)
        if args.log:  # extra log-y copy for distributions only; the linear PNG is never replaced
            if path.name.startswith(("dist_", "overlay_dist")):
                for a in fig.axes:
                    a.set_yscale("log")
                fig.savefig(path.with_name(path.stem + "_log.png"), dpi=110)
        plt.close(fig)

    def ratio(group, num, den, var):
        n, edges = h(group, "%s__%s" % (num, var))
        d, _ = h(group, "%s__%s" % (den, var))
        n, d = n[1:-1], d[1:-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(d > 0, n / d, np.nan)
        lo, hi = cp(n, d)
        return edges, r, lo, hi, d

    def curve(ax, edges, r, lo, hi, d, label, color=None):
        x = 0.5 * (edges[1:] + edges[:-1])
        ok = d >= 20
        ax.errorbar(x[ok], r[ok], yerr=[(r - lo)[ok], (hi - r)[ok]], fmt="o", ms=3, lw=1, label=label, color=color)

    METRICS = (("matched", "den", "tmatched_", "Both matched / accepted"),
               ("valid", "matched", "tvalid_", "Vertexed / both matched"),
               ("valid", "den", "tvalid_", "Vertexed / accepted"))

    def eff_overall(group):
        for var in EFF_EDGES:
            fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
            for i, (num, den, typed, title) in enumerate(METRICS):
                curve(ax[i], *ratio(group, num, den, var), "best–best (hierarchy)", "k")
                for j, c in enumerate(CATS):
                    n_, d_ = typed + c, ("tmatched_" + c if den == "matched" else "den")
                    curve(ax[i], *ratio(group, n_, d_, var), CAT_NAMES[c], plt.get_cmap("tab10")(j))
                if den == "matched":
                    curve(ax[i], *ratio(group, "anyMatchedValid", "matched", var), "any pair (≥1 valid, matched events)", "0.5")
                if den == "den" and num == "valid":
                    curve(ax[i], *ratio(group, "anyValid", "den", var), "any pair (≥1 valid)", "0.5")
                ax[i].set_xlabel(EFF_LABELS[var]); ax[i].set_ylabel("Efficiency"); ax[i].set_title(title, fontsize=9)
                ax[i].set_ylim(0, 1.05); ax[i].grid(alpha=.3)
            ax[2].legend(fontsize=6.5)
            fig.suptitle("overall (typed sets: own matched count in the vertexing-given-matched panel)", fontsize=8)
            finish(fig, out / ("eff_overall__%s.png" % var), note=foot_eff)

    def eff_overlay(prefix, tag):
        gs = sorted([g for g in groups if g.startswith(prefix)], key=numkey)
        for var in EFF_EDGES:
            fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
            for i, (num, den, typed, title) in enumerate(METRICS):
                for j, g in enumerate(gs):
                    if len(gs) > 10:  # many groups: lines without error bars, otherwise the points are unreadable
                        edges, r_, l_, u_, b_ = ratio(g, num, den, var)
                        xc = 0.5 * (edges[1:] + edges[:-1]); ok_ = b_ >= 20
                        ax[i].plot(xc[ok_], r_[ok_], "-o", ms=2.5, lw=1, color=cmap(j, len(gs)), label=glabel(g))
                    else:
                        curve(ax[i], *ratio(g, num, den, var), glabel(g), cmap(j, len(gs)))
                ax[i].set_xlabel(EFF_LABELS[var]); ax[i].set_ylabel("Efficiency"); ax[i].set_ylim(0, 1.05)
                ax[i].set_title("Best–best: " + title, fontsize=9); ax[i].grid(alpha=.3)
            hd, lb = ax[2].get_legend_handles_labels()
            fig.legend(hd, lb, loc="lower center", ncol=6, fontsize=7, bbox_to_anchor=(0.5, 0.02))
            finish(fig, out / ("overlay_%s_eff__%s.png" % (tag, var)), note=foot_eff if len(gs) <= 10 else foot + "; lines connect bins (no error bars)",
                   bottom=0.12)

    def cutflow_fig(group):
        v, _ = h(group, "cutflow")
        v = v[1:-1]
        fig, ax = plt.subplots(figsize=(9, 6.5))
        y = np.arange(len(CUTFLOW))[::-1]
        fam = ["0.6", "0.6", "C0", "C1", "C1"] + ["C2"] * 6 + ["C3"] * 6 + ["C4"] * 3
        ax.barh(y, v, color=fam)
        for yi, c in zip(y, v):
            ax.text(c, yi, " %d  (%.2f%% of initial, %.2f%% of accepted)" % (c, 100 * c / max(v[0], 1), 100 * c / max(v[2], 1)),
                    va="center", fontsize=7)
        ax.set_yticks(y); ax.set_yticklabels(CUTFLOW, fontsize=8); ax.set_xlabel("Events (raw)")
        ax.set_xlim(0, v.max() * 1.9 if v.max() else 1); ax.set_title("Cutflow: " + group, fontsize=9)
        finish(fig, out / ("cutflow_%s.png" % group))

    def panels(groupnames, labels, s, norm, fname, title):
        fig, ax = plt.subplots(2, 3, figsize=(14, 7.5))
        ax = ax.ravel()
        for pi, var in enumerate(PLOT_DISTS):
            edges = DIST_EDGES[var]
            if var == "chi2":
                edges = edges[edges <= 10]
            for gi, (g, lab) in enumerate(zip(groupnames, labels)):
                v, e_full = h(g, "%s__%s" % (s, var))
                tot = v.sum()
                nb = len(edges) - 1
                vals = v[1:nb + 1]
                if tot == 0:
                    continue
                ax[pi].stairs(vals / (tot if norm else 1), edges, label=lab, color=cmap(gi, len(groupnames)))
            ax[pi].set_xlabel(DIST_LABELS[var]); ax[pi].set_ylabel("Unit area (incl. overflow)" if norm else "Entries (raw)")
            if var == "chi2":
                ax[pi].set_title("range 0–10 (uniform bins); larger values in ROOT overflow/tail bins", fontsize=7)
        if len(groupnames) > 1:
            ax[5].legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=2, fontsize=7)
        ax[5].axis("off")
        v, _ = h(groupnames[0], "%s__chi2" % s)
        if len(groupnames) == 1:
            ax[5].text(0, 0.5, "%s\n%s\nvalid fits: %d" % (SET_NAMES[s], title, v.sum()), fontsize=10)
        else:
            ax[5].text(0, 0.0, "%s, %s" % (SET_NAMES[s], title), fontsize=9)
        finish(fig, out / fname)

    cutflow_fig("overall")
    for g in groups:
        if g.startswith(("mass_", "ctau_")):
            cutflow_fig(g)
    eff_overall("overall")
    eff_overlay("mass_", "mass")
    eff_overlay("ctau_", "lifetime")
    mg = sorted([g for g in groups if g.startswith("mass_")], key=numkey)
    lg = sorted([g for g in groups if g.startswith("ctau_")], key=numkey)
    for s in SETS:
        panels(["overall"], ["overall"], s, False, "dist_overall__%s.png" % s, "overall")
        for tag, gs, pre in (("mass", mg, "mass_"), ("lifetime", lg, "ctau_")):
            for norm in (False, True):
                panels(gs, [glabel(g) for g in gs], s, norm,
                       "overlay_dist_%s__%s%s.png" % (tag, s, "_norm" if norm else ""), tag + " overlay")
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
    if args.mode == "analyze":
        analyze(args)
    else:
        plot(args)


if __name__ == "__main__":
    main()
