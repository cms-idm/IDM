#!/usr/bin/env python3
"""Cutflow, efficiencies and vertex-category distributions from PreselectedVertexing trees.

python3 preselected_vertexing_analysis.py analyze --input-dir RUN/inputs --report-dir RUN/logs --revision <sha> --output RUN/vertexing.root
python3 preselected_vertexing_analysis.py plot --output RUN/vertexing.root --plot-dir RUN/plots [--log]
Study note: kb/studies/preselected-vertexing-efficiency.md. ROOT is the numerical source (raw unweighted counts,
Sumw2, flow bins kept); `plot` reads only that ROOT file. --log adds log copies, never replaces linear PNGs.
Efficiency denominators: events with two accepted gen ('den') unless stated 'matched' (both signal muons matched
by the hierarchy). Any-pair distributions have one entry per valid pair; best-best one entry per event.
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
SETS = ("bb_all",) + tuple("bb_" + c for c in CATS) + ("any_all",) + tuple("any_" + c for c in CATS)
SET_NAMES = {"bb": "Best–best (hierarchy)", "any": "Any two pre-selected objects"}
EFF_EDGES = {
    "pt": np.r_[np.arange(0, 31, 1), np.arange(35, 105, 5), np.arange(125, 525, 25), np.arange(600, 2100, 100)],
    "eta": np.linspace(0, 2.4, 13),
    "lxy": np.r_[np.arange(0, 21, 1), np.arange(30, 751, 10)],
    "d0": np.r_[np.arange(0, 10.5, .5), np.arange(15, 105, 5), np.arange(120, 741, 20)],
    "lxy30": np.arange(31, dtype=float),
    "d030": np.arange(31, dtype=float),
}
EFF_LABELS = {"pt": r"Min gen signal-muon $p_T$ [GeV]", "eta": r"Max gen signal-muon $|\eta|$",
              "lxy": r"Gen production $L_{xy}$ [cm]", "d0": r"Max gen $|d_0|$ to origin [cm]",
              "lxy30": r"Gen production $L_{xy}$ [cm] (0–30)", "d030": r"Max gen $|d_0|$ to origin [cm] (0–30)"}
DIST_EDGES = {
    "chi2": np.r_[np.arange(0, 10.01, .25), 12, 15, 20, 30, 50, 100, 1000],
    "drGen": np.arange(0, 5.01, .1), "drReco": np.arange(0, 5.01, .1),
    "dphiGen": np.linspace(-np.pi, np.pi, 33), "dphiPF": np.linspace(-np.pi, np.pi, 33),
    "absdphiGen": np.linspace(0, np.pi, 33), "absdphiPF": np.linspace(0, np.pi, 33),
}
DIST_LABELS = {"chi2": r"Vertex $\chi^2$/ndof", "drGen": r"$\Delta R$(gen signal pair)",
               "drReco": r"$\Delta R$(reco pair, track reference point)",
               "dphiGen": r"$\Delta\phi$(gen MET, gen $\mu\mu$ $p_T$) [rad]",
               "dphiPF": r"$\Delta\phi$(PF MET, reco $\mu\mu$ $p_T$) [rad]",
               "absdphiGen": r"$|\Delta\phi|$(gen MET, gen $\mu\mu$ $p_T$) [rad]",
               "absdphiPF": r"$|\Delta\phi|$(PF MET, reco $\mu\mu$ $p_T$) [rad]"}
CUTFLOW = ("all events", "two signal muons (mu-,mu+)", "both accepted", "both matched (hierarchy)",
           "best-best vertex valid") + tuple("matched " + c for c in CATS) + tuple("valid " + c for c in CATS) + (
          "accepted & >=2 pre-selected objects", "accepted & >=1 valid any-pair", "matched & >=1 valid any-pair")


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def load_samples(args):
    import awkward as ak
    import uproot
    paths = sorted(Path(args.input_dir).glob("*.root"))
    if len(paths) != args.expected_samples:
        raise RuntimeError("Unexpected number of sample ROOT inputs: %d" % len(paths))
    names, revisions, cols, pcols, counts = [], set(), [], [], []
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
                matched=np.asarray(a["bbStatus"]) > 0, status=np.asarray(a["bbStatus"]), cat=np.asarray(a["bbCat"]),
                nElig=np.asarray(a["nElig"]), anyN=np.asarray(a["anyNPairs"]), anyV=np.asarray(a["anyNValid"]),
                pt=per2(a["sigPt"], lambda v: v.min(axis=1)), eta=per2(a["sigEta"], lambda v: np.abs(v).max(axis=1)),
                d0=per2(a["sigD0"], lambda v: v.max(axis=1)), lxy=per2(a["sigLxy"], lambda v: v[:, 0]),
                genDR=np.asarray(a["genDR"]), genPhi=np.asarray(a["genMuMuPhi"]), genMetPhi=np.asarray(a["genMetPhi"]),
                pfMetPhi=np.asarray(a["pfMetPhi"]), chi2=np.asarray(a["bbChi2"]), drReco=np.asarray(a["bbRecoDR"]),
                recoPhi=np.asarray(a["bbRecoMuMuPhi"])))
            p = f["preselectedVertexing/pairs"].arrays(library="np")
            p["pEvent"] = p["pEvent"] + base
            pcols.append(p)
            counts.append(n)
            base += n
    cat_cols = {k: np.concatenate([c[k] for c in cols]) for k in cols[0]}
    pair_cols = {k: np.concatenate([p[k] for p in pcols]) for k in pcols[0]}
    if len(revisions) != 1 or "unspecified" in revisions:
        raise RuntimeError("Mixed/unspecified processing revision")
    return names, counts, cat_cols, pair_cols, next(iter(revisions))


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
    names, counts, e, p, revision = load_samples(args)
    total = sum(counts)
    if args.expected_events and total != args.expected_events:
        raise RuntimeError("Event total %d != expected %d" % (total, args.expected_events))
    # Consistency checks on the arrays
    st = e["status"]
    if np.any(e["matched"] & ~e["two"]) or np.any((e["cat"] >= 0) != e["matched"]):
        raise RuntimeError("Matched/category/acceptance inconsistency")
    if np.any(e["two"] & ~e["sig2"]):
        raise RuntimeError("twoAcc without two opposite-charge signal muons")
    if int(p["pStatus"].size) != int(e["anyN"].sum()):
        raise RuntimeError("Pair rows != sum anyNPairs")
    if int((p["pStatus"] == 3).sum()) != int(e["anyV"].sum()):
        raise RuntimeError("Valid pair rows != sum anyNValid")
    anyvalid = e["two"] & (e["anyV"] > 0)
    # Event-level masks
    masks = {"den": e["two"], "matched": e["matched"], "valid": st == 3,
             "anyValid": anyvalid, "anyMatchedValid": e["matched"] & anyvalid}
    for i, c in enumerate(CATS):
        masks["matched_" + c] = e["matched"] & (e["cat"] == i)
        masks["valid_" + c] = (st == 3) & (e["cat"] == i)
    if sum(int(masks["matched_" + c].sum()) for c in CATS) != int(masks["matched"].sum()) or \
       sum(int(masks["valid_" + c].sum()) for c in CATS) != int(masks["valid"].sum()):
        raise RuntimeError("Categories do not partition best-best")
    cutflow = [np.ones(total, bool), e["sig2"], e["two"], e["matched"], st == 3] + \
        [masks["matched_" + c] for c in CATS] + [masks["valid_" + c] for c in CATS] + \
        [e["two"] & (e["nElig"] >= 2), anyvalid, e["matched"] & anyvalid]
    # Distribution entries
    evsel = st == 3
    ev_vals = {"chi2": e["chi2"], "drGen": e["genDR"], "drReco": e["drReco"],
               "dphiGen": wrap(e["genMetPhi"] - e["genPhi"]), "dphiPF": wrap(e["pfMetPhi"] - e["recoPhi"])}
    pv = p["pStatus"] == 3
    pe = p["pEvent"]
    pair_vals = {"chi2": p["pChi2"], "drGen": e["genDR"][pe], "drReco": p["pRecoDR"],
                 "dphiGen": wrap(e["genMetPhi"][pe] - e["genPhi"][pe]),
                 "dphiPF": wrap(e["pfMetPhi"][pe] - p["pRecoMuMuPhi"])}
    for d in (ev_vals, pair_vals):
        d["absdphiGen"] = np.abs(d["dphiGen"])
        d["absdphiPF"] = np.abs(d["dphiPF"])
    dist = {}
    for s in SETS:
        kind, c = s.split("_")
        if kind == "bb":
            sel = evsel & (True if c == "all" else e["cat"] == CATS.index(c))
            dist[s] = (sel, ev_vals, e["sample"])
        else:
            sel = pv & (True if c == "all" else p["pCat"] == CATS.index(c))
            dist[s] = (sel, pair_vals, e["sample"][pe])
    # Groups
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
        "(|eta|<2.4,0<=Lxy<740,|vz|<960,no pT). Hierarchy global>outer>inner, station-2, same charge, dR<0.06 (I,G) <0.10 (O), "
        "whole-object claim. Eligible component = own ID and own |eta|<2.4. Best-best = fit of matched components; "
        "categories partition best-best. Any = all pairs of distinct eligible objects at best component (G>O>I). "
        "Raw unweighted. Any distributions one entry per valid pair; best-best one per event. Flow kept.").Write()
    ROOT.TNamed("analysisRevision", args.revision).Write()
    ROOT.TNamed("inputRevision", revision).Write()
    ROOT.TNamed("samples", "\n".join(names)).Write()
    ROOT.TNamed("coverage", "\n".join("%s: %d events" % x for x in zip(names, counts))).Write()
    top = root.mkdir("groups")
    for gname, ids in groups.items():
        d = top.mkdir(gname)
        d.cd()
        ev_sel = np.ones(total, bool) if ids is None else np.isin(e["sample"], list(ids))
        # cutflow
        write_hist("cutflow", "cutflow", np.arange(len(CUTFLOW) + 1, dtype=float),
                   np.r_[0, [int((m & ev_sel).sum()) for m in cutflow], 0])
        for var, edges in EFF_EDGES.items():
            key = {"lxy30": "lxy", "d030": "d0"}.get(var, var)
            for mname, m in masks.items():
                write_hist("%s__%s" % (mname, var), EFF_LABELS[var], edges,
                           flowcounts(e[key][m & ev_sel], edges))
        for s, (sel, vals, sid) in dist.items():
            gsel = sel & (np.ones(len(sel), bool) if ids is None else np.isin(sid, list(ids)))
            for var, edges in DIST_EDGES.items():
                write_hist("%s__%s" % (s, var), DIST_LABELS[var], edges, flowcounts(vals[var][gsel], edges))
    root.Close()
    print("VALIDATED: events=%d groups=%d" % (total, len(groups)), flush=True)
    print("CUTFLOW (overall):")
    allc = [int(m.sum()) for m in cutflow]
    for label, c in zip(CUTFLOW, allc):
        print("  %-40s %9d  %.4f of all  %.4f of accepted" % (label, c, c / allc[0], c / max(allc[2], 1)), flush=True)


# ---------------------------------------------------------------------------------------------- plotting
def cp(k, n, level=0.6827):
    from scipy.stats import beta
    a = (1 - level) / 2
    lo = np.where(k > 0, beta.ppf(a, k, np.maximum(n - k + 1, 1e-9)), 0.0)
    hi = np.where(k < n, beta.ppf(1 - a, k + 1, np.maximum(n - k, 1e-9)), 1.0)
    return lo, hi


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import uproot
    out = Path(args.plot_dir)
    out.mkdir(parents=True, exist_ok=True)
    f = uproot.open(args.output)
    groups = sorted({k.split("/")[1] for k in f.keys() if k.startswith("groups/") and k.count("/") == 2})
    definitions = str(f["definitions"].member("fTitle"))
    foot = "Pre-selected displacedMuons (ID+|eta|<2.4), hierarchy G>O>I; accepted gen = exactly 2 signal, both accepted; raw unweighted"

    def h(group, name):
        o = f["groups/%s/%s" % (group, name)]
        return o.values(flow=True), o.axis().edges()

    def finish(fig, path, logx=False, logy=False):
        fig.text(0.01, 0.005, foot, fontsize=6.5, color="0.35")
        fig.tight_layout(rect=(0, 0.02, 1, 1))
        fig.savefig(path, dpi=110)
        if args.log and (logx or logy):  # extra copy only; the linear PNG above is never replaced
            for a in fig.axes:
                if logx:
                    a.set_xscale("symlog", linthresh=1)
                if logy:
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
        ok = d > 0
        ax.errorbar(x[ok], r[ok], yerr=[(r - lo)[ok], (hi - r)[ok]], fmt="o", ms=3, lw=1, label=label, color=color,
                    xerr=0.5 * np.diff(edges)[ok] if len(x) < 40 else None)

    def eff_figs(group):
        for var in EFF_EDGES:
            fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
            for num, den, lab in (("matched", "den", "both matched / accepted"),
                                  ("valid", "den", "best–best vertexed / accepted"),
                                  ("anyValid", "den", "≥1 valid any-pair / accepted")):
                curve(ax[0], *ratio(group, num, den, var), lab)
            for num, den, lab in (("valid", "matched", "best–best vertexed / matched"),
                                  ("anyMatchedValid", "matched", "matched & ≥1 valid any-pair / matched")):
                curve(ax[1], *ratio(group, num, den, var), lab)
            for a, t in zip(ax, ("Relative to events with two accepted gen", "Relative to events with both matched")):
                a.set_xlabel(EFF_LABELS[var]); a.set_ylabel("Efficiency"); a.set_title(t, fontsize=9)
                a.set_ylim(0, 1.05); a.legend(fontsize=7); a.grid(alpha=.3)
            fig.suptitle(group, fontsize=9)
            finish(fig, out / ("eff_%s__%s.png" % (group, var)), logx=var in ("pt", "lxy", "d0"))
            fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
            for c in CATS:
                curve(ax[0], *ratio(group, "valid_" + c, "matched_" + c, var), CAT_NAMES[c])
                curve(ax[1], *ratio(group, "valid_" + c, "den", var), CAT_NAMES[c])
            for a, t in zip(ax, ("Vertexed / matched in that category", "Vertexed in category / events with two accepted gen")):
                a.set_xlabel(EFF_LABELS[var]); a.set_ylabel("Efficiency"); a.set_title(t, fontsize=9)
                a.set_ylim(0, 1.05); a.legend(fontsize=7); a.grid(alpha=.3)
            fig.suptitle(group, fontsize=9)
            finish(fig, out / ("effcat_%s__%s.png" % (group, var)), logx=var in ("pt", "lxy", "d0"))

    def cutflow_fig(group):
        v, _ = h(group, "cutflow")
        v = v[1:-1]
        fig, ax = plt.subplots(figsize=(8, 6))
        y = np.arange(len(CUTFLOW))[::-1]
        ax.barh(y, v)
        for yi, c in zip(y, v):
            ax.text(c, yi, " %d (%.2f%% of all, %.2f%% of accepted)" % (c, 100 * c / max(v[0], 1), 100 * c / max(v[2], 1)),
                    va="center", fontsize=7)
        ax.set_yticks(y); ax.set_yticklabels(CUTFLOW, fontsize=8); ax.set_xlabel("Events (raw)")
        ax.set_xlim(0, v.max() * 1.6 if v.max() else 1); ax.set_title("Cutflow: " + group, fontsize=9)
        finish(fig, out / ("cutflow_%s.png" % group))

    def dist_figs(group, with_sample):
        for s in SETS:
            for var, edges in DIST_EDGES.items():
                v, _ = h(group, "%s__%s" % (s, var))
                if v[1:-1].sum() == 0:
                    continue
                fig, ax = plt.subplots(figsize=(5.6, 4.2))
                ax.stairs(v[1:-1], edges)
                ax.set_xlabel(DIST_LABELS[var]); ax.set_ylabel("Entries (raw)")
                kind, c = s.split("_")
                ax.set_title("%s %s: %s\nunder/overflow: %d/%d" % (SET_NAMES[kind], "" if c == "all" else CAT_NAMES[c],
                             group, v[0], v[-1]), fontsize=8)
                finish(fig, out / ("dist_%s__%s__%s.png" % (group, s, var)), logy=True)

    def overlay_figs(prefix, tag):
        gs = [g for g in groups if g.startswith(prefix)]
        for var in ("d0", "d030"):
            for num, den, lab in (("matched", "den", "both matched / accepted"), ("valid", "den", "vertexed / accepted"),
                                  ("valid", "matched", "vertexed / matched")):
                fig, ax = plt.subplots(figsize=(7, 4.5))
                for g in gs:
                    curve(ax, *ratio(g, num, den, var), g[len(prefix):])
                ax.set_xlabel(EFF_LABELS[var]); ax.set_ylabel(lab); ax.set_ylim(0, 1.05); ax.grid(alpha=.3)
                ax.legend(fontsize=6, ncol=2); ax.set_title("%s overlay: %s" % (tag, lab), fontsize=9)
                finish(fig, out / ("overlay_%s_%s_%s_%s.png" % (tag, num, den, var)))
        for s in SETS:
            for var, edges in DIST_EDGES.items():
                for norm in (False, True):
                    fig, ax = plt.subplots(figsize=(6.5, 4.5))
                    drawn = False
                    for g in gs:
                        v, _ = h(g, "%s__%s" % (s, var))
                        tot = v.sum()
                        if v[1:-1].sum() == 0:
                            continue
                        ax.stairs(v[1:-1] / (tot if norm and tot else 1), edges, label=g[len(prefix):])
                        drawn = True
                    if drawn:
                        kind, c = s.split("_")
                        ax.set_xlabel(DIST_LABELS[var]); ax.set_ylabel("Unit area (incl. flow)" if norm else "Entries (raw)")
                        ax.legend(fontsize=6, ncol=2)
                        ax.set_title("%s %s, %s overlay%s" % (SET_NAMES[kind], "" if c == "all" else CAT_NAMES[c], tag,
                                     ", unit area" if norm else ""), fontsize=8)
                        finish(fig, out / ("overlay_%s_dist_%s__%s%s.png" % (tag, s, var, "_norm" if norm else "")), logy=True)
                    else:
                        plt.close(fig)

    for g in groups:
        eff_figs(g)
        cutflow_fig(g)
        if not g.startswith("sample_"):
            dist_figs(g, False)
    overlay_figs("mass_", "mass")
    overlay_figs("ctau_", "lifetime")
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
