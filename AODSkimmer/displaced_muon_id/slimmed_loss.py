#!/usr/bin/env python3
"""Signal-matched muon loss from slimmedDisplacedMuons (study note: kb/studies/slimmed-displaced-muon-loss.md).

python3 slimmed_loss.py analyze --input-dir RUN/inputs --log-dir RUN/logs --reference-dir <G>I>DSA PreselectedVertexing inputs> --output RUN/slimmed_loss.root
python3 slimmed_loss.py plot --output RUN/slimmed_loss.root --plot-dir RUN/plots [--log]
Inputs: slimmed_loss_cfg.py outputs (preselectedVertexing = case A, preselectedVertexingB = case B,
filteredDisplacedMuons = kept-in-slimmed bookkeeping). Population: the two accepted signal muons of each twoAcc event
(exactly two signal muons, opposite charge, both accepted), object level, raw unweighted, propagation failures included.
ROOT is the numerical source (raw counts, Sumw2, flow bins kept); `plot` reads only that file. --log adds log-x copies.
"""
import argparse
import re
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preselected_vertexing_analysis import cp, flowcounts, write_hist  # noqa: E402

EDGES = {"pt": np.arange(0, 31, 1.0), "d0": np.arange(0, 31, 1.0), "lxy": np.arange(0, 61, 2.0)}
LABELS = {"pt": r"Gen signal-muon $p_T$ [GeV]", "d0": r"Gen signal-muon $|d_0|$ to origin [cm]",
          "lxy": r"Gen signal-muon $L_{xy}$ [cm]"}
NUMS = ("den", "matchA", "matchB", "lost", "gained")
TYPES = {0: "Inner", 1: "DSA", 2: "Global"}
# Fate of A's matched raw object for lost gen muons. Filter codes as in SlimmedDisplacedFilterCheck.
FATES = ("filter: STA+inner both pT<3.5", "filter: STA-only pT<3.5", "filter: STA-only <2 DT+CSC hits (hitPattern)",
         "filter: no-STA inner pT<3.5", "filter: no-STA not tracker muon", "filter: STA-only, passes replicated cuts (segment count)",
         "filter: not STA-only, passes replicated cuts (unexplained)", "kept, removed by B cleaner", "kept and cleaned, match lost")
COUNTS = (("den", "accepted gen (twoAcc events)"), ("matchA", "matched A"), ("matchB", "matched B"), ("lost", "lost"),
          ("gained", "gained")) + tuple(("lostType%d" % t, "lost, A type %s" % n) for t, n in TYPES.items()) + tuple(
          ("lostFate%d" % i, "lost: " + f) for i, f in enumerate(FATES)) + (
          ("rawMuons", "raw displacedMuons"), ("keptMuons", "kept in slimmed"), ("refMismatch", "A vs reference mismatches"))


def per_gen(arr, two):
    """Concatenate the two per-gen entries of each twoAcc event (sig arrays: sig order = matching gen order)."""
    return np.concatenate([np.asarray(x[:2]) for x, t in zip(arr, two) if t]) if two.any() else np.zeros(0)


def analyze_sample(path, ref_dir, log_dir, flist_dir):
    import uproot
    f = uproot.open(path)
    sample = str(f["preselectedVertexing/sample"].member("fTitle"))
    if sample != path.stem:
        raise RuntimeError("%s: file holds sample %s" % (path, sample))
    rev = str(f["preselectedVertexing/revision"].member("fTitle"))
    br = ["run", "lumi", "event", "twoAcc", "sigPt", "sigD0", "sigLxy", "sigQ", "matchObj", "matchType", "cObj", "cType"]
    a = f["preselectedVertexing/events"].arrays(br, library="np")
    b = f["preselectedVertexingB/events"].arrays(br, library="np")
    c = f["filteredDisplacedMuons/events"].arrays(library="np")
    n = len(a["run"])
    if n != 10000 or len(b["run"]) != n or len(c["run"]) != n:
        raise RuntimeError("%s: event count A %d B %d check %d" % (sample, n, len(b["run"]), len(c["run"])))
    for k in ("run", "lumi", "event", "twoAcc"):
        if not (np.array_equal(a[k], b[k]) and (k == "twoAcc" or np.array_equal(a[k], c[k]))):
            raise RuntimeError("%s: A/B/check event alignment differs in %s" % (sample, k))
    for k in ("sigPt", "sigD0", "sigLxy", "sigQ"):
        if any(not np.array_equal(x, y) for x, y in zip(a[k], b[k])):
            raise RuntimeError("%s: A/B gen arrays differ in %s" % (sample, k))
    bad = int(c["badNoPartner"].sum() + c["badAmbig"].sum() + c["badOrder"].sum())
    kept_n = np.array([(x >= 0).sum() for x in c["rawToFiltered"]])
    if bad or not np.array_equal(kept_n, c["nSlimmed"]):
        raise RuntimeError("%s: slimmed partner validation failed (%d bad, %d count mismatches)"
                           % (sample, bad, int((kept_n != c["nSlimmed"]).sum())))
    # Coverage: every listed input file opened and closed once (cmsRun log).
    files = [l.strip() for l in (flist_dir / (sample + "_flist.txt")).read_text().splitlines()
             if l.strip() and not l.lstrip().startswith("#")]
    # Complete paths only: interleaved log streams can cut a path and glue the next timestamp onto it.
    closed = re.findall(r"Closed file (\S+\.root)(?!\S)", (log_dir / (sample + ".log")).read_text())
    canon = lambda s: s[s.find("/store/"):]
    if sorted(map(canon, closed)) != sorted(map(canon, files)):
        raise RuntimeError("%s: input coverage mismatch (%d listed, %d closed)" % (sample, len(files), len(closed)))

    two = a["twoAcc"].astype(bool)
    ev = np.repeat(np.flatnonzero(two), 2)
    g = {k: per_gen(a[k], two) for k in ("sigPt", "sigD0", "sigLxy")}
    objA, typA = per_gen(a["matchObj"], two).astype(int), per_gen(a["matchType"], two).astype(int)
    objB = per_gen(b["matchObj"], two).astype(int)
    mA, mB = objA >= 0, objB >= 0
    lost, gained = mA & ~mB, ~mA & mB

    # Reference: A must reproduce the existing G > I > DSA run on the same events.
    r = uproot.open(ref_dir / (sample + ".root"))["preselectedVertexing/events"].arrays(
        ["run", "lumi", "event", "twoAcc", "matchObj", "matchType"], library="np")
    if any(not np.array_equal(r[k], a[k]) for k in ("run", "lumi", "event", "twoAcc")):
        raise RuntimeError("%s: reference events differ" % sample)
    ref_mis = int(((per_gen(r["matchObj"], two) != objA) | (per_gen(r["matchType"], two) != typA)).sum())
    if ref_mis:
        raise RuntimeError("%s: case A differs from the reference matching for %d gen" % (sample, ref_mis))

    fate = np.full(len(objA), -1)
    for i in np.flatnonzero(lost):
        e, r = ev[i], objA[i]
        code = int(c["filterCode"][e][r])
        if c["rawToFiltered"][e][r] < 0:
            types = set(a["cType"][e][a["cObj"][e] == r].tolist())
            fate[i] = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}.get(code, 5 if 0 not in types else 6)
        elif c["rawToFiltered"][e][r] not in set(b["cObj"][e].tolist()):
            fate[i] = 7
        else:
            fate[i] = 8
    counts = dict(den=len(objA), matchA=int(mA.sum()), matchB=int(mB.sum()), lost=int(lost.sum()),
                  gained=int(gained.sum()), rawMuons=int(c["nRaw"].sum()), keptMuons=int(kept_n.sum()),
                  refMismatch=ref_mis)
    for t in TYPES:
        counts["lostType%d" % t] = int((lost & (typA == t)).sum())
    for i in range(len(FATES)):
        counts["lostFate%d" % i] = int((fate == i).sum())
    if counts["matchA"] - counts["matchB"] != counts["lost"] - counts["gained"] or (fate[lost] < 0).any():
        raise RuntimeError("%s: bookkeeping identity failed" % sample)
    masks = dict(den=np.ones(len(objA), bool), matchA=mA, matchB=mB, lost=lost, gained=gained)
    vals = dict(pt=g["sigPt"], d0=np.abs(g["sigD0"]), lxy=g["sigLxy"])
    hists = {(m, v): flowcounts(vals[v][masks[m]], EDGES[v]) for m in NUMS for v in EDGES}
    return sample, rev, counts, hists


def analyze(args):
    import ROOT
    paths = sorted(Path(args.input_dir).glob("*.root"))
    if len(paths) != args.expected_samples:
        raise RuntimeError("Expected %d sample files, found %d" % (args.expected_samples, len(paths)))
    flist_dir = Path(__file__).resolve().parents[1] / "fileLists/signal/2022EE/iDMmu"
    expected = sorted(f.name[:-len("_flist.txt")] for f in flist_dir.glob("*_flist.txt"))
    if [p.stem for p in paths] != expected:
        raise RuntimeError("Input samples differ from the signal file lists: missing %s, extra %s" % (
            sorted(set(expected) - {p.stem for p in paths}), sorted({p.stem for p in paths} - set(expected))))
    if not args.reference_dir:
        raise RuntimeError("analyze needs --reference-dir (case A must reproduce the reference matching)")
    ref = Path(args.reference_dir)
    results, revs = [], set()
    for p in paths:
        s, rev, counts, hists = analyze_sample(p, ref, Path(args.log_dir), flist_dir)
        results.append((s, counts, hists))
        revs.add(rev)
        print("OK %s den %d A %d B %d lost %d gained %d refMismatch %d" % (
            s, counts["den"], counts["matchA"], counts["matchB"], counts["lost"], counts["gained"], counts["refMismatch"]),
            flush=True)
    if len(revs) != 1:
        raise RuntimeError("Mixed processing revisions: %s" % revs)
    groups = {"overall": [r for r in results]}
    for r in results:
        mass, ctau = re.fullmatch(r"(Mchi-[\dp]+_dMchi-[\dp]+)_ctau-([\dp]+)", r[0]).groups()
        groups.setdefault(mass, []).append(r)
        groups.setdefault("ctau_" + ctau, []).append(r)
        groups.setdefault("sample_" + r[0], []).append(r)
    out = ROOT.TFile(args.output, "RECREATE")
    ROOT.TNamed("revision", revs.pop()).Write()
    ROOT.TNamed("definitions", __doc__.split("\n\n", 1)[1]).Write()
    for gname, rs in groups.items():
        out.mkdir("groups/" + gname)  # returns the top directory ("groups"), so cd by path
        out.cd("groups/" + gname)
        hc = ROOT.TH1D("counts", "raw unweighted counts", len(COUNTS), 0, len(COUNTS))
        for i, (k, label) in enumerate(COUNTS):
            hc.GetXaxis().SetBinLabel(i + 1, label)
            hc.SetBinContent(i + 1, sum(r[1][k] for r in rs))
        hc.Write()
        for m in NUMS:
            for v in EDGES:
                cnt = sum(r[2][(m, v)] for r in rs)
                write_hist("%s__%s" % (m, v), LABELS[v], EDGES[v], cnt)
        if gname == "overall":
            for i, (k, label) in enumerate(COUNTS):
                print("%-70s %d" % (label, hc.GetBinContent(i + 1)))
    out.Close()


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import uproot
    out = Path(args.plot_dir)
    out.mkdir(parents=True, exist_ok=True)
    f = uproot.open(args.output)
    groups = sorted({k.split("/")[1] for k in f.keys() if k.startswith("groups/") and k.count("/") == 2})
    foot = ("Accepted signal muons of events with exactly 2, opposite-sign, both accepted; hierarchy G>I>DSA, station 2, "
            "same charge; A = displacedMuons, B = those kept in slimmedDisplacedMuons; raw unweighted, unweighted pooling; "
            "68% Clopper-Pearson; bins with < 20 gen omitted")

    def h(group, name):
        o = f["groups/%s/%s" % (group, name)]
        return o.values(flow=True)[1:-1], o.axis().edges()

    def frac(group, num, var):
        n, edges = h(group, "%s__%s" % (num, var))
        d, _ = h(group, "den__%s" % var)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(d > 0, n / d, np.nan)
        lo, hi = cp(n, d)
        ok = d >= 20
        x = 0.5 * (edges[1:] + edges[:-1])
        return x[ok], r[ok], (r - lo)[ok], (hi - r)[ok]

    def save(fig, path):
        fig.text(0.01, 0.005, foot, fontsize=6, color="0.35")
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(path, dpi=110)
        if args.log:
            for ax in fig.axes:
                ax.set_xscale("symlog", linthresh=1)
            fig.savefig(path.with_name(path.stem + "_log.png"), dpi=110)
        plt.close(fig)

    def numkey(g):
        return [float(x.replace("p", ".")) for x in re.findall(r"[\d]+(?:p\d+)?", g)]

    def glabel(g):
        if g.startswith("ctau_"):
            return "cτ = %s cm" % g.split("_")[1].replace("p", ".")
        a, b = (float(x.replace("p", ".")) for x in re.findall(r"Mchi-([\dp]+)", g))
        return "Mχ=%g, ΔM=%g GeV" % (a, b)

    for var in EDGES:
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(6.4, 6), sharex=True, gridspec_kw=dict(height_ratios=(2, 1)))
        for num, lab, col in (("matchA", "A: displacedMuons", "k"), ("matchB", "B: kept in slimmedDisplacedMuons", "tab:red")):
            x, r, lo, hi = frac("overall", num, var)
            ax0.errorbar(x, r, yerr=[lo, hi], fmt="o", ms=3, lw=1, color=col, label=lab)
        x, r, lo, hi = frac("overall", "lost", var)
        ax1.errorbar(x, r, yerr=[lo, hi], fmt="o", ms=3, lw=1, color="tab:blue")
        ax0.set_ylabel("Matched / accepted"); ax0.set_ylim(0, 1.05); ax0.grid(alpha=.3); ax0.legend(fontsize=8)
        ax1.set_ylabel("Lost / accepted"); ax1.set_ylim(bottom=0); ax1.grid(alpha=.3); ax1.set_xlabel(LABELS[var])
        ax0.set_title("All 72 signal points, pooled", fontsize=9)
        save(fig, out / ("eff_overall__%s.png" % var))
        for tag, sel in (("mass", lambda g: g.startswith("Mchi")), ("lifetime", lambda g: g.startswith("ctau_"))):
            gs = sorted([g for g in groups if sel(g)], key=numkey)
            fig, ax = plt.subplots(figsize=(7.5, 4.6))
            for i, g in enumerate(gs):
                x, r, lo, hi = frac(g, "lost", var)
                col = plt.get_cmap("turbo")(0.05 + 0.9 * i / max(len(gs) - 1, 1)) if len(gs) > 10 else plt.get_cmap("tab10")(i)
                ax.plot(x, r, "-o", ms=2.5, lw=1, color=col, label=glabel(g))
            ax.set_xlabel(LABELS[var]); ax.set_ylabel("Lost / accepted"); ax.set_ylim(bottom=0); ax.grid(alpha=.3)
            ax.set_title("Lost fraction per %s (%s)" % (tag, "pooled over lifetimes" if tag == "mass" else "pooled over masses"),
                         fontsize=9)
            ax.legend(fontsize=6, ncol=2 if len(gs) > 6 else 1)
            save(fig, out / ("overlay_%s_lost__%s.png" % (tag, var)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("analyze", "plot"))
    ap.add_argument("--input-dir")
    ap.add_argument("--log-dir")
    ap.add_argument("--reference-dir")
    ap.add_argument("--output", required=True)
    ap.add_argument("--plot-dir")
    ap.add_argument("--expected-samples", type=int, default=72)
    ap.add_argument("--log", action="store_true")
    args = ap.parse_args()
    analyze(args) if args.mode == "analyze" else plot(args)


if __name__ == "__main__":
    main()
