#!/usr/bin/env python3
"""Match ID-passing cleaned components, retain ROOT counts/rows, plot or replot without MINIAOD scans.

python3 component_matching_efficiency.py --input-dir RUN/inputs --report-dir RUN/logs --revision <sha> --output RUN/efficiency.root --plot-dir RUN/plots
python3 component_matching_efficiency.py --mode replot --output RUN/efficiency.root --plot-dir RUN/new_plots
--no-plots supports the one-file runtime check; --log adds log-x copies, never replaces linear PNGs.
Every plotted fraction uses ALL accepted signal gen. Comparable-match counts are stored separately.
"""
import argparse
from array import array
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import numpy as np

TYPES = ("inner", "outer", "global")
EFF = (*TYPES, "hierarchy")
PAIRS = ((2, 1), (2, 0), (1, 0))
DISC = ("global_outer", "global_inner", "outer_inner", "independent_any",
        "hier_global", "hier_outer", "hier_inner", "hier_any")
COMPARE = ("both_global_outer", "both_global_inner", "both_outer_inner", "multiple_independent",
           "both_hier_global", "both_hier_outer", "both_hier_inner", "any_hier_comparison")
METRICS = EFF + DISC + COMPARE
EDGES = {
    "pt": np.r_[np.arange(0, 31, 1), np.arange(35, 105, 5), np.arange(125, 525, 25), np.arange(600, 2100, 100)],
    "eta": np.linspace(-2.4, 2.4, 25),
    "lxy": np.r_[np.arange(0, 21, 1), np.arange(30, 751, 10)],
    "d0": np.r_[np.arange(0, 10.5, .5), np.arange(15, 105, 5), np.arange(120, 741, 20)],
}
LABELS = {"pt": r"Gen signal-muon $p_T$ [GeV]", "eta": r"Gen signal-muon $\eta$",
          "lxy": r"Gen production $L_{xy}$ [cm]", "d0": r"Gen $|d_0|$ to origin [cm]"}
NICE = dict(zip(EFF + DISC, ("Inner (tracker ID)", "Outer (DSA ID)", "Global (global ID)",
    "Hierarchy: global → outer → inner", "Global vs outer", "Global vs inner", "Outer vs inner",
    "Any independent disagreement", "Hierarchy vs global", "Hierarchy vs outer", "Hierarchy vs inner",
    "Any hierarchy disagreement")))


def matching(gq, gp, ge, gf, ct, co, cq, cp, ce, cf):
    """Sorted compatible pairs; independent component reservations, staged whole-object hierarchy."""
    n = len(gq)
    independent = np.full((n, 3), -1, dtype=np.int32)
    hierarchy = np.full(n, -1, dtype=np.int32)
    tier = np.full(n, -1, dtype=np.int32)
    distances = np.full((n, 4), np.nan, dtype=np.float64)
    pairs = []
    for c in range(3):
        selected = np.flatnonzero((ct == c) & (cp == 1) & np.isfinite(ce) & np.isfinite(cf))
        if not n or not len(selected):
            pairs.append([])
            continue
        dphi = (gf[:, None] - cf[selected] + np.pi) % (2 * np.pi) - np.pi
        dr = np.hypot(ge[:, None] - ce[selected], dphi)
        compatible = ((gp[:, None] == 1) & np.isfinite(ge[:, None]) & np.isfinite(gf[:, None]) &
                      (gq[:, None] == cq[selected]) & (dr < (.10 if c == 1 else .06)))
        gi, ci = np.nonzero(compatible)
        ranked = sorted((float(dr[g, k]), int(g), int(co[selected[k]]), int(selected[k]))
                        for g, k in zip(gi, ci))
        pairs.append(ranked)
        used = set()
        for d, g, obj, k in ranked:
            if independent[g, c] < 0 and obj not in used:
                independent[g, c] = obj
                distances[g, c] = d
                used.add(obj)
    used = set()
    for c in (2, 1, 0):
        for d, g, obj, k in pairs[c]:
            if hierarchy[g] < 0 and obj not in used:
                hierarchy[g], tier[g], distances[g, 3] = obj, c, d
                used.add(obj)
    return independent, hierarchy, tier, distances


def oracle(gq, gp, ge, gf, ct, co, cq, cp, ce, cf):
    """Bounded real-event check: enumerate pairs with scalar periodic distance, repeatedly take minimum."""
    import math
    pairs = [[] for _ in range(3)]
    for g in range(len(gq)):
        if gp[g] != 1 or not math.isfinite(ge[g]) or not math.isfinite(gf[g]):
            continue
        for k in range(len(ct)):
            if cp[k] != 1 or gq[g] != cq[k] or not math.isfinite(ce[k]) or not math.isfinite(cf[k]):
                continue
            d = math.hypot(ge[g] - ce[k], math.remainder(gf[g] - cf[k], 2 * math.pi))
            if d < (.10 if ct[k] == 1 else .06):
                pairs[int(ct[k])].append((d, g, int(co[k]), k))
    def assign(candidates, assignments, used, category=None):
        work = list(candidates)
        while work:
            best = min(work)
            work.remove(best)
            d, g, obj, _ = best
            if assignments[g] == -1 and obj not in used:
                assignments[g] = obj
                used.add(obj)
                if category is not None:
                    tiers[g] = category
    indep = np.full((len(gq), 3), -1, dtype=np.int32)
    for c in range(3):
        assign(pairs[c], indep[:, c], set())
    hier, tiers = np.full(len(gq), -1, dtype=np.int32), np.full(len(gq), -1, dtype=np.int32)
    used = set()
    for c in (2, 1, 0):
        assign(pairs[c], hier, used, c)
    return indep, hier, tiers


def analyze(args):
    import awkward as ak
    import uproot
    import ROOT
    paths = sorted(Path(args.input_dir).glob("*.root"))
    if len(paths) != args.expected_samples:
        raise RuntimeError("Unexpected number of sample ROOT inputs")
    if args.output.exists():
        raise RuntimeError("Refusing to overwrite numerical ROOT source")
    records, event_counts, samples, revisions = [], [], [], set()
    names = ("gQ", "gProp", "gEta2", "gPhi2", "cType", "cObject", "cQ", "cProp", "cEta2", "cPhi2",
             "gPt", "gEta", "gLxy", "gD0", "gIndex")
    events_total = checks = 0
    for sample_id, path in enumerate(paths):
        with uproot.open(path) as source:
            sample = str(source["componentInputs/sample"].member("fTitle"))
            if sample in samples or not re.fullmatch(r"Mchi-[\dp]+_dMchi-[\dp]+_ctau-[\dp]+", sample):
                raise RuntimeError("Unexpected/duplicate sample name: " + sample)
            samples.append(sample)
            revisions.add(str(source["componentInputs/revision"].member("fTitle")))
            tree = source["componentInputs/events"]
            event_counts.append(tree.num_entries)
            events_total += tree.num_entries
            if args.report_dir is not None:
                report = ET.parse(args.report_dir / (sample + ".xml")).getroot()
                failures = [e for e in report.findall(".//FrameworkError") if e.get("ExitStatus") != "0"]
                if failures:
                    raise RuntimeError("FrameworkJobReport reports a processing error: " + sample)
                flist = Path(__file__).resolve().parents[2] / "AODSkimmer/fileLists/signal/2022EE/iDMmu" / (sample + "_flist.txt")
                originals = [line.strip() for line in flist.read_text().splitlines()
                             if line.strip() and not line.lstrip().startswith("#")]
                def canonical(path):
                    index = path.find("/store/")
                    if index < 0:
                        raise RuntimeError("Unexpected input PFN: " + path)
                    return path[index:]
                observed = [(canonical(f.findtext("PFN", "")), int(f.findtext("EventsRead", "-1")))
                            for f in report.findall(".//InputFile")]
                if (len(originals) != 4 or len(observed) != 4 or len(set(p for p, n in observed)) != 4
                        or set(p for p, n in observed) != set(map(canonical, originals))
                        or any(n <= 0 for p, n in observed) or sum(n for p, n in observed) != tree.num_entries):
                    raise RuntimeError("Per-file coverage/events mismatch: " + sample)
                print("COVERAGE %s: 4 original files, %d events verified" % (sample, tree.num_entries), flush=True)
            columns = tree.arrays(names, library="ak")
            values = [ak.to_list(columns[name]) for name in names]
            for ei, row in enumerate(zip(*values)):
                a = [np.asarray(v) for v in row]
                lengths = [len(v) for v in a]
                if len(set(lengths[:4] + lengths[10:])) != 1 or len(set(lengths[4:10])) != 1:
                    raise RuntimeError("Nonparallel event columns")
                if any(t not in (0, 1, 2) for t in a[4]) or len(set(zip(a[4], a[5]))) != lengths[4]:
                    raise RuntimeError("Invalid/duplicate component category-parent pair")
                independent, hierarchy, tier, distances = matching(*a[:10])
                if ei < args.validation_events:
                    reference = oracle(*a[:10])
                    if any(not np.array_equal(x, y) for x, y in zip((independent, hierarchy, tier), reference)):
                        raise RuntimeError("Independent scalar greedy oracle mismatch")
                    checks += 1
                for c in range(3):
                    chosen = independent[:, c][independent[:, c] >= 0]
                    if len(set(chosen)) != len(chosen):
                        raise RuntimeError("Independent category reused a component")
                chosen = hierarchy[hierarchy >= 0]
                if len(set(chosen)) != len(chosen) or not np.array_equal(independent[:, 2] >= 0, tier == 2):
                    raise RuntimeError("Hierarchy reservation/global stage mismatch")
                if np.any(independent[:, 2][tier == 2] != hierarchy[tier == 2]):
                    raise RuntimeError("Hierarchy global parent mismatch")
                for g in range(lengths[0]):
                    records.append((sample_id, *[a[k][g] for k in (10, 11, 12, 13)],
                                    *independent[g], hierarchy[g], tier[g], *distances[g], ei, a[14][g]))
        print("Matched %s: events=%d" % (sample, tree.num_entries), flush=True)
    if events_total != args.expected_events or len(revisions) != 1 or "unspecified" in revisions:
        raise RuntimeError("Incomplete event coverage or mixed/unspecified processing revision")
    rows = np.asarray(records, dtype=np.float64).reshape((-1, 16))
    del records
    matched = rows[:, 5:9].astype(np.int32)
    good = matched >= 0
    masks = {name: good[:, c] for c, name in enumerate(EFF)}
    comparable, differing = [], []
    for name, pair in zip(DISC[:3], PAIRS):
        x, y = pair
        both = good[:, x] & good[:, y]
        diff = both & (matched[:, x] != matched[:, y])
        masks[name] = diff
        comparable.append(both)
        differing.append(diff)
    masks["independent_any"] = np.any(differing, axis=0)
    masks["multiple_independent"] = np.sum(good[:, :3], axis=1) >= 2
    for name, both in zip(COMPARE[:3], comparable):
        masks[name] = both
    hd = []
    hc = []
    for c, name in zip((2, 1, 0), DISC[4:7]):
        both = good[:, 3] & good[:, c]
        masks[name] = both & (matched[:, 3] != matched[:, c])
        hc.append(both); hd.append(masks[name])
    masks["hier_any"] = np.any(hd, axis=0)
    for name, both in zip(COMPARE[4:7], hc):
        masks[name] = both
    masks["any_hier_comparison"] = np.any(hc, axis=0)

    groups = {"overall": np.ones(len(rows), dtype=bool)}
    for si, sample in enumerate(samples):
        groups["sample_" + sample] = rows[:, 0] == si
        mass, lifetime = sample.rsplit("_ctau-", 1)
        groups.setdefault("mass_" + mass, np.zeros(len(rows), dtype=bool))[:] |= rows[:, 0] == si
        groups.setdefault("ctau_" + lifetime, np.zeros(len(rows), dtype=bool))[:] |= rows[:, 0] == si
    root = ROOT.TFile.Open(str(args.output), "CREATE")
    if not root or root.IsZombie():
        raise RuntimeError("Cannot create numerical ROOT")
    ROOT.TNamed("definitions", "All plotted fractions / all accepted signal gen; same-charge station2; "
        "strict dR<0.06 inner/global,<0.10 outer; no eta/pT reco cut; ID before matching; "
        "independent categories one-to-one; hierarchy global>outer>inner reserves whole parent; "
        "raw/unweighted; no surface fallback; flow bins preserved.").Write()
    ROOT.TNamed("inputRevision", next(iter(revisions))).Write()
    ROOT.TNamed("analysisRevision", args.revision).Write()
    ROOT.TNamed("samples", "\n".join(samples)).Write()
    ROOT.TNamed("coverage", "\n".join("%s: %d events" % item for item in zip(samples, event_counts))).Write()
    summary = root.mkdir("groups")
    for group, selection in groups.items():
        directory = summary.mkdir(group)
        directory.cd()
        for vi, (var, edges) in enumerate(EDGES.items(), 1):
            bins = np.searchsorted(edges, rows[:, vi], side="right")
            for metric in ("den",) + METRICS:
                selected = selection if metric == "den" else selection & masks[metric]
                counts = np.bincount(bins[selected], minlength=len(edges) + 1)
                h = ROOT.TH1D(metric + "__" + var, LABELS[var], len(edges) - 1, array("d", edges))
                h.Sumw2()
                h.GetXaxis().SetTitle(LABELS[var]); h.GetYaxis().SetTitle("Gen muons (unweighted count)")
                for bi, count in enumerate(counts):
                    h.SetBinContent(bi, float(count)); h.SetBinError(bi, float(np.sqrt(count)))
                h.SetEntries(float(np.sum(selected))); h.Write()
                if metric != "den" and np.any(counts > np.bincount(bins[selection], minlength=len(edges) + 1)):
                    raise RuntimeError("Numerator exceeds accepted-gen denominator")
                if int(h.Integral(0, h.GetNbinsX() + 1)) != int(np.sum(selected)):
                    raise RuntimeError("Histogram sum including flow differs from row count")
    root.cd()
    tree = ROOT.TTree("matches", "One row per accepted signal gen; sample index in samples metadata")
    fields = ("sample", "pt", "eta", "lxy", "d0", "inner", "outer", "global", "hierarchy", "hierType",
              "innerDR", "outerDR", "globalDR", "hierDR", "inputEntry", "genParticleIndex")
    buffers = {field: array("d", [0]) for field in fields}
    for field, buffer in buffers.items():
        tree.Branch(field, buffer, field + "/D")
    for row in rows:
        for field, value in zip(fields, row):
            buffers[field][0] = float(value)
        tree.Fill()
    tree.Write()
    root.Close()
    print("VALIDATED: events=%d accepted_gen=%d scalar_oracle_events=%d groups=%d" %
          (events_total, len(rows), checks, len(groups)), flush=True)
    for metric in EFF + DISC + COMPARE:
        print("%s: %d / %d accepted gen" % (metric, np.sum(masks[metric]), len(rows)), flush=True)


def plot(args):
    import ROOT
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    source = ROOT.TFile.Open(str(args.output))
    if not source or source.IsZombie():
        raise RuntimeError("Missing saved numerical ROOT source")
    groups = [key.GetName() for key in source.Get("groups").GetListOfKeys()]
    args.plot_dir.mkdir(parents=True, exist_ok=False)
    def counts(group, metric, var):
        h = source.Get("groups/%s/%s__%s" % (group, metric, var))
        edges = np.array([h.GetXaxis().GetBinLowEdge(b) for b in range(1, h.GetNbinsX() + 2)])
        return np.array([h.GetBinContent(b) for b in range(1, h.GetNbinsX() + 1)]), edges
    def draw(curves, var, title, path, ylabel, logarithmic=False):
        fig, ax = plt.subplots(figsize=(10, 6.5))
        ptmax = 1.
        colors = plt.cm.get_cmap("tab20" if len(curves) > 10 else "tab10", max(1, len(curves)))
        for ci, (label, group, metric) in enumerate(curves):
            k, edges = counts(group, metric, var)
            n, _ = counts(group, "den", var)
            ok = n > 0
            centers = .5 * (edges[:-1] + edges[1:])
            if var == "pt" and np.any(ok):
                ptmax = max(ptmax, float(edges[1:][ok][-1]))
            if logarithmic:
                ok &= centers > 0
            kval, nval = k[ok].astype(int), n[ok].astype(int)
            y = kval / nval
            low = np.array([ROOT.TEfficiency.ClopperPearson(int(nn), int(kk), .682689492, False)
                            for kk, nn in zip(kval, nval)])
            high = np.array([ROOT.TEfficiency.ClopperPearson(int(nn), int(kk), .682689492, True)
                             for kk, nn in zip(kval, nval)])
            ax.errorbar(centers[ok], y, yerr=[y - low, high - y], marker=".", ms=4, lw=1,
                        color=colors(ci), label=label)
        ax.set_xlabel(LABELS[var]); ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.02); ax.grid(alpha=.25)
        if var == "pt": ax.set_xlim(0.1 if logarithmic else 0, ptmax)
        if logarithmic: ax.set_xscale("log")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, ncol=3 if len(curves) > 8 else 2, loc="upper center", bbox_to_anchor=(.5, -.14))
        fig.text(.01, .01, "All accepted signal gen; unweighted; ID before matching; same-charge station 2\n"
                 "ΔR <0.06 inner/global, <0.10 outer; propagation failures in denominator; flow bins saved in ROOT; 68.27% intervals",
                 fontsize=7)
        fig.subplots_adjust(bottom=.30)
        fig.savefig(path, dpi=130); plt.close(fig)
    panels = (("efficiency", EFF), ("independent_discrepancy", DISC[:4]), ("hierarchy_discrepancy", DISC[4:]))
    for group in groups:
        target = args.plot_dir / group
        target.mkdir()
        for var in EDGES:
            for panel, metrics in panels:
                curves = [(NICE[m], group, m) for m in metrics]
                for log in ([False, True] if args.log and var != "eta" else [False]):
                    draw(curves, var, group.replace("_", " "), target / (panel + "_" + var + ("_log" if log else "") + ".png"),
                         "Matched / accepted gen" if panel == "efficiency" else "Different parent objects / accepted gen", log)
    for kind in ("mass_", "ctau_"):
        selected = sorted(g for g in groups if g.startswith(kind))
        for metric in EFF + DISC:
            for var in EDGES:
                curves = [(g[len(kind):], g, metric) for g in selected]
                for log in ([False, True] if args.log and var != "eta" else [False]):
                    draw(curves, var, NICE[metric] + "; " + ("pooled over lifetimes" if kind == "mass_" else "pooled over masses"),
                         args.plot_dir / (kind + metric + "_" + var + ("_log" if log else "") + ".png"),
                         "Fraction of accepted gen", log)
    source.Close()
    print("PNG plots:", len(list(args.plot_dir.rglob("*.png"))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("analyze", "replot"), default="analyze")
    parser.add_argument("--revision", default="unspecified", help="analysis code revision stored in numerical ROOT")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--report-dir", type=Path, help="per-sample FrameworkJobReport directory (required for full analysis)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot-dir", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--expected-samples", type=int, default=72)
    parser.add_argument("--expected-events", type=int, default=720000)
    parser.add_argument("--validation-events", type=int, default=200)
    args = parser.parse_args()
    if args.mode == "analyze":
        if args.input_dir is None:
            parser.error("--input-dir required in analyze mode")
        if args.expected_samples > 1 and args.report_dir is None:
            parser.error("--report-dir required to verify full original-input coverage")
        if args.expected_samples > 1 and args.revision == "unspecified":
            parser.error("--revision required for full analysis provenance")
        analyze(args)
    if not args.no_plots:
        if args.plot_dir is None:
            parser.error("--plot-dir required unless --no-plots")
        plot(args)


if __name__ == "__main__":
    main()
