#!/usr/bin/env python3
"""Ten overall gen/reco direction plots; same-pair residual tables, no residual plots.

Gen directions at production vs original hf tracks at 3D PCA to the fitted vertex.
Same-charge station-2 hierarchy G>I>DSA with fit fallback is fixed before propagation.
All plotted comparisons require two accepted, matched gen muons, a valid hf vertex and
two valid propagated states. Raw unweighted pooling, one entry/pair (two for phi/eta).
Pair sign mu+ minus mu-; residual reco minus gen, wrapped again for dphi.
ROOT stores exact event values, histograms with flow bins, accounting and summary tables.
Study: kb/studies/dimuon-vertex-directions.md. plot reads ROOT only; --log adds copies.
"""
import argparse
from array import array
from pathlib import Path
import re
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preselected_vertexing_analysis import check_report, flowcounts, write_hist, CAT_NAMES, ORDERINGS

EDGES = dict(phi=np.linspace(-np.pi, np.pi, 65), eta=np.linspace(-5, 5, 101),
             dphi=np.linspace(-np.pi, np.pi, 65), deta=np.linspace(-10, 10, 101), dr=np.linspace(0, 11, 111))
LABEL = dict(phi=r"Muon momentum $\phi$ [rad]", eta=r"Muon momentum $\eta$",
             dphi=r"$\Delta\phi = \mathrm{wrap}(\phi_{\mu^+}-\phi_{\mu^-})$ [rad]",
             deta=r"$\Delta\eta = \eta_{\mu^+}-\eta_{\mu^-}$", dr=r"Dimuon $\Delta R$")
D0_EDGES = np.array([0, 1, 3, 10, 30, 100, 300, 740.0])
VECTOR = ("sigQ", "sigEta", "sigPhi", "sigD0", "sigVx", "sigVy", "sigVz", "hfRefEta", "hfRefPhi",
          "hfVertexEta", "hfVertexPhi", "hfVertexValid", "hfVertexStatus", "hfVertexPath", "hfVertexDistance")
SCALAR = ("run", "lumi", "event", "twoAcc", "bbStatus", "hfStatus", "hfTypeA", "hfTypeB", "hfTried", "hfFirstIsBB")


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def load(args):
    import awkward as ak
    import uproot
    paths = sorted(Path(args.input_dir).glob("*.root"))
    if len(paths) != args.expected_samples:
        raise RuntimeError("Input sample count mismatch: %d" % len(paths))
    names, revisions, chunks = [], set(), []
    for si, path in enumerate(paths):
        with uproot.open(path) as f:
            prefix = "preselectedVertexing/"
            name = str(f[prefix + "sample"].member("fTitle"))
            if name in names or not re.fullmatch(r"Mchi-[\dp]+_dMchi-[\dp]+_ctau-[\dp]+", name):
                raise RuntimeError("Unexpected or duplicate sample: " + name)
            names.append(name)
            revisions.add(str(f[prefix + "revision"].member("fTitle")))
            if str(f[prefix + "hierarchyOrder"].member("fTitle")) != "global>inner>outer(DSA)":
                raise RuntimeError("Unexpected hierarchy order: " + name)
            if str(f[prefix + "vertexDirections"].member("fTitle")) == "disabled":
                raise RuntimeError("Point propagation was disabled: " + name)
            t = f[prefix + "events"]
            n = t.num_entries
            check_report(Path(args.report_dir), name, n)
            a = t.arrays(list(SCALAR + VECTOR), library="ak")
            c = {k: np.asarray(a[k]) for k in SCALAR}
            c["sample"] = np.full(n, si, dtype=np.int32)
            two, fit = c["twoAcc"].astype(bool), c["hfStatus"] == 3
            for k in VECTOR:
                selected = two if k.startswith("sig") else fit
                if np.any(np.asarray(ak.num(a[k]))[selected] != 2):
                    raise RuntimeError("Non-pair vector %s in %s" % (k, name))
                if k.startswith("hf") and np.any(np.asarray(ak.num(a[k]))[~fit] != 0):
                    raise RuntimeError("Propagation attempted without a valid fit")
                c[k] = np.full((n, 2), np.nan)
                if np.any(selected):
                    c[k][selected] = ak.to_numpy(a[k][selected])
            if np.any(fit & ~two) or np.any(two & (np.prod(c["sigQ"], axis=1) != -1)):
                raise RuntimeError("Fit/acceptance/charge inconsistency")
            if np.any(fit & ~c["hfFirstIsBB"]):
                raise RuntimeError("Unexpected hierarchy first-pair ordering")
            vdiff = np.max(np.abs(np.stack([c[k][:, 0] - c[k][:, 1] for k in
                                           ("sigVx", "sigVy", "sigVz")], axis=1)), axis=1)
            if np.any(two & ~(vdiff <= 1e-4)):
                raise RuntimeError("Gen production vertices disagree by > 1e-4 cm")
            chunks.append(c)
    if len(revisions) != 1 or "unspecified" in revisions:
        raise RuntimeError("Mixed/unspecified input revisions")
    e = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    if len(e["event"]) != args.expected_events:
        raise RuntimeError("Event total mismatch")
    return names, e, revisions.pop()


def quantities(eta, phi, charge):
    # q is physical electric charge, not PDG ID; all comparisons have (+1,-1).
    plus = charge[:, 0] == 1
    de = np.where(plus, eta[:, 0] - eta[:, 1], eta[:, 1] - eta[:, 0])
    dp = wrap(np.where(plus, phi[:, 0] - phi[:, 1], phi[:, 1] - phi[:, 0]))
    return dict(phi=wrap(phi), eta=eta, dphi=dp, deta=de, dr=np.hypot(de, dp))


def analyze(args):
    import ROOT
    ROOT.gROOT.SetBatch(True)
    if Path(args.output).exists():
        raise RuntimeError("Refusing to overwrite numerical ROOT output")
    names, e, revision = load(args)
    n = len(e["event"])
    two, fit = e["twoAcc"].astype(bool), e["hfStatus"] == 3
    leg_ok = e["hfVertexValid"] == 1
    good = fit & leg_ok.all(axis=1)
    for k in ("hfVertexEta", "hfVertexPhi", "hfVertexDistance", "hfVertexPath"):
        if np.any(leg_ok & (~np.isfinite(e[k]) | ((e[k] == -999) & (k != "hfVertexPath")))):
            raise RuntimeError("Valid propagation with invalid state: " + k)
    if np.any(leg_ok & (e["hfVertexDistance"] < 0)):
        raise RuntimeError("Negative PCA-target distance")
    q = {"gen": quantities(e["sigEta"], e["sigPhi"], e["sigQ"]),
         "reference": quantities(e["hfRefEta"], e["hfRefPhi"], e["sigQ"]),
         "reco": quantities(np.where(leg_ok, e["hfVertexEta"], np.nan),
                            np.where(leg_ok, e["hfVertexPhi"], np.nan), e["sigQ"])}
    if any(np.any(~np.isfinite(x[good])) for z in q.values() for x in z.values()):
        raise RuntimeError("Nonfinite direction in complete-pair comparison")
    pair = np.full(n, "none", dtype="U4")
    for cat, orderings in ORDERINGS.items():
        for a, b in orderings:
            pair[(e["hfTypeA"] == a) & (e["hfTypeB"] == b)] = cat
    if np.any(fit & (pair == "none")):
        raise RuntimeError("Valid fit has unknown component types")
    maxd0 = np.max(np.where(two[:, None], e["sigD0"], -1), axis=1)
    groups = {"overall": np.ones(n, bool)}
    for si, name in enumerate(names):
        groups["sample/" + name] = e["sample"] == si
    for kind, fn in (("mass", lambda s: s.rsplit("_ctau-", 1)[0]),
                     ("lifetime", lambda s: "ctau-" + s.rsplit("_ctau-", 1)[1])):
        for key in sorted({fn(s) for s in names}):
            groups[kind + "/" + key] = np.isin(e["sample"], [i for i, s in enumerate(names) if fn(s) == key])
    for lo, hi in zip(D0_EDGES[:-1], D0_EDGES[1:]):
        groups["max_gen_abs_d0_cm/%g-%g" % (lo, hi)] = two & (maxd0 >= lo) & (maxd0 < hi)
    strata = {"all": np.ones(n, bool), **{c: pair == c for c in ORDERINGS},
              "first_fit": fit & (e["hfTried"] == 1), "fallback": fit & (e["hfTried"] > 1)}
    root = ROOT.TFile.Open(str(args.output), "CREATE")
    ROOT.TNamed("definitions", __doc__).Write()
    ROOT.TNamed("inputRevision", revision).Write()
    ROOT.TNamed("analysisRevision", args.revision).Write()
    ROOT.TNamed("samples", "\n".join(names)).Write()
    root.mkdir("distributions"); root.cd("distributions")
    for source in ("gen", "reco"):
        for var, edges in EDGES.items():
            x = q[source][var][good].ravel()
            write_hist(source + "_" + var, ";%s;%s / bin" % (LABEL[var], "Muons" if var in ("eta", "phi") else "Pairs"),
                       edges, flowcounts(x, edges))
    root.cd()

    # Exact per-event source supports quantiles and replotting without rereading MINIAOD.
    t = ROOT.TTree("events", "All processed events; same-pair comparisons require complete=1")
    ints = {k: array("i", [0]) for k in ("sample", "pair", "twoAcc", "hfStatus", "bbStatus", "fallback", "complete")}
    floats = {"maxD0": maxd0}
    for source in q:
        for var in ("dphi", "deta", "dr"):
            floats[source + "_" + var] = q[source][var]
        for var in ("phi", "eta"):
            for g in range(2):
                floats["%s_%s%d" % (source, var, g)] = q[source][var][:, g]
    for k in ("hfVertexValid", "hfVertexStatus", "hfVertexPath", "hfVertexDistance"):
        for g in range(2):
            floats[k + str(g)] = e[k][:, g]
    fb = {k: array("d", [0]) for k in floats}
    for k, b in ints.items(): t.Branch(k, b, k + "/I")
    for k, b in fb.items(): t.Branch(k, b, k + "/D")
    ids = {k: array("Q", [0]) for k in ("run", "lumi", "event")}
    for k, b in ids.items(): t.Branch(k, b, k + "/l")
    cats = tuple(ORDERINGS)
    ROOT.TNamed("pairCodes", "-1 none; " + "; ".join("%d %s" % (i, c) for i, c in enumerate(cats))).Write()
    for i in range(n):
        for k in ("sample", "twoAcc", "hfStatus", "bbStatus"): ints[k][0] = int(e[k][i])
        ints["pair"][0] = cats.index(pair[i]) if pair[i] in cats else -1
        ints["fallback"][0], ints["complete"][0] = int(fit[i] and e["hfTried"][i] > 1), int(good[i])
        for k in floats: fb[k][0] = float(floats[k][i])
        for k in ids: ids[k][0] = int(e[k][i])
        t.Fill()
    t.Write()

    summary = ROOT.TTree("summary", "Same-complete-pair residuals; reference and reco use identical events")
    texts = {k: ROOT.std.string() for k in ("group", "stratum", "estimator", "quantity")}
    nb = array("i", [0]); summary.Branch("n", nb, "n/I")
    stats = {k: array("d", [0]) for k in ("median", "q16", "q84", "q95abs")}
    for k, b in texts.items(): summary.Branch(k, b)
    for k, b in stats.items(): summary.Branch(k, b, k + "/D")
    accounting = ROOT.TTree("accounting", "Nonexclusive named populations; all counts raw events unless marked legs")
    ag, ac = ROOT.std.string(), ROOT.std.string()
    av = array("i", [0]); accounting.Branch("group", ag); accounting.Branch("condition", ac); accounting.Branch("n", av, "n/I")
    counters = {"processed": np.ones(n, bool), "two_accepted": two, "both_matched": e["bbStatus"] > 0,
                "shared_tracks": e["hfStatus"] == 1, "no_valid_hf_fit": e["hfStatus"] == 2,
                "valid_hf_fit": fit, "fallback_attempted": e["hfTried"] > 1,
                "fallback_no_valid_fit": (e["hfTried"] > 1) & ~fit,
                "fit_fallback": fit & (e["hfTried"] > 1), "complete": good,
                "one_leg_failed": fit & (leg_ok.sum(axis=1) == 1), "two_legs_failed": fit & ~leg_ok.any(axis=1)}
    print("RESIDUAL SUMMARY: raw, unweighted, same complete pairs; residual=reco-gen; phi in rad", flush=True)
    print("group | stratum | estimator | quantity | N | median | q16 | q84 | q95abs", flush=True)
    for group, gm in groups.items():
        for condition, mask in counters.items():
            ag.assign(group); ac.assign(condition); av[0] = int((gm & mask).sum()); accounting.Fill()
        for cat in cats:
            for label, mask in (("valid", fit), ("complete", good), ("fallback", counters["fit_fallback"]),
                                ("one_leg_failed", counters["one_leg_failed"]), ("two_legs_failed", counters["two_legs_failed"])):
                ag.assign(group); ac.assign(cat + "/" + label); av[0] = int((gm & (pair == cat) & mask).sum()); accounting.Fill()
        for stratum, sm in strata.items():
            sel = gm & sm & good
            for estimator in ("reference", "reco"):
                for var in ("dphi", "deta", "dr"):
                    x = q[estimator][var][sel] - q["gen"][var][sel]
                    if var == "dphi": x = wrap(x)
                    values = (*np.percentile(x, [50, 16, 84]), np.percentile(np.abs(x), 95)) if len(x) else (np.nan,) * 4
                    nb[0] = len(x)
                    for key, value in zip(texts, (group, stratum, estimator, var)): texts[key].assign(value)
                    for key, value in zip(stats, values): stats[key][0] = float(value)
                    summary.Fill()
                    print("%s | %s | %s | %s | %d | %.7g | %.7g | %.7g | %.7g" %
                          (group, stratum, estimator, var, len(x), *values))
    accounting.Write(); summary.Write()
    for label, mask in counters.items(): print("ACCOUNT overall %-20s %d" % (label, int(mask.sum())))
    for label, mask in strata.items():
        selected = fit & mask
        for g in range(2):
            sel = selected & leg_ok[:, g]
            path, dist = e["hfVertexPath"][sel, g], e["hfVertexDistance"][sel, g]
            print("TRANSPORT %s gen-index%d valid %d forward %d backward %d zero %d distance_cm_q50_q95 %s" %
                  (label, g, len(path), (path > 0).sum(), (path < 0).sum(), (path == 0).sum(),
                   np.percentile(dist, [50, 95]) if len(dist) else "n/a"))
    root.Close()
    print("VALIDATED %d samples, %d processed events; all file coverage and comparison checks passed" % (len(names), n), flush=True)


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import uproot
    out = Path(args.plot_dir)
    out.mkdir(parents=True, exist_ok=True)
    with uproot.open(args.output) as f:
        if args.dr_only:
            events = f["events"].arrays(["complete", "gen_dr", "reco_dr"], library="np")
        for source in ("gen", "reco"):
            for var in (("dr",) if args.dr_only else EDGES):
                if args.dr_only:
                    x = events[source + "_dr"][events["complete"] == 1]
                    if not np.all(np.isfinite(x)):
                        raise ValueError("Nonfinite dR in complete-pair population")
                    edges = np.r_[np.arange(32, dtype=float) / 10., 3.14]
                    values, _ = np.histogram(x[(x >= 0) & (x < 3.14)], bins=edges)
                    flow = np.r_[(x < 0).sum(), values, (x >= 3.14).sum()]
                else:
                    h = f["distributions/%s_%s" % (source, var)]
                    values, edges = h.to_numpy()
                    flow = h.values(flow=True)
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.stairs(values, edges, lw=1.6, color="#0072B2" if source == "gen" else "#D55E00")
                if args.dr_only:
                    ax.set_xlim(0, 3.14)
                    ax.set_xticks(np.arange(32, dtype=float) / 10.)
                    ax.tick_params(axis="x", labelrotation=90, labelsize=6)
                ax.set_xlabel(LABEL[var]); ax.set_ylabel(("Muons" if var in ("phi", "eta") else "Pairs") + " / bin")
                ax.set_ylim(bottom=0)
                ax.set_title("Gen at production" if source == "gen" else "Original reco tracks at PCA to fitted vertex")
                fig.text(.02, .025, "2022EE signal MC | same complete pairs | raw, unweighted" +
                         (" | last bin 3.10–3.14" if args.dr_only else ""), fontsize=8)
                fig.text(.02, .005, "Entries: %d | underflow: %d | overflow: %d" % (flow.sum(), flow[0], flow[-1]), fontsize=8)
                fig.tight_layout(rect=(0, .06, 1, 1))
                fig.savefig(out / (source + "_" + var + ".png"), dpi=140)
                if args.log:
                    ax.set_yscale("log"); fig.savefig(out / (source + "_" + var + "_log.png"), dpi=140)
                plt.close(fig)
    count = 2 if args.dr_only else 10
    print("PLOTTED %d linear PNGs" % count + (" and %d log copies" % count if args.log else ""))


def plot_dr(args):
    """Gen vs reco-at-PCA dR in 0.05 bins to 3.15, overlays and per-pair residuals; all pairs and final DSA-DSA."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import uproot
    out = Path(args.plot_dir)
    out.mkdir(parents=True, exist_ok=True)
    with uproot.open(args.output) as f:
        ev = f["events"].arrays(["complete", "pair", "fallback", "gen_dr", "reco_dr"], library="np")
    complete = ev["complete"] == 1
    oo = tuple(ORDERINGS).index("OO")
    edges = np.round(np.arange(64) * 0.05, 2)
    redges = np.round(np.arange(-50, 51) * 0.02, 2)
    note = ("2022EE signal MC | two accepted gen muons, both matched (G>I>DSA), valid hf vertex, two valid PCA states"
            " | raw, unweighted")
    color = dict(gen="#0072B2", reco="#D55E00", residual="#009E73")
    title = dict(gen="Gen at production", reco="Original reco tracks at PCA to fitted vertex",
                 residual="Per pair: reco at PCA minus gen")

    def hist(x, e):
        if not np.all(np.isfinite(x)):
            raise ValueError("Nonfinite dR in complete-pair population")
        values, _ = np.histogram(x, bins=e)  # numpy's last bin is closed; move x == upper edge to overflow
        values[-1] -= (x == e[-1]).sum()
        return values, (x < e[0]).sum(), (x >= e[-1]).sum()

    def save(fig, ax, name, bottom=0.5):
        fig.savefig(out / (name + ".png"), dpi=140)
        if args.log:
            top = ax.get_ylim()[1]
            ax.set_yscale("log"); ax.set_ylim(bottom, 3 * top); fig.savefig(out / (name + "_log.png"), dpi=140)
        plt.close(fig)

    count = 0
    for tag, mask, label in (("all", complete, "All vertex types"), ("dsadsa", complete & (ev["pair"] == oo), "DSA–DSA vertex")):
        n, nfb = mask.sum(), (mask & (ev["fallback"] == 1)).sum()
        pop = "%s: %d pairs (%d via fit fallback)" % (label, n, nfb)
        x = {s: ev[s + "_dr"][mask] for s in ("gen", "reco")}
        x["residual"] = x["reco"] - x["gen"]
        h = {s: hist(x[s], redges if s == "residual" else edges) for s in x}
        for s in x:
            values, under, over = h[s]
            e = redges if s == "residual" else edges
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.stairs(values, e, lw=1.6, color=color[s])
            ax.set_xlim(e[0], e[-1])
            ax.set_xlabel(r"$\Delta R_\mathrm{reco} - \Delta R_\mathrm{gen}$" if s == "residual" else LABEL["dr"])
            ax.set_ylabel("Pairs / %g" % (e[1] - e[0]))
            ax.set_ylim(bottom=0)
            ax.set_title(title[s] + " — " + label)
            fig.text(.02, .025, note, fontsize=7)
            fig.text(.02, .005, "%s | underflow %d | overflow %d" % (pop, under, over), fontsize=7)
            fig.tight_layout(rect=(0, .06, 1, 1))
            save(fig, ax, "%s_dr_%s" % (s, tag)); count += 1
        for norm in (False, True):
            fig, ax = plt.subplots(figsize=(7, 5))
            for s in ("gen", "reco"):
                values, under, over = h[s]
                w = values / float(n) if norm else values
                ax.stairs(w, edges, lw=1.6, color=color[s],
                          label="%s (overflow %d)" % ("Gen at production" if s == "gen" else "Reco at PCA to vertex", over))
            ax.set_xlim(edges[0], edges[-1])
            ax.set_xlabel(LABEL["dr"])
            ax.set_ylabel("Fraction of pairs / 0.05 (unit area incl. overflow)" if norm else "Pairs / 0.05")
            ax.set_ylim(bottom=0)
            ax.set_title(("Unit-area normalized" if norm else "Unnormalized") + " — " + label)
            ax.legend(frameon=False)
            fig.text(.02, .025, note, fontsize=7)
            fig.text(.02, .005, pop + " | same pairs in both series", fontsize=7)
            fig.tight_layout(rect=(0, .06, 1, 1))
            save(fig, ax, "overlay_dr_%s%s" % (tag, "_unitarea" if norm else ""), 0.5 / n if norm else 0.5); count += 1
        print("%s N=%d fallback=%d gen_over=%d reco_over=%d residual_under=%d residual_over=%d" %
              (tag, n, nfb, h["gen"][2], h["reco"][2], h["residual"][1], h["residual"][2]))
    print("PLOTTED %d linear PNGs" % count + (" and %d log copies" % count if args.log else ""))


def plot_dr_cms(args):
    """plot-dr content in PyROOT, CMS style: same bins, populations and outputs as plot_dr."""
    import ROOT
    import uproot
    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0); ROOT.gStyle.SetOptTitle(0)
    ROOT.gStyle.SetTextFont(42); ROOT.gStyle.SetLabelFont(42, "XYZ"); ROOT.gStyle.SetTitleFont(42, "XYZ")
    ROOT.gStyle.SetPadTickX(1); ROOT.gStyle.SetPadTickY(1)
    ROOT.TH1.AddDirectory(False)
    ROOT.TGaxis.SetMaxDigits(6)  # no x10^n exponent colliding with the CMS label
    out = Path(args.plot_dir)
    out.mkdir(parents=True, exist_ok=True)
    with uproot.open(args.output) as f:
        ev = f["events"].arrays(["complete", "pair", "fallback", "gen_dr", "reco_dr", "run", "lumi", "event"], library="np")
    complete = ev["complete"] == 1
    cut = ""
    if args.max_chi2ndof is not None:
        # hf vertex chi2/ndof is not in the events tree; take it from the inputs, concatenated in the same sorted order.
        chi2, ndof, ids = [], [], []
        for path in sorted(Path(args.input_dir).glob("*.root")):
            with uproot.open(path) as f:
                a = f["preselectedVertexing/events"].arrays(["hfChi2", "hfNdof", "run", "lumi", "event"], library="np")
            chi2.append(a["hfChi2"]); ndof.append(a["hfNdof"]); ids.append(np.stack([a[k] for k in ("run", "lumi", "event")], 1))
        ids = np.concatenate(ids)
        if len(ids) != len(complete) or np.any(ids != np.stack([ev[k] for k in ("run", "lumi", "event")], 1)):
            raise RuntimeError("Input events do not align with the events tree")
        chi2, ndof = np.concatenate(chi2), np.concatenate(ndof)
        if np.any(complete & ~(ndof > 0)):
            raise RuntimeError("Complete pair with ndof <= 0")
        complete &= chi2 / np.where(ndof > 0, ndof, 1) < args.max_chi2ndof
        cut = ", #chi^{2}/ndof < %g" % args.max_chi2ndof
    oo = tuple(ORDERINGS).index("OO")
    edges = np.round(np.arange(64) * 0.05, 2)
    redges = np.round(np.arange(-50, 51) * 0.02, 2)
    color = dict(gen=ROOT.TColor.GetColor("#0072B2"), reco=ROOT.TColor.GetColor("#D55E00"),
                 residual=ROOT.TColor.GetColor("#009E73"))
    legend = dict(gen="Gen at production", reco="Reco at PCA to fitted vertex", residual="Per pair")
    xtitle = dict(dr="Dimuon #DeltaR", residual="#DeltaR_{reco} #minus #DeltaR_{gen}")
    keep = []

    def th1(name, x, e):
        if not np.all(np.isfinite(x)):
            raise ValueError("Nonfinite dR in complete-pair population")
        h = ROOT.TH1D(name, "", len(e) - 1, array("d", e))
        values, _ = np.histogram(x, bins=e)
        values[-1] -= (x == e[-1]).sum()  # numpy's last bin is closed; x == upper edge is overflow, as in TH1
        for i, v in enumerate(values, 1): h.SetBinContent(i, float(v))
        h.SetBinContent(0, float((x < e[0]).sum())); h.SetBinContent(len(e), float((x >= e[-1]).sum()))
        h.SetEntries(len(x))
        return h

    def frame(h, ytitle):
        h.GetXaxis().SetTitle(xtitle["residual" if h.GetXaxis().GetXmin() < 0 else "dr"])
        h.GetYaxis().SetTitle(ytitle)
        for ax in (h.GetXaxis(), h.GetYaxis()):
            ax.SetTitleSize(0.05); ax.SetLabelSize(0.042)
        h.GetXaxis().SetTitleOffset(1.1); h.GetYaxis().SetTitleOffset(1.65)

    def labels(c, lines):
        c.cd()
        t = ROOT.TLatex(); t.SetNDC(); keep.append(t)
        t.SetTextFont(61); t.SetTextSize(0.05); t.SetTextAlign(11); t.DrawLatex(0.17, 0.94, "CMS")
        t.SetTextFont(52); t.SetTextSize(0.038); t.DrawLatex(0.275, 0.94, "Simulation Private Work")
        t.SetTextFont(42); t.SetTextAlign(31); t.DrawLatex(0.96, 0.94, "2022EE (13.6 TeV)")
        t.SetTextSize(0.033); t.SetTextAlign(13)
        for i, line in enumerate(lines): t.DrawLatex(0.21, 0.88 - 0.045 * i, line)

    def save(c, hists, name, bottom=0.5):
        top = max(h.GetBinContent(h.GetMaximumBin()) for h in hists)  # GetMaximum() would return a SetMaximum value
        hists[0].SetMinimum(0); hists[0].SetMaximum(1.45 * top)
        c.Modified(); c.SaveAs(str(out / (name + ".png")))
        if args.log:
            c.SetLogy(1); hists[0].SetMinimum(bottom); hists[0].SetMaximum(top * 60)
            c.Modified(); c.SaveAs(str(out / (name + "_log.png"))); c.SetLogy(0)

    def canvas(name):
        c = ROOT.TCanvas(name, "", 800, 700)
        c.SetLeftMargin(0.17); c.SetRightMargin(0.04); c.SetTopMargin(0.08); c.SetBottomMargin(0.13)
        keep.append(c)
        return c

    count = 0
    for tag, mask, label in (("all", complete, "All vertex types"), ("dsadsa", complete & (ev["pair"] == oo), "DSA#minusDSA vertex")):
        n, nfb = int(mask.sum()), int((mask & (ev["fallback"] == 1)).sum())
        head = "%s%s: %d pairs (%d via fit fallback)" % (label, cut, n, nfb)
        x = {s: ev[s + "_dr"][mask] for s in ("gen", "reco")}
        x["residual"] = x["reco"] - x["gen"]
        g9, r9 = x["gen"] < 0.9, x["reco"] < 0.9
        print("%s dR<0.9 of N=%d: gen %d reco %d both %d gen_only %d reco_only %d" %
              (tag, n, g9.sum(), r9.sum(), (g9 & r9).sum(), (g9 & ~r9).sum(), (~g9 & r9).sum()))
        h = {s: th1("h_%s_%s" % (s, tag), x[s], redges if s == "residual" else edges) for s in x}
        for s in x:
            c = canvas("c_%s_%s" % (s, tag))
            hs = h[s]; hs.SetLineColor(color[s]); hs.SetLineWidth(3)
            frame(hs, "Pairs / %g" % (redges[1] - redges[0] if s == "residual" else 0.05))
            hs.Draw("hist")
            under, over = int(hs.GetBinContent(0)), int(hs.GetBinContent(hs.GetNbinsX() + 1))
            flow = ("Underflow %d, overflow %d" % (under, over) if s == "residual"
                    else "Overflow at #DeltaR #geq 3.15: %d" % over)
            labels(c, [head, legend[s] if s != "residual" else "Reco at PCA minus gen, per pair", flow])
            save(c, [hs], "%s_dr_%s" % (s, tag)); count += 1
        for norm in (False, True):
            c = canvas("c_overlay_%s_%d" % (tag, norm))
            hs = [h[s].Clone("o_%s_%s_%d" % (s, tag, norm)) for s in ("gen", "reco")]
            for s, hh in zip(("gen", "reco"), hs):
                hh.SetLineColor(color[s]); hh.SetLineWidth(3)
                if norm: hh.Scale(1.0 / n)  # n includes overflow: drawn bins sum to 1 - overflow fraction
            frame(hs[0], "Fraction of pairs / 0.05" if norm else "Pairs / 0.05")
            hs[0].Draw("hist"); hs[1].Draw("hist same")
            leg = ROOT.TLegend(0.52, 0.60, 0.94, 0.72); keep.append(leg)
            leg.SetBorderSize(0); leg.SetFillStyle(0); leg.SetTextSize(0.033)
            for s, hh in zip(("gen", "reco"), hs):
                leg.AddEntry(hh, "%s" % legend[s], "l")
            leg.Draw()
            over = [int(h[s].GetBinContent(h[s].GetNbinsX() + 1)) for s in ("gen", "reco")]
            labels(c, [head, "Overflow at #DeltaR #geq 3.15: gen %d, reco %d" % tuple(over)] +
                   (["Unit area incl. overflow"] if norm else []))
            save(c, hs, "overlay_dr_%s%s" % (tag, "_unitarea" if norm else ""), 0.5 / n if norm else 0.5); count += 1
        print("%s N=%d fallback=%d gen_over=%d reco_over=%d residual_under=%d residual_over=%d" %
              (tag, n, nfb, h["gen"].GetBinContent(64), h["reco"].GetBinContent(64),
               h["residual"].GetBinContent(0), h["residual"].GetBinContent(101)))
    print("PLOTTED %d linear PNGs" % count + (" and %d log copies" % count if args.log else ""))


def plot_dphi(args):
    """|dphi(X, reco mumu)|, X = PF/PUPPI/gen MET, gen chi1chi1 pT; complete pairs with chi2/ndof and PCA dR cuts.

    PyROOT, CMS style. Main mumu pT = sum of hf-leg pT at the 3D PCA to the fitted vertex (hfVertexPt/Phi);
    cross-check (printed only) = hfRecoMuMuPhi at the track reference points. Raw, unweighted, one entry per pair.
    Study: kb/studies/met-dimuon-dphi.md Part 3.
    """
    import awkward as ak
    import ROOT
    import uproot
    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0); ROOT.gStyle.SetOptTitle(0)
    ROOT.gStyle.SetTextFont(42); ROOT.gStyle.SetLabelFont(42, "XYZ"); ROOT.gStyle.SetTitleFont(42, "XYZ")
    ROOT.gStyle.SetPadTickX(1); ROOT.gStyle.SetPadTickY(1)
    ROOT.TH1.AddDirectory(False)
    ROOT.TGaxis.SetMaxDigits(6)
    out = Path(args.plot_dir)
    out.mkdir(parents=True, exist_ok=True)
    ids = ("run", "lumi", "event")
    with uproot.open(args.output) as f:
        ev = f["events"].arrays(["complete", "pair", "fallback", "reco_dr", *ids], library="np")
    scalars = ("hfChi2", "hfNdof", "hfStatus", "hfRecoMuMuPhi", "pfMetPt", "pfMetPhi", "puppiMetPt", "puppiMetPhi",
               "genMetPt", "genMetPhi", "chiOK", "nChi", "chiPx", "chiPy") + ids
    vectors = ("hfVertexValid", "hfVertexPt", "hfVertexPhi")
    parts = []
    for path in sorted(Path(args.input_dir).glob("*.root")):
        with uproot.open(path) as f:
            a = f["preselectedVertexing/events"].arrays(list(scalars + vectors), library="ak")
        c = {k: np.asarray(a[k]) for k in scalars}
        fit = c["hfStatus"] == 3
        for k in vectors:  # two legs for valid fits, empty otherwise (PreselectedVertexing)
            if np.any(np.asarray(ak.num(a[k]))[fit] != 2) or np.any(np.asarray(ak.num(a[k]))[~fit] != 0):
                raise RuntimeError("Unexpected %s length in %s" % (k, path.name))
            c[k] = np.full((len(fit), 2), np.nan)
            if np.any(fit):
                c[k][fit] = ak.to_numpy(a[k][fit])
        parts.append(c)
    e = {k: np.concatenate([c[k] for c in parts]) for k in parts[0]}
    if len(e["run"]) != len(ev["run"]) or any(np.any(e[k] != ev[k]) for k in ids):
        raise RuntimeError("Input events do not align with the events tree")
    complete = ev["complete"] == 1
    if np.any(complete & ((e["hfStatus"] != 3) | ~(e["hfNdof"] > 0) | np.any(e["hfVertexValid"] != 1, axis=1))):
        raise RuntimeError("Complete pair without a valid fit, ndof > 0 and two valid PCA states")
    chi2ndof = e["hfChi2"] / np.where(e["hfNdof"] > 0, e["hfNdof"], 1)
    sel = complete & (chi2ndof < args.max_chi2ndof) & (ev["reco_dr"] < args.max_dr)
    if np.any(sel & ~np.all(e["hfVertexPt"] > 0, axis=1)):
        raise RuntimeError("Selected pair with non-positive PCA pT")
    pt, ph = e["hfVertexPt"], e["hfVertexPhi"]
    mumu = {"pca": np.arctan2((pt * np.sin(ph)).sum(axis=1), (pt * np.cos(ph)).sum(axis=1)),
            "ref": e["hfRecoMuMuPhi"]}
    # Validity from flags, not finiteness: atan2/wrap of -999 sentinels give plausible angles.
    xphi = {"pf": (e["pfMetPhi"], e["pfMetPt"] != -999), "puppi": (e["puppiMetPhi"], e["puppiMetPt"] != -999),
            "gen": (e["genMetPhi"], e["genMetPt"] != -999),
            "chi": (np.arctan2(e["chiPy"], e["chiPx"]), e["chiOK"].astype(bool))}
    keys = ("pf", "puppi", "gen", "chi")
    legend = dict(pf="PF p_{T}^{miss} (Type-1)", puppi="PUPPI p_{T}^{miss} (Type-1)", gen="Gen p_{T}^{miss}",
                  chi="Gen #chi_{1}#chi_{1} p_{T}")
    xtitle = dict(pf="|#Delta#phi(PF p_{T}^{miss}, p_{T}^{#mu#mu})|",
                  puppi="|#Delta#phi(PUPPI p_{T}^{miss}, p_{T}^{#mu#mu})|",
                  gen="|#Delta#phi(gen p_{T}^{miss}, p_{T}^{#mu#mu})|",
                  chi="|#Delta#phi(gen p_{T}^{#chi_{1}#chi_{1}}, p_{T}^{#mu#mu})|",
                  overlay="|#Delta#phi(X, p_{T}^{#mu#mu})|")
    color = dict(pf=ROOT.TColor.GetColor("#0072B2"), puppi=ROOT.TColor.GetColor("#E69F00"),
                 gen=ROOT.TColor.GetColor("#009E73"), chi=ROOT.TColor.GetColor("#CC79A7"))
    style = dict(pf=1, puppi=2, gen=1, chi=7)
    edges = np.round(np.arange(64) * 0.05, 2)  # 0-3.15 in 0.05; |dphi| <= pi, so no overflow; 0.5 is an edge
    oo = tuple(ORDERINGS).index("OO")
    keep = []

    def th1(name, x):
        h = ROOT.TH1D(name, "", len(edges) - 1, array("d", edges))
        values, _ = np.histogram(x, bins=edges)
        values[-1] -= (x == edges[-1]).sum()
        for i, v in enumerate(values, 1): h.SetBinContent(i, float(v))
        h.SetBinContent(0, float((x < edges[0]).sum())); h.SetBinContent(len(edges), float((x >= edges[-1]).sum()))
        h.SetEntries(len(x))
        return h

    def frame(h, xt, yt):
        h.GetXaxis().SetTitle(xt); h.GetYaxis().SetTitle(yt)
        for ax in (h.GetXaxis(), h.GetYaxis()):
            ax.SetTitleSize(0.05); ax.SetLabelSize(0.042)
        h.GetXaxis().SetTitleOffset(1.15); h.GetYaxis().SetTitleOffset(1.65)

    def labels(c, lines):
        c.cd()
        t = ROOT.TLatex(); t.SetNDC(); keep.append(t)
        t.SetTextFont(61); t.SetTextSize(0.05); t.SetTextAlign(11); t.DrawLatex(0.17, 0.94, "CMS")
        t.SetTextFont(52); t.SetTextSize(0.038); t.DrawLatex(0.275, 0.94, "Simulation Private Work")
        t.SetTextFont(42); t.SetTextAlign(31); t.DrawLatex(0.96, 0.94, "2022EE (13.6 TeV)")
        t.SetTextSize(0.033); t.SetTextAlign(33)
        for i, line in enumerate(lines): t.DrawLatex(0.93, 0.88 - 0.045 * i, line)

    def cutline(top):
        ln = ROOT.TLine(args.cut, 0, args.cut, top); keep.append(ln)
        ln.SetLineColor(ROOT.kGray + 2); ln.SetLineStyle(3); ln.SetLineWidth(2)
        ln.Draw()

    def save(c, hists, name, scale, bottom):
        top = max(h.GetBinContent(h.GetMaximumBin()) for h in hists)
        hists[0].SetMinimum(0); hists[0].SetMaximum(scale * top)
        cutline(scale * top)
        c.Modified(); c.SaveAs(str(out / (name + ".png")))
        if args.log:
            c.SetLogy(1); hists[0].SetMinimum(bottom); hists[0].SetMaximum(top * 150)
            keep[-1].SetY2(top * 150)
            c.Modified(); c.SaveAs(str(out / (name + "_log.png"))); c.SetLogy(0)

    def canvas(name):
        c = ROOT.TCanvas(name, "", 800, 700)
        c.SetLeftMargin(0.17); c.SetRightMargin(0.04); c.SetTopMargin(0.08); c.SetBottomMargin(0.13)
        keep.append(c)
        return c

    cuts = "#chi^{2}/ndof < %g, #DeltaR_{PCA} < %g" % (args.max_chi2ndof, args.max_dr)
    print("complete=%d selected(chi2/ndof<%g & reco_dr<%g)=%d" % (complete.sum(), args.max_chi2ndof, args.max_dr, sel.sum()))
    count = 0
    for tag, label in (("all", "All vertex types"), ("dsadsa", "DSA#minusDSA vertex")):
        mask = sel & (ev["pair"] == oo) if tag == "dsadsa" else sel
        n, nfb = int(mask.sum()), int((mask & (ev["fallback"] == 1)).sum())
        print("%s N=%d fallback=%d" % (tag, n, nfb))
        h, frac = {}, {}
        for k in keys:
            phi, ok = xphi[k]
            bad = mask & ~ok
            if k == "chi":
                print("  chi excluded %d (nChi!=2: %d, wrong parents: %d)" %
                      (bad.sum(), (bad & (e["nChi"] != 2)).sum(), (bad & (e["nChi"] == 2)).sum()))
            elif bad.any():
                raise RuntimeError("%s MET missing for %d selected pairs" % (k, bad.sum()))
            for mm in ("pca", "ref"):
                x = np.abs(wrap(phi[mask & ok] - mumu[mm][mask & ok]))
                if not np.all(np.isfinite(x)):
                    raise ValueError("Nonfinite dphi")
                npass = int((x < args.cut).sum())
                print("  %s mumu=%s |dphi|<%g: %d / %d = %.4f" % (k, mm, args.cut, npass, len(x), npass / len(x)))
                if mm == "pca":
                    h[k], frac[k] = th1("h_%s_%s" % (k, tag), x), (npass, len(x))
        head = ["%s: %d pairs (%d via fit fallback)" % (label, n, nfb), cuts, "p_{T}^{#mu#mu} from tracks at PCA to vertex"]
        for k in keys:
            c = canvas("c_%s_%s" % (k, tag))
            hk = h[k]; hk.SetLineColor(color[k]); hk.SetLineWidth(3)
            frame(hk, xtitle[k], "Pairs / 0.05")
            hk.Draw("hist")
            extra = ["%s, N = %d" % (legend[k], frac[k][1]),
                     "|#Delta#phi| < %g: %.1f%%" % (args.cut, 100.0 * frac[k][0] / frac[k][1])]
            labels(c, head + extra)
            save(c, [hk], "dphi_%s_%s" % (k, tag), 1.45, 0.5); count += 1
        for norm in (False, True):
            c = canvas("c_overlay_%s_%d" % (tag, norm))
            hs = [h[k].Clone("o_%s_%s_%d" % (k, tag, norm)) for k in keys]
            for k, hh in zip(keys, hs):
                hh.SetLineColor(color[k]); hh.SetLineWidth(3); hh.SetLineStyle(style[k])
                if norm: hh.Scale(1.0 / frac[k][1])
            frame(hs[0], xtitle["overlay"], "Fraction of pairs / 0.05" if norm else "Pairs / 0.05")
            hs[0].Draw("hist")
            for hh in hs[1:]: hh.Draw("hist same")
            leg = ROOT.TLegend(0.40, 0.50, 0.93, 0.70); keep.append(leg)
            leg.SetBorderSize(0); leg.SetFillStyle(0); leg.SetTextSize(0.031)
            leg.SetHeader("X  (fraction |#Delta#phi| < %g)" % args.cut)
            for k, hh in zip(keys, hs):
                leg.AddEntry(hh, "%s  (%.1f%%)" % (legend[k], 100.0 * frac[k][0] / frac[k][1]), "l")
            leg.Draw()
            labels(c, head + (["Each unit area"] if norm else []))
            save(c, hs, "overlay_dphi_%s%s" % (tag, "_unitarea" if norm else ""), 1.9, 0.5 / n if norm else 0.5); count += 1
    print("PLOTTED %d linear PNGs" % count + (" and %d log copies" % count if args.log else ""))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--input-dir", required=True); a.add_argument("--report-dir", required=True)
    a.add_argument("--output", required=True); a.add_argument("--revision", required=True)
    a.add_argument("--expected-samples", type=int, default=72); a.add_argument("--expected-events", type=int, default=720000)
    b = sub.add_parser("plot")
    b.add_argument("--output", required=True); b.add_argument("--plot-dir", required=True)
    b.add_argument("--log", action="store_true")
    b.add_argument("--dr-only", action="store_true", help="dR only: 0.1 bins/ticks to 3.1, last bin ends at 3.14")
    c = sub.add_parser("plot-dr", help="dR 0.05 bins to 3.15: gen, reco, overlays, residual; all and DSA-DSA")
    c.add_argument("--output", required=True); c.add_argument("--plot-dir", required=True)
    c.add_argument("--log", action="store_true")
    c.add_argument("--cms", action="store_true", help="PyROOT, CMS style")
    c.add_argument("--max-chi2ndof", type=float, help="--cms only: keep pairs with hf vertex chi2/ndof < this")
    c.add_argument("--input-dir", help="per-sample inputs (needed for --max-chi2ndof)")
    d = sub.add_parser("plot-dphi", help="|dphi(MET or chi1chi1, reco mumu at PCA)|, CMS style; all and DSA-DSA")
    d.add_argument("--output", required=True, help="analyze output (events tree)")
    d.add_argument("--input-dir", required=True, help="per-sample inputs of the same run (MET, chi, PCA pT branches)")
    d.add_argument("--plot-dir", required=True); d.add_argument("--log", action="store_true")
    d.add_argument("--max-chi2ndof", type=float, default=4.0); d.add_argument("--max-dr", type=float, default=0.9)
    d.add_argument("--cut", type=float, default=0.5, help="|dphi| threshold for the printed fractions")
    args = p.parse_args()
    if args.cmd == "plot-dr" and args.max_chi2ndof is not None and not (args.cms and args.input_dir):
        p.error("--max-chi2ndof needs --cms and --input-dir")
    if args.cmd == "plot-dr" and args.cms:
        plot_dr_cms(args)
    else:
        {"analyze": analyze, "plot": plot, "plot-dr": plot_dr, "plot-dphi": plot_dphi}[args.cmd](args)


if __name__ == "__main__":
    main()
