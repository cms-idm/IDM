#!/usr/bin/env python3
"""Talk-oriented summary of the typed pre-selected vertexing run (no new MINIAOD processing).

python3 preselected_vertexing_talk.py analyze --input-dir RUN/inputs --output RUN/talk.root --revision <sha>
python3 preselected_vertexing_talk.py plot --output RUN/talk.root --plot-dir RUN/plots_talk
Narrative: accepted events -> both muons matched (typed) -> vertexed -> vertex quality. Study note:
kb/studies/preselected-vertexing-efficiency.md. Reads the per-sample trees written by PreselectedVertexing.
Event level, raw, unweighted. den = events with two accepted gen. Ordered categories use the muon CHARGE:
GmOp = global leg for mu-, DSA (outer) leg for mu+. "any" = union of all typed combos (any component).
"h" = staged hierarchy best-best (comparison). Quality: valid typed fits, six unordered sets, one entry per valid
ordered fit; reco vxy = hypot(fit vx, vy) from the origin, truth = gen production Lxy from the origin.
"""
import argparse
from array import array
from pathlib import Path
import re
import numpy as np

LET = {2: "G", 1: "O", 0: "I"}
NAME = {"G": "Global", "O": "DSA", "I": "Inner"}
PURE = ("GG", "II", "OO")
MIXED = ("GmOp", "OmGp", "GmIp", "ImGp", "OmIp", "ImOp")
CATS = PURE + MIXED
UNORD = {"GG": ((2, 2),), "II": ((0, 0),), "OO": ((1, 1),), "GO": ((2, 1), (1, 2)), "GI": ((2, 0), (0, 2)),
         "OI": ((1, 0), (0, 1))}
TRK_SETS, DSA_SETS = ("GG", "II", "GI"), ("OO", "GO", "OI")
EFF_EDGES = {"d0": np.arange(0, 16, 1.0), "pt": np.arange(0, 51, 1.0), "lxy": np.arange(0, 31, 1.0),
             "eta": np.linspace(0, 2.4, 13)}
EFF_LABELS = {"d0": r"Max gen $|d_0|$ to origin [cm]", "pt": r"Gen dimuon $p_T$ [GeV]",
              "lxy": r"Gen production $L_{xy}$ [cm]", "eta": r"Max gen $|\eta|$"}
Q_EDGES = {"chi2": np.arange(0, 10.01, .25),
           "dvxy_fine": np.linspace(-0.2, 0.2, 41), "dvxy_wide": np.linspace(-100, 300, 81),
           "dvxyrel_fine": np.linspace(-0.1, 0.1, 41), "dvxyrel_wide": np.linspace(-1, 10, 45),
           "dpt_fine": np.linspace(-3, 3, 61), "dpt_wide": np.linspace(-30, 30, 61),
           "dptrel_fine": np.linspace(-0.1, 0.1, 41), "dptrel_wide": np.linspace(-1, 1, 41)}


def label(c):
    if len(c) == 2:
        return NAME[c[0]] + "–" + NAME[c[1]]
    return "%s(μ⁻)–%s(μ⁺)" % (NAME[c[0]], NAME[c[2]])


def key_for(a, b, q0):
    tm, tp = (a, b) if q0 < 0 else (b, a)
    return LET[tm] * 2 if tm == tp else LET[tm] + "m" + LET[tp] + "p"


def flowcounts(values, edges):
    return np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1)


def write_hist(name, title, edges, counts):
    import ROOT
    h = ROOT.TH1D(name, title, len(edges) - 1, array("d", edges))
    h.Sumw2()
    for b, c in enumerate(counts):
        h.SetBinContent(b, float(c)); h.SetBinError(b, float(np.sqrt(c)))
    h.SetEntries(float(np.sum(counts)))
    h.Write()


def load(args):
    import awkward as ak
    import uproot
    paths = sorted(Path(args.input_dir).glob("*.root"))
    if len(paths) != args.expected_samples:
        raise RuntimeError("Unexpected number of inputs: %d" % len(paths))
    names, ev, cb, base, revs = [], [], [], 0, set()
    for si, path in enumerate(paths):
        with uproot.open(path) as f:
            sample = str(f["preselectedVertexing/sample"].member("fTitle"))
            if sample in names or not re.fullmatch(r"Mchi-[\dp]+_dMchi-[\dp]+_ctau-[\dp]+", sample):
                raise RuntimeError("Bad/duplicate sample " + sample)
            names.append(sample)
            revs.add(str(f["preselectedVertexing/revision"].member("fTitle")))
            a = f["preselectedVertexing/events"].arrays(
                ["twoAcc", "bbStatus", "sigQ", "sigD0", "sigLxy", "sigEta", "genMuMuPt"], library="ak")
            n = len(a["twoAcc"])
            two = np.asarray(a["twoAcc"])
            sq = a["sigQ"]
            sig2 = np.asarray((ak.num(sq) == 2) & (ak.fill_none(ak.prod(sq, axis=1, mask_identity=False), 0) < 0))
            def pad(x):
                return ak.to_numpy(ak.pad_none(x, 2, clip=True))
            q = pad(sq).astype(float)
            d0 = np.nanmax(pad(a["sigD0"]).astype(float), axis=1)
            eta = np.nanmax(np.abs(pad(a["sigEta"]).astype(float)), axis=1)
            lxy = pad(a["sigLxy"]).astype(float)[:, 0]
            ev.append(dict(sample=np.full(n, si), two=two, sig2=sig2, hm=np.asarray(a["bbStatus"]) > 0,
                           hv=np.asarray(a["bbStatus"]) == 3, q0=q[:, 0], d0=d0, eta=eta, lxy=lxy,
                           pt=np.asarray(a["genMuMuPt"]).astype(float)))
            k = f["preselectedVertexing/combos"].arrays(library="np")
            k["kEvent"] = k["kEvent"] + base
            cb.append(k)
            base += n
    e = {key: np.concatenate([x[key] for x in ev]) for key in ev[0]}
    k = {key: np.concatenate([x[key] for x in cb]) for key in cb[0]}
    if len(revs) != 1 or "unspecified" in revs:
        raise RuntimeError("Mixed/unspecified revision")
    return names, e, k, next(iter(revs))


def analyze(args):
    import ROOT
    if Path(args.output).exists():
        raise RuntimeError("Refusing to overwrite ROOT output")
    names, e, k, rev = load(args)
    total = len(e["two"])
    if args.expected_events and total != args.expected_events:
        raise RuntimeError("Event total mismatch")
    if not np.all(e["two"][k["kEvent"]]):
        raise RuntimeError("combo in non-accepted event")
    ke = k["kEvent"]
    valid = k["kStatus"] == 3
    keys = np.array([key_for(int(a), int(b), e["q0"][i]) for a, b, i in zip(k["kTypeA"], k["kTypeB"], ke)])
    masks = {"den": e["two"], "matched_any": np.zeros(total, bool), "valid_any": np.zeros(total, bool),
             "matched_h": e["hm"], "valid_h": e["hv"]}
    masks["matched_any"][ke] = True
    masks["valid_any"][ke[valid]] = True
    for c in CATS:
        m = np.zeros(total, bool); v = np.zeros(total, bool)
        sel = keys == c
        m[ke[sel]] = True; v[ke[sel & valid]] = True
        masks["m_" + c] = m; masks["v_" + c] = v
        if np.any(v & ~m):
            raise RuntimeError("valid outside matched: " + c)
    if np.any(masks["valid_any"] & ~masks["matched_any"]):
        raise RuntimeError("valid_any outside matched_any")
    # quality entries (valid typed fits)
    rv = np.hypot(k["kVx"], k["kVy"])
    truth = e["lxy"][ke]
    q = {"chi2": k["kChi2"], "dvxy": rv - truth, "dpt": k["kRecoMuMuPt"] - e["pt"][ke]}
    with np.errstate(divide="ignore", invalid="ignore"):
        q["dvxyrel"] = np.where(truth > 1.0, (rv - truth) / truth, np.nan)
        q["dptrel"] = (k["kRecoMuMuPt"] - e["pt"][ke]) / e["pt"][ke]
    uk = {}
    for u, orders in UNORD.items():
        sel = np.zeros(len(ke), bool)
        for a, b in orders:
            sel |= (k["kTypeA"] == a) & (k["kTypeB"] == b)
        uk[u] = sel & valid
    groups = {"overall": None}
    for si, s in enumerate(names):
        mass, tau = s.rsplit("_ctau-", 1)
        groups.setdefault("mass_" + mass, set()).add(si)
        groups.setdefault("ctau_" + tau, set()).add(si)
    root = ROOT.TFile.Open(str(args.output), "CREATE")
    ROOT.TNamed("definitions", "Typed ordered categories by muon charge (GmOp = global for mu-, DSA for mu+); any = union of "
                "typed combos; h = hierarchy; den = events with two accepted gen; quality = valid typed fits, six unordered "
                "sets, one entry per valid ordered fit; vxy from origin vs gen production Lxy; relative vxy for truth vxy>1cm. "
                "Raw unweighted; flow kept.").Write()
    ROOT.TNamed("analysisRevision", args.revision).Write()
    ROOT.TNamed("inputRevision", rev).Write()
    ROOT.TNamed("samples", "\n".join(names)).Write()
    top = root.mkdir("groups")
    ev_pass = {"initial": np.ones(total, bool), "sig2": e["sig2"]}
    for gname, ids in groups.items():
        d = top.mkdir(gname); d.cd()
        gs = np.ones(total, bool) if ids is None else np.isin(e["sample"], list(ids))
        stages = [ev_pass["initial"], ev_pass["sig2"], masks["den"], masks["matched_any"], masks["valid_any"],
                  masks["matched_h"], masks["valid_h"]]
        write_hist("cutflow", "cutflow", np.arange(len(stages) + 1, dtype=float),
                   np.r_[0, [int((m & gs).sum()) for m in stages], 0])
        for var, edges in EFF_EDGES.items():
            for mname, m in masks.items():
                write_hist("%s__%s" % (mname, var), EFF_LABELS[var], edges, flowcounts(e[var][m & gs], edges))
        ks = np.ones(len(ke), bool) if ids is None else np.isin(e["sample"][ke], list(ids))
        for u in UNORD:
            sel = uk[u] & ks
            write_hist("%s__chi2" % u, "chi2", Q_EDGES["chi2"], flowcounts(q["chi2"][sel], Q_EDGES["chi2"]))
            for var in ("dvxy", "dvxyrel", "dpt", "dptrel"):
                ok = sel & np.isfinite(q[var])
                for rng in ("fine", "wide"):
                    ed = Q_EDGES[var + "_" + rng]
                    write_hist("%s__%s_%s" % (u, var, rng), var, ed, flowcounts(q[var][ok], ed))
    root.Close()
    chk = ROOT.TFile.Open(str(args.output))
    for mname, m in masks.items():
        hh = chk.Get("groups/overall/%s__d0" % mname)
        if int(hh.Integral(0, hh.GetNbinsX() + 1)) != int(m.sum()):
            raise RuntimeError("Histogram total differs: " + mname)
    for u in UNORD:
        hh = chk.Get("groups/overall/%s__chi2" % u)
        if int(hh.Integral(0, hh.GetNbinsX() + 1)) != int(uk[u].sum()):
            raise RuntimeError("Quality histogram total differs: " + u)
    chk.Close()
    c = lambda m: int(m.sum())
    print("VALIDATED: events=%d" % total)
    print("initial %d, two signal %d, accepted %d" % (total, c(e["sig2"]), c(masks["den"])))
    print("matched any(union) %d  hierarchy %d | valid any %d  hierarchy %d | hier-matched not in union %d | hier-valid not in any-valid %d | any-valid not hier-valid %d" % (
        c(masks["matched_any"]), c(masks["matched_h"]), c(masks["valid_any"]), c(masks["valid_h"]),
        c(masks["matched_h"] & ~masks["matched_any"]), c(masks["valid_h"] & ~masks["valid_any"]),
        c(masks["valid_any"] & ~masks["valid_h"])))
    print("category matched valid valid/matched matched/accepted valid/anymatched")
    for cat in CATS:
        m, v = c(masks["m_" + cat]), c(masks["v_" + cat])
        print("  %-5s %7d %7d %.4f %.4f %.4f" % (cat, m, v, v / max(m, 1), m / c(masks["den"]), v / c(masks["matched_any"])))
    print("quality (valid fits): set N | dvxy[cm] median, 16-84 | dvxyrel median, 16-84 | dpt[GeV] median, 16-84 | dptrel median, 16-84")
    for u in UNORD:
        sel = uk[u]
        row = []
        for var in ("dvxy", "dvxyrel", "dpt", "dptrel"):
            x = q[var][sel & np.isfinite(q[var])]
            p16, p50, p84 = np.percentile(x, [16, 50, 84])
            row.append("%.4g [%.4g, %.4g]" % (p50, p16, p84))
        print("  %-3s %7d | %s" % (u, c(sel), " | ".join(row)))


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
    foot = ("Signal MC (2022EE), pre-selected displacedMuons; events with exactly 2 signal muons, both accepted (|η|<2.4, "
            "Lxy<740 cm, no pT cut); raw unweighted; 68% Clopper–Pearson")

    def h(name, group="overall"):
        o = f["groups/%s/%s" % (group, name)]
        return o.values(flow=True), o.axis().edges()

    def finish(fig, path, bottom=0.02):
        fig.text(0.01, 0.005, foot, fontsize=7, color="0.35")
        fig.tight_layout(rect=(0, bottom, 1, 0.97))
        fig.savefig(path, dpi=120)
        if args.log and path.name.startswith("slide4"):  # extra log-y copy of the distributions only
            for a_ in fig.axes:
                a_.set_yscale("log")
            fig.savefig(path.with_name(path.stem + "_log.png"), dpi=120)
        plt.close(fig)

    tab = plt.get_cmap("tab10")
    style = {"any": ("any component (union of typed)", "k", "-", "o"), "h": ("hierarchy best–best (one-to-one)", "0.5", ":", "s")}
    for i, c in enumerate(PURE):
        style[c] = (label(c), tab(i), "-", "o")
    for i, pair in enumerate(("GO", "GI", "OI")):
        for c, ls in zip(MIXED[2 * i:2 * i + 2], ("-", "--")):
            style[c] = (label(c), tab(3 + i), ls, "^" if ls == "-" else "v")
    order = ("any", "h") + CATS

    def series(num, den):
        """num, den: functions key -> hist name; returns key -> (total ratio, lo, hi, var curves)."""
        res = {}
        for key in order:
            n_name, d_name = num(key), den(key)
            if n_name is None:
                continue
            nv, _ = h(n_name + "__d0"); dv, _ = h(d_name + "__d0")
            n, d = nv.sum(), dv.sum()
            lo, hi = cp(np.array(n), np.array(d))
            curves = {}
            for var in EFF_EDGES:
                a, edges = h("%s__%s" % (n_name, var)); b, _ = h("%s__%s" % (d_name, var))
                a, b = a[1:-1], b[1:-1]
                with np.errstate(divide="ignore", invalid="ignore"):
                    r = np.where(b > 0, a / b, np.nan)
                l, u = cp(a, b)
                curves[var] = (edges, r, l, u, b)
            res[key] = (n / d if d else np.nan, float(lo), float(hi), curves, int(n), int(d))
        return res

    def slide(res, title, ylabel, fname):
        fig = plt.figure(figsize=(18, 8.2))
        gs = fig.add_gridspec(2, 4)
        axb = fig.add_subplot(gs[:, :2])
        keys = list(res)
        xs = np.arange(len(keys))
        for x, key in zip(xs, keys):
            r, lo, hi, _, n, d = res[key]
            lab, col, ls, mk = style[key]
            axb.bar(x, r, color=col, hatch="//" if ls == "--" else None, alpha=.85, edgecolor="white")
            axb.errorbar(x, r, yerr=[[r - lo], [hi - r]], color="k", lw=1)
            axb.text(x, hi + 0.01, "%.1f%%" % (100 * r), ha="center", fontsize=8)
        axb.set_xticks(xs); axb.set_xticklabels([style[k][0] for k in keys], rotation=60, ha="right", fontsize=8)
        axb.set_ylabel(ylabel); axb.set_ylim(0, min(1.12, max(0.2, max(res[k][2] for k in keys) * 1.15)))
        axb.set_title("Overall", fontsize=10); axb.grid(axis="y", alpha=.3)
        for ax, var in zip((fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 3]), fig.add_subplot(gs[1, 2]),
                            fig.add_subplot(gs[1, 3])), EFF_EDGES):
            for key in keys:
                lab, col, ls, mk = style[key]
                edges, r, l, u, b = res[key][3][var]
                x = 0.5 * (edges[1:] + edges[:-1]); ok = b > 0
                ax.errorbar(x[ok], r[ok], yerr=[(r - l)[ok], (u - r)[ok]], color=col, ls=ls, marker=mk, ms=3, lw=1,
                            label=lab, alpha=.9)
            ax.set_xlabel(EFF_LABELS[var]); ax.set_ylabel(ylabel); ax.grid(alpha=.3); ax.set_ylim(0, 1.05)
        hd, lb = fig.axes[1].get_legend_handles_labels()
        fig.legend(hd, lb, loc="lower center", ncol=6, fontsize=7.5, bbox_to_anchor=(0.5, 0.025))
        fig.suptitle(title, fontsize=12)
        finish(fig, out / fname, bottom=0.1)

    A = lambda key: {"any": "matched_any", "h": "matched_h"}.get(key, "m_" + key)
    V = lambda key: {"any": "valid_any", "h": "valid_h"}.get(key, "v_" + key)
    # Slide 1: matched / accepted
    slide(series(A, lambda k: "den"), "1. Both muons reconstructed (matched) per leg-type set, of events with two accepted gen",
          "Fraction of accepted events", "slide1_matched_over_accepted.png")
    # Slide 2: valid / any-matched (hierarchy uses its own matched count)
    slide(series(V, lambda k: "matched_h" if k == "h" else "matched_any"),
          "2. Of events with both muons matched (any component), fraction vertexed with each leg-type set "
          "(includes the probability of having those legs)", "Fraction of both-matched events", "slide2_vertexed_over_matched_any.png")
    # Slide 3: valid / own matched
    slide(series(V, lambda k: A(k)), "3. Given the typed legs exist, vertexing efficiency (own denominator per set)",
          "Vertexed / matched in the same set", "slide3_vertexed_over_own_matched.png")

    # Cutflow
    v, _ = h("cutflow"); v = v[1:-1]
    labels = ("initial events", "two signal muons (μ⁻μ⁺)", "both accepted", "both matched, any component (union)",
              "vertexed, any component (union)", "both matched, hierarchy", "vertexed, hierarchy")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    y = np.arange(len(labels))[::-1]
    for yi, c, col, al in zip(y, v, ["0.6", "0.6", "C0", "C1", "C2", "C1", "C2"], [1, 1, 1, 1, 1, .45, .45]):
        ax.barh(yi, c, color=col, alpha=al)
    for yi, c in zip(y, v):
        ax.text(c, yi, "  %d  (%.2f%% of initial, %.2f%% of accepted)" % (c, 100 * c / v[0], 100 * c / v[2]), va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.set_xlim(0, v.max() * 1.8); ax.set_xlabel("Events (raw)")
    ax.set_title("Cutflow (720000 initial events, 10000 per sample)", fontsize=10)
    finish(fig, out / "slide0_cutflow.png")

    # Slide 4: quality
    def qpanels(var_fine, var_wide, xlabel_f, xlabel_w, fname, title, extra=None):
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
        for a, sets, var, xl, rng in ((ax[0], TRK_SETS, var_fine, xlabel_f, "fine"), (ax[1], DSA_SETS, var_wide, xlabel_w, "wide")):
            for i, u in enumerate(sets):
                v, e_ = h("%s__%s_%s" % (u, var, rng))
                n = v.sum()
                if n == 0:
                    continue
                a.stairs(v[1:-1] / n, e_, label="%s (N=%d, %.1f%% outside)" % (label(u), n, 100 * (v[0] + v[-1]) / n))
            a.set_xlabel(xl); a.set_ylabel("Unit area (incl. outside range)"); a.legend(fontsize=7, loc="upper left"); a.grid(alpha=.3)
        ax[0].set_title("Tracker-leg sets (no DSA leg): narrow range", fontsize=9)
        ax[1].set_title("DSA-leg sets: wide range", fontsize=9)
        fig.suptitle(title, fontsize=11)
        finish(fig, out / fname)

    fig, ax = plt.subplots(figsize=(7, 4.8))
    for i, u in enumerate(UNORD):
        v, e_ = h("%s__chi2" % u); n = v.sum()
        ax.stairs(v[1:-1] / n, e_, label="%s (N=%d)" % (label(u), n), color=tab(i))
    ax.set_xlabel(r"Vertex $\chi^2$/ndof (ndof = 1)"); ax.set_ylabel("Unit area (incl. > 10)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax.set_title("4a. Vertex quality: χ²/dof of valid typed fits", fontsize=10)
    finish(fig, out / "slide4a_chi2.png")
    qpanels("dvxy", "dvxy", r"$v_{xy}^{reco} - v_{xy}^{gen}$ [cm]", r"$v_{xy}^{reco} - v_{xy}^{gen}$ [cm]",
            "slide4b_vxy_absolute.png", "4b. Vertex vxy resolution, absolute (reco fit radius − gen production radius, from the origin)")
    qpanels("dvxyrel", "dvxyrel", r"$(v_{xy}^{reco} - v_{xy}^{gen})/v_{xy}^{gen}$  (gen $v_{xy}$ > 1 cm)",
            r"$(v_{xy}^{reco} - v_{xy}^{gen})/v_{xy}^{gen}$  (gen $v_{xy}$ > 1 cm)", "slide4c_vxy_relative.png",
            "4c. Vertex vxy resolution, relative (gen vxy > 1 cm)")
    qpanels("dpt", "dpt", r"$p_T^{\mu\mu,reco} - p_T^{\mu\mu,gen}$ [GeV]", r"$p_T^{\mu\mu,reco} - p_T^{\mu\mu,gen}$ [GeV]",
            "slide4d_dimuon_pt_absolute.png", "4d. Dimuon pT resolution, absolute (sum of the two legs' track pT at their reference points)")
    qpanels("dptrel", "dptrel", r"$(p_T^{reco} - p_T^{gen})/p_T^{gen}$", r"$(p_T^{reco} - p_T^{gen})/p_T^{gen}$",
            "slide4e_dimuon_pt_relative.png", "4e. Dimuon pT resolution, relative")
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
    ap.add_argument("--log", action="store_true", help="add log-y copies of the slide-4 distributions")
    args = ap.parse_args()
    analyze(args) if args.mode == "analyze" else plot(args)


if __name__ == "__main__":
    main()
