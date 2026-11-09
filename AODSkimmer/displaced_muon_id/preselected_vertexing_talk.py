#!/usr/bin/env python3
"""Talk-oriented summary of the typed pre-selected vertexing run (no new MINIAOD processing).

python3 preselected_vertexing_talk.py analyze --input-dir RUN/inputs --output RUN/talk.root --revision <sha>
python3 preselected_vertexing_talk.py plot --output RUN/talk.root --plot-dir RUN/plots_talk
Narrative: accepted events -> both muons matched (typed) -> vertexed -> vertex quality. Study note:
kb/studies/preselected-vertexing-efficiency.md. Reads the per-sample trees written by PreselectedVertexing.
Event level, raw, unweighted. den = events with two accepted gen. Ordered categories use the muon CHARGE:
GmOp = global leg for mu-, DSA (outer) leg for mu+. "any" = union of all typed combos (any component).
"P1"/"P2" = vertex-level leg priority per muon among its matched component types: P1 Global>Inner>DSA, P2 Global>DSA>Inner
(matching itself is priority independent: a muon is matched if ANY component matches). "h" (staged hierarchy) is stored in ROOT only. Quality: valid typed fits, six unordered sets, one entry per valid
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
# Extended x variables (Priority 2, 2026-10-21): full-acceptance Lxy, 0-10 GeV zooms, leading/subleading gen muon pT.
# Each maps to its source array (dimuon pT = |vector sum| of the two gen signal muon pT; lead/sub = max/min gen muon pT).
EXT_EDGES = {"lxyfull": np.arange(0, 741, 10.0), "ptlow": np.arange(0, 10.01, 0.25),
             "lead": np.arange(0, 51, 1.0), "sub": np.arange(0, 51, 1.0),
             "leadlow": np.arange(0, 10.01, 0.25), "sublow": np.arange(0, 10.01, 0.25)}
EXT_SRC = {"lxyfull": "lxy", "ptlow": "pt", "lead": "lead", "sub": "sub", "leadlow": "lead", "sublow": "sub"}
EXT_LABELS = {"lxyfull": r"Gen production $L_{xy}$ [cm] (full acceptance)", "ptlow": r"Gen dimuon $p_T$ [GeV] (0–10 zoom)",
              "lead": r"Leading gen muon $p_T$ [GeV]", "sub": r"Subleading gen muon $p_T$ [GeV]",
              "leadlow": r"Leading gen muon $p_T$ [GeV] (0–10 zoom)", "sublow": r"Subleading gen muon $p_T$ [GeV] (0–10 zoom)"}
ALL_EDGES = {**EFF_EDGES, **EXT_EDGES}
ALL_LABELS = {**EFF_LABELS, **EXT_LABELS}
XSRC = {**{v: v for v in EFF_EDGES}, **EXT_SRC}
# Dip decomposition: gen opening angle classes (genDR from the gen momenta at production), dimuon pT axes
DR_CLASSES = ((0.0, 0.3), (0.3, 1.0), (1.0, np.inf))
DR_LABELS = (r"$\Delta R_{gen}<0.3$", r"$0.3\leq\Delta R_{gen}<1$", r"$\Delta R_{gen}\geq 1$")
# Colour-blind-safe palette (Okabe-Ito based). Sets whose curves nearly coincide also differ in line width and marker.
SETCOL = {"GG": "#0072B2", "II": "#E69F00", "OO": "#009E73", "GO": "#D55E00", "GI": "#CC79A7", "OI": "#6A3D9A"}
SETLW = {"GG": 3.4, "II": 2.2, "GI": 1.1, "OO": 3.4, "GO": 2.2, "OI": 1.1}
SETMK = {"GG": "o", "II": "s", "GI": "^", "OO": "o", "GO": "s", "OI": "^"}
QMETRICS = ("dvxy", "dvxyrel", "dpt", "dptrel")
Q_NMIN = 100  # minimum fits per bin for a quantile
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
                ["twoAcc", "bbStatus", "sigQ", "sigD0", "sigLxy", "sigEta", "sigPt", "genMuMuPt", "genDR"], library="ak")
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
            spt = pad(a["sigPt"]).astype(float)
            ev.append(dict(sample=np.full(n, si), two=two, sig2=sig2, hm=np.asarray(a["bbStatus"]) > 0,
                           hv=np.asarray(a["bbStatus"]) == 3, q0=q[:, 0], d0=d0, eta=eta, lxy=lxy,
                           pt=np.asarray(a["genMuMuPt"]).astype(float), lead=np.nanmax(spt, axis=1),
                           sub=np.nanmin(spt, axis=1), dr=np.asarray(a["genDR"]).astype(float)))
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
    # vertex-level leg priority per muon: choose, for each gen muon, its best matched component type; fit that typed combo
    RANKS = {"P1": {2: 0, 0: 1, 1: 2}, "P2": {2: 0, 1: 1, 0: 2}}
    for pname, rk in RANKS.items():
        lut = np.array([rk[t] for t in (0, 1, 2)])
        ra, rb = lut[k["kTypeA"]], lut[k["kTypeB"]]
        mina = np.full(total, 9); minb = np.full(total, 9)
        np.minimum.at(mina, ke, ra); np.minimum.at(minb, ke, rb)
        chosen = (ra == mina[ke]) & (rb == minb[ke])
        if not np.array_equal(np.bincount(ke[chosen], minlength=total) > 0, masks["matched_any"]) or \
           np.bincount(ke[chosen], minlength=total).max() != 1:
            raise RuntimeError("priority selection is not one combo per matched event: " + pname)
        uk[pname] = chosen & valid
        v_ev = np.zeros(total, bool); v_ev[ke[chosen & valid]] = True
        masks["v_" + pname] = v_ev
    groups = {"overall": None}
    for si, s in enumerate(names):
        mass, tau = s.rsplit("_ctau-", 1)
        groups.setdefault("mass_" + mass, set()).add(si)
        groups.setdefault("ctau_" + tau, set()).add(si)
    root = ROOT.TFile.Open(str(args.output), "CREATE")
    ROOT.TNamed("definitions", "Typed ordered categories by muon charge (GmOp = global for mu-, DSA for mu+); any = union of "
                "typed combos; P1/P2 = per-muon vertex leg priority Global>Inner>DSA / Global>DSA>Inner; den = events with two accepted gen; quality = valid typed fits, six unordered "
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
        stages = [ev_pass["initial"], ev_pass["sig2"], masks["den"], masks["matched_any"], masks["v_P1"], masks["v_P2"],
                  masks["valid_any"]]
        write_hist("cutflow", "cutflow", np.arange(len(stages) + 1, dtype=float),
                   np.r_[0, [int((m & gs).sum()) for m in stages], 0])
        for var, edges in ALL_EDGES.items():
            for mname, m in masks.items():
                write_hist("%s__%s" % (mname, var), ALL_LABELS[var], edges, flowcounts(e[XSRC[var]][m & gs], edges))
        if gname == "overall":  # dip decomposition: every event-level mask versus dimuon pT, per gen opening-angle class
            for ci, (lo_, hi_) in enumerate(DR_CLASSES):
                dsel = (e["dr"] >= lo_) & (e["dr"] < hi_)
                for var in ("pt", "ptlow"):
                    for mname, m in masks.items():
                        write_hist("dr%d__%s__%s" % (ci, mname, var), ALL_LABELS[var], ALL_EDGES[var],
                                   flowcounts(e[XSRC[var]][m & dsel], ALL_EDGES[var]))
        ks = np.ones(len(ke), bool) if ids is None else np.isin(e["sample"][ke], list(ids))
        for u in uk:
            sel = uk[u] & ks
            write_hist("%s__chi2" % u, "chi2", Q_EDGES["chi2"], flowcounts(q["chi2"][sel], Q_EDGES["chi2"]))
            for var in ("dvxy", "dvxyrel", "dpt", "dptrel"):
                ok = sel & np.isfinite(q[var])
                for rng in ("fine", "wide"):
                    ed = Q_EDGES[var + "_" + rng]
                    write_hist("%s__%s_%s" % (u, var, rng), var, ed, flowcounts(q[var][ok], ed))
    root.Close()
    # 90% interval (5th, 50th, 95th percentiles) of each quality metric versus each plotting variable, per set and group
    qroot = ROOT.TFile.Open(str(args.output), "UPDATE")
    qtop = qroot.mkdir("quantiles")
    qgroups = {g: ids for g, ids in groups.items() if g == "overall" or g.startswith("ctau_")}
    for gname, ids in qgroups.items():
        d = qtop.mkdir(gname); d.cd()
        ks = np.ones(len(ke), bool) if ids is None else np.isin(e["sample"][ke], list(ids))
        for u in uk:
            sel0 = uk[u] & ks
            for var in QMETRICS:
                ok = sel0 & np.isfinite(q[var])
                for xv, edges in ALL_EDGES.items():
                    nb = len(edges) - 1
                    bins = np.searchsorted(edges, e[XSRC[xv]][ke][ok], side="right") - 1
                    vals_ok = q[var][ok]
                    qs = np.zeros((3, nb)); ns = np.zeros(nb)
                    order_ = np.argsort(bins, kind="stable")
                    bounds = np.searchsorted(bins[order_], np.arange(nb + 1))
                    for b in range(nb):
                        vv = vals_ok[order_[bounds[b]:bounds[b + 1]]]
                        ns[b] = len(vv)
                        if len(vv) >= Q_NMIN:
                            qs[:, b] = np.percentile(vv, [5, 50, 95])
                    for name, vals in (("q05", qs[0]), ("q50", qs[1]), ("q95", qs[2]), ("n", ns)):
                        hq = ROOT.TH1D("%s__%s__%s__%s" % (name, u, var, xv), name, nb, array("d", edges))
                        for b in range(nb):
                            hq.SetBinContent(b + 1, float(vals[b]))
                        hq.Write()
    qroot.Close()
    chk = ROOT.TFile.Open(str(args.output))
    for mname, m in masks.items():
        hh = chk.Get("groups/overall/%s__d0" % mname)
        if int(hh.Integral(0, hh.GetNbinsX() + 1)) != int(m.sum()):
            raise RuntimeError("Histogram total differs: " + mname)
    for u in uk:
        hh = chk.Get("groups/overall/%s__chi2" % u)
        if int(hh.Integral(0, hh.GetNbinsX() + 1)) != int(uk[u].sum()):
            raise RuntimeError("Quality histogram total differs: " + u)
    chk.Close()
    c = lambda m: int(m.sum())
    print("VALIDATED: events=%d" % total)
    print("initial %d, two signal %d, accepted %d" % (total, c(e["sig2"]), c(masks["den"])))
    print("matched any(union) %d | vertexed: any valid combo %d, priority G>I>DSA %d, priority G>DSA>I %d | valid/matched: %.4f, %.4f, %.4f" % (
        c(masks["matched_any"]), c(masks["valid_any"]), c(masks["v_P1"]), c(masks["v_P2"]), c(masks["valid_any"]) / c(masks["matched_any"]),
        c(masks["v_P1"]) / c(masks["matched_any"]), c(masks["v_P2"]) / c(masks["matched_any"])))
    print("staged hierarchy (not plotted): matched %d valid %d" % (c(masks["matched_h"]), c(masks["valid_h"])))
    print("category matched valid valid/matched matched/accepted valid/anymatched")
    for cat in CATS:
        m, v = c(masks["m_" + cat]), c(masks["v_" + cat])
        print("  %-5s %7d %7d %.4f %.4f %.4f" % (cat, m, v, v / max(m, 1), m / c(masks["den"]), v / c(masks["matched_any"])))
    print("quality (valid fits): set N | dvxy[cm] median, 16-84 | dvxyrel median, 16-84 | dpt[GeV] median, 16-84 | dptrel median, 16-84")
    for u in uk:
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
    foot_eff = ("Signal MC (2022EE), pre-selected displacedMuons; events with exactly 2 signal muons, both accepted (|η|<2.4, "
                "Lxy<740 cm, |vz|<960 cm, no pT cut); raw unweighted; error bars: 68% Clopper–Pearson; bins with < 20 denominator events omitted")
    foot = "Signal MC (2022EE), pre-selected displacedMuons; events with two accepted signal muons; raw unweighted"

    def h(name, group="overall"):
        o = f["groups/%s/%s" % (group, name)]
        return o.values(flow=True), o.axis().edges()

    def finish(fig, path, bottom=0.02, note=None):
        fig.text(0.01, 0.005, note or foot, fontsize=7, color="0.35")
        fig.tight_layout(rect=(0, bottom, 1, 0.97))
        fig.savefig(path, dpi=120)
        if args.log and path.name.startswith("slide4"):  # extra log-y copy of the distributions only
            for a_ in fig.axes:
                a_.set_yscale("log")
            fig.savefig(path.with_name(path.stem + "_log.png"), dpi=120)
        plt.close(fig)

    tab = plt.get_cmap("tab10")
    style = {"any": ("any component (union of typed)", "k", "-", "o"), "P1": ("vertex priority Global>Inner>DSA", "#444444", ":", "s"),
             "P2": ("vertex priority Global>DSA>Inner", "#9a9a9a", "-.", "D")}
    for i, c in enumerate(PURE):
        style[c] = (label(c), SETCOL[c], "-", "o")
    for i, pair in enumerate(("GO", "GI", "OI")):
        for c, ls in zip(MIXED[2 * i:2 * i + 2], ("-", "--")):
            style[c] = (label(c), SETCOL[pair], ls, "^" if ls == "-" else "v")
    order = ("any", "P1", "P2") + CATS

    def series(num, den, keys=order):
        """num, den: functions key -> hist name; returns key -> (total ratio, lo, hi, var curves)."""
        res = {}
        for key in keys:
            n_name, d_name = num(key), den(key)
            if n_name is None:
                continue
            nv, _ = h(n_name + "__d0"); dv, _ = h(d_name + "__d0")
            n, d = nv.sum(), dv.sum()
            lo, hi = cp(np.array(n), np.array(d))
            curves = {}
            for var in ALL_EDGES:
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
                x = 0.5 * (edges[1:] + edges[:-1]); ok = b >= 20
                ax.errorbar(x[ok], r[ok], yerr=[(r - l)[ok], (u - r)[ok]], color=col, ls=ls, marker=mk, ms=3, lw=1,
                            label=lab, alpha=.9)
            ax.set_xlabel(EFF_LABELS[var]); ax.set_ylabel(ylabel); ax.grid(alpha=.3); ax.set_ylim(0, 1.05)
        hd, lb = fig.axes[1].get_legend_handles_labels()
        fig.legend(hd, lb, loc="lower center", ncol=6, fontsize=7.5, bbox_to_anchor=(0.5, 0.025))
        fig.suptitle(title, fontsize=12)
        finish(fig, out / fname, bottom=0.1, note=foot_eff)

    EXT_GRID = ("lxyfull", "ptlow", "lead", "sub", "leadlow", "sublow")

    def slide_ext(res, title, ylabel, fname):
        """Extended x variables of slides 1-3: full-acceptance Lxy, 0-10 GeV dimuon pT, leading/subleading pT (0-50 and 0-10)."""
        fig, axs = plt.subplots(2, 3, figsize=(18, 9.4))
        for ax, var in zip(axs.flat, EXT_GRID):
            for key in res:
                lab, col, ls, mk = style[key]
                edges, r, l, u, b = res[key][3][var]
                x = 0.5 * (edges[1:] + edges[:-1]); ok = b >= 20
                ax.errorbar(x[ok], r[ok], yerr=[(r - l)[ok], (u - r)[ok]], color=col, ls=ls, marker=mk, ms=3, lw=1,
                            label=lab, alpha=.9)
            ax.set_xlabel(ALL_LABELS[var]); ax.set_ylabel(ylabel); ax.grid(alpha=.3); ax.set_ylim(0, 1.05)
            ax.set_xlim(ALL_EDGES[var][0], ALL_EDGES[var][-1])
        hd, lb = axs[0, 0].get_legend_handles_labels()
        fig.legend(hd, lb, loc="lower center", ncol=6, fontsize=7.5, bbox_to_anchor=(0.5, 0.025))
        fig.suptitle(title, fontsize=12)
        finish(fig, out / fname, bottom=0.1, note=foot_eff)

    A1 = lambda key: {"any": "matched_any"}.get(key, "m_" + key)   # slide 1: matching is priority independent
    A3 = lambda key: "matched_any" if key in ("any", "P1", "P2") else "m_" + key
    V = lambda key: {"any": "valid_any", "P1": "v_P1", "P2": "v_P2"}.get(key, "v_" + key)
    # Slide 1: matched / accepted
    slide(series(A1, lambda k: "den", ("any",) + CATS), "1. Both muons reconstructed (matched) per leg-type set, of events with two accepted gen",
          "Fraction of accepted events", "slide1_matched_over_accepted.png")
    # Slide 2: valid / any-matched (hierarchy uses its own matched count)
    slide(series(V, lambda k: "matched_any"),
          "2. Of events with both muons matched (any component), fraction vertexed with each leg-type set "
          "(includes the probability of having those legs)", "Fraction of both-matched events", "slide2_vertexed_over_matched_any.png")
    # Slide 3: valid / own matched
    slide(series(V, A3), "3. Given the typed legs exist, vertexing efficiency (own denominator per set)",
          "Vertexed / matched in the same set", "slide3_vertexed_over_own_matched.png")
    # Extended versions of slides 1-3 (Priority 2)
    slide_ext(series(A1, lambda k: "den", ("any",) + CATS),
              "1x. Both muons matched / accepted: full-acceptance Lxy, low dimuon pT, leading and subleading gen muon pT",
              "Fraction of accepted events", "slide1x_matched_over_accepted_extended.png")
    slide_ext(series(V, lambda k: "matched_any"),
              "2x. Vertexed with each leg-type set / both matched (any component): full-acceptance Lxy, low dimuon pT, leading/subleading pT",
              "Fraction of both-matched events", "slide2x_vertexed_over_matched_any_extended.png")
    slide_ext(series(V, A3), "3x. Vertexed / matched in the same set: full-acceptance Lxy, low dimuon pT, leading/subleading pT",
              "Vertexed / matched in the same set", "slide3x_vertexed_over_own_matched_extended.png")

    # Slide 6: dimuon-pT dip decomposition by gen opening angle (vertexed / matched in the same set, and class mix)
    DIP_SETS = ("OO", "GmOp", "OmIp", "GG")
    dcol = ("#D55E00", "#0072B2", "#009E73")
    fig, axs = plt.subplots(3, len(DIP_SETS), figsize=(20, 12.5))
    for c_, key in enumerate(DIP_SETS):
        for r_, var in enumerate(("pt", "ptlow")):
            ax = axs[r_, c_]
            for ci in range(len(DR_CLASSES)):
                a, edges = h("dr%d__%s__%s" % (ci, V(key), var)); b, _ = h("dr%d__%s__%s" % (ci, "m_" + key, var))
                a, b = a[1:-1], b[1:-1]
                with np.errstate(divide="ignore", invalid="ignore"):
                    r = np.where(b > 0, a / b, np.nan)
                l, u = cp(a, b); x = 0.5 * (edges[1:] + edges[:-1]); ok = b >= 20
                ax.errorbar(x[ok], r[ok], yerr=[(r - l)[ok], (u - r)[ok]], color=dcol[ci], marker="o", ms=3, lw=1,
                            label=DR_LABELS[ci])
            a, edges = h("%s__%s" % (V(key), var)); b, _ = h("%s__%s" % ("m_" + key, var))
            a, b = a[1:-1], b[1:-1]
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where(b > 0, a / b, np.nan)
            x = 0.5 * (edges[1:] + edges[:-1]); ok = b >= 20
            ax.plot(x[ok], r[ok], color="k", lw=1.6, ls="--", label="all ΔR (as slide 3)")
            # Fixed ΔR mix: per-class efficiencies weighted by the set's overall class fractions (all dimuon pT, flow included),
            # so the gap to the dashed curve is the part due to the changing ΔR mix along the axis (between-class contribution).
            fr = np.array([h("dr%d__%s__pt" % (ci, "m_" + key))[0].sum() for ci in range(len(DR_CLASSES))])
            fr = fr / fr.sum()
            effs, oks = [], []
            for ci in range(len(DR_CLASSES)):
                a_, _ = h("dr%d__%s__%s" % (ci, V(key), var)); b_, _ = h("dr%d__%s__%s" % (ci, "m_" + key, var))
                a_, b_ = a_[1:-1], b_[1:-1]
                with np.errstate(divide="ignore", invalid="ignore"):
                    effs.append(np.where(b_ > 0, a_ / b_, np.nan))
                oks.append(b_ >= 20)
            fixed = np.sum([f_ * e_ for f_, e_ in zip(fr, effs)], axis=0); okf = np.all(oks, axis=0)
            ax.plot(x, np.where(okf, fixed, np.nan), color="0.45", lw=1.6, ls=":", marker=".", ms=4,
                    label="fixed ΔR mix (overall class fractions; bins with all classes ≥ 20)")
            ax.set_ylim(0.4, 1.03); ax.grid(alpha=.3); ax.set_xlabel(ALL_LABELS[var])
            ax.set_xlim(edges[0], edges[-1])
            if r_ == 0:
                ax.set_title(style[key][0], fontsize=11, color=style[key][1], fontweight="bold")
            if c_ == 0:
                ax.set_ylabel("Vertexed / matched in the same set")
        ax = axs[2, c_]
        tot = h("%s__pt" % ("m_" + key))[0][1:-1]
        for ci in range(len(DR_CLASSES)):
            a, edges = h("dr%d__%s__pt" % (ci, "m_" + key)); a = a[1:-1]
            x = 0.5 * (edges[1:] + edges[:-1]); ok = tot >= 20
            ax.plot(x[ok], (a / np.where(tot > 0, tot, 1))[ok], color=dcol[ci], marker="o", ms=3, lw=1, label=DR_LABELS[ci])
        ax.set_ylim(0, 1.03); ax.grid(alpha=.3); ax.set_xlabel(ALL_LABELS["pt"])
        if c_ == 0:
            ax.set_ylabel("Fraction of matched events in the class")
    hd, lb = axs[0, 0].get_legend_handles_labels()
    fig.legend(hd, lb, loc="lower center", ncol=5, fontsize=10, bbox_to_anchor=(0.5, 0.025))
    fig.suptitle("6. Dimuon-pT dip of the vertexing efficiency, split by gen opening angle ΔR(μμ) at production: "
                 "rows 1-2 vertexed / matched per ΔR class (dashed: all ΔR), row 3 ΔR-class mix of the matched events", fontsize=11)
    finish(fig, out / "slide6_dimuon_pt_dip_by_gen_dR.png", bottom=0.07,
           note="Signal MC (2022EE), pre-selected displacedMuons; events with two accepted signal muons matched in the set; dimuon pT = "
                "|vector sum| of the two gen muon pT; charge-ordered sets as slide 3; raw unweighted; 68% Clopper–Pearson; bins with < 20 matched "
                "events omitted (a missing point is not zero efficiency)")

    # Cutflow
    v, _ = h("cutflow"); v = v[1:-1]
    labels = ("initial events", "two signal muons (μ⁻μ⁺)", "both accepted", "both matched, any component (union)",
              "vertexed, priority Global>Inner>DSA", "vertexed, priority Global>DSA>Inner", "vertexed, any valid combo (fallback)")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    y = np.arange(len(labels))[::-1]
    for yi, c, col, al in zip(y, v, ["0.6", "0.6", "C0", "C1", "tab:olive", "tab:cyan", "C2"], [1, 1, 1, 1, 1, 1, 1]):
        ax.barh(yi, c, color=col, alpha=al)
    for yi, c in zip(y, v):
        ax.text(c, yi, "  %d  (%.2f%% of initial, %.2f%% of accepted)" % (c, 100 * c / v[0], 100 * c / v[2]), va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.set_xlim(0, v.max() * 1.8); ax.set_xlabel("Events (raw)")
    ax.set_title("Cutflow (720000 initial events, 10000 per sample)", fontsize=10)
    finish(fig, out / "slide0_cutflow.png")

    # Slide 4: quality
    # Slide 4a-4e: one panel per set (tracker-leg sets on the top row, DSA-leg sets on the bottom row), all unit-area
    # (fraction of fits per bin, incl. the fraction outside the plotted range, quoted in each panel), shared y per row,
    # dashed black outline = first set of the row as reference, so differences between sets are visible directly.
    def qpanels(var, xlabel, fname, title):
        fig, axs = plt.subplots(2, 3, figsize=(15, 7.6), sharey="row")
        for r_, (sets, rng) in enumerate(((TRK_SETS, "fine"), (DSA_SETS, "wide"))):
            ref = None
            for c_, u in enumerate(sets):
                a = axs[r_, c_]
                nm = "%s__chi2" % u if var == "chi2" else "%s__%s_%s" % (u, var, rng)
                v, e_ = h(nm); n = v.sum()
                if c_ == 0:
                    ref = (v[1:-1] / n, e_)
                a.stairs(v[1:-1] / n, e_, fill=True, color=SETCOL[u], alpha=0.5)
                a.stairs(v[1:-1] / n, e_, color=SETCOL[u], lw=1.6)
                if c_ > 0:
                    a.stairs(ref[0], ref[1], color="k", lw=1.0, ls="--", label=label(sets[0]) + " (reference)")
                    a.legend(fontsize=7, loc="upper right")
                a.set_title("%s   N=%d, %.1f%% outside range" % (label(u), n, 100 * (v[0] + v[-1]) / n), fontsize=9)
                a.set_xlabel(xlabel); a.grid(alpha=.3)
                if c_ == 0:
                    a.set_ylabel(("Tracker-leg sets" if r_ == 0 else "DSA-leg sets") + "\nfraction of fits per bin (unit area)")
        fig.suptitle(title, fontsize=11)
        finish(fig, out / fname)

    qpanels("chi2", r"Vertex $\chi^2$/ndof (ndof = 1; plotted 0–10)", "slide4a_chi2.png", "4a. Vertex quality: χ²/dof of valid typed fits")
    qpanels("dvxy", r"$v_{xy}^{reco} - v_{xy}^{gen}$ [cm]", "slide4b_vxy_absolute.png",
            "4b. Vertex vxy resolution, absolute (reco fit radius − gen production radius, from the origin); top row narrow range, bottom row wide range")
    qpanels("dvxyrel", r"$(v_{xy}^{reco} - v_{xy}^{gen})/v_{xy}^{gen}$  (gen $v_{xy}$ > 1 cm)", "slide4c_vxy_relative.png",
            "4c. Vertex vxy resolution, relative (gen vxy > 1 cm); top row narrow range, bottom row wide range")
    qpanels("dpt", r"$p_T^{\mu\mu,reco} - p_T^{\mu\mu,gen}$ [GeV]", "slide4d_dimuon_pt_absolute.png",
            "4d. Dimuon pT resolution, absolute (sum of the two legs' track pT at their reference points); top row narrow, bottom row wide")
    qpanels("dptrel", r"$(p_T^{reco} - p_T^{gen})/p_T^{gen}$", "slide4e_dimuon_pt_relative.png",
            "4e. Dimuon pT resolution, relative; top row narrow range, bottom row wide range")

    # Slide 4f-4i: 90% interval (5th-95th percentile band, median line) versus the plotting variables, one panel per set
    # (rows) and variable (columns); y shared within the tracker rows and within the DSA rows of each column.
    YLAB = {"dvxy": r"$v_{xy}^{reco}-v_{xy}^{gen}$ [cm]", "dvxyrel": r"$(v_{xy}^{reco}-v_{xy}^{gen})/v_{xy}^{gen}$",
            "dpt": r"$p_T^{\mu\mu,reco}-p_T^{\mu\mu,gen}$ [GeV]", "dptrel": r"$(p_T^{reco}-p_T^{gen})/p_T^{gen}$"}
    TTL = {"dvxy": "vxy residual (absolute)", "dvxyrel": "vxy residual (relative, gen vxy > 1 cm)",
           "dpt": "dimuon pT residual (absolute)", "dptrel": "dimuon pT residual (relative)"}
    LETTER = {"dvxy": "f", "dvxyrel": "g", "dpt": "h", "dptrel": "i"}
    rows = TRK_SETS + DSA_SETS
    for metric, xvars, suffix in [(m_, tuple(EFF_EDGES), "") for m_ in QMETRICS] + \
                                 [(m_, EXT_GRID, "_extended") for m_ in QMETRICS]:
        fig, axs = plt.subplots(6, len(xvars), figsize=(4.5 * len(xvars), 15), sharex="col")
        for c_, xv in enumerate(xvars):
            lims = {0: [], 1: []}
            for r_, u in enumerate(rows):
                ax = axs[r_, c_]
                lo = f["quantiles/overall/q05__%s__%s__%s" % (u, metric, xv)]
                md = f["quantiles/overall/q50__%s__%s__%s" % (u, metric, xv)]
                hi = f["quantiles/overall/q95__%s__%s__%s" % (u, metric, xv)]
                nn = f["quantiles/overall/n__%s__%s__%s" % (u, metric, xv)]
                edges = lo.axis().edges(); x = 0.5 * (edges[1:] + edges[:-1])
                good = nn.values() >= Q_NMIN
                ax.fill_between(x[good], lo.values()[good], hi.values()[good], color=SETCOL[u], alpha=0.4, lw=0)
                ax.plot(x[good], md.values()[good], color=SETCOL[u], lw=1.8)
                ax.plot(x[good], lo.values()[good], color=SETCOL[u], lw=0.8)
                ax.plot(x[good], hi.values()[good], color=SETCOL[u], lw=0.8)
                ax.axhline(0, color="k", lw=0.5); ax.grid(alpha=.3)
                lims[0 if r_ < 3 else 1] += [lo.values()[good], hi.values()[good]]
                if c_ == 0:
                    ax.set_ylabel(label(u), fontsize=10, color=SETCOL[u], fontweight="bold")
                if r_ == 0:
                    ax.set_title(ALL_LABELS[xv], fontsize=10)
                if r_ == len(rows) - 1:
                    ax.set_xlim(edges[0], edges[-1])
            for g_, (r0, r1) in enumerate(((0, 3), (3, 6))):  # robust shared y range per group and column
                vals_ = np.concatenate(lims[g_])
                if len(vals_):
                    lo_, hi_ = np.percentile(vals_, [3, 97]); pad_ = 0.15 * (hi_ - lo_)
                    for r_ in range(r0, r1):
                        axs[r_, c_].set_ylim(lo_ - pad_, hi_ + pad_)
        fig.suptitle("4%s%s. %s: median (line) and 90%% interval, 5th–95th percentile (band), versus the plotting variables; "
                     "bins with < %d valid fits omitted; y axis shared within the tracker rows and within the DSA rows"
                     % (LETTER[metric], "x" if suffix else "", TTL[metric], Q_NMIN), fontsize=11)
        finish(fig, out / ("slide4%s_%s_90pc_vs_variables%s.png" % (LETTER[metric], metric, suffix)),
               note="Signal MC (2022EE); valid typed fits, mixed sets merged (both orderings); raw unweighted; y range clipped to the "
                    "3rd-97th percentile of the plotted band edges (extreme bins can fall outside)")

    # Slide 5: vertex quality by leg priority (one entry per vertexed event; scales mix tracker and DSA legs, so log y)
    fig, ax = plt.subplots(1, 5, figsize=(20, 4.2))
    for a, (var, xl) in zip(ax, (("chi2", r"Vertex $\chi^2$/ndof"), ("dvxy_wide", r"$v_{xy}^{reco}-v_{xy}^{gen}$ [cm]"),
                                 ("dvxyrel_wide", r"$(v_{xy}^{reco}-v_{xy}^{gen})/v_{xy}^{gen}$ (gen $v_{xy}$>1 cm)"),
                                 ("dpt_wide", r"$p_T^{\mu\mu,reco}-p_T^{\mu\mu,gen}$ [GeV]"),
                                 ("dptrel_wide", r"$(p_T^{reco}-p_T^{gen})/p_T^{gen}$"))):
        for pk in ("P1", "P2"):
            nm = "%s__%s" % (pk, var) if var != "chi2" else "%s__chi2" % pk
            v, e_ = h(nm); n = v.sum()
            a.stairs(v[1:-1] / n, e_, color=style[pk][1], ls="-" if pk == "P1" else "--",
                     label="%s (N=%d, %.1f%% outside)" % (style[pk][0], n, 100 * (v[0] + v[-1]) / n))
        a.set_xlabel(xl); a.set_ylabel("Fraction of fits per bin (unit area)"); a.set_yscale("log"); a.grid(alpha=.3)
    hd, lb = ax[0].get_legend_handles_labels()
    fig.legend(hd, lb, loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, 0.03))
    fig.suptitle("5. Vertex quality for the two vertex-level leg priorities (valid fits, one per event; log y); the two curves nearly coincide",
                 fontsize=11)
    finish(fig, out / "slide5_priority_quality.png", bottom=0.1)
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
