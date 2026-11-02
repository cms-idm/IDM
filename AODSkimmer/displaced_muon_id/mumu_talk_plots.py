#!/usr/bin/env python3
"""mumu talk plots from saved PreselectedVertexing outputs (no new MINIAOD processing).

python3 mumu_talk_plots.py analyze --input-dir RUN/inputs --output OUT/mumu_talk.root --revision <sha>
python3 mumu_talk_plots.py plot --output OUT/mumu_talk.root --plot-dir OUT/plots [--log]
python3 mumu_talk_plots.py validate --input-dir RUN/inputs --output OUT/mumu_talk.root
Study note: kb/studies/mumu-talk-matching-vertexing.md. Inputs: met_proxy_20261022 (hierarchy G > I > DSA, 5a2b10e).
Population: two-accepted events (twoAcc: exactly two signal muons, opposite charge, both accepted). Raw, unweighted.
Slide 1 (per muon): den = signal muons in twoAcc events; num = hierarchy match (matchObj >= 0) or independent
Global / Inner / DSA match (indObj, layout [type I, O, G][gen 0, 1]). Slide 2 (per event): den = twoAcc events;
matched = both hierarchy-matched; valid = best-best fit valid (bbStatus == 3, no fallback); *fixed = additionally both
matched legs are their object's top eligible component (G > I > DSA). Residual: valid best-best fits,
dr = hypot(bbVx, bbVy) - gen Lxy (both from the origin); tracker = no DSA leg, anyDSA = at least one DSA leg.
"""
import argparse
from pathlib import Path
import re
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # run with python3 -I, which drops the script directory
from preselected_vertexing_talk import cp, flowcounts, write_hist

INPUT_REV, INPUT_ORDER = "5a2b10e", "global>inner>outer(DSA)"

S1_VIEWS = {"pt30": ("pt", np.arange(0, 30.01, 1.0)), "pt100": ("pt", np.arange(0, 100.01, 5.0)),
            "d030": ("d0", np.arange(0, 30.01, 1.0)), "d0740": ("d0", np.arange(0, 740.01, 10.0))}
S2_VIEWS = {"lxy30": ("lxy", np.arange(0, 30.01, 1.0)), "lxy740": ("lxy", np.arange(0, 740.01, 10.0)),
            "d015": ("d0max", np.arange(0, 15.01, 1.0))}
XLAB = {"pt30": r"Gen muon $p_T$ [GeV]", "pt100": r"Gen muon $p_T$ [GeV]", "d030": r"Gen muon $|d_0|$ to origin [cm]",
        "d0740": r"Gen muon $|d_0|$ to origin [cm]", "lxy30": r"Gen production $L_{xy}$ [cm]",
        "lxy740": r"Gen production $L_{xy}$ [cm]", "d015": r"Max gen $|d_0|$ of the two muons [cm]"}
S1_NUMS = {"h": ("Hierarchy G > I > DSA", "k"), "G": ("Global (independent)", "#0072B2"),
           "I": ("Inner (independent)", "#E69F00"), "O": ("DSA (independent)", "#009E73")}
S2_NUMS = {"matched": ("Both muons matched", "#0072B2"), "valid": ("Both matched and vertex valid", "#D55E00")}
CLS = ("tracker", "anyDSA")
CLS_LAB = {"tracker": "Tracker legs (GG, GI, II)", "anyDSA": "At least one DSA leg"}
RANK = {2: 0, 0: 1, 1: 2}  # G > I > DSA; types 0 I, 1 O(DSA), 2 G
TALK = ("Mchi-5p25_dMchi-0p5", "Mchi-55p0_dMchi-10p0")
TALK_CT = ("10", "100")
Q_NMIN = 100
EFF_NMIN = 20


def top_type(obj, cobj, ctype, cid, ceta):
    el = [t for o, t, i, e in zip(cobj, ctype, cid, ceta) if o == obj and i and e]
    return min(el, key=lambda t: RANK[t]) if el else -1


def load(args):
    import awkward as ak
    import uproot
    paths = sorted(Path(args.input_dir).glob("*.root"))
    if len(paths) != args.expected_samples:
        raise RuntimeError("Unexpected number of inputs: %d" % len(paths))
    names, mu, ev, revs, orders, total = [], [], [], set(), set(), 0
    br = ["twoAcc", "sigPt", "sigD0", "sigLxy", "matchObj", "matchType", "indObj", "bbStatus", "bbVx", "bbVy",
          "cObj", "cType", "cIdPass", "cInEta"]
    for si, path in enumerate(paths):
        with uproot.open(path) as f:
            sample = str(f["preselectedVertexing/sample"].member("fTitle"))
            if sample in names or not re.fullmatch(r"Mchi-[\dp]+_dMchi-[\dp]+_ctau-[\dp]+", sample):
                raise RuntimeError("Bad/duplicate sample " + sample)
            names.append(sample)
            revs.add(str(f["preselectedVertexing/revision"].member("fTitle")))
            orders.add(str(f["preselectedVertexing/hierarchyOrder"].member("fTitle")))
            a = f["preselectedVertexing/events"].arrays(br, library="ak")
        total += len(a)
        two = np.asarray(a["twoAcc"]).astype(bool)
        if np.any(ak.num(a["matchObj"][~two]) > 0):
            raise RuntimeError("matching filled outside twoAcc in " + sample)
        a = a[two]
        n = len(a)
        for b, k in (("sigPt", 2), ("matchObj", 2), ("matchType", 2), ("indObj", 6)):
            if np.any(ak.num(a[b]) != k):
                raise RuntimeError("%s length != %d in %s" % (b, k, sample))
        mo, mt = ak.to_numpy(a["matchObj"]), ak.to_numpy(a["matchType"])
        io = ak.to_numpy(a["indObj"])
        both = np.all(mo >= 0, axis=1)
        valid = np.asarray(a["bbStatus"]) == 3
        if np.any(valid & ~both):
            raise RuntimeError("valid best-best without both matched in " + sample)
        fixed = np.zeros(n, bool)
        for i in np.flatnonzero(both):
            c = (list(a["cObj"][i]), list(a["cType"][i]), list(a["cIdPass"][i]), list(a["cInEta"][i]))
            fixed[i] = all(top_type(mo[i, g], *c) == mt[i, g] for g in (0, 1))
        lxy = ak.to_numpy(a["sigLxy"])[:, 0]
        d0 = ak.to_numpy(a["sigD0"])
        for g in (0, 1):
            mu.append(dict(sample=np.full(n, si), pt=ak.to_numpy(a["sigPt"])[:, g], d0=d0[:, g], h=mo[:, g] >= 0,
                           I=io[:, g] >= 0, O=io[:, 2 + g] >= 0, G=io[:, 4 + g] >= 0))
        ev.append(dict(sample=np.full(n, si), lxy=lxy, d0max=d0.max(axis=1), matched=both, valid=valid,
                       matchedfixed=both & fixed, validfixed=valid & fixed, dsa=np.any(mt == 1, axis=1),
                       vx=np.asarray(a["bbVx"]).astype(float), vy=np.asarray(a["bbVy"]).astype(float)))
    if len(revs) != 1 or not next(iter(revs)).startswith(INPUT_REV) or orders != {INPUT_ORDER}:
        raise RuntimeError("Unexpected revision/hierarchy: %s %s" % (revs, orders))
    cat = lambda xs: {k: np.concatenate([x[k] for x in xs]) for k in xs[0]}
    return names, cat(mu), cat(ev), next(iter(revs)), next(iter(orders)), total


def groups_of(names):
    groups = {"overall": None}
    for si, s in enumerate(names):
        mass, tau = s.rsplit("_ctau-", 1)
        groups.setdefault("mass_" + mass, set()).add(si)
        groups.setdefault("ctau_" + tau, set()).add(si)
        groups["sample_" + s] = {si}
    return groups


def analyze(args):
    import ROOT
    if Path(args.output).exists():
        raise RuntimeError("Refusing to overwrite ROOT output")
    names, mu, ev, rev, order, total = load(args)
    if args.expected_events and total != args.expected_events:
        raise RuntimeError("Event total mismatch: %d" % total)
    fit = ev["valid"]
    dr = np.hypot(ev["vx"], ev["vy"]) - ev["lxy"]
    if not np.all(np.isfinite(dr[fit])):
        raise RuntimeError("non-finite residual")
    root = ROOT.TFile.Open(str(args.output), "CREATE")
    ROOT.TNamed("definitions", __doc__.split("Population:", 1)[1].strip()).Write()
    ROOT.TNamed("analysisRevision", args.revision).Write()
    ROOT.TNamed("inputRevision", rev).Write()
    ROOT.TNamed("hierarchyOrder", order).Write()
    ROOT.TNamed("samples", "\n".join(names)).Write()
    c = ROOT.TH1D("counts", "initial, twoAcc, matched, valid, matchedfixed, validfixed", 6, 0, 6)
    for i, v in enumerate((total, len(ev["lxy"]), ev["matched"].sum(), fit.sum(), ev["matchedfixed"].sum(),
                           ev["validfixed"].sum())):
        c.SetBinContent(i + 1, float(v))
    c.Write()
    top = root.mkdir("groups")
    for gname, ids in groups_of(names).items():
        d = top.mkdir(gname); d.cd()
        gm = np.ones(len(mu["pt"]), bool) if ids is None else np.isin(mu["sample"], list(ids))
        ge = np.ones(len(ev["lxy"]), bool) if ids is None else np.isin(ev["sample"], list(ids))
        for v, (src, edges) in S1_VIEWS.items():
            for k in ("den",) + tuple(S1_NUMS):
                sel = gm if k == "den" else gm & mu[k]
                write_hist("s1__%s__%s" % (k, v), k, edges, flowcounts(mu[src][sel], edges))
        for v, (src, edges) in S2_VIEWS.items():
            for k in ("den", "matched", "valid", "matchedfixed", "validfixed"):
                sel = ge if k == "den" else ge & ev[k]
                write_hist("s2__%s__%s" % (k, v), k, edges, flowcounts(ev[src][sel], edges))
    root.cd()
    t = ROOT.TTree("fits", "valid best-best fits: residual dr = hypot(vx, vy) - gen Lxy [cm]")
    from array import array
    bs, bl, bd, br_, bc = array("i", [0]), array("d", [0]), array("d", [0]), array("d", [0]), array("i", [0])
    t.Branch("sample", bs, "sample/I"); t.Branch("lxy", bl, "lxy/D"); t.Branch("d0max", bd, "d0max/D")
    t.Branch("dr", br_, "dr/D"); t.Branch("anyDSA", bc, "anyDSA/I")
    for i in np.flatnonzero(fit):
        bs[0], bl[0], bd[0], br_[0], bc[0] = int(ev["sample"][i]), ev["lxy"][i], ev["d0max"][i], dr[i], int(ev["dsa"][i])
        t.Fill()
    t.Write()
    root.Close()
    print("events %d twoAcc %d matched %d valid %d matchedfixed %d validfixed %d muons %d fits %d" % (
        total, len(ev["lxy"]), ev["matched"].sum(), fit.sum(), ev["matchedfixed"].sum(), ev["validfixed"].sum(),
        len(mu["pt"]), fit.sum()))


def qband(x, y, edges, nmin=Q_NMIN):
    """Per x bin (no flow): median, 16% and 84% quantiles of y, NaN with fewer than nmin entries."""
    idx = np.searchsorted(edges, x, side="right") - 1
    out = np.full((3, len(edges) - 1), np.nan)
    for b in range(len(edges) - 1):
        v = y[idx == b]
        if len(v) >= nmin:
            out[:, b] = np.quantile(v, (0.5, 0.16, 0.84))
    return out


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import uproot
    out = Path(args.plot_dir)
    for sub in ("overall", "mass", "ctau", "sample", "overlays", "talk"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    f = uproot.open(args.output)
    names = str(f["samples"].member("fTitle")).split("\n")
    fits = f["fits"].arrays(library="np")
    foot = ("Signal MC 2022EE (2026-09-28), raw unweighted; events with exactly two signal muons (opposite charge) both "
            "accepted; cleanedDisplacedMuons, eligible = own ID and |η|<2.4; station-2 matching, same charge, ΔR<0.06 G/I, "
            "<0.10 DSA; input %s, analysis %s" % (f["inputRevision"].member("fTitle"), f["analysisRevision"].member("fTitle")))
    note1 = "Per muon; curves overlap (not additive). Bars: nominal Clopper–Pearson 68%, the two muons of an event treated as independent."
    note2 = "Per event; best–best = Kalman fit of the two hierarchy-matched legs, no fallback. Bars: Clopper–Pearson 68%."
    note3 = "Valid best–best fits; Δr = reco vertex radius − gen Lxy (both from origin); median and 16–84%% band, bins with ≥%d fits." % Q_NMIN

    def eff(g, slide, k, v):
        a, edges = f["groups/%s/s%d__%s__%s" % (g, slide, k, v)].to_numpy()
        b, _ = f["groups/%s/s%d__den__%s" % (g, slide, v)].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(b > 0, a / b, np.nan)
        lo, hi = cp(a, b)
        x = 0.5 * (edges[1:] + edges[:-1]); ok = b >= EFF_NMIN
        return x[ok], r[ok], (r - lo)[ok], (hi - r)[ok], edges

    def finish(fig, path, note, pdf=False, logy=True):
        fig.text(0.01, 0.025, note, fontsize=7, color="0.3")
        fig.text(0.01, 0.005, foot, fontsize=6, color="0.45")
        fig.tight_layout(rect=(0, 0.05, 1, 0.96))
        fig.savefig(path, dpi=130)
        if pdf:
            fig.savefig(path.with_suffix(".pdf"))
        if args.log and logy:
            for ax in fig.axes:
                ax.set_yscale("log"); ax.set_ylim(1e-3, 1.1)  # efficiencies only; the linear y floor of 0 is not loggable
            fig.savefig(path.with_name(path.stem + "_log.png"), dpi=130)
        plt.close(fig)

    def eff_panels(series, slide, views, title, path, note, ylabel, pdf=False):
        """series: list of (group, key, label, colour, linestyle)."""
        nc = 2 if len(views) == 4 else len(views)
        nr = (len(views) + nc - 1) // nc
        fig, axs = plt.subplots(nr, nc, figsize=(6.2 * nc, 4.6 * nr), squeeze=False)
        for ax, v in zip(axs.flat, views):
            for g, k, lab, col, ls in series:
                x, r, el, eu, edges = eff(g, slide, k, v)
                ax.errorbar(x, r, yerr=[el, eu], color=col, ls=ls, marker="o", ms=3, lw=1.2, label=lab)
            ax.set_xlabel(XLAB[v]); ax.set_ylabel(ylabel); ax.set_ylim(0, 1.05); ax.set_xlim(edges[0], edges[-1])
            ax.grid(alpha=.3)
        axs.flat[0].legend(fontsize=8, loc="best")
        fig.suptitle(title, fontsize=12)
        finish(fig, path, note, pdf)

    def resid(series, title, path, pdf=False):
        """series: list of (sample-id set or None, label, colour)."""
        fig, axs = plt.subplots(2, len(S2_VIEWS), figsize=(6 * len(S2_VIEWS), 8.4), squeeze=False)
        for row, cl in enumerate(CLS):
            for ax, (v, (src, edges)) in zip(axs[row], S2_VIEWS.items()):
                for ids, lab, col in series:
                    m = fits["anyDSA"] == row
                    if ids is not None:
                        m &= np.isin(fits["sample"], list(ids))
                    q = qband(fits[src][m], fits["dr"][m], edges)
                    x = 0.5 * (edges[1:] + edges[:-1])
                    ax.plot(x, q[0], color=col, marker="o", ms=3, lw=1.2, label=lab + " median")
                    ax.fill_between(x, q[1], q[2], color=col, alpha=.2, step=None, label=lab + " 16–84%")
                ax.axhline(0, color="0.5", lw=.8)
                ax.set_xlabel(XLAB[v]); ax.set_ylabel("Δr = reco radius − gen Lxy [cm]"); ax.set_xlim(edges[0], edges[-1])
                ax.set_title(CLS_LAB[cl], fontsize=10); ax.grid(alpha=.3)
        axs[0, 0].legend(fontsize=7.5)
        fig.suptitle(title, fontsize=12)
        finish(fig, path, note3, pdf, logy=False)

    s1 = lambda g: [(g, k, lab, col, "-") for k, (lab, col) in S1_NUMS.items()]
    s2 = lambda g: [(g, k, lab, col, "-") for k, (lab, col) in S2_NUMS.items()]
    sid = {s: i for i, s in enumerate(names)}
    gl = {"overall": ("overall", "overall", None)}
    for s in names:
        mass, tau = s.rsplit("_ctau-", 1)
        gl["mass_" + mass] = ("mass", mass + " (all lifetimes)", {i for i, x in enumerate(names) if x.startswith(mass + "_")})
        gl["ctau_" + tau] = ("ctau", "ctau-" + tau + " (all masses)",
                             {i for i, x in enumerate(names) if x.endswith("_ctau-" + tau)})
        gl["sample_" + s] = ("sample", s, {sid[s]})
    for g, (sub, lab, ids) in gl.items():
        pdf = g == "overall"
        eff_panels(s1(g), 1, S1_VIEWS, "Signal muon matching efficiency: " + lab, out / sub / ("%s_s1_matching.png" % g),
                   note1, "Matched muons / muons in two-accepted events", pdf)
        eff_panels(s2(g), 2, S2_VIEWS, "Dimuon matching and vertexing efficiency: " + lab, out / sub / ("%s_s2_vertexing.png" % g),
                   note2, "Fraction of two-accepted events", pdf)
        if sub != "sample":  # per-sample residuals dropped (user, 2026-11-02): too few fits per bin
            resid([(ids, "", "#0072B2" if ids is None else "#D55E00")], "Vertex radius residual: " + lab,
                  out / sub / ("%s_s2_residual.png" % g), pdf)
    # Overlays: all masses (pooled over lifetimes) and all lifetimes (pooled over masses), hierarchy / valid only.
    masses = sorted({g for g in gl if g.startswith("mass_")}, key=lambda g: float(g[10:].split("_")[0].replace("p", ".")))
    taus = sorted({g for g in gl if g.startswith("ctau_")}, key=lambda g: float(g[5:]))
    for kind, gs in (("mass", masses), ("ctau", taus)):
        cm = plt.get_cmap("tab20" if len(gs) > 10 else "tab10")
        ser = lambda k: [(g, k, gl[g][1], cm(i % cm.N), "-" if i < 10 else "--") for i, g in enumerate(gs)]
        eff_panels(ser("h"), 1, S1_VIEWS, "Hierarchy matching efficiency, all %ss overlaid" % kind,
                   out / "overlays" / ("%s_s1_hierarchy.png" % kind), note1, "Matched muons / muons in two-accepted events")
        eff_panels(ser("valid"), 2, S2_VIEWS, "Both matched and vertex valid, all %ss overlaid" % kind,
                   out / "overlays" / ("%s_s2_valid.png" % kind), note2, "Fraction of two-accepted events")
    # Talk: two masses per lifetime.
    tcol = {TALK[0]: "#0072B2", TALK[1]: "#D55E00"}
    for tau in TALK_CT:
        ss = [m + "_ctau-" + tau for m in TALK]
        eff_panels([("sample_" + s, "h", s, tcol[m], "-") for s, m in zip(ss, TALK)], 1, S1_VIEWS,
                   "Hierarchy G > I > DSA matching efficiency, ctau-" + tau, out / "talk" / ("ctau%s_s1_hierarchy.png" % tau),
                   note1, "Matched muons / muons in two-accepted events", True)
        eff_panels([("sample_" + s, k, "%s: %s" % (s, S2_NUMS[k][0]), tcol[m], "-" if k == "valid" else ":")
                    for s, m in zip(ss, TALK) for k in S2_NUMS], 2, S2_VIEWS,
                   "Dimuon matching and vertexing efficiency, ctau-" + tau, out / "talk" / ("ctau%s_s2_vertexing.png" % tau),
                   note2, "Fraction of two-accepted events", True)
    print("plots written to", out)


def plot_cms(args):
    """Slide plots in PyROOT CMS style (as vertex_directions.py plot-dr --cms): one panel per figure, overall and the
    talk samples; same stored numerators/denominators and residual rows as plot()."""
    import ROOT
    import uproot
    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0); ROOT.gStyle.SetOptTitle(0)
    ROOT.gStyle.SetTextFont(42); ROOT.gStyle.SetLabelFont(42, "XYZ"); ROOT.gStyle.SetTitleFont(42, "XYZ")
    ROOT.gStyle.SetPadTickX(1); ROOT.gStyle.SetPadTickY(1)
    ROOT.TGaxis.SetMaxDigits(6)
    out = Path(args.plot_dir)
    out.mkdir(parents=True, exist_ok=True)
    f = uproot.open(args.output)
    names = str(f["samples"].member("fTitle")).split("\n")
    fits = f["fits"].arrays(library="np")
    col = lambda h: ROOT.TColor.GetColor(h)
    xt = {"pt30": "Gen muon p_{T} [GeV]", "pt100": "Gen muon p_{T} [GeV]", "d030": "Gen muon |d_{0}| [cm]",
          "d0740": "Gen muon |d_{0}| [cm]", "lxy30": "Gen L_{xy} [cm]", "lxy740": "Gen L_{xy} [cm]",
          "d015": "Max gen muon |d_{0}| [cm]"}
    s1lab = {"h": "Global > Inner > DSA", "G": "Global", "I": "Inner", "O": "DSA"}
    s2lab = {"matched": "Both muons matched", "valid": "Matched and vertexed"}
    keep = []

    def mass_label(m):
        a, b = re.fullmatch(r"Mchi-([\dp]+)_dMchi-([\dp]+)", m).groups()
        return "m_{#chi} = %s GeV, #Deltam = %s GeV" % (a.replace("p", "."), b.replace("p", "."))

    def canvas(name):
        c = ROOT.TCanvas(name, "", 800, 700)
        c.SetLeftMargin(0.15); c.SetRightMargin(0.04); c.SetTopMargin(0.08); c.SetBottomMargin(0.13)
        keep.append(c)
        return c

    def frame(c, edges, ylo, yhi, xtitle, ytitle):
        h = c.DrawFrame(edges[0], ylo, edges[-1], yhi); keep.append(h)
        h.GetXaxis().SetTitle(xtitle); h.GetYaxis().SetTitle(ytitle)
        for ax in (h.GetXaxis(), h.GetYaxis()):
            ax.SetTitleSize(0.05); ax.SetLabelSize(0.042)
        h.GetXaxis().SetTitleOffset(1.1); h.GetYaxis().SetTitleOffset(1.4)
        return h

    def labels(c, lines=()):
        c.cd()
        t = ROOT.TLatex(); t.SetNDC(); keep.append(t)
        t.SetTextFont(61); t.SetTextSize(0.05); t.SetTextAlign(11); t.DrawLatex(0.15, 0.94, "CMS")
        t.SetTextFont(52); t.SetTextSize(0.038); t.DrawLatex(0.255, 0.94, "Simulation Private Work")
        t.SetTextFont(42); t.SetTextAlign(31); t.DrawLatex(0.96, 0.94, "2022EE (13.6 TeV)")
        t.SetTextSize(0.035); t.SetTextAlign(13)
        for i, line in enumerate(lines):
            t.DrawLatex(0.19, 0.88 - 0.045 * i, line)

    def legend(entries, y2=0.80):
        leg = ROOT.TLegend(0.45, y2 - 0.055 * len(entries), 0.94, y2); keep.append(leg)
        leg.SetBorderSize(0); leg.SetFillStyle(0); leg.SetTextSize(0.035)
        for o, lab, opt in entries:
            leg.AddEntry(o, lab, opt)
        leg.Draw()

    def eff_graph(g, slide, k, v, color, style=1, marker=20):
        a, edges = f["groups/%s/s%d__%s__%s" % (g, slide, k, v)].to_numpy()
        b, _ = f["groups/%s/s%d__den__%s" % (g, slide, v)].to_numpy()
        lo, hi = cp(a, b)
        gr = ROOT.TGraphAsymmErrors(); keep.append(gr)
        for i in np.flatnonzero(b >= EFF_NMIN):
            r = a[i] / b[i]; x = 0.5 * (edges[i] + edges[i + 1])
            n = gr.GetN(); gr.SetPoint(n, x, r)
            gr.SetPointError(n, 0.5 * (edges[i + 1] - edges[i]), 0.5 * (edges[i + 1] - edges[i]), r - lo[i], hi[i] - r)
        gr.SetLineColor(color); gr.SetMarkerColor(color); gr.SetMarkerStyle(marker); gr.SetMarkerSize(1.0)
        gr.SetLineWidth(2); gr.SetLineStyle(style)
        return gr, edges

    def save(c, name, log=True):
        c.Modified(); c.SaveAs(str(out / (name + ".png"))); c.SaveAs(str(out / (name + ".pdf")))
        if args.log and log:
            fr = [p for p in c.GetListOfPrimitives() if p.InheritsFrom("TH1")][0]
            fr.SetMinimum(1e-3); fr.SetMaximum(5); c.SetLogy(1); c.Modified()
            c.SaveAs(str(out / (name + "_log.png"))); c.SetLogy(0)

    def eff_plot(name, series, slide, v, ytitle, lines=()):
        c = canvas("c_" + name)
        edges = S1_VIEWS[v][1] if slide == 1 else S2_VIEWS[v][1]
        frame(c, edges, 0, 1.3, xt[v], ytitle)
        ent = []
        for g, k, lab, color, style, marker in series:
            gr, _ = eff_graph(g, slide, k, v, color, style, marker)
            gr.Draw("P same"); ent.append((gr, lab, "lp"))
        labels(c, lines); legend(ent)
        save(c, name)

    def resid_plot(name, cls, v, series, lines=()):
        src, edges = S2_VIEWS[v]
        c = canvas("c_" + name)
        qs = []
        for ids, lab, color in series:
            m = fits["anyDSA"] == cls
            if ids is not None:
                m &= np.isin(fits["sample"], list(ids))
            qs.append(qband(fits[src][m], fits["dr"][m], edges))
        lo = np.nanmin([q[1] for q in qs]); hi = np.nanmax([q[2] for q in qs])
        pad = 0.15 * (hi - lo)
        frame(c, edges, lo - pad, hi + 0.8 * (hi - lo), xt[v], "Reco vertex r #minus gen L_{xy} [cm]")
        ent = []
        x = 0.5 * (edges[1:] + edges[:-1]); w = 0.5 * (edges[1:] - edges[:-1])
        for q, (ids, lab, color) in zip(qs, series):
            band, med = ROOT.TGraphAsymmErrors(), ROOT.TGraph(); keep.extend([band, med])
            for i in np.flatnonzero(np.isfinite(q[0])):
                n = band.GetN(); band.SetPoint(n, x[i], q[0, i]); band.SetPointError(n, w[i], w[i], q[0, i] - q[1, i], q[2, i] - q[0, i])
                med.SetPoint(med.GetN(), x[i], q[0, i])
            band.SetFillColorAlpha(color, 0.3); band.SetLineColor(color)
            med.SetLineColor(color); med.SetMarkerColor(color); med.SetMarkerStyle(20); med.SetLineWidth(2)
            band.Draw("2 same"); med.Draw("LP same")
            ent += [(med, (lab + ": " if lab else "") + "median", "lp"), (band, (lab + ": " if lab else "") + "16#minus84%", "f")]
        zero = ROOT.TLine(edges[0], 0, edges[-1], 0); zero.SetLineStyle(2); zero.SetLineColor(ROOT.kGray + 2); zero.Draw()
        keep.append(zero)
        labels(c, lines); legend(ent)
        save(c, name, log=False)

    blue, orange, green, yellow = col("#0072B2"), col("#D55E00"), col("#009E73"), col("#E69F00")
    pop = "Both signal muons accepted"
    s1c = {"h": (ROOT.kBlack, 20), "G": (blue, 21), "I": (yellow, 22), "O": (green, 23)}
    for v in S1_VIEWS:
        eff_plot("overall_s1_%s" % v, [("overall", k, s1lab[k], s1c[k][0], 1, s1c[k][1]) for k in S1_NUMS], 1, v,
                 "Muon matching efficiency", [pop])
    for v in S2_VIEWS:
        eff_plot("overall_s2_%s" % v, [("overall", k, s2lab[k], c_, 1, m_) for k, c_, m_ in
                                       (("matched", blue, 21), ("valid", orange, 20))], 2, v, "Event efficiency", [pop])
        for cls, cname in ((0, "tracker"), (1, "anydsa")):
            resid_plot("overall_residual_%s_%s" % (cname, v), cls, v, [(None, "", blue if cls == 0 else orange)],
                       [pop, "Tracker legs" if cls == 0 else "#geq 1 DSA leg"])
    tcol = {TALK[0]: blue, TALK[1]: orange}
    for tau in TALK_CT:
        lines = ["c#tau = %s mm" % tau]
        for v in S1_VIEWS:
            eff_plot("ctau%s_s1_%s" % (tau, v), [("sample_%s_ctau-%s" % (m, tau), "h", mass_label(m), tcol[m], 1, 20)
                                                 for m in TALK], 1, v, "Muon matching efficiency", lines)
        for v in S2_VIEWS:
            eff_plot("ctau%s_s2_%s" % (tau, v), [("sample_%s_ctau-%s" % (m, tau), k, "%s%s" % (mass_label(m), "" if k == "valid" else " (matched)"),
                                                  tcol[m], 1 if k == "valid" else 2, 20 if k == "valid" else 24)
                                                 for m in TALK for k in ("matched", "valid")], 2, v, "Event efficiency",
                     lines)
    print("CMS-style plots written to", out)


def validate(args):
    """Independent PyROOT loop with explicit gen-leg indexing; compares every bin (flow included) of the overall group
    and the four talk samples, and every residual row."""
    import ROOT
    from array import array
    paths = sorted(Path(args.input_dir).glob("*.root"))
    if len(paths) != args.expected_samples:
        raise SystemExit("Unexpected number of inputs: %d" % len(paths))
    res = ROOT.TFile.Open(args.output)
    names = res.Get("samples").GetTitle().split("\n")
    check = ["overall"] + ["sample_%s_ctau-%s" % (m, t) for m in TALK for t in TALK_CT]
    ref = {}
    for g in check:
        for sl, views, keys in ((1, S1_VIEWS, ("den",) + tuple(S1_NUMS)),
                                (2, S2_VIEWS, ("den", "matched", "valid", "matchedfixed", "validfixed"))):
            for v, (src, edges) in views.items():
                for k in keys:
                    h = ROOT.TH1D("ref_%s_%d_%s_%s" % (g, sl, k, v), "", len(edges) - 1, array("d", edges))
                    h.SetDirectory(0); ref[(g, sl, k, v)] = h
    mine = {}
    n0 = n2 = nm = nv = nmf = nvf = 0
    for path in paths:
        f = ROOT.TFile.Open(str(path))
        if not f.Get("preselectedVertexing/revision").GetTitle().startswith(INPUT_REV) or \
                f.Get("preselectedVertexing/hierarchyOrder").GetTitle() != INPUT_ORDER:
            raise SystemExit("unexpected input revision/hierarchy in " + str(path))
        n0 += f.Get("preselectedVertexing/events").GetEntries()
        s = f.Get("preselectedVertexing/sample").GetTitle()
        si = names.index(s)
        gs = ["overall"] + (["sample_" + s] if ("sample_" + s) in check else [])
        rows = []
        for e in f.Get("preselectedVertexing/events"):
            if not e.twoAcc:
                continue
            n2 += 1
            pt = [e.sigPt[0], e.sigPt[1]]; d0 = [e.sigD0[0], e.sigD0[1]]; lxy = e.sigLxy[0]
            mo = [e.matchObj[0], e.matchObj[1]]; mt = [e.matchType[0], e.matchType[1]]
            ind = {"I": [e.indObj[0], e.indObj[1]], "O": [e.indObj[2], e.indObj[3]], "G": [e.indObj[4], e.indObj[5]]}
            both = mo[0] >= 0 and mo[1] >= 0
            valid = e.bbStatus == 3
            fixed = both and all(top_type(mo[g], list(e.cObj), list(e.cType), list(e.cIdPass), list(e.cInEta)) == mt[g]
                                 for g in (0, 1))
            nm += both; nv += valid; nmf += both and fixed; nvf += valid and fixed
            flags = {"den": True, "matched": both, "valid": valid, "matchedfixed": both and fixed, "validfixed": valid and fixed}
            xs2 = {"lxy": lxy, "d0max": max(d0)}
            for g in gs:
                for gi in (0, 1):
                    x1 = {"pt": pt[gi], "d0": d0[gi]}
                    fl1 = {"den": True, "h": mo[gi] >= 0, "G": ind["G"][gi] >= 0, "I": ind["I"][gi] >= 0, "O": ind["O"][gi] >= 0}
                    for v, (src, _) in S1_VIEWS.items():
                        for k, ok in fl1.items():
                            if ok:
                                ref[(g, 1, k, v)].Fill(x1[src])
                for v, (src, _) in S2_VIEWS.items():
                    for k, ok in flags.items():
                        if ok:
                            ref[(g, 2, k, v)].Fill(xs2[src])
            if valid:
                d = float(ROOT.TMath.Sqrt(e.bbVx ** 2 + e.bbVy ** 2) - lxy)
                if not all(np.isfinite((d, lxy, max(d0)))):
                    raise SystemExit("non-finite residual input in " + s)
                rows.append((d, int(mt[0] == 1 or mt[1] == 1), float(lxy), float(max(d0))))
        mine[si] = sorted(rows)
    bad = 0
    for key, h in ref.items():
        g, sl, k, v = key
        hs = res.Get("groups/%s/s%d__%s__%s" % (g, sl, k, v))
        ax, rx = hs.GetXaxis(), h.GetXaxis()
        if ax.GetNbins() != rx.GetNbins() or any(abs(ax.GetBinLowEdge(b) - rx.GetBinLowEdge(b)) > 1e-9
                                                  for b in range(1, rx.GetNbins() + 2)):
            bad += 1
            print("AXIS MISMATCH", key)
            continue
        for b in range(h.GetNbinsX() + 2):
            if h.GetBinContent(b) != hs.GetBinContent(b):
                bad += 1
                if bad < 10:
                    print("MISMATCH", key, b, h.GetBinContent(b), hs.GetBinContent(b))
    print("bin check: %d histograms, %d mismatching bins" % (len(ref), bad))
    t = res.Get("fits")
    stored = {}
    for r in t:
        stored.setdefault(r.sample, []).append((r.dr, r.anyDSA, r.lxy, r.d0max))

    def differ(a, b):
        a, b = sorted(a), sorted(b)
        return len(a) != len(b) or any(abs(x[0] - y[0]) > 1e-6 or x[1] != y[1] or abs(x[2] - y[2]) > 1e-6 or
                                       abs(x[3] - y[3]) > 1e-6 for x, y in zip(a, b))
    rbad = sum(differ(stored.get(si, []), mine[si]) for si in mine)
    print("residual rows (dr, class, lxy, d0max): %d stored, %d independent, %d samples differ" % (
        t.GetEntries(), sum(len(x) for x in mine.values()), rbad))
    # Quantiles of the overall gen-Lxy 0-30 view: plotting code (qband) on the STORED tree versus explicit
    # sort + linear interpolation on the INDEPENDENT input rows.
    lx = np.array([r.lxy for r in t]); dr = np.array([r.dr for r in t]); cl = np.array([r.anyDSA for r in t])
    ind = np.array([x for v in mine.values() for x in v])
    edges = S2_VIEWS["lxy30"][1]; qbad = 0
    for c in (0, 1):
        q = qband(lx[cl == c], dr[cl == c], edges)
        for b in range(len(edges) - 1):
            v = np.sort(ind[(ind[:, 1] == c) & (ind[:, 2] >= edges[b]) & (ind[:, 2] < edges[b + 1]), 0])
            if len(v) < Q_NMIN:
                qbad += not np.all(np.isnan(q[:, b])); continue
            for j, p in enumerate((0.5, 0.16, 0.84)):
                pos = p * (len(v) - 1); lo = int(np.floor(pos))
                val = v[lo] + (pos - lo) * (v[min(lo + 1, len(v) - 1)] - v[lo])
                qbad += abs(val - q[j, b]) > 1e-6
    print("quantile check (overall lxy30, both classes): %d mismatches" % qbad)
    # Count closure: independent loop vs expected study totals vs the stored counts histogram.
    got = (n0, n2, nm, nv, nmf, nvf)
    stored_counts = tuple(int(res.Get("counts").GetBinContent(i + 1)) for i in range(6))
    expected = (args.expected_events, 494255, 199931, 189460, 186132, None)
    cbad = stored_counts != got or any(e_ is not None and e_ != g_ for e_, g_ in zip(expected, got))
    print("counts (initial, twoAcc, matched, valid, matchedfixed, validfixed): independent %s stored %s expected %s"
          % (got, stored_counts, expected))
    if bad or rbad or qbad or cbad:
        raise SystemExit(1)
    print("VALIDATION PASSED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("analyze", "plot", "plot-cms", "validate"))
    ap.add_argument("--input-dir")
    ap.add_argument("--revision", default="unspecified")
    ap.add_argument("--output", required=True)
    ap.add_argument("--plot-dir")
    ap.add_argument("--expected-samples", type=int, default=72)
    ap.add_argument("--expected-events", type=int, default=720000)
    ap.add_argument("--log", action="store_true", help="add log-y copies of the efficiency plots")
    args = ap.parse_args()
    {"analyze": analyze, "plot": plot, "plot-cms": plot_cms, "validate": validate}[args.mode](args)
