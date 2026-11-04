#!/usr/bin/env python3
"""Cutflow, selected-fraction and figure-of-merit plots for the reco-only dimuon chain plus the MET > 80 GeV and
leading-jet pT > 80 GeV, |eta| < 2.4 cut, signal and background (kb/studies/bkg-reco-dimuons.md Part 3; LPC only).

python3 -I met_jet_plots.py --signal-summary recovery.root --bkg-summary bkg_reco_dimuons.root \
    --met-jet met_jet_cut.root --style-dir RUNDIR --out-dir RUNDIR/plots [--log]

Saved ROOT counts only (no rescan). Stage counts: per-slice `counts` of recovery.root (signal: overall, lifetime/,
mass/) and per-sample `counts` of bkg_reco_dimuons.root summed over the background samples present in met_jet_cut.root
(the Part 2 inclusion); last stage = selected_met_and_jet of met_jet_cut.root. Every fraction is over all events of
the slice. F = eps_s / sqrt(eps_b), eps = selected (or selected and cut) / all events; relative and unweighted, not an
expected significance. Raw counts, Clopper-Pearson 68% bars; F bars are first-order propagation of those intervals.
"""
import argparse
import math
from pathlib import Path
import re
import sys

import ROOT

STAGES = [("events", "All events"), ("events_eligible_pair", "Eligible pair"), ("events_opposite_sign", "Opposite sign"),
          ("events_valid_vertex", "Valid vertex"), ("events_chi2", "#chi^{2}/ndof < 4"), ("events_pca_valid", "PCA valid"),
          ("events_dr", "PCA #DeltaR < 0.9"), ("events_met_angle", "|#Delta#phi(MET,#mu#mu)| < 0.5"),
          ("selected_met_and_jet", "+ MET, jet")]
GROUPS = ["ZJets", "WJets", "QCD", "TTbar", "SingleTop", "DYJets", "Multiboson", "WWZ", "WZZ", "ZZ", "ZZZ", "ZGamma"]
MAJOR = GROUPS[:5]  # FOM overlays split in two for legibility: the five largest groups, then the rest
LEVEL = 0.6827
POP = "Raw MC counts, unweighted"
BKG_POP = None  # set in main() from the background samples present in --met-jet
FOM_LINE = "Relative, unweighted, not a significance; bars approx."
CUT_LINE = "Last step: Type-1 PF MET > 80 GeV, leading jet p_{T} > 80 GeV, |#eta| < 2.4"  # band / above layout
CUT_LINES = ["MET, jet cut: Type-1 PF MET > 80 GeV,", "leading jet p_{T} > 80 GeV, |#eta| < 2.4"]  # frame layout
M_HEADER, M_KEYS = "M = m_{#chi} - #Deltam/2", ("#Deltam = 0.1 M", "0.2 M", "0.4 M")


def counts(f, path):
    h = f.Get(path)
    if not h:
        raise RuntimeError("Missing " + path + " in " + f.GetName())
    ax = h.GetXaxis()
    return {ax.GetBinLabel(i): h.GetBinContent(i) for i in range(1, h.GetNbinsX() + 1)}


def mass_key(name):
    """(Mchi - dMchi/2, dMchi) of 'Mchi-5p25_dMchi-0p5'."""
    m = re.fullmatch(r"Mchi-([\dp]+)_dMchi-([\dp]+)", name)
    mchi, dm = (float(x.replace("p", ".")) for x in m.groups())
    return (round(mchi - dm / 2, 6), dm)


def mass_label(name):
    mchi, dm = (x.replace("p", ".") for x in re.fullmatch(r"Mchi-([\dp]+)_dMchi-([\dp]+)", name).groups())
    return "%s / %s" % (mchi.rstrip("0").rstrip(".") if "." in mchi else mchi, dm.rstrip("0").rstrip(".") if "." in dm else dm)


def load(args):
    """slices: name -> list of 9 cumulative event counts (STAGES order)."""
    sig_f, bkg_f, mj = (ROOT.TFile.Open(p) for p in (args.signal_summary, args.bkg_summary, args.met_jet))
    sig, bkg = {}, {}
    dirs = lambda f, d: sorted(k.GetName() for k in f.Get(d).GetListOfKeys())
    for path in ["overall"] + ["lifetime/" + n for n in dirs(sig_f, "lifetime")] + ["mass/" + n for n in dirs(sig_f, "mass")]:
        c, m = counts(sig_f, path + "/counts"), counts(mj, "signal/" + path + "/counts")
        sig[path] = [c[k] for k, _ in STAGES[:-1]] + [m["selected_met_and_jet"]]
        if (c["events"], c["events_met_angle"]) != (m["events"], m["selected"]):
            raise RuntimeError("Signal %s: recovery.root and met_jet_cut.root disagree" % path)
    included = dirs(mj, "background/sample")
    for g in ["overall"] + ["group/" + x for x in GROUPS]:
        members = [s for s in included if g == "overall" or s.split("_")[0] == g[6:]]
        total = [0.] * (len(STAGES) - 1)
        for s in members:
            c = counts(bkg_f, "sample/%s/counts" % s)
            total = [t + c[k] for t, (k, _) in zip(total, STAGES[:-1])]
        m = counts(mj, "background/%s/counts" % g)
        if not members or (total[0], total[-1]) != (m["events"], m["selected"]):
            raise RuntimeError("Background %s: summed bkg_reco_dimuons.root counts disagree with met_jet_cut.root" % g)
        bkg[g] = total + [m["selected_met_and_jet"]]
    if sorted(s.split("_")[0] for s in included) != sorted(s.split("_")[0] for s in included if s.split("_")[0] in GROUPS):
        raise RuntimeError("Background sample outside the 12 groups")
    for name, v in list(sig.items()) + list(bkg.items()):
        if any(b > a for a, b in zip(v, v[1:])) or v[-1] > v[-2] or any(x != round(x) or x < 0 for x in v):
            raise RuntimeError("Cutflow not non-increasing or not integer: %s %r" % (name, v))
    return sig, bkg, len(included)


def cp(k, n, up):
    return ROOT.TEfficiency.ClopperPearson(int(n), int(k), LEVEL, up)


def fom(ks, ns, kb, nb):
    """F = (ks/ns)/sqrt(kb/nb) with first-order propagation of the Clopper-Pearson intervals; None if kb == 0."""
    if kb == 0:
        return None
    es, eb = ks / ns, kb / nb
    s_up, s_dn = cp(ks, ns, True) - es, es - cp(ks, ns, False)
    b_up, b_dn = cp(kb, nb, True) - eb, eb - cp(kb, nb, False)
    f = es / math.sqrt(eb)
    return f, f * math.hypot(s_up / es, b_dn / (2 * eb)), f * math.hypot(s_dn / es, b_up / (2 * eb))


def fom_check(ks, ns, kb, nb):
    """Independent recomputation through TEfficiency objects."""
    def eff(k, n):
        p, t = ROOT.TH1D("p", "", 1, 0, 1), ROOT.TH1D("t", "", 1, 0, 1)
        p.SetBinContent(1, k); t.SetBinContent(1, n)
        e = ROOT.TEfficiency(p, t); e.SetConfidenceLevel(LEVEL); e.SetStatisticOption(ROOT.TEfficiency.kFCP)
        return e.GetEfficiency(1), e.GetEfficiencyErrorUp(1), e.GetEfficiencyErrorLow(1)
    (es, su, sd), (eb, bu, bd) = eff(ks, ns), eff(kb, nb)
    f = es / eb ** 0.5
    return f, f * ((su / es) ** 2 + (bd / eb / 2) ** 2) ** 0.5, f * ((sd / es) ** 2 + (bu / eb / 2) ** 2) ** 0.5


class Plotter:
    def __init__(self, cs, out, log):
        self.cs, self.out, self.log = cs, Path(out), log

    def hist(self, values, name):
        h = ROOT.TH1D(name, "", len(values), 0, len(values)); h.SetDirectory(0)
        for i, v in enumerate(values, 1):
            h.SetBinContent(i, v); h.SetBinError(i, v ** 0.5)
        self.cs._keep.append(h)
        return h

    def eff(self, num, den, color, style, name):
        n = len(num)
        return self.cs.eff_graph(self.hist(num, name + "_n"), self.hist([den] * n if not isinstance(den, list) else den,
                                                                          name + "_d"), color, *style)

    def graph(self, points, color, style):
        """points: list of (x index, (f, up, down)) -> asymmetric graph, styled like the other series."""
        gr = ROOT.TGraphAsymmErrors()
        for i, (f, up, dn) in points:
            j = gr.GetN(); gr.SetPoint(j, i + 0.5, f); gr.SetPointError(j, 0.5, 0.5, min(dn, f), up)  # bar capped at 0
        return self.cs._style(gr, color, style[0], style[1], [0, 1], kind="hist")

    def draw(self, base, graphs, labels, xlabels, ytitle, lines, xtitle="", band=None, log=False, above=False):
        """Categorical x axis (one bin per label, vertical labels). Context lines top left; legend below them in up to
        two columns (frame), or colour and style keys placed above the frame (band); y max keeps the data below them.
        above=True (cutflows): context lines and legend in a band above the frame, so the frame holds only data."""
        cs = self.cs
        n = len(xlabels)
        lsize = 0.034 if n < 15 else 0.028
        labels_bottom = min(0.33, 0.09 + 0.30 * lsize * max(len(re.sub(r"#\w+|[{}_^]", "", l)) for l in xlabels))
        bottom = labels_bottom + (0.07 if xtitle else 0)
        top = 0.335 if band else 0.08
        if above:
            ncol = 1 if len(graphs) <= 2 else (2 if len(graphs) <= 8 else 3)
            rows = -(-len(graphs) // ncol)
            top = 0.10 + 0.035 * len(lines) + 0.032 * rows + 0.02
        c = ROOT.TCanvas("c_" + base.replace("/", "_"), "", 900, 1000 if band else (1100 if above else 900))
        c.SetLeftMargin(0.15); c.SetRightMargin(0.04); c.SetBottomMargin(bottom); c.SetTopMargin(top)
        cs._keep.append(c)
        pts = cs._points(graphs)
        hi = max([p[3] for p in pts] or [1])
        if band or above:
            key_bottom = 1 - top  # keys sit above the frame
        else:
            ncol = 1 if len(graphs) <= 3 else 2
            rows = -(-len(graphs) // ncol)
            row = 0.042 if rows <= 4 else 0.033
            y2 = 0.89 - 0.045 * len(lines) - 0.01
            key_bottom = y2 - row * rows
        frac = (key_bottom - 0.02 - bottom) / (1 - top - bottom)
        ymax = cs._round_up(hi / max(frac, 0.2))
        if ymax < 0.1:  # 5-character tick labels: room for the y title
            c.SetLeftMargin(0.18)
        fr = cs.frame(c, 0, n, 0, ymax, xtitle, ytitle)
        if ymax < 0.1:
            fr.GetYaxis().SetTitleOffset(1.75)
        ax = fr.GetXaxis(); ax.Set(n, 0, n)
        for i, lab in enumerate(xlabels, 1):
            ax.SetBinLabel(i, lab)
        ax.LabelsOption("v"); ax.SetLabelSize(lsize)
        ax.SetTitleOffset(1.0 + (labels_bottom - 0.10) * 10)
        c.Update()
        for g in graphs:
            g.Draw(cs._info[id(g)][1] + " same")
        hb = band or above
        t = cs.header(c, 0.15, 0.95 if hb else 0.94, (0.04, 0.031) if hb else (0.05, 0.038), 0.96)
        t.SetTextSize(0.028 if hb else 0.032); t.SetTextAlign(13)
        y0 = 0.90 if hb else 0.89
        for i, line in enumerate(lines):
            t.DrawLatex(c.GetLeftMargin() + 0.03, y0 - (0.035 if hb else 0.045) * i, line)
        if band:
            groups, stages, header = band
            ytop = 0.90 - 0.035 * len(lines) - 0.005  # keys go below the context lines
            cs.legend([(cs._proxy(col, 1, 20), lab, "l") for lab, col in groups.items()], y2=ytop, y1=1 - top + 0.01,
                      x1=0.50, header=header, ncol=2, size=0.026)
            cs.legend([(cs._proxy(cs.BLACK, ls, ms), lab, "lp") for lab, (ls, ms) in stages.items()],
                      y2=ytop, y1=1 - top + 0.01, x1=0.17, x2=0.47, size=0.026)
        elif above:
            ytop = 0.90 - 0.035 * len(lines) - 0.005
            cs.legend([(g, lab, cs._info[id(g)][2]) for g, lab in zip(graphs, labels)], y2=ytop, y1=1 - top + 0.01,
                      x1=0.17, x2=0.96, ncol=ncol, size=0.026)
        else:
            cs.legend([(g, lab, cs._info[id(g)][2]) for g, lab in zip(graphs, labels)], y2=y2, y1=key_bottom, x1=0.18,
                      x2=0.94, ncol=ncol, size=0.030 if rows <= 4 else 0.026)
        logy = None
        if log:
            lo = min([p[2] for p in pts if p[2] > 0] or [1e-5])
            ylo = 10 ** math.floor(math.log10(lo / 2))
            f = max(frac, 0.45)  # data below the legend as in the linear plot, but at least 45% of the frame
            logy = (ylo, ylo * (hi * 1.05 / ylo) ** (1 / f))
        cs.save(c, self.out / base, log=log, logy=logy)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("--signal-summary", "--bkg-summary", "--met-jet", "--style-dir", "--out-dir"):
        ap.add_argument(k, required=True)
    ap.add_argument("--log", action="store_true", help="add log-y copies of the cutflows")
    args = ap.parse_args()
    sys.path.insert(0, args.style_dir)
    import cms_style as cs
    cs.setup(era="2022EE (13.6 TeV)", status="Simulation Private Work")
    ROOT.gErrorIgnoreLevel = ROOT.kWarning
    out = Path(args.out_dir)
    if out.exists():
        raise FileExistsError(out)
    global BKG_POP
    sig, bkg, n_bkg = load(args)
    ALLB = "all %d" % n_bkg
    BKG_POP = "Background: %s 2022EE samples" % ("all 50" if n_bkg == 50 else "%d of 50" % n_bkg)
    P = Plotter(cs, out, args.log)
    stage_labels = [lab for _, lab in STAGES]
    taus = sorted((k for k in sig if k.startswith("lifetime/")), key=lambda k: float(k.split("-")[1]))
    masses = sorted((k for k in sig if k.startswith("mass/")), key=lambda k: mass_key(k[5:]))
    tau_label = lambda k: "c#tau = %s mm" % k.split("-")[1]
    sig_label = {"overall": "Signal, all 72 samples", **{k: "Signal, " + tau_label(k) for k in taus},
                 **{k: "Signal, m_{#chi} / #Deltam = %s GeV" % mass_label(k[5:]) for k in masses}}
    bkg_label = {"overall": "Background, " + ALLB, **{"group/" + g: g for g in GROUPS}}
    S, B = cs.FILLED_SOLID, cs.OPEN_DASHED
    cut_lines = ["Reco-only #mu#mu chain, cumulative; " + POP, CUT_LINE]
    print("Cutflows (event counts per stage):")
    for k, v in list(sig.items()) + list(bkg.items()):
        print("  %-40s %s" % (k, " ".join("%d" % x for x in v)))

    # 1. Cutflows.
    for k in ["overall"] + taus + masses:
        d = "cutflow/" + ("overall" if k == "overall" else k)
        v = sig[k]
        P.draw(d + "_signal", [P.eff(v, v[0], cs.BLUE, S, d + "s")], [sig_label[k]], stage_labels,
               "Fraction of all events", cut_lines, log=args.log, above=True)
        b = bkg["overall"]
        P.draw(d + "_signal_bkg", [P.eff(v, v[0], cs.BLUE, S, d + "s2"), P.eff(b, b[0], cs.BLACK, B, d + "b")],
               [sig_label[k], bkg_label["overall"]], stage_labels, "Fraction of all events", cut_lines + [BKG_POP],
               log=args.log, above=True)
    palette = [cs.BLUE, cs.ORANGE, cs.GREEN, cs.YELLOW, cs.SKY, cs.PURPLE]
    bstyle = {g: (palette[i % 6], (1, 20 + i % 6) if i < 6 else (2, 24 + i % 6)) for i, g in enumerate(GROUPS)}
    for g in ["overall"] + ["group/" + x for x in GROUPS]:
        b = bkg[g]
        P.draw("cutflow/background_" + g.replace("group/", ""), [P.eff(b, b[0], cs.BLACK, S, "cb" + g)], [bkg_label[g]],
               stage_labels, "Fraction of all events", cut_lines + [BKG_POP], log=args.log, above=True)
    gr = [P.eff(bkg["overall"], bkg["overall"][0], cs.BLACK, S, "cbo")]
    gr += [P.eff(bkg["group/" + g], bkg["group/" + g][0], bstyle[g][0], bstyle[g][1], "cbg" + g) for g in GROUPS]
    P.draw("cutflow/overlay_background_groups", gr, [bkg_label["overall"]] + GROUPS, stage_labels,
           "Fraction of all events", cut_lines + [BKG_POP], log=args.log, above=True)
    gr = [P.eff(sig[k], sig[k][0], palette[i], (1, 20 + i), "cto" + k) for i, k in enumerate(taus)]
    gr.append(P.eff(bkg["overall"], bkg["overall"][0], cs.BLACK, B, "ctob"))
    P.draw("cutflow/overlay_ctau", gr, ["Signal, " + tau_label(k) for k in taus] + [bkg_label["overall"]], stage_labels,
           "Fraction of all events", cut_lines + [BKG_POP + "; signal pooled over masses"], log=args.log, above=True)
    fam = sorted({mass_key(k[5:])[0] for k in masses})
    fstyle = [(3, 25), (2, 24), (1, 20)]  # Delta = 0.1, 0.2, 0.4 x (Mchi - Delta/2)
    gr = []
    for k in masses:
        m1, dm = mass_key(k[5:])
        gr.append(P.eff(sig[k], sig[k][0], palette[fam.index(m1)], fstyle[[0.1, 0.2, 0.4].index(round(dm / m1, 3))],
                        "cmo" + k))
    gr.append(P.eff(bkg["overall"], bkg["overall"][0], cs.BLACK, (1, 21), "cmob"))
    groups = {"%g GeV" % m: palette[i] for i, m in enumerate(fam)}
    groups["Background, " + ALLB] = cs.BLACK
    stages = dict(zip(M_KEYS, fstyle))
    P.draw("cutflow/overlay_mass", gr, None, stage_labels, "Fraction of all events",
           ["Reco-only #mu#mu chain, cumulative; raw MC counts, unweighted", BKG_POP + "; signal pooled over lifetimes",
            CUT_LINE], band=(groups, stages, M_HEADER), log=args.log)

    # 2. Selected / all events, uncut and with the cut.
    axes = {"ctau": ["overall"] + taus, "mass": ["overall"] + masses}
    xl = {"ctau": ["All"] + [k.split("-")[1] + " mm" for k in taus],
          "mass": ["All"] + [mass_label(k[5:]) for k in masses]}
    xt = {"ctau": "Signal c#tau (pooled over masses)", "mass": "Signal m_{#chi} / #Deltam [GeV] (pooled over lifetimes)"}
    for a, keys in axes.items():
        ns = [sig[k][0] for k in keys]
        unc, cut = [sig[k][7] for k in keys], [sig[k][8] for k in keys]
        b = bkg["overall"]
        s_unc, s_cut = P.eff(unc, ns, cs.BLUE, B, "f%su" % a), P.eff(cut, ns, cs.BLUE, S, "f%sc" % a)
        lines = [POP] + CUT_LINES
        P.draw("fraction/signal_vs_%s" % a, [s_unc, s_cut], ["Signal, selected #mu#mu", "Signal, + MET, jet cut"], xl[a],
               "Events / all events", lines, xtitle=xt[a])
        if a == "mass":  # background reference squashed at 0 next to signal; the background plot shows it (user 2026-11-04)
            continue
        P.draw("fraction/signal_bkg_vs_%s" % a,
               [P.eff(unc, ns, cs.BLUE, B, "g%su" % a), P.eff(cut, ns, cs.BLUE, S, "g%sc" % a),
                P.eff([b[7]] * len(keys), b[0], cs.BLACK, B, "g%sbu" % a),
                P.eff([b[8]] * len(keys), b[0], cs.BLACK, S, "g%sbc" % a)],
               ["Signal, selected #mu#mu", "Signal, + MET, jet cut", "Bkg (%s), selected #mu#mu" % ALLB,
                "Bkg (%s), + MET, jet cut" % ALLB], xl[a], "Events / all events", lines + [BKG_POP], xtitle=xt[a])
    bkeys = ["overall"] + ["group/" + g for g in GROUPS]
    nb = [bkg[g][0] for g in bkeys]
    P.draw("fraction/background_vs_group",
           [P.eff([bkg[g][7] for g in bkeys], nb, cs.BLACK, B, "fbu"), P.eff([bkg[g][8] for g in bkeys], nb, cs.BLACK, S, "fbc")],
           ["Bkg, selected #mu#mu", "Bkg, + MET, jet cut"], [ALLB.capitalize()] + GROUPS, "Events / all events",
           [POP, BKG_POP] + CUT_LINES, xtitle="Background process group")

    # 3. F = eps_s / sqrt(eps_b).
    print("F = eps_s/sqrt(eps_b) (value +up -down), uncut | cut:")
    zero = {}
    for a, keys in axes.items():
        over = {"uncut": [], "cut": []}
        for g in bkeys:
            b = bkg[g]
            series = {}
            for tag, st in (("uncut", 7), ("cut", 8)):
                pts = []
                for i, k in enumerate(keys):
                    r = fom(sig[k][st], sig[k][0], b[st], b[0])
                    chk = fom_check(sig[k][st], sig[k][0], b[st], b[0]) if r else None
                    if r and max(abs(x - y) for x, y in zip(r, chk)) > 1e-9 * max(1, r[0]):
                        raise RuntimeError("F recomputation mismatch for %s %s %s" % (g, k, tag))
                    if r:
                        pts.append((i, r))
                    if a == "ctau" and k == "overall":
                        print("  bkg %-12s %-5s %s" % (g, tag, "no background events survive" if not r else
                                                     "%.4f +%.4f -%.4f" % r))
                if not pts:
                    zero.setdefault(g, set()).add(tag)
                series[tag] = pts
            lines = [("Background: " + bkg_label[g]) if g != "overall" else BKG_POP, FOM_LINE] + CUT_LINES
            if not series["cut"] or not series["uncut"]:
                lines.append("No background events survive the %s selection" % ("cut" if not series["cut"] else "uncut"))
            gr, labs = [], []
            for tag, sty, lab in (("uncut", B, "Selected #mu#mu"), ("cut", S, "+ MET, jet cut")):
                if series[tag]:
                    gr.append(P.graph(series[tag], cs.BLUE, sty)); labs.append(lab)
            P.draw("fom/%s_vs_%s" % (g.replace("group/", "bkg_") if g != "overall" else "bkg_all", a), gr, labs, xl[a],
                   "#varepsilon_{s} / #sqrt{#varepsilon_{b}}", lines, xtitle=xt[a])
            for tag in over:
                if series[tag]:
                    color, sty = (cs.BLACK, (1, 20)) if g == "overall" else bstyle[g[6:]]
                    over[tag].append((g, P.graph(series[tag], color, sty), bkg_label[g]))
        for tag in over:
            for part, members in (("major", MAJOR), ("minor", [g for g in GROUPS if g not in MAJOR])):
                keep = [o for o in over[tag] if o[0] == "overall" or o[0][6:] in members]
                missing = [g for g in members if tag in zero.get("group/" + g, ())]
                lines = ["Signal and bkg: " + ("selected #mu#mu" if tag == "uncut" else "+ MET, jet cut"),
                         FOM_LINE] + CUT_LINES + (["No surviving background: " + ", ".join(missing)] if missing else [])
                P.draw("fom/overlay_%s_%s_vs_%s" % (tag, part, a), [o[1] for o in keep], [o[2] for o in keep], xl[a],
                       "#varepsilon_{s} / #sqrt{#varepsilon_{b}}", lines, xtitle=xt[a])
    print("PLOTS DONE", out)


if __name__ == "__main__":
    main()
