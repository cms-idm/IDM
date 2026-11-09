#!/usr/bin/env python3
"""Event-level reco and vertexing loss from slimmedDisplacedMuons (study note:
kb/studies/slimmed-displaced-vertexing-loss.md).

python3 slimmed_vertexing.py analyze --input-dir SLIMMED_LOSS_RUN/inputs --revision <sha> --output RUN/slimmed_vertexing.root
python3 slimmed_vertexing.py validate --input-dir SLIMMED_LOSS_RUN/inputs --output RUN/slimmed_vertexing.root
python3 slimmed_vertexing.py plot --output RUN/slimmed_vertexing.root --style-dir RUN --plot-dir RUN/plots [--log]

Inputs: slimmed_loss_cfg.py outputs (cmsRun 98f29d7): preselectedVertexing/events = case A (all displacedMuons ->
cleanedDisplacedMuons), preselectedVertexingB/events = case B (the raw displacedMuons kept in slimmedDisplacedMuons ->
same cleaner); same events, same job. Read-only.
Population: two-accepted events (twoAcc: exactly two signal muons, opposite charge, both accepted; no pT minimum).
Event level, raw unweighted; the denominator is the same for A and B.
reco(X) = both signal muons hierarchy-matched (G > I > DSA, station 2, same charge, strict dR < 0.06 G/I, < 0.10 DSA,
global-nearest greedy one-to-one, whole-object claim).
vertexed(X) = reco(X) and best-best fit valid (bbStatus == 3: Kalman refit of the two matched components' tracks, no
fallback) and chi2/ndof < 4 and reco dR(mumu) < 0.9 (track reference points) and |dphi(Type-1 PF MET, reco pT^mumu)| <
0.5 (pT^mumu = vector sum of the two legs' reference-point track pT); non-finite values, chi2 < 0, ndof <= 0, dR < 0,
MET <= 0 or pT^mumu <= 0 fail.
lost = pass in A and not in B; gained = pass in B and not in A; per stage.
x (gen, per event): max |d0| of the two signal muons (straight line to the origin), gen production Lxy, leading and
subleading gen muon pT. Groups: overall, mass point (pooled over lifetimes), lifetime (pooled over masses), sample;
unweighted pooling. ROOT is the numerical source (raw counts, Sumw2, flow bins kept); plot reads only that file.
"""
import argparse
import re
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python3 -I drops the script directory
from preselected_vertexing_analysis import flowcounts, write_hist  # noqa: E402

CHI2_CUT, DR_CUT, DPHI_CUT = 4.0, 0.9, 0.5
EDGES = {"d0": np.arange(0, 31, 1.0), "d0wide": np.arange(0, 741, 10.0), "lxy": np.arange(0, 31, 1.0),
         "lxywide": np.arange(0, 741, 10.0), "ptlead": np.arange(0, 31, 1.0), "ptsub": np.arange(0, 31, 1.0)}
XTITLE = {"d0": "Max gen |d_{0}| of the two muons [cm]", "lxy": "Gen L_{xy} [cm]",
          "ptlead": "Leading gen muon p_{T} [GeV]", "ptsub": "Subleading gen muon p_{T} [GeV]"}
XTITLE["d0wide"], XTITLE["lxywide"] = XTITLE["d0"], XTITLE["lxy"]
KEYS = ("den", "recoA", "recoB", "vtxA", "vtxB", "lostReco", "lostVtx", "gainedReco", "gainedVtx")
COUNTS = KEYS + ("muMatchA", "muMatchB", "bbValidA", "events")
BR = ["run", "lumi", "event", "twoAcc", "sigPt", "sigD0", "sigLxy", "sigQ", "matchObj", "bbStatus", "bbChi2", "bbNdof",
      "bbRecoDR", "bbRecoMuMuPt", "bbRecoMuMuPhi", "pfMetPt", "pfMetPhi"]
# Reference counts on all 72 samples (studies/mumu-talk-matching-vertexing.md; studies/slimmed-displaced-muon-loss.md).
REFERENCE = {"den": 494255, "recoA": 199931, "bbValidA": 189460, "muMatchA": 581703, "muMatchB": 520306}
SAMPLE_RE = re.compile(r"(Mchi-[\dp]+_dMchi-[\dp]+)_ctau-([\dp]+)")


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def stages(t):
    """reco and vertexed masks (all events) for one case's arrays."""
    two = t["twoAcc"] > 0
    reco = np.array([bool(tw) and len(m) == 2 and m[0] >= 0 and m[1] >= 0 for tw, m in zip(two, t["matchObj"])])
    st = t["bbStatus"]
    with np.errstate(invalid="ignore", divide="ignore"):
        fin = np.isfinite(t["bbChi2"]) & np.isfinite(t["bbNdof"]) & np.isfinite(t["bbRecoDR"]) & \
            np.isfinite(t["bbRecoMuMuPt"]) & np.isfinite(t["bbRecoMuMuPhi"]) & np.isfinite(t["pfMetPt"]) & \
            np.isfinite(t["pfMetPhi"])
        ok = fin & (t["bbChi2"] >= 0) & (t["bbNdof"] > 0) & (t["bbRecoDR"] >= 0) & (t["pfMetPt"] > 0) & \
            (t["bbRecoMuMuPt"] > 0)
        chi2 = np.where(ok, t["bbChi2"] / np.where(t["bbNdof"] > 0, t["bbNdof"], 1), np.inf)
        dphi = np.where(ok, np.abs(wrap(t["pfMetPhi"] - t["bbRecoMuMuPhi"])), np.inf)
    vtx = reco & (st == 3) & ok & (chi2 < CHI2_CUT) & (t["bbRecoDR"] < DR_CUT) & (dphi < DPHI_CUT)
    return reco, vtx


def analyze_sample(path):
    import uproot
    f = uproot.open(path)
    sample = str(f["preselectedVertexing/sample"].member("fTitle"))
    if sample != path.stem:
        raise RuntimeError("%s: file holds sample %s" % (path, sample))
    rev = str(f["preselectedVertexing/revision"].member("fTitle"))
    a = f["preselectedVertexing/events"].arrays(BR, library="np")
    b = f["preselectedVertexingB/events"].arrays(BR, library="np")
    n = len(a["run"])
    if n != 10000 or len(b["run"]) != n:
        raise RuntimeError("%s: entries A %d B %d (expected 10000)" % (sample, n, len(b["run"])))
    keys = set(zip(a["run"].tolist(), a["lumi"].tolist(), a["event"].tolist()))
    if len(keys) != n:
        raise RuntimeError("%s: %d duplicate (run, lumi, event)" % (sample, n - len(keys)))
    for k in ("run", "lumi", "event", "twoAcc", "pfMetPt", "pfMetPhi"):
        if not np.array_equal(a[k], b[k]):
            raise RuntimeError("%s: A/B differ in %s" % (sample, k))
    for k in ("sigPt", "sigD0", "sigLxy", "sigQ"):
        if any(not np.array_equal(x, y) for x, y in zip(a[k], b[k])):
            raise RuntimeError("%s: A/B gen arrays differ in %s" % (sample, k))
    two = a["twoAcc"] > 0
    if any(len(a[k][i]) != 2 for i in np.flatnonzero(two) for k in ("sigPt", "sigD0", "sigLxy", "matchObj")):
        raise RuntimeError("%s: twoAcc event without exactly two gen entries" % sample)
    recoA, vtxA = stages(a)
    recoB, vtxB = stages(b)
    for tag, t, reco, vtx in (("A", a, recoA, vtxA), ("B", b, recoB, vtxB)):
        if (vtx & ~reco).any() or ((t["bbStatus"] != 0) != reco).any():
            raise RuntimeError("%s: case %s: vertexed outside reco, or bbStatus set without both matched" % (sample, tag))
    idx = np.flatnonzero(two)
    pt = np.array([a["sigPt"][i] for i in idx], dtype=float).reshape(-1, 2)
    d0 = np.abs(np.array([a["sigD0"][i] for i in idx], dtype=float).reshape(-1, 2))
    lxy = np.array([a["sigLxy"][i] for i in idx], dtype=float).reshape(-1, 2)
    if np.abs(lxy[:, 0] - lxy[:, 1]).max(initial=0) > 1e-3:
        raise RuntimeError("%s: the two signal muons have different gen Lxy" % sample)
    x = {"d0": d0.max(1), "lxy": lxy[:, 0], "ptlead": pt.max(1), "ptsub": pt.min(1)}
    x["d0wide"], x["lxywide"] = x["d0"], x["lxy"]
    if not all(np.isfinite(v).all() for v in x.values()):
        raise RuntimeError("%s: non-finite gen x-variable" % sample)
    m = dict(den=np.ones(len(idx), bool), recoA=recoA[idx], recoB=recoB[idx], vtxA=vtxA[idx], vtxB=vtxB[idx])
    m.update(lostReco=m["recoA"] & ~m["recoB"], lostVtx=m["vtxA"] & ~m["vtxB"], gainedReco=~m["recoA"] & m["recoB"],
             gainedVtx=~m["vtxA"] & m["vtxB"])
    counts = {k: int(v.sum()) for k, v in m.items()}
    counts.update(events=n, bbValidA=int((recoA & (a["bbStatus"] == 3)).sum()),
                  muMatchA=int(sum((np.asarray(a["matchObj"][i]) >= 0).sum() for i in idx)),
                  muMatchB=int(sum((np.asarray(b["matchObj"][i]) >= 0).sum() for i in idx)))
    for st, s in (("reco", "Reco"), ("vtx", "Vtx")):
        if counts[st + "A"] - counts[st + "B"] != counts["lost" + s] - counts["gained" + s]:
            raise RuntimeError("%s: pass(A) - pass(B) != lost - gained (%s)" % (sample, s))
    hists = {(k, v): flowcounts(x[v][m[k]], EDGES[v]) for k in KEYS for v in EDGES}
    return sample, rev, counts, hists


def group_names(sample):
    mass, ctau = SAMPLE_RE.fullmatch(sample).groups()
    return ["overall", "mass_" + mass, "ctau_" + ctau, "sample_" + sample]


def analyze(args):
    import ROOT
    if Path(args.output).exists():
        raise RuntimeError("Refusing to overwrite numerical ROOT source %s" % args.output)
    paths = sorted(Path(args.input_dir).glob("*.root"))
    if len(paths) != args.expected_samples:
        raise RuntimeError("Expected %d sample files, found %d" % (args.expected_samples, len(paths)))
    if args.expected_samples == 72:
        flist_dir = Path(__file__).resolve().parents[1] / "fileLists/signal/2022EE/iDMmu"
        expected = sorted(f.name[:-len("_flist.txt")] for f in flist_dir.glob("*_flist.txt"))
        if [p.stem for p in paths] != expected:
            raise RuntimeError("Input samples differ from the 72 signal file lists")
    groups, revs = {}, set()
    for p in paths:
        s, rev, counts, hists = analyze_sample(p)
        revs.add(rev)
        for g in group_names(s):
            acc = groups.setdefault(g, [dict.fromkeys(COUNTS, 0), {}])
            for k in COUNTS:
                acc[0][k] += counts[k]
            for hk, h in hists.items():
                acc[1][hk] = acc[1].get(hk, 0) + h
        print("OK %s twoAcc %d recoA %d recoB %d vtxA %d vtxB %d lostReco %d lostVtx %d gained %d/%d" % (
            s, counts["den"], counts["recoA"], counts["recoB"], counts["vtxA"], counts["vtxB"], counts["lostReco"],
            counts["lostVtx"], counts["gainedReco"], counts["gainedVtx"]), flush=True)
    if len(revs) != 1:
        raise RuntimeError("Mixed input revisions: %s" % revs)
    tot = groups["overall"][0]
    if args.expected_samples == 72:
        bad = {k: (tot[k], v) for k, v in REFERENCE.items() if tot[k] != v}
        if bad:
            raise RuntimeError("Reference counts differ (got, expected): %s" % bad)
        print("REFERENCE OK: %s" % REFERENCE)
    for g, (c, h) in groups.items():
        for k in KEYS:
            for v in EDGES:
                if h[(k, v)].sum() != c[k]:
                    raise RuntimeError("%s: histogram %s__%s total %d != count %d" % (g, k, v, h[(k, v)].sum(), c[k]))
    out = ROOT.TFile(args.output, "CREATE")
    ROOT.TNamed("inputRevision", revs.pop()).Write()
    ROOT.TNamed("revision", args.revision).Write()
    ROOT.TNamed("inputDir", str(Path(args.input_dir).resolve())).Write()
    ROOT.TNamed("definitions", __doc__.split("\n\n", 2)[2]).Write()
    for g, (c, h) in groups.items():
        out.mkdir("groups/" + g)
        out.cd("groups/" + g)
        hc = ROOT.TH1D("counts", "raw unweighted event counts", len(COUNTS), 0, len(COUNTS))
        for i, k in enumerate(COUNTS):
            hc.GetXaxis().SetBinLabel(i + 1, k)
            hc.SetBinContent(i + 1, c[k])
        hc.Write()
        for k in KEYS:
            for v in EDGES:
                write_hist("%s__%s" % (k, v), XTITLE[v], EDGES[v], h[(k, v)])
    out.Close()
    print("OVERALL " + " ".join("%s=%d" % (k, tot[k]) for k in COUNTS))


def validate(args):
    """Independent recount: PyROOT TTree loop over the inputs (no uproot, no analyze code), for overall, one mass point,
    one lifetime and one sample; every key, every x-variable, every bin including flow, and the counts."""
    import math
    from array import array
    import ROOT
    ROOT.gROOT.SetBatch(True)
    out = ROOT.TFile.Open(args.output)
    paths = sorted(Path(args.input_dir).glob("*.root"))
    first = paths[0].stem
    mass, ctau = SAMPLE_RE.fullmatch(first).groups()
    check = ["overall", "mass_" + mass, "ctau_" + ctau, "sample_" + first]
    ref = {}
    for g in check:
        ref[g] = {(k, v): ROOT.TH1D("v_%s_%s_%s" % (g, k, v), "", len(EDGES[v]) - 1, array("d", EDGES[v]))
                  for k in KEYS for v in EDGES}
    cnt = {g: dict.fromkeys(KEYS, 0) for g in check}

    def passes(t):
        if not t.twoAcc or len(t.matchObj) != 2 or t.matchObj[0] < 0 or t.matchObj[1] < 0:
            return False, False
        vals = (t.bbChi2, t.bbNdof, t.bbRecoDR, t.bbRecoMuMuPt, t.bbRecoMuMuPhi, t.pfMetPt, t.pfMetPhi)
        if t.bbStatus != 3 or not all(math.isfinite(v) for v in vals):
            return True, False
        if t.bbChi2 < 0 or t.bbNdof <= 0 or t.bbRecoDR < 0 or t.pfMetPt <= 0 or t.bbRecoMuMuPt <= 0:
            return True, False
        dphi = abs(math.remainder(t.pfMetPhi - t.bbRecoMuMuPhi, 2 * math.pi))
        return True, (t.bbChi2 / t.bbNdof < CHI2_CUT and t.bbRecoDR < DR_CUT and dphi < DPHI_CUT)

    for p in paths:
        gs = [g for g in group_names(p.stem) if g in check]
        if not gs:
            continue
        f = ROOT.TFile.Open(str(p))
        ta, tb = f.Get("preselectedVertexing/events"), f.Get("preselectedVertexingB/events")
        for i in range(ta.GetEntries()):
            ta.GetEntry(i); tb.GetEntry(i)
            if (ta.run, ta.lumi, ta.event) != (tb.run, tb.lumi, tb.event):
                raise RuntimeError("validate: A/B misaligned in %s entry %d" % (p.stem, i))
            if not ta.twoAcc:
                continue
            ra, va = passes(ta)
            rb, vb = passes(tb)
            pts = [ta.sigPt[0], ta.sigPt[1]]
            xv = {"d0": max(abs(ta.sigD0[0]), abs(ta.sigD0[1])), "lxy": ta.sigLxy[0], "ptlead": max(pts), "ptsub": min(pts)}
            xv["d0wide"], xv["lxywide"] = xv["d0"], xv["lxy"]
            flags = dict(den=True, recoA=ra, recoB=rb, vtxA=va, vtxB=vb, lostReco=ra and not rb, lostVtx=va and not vb,
                         gainedReco=rb and not ra, gainedVtx=vb and not va)
            for g in gs:
                for k, on in flags.items():
                    if on:
                        cnt[g][k] += 1
                        for v in EDGES:
                            ref[g][(k, v)].Fill(xv[v])
        f.Close()
        print("validate read %s" % p.stem, flush=True)
    bad = 0
    for g in check:
        hc = out.Get("groups/%s/counts" % g)
        for i, k in enumerate(COUNTS):
            if k in KEYS and hc.GetBinContent(i + 1) != cnt[g][k]:
                print("MISMATCH %s count %s: file %d independent %d" % (g, k, hc.GetBinContent(i + 1), cnt[g][k]))
                bad += 1
        for (k, v), h in ref[g].items():
            s = out.Get("groups/%s/%s__%s" % (g, k, v))
            if s.GetNbinsX() != h.GetNbinsX() or any(abs(s.GetXaxis().GetBinLowEdge(b) - h.GetXaxis().GetBinLowEdge(b))
                                                     > 1e-9 for b in range(1, h.GetNbinsX() + 2)):
                print("MISMATCH %s %s__%s binning" % (g, k, v))
                bad += 1
                continue
            for b in range(h.GetNbinsX() + 2):
                if s.GetBinContent(b) != h.GetBinContent(b):
                    print("MISMATCH %s %s__%s bin %d: file %g independent %g" % (g, k, v, b, s.GetBinContent(b),
                                                                                 h.GetBinContent(b)))
                    bad += 1
        print("validate %s: %s" % (g, " ".join("%s=%d" % (k, cnt[g][k]) for k in KEYS)))
    if bad:
        raise RuntimeError("validate: %d mismatches" % bad)
    print("VALIDATED: groups %s, %d keys x %d x-variables, all bins incl. flow and counts match an independent PyROOT "
          "loop" % (check, len(KEYS), len(EDGES)))


def plot(args):
    sys.path.insert(0, str(Path(args.style_dir).resolve()))
    import ROOT
    import cms_style as cs
    cs.setup(era="2022EE (13.6 TeV)", status="Simulation Private Work")
    out = Path(args.plot_dir)
    f = ROOT.TFile.Open(args.output)
    groups = [k.GetName() for k in f.Get("groups").GetListOfKeys()]
    get = lambda g, k, v: f.Get("groups/%s/%s__%s" % (g, k, v))
    counts = {g: {f.Get("groups/%s/counts" % g).GetXaxis().GetBinLabel(i + 1): f.Get("groups/%s/counts" % g).GetBinContent(i + 1)
                  for i in range(len(COUNTS))} for g in groups}
    num = lambda s: float(s.replace("p", "."))

    nsamp = sum(g.startswith("sample_") for g in groups)

    def lab(g):
        if g == "overall":
            return "All %d signal samples" % nsamp
        k, _, v = g.partition("_")
        if k == "ctau":
            return "c#tau = %s mm (pooled over masses)" % v.replace("p", ".")
        m = SAMPLE_RE.fullmatch(v) if k == "sample" else re.fullmatch(r"Mchi-([\dp]+)_dMchi-([\dp]+)", v)
        if k == "sample":
            ms, ct = m.groups()
            a, d = re.fullmatch(r"Mchi-([\dp]+)_dMchi-([\dp]+)", ms).groups()
            return "m_{#chi} / #Deltam = %g / %g GeV, c#tau = %g mm" % (num(a), num(d), num(ct))
        return "m_{#chi} / #Deltam = %g / %g GeV (pooled over c#tau)" % (num(m.group(1)), num(m.group(2)))

    def mass_key(g):
        a, d = re.search(r"Mchi-([\dp]+)_dMchi-([\dp]+)", g).groups()
        return num(a), num(d)

    def ymax_of(graphs, frac):
        hi = max([gr.GetPointY(i) + gr.GetErrorYhigh(i) for gr in graphs for i in range(gr.GetN())] or [0.1])
        return cs._round_up(max(hi / frac, 0.02))

    COL = {"A": cs.BLUE, "B": cs.ORANGE}
    NAME = {"A": "displacedMuons", "B": "slimmedDisplacedMuons"}
    STAGE = {"reco": ("Reco", cs.OPEN_DASHED), "vtx": ("Vertexed", cs.FILLED_SOLID)}

    def two_panel(base, g, v, log):
        den = get(g, "den", v)
        top = [(cs.eff_graph(get(g, s + X, v), den, COL[X], *STAGE[s][1]), "%s, %s" % (NAME[X], STAGE[s][0].lower()))
               for X in "AB" for s in ("reco", "vtx")]
        bot = [(cs.eff_graph(get(g, "lost" + ("Reco" if s == "reco" else "Vtx"), v), den, cs.BLACK, *STAGE[s][1]),
                "Lost by slimming, %s" % STAGE[s][0].lower()) for s in ("reco", "vtx")]
        if not any(gr.GetN() for gr, _ in top):
            return False
        c = ROOT.TCanvas("c_" + base.name + ("_log" if log else ""), "", 800, 850)
        p1 = ROOT.TPad("p1_" + base.name, "", 0, 0.33, 1, 1); p2 = ROOT.TPad("p2_" + base.name, "", 0, 0, 1, 0.33)
        for p in (p1, p2):
            cs._keep.append(p); p.SetLeftMargin(0.15); p.SetRightMargin(0.04); p.SetTicks(1, 1)
        p1.SetTopMargin(0.09); p1.SetBottomMargin(0.025); p2.SetTopMargin(0.03); p2.SetBottomMargin(0.33)
        p1.Draw(); p2.Draw()
        xlo, xhi = EDGES[v][0], EDGES[v][-1]
        for pad, ser, ylo, yhi, yt, s in (
                (p1, top, 1e-3 if log else 0, 2 if log else ymax_of([x[0] for x in top], 0.62),
                 "Fraction of two-accepted events", 1.0),
                (p2, bot, 1e-4 if log else 0, 3 if log else ymax_of([x[0] for x in bot], 0.70), "Lost fraction", 0.67 / 0.33)):
            pad.cd(); pad.SetLogy(int(log))
            fr = pad.DrawFrame(xlo, ylo, xhi, yhi); cs._keep.append(fr)
            fr.GetYaxis().SetTitle(yt); fr.GetXaxis().SetTitle(XTITLE[v] if pad is p2 else "")
            for ax in (fr.GetXaxis(), fr.GetYaxis()):
                ax.SetTitleSize(0.05 * s * 0.8); ax.SetLabelSize(0.042 * s * 0.8)
            fr.GetYaxis().SetTitleOffset(1.45 / s / 0.8 * 0.8); fr.GetYaxis().SetNdivisions(505)
            fr.GetXaxis().SetTitleOffset(1.0)
            if pad is p1:
                fr.GetXaxis().SetLabelSize(0)
            for gr, _ in ser:
                gr.Draw("P same")
            pad.RedrawAxis()
        p1.cd()
        cs.header(p1, 0.15, 0.935, (0.055, 0.042), 0.96)
        t = ROOT.TLatex(); t.SetNDC(); t.SetTextFont(42); t.SetTextSize(0.036); t.SetTextAlign(13); cs._keep.append(t)
        t.DrawLatex(0.19, 0.87, lab(g))
        cs.legend([(gr, l, "lp") for gr, l in top], y2=0.82, y1=0.58, x1=0.46, x2=0.90, size=0.033)
        p2.cd()
        cs.legend([(gr, l, "lp") for gr, l in bot], y2=0.93, y1=0.70, x1=0.50, size=0.07)
        base.parent.mkdir(parents=True, exist_ok=True)
        c.SaveAs(str(base) + ("_log.png" if log else ".png"))
        c.Close()
        return True

    for g in groups:
        kind = g.split("_")[0]
        d = out / "efficiency" / kind
        for v in EDGES:
            for log in ((False, True) if args.log else (False,)):
                two_panel(d / ("eff_%s_%s" % (v, g)), g, v, log)

    # Summaries of lost / two-accepted per group, overall as dashed lines.
    sdir = out / "summary"
    sdir.mkdir(parents=True, exist_ok=True)

    def cp_point(k, n):
        lo = ROOT.TEfficiency.ClopperPearson(int(n), int(k), 0.6827, False)
        hi = ROOT.TEfficiency.ClopperPearson(int(n), int(k), 0.6827, True)
        return k / n, k / n - lo, hi - k / n

    masses = sorted([g for g in groups if g.startswith("mass_")], key=mass_key)
    ctaus = sorted([g for g in groups if g.startswith("ctau_")], key=lambda g: num(g.split("_")[1]))
    samples = sorted([g for g in groups if g.startswith("sample_")], key=lambda g: (mass_key(g), num(g.split("ctau-")[1])))
    mlab = lambda g: "%g / %g" % mass_key(g)
    separators = lambda cats: [i for i in range(1, len(cats)) if cats[i].startswith("sample_")
                               and mass_key(cats[i - 1]) != mass_key(cats[i])]

    def summary(base, cats, labels, series_of, xtitle, legend_entries, vertical):
        nb = len(cats)
        c = ROOT.TCanvas("c_" + base, "", 1000 if nb > 20 else 800, 700); cs._keep.append(c)
        c.SetLeftMargin(0.12 if nb > 20 else 0.15); c.SetRightMargin(0.03); c.SetTopMargin(0.08)
        c.SetBottomMargin(0.24 if vertical else 0.13)
        series = series_of()
        hi = max(y + eh for gr in series for y, eh in ((gr.GetPointY(i), gr.GetErrorYhigh(i)) for i in range(gr.GetN())))
        fr = ROOT.TH1D("f_" + base, "", nb, 0, nb); cs._keep.append(fr)
        fr.SetMinimum(0); fr.SetMaximum(cs._round_up(hi / 0.6))
        for i, l in enumerate(labels):
            fr.GetXaxis().SetBinLabel(i + 1, l)
        fr.GetXaxis().SetTitle(xtitle); fr.GetYaxis().SetTitle("Lost / two-accepted events")
        fr.GetXaxis().SetLabelSize(0.04 if vertical else 0.05); fr.GetYaxis().SetLabelSize(0.042)
        fr.GetXaxis().SetTitleSize(0.045); fr.GetYaxis().SetTitleSize(0.05)
        fr.GetXaxis().SetTitleOffset(2.6 if vertical else 1.1); fr.GetYaxis().SetTitleOffset(1.1 if nb > 20 else 1.4)
        if vertical:
            fr.LabelsOption("v")
        fr.Draw("AXIS")
        for xs in separators(cats):
            sep = ROOT.TLine(xs, 0, xs, fr.GetMaximum()); sep.SetLineColor(ROOT.kGray); sep.SetLineStyle(3)
            sep.Draw(); cs._keep.append(sep)
        tot = counts["overall"]
        lines = []
        for k, (ls, _) in (("lostReco", cs.OPEN_DASHED), ("lostVtx", cs.FILLED_SOLID)):
            y = tot[k] / tot["den"]
            ln = ROOT.TLine(0, y, nb, y); ln.SetLineStyle(2 if k == "lostReco" else 7); ln.SetLineColor(ROOT.kGray + 2)
            ln.SetLineWidth(2); ln.Draw(); cs._keep.append(ln); lines.append(ln)
        for gr in series:
            gr.Draw("P same")
        cs.context(c, [])
        ents = legend_entries(series, lines)
        cs.legend(ents, y2=0.89, x1=0.45 if nb > 20 else 0.40, ncol=2 if len(ents) > 4 else 1, size=0.030)
        c.RedrawAxis()
        c.SaveAs(str(sdir / base) + ".png")

    def stage_graphs(cats, color_of=None):
        out_g = []
        for k, (ls, mk) in (("lostReco", cs.OPEN_DASHED), ("lostVtx", cs.FILLED_SOLID)):
            gr = ROOT.TGraphAsymmErrors(); cs._keep.append(gr)
            for i, g in enumerate(cats):
                y, el, eh = cp_point(counts[g][k], counts[g]["den"])
                j = gr.GetN(); gr.SetPoint(j, i + 0.5, y); gr.SetPointError(j, 0, 0, el, eh)
            col = ROOT.TColor.GetColor(cs.BLACK)
            gr.SetMarkerStyle(mk); gr.SetMarkerColor(col); gr.SetLineColor(col); gr.SetMarkerSize(1.1)
            out_g.append(gr)
        return out_g

    std_leg = lambda s, l: [(s[0], "Lost, reco", "p"), (s[1], "Lost, vertexed", "p"), (l[0], "All samples, reco", "l"),
                            (l[1], "All samples, vertexed", "l")]
    summary("summary_mass", masses, [mlab(g) for g in masses], lambda: stage_graphs(masses),
            "m_{#chi} / #Deltam [GeV] (pooled over c#tau)", std_leg, True)
    summary("summary_lifetime", ctaus, ["%g" % num(g.split("_")[1]) for g in ctaus], lambda: stage_graphs(ctaus),
            "c#tau [mm] (pooled over masses)", std_leg, False)

    def sample_graphs():
        gs = []
        for ci, ct in enumerate(ctaus):
            cv = ct.split("_")[1]
            for k, (ls, mk) in (("lostReco", cs.OPEN_DASHED), ("lostVtx", cs.FILLED_SOLID)):
                gr = ROOT.TGraphAsymmErrors(); cs._keep.append(gr)
                for i, g in enumerate(samples):
                    if g.endswith("_ctau-" + cv):
                        y, el, eh = cp_point(counts[g][k], counts[g]["den"])
                        j = gr.GetN(); gr.SetPoint(j, i + 0.5, y); gr.SetPointError(j, 0, 0, el, eh)
                col = ROOT.TColor.GetColor(cs.COLORS[ci])
                gr.SetMarkerStyle(mk); gr.SetMarkerColor(col); gr.SetLineColor(col); gr.SetMarkerSize(0.9)
                gs.append(gr)
        return gs

    def sample_leg(s, l):
        ents = []
        for ci, ct in enumerate(ctaus):
            v = "%g" % num(ct.split("_")[1])
            ents += [(s[2 * ci], "c#tau %s mm, reco" % v, "p"), (s[2 * ci + 1], "c#tau %s mm, vertexed" % v, "p")]
        return ents + [(l[0], "All samples, reco", "l"), (l[1], "All samples, vertexed", "l")]
    # One label per mass point, on the first sample of its group; dotted separators between mass groups.
    slabels = [mlab(g) if i == 0 or mass_key(samples[i - 1]) != mass_key(g) else "" for i, g in enumerate(samples)]
    summary("summary_sample", samples, slabels, sample_graphs, "m_{#chi} / #Deltam [GeV] (grouped by mass, colored by c#tau)", sample_leg, True)

    # Overlays: all masses (pooled over lifetimes) and all lifetimes (pooled over masses).
    odir = out / "overlay"
    odir.mkdir(parents=True, exist_ok=True)
    QUANT = {"lostVtx": ("Lost / two-accepted events", "Lost by slimming, vertexed"),
             "vtxB": ("Vertexed / two-accepted events", "slimmedDisplacedMuons"),
             "vtxA": ("Vertexed / two-accepted events", "displacedMuons")}
    for kind, gs, hdr in (("mass", masses, "m_{#chi} / #Deltam [GeV], pooled over c#tau"),
                          ("lifetime", ctaus, "c#tau [mm], pooled over masses")):
        many = len(gs) > 7
        styles = [(cs.COLORS[i // 3 % 7], (1, 2, 3)[i % 3], (20, 24, 21)[i % 3]) if many else (cs.COLORS[i], 1, 20)
                  for i in range(len(gs))]
        labs = [mlab(g) if kind == "mass" else "%g" % num(g.split("_")[1]) for g in gs]
        for q, (yt, what) in QUANT.items():
            for v in EDGES:
                ss = [(cs.eff_graph(get(g, q, v), get(g, "den", v), col, ls, mk), l) for g, (col, ls, mk), l in zip(gs, styles, labs)]
                ss = [x for x in ss if x[0].GetN()]
                base = odir / ("%s_%s_%s" % (kind, q, v))
                if not many:
                    cs.plot(str(base), [x[0] for x in ss], [x[1] for x in ss], XTITLE[v], yt, lines=[what], header=hdr,
                            log=args.log)
                    continue
                rows = (len(ss) + 2) // 3 + 1
                y1 = 0.89 - rows * 0.036
                hi = max(gr.GetPointY(i) + gr.GetErrorYhigh(i) for gr, _ in ss for i in range(gr.GetN()))
                c = cs.canvas("c_" + base.name)
                cs.frame(c, EDGES[v][0], EDGES[v][-1], 0, cs._round_up(hi * 0.78 / (y1 - 0.15)), XTITLE[v], yt)
                c.Update()
                for gr, _ in ss:
                    gr.Draw("P same")
                cs.context(c, ())
                cs.legend([(gr, l, "lp") for gr, l in ss], y2=0.89, y1=y1, x1=0.20, ncol=3, size=0.024,
                          header="%s; %s" % (what, hdr))
                cs.save(c, str(base), False, args.log)

    pct = lambda k, n: "%d (%.2f%%)" % (k, 100.0 * k / n) if n else "%d" % k
    row = lambda name, g: [name, int(counts[g]["den"]), pct(counts[g]["recoA"], counts[g]["den"]),
                           pct(counts[g]["recoB"], counts[g]["den"]), pct(counts[g]["vtxA"], counts[g]["den"]),
                           pct(counts[g]["vtxB"], counts[g]["den"]), pct(counts[g]["lostReco"], counts[g]["den"]),
                           pct(counts[g]["lostVtx"], counts[g]["den"]),
                           "%d / %d" % (counts[g]["gainedReco"], counts[g]["gainedVtx"])]
    hdr = ["Group", "Two-accepted", "Reco A", "Reco B", "Vertexed A", "Vertexed B", "Lost reco", "Lost vertexed",
           "Gained reco / vertexed"]
    cap = cs.Captions("Reco and vertexing loss from slimmedDisplacedMuons (signal MC, 2022EE, 2026-09-28 production)", [
        "Population: two-accepted events (exactly two signal muons, opposite charge, both with |eta| < 2.4, 0 <= Lxy < "
        "740 cm, |vz| < 960 cm; no pT minimum). Event level, raw unweighted; pooling over samples unweighted.",
        "A = displacedMuons -> cleanedDisplacedMuons; B = only the displacedMuons kept in slimmedDisplacedMuons, then the "
        "same cleaner (a best case for a MiniAOD-only workflow).",
        "Reco = both signal muons matched by the G > I > DSA hierarchy (eligible components: own ID and own |eta| < 2.4; "
        "station 2; same charge; dR < 0.06 G/I, < 0.10 DSA; greedy one-to-one).",
        "Vertexed = reco and valid best-best Kalman fit of the two matched components (no fallback) with chi2/ndof < 4, "
        "reco dR(mumu) < 0.9 and |dphi(Type-1 PF MET, reco pT^mumu)| < 0.5.",
        "Lost by slimming = passes in A and not in B; fractions of two-accepted events. Clopper-Pearson 68.3%; bins with "
        "< 20 two-accepted events not drawn.",
        "x: max gen |d0| of the two muons (straight line to the origin); gen production Lxy; leading/subleading gen pT."])
    cap.add("efficiency/<group kind>/eff_<x>_<group>.png", "Top: reco and vertexed fraction for A and B. Bottom: lost "
            "by slimming at each stage. Counts per group: summary tables below. `d0wide`/`lxywide`: 0-740 cm in 10 cm "
            "bins. `_log` copies: log y; a bin with a zero fraction has no marker there and its interval is clipped at "
            "the lower edge.")
    cap.add("summary/summary_mass.png", "Lost / two-accepted per mass point (pooled over lifetimes); dashed lines: all "
            "samples.", hdr, [row(mlab(g), g) for g in masses] + [row("All samples", "overall")])
    cap.add("summary/summary_lifetime.png", "Lost / two-accepted per lifetime (pooled over masses).", hdr,
            [row("ctau %g mm" % num(g.split("_")[1]), g) for g in ctaus] + [row("All samples", "overall")])
    cap.add("summary/summary_sample.png", "Lost / two-accepted per sample, grouped by mass point, colored by lifetime.",
            hdr, [row(g[len("sample_"):], g) for g in samples])
    cap.add("overlay/<mass|lifetime>_<quantity>_<x>.png", "One curve per mass point (pooled over lifetimes) or per "
            "lifetime (pooled over masses). Quantity, each divided by two-accepted events: lostVtx = lost by slimming at "
            "the vertexed stage; vtxA = vertexed with displacedMuons; vtxB = vertexed with slimmedDisplacedMuons.")
    cap.write(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("analyze", "validate", "plot"))
    ap.add_argument("--input-dir")
    ap.add_argument("--output", required=True)
    ap.add_argument("--revision", default="unknown")
    ap.add_argument("--plot-dir")
    ap.add_argument("--style-dir")
    ap.add_argument("--expected-samples", type=int, default=72)
    ap.add_argument("--log", action="store_true")
    args = ap.parse_args()
    {"analyze": analyze, "validate": validate, "plot": plot}[args.mode](args)


if __name__ == "__main__":
    main()
