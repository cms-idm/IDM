#!/usr/bin/env python3
"""METNoMu HLT fraction of chain-A events (study note: kb/studies/metnomu-trigger-chain-a.md; LPC only).

python3 metnomu_chain_a.py --input-dir MET_PROXY_RUN/inputs --input-revision <sha> --revision <sha> --run-dir RUN \
    --proxy-root MET_PROXY_RUN/met_proxy_v2.root --trigger-root TRIGGER_RUN/trigger_counts.root [--workers 4] [--test]
Full runs check: chain-A and base counts equal met_proxy chains/chain_A_PF__<pair>; per-path counts over all events equal
trigger_counts.root overall/accepted.

Chain A exactly as met_proxy.py (imported derive()): base (two accepted gen signal muons, valid G > I > DSA hierarchy
vertex) -> chi2/ndof < 4 -> reco dR(mu, mu) < 0.9 -> finite |dphi(Type-1 PF MET, reco pT_mumu)| < 0.5.
Triggers: TriggerResults::HLT accept bits of every path matching PATH_RE, required to equal EXPECTED (17 names);
S14 = S10 (pure PFMETNoMu) + MU3 (Mu3er1p5_PFJet100er2p5) = headline set (user 2026-10-22); S17 adds the 3 VBF-type
Mu+DiPFJet+PFMETNoMu60 paths, context only. A path absent from an event's menu counts as not fired (counted separately).
Join: (run, lumi, event) within each sample, unique and one-to-one. Raw, unweighted event counts; pooling unweighted.
"""
import argparse
import concurrent.futures
import glob
from pathlib import Path
import re
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "met_dphi"))
import met_proxy  # noqa: E402

PATH_RE = re.compile(r"^HLT_.*PFMETNoMu\d+.*_v\d+$")  # every METNoMu path; must equal EXPECTED
S10 = ["HLT_PFMETNoMu%s_PFMHTNoMu%s_IDTight%s" % (x, x, v) for x, v in (
    ("100", "_PFHT60_v11"), ("110", "_v22"), ("110", "_FilterHF_v2"), ("120", "_v22"), ("120", "_FilterHF_v2"),
    ("120", "_PFHT60_v11"), ("130", "_v21"), ("130", "_FilterHF_v2"), ("140", "_v21"), ("140", "_FilterHF_v2"))]
MU3 = ["HLT_Mu3er1p5_PFJet100er2p5_PFMETNoMu%d_PFMHTNoMu%d_IDTight_v4" % (x, x) for x in (70, 80, 90, 100)]
VBF = ["HLT_Mu10_TrkIsoVVL_DiPFJet40_DEta3p5_MJJ750_HTT350_PFMETNoMu60_v17",  # context only, not in S14
       "HLT_Mu4_TrkIsoVVL_DiPFJet90_40_DEta3p5_MJJ750_HTT300_PFMETNoMu60_v17",
       "HLT_Mu8_TrkIsoVVL_DiPFJet40_DEta3p5_MJJ750_HTT300_PFMETNoMu60_v18"]
EXPECTED = sorted(S10 + MU3 + VBF)
POPS = ("all", "base", "chainA")


def read_hlt(job):
    """Per event of the sample's MINIAOD: (run, lumi, event), accept bits of PATH_RE paths, absent counter."""
    import ROOT
    ROOT.gROOT.SetBatch(True)
    ROOT.gSystem.Load("libFWCoreFWLite")
    ROOT.FWLiteEnabler.enable()
    from DataFormats.FWLite import Events, Handle
    sample, files = job
    handle = Handle("edm::TriggerResults")
    keys, bits, names, absent, processed = [], [], None, 0, []
    try:
        for filename in files:
            events = Events(filename)
            expected, seen = int(events.size()), 0
            menu, index = None, None
            for event in events:
                event.getByLabel("TriggerResults", "", "HLT", handle)
                if not handle.isValid():
                    raise RuntimeError("Missing TriggerResults::HLT: " + filename)
                res = handle.product()
                if menu is None or res.parameterSetID() != menu:
                    tn = event.object().triggerNames(res)
                    menu_names = [str(tn.triggerName(i)) for i in range(tn.size())]
                    found = sorted(n for n in menu_names if PATH_RE.match(n))
                    if found != EXPECTED:
                        raise RuntimeError("Unexpected METNoMu path set in %s: %r" % (filename, found))
                    names = found
                    index = [menu_names.index(n) for n in names]
                    menu = res.parameterSetID()
                aux = event.object().id()
                keys.append((aux.run(), aux.luminosityBlock(), aux.event()))
                row = [bool(res.accept(i)) if i >= 0 else False for i in index]
                absent += sum(i < 0 for i in index)
                bits.append(row)
                seen += 1
            if seen != expected or seen == 0:
                raise RuntimeError("Incomplete/empty input: %s: %d/%d" % (filename, seen, expected))
            processed.append((filename, seen))
        return sample, processed, names, np.array(keys, dtype=np.int64), np.array(bits, dtype=bool), absent, None
    except Exception:
        return sample, processed, names, None, None, absent, traceback.format_exc()


def chain_a(path, input_revision):
    """Chain-A, base masks and pair index for one PreselectedVertexing output, plus its event keys."""
    import uproot
    with uproot.open(path) as f:
        d = f["preselectedVertexing"]
        sample = str(d["sample"].member("fTitle"))
        if str(d["hierarchyOrder"].member("fTitle")) != met_proxy.ORDER_EXPECTED or \
                str(d["revision"].member("fTitle")) != input_revision:
            raise RuntimeError("Unexpected hierarchyOrder/revision: %s" % path)
        a = d["events"].arrays(met_proxy.BRANCHES + ["run", "lumi", "event"], library="np")
    c = {k: (np.array([max(v) if len(v) == 2 else np.nan for v in a[k]], dtype=float) if k == "sigD0"
             else np.asarray(a[k], dtype=float)) for k in met_proxy.BRANCHES}
    q, pair, m = met_proxy.derive(c)
    s1 = m["base"] & (q["chi2ndof"] < met_proxy.CHI2_CUT)
    s2 = s1 & (q["recoDR"] < met_proxy.DR_CUT)
    ca = s2 & np.isfinite(q["PF_dphiMM"]) & (np.abs(q["PF_dphiMM"]) < met_proxy.DPHI_CUT)
    keys = np.stack([np.asarray(a[k], dtype=np.int64) for k in ("run", "lumi", "event")], axis=1)
    return sample, keys, m["base"], ca, pair


def join(vkeys, hkeys, sample):
    """Index into hkeys for each vkeys row; one-to-one or raise."""
    for k, lab in ((vkeys, "vertexing"), (hkeys, "MINIAOD")):
        if len(np.unique(k, axis=0)) != len(k):
            raise RuntimeError("Duplicate event keys in %s for %s" % (lab, sample))
    if len(vkeys) != len(hkeys):
        raise RuntimeError("Event count mismatch for %s: %d vs %d" % (sample, len(vkeys), len(hkeys)))
    lookup = {tuple(r): i for i, r in enumerate(hkeys)}
    idx = np.array([lookup.get(tuple(r), -1) for r in vkeys])
    if (idx < 0).any():
        raise RuntimeError("%d vertexing events not found in MINIAOD for %s" % ((idx < 0).sum(), sample))
    return idx


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--input-revision", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--proxy-root", required=True)
    ap.add_argument("--trigger-root", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--test", action="store_true", help="Runtime test: first sample only")
    args = ap.parse_args()
    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    out = run / "metnomu_chain_a.root"
    if out.exists():
        raise RuntimeError("Refusing to overwrite " + str(out))
    repo = Path(__file__).resolve().parents[2]
    jobs = []
    for src in sorted(glob.glob(str(repo / "AODSkimmer/fileLists/signal/2022EE/iDMmu/*_flist.txt"))):
        files = [l.strip() for l in Path(src).read_text().splitlines() if l.strip() and not l.lstrip().startswith("#")]
        if len(files) != 4 or any("260928_" not in p for p in files):
            raise RuntimeError("Unexpected file list " + src)
        jobs.append((Path(src).name.removesuffix("_flist.txt"), files))
    if len(jobs) != 72:
        raise RuntimeError("Expected 72 samples")
    vpaths = {Path(p).stem: p for p in glob.glob(str(Path(args.input_dir) / "*.root"))}
    if args.test:
        jobs = jobs[:1]
    start = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        hlt = list(pool.map(read_hlt, jobs))
    failed = [h[0] for h in hlt if h[6]]
    for h in hlt:
        if h[6]:
            print(h[6], flush=True)
    if failed:
        raise RuntimeError("Failed samples: %r; no output" % failed)
    if any(h[5] for h in hlt):
        raise RuntimeError("Absent METNoMu paths in some events; no output")
    names = hlt[0][2]
    if names != EXPECTED or any(h[2] != names for h in hlt):
        raise RuntimeError("Unexpected METNoMu path set: %r" % names)
    s10 = np.array([n in S10 for n in names])
    s14 = np.array([n in S10 or n in MU3 for n in names])

    rows = []  # per sample: sample, pair, pop masks, bits
    for (sample, processed, _, hkeys, bits, absent, _) in hlt:
        if sample not in vpaths:
            raise RuntimeError("No vertexing output for " + sample)
        vs, vkeys, base, ca, pair = chain_a(vpaths[sample], args.input_revision)
        if vs != sample:
            raise RuntimeError("Sample name mismatch %s vs %s" % (vs, sample))
        idx = join(vkeys, hkeys, sample)
        rows.append(dict(sample=sample, files=processed, absent=absent, pair=pair,
                         masks={"all": np.ones(len(base), bool), "base": base, "chainA": ca}, bits=bits[idx]))
        print("SAMPLE %s events=%d base=%d chainA=%d absent=%d" % (sample, len(base), base.sum(), ca.sum(), absent),
              flush=True)
    total = sum(len(r["pair"]) for r in rows)
    if not args.test and (total != 720000 or any(len(r["pair"]) != 10000 for r in rows)):
        raise RuntimeError("Incomplete coverage: %d events" % total)

    def counts(rsel, keyfn):
        """{group: {pop: [N, OR14, OR10, OR17, per path...]}}"""
        res = {}
        for r in rsel:
            for g, gmask in keyfn(r):
                for pop in POPS:
                    sel = r["masks"][pop] & gmask
                    b = r["bits"][sel]
                    v = [sel.sum(), b[:, s14].any(axis=1).sum(), b[:, s10].any(axis=1).sum(), b.any(axis=1).sum()] + list(b.sum(axis=0))
                    acc = res.setdefault(g, {}).setdefault(pop, np.zeros(len(v), np.int64))
                    acc += np.array(v, np.int64)
        return res

    def mass(s):
        return re.match(r"(Mchi-[^_]+_dMchi-[^_]+)_", s).group(1)

    def life(s):
        return re.search(r"(ctau-[^_]+)$", s).group(1)

    groups = {}
    groups.update(counts(rows, lambda r: [("overall", np.ones(len(r["pair"]), bool))]))
    groups.update(counts(rows, lambda r: [("pair/" + p, r["pair"] == i) for i, p in enumerate(met_proxy.PAIRS)]))
    groups.update(counts(rows, lambda r: [("mass/" + mass(r["sample"]), np.ones(len(r["pair"]), bool))]))
    groups.update(counts(rows, lambda r: [("lifetime/" + life(r["sample"]), np.ones(len(r["pair"]), bool))]))
    groups.update(counts(rows, lambda r: [("sample/" + r["sample"], np.ones(len(r["pair"]), bool))]))

    import uproot
    labels = ["events", "OR of S14", "OR of S10", "OR of S17"] + names
    if not args.test:
        with uproot.open(args.proxy_root) as fp:
            for i, p in enumerate(("all",) + met_proxy.PAIRS):
                ref = fp["chains/chain_A_PF__" + p].values()
                g = groups["overall" if p == "all" else "pair/" + p]
                if int(ref[0]) != int(g["base"][0]) or int(ref[3]) != int(g["chainA"][0]):
                    raise RuntimeError("Chain A mismatch with met_proxy for %s: %s vs base %d chainA %d"
                                       % (p, ref, g["base"][0], g["chainA"][0]))
        with uproot.open(args.trigger_root) as ft:
            h = ft["overall/accepted"]
            ref = dict(zip(h.axis().labels(), h.values()))
            for lab, x in zip(names, groups["overall"]["all"][4:]):
                if int(ref[lab]) != int(x):
                    raise RuntimeError("Path count mismatch %s: %d vs trigger_counts %d" % (lab, x, ref[lab]))
        print("VALIDATED: chain A and base match met_proxy per pair type; 17 path counts match trigger_counts.root")

    import ROOT
    ROOT.gROOT.SetBatch(True)
    labels = ["events", "OR of S14", "OR of S10", "OR of S17"] + names
    f = ROOT.TFile.Open(str(out), "CREATE")
    ROOT.TNamed("definitions", __doc__).Write()
    ROOT.TNamed("analysisRevision", args.revision).Write()
    ROOT.TNamed("inputRevision", args.input_revision).Write()
    ROOT.TNamed("inputs", "\n".join("%s\t%s\t%d" % (r["sample"], p, n) for r in rows for p, n in r["files"])).Write()
    ROOT.TNamed("absentPathEvents", str(sum(r["absent"] for r in rows))).Write()
    for g, pops in groups.items():
        f.mkdir(g)
        f.cd(g)
        for pop, v in pops.items():
            h = ROOT.TH1D(pop, "%s: %s;;Events" % (g, pop), len(v), 0, len(v))
            for i, (lab, x) in enumerate(zip(labels, v), 1):
                h.GetXaxis().SetBinLabel(i, lab)
                h.SetBinContent(i, float(x))
                h.SetBinError(i, float(x) ** 0.5)
            h.Write()
        f.cd()
    f.Close()

    from scipy.stats import beta

    def cp(k, n):
        if n == 0:
            return float("nan"), float("nan")
        lo = beta.ppf(0.16, k, n - k + 1) if k > 0 else 0.0
        hi = beta.ppf(0.84, k + 1, n - k) if k < n else 1.0
        return lo, hi

    print("COMPLETE events=%d absentPathEvents=%d elapsed=%.1fs root=%s" %
          (total, sum(r["absent"] for r in rows), time.monotonic() - start, out))
    for g in ["overall"] + ["pair/" + p for p in met_proxy.PAIRS] + sorted(k for k in groups if k.startswith(("mass/", "lifetime/"))):
        for pop in POPS:
            v = groups[g][pop]
            n = int(v[0])
            txt = " ".join("%s %d (%.2f%%)" % (lab, x, 100.0 * x / n if n else float("nan")) for lab, x in zip(labels[1:4], v[1:4]))
            lo, hi = cp(int(v[1]), n)
            print("%-28s %-7s N=%-7d %s  [S14 68%% CP %.2f-%.2f%%]" % (g, pop, n, txt, 100 * lo, 100 * hi))
    for pop in POPS:
        v = groups["overall"][pop]
        for lab, x in zip(labels[4:], v[4:]):
            print("PATH %-7s %-62s %7d / %7d = %.2f%%" % (pop, lab, x, v[0], 100.0 * x / v[0] if v[0] else float("nan")))


if __name__ == "__main__":
    main()
