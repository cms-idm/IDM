#!/usr/bin/env python3
"""Independent check of DisplacedMuonDuplicateCleaner (kb/studies/displaced-muon-duplicate-cleaner.md).

Recomputes the kept set from raw displacedMuons and compares it, event by event, with the cleaner's
output, read either from an ntuple of scripts/ElectronNtuplizer_cfg.py (RecoDisplacedMuon_originalIndex,
nRecoDisplacedMuonRaw; events matched by run, lumi, event) or from the EDM output of
cleaner_only_cfg.py (read in lockstep with the raw files, since event IDs repeat across samples).
Also checks that no two kept muons share a TrackRef, counts muon pairs sharing each track type, events
with duplicates, chains and the non-transitive case, and which copy is kept for two-user shared inner
tracks (lower or higher index).

  python3 check_cleaner.py --aod <MiniAOD files in cmsRun order, or a file list> --ntuple <ntuple.root>
  python3 check_cleaner.py --aod <MiniAOD files in cmsRun order, or a file list> --edm <cleaned.root ...>
"""
import argparse
import collections
import sys

TYPES = ("inner", "outer", "global")


def refs(mu):
    """{type: (processIndex, productIndex, key)} for the non-null components."""
    out = {}
    for name, ref in zip(TYPES, (mu.innerTrack(), mu.outerTrack(), mu.globalTrack())):
        if ref.isNonnull():
            pid = ref.id()
            out[name] = (pid.processIndex(), pid.productIndex(), ref.key())
    return out


def expected_kept(muons, comp):
    # Rank: global track, then outer track, then lower index (sort key, best first).
    order = sorted(range(len(muons)), key=lambda i: (muons[i].globalTrack().isNull(),
                                                     muons[i].outerTrack().isNull(), i))
    kept = []
    for i in order:
        if not any(comp[i] & comp[k] for k in kept):
            kept.append(i)
    return sorted(kept)


def ntuple_results(path):
    import ROOT
    res = {}
    f = ROOT.TFile.Open(path)
    for t in f.Get("ntuples/outT"):
        res[(t.runNum, t.lumiSec, t.eventNum)] = (list(t.RecoDisplacedMuon_originalIndex),
                                                  int(t.nRecoDisplacedMuonRaw), int(t.nRecoDisplacedMuon),
                                                  int(t.RecoDisplacedMuon_available))
    return res


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--aod", required=True, nargs="+", help="raw MiniAOD files, or one text file list")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ntuple")
    group.add_argument("--edm", nargs="+")
    parser.add_argument("--muons", default="displacedMuons")
    parser.add_argument("--process", default="RECO")
    parser.add_argument("--cleanedProcess", default="DUPCLEAN")
    args = parser.parse_args()
    aod = args.aod
    if len(aod) == 1 and not aod[0].endswith(".root"):
        with open(aod[0]) as src:
            aod = [line.strip() for line in src if line.strip() and not line.startswith("#")]

    import ROOT
    ROOT.gSystem.Load("libFWCoreFWLite")
    ROOT.FWLiteEnabler.enable()
    from DataFormats.FWLite import Events, Handle

    raw_h = Handle("std::vector<reco::Muon>")
    if args.ntuple:
        ntuple = ntuple_results(args.ntuple)
        cleaned_iter = None
    else:
        ntuple = None
        idx_h, nraw_h, mu_h = Handle("std::vector<int>"), Handle("int"), Handle("std::vector<reco::Muon>")
        cleaned_iter = iter(Events(args.edm))

    n_ev = n_dup_ev = n_chain_ev = n_nontrans_ev = n_raw = n_drop = 0
    share_pairs = collections.Counter()
    two_user_inner = collections.Counter()
    bad = []
    for ev in Events(aod):
        aux = ev.eventAuxiliary()
        eid = (aux.run(), aux.luminosityBlock(), aux.event())
        if ntuple is not None:
            if eid not in ntuple:
                continue
            idx, nraw, n, avail = ntuple[eid]
        else:
            cev = next(cleaned_iter, None)
            if cev is None:
                bad.append((eid, ["no cleaned event left"]))
                break
            caux = cev.eventAuxiliary()
            ceid = (caux.run(), caux.luminosityBlock(), caux.event())
            if ceid != eid:
                bad.append((eid, ["cleaned event out of step: %s" % (ceid,)]))
                break
            cev.getByLabel("cleanedDisplacedMuons", "originalIndex", args.cleanedProcess, idx_h)
            cev.getByLabel("cleanedDisplacedMuons", "nRaw", args.cleanedProcess, nraw_h)
            cev.getByLabel("cleanedDisplacedMuons", "", args.cleanedProcess, mu_h)
            nraw_v = nraw_h.product()  # PyROOT may return the int itself or a one-element proxy
            nraw_v = nraw_v[0] if hasattr(nraw_v, "__getitem__") else nraw_v
            idx, nraw, n, avail = list(idx_h.product()), int(nraw_v), mu_h.product().size(), 1
        n_ev += 1
        ev.getByLabel(args.muons, "", args.process, raw_h)
        muons = list(raw_h.product())
        typed = [refs(mu) for mu in muons]
        comp = [set(r.values()) for r in typed]
        kept = expected_kept(muons, comp)
        keptset = set(kept)
        n_raw += len(muons)
        n_drop += len(muons) - len(kept)

        degree = [0] * len(muons)
        for i in range(len(muons)):
            for j in range(i + 1, len(muons)):
                if comp[i] & comp[j]:
                    common = [t for t in TYPES if t in typed[i] and typed[i][t] == typed[j].get(t)]
                    degree[i] += 1
                    degree[j] += 1
                    share_pairs["+".join(common) if common else "cross-type"] += 1
        n_dup_ev += any(degree)
        n_chain_ev += any(d >= 2 for d in degree)
        # Non-transitive case: a dropped B shares with two kept muons A and C (kept muons never share,
        # so A and C share nothing); transitive grouping would have dropped one of them.
        n_nontrans_ev += any(b not in keptset and sum(1 for k in kept if comp[b] & comp[k]) >= 2
                             for b in range(len(muons)))
        users = collections.defaultdict(list)
        for i, r in enumerate(typed):
            if "inner" in r:
                users[r["inner"]].append(i)
        for u in users.values():
            if len(u) == 2:
                lo, hi = u
                two_user_inner["lower kept" if lo in keptset and hi not in keptset else
                               "higher kept" if hi in keptset and lo not in keptset else "other"] += 1

        problems = []
        if avail != 1:
            problems.append("RecoDisplacedMuon_available=%d" % avail)
        if nraw != len(muons):
            problems.append("nRaw %d != raw %d" % (nraw, len(muons)))
        if n != len(idx):
            problems.append("n %d != len(originalIndex) %d" % (n, len(idx)))
        if idx != kept:
            problems.append("originalIndex %s != expected %s" % (idx, kept))
        if any(comp[a] & comp[b] for ai, a in enumerate(idx) for b in idx[ai + 1:] if a < len(comp) and b < len(comp)):
            problems.append("kept muons share a track")
        if problems:
            bad.append((eid, problems))

    leftover = 0 if cleaned_iter is None else sum(1 for _ in cleaned_iter)
    print("events compared: %d" % n_ev + (" (ntuple entries %d)" % len(ntuple) if ntuple is not None else
                                          " (cleaned events left over %d)" % leftover))
    print("raw muons: %d, dropped: %d" % (n_raw, n_drop))
    print("muon pairs sharing a track, by shared type(s): %s" % dict(share_pairs))
    print("events with duplicates: %d, with chains (a muon sharing with >= 2 others): %d" % (n_dup_ev, n_chain_ev))
    print("events where a dropped muon shares with two kept muons (non-transitive rule exercised): %d" % n_nontrans_ev)
    print("two-user shared inner tracks: %s" % dict(two_user_inner))
    print("events with mismatches: %d" % len(bad))
    for eid, problems in bad[:20]:
        print("  %s: %s" % (eid, "; ".join(problems)))
    missing = (len(ntuple) - n_ev) if ntuple is not None else leftover
    if missing:
        print("events not compared: %d" % missing)
    return 1 if bad or missing else 0


if __name__ == "__main__":
    sys.exit(main())
