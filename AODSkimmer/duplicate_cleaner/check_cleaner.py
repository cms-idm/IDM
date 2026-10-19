#!/usr/bin/env python3
"""Independent check of DisplacedMuonDuplicateCleaner (kb/studies/displaced-muon-duplicate-cleaner.md).

Recomputes the kept set from raw displacedMuons in a MiniAOD file and compares it, event by event
(run, lumi, event), with RecoDisplacedMuon_originalIndex and nRecoDisplacedMuonRaw in the ntuple made
from the same file by scripts/ElectronNtuplizer_cfg.py. Also checks that no two kept muons share a
TrackRef and counts events with duplicates and with chains (a muon sharing with two or more others).

  python3 check_cleaner.py --aod <MiniAOD.root> [more MiniAOD files] --ntuple <ntuple.root>

Every ntuple event must be found in the --aod files; extra AOD events are ignored.
"""
import argparse
import sys


def keys(mu):
    out = set()
    for ref in (mu.innerTrack(), mu.outerTrack(), mu.globalTrack()):
        if ref.isNonnull():
            pid = ref.id()
            out.add((pid.processIndex(), pid.productIndex(), ref.key()))
    return out


def expected_kept(muons):
    comp = [keys(mu) for mu in muons]
    # Rank: global track, then outer track, then lower index (sort key, best first).
    order = sorted(range(len(muons)), key=lambda i: (muons[i].globalTrack().isNull(),
                                                     muons[i].outerTrack().isNull(), i))
    kept = []
    for i in order:
        if not any(comp[i] & comp[k] for k in kept):
            kept.append(i)
    return sorted(kept), comp


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--aod", required=True, nargs="+")
    parser.add_argument("--ntuple", required=True)
    parser.add_argument("--muons", default="displacedMuons")
    parser.add_argument("--process", default="RECO")
    args = parser.parse_args()

    import ROOT
    ROOT.gSystem.Load("libFWCoreFWLite")
    ROOT.FWLiteEnabler.enable()
    from DataFormats.FWLite import Events, Handle

    ntuple = {}
    f = ROOT.TFile.Open(args.ntuple)
    tree = f.Get("ntuples/outT")
    for t in tree:
        ntuple[(t.runNum, t.lumiSec, t.eventNum)] = (list(t.RecoDisplacedMuon_originalIndex),
                                                     int(t.nRecoDisplacedMuonRaw), int(t.nRecoDisplacedMuon),
                                                     int(t.RecoDisplacedMuon_available))

    handle = Handle("std::vector<reco::Muon>")
    n_ev = n_dup_ev = n_chain_ev = n_nontrans_ev = n_raw = n_drop = 0
    bad = []
    for ev in Events(args.aod):
        aux = ev.eventAuxiliary()
        eid = (aux.run(), aux.luminosityBlock(), aux.event())
        if eid not in ntuple:
            continue
        n_ev += 1
        ev.getByLabel(args.muons, "", args.process, handle)
        muons = list(handle.product())
        kept, comp = expected_kept(muons)
        n_raw += len(muons)
        n_drop += len(muons) - len(kept)
        degree = [sum(1 for j in range(len(muons)) if j != i and comp[i] & comp[j]) for i in range(len(muons))]
        n_dup_ev += any(degree)
        n_chain_ev += any(d >= 2 for d in degree)
        # Non-transitive case: a dropped B shares with two kept muons A and C (kept muons never share,
        # so A and C share nothing); transitive grouping would have dropped one of them.
        keptset = set(kept)
        n_nontrans_ev += any(b not in keptset and sum(1 for k in kept if comp[b] & comp[k]) >= 2
                             for b in range(len(muons)))
        idx, nraw, n, avail = ntuple[eid]
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

    print("events compared: %d (ntuple entries %d)" % (n_ev, len(ntuple)))
    print("raw muons: %d, dropped: %d" % (n_raw, n_drop))
    print("events with duplicates: %d, with chains (a muon sharing with >= 2 others): %d" % (n_dup_ev, n_chain_ev))
    print("events where a dropped muon shares with two kept muons (non-transitive rule exercised): %d" % n_nontrans_ev)
    print("events with mismatches: %d" % len(bad))
    if n_ev != len(ntuple):
        print("ntuple events not found in --aod: %d" % (len(ntuple) - n_ev))
    for eid, problems in bad[:20]:
        print("  %s: %s" % (eid, "; ".join(problems)))
    return 1 if bad or n_ev != len(ntuple) else 0


if __name__ == "__main__":
    sys.exit(main())
