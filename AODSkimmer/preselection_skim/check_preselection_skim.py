#!/usr/bin/env python3
"""Independently check an MC MINIAOD skim against its ordered original inputs, using CMSSW FWLite.

python3 check_preselection_skim.py --input <list.txt|input.root> --output <skim.root> [--max-events N]
Checks the passing event subsequence, every flag, independent duplicate removal, cleaner TrackRefs,
raw/clean muon kinematics, referenced track kinematics/availability, and original event branch inventory.
Only use on the explicitly authorized signal MC; no collision data. A bounded check is a test, not
production coverage. max-events must match the cmsRun input event limit (default: all).
"""
import argparse


COMPONENTS = ("inner", "outer", "global")
FLAGS = [c + field for c in COMPONENTS for field in ("Present", "PassID", "InEta")]
FLAGS += ["keptAfterCleaning", "preselected", "preselectedEta"]


def inputs(path):
    if path.endswith(".root"):
        return [path]
    with open(path) as source:
        return [line.strip() for line in source if line.strip() and not line.lstrip().startswith("#")]


def refs(muon):
    return (muon.innerTrack(), muon.outerTrack(), muon.globalTrack())


def key(ref):
    if ref.isNull():
        return None
    return (int(ref.id().processIndex()), int(ref.id().productIndex()), int(ref.key()))


def expected(muons):
    import math
    result = {name: [0] * len(muons) for name in FLAGS}
    identities = [{key(ref) for ref in refs(muon) if ref.isNonnull()} for muon in muons]
    ranked = sorted(range(len(muons)), key=lambda i: (muons[i].globalTrack().isNull(),
                                                     muons[i].outerTrack().isNull(), i))
    survivors = []
    for i in ranked:
        if all(identities[i].isdisjoint(identities[j]) for j in survivors):
            survivors.append(i)
            result["keptAfterCleaning"][i] = 1
    for i, muon in enumerate(muons):
        same_component = False
        any_id = False
        for name, ref in zip(COMPONENTS, refs(muon)):
            if ref.isNull():
                continue
            if not ref.isAvailable():
                raise RuntimeError("Unavailable original component: %s[%d]" % (name, i))
            track = ref.get()
            if name == "inner":
                passed = bool(muon.isTrackerMuon())
            elif name == "global":
                passed = bool(muon.isGlobalMuon())
            else:
                csc = int(track.hitPattern().numberOfValidMuonCSCHits())
                dt = int(track.hitPattern().numberOfValidMuonDTHits())
                passed = (csc + dt > 12 and (csc > 0 or dt > 18) and track.normalizedChi2() < 2.5
                          and track.pt() > 0 and track.ptError() / track.pt() < 1)
            in_eta = math.isfinite(track.eta()) and abs(track.eta()) < 2.4
            result[name + "Present"][i] = 1
            result[name + "PassID"][i] = int(passed)
            result[name + "InEta"][i] = int(in_eta)
            any_id |= passed
            same_component |= passed and in_eta
        result["preselected"][i] = int(result["keptAfterCleaning"][i] and any_id)
        result["preselectedEta"][i] = int(result["keptAfterCleaning"][i] and same_component)
    return sorted(survivors), result


def event_id(event):
    aux = event.eventAuxiliary()
    return (int(aux.run()), int(aux.luminosityBlock()), int(aux.event()))


def check_muons(original, copied):
    import math
    if len(original) != len(copied):
        raise RuntimeError("Muon collection length changed")
    def same(a, b):
        return a == b or (math.isnan(a) and math.isnan(b))
    for a, b in zip(original, copied):
        for field in ("pt", "eta", "phi", "energy", "charge", "type"):
            if not same(getattr(a, field)(), getattr(b, field)()):
                raise RuntimeError("Changed original muon field " + field)
        for ra, rb in zip(refs(a), refs(b)):
            if key(ra) != key(rb):
                raise RuntimeError("Changed muon TrackRef identity")
            if ra.isNonnull():
                if not rb.isAvailable():
                    raise RuntimeError("Output TrackRef target unavailable")
                for field in ("pt", "eta", "phi", "ptError", "normalizedChi2"):
                    if not same(getattr(ra.get(), field)(), getattr(rb.get(), field)()):
                        raise RuntimeError("Changed referenced track field " + field)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-events", type=int, default=-1)
    args = parser.parse_args()
    import ROOT
    ROOT.gSystem.Load("libFWCoreFWLite")
    ROOT.FWLiteEnabler.enable()
    from DataFormats.FWLite import Events, Handle

    source_files = inputs(args.input)
    def branches(path):
        source = ROOT.TFile.Open(path)
        if not source or source.IsZombie():
            raise RuntimeError("Cannot read " + path)
        tree = source.Get("Events")
        if not tree:
            raise RuntimeError("Missing Events tree in " + path)
        names = {branch.GetName() for branch in tree.GetListOfBranches()}
        source.Close()
        return names
    output_branches = branches(args.output)
    for path in source_files:
        missing = branches(path) - output_branches
        if missing:
            raise RuntimeError("Original event branches dropped: " + repr(sorted(missing)))

    raw_handle = Handle("std::vector<reco::Muon>")
    copied_handle = Handle("std::vector<reco::Muon>")
    clean_handle = Handle("std::vector<reco::Muon>")
    index_handle = Handle("std::vector<int>")
    handles = {flag: Handle("std::vector<int>") for flag in FLAGS}
    output = iter(Events([args.output]))
    processed = retained = 0
    for event in Events(source_files):
        if args.max_events >= 0 and processed >= args.max_events:
            break
        processed += 1
        event.getByLabel("displacedMuons", "", "RECO", raw_handle)
        original = list(raw_handle.product())
        survivors, flags = expected(original)
        if not any(flags["preselected"]):
            continue
        copied_event = next(output, None)
        if copied_event is None or event_id(event) != event_id(copied_event):
            raise RuntimeError("Output event subsequence mismatch at input event " + str(event_id(event)))
        for name, handle in handles.items():
            copied_event.getByLabel("displacedMuonPreselection", name, "PRESKIM", handle)
            if list(handle.product()) != flags[name]:
                raise RuntimeError("Flag mismatch for " + name + " at " + str(event_id(event)))
        copied_event.getByLabel("cleanedDisplacedMuons", "originalIndex", "PRESKIM", index_handle)
        if list(index_handle.product()) != survivors:
            raise RuntimeError("Cleaner survivor mismatch")
        copied_event.getByLabel("displacedMuons", "", "RECO", copied_handle)
        check_muons(original, list(copied_handle.product()))
        copied_event.getByLabel("cleanedDisplacedMuons", "", "PRESKIM", clean_handle)
        check_muons([original[i] for i in survivors], list(clean_handle.product()))
        retained += 1
    if next(output, None) is not None:
        raise RuntimeError("Unexpected extra output events")
    print("PASS: input_events=%d retained_events=%d; all flags, event order, original event branches, "
          "raw/clean muons and TrackRef targets checked" % (processed, retained))


if __name__ == "__main__":
    main()
