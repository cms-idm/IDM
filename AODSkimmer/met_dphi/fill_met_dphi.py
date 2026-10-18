"""Per-event tree for Delta phi(MET, gen signal dimuon) from MINIAOD (FWLite; CMSSW environment).

  python3 fill_met_dphi.py FLIST SAMPLE OUT.root [--max-events N]

FLIST: a file list (one MINIAOD path per line) or a single .root file. One entry per event, all events
(nothing dropped; the plotter selects). Branches:
  run, lumi, event
  nSig        number of signal muons (|pdgId| 13, status 1, last copy, first different-PDG ancestor
              |pdgId| 1000023; same as ElectronSkimmer.cc / DimuonVertexTruth.cc); nSigMinus, nSigPlus
  valid       exactly two signal muons, one mu- and one mu+ (the dimuon below is filled only then)
  mu{0,1}_{pt,eta,phi,vxy,vz,d0}  0 = mu-, 1 = mu+; vxy/vz = gen production vertex; d0 = |vx sin(phi) -
              vy cos(phi)|, straight line to (0,0)
  mm_pt, mm_phi              vector sum of the two signal muons
  pfmet_pt, pfmet_phi        slimmedMETs[0] corPt/corPhi(Type1)
  genmet_pt, genmet_phi      genMetTrue[0]
Sentinel -999 for unfilled values (invalid dimuon, empty MET collection).
"""
import argparse
import math
import sys
import time
from array import array

import ROOT

ROOT.gROOT.SetBatch(True)
from DataFormats.FWLite import Events, Handle

SENT = -999.0
FLOATS = ["mm_pt", "mm_phi", "pfmet_pt", "pfmet_phi", "genmet_pt", "genmet_phi"] + [
    f"mu{i}_{v}" for i in (0, 1) for v in ("pt", "eta", "phi", "vxy", "vz", "d0")]
INTS = ["nSig", "nSigMinus", "nSigPlus", "valid"]


def first_different_mother(p):
    cur, seen = p, 0
    while cur and seen < 1000:
        seen += 1
        if cur.numberOfMothers() == 0:
            return None
        mom = cur.mother(0)
        if not mom:
            return None
        if mom.pdgId() != p.pdgId():
            return mom
        cur = mom
    return None


def is_signal_muon(p):
    if abs(p.pdgId()) != 13 or p.status() != 1 or not p.isLastCopy():
        return False
    mom = first_different_mother(p)
    return mom is not None and abs(mom.pdgId()) == 1000023


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("flist")
    ap.add_argument("sample")
    ap.add_argument("out")
    ap.add_argument("--max-events", type=int, default=-1)
    args = ap.parse_args()
    if args.flist.endswith(".root"):
        files = [args.flist]
    else:
        files = [l.strip() for l in open(args.flist) if l.strip() and not l.startswith("#")]

    fout = ROOT.TFile(args.out, "RECREATE")
    ROOT.TNamed("definitions", __doc__).Write()
    ROOT.TNamed("sample", args.sample).Write()
    ROOT.TNamed("inputs", "\n".join(files)).Write()
    t = ROOT.TTree("events", "one entry per event")
    buf = {}
    for k in ("run", "lumi", "event"):
        buf[k] = array("L", [0])
        t.Branch(k, buf[k], f"{k}/l")
    for k in INTS:
        buf[k] = array("i", [0])
        t.Branch(k, buf[k], f"{k}/I")
    for k in FLOATS:
        buf[k] = array("f", [0])
        t.Branch(k, buf[k], f"{k}/F")

    hGen, hMet, hGenMet = (Handle("std::vector<reco::GenParticle>"), Handle("std::vector<pat::MET>"),
                           Handle("std::vector<reco::GenMET>"))
    type1 = ROOT.pat.MET.Type1
    n, t0 = 0, time.time()
    for fn in files:
        # One file at a time so a failing input is named; any failure stops the run (never skipped silently).
        for ev in Events(fn):
            if 0 <= args.max_events <= n:
                break
            aux = ev.eventAuxiliary()
            buf["run"][0], buf["lumi"][0], buf["event"][0] = aux.run(), aux.luminosityBlock(), aux.event()
            for k in FLOATS:
                buf[k][0] = SENT
            ev.getByLabel("prunedGenParticles", hGen)
            sig = [p for p in hGen.product() if is_signal_muon(p)]
            minus = [p for p in sig if p.pdgId() == 13]
            plus = [p for p in sig if p.pdgId() == -13]
            buf["nSig"][0], buf["nSigMinus"][0], buf["nSigPlus"][0] = len(sig), len(minus), len(plus)
            valid = len(sig) == 2 and len(minus) == 1 and len(plus) == 1
            buf["valid"][0] = int(valid)
            if valid:
                for i, p in enumerate((minus[0], plus[0])):
                    buf[f"mu{i}_pt"][0], buf[f"mu{i}_eta"][0], buf[f"mu{i}_phi"][0] = p.pt(), p.eta(), p.phi()
                    buf[f"mu{i}_vxy"][0], buf[f"mu{i}_vz"][0] = math.hypot(p.vx(), p.vy()), p.vz()
                    buf[f"mu{i}_d0"][0] = abs(p.vx() * math.sin(p.phi()) - p.vy() * math.cos(p.phi()))
                px = minus[0].px() + plus[0].px()
                py = minus[0].py() + plus[0].py()
                buf["mm_pt"][0], buf["mm_phi"][0] = math.hypot(px, py), math.atan2(py, px)
            ev.getByLabel("slimmedMETs", hMet)
            if hMet.product().size() > 0:
                m = hMet.product()[0]
                buf["pfmet_pt"][0], buf["pfmet_phi"][0] = m.corPt(type1), m.corPhi(type1)
            ev.getByLabel("genMetTrue", hGenMet)
            if hGenMet.product().size() > 0:
                g = hGenMet.product()[0]
                buf["genmet_pt"][0], buf["genmet_phi"][0] = g.pt(), g.phi()
            t.Fill()
            n += 1
        if 0 <= args.max_events <= n:
            break
        print(f"{args.sample}: {fn} done, {n} events, {time.time() - t0:.0f} s", flush=True)
    fout.cd()
    t.Write()
    fout.Close()
    print(f"{args.sample}: {n} events in {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
