#!/usr/bin/env python3
"""Dump displacedMuons inner-track quality and nearest charged gen particles for one event.

In CMSSW (cmsenv + voms proxy):
  python3 dump_event_tracks_gen.py --file root://cmseos.fnal.gov//store/.../Run3Summer22EEMiniAOD_3.root \
      --event 1:10003:2316
Gen candidates: status-1 charged packedGenParticles (hadrons included), nearest in dR to the
track momentum at its point of closest approach. Decay-in-flight muons are SIM-level and never
appear here; a matched pi/K is the hint for those.
"""
import argparse
import math


def dphi(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", required=True)
    parser.add_argument("--event", required=True, help="run:lumi:event")
    parser.add_argument("--muons", default="displacedMuons")
    parser.add_argument("--gen", default="packedGenParticles")
    parser.add_argument("--pruned", default="prunedGenParticles")
    parser.add_argument("--nearest", type=int, default=3)
    args = parser.parse_args()
    target = tuple(int(x) for x in args.event.split(":"))

    import ROOT
    ROOT.gSystem.Load("libFWCoreFWLite")
    ROOT.FWLiteEnabler.enable()
    from DataFormats.FWLite import Events, Handle

    muons_h = Handle("std::vector<reco::Muon>")
    gen_h = Handle("std::vector<pat::PackedGenParticle>")
    for ev in Events(args.file):
        aux = ev.eventAuxiliary()
        if (aux.run(), aux.luminosityBlock(), aux.event()) != target:
            continue
        ev.getByLabel(args.muons, muons_h)
        ev.getByLabel(args.gen, gen_h)
        gens = [g for g in gen_h.product() if g.charge() != 0]

        tracks = {}  # (productID, key) -> muon indices
        for i, mu in enumerate(muons_h.product()):
            ref = mu.innerTrack()
            if ref.isNonnull():
                tracks.setdefault((ref.id().id(), ref.key()), (ref, []))[1].append(i)

        for (pid, key), (ref, users) in sorted(tracks.items()):
            tk = ref.get()
            hp = tk.hitPattern()
            print(f"innerTrack ({pid},{key}) used by displacedMuons {users}")
            print(f"  q={tk.charge():+d} pt={tk.pt():.3f} eta={tk.eta():.3f} phi={tk.phi():.3f} "
                  f"dxy={tk.dxy():.3f} dz={tk.dz():.3f} cm  algo={tk.algoName()}")
            print(f"  validHits={tk.numberOfValidHits()} lostHits={tk.numberOfLostHits()} "
                  f"pixelLayers={hp.pixelLayersWithMeasurement()} "
                  f"trackerLayers={hp.trackerLayersWithMeasurement()} "
                  f"chi2/ndof={tk.normalizedChi2():.2f} "
                  f"highPurity={int(tk.quality(ROOT.reco.TrackBase.highPurity))}")
            if tk.extra().isAvailable():
                ip, op = tk.innerPosition(), tk.outerPosition()
                print(f"  first hit rho={ip.rho():.1f} z={ip.z():.1f}  last hit rho={op.rho():.1f} z={op.z():.1f} cm")
                # Same predicate as TrackDetectorAssociator::crossedIP, on the reconstructed endpoint states
                im, om = tk.innerMomentum(), tk.outerMomentum()
                r_dot_p_in = ip.x() * im.x() + ip.y() * im.y()
                r_dot_p_out = op.x() * om.x() + op.y() * om.y()
                crossed = ip.rho() > 3 and op.rho() > 3 and ((r_dot_p_in < 0) != (r_dot_p_out < 0))
                print(f"  crossedIP inputs: first-hit r.p={r_dot_p_in:+.2f} last-hit r.p={r_dot_p_out:+.2f} "
                      f"(cm*GeV) -> crossedIP={int(crossed)}  first-hit phi={ip.phi():.3f} last-hit phi={op.phi():.3f}")
            near = sorted(gens, key=lambda g: math.hypot(g.eta() - tk.eta(), dphi(g.phi(), tk.phi())))
            for g in near[:args.nearest]:
                dr = math.hypot(g.eta() - tk.eta(), dphi(g.phi(), tk.phi()))
                mother = g.mother(0)
                mother_id = mother.pdgId() if mother else 0
                print(f"    gen dR={dr:.3f} pdgId={g.pdgId()} q={g.charge():+d} pt={g.pt():.3f} "
                      f"eta={g.eta():.3f} phi={g.phi():.3f} mother={mother_id}")
        pruned_h = Handle("std::vector<reco::GenParticle>")
        ev.getByLabel(args.pruned, pruned_h)
        print("all prunedGenParticles muons (any status):")
        for i, p in enumerate(pruned_h.product()):
            if abs(p.pdgId()) == 13:
                mother = p.mother(0)
                print(f"  [{i}] pdgId={p.pdgId()} status={p.status()} pt={p.pt():.3f} eta={p.eta():.3f} "
                      f"phi={p.phi():.3f} vtx=({p.vx():.2f},{p.vy():.2f},{p.vz():.2f}) "
                      f"mother={mother.pdgId() if mother else 0}")
        return
    raise RuntimeError(f"Event {args.event} not found in {args.file}")


if __name__ == "__main__":
    main()
