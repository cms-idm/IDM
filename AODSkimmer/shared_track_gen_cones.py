#!/usr/bin/env python3
"""Nearby charged gen candidates around each exactly shared displacedMuons innerTrack.

In CMSSW (cmsenv + voms proxy):
  python3 shared_track_gen_cones.py --input shared_inner_track_muon_locations.csv \
      --output shared_track_gen_cones.csv

Convention: the track's eta/phi are its momentum direction at its stored reference point (closest
approach to the beam spot); its dxy/dz are taken relative to the beam spot. Each status-1 charged
packedGenParticle is put on the same convention: closest approach of its ideal helix (uniform
Bz = +3.8 T, no material) to the beam line through the beam spot (tilt ignored). eta is unchanged
along that helix; phi, dxy and dz are recomputed. packedGenParticles carry no vertex, so the
production vertex is borrowed from the closest status-1 prunedGenParticles entry with the same
pdgId, dR < 0.01 and pT within 2% (n_pruned_match counts how many qualified); without one the
vertex is unknown and the particle is placed at the generated hard-scatter vertex (taken from the
first prunedGenParticles entry, the beam proton; "unknown-vertex-at-gen-PV").
The gen mother is the first ancestor kept in prunedGenParticles, not necessarily the direct parent.
These are nearby gen candidates, not truth matches. The MiniAOD gen collections hold only the hard
scatter: pileup collisions and GEANT secondaries (decays in flight, material interactions) are
absent, so an empty cone does not mean a fake track.
"""
import argparse
import csv
import math
from collections import Counter, defaultdict

CONES = (0.06, 0.1, 0.3)
R_CM_PER_GEV = 100. / (0.299792458 * 3.8)  # helix radius per GeV of pT, uniform 3.8 T


def dphi(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def helix_pca(q, px, py, pz, vx, vy, vz, x0, y0, z0):
    """phi, dxy, dz at the closest approach to the beam line through (x0, y0); dz relative to z0."""
    pt = math.hypot(px, py)
    x, y = vx - x0, vy - y0
    phi0 = math.atan2(py, px)
    radius = pt * R_CM_PER_GEV
    cx, cy = x + radius * q * py / pt, y - radius * q * px / pt   # circle centre (q*v x B side)
    c = math.hypot(cx, cy)
    if c == 0:
        return phi0, 0., vz - z0
    ux, uy = cx / c, cy / c
    phi_pca = math.atan2(q * ux, -q * uy)                        # tangent at the PCA point c - R*c_hat
    pca_x, pca_y = cx - radius * ux, cy - radius * uy
    dxy = -pca_x * math.sin(phi_pca) + pca_y * math.cos(phi_pca)
    arc = -q * dphi(phi_pca, phi0) * radius                      # signed transverse path to the PCA
    return phi_pca, dxy, vz + arc * pz / pt - z0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--muons", default="displacedMuons")
    parser.add_argument("--packed", default="packedGenParticles")
    parser.add_argument("--pruned", default="prunedGenParticles")
    parser.add_argument("--beamspot", default="offlineBeamSpot")
    args = parser.parse_args()

    import ROOT
    ROOT.gSystem.Load("libFWCoreFWLite")
    ROOT.FWLiteEnabler.enable()
    from DataFormats.FWLite import Events, Handle

    # One entry per shared track (pair)
    tracks = {}
    for row in csv.DictReader(open(args.input)):
        key = (row["file_url"], int(row["run"]), int(row["lumi"]), int(row["event"]),
               int(row["inner_track_product"]), int(row["inner_track_key"]))
        entry = tracks.setdefault(key, dict(sample=row["sample"], muons=set()))
        entry["muons"].add(int(row["muon_index"]))
    by_file = defaultdict(lambda: defaultdict(list))
    for key in tracks:
        by_file[key[0]][key[1:4]].append(key)

    muons_h = Handle("std::vector<reco::Muon>")
    packed_h = Handle("std::vector<pat::PackedGenParticle>")
    pruned_h = Handle("std::vector<reco::GenParticle>")
    bs_h = Handle("reco::BeamSpot")
    out = []
    for number, (url, targets) in enumerate(sorted(by_file.items()), 1):
        found = set()
        for ev in Events(url):
            aux = ev.eventAuxiliary()
            ident = (aux.run(), aux.luminosityBlock(), aux.event())
            if ident not in targets:
                continue
            ev.getByLabel(args.muons, muons_h)
            ev.getByLabel(args.packed, packed_h)
            ev.getByLabel(args.pruned, pruned_h)
            ev.getByLabel(args.beamspot, bs_h)
            muons = muons_h.product()
            bs = bs_h.product().position()
            x0, y0, z0 = bs.x(), bs.y(), bs.z()

            pruned = pruned_h.product()
            gen_pv = (pruned[0].vx(), pruned[0].vy(), pruned[0].vz()) if pruned.size() else (0., 0., 0.)
            pruned_by_id = defaultdict(list)
            for p in pruned:
                if p.status() == 1 and p.charge() != 0:
                    pruned_by_id[p.pdgId()].append(p)
            gens = []
            for g in packed_h.product():
                if g.charge() == 0:
                    continue
                matches = []
                for p in pruned_by_id.get(g.pdgId(), ()):
                    dr = math.hypot(p.eta() - g.eta(), dphi(p.phi(), g.phi()))
                    if dr < 0.01 and abs(p.pt() - g.pt()) < 0.02 * g.pt():
                        matches.append((dr, p))
                if matches:
                    p = min(matches, key=lambda m: m[0])[1]
                    vx, vy, vz, source = p.vx(), p.vy(), p.vz(), "pruned-vertex"
                else:
                    (vx, vy, vz), source = gen_pv, "unknown-vertex-at-gen-PV"
                phi_pca, dxy, dz = helix_pca(g.charge(), g.px(), g.py(), g.pz(), vx, vy, vz, x0, y0, z0)
                gens.append(dict(pdgId=g.pdgId(), q=g.charge(), pt=g.pt(), eta=g.eta(), phi_pca=phi_pca,
                                 phi_prod=g.phi(), dxy=dxy, dz=dz, vtx=(vx, vy, vz), source=source,
                                 n_pruned=len(matches), mother=g.mother(0).pdgId() if g.mother(0) else 0))

            for key in targets[ident]:
                entry = tracks[key]
                ref = muons[min(entry["muons"])].innerTrack()
                if (ref.id().id(), ref.key()) != key[4:]:
                    raise RuntimeError(f"innerTrack ref mismatch {ident} in {url}")
                tk = ref.get()
                for g in gens:
                    g["dR"] = math.hypot(g["eta"] - tk.eta(), dphi(g["phi_pca"], tk.phi()))
                ranked = sorted(gens, key=lambda g: g["dR"])
                row = dict(sample=entry["sample"], run=ident[0], lumi=ident[1], event=ident[2],
                           muon_indices=";".join(map(str, sorted(entry["muons"]))),
                           inner_track_product=key[4], inner_track_key=key[5],
                           trk_q=tk.charge(), trk_pt=round(tk.pt(), 3), trk_eta=round(tk.eta(), 3),
                           trk_phi=round(tk.phi(), 3), trk_dxy_bs=round(tk.dxy(bs), 3),
                           trk_dz_bs=round(tk.dz(bs), 3),
                           trk_highPurity=int(tk.quality(ROOT.reco.TrackBase.highPurity)), file_url=url)
                for cone in CONES:
                    inside = [g for g in ranked if g["dR"] < cone]
                    tag = str(cone).replace(".", "p")
                    row[f"n_{tag}"] = len(inside)
                    row[f"n_same_q_{tag}"] = sum(g["q"] == tk.charge() for g in inside)
                same_q = [g for g in ranked if g["q"] == tk.charge()]
                for prefix, g in (("near", ranked[0] if ranked else None), ("near_same_q", same_q[0] if same_q else None)):
                    if g is None:
                        continue
                    row.update({f"{prefix}_dR": round(g["dR"], 4), f"{prefix}_pdgId": g["pdgId"],
                                f"{prefix}_q": g["q"], f"{prefix}_pt": round(g["pt"], 3),
                                f"{prefix}_eta": round(g["eta"], 3), f"{prefix}_phi_pca": round(g["phi_pca"], 3),
                                f"{prefix}_phi_prod": round(g["phi_prod"], 3), f"{prefix}_dxy_bs": round(g["dxy"], 3),
                                f"{prefix}_dz_bs": round(g["dz"], 3),
                                f"{prefix}_vtx": "{:.2f};{:.2f};{:.2f}".format(*g["vtx"]),
                                f"{prefix}_vertex_source": g["source"], f"{prefix}_n_pruned_match": g["n_pruned"],
                                f"{prefix}_mother": g["mother"]})
                if ranked:
                    row["near_q_agrees"] = int(ranked[0]["q"] == tk.charge())
                    row["_near_dR"] = ranked[0]["dR"]  # unrounded, summary only
                # every candidate in the widest cone: pdgId:q:pt:dR:dxy_bs:dz_bs:vertex_source
                row["cone0p3_candidates"] = " ".join(
                    f"{g['pdgId']}:{g['q']:+d}:{g['pt']:.2f}:{g['dR']:.3f}:{g['dxy']:.2f}:{g['dz']:.2f}:"
                    f"{'P' if g['source'] == 'pruned-vertex' else 'U'}" for g in ranked if g["dR"] < 0.3)
                out.append(row)
            found.add(ident)
            if found == set(targets):
                break
        if found != set(targets):
            raise RuntimeError(f"Missing events in {url}: {sorted(set(targets) - found)}")
        print(f"File {number}/{len(by_file)}", flush=True)

    columns = []
    for row in out:
        columns += [c for c in row if c not in columns and not c.startswith("_")]
    with open(args.output, "w", newline="") as dest:
        writer = csv.DictWriter(dest, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out)

    print(f"\n{len(out)} shared tracks -> {args.output}")
    for cone in CONES:
        tag = str(cone).replace(".", "p")
        has = [r for r in out if r[f"n_{tag}"] > 0]
        nearest_in = [r for r in out if r.get("_near_dR", 99) < cone]
        agree = sum(r["near_q_agrees"] for r in nearest_in)
        any_same = sum(r[f"n_same_q_{tag}"] > 0 for r in has)
        ids = Counter(abs(r["near_pdgId"]) for r in nearest_in)
        print(f"  dR < {cone}: {len(has)} tracks with >=1 charged gen candidate | nearest charge agrees {agree}, "
              f"disagrees {len(nearest_in) - agree} | any same-charge in cone {any_same} | nearest |pdgId| {dict(ids)}")


if __name__ == "__main__":
    main()
