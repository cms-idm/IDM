#!/usr/bin/env python3
"""Compare two direction tests on displacedMuons that share an exact innerTrack.

In CMSSW (cmsenv + voms proxy):
  python3 check_shared_track_direction.py --input shared_inner_track_muon_locations.csv \
      --output shared_inner_track_direction_tests.csv

Per muon:
  region_test   +1/-1/0: first DT/CSC chamber with a segment, sector phi (MuonIdProducer::sectorPhi)
                within pi/2 of innerTrack outerPosition phi (PCA phi if no TrackExtra); 0 = no such chamber.
  forward_score sum over chambers with a segment/RPC/GEM hit: endcap +1/-1 by z side vs sign(pz),
                barrel +1/-1 by sector phi within pi/2 of outerMomentum phi.
  endcap_z      sign of the endcap-only part of forward_score; 0 = no endcap chamber with a hit.
Per pair: "split" = one +1 and one -1 (scores: one >0, one <0).
"""
import argparse
import csv
import math
from collections import Counter, defaultdict

DT, CSC, RPC, GEM = 1, 2, 3, 4


def dphi(a, b):
    d = a - b
    while d > math.pi:
        d -= 2 * math.pi
    while d <= -math.pi:
        d += 2 * math.pi
    return d


def sector_phi(sector):
    phi = (sector - 1) / 6. * math.pi
    if sector == 13:
        phi = 3 / 6. * math.pi
    if sector == 14:
        phi = 9 / 6. * math.pi
    return phi - 2 * math.pi if phi > math.pi else phi


def sign(x):
    return (x > 0) - (x < 0)


def evaluate(ROOT, mu):
    tk = mu.innerTrack().get()
    has_extra = tk.extra().isAvailable()
    phi_exit_pos = tk.outerPosition().phi() if has_extra else tk.phi()
    phi_exit_mom = tk.outerMomentum().phi() if has_extra else tk.phi()
    pz_sign = 1 if tk.pz() >= 0 else -1

    region_phi = None
    if mu.isStandAloneMuon():
        region_phi = mu.standAloneMuon().innerPosition().phi()
    forward = endcap = 0
    chambers = []
    for m in mu.matches():
        det, raw = m.detector(), m.id.rawId()
        n_seg = m.segmentMatches.size()
        n_hit = n_seg + m.rpcMatches.size() + m.gemMatches.size()
        if hasattr(m, "gemHitMatches"):
            n_hit += m.gemHitMatches.size()
        if det == DT:
            sec_phi, z_side = sector_phi(ROOT.DTChamberId(raw).sector()), 0
        elif det == CSC:
            c = ROOT.CSCDetId(raw)
            sec_phi = math.pi / 4 + (c.triggerSector() - 1) / 3. * math.pi
            sec_phi = sec_phi - 2 * math.pi if sec_phi > math.pi else sec_phi
            z_side = 1 if c.endcap() == 1 else -1
        elif det == RPC:
            r = ROOT.RPCDetId(raw)
            z_side = r.region()
            sec_phi = sector_phi(r.sector()) if z_side == 0 else None
        elif det == GEM:
            sec_phi, z_side = None, ROOT.GEMDetId(raw).region()
        else:
            continue
        chambers.append(f"{det}:{m.station()}:{z_side:+d}:{n_hit}")
        if region_phi is None and det in (DT, CSC) and n_seg > 0:
            region_phi = sec_phi
        if n_hit == 0:
            continue
        if z_side != 0:
            vote = 1 if z_side == pz_sign else -1
            endcap += vote
        else:
            vote = 1 if abs(dphi(sec_phi, phi_exit_mom)) < math.pi / 2 else -1
        forward += vote

    region = 0 if region_phi is None else (1 if abs(dphi(region_phi, phi_exit_pos)) < math.pi / 2 else -1)
    return dict(pt=round(mu.pt(), 3), eta=round(mu.eta(), 3), phi=round(mu.phi(), 3),
                is_global=int(mu.isGlobalMuon()), is_tracker=int(mu.isTrackerMuon()),
                is_standalone=int(mu.isStandAloneMuon()), has_extra=int(has_extra),
                region_test=region, forward_score=forward, endcap_z=sign(endcap),
                chambers=" ".join(chambers),
                _ref=(mu.innerTrack().id().id(), mu.innerTrack().key()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", default="displacedMuons")
    args = parser.parse_args()

    import ROOT
    ROOT.gSystem.Load("libFWCoreFWLite")
    ROOT.FWLiteEnabler.enable()
    from DataFormats.FWLite import Events, Handle

    rows = list(csv.DictReader(open(args.input)))
    by_file = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_file[row["file_url"]][int(row["run"]), int(row["lumi"]), int(row["event"])].append(row)

    handle = Handle("std::vector<reco::Muon>")
    for number, (url, targets) in enumerate(sorted(by_file.items()), 1):
        found = set()
        for ev in Events(url):
            aux = ev.eventAuxiliary()
            ident = (aux.run(), aux.luminosityBlock(), aux.event())
            if ident not in targets:
                continue
            ev.getByLabel(args.label, handle)
            muons = handle.product()
            for row in targets[ident]:
                result = evaluate(ROOT, muons[int(row["muon_index"])])
                expected = (int(row["inner_track_product"]), int(row["inner_track_key"]))
                if result.pop("_ref") != expected:
                    raise RuntimeError(f"innerTrack ref mismatch {ident} muon {row['muon_index']} in {url}")
                row.update(result)
            found.add(ident)
            if found == set(targets):
                break
        if found != set(targets):
            raise RuntimeError(f"Missing events in {url}: {sorted(set(targets) - found)}")
        print(f"File {number}/{len(by_file)}", flush=True)

    with open(args.output, "w", newline="") as dest:
        writer = csv.DictWriter(dest, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Pair-level summary (every group in the input is a pair)
    pairs = defaultdict(list)
    for row in rows:
        pairs[row["file_url"], row["run"], row["lumi"], row["event"],
              row["inner_track_product"], row["inner_track_key"]].append(row)
    tally = Counter()
    for group in pairs.values():
        if len(group) != 2:
            tally["group size != 2 (skipped)"] += 1
            continue
        a, b = group
        tally["global muons in pair: %d" % (a["is_global"] + b["is_global"])] += 1
        region_split = {a["region_test"], b["region_test"]} == {1, -1}
        fwd_split = sign(a["forward_score"]) * sign(b["forward_score"]) < 0
        tally["region test: split +1/-1"] += region_split
        tally[f"region test values {tuple(sorted((a['region_test'], b['region_test'])))}"] += 1
        tally["forward score: split"] += fwd_split
        if region_split and fwd_split:
            same = sign(a["forward_score"]) == a["region_test"]
            tally["both split: pick same keeper" if same else "both split: DISAGREE"] += 1
        if a["endcap_z"] != 0 and b["endcap_z"] != 0:
            tally["pairs with endcap hits on both"] += 1
            if region_split:
                same = a["endcap_z"] == a["region_test"] and b["endcap_z"] == b["region_test"]
                tally["  region split, agrees with endcap z" if same else "  region split, DISAGREES with endcap z"] += 1
    print(f"\n{len(pairs)} pairs, {len(rows)} muons -> {args.output}")
    for key in sorted(tally):
        print(f"  {key}: {tally[key]}")


if __name__ == "__main__":
    main()
