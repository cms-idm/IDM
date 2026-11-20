#!/usr/bin/env python3
"""
Step 1 of the event maps: pull the per-event objects for an event list out of
MiniAOD with FWLite and dump them to JSON. Step 2 (the plotting) is
python_analysis/eventmap_plots.py, run in the coffea environment.

Input: a CSV with at least run, lumi, event, ntuple_file, gen_ele_pt and
gen_pos_pt columns, e.g. the output of python_analysis/eventmaps/merged_eventlist.py.
Only events with both gen electrons pT > --gen-pt (default 2 GeV) are dumped;
the cut is applied before --max-per-sample / --max-events. Sample metadata
columns (sample, year, m1, delta, mZD, ctau, gen_lxy, gen_dR_ee, ...) are
copied through to the JSON.

Each event is located like this:
  ntuple_file  .../<year>/<M1-..._dM-..._mZD-...>/<ctau-N>/ntuples_..._flist_XX.root
  -> MiniAOD file list  AODSkimmer/fileLists/signal/<year>/<point>_<ctau>_flist.txt
  -> (run, lumi) -> MiniAOD file, from an index of each file's lumi blocks
     (built on first use and cached as JSON)
  -> fwlite ChainEvent::to(run, lumi, event).

Objects dumped (eta/phi are momentum directions):
  reco_jets      slimmedJets                   pT > --jet-pt
  gen_jets       slimmedGenJets                pT > --genjet-pt
  reco_lpt_eles  slimmedLowPtElectrons         pT > --ele-pt (no ID cut)
  gen_eles       prunedGenParticles, |pdgId| == 11 with mother chi2 (1000023)
                 (same definition as ElectronSkimmer's GenEle/GenPos)
  gen_ee         pt/eta/phi of the sum of the two gen_eles 4-vectors
                 (null unless exactly two are found)
  met            PF  (slimmedMETs, Type-1), PUPPI (slimmedMETsPuppi, Type-1),
                 gen (genMetTrue, as in the ntuple's genLeadMET)
  ecal_scs       reducedEgamma:reducedSuperClusters, all, no energy cut (EB+EE, in-time;
                 MiniAOD keeps only SCs linked to saved e/gamma objects).
                 eta/phi here are the SC position seen from (0,0,0), not a momentum.

Objects within dR < --dr of the gen_ee direction, under 'near_gen_ee' (null if
gen_ee is null; no pT/energy cuts):
  ecal_rechits   reducedEgamma:reduced{EB,EE}RecHits; eta/phi of the crystal
                 front-face centre seen from (0,0,0), from the Fireworks Run 3
                 geometry (cmsGeom2021.root). Only hits MiniAOD kept around
                 saved e/gamma objects.
  pf_cands       packedPFCandidates
  photons        slimmedPhotons
  conversions    reducedEgamma:reducedConversions        dR on pairMomentum()
  single_leg_conversions  reducedEgamma:reducedSingleLegConversions  (same)
  gsf_tracks     reducedEgamma:reducedGsfTracks           dR on the momentum at
                 the reference point (vx, vy, vz)
  open_conversions   gsfTracksOpenConversions (GSF-track pairs)  dR on pairMomentum()
  lost_tracks        lostTracks             (tracks not made into PF candidates)
  lost_ele_tracks    lostTracks:eleTracks
  secondary_vertices slimmedSecondaryVertices (IVF); dR on the vertex momentum,
                     'vtx' is the vertex position
  ebee_clusters      reducedEgamma:reducedEBEEClusters  } eta/phi are the cluster
  es_clusters        reducedEgamma:reducedESClusters    } position from (0,0,0)
  isolated_tracks    isolatedTracks
  displaced_tracks   displacedTracks        dR on the momentum at (vx, vy, vz)
Charged pf_cands, gsf_tracks, lost(_ele)_tracks, isolated_tracks and
displaced_tracks also get 'ecal', propagated exactly as for the electrons below
(null for neutral PF candidates). Start points: the object's reference point
(vx, vy, vz) and momentum there; GSF tracks use the mode momentum and charge
(as the electrons' trackMomentumAtVtx does); isolated tracks, whose vertex() is
not filled, use their linked PF candidate/lost track, else the point rebuilt
from dxy/dz and offlineSlimmedPrimaryVertices[0]. The helix doesn't depend on
the particle mass, so using the electron mass for hadrons doesn't matter.

Every gen and reco low-pT electron also gets 'ecal': its trajectory propagated
to the ECAL front face with CMSSW's BaseParticlePropagator
(propagateToEcalEntrance: barrel R = 129 cm, endcap |z| = 320.9 cm), as a helix
in a uniform --bfield starting from the particle's own vertex, so displaced
electrons are handled. No energy loss or bremsstrahlung along the way. Starting
state: gen = production vertex + gen momentum; reco = GSF track at its PCA to
the beamspot (trackPositionAtVtx, trackMomentumAtVtx), not the electron p4,
whose energy is ECAL-corrected. 'ecal' is null if the propagation doesn't reach
the ECAL or ends outside |eta| < 3. Reco electrons also carry 'sc', their own
supercluster position and energy (from LowPtGsfElectronSCProducer, embedded in
the pat::Electron; not in reducedSuperClusters). ecal/sc eta and phi are
positions seen from (0,0,0).

Run under cmsenv, from anywhere:
  python3 scripts/dump_eventmap_objects.py --max-per-sample 5
  python3 scripts/dump_eventmap_objects.py --events 1:1927128:297 1:1927128:40
"""
import argparse
import array
import json
import math
import os
import re
import sys
import time

import pandas as pd
import ROOT
from DataFormats.FWLite import Events, Handle

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kError
ROOT.gSystem.Load('libCommonToolsBaseParticlePropagator')
ROOT.gInterpreter.Declare('#include "CommonTools/BaseParticlePropagator/interface/BaseParticlePropagator.h"')

AODSKIM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYANA = os.path.join(os.path.dirname(AODSKIM), 'python_analysis')
OUTDIR = os.path.join(PYANA, 'eventmaps', 'outputs')

DEFAULT_CSV = os.path.join(OUTDIR, 'merged_events_sigNov2026_anabrv-sel_cut3.csv')
DEFAULT_FLIST_DIR = os.path.join(AODSKIM, 'fileLists', 'signal')
DEFAULT_INDEX = os.path.join(OUTDIR, 'miniaod_lumi_index.json')
REDIRECTOR = 'root://cmseos.fnal.gov/'

CHI2_PDGID = 1000023
ELE_MASS = 0.000511  # GeV
NTUPLE_PATH_RE = re.compile(r'/(\d{4})/+(M1-[^/]+)/+(ctau-[^/]+)/')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--csv', default=DEFAULT_CSV, help='event list CSV (run, lumi, event, ntuple_file, ...)')
    p.add_argument('--out', default=None,
                   help='output JSON (default: eventmaps/outputs/eventmap_objects_<csv stem>.json)')
    p.add_argument('--samples', nargs='*', default=None,
                   help="keep only rows whose 'sample' contains one of these substrings")
    p.add_argument('--events', nargs='*', default=None, metavar='RUN:LUMI:EVENT',
                   help='keep only these events')
    p.add_argument('--gen-pt', type=float, default=2.,
                   help='keep only events with both gen electrons pT > this (GeV; CSV gen_ele_pt/gen_pos_pt)')
    p.add_argument('--max-per-sample', type=int, default=10, help='events per sample (-1 = all)')
    p.add_argument('--max-events', type=int, default=-1, help='total events (-1 = all)')
    p.add_argument('--jet-pt', type=float, default=20., help='reco jet pT threshold (GeV)')
    p.add_argument('--genjet-pt', type=float, default=20., help='gen jet pT threshold (GeV)')
    p.add_argument('--ele-pt', type=float, default=1., help='reco low-pT electron pT threshold (GeV)')
    p.add_argument('--bfield', type=float, default=3.8, help='solenoid field for the ECAL propagation (T)')
    p.add_argument('--dr', type=float, default=0.5,
                   help='dR around the gen_ee direction for the near_gen_ee objects')
    p.add_argument('--flist-dir', default=DEFAULT_FLIST_DIR, help='MiniAOD file-list directory')
    p.add_argument('--index-cache', default=DEFAULT_INDEX, help='(run, lumi) -> MiniAOD file index cache')
    p.add_argument('--redirector', default=REDIRECTOR, help='xrootd redirector for /store paths')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Event list
# ---------------------------------------------------------------------------

def load_event_list(args):
    df = pd.read_csv(args.csv)
    for c in ('run', 'lumi', 'event'):
        df[c] = df[c].astype('int64')
    if args.samples:
        df = df[df['sample'].apply(lambda s: any(sub in s for sub in args.samples))]
    if args.events:
        wanted = {tuple(int(x) for x in e.split(':')) for e in args.events}
        df = df[[(r, l, e) in wanted for r, l, e in zip(df.run, df.lumi, df.event)]]
    df = df[(df.gen_ele_pt > args.gen_pt) & (df.gen_pos_pt > args.gen_pt)]
    if args.max_per_sample >= 0:
        df = df.groupby('sample', sort=False).head(args.max_per_sample)
    if args.max_events >= 0:
        df = df.head(args.max_events)
    return df


def flist_for(ntuple_file, flist_dir):
    """MiniAOD file list for the signal point an ntuple was made from."""
    m = NTUPLE_PATH_RE.search(ntuple_file)
    if m is None:
        raise ValueError(f"can't parse signal point from ntuple path {ntuple_file}")
    year, point, ctau = m.groups()
    path = os.path.join(flist_dir, year, f'{point}_{ctau}_flist.txt')
    if not os.path.exists(path):
        raise FileNotFoundError(f'no MiniAOD file list {path} (from {ntuple_file})')
    return path


# ---------------------------------------------------------------------------
# (run, lumi) -> MiniAOD file index
# ---------------------------------------------------------------------------

def _pfn(lfn, redirector):
    return redirector + lfn if lfn.startswith('/store') else lfn


def lumis_in_file(pfn):
    tf = ROOT.TFile.Open(pfn)
    if not tf or tf.IsZombie():
        print(f'  WARNING: could not open {pfn}')
        return []
    t = tf.Get('LuminosityBlocks')
    out = set()
    for i in range(t.GetEntries()):
        t.GetEntry(i)
        aux = t.LuminosityBlockAuxiliary
        out.add((int(aux.run()), int(aux.luminosityBlock())))
    tf.Close()
    return sorted(out)


def load_index(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def index_flist(flist, index, args):
    """{'run:lumi': [pfn, ...]} for every file in a MiniAOD file list, cached in `index`."""
    key = os.path.abspath(flist)
    if key in index:
        return index[key]
    with open(flist) as f:
        files = [_pfn(l.strip(), args.redirector) for l in f if l.strip()]
    print(f'Indexing lumis of {len(files)} MiniAOD files in {os.path.basename(flist)}')
    lumimap = {}
    for pfn in files:
        for run, lumi in lumis_in_file(pfn):
            lumimap.setdefault(f'{run}:{lumi}', []).append(pfn)
    index[key] = lumimap
    os.makedirs(os.path.dirname(args.index_cache), exist_ok=True)
    with open(args.index_cache, 'w') as f:
        json.dump(index, f, indent=1)
    return lumimap


# ---------------------------------------------------------------------------
# Object extraction
# ---------------------------------------------------------------------------

HANDLES = {
    'jets':    ('slimmedJets',           Handle('std::vector<pat::Jet>')),
    'genjets': ('slimmedGenJets',        Handle('std::vector<reco::GenJet>')),
    'lpteles': ('slimmedLowPtElectrons', Handle('std::vector<pat::Electron>')),
    'genpart': ('prunedGenParticles',    Handle('std::vector<reco::GenParticle>')),
    'pfmet':   ('slimmedMETs',           Handle('std::vector<pat::MET>')),
    'puppimet': ('slimmedMETsPuppi',     Handle('std::vector<pat::MET>')),
    'genmet':  ('genMetTrue',            Handle('std::vector<reco::GenMET>')),
    'scs':     (('reducedEgamma', 'reducedSuperClusters'), Handle('std::vector<reco::SuperCluster>')),
    'ebhits':  (('reducedEgamma', 'reducedEBRecHits'),
                Handle('edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit> >')),
    'eehits':  (('reducedEgamma', 'reducedEERecHits'),
                Handle('edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit> >')),
    'pfcands': ('packedPFCandidates',    Handle('std::vector<pat::PackedCandidate>')),
    'photons': ('slimmedPhotons',        Handle('std::vector<pat::Photon>')),
    'convs':   (('reducedEgamma', 'reducedConversions'), Handle('std::vector<reco::Conversion>')),
    'slconvs': (('reducedEgamma', 'reducedSingleLegConversions'), Handle('std::vector<reco::Conversion>')),
    'gsftrks': (('reducedEgamma', 'reducedGsfTracks'), Handle('std::vector<reco::GsfTrack>')),
    'openconvs': (('gsfTracksOpenConversions', 'gsfTracksOpenConversions'),
                  Handle('std::vector<reco::Conversion>')),
    'losttrks':  ('lostTracks',            Handle('std::vector<pat::PackedCandidate>')),
    'lostetrks': (('lostTracks', 'eleTracks'), Handle('std::vector<pat::PackedCandidate>')),
    'svs':       ('slimmedSecondaryVertices', Handle('std::vector<reco::VertexCompositePtrCandidate>')),
    'ebeeclus':  (('reducedEgamma', 'reducedEBEEClusters'), Handle('std::vector<reco::CaloCluster>')),
    'esclus':    (('reducedEgamma', 'reducedESClusters'), Handle('std::vector<reco::CaloCluster>')),
    'isotrks':   ('isolatedTracks',        Handle('std::vector<pat::IsolatedTrack>')),
    'disptrks':  ('displacedTracks',       Handle('std::vector<reco::Track>')),
    'pvs':       ('offlineSlimmedPrimaryVertices', Handle('std::vector<reco::Vertex>')),
}

FWGEOM_REL = 'Fireworks/Geometry/data/cmsGeom2021.root'
_ECAL_GEOM = None  # rawId -> (eta, phi) of the crystal front-face centre


def ecal_geometry():
    """{rawId: (eta, phi)} for every EB/EE crystal, from the Fireworks Run 3 geometry file."""
    global _ECAL_GEOM
    if _ECAL_GEOM is not None:
        return _ECAL_GEOM
    path = next((os.path.join(d, FWGEOM_REL) for d in os.environ.get('CMSSW_SEARCH_PATH', '').split(':')
                 if os.path.exists(os.path.join(d, FWGEOM_REL))), None)
    if path is None:
        raise FileNotFoundError(f'{FWGEOM_REL} not found in CMSSW_SEARCH_PATH (run under cmsenv)')
    tf = ROOT.TFile.Open(path)
    t = tf.Get('idToGeo')
    rawid, pts = array.array('I', [0]), array.array('f', [0.] * 24)
    t.SetBranchStatus('*', 0)
    t.SetBranchStatus('id', 1)
    t.SetBranchStatus('points', 1)
    t.SetBranchAddress('id', rawid)
    t.SetBranchAddress('points', pts)
    bid = t.GetBranch('id')
    geo = {}
    for i in range(t.GetEntries()):
        bid.GetEntry(i)
        d = rawid[0]
        if d >> 28 != 3 or (d >> 25) & 7 not in (1, 2):  # DetId::Ecal, EcalBarrel/EcalEndcap
            continue
        t.GetEntry(i)
        # corners 0-3 are the front face
        v = ROOT.TVector3(sum(pts[0:12:3]) / 4, sum(pts[1:12:3]) / 4, sum(pts[2:12:3]) / 4)
        geo[d] = (v.Eta(), v.Phi())
    tf.Close()
    _ECAL_GEOM = geo
    return geo


def _product(events, key):
    label, h = HANDLES[key]
    events.getByLabel(label, h)
    return h.product() if h.isValid() else None


def _kin(o):
    return {'pt': o.pt(), 'eta': o.eta(), 'phi': o.phi()}


def _point(x, y, z):
    v = ROOT.TVector3(x, y, z)
    return {'x': x, 'y': y, 'z': z, 'eta': v.Eta(), 'phi': v.Phi()}


def propagate_to_ecal(vtx, mom, charge, bfield):
    """ECAL-front position of a particle starting at vtx (cm) with momentum mom (GeV).

    vtx and mom are (x, y, z) tuples. Uses BaseParticlePropagator::propagateToEcalEntrance.
    Returns None if the particle doesn't reach the ECAL or lands outside |eta| < 3.
    """
    px, py, pz = mom
    e = (px * px + py * py + pz * pz + ELE_MASS * ELE_MASS) ** 0.5
    part = ROOT.RawParticle(ROOT.math.XYZTLorentzVector(px, py, pz, e),
                            ROOT.math.XYZTLorentzVector(vtx[0], vtx[1], vtx[2], 0.))
    part.setCharge(float(charge))
    prop = ROOT.BaseParticlePropagator(part, 0., 0., bfield)
    prop.propagateToEcalEntrance(False)
    if prop.getSuccess() == 0:
        return None
    v = prop.particle().vertex()
    return _point(v.x(), v.y(), v.z())


def _gen_ele(g, args):
    return {**_kin(g), 'pdgId': g.pdgId(), 'vxy': (g.vx() ** 2 + g.vy() ** 2) ** 0.5,
            'ecal': propagate_to_ecal((g.vx(), g.vy(), g.vz()), (g.px(), g.py(), g.pz()),
                                      g.charge(), args.bfield)}


def _reco_ele(e, args):
    pos, mom = e.trackPositionAtVtx(), e.trackMomentumAtVtx()
    sc = e.superCluster()
    return {**_kin(e), 'charge': e.charge(),
            'ecal': propagate_to_ecal((pos.x(), pos.y(), pos.z()), (mom.x(), mom.y(), mom.z()),
                                      e.charge(), args.bfield),
            'sc': {**_point(sc.x(), sc.y(), sc.z()), 'energy': sc.energy()}
                  if sc.isNonnull() and sc.isAvailable() else None}


def _met(coll, corrected):
    if coll is None or coll.size() == 0:
        return None
    m = coll.at(0)
    if corrected:
        return {'pt': m.corPt(ROOT.pat.MET.Type1), 'phi': m.corPhi(ROOT.pat.MET.Type1)}
    return {'pt': m.pt(), 'phi': m.phi()}


def _dr(eta1, phi1, eta2, phi2):
    dphi = (phi1 - phi2 + math.pi) % (2 * math.pi) - math.pi
    return math.hypot(eta1 - eta2, dphi)


def _rechits(coll, det, ref, args):
    if coll is None:
        return []
    geo = ecal_geometry()
    out = []
    for h in coll:
        pos = geo.get(h.id().rawId())
        if pos is not None and _dr(pos[0], pos[1], ref['eta'], ref['phi']) < args.dr:
            out.append({'det': det, 'eta': pos[0], 'phi': pos[1], 'energy': h.energy()})
    return out


def _conversion(c):
    p = c.pairMomentum()
    v = c.conversionVertex()
    return {'pt': p.rho(), 'eta': p.eta(), 'phi': p.phi(), 'ntracks': c.tracksPin().size(),
            'vtx': {'x': v.x(), 'y': v.y(), 'z': v.z(), 'rxy': math.hypot(v.x(), v.y())}
                   if v.isValid() else None}


def _isotrk_vertex(t, pv):
    """Start point for a pat::IsolatedTrack, whose own vertex() is always (0,0,0).

    Uses the linked packedPFCandidate/lostTrack if there is one; otherwise rebuilds the
    point of closest approach from dxy/dz, which the producer measures w.r.t. PV[0].
    """
    pc = t.packedCandRef()
    if pc.isNonnull() and pc.isAvailable():
        c = pc.get()
        return (c.vx(), c.vy(), c.vz())
    if pv is None:
        return None
    return (pv.x() - t.dxy() * math.sin(t.phi()), pv.y() + t.dxy() * math.cos(t.phi()), pv.z() + t.dz())


def near_gen_ee(events, gen_ee, args):
    """Objects within dR < args.dr of the gen_ee direction (see the module docstring)."""
    if gen_ee is None:
        return None

    def near(o):
        return _dr(o['eta'], o['phi'], gen_ee['eta'], gen_ee['phi']) < args.dr

    def near_coll(key):
        """Objects of a collection whose own eta()/phi() are within dR of gen_ee."""
        return [o for o in (_product(events, key) or [])
                if _dr(o.eta(), o.phi(), gen_ee['eta'], gen_ee['phi']) < args.dr]

    def ecal(vtx, mom, charge):
        """Propagation to the ECAL front face, as for the electrons; None if neutral or no start point."""
        if charge == 0 or vtx is None:
            return None
        return propagate_to_ecal(vtx, mom, charge, args.bfield)

    def ecal_cand(c):
        return ecal((c.vx(), c.vy(), c.vz()), (c.px(), c.py(), c.pz()), c.charge())

    pvs = _product(events, 'pvs')
    pv = pvs.at(0) if pvs is not None and pvs.size() > 0 else None

    pfcands = [{**_kin(c), 'energy': c.energy(), 'pdgId': c.pdgId(), 'charge': c.charge(),
                'caloFraction': c.caloFraction(), 'hcalFraction': c.hcalFraction(), 'ecal': ecal_cand(c)}
               for c in near_coll('pfcands')]
    photons = [{**_kin(g), 'energy': g.energy()} for g in (_product(events, 'photons') or [])]
    convs = [_conversion(c) for c in (_product(events, 'convs') or []) if c.tracksPin().size() > 0]
    slconvs = [_conversion(c) for c in (_product(events, 'slconvs') or []) if c.tracksPin().size() > 0]
    # GSF tracks start from the mode momentum/charge, like trackMomentumAtVtx for the electrons
    gsftrks = [{**_kin(t), 'charge': t.charge(), 'vx': t.vx(), 'vy': t.vy(), 'vz': t.vz(),
                'ecal': ecal((t.vx(), t.vy(), t.vz()), (t.pxMode(), t.pyMode(), t.pzMode()), t.chargeMode())}
               for t in near_coll('gsftrks')]
    openconvs = [_conversion(c) for c in (_product(events, 'openconvs') or []) if c.tracksPin().size() > 0]

    def lost(key):
        return [{**_kin(c), 'charge': c.charge(), 'pdgId': c.pdgId(), 'dxy': c.dxy(), 'dz': c.dz(),
                 'ecal': ecal_cand(c)}
                for c in near_coll(key)]

    svs = [{**_kin(s), 'mass': s.mass(), 'ntracks': s.numberOfDaughters(),
            'chi2ndof': s.vertexNormalizedChi2(),
            'vtx': {'x': s.vx(), 'y': s.vy(), 'z': s.vz(), 'rxy': math.hypot(s.vx(), s.vy())}}
           for s in (_product(events, 'svs') or [])]

    def clusters(key):
        return [{'energy': c.energy(), 'eta': c.eta(), 'phi': c.phi(), 'x': c.x(), 'y': c.y(), 'z': c.z()}
                for c in (_product(events, key) or [])]

    isotrks = [{**_kin(t), 'charge': t.charge(), 'pdgId': t.pdgId(), 'dxy': t.dxy(), 'dz': t.dz(),
                'ecal': ecal(_isotrk_vertex(t, pv), (t.px(), t.py(), t.pz()), t.charge())}
               for t in near_coll('isotrks')]
    disptrks = [{**_kin(t), 'charge': t.charge(), 'vx': t.vx(), 'vy': t.vy(), 'vz': t.vz(),
                 'ecal': ecal((t.vx(), t.vy(), t.vz()), (t.px(), t.py(), t.pz()), t.charge())}
                for t in near_coll('disptrks')]
    return {
        'ecal_rechits': (_rechits(_product(events, 'ebhits'), 'EB', gen_ee, args)
                         + _rechits(_product(events, 'eehits'), 'EE', gen_ee, args)),
        'pf_cands':               [o for o in pfcands if near(o)],
        'photons':                [o for o in photons if near(o)],
        'conversions':            [o for o in convs if near(o)],
        'single_leg_conversions': [o for o in slconvs if near(o)],
        'gsf_tracks':             [o for o in gsftrks if near(o)],
        'open_conversions':       [o for o in openconvs if near(o)],
        'lost_tracks':            [o for o in lost('losttrks') if near(o)],
        'lost_ele_tracks':        [o for o in lost('lostetrks') if near(o)],
        'secondary_vertices':     [o for o in svs if near(o)],
        'ebee_clusters':          [o for o in clusters('ebeeclus') if near(o)],
        'es_clusters':            [o for o in clusters('esclus') if near(o)],
        'isolated_tracks':        [o for o in isotrks if near(o)],
        'displaced_tracks':       [o for o in disptrks if near(o)],
    }


def extract_objects(events, args):
    jets = _product(events, 'jets') or []
    genjets = _product(events, 'genjets') or []
    eles = _product(events, 'lpteles') or []
    genpart = _product(events, 'genpart') or []
    scs = _product(events, 'scs') or []

    gen_parts = [g for g in genpart
                 if abs(g.pdgId()) == 11 and g.numberOfMothers() > 0 and g.mother(0).pdgId() == CHI2_PDGID]
    gen_eles = [_gen_ele(g, args) for g in gen_parts]
    gen_ee = None
    if len(gen_parts) == 2:
        ee = gen_parts[0].p4() + gen_parts[1].p4()
        gen_ee = {'pt': ee.pt(), 'eta': ee.eta(), 'phi': ee.phi()}

    return {
        'reco_jets':     [_kin(j) for j in jets if j.pt() > args.jet_pt],
        'gen_jets':      [_kin(j) for j in genjets if j.pt() > args.genjet_pt],
        'reco_lpt_eles': [_reco_ele(e, args) for e in eles if e.pt() > args.ele_pt],
        'gen_eles':      gen_eles,
        'gen_ee':        gen_ee,
        'ecal_scs':      [{'energy': s.energy(), 'raw_energy': s.rawEnergy(),
                           'eta': s.eta(), 'phi': s.phi(), 'x': s.x(), 'y': s.y(), 'z': s.z()}
                          for s in scs],
        'near_gen_ee':   near_gen_ee(events, gen_ee, args),
        'met': {
            'pf':    _met(_product(events, 'pfmet'), corrected=True),
            'puppi': _met(_product(events, 'puppimet'), corrected=True),
            'gen':   _met(_product(events, 'genmet'), corrected=False),
        },
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _jsonable(v):
    return v.item() if hasattr(v, 'item') else v


def main():
    args = parse_args()
    out = args.out or os.path.join(
        OUTDIR, 'eventmap_objects_' + os.path.splitext(os.path.basename(args.csv))[0] + '.json')

    df = load_event_list(args)
    print(f'{len(df)} events selected from {args.csv}')
    if df.empty:
        return

    index = load_index(args.index_cache)
    df = df.assign(flist=[flist_for(f, args.flist_dir) for f in df.ntuple_file])

    # Resolve each event's MiniAOD file(s) first, so each file is opened once.
    todo = {}  # pfn -> [row, ...]
    missing = []
    for _, row in df.iterrows():
        lumimap = index_flist(row.flist, index, args)
        cands = lumimap.get(f'{row.run}:{row.lumi}')
        if not cands:
            missing.append(row)
            continue
        todo.setdefault(tuple(cands), []).append(row)

    records, not_found = [], []
    t0 = time.time()
    for cands, rows in todo.items():
        events = Events(list(cands))
        events.size()  # creates the underlying fwlite ChainEvent
        ev = events.object()
        for row in rows:
            if not ev.to(int(row.run), int(row.lumi), int(row.event)):
                not_found.append(row)
                continue
            rec = {k: _jsonable(v) for k, v in row.items() if k != 'flist'}
            rec['miniaod_file'] = cands[ev.fileIndex()] if len(cands) > 1 else cands[0]
            rec['objects'] = extract_objects(events, args)
            records.append(rec)
            print(f'  [{len(records)}/{len(df)}] {row["sample"]} {row.run}:{row.lumi}:{row.event}')
        del events

    for row in missing:
        print(f'  WARNING: no MiniAOD file has lumi {row.run}:{row.lumi} ({row["sample"]} event {row.event})')
    for row in not_found:
        print(f'  WARNING: event {row.run}:{row.lumi}:{row.event} not found in its MiniAOD file(s)')

    meta = {'csv': os.path.abspath(args.csv), 'jet_pt': args.jet_pt, 'genjet_pt': args.genjet_pt,
            'gen_pt': args.gen_pt, 'ele_pt': args.ele_pt, 'bfield': args.bfield, 'dr': args.dr}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, 'w') as f:
        json.dump({'meta': meta, 'events': records}, f, indent=1)
    print(f'Wrote {len(records)} events to {out} ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    sys.exit(main())
