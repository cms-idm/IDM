// DimuonVertexTruth: truth-matched dimuon vertex study (vertexing step 1).
//
// Events with exactly two final signal muons (|pdgId|==13, status 1, last copy,
// first different-PDG ancestor |pdgId|==1000023), one of each charge.
// Gen acceptance: |eta|<2.4, 0<=vxy<740 cm, |vz|<960 cm, no pT minimum.
//
// Reco units: unique tracks referenced by displacedMuons, per type
//   inner  = innerTrack  (displacedTracks)
//   global = globalTrack (displacedGlobalMuons)
//   outer  = outerTrack  (displacedStandAloneMuons, not UpdatedAtVtx)
// Several displacedMuons pointing to one track give one unit (ID flags OR-ed).
// Interim duplicate removal (user decisions 2026-10-15), before units are built:
// for an inner track used by exactly two displacedMuons, if the track crossed the
// IP (TrackDetectorAssociator::crossedIP, needs the TrackExtra) the higher-index
// muon is dropped; otherwise, if exactly one of the two has a global track, the
// other is dropped. Unresolved cases (no TrackExtra, both or neither global,
// more than two users) keep all muons.
//
// Matching: station-2 position eta/phi (same PropagateToMuon setup as the
// ntuplizer), same charge, dR < 0.06 inner/global and < 0.10 outer, strict
// one-to-one global-nearest greedy with compatible fallback, each type alone.
// Fallback (second greedy pass, only gen muons whose station-2 propagation
// failed, only tracks not taken in the station-2 pass, same charge and dR):
//   inner         -> ECAL inner boundary (EcalDetIdAssociator volume, first
//                    along-momentum crossing; reco from the track's reference state)
//   global, outer -> muon station 1 (PropagateToMuon, useStation2=false)
// Every match stores its surface and the gen/reco eta/phi there.
//
// Fits: KalmanVertexFitter(refit=true) for the 9 ordered combinations
// (mu- type, mu+ type), index 3*typeMuMinus + typeMuPlus. No ID is applied;
// ID pass flags are stored per matched unit.
//
// Object-level hierarchy (vertexing step 2, user decisions 2026-10-15; branches hier*):
// each displacedMuon kept by the dedup gets one representative track:
//   isGlobalMuon -> global; else isTrackerMuon -> inner; else outer passing the
//   DSA ID -> outer; else the object is not used. Objects with the same
//   representative track are one candidate. Candidates are matched to gen like the
//   units (same charge, dR of the representative's type, greedy one-to-one over
//   all candidates), at station 2 with a station-1 fallback for every type (no
//   ECAL fallback). The two matched representatives are fitted as above.
// Merging check (hierOuterCheck): DSA-ID'd tracks of the raw displacedStandAloneMuons
// collection are matched to gen the same way (dR < drOuter); for each matched
// representative: 1 no ID'd DSA matched to that gen muon, 2 it is an outer track of
// the representative's objects, 3 it is not but the objects have an outer track,
// 4 the objects have no outer track.
// Merging diagnostics (dsa*, hierDsa*, per gen muon with a matched DSA): DSA
// kinematics, hits and owner displacedMuons; station-2 dR of the DSA and of the
// representative to the other gen muon; station-2 dR between the representative and
// the DSA; whether the DSA belongs to the representative matched to the other gen muon.
//
// PF muons (branches pf*, user decisions 2026-10-18): slimmedMuons with isPFMuon and
// (isGlobalMuon or isTrackerMuon); representative global track if isGlobalMuon, else
// inner track; matched to gen exactly like the hierarchy candidates (station 2 with
// station-1 fallback, same charge, dR < drGlobal / drInner, greedy one-to-one over all
// PF candidates), independently of the displacedMuons. pfMet* = slimmedMETs[0] Type-1.
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MuonAnalysis/MuonAssociators/interface/PropagateToMuonSetup.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/DetIdAssociatorRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrackAssociator/interface/DetIdAssociator.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "TMath.h"
#include "TNamed.h"
#include "TTree.h"

namespace {

constexpr int kNTypes = 3;  // 0 inner, 1 global, 2 outer
constexpr int kNCombos = kNTypes * kNTypes;
constexpr double kMuonMass = 0.1056583755;

enum FitStatus { kNotMatched = 0, kSharedTrack = 1, kFitFailed = 2, kFitValid = 3 };
enum MatchSurface { kNoMatch = 0, kStation2 = 1, kStation1Fallback = 2, kEcalFallback = 3 };
enum EcalStatus { kEcalNotRun = 0, kEcalOk = 1, kEcalStartBeyond = 2, kEcalFailed = 3 };

struct Pos {
  bool valid = false;
  float eta = -999.f;
  float phi = -999.f;
  int status = 0;  // ECAL only: EcalStatus
};

Pos toPos(const TrajectoryStateOnSurface& tsos) {
  Pos out;
  if (!tsos.isValid()) return out;
  out.valid = true;
  out.eta = tsos.globalPosition().eta();
  out.phi = tsos.globalPosition().phi();
  return out;
}

// ECAL inner boundary, as in SharedTrackTrajectoryCheck.cc: barrel cylinder
// rho = minR (|z| <= minZ) or endcap planes |z| = minZ (rho <= minR); first
// along-momentum crossing.
Pos toEcal(const FreeTrajectoryState& start, const Propagator& along, double minR, double minZ) {
  Pos out;
  out.status = kEcalFailed;
  if (start.position().perp() >= minR || std::abs(start.position().z()) >= minZ) {
    out.status = kEcalStartBeyond;
    return out;
  }
  double bestPath = std::numeric_limits<double>::max();
  auto consider = [&](const std::pair<TrajectoryStateOnSurface, double>& r, bool barrel) {
    if (!r.first.isValid()) return;
    const GlobalPoint p = r.first.globalPosition();
    if (barrel ? std::abs(p.z()) > minZ : p.perp() > minR) return;
    if (r.second < bestPath) {
      bestPath = r.second;
      out.valid = true;
      out.status = kEcalOk;
      out.eta = p.eta();
      out.phi = p.phi();
    }
  };
  const auto cylinder = Cylinder::build(minR, Cylinder::PositionType(0, 0, 0), Cylinder::RotationType());
  consider(along.propagateWithPath(start, *cylinder), true);
  for (const double z : {minZ, -minZ}) {
    const auto plane = Plane::build(Plane::PositionType(0, 0, z), Plane::RotationType());
    consider(along.propagateWithPath(start, *plane), false);
  }
  return out;
}

// Same definitions as ElectronSkimmer.cc.
const reco::GenParticle* firstDifferentMother(const reco::GenParticle& p) {
  const reco::GenParticle* cur = &p;
  std::set<const reco::GenParticle*> seen;
  while (cur) {
    if (seen.count(cur)) return nullptr;
    seen.insert(cur);
    if (cur->numberOfMothers() == 0) return nullptr;
    const auto* mom = dynamic_cast<const reco::GenParticle*>(cur->mother(0));
    if (!mom) return nullptr;
    if (mom->pdgId() != p.pdgId()) return mom;
    cur = mom;
  }
  return nullptr;
}

bool isSignalMuon(const reco::GenParticle& p) {
  if (std::abs(p.pdgId()) != 13 || p.status() != 1 || !p.isLastCopy()) return false;
  const reco::GenParticle* mom = firstDifferentMother(p);
  return mom && std::abs(mom->pdgId()) == 1000023;
}

// ElectronSkimmer::passesDisplacedID (DSA ID).
bool passesDisplacedID(const reco::Track& t) {
  const int csc = t.hitPattern().numberOfValidMuonCSCHits();
  const int dt = t.hitPattern().numberOfValidMuonDTHits();
  return csc + dt > 12 && (csc != 0 || dt > 18) && t.normalizedChi2() < 2.5 && t.ptError() / t.pt() < 1;
}

using RefKey = std::pair<edm::ProductID, unsigned int>;
RefKey refKey(const reco::TrackRef& r) { return {r.id(), static_cast<unsigned int>(r.key())}; }

struct Unit {
  reco::TrackRef ref;
  int charge = 0;
  Pos st2;
  Pos fallback;  // inner: ECAL; global/outer: station 1 (filled only when a gen muon needs it)
  bool parentTracker = false;
  bool parentGlobal = false;
  int nParents = 0;
  // Tracks this leg is built from: the track itself; for a global track also
  // its parent muons' inner and outer tracks; for an outer track also its
  // parent muons' inner tracks. Legs sharing any are not fitted, so muons that
  // share an inner track are never vertexed together, whatever the leg types.
  std::set<RefKey> constituents;
  std::set<RefKey> ownOuters;  // hierarchy candidates: outer tracks of the parent muons
  int type = -1;               // hierarchy candidates: representative type
  int ownOuterId = -1;         // hierarchy candidates: -1 no own outer, else OR of DSA ID over own outers
  int firstParent = -1;        // hierarchy candidates: lowest displacedMuons index
};

struct Gen {
  const reco::GenParticle* p = nullptr;
  bool accepted = false;
  Pos st2, st1, ecal;
};

// Strict one-to-one global-nearest greedy matching with compatible fallback.
// genPos/unitPos give the surface positions; only eligible objects take part.
std::array<int, 2> greedyMatch(const std::array<Gen, 2>& gen,
                               const std::array<const Pos*, 2>& genPos,
                               const std::array<bool, 2>& genEligible,
                               const std::vector<Unit>& units,
                               const std::vector<const Pos*>& unitPos,
                               const std::vector<bool>& unitEligible,
                               double threshold,
                               std::array<float, 2>& dROut) {
  std::vector<std::tuple<float, int, int>> pairs;
  for (int g = 0; g < 2; ++g) {
    if (!genEligible[g] || !genPos[g]->valid) continue;
    for (size_t u = 0; u < units.size(); ++u) {
      if (!unitEligible[u] || !unitPos[u]->valid || units[u].charge != gen[g].p->charge()) continue;
      const float dr = reco::deltaR(genPos[g]->eta, genPos[g]->phi, unitPos[u]->eta, unitPos[u]->phi);
      if (dr < threshold) pairs.emplace_back(dr, g, static_cast<int>(u));
    }
  }
  std::sort(pairs.begin(), pairs.end());
  std::array<int, 2> match{{-1, -1}};
  dROut = {{-1.f, -1.f}};
  std::vector<bool> used(units.size(), false);
  for (const auto& [dr, g, u] : pairs) {
    if (match[g] >= 0 || used[u]) continue;
    match[g] = u;
    used[u] = true;
    dROut[g] = dr;
  }
  return match;
}

struct FitResult {
  int status = kNotMatched;
  float vx = -999.f, vy = -999.f, vz = -999.f, cxx = -999.f, cyy = -999.f, czz = -999.f, cxy = -999.f,
        cxz = -999.f, cyz = -999.f, chi2 = -999.f, ndof = -999.f, prob = -999.f, resAlong = -999.f,
        resPerp = -999.f, resZ = -999.f, sigAlong = -999.f, sigPerp = -999.f, lxyBS = -999.f, sigLxyBS = -999.f,
        mass = -999.f, massRefit = -999.f;
};

// Fit two legs; legs sharing a constituent are not fitted. Residuals relative to the
// gen decay vertex along/perpendicular to the gen transverse flight direction (ux, uy).
FitResult fitPair(const Unit& u0, const Unit& u1, const TransientTrackBuilder& ttb, const KalmanVertexFitter& kvf,
                  const reco::BeamSpot& bs, const float* genVtx, double ux, double uy) {
  FitResult f;
  for (const auto& k : u0.constituents)
    if (u1.constituents.count(k)) {
      f.status = kSharedTrack;
      return f;
    }
  std::vector<reco::TransientTrack> tts{ttb.build(u0.ref), ttb.build(u1.ref)};
  const TransientVertex tv = kvf.vertex(tts);
  if (!tv.isValid()) {
    f.status = kFitFailed;
    return f;
  }
  f.status = kFitValid;
  const reco::Vertex v(tv);
  f.vx = v.x();
  f.vy = v.y();
  f.vz = v.z();
  f.cxx = v.covariance(0, 0);
  f.cyy = v.covariance(1, 1);
  f.czz = v.covariance(2, 2);
  f.cxy = v.covariance(0, 1);
  f.cxz = v.covariance(0, 2);
  f.cyz = v.covariance(1, 2);
  f.chi2 = v.chi2();
  f.ndof = v.ndof();
  f.prob = TMath::Prob(v.chi2(), static_cast<int>(std::lround(v.ndof())));
  const double dx = v.x() - genVtx[0], dy = v.y() - genVtx[1];
  f.resAlong = dx * ux + dy * uy;
  f.resPerp = -dx * uy + dy * ux;
  f.resZ = v.z() - genVtx[2];
  const double cxx = v.covariance(0, 0), cyy = v.covariance(1, 1), cxy = v.covariance(0, 1);
  f.sigAlong = std::sqrt(std::max(0., ux * ux * cxx + uy * uy * cyy + 2 * ux * uy * cxy));
  f.sigPerp = std::sqrt(std::max(0., uy * uy * cxx + ux * ux * cyy - 2 * ux * uy * cxy));
  const double bx = v.x() - bs.x(v.z()), by = v.y() - bs.y(v.z());
  const double lxy = std::hypot(bx, by);
  f.lxyBS = lxy;
  if (lxy > 0) f.sigLxyBS = std::sqrt(std::max(0., bx * bx * cxx + by * by * cyy + 2 * bx * by * cxy)) / lxy;
  auto p4 = [](const reco::Track& t) { return math::PtEtaPhiMLorentzVector(t.pt(), t.eta(), t.phi(), kMuonMass); };
  f.mass = (p4(*u0.ref) + p4(*u1.ref)).mass();
  if (tv.hasRefittedTracks() && tv.refittedTracks().size() == 2) {
    const reco::Track r0 = tv.refittedTracks()[0].track();
    const reco::Track r1 = tv.refittedTracks()[1].track();
    f.massRefit = (p4(r0) + p4(r1)).mass();
  }
  return f;
}

// Station 2 pass, then station-1 pass for accepted gen muons failing station 2 on
// candidates unused in pass 1. surface: kStation2 / kStation1Fallback / kNoMatch.
std::array<int, 2> matchSt2St1(const std::array<Gen, 2>& gen, const std::vector<Unit>& cands,
                               const std::vector<double>& thr, std::array<float, 2>& dROut,
                               std::array<int, 2>& surface) {
  std::vector<const Pos*> st2(cands.size()), st1(cands.size());
  for (size_t u = 0; u < cands.size(); ++u) {
    st2[u] = &cands[u].st2;
    st1[u] = &cands[u].fallback;
  }
  auto pass = [&](const std::array<const Pos*, 2>& genPos, const std::array<bool, 2>& genOk,
                  const std::vector<const Pos*>& pos, const std::vector<bool>& candOk, std::array<float, 2>& dr) {
    // Pairs from all candidates with their own thresholds, then one greedy assignment.
    std::vector<std::tuple<float, int, int>> pairs;
    for (int g = 0; g < 2; ++g) {
      if (!genOk[g] || !genPos[g]->valid) continue;
      for (size_t u = 0; u < cands.size(); ++u) {
        if (!candOk[u] || !pos[u]->valid || cands[u].charge != gen[g].p->charge()) continue;
        const float d = reco::deltaR(genPos[g]->eta, genPos[g]->phi, pos[u]->eta, pos[u]->phi);
        if (d < thr[u]) pairs.emplace_back(d, g, static_cast<int>(u));
      }
    }
    std::sort(pairs.begin(), pairs.end());
    std::array<int, 2> m{{-1, -1}};
    dr = {{-1.f, -1.f}};
    std::vector<bool> used(cands.size(), false);
    for (const auto& [d, g, u] : pairs) {
      if (m[g] >= 0 || used[u]) continue;
      m[g] = u;
      used[u] = true;
      dr[g] = d;
    }
    return m;
  };
  std::array<float, 2> dr2, dr1;
  std::array<int, 2> m = pass({{&gen[0].st2, &gen[1].st2}}, {{gen[0].accepted, gen[1].accepted}}, st2,
                              std::vector<bool>(cands.size(), true), dr2);
  std::vector<bool> unused(cands.size(), true);
  for (int g = 0; g < 2; ++g)
    if (m[g] >= 0) unused[m[g]] = false;
  const auto m1 = pass({{&gen[0].st1, &gen[1].st1}},
                       {{gen[0].accepted && !gen[0].st2.valid, gen[1].accepted && !gen[1].st2.valid}}, st1, unused,
                       dr1);
  for (int g = 0; g < 2; ++g) {
    surface[g] = kNoMatch;
    dROut[g] = -1.f;
    if (m[g] >= 0) {
      surface[g] = kStation2;
      dROut[g] = dr2[g];
    } else if (m1[g] >= 0) {
      m[g] = m1[g];
      surface[g] = kStation1Fallback;
      dROut[g] = dr1[g];
    }
  }
  return m;
}

}  // namespace

class DimuonVertexTruth : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit DimuonVertexTruth(const edm::ParameterSet&);
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

private:
  void resetBranches();
  void setFit(int c, const FitResult& f);

  const edm::EDGetTokenT<std::vector<reco::Muon>> muonToken_;
  const edm::EDGetTokenT<std::vector<reco::GenParticle>> genToken_;
  const edm::EDGetTokenT<reco::BeamSpot> beamspotToken_;
  const edm::EDGetTokenT<std::vector<reco::Track>> dsaToken_;
  const edm::EDGetTokenT<std::vector<pat::Muon>> pfMuonToken_;
  const edm::EDGetTokenT<std::vector<pat::MET>> metToken_;
  const edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> ttbToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magFieldToken_;
  const PropagateToMuonSetup propagatorSetup_;
  const PropagateToMuonSetup propagatorSetupSt1_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> alongToken_;
  const edm::ESGetToken<DetIdAssociator, DetIdAssociatorRecord> ecalToken_;
  const std::string sample_;
  const std::array<double, kNTypes> drThreshold_;

  TTree* tree_ = nullptr;
  TTree* totals_ = nullptr;
  Long64_t nEvents_ = 0, nTwoSignal_ = 0, nBothAccepted_ = 0, nBothAcceptedPropagated_ = 0;

  // Event
  unsigned int run_ = 0, lumi_ = 0;
  unsigned long long event_ = 0;
  float genPV_[3], genVtx_[3], genVtxSep_, genLxy_, genLxyPV_, genDimuonPt_, genDimuonEta_, genDimuonMass_,
      genDimuonDR_, flightDirT_[2];
  // Gen muons: index 0 = mu- (pdgId 13), 1 = mu+ (pdgId -13)
  float gPt_[2], gEta_[2], gPhi_[2], gVx_[2], gVy_[2], gVz_[2], gD0_[2], gSt2Eta_[2], gSt2Phi_[2];
  int gCharge_[2], gAccepted_[2], gSt2Valid_[2];
  // Fallback surfaces, computed for every gen muon (used only when station 2 fails)
  float gSt1Eta_[2], gSt1Phi_[2], gEcalEta_[2], gEcalPhi_[2];
  int gSt1Valid_[2], gEcalStatus_[2];
  // Matches: index 3*g + type
  int nUnits_[kNTypes];
  int nDupDropped_, nDupUnresolved_;  // muons dropped; shared inner tracks left unresolved
  int mMatched_[2 * kNTypes], mIdPass_[2 * kNTypes], mNParents_[2 * kNTypes], mCharge_[2 * kNTypes];
  float mDR_[2 * kNTypes], mPt_[2 * kNTypes], mEta_[2 * kNTypes], mPhi_[2 * kNTypes];
  // Surface of the match (MatchSurface) and gen/reco eta/phi on it
  int mSurface_[2 * kNTypes];
  float mGenSurfEta_[2 * kNTypes], mGenSurfPhi_[2 * kNTypes], mRecoSurfEta_[2 * kNTypes],
      mRecoSurfPhi_[2 * kNTypes];
  // Fits: index 3*typeMuMinus + typeMuPlus
  int fStatus_[kNCombos];
  float fVx_[kNCombos], fVy_[kNCombos], fVz_[kNCombos], fCxx_[kNCombos], fCyy_[kNCombos], fCzz_[kNCombos],
      fCxy_[kNCombos], fCxz_[kNCombos], fCyz_[kNCombos], fChi2_[kNCombos], fNdof_[kNCombos], fProb_[kNCombos],
      fResAlong_[kNCombos], fResPerp_[kNCombos], fResZ_[kNCombos], fSigAlong_[kNCombos], fSigPerp_[kNCombos],
      fLxyBS_[kNCombos], fSigLxyBS_[kNCombos], fMass_[kNCombos], fMassRefit_[kNCombos];
  // Object-level hierarchy: candidates, per-gen matches (index g), one fit
  int hierNCand_, hierNCandType_[kNTypes], nDSA_, nDSAId_, nMuOuter_, nMuOuterInDSA_;
  int hMatched_[2], hType_[2], hSurface_[2], hNParents_[2], hCharge_[2], hOuterCheck_[2], hOwnOuterId_[2];
  float hDR_[2], hPt_[2], hEta_[2], hPhi_[2];
  // Merging diagnostics for the DSA matched to each gen muon
  int dSurface_[2], dNDT_[2], dNCSC_[2], dNStations_[2], dOwnerN_[2], dOwnerTracker_[2], dOwnerGlobal_[2],
      dOwnerInner_[2], dOtherRep_[2], hNMatchedStations_[2], dOwnerCandType_[2], dOwnerInnerSame_[2];
  float dDR_[2], dPt_[2], dEta_[2], dPhi_[2], dPtErr_[2], dNormChi2_[2], dDROtherGen_[2], hDROtherGen_[2],
      hDsaDRSt2_[2], dOwnerInnerDRRep_[2], dOwnerInnerPtRatio_[2], dOwnerCandDR_[2];
  FitResult hFit_;
  // PF muons (index g) and PF MET
  int pfNCand_, pfNCandType_[2], pfNoTrack_;
  int pfMatched_[2], pfType_[2], pfSurface_[2], pfCharge_[2];
  float pfDR_[2], pfPt_[2], pfEta_[2], pfPhi_[2], pfMuPt_[2], pfMuPhi_[2], pfMetPt_, pfMetPhi_;
};

DimuonVertexTruth::DimuonVertexTruth(const edm::ParameterSet& ps)
    : muonToken_(consumes<std::vector<reco::Muon>>(ps.getParameter<edm::InputTag>("muons"))),
      genToken_(consumes<std::vector<reco::GenParticle>>(ps.getParameter<edm::InputTag>("genParticles"))),
      beamspotToken_(consumes<reco::BeamSpot>(ps.getParameter<edm::InputTag>("beamspot"))),
      dsaToken_(consumes<std::vector<reco::Track>>(ps.getParameter<edm::InputTag>("standAloneMuons"))),
      pfMuonToken_(consumes<std::vector<pat::Muon>>(ps.getParameter<edm::InputTag>("pfMuons"))),
      metToken_(consumes<std::vector<pat::MET>>(ps.getParameter<edm::InputTag>("met"))),
      ttbToken_(esConsumes(edm::ESInputTag("", "TransientTrackBuilder"))),
      magFieldToken_(esConsumes()),
      propagatorSetup_(ps.getParameter<edm::ParameterSet>("propagatorSt2"), consumesCollector()),
      propagatorSetupSt1_(ps.getParameter<edm::ParameterSet>("propagatorSt1"), consumesCollector()),
      alongToken_(esConsumes(edm::ESInputTag("", "SteppingHelixPropagatorAlong"))),
      ecalToken_(esConsumes(edm::ESInputTag("", "EcalDetIdAssociator"))),
      sample_(ps.getParameter<std::string>("sample")),
      drThreshold_{{ps.getParameter<double>("drInner"), ps.getParameter<double>("drGlobal"),
                    ps.getParameter<double>("drOuter")}} {
  usesResource(TFileService::kSharedResource);
  edm::Service<TFileService> fs;
  fs->make<TNamed>("sample", sample_.c_str());
  fs->make<TNamed>("definitions",
                   "types: 0 inner(displacedTracks) 1 global(displacedGlobalMuons) 2 outer(displacedStandAloneMuons); "
                   "gen 0 = mu-, 1 = mu+; match index 3*g+type; fit index 3*typeMuMinus+typeMuPlus; "
                   "interim dedup of displacedMuons sharing an inner track (2 users only; TrackExtra needed): crossedIP -> drop higher index, "
                   "else drop the one without a global track if exactly one has it; others unresolved (nDupDropped, nDupUnresolved); "
                   "fitStatus 0 not both matched, 1 legs share a track, global legs incl. parent inner/outer, outer legs incl. parent inner (not fitted), 2 fit failed, 3 valid; "
                   "match: station 2 position, same charge, greedy 1-to-1, dR<0.06/0.06/0.10; "
                   "fallback: 2nd greedy pass for gen muons failing station 2, unused tracks, same charge and dR, "
                   "inner at ECAL inner boundary (reco from reference state), global/outer at station 1; "
                   "matchSurface 0 none, 1 station 2, 2 station-1 fallback, 3 ECAL fallback; "
                   "genEcalStatus 0 not run, 1 ok, 2 start beyond boundary, 3 failed; "
                   "residual = reco - gen decay vertex, along/perp = gen transverse flight direction (decay - gen PV); "
                   "idPass: inner parent isTracker, global parent isGlobal, outer DSA ID (OR over parents); "
                   "hier*: object-level hierarchy per kept displacedMuon: isGlobal -> global, else isTracker -> inner, "
                   "else DSA-ID'd outer -> outer, else unused; same representative track = one candidate; "
                   "matched at station 2 with station-1 fallback for all types (hierSurface 1/2), dR by representative type; "
                   "hierType 0 inner 1 global 2 outer -1 none; hierFit* = fit of the two matched representatives (status codes as fitStatus); "
                   "hierOuterCheck: DSA-ID'd raw displacedStandAloneMuons matched to the same gen muon (st2 + st1 fallback, dR<drOuter): "
                   "0 not matched, 1 none matched, 2 it is an outer of the representative's objects, 3 it is not but they have an outer, "
                   "4 they have no outer; hierOwnOuterId -1 no own outer, else DSA ID of the own outer(s) (OR); "
                   "dsa* (filled when a DSA is matched to the gen muon): surface, dR, kinematics, DT/CSC hits, stations, "
                   "owners among kept displacedMuons (count, OR isTracker/isGlobal/has inner), dsaOfOtherRep = DSA belongs to the "
                   "representative matched to the other gen muon; dsaDROtherGen / hierDROtherGen = station-2 dR to the other gen muon; "
                   "hierDsaDRSt2 = station-2 dR between representative and DSA; hierNMatchedStations of the lowest-index parent; "
                   "lowest-index DSA owner: dsaOwnerInnerSame (its inner is a constituent of the representative; -1 no inner), "
                   "dsaOwnerInnerDRRep / PtRatio (its inner vs the representative track, at the reference point), "
                   "dsaOwnerCandType / CandDR (its hierarchy candidate's type and station-2 dR to this gen muon; -1/-999 none); "
                   "pf*: slimmedMuons with isPFMuon and (isGlobalMuon or isTrackerMuon), representative global track if "
                   "isGlobalMuon else inner track (pfType 1 / 0, -1 none), matched like hier* (st2 + st1 fallback, same charge, "
                   "dR<drGlobal/drInner, greedy 1-to-1 over PF candidates), independent of displacedMuons; pfPt/Eta/Phi = "
                   "representative track at its reference point, pfMuPt/Phi = pat::Muon p4; pfNoTrack = PF candidates "
                   "skipped because the flagged track is missing; pfMetPt/Phi = slimmedMETs[0] Type-1 (-999 if empty)");
  tree_ = fs->make<TTree>("events", "truth-matched dimuon vertex fits");
  auto& t = *tree_;
  t.Branch("run", &run_);
  t.Branch("lumi", &lumi_);
  t.Branch("event", &event_);
  t.Branch("genPV", genPV_, "genPV[3]/F");
  t.Branch("genVtx", genVtx_, "genVtx[3]/F");
  t.Branch("genVtxSep", &genVtxSep_);
  t.Branch("genLxy", &genLxy_);
  t.Branch("genLxyPV", &genLxyPV_);
  t.Branch("flightDirT", flightDirT_, "flightDirT[2]/F");
  t.Branch("genDimuonPt", &genDimuonPt_);
  t.Branch("genDimuonEta", &genDimuonEta_);
  t.Branch("genDimuonMass", &genDimuonMass_);
  t.Branch("genDimuonDR", &genDimuonDR_);
  t.Branch("genPt", gPt_, "genPt[2]/F");
  t.Branch("genEta", gEta_, "genEta[2]/F");
  t.Branch("genPhi", gPhi_, "genPhi[2]/F");
  t.Branch("genCharge", gCharge_, "genCharge[2]/I");
  t.Branch("genVx", gVx_, "genVx[2]/F");
  t.Branch("genVy", gVy_, "genVy[2]/F");
  t.Branch("genVz", gVz_, "genVz[2]/F");
  t.Branch("genD0", gD0_, "genD0[2]/F");
  t.Branch("genAccepted", gAccepted_, "genAccepted[2]/I");
  t.Branch("genSt2Valid", gSt2Valid_, "genSt2Valid[2]/I");
  t.Branch("genSt2Eta", gSt2Eta_, "genSt2Eta[2]/F");
  t.Branch("genSt2Phi", gSt2Phi_, "genSt2Phi[2]/F");
  t.Branch("genSt1Valid", gSt1Valid_, "genSt1Valid[2]/I");
  t.Branch("genSt1Eta", gSt1Eta_, "genSt1Eta[2]/F");
  t.Branch("genSt1Phi", gSt1Phi_, "genSt1Phi[2]/F");
  t.Branch("genEcalStatus", gEcalStatus_, "genEcalStatus[2]/I");
  t.Branch("genEcalEta", gEcalEta_, "genEcalEta[2]/F");
  t.Branch("genEcalPhi", gEcalPhi_, "genEcalPhi[2]/F");
  t.Branch("nUnits", nUnits_, "nUnits[3]/I");
  t.Branch("nDupDropped", &nDupDropped_);
  t.Branch("nDupUnresolved", &nDupUnresolved_);
  t.Branch("matched", mMatched_, "matched[6]/I");
  t.Branch("matchDR", mDR_, "matchDR[6]/F");
  t.Branch("matchPt", mPt_, "matchPt[6]/F");
  t.Branch("matchEta", mEta_, "matchEta[6]/F");
  t.Branch("matchPhi", mPhi_, "matchPhi[6]/F");
  t.Branch("matchCharge", mCharge_, "matchCharge[6]/I");
  t.Branch("matchIdPass", mIdPass_, "matchIdPass[6]/I");
  t.Branch("matchNParents", mNParents_, "matchNParents[6]/I");
  t.Branch("matchSurface", mSurface_, "matchSurface[6]/I");
  t.Branch("matchGenSurfEta", mGenSurfEta_, "matchGenSurfEta[6]/F");
  t.Branch("matchGenSurfPhi", mGenSurfPhi_, "matchGenSurfPhi[6]/F");
  t.Branch("matchRecoSurfEta", mRecoSurfEta_, "matchRecoSurfEta[6]/F");
  t.Branch("matchRecoSurfPhi", mRecoSurfPhi_, "matchRecoSurfPhi[6]/F");
  t.Branch("fitStatus", fStatus_, "fitStatus[9]/I");
  t.Branch("fitVx", fVx_, "fitVx[9]/F");
  t.Branch("fitVy", fVy_, "fitVy[9]/F");
  t.Branch("fitVz", fVz_, "fitVz[9]/F");
  t.Branch("fitCxx", fCxx_, "fitCxx[9]/F");
  t.Branch("fitCyy", fCyy_, "fitCyy[9]/F");
  t.Branch("fitCzz", fCzz_, "fitCzz[9]/F");
  t.Branch("fitCxy", fCxy_, "fitCxy[9]/F");
  t.Branch("fitCxz", fCxz_, "fitCxz[9]/F");
  t.Branch("fitCyz", fCyz_, "fitCyz[9]/F");
  t.Branch("fitChi2", fChi2_, "fitChi2[9]/F");
  t.Branch("fitNdof", fNdof_, "fitNdof[9]/F");
  t.Branch("fitProb", fProb_, "fitProb[9]/F");
  t.Branch("fitResAlong", fResAlong_, "fitResAlong[9]/F");
  t.Branch("fitResPerp", fResPerp_, "fitResPerp[9]/F");
  t.Branch("fitResZ", fResZ_, "fitResZ[9]/F");
  t.Branch("fitSigAlong", fSigAlong_, "fitSigAlong[9]/F");
  t.Branch("fitSigPerp", fSigPerp_, "fitSigPerp[9]/F");
  t.Branch("fitLxyBS", fLxyBS_, "fitLxyBS[9]/F");
  t.Branch("fitSigLxyBS", fSigLxyBS_, "fitSigLxyBS[9]/F");
  t.Branch("fitMass", fMass_, "fitMass[9]/F");
  t.Branch("fitMassRefit", fMassRefit_, "fitMassRefit[9]/F");
  t.Branch("hierNCand", &hierNCand_);
  t.Branch("hierNCandType", hierNCandType_, "hierNCandType[3]/I");
  t.Branch("nDSA", &nDSA_);
  t.Branch("nDSAId", &nDSAId_);
  t.Branch("nMuOuter", &nMuOuter_);
  t.Branch("nMuOuterInDSA", &nMuOuterInDSA_);
  t.Branch("hierMatched", hMatched_, "hierMatched[2]/I");
  t.Branch("hierType", hType_, "hierType[2]/I");
  t.Branch("hierSurface", hSurface_, "hierSurface[2]/I");
  t.Branch("hierNParents", hNParents_, "hierNParents[2]/I");
  t.Branch("hierCharge", hCharge_, "hierCharge[2]/I");
  t.Branch("hierOuterCheck", hOuterCheck_, "hierOuterCheck[2]/I");
  t.Branch("hierOwnOuterId", hOwnOuterId_, "hierOwnOuterId[2]/I");
  t.Branch("hierDR", hDR_, "hierDR[2]/F");
  t.Branch("hierPt", hPt_, "hierPt[2]/F");
  t.Branch("hierEta", hEta_, "hierEta[2]/F");
  t.Branch("hierPhi", hPhi_, "hierPhi[2]/F");
  t.Branch("dsaSurface", dSurface_, "dsaSurface[2]/I");
  t.Branch("dsaNDT", dNDT_, "dsaNDT[2]/I");
  t.Branch("dsaNCSC", dNCSC_, "dsaNCSC[2]/I");
  t.Branch("dsaNStations", dNStations_, "dsaNStations[2]/I");
  t.Branch("dsaOwnerN", dOwnerN_, "dsaOwnerN[2]/I");
  t.Branch("dsaOwnerTracker", dOwnerTracker_, "dsaOwnerTracker[2]/I");
  t.Branch("dsaOwnerGlobal", dOwnerGlobal_, "dsaOwnerGlobal[2]/I");
  t.Branch("dsaOwnerInner", dOwnerInner_, "dsaOwnerInner[2]/I");
  t.Branch("dsaOfOtherRep", dOtherRep_, "dsaOfOtherRep[2]/I");
  t.Branch("dsaDR", dDR_, "dsaDR[2]/F");
  t.Branch("dsaPt", dPt_, "dsaPt[2]/F");
  t.Branch("dsaEta", dEta_, "dsaEta[2]/F");
  t.Branch("dsaPhi", dPhi_, "dsaPhi[2]/F");
  t.Branch("dsaPtErr", dPtErr_, "dsaPtErr[2]/F");
  t.Branch("dsaNormChi2", dNormChi2_, "dsaNormChi2[2]/F");
  t.Branch("dsaDROtherGen", dDROtherGen_, "dsaDROtherGen[2]/F");
  t.Branch("hierDROtherGen", hDROtherGen_, "hierDROtherGen[2]/F");
  t.Branch("hierDsaDRSt2", hDsaDRSt2_, "hierDsaDRSt2[2]/F");
  t.Branch("hierNMatchedStations", hNMatchedStations_, "hierNMatchedStations[2]/I");
  t.Branch("dsaOwnerCandType", dOwnerCandType_, "dsaOwnerCandType[2]/I");
  t.Branch("dsaOwnerInnerSame", dOwnerInnerSame_, "dsaOwnerInnerSame[2]/I");
  t.Branch("dsaOwnerInnerDRRep", dOwnerInnerDRRep_, "dsaOwnerInnerDRRep[2]/F");
  t.Branch("dsaOwnerInnerPtRatio", dOwnerInnerPtRatio_, "dsaOwnerInnerPtRatio[2]/F");
  t.Branch("dsaOwnerCandDR", dOwnerCandDR_, "dsaOwnerCandDR[2]/F");
  t.Branch("hierFitStatus", &hFit_.status);
  t.Branch("hierFitVx", &hFit_.vx);
  t.Branch("hierFitVy", &hFit_.vy);
  t.Branch("hierFitVz", &hFit_.vz);
  t.Branch("hierFitCxx", &hFit_.cxx);
  t.Branch("hierFitCyy", &hFit_.cyy);
  t.Branch("hierFitCzz", &hFit_.czz);
  t.Branch("hierFitCxy", &hFit_.cxy);
  t.Branch("hierFitCxz", &hFit_.cxz);
  t.Branch("hierFitCyz", &hFit_.cyz);
  t.Branch("hierFitChi2", &hFit_.chi2);
  t.Branch("hierFitNdof", &hFit_.ndof);
  t.Branch("hierFitProb", &hFit_.prob);
  t.Branch("hierFitResAlong", &hFit_.resAlong);
  t.Branch("hierFitResPerp", &hFit_.resPerp);
  t.Branch("hierFitResZ", &hFit_.resZ);
  t.Branch("hierFitSigAlong", &hFit_.sigAlong);
  t.Branch("hierFitSigPerp", &hFit_.sigPerp);
  t.Branch("hierFitLxyBS", &hFit_.lxyBS);
  t.Branch("hierFitSigLxyBS", &hFit_.sigLxyBS);
  t.Branch("hierFitMass", &hFit_.mass);
  t.Branch("hierFitMassRefit", &hFit_.massRefit);
  t.Branch("pfNCand", &pfNCand_);
  t.Branch("pfNCandType", pfNCandType_, "pfNCandType[2]/I");
  t.Branch("pfNoTrack", &pfNoTrack_);
  t.Branch("pfMatched", pfMatched_, "pfMatched[2]/I");
  t.Branch("pfType", pfType_, "pfType[2]/I");
  t.Branch("pfSurface", pfSurface_, "pfSurface[2]/I");
  t.Branch("pfCharge", pfCharge_, "pfCharge[2]/I");
  t.Branch("pfDR", pfDR_, "pfDR[2]/F");
  t.Branch("pfPt", pfPt_, "pfPt[2]/F");
  t.Branch("pfEta", pfEta_, "pfEta[2]/F");
  t.Branch("pfPhi", pfPhi_, "pfPhi[2]/F");
  t.Branch("pfMuPt", pfMuPt_, "pfMuPt[2]/F");
  t.Branch("pfMuPhi", pfMuPhi_, "pfMuPhi[2]/F");
  t.Branch("pfMetPt", &pfMetPt_);
  t.Branch("pfMetPhi", &pfMetPhi_);

  totals_ = fs->make<TTree>("totals", "event counters");
  totals_->Branch("events", &nEvents_);
  totals_->Branch("twoSignalMuons", &nTwoSignal_);
  totals_->Branch("bothAccepted", &nBothAccepted_);
  totals_->Branch("bothAcceptedPropagated", &nBothAcceptedPropagated_);
}

void DimuonVertexTruth::resetBranches() {
  auto fill = [](auto* arr, int n, auto v) { std::fill(arr, arr + n, v); };
  fill(genPV_, 3, -999.f);
  fill(genVtx_, 3, -999.f);
  fill(flightDirT_, 2, 0.f);
  genVtxSep_ = genLxy_ = genLxyPV_ = genDimuonPt_ = genDimuonEta_ = genDimuonMass_ = genDimuonDR_ = -999.f;
  for (auto* a : {gPt_, gEta_, gPhi_, gVx_, gVy_, gVz_, gD0_, gSt2Eta_, gSt2Phi_, gSt1Eta_, gSt1Phi_, gEcalEta_,
                  gEcalPhi_})
    fill(a, 2, -999.f);
  for (auto* a : {gCharge_, gAccepted_, gSt2Valid_, gSt1Valid_, gEcalStatus_}) fill(a, 2, 0);
  fill(nUnits_, kNTypes, 0);
  nDupDropped_ = nDupUnresolved_ = 0;
  for (auto* a : {mMatched_, mIdPass_, mNParents_, mCharge_, mSurface_}) fill(a, 2 * kNTypes, 0);
  for (auto* a : {mDR_, mPt_, mEta_, mPhi_, mGenSurfEta_, mGenSurfPhi_, mRecoSurfEta_, mRecoSurfPhi_})
    fill(a, 2 * kNTypes, -999.f);
  fill(fStatus_, kNCombos, static_cast<int>(kNotMatched));
  for (auto* a : {fVx_, fVy_, fVz_, fCxx_, fCyy_, fCzz_, fCxy_, fCxz_, fCyz_, fChi2_, fNdof_, fProb_, fResAlong_,
                  fResPerp_, fResZ_, fSigAlong_, fSigPerp_, fLxyBS_, fSigLxyBS_, fMass_, fMassRefit_})
    fill(a, kNCombos, -999.f);
  hierNCand_ = nDSA_ = nDSAId_ = nMuOuter_ = nMuOuterInDSA_ = 0;
  fill(hierNCandType_, kNTypes, 0);
  for (auto* a : {hMatched_, hSurface_, hNParents_, hCharge_, hOuterCheck_}) fill(a, 2, 0);
  fill(hType_, 2, -1);
  fill(hOwnOuterId_, 2, -1);
  for (auto* a : {hDR_, hPt_, hEta_, hPhi_}) fill(a, 2, -999.f);
  for (auto* a : {dSurface_, dNDT_, dNCSC_, dNStations_, dOwnerN_, dOwnerTracker_, dOwnerGlobal_, dOwnerInner_,
                  dOtherRep_})
    fill(a, 2, 0);
  fill(hNMatchedStations_, 2, -1);
  fill(dOwnerCandType_, 2, -1);
  fill(dOwnerInnerSame_, 2, -1);
  for (auto* a : {dDR_, dPt_, dEta_, dPhi_, dPtErr_, dNormChi2_, dDROtherGen_, hDROtherGen_, hDsaDRSt2_,
                  dOwnerInnerDRRep_, dOwnerInnerPtRatio_, dOwnerCandDR_})
    fill(a, 2, -999.f);
  hFit_ = FitResult();
  pfNCand_ = pfNoTrack_ = 0;
  fill(pfNCandType_, 2, 0);
  for (auto* a : {pfMatched_, pfSurface_, pfCharge_}) fill(a, 2, 0);
  fill(pfType_, 2, -1);
  for (auto* a : {pfDR_, pfPt_, pfEta_, pfPhi_, pfMuPt_, pfMuPhi_}) fill(a, 2, -999.f);
  pfMetPt_ = pfMetPhi_ = -999.f;
}

void DimuonVertexTruth::setFit(int c, const FitResult& f) {
  fStatus_[c] = f.status;
  fVx_[c] = f.vx;
  fVy_[c] = f.vy;
  fVz_[c] = f.vz;
  fCxx_[c] = f.cxx;
  fCyy_[c] = f.cyy;
  fCzz_[c] = f.czz;
  fCxy_[c] = f.cxy;
  fCxz_[c] = f.cxz;
  fCyz_[c] = f.cyz;
  fChi2_[c] = f.chi2;
  fNdof_[c] = f.ndof;
  fProb_[c] = f.prob;
  fResAlong_[c] = f.resAlong;
  fResPerp_[c] = f.resPerp;
  fResZ_[c] = f.resZ;
  fSigAlong_[c] = f.sigAlong;
  fSigPerp_[c] = f.sigPerp;
  fLxyBS_[c] = f.lxyBS;
  fSigLxyBS_[c] = f.sigLxyBS;
  fMass_[c] = f.mass;
  fMassRefit_[c] = f.massRefit;
}

void DimuonVertexTruth::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  ++nEvents_;
  const auto& genParticles = iEvent.get(genToken_);

  // Exactly two final signal muons, one of each charge.
  std::array<Gen, 2> gen;
  int nSignal = 0;
  for (const auto& p : genParticles) {
    if (!isSignalMuon(p)) continue;
    ++nSignal;
    gen[p.pdgId() == 13 ? 0 : 1].p = &p;
  }
  if (nSignal != 2 || !gen[0].p || !gen[1].p) return;
  ++nTwoSignal_;

  resetBranches();
  run_ = iEvent.id().run();
  lumi_ = iEvent.id().luminosityBlock();
  event_ = iEvent.id().event();

  const MagneticField* bField = &iSetup.getData(magFieldToken_);
  const PropagateToMuon propagator = propagatorSetup_.init(iSetup);
  const PropagateToMuon propagatorSt1 = propagatorSetupSt1_.init(iSetup);
  const Propagator& along = iSetup.getData(alongToken_);
  const auto& ecalVolume = iSetup.getData(ecalToken_).volume();
  const double ecalMinR = ecalVolume.minR(), ecalMinZ = ecalVolume.minZ();

  for (int g = 0; g < 2; ++g) {
    const reco::GenParticle& p = *gen[g].p;
    gen[g].accepted = std::abs(p.eta()) < 2.4 && p.vertex().rho() >= 0 && p.vertex().rho() < 740 &&
                      std::abs(p.vz()) < 960;
    const FreeTrajectoryState fts(GlobalPoint(p.vx(), p.vy(), p.vz()), GlobalVector(p.px(), p.py(), p.pz()),
                                  TrackCharge(p.charge()), bField);
    gen[g].st2 = toPos(propagator.extrapolate(fts));
    gen[g].st1 = toPos(propagatorSt1.extrapolate(fts));
    gen[g].ecal = toEcal(fts, along, ecalMinR, ecalMinZ);
    gPt_[g] = p.pt();
    gEta_[g] = p.eta();
    gPhi_[g] = p.phi();
    gCharge_[g] = p.charge();
    gVx_[g] = p.vx();
    gVy_[g] = p.vy();
    gVz_[g] = p.vz();
    gD0_[g] = std::abs(p.vx() * std::sin(p.phi()) - p.vy() * std::cos(p.phi()));
    gAccepted_[g] = gen[g].accepted;
    gSt2Valid_[g] = gen[g].st2.valid;
    gSt2Eta_[g] = gen[g].st2.eta;
    gSt2Phi_[g] = gen[g].st2.phi;
    gSt1Valid_[g] = gen[g].st1.valid;
    gSt1Eta_[g] = gen[g].st1.eta;
    gSt1Phi_[g] = gen[g].st1.phi;
    gEcalStatus_[g] = gen[g].ecal.status;
    gEcalEta_[g] = gen[g].ecal.eta;
    gEcalPhi_[g] = gen[g].ecal.phi;
  }
  if (gen[0].accepted && gen[1].accepted) {
    ++nBothAccepted_;
    if (gen[0].st2.valid && gen[1].st2.valid) ++nBothAcceptedPropagated_;
  }

  // Gen decay vertex (mu- production point), gen PV (chi2 production point).
  const reco::GenParticle& m0 = *gen[0].p;
  const reco::GenParticle& m1 = *gen[1].p;
  genVtx_[0] = m0.vx();
  genVtx_[1] = m0.vy();
  genVtx_[2] = m0.vz();
  genVtxSep_ = std::hypot(std::hypot(m0.vx() - m1.vx(), m0.vy() - m1.vy()), m0.vz() - m1.vz());
  genLxy_ = m0.vertex().rho();
  if (const reco::GenParticle* chi2 = firstDifferentMother(m0)) {
    genPV_[0] = chi2->vx();
    genPV_[1] = chi2->vy();
    genPV_[2] = chi2->vz();
    genLxyPV_ = std::hypot(m0.vx() - chi2->vx(), m0.vy() - chi2->vy());
  }
  const auto dimuon = m0.p4() + m1.p4();
  genDimuonPt_ = dimuon.pt();
  genDimuonEta_ = dimuon.eta();
  genDimuonMass_ = dimuon.mass();
  genDimuonDR_ = reco::deltaR(m0, m1);
  // Transverse flight direction: decay - gen PV; dimuon pT direction if undefined.
  double ux = 0, uy = 0;
  if (genLxyPV_ > 0) {
    ux = (m0.vx() - genPV_[0]) / genLxyPV_;
    uy = (m0.vy() - genPV_[1]) / genLxyPV_;
  } else if (dimuon.pt() > 0) {
    ux = dimuon.px() / dimuon.pt();
    uy = dimuon.py() / dimuon.pt();
  }
  flightDirT_[0] = ux;
  flightDirT_[1] = uy;

  // Reco fallback positions are needed only if an accepted gen muon failed station 2.
  const bool needFallback = (gen[0].accepted && !gen[0].st2.valid) || (gen[1].accepted && !gen[1].st2.valid);

  const auto& muons = iEvent.get(muonToken_);
  // Interim duplicate removal for displacedMuons sharing an inner track (see header).
  std::vector<bool> dropped(muons.size(), false);
  {
    std::map<RefKey, std::vector<size_t>> users;
    for (size_t i = 0; i < muons.size(); ++i)
      if (muons[i].innerTrack().isNonnull()) users[refKey(muons[i].innerTrack())].push_back(i);
    for (const auto& [key, idx] : users) {
      if (idx.size() < 2) continue;
      int drop = -1;
      if (idx.size() == 2) {
        const reco::Track& tk = *muons[idx[0]].innerTrack();
        if (tk.extra().isAvailable()) {
          // crossedIP = TrackDetectorAssociator::crossedIP(track), as in SharedInnerTrackDuplicatePolicy.cc
          const auto ip = tk.innerPosition(), op = tk.outerPosition();
          const auto im = tk.innerMomentum(), om = tk.outerMomentum();
          const double rpIn = ip.x() * im.x() + ip.y() * im.y(), rpOut = op.x() * om.x() + op.y() * om.y();
          const bool crossedIP = ip.rho() > 3 && op.rho() > 3 && ((rpIn < 0) != (rpOut < 0));
          const bool g0 = muons[idx[0]].globalTrack().isNonnull(), g1 = muons[idx[1]].globalTrack().isNonnull();
          if (crossedIP)
            drop = static_cast<int>(idx[1]);  // indices are ascending: keep the lower one
          else if (g0 != g1)
            drop = static_cast<int>(g0 ? idx[1] : idx[0]);  // keep the one with a global track
        }
      }
      if (drop < 0) {
        ++nDupUnresolved_;
        continue;
      }
      dropped[drop] = true;
      ++nDupDropped_;
    }
  }

  // Reco units, unique per track.
  std::array<std::vector<Unit>, kNTypes> units;
  for (size_t iMu = 0; iMu < muons.size(); ++iMu) {
    if (dropped[iMu]) continue;
    const auto& mu = muons[iMu];
    const std::array<reco::TrackRef, kNTypes> refs{{mu.innerTrack(), mu.globalTrack(), mu.outerTrack()}};
    for (int t = 0; t < kNTypes; ++t) {
      const reco::TrackRef& ref = refs[t];
      if (ref.isNull()) continue;
      auto it = std::find_if(units[t].begin(), units[t].end(), [&](const Unit& u) { return u.ref == ref; });
      if (it == units[t].end()) {
        Unit u;
        u.ref = ref;
        u.charge = ref->charge();
        u.st2 = toPos(propagator.extrapolate(*ref));
        if (needFallback)
          u.fallback = t == 0 ? toEcal(trajectoryStateTransform::initialFreeState(*ref, bField), along, ecalMinR,
                                     ecalMinZ)
                              : toPos(propagatorSt1.extrapolate(*ref));
        units[t].push_back(u);
        it = units[t].end() - 1;
      }
      it->nParents += 1;
      it->parentTracker |= mu.isTrackerMuon();
      it->parentGlobal |= mu.isGlobalMuon();
      it->constituents.insert(refKey(ref));
      if (t == 1)
        for (const auto& r : refs)
          if (r.isNonnull()) it->constituents.insert(refKey(r));
      if (t == 2 && refs[0].isNonnull()) it->constituents.insert(refKey(refs[0]));
    }
  }

  std::array<std::array<int, 2>, kNTypes> match;
  for (int t = 0; t < kNTypes; ++t) {
    const size_t n = units[t].size();
    nUnits_[t] = n;
    std::vector<const Pos*> unitSt2(n), unitFallback(n);
    for (size_t u = 0; u < n; ++u) {
      unitSt2[u] = &units[t][u].st2;
      unitFallback[u] = &units[t][u].fallback;
    }
    // Pass 1: station 2.
    std::array<float, 2> dr, drFallback;
    match[t] = greedyMatch(gen, {{&gen[0].st2, &gen[1].st2}}, {{gen[0].accepted, gen[1].accepted}}, units[t],
                           unitSt2, std::vector<bool>(n, true), drThreshold_[t], dr);
    // Pass 2: gen muons that failed station 2, tracks not taken in pass 1.
    std::vector<bool> unused(n, true);
    for (int g = 0; g < 2; ++g)
      if (match[t][g] >= 0) unused[match[t][g]] = false;
    const Pos* gf0 = t == 0 ? &gen[0].ecal : &gen[0].st1;
    const Pos* gf1 = t == 0 ? &gen[1].ecal : &gen[1].st1;
    const auto fallback = greedyMatch(gen, {{gf0, gf1}},
                                      {{gen[0].accepted && !gen[0].st2.valid, gen[1].accepted && !gen[1].st2.valid}},
                                      units[t], unitFallback, unused, drThreshold_[t], drFallback);
    for (int g = 0; g < 2; ++g) {
      const int i = 3 * g + t;
      if (match[t][g] >= 0) {
        mSurface_[i] = kStation2;
      } else if (fallback[g] >= 0) {
        match[t][g] = fallback[g];
        dr[g] = drFallback[g];
        mSurface_[i] = t == 0 ? kEcalFallback : kStation1Fallback;
      } else {
        continue;
      }
      const Unit& u = units[t][match[t][g]];
      const Pos& genSurf = mSurface_[i] == kStation2 ? gen[g].st2 : *(g == 0 ? gf0 : gf1);
      const Pos& recoSurf = mSurface_[i] == kStation2 ? u.st2 : u.fallback;
      mGenSurfEta_[i] = genSurf.eta;
      mGenSurfPhi_[i] = genSurf.phi;
      mRecoSurfEta_[i] = recoSurf.eta;
      mRecoSurfPhi_[i] = recoSurf.phi;
      mMatched_[i] = 1;
      mDR_[i] = dr[g];
      mPt_[i] = u.ref->pt();
      mEta_[i] = u.ref->eta();
      mPhi_[i] = u.ref->phi();
      mCharge_[i] = u.charge;
      mNParents_[i] = u.nParents;
      mIdPass_[i] = t == 0 ? u.parentTracker : t == 1 ? u.parentGlobal : passesDisplacedID(*u.ref);
    }
  }

  // Vertex fits for the 9 ordered combinations.
  const TransientTrackBuilder& ttb = iSetup.getData(ttbToken_);
  const reco::BeamSpot& bs = iEvent.get(beamspotToken_);
  KalmanVertexFitter kvf(true);
  for (int a = 0; a < kNTypes; ++a) {
    for (int b = 0; b < kNTypes; ++b) {
      if (match[a][0] < 0 || match[b][1] < 0) continue;
      setFit(3 * a + b, fitPair(units[a][match[a][0]], units[b][match[b][1]], ttb, kvf, bs, genVtx_, ux, uy));
    }
  }

  // Object-level hierarchy (see header).
  std::vector<Unit> cands;
  for (size_t iMu = 0; iMu < muons.size(); ++iMu) {
    if (dropped[iMu]) continue;
    const auto& mu = muons[iMu];
    const std::array<reco::TrackRef, kNTypes> refs{{mu.innerTrack(), mu.globalTrack(), mu.outerTrack()}};
    int t = -1;
    if (mu.isGlobalMuon() && refs[1].isNonnull())
      t = 1;
    else if (mu.isTrackerMuon() && refs[0].isNonnull())
      t = 0;
    else if (refs[2].isNonnull() && passesDisplacedID(*refs[2]))
      t = 2;
    if (t < 0) continue;
    const reco::TrackRef& ref = refs[t];
    auto it = std::find_if(cands.begin(), cands.end(), [&](const Unit& u) { return u.ref == ref; });
    if (it == cands.end()) {
      Unit u;
      u.ref = ref;
      u.type = t;
      u.charge = ref->charge();
      u.st2 = toPos(propagator.extrapolate(*ref));
      if (needFallback) u.fallback = toPos(propagatorSt1.extrapolate(*ref));
      u.firstParent = static_cast<int>(iMu);
      cands.push_back(u);
      ++hierNCandType_[t];
      it = cands.end() - 1;
    }
    it->nParents += 1;
    it->constituents.insert(refKey(ref));
    if (t == 1)
      for (const auto& r : refs)
        if (r.isNonnull()) it->constituents.insert(refKey(r));
    if (t == 2 && refs[0].isNonnull()) it->constituents.insert(refKey(refs[0]));
    if (refs[2].isNonnull()) {
      it->ownOuters.insert(refKey(refs[2]));
      it->ownOuterId = std::max(it->ownOuterId, static_cast<int>(passesDisplacedID(*refs[2])));
    }
  }
  hierNCand_ = cands.size();
  std::array<float, 2> hdr;
  std::array<int, 2> hsurf;
  std::vector<double> hthr;
  for (const auto& c : cands) hthr.push_back(drThreshold_[c.type]);
  const auto hmatch = matchSt2St1(gen, cands, hthr, hdr, hsurf);

  // DSA-ID'd raw standalone tracks, matched to gen for the merging check.
  const auto dsaHandle = iEvent.getHandle(dsaToken_);
  std::vector<Unit> dsa;
  for (size_t i = 0; i < dsaHandle->size(); ++i) {
    ++nDSA_;
    const reco::TrackRef ref(dsaHandle, i);
    if (!passesDisplacedID(*ref)) continue;
    ++nDSAId_;
    Unit u;
    u.ref = ref;
    u.type = 2;
    u.charge = ref->charge();
    u.st2 = toPos(propagator.extrapolate(*ref));
    if (needFallback) u.fallback = toPos(propagatorSt1.extrapolate(*ref));
    dsa.push_back(u);
  }
  // Owners of each outer track among the kept displacedMuons: count, OR of isTracker / isGlobal / has inner,
  // lowest owner index.
  std::map<RefKey, std::array<int, 5>> owners;
  for (size_t iMu = 0; iMu < muons.size(); ++iMu) {
    const auto& mu = muons[iMu];
    if (mu.outerTrack().isNull()) continue;
    ++nMuOuter_;
    if (mu.outerTrack().id() == dsaHandle.id()) ++nMuOuterInDSA_;
    if (dropped[iMu]) continue;
    auto& o = owners.try_emplace(refKey(mu.outerTrack()), std::array<int, 5>{{0, 0, 0, 0, static_cast<int>(iMu)}})
                  .first->second;
    o[0] += 1;
    o[1] |= mu.isTrackerMuon();
    o[2] |= mu.isGlobalMuon();
    o[3] |= mu.innerTrack().isNonnull();
  }
  std::array<float, 2> ddr;
  std::array<int, 2> dsurf;
  const auto dmatch = matchSt2St1(gen, dsa, std::vector<double>(dsa.size(), drThreshold_[2]), ddr, dsurf);

  for (int g = 0; g < 2; ++g) {
    if (hmatch[g] < 0) continue;
    const Unit& c = cands[hmatch[g]];
    hMatched_[g] = 1;
    hType_[g] = c.type;
    hSurface_[g] = hsurf[g];
    hNParents_[g] = c.nParents;
    hCharge_[g] = c.charge;
    hDR_[g] = hdr[g];
    hPt_[g] = c.ref->pt();
    hEta_[g] = c.ref->eta();
    hPhi_[g] = c.ref->phi();
    hOwnOuterId_[g] = c.ownOuterId;
    hNMatchedStations_[g] = muons[c.firstParent].numberOfMatchedStations();
    const Pos& other = gen[1 - g].st2;
    if (c.st2.valid && other.valid) hDROtherGen_[g] = reco::deltaR(c.st2.eta, c.st2.phi, other.eta, other.phi);
    if (dmatch[g] < 0) {
      hOuterCheck_[g] = 1;
      continue;
    }
    const Unit& d = dsa[dmatch[g]];
    const RefKey dk = refKey(d.ref);
    if (c.ownOuters.count(dk))
      hOuterCheck_[g] = 2;
    else
      hOuterCheck_[g] = c.ownOuters.empty() ? 4 : 3;
    const reco::Track& dt = *d.ref;
    dSurface_[g] = dsurf[g];
    dDR_[g] = ddr[g];
    dPt_[g] = dt.pt();
    dEta_[g] = dt.eta();
    dPhi_[g] = dt.phi();
    dPtErr_[g] = dt.ptError();
    dNormChi2_[g] = dt.normalizedChi2();
    dNDT_[g] = dt.hitPattern().numberOfValidMuonDTHits();
    dNCSC_[g] = dt.hitPattern().numberOfValidMuonCSCHits();
    dNStations_[g] = dt.hitPattern().muonStationsWithValidHits();
    if (const auto o = owners.find(dk); o != owners.end()) {
      dOwnerN_[g] = o->second[0];
      dOwnerTracker_[g] = o->second[1];
      dOwnerGlobal_[g] = o->second[2];
      dOwnerInner_[g] = o->second[3];
      // Lowest-index owner: its inner track against the representative, and its own candidate.
      const reco::Muon& ow = muons[o->second[4]];
      if (ow.innerTrack().isNonnull()) {
        dOwnerInnerSame_[g] = c.constituents.count(refKey(ow.innerTrack())) > 0;
        dOwnerInnerDRRep_[g] = reco::deltaR(*ow.innerTrack(), *c.ref);
        dOwnerInnerPtRatio_[g] = ow.innerTrack()->pt() / c.ref->pt();
      }
      const reco::TrackRef owRep = ow.isGlobalMuon() && ow.globalTrack().isNonnull()   ? ow.globalTrack()
                                   : ow.isTrackerMuon() && ow.innerTrack().isNonnull() ? ow.innerTrack()
                                   : passesDisplacedID(*ow.outerTrack())               ? ow.outerTrack()
                                                                                       : reco::TrackRef();
      if (owRep.isNonnull())
        for (const auto& oc : cands)
          if (oc.ref == owRep) {
            dOwnerCandType_[g] = oc.type;
            if (oc.st2.valid && gen[g].st2.valid)
              dOwnerCandDR_[g] = reco::deltaR(oc.st2.eta, oc.st2.phi, gen[g].st2.eta, gen[g].st2.phi);
          }
    }
    if (hmatch[1 - g] >= 0) {
      const Unit& oc = cands[hmatch[1 - g]];
      dOtherRep_[g] = oc.ownOuters.count(dk) || oc.constituents.count(dk);
    }
    if (d.st2.valid && other.valid) dDROtherGen_[g] = reco::deltaR(d.st2.eta, d.st2.phi, other.eta, other.phi);
    if (c.st2.valid && d.st2.valid) hDsaDRSt2_[g] = reco::deltaR(c.st2.eta, c.st2.phi, d.st2.eta, d.st2.phi);
  }
  if (hmatch[0] >= 0 && hmatch[1] >= 0)
    hFit_ = fitPair(cands[hmatch[0]], cands[hmatch[1]], ttb, kvf, bs, genVtx_, ux, uy);

  // PF muons (see header). Units carry charge and positions only; kinematics kept per candidate.
  const auto& pfMuons = iEvent.get(pfMuonToken_);
  std::vector<Unit> pfCands;
  std::vector<const pat::Muon*> pfMu;
  std::vector<const reco::Track*> pfTrk;
  for (const auto& mu : pfMuons) {
    if (!mu.isPFMuon() || !(mu.isGlobalMuon() || mu.isTrackerMuon())) continue;
    int t = -1;
    if (mu.isGlobalMuon() && mu.globalTrack().isNonnull())
      t = 1;
    else if (mu.isTrackerMuon() && mu.innerTrack().isNonnull())
      t = 0;
    if (t < 0) {
      ++pfNoTrack_;
      continue;
    }
    const reco::Track& tk = t == 1 ? *mu.globalTrack() : *mu.innerTrack();
    Unit u;
    u.type = t;
    u.charge = tk.charge();
    u.st2 = toPos(propagator.extrapolate(tk));
    if (needFallback) u.fallback = toPos(propagatorSt1.extrapolate(tk));
    pfCands.push_back(u);
    pfMu.push_back(&mu);
    pfTrk.push_back(&tk);
    ++pfNCandType_[t];
  }
  pfNCand_ = pfCands.size();
  std::vector<double> pthr;
  for (const auto& c : pfCands) pthr.push_back(drThreshold_[c.type]);
  std::array<float, 2> pdr;
  std::array<int, 2> psurf;
  const auto pmatch = matchSt2St1(gen, pfCands, pthr, pdr, psurf);
  for (int g = 0; g < 2; ++g) {
    if (pmatch[g] < 0) continue;
    pfMatched_[g] = 1;
    pfType_[g] = pfCands[pmatch[g]].type;
    pfSurface_[g] = psurf[g];
    pfCharge_[g] = pfCands[pmatch[g]].charge;
    pfDR_[g] = pdr[g];
    pfPt_[g] = pfTrk[pmatch[g]]->pt();
    pfEta_[g] = pfTrk[pmatch[g]]->eta();
    pfPhi_[g] = pfTrk[pmatch[g]]->phi();
    pfMuPt_[g] = pfMu[pmatch[g]]->pt();
    pfMuPhi_[g] = pfMu[pmatch[g]]->phi();
  }
  const auto& mets = iEvent.get(metToken_);
  if (!mets.empty()) {
    pfMetPt_ = mets[0].corPt(pat::MET::Type1);
    pfMetPhi_ = mets[0].corPhi(pat::MET::Type1);
  }
  tree_->Fill();
}

void DimuonVertexTruth::endJob() { totals_->Fill(); }

DEFINE_FWK_MODULE(DimuonVertexTruth);
