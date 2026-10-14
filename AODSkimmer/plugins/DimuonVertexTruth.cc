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
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
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
  Pos fallback;  // inner: ECAL; global/outer: station 1
  bool parentTracker = false;
  bool parentGlobal = false;
  int nParents = 0;
  // Tracks this leg is built from: the track itself; for a global track also
  // its parent muons' inner and outer tracks. Legs sharing any are not fitted.
  std::set<RefKey> constituents;
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

}  // namespace

class DimuonVertexTruth : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit DimuonVertexTruth(const edm::ParameterSet&);
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

private:
  void resetBranches();

  const edm::EDGetTokenT<std::vector<reco::Muon>> muonToken_;
  const edm::EDGetTokenT<std::vector<reco::GenParticle>> genToken_;
  const edm::EDGetTokenT<reco::BeamSpot> beamspotToken_;
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
};

DimuonVertexTruth::DimuonVertexTruth(const edm::ParameterSet& ps)
    : muonToken_(consumes<std::vector<reco::Muon>>(ps.getParameter<edm::InputTag>("muons"))),
      genToken_(consumes<std::vector<reco::GenParticle>>(ps.getParameter<edm::InputTag>("genParticles"))),
      beamspotToken_(consumes<reco::BeamSpot>(ps.getParameter<edm::InputTag>("beamspot"))),
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
                   "fitStatus 0 not both matched, 1 legs share a track, global legs incl. parent inner/outer (not fitted), 2 fit failed, 3 valid; "
                   "match: station 2 position, same charge, greedy 1-to-1, dR<0.06/0.06/0.10; "
                   "fallback: 2nd greedy pass for gen muons failing station 2, unused tracks, same charge and dR, "
                   "inner at ECAL inner boundary (reco from reference state), global/outer at station 1; "
                   "matchSurface 0 none, 1 station 2, 2 station-1 fallback, 3 ECAL fallback; "
                   "genEcalStatus 0 not run, 1 ok, 2 start beyond boundary, 3 failed; "
                   "residual = reco - gen decay vertex, along/perp = gen transverse flight direction (decay - gen PV); "
                   "idPass: inner parent isTracker, global parent isGlobal, outer DSA ID (OR over parents)");
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
  for (auto* a : {mMatched_, mIdPass_, mNParents_, mCharge_, mSurface_}) fill(a, 2 * kNTypes, 0);
  for (auto* a : {mDR_, mPt_, mEta_, mPhi_, mGenSurfEta_, mGenSurfPhi_, mRecoSurfEta_, mRecoSurfPhi_})
    fill(a, 2 * kNTypes, -999.f);
  fill(fStatus_, kNCombos, static_cast<int>(kNotMatched));
  for (auto* a : {fVx_, fVy_, fVz_, fCxx_, fCyy_, fCzz_, fCxy_, fCxz_, fCyz_, fChi2_, fNdof_, fProb_, fResAlong_,
                  fResPerp_, fResZ_, fSigAlong_, fSigPerp_, fLxyBS_, fSigLxyBS_, fMass_, fMassRefit_})
    fill(a, kNCombos, -999.f);
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

  // Reco units, unique per track.
  const auto& muons = iEvent.get(muonToken_);
  std::array<std::vector<Unit>, kNTypes> units;
  for (const auto& mu : muons) {
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
      const int c = 3 * a + b;
      if (match[a][0] < 0 || match[b][1] < 0) continue;
      const Unit& u0 = units[a][match[a][0]];
      const Unit& u1 = units[b][match[b][1]];
      bool shared = false;
      for (const auto& k : u0.constituents)
        if (u1.constituents.count(k)) shared = true;
      if (shared) {
        fStatus_[c] = kSharedTrack;
        continue;
      }
      std::vector<reco::TransientTrack> tts{ttb.build(u0.ref), ttb.build(u1.ref)};
      const TransientVertex tv = kvf.vertex(tts);
      if (!tv.isValid()) {
        fStatus_[c] = kFitFailed;
        continue;
      }
      fStatus_[c] = kFitValid;
      const reco::Vertex v(tv);
      fVx_[c] = v.x();
      fVy_[c] = v.y();
      fVz_[c] = v.z();
      fCxx_[c] = v.covariance(0, 0);
      fCyy_[c] = v.covariance(1, 1);
      fCzz_[c] = v.covariance(2, 2);
      fCxy_[c] = v.covariance(0, 1);
      fCxz_[c] = v.covariance(0, 2);
      fCyz_[c] = v.covariance(1, 2);
      fChi2_[c] = v.chi2();
      fNdof_[c] = v.ndof();
      fProb_[c] = TMath::Prob(v.chi2(), static_cast<int>(std::lround(v.ndof())));
      const double dx = v.x() - genVtx_[0], dy = v.y() - genVtx_[1];
      fResAlong_[c] = dx * ux + dy * uy;
      fResPerp_[c] = -dx * uy + dy * ux;
      fResZ_[c] = v.z() - genVtx_[2];
      const double cxx = v.covariance(0, 0), cyy = v.covariance(1, 1), cxy = v.covariance(0, 1);
      fSigAlong_[c] = std::sqrt(std::max(0., ux * ux * cxx + uy * uy * cyy + 2 * ux * uy * cxy));
      fSigPerp_[c] = std::sqrt(std::max(0., uy * uy * cxx + ux * ux * cyy - 2 * ux * uy * cxy));
      const double bx = v.x() - bs.x(v.z()), by = v.y() - bs.y(v.z());
      const double lxy = std::hypot(bx, by);
      fLxyBS_[c] = lxy;
      if (lxy > 0)
        fSigLxyBS_[c] = std::sqrt(std::max(0., bx * bx * cxx + by * by * cyy + 2 * bx * by * cxy)) / lxy;
      auto p4 = [](const reco::Track& t) {
        return math::PtEtaPhiMLorentzVector(t.pt(), t.eta(), t.phi(), kMuonMass);
      };
      fMass_[c] = (p4(*u0.ref) + p4(*u1.ref)).mass();
      if (tv.hasRefittedTracks() && tv.refittedTracks().size() == 2) {
        const reco::Track r0 = tv.refittedTracks()[0].track();
        const reco::Track r1 = tv.refittedTracks()[1].track();
        fMassRefit_[c] = (p4(r0) + p4(r1)).mass();
      }
    }
  }
  tree_->Fill();
}

void DimuonVertexTruth::endJob() { totals_->Fill(); }

DEFINE_FWK_MODULE(DimuonVertexTruth);
