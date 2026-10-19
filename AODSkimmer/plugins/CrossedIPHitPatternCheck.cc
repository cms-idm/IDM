// CrossedIPHitPatternCheck: compares the exact crossedIP of a displacedMuon inner track (TrackExtra) with a
// proxy from the track's HitPattern alone, which is also stored in data AOD (design:
// kb/studies/shared-track-duplicates.md, "HitPattern crossedIP for data AOD").
//
// One tree entry per distinct inner track (ProductID + key) used by at least one displacedMuon; users are
// grouped exactly as in DimuonVertexTruth.cc. Policy decisions are derived offline from nUsers and the
// users' global flags.
//
// exact (needs TrackExtra, else -1): ip.rho() > 3 && op.rho() > 3 && radial momentum sign differs between
//   inner and outer state, as in SharedInnerTrackDuplicatePolicy.cc / DimuonVertexTruth.cc.
// proxy (-1 = undefined): valid tracker hits in HitPattern order (fit order, along the momentum). Hit
//   distance d: barrel (PXB, TIB, TOB) = mean module radius of the layer; endcap (PXF, TID, TEC) = mean
//   module |z| of the disk times pt/|pz|. Layer means from TrackerGeometry detUnits, keyed by
//   (subdetId, TrackerTopology::layer), the key HitPattern encodes. proxy = 1 iff d(first) > 3, d(last) > 3
//   and some hit has a strictly larger d both before and after it (interior minimum).
//   Undefined (undefReason): 1 fewer than 3 valid tracker hits, 2 HitPattern full (MaxHits), 3 endcap hit
//   with pz = 0 or non-finite d, 4 (subdet, layer) not in the geometry map.
// proxyFix = proxy OR rule E (one fix, design in the same study note): first valid hit on an endcap disk
//   (mean |z| Z1) with sign(pz) * vz > Z1 (PCA to the beamline beyond that disk along the travel: the first
//   hit precedes the PCA on either z side); the last hit follows the PCA (endcap: Z_L > sign(pz) * vz;
//   barrel: assumed); rho lower bound at the first hit sqrt(dxy^2 + ((sign(pz) * vz - Z1) * pt / |pz|)^2) > 3
//   and d(last) > 3. Evaluated only when the proxy is defined.
// Muon-system side test (two-user tracks only; design "Muon-system side test vs crossedIP" in the same note):
//   s = (X - r0) . p_hat, r0 = shared track reference point (PCA to the beamline), p_hat its unit momentum.
//   Per copy (A = lower index, B) and source: M = every matches() chamber with >= 1 segment/RPC/GEM match,
//   X = chamber surface centre (GlobalTrackingGeometry); Arb = DT/CSC chambers with a segment passing
//   BestInChamberByDR && BelongsToTrackByDR (SegmentAndTrackArbitration); S = standalone innerPosition and
//   outerPosition (outerTrack with available extra). Counts: ahead (s > 0), behind (s < 0), plane (s == 0),
//   dropped (no geometry or non-finite s). Classes are formed offline.
// Direction-consistency selection inputs (two-user tracks only; design "Direction-consistency selection" in
//   the same note). Per copy, items = chambers: source M (as above); if M gives no chamber with geometry,
//   fallback = the chamber of the global track's outerDetId (TrackExtra; the MC MINIAOD has no global rechits,
//   so its outermost hit only). Per item, T: s = (centre - r0) . p_hat as above; P: from the shared track's
//   state at r0 (initialFreeState, no TrackExtra), SteppingHelixPropagatorAlong and ...Opposite each to the
//   chamber plane, each attempt in (valid, inside the bounds within muonMaxDistanceX/Y, distanceToChamber as
//   SharedTrackTrajectoryCheck.cc), out (valid, outside) or invalid; ahead = along in and opposite not in,
//   behind = opposite in and along not in, else bothIn / neitherIn. Only the first plane crossing is tested.
//   Verdicts and the pair decision are formed offline.
#include <algorithm>
#include <cmath>
#include <map>
#include <utility>
#include <vector>

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CSCGeometry/interface/CSCChamber.h"
#include "Geometry/CSCGeometry/interface/CSCChamberSpecs.h"
#include "Geometry/CSCGeometry/interface/CSCLayerGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCWireTopology.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMChamber.h"
#include "Geometry/GEMGeometry/interface/GEMSuperChamber.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TTree.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

namespace {
using RefKey = std::pair<edm::ProductID, unsigned int>;  // as in DimuonVertexTruth.cc
RefKey refKey(const reco::TrackRef& r) { return {r.id(), static_cast<unsigned int>(r.key())}; }
bool isBarrel(int subdet) { return subdet == 1 || subdet == 3 || subdet == 5; }

// Muon detector id -> the chamber id used by matches() (DT/CSC chamber, RPC roll, GEM chamber); 0 otherwise.
uint32_t muonChamberId(DetId id) {
  if (id.det() != DetId::Muon) return 0;
  switch (id.subdetId()) {
    case MuonSubdetId::DT:
      return DTChamberId(id.rawId()).rawId();
    case MuonSubdetId::CSC:
      return CSCDetId(id.rawId()).chamberId().rawId();
    case MuonSubdetId::RPC:
      return id.rawId();
    case MuonSubdetId::GEM:
      return GEMDetId(id.rawId()).chamberId().rawId();
    default:
      return 0;
  }
}

// Chamber bounds: copied from SharedTrackTrajectoryCheck.cc (TrackDetectorAssociator distance definition).
void distanceToChamber(const GeomDet* det, const LocalPoint& lp, double& dx, double& dy) {
  if (const auto* csc = dynamic_cast<const CSCChamber*>(det)) {
    const auto* wire = csc->specs()->oddLayerGeometry(1)->wireTopology();
    const float wide = wire->wideWidthOfPlane(), narrow = wire->narrowWidthOfPlane(), length = wire->lengthOfPlane();
    const float yFirst = std::abs(wire->wireAngle()) > 1.E-06f ? -0.5 * length : wire->yOfWire(1);
    const float yOffset = yFirst + 0.5f * length;
    const float tangent = (wide - narrow) / (2.f * length);
    const float halfWidth = 0.5f * narrow + (lp.y() + std::abs(yFirst)) * tangent;
    dx = std::abs(lp.x()) - halfWidth;
    dy = std::abs(lp.y() - yOffset) - 0.5f * length;
  } else if (dynamic_cast<const GEMChamber*>(det) || dynamic_cast<const GEMSuperChamber*>(det)) {
    const auto* b = dynamic_cast<const TrapezoidalPlaneBounds*>(&det->surface().bounds());
    const float wide = b->width(), narrow = 2.f * b->widthAtHalfLength() - wide, length = b->length();
    const float tangent = (wide - narrow) / (2.f * length);
    dx = std::abs(lp.x()) - (tangent * lp.y() + 0.25f * (narrow + wide));
    dy = std::abs(lp.y()) - 0.5f * length;
  } else {
    dx = std::abs(lp.x()) - 0.5f * det->surface().bounds().width();
    dy = std::abs(lp.y()) - 0.5f * det->surface().bounds().length();
  }
}
}  // namespace

class CrossedIPHitPatternCheck : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit CrossedIPHitPatternCheck(const edm::ParameterSet& cfg)
      : muonToken_(consumes<std::vector<reco::Muon>>(cfg.getParameter<edm::InputTag>("muons"))),
        geomToken_(esConsumes()),
        topoToken_(esConsumes()),
        globalGeomToken_(esConsumes()),
        fieldToken_(esConsumes()),
        alongToken_(esConsumes(edm::ESInputTag("", "SteppingHelixPropagatorAlong"))),
        oppositeToken_(esConsumes(edm::ESInputTag("", "SteppingHelixPropagatorOpposite"))),
        maxDistX_(cfg.getParameter<double>("muonMaxDistanceX")),
        maxDistY_(cfg.getParameter<double>("muonMaxDistanceY")) {
    usesResource(TFileService::kSharedResource);
  }

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void buildLayerMap(const edm::EventSetup&);

  edm::EDGetTokenT<std::vector<reco::Muon>> muonToken_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  edm::ESGetToken<GlobalTrackingGeometry, GlobalTrackingGeometryRecord> globalGeomToken_;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> fieldToken_;
  edm::ESGetToken<Propagator, TrackingComponentsRecord> alongToken_, oppositeToken_;
  const double maxDistX_, maxDistY_;
  std::map<std::pair<int, int>, double> layerPos_;  // (subdet, layer) -> mean radius (barrel) or |z| (endcap)

  TTree* tree_ = nullptr;
  unsigned int run_, lumi_;
  unsigned long long event_;
  int nUsers_, idxA_, idxB_, globalA_, globalB_;
  int hasExtra_, exact_, proxy_, undefReason_, seqClass_, nValid_, nTrackHits_, saturated_, interiorMin_;
  int firstType_, lastType_, firstValidPXB1_, lastValidPXB1_;
  int firstEndcap_, lastEndcap_, ruleE_, proxyFix_;
  float pt_, eta_, pz_, innerRho_, outerRho_, innerZ_, outerZ_, dFirst_, dLast_;
  float vz_, dxy_, dzError_, marginFirst_, marginLast_, rhoFirstBound_;
  std::vector<int> hitCode_;
  std::vector<float> hitD_;
  // side test: [copy A/B][source M/Arb/S]
  int sideN_[2][3][4];  // ahead, behind, plane, dropped
  float sideS_[2][3][3];  // s min, s max, min |s|
  int identicalM_;        // copies' source-M chamber sets identical (-1: not a two-user track)
  // direction-consistency inputs: [copy A/B]
  int dcSrc_[2];  // 0 source M, 1 global-outer fallback, -1 no item (-1 for all below if not two-user)
  int dcFb_[2];   // fallback: 0 not needed, 1 used, 2 no global track, 3 no global extra, 4 no muon chamber/geometry
  int dcT_[2][4];  // T items: ahead, behind, plane, dropped
  int dcP_[2][6];  // P items: ahead, behind, bothIn, neitherIn; attempts invalid: along, opposite
  int dcAgree_[2];  // items with T and P both ahead or both behind
};

void CrossedIPHitPatternCheck::beginJob() {
  edm::Service<TFileService> fs;
  tree_ = fs->make<TTree>("tracks", "displacedMuon inner tracks: exact vs HitPattern crossedIP");
  tree_->Branch("run", &run_);
  tree_->Branch("lumi", &lumi_);
  tree_->Branch("event", &event_);
  tree_->Branch("nUsers", &nUsers_);  // displacedMuons with this inner track
  tree_->Branch("idxA", &idxA_);      // lowest user index
  tree_->Branch("idxB", &idxB_);      // second user index (-1 if one user)
  tree_->Branch("globalA", &globalA_);
  tree_->Branch("globalB", &globalB_);  // -1 if one user
  tree_->Branch("pt", &pt_);
  tree_->Branch("eta", &eta_);
  tree_->Branch("pz", &pz_);
  tree_->Branch("hasExtra", &hasExtra_);
  tree_->Branch("exact", &exact_);  // -1 no TrackExtra
  tree_->Branch("innerRho", &innerRho_);
  tree_->Branch("outerRho", &outerRho_);
  tree_->Branch("innerZ", &innerZ_);
  tree_->Branch("outerZ", &outerZ_);
  tree_->Branch("proxy", &proxy_);  // -1 undefined
  tree_->Branch("undefReason", &undefReason_);
  tree_->Branch("interiorMin", &interiorMin_);  // interior minimum, before the 3 cm endpoint cut
  tree_->Branch("seqClass", &seqClass_);        // valid hits: 0 barrel only, 1 endcap only, 2 mixed
  tree_->Branch("nValid", &nValid_);
  tree_->Branch("nTrackHits", &nTrackHits_);  // TRACK_HITS entries, any type
  tree_->Branch("saturated", &saturated_);
  tree_->Branch("firstType", &firstType_);  // hit type of the first / last TRACK_HITS entry (0 = valid)
  tree_->Branch("lastType", &lastType_);
  tree_->Branch("firstValidPXB1", &firstValidPXB1_);
  tree_->Branch("lastValidPXB1", &lastValidPXB1_);
  tree_->Branch("dFirst", &dFirst_);
  tree_->Branch("dLast", &dLast_);
  tree_->Branch("vz", &vz_);  // reference point = PCA to the beamline
  tree_->Branch("dxy", &dxy_);  // about (0,0)
  tree_->Branch("dzError", &dzError_);
  tree_->Branch("firstEndcap", &firstEndcap_);  // first / last valid hit on an endcap disk
  tree_->Branch("lastEndcap", &lastEndcap_);
  tree_->Branch("marginFirst", &marginFirst_);  // sign(pz) * vz - Z1 (first hit endcap, else NaN)
  tree_->Branch("marginLast", &marginLast_);    // Z_L - sign(pz) * vz (last hit endcap, else NaN)
  tree_->Branch("rhoFirstBound", &rhoFirstBound_);
  tree_->Branch("ruleE", &ruleE_);
  tree_->Branch("proxyFix", &proxyFix_);  // proxy OR ruleE; -1 undefined
  const char* copies[2] = {"A", "B"};
  const char* sources[3] = {"M", "Arb", "S"};
  const char* counts[4] = {"nAhead", "nBehind", "nPlane", "nDropped"};
  const char* svals[3] = {"sMin", "sMax", "absSMin"};
  for (int c = 0; c < 2; ++c)
    for (int k = 0; k < 3; ++k) {
      for (int i = 0; i < 4; ++i)
        tree_->Branch((std::string("side") + sources[k] + "_" + copies[c] + "_" + counts[i]).c_str(), &sideN_[c][k][i]);
      for (int i = 0; i < 3; ++i)
        tree_->Branch((std::string("side") + sources[k] + "_" + copies[c] + "_" + svals[i]).c_str(), &sideS_[c][k][i]);
    }
  tree_->Branch("sideIdenticalM", &identicalM_);
  const char* tCounts[4] = {"nAhead", "nBehind", "nPlane", "nDropped"};
  const char* pCounts[6] = {"nAhead", "nBehind", "nBothIn", "nNeitherIn", "nInvalidAlong", "nInvalidOpposite"};
  for (int c = 0; c < 2; ++c) {
    tree_->Branch((std::string("dc_") + copies[c] + "_src").c_str(), &dcSrc_[c]);
    tree_->Branch((std::string("dc_") + copies[c] + "_fb").c_str(), &dcFb_[c]);
    tree_->Branch((std::string("dc_") + copies[c] + "_nTPAgree").c_str(), &dcAgree_[c]);
    for (int i = 0; i < 4; ++i) tree_->Branch((std::string("dcT_") + copies[c] + "_" + tCounts[i]).c_str(), &dcT_[c][i]);
    for (int i = 0; i < 6; ++i) tree_->Branch((std::string("dcP_") + copies[c] + "_" + pCounts[i]).c_str(), &dcP_[c][i]);
  }
  tree_->Branch("hitCode", &hitCode_);  // valid tracker hits in order: 10 * subdet + layer
  tree_->Branch("hitD", &hitD_);
}

void CrossedIPHitPatternCheck::buildLayerMap(const edm::EventSetup& iSetup) {
  const auto& geom = iSetup.getData(geomToken_);
  const auto& topo = iSetup.getData(topoToken_);
  std::map<std::pair<int, int>, std::pair<double, int>> sums;
  for (const auto* det : geom.detUnits()) {
    const DetId id = det->geographicalId();
    const int subdet = id.subdetId();
    const auto pos = det->surface().position();
    auto& s = sums[{subdet, static_cast<int>(topo.layer(id))}];
    s.first += isBarrel(subdet) ? pos.perp() : std::abs(pos.z());
    ++s.second;
  }
  for (const auto& [key, s] : sums) layerPos_[key] = s.first / s.second;
}

void CrossedIPHitPatternCheck::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  if (layerPos_.empty()) buildLayerMap(iSetup);
  const auto& muons = iEvent.get(muonToken_);
  std::map<RefKey, std::vector<size_t>> users;
  for (size_t i = 0; i < muons.size(); ++i)
    if (muons[i].innerTrack().isNonnull()) users[refKey(muons[i].innerTrack())].push_back(i);

  run_ = iEvent.id().run();
  lumi_ = iEvent.id().luminosityBlock();
  event_ = iEvent.id().event();
  for (const auto& [key, idx] : users) {
    const reco::Track& tk = *muons[idx[0]].innerTrack();
    nUsers_ = idx.size();
    idxA_ = idx[0];
    idxB_ = idx.size() > 1 ? static_cast<int>(idx[1]) : -1;
    globalA_ = muons[idx[0]].globalTrack().isNonnull();
    globalB_ = idx.size() > 1 ? static_cast<int>(muons[idx[1]].globalTrack().isNonnull()) : -1;
    pt_ = tk.pt();
    eta_ = tk.eta();
    pz_ = tk.pz();

    // exact
    hasExtra_ = tk.extra().isAvailable();
    exact_ = -1;
    innerRho_ = outerRho_ = innerZ_ = outerZ_ = std::nanf("");
    if (hasExtra_) {
      const auto ip = tk.innerPosition(), op = tk.outerPosition();
      const auto im = tk.innerMomentum(), om = tk.outerMomentum();
      const double rpIn = ip.x() * im.x() + ip.y() * im.y(), rpOut = op.x() * om.x() + op.y() * om.y();
      exact_ = ip.rho() > 3 && op.rho() > 3 && ((rpIn < 0) != (rpOut < 0));
      innerRho_ = ip.rho();
      outerRho_ = op.rho();
      innerZ_ = ip.z();
      outerZ_ = op.z();
    }

    // proxy
    const reco::HitPattern& hp = tk.hitPattern();
    nTrackHits_ = hp.numberOfAllHits(reco::HitPattern::TRACK_HITS);
    saturated_ = nTrackHits_ + hp.numberOfAllHits(reco::HitPattern::MISSING_INNER_HITS) +
                     hp.numberOfAllHits(reco::HitPattern::MISSING_OUTER_HITS) >=
                 reco::HitPattern::MaxHits;
    firstType_ = nTrackHits_ ? reco::HitPattern::getHitType(hp.getHitPattern(reco::HitPattern::TRACK_HITS, 0)) : -1;
    lastType_ = nTrackHits_ ? reco::HitPattern::getHitType(
                                  hp.getHitPattern(reco::HitPattern::TRACK_HITS, nTrackHits_ - 1))
                            : -1;
    hitCode_.clear();
    hitD_.clear();
    std::vector<float> hitPos;  // layer mean radius (barrel) or |z| (endcap)
    undefReason_ = 0;
    bool anyBarrel = false, anyEndcap = false;
    const double ptOverPz = tk.pz() != 0 ? tk.pt() / std::abs(tk.pz()) : std::nan("");
    for (int i = 0; i < nTrackHits_; ++i) {
      const uint16_t p = hp.getHitPattern(reco::HitPattern::TRACK_HITS, i);
      if (!reco::HitPattern::trackerHitFilter(p) || !reco::HitPattern::validHitFilter(p)) continue;
      const int subdet = reco::HitPattern::getSubStructure(p), layer = reco::HitPattern::getLayer(p);
      hitCode_.push_back(10 * subdet + layer);
      const auto it = layerPos_.find({subdet, layer});
      double d = std::nan("");
      if (it == layerPos_.end()) {
        if (!undefReason_) undefReason_ = 4;
      } else if (isBarrel(subdet)) {
        d = it->second;
        anyBarrel = true;
      } else {
        d = it->second * ptOverPz;
        anyEndcap = true;
        if (!std::isfinite(d) && !undefReason_) undefReason_ = 3;
      }
      hitD_.push_back(d);
      hitPos.push_back(it == layerPos_.end() ? std::nanf("") : it->second);
    }
    nValid_ = hitD_.size();
    seqClass_ = anyBarrel && anyEndcap ? 2 : anyEndcap ? 1 : 0;
    firstValidPXB1_ = nValid_ && hitCode_.front() == 11;
    lastValidPXB1_ = nValid_ && hitCode_.back() == 11;
    dFirst_ = nValid_ ? hitD_.front() : std::nanf("");
    dLast_ = nValid_ ? hitD_.back() : std::nanf("");
    if (!undefReason_ && nValid_ < 3) undefReason_ = 1;
    if (!undefReason_ && saturated_) undefReason_ = 2;
    interiorMin_ = 0;
    if (nValid_ >= 3 && (undefReason_ == 0 || undefReason_ == 2)) {
      std::vector<float> prefixMax(nValid_), suffixMax(nValid_);
      prefixMax[0] = hitD_[0];
      for (int i = 1; i < nValid_; ++i) prefixMax[i] = std::max(prefixMax[i - 1], hitD_[i]);
      suffixMax[nValid_ - 1] = hitD_[nValid_ - 1];
      for (int i = nValid_ - 2; i >= 0; --i) suffixMax[i] = std::max(suffixMax[i + 1], hitD_[i]);
      for (int j = 1; j + 1 < nValid_ && !interiorMin_; ++j)
        interiorMin_ = prefixMax[j - 1] > hitD_[j] && suffixMax[j + 1] > hitD_[j];
    }
    proxy_ = undefReason_ ? -1 : (dFirst_ > 3 && dLast_ > 3 && interiorMin_);

    // rule E
    vz_ = tk.vz();
    dxy_ = tk.dxy();
    dzError_ = tk.dzError();
    const double sigma = tk.pz() >= 0 ? 1 : -1;
    firstEndcap_ = nValid_ && !isBarrel(hitCode_.front() / 10);
    lastEndcap_ = nValid_ && !isBarrel(hitCode_.back() / 10);
    marginFirst_ = firstEndcap_ ? sigma * vz_ - hitPos.front() : std::nanf("");
    marginLast_ = lastEndcap_ ? hitPos.back() - sigma * vz_ : std::nanf("");
    rhoFirstBound_ = firstEndcap_ ? std::hypot(dxy_, marginFirst_ * ptOverPz) : std::nanf("");
    ruleE_ = !undefReason_ && firstEndcap_ && marginFirst_ > 0 && (!lastEndcap_ || marginLast_ > 0) &&
             rhoFirstBound_ > 3 && dLast_ > 3;
    proxyFix_ = undefReason_ ? -1 : (proxy_ || ruleE_);

    // muon-system side test
    for (auto& c : sideN_)
      for (auto& k : c)
        for (int& n : k) n = -1;
    for (auto& c : sideS_)
      for (auto& k : c)
        for (float& v : k) v = std::nanf("");
    identicalM_ = -1;
    for (int c = 0; c < 2; ++c) {
      dcSrc_[c] = dcFb_[c] = dcAgree_[c] = -1;
      for (int& n : dcT_[c]) n = -1;
      for (int& n : dcP_[c]) n = -1;
    }
    if (idx.size() == 2) {
      const auto& gg = iSetup.getData(globalGeomToken_);
      const auto r0 = tk.referencePoint();
      const auto pHat = tk.momentum().unit();
      std::vector<uint32_t> chambersM[2];
      for (int c = 0; c < 2; ++c) {
        const reco::Muon& mu = muons[idx[c]];
        for (int k = 0; k < 3; ++k) {
          for (int& n : sideN_[c][k]) n = 0;
          sideS_[c][k][0] = sideS_[c][k][1] = sideS_[c][k][2] = std::nanf("");
        }
        auto add = [&](int k, double x, double y, double z, bool valid) {
          const double sv = (x - r0.x()) * pHat.x() + (y - r0.y()) * pHat.y() + (z - r0.z()) * pHat.z();
          if (!valid || !std::isfinite(sv)) {
            ++sideN_[c][k][3];
            return;
          }
          ++sideN_[c][k][sv > 0 ? 0 : sv < 0 ? 1 : 2];
          float& lo = sideS_[c][k][0];
          float& hi = sideS_[c][k][1];
          float& am = sideS_[c][k][2];
          lo = std::isnan(lo) ? sv : std::min<float>(lo, sv);
          hi = std::isnan(hi) ? sv : std::max<float>(hi, sv);
          am = std::isnan(am) ? std::abs(sv) : std::min<float>(am, std::abs(sv));
        };
        for (const auto& m : mu.matches()) {
          const int det = m.detector();
          if (det != MuonSubdetId::DT && det != MuonSubdetId::CSC && det != MuonSubdetId::RPC &&
              det != MuonSubdetId::GEM)
            continue;
          const bool anyHit = !m.segmentMatches.empty() || !m.rpcMatches.empty() || !m.gemMatches.empty() ||
                              !m.gemHitMatches.empty();
          bool arbitrated = false;
          if (det == MuonSubdetId::DT || det == MuonSubdetId::CSC)
            for (const auto& seg : m.segmentMatches)
              arbitrated |= seg.isMask(reco::MuonSegmentMatch::BestInChamberByDR) &&
                            seg.isMask(reco::MuonSegmentMatch::BelongsToTrackByDR);
          if (!anyHit) continue;
          const GeomDet* gd = gg.idToDet(m.id);
          const GlobalPoint x = gd ? gd->position() : GlobalPoint();
          add(0, x.x(), x.y(), x.z(), gd != nullptr);
          if (arbitrated) add(1, x.x(), x.y(), x.z(), gd != nullptr);
          chambersM[c].push_back(m.id.rawId());
        }
        if (mu.outerTrack().isNonnull() && mu.outerTrack()->extra().isAvailable()) {
          const auto& ot = *mu.outerTrack();
          add(2, ot.innerPosition().x(), ot.innerPosition().y(), ot.innerPosition().z(), true);
          add(2, ot.outerPosition().x(), ot.outerPosition().y(), ot.outerPosition().z(), true);
        }
      }
      for (auto& v : chambersM) {
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
      }
      identicalM_ = chambersM[0] == chambersM[1];

      // direction-consistency inputs
      const auto& field = iSetup.getData(fieldToken_);
      const Propagator& along = iSetup.getData(alongToken_);
      const Propagator& opposite = iSetup.getData(oppositeToken_);
      const FreeTrajectoryState fts = trajectoryStateTransform::initialFreeState(tk, &field);
      auto attempt = [&](const Propagator& prop, const GeomDet* det) {  // 0 invalid, 1 out, 2 in
        const auto tsos = prop.propagateWithPath(fts, det->surface()).first;
        if (!tsos.isValid()) return 0;
        double dx = 0, dy = 0;
        distanceToChamber(det, tsos.localPosition(), dx, dy);
        return dx < maxDistX_ && dy < maxDistY_ ? 2 : 1;
      };
      for (int c = 0; c < 2; ++c) {
        const reco::Muon& mu = muons[idx[c]];
        for (int& n : dcT_[c]) n = 0;
        for (int& n : dcP_[c]) n = 0;
        dcFb_[c] = dcAgree_[c] = 0;
        std::vector<const GeomDet*> items;
        int droppedM = 0;
        for (const auto& m : mu.matches()) {
          const int det = m.detector();
          if (det != MuonSubdetId::DT && det != MuonSubdetId::CSC && det != MuonSubdetId::RPC &&
              det != MuonSubdetId::GEM)
            continue;
          if (m.segmentMatches.empty() && m.rpcMatches.empty() && m.gemMatches.empty() && m.gemHitMatches.empty())
            continue;
          if (const GeomDet* gd = gg.idToDet(m.id))
            items.push_back(gd);
          else
            ++droppedM;
        }
        dcSrc_[c] = items.empty() ? -1 : 0;
        dcT_[c][3] = droppedM;
        if (items.empty()) {
          if (mu.globalTrack().isNull())
            dcFb_[c] = 2;
          else if (!mu.globalTrack()->extra().isAvailable())
            dcFb_[c] = 3;
          else {
            const uint32_t ch = muonChamberId(mu.globalTrack()->outerDetId());
            const GeomDet* gd = ch ? gg.idToDet(DetId(ch)) : nullptr;
            if (gd) {
              items.push_back(gd);
              dcFb_[c] = 1;
              dcSrc_[c] = 1;
            } else
              dcFb_[c] = 4;
          }
        }
        for (const GeomDet* gd : items) {
          const GlobalPoint x = gd->position();
          const double sv = (x.x() - r0.x()) * pHat.x() + (x.y() - r0.y()) * pHat.y() + (x.z() - r0.z()) * pHat.z();
          const int t = !std::isfinite(sv) ? 3 : sv > 0 ? 0 : sv < 0 ? 1 : 2;
          ++dcT_[c][t];
          const int a = attempt(along, gd), o = attempt(opposite, gd);
          dcP_[c][4] += a == 0;
          dcP_[c][5] += o == 0;
          const int pv = a == 2 && o != 2 ? 0 : o == 2 && a != 2 ? 1 : a == 2 && o == 2 ? 2 : 3;
          ++dcP_[c][pv];
          dcAgree_[c] += t < 2 && t == pv;
        }
      }
    }
    tree_->Fill();
  }
}

DEFINE_FWK_MODULE(CrossedIPHitPatternCheck);
