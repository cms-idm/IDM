// For displacedMuons that share one innerTrack (same ProductID + key):
//  1. match the shared track to status-1 gen muons (prunedGenParticles, |pdgId| == 13, no ancestry
//     requirement) at the ECAL inner boundary;
//  2. propagate the matched gen muon forward from its production vertex to every chamber where each
//     copy has hits, and test the copy's chambers / segments / RPC hits against that prediction;
//  3. record the existing forward score (check_shared_track_direction.py) for comparison.
//
// ECAL boundary: the ECAL fiducial volume of the EcalDetIdAssociator, the same volume
// TrackDetectorAssociator uses for getStateAtEcal(): barrel cylinder rho = minR (accepted if |z| <= minZ)
// and endcap planes |z| = minZ (accepted if rho <= minR). The first along-momentum crossing is used.
//   reco: start from the track's outermost state (TrackExtra required; no substitute if absent)
//   gen:  start from the production vertex
// Along-momentum only: a track fitted with reversed direction will not match here.
//
// Tolerances are the ones MuonIdProducer / TrackDetectorAssociator used to build the associations
// (passed from the cfg). The gen state has no covariance, so only the absolute-distance criteria apply.
// No direction cut exists in MuonIdProducer: direction differences are written but never used.
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrajectoryState/interface/TrackCharge.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CSCGeometry/interface/CSCChamber.h"
#include "Geometry/CSCGeometry/interface/CSCChamberSpecs.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCLayerGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCWireTopology.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMChamber.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMSuperChamber.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/DetIdAssociatorRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrackAssociator/interface/DetIdAssociator.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

namespace {
const double kNaN = std::numeric_limits<double>::quiet_NaN();

// ---------- ECAL inner boundary ----------
struct EcalPoint {
  std::string status = "not_run";  // ok | start_beyond_boundary | propagation_failure | no_track_extra
  double eta = kNaN, phi = kNaN;
};

EcalPoint toEcal(const FreeTrajectoryState& start, const Propagator& along, double minR, double minZ) {
  EcalPoint out;
  if (start.position().perp() >= minR || std::abs(start.position().z()) >= minZ) {
    out.status = "start_beyond_boundary";
    return out;
  }
  double bestPath = std::numeric_limits<double>::max();
  auto consider = [&](const std::pair<TrajectoryStateOnSurface, double>& r, bool barrel) {
    if (!r.first.isValid()) return;
    const GlobalPoint p = r.first.globalPosition();
    if (barrel ? std::abs(p.z()) > minZ : p.perp() > minR) return;
    if (r.second < bestPath) {
      bestPath = r.second;
      out.status = "ok";
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
  if (out.status != "ok") out.status = "propagation_failure";
  return out;
}

// ---------- forward score: same as check_shared_track_direction.py ----------
double sectorPhi(int sector) {
  double phi = (sector - 1) / 6. * M_PI;
  if (sector == 13) phi = 3 / 6. * M_PI;
  if (sector == 14) phi = 9 / 6. * M_PI;
  return phi > M_PI ? phi - 2 * M_PI : phi;
}

int hitsInChamber(const reco::MuonChamberMatch& m) {
  return m.segmentMatches.size() + m.rpcMatches.size() + m.gemMatches.size() + m.gemHitMatches.size();
}

// z side (+1/-1, 0 = barrel) and barrel sector phi; false for detectors the score ignores.
bool chamberSide(const reco::MuonChamberMatch& m, int& zSide, double& secPhi) {
  const uint32_t raw = m.id.rawId();
  secPhi = kNaN;
  switch (m.detector()) {
    case MuonSubdetId::DT:
      zSide = 0;
      secPhi = sectorPhi(DTChamberId(raw).sector());
      return true;
    case MuonSubdetId::CSC: {
      const CSCDetId c(raw);
      zSide = c.endcap() == 1 ? 1 : -1;
      return true;
    }
    case MuonSubdetId::RPC: {
      const RPCDetId r(raw);
      zSide = r.region();
      if (zSide == 0) secPhi = sectorPhi(r.sector());
      return true;
    }
    case MuonSubdetId::GEM:
      zSide = GEMDetId(raw).region();
      return true;
    default:
      return false;
  }
}

int forwardScore(const reco::Muon& mu) {
  const auto& tk = *mu.innerTrack();
  const bool extra = tk.extra().isAvailable();
  const double phiExitMom = extra ? tk.outerMomentum().phi() : tk.phi();
  const int pzSign = tk.pz() >= 0 ? 1 : -1;
  int score = 0;
  for (const auto& m : mu.matches()) {
    int zSide = 0;
    double secPhi = kNaN;
    if (!chamberSide(m, zSide, secPhi) || hitsInChamber(m) == 0) continue;
    if (zSide != 0) score += zSide == pzSign ? 1 : -1;
    else score += std::abs(reco::deltaPhi(secPhi, phiExitMom)) < M_PI / 2 ? 1 : -1;
  }
  return score;
}

// ---------- chamber bounds: same distance definition as TrackDetectorAssociator ----------
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

// ---------- per-copy verdicts ----------
struct Tally {
  int compatible = 0, incompatible = 0, failure = 0;
  std::string verdict() const {
    if (failure) return "untestable_propagation_failure";
    if (compatible + incompatible == 0) return "untestable_no_items";
    if (incompatible == 0) return "consistent";
    if (compatible == 0) return "inconsistent";
    return "mixed";
  }
};

std::string conclusion(const std::string& a, const std::string& b) {
  if (a.rfind("untestable", 0) == 0 || b.rfind("untestable", 0) == 0) return "untestable";
  const bool ca = a == "consistent", cb = b == "consistent";
  if (ca && cb) return "both";
  if (ca) return "A_only";
  if (cb) return "B_only";
  return "neither";
}

const reco::GenParticle* firstDifferentMother(const reco::GenParticle& p) {
  const reco::Candidate* cur = &p;
  for (int depth = 0; depth < 100 && cur && cur->numberOfMothers(); ++depth) {
    const auto* mom = cur->mother(0);
    if (!mom) return nullptr;
    if (mom->pdgId() != p.pdgId()) return dynamic_cast<const reco::GenParticle*>(mom);
    cur = mom;
  }
  return nullptr;
}

std::string num(double v) {
  if (!std::isfinite(v)) return "";
  std::ostringstream s;
  s.precision(6);
  s << v;
  return s.str();
}
}  // namespace

class SharedTrackTrajectoryCheck : public edm::one::EDAnalyzer<> {
public:
  explicit SharedTrackTrajectoryCheck(const edm::ParameterSet& cfg)
      : muonsToken_(consumes<std::vector<reco::Muon>>(cfg.getParameter<edm::InputTag>("muons"))),
        genToken_(consumes<std::vector<reco::GenParticle>>(cfg.getParameter<edm::InputTag>("genParticles"))),
        fieldToken_(esConsumes()),
        alongToken_(esConsumes(edm::ESInputTag("", "SteppingHelixPropagatorAlong"))),
        oppositeToken_(esConsumes(edm::ESInputTag("", "SteppingHelixPropagatorOpposite"))),
        ecalToken_(esConsumes(edm::ESInputTag("", "EcalDetIdAssociator"))),
        dtToken_(esConsumes()),
        cscToken_(esConsumes()),
        rpcToken_(esConsumes()),
        gemToken_(esConsumes()),
        thresholds_(cfg.getParameter<std::vector<double>>("drThresholds")),
        maxAbsDx_(cfg.getParameter<double>("maxAbsDx")),
        maxAbsDy_(cfg.getParameter<double>("maxAbsDy")),
        muonMaxDistanceX_(cfg.getParameter<double>("muonMaxDistanceX")),
        muonMaxDistanceY_(cfg.getParameter<double>("muonMaxDistanceY")),
        rpcMaxAbsDx_(cfg.getParameter<double>("rpcMaxAbsDx")),
        tracksFile_(cfg.getParameter<std::string>("tracksFile")),
        itemsFile_(cfg.getParameter<std::string>("itemsFile")) {}

private:
  void beginJob() override {
    tracks_.open(tracksFile_);
    items_.open(itemsFile_);
    if (!tracks_ || !items_) throw cms::Exception("FileOpenError") << tracksFile_ << " / " << itemsFile_;
    tracks_ << "run,lumi,event,track_product,track_key,n_users,user_indices,trk_charge,trk_pt,trk_eta,trk_phi,"
               "has_extra,first_hit_rho,first_hit_z,first_hit_rdotp,last_hit_rdotp,crossedIP,"
               "reco_ecal_status,reco_ecal_eta,reco_ecal_phi,"
               "n_gen_eligible,n_gen_ecal_ok,n_gen_ecal_start_beyond,n_gen_ecal_failure,";
    for (const double t : thresholds_) tracks_ << "n_gen_dR_lt_" << num(t) << ",";
    tracks_ << "gen_index,gen_pdgId,gen_charge,gen_pt,gen_eta,gen_phi,gen_vxy,gen_vz,gen_first_other_mother,"
               "gen_dR_ecal,charge_relation,"
               "A_index,B_index,A_global,B_global,A_score,B_score,raw_preference,tiebreak_preference,"
               "identical_hit_chambers,A_chamber_verdict,B_chamber_verdict,chamber_conclusion,"
               "A_hit_verdict,B_hit_verdict,hit_conclusion,A_gem_hits_not_tested,B_gem_hits_not_tested\n";
    items_ << "run,lumi,event,track_product,track_key,copy,muon_index,detector,station,z_side,chamber_raw_id,item,"
              "gen_propagation,pred_x,pred_y,meas_x,meas_y,abs_dx,abs_dy,edge_dist_x,edge_dist_y,"
              "d_dxdz,d_dydz,result\n";
  }

  // Gen muon propagated forward to a chamber plane. "behind" = the plane is reached only opposite to
  // the momentum from the production vertex; that state is used for classification only, never compared.
  struct ChamberPrediction {
    std::string status = "failure";  // forward | behind | failure
    LocalPoint pos;
    double dxdz = kNaN, dydz = kNaN;
    double edgeX = kNaN, edgeY = kNaN;
  };

  ChamberPrediction predict(const FreeTrajectoryState& gen, const GeomDet* det, const Propagator& along,
                            const Propagator& opposite) const {
    ChamberPrediction out;
    if (!det) return out;
    const auto fwd = along.propagateWithPath(gen, det->surface());
    if (fwd.first.isValid()) {
      out.status = "forward";
      out.pos = fwd.first.localPosition();
      const LocalVector dir = fwd.first.localDirection();
      if (dir.z() != 0) {
        out.dxdz = dir.x() / dir.z();
        out.dydz = dir.y() / dir.z();
      }
      double dx = 0, dy = 0;
      distanceToChamber(det, out.pos, dx, dy);
      out.edgeX = dx;
      out.edgeY = dy;
      return out;
    }
    if (opposite.propagateWithPath(gen, det->surface()).first.isValid()) out.status = "behind";
    return out;
  }

  const GeomDet* chamberDet(const DetId& id, const edm::EventSetup& setup) const {
    switch (id.subdetId()) {
      case MuonSubdetId::DT: return setup.getData(dtToken_).idToDet(id);
      case MuonSubdetId::CSC: return setup.getData(cscToken_).idToDet(id);
      case MuonSubdetId::RPC: return setup.getData(rpcToken_).idToDet(id);
      case MuonSubdetId::GEM: return setup.getData(gemToken_).idToDet(id);
      default: return nullptr;
    }
  }

  // Tests one copy; writes item rows; returns chamber-level and hit-level tallies.
  void testCopy(const std::string& prefix, const char* copy, size_t index, const reco::Muon& mu,
                const FreeTrajectoryState& gen, const edm::EventSetup& setup, const Propagator& along,
                const Propagator& opposite, Tally& chambers, Tally& hits, int& gemHitsNotTested) {
    for (const auto& m : mu.matches()) {
      int zSide = 0;
      double secPhi = kNaN;
      if (!chamberSide(m, zSide, secPhi) || hitsInChamber(m) == 0) continue;
      const auto pred = predict(gen, chamberDet(m.id, setup), along, opposite);
      const bool fwd = pred.status == "forward";
      const std::string head = prefix + copy + "," + std::to_string(index) + "," + std::to_string(m.detector()) +
                               "," + std::to_string(m.station()) + "," + std::to_string(zSide) + "," +
                               std::to_string(m.id.rawId()) + ",";
      auto result = [&](bool pass) {
        if (pred.status == "failure") return std::string("untestable");
        return std::string(fwd && pass ? "compatible" : "incompatible");
      };
      auto count = [](Tally& t, const std::string& r) {
        if (r == "compatible") ++t.compatible;
        else if (r == "incompatible") ++t.incompatible;
        else ++t.failure;
      };
      // chamber bounds
      const std::string cr = result(fwd && pred.edgeX < muonMaxDistanceX_ && pred.edgeY < muonMaxDistanceY_);
      count(chambers, cr);
      items_ << head << "chamber," << pred.status << "," << (fwd ? num(pred.pos.x()) : "") << ","
             << (fwd ? num(pred.pos.y()) : "") << ",,,,," << num(pred.edgeX) << "," << num(pred.edgeY) << ",,,"
             << cr << "\n";
      // DT/CSC segments and GEM segments: MuonIdProducer criterion with zero gen error
      auto segments = [&](const std::vector<reco::MuonSegmentMatch>& segs, const char* name) {
        for (const auto& s : segs) {
          const double adx = fwd ? std::abs(s.x - pred.pos.x()) : kNaN;
          const double ady = fwd ? std::abs(s.y - pred.pos.y()) : kNaN;
          const std::string r = result(fwd && adx < maxAbsDx_ && ady < maxAbsDy_);
          count(hits, r);
          items_ << head << name << "," << pred.status << "," << (fwd ? num(pred.pos.x()) : "") << ","
                 << (fwd ? num(pred.pos.y()) : "") << "," << num(s.x) << "," << num(s.y) << "," << num(adx) << ","
                 << num(ady) << ",,," << (fwd ? num(s.dXdZ - pred.dxdz) : "") << ","
                 << (fwd ? num(s.dYdZ - pred.dydz) : "") << "," << r << "\n";
        }
      };
      segments(m.segmentMatches, "segment");
      segments(m.gemMatches, "gem_segment");
      // RPC hits: MuonIdProducer accepts |dx| <= 20 cm (hard-coded there)
      for (const auto& h : m.rpcMatches) {
        const double adx = fwd ? std::abs(h.x - pred.pos.x()) : kNaN;
        const std::string r = result(fwd && adx <= rpcMaxAbsDx_);
        count(hits, r);
        items_ << head << "rpc_hit," << pred.status << "," << (fwd ? num(pred.pos.x()) : "") << ",," << num(h.x)
               << ",," << num(adx) << ",,,,,," << r << "\n";
      }
      // GEM rechit matches: the producer's eta-partition border cut needs data not stored in the match.
      for (size_t k = 0; k < m.gemHitMatches.size(); ++k) {
        ++gemHitsNotTested;
        items_ << head << "gem_hit," << pred.status << ",,,,,,,,,,,not_tested\n";
      }
    }
  }

  void analyze(const edm::Event& event, const edm::EventSetup& setup) override {
    const auto& muons = event.get(muonsToken_);
    std::map<std::pair<unsigned int, unsigned int>, std::vector<size_t>> groups;
    for (size_t i = 0; i < muons.size(); ++i) {
      const auto ref = muons[i].innerTrack();
      if (ref.isNull()) continue;
      if (!ref.isAvailable()) {
        ++unavailableInner_;
        continue;
      }
      groups[{ref.id().id(), static_cast<unsigned int>(ref.key())}].push_back(i);
    }
    std::vector<std::pair<std::pair<unsigned int, unsigned int>, std::vector<size_t>>> shared;
    for (const auto& g : groups)
      if (g.second.size() >= 2) shared.push_back(g);
    if (shared.empty()) return;

    const MagneticField* field = &setup.getData(fieldToken_);
    const auto& along = setup.getData(alongToken_);
    const auto& opposite = setup.getData(oppositeToken_);
    const auto& ecalVolume = setup.getData(ecalToken_).volume();
    const double minR = ecalVolume.minR(), minZ = ecalVolume.minZ();
    if (!loggedBoundary_) {
      std::cout << "ECAL inner boundary (EcalDetIdAssociator volume): minR=" << minR << " cm, minZ=" << minZ
                << " cm\n";
      loggedBoundary_ = true;
    }

    // eligible gen muons: prunedGenParticles, status 1, |pdgId| 13
    const auto& gen = event.get(genToken_);
    struct GenMuon {
      size_t index;
      FreeTrajectoryState fts;
      EcalPoint ecal;
    };
    std::vector<GenMuon> genMuons;
    int nStartBeyond = 0, nFailure = 0;
    for (size_t k = 0; k < gen.size(); ++k) {
      const auto& p = gen[k];
      if (p.status() != 1 || std::abs(p.pdgId()) != 13) continue;
      const FreeTrajectoryState fts(GlobalPoint(p.vx(), p.vy(), p.vz()), GlobalVector(p.px(), p.py(), p.pz()),
                                    TrackCharge(p.charge()), field);
      const auto ecal = toEcal(fts, along, minR, minZ);
      if (ecal.status == "start_beyond_boundary") ++nStartBeyond;
      if (ecal.status == "propagation_failure") ++nFailure;
      genMuons.push_back({k, fts, ecal});
    }
    const int nGenOk = static_cast<int>(genMuons.size()) - nStartBeyond - nFailure;

    const double maxThreshold = *std::max_element(thresholds_.begin(), thresholds_.end());
    for (const auto& [ref, users] : shared) {
      const reco::Muon& first = muons[users[0]];
      const reco::Track& tk = *first.innerTrack();
      std::ostringstream row;
      row << event.id().run() << "," << event.luminosityBlock() << "," << event.id().event() << "," << ref.first
          << "," << ref.second << "," << users.size() << ",";
      for (size_t u = 0; u < users.size(); ++u) row << (u ? ";" : "") << users[u];
      row << "," << tk.charge() << "," << num(tk.pt()) << "," << num(tk.eta()) << "," << num(tk.phi()) << ",";

      // existing first-hit / endpoint (crossedIP) inputs, only when TrackExtra is there
      const bool extra = tk.extra().isAvailable();
      row << extra << ",";
      if (extra) {
        const auto ip = tk.innerPosition(), op = tk.outerPosition();
        const auto im = tk.innerMomentum(), om = tk.outerMomentum();
        const double rpIn = ip.x() * im.x() + ip.y() * im.y(), rpOut = op.x() * om.x() + op.y() * om.y();
        const bool crossed = ip.rho() > 3 && op.rho() > 3 && ((rpIn < 0) != (rpOut < 0));
        row << num(ip.rho()) << "," << num(ip.z()) << "," << num(rpIn) << "," << num(rpOut) << "," << crossed << ",";
      } else {
        row << ",,,,,";
      }

      // reco ECAL point: outermost state, along momentum
      EcalPoint recoEcal;
      recoEcal.status = "no_track_extra";
      if (extra) recoEcal = toEcal(trajectoryStateTransform::outerFreeState(tk, field, false), along, minR, minZ);
      row << recoEcal.status << "," << num(recoEcal.eta) << "," << num(recoEcal.phi) << "," << genMuons.size()
          << "," << nGenOk << "," << nStartBeyond << "," << nFailure << ",";

      // gen candidates at the ECAL boundary (no charge requirement)
      const GenMuon* nearest = nullptr;
      double nearestDR = kNaN;
      std::vector<int> within(thresholds_.size(), 0);
      if (recoEcal.status == "ok") {
        for (const auto& g : genMuons) {
          if (g.ecal.status != "ok") continue;
          const double dr = reco::deltaR(recoEcal.eta, recoEcal.phi, g.ecal.eta, g.ecal.phi);
          for (size_t t = 0; t < thresholds_.size(); ++t)
            if (dr < thresholds_[t]) ++within[t];
          if (!nearest || dr < nearestDR) {
            nearest = &g;
            nearestDR = dr;
          }
        }
      }
      for (const int n : within) row << n << ",";
      const bool matched = nearest && nearestDR < maxThreshold;
      if (matched) {
        const auto& p = gen[nearest->index];
        const auto* mom = firstDifferentMother(p);
        row << nearest->index << "," << p.pdgId() << "," << p.charge() << "," << num(p.pt()) << "," << num(p.eta())
            << "," << num(p.phi()) << "," << num(p.vertex().rho()) << "," << num(p.vz()) << ","
            << (mom ? std::to_string(mom->pdgId()) : "") << "," << num(nearestDR) << ","
            << (p.charge() == tk.charge() ? "same" : "opposite") << ",";
      } else {
        row << ",,,,,,,,," << num(nearestDR) << ",,";
      }

      // A/B comparison only for two-user tracks with a gen muon within the largest threshold
      if (users.size() == 2 && matched) {
        const size_t ia = users[0], ib = users[1];
        const reco::Muon& a = muons[ia];
        const reco::Muon& b = muons[ib];
        const int sa = forwardScore(a), sb = forwardScore(b);
        const std::string raw = sa > sb ? "A" : sb > sa ? "B" : "tie";
        std::string tiebreak = raw;
        if (raw == "tie") {
          if (a.isGlobalMuon() != b.isGlobalMuon()) tiebreak = a.isGlobalMuon() ? "A" : "B";
          else tiebreak = "A";  // earlier index (bookkeeping only)
        }
        std::set<uint32_t> chA, chB;
        for (const auto& m : a.matches())
          if (hitsInChamber(m)) chA.insert(m.id.rawId());
        for (const auto& m : b.matches())
          if (hitsInChamber(m)) chB.insert(m.id.rawId());

        std::ostringstream prefix;
        prefix << event.id().run() << "," << event.luminosityBlock() << "," << event.id().event() << ","
               << ref.first << "," << ref.second << ",";
        Tally chamA, hitA, chamB, hitB;
        int gemA = 0, gemB = 0;
        testCopy(prefix.str(), "A", ia, a, nearest->fts, setup, along, opposite, chamA, hitA, gemA);
        testCopy(prefix.str(), "B", ib, b, nearest->fts, setup, along, opposite, chamB, hitB, gemB);
        row << ia << "," << ib << "," << a.isGlobalMuon() << "," << b.isGlobalMuon() << "," << sa << "," << sb
            << "," << raw << "," << tiebreak << "," << (chA == chB) << "," << chamA.verdict() << ","
            << chamB.verdict() << "," << conclusion(chamA.verdict(), chamB.verdict()) << "," << hitA.verdict()
            << "," << hitB.verdict() << "," << conclusion(hitA.verdict(), hitB.verdict()) << "," << gemA << ","
            << gemB << "\n";
      } else {
        row << ",,,,,,,,,,,,,,,,\n";
      }
      tracks_ << row.str();
    }
  }

  void endJob() override {
    tracks_.close();
    items_.close();
    std::cout << "SharedTrackTrajectoryCheck: displacedMuons with an innerTrack ref that is not available: "
              << unavailableInner_ << "\nTolerances used: maxAbsDx=" << maxAbsDx_ << " maxAbsDy=" << maxAbsDy_
              << " muonMaxDistanceX=" << muonMaxDistanceX_ << " muonMaxDistanceY=" << muonMaxDistanceY_
              << " rpcMaxAbsDx=" << rpcMaxAbsDx_ << "\nWrote " << tracksFile_ << " and " << itemsFile_ << "\n";
  }

  const edm::EDGetTokenT<std::vector<reco::Muon>> muonsToken_;
  const edm::EDGetTokenT<std::vector<reco::GenParticle>> genToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> fieldToken_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> alongToken_, oppositeToken_;
  const edm::ESGetToken<DetIdAssociator, DetIdAssociatorRecord> ecalToken_;
  const edm::ESGetToken<DTGeometry, MuonGeometryRecord> dtToken_;
  const edm::ESGetToken<CSCGeometry, MuonGeometryRecord> cscToken_;
  const edm::ESGetToken<RPCGeometry, MuonGeometryRecord> rpcToken_;
  const edm::ESGetToken<GEMGeometry, MuonGeometryRecord> gemToken_;
  const std::vector<double> thresholds_;
  const double maxAbsDx_, maxAbsDy_, muonMaxDistanceX_, muonMaxDistanceY_, rpcMaxAbsDx_;
  const std::string tracksFile_, itemsFile_;
  std::ofstream tracks_, items_;
  unsigned long long unavailableInner_ = 0;
  bool loggedBoundary_ = false;
};

DEFINE_FWK_MODULE(SharedTrackTrajectoryCheck);
