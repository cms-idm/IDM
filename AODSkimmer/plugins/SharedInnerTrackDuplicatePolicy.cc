// Duplicate-removal policy for displacedMuons that share one innerTrack (same ProductID + key).
// Counting unit: one distinct shared innerTrack (event + ProductID + key); A = lower displacedMuon index,
// B = higher. Tracks with more than two users are written (per-user components and scores listed) with the
// policy columns blank: they are never reduced to an A/B pair.
//
// Policy written in the CSV (original definition) for a two-user track:
//  1. components: if one copy's non-null {outerTrack, globalTrack} set is a strict superset of the other's,
//     keep it (decision_stage = components);
//  2. otherwise (equal or incomparable components) the forward score of SharedTrackTrajectoryCheck.cc decides
//     (decision_stage = score): higher score kept; equal scores -> final_choice = tie, no further tie-breaker.
// score_preference is recorded for every two-user track, so other rules (e.g. score for crossedIP tracks,
// components otherwise) can be evaluated from the CSV.
//
// crossedIP: TrackDetectorAssociator::crossedIP(track), written out as in SharedTrackTrajectoryCheck.cc
// (needs TrackExtra; blank without it).
//
// Gen matching (second CSV only, diagnostic):
//   gen:  prunedGenParticles, status 1, |pdgId| == 13, no ancestry requirement, propagated along momentum from
//         the production vertex to the ECAL inner boundary (EcalDetIdAssociator volume);
//   reco: inner tracks, from the outermost state (TrackExtra required), along momentum. Every distinct inner
//         track used by any displacedMuon is propagated, shared or not; the copies of a shared track are one
//         reco candidate. displacedMuons without an innerTrack (standalone-only) are not candidates.
//   nearest:   the closest gen muon to the shared track (as in SharedTrackTrajectoryCheck.cc).
//   exclusive: only gen muons whose closest reco inner track (any charge) is this shared track are eligible;
//              a gen muon closer to another displacedMuon's inner track is never matched here.
//   Both are counted for any charge and for same sign (gen charge == innerTrack charge).
//   A track enters the gen-matched CSV when the nearest any-charge gen muon is within the largest threshold
//   (unchanged inclusion; the exclusive columns select a subset).
//
// Forward/behind (two-user tracks, per charge definition, exclusive nearest gen muon within the largest
// threshold): that gen muon is propagated from its vertex to each chamber where a copy has match items
// (the chambers the forward score counts): forward = reached along the momentum, behind = reached only
// opposite to it, failure = neither (SharedTrackTrajectoryCheck.cc predict()). No edge or hit tolerances.
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
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
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
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
using TrackId = std::pair<unsigned int, unsigned int>;  // ProductID index, key

// ---------- ECAL inner boundary: copied unchanged from SharedTrackTrajectoryCheck.cc ----------
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

// ---------- forward score: copied unchanged from SharedTrackTrajectoryCheck.cc ----------
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

// ---------- gen ancestry: copied unchanged from SharedTrackTrajectoryCheck.cc ----------
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

// threshold as written in column names: 0.1 -> "0.10"
std::string thresholdLabel(double t) {
  std::ostringstream s;
  s << std::fixed << std::setprecision(2) << t;
  return s.str();
}

std::string join(const std::vector<std::string>& fields) {
  std::string out;
  for (size_t i = 0; i < fields.size(); ++i) out += (i ? "," : "") + fields[i];
  return out;
}

// track components of a copy (innerTrack is always there: copies are grouped by it)
std::string components(const reco::Muon& mu) {
  return std::string("I") + (mu.outerTrack().isNonnull() ? "+O" : "") + (mu.globalTrack().isNonnull() ? "+G" : "");
}

// ---------- the duplicate policy ----------
struct Decision {
  std::string relation, stage, scorePreference, finalChoice, reason;
};

Decision decide(const reco::Muon& a, const reco::Muon& b, int scoreA, int scoreB) {
  const bool hasOuterA = a.outerTrack().isNonnull(), hasGlobalA = a.globalTrack().isNonnull();
  const bool hasOuterB = b.outerTrack().isNonnull(), hasGlobalB = b.globalTrack().isNonnull();
  const bool aDominatesB = (!hasOuterB || hasOuterA) && (!hasGlobalB || hasGlobalA) &&
                           ((hasOuterA && !hasOuterB) || (hasGlobalA && !hasGlobalB));
  const bool bDominatesA = (!hasOuterA || hasOuterB) && (!hasGlobalA || hasGlobalB) &&
                           ((hasOuterB && !hasOuterA) || (hasGlobalB && !hasGlobalA));
  Decision d;
  d.relation = aDominatesB                                          ? "A_dominates"
               : bDominatesA                                        ? "B_dominates"
               : (hasOuterA == hasOuterB && hasGlobalA == hasGlobalB) ? "equal"
                                                                    : "incomparable";
  d.scorePreference = scoreA > scoreB ? "A" : scoreB > scoreA ? "B" : "tie";
  const std::string comps = "A=" + components(a) + " B=" + components(b);
  const std::string scores = "score A=" + std::to_string(scoreA) + " B=" + std::to_string(scoreB);
  if (aDominatesB || bDominatesA) {
    d.stage = "components";
    d.finalChoice = aDominatesB ? "A" : "B";
    d.reason = d.finalChoice + " components strict superset (" + comps + ")";
  } else {
    d.stage = "score";
    d.finalChoice = d.scorePreference;
    d.reason = d.relation + " components (" + comps + "); " + scores +
               (d.finalChoice == "tie" ? " -> unresolved tie" : " -> " + d.finalChoice + " higher score");
  }
  return d;
}

// ---------- forward/behind of a copy's chambers relative to a gen muon ----------
struct Directions {
  int forward = 0, behind = 0, failure = 0;
};
}  // namespace

class SharedInnerTrackDuplicatePolicy : public edm::one::EDAnalyzer<> {
public:
  explicit SharedInnerTrackDuplicatePolicy(const edm::ParameterSet& cfg)
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
        policyFile_(cfg.getParameter<std::string>("policyFile")),
        genMatchedFile_(cfg.getParameter<std::string>("genMatchedFile")) {
    if (thresholds_.empty()) throw cms::Exception("Configuration") << "drThresholds is empty";
  }

private:
  void beginJob() override {
    policy_.open(policyFile_);
    genMatched_.open(genMatchedFile_);
    if (!policy_ || !genMatched_) throw cms::Exception("FileOpenError") << policyFile_ << " / " << genMatchedFile_;
    const std::string policyHeader =
        "run,lumi,event,track_product,track_key,n_users,user_indices,user_components,user_scores,"
        "trk_charge,trk_pt,trk_eta,trk_phi,has_extra,crossedIP,"
        "A_index,B_index,A_has_outer,B_has_outer,A_has_global,B_has_global,"
        "A_isTrackerMuon,B_isTrackerMuon,A_isStandAloneMuon,B_isStandAloneMuon,A_isGlobalMuon,B_isGlobalMuon,"
        "A_score,B_score,component_relation,decision_stage,score_preference,final_choice,kept_muon_index,"
        "decision_reason";
    const std::vector<std::string> genFields = {"gen_index", "gen_pdgId", "gen_charge", "gen_pt", "gen_eta",
                                                "gen_phi",   "gen_vxy",   "gen_vz",     "first_non_muon_mother",
                                                "dR_ecal"};
    std::vector<std::string> gen = {"reco_ecal_status", "reco_ecal_eta", "reco_ecal_phi", "n_gen_eligible",
                                    "n_gen_ecal_ok", "n_gen_ecal_start_beyond", "n_gen_ecal_failure",
                                    "n_reco_inner_tracks", "n_reco_inner_ecal_not_ok", "n_standalone_only_muons"};
    // nearest (unchanged)
    for (const std::string kind : {"anycharge", "samesign"})
      for (const double t : thresholds_) gen.push_back("n_gen_" + kind + "_dR_lt_" + thresholdLabel(t));
    for (const std::string kind : {"anycharge", "samesign"})
      for (const auto& field : genFields) gen.push_back("nearest_" + kind + "_" + field);
    // owner of the nearest gen muon: closest reco inner track to it
    for (const std::string kind : {"anycharge", "samesign"}) {
      gen.push_back("nearest_" + kind + "_owner");
      gen.push_back("nearest_" + kind + "_owner_dR_ecal");
    }
    // exclusive
    for (const std::string kind : {"anycharge", "samesign"})
      for (const double t : thresholds_) gen.push_back("n_gen_" + kind + "_excl_dR_lt_" + thresholdLabel(t));
    for (const std::string kind : {"anycharge", "samesign"})
      for (const auto& field : genFields) gen.push_back("nearest_" + kind + "_excl_" + field);
    // forward/behind with the exclusive nearest gen muon
    for (const std::string kind : {"anycharge", "samesign"})
      for (const char* field : {"A_forward", "A_behind", "A_failure", "B_forward", "B_behind", "B_failure"})
        gen.push_back(kind + "_excl_" + field);
    policy_ << policyHeader << "\n";
    genMatched_ << policyHeader << "," << join(gen) << "\n";
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

  // Same chambers as the forward score; same propagation as SharedTrackTrajectoryCheck.cc predict().
  Directions directions(const reco::Muon& mu, const FreeTrajectoryState& gen, const edm::EventSetup& setup,
                        const Propagator& along, const Propagator& opposite) const {
    Directions d;
    for (const auto& m : mu.matches()) {
      int zSide = 0;
      double secPhi = kNaN;
      if (!chamberSide(m, zSide, secPhi) || hitsInChamber(m) == 0) continue;
      const GeomDet* det = chamberDet(m.id, setup);
      if (det && along.propagateWithPath(gen, det->surface()).first.isValid()) ++d.forward;
      else if (det && opposite.propagateWithPath(gen, det->surface()).first.isValid()) ++d.behind;
      else ++d.failure;
    }
    return d;
  }

  void analyze(const edm::Event& event, const edm::EventSetup& setup) override {
    const auto& muons = event.get(muonsToken_);
    std::map<TrackId, std::vector<size_t>> groups;
    int nStandaloneOnly = 0;
    for (size_t i = 0; i < muons.size(); ++i) {
      const auto ref = muons[i].innerTrack();
      if (ref.isNull()) {
        ++nStandaloneOnly;
        continue;
      }
      if (!ref.isAvailable()) {
        ++unavailableInner_;
        continue;
      }
      groups[{ref.id().id(), static_cast<unsigned int>(ref.key())}].push_back(i);
    }
    std::vector<std::pair<TrackId, std::vector<size_t>>> shared;
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

    // ECAL point of every distinct inner track used by a displacedMuon (the reco candidates)
    std::map<TrackId, EcalPoint> recoEcal;
    int nRecoNotOk = 0;
    for (const auto& [id, users] : groups) {
      const reco::Track& tk = *muons[users[0]].innerTrack();
      EcalPoint e;
      e.status = "no_track_extra";
      if (tk.extra().isAvailable()) e = toEcal(trajectoryStateTransform::outerFreeState(tk, field, false), along, minR, minZ);
      if (e.status != "ok") ++nRecoNotOk;
      recoEcal[id] = e;
    }

    // eligible gen muons: prunedGenParticles, status 1, |pdgId| 13; each with its closest reco inner track
    const auto& gen = event.get(genToken_);
    struct GenMuon {
      size_t index;
      int charge;
      FreeTrajectoryState fts;
      EcalPoint ecal;
      bool hasOwner = false;
      TrackId owner;
      double ownerDR = kNaN;
    };
    std::vector<GenMuon> genMuons;
    int nStartBeyond = 0, nFailure = 0;
    for (size_t k = 0; k < gen.size(); ++k) {
      const auto& p = gen[k];
      if (p.status() != 1 || std::abs(p.pdgId()) != 13) continue;
      GenMuon g{k,
                p.charge(),
                FreeTrajectoryState(GlobalPoint(p.vx(), p.vy(), p.vz()), GlobalVector(p.px(), p.py(), p.pz()),
                                    TrackCharge(p.charge()), field),
                EcalPoint()};
      g.ecal = toEcal(g.fts, along, minR, minZ);
      if (g.ecal.status == "start_beyond_boundary") ++nStartBeyond;
      if (g.ecal.status == "propagation_failure") ++nFailure;
      if (g.ecal.status == "ok") {
        for (const auto& [id, e] : recoEcal) {
          if (e.status != "ok") continue;
          const double dr = reco::deltaR(e.eta, e.phi, g.ecal.eta, g.ecal.phi);
          if (!g.hasOwner || dr < g.ownerDR) {
            g.hasOwner = true;
            g.owner = id;
            g.ownerDR = dr;
          }
        }
      }
      genMuons.push_back(g);
    }
    const int nGenOk = static_cast<int>(genMuons.size()) - nStartBeyond - nFailure;
    const double maxThreshold = *std::max_element(thresholds_.begin(), thresholds_.end());

    for (const auto& [ref, users] : shared) {
      const reco::Track& tk = *muons[users[0]].innerTrack();
      std::vector<std::string> f = {std::to_string(event.id().run()), std::to_string(event.luminosityBlock()),
                                    std::to_string(event.id().event()), std::to_string(ref.first),
                                    std::to_string(ref.second), std::to_string(users.size())};
      std::string indices, comps, scores;
      for (size_t u = 0; u < users.size(); ++u) {
        const std::string sep = u ? ";" : "";
        const std::string idx = std::to_string(users[u]);
        indices += sep + idx;
        comps += sep + idx + ":" + components(muons[users[u]]);
        scores += sep + idx + ":" + std::to_string(forwardScore(muons[users[u]]));
      }
      f.insert(f.end(), {indices, comps, scores, std::to_string(tk.charge()), num(tk.pt()), num(tk.eta()),
                         num(tk.phi())});

      // crossedIP = TrackDetectorAssociator::crossedIP(track), as in SharedTrackTrajectoryCheck.cc
      const bool extra = tk.extra().isAvailable();
      std::string crossedIP;
      if (extra) {
        const auto ip = tk.innerPosition(), op = tk.outerPosition();
        const auto im = tk.innerMomentum(), om = tk.outerMomentum();
        const double rpIn = ip.x() * im.x() + ip.y() * im.y(), rpOut = op.x() * om.x() + op.y() * om.y();
        crossedIP = std::to_string(ip.rho() > 3 && op.rho() > 3 && ((rpIn < 0) != (rpOut < 0)));
      }
      f.insert(f.end(), {std::to_string(extra), crossedIP});

      // policy: two-user tracks only
      if (users.size() == 2) {
        const size_t ia = users[0], ib = users[1];
        const reco::Muon& a = muons[ia];
        const reco::Muon& b = muons[ib];
        const int sa = forwardScore(a), sb = forwardScore(b);
        const Decision d = decide(a, b, sa, sb);
        const std::string kept = d.finalChoice == "A" ? std::to_string(ia)
                                 : d.finalChoice == "B" ? std::to_string(ib)
                                                        : "";
        auto flag = [](bool x) { return std::to_string(x); };
        f.insert(f.end(), {std::to_string(ia), std::to_string(ib), flag(a.outerTrack().isNonnull()),
                           flag(b.outerTrack().isNonnull()), flag(a.globalTrack().isNonnull()),
                           flag(b.globalTrack().isNonnull()), flag(a.isTrackerMuon()), flag(b.isTrackerMuon()),
                           flag(a.isStandAloneMuon()), flag(b.isStandAloneMuon()), flag(a.isGlobalMuon()),
                           flag(b.isGlobalMuon()), std::to_string(sa), std::to_string(sb), d.relation, d.stage,
                           d.scorePreference, d.finalChoice, kept, d.reason});
      } else {
        f.insert(f.end(), 19, "");
        f.push_back("n_users=" + std::to_string(users.size()) + ": not reduced to an A/B pair");
      }
      const std::string policyRow = join(f);
      policy_ << policyRow << "\n";
      ++nPolicyRows_;

      // gen matching at the ECAL inner boundary
      const EcalPoint& trkEcal = recoEcal.at(ref);
      struct Nearest {
        const GenMuon* g = nullptr;
        double dr = kNaN;
        std::vector<int> within;
      };
      auto scan = [&](bool sameSign, bool exclusive) {
        Nearest n;
        n.within.assign(thresholds_.size(), 0);
        if (trkEcal.status != "ok") return n;
        for (const auto& g : genMuons) {
          if (g.ecal.status != "ok") continue;
          if (sameSign && g.charge != tk.charge()) continue;
          if (exclusive && !(g.hasOwner && g.owner == ref)) continue;
          const double dr = reco::deltaR(trkEcal.eta, trkEcal.phi, g.ecal.eta, g.ecal.phi);
          for (size_t t = 0; t < thresholds_.size(); ++t)
            if (dr < thresholds_[t]) ++n.within[t];
          if (!n.g || dr < n.dr) {
            n.g = &g;
            n.dr = dr;
          }
        }
        return n;
      };
      const Nearest any = scan(false, false), same = scan(true, false);
      const Nearest anyExcl = scan(false, true), sameExcl = scan(true, true);
      if (!(any.g && any.dr < maxThreshold)) continue;  // gen-matched CSV: nearest any-charge within 0.10

      std::vector<std::string> g = {trkEcal.status,
                                    num(trkEcal.eta),
                                    num(trkEcal.phi),
                                    std::to_string(genMuons.size()),
                                    std::to_string(nGenOk),
                                    std::to_string(nStartBeyond),
                                    std::to_string(nFailure),
                                    std::to_string(recoEcal.size()),
                                    std::to_string(nRecoNotOk),
                                    std::to_string(nStandaloneOnly)};
      auto details = [&](const Nearest& n) {
        if (n.g && n.dr < maxThreshold) {
          const auto& p = gen[n.g->index];
          const auto* mom = firstDifferentMother(p);
          g.insert(g.end(), {std::to_string(n.g->index), std::to_string(p.pdgId()), std::to_string(p.charge()),
                             num(p.pt()), num(p.eta()), num(p.phi()), num(p.vertex().rho()), num(p.vz()),
                             mom ? std::to_string(mom->pdgId()) : "", num(n.dr)});
        } else {
          g.insert(g.end(), 9, "");
          g.push_back(num(n.dr));
        }
      };
      for (const Nearest* n : {&any, &same})
        for (const int c : n->within) g.push_back(std::to_string(c));
      for (const Nearest* n : {&any, &same}) details(*n);
      for (const Nearest* n : {&any, &same}) {
        if (n->g && n->g->hasOwner) {
          g.push_back(n->g->owner == ref ? "this" : "other");
          g.push_back(num(n->g->ownerDR));
        } else {
          g.insert(g.end(), 2, "");
        }
      }
      for (const Nearest* n : {&anyExcl, &sameExcl})
        for (const int c : n->within) g.push_back(std::to_string(c));
      for (const Nearest* n : {&anyExcl, &sameExcl}) details(*n);
      for (const Nearest* n : {&anyExcl, &sameExcl}) {
        if (users.size() == 2 && n->g && n->dr < maxThreshold) {
          const Directions da = directions(muons[users[0]], n->g->fts, setup, along, opposite);
          const Directions db = directions(muons[users[1]], n->g->fts, setup, along, opposite);
          for (const int c : {da.forward, da.behind, da.failure, db.forward, db.behind, db.failure})
            g.push_back(std::to_string(c));
        } else {
          g.insert(g.end(), 6, "");
        }
      }
      genMatched_ << policyRow << "," << join(g) << "\n";
      ++nGenMatchedRows_;
    }
  }

  void endJob() override {
    policy_.close();
    genMatched_.close();
    std::cout << "SharedInnerTrackDuplicatePolicy: displacedMuons with an innerTrack ref that is not available: "
              << unavailableInner_ << "\nWrote " << nPolicyRows_ << " shared innerTracks to " << policyFile_ << " and "
              << nGenMatchedRows_ << " gen-matched ones to " << genMatchedFile_ << "\n";
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
  const std::string policyFile_, genMatchedFile_;
  std::ofstream policy_, genMatched_;
  unsigned long long unavailableInner_ = 0, nPolicyRows_ = 0, nGenMatchedRows_ = 0;
  bool loggedBoundary_ = false;
};

DEFINE_FWK_MODULE(SharedInnerTrackDuplicatePolicy);
