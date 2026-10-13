// Mini InspectSharedMuonDuplicate: for selected displacedMuon pairs (A, B) sharing one innerTrack, print
//  1. flags / components of both copies;
//  2. A's (and B's) outer and global track: innermost / outermost hit DetIds (TrackExtra) decoded to chambers,
//     inner / outer positions, and the HitPattern muon hits per station (MiniAOD keeps no rechits, so the
//     intermediate hits are known only at station / ring / layer level);
//  3. each copy's matched chambers with segments / RPC hits (the chambers the forward score counts);
//  4. phiOfMuonInteractionRegion exactly as MuonIdProducer (CMSSW_12_4_14_patch3) computes it, the merge test
//     |dphi(A, tracker candidate B)| < pi/2, and the direction test used to refill A's matches;
//  5. position and momentum phi of A.outer, A.global, B.inner (the shared track) and the gen muon, all propagated
//     to the same surfaces: the chamber of A.outer's innermost hit (where phi_IR(A) is taken) and B's first
//     matched chamber (whose sector phi is phi_IR(B)), with the differences.
// Read-only diagnostics; nothing is selected.
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrajectoryState/interface/TrackCharge.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

namespace {
const double kNaN = std::numeric_limits<double>::quiet_NaN();

// ---------- copied from MuonIdProducer.cc (CMSSW_12_4_14_patch3) ----------
double sectorPhi(const DetId& id) {
  double phi = 0;
  if (id.subdetId() == MuonSubdetId::DT) {
    DTChamberId muonId(id.rawId());
    if (muonId.sector() <= 12) phi = (muonId.sector() - 1) / 6. * M_PI;
    if (muonId.sector() == 13) phi = 3 / 6. * M_PI;
    if (muonId.sector() == 14) phi = 9 / 6. * M_PI;
  }
  if (id.subdetId() == MuonSubdetId::CSC) {
    CSCDetId muonId(id.rawId());
    phi = M_PI / 4 + (muonId.triggerSector() - 1) / 3. * M_PI;
  }
  if (phi > M_PI) phi -= 2 * M_PI;
  return phi;
}

double phiOfMuonInteractionRegion(const reco::Muon& muon) {
  if (muon.isStandAloneMuon()) return muon.standAloneMuon()->innerPosition().phi();
  if (muon.matches().empty()) {
    if (muon.innerTrack().isAvailable() && muon.innerTrack()->extra().isAvailable())
      return muon.innerTrack()->outerPosition().phi();
    else
      return muon.phi();
  }
  return sectorPhi(muon.matches().at(0).id);
}

// The tracker candidate as MuonIdProducer saw it at the merge test: no standalone track yet.
double phiOfTrackerCandidate(const reco::Muon& muon) {
  if (muon.matches().empty()) {
    if (muon.innerTrack().isAvailable() && muon.innerTrack()->extra().isAvailable())
      return muon.innerTrack()->outerPosition().phi();
    return muon.phi();
  }
  return sectorPhi(muon.matches().at(0).id);
}

// ---------- chamber helpers ----------
DetId chamberOf(const DetId& id) {
  if (id.det() != DetId::Muon) return id;
  switch (id.subdetId()) {
    case MuonSubdetId::DT: return DTChamberId(id.rawId());
    case MuonSubdetId::CSC: return CSCDetId(id.rawId()).chamberId();
    case MuonSubdetId::GEM: return GEMDetId(id.rawId()).chamberId();
    default: return id;  // RPC: roll
  }
}

std::string chamberName(const DetId& id) {
  std::ostringstream s;
  if (id.det() != DetId::Muon) {
    s << "non-muon DetId " << id.rawId();
    return s.str();
  }
  switch (id.subdetId()) {
    case MuonSubdetId::DT: {
      const DTChamberId c(id.rawId());
      s << "DT MB" << c.station() << " wheel " << c.wheel() << " sector " << c.sector();
      break;
    }
    case MuonSubdetId::CSC: {
      const CSCDetId c(id.rawId());
      s << "CSC ME" << (c.endcap() == 1 ? "+" : "-") << c.station() << "/" << c.ring() << " chamber " << c.chamber()
        << " (trigger sector " << c.triggerSector() << ")";
      break;
    }
    case MuonSubdetId::RPC: {
      const RPCDetId c(id.rawId());
      if (c.region() == 0)
        s << "RPC barrel ring(wheel) " << c.ring() << " station " << c.station() << " sector " << c.sector() << " layer "
          << c.layer();
      else
        s << "RPC RE" << (c.region() > 0 ? "+" : "-") << c.station() << "/" << c.ring() << " sector " << c.sector();
      break;
    }
    case MuonSubdetId::GEM: {
      const GEMDetId c(id.rawId());
      s << "GEM GE" << (c.region() > 0 ? "+" : "-") << c.station() << "/" << c.ring() << " chamber " << c.chamber();
      break;
    }
    default:
      s << "muon subdet " << id.subdetId() << " rawId " << id.rawId();
  }
  s << " [rawId " << id.rawId() << "]";
  return s.str();
}

std::string components(const reco::Muon& mu) {
  return std::string("I") + (mu.outerTrack().isNonnull() ? "+O" : "") + (mu.globalTrack().isNonnull() ? "+G" : "");
}

void printHitPattern(const reco::Track& t) {
  const auto& hp = t.hitPattern();
  std::map<std::string, std::array<int, 4>> counts;  // valid, missing, inactive, bad
  for (int i = 0; i < hp.numberOfAllHits(reco::HitPattern::TRACK_HITS); ++i) {
    const uint16_t p = hp.getHitPattern(reco::HitPattern::TRACK_HITS, i);
    if (!reco::HitPattern::muonHitFilter(p)) continue;
    std::ostringstream key;
    const uint32_t sub = reco::HitPattern::getSubStructure(p);
    switch (sub) {
      case MuonSubdetId::DT:
        key << "DT station " << reco::HitPattern::getMuonStation(p) << " superlayer "
            << reco::HitPattern::getDTSuperLayer(p);
        break;
      case MuonSubdetId::CSC:
        key << "CSC station " << reco::HitPattern::getMuonStation(p) << " ring " << reco::HitPattern::getCSCRing(p);
        break;
      case MuonSubdetId::RPC:
        key << "RPC " << (reco::HitPattern::getRPCregion(p) ? "endcap" : "barrel") << " station "
            << reco::HitPattern::getMuonStation(p) << " layer " << reco::HitPattern::getRPCLayer(p);
        break;
      case MuonSubdetId::GEM:
        key << "GEM station " << reco::HitPattern::getGEMStation(p) << " layer " << reco::HitPattern::getGEMLayer(p);
        break;
      default:
        key << "muon subdet " << sub;
    }
    const uint32_t type = reco::HitPattern::getHitType(p);
    if (type < 4) ++counts[key.str()][type];
  }
  std::cout << "      HitPattern muon hits (valid / missing / inactive / bad); no z side or sector stored:\n";
  if (counts.empty()) std::cout << "        none\n";
  for (const auto& [k, c] : counts)
    std::cout << "        " << std::left << std::setw(36) << k << c[0] << " / " << c[1] << " / " << c[2] << " / " << c[3]
              << "\n";
}

void printTrack(const char* label, const reco::TrackRef& ref) {
  std::cout << "    " << label << ": ";
  if (ref.isNull()) {
    std::cout << "none\n";
    return;
  }
  if (!ref.isAvailable()) {
    std::cout << "ref not available in this file\n";
    return;
  }
  const reco::Track& t = *ref;
  std::cout << "q=" << t.charge() << " pt=" << t.pt() << " eta=" << t.eta() << " phi=" << t.phi()
            << " validMuonHits=" << t.hitPattern().numberOfValidMuonHits() << "\n";
  if (t.extra().isAvailable()) {
    const auto ip = t.innerPosition(), op = t.outerPosition();
    std::cout << "      innermost hit: " << chamberName(chamberOf(t.innerDetId())) << "\n"
              << "        position rho=" << ip.rho() << " z=" << ip.z() << " phi=" << ip.phi()
              << " | momentum phi=" << t.innerMomentum().phi() << "\n"
              << "      outermost hit: " << chamberName(chamberOf(t.outerDetId())) << "\n"
              << "        position rho=" << op.rho() << " z=" << op.z() << " phi=" << op.phi()
              << " | momentum phi=" << t.outerMomentum().phi() << "\n";
  } else {
    std::cout << "      TrackExtra not available\n";
  }
  printHitPattern(t);
}

void printMatches(const reco::Muon& mu) {
  std::cout << "    matched chambers (" << mu.matches().size() << " entries; * = has segments / hits, counted by the score):\n";
  if (mu.matches().empty()) std::cout << "      none\n";
  for (size_t i = 0; i < mu.matches().size(); ++i) {
    const auto& m = mu.matches()[i];
    const size_t items = m.segmentMatches.size() + m.rpcMatches.size() + m.gemMatches.size() + m.gemHitMatches.size();
    std::cout << "      [" << i << "] " << (items ? "* " : "  ") << chamberName(m.id) << "  segments=" << m.segmentMatches.size()
              << " rpcHits=" << m.rpcMatches.size() << " gemSegments=" << m.gemMatches.size()
              << " gemHits=" << m.gemHitMatches.size() << "\n";
  }
}

struct State {
  std::string how = "failure";  // along | opposite | failure
  double posPhi = kNaN, momPhi = kNaN, rho = kNaN, z = kNaN;
};
}  // namespace

class InspectSharedMuonPairChambers : public edm::one::EDAnalyzer<> {
public:
  explicit InspectSharedMuonPairChambers(const edm::ParameterSet& cfg)
      : muonsToken_(consumes<std::vector<reco::Muon>>(cfg.getParameter<edm::InputTag>("muons"))),
        genToken_(consumes<std::vector<reco::GenParticle>>(cfg.getParameter<edm::InputTag>("genParticles"))),
        fieldToken_(esConsumes()),
        alongToken_(esConsumes(edm::ESInputTag("", "SteppingHelixPropagatorAlong"))),
        oppositeToken_(esConsumes(edm::ESInputTag("", "SteppingHelixPropagatorOpposite"))),
        dtToken_(esConsumes()),
        cscToken_(esConsumes()),
        rpcToken_(esConsumes()),
        gemToken_(esConsumes()) {
    for (const auto& p : cfg.getParameter<std::vector<edm::ParameterSet>>("pairs"))
      pairs_.push_back({p.getParameter<std::string>("event"), p.getParameter<unsigned int>("A"),
                        p.getParameter<unsigned int>("B"), p.getParameter<int>("genIndex")});
  }

private:
  struct Pair {
    std::string event;
    unsigned int a, b;
    int genIndex;
  };

  const GeomDet* chamberDet(const DetId& id, const edm::EventSetup& setup) const {
    switch (id.subdetId()) {
      case MuonSubdetId::DT: return setup.getData(dtToken_).idToDet(id);
      case MuonSubdetId::CSC: return setup.getData(cscToken_).idToDet(id);
      case MuonSubdetId::RPC: return setup.getData(rpcToken_).idToDet(id);
      case MuonSubdetId::GEM: return setup.getData(gemToken_).idToDet(id);
      default: return nullptr;
    }
  }

  State propagate(const FreeTrajectoryState& fts, const GeomDet* det, const Propagator& along,
                  const Propagator& opposite) const {
    State s;
    if (!det) return s;
    auto r = along.propagateWithPath(fts, det->surface());
    if (r.first.isValid()) s.how = "along";
    else {
      r = opposite.propagateWithPath(fts, det->surface());
      if (r.first.isValid()) s.how = "opposite";
    }
    if (s.how != "failure") {
      const auto p = r.first.globalPosition();
      s.posPhi = p.phi();
      s.rho = p.perp();
      s.z = p.z();
      s.momPhi = r.first.globalMomentum().phi();
    }
    return s;
  }

  void analyze(const edm::Event& event, const edm::EventSetup& setup) override {
    std::ostringstream id;
    id << event.id().run() << ":" << event.luminosityBlock() << ":" << event.id().event();
    const auto& muons = event.get(muonsToken_);
    const auto& gen = event.get(genToken_);
    const MagneticField* field = &setup.getData(fieldToken_);
    const auto& along = setup.getData(alongToken_);
    const auto& opposite = setup.getData(oppositeToken_);

    for (const auto& pr : pairs_) {
      if (pr.event != id.str()) continue;
      std::cout << std::setprecision(5) << "\n################ " << pr.event << "  displacedMuons A=" << pr.a
                << " B=" << pr.b << " ################\n";
      if (pr.a >= muons.size() || pr.b >= muons.size()) {
        std::cout << "  index out of range (" << muons.size() << " displacedMuons)\n";
        continue;
      }
      const reco::Muon& A = muons[pr.a];
      const reco::Muon& B = muons[pr.b];
      const bool same = A.innerTrack().isNonnull() && B.innerTrack().isNonnull() && A.innerTrack() == B.innerTrack();
      std::cout << "  same innerTrack: " << (same ? "yes" : "NO") << "\n";

      // 1-3: per copy
      const std::vector<std::pair<const char*, const reco::Muon*>> copies = {{"A", &A}, {"B", &B}};
      for (const auto& [label, mu] : copies) {
        std::cout << "\n  " << label << ": components " << components(*mu) << " | isGlobal=" << mu->isGlobalMuon()
                  << " isTracker=" << mu->isTrackerMuon() << " isStandAlone=" << mu->isStandAloneMuon()
                  << " isRPC=" << mu->isRPCMuon() << " | pt=" << mu->pt() << " eta=" << mu->eta() << " phi=" << mu->phi()
                  << "\n";
        printTrack("outer (standalone) track", mu->outerTrack());
        printTrack("global track", mu->globalTrack());
        printMatches(*mu);
      }
      std::cout << "\n  shared inner track:\n";
      printTrack("inner track", A.innerTrack());

      // 4: MuonIdProducer phi tests
      const double phiA = phiOfMuonInteractionRegion(A), phiBmerge = phiOfTrackerCandidate(B);
      const double dMerge = reco::deltaPhi(phiA, phiBmerge);
      std::cout << "\n  MuonIdProducer phi_interactionRegion:\n"
                << "    A (" << (A.isStandAloneMuon() ? "standalone innerPosition phi" : "tracker rules") << "): " << phiA
                << "\n    B as tracker candidate at the merge test ("
                << (B.matches().empty() ? "no matches: inner track outerPosition phi"
                                        : "sectorPhi of first match: " + chamberName(B.matches()[0].id))
                << "): " << phiBmerge << "\n";
      if (!B.matches().empty() && (B.matches()[0].id.subdetId() == MuonSubdetId::RPC ||
                                   B.matches()[0].id.subdetId() == MuonSubdetId::GEM))
        std::cout << "    note: sectorPhi returns 0 for an RPC/GEM first match\n";
      std::cout << "    merge test |dphi(A, B)| = " << std::abs(dMerge) << " < pi/2 ? "
                << (std::abs(dMerge) < M_PI_2 ? "yes (would merge)" : "NO (B kept as separate muon)") << "\n";
      const double dFill = reco::deltaPhi(phiA, A.phi());
      std::cout << "    A refill direction: |dphi(phi_IR(A), A.phi())| = " << std::abs(dFill) << " -> "
                << (std::abs(dFill) < M_PI_2 ? "InsideOut" : "OutsideIn")
                << " (used only if A had no matches after the tracker-muon step)\n";

      // 5: common surfaces
      std::vector<std::pair<std::string, DetId>> surfaces;
      if (A.outerTrack().isNonnull() && A.outerTrack().isAvailable() && A.outerTrack()->extra().isAvailable())
        surfaces.push_back({"A.outer innermost-hit chamber (phi_IR(A) is taken here)", chamberOf(A.outerTrack()->innerDetId())});
      if (!B.matches().empty())
        surfaces.push_back({"B first matched chamber (its sector phi is phi_IR(B))", B.matches()[0].id});

      std::vector<std::pair<std::string, FreeTrajectoryState>> objects;
      if (A.outerTrack().isNonnull() && A.outerTrack().isAvailable() && A.outerTrack()->extra().isAvailable())
        objects.push_back({"A.outer", trajectoryStateTransform::innerFreeState(*A.outerTrack(), field)});
      if (A.globalTrack().isNonnull() && A.globalTrack().isAvailable() && A.globalTrack()->extra().isAvailable())
        objects.push_back({"A.global", trajectoryStateTransform::outerFreeState(*A.globalTrack(), field)});
      if (A.innerTrack().isAvailable() && A.innerTrack()->extra().isAvailable())
        objects.push_back({"B.inner (shared)", trajectoryStateTransform::outerFreeState(*A.innerTrack(), field)});
      // kinematics: gen at generation (production vertex) vs reco objects (reference point / muon p4)
      std::cout << "\n  kinematics (q, pt, eta, phi):\n";
      auto kin = [](const std::string& name, int q, double pt, double eta, double phi) {
        std::cout << "    " << std::left << std::setw(34) << name << std::setw(4) << q << " pt=" << std::setw(10) << pt
                  << " eta=" << std::setw(10) << eta << " phi=" << phi << "\n";
      };
      if (pr.genIndex >= 0 && static_cast<size_t>(pr.genIndex) < gen.size()) {
        const auto& p = gen[pr.genIndex];
        kin("gen [" + std::to_string(pr.genIndex) + "] at production vertex", p.charge(), p.pt(), p.eta(), p.phi());
      }
      kin("A (displacedMuon p4)", A.charge(), A.pt(), A.eta(), A.phi());
      kin("B (displacedMuon p4)", B.charge(), B.pt(), B.eta(), B.phi());
      if (A.innerTrack().isAvailable())
        kin("shared inner track", A.innerTrack()->charge(), A.innerTrack()->pt(), A.innerTrack()->eta(), A.innerTrack()->phi());
      if (A.outerTrack().isNonnull() && A.outerTrack().isAvailable())
        kin("A outer (standalone) track", A.outerTrack()->charge(), A.outerTrack()->pt(), A.outerTrack()->eta(),
            A.outerTrack()->phi());
      if (A.globalTrack().isNonnull() && A.globalTrack().isAvailable())
        kin("A global track", A.globalTrack()->charge(), A.globalTrack()->pt(), A.globalTrack()->eta(),
            A.globalTrack()->phi());
      if (B.outerTrack().isNonnull() && B.outerTrack().isAvailable())
        kin("B outer (standalone) track", B.outerTrack()->charge(), B.outerTrack()->pt(), B.outerTrack()->eta(),
            B.outerTrack()->phi());

      if (pr.genIndex >= 0 && static_cast<size_t>(pr.genIndex) < gen.size()) {
        const auto& p = gen[pr.genIndex];
        std::cout << "\n  gen muon [" << pr.genIndex << "]: pdgId=" << p.pdgId() << " pt=" << p.pt() << " eta=" << p.eta()
                  << " phi=" << p.phi() << " vxy=" << p.vertex().rho() << " vz=" << p.vz() << "\n";
        objects.push_back({"gen", FreeTrajectoryState(GlobalPoint(p.vx(), p.vy(), p.vz()),
                                                       GlobalVector(p.px(), p.py(), p.pz()), TrackCharge(p.charge()), field)});
      }

      for (const auto& [name, detId] : surfaces) {
        const GeomDet* det = chamberDet(detId, setup);
        std::cout << "\n  surface: " << name << "\n    " << chamberName(detId) << (det ? "" : "  (no geometry)") << "\n";
        std::map<std::string, State> states;
        for (const auto& [obj, fts] : objects) {
          const State s = propagate(fts, det, along, opposite);
          states[obj] = s;
          std::cout << "    " << std::left << std::setw(18) << obj << std::setw(9) << s.how << " position phi="
                    << std::setw(10) << s.posPhi << " momentum phi=" << std::setw(10) << s.momPhi << " (rho=" << s.rho
                    << ", z=" << s.z << ")\n";
        }
        auto diff = [&](const std::string& x, const std::string& y) {
          if (!states.count(x) || !states.count(y) || states[x].how == "failure" || states[y].how == "failure") return;
          std::cout << "    dphi(" << x << ", " << y << "): position " << reco::deltaPhi(states[x].posPhi, states[y].posPhi)
                    << "  momentum " << reco::deltaPhi(states[x].momPhi, states[y].momPhi) << "\n";
        };
        diff("A.outer", "B.inner (shared)");
        diff("A.global", "B.inner (shared)");
        diff("gen", "A.outer");
        diff("gen", "A.global");
        diff("gen", "B.inner (shared)");
      }
    }
  }

  const edm::EDGetTokenT<std::vector<reco::Muon>> muonsToken_;
  const edm::EDGetTokenT<std::vector<reco::GenParticle>> genToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> fieldToken_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> alongToken_, oppositeToken_;
  const edm::ESGetToken<DTGeometry, MuonGeometryRecord> dtToken_;
  const edm::ESGetToken<CSCGeometry, MuonGeometryRecord> cscToken_;
  const edm::ESGetToken<RPCGeometry, MuonGeometryRecord> rpcToken_;
  const edm::ESGetToken<GEMGeometry, MuonGeometryRecord> gemToken_;
  std::vector<Pair> pairs_;
};

DEFINE_FWK_MODULE(InspectSharedMuonPairChambers);
