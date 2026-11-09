#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <set>
#include <string>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "DataFormats/Math/interface/deltaPhi.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonChamberMatch.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/MuonDetId/interface/ME0DetId.h"


namespace {

std::string yesNo(bool x) {
  return x ? "YES" : "NO";
}


std::string muonSubdetName(unsigned int subdet) {
  switch (subdet) {
    case MuonSubdetId::DT:
      return "DT";
    case MuonSubdetId::CSC:
      return "CSC";
    case MuonSubdetId::RPC:
      return "RPC";
    case MuonSubdetId::GEM:
      return "GEM";
    case MuonSubdetId::ME0:
      return "ME0";
    default:
      return "UNKNOWN";
  }
}


// Exact helper used by MuonIdProducer.cc in CMSSW_12_4_14_patch3.
double sectorPhi(const DetId& id) {
  double phi = 0.0;

  if (id.subdetId() == MuonSubdetId::DT) {
    DTChamberId muonId(id.rawId());

    if (muonId.sector() <= 12)
      phi = (muonId.sector() - 1) / 6.0 * M_PI;

    if (muonId.sector() == 13)
      phi = 3.0 / 6.0 * M_PI;

    if (muonId.sector() == 14)
      phi = 9.0 / 6.0 * M_PI;
  }

  if (id.subdetId() == MuonSubdetId::CSC) {
    CSCDetId muonId(id.rawId());

    phi =
        M_PI / 4.0 +
        (muonId.triggerSector() - 1) / 3.0 * M_PI;
  }

  if (phi > M_PI)
    phi -= 2.0 * M_PI;

  return phi;
}


// This reproduces MuonIdProducer::phiOfMuonInteractionRegion()
// for the FINAL stored reco::Muon object.
double phiOfMuonInteractionRegionFinal(const reco::Muon& muon) {
  if (muon.isStandAloneMuon() &&
      muon.standAloneMuon().isNonnull()) {
    return muon.standAloneMuon()->innerPosition().phi();
  }

  if (muon.matches().empty()) {
    if (muon.innerTrack().isAvailable() &&
        muon.innerTrack()->extra().isAvailable()) {
      return muon.innerTrack()->outerPosition().phi();
    }

    return muon.phi();
  }

  return sectorPhi(muon.matches().at(0).id);
}


// What the trackerMuon candidate would have used at the point
// where MuonIdProducer checked:
//
//   same innerTrack && abs(deltaPhi(...)) < pi/2
//
// At that stage the trackerMuon candidate is constructed from the
// inner track and filled with chamber matches. It has not yet acquired
// a later standalone-track association.
double phiTrackerCandidateAtMergeTime(const reco::Muon& muon) {
  if (!muon.matches().empty())
    return sectorPhi(muon.matches().at(0).id);

  if (muon.innerTrack().isAvailable() &&
      muon.innerTrack()->extra().isAvailable()) {
    return muon.innerTrack()->outerPosition().phi();
  }

  return muon.phi();
}


uint32_t chamberRawId(const DetId& id) {
  if (id.det() != DetId::Muon)
    return 0;

  switch (id.subdetId()) {
    case MuonSubdetId::DT:
      return DTChamberId(id.rawId()).rawId();

    case MuonSubdetId::CSC:
      return CSCDetId(id.rawId()).chamberId().rawId();

    case MuonSubdetId::RPC: {
      const RPCDetId rpc(id.rawId());

      const RPCDetId chamberId(
          rpc.region(),
          rpc.ring(),
          rpc.station(),
          rpc.sector(),
          rpc.layer(),
          rpc.subsector(),
          0);

      return chamberId.rawId();
    }

    case MuonSubdetId::GEM:
      return GEMDetId(id.rawId()).chamberId().rawId();

    case MuonSubdetId::ME0:
      return ME0DetId(id.rawId()).chamberId().rawId();

    default:
      return id.rawId();
  }
}


void printTrackRef(const std::string& label,
                   const reco::TrackRef& ref) {
  std::cout
      << "  "
      << std::left
      << std::setw(14)
      << label;

  if (ref.isNull()) {
    std::cout << "NULL\n";
    return;
  }

  std::cout
      << "ProductID=" << ref.id()
      << " key=" << ref.key()
      << " q=" << ref->charge()
      << " pt=" << ref->pt()
      << " eta=" << ref->eta()
      << " phi=" << ref->phi()
      << " validHits=" << ref->numberOfValidHits();

  if (ref->ndof() > 0.0) {
    std::cout
        << " chi2/ndof="
        << ref->chi2() / ref->ndof();
  } else {
    std::cout << " chi2/ndof=n/a";
  }

  std::cout << "\n";

  // These are particularly important for the standalone track.
  std::cout
      << "      innerPosition = ("
      << ref->innerPosition().x() << ", "
      << ref->innerPosition().y() << ", "
      << ref->innerPosition().z() << ")"
      << " phi=" << ref->innerPosition().phi()
      << "\n";

  std::cout
      << "      outerPosition = ("
      << ref->outerPosition().x() << ", "
      << ref->outerPosition().y() << ", "
      << ref->outerPosition().z() << ")"
      << " phi=" << ref->outerPosition().phi()
      << "\n";
}


// Do not dereference Track::recHits() in MiniAOD.
// The underlying TrackingRecHit product is not stored.
void printTrackHitPattern(const reco::TrackRef& trk,
                          const std::string& label) {
  std::cout
      << "\n  "
      << label
      << " hit-pattern summary:\n";

  if (trk.isNull()) {
    std::cout << "    [track is NULL]\n";
    return;
  }

  const auto& hp = trk->hitPattern();

  std::cout
      << "    validHits     = "
      << trk->numberOfValidHits()
      << "\n"
      << "    validMuonHits = "
      << hp.numberOfValidMuonHits()
      << "\n";
}


struct MuonMatchSummary {
  std::set<uint32_t> allChambers;
  std::set<uint32_t> chambersWithItems;

  unsigned int nSegmentMatches = 0;
  unsigned int nRPCMatches = 0;
  unsigned int nGEMMatches = 0;
  unsigned int nME0Matches = 0;
};


MuonMatchSummary printMuonMatches(const reco::Muon& mu) {
  MuonMatchSummary summary;

  std::cout << "\n  reco::Muon chamber matches:\n";

  std::cout
      << "    matches().size() = "
      << mu.matches().size()
      << "\n";

  unsigned int i = 0;

  for (const auto& match : mu.matches()) {
    const DetId id(match.id.rawId());
    const uint32_t chamber = chamberRawId(id);

    const unsigned int nSegments =
        match.segmentMatches.size();

    const unsigned int nRPC =
        match.rpcMatches.size();

    const unsigned int nGEM =
        match.gemMatches.size();

    const unsigned int nME0 =
        match.me0Matches.size();

    summary.allChambers.insert(chamber);

    summary.nSegmentMatches += nSegments;
    summary.nRPCMatches += nRPC;
    summary.nGEMMatches += nGEM;
    summary.nME0Matches += nME0;

    if (nSegments + nRPC + nGEM + nME0 > 0)
      summary.chambersWithItems.insert(chamber);

    std::cout
        << "    match[" << i << "]"
        << " rawId=" << match.id.rawId()
        << " subdet=" << muonSubdetName(id.subdetId())
        << " chamberRawId=" << chamber;

    if (id.subdetId() == MuonSubdetId::CSC) {
      CSCDetId csc(id.rawId());

      std::cout
          << " CSC(endcap=" << csc.endcap()
          << ", station=" << csc.station()
          << ", ring=" << csc.ring()
          << ", chamber=" << csc.chamber()
          << ", triggerSector=" << csc.triggerSector()
          << ")";
    }

    if (id.subdetId() == MuonSubdetId::DT) {
      DTChamberId dt(id.rawId());

      std::cout
          << " DT(wheel=" << dt.wheel()
          << ", station=" << dt.station()
          << ", sector=" << dt.sector()
          << ")";
    }

    std::cout
        << " sectorPhi=" << sectorPhi(id)
        << " nSegments=" << nSegments
        << " nRPC=" << nRPC
        << " nGEM=" << nGEM
        << " nME0=" << nME0
        << "\n";

    ++i;
  }

  std::cout
      << "    -> distinct chamber entries = "
      << summary.allChambers.size()
      << "\n"
      << "    -> chambers with actual match items = "
      << summary.chambersWithItems.size()
      << "\n"
      << "    -> total segment matches = "
      << summary.nSegmentMatches
      << "\n"
      << "    -> total RPC matches = "
      << summary.nRPCMatches
      << "\n"
      << "    -> total GEM matches = "
      << summary.nGEMMatches
      << "\n"
      << "    -> total ME0 matches = "
      << summary.nME0Matches
      << "\n";

  return summary;
}


void printSet(const std::string& label,
              const std::set<uint32_t>& values) {
  std::cout
      << "  "
      << label
      << " ("
      << values.size()
      << "):";

  if (values.empty()) {
    std::cout << " none";
  } else {
    for (const auto value : values)
      std::cout << " " << value;
  }

  std::cout << "\n";
}


std::set<uint32_t> intersection(
    const std::set<uint32_t>& a,
    const std::set<uint32_t>& b) {
  std::set<uint32_t> out;

  std::set_intersection(
      a.begin(),
      a.end(),
      b.begin(),
      b.end(),
      std::inserter(out, out.begin()));

  return out;
}


MuonMatchSummary inspectMuon(const reco::Muon& mu,
                             unsigned int index) {
  std::cout
      << "\n============================================================\n"
      << " displacedMuon[" << index << "]\n"
      << "============================================================\n";

  std::cout
      << "  reco pt=" << mu.pt()
      << " eta=" << mu.eta()
      << " phi=" << mu.phi()
      << " charge=" << mu.charge()
      << "\n";

  std::cout
      << "  isTrackerMuon     = "
      << mu.isTrackerMuon()
      << "\n"
      << "  isStandAloneMuon  = "
      << mu.isStandAloneMuon()
      << "\n"
      << "  isGlobalMuon      = "
      << mu.isGlobalMuon()
      << "\n"
      << "  numberOfMatches   = "
      << mu.numberOfMatches()
      << "\n"
      << "  matchedStations   = "
      << mu.numberOfMatchedStations()
      << "\n";

  std::cout
      << "\n  Track-ref existence:\n"
      << "    has innerTrack  = "
      << yesNo(mu.innerTrack().isNonnull())
      << "\n"
      << "    has outerTrack  = "
      << yesNo(mu.outerTrack().isNonnull())
      << "\n"
      << "    has globalTrack = "
      << yesNo(mu.globalTrack().isNonnull())
      << "\n";

  std::cout << "\n  Track references:\n";

  printTrackRef(
      "innerTrack",
      mu.innerTrack());

  printTrackRef(
      "outerTrack",
      mu.outerTrack());

  printTrackRef(
      "globalTrack",
      mu.globalTrack());

  const MuonMatchSummary matchSummary =
      printMuonMatches(mu);

  printTrackHitPattern(
      mu.innerTrack(),
      "innerTrack");

  printTrackHitPattern(
      mu.outerTrack(),
      "outerTrack");

  printTrackHitPattern(
      mu.globalTrack(),
      "globalTrack");

  const double phiFinal =
      phiOfMuonInteractionRegionFinal(mu);

  const double phiTrackerCandidate =
      phiTrackerCandidateAtMergeTime(mu);

  std::cout
      << "\n  phi diagnostics:\n"
      << "    phi(muon) = "
      << mu.phi()
      << "\n"
      << "    phiOfMuonInteractionRegion(final object) = "
      << phiFinal
      << "\n"
      << "    tracker-candidate phi at merge stage = "
      << phiTrackerCandidate
      << "\n";

  if (mu.outerTrack().isNonnull()) {
    std::cout
        << "    outerTrack innerPosition phi = "
        << mu.outerTrack()->innerPosition().phi()
        << "\n";
  }

  if (!mu.matches().empty()) {
    const DetId firstId(mu.matches().front().id.rawId());

    std::cout
        << "    first match rawId = "
        << firstId.rawId()
        << "\n"
        << "    first match subdet = "
        << muonSubdetName(firstId.subdetId())
        << "\n"
        << "    sectorPhi(first match) = "
        << sectorPhi(firstId)
        << "\n";
  }

  return matchSummary;
}

}  // namespace


class InspectSharedMuonDuplicate
    : public edm::one::EDAnalyzer<> {
public:
  explicit InspectSharedMuonDuplicate(
      const edm::ParameterSet& cfg)
      : muonsToken_(
            consumes<reco::MuonCollection>(
                cfg.getParameter<edm::InputTag>("muons"))),
        firstIndex_(
            cfg.getParameter<unsigned int>("firstIndex")),
        secondIndex_(
            cfg.getParameter<unsigned int>("secondIndex")) {}


  void analyze(const edm::Event& event,
               const edm::EventSetup&) override {
    edm::Handle<reco::MuonCollection> muons;
    event.getByToken(muonsToken_, muons);

    std::cout
        << "\n\n################################################################\n"
        << " EVENT "
        << event.id().run()
        << ":"
        << event.luminosityBlock()
        << ":"
        << event.id().event()
        << "\n"
        << "################################################################\n";

    std::cout
        << "\nTotal displacedMuons = "
        << muons->size()
        << "\n";


    // ------------------------------------------------------------
    // First: inspect outerTrack/globalTrack/innerTrack presence for
    // EVERY displacedMuon in the event.
    // ------------------------------------------------------------
    std::cout
        << "\n============================================================\n"
        << " ALL displacedMuon TRACK-REFERENCE SUMMARY\n"
        << "============================================================\n";

    std::cout
        << std::left
        << std::setw(8)  << "index"
        << std::setw(12) << "pt"
        << std::setw(12) << "eta"
        << std::setw(12) << "phi"
        << std::setw(10) << "tracker"
        << std::setw(10) << "STA"
        << std::setw(10) << "global"
        << std::setw(12) << "innerTrack"
        << std::setw(12) << "outerTrack"
        << std::setw(12) << "globalTrack"
        << "\n";

    for (unsigned int i = 0; i < muons->size(); ++i) {
      const auto& mu = muons->at(i);

      std::cout
          << std::left
          << std::setw(8)  << i
          << std::setw(12) << mu.pt()
          << std::setw(12) << mu.eta()
          << std::setw(12) << mu.phi()
          << std::setw(10) << yesNo(mu.isTrackerMuon())
          << std::setw(10) << yesNo(mu.isStandAloneMuon())
          << std::setw(10) << yesNo(mu.isGlobalMuon())
          << std::setw(12) << yesNo(mu.innerTrack().isNonnull())
          << std::setw(12) << yesNo(mu.outerTrack().isNonnull())
          << std::setw(12) << yesNo(mu.globalTrack().isNonnull())
          << "\n";
    }


    if (firstIndex_ >= muons->size() ||
        secondIndex_ >= muons->size()) {
      std::cout
          << "\nRequested displacedMuon indices "
          << firstIndex_
          << " and "
          << secondIndex_
          << ", but collection size is "
          << muons->size()
          << "\n";

      return;
    }

    const reco::Muon& A =
        muons->at(firstIndex_);

    const reco::Muon& B =
        muons->at(secondIndex_);


    const bool sameInnerTrack =
        A.innerTrack().isNonnull() &&
        B.innerTrack().isNonnull() &&
        A.innerTrack().id() ==
            B.innerTrack().id() &&
        A.innerTrack().key() ==
            B.innerTrack().key();


    std::cout
        << "\n============================================================\n"
        << " SELECTED DUPLICATE PAIR\n"
        << "============================================================\n";

    std::cout
        << "Same exact innerTrack? "
        << yesNo(sameInnerTrack)
        << "\n";

    if (A.innerTrack().isNonnull() &&
        B.innerTrack().isNonnull()) {
      std::cout
          << "  A innerTrack: "
          << A.innerTrack().id()
          << " key="
          << A.innerTrack().key()
          << "\n";

      std::cout
          << "  B innerTrack: "
          << B.innerTrack().id()
          << " key="
          << B.innerTrack().key()
          << "\n";
    }


    const MuonMatchSummary A_matches =
        inspectMuon(A, firstIndex_);

    const MuonMatchSummary B_matches =
        inspectMuon(B, secondIndex_);


    // ------------------------------------------------------------
    // Reproduce the actual MuonIdProducer duplicate comparison.
    //
    // At this point in MuonIdProducer:
    //
    //   A = already-created global muon
    //   B = trackerMuon candidate
    //
    // A uses standaloneMuon()->innerPosition().phi()
    // because it is a StandAloneMuon.
    //
    // B uses sectorPhi(matches()[0].id) if it has matches.
    // ------------------------------------------------------------

    std::cout
        << "\n============================================================\n"
        << " MuonIdProducer MERGE-TIME dPhi TEST\n"
        << "============================================================\n";

    const double phiA_merge =
        phiOfMuonInteractionRegionFinal(A);

    const double phiB_merge =
        phiTrackerCandidateAtMergeTime(B);

    const double dphi =
        reco::deltaPhi(phiA_merge, phiB_merge);

    const double absDphi =
        std::abs(dphi);

    std::cout
        << std::setprecision(10)
        << "  phi_A(existing global) = "
        << phiA_merge
        << "\n"
        << "  phi_B(tracker candidate) = "
        << phiB_merge
        << "\n"
        << "  deltaPhi(A,B) = "
        << dphi
        << "\n"
        << "  abs(deltaPhi) = "
        << absDphi
        << "\n"
        << "  pi/2 = "
        << M_PI_2
        << "\n"
        << "  abs(deltaPhi) < pi/2 ? "
        << yesNo(absDphi < M_PI_2)
        << "\n";

    std::cout
        << "\n  MuonIdProducer logic:\n"
        << "    same innerTrack && abs(deltaPhi) < pi/2\n"
        << "    => merge tracker candidate into existing muon\n";

    std::cout
        << "\n  For this pair:\n"
        << "    same innerTrack = "
        << yesNo(sameInnerTrack)
        << "\n"
        << "    dPhi condition  = "
        << yesNo(absDphi < M_PI_2)
        << "\n"
        << "    expected merge  = "
        << yesNo(
               sameInnerTrack &&
               absDphi < M_PI_2)
        << "\n";


    // ------------------------------------------------------------
    // Chamber overlap.
    // ------------------------------------------------------------

    std::cout
        << "\n============================================================\n"
        << " MUON-CHAMBER COMPARISON\n"
        << "============================================================\n";

    printSet(
        "A all muon.matches() chambers",
        A_matches.allChambers);

    printSet(
        "A chambers with actual match items",
        A_matches.chambersWithItems);

    printSet(
        "B all muon.matches() chambers",
        B_matches.allChambers);

    printSet(
        "B chambers with actual match items",
        B_matches.chambersWithItems);


    const auto commonAllChambers =
        intersection(
            A_matches.allChambers,
            B_matches.allChambers);

    const auto Aall_Bactive =
        intersection(
            A_matches.allChambers,
            B_matches.chambersWithItems);

    const auto activeBoth =
        intersection(
            A_matches.chambersWithItems,
            B_matches.chambersWithItems);


    std::cout << "\nOverlap tests:\n";

    printSet(
        "A all chambers intersect B all chambers",
        commonAllChambers);

    printSet(
        "A all chambers intersect B active chambers",
        Aall_Bactive);

    printSet(
        "A active chambers intersect B active chambers",
        activeBoth);


    // ------------------------------------------------------------
    // Compact final summary.
    // ------------------------------------------------------------

    std::cout
        << "\n============================================================\n"
        << " FINAL SUMMARY\n"
        << "============================================================\n";

    std::cout
        << "  same innerTrack = "
        << yesNo(sameInnerTrack)
        << "\n";

    std::cout
        << "\n  displacedMuon[" << firstIndex_ << "]:\n"
        << "    isTrackerMuon    = "
        << yesNo(A.isTrackerMuon())
        << "\n"
        << "    isStandAloneMuon = "
        << yesNo(A.isStandAloneMuon())
        << "\n"
        << "    isGlobalMuon     = "
        << yesNo(A.isGlobalMuon())
        << "\n"
        << "    has innerTrack   = "
        << yesNo(A.innerTrack().isNonnull())
        << "\n"
        << "    has outerTrack   = "
        << yesNo(A.outerTrack().isNonnull())
        << "\n"
        << "    has globalTrack  = "
        << yesNo(A.globalTrack().isNonnull())
        << "\n";

    std::cout
        << "\n  displacedMuon[" << secondIndex_ << "]:\n"
        << "    isTrackerMuon    = "
        << yesNo(B.isTrackerMuon())
        << "\n"
        << "    isStandAloneMuon = "
        << yesNo(B.isStandAloneMuon())
        << "\n"
        << "    isGlobalMuon     = "
        << yesNo(B.isGlobalMuon())
        << "\n"
        << "    has innerTrack   = "
        << yesNo(B.innerTrack().isNonnull())
        << "\n"
        << "    has outerTrack   = "
        << yesNo(B.outerTrack().isNonnull())
        << "\n"
        << "    has globalTrack  = "
        << yesNo(B.globalTrack().isNonnull())
        << "\n";

    std::cout
        << "\n  exact merge-time abs(dPhi_inter) = "
        << absDphi
        << "\n"
        << "  pi/2 = "
        << M_PI_2
        << "\n"
        << "  should have merged according to "
           "MuonIdProducer = "
        << yesNo(
               sameInnerTrack &&
               absDphi < M_PI_2)
        << "\n";

    std::cout << "\n";
  }


private:
  edm::EDGetTokenT<reco::MuonCollection> muonsToken_;

  unsigned int firstIndex_;
  unsigned int secondIndex_;
};


DEFINE_FWK_MODULE(InspectSharedMuonDuplicate);