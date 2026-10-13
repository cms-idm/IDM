// Export all displacedMuons for gen-matched residuals; diagnose exact shared inner/outer/global TrackRefs independently.
// Signal identification and Station-2 position propagation follow ElectronSkimmer.
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrajectoryState/interface/TrackCharge.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MuonAnalysis/MuonAssociators/interface/PropagateToMuonSetup.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TDirectory.h"
#include "TFile.h"
#include "TTree.h"
#include "TParameter.h"

namespace {
const reco::GenParticle* firstDifferentMotherInSamePdgChain(const reco::GenParticle& p) {
  const reco::GenParticle* cur = &p;
  std::set<const reco::GenParticle*> seen;
  while (cur) {
    if (seen.count(cur)) return nullptr;
    seen.insert(cur);
    if (!cur->numberOfMothers()) return nullptr;
    const auto* mom = dynamic_cast<const reco::GenParticle*>(cur->mother(0));
    if (!mom) return nullptr;
    if (mom->pdgId() != p.pdgId()) return mom;
    cur = mom;
  }
  return nullptr;
}

bool isFinalSignalMuonFromChi2(const reco::GenParticle& p) {
  if (std::abs(p.pdgId()) != 13 || p.status() != 1 || !p.isLastCopy()) return false;
  const auto* mom = firstDifferentMotherInSamePdgChain(p);
  return mom && std::abs(mom->pdgId()) == 1000023;
}

struct Station2Position {
  bool valid = false;
  float eta = -999.f;
  float phi = -999.f;
};

Station2Position position(const TrajectoryStateOnSurface& tsos) {
  if (!tsos.isValid()) return {};
  return {true, tsos.globalPosition().eta(), tsos.globalPosition().phi()};
}

Station2Position propagateGen(const reco::GenParticle& p, const MagneticField* field,
                              const PropagateToMuon& propagator) {
  if (!field || p.charge() == 0) return {};
  const GlobalPoint startPos(p.vx(), p.vy(), p.vz());
  const GlobalVector startMom(p.px(), p.py(), p.pz());
  const FreeTrajectoryState state(startPos, startMom, TrackCharge(p.charge()), field);
  return position(propagator.extrapolate(state));
}

Station2Position propagateTrack(const reco::Track& track, const PropagateToMuon& propagator) {
  return position(propagator.extrapolate(track));
}

const double unavailable = std::numeric_limits<double>::quiet_NaN();

bool usable(const Station2Position& p) {
  return p.valid && std::isfinite(p.eta) && std::isfinite(p.phi);
}

double distance(const Station2Position& a, const Station2Position& b) {
  return usable(a) && usable(b) ? reco::deltaR(a.eta, a.phi, b.eta, b.phi) : unavailable;
}

void number(std::ostream& out, double value) {
  if (std::isfinite(value)) out << value;
  else out << "N/A";
}

// Sector convention copied from CMSSW_13_0_13 MuonIdProducer::sectorPhi.
double sectorPhi(const DetId& id) {
  double phi = 0;
  if (id.subdetId() == MuonSubdetId::DT) {
    const DTChamberId muonId(id.rawId());
    if (muonId.sector() <= 12) phi = (muonId.sector() - 1) / 6. * M_PI;
    if (muonId.sector() == 13) phi = 3 / 6. * M_PI;
    if (muonId.sector() == 14) phi = 9 / 6. * M_PI;
  }
  if (id.subdetId() == MuonSubdetId::CSC) {
    const CSCDetId muonId(id.rawId());
    phi = M_PI / 4 + (muonId.triggerSector() - 1) / 3. * M_PI;
  }
  if (phi > M_PI) phi -= 2 * M_PI;
  return phi;
}

double phiOfMuonInteractionRegion(const reco::Muon& mu) {
  if (mu.isStandAloneMuon()) {
    const auto sta = mu.standAloneMuon();
    // Preserve the producer's branch; do not substitute another phi if data are unavailable.
    if (sta.isNull() || !sta.isAvailable() || sta->extra().isNull() ||
        !sta->extra().isAvailable()) return unavailable;
    return sta->innerPosition().phi();
  }
  if (!mu.matches().empty()) return sectorPhi(mu.matches().at(0).id);
  const auto inner = mu.innerTrack();
  if (inner.isNonnull() && inner.isAvailable() && inner->extra().isNonnull() &&
      inner->extra().isAvailable()) return inner->outerPosition().phi();
  return mu.phi();
}

void printTrackRef(std::ostream& out, const char* name, const reco::TrackRef& ref) {
  out << "    " << name << ": present=" << ref.isNonnull();
  if (ref.isNonnull()) out << " product=" << ref.id().id() << " key=" << ref.key();
  out << '\n';
}

void printTrackDetails(std::ostream& out, const char* name, const reco::TrackRef& ref) {
  printTrackRef(out, name, ref);
  if (ref.isNull()) return;
  if (!ref.isAvailable()) { out << "      track data: N/A (unavailable)\n"; return; }
  out << "      charge=" << ref->charge() << " pt=" << ref->pt()
      << " eta=" << ref->eta() << " phi=" << ref->phi() << '\n';
  const auto& pattern = ref->hitPattern();
  out << "      hitPattern TRACK_HITS:\n";
  pattern.print(reco::HitPattern::TRACK_HITS, out);
  out << "      hitPattern MISSING_INNER_HITS:\n";
  pattern.print(reco::HitPattern::MISSING_INNER_HITS, out);
  out << "      hitPattern MISSING_OUTER_HITS:\n";
  pattern.print(reco::HitPattern::MISSING_OUTER_HITS, out);
}

template <typename Ref>
void printAssociation(std::ostream& out, const char* name, const Ref& ref) {
  out << " " << name << " present=" << ref.isNonnull();
  if (ref.isNonnull()) out << " product=" << ref.id().id() << " key=" << ref.key();
}

const char* detectorName(int detector) {
  switch (detector) {
    case MuonSubdetId::DT: return "DT";
    case MuonSubdetId::CSC: return "CSC";
    case MuonSubdetId::RPC: return "RPC";
    case MuonSubdetId::GEM: return "GEM";
    case MuonSubdetId::ME0: return "ME0";
    default: return "other";
  }
}

void printMuon(std::ostream& out, size_t index, const reco::Muon& mu, double interactionPhi) {
  out << "  displacedMuon index=" << index << '\n'
      << "    object charge=" << mu.charge() << " pt=" << mu.pt()
      << " eta=" << mu.eta() << " phi=" << mu.phi() << '\n'
      << "    Global=" << mu.isGlobalMuon() << " Tracker=" << mu.isTrackerMuon()
      << " StandAlone=" << mu.isStandAloneMuon() << " RPC=" << mu.isRPCMuon()
      << " GEM=" << mu.isGEMMuon() << " ME0=" << mu.isME0Muon() << '\n';
  out << "    track associations (hit patterns belong to tracks):\n";
  printTrackDetails(out, "innerTrack", mu.innerTrack());
  printTrackDetails(out, "outerTrack", mu.outerTrack());
  printTrackDetails(out, "globalTrack", mu.globalTrack());
  printTrackDetails(out, "tpfmsTrack", mu.tpfmsTrack());
  printTrackDetails(out, "pickyTrack", mu.pickyTrack());
  printTrackDetails(out, "dytTrack", mu.dytTrack());
  out << "    muonBestTrackType=" << static_cast<int>(mu.muonBestTrackType())
      << " tunePMuonBestTrackType=" << static_cast<int>(mu.tunePMuonBestTrackType()) << '\n';
  printTrackDetails(out, "muonBestTrack", mu.muonBestTrack());
  printTrackDetails(out, "tunePMuonBestTrack", mu.tunePMuonBestTrack());
  out << "    phiOfMuonInteractionRegion=";
  number(out, interactionPhi);
  out << "\n    matched chambers (in mu.matches() order): " << mu.matches().size() << '\n';
  for (size_t k = 0; k < mu.matches().size(); ++k) {
    const auto& match = mu.matches()[k];
    out << "      match=" << k << " chamberId=" << match.id.rawId()
        << " detector=" << detectorName(match.detector())
        << " subdetId=" << match.detector() << " station=" << match.station()
        << " rpcMatches=" << match.rpcMatches.size()
        << " gemHitMatches=" << match.gemHitMatches.size() << '\n';
    const auto printSegments = [&](const char* label, const std::vector<reco::MuonSegmentMatch>& segments) {
      out << "        " << label << " count=" << segments.size() << '\n';
      for (size_t s = 0; s < segments.size(); ++s) {
        const auto& segment = segments[s];
        out << "          segment=" << s << " arbitrationMask=" << segment.mask;
        printAssociation(out, "DT", segment.dtSegmentRef);
        printAssociation(out, "CSC", segment.cscSegmentRef);
        printAssociation(out, "GEM", segment.gemSegmentRef);
        printAssociation(out, "ME0", segment.me0SegmentRef);
        out << '\n';
      }
    };
    printSegments("segmentMatches", match.segmentMatches);
    printSegments("gemMatches", match.gemMatches);
    printSegments("me0Matches", match.me0Matches);
    printSegments("truthMatches", match.truthMatches);
  }
}

void printPosition(std::ostream& out, const Station2Position& p) {
  if (usable(p)) out << "eta=" << p.eta << " phi=" << p.phi;
  else out << "N/A (invalid or non-finite propagation)";
}
}  // namespace

class DisplacedMuonDoubleCounting : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit DisplacedMuonDoubleCounting(const edm::ParameterSet& cfg)
      : muons_(consumes<std::vector<reco::Muon>>(cfg.getParameter<edm::InputTag>("displacedMuons"))),
        gen_(consumes<std::vector<reco::GenParticle>>(cfg.getParameter<edm::InputTag>("genParticle"))),
        field_(esConsumes<MagneticField, IdealMagneticFieldRecord>()),
        station2Setup_(cfg.getParameter<edm::ParameterSet>("genMuonPropagatorSt2"), consumesCollector()),
        outputFile_(cfg.getUntrackedParameter<std::string>("sharedTracksFile", "displaced_shared_tracks.root")) {
    usesResource("TFileService");
  }

private:
  // One row per displacedMuon in every event, independently of duplicate pairs.
  // Raw kinematics allow the companion plotter to apply the supplied acceptance.
  struct TrackRecord {
    bool present = false, valid = false, propValid = false;
    int charge = 0;
    double pt = unavailable, eta = unavailable, vxy = unavailable, vz = unavailable;
    double st2Eta = unavailable, st2Phi = unavailable;
  };

  void beginJob() override {
    TDirectory::TContext context;
    output_.reset(TFile::Open(outputFile_.c_str(), "RECREATE"));
    if (!output_ || output_->IsZombie())
      throw cms::Exception("FileOpenError") << "Cannot create " << outputFile_;
    output_->cd();
    sharedTree_ = new TTree("reco_tracks", "All displacedMuons for independent inner/global gen matching");
    sharedTree_->Branch("run", &run_, "run/i");
    sharedTree_->Branch("lumi", &lumi_, "lumi/i");
    sharedTree_->Branch("event", &event_, "event/l");
    sharedTree_->Branch("muon_index", &muonIndex_, "muon_index/i");
    sharedTree_->Branch("inner_product", &innerProduct_, "inner_product/i");
    sharedTree_->Branch("inner_key", &innerKey_, "inner_key/l");
    sharedTree_->Branch("group_size", &groupSize_, "group_size/i");
    sharedTree_->Branch("best_choice", &bestChoice_, "best_choice/I");
    sharedTree_->Branch("best_type", &bestType_, "best_type/I");
    const auto branchTrack = [&](const std::string& prefix, TrackRecord& t) {
      const auto branch = [&](const char* field, auto* address, const char* type) {
        const auto name = prefix + "_" + field;
        sharedTree_->Branch(name.c_str(), address, (name + "/" + type).c_str());
      };
      branch("present", &t.present, "O"); branch("valid", &t.valid, "O");
      branch("propValid", &t.propValid, "O"); branch("charge", &t.charge, "I");
      branch("pt", &t.pt, "D"); branch("eta", &t.eta, "D");
      branch("vxy", &t.vxy, "D"); branch("vz", &t.vz, "D");
      branch("st2Eta", &t.st2Eta, "D"); branch("st2Phi", &t.st2Phi, "D");
    };
    branchTrack("inner", inner_); branchTrack("global", global_); branchTrack("best", best_);
    sharedTree_->Branch("gen_valid", genValid_, "gen_valid[2]/O");
    sharedTree_->Branch("gen_propValid", genPropValid_, "gen_propValid[2]/O");
    sharedTree_->Branch("gen_charge", genCharge_, "gen_charge[2]/I");
    sharedTree_->Branch("gen_pt", genPt_, "gen_pt[2]/D");
    sharedTree_->Branch("gen_eta", genEta_, "gen_eta[2]/D");
    sharedTree_->Branch("gen_vxy", genVxy_, "gen_vxy[2]/D");
    sharedTree_->Branch("gen_vz", genVz_, "gen_vz[2]/D");
    sharedTree_->Branch("gen_st2Eta", genSt2Eta_, "gen_st2Eta[2]/D");
    sharedTree_->Branch("gen_st2Phi", genSt2Phi_, "gen_st2Phi[2]/D");
  }

  TrackRecord trackRecord(const reco::TrackRef& ref, const PropagateToMuon& propagator) const {
    TrackRecord t;
    t.present = ref.isNonnull();
    t.valid = t.present && ref.isAvailable();
    if (!t.valid) return t;
    t.charge = ref->charge(); t.pt = ref->pt(); t.eta = ref->eta();
    t.vxy = ref->referencePoint().rho(); t.vz = ref->vz();
    const auto p = propagateTrack(*ref, propagator);
    t.propValid = usable(p);
    if (t.propValid) { t.st2Eta = p.eta; t.st2Phi = p.phi; }
    return t;
  }

  void analyze(const edm::Event& event, const edm::EventSetup& setup) override {
    ++events_;
    const auto& muons = event.get(muons_);
    std::vector<std::pair<size_t, size_t>> pairs;
    for (size_t i = 0; i < muons.size(); ++i) {
      for (size_t j = i + 1; j < muons.size(); ++j) {
        const bool inner = muons[i].innerTrack().isNonnull() &&
                           muons[i].innerTrack() == muons[j].innerTrack();
        const bool outer = muons[i].outerTrack().isNonnull() &&
                           muons[i].outerTrack() == muons[j].outerTrack();
        const bool global = muons[i].globalTrack().isNonnull() &&
                            muons[i].globalTrack() == muons[j].globalTrack();
        if (inner) ++innerPairs_;
        if (outer) ++outerPairs_;
        if (global) ++globalPairs_;
        if (inner || outer || global) pairs.emplace_back(i, j);
      }
    }

    const auto& gen = event.get(gen_);
    std::vector<size_t> finalMuons;
    for (size_t k = 0; k < gen.size(); ++k)
      if (isFinalSignalMuonFromChi2(gen[k])) finalMuons.push_back(k);
    const reco::GenParticle* sig = nullptr;
    const reco::GenParticle* anti = nullptr;
    size_t sigIndex = 0, antiIndex = 0;
    // Preserve the existing scalar GenSigMuon/GenSigAntiMuon identification rule.
    if (finalMuons.size() == 2) {
      for (const auto k : finalMuons) {
        if (gen[k].pdgId() == 13) { sig = &gen[k]; sigIndex = k; }
        else if (gen[k].pdgId() == -13) { anti = &gen[k]; antiIndex = k; }
      }
    }
    const auto& field = setup.getData(field_);
    const auto station2 = station2Setup_.init(setup);
    const auto sigPos = sig ? propagateGen(*sig, &field, station2) : Station2Position{};
    const auto antiPos = anti ? propagateGen(*anti, &field, station2) : Station2Position{};

    const reco::GenParticle* truth[2] = {sig, anti};
    const Station2Position truthPos[2] = {sigPos, antiPos};
    for (unsigned int k = 0; k < 2; ++k) {
      genValid_[k] = truth[k] != nullptr;
      genPropValid_[k] = usable(truthPos[k]);
      genCharge_[k] = truth[k] ? truth[k]->charge() : 0;
      genPt_[k] = truth[k] ? truth[k]->pt() : unavailable;
      genEta_[k] = truth[k] ? truth[k]->eta() : unavailable;
      genVxy_[k] = truth[k] ? truth[k]->vertex().rho() : unavailable;
      genVz_[k] = truth[k] ? truth[k]->vz() : unavailable;
      genSt2Eta_[k] = genPropValid_[k] ? truthPos[k].eta : unavailable;
      genSt2Phi_[k] = genPropValid_[k] ? truthPos[k].phi : unavailable;
    }
    run_ = event.id().run(); lumi_ = event.luminosityBlock(); event_ = event.id().event();
    for (size_t index = 0; index < muons.size(); ++index) {
      const auto& mu = muons[index];
      muonIndex_ = index;
      innerProduct_ = mu.innerTrack().isNonnull() ? mu.innerTrack().id().id() : 0;
      innerKey_ = mu.innerTrack().isNonnull() ? mu.innerTrack().key() : 0;
      groupSize_ = 0;
      if (mu.innerTrack().isNonnull())
        for (const auto& member : muons)
          if (member.innerTrack() == mu.innerTrack()) ++groupSize_;
      inner_ = trackRecord(mu.innerTrack(), station2);
      global_ = trackRecord(mu.globalTrack(), station2);
      best_ = trackRecord(mu.muonBestTrack(), station2);
      bestType_ = static_cast<int>(mu.muonBestTrackType());
      // Classify the actual BestTrack reference, independently of propagation/matching.
      bestChoice_ = !best_.valid ? 3 : mu.muonBestTrack() == mu.innerTrack() ? 0
                    : mu.muonBestTrack() == mu.globalTrack() ? 1 : 2;
      sharedTree_->Fill();
      ++recoObjects_;
    }

    for (const auto& pair : pairs) {
      const auto& a = muons[pair.first];
      const auto& b = muons[pair.second];
      const bool sharesInner = a.innerTrack().isNonnull() && a.innerTrack() == b.innerTrack();
      const bool sharesOuter = a.outerTrack().isNonnull() && a.outerTrack() == b.outerTrack();
      const bool sharesGlobal = a.globalTrack().isNonnull() && a.globalTrack() == b.globalTrack();
      const char* sharedName = sharesInner ? "innerTrack" : sharesOuter ? "outerTrack" : "globalTrack";
      const auto shared = sharesInner ? a.innerTrack() : sharesOuter ? a.outerTrack() : a.globalTrack();
      const double phiA = phiOfMuonInteractionRegion(a), phiB = phiOfMuonInteractionRegion(b);
      const double dPhi = std::isfinite(phiA) && std::isfinite(phiB)
                               ? std::abs(reco::deltaPhi(phiA, phiB)) : unavailable;
      const auto sharedPos = shared.isAvailable() ? propagateTrack(*shared, station2) : Station2Position{};
      const double drSig = distance(sharedPos, sigPos), drAnti = distance(sharedPos, antiPos);
      const double drTruthSt2 = distance(sigPos, antiPos);
      const bool hasOuter = a.outerTrack().isNonnull() || b.outerTrack().isNonnull();
      const bool hasGlobal = a.globalTrack().isNonnull() || b.globalTrack().isNonnull();
      std::ostringstream out, row;
      out << std::boolalpha << std::setprecision(9)
          << "  run:lumi:event: " << event.id().run() << ':' << event.luminosityBlock()
          << ':' << event.id().event() << " displacedMuon indices=" << pair.first << ',' << pair.second << '\n';
      out << "    exact shared references: inner=" << sharesInner
          << " outer=" << sharesOuter << " global=" << sharesGlobal << '\n';
      printTrackRef(out, (std::string("shared ") + sharedName).c_str(), shared);
      if (shared.isAvailable())
        out << "    charge=" << shared->charge() << " pt=" << shared->pt()
            << " eta=" << shared->eta() << " phi=" << shared->phi() << '\n';
      else out << "    shared track kinematics: N/A (TrackRef unavailable)\n";
      out << "    shared track Station-2 position: ";
      printPosition(out, sharedPos);
      out << '\n';
      printMuon(out, pair.first, a, phiA);
      printMuon(out, pair.second, b, phiB);
      out << "  pair comparison: dR(displacedMuon i, displacedMuon j)="
          << reco::deltaR(a.eta(), a.phi(), b.eta(), b.phi())
          << " abs(dPhi_interaction)=";
      number(out, dPhi);
      out << '\n';
      const auto printGen = [&](const char* identity, const reco::GenParticle* p, size_t index,
                                const Station2Position& pos) {
        out << "  " << identity << ": ";
        if (!p) { out << "N/A (scalar signal identification failed)\n"; return; }
        out << "gen collection index=" << index << " charge=" << p->charge()
            << " pt=" << p->pt() << " eta=" << p->eta() << " phi=" << p->phi()
            << " vxy=" << p->vertex().rho() << " vz=" << p->vz() << '\n'
            << "    Station-2 position: ";
        printPosition(out, pos);
        out << '\n';
      };
      printGen("GenSigMuon", sig, sigIndex, sigPos);
      printGen("GenSigAntiMuon", anti, antiIndex, antiPos);
      out << "  shared-" << sharedName << " dR to GenSigMuon at Station 2=";
      number(out, drSig);
      out << "\n  shared-" << sharedName << " dR to GenSigAntiMuon at Station 2=";
      number(out, drAnti);
      out << "\n  truth-muon production dR=";
      number(out, sig && anti ? reco::deltaR(sig->eta(), sig->phi(), anti->eta(), anti->phi()) : unavailable);
      out << "\n  truth-muon Station-2 dR=";
      number(out, drTruthSt2);
      out << "\n";
      // Also print the other shared associations when a pair shares multiple types.
      const auto extraShared = [&](const char* name, const reco::TrackRef& ref, bool equal) {
        if (!equal || ref == shared) return;
        printTrackDetails(out, (std::string("shared ") + name).c_str(), ref);
        const auto pos = ref.isAvailable() ? propagateTrack(*ref, station2) : Station2Position{};
        out << "    shared " << name << " Station-2 position: "; printPosition(out, pos);
        out << "\n    dR to GenSigMuon="; number(out, distance(pos, sigPos));
        out << " dR to GenSigAntiMuon="; number(out, distance(pos, antiPos)); out << '\n';
      };
      extraShared("outerTrack", a.outerTrack(), sharesOuter);
      extraShared("globalTrack", a.globalTrack(), sharesGlobal);
      out << '\n';
      details_.push_back(out.str());
      row << std::setprecision(9) << event.id().run() << ':' << event.luminosityBlock()
          << ':' << event.id().event() << " [" << pair.first << ',' << pair.second << "] | " << sharesInner << "," << sharesOuter << "," << sharesGlobal
          << " | " << sharedName << " | ";
      number(row, dPhi); row << " | ";
      number(row, drSig); row << " | ";
      number(row, drAnti); row << " | ";
      number(row, drTruthSt2);
      row << " | " << (hasOuter ? "yes" : "no") << " | " << (hasGlobal ? "yes" : "no") << '\n';
      rows_.push_back(row.str());
    }
  }

  void endJob() override {
    // Keep these delimiters compatible with the existing log-to-summary extraction.
    std::cout << "\n=== displacedMuons double-counting diagnostic ===\n"
              << "Events: " << events_ << "; reco objects exported: " << recoObjects_ << '\n'
              << "Exact shared-innerTrack pairs: " << innerPairs_
              << "; shared-outerTrack pairs: " << outerPairs_
              << "; shared-globalTrack pairs: " << globalPairs_
              << "; distinct pairs sharing any track: " << details_.size() << '\n';
    for (const auto& detail : details_) std::cout << detail;
    std::cout << "Exact shared-track pair summary (distances have no charge requirement):\n"
              << "event [i,j] | shared inner,outer,global | reference used for distances | abs(dPhi_interaction) | track-genMuon dR St2 | track-genAntiMuon dR St2"
              << " | genMuon-genAntiMuon dR St2 | outer present either | global present either\n";
    for (const auto& row : rows_) std::cout << row;
    std::cout << "===============================================\n";
    {
      TDirectory::TContext context;
      output_->cd();
      sharedTree_->Write();
      TParameter<int>("schema_version", 2).Write();
      TParameter<Long64_t>("events_processed", events_).Write();
      TParameter<Long64_t>("reco_objects", recoObjects_).Write();
      TParameter<Long64_t>("duplicate_inner_pairs", innerPairs_).Write();
      TParameter<Long64_t>("duplicate_outer_pairs", outerPairs_).Write();
      TParameter<Long64_t>("duplicate_global_pairs", globalPairs_).Write();
      TParameter<Long64_t>("duplicate_any_pairs", details_.size()).Write();
      output_->Close();
    }
    std::cout << "Wrote " << recoObjects_ << " reco objects (duplicate-independent) to " << outputFile_
              << "; plot with displaced_shared_track_pt_residuals.py --input " << outputFile_ << '\n';
  }

  const edm::EDGetTokenT<std::vector<reco::Muon>> muons_;
  const edm::EDGetTokenT<std::vector<reco::GenParticle>> gen_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> field_;
  const PropagateToMuonSetup station2Setup_;
  const std::string outputFile_;
  std::unique_ptr<TFile> output_;
  TTree* sharedTree_ = nullptr;
  TrackRecord inner_, global_, best_;
  unsigned int run_ = 0, lumi_ = 0, muonIndex_ = 0, innerProduct_ = 0, groupSize_ = 0;
  unsigned long long event_ = 0, innerKey_ = 0, recoObjects_ = 0;
  unsigned long long innerPairs_ = 0, outerPairs_ = 0, globalPairs_ = 0;
  int bestChoice_ = 3, bestType_ = 0;
  bool genValid_[2] = {}, genPropValid_[2] = {};
  int genCharge_[2] = {};
  double genPt_[2] = {}, genEta_[2] = {}, genVxy_[2] = {}, genVz_[2] = {};
  double genSt2Eta_[2] = {}, genSt2Phi_[2] = {};
  unsigned long long events_ = 0;
  std::vector<std::string> details_, rows_;
};

DEFINE_FWK_MODULE(DisplacedMuonDoubleCounting);
