// Minimal primitive-column inputs for ID-passing component efficiencies. No event/eta/pT filter.
// Read ALL original events so skim-rejected events retain their accepted signal-gen denominator.
#include <array>
#include <cmath>
#include <set>
#include <string>
#include <vector>
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MuonAnalysis/MuonAssociators/interface/PropagateToMuonSetup.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TNamed.h"
#include "TTree.h"

namespace {
  const reco::GenParticle* firstDifferentMother(const reco::GenParticle& p) {
    const reco::GenParticle* cur = &p;
    std::set<const reco::GenParticle*> seen;
    while (cur && seen.insert(cur).second) {
      if (!cur->numberOfMothers()) return nullptr;
      const auto* mom = dynamic_cast<const reco::GenParticle*>(cur->mother(0));
      if (!mom) return nullptr;
      if (mom->pdgId() != p.pdgId()) return mom;
      cur = mom;
    }
    return nullptr;
  }
  bool outerID(const reco::Track& t) {
    const int csc = t.hitPattern().numberOfValidMuonCSCHits(), dt = t.hitPattern().numberOfValidMuonDTHits();
    return csc + dt > 12 && (csc > 0 || dt > 18) && t.normalizedChi2() < 2.5 &&
           t.pt() > 0 && t.ptError() / t.pt() < 1;
  }
  std::array<float, 3> position(const TrajectoryStateOnSurface& state) {
    if (!state.isValid()) return {{0, -999, -999}};
    const auto p = state.globalPosition();
    if (!std::isfinite(p.eta()) || !std::isfinite(p.phi())) return {{0, -999, -999}};
    return {{1, static_cast<float>(p.eta()), static_cast<float>(p.phi())}};
  }
}

class ComponentMatchingInputs : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit ComponentMatchingInputs(const edm::ParameterSet& ps)
      : gen_(consumes<std::vector<reco::GenParticle>>(ps.getParameter<edm::InputTag>("genParticles"))),
        muons_(consumes<std::vector<reco::Muon>>(ps.getParameter<edm::InputTag>("muons"))),
        indices_(consumes<std::vector<int>>(ps.getParameter<edm::InputTag>("originalIndex"))),
        field_(esConsumes()), setup_(ps.getParameter<edm::ParameterSet>("propagatorSt2"), consumesCollector()) {
    usesResource(TFileService::kSharedResource);
    edm::Service<TFileService> fs;
    fs->make<TNamed>("sample", ps.getParameter<std::string>("sample").c_str());
    fs->make<TNamed>("revision", ps.getParameter<std::string>("revision").c_str());
    fs->make<TNamed>("definitions", "All original events; accepted signal gen: status1,lastCopy,abs(pdgId)13,"
      "firstDifferentMother abs1000023; abs(eta)<2.4,0<=Lxy<740cm,abs(vz)<960cm,no pT cut. "
      "cType:0 inner,1 outer,2 global. ID-passing components of cleanedDisplacedMuons; no reco eta/pT cut. "
      "Station2 POSITION eta/phi; no fallback. cObject is ORIGINAL displacedMuons index.");
    tree_ = fs->make<TTree>("events", "All events, accepted signal gen and ID-passing cleaned components");
    tree_->Branch("run", &run_); tree_->Branch("lumi", &lumi_); tree_->Branch("event", &event_);
#define BRANCH(x) tree_->Branch(#x, &x##_)
    BRANCH(gIndex); BRANCH(gPt); BRANCH(gEta); BRANCH(gLxy); BRANCH(gD0); BRANCH(gQ);
    BRANCH(gProp); BRANCH(gEta2); BRANCH(gPhi2);
    BRANCH(cType); BRANCH(cObject); BRANCH(cQ); BRANCH(cProp); BRANCH(cEta2); BRANCH(cPhi2);
#undef BRANCH
  }

  void analyze(const edm::Event& ev, const edm::EventSetup& es) override {
    run_ = ev.id().run(); lumi_ = ev.luminosityBlock(); event_ = ev.id().event();
    gIndex_.clear(); gPt_.clear(); gEta_.clear(); gLxy_.clear(); gD0_.clear(); gQ_.clear();
    gProp_.clear(); gEta2_.clear(); gPhi2_.clear();
    cType_.clear(); cObject_.clear(); cQ_.clear(); cProp_.clear(); cEta2_.clear(); cPhi2_.clear();
    const auto propagator = setup_.init(es);
    const auto* field = &es.getData(field_);
    const auto& gen = ev.get(gen_);
    for (size_t i = 0; i < gen.size(); ++i) {
      const auto& p = gen[i];
      if (std::abs(p.pdgId()) != 13 || p.status() != 1 || !p.isLastCopy()) continue;
      const FreeTrajectoryState start(GlobalPoint(p.vx(), p.vy(), p.vz()), GlobalVector(p.px(), p.py(), p.pz()),
                                      TrackCharge(p.charge()), field);
      const auto pos = position(propagator.extrapolate(start));  // includes nonsignal final muons
      const auto* mom = firstDifferentMother(p);
      if (!mom || std::abs(mom->pdgId()) != 1000023) continue;
      const double lxy = p.vertex().rho();
      if (!(std::abs(p.eta()) < 2.4 && lxy >= 0 && lxy < 740 && std::abs(p.vz()) < 960)) continue;
      gIndex_.push_back(i); gPt_.push_back(p.pt()); gEta_.push_back(p.eta()); gLxy_.push_back(lxy);
      gD0_.push_back(std::abs(p.vx() * std::sin(p.phi()) - p.vy() * std::cos(p.phi())));
      gQ_.push_back(p.charge()); gProp_.push_back(pos[0]); gEta2_.push_back(pos[1]); gPhi2_.push_back(pos[2]);
    }
    const auto& muons = ev.get(muons_);
    const auto& indices = ev.get(indices_);
    if (indices.size() != muons.size()) throw cms::Exception("ComponentMatchingInputs") << "Cleaner index size";
    for (size_t i = 0; i < muons.size(); ++i) {
      const auto& mu = muons[i];
      const std::array<reco::TrackRef, 3> refs{{mu.innerTrack(), mu.outerTrack(), mu.globalTrack()}};
      for (int c = 0; c < 3; ++c) {
        if (refs[c].isNull()) continue;
        if (!refs[c].isAvailable()) throw cms::Exception("ComponentMatchingInputs") << "Unavailable TrackRef";
        const bool pass = c == 0 ? mu.isTrackerMuon() : c == 1 ? outerID(*refs[c]) : mu.isGlobalMuon();
        const auto pos = position(propagator.extrapolate(*refs[c]));
        if (!pass) continue;
        cType_.push_back(c); cObject_.push_back(indices[i]); cQ_.push_back(refs[c]->charge());
        cProp_.push_back(pos[0]); cEta2_.push_back(pos[1]); cPhi2_.push_back(pos[2]);
      }
    }
    tree_->Fill();
  }
private:
  const edm::EDGetTokenT<std::vector<reco::GenParticle>> gen_;
  const edm::EDGetTokenT<std::vector<reco::Muon>> muons_;
  const edm::EDGetTokenT<std::vector<int>> indices_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> field_;
  const PropagateToMuonSetup setup_;
  TTree* tree_ = nullptr;
  unsigned int run_, lumi_;
  unsigned long long event_;
  std::vector<int> gIndex_, gQ_, gProp_, cType_, cObject_, cQ_, cProp_;
  std::vector<float> gPt_, gEta_, gLxy_, gD0_, gEta2_, gPhi2_, cEta2_, cPhi2_;
};
DEFINE_FWK_MODULE(ComponentMatchingInputs);
