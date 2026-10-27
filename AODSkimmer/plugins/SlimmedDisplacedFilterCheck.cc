// Replicated MiniAOD displaced-muon filter for the slimmedDisplacedMuons loss study
// (kb/studies/slimmed-displaced-muon-loss.md).
// Produces copies (TrackRefs unchanged) of the raw displacedMuons that pass the selection of CMSSW_13_0_13
// DisplacedMuonFilterProducer (the input of patDisplacedMuons -> slimmedDisplacedMuons), in raw order. The producer
// counts valid DT+CSC rechits of the STA track via its TrackExtra, which signal MINIAOD does not store; here the count
// is hitPattern().numberOfValidMuonDTHits() + numberOfValidMuonCSCHits() of the STA track.
// Validation: stored slimmedDisplacedMuons must correspond one-to-one, by originalObjectRef key (index in the
// production-time filtered collection, raw order), to this collection, with exactly equal charge, muon type flags,
// component presence, component pT/eta/phi/charge and the outer-track DSA-ID inputs (valid CSC/DT hits, normalized
// chi2, ptError). Tree "events" (one row per event, same order as PreselectedVertexing): filterCode per raw muon
// (0 kept; 1 STA+inner both pT < min; 2 STA-only pT < minPtSTA; 3 STA-only < minMatches valid DT+CSC hits;
// 4 no-STA inner pT < minPtTK; 5 no-STA not isTrackerMuon; 6 no STA and no inner track), rawToFiltered per raw muon
// (-1 if removed), and the mismatch counts.
#include <memory>
#include <vector>

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "TTree.h"

class SlimmedDisplacedFilterCheck : public edm::one::EDProducer<edm::one::SharedResources> {
public:
  explicit SlimmedDisplacedFilterCheck(const edm::ParameterSet& ps)
      : raw_(consumes<std::vector<reco::Muon>>(ps.getParameter<edm::InputTag>("raw"))),
        slimmed_(consumes<std::vector<pat::Muon>>(ps.getParameter<edm::InputTag>("slimmed"))),
        minPtSTA_(ps.getParameter<double>("minPtSTA")),
        minPtTK_(ps.getParameter<double>("minPtTK")),
        minMatches_(ps.getParameter<double>("minMatches")) {
    produces<std::vector<reco::Muon>>();
    usesResource("TFileService");
    edm::Service<TFileService> fs;
    tree_ = fs->make<TTree>("events", "One row per event");
    tree_->Branch("run", &run_);
    tree_->Branch("lumi", &lumi_);
    tree_->Branch("event", &event_);
    tree_->Branch("filterCode", &filterCode_);
    tree_->Branch("rawToFiltered", &rawToFiltered_);
    tree_->Branch("nRaw", &nRaw_);
    tree_->Branch("nFiltered", &nFiltered_);
    tree_->Branch("nSlimmed", &nSlimmed_);
    tree_->Branch("badSlimKey", &badSlimKey_);  // size mismatch, null, out-of-range or repeated originalObjectRef key
    tree_->Branch("badSlimObj", &badSlimObj_);  // slimmed objects differing from their filtered partner
  }

  void produce(edm::Event& ev, const edm::EventSetup&) override {
    run_ = ev.id().run();
    lumi_ = ev.id().luminosityBlock();
    event_ = ev.id().event();
    const auto& raw = ev.get(raw_);
    const auto& slimmed = ev.get(slimmed_);
    auto filtered = std::make_unique<std::vector<reco::Muon>>();
    filterCode_.assign(raw.size(), 0);
    rawToFiltered_.assign(raw.size(), -1);
    badSlimKey_ = badSlimObj_ = 0;

    // Same branch order as DisplacedMuonFilterProducer::produce.
    for (unsigned int i = 0; i < raw.size(); ++i) {
      const reco::Muon& mu = raw[i];
      int code = 0;
      if (mu.isStandAloneMuon()) {
        if (mu.innerTrack().isNonnull()) {
          if (mu.innerTrack()->pt() < minPtTK_ && mu.standAloneMuon()->pt() < minPtSTA_)
            code = 1;
        } else if (mu.standAloneMuon()->pt() < minPtSTA_) {
          code = 2;
        } else {
          const auto& hp = mu.standAloneMuon()->hitPattern();
          if (hp.numberOfValidMuonDTHits() + hp.numberOfValidMuonCSCHits() < minMatches_)
            code = 3;
        }
      } else if (mu.innerTrack().isNonnull()) {
        if (mu.innerTrack()->pt() < minPtTK_)
          code = 4;
        else if (!mu.isTrackerMuon())
          code = 5;
      } else {
        code = 6;
      }
      filterCode_[i] = code;
      if (code == 0) {
        rawToFiltered_[i] = filtered->size();
        filtered->push_back(mu);
      }
    }

    nRaw_ = raw.size();
    nFiltered_ = filtered->size();
    nSlimmed_ = slimmed.size();
    if (slimmed.size() != filtered->size())
      ++badSlimKey_;
    std::vector<bool> used(filtered->size(), false);
    for (const pat::Muon& s : slimmed) {
      const auto orig = s.originalObjectRef();
      if (orig.isNull() || orig.key() >= filtered->size() || used[orig.key()]) {
        ++badSlimKey_;
        continue;
      }
      used[orig.key()] = true;
      if (!sameMuon((*filtered)[orig.key()], s))
        ++badSlimObj_;
    }
    tree_->Fill();
    ev.put(std::move(filtered));
  }

private:
  // Exact equality of the track quantities the matching and the component IDs use (embedded copies are exact).
  static bool sameTrack(const reco::TrackRef& a, const reco::TrackRef& b, bool dsaId) {
    if (a.isNull() || b.isNull())
      return a.isNull() == b.isNull();
    if (a->pt() != b->pt() || a->eta() != b->eta() || a->phi() != b->phi() || a->charge() != b->charge())
      return false;
    if (!dsaId)
      return true;
    const auto& ha = a->hitPattern();
    const auto& hb = b->hitPattern();
    return ha.numberOfValidMuonCSCHits() == hb.numberOfValidMuonCSCHits() &&
           ha.numberOfValidMuonDTHits() == hb.numberOfValidMuonDTHits() &&
           a->normalizedChi2() == b->normalizedChi2() && a->ptError() == b->ptError();
  }

  static bool sameMuon(const reco::Muon& f, const pat::Muon& s) {
    return f.charge() == s.charge() && f.isGlobalMuon() == s.isGlobalMuon() &&
           f.isTrackerMuon() == s.isTrackerMuon() && f.isStandAloneMuon() == s.isStandAloneMuon() &&
           sameTrack(f.innerTrack(), s.innerTrack(), false) && sameTrack(f.outerTrack(), s.outerTrack(), true) &&
           sameTrack(f.globalTrack(), s.globalTrack(), false);
  }

  const edm::EDGetTokenT<std::vector<reco::Muon>> raw_;
  const edm::EDGetTokenT<std::vector<pat::Muon>> slimmed_;
  const double minPtSTA_, minPtTK_, minMatches_;
  TTree* tree_;
  unsigned int run_ = 0, lumi_ = 0;
  unsigned long long event_ = 0;
  std::vector<int> filterCode_, rawToFiltered_;
  int nRaw_ = 0, nFiltered_ = 0, nSlimmed_ = 0, badSlimKey_ = 0, badSlimObj_ = 0;
};

DEFINE_FWK_MODULE(SlimmedDisplacedFilterCheck);
