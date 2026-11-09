// Raw displacedMuons kept in slimmedDisplacedMuons, for the slimmedDisplacedMuons loss study
// (kb/studies/slimmed-displaced-muon-loss.md).
// Produces copies (TrackRefs unchanged) of the raw displacedMuons that have an exact partner in the stored
// slimmedDisplacedMuons, in raw order. Partner: equal charge, muon type flags (global/tracker/standalone), component
// presence, and exactly equal pT/eta/phi/charge of the inner, outer and global tracks (PAT embeds exact copies), plus
// equal outer-track DSA-ID inputs (valid CSC/DT hits, normalized chi2, ptError). Every slimmed muon must have exactly
// one raw partner (or belong to a group of k identical slimmed muons with the same k identical raw candidates), no raw
// muon may be claimed twice, and partners must be in increasing raw index in
// originalObjectRef-key order (the production filter keeps raw order); violations are counted per event.
// Diagnostic only: filterCode per raw muon replicates CMSSW_13_0_13 DisplacedMuonFilterProducer cuts that can be
// evaluated on MINIAOD (0 passes; 1 STA+inner both pT < min; 2 STA-only pT < minPtSTA; 3 STA-only < minMatches valid
// DT+CSC hits in hitPattern; 4 no-STA inner pT < minPtTK; 5 no-STA not isTrackerMuon; 6 neither STA nor inner track).
// The producer's STA-only count uses TrackExtra rechits, not stored in MINIAOD, so code 0 does not imply kept.
#include <algorithm>
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
    tree_->Branch("nSlimmed", &nSlimmed_);
    tree_->Branch("badNoPartner", &badNoPartner_);  // slimmed muons with no raw partner
    tree_->Branch("badAmbig", &badAmbig_);          // slimmed muons with > 1 raw partner, or raw claimed twice
    tree_->Branch("badOrder", &badOrder_);          // partners not increasing in originalObjectRef-key order
  }

  void produce(edm::Event& ev, const edm::EventSetup&) override {
    run_ = ev.id().run();
    lumi_ = ev.id().luminosityBlock();
    event_ = ev.id().event();
    const auto& raw = ev.get(raw_);
    const auto& slimmed = ev.get(slimmed_);
    nRaw_ = raw.size();
    nSlimmed_ = slimmed.size();
    badNoPartner_ = badAmbig_ = badOrder_ = 0;
    filterCode_.resize(raw.size());
    for (unsigned int i = 0; i < raw.size(); ++i)
      filterCode_[i] = code(raw[i]);

    // Exact partners of each slimmed muon among the raw muons. Raw displacedMuons can contain identical copies (same
    // TrackRefs); when k slimmed muons share the same k raw candidates, all k are kept, paired in key/raw order.
    std::vector<bool> kept(raw.size(), false);
    std::vector<std::pair<unsigned int, int>> keyToRaw;  // (originalObjectRef key, raw index)
    std::vector<std::vector<unsigned int>> cand(slimmed.size());
    for (unsigned int j = 0; j < slimmed.size(); ++j)
      for (unsigned int i = 0; i < raw.size(); ++i)
        if (sameMuon(raw[i], slimmed[j]))
          cand[j].push_back(i);
    std::vector<bool> done(slimmed.size(), false);
    for (unsigned int j = 0; j < slimmed.size(); ++j) {
      if (done[j])
        continue;
      if (cand[j].empty()) {
        ++badNoPartner_;
        continue;
      }
      std::vector<unsigned int> group, keys;
      for (unsigned int k = j; k < slimmed.size(); ++k)
        if (cand[k] == cand[j]) {
          group.push_back(k);
          keys.push_back(slimmed[k].originalObjectRef().key());
          done[k] = true;
        }
      bool clash = group.size() != cand[j].size();
      for (unsigned int i : cand[j])
        clash = clash || kept[i];
      if (clash) {
        badAmbig_ += group.size();
        continue;
      }
      std::sort(keys.begin(), keys.end());
      for (unsigned int k = 0; k < keys.size(); ++k) {
        kept[cand[j][k]] = true;
        keyToRaw.emplace_back(keys[k], cand[j][k]);
      }
    }
    std::sort(keyToRaw.begin(), keyToRaw.end());
    for (unsigned int k = 1; k < keyToRaw.size(); ++k)
      if (keyToRaw[k].first == keyToRaw[k - 1].first || keyToRaw[k].second <= keyToRaw[k - 1].second)
        ++badOrder_;

    auto filtered = std::make_unique<std::vector<reco::Muon>>();
    rawToFiltered_.assign(raw.size(), -1);
    for (unsigned int i = 0; i < raw.size(); ++i)
      if (kept[i]) {
        rawToFiltered_[i] = filtered->size();
        filtered->push_back(raw[i]);
      }
    tree_->Fill();
    ev.put(std::move(filtered));
  }

private:
  int code(const reco::Muon& mu) const {
    if (mu.isStandAloneMuon()) {
      if (mu.innerTrack().isNonnull())
        return mu.innerTrack()->pt() < minPtTK_ && mu.standAloneMuon()->pt() < minPtSTA_ ? 1 : 0;
      if (mu.standAloneMuon()->pt() < minPtSTA_)
        return 2;
      const auto& hp = mu.standAloneMuon()->hitPattern();
      return hp.numberOfValidMuonDTHits() + hp.numberOfValidMuonCSCHits() < minMatches_ ? 3 : 0;
    }
    if (mu.innerTrack().isNonnull())
      return mu.innerTrack()->pt() < minPtTK_ ? 4 : (!mu.isTrackerMuon() ? 5 : 0);
    return 6;
  }

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

  static bool sameMuon(const reco::Muon& r, const pat::Muon& s) {
    return r.charge() == s.charge() && r.isGlobalMuon() == s.isGlobalMuon() &&
           r.isTrackerMuon() == s.isTrackerMuon() && r.isStandAloneMuon() == s.isStandAloneMuon() &&
           sameTrack(r.innerTrack(), s.innerTrack(), false) && sameTrack(r.outerTrack(), s.outerTrack(), true) &&
           sameTrack(r.globalTrack(), s.globalTrack(), false);
  }

  const edm::EDGetTokenT<std::vector<reco::Muon>> raw_;
  const edm::EDGetTokenT<std::vector<pat::Muon>> slimmed_;
  const double minPtSTA_, minPtTK_, minMatches_;
  TTree* tree_;
  unsigned int run_ = 0, lumi_ = 0;
  unsigned long long event_ = 0;
  std::vector<int> filterCode_, rawToFiltered_;
  int nRaw_ = 0, nSlimmed_ = 0, badNoPartner_ = 0, badAmbig_ = 0, badOrder_ = 0;
};

DEFINE_FWK_MODULE(SlimmedDisplacedFilterCheck);
