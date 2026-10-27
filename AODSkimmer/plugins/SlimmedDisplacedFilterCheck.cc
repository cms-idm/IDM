// Per-event bookkeeping for the slimmedDisplacedMuons loss study (kb/studies/slimmed-displaced-muon-loss.md).
// (a) Replicates DisplacedMuonFilterProducer (CMSSW_13_0_13 PhysicsTools/PatAlgos) on raw displacedMuons and checks that
//     the kept raw indices map, in order, onto the in-job filtered collection with identical inner/outer/global
//     TrackRefs (ProductID + key).
// (b) Checks that stored slimmedDisplacedMuons correspond one-to-one, by originalObjectRef key, to the filtered
//     collection, with exactly equal charge, muon type flags, component presence, component pT/eta/phi and the
//     outer-track DSA-ID inputs (valid CSC/DT hits, normalized chi2, ptError).
// Tree "events" (one row per event, same order as PreselectedVertexing): filterCode per raw muon (0 kept; 1 STA+inner
// both pT < minPt; 2 STA-only pT < minPtSTA; 3 STA-only < minMatches valid DT+CSC hits; 4 no-STA inner pT < minPtTK;
// 5 no-STA not isTrackerMuon), rawToFiltered per raw muon (-1 if removed), and the mismatch counts of (a) and (b).
#include <algorithm>
#include <vector>

#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TTree.h"

class SlimmedDisplacedFilterCheck : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit SlimmedDisplacedFilterCheck(const edm::ParameterSet& ps)
      : raw_(consumes<std::vector<reco::Muon>>(ps.getParameter<edm::InputTag>("raw"))),
        filtered_(consumes<std::vector<reco::Muon>>(ps.getParameter<edm::InputTag>("filtered"))),
        slimmed_(consumes<std::vector<pat::Muon>>(ps.getParameter<edm::InputTag>("slimmed"))),
        minPtSTA_(ps.getParameter<double>("minPtSTA")),
        minPtTK_(ps.getParameter<double>("minPtTK")),
        minMatches_(ps.getParameter<double>("minMatches")) {
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
    tree_->Branch("badFilter", &badFilter_);    // (a) size or TrackRef mismatches
    tree_->Branch("badSlimKey", &badSlimKey_);  // (b) size mismatch, out-of-range or repeated originalObjectRef key
    tree_->Branch("badSlimObj", &badSlimObj_);  // (b) slimmed objects differing from their filtered partner
  }

  void analyze(const edm::Event& ev, const edm::EventSetup&) override {
    run_ = ev.id().run();
    lumi_ = ev.id().luminosityBlock();
    event_ = ev.id().event();
    const auto& raw = ev.get(raw_);
    const auto& filtered = ev.get(filtered_);
    const auto& slimmed = ev.get(slimmed_);
    nRaw_ = raw.size();
    nFiltered_ = filtered.size();
    nSlimmed_ = slimmed.size();
    filterCode_.assign(raw.size(), 0);
    rawToFiltered_.assign(raw.size(), -1);
    badFilter_ = badSlimKey_ = badSlimObj_ = 0;

    // (a) Replicated filter decision, same branch order as DisplacedMuonFilterProducer::produce.
    std::vector<int> kept;
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
          unsigned int nseg = 0;
          for (auto hit = mu.standAloneMuon()->recHitsBegin(); hit != mu.standAloneMuon()->recHitsEnd(); ++hit) {
            if (!(*hit)->isValid())
              continue;
            const DetId id = (*hit)->geographicalId();
            if (id.det() == DetId::Muon && (id.subdetId() == MuonSubdetId::DT || id.subdetId() == MuonSubdetId::CSC))
              ++nseg;
          }
          if (nseg < minMatches_)
            code = 3;
        }
      } else if (mu.innerTrack().isNonnull()) {
        if (mu.innerTrack()->pt() < minPtTK_)
          code = 4;
        else if (!mu.isTrackerMuon())
          code = 5;
      } else {
        code = 6;  // neither STA nor inner track; the producer drops it with a warning
      }
      filterCode_[i] = code;
      if (code == 0) {
        rawToFiltered_[i] = kept.size();
        kept.push_back(i);
      }
    }
    if (kept.size() != filtered.size())
      ++badFilter_;
    for (unsigned int k = 0; k < std::min(kept.size(), filtered.size()); ++k) {
      const reco::Muon& a = raw[kept[k]];
      const reco::Muon& b = filtered[k];
      if (!sameRef(a.innerTrack(), b.innerTrack()) || !sameRef(a.outerTrack(), b.outerTrack()) ||
          !sameRef(a.globalTrack(), b.globalTrack()))
        ++badFilter_;
    }

    // (b) Slimmed <-> filtered by originalObjectRef key.
    if (slimmed.size() != filtered.size())
      ++badSlimKey_;
    std::vector<bool> used(filtered.size(), false);
    for (const pat::Muon& s : slimmed) {
      const auto orig = s.originalObjectRef();
      if (orig.isNull() || orig.key() >= filtered.size() || used[orig.key()]) {
        ++badSlimKey_;
        continue;
      }
      used[orig.key()] = true;
      if (!sameMuon(filtered[orig.key()], s))
        ++badSlimObj_;
    }
    tree_->Fill();
  }

private:
  static bool sameRef(const reco::TrackRef& a, const reco::TrackRef& b) {
    if (a.isNull() || b.isNull())
      return a.isNull() == b.isNull();
    return a.id() == b.id() && a.key() == b.key();
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

  static bool sameMuon(const reco::Muon& f, const pat::Muon& s) {
    return f.charge() == s.charge() && f.isGlobalMuon() == s.isGlobalMuon() &&
           f.isTrackerMuon() == s.isTrackerMuon() && f.isStandAloneMuon() == s.isStandAloneMuon() &&
           sameTrack(f.innerTrack(), s.innerTrack(), false) && sameTrack(f.outerTrack(), s.outerTrack(), true) &&
           sameTrack(f.globalTrack(), s.globalTrack(), false);
  }

  const edm::EDGetTokenT<std::vector<reco::Muon>> raw_, filtered_;
  const edm::EDGetTokenT<std::vector<pat::Muon>> slimmed_;
  const double minPtSTA_, minPtTK_, minMatches_;
  TTree* tree_;
  unsigned int run_ = 0, lumi_ = 0;
  unsigned long long event_ = 0;
  std::vector<int> filterCode_, rawToFiltered_;
  int nRaw_ = 0, nFiltered_ = 0, nSlimmed_ = 0, badFilter_ = 0, badSlimKey_ = 0, badSlimObj_ = 0;
};

DEFINE_FWK_MODULE(SlimmedDisplacedFilterCheck);
