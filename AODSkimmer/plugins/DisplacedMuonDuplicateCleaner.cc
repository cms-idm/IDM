// Deterministic duplicate removal for displacedMuons (kb/DECISIONS.md 2026-10-19; design in
// kb/studies/displaced-muon-duplicate-cleaner.md). Two muons are duplicates if they share an inner,
// outer or global TrackRef directly (ProductID + key; no transitive grouping). Rank: non-null
// globalTrack() first, then non-null outerTrack(), then lower original index. Visit muons best rank
// first and keep one unless it shares a track with a muon already kept.
// Products: the kept muons as copies in ascending original index (TrackRefs unchanged), their original
// indices (instance "originalIndex") and the raw collection size (instance "nRaw").
#include <algorithm>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace {
  using RefKey = std::pair<edm::ProductID, unsigned int>;

  // Non-null component refs of a muon, compared by ProductID + key only (never dereferenced).
  std::vector<RefKey> componentKeys(const reco::Muon& mu) {
    std::vector<RefKey> keys;
    for (const reco::TrackRef& ref : {mu.innerTrack(), mu.outerTrack(), mu.globalTrack()})
      if (ref.isNonnull())
        keys.emplace_back(ref.id(), static_cast<unsigned int>(ref.key()));
    return keys;
  }

  bool share(const std::vector<RefKey>& a, const std::vector<RefKey>& b) {
    for (const auto& ka : a)
      if (std::find(b.begin(), b.end(), ka) != b.end())
        return true;
    return false;
  }
}  // namespace

class DisplacedMuonDuplicateCleaner : public edm::global::EDProducer<> {
public:
  explicit DisplacedMuonDuplicateCleaner(const edm::ParameterSet& ps)
      : muonToken_(consumes<std::vector<reco::Muon>>(ps.getParameter<edm::InputTag>("muons"))),
        validate_(ps.getParameter<bool>("validate")),
        muonPut_(produces<std::vector<reco::Muon>>()),
        indexPut_(produces<std::vector<int>>("originalIndex")),
        nRawPut_(produces<int>("nRaw")) {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("muons", edm::InputTag("displacedMuons", "", "RECO"));
    desc.add<bool>("validate", false);  // re-check the invariants per event and throw on failure
    descriptions.add("displacedMuonDuplicateCleaner", desc);
  }

  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const override {
    const auto& muons = iEvent.get(muonToken_);
    const size_t n = muons.size();

    std::vector<std::vector<RefKey>> keys(n);
    std::vector<int> rank(n);  // 2 = global, 1 = outer only, 0 = neither
    for (size_t i = 0; i < n; ++i) {
      keys[i] = componentKeys(muons[i]);
      rank[i] = muons[i].globalTrack().isNonnull() ? 2 : (muons[i].outerTrack().isNonnull() ? 1 : 0);
    }

    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      return rank[a] != rank[b] ? rank[a] > rank[b] : a < b;
    });

    std::vector<bool> kept(n, false);
    std::vector<size_t> keptSoFar;
    for (size_t i : order) {
      bool dup = false;
      for (size_t k : keptSoFar)
        if (share(keys[i], keys[k])) {
          dup = true;
          break;
        }
      if (!dup) {
        kept[i] = true;
        keptSoFar.push_back(i);
      }
    }

    if (validate_)
      check(keys, rank, kept);

    auto outMuons = std::make_unique<std::vector<reco::Muon>>();
    auto outIndex = std::make_unique<std::vector<int>>();
    for (size_t i = 0; i < n; ++i)
      if (kept[i]) {
        outMuons->push_back(muons[i]);
        outIndex->push_back(static_cast<int>(i));
      }
    iEvent.put(muonPut_, std::move(outMuons));
    iEvent.put(indexPut_, std::move(outIndex));
    iEvent.emplace(nRawPut_, static_cast<int>(n));
  }

private:
  // Invariants: no two kept muons share a track; every dropped muon shares a track with a kept muon of
  // better rank (higher type rank, or equal type rank and lower index).
  static void check(const std::vector<std::vector<RefKey>>& keys,
                    const std::vector<int>& rank,
                    const std::vector<bool>& kept) {
    const size_t n = keys.size();
    auto better = [&](size_t a, size_t b) { return rank[a] != rank[b] ? rank[a] > rank[b] : a < b; };
    for (size_t i = 0; i < n; ++i) {
      bool justified = kept[i];
      for (size_t j = 0; j < n; ++j) {
        if (j == i || !kept[j] || !share(keys[i], keys[j]))
          continue;
        if (kept[i])
          throw cms::Exception("DisplacedMuonDuplicateCleaner")
              << "kept muons " << i << " and " << j << " share a track";
        if (better(j, i))
          justified = true;
      }
      if (!justified)
        throw cms::Exception("DisplacedMuonDuplicateCleaner")
            << "muon " << i << " dropped without a better-ranked kept sharer";
    }
  }

  const edm::EDGetTokenT<std::vector<reco::Muon>> muonToken_;
  const bool validate_;
  const edm::EDPutTokenT<std::vector<reco::Muon>> muonPut_;
  const edm::EDPutTokenT<std::vector<int>> indexPut_;
  const edm::EDPutTokenT<int> nRawPut_;
};

DEFINE_FWK_MODULE(DisplacedMuonDuplicateCleaner);
