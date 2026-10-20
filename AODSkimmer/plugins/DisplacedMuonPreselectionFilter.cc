// Event skim after DisplacedMuonDuplicateCleaner. All flag vectors are aligned to the ORIGINAL
// displacedMuons collection, including dropped duplicates and objects failing every ID.
// Missing components: Present=0, PassID=0, InEta=0. Non-null unavailable refs throw, not silently fail.
// preselected = keptAfterCleaning && any component PassID (no eta or pT selection).
// preselectedEta additionally requires that SAME passing component to have finite |track eta| < 2.4.
#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace {
  // Existing POG DSA ID (ElectronSkimmer::passesDisplacedID); pt > 0 is a division guard,
  // as in displaced_muon_id.C, not an analysis pT threshold.
  bool passesDisplacedID(const reco::Track& track) {
    const auto& hits = track.hitPattern();
    const int csc = hits.numberOfValidMuonCSCHits();
    const int dt = hits.numberOfValidMuonDTHits();
    return csc + dt > 12 && (csc > 0 || dt > 18) && track.normalizedChi2() < 2.5 &&
           track.pt() > 0 && track.ptError() / track.pt() < 1;
  }
  const std::array<std::string, 12> flagNames{{
      "innerPresent", "innerPassID", "innerInEta", "outerPresent", "outerPassID", "outerInEta",
      "globalPresent", "globalPassID", "globalInEta", "keptAfterCleaning", "preselected", "preselectedEta"}};
}

class DisplacedMuonPreselectionFilter : public edm::global::EDFilter<> {
public:
  explicit DisplacedMuonPreselectionFilter(const edm::ParameterSet& ps)
      : muons_(consumes<std::vector<reco::Muon>>(ps.getParameter<edm::InputTag>("muons"))),
        indices_(consumes<std::vector<int>>(ps.getParameter<edm::InputTag>("originalIndex"))) {
    for (size_t f = 0; f < flagNames.size(); ++f)
      flags_[f] = produces<std::vector<int>>(flagNames[f]);
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("muons", edm::InputTag("displacedMuons", "", "RECO"));
    desc.add<edm::InputTag>("originalIndex", edm::InputTag("cleanedDisplacedMuons", "originalIndex"));
    descriptions.add("displacedMuonPreselectionFilter", desc);
  }

  bool filter(edm::StreamID, edm::Event& event, const edm::EventSetup&) const override {
    const auto& muons = event.get(muons_);
    const auto& indices = event.get(indices_);
    std::array<std::vector<int>, 12> values;
    for (auto& value : values)
      value.assign(muons.size(), 0);
    int previous = -1;
    for (const int index : indices) {
      if (index <= previous || index < 0 || static_cast<size_t>(index) >= muons.size())
        throw cms::Exception("DisplacedMuonPreselectionFilter") << "Invalid cleaner originalIndex " << index;
      values[9][index] = 1;
      previous = index;
    }

    bool accept = false;
    for (size_t i = 0; i < muons.size(); ++i) {
      const auto& muon = muons[i];
      const std::array<reco::TrackRef, 3> refs{{muon.innerTrack(), muon.outerTrack(), muon.globalTrack()}};
      bool anyID = false, anyIDEta = false;
      for (size_t c = 0; c < refs.size(); ++c) {
        if (refs[c].isNull())
          continue;
        values[3 * c][i] = 1;
        if (!refs[c].isAvailable())
          throw cms::Exception("DisplacedMuonPreselectionFilter")
              << "Unavailable component " << c << " of original displacedMuon " << i;
        const auto& track = *refs[c];
        const bool passID = c == 0 ? muon.isTrackerMuon() : c == 1 ? passesDisplacedID(track) : muon.isGlobalMuon();
        const bool inEta = std::isfinite(track.eta()) && std::abs(track.eta()) < 2.4;
        values[3 * c + 1][i] = passID;
        values[3 * c + 2][i] = inEta;
        anyID |= passID;
        anyIDEta |= passID && inEta;
      }
      values[10][i] = values[9][i] && anyID;
      values[11][i] = values[9][i] && anyIDEta;
      accept |= values[10][i];
    }
    for (size_t f = 0; f < flags_.size(); ++f)
      event.emplace(flags_[f], std::move(values[f]));
    return accept;
  }

private:
  const edm::EDGetTokenT<std::vector<reco::Muon>> muons_;
  const edm::EDGetTokenT<std::vector<int>> indices_;
  std::array<edm::EDPutTokenT<std::vector<int>>, 12> flags_;
};

DEFINE_FWK_MODULE(DisplacedMuonPreselectionFilter);
