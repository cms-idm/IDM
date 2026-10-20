// Hierarchy matching of the two accepted signal muons to pre-selected cleanedDisplacedMuons and
// KalmanVertexFitter(refit=true) of (a) the two matched legs (best-best) and (b) every pair of distinct
// pre-selected objects (Any pair). Study note: kb/studies/preselected-vertexing-efficiency.md.
// All original events are read; nothing is filtered. Components: 0 inner, 1 outer (embedded DSA), 2 global.
// Eligible component = passes its ID AND own track |eta|<2.4. A second hierarchy match without the
// reco-eta gate is stored only as a cross-check against the component-matching study.
// Positions are station-2 eta/phi, rounded to float before dR (as the stored component-study inputs).
#include <algorithm>
#include <array>
#include <cmath>
#include <set>
#include <string>
#include <tuple>
#include <utility>
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
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MuonAnalysis/MuonAssociators/interface/PropagateToMuonSetup.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TMath.h"
#include "TNamed.h"
#include "TTree.h"

namespace {
  constexpr double kMuonMass = 0.1056583755;
  enum Status { kNotFitted = 0, kShared = 1, kFailed = 2, kValid = 3 };
  // Unordered leg-type pair -> category: 0 GG, 1 II, 2 OO, 3 GO, 4 GI, 5 OI. Types: 0 I, 1 O, 2 G.
  int categoryOf(int t1, int t2) {
    const int lo = std::min(t1, t2), hi = std::max(t1, t2);
    if (lo == 2 && hi == 2) return 0;  // GG
    if (lo == 0 && hi == 0) return 1;  // II
    if (lo == 1 && hi == 1) return 2;  // OO
    if (lo == 1 && hi == 2) return 3;  // GO
    if (lo == 0 && hi == 2) return 4;  // GI
    return 5;                          // OI
  }

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
    return csc + dt > 12 && (csc > 0 || dt > 18) && t.normalizedChi2() < 2.5 && t.pt() > 0 &&
           t.ptError() / t.pt() < 1;
  }
  struct Pos {
    bool ok = false;
    float eta = -999.f, phi = -999.f;
  };
  Pos position(const TrajectoryStateOnSurface& state) {
    Pos r;
    if (!state.isValid()) return r;
    const auto p = state.globalPosition();
    if (!std::isfinite(p.eta()) || !std::isfinite(p.phi())) return r;
    r.ok = true;
    r.eta = static_cast<float>(p.eta());
    r.phi = static_cast<float>(p.phi());
    return r;
  }

  struct Gen {
    int index = -1, q = 0;
    float pt = 0, eta = 0, phi = 0, px = 0, py = 0, lxy = 0, d0 = 0;
    bool accepted = false;
    Pos st2;
  };
  struct Comp {
    int obj = -1, type = -1, q = 0;
    bool idPass = false, inEta = false;
    Pos st2;
    reco::TrackRef ref;
  };
  struct Match {
    std::array<int, 2> comp{{-1, -1}};  // index into the component vector
    std::array<int, 2> type{{-1, -1}};
    std::array<int, 2> obj{{-1, -1}};
    std::array<float, 2> dr{{-1.f, -1.f}};
  };

  // Staged global -> outer -> inner globally-nearest greedy one-to-one; a matched object is claimed whole.
  // Sorted by dR, then gen index, then original object index.
  Match hierarchy(const std::array<Gen, 2>& gen, const std::vector<Comp>& comps, bool requireEta) {
    Match m;
    std::set<int> claimed;
    for (const int type : {2, 1, 0}) {
      const double threshold = type == 1 ? 0.10 : 0.06;
      std::vector<std::tuple<double, int, int, int>> pairs;  // dR, genParticleIndex, obj, comp
      for (int g = 0; g < 2; ++g) {
        if (m.comp[g] >= 0 || !gen[g].st2.ok) continue;
        for (size_t c = 0; c < comps.size(); ++c) {
          const auto& k = comps[c];
          if (k.type != type || !k.idPass || (requireEta && !k.inEta) || !k.st2.ok || k.q != gen[g].q ||
              claimed.count(k.obj))
            continue;
          const double d = std::hypot(static_cast<double>(gen[g].st2.eta) - k.st2.eta,
                                      reco::deltaPhi(static_cast<double>(gen[g].st2.phi), static_cast<double>(k.st2.phi)));
          if (d < threshold) pairs.emplace_back(d, gen[g].index, k.obj, static_cast<int>(c));
        }
      }
      std::sort(pairs.begin(), pairs.end());
      for (const auto& [d, gi, obj, c] : pairs) {
        const int g = gen[0].index == gi ? 0 : 1;
        if (m.comp[g] >= 0 || claimed.count(obj)) continue;
        m.comp[g] = c;
        m.type[g] = type;
        m.obj[g] = obj;
        m.dr[g] = static_cast<float>(d);
        claimed.insert(obj);
      }
    }
    return m;
  }

  struct Fit {
    int status = kNotFitted;
    float vx = -999, vy = -999, vz = -999, chi2 = -999, ndof = -999, prob = -999, lxyBS = -999, mass = -999;
  };
  Fit fitLegs(const reco::TrackRef& a, const reco::TrackRef& b, const TransientTrackBuilder& ttb,
              const KalmanVertexFitter& kvf, const reco::BeamSpot& bs) {
    Fit f;
    std::vector<reco::TransientTrack> tts{ttb.build(a), ttb.build(b)};
    const TransientVertex tv = kvf.vertex(tts);
    if (!tv.isValid()) {
      f.status = kFailed;
      return f;
    }
    f.status = kValid;
    const reco::Vertex v(tv);
    f.vx = v.x();
    f.vy = v.y();
    f.vz = v.z();
    f.chi2 = v.chi2();
    f.ndof = v.ndof();
    f.prob = TMath::Prob(v.chi2(), static_cast<int>(std::lround(v.ndof())));
    f.lxyBS = std::hypot(v.x() - bs.x(v.z()), v.y() - bs.y(v.z()));
    auto p4 = [](const reco::Track& t) { return math::PtEtaPhiMLorentzVector(t.pt(), t.eta(), t.phi(), kMuonMass); };
    f.mass = (p4(*a) + p4(*b)).mass();
    return f;
  }

  // Track keys of all non-null components of an object, for the direct-sharing guard.
  std::set<std::pair<unsigned int, unsigned int>> keys(const reco::Muon& mu) {
    std::set<std::pair<unsigned int, unsigned int>> s;
    for (const auto& r : {mu.innerTrack(), mu.outerTrack(), mu.globalTrack()})
      if (r.isNonnull()) s.emplace(r.id().id(), r.key());
    return s;
  }
}  // namespace

class PreselectedVertexing : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit PreselectedVertexing(const edm::ParameterSet& ps)
      : gen_(consumes<std::vector<reco::GenParticle>>(ps.getParameter<edm::InputTag>("genParticles"))),
        muons_(consumes<std::vector<reco::Muon>>(ps.getParameter<edm::InputTag>("muons"))),
        indices_(consumes<std::vector<int>>(ps.getParameter<edm::InputTag>("originalIndex"))),
        beamspot_(consumes<reco::BeamSpot>(ps.getParameter<edm::InputTag>("beamspot"))),
        met_(consumes<std::vector<pat::MET>>(ps.getParameter<edm::InputTag>("met"))),
        genMet_(consumes<std::vector<reco::GenMET>>(ps.getParameter<edm::InputTag>("genMet"))),
        field_(esConsumes()),
        ttb_(esConsumes(edm::ESInputTag("", "TransientTrackBuilder"))),
        setup_(ps.getParameter<edm::ParameterSet>("propagatorSt2"), consumesCollector()) {
    usesResource(TFileService::kSharedResource);
    edm::Service<TFileService> fs;
    fs->make<TNamed>("sample", ps.getParameter<std::string>("sample").c_str());
    fs->make<TNamed>("revision", ps.getParameter<std::string>("revision").c_str());
    fs->make<TNamed>("definitions",
      "All original events. Signal gen: status1,lastCopy,abs(pdgId)13,firstDifferentMother abs1000023; accepted: "
      "abs(eta)<2.4,0<=Lxy<740,abs(vz)<960,no pT cut. twoAcc: exactly two signal muons, opposite charge, both accepted. "
      "Comp types 0 inner,1 outer(DSA),2 global; eligible = own ID and own |eta|<2.4. Hierarchy global,outer,inner, "
      "station-2 position eta/phi, dR<0.06 (I,G) <0.10 (O), same charge, whole-object claim, no fallback. "
      "Best-best = fit of the matched components' tracks; categories 0 GG,1 II,2 OO,3 GO,4 GI,5 OI. "
      "Pairs = all unordered pairs of distinct eligible-object best components (global>outer>inner) in twoAcc events. "
      "KalmanVertexFitter(refit), status 0 not fitted,1 shared,2 failed,3 valid. *NoEta = cross-check without reco-eta gate. "
      "Reco dR/pT from track reference-point momenta. dPhi = phi(MET)-phi(mumu) wrapped.");
    evt_ = fs->make<TTree>("events", "One row per event");
    pair_ = fs->make<TTree>("pairs", "Any-pair fits in twoAcc events; event = entry in events");
#define E(x) evt_->Branch(#x, &x##_)
    E(run); E(lumi); E(event); E(nSig); E(twoAcc); E(nObj); E(nElig);
    E(sigIndex); E(sigQ); E(sigAcc); E(sigPt); E(sigEta); E(sigPhi); E(sigLxy); E(sigD0); E(sigProp); E(sigEta2); E(sigPhi2);
    E(cObj); E(cType); E(cQ); E(cIdPass); E(cInEta); E(cProp); E(cEta2); E(cPhi2);
    E(matchComp); E(matchType); E(matchObj); E(matchDR);
    E(matchCompNoEta); E(matchTypeNoEta); E(matchObjNoEta);
    E(genDR); E(genMuMuPt); E(genMuMuPhi); E(genMetPt); E(genMetPhi); E(pfMetPt); E(pfMetPhi);
    E(bbStatus); E(bbCat); E(bbChi2); E(bbNdof); E(bbProb); E(bbVx); E(bbVy); E(bbVz); E(bbLxyBS); E(bbMass);
    E(bbRecoDR); E(bbRecoMuMuPt); E(bbRecoMuMuPhi);
    E(anyNPairs); E(anyNValid);
#undef E
#define P(x) pair_->Branch(#x, &x##_)
    P(pEvent); P(pObjA); P(pObjB); P(pTypeA); P(pTypeB); P(pCat); P(pIsBB); P(pStatus); P(pChi2); P(pNdof); P(pProb);
    P(pVx); P(pVy); P(pVz); P(pLxyBS); P(pMass); P(pRecoDR); P(pRecoMuMuPt); P(pRecoMuMuPhi);
#undef P
  }

  void analyze(const edm::Event& ev, const edm::EventSetup& es) override {
    reset();
    run_ = ev.id().run();
    lumi_ = ev.luminosityBlock();
    event_ = ev.id().event();
    const auto propagator = setup_.init(es);
    const auto* field = &es.getData(field_);
    const auto& ttb = es.getData(ttb_);
    const auto& bs = ev.get(beamspot_);

    // Signal gen: all signal muons are stored; acceptance is a flag.
    std::vector<Gen> sig;
    for (const auto& p : ev.get(gen_)) {
      const int i = static_cast<int>(&p - &ev.get(gen_)[0]);
      if (std::abs(p.pdgId()) != 13 || p.status() != 1 || !p.isLastCopy()) continue;
      const auto* mom = firstDifferentMother(p);
      if (!mom || std::abs(mom->pdgId()) != 1000023) continue;
      Gen g;
      g.index = i;
      g.q = p.charge();
      g.pt = p.pt(); g.eta = p.eta(); g.phi = p.phi(); g.px = p.px(); g.py = p.py();
      g.lxy = p.vertex().rho();
      g.d0 = std::abs(p.vx() * std::sin(p.phi()) - p.vy() * std::cos(p.phi()));
      g.accepted = std::abs(p.eta()) < 2.4 && g.lxy >= 0 && g.lxy < 740 && std::abs(p.vz()) < 960;
      const FreeTrajectoryState start(GlobalPoint(p.vx(), p.vy(), p.vz()), GlobalVector(p.px(), p.py(), p.pz()),
                                      TrackCharge(p.charge()), field);
      g.st2 = position(propagator.extrapolate(start));
      sig.push_back(g);
    }
    nSig_ = sig.size();
    for (const auto& g : sig) {
      sigIndex_.push_back(g.index); sigQ_.push_back(g.q); sigAcc_.push_back(g.accepted); sigPt_.push_back(g.pt);
      sigEta_.push_back(g.eta); sigPhi_.push_back(g.phi); sigLxy_.push_back(g.lxy); sigD0_.push_back(g.d0);
      sigProp_.push_back(g.st2.ok); sigEta2_.push_back(g.st2.eta); sigPhi2_.push_back(g.st2.phi);
    }
    twoAcc_ = sig.size() == 2 && sig[0].q * sig[1].q < 0 && sig[0].accepted && sig[1].accepted;

    // Components of cleaned objects (all components that exist; ID/eta flags stored; station 2 only if ID passes).
    const auto& muons = ev.get(muons_);
    const auto& indices = ev.get(indices_);
    if (indices.size() != muons.size()) throw cms::Exception("PreselectedVertexing") << "Cleaner index size";
    std::vector<Comp> comps;
    std::vector<int> objBest(muons.size(), -1);  // best eligible component per cleaned-object position (global>outer>inner)
    for (size_t i = 0; i < muons.size(); ++i) {
      const auto& mu = muons[i];
      const std::array<reco::TrackRef, 3> refs{{mu.innerTrack(), mu.outerTrack(), mu.globalTrack()}};
      for (int t = 0; t < 3; ++t) {
        if (refs[t].isNull()) continue;
        if (!refs[t].isAvailable()) throw cms::Exception("PreselectedVertexing") << "Unavailable TrackRef";
        Comp c;
        c.obj = indices[i];
        c.type = t;
        c.q = refs[t]->charge();
        c.ref = refs[t];
        c.idPass = t == 0 ? mu.isTrackerMuon() : t == 1 ? outerID(*refs[t]) : mu.isGlobalMuon();
        const double eta = refs[t]->eta();
        c.inEta = std::isfinite(eta) && std::abs(eta) < 2.4;
        if (c.idPass) c.st2 = position(propagator.extrapolate(*refs[t]));
        comps.push_back(c);
      }
    }
    for (const auto& c : comps) {
      cObj_.push_back(c.obj); cType_.push_back(c.type); cQ_.push_back(c.q); cIdPass_.push_back(c.idPass);
      cInEta_.push_back(c.inEta); cProp_.push_back(c.st2.ok); cEta2_.push_back(c.st2.eta); cPhi2_.push_back(c.st2.phi);
    }
    std::set<int> eligibleObjs;
    for (size_t c = 0; c < comps.size(); ++c)
      if (comps[c].idPass && comps[c].inEta) {
        eligibleObjs.insert(comps[c].obj);
        const int pos = static_cast<int>(std::lower_bound(indices.begin(), indices.end(), comps[c].obj) - indices.begin());
        if (objBest[pos] < 0 || comps[c].type > comps[objBest[pos]].type) objBest[pos] = static_cast<int>(c);
      }
    nObj_ = static_cast<int>(indices.size());
    nElig_ = static_cast<int>(eligibleObjs.size());

    const auto& mets = ev.get(met_);
    if (!mets.empty()) {
      pfMetPt_ = mets[0].corPt(pat::MET::Type1);
      pfMetPhi_ = mets[0].corPhi(pat::MET::Type1);
    }
    const auto& gmets = ev.get(genMet_);
    if (!gmets.empty()) {
      genMetPt_ = gmets[0].pt();
      genMetPhi_ = gmets[0].phi();
    }

    KalmanVertexFitter kvf(true);
    int eventRow = tree_entries_;
    if (twoAcc_) {
      const std::array<Gen, 2> gen{{sig[0], sig[1]}};
      const double sx = gen[0].px + gen[1].px, sy = gen[0].py + gen[1].py;
      genMuMuPt_ = std::hypot(sx, sy);
      genMuMuPhi_ = std::atan2(sy, sx);
      genDR_ = reco::deltaR(gen[0].eta, gen[0].phi, gen[1].eta, gen[1].phi);

      const Match m = hierarchy(gen, comps, true);
      const Match mn = hierarchy(gen, comps, false);
      for (int g = 0; g < 2; ++g) {
        matchComp_.push_back(m.comp[g]); matchType_.push_back(m.type[g]); matchObj_.push_back(m.obj[g]);
        matchDR_.push_back(m.dr[g]);
        matchCompNoEta_.push_back(mn.comp[g]); matchTypeNoEta_.push_back(mn.type[g]);
        matchObjNoEta_.push_back(mn.obj[g]);
      }
      const bool both = m.comp[0] >= 0 && m.comp[1] >= 0;
      std::array<int, 2> bbPos{{-1, -1}};  // best-best legs by component index
      if (both) {
        bbPos = m.comp;
        bbCat_ = categoryOf(m.type[0], m.type[1]);
        const auto& a = comps[m.comp[0]];
        const auto& b = comps[m.comp[1]];
        const auto ka = keys(muons[std::lower_bound(indices.begin(), indices.end(), a.obj) - indices.begin()]);
        const auto kb = keys(muons[std::lower_bound(indices.begin(), indices.end(), b.obj) - indices.begin()]);
        bool shared = false;
        for (const auto& k : ka) shared |= kb.count(k) > 0;
        Fit f;
        if (shared)
          f.status = kShared;
        else
          f = fitLegs(a.ref, b.ref, ttb, kvf, bs);
        bbStatus_ = f.status;
        bbChi2_ = f.chi2; bbNdof_ = f.ndof; bbProb_ = f.prob; bbVx_ = f.vx; bbVy_ = f.vy; bbVz_ = f.vz;
        bbLxyBS_ = f.lxyBS; bbMass_ = f.mass;
        bbRecoDR_ = reco::deltaR(a.ref->eta(), a.ref->phi(), b.ref->eta(), b.ref->phi());
        const double rx = a.ref->px() + b.ref->px(), ry = a.ref->py() + b.ref->py();
        bbRecoMuMuPt_ = std::hypot(rx, ry);
        bbRecoMuMuPhi_ = std::atan2(ry, rx);
      }

      // Any pairs: distinct eligible objects, each at its best component.
      std::vector<int> best;
      for (size_t i = 0; i < objBest.size(); ++i)
        if (objBest[i] >= 0) best.push_back(objBest[i]);
      for (size_t i = 0; i < best.size(); ++i)
        for (size_t j = i + 1; j < best.size(); ++j) {
          const auto& a = comps[best[i]];
          const auto& b = comps[best[j]];
          const auto ka = keys(muons[std::lower_bound(indices.begin(), indices.end(), a.obj) - indices.begin()]);
          const auto kb = keys(muons[std::lower_bound(indices.begin(), indices.end(), b.obj) - indices.begin()]);
          bool shared = false;
          for (const auto& k : ka) shared |= kb.count(k) > 0;
          Fit f;
          if (shared)
            f.status = kShared;
          else
            f = fitLegs(a.ref, b.ref, ttb, kvf, bs);
          ++anyNPairs_;
          if (f.status == kValid) ++anyNValid_;
          const double rx = a.ref->px() + b.ref->px(), ry = a.ref->py() + b.ref->py();
          pEvent_ = eventRow; pObjA_ = a.obj; pObjB_ = b.obj; pTypeA_ = a.type; pTypeB_ = b.type;
          pCat_ = categoryOf(a.type, b.type);
          pIsBB_ = both && ((a.obj == comps[bbPos[0]].obj && b.obj == comps[bbPos[1]].obj) ||
                            (a.obj == comps[bbPos[1]].obj && b.obj == comps[bbPos[0]].obj));  // same OBJECTS
          pStatus_ = f.status; pChi2_ = f.chi2; pNdof_ = f.ndof; pProb_ = f.prob; pVx_ = f.vx; pVy_ = f.vy;
          pVz_ = f.vz; pLxyBS_ = f.lxyBS; pMass_ = f.mass;
          pRecoDR_ = reco::deltaR(a.ref->eta(), a.ref->phi(), b.ref->eta(), b.ref->phi());
          pRecoMuMuPt_ = std::hypot(rx, ry);
          pRecoMuMuPhi_ = std::atan2(ry, rx);
          pair_->Fill();
        }
    }
    evt_->Fill();
    ++tree_entries_;
  }

private:
  void reset() {
    nSig_ = 0; twoAcc_ = false; nObj_ = nElig_ = 0;
    for (auto* v : {&sigIndex_, &sigQ_, &sigAcc_, &sigProp_, &cObj_, &cType_, &cQ_, &cIdPass_, &cInEta_, &cProp_,
                    &matchComp_, &matchType_, &matchObj_, &matchCompNoEta_, &matchTypeNoEta_, &matchObjNoEta_})
      v->clear();
    for (auto* v : {&sigPt_, &sigEta_, &sigPhi_, &sigLxy_, &sigD0_, &sigEta2_, &sigPhi2_, &cEta2_, &cPhi2_, &matchDR_})
      v->clear();
    genDR_ = genMuMuPt_ = genMuMuPhi_ = genMetPt_ = genMetPhi_ = pfMetPt_ = pfMetPhi_ = -999.f;
    bbStatus_ = kNotFitted; bbCat_ = -1;
    bbChi2_ = bbNdof_ = bbProb_ = bbVx_ = bbVy_ = bbVz_ = bbLxyBS_ = bbMass_ = bbRecoDR_ = bbRecoMuMuPt_ =
        bbRecoMuMuPhi_ = -999.f;
    anyNPairs_ = anyNValid_ = 0;
  }

  const edm::EDGetTokenT<std::vector<reco::GenParticle>> gen_;
  const edm::EDGetTokenT<std::vector<reco::Muon>> muons_;
  const edm::EDGetTokenT<std::vector<int>> indices_;
  const edm::EDGetTokenT<reco::BeamSpot> beamspot_;
  const edm::EDGetTokenT<std::vector<pat::MET>> met_;
  const edm::EDGetTokenT<std::vector<reco::GenMET>> genMet_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> field_;
  const edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> ttb_;
  const PropagateToMuonSetup setup_;
  TTree *evt_ = nullptr, *pair_ = nullptr;
  int tree_entries_ = 0;

  unsigned int run_ = 0, lumi_ = 0;
  unsigned long long event_ = 0;
  int nSig_ = 0, nObj_ = 0, nElig_ = 0;
  bool twoAcc_ = false;
  std::vector<int> sigIndex_, sigQ_, sigAcc_, sigProp_, cObj_, cType_, cQ_, cIdPass_, cInEta_, cProp_;
  std::vector<float> sigPt_, sigEta_, sigPhi_, sigLxy_, sigD0_, sigEta2_, sigPhi2_, cEta2_, cPhi2_;
  std::vector<int> matchComp_, matchType_, matchObj_, matchCompNoEta_, matchTypeNoEta_, matchObjNoEta_;
  std::vector<float> matchDR_;
  float genDR_ = -999, genMuMuPt_ = -999, genMuMuPhi_ = -999, genMetPt_ = -999, genMetPhi_ = -999,
        pfMetPt_ = -999, pfMetPhi_ = -999;
  int bbStatus_ = 0, bbCat_ = -1, anyNPairs_ = 0, anyNValid_ = 0;
  float bbChi2_ = -999, bbNdof_ = -999, bbProb_ = -999, bbVx_ = -999, bbVy_ = -999, bbVz_ = -999,
        bbLxyBS_ = -999, bbMass_ = -999, bbRecoDR_ = -999, bbRecoMuMuPt_ = -999, bbRecoMuMuPhi_ = -999;
  int pEvent_ = 0, pObjA_ = 0, pObjB_ = 0, pTypeA_ = 0, pTypeB_ = 0, pCat_ = 0, pStatus_ = 0;
  bool pIsBB_ = false;
  float pChi2_ = 0, pNdof_ = 0, pProb_ = 0, pVx_ = 0, pVy_ = 0, pVz_ = 0, pLxyBS_ = 0, pMass_ = 0, pRecoDR_ = 0,
        pRecoMuMuPt_ = 0, pRecoMuMuPhi_ = 0;
};
DEFINE_FWK_MODULE(PreselectedVertexing);
