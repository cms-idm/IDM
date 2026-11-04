// recoOnly mode: all events, fixed eligible components, OS pairs, no fit fallback; original-track PCA observables.
// Hierarchy matching of the two accepted signal muons to pre-selected cleanedDisplacedMuons and
// KalmanVertexFitter(refit=true) of (a) the two matched legs (best-best), (b) the hierarchy fallback (hf*: other matched
// components of the same two objects, tried in hierarchy order until a fit is valid) and (c) every pair of distinct
// pre-selected objects (Any pair). The hierarchy order is the hierarchyOrder parameter (types, best first).
// Study notes: kb/studies/preselected-vertexing-efficiency.md, kb/studies/met-proxy-residual.md (hf*, pfMu*, chi*),
// kb/studies/vertex-fit-field.md (sigV*, sigB*, bbSigmaLxy, bbB*), kb/studies/displaced-pf-muon-overlap.md (leg*; Part 2: hfLeg*, pfMuPt/St2Ok).
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
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MuonAnalysis/MuonAssociators/interface/PropagateToMuonSetup.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
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
    float pt = 0, eta = 0, phi = 0, px = 0, py = 0, lxy = 0, d0 = 0, vx = 0, vy = 0, vz = 0, bx = 0, by = 0, bz = 0;
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

  // Staged (types in the given order, best first) globally-nearest greedy one-to-one; a matched object is claimed whole.
  // Sorted by dR, then gen index, then original object index.
  Match hierarchy(const std::array<Gen, 2>& gen, const std::vector<Comp>& comps, bool requireEta,
                  const std::array<int, 3>& order) {
    Match m;
    std::set<int> claimed;
    for (const int type : order) {
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

  // Same-charge station-2 dR match of one component to one gen muon, with the type's threshold (as hierarchy()).
  bool matchesGen(const Gen& g, const Comp& k) {
    if (!g.st2.ok || !k.st2.ok || k.q != g.q) return false;
    const double d = std::hypot(static_cast<double>(g.st2.eta) - k.st2.eta,
                                reco::deltaPhi(static_cast<double>(g.st2.phi), static_cast<double>(k.st2.phi)));
    return d < (k.type == 1 ? 0.10 : 0.06);
  }

  // Independent per-type matching (no cross-type reservation): one-to-one within the type, objects used at most once.
  std::array<int, 2> independentMatch(const std::array<Gen, 2>& gen, const std::vector<Comp>& comps, int type,
                                      bool requireEta) {
    const double threshold = type == 1 ? 0.10 : 0.06;
    std::vector<std::tuple<double, int, int, int>> pairs;  // dR, genParticleIndex, obj, comp
    for (int g = 0; g < 2; ++g) {
      if (!gen[g].st2.ok) continue;
      for (size_t c = 0; c < comps.size(); ++c) {
        const auto& k = comps[c];
        if (k.type != type || !k.idPass || (requireEta && !k.inEta) || !k.st2.ok || k.q != gen[g].q) continue;
        const double d = std::hypot(static_cast<double>(gen[g].st2.eta) - k.st2.eta,
                                    reco::deltaPhi(static_cast<double>(gen[g].st2.phi), static_cast<double>(k.st2.phi)));
        if (d < threshold) pairs.emplace_back(d, gen[g].index, k.obj, static_cast<int>(c));
      }
    }
    std::sort(pairs.begin(), pairs.end());
    std::array<int, 2> m{{-1, -1}};
    std::set<int> used;
    for (const auto& [d, gi, obj, c] : pairs) {
      const int g = gen[0].index == gi ? 0 : 1;
      if (m[g] >= 0 || used.count(obj)) continue;
      m[g] = c;
      used.insert(obj);
    }
    return m;
  }

  struct Fit {
    int status = kNotFitted;
    float vx = -999, vy = -999, vz = -999, chi2 = -999, ndof = -999, prob = -999, lxyBS = -999, mass = -999;
    float sigmaLxy = -999, bx = -999, by = -999, bz = -999;
  };
  // sigma(Lxy): full 3D propagation of the vertex covariance, beamspot fixed; Lxy = |v - bs(vz)| in xy, so
  // g = (dx, dy, -dxdz*dx - dydz*dy)/L. -999 if L = 0, the covariance is non-finite or the variance is not positive.
  float sigmaLxyBS(const reco::Vertex& v, const reco::BeamSpot& bs) {
    const double dx = v.x() - bs.x(v.z()), dy = v.y() - bs.y(v.z()), l = std::hypot(dx, dy);
    if (!(l > 0)) return -999.f;
    const std::array<double, 3> g{{dx / l, dy / l, -(bs.dxdz() * dx + bs.dydz() * dy) / l}};
    double var = 0;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) var += g[i] * v.covariance(i, j) * g[j];
    if (!std::isfinite(var) || !(var > 0)) return -999.f;
    return static_cast<float>(std::sqrt(var));
  }
  Fit fitLegs(const reco::TrackRef& a, const reco::TrackRef& b, const TransientTrackBuilder& ttb,
              const KalmanVertexFitter& kvf, const reco::BeamSpot& bs, const MagneticField& field) {
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
    f.sigmaLxy = sigmaLxyBS(v, bs);
    const auto bf = field.inTesla(GlobalPoint(v.x(), v.y(), v.z()));
    f.bx = bf.x();
    f.by = bf.y();
    f.bz = bf.z();
    auto p4 = [](const reco::Track& t) { return math::PtEtaPhiMLorentzVector(t.pt(), t.eta(), t.phi(), kMuonMass); };
    f.mass = (p4(*a) + p4(*b)).mass();
    return f;
  }

  // DT/CSC segment fingerprints of an object's ID chamber matches (refs are not stored in MINIAOD): chamber rawId,
  // hasPhi/hasZed and local x, y. Non-finite x/y dropped (counted); de-duplicated; arb keeps only segments with
  // BestInChamberByDR and BelongsToTrackByDR.
  struct Seg {
    unsigned int id = 0;
    bool phi = false, zed = false;
    float x = 0, y = 0;
    bool same(const Seg& o) const {
      return id == o.id && phi == o.phi && zed == o.zed && std::abs(x - o.x) < 1e-4f && std::abs(y - o.y) < 1e-4f;
    }
  };
  std::vector<Seg> segments(const reco::Muon& mu, bool arb, int& dropped) {
    std::vector<Seg> out;
    for (const auto& ch : mu.matches()) {
      if (ch.detector() != MuonSubdetId::DT && ch.detector() != MuonSubdetId::CSC) continue;
      for (const auto& m : ch.segmentMatches) {
        if (arb && !(m.isMask(reco::MuonSegmentMatch::BestInChamberByDR) &&
                     m.isMask(reco::MuonSegmentMatch::BelongsToTrackByDR)))
          continue;
        if (!std::isfinite(m.x) || !std::isfinite(m.y)) {
          ++dropped;
          continue;
        }
        Seg g;
        g.id = ch.id.rawId(); g.phi = m.hasPhi(); g.zed = m.hasZed(); g.x = m.x; g.y = m.y;
        if (std::none_of(out.begin(), out.end(), [&](const Seg& o) { return o.same(g); })) out.push_back(g);
      }
    }
    return out;
  }
  // Fraction of the reference segments whose fingerprint appears in the other list; -1 if the reference list is empty.
  double sharedFraction(const std::vector<Seg>& ref, const std::vector<Seg>& other) {
    if (ref.empty()) return -1;
    int n = 0;
    for (const auto& r : ref) n += std::any_of(other.begin(), other.end(), [&](const Seg& o) { return o.same(r); });
    return static_cast<double>(n) / ref.size();
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
        puppiMet_(consumes<std::vector<pat::MET>>(ps.getParameter<edm::InputTag>("puppiMet"))),
        genMet_(consumes<std::vector<reco::GenMET>>(ps.getParameter<edm::InputTag>("genMet"))),
        packed_(consumes<std::vector<pat::PackedCandidate>>(ps.getParameter<edm::InputTag>("packedCandidates"))),
        slimmed_(consumes<std::vector<pat::Muon>>(ps.getUntrackedParameter<edm::InputTag>("slimmedMuons", edm::InputTag("slimmedMuons")))),
        field_(esConsumes()),
        ttb_(esConsumes(edm::ESInputTag("", "TransientTrackBuilder"))),
        vertexPropagator_(esConsumes(edm::ESInputTag("", "SteppingHelixPropagatorAny"))),
        recoOnly_(ps.getUntrackedParameter<bool>("recoOnly", false)),
        vertexDirections_(ps.getUntrackedParameter<bool>("vertexDirections", false)),
        pfNearest_(ps.getUntrackedParameter<bool>("pfNearest", false)),
        pfStation2_(ps.getUntrackedParameter<bool>("pfStation2", false)),
        pfHitMatch_(ps.getUntrackedParameter<bool>("pfHitMatch", false)),
        setup_(ps.getParameter<edm::ParameterSet>("propagatorSt2"), consumesCollector()) {
    usesResource(TFileService::kSharedResource);
    const auto order = ps.getParameter<std::vector<int>>("hierarchyOrder");
    if (order.size() != 3 || std::set<int>(order.begin(), order.end()) != std::set<int>{0, 1, 2})
      throw cms::Exception("PreselectedVertexing") << "hierarchyOrder must be a permutation of 0 1 2";
    for (int r = 0; r < 3; ++r) {
      order_[r] = order[r];
      rank_[order[r]] = r;
    }
    edm::Service<TFileService> fs;
    fs->make<TNamed>("sample", ps.getParameter<std::string>("sample").c_str());
    fs->make<TNamed>("revision", ps.getParameter<std::string>("revision").c_str());
    fs->make<TNamed>("recoOnly", recoOnly_ ? "all events; fixed eligible G>I>DSA legs; OS before fit; no fallback; PCA momenta" : "disabled");
    fs->make<TNamed>("vertexDirections", vertexDirections_ ?
      "Original hf tracks, no vertex constraint; SteppingHelixPropagatorAny to 3D PCA to fitted vertex. "
      "Vectors in gen-index order. hfVertexValid per leg: 0 failure,1 valid; empty if no valid hf fit. "
      "hfVertexStatus = SteppingHelixStateInfo::Result; hfVertexPath signed cm; hfVertexDistance 3D miss distance cm. "
      "hfRefEta/Phi at stored original-track reference; hfVertexEta/Phi/Pt from propagated momentum; -999 on failure. "
      "Material/field settings are those of SteppingHelixPropagatorAny loaded by the cfg. No rematching or fallback on propagation failure."
      : "disabled");
    fs->make<TNamed>("pfNearest", pfNearest_ ?
      "twoAcc events, vectors in gen-index order. leg = hierarchy-matched component (matchComp) track at its reference "
      "point: legPt/Eta/Phi/Q, -999/0 if unmatched. Nearest packedPFCandidates abs(pdgId)13 by dR(track eta/phi, "
      "candidate eta/phi), no cut: legPfIdx (event-local packed index, -1 if unmatched leg or no PF muon), legPfDR/Pt/Eta/Phi "
      "(-999), legPfQ (0); legPfN02 = PF muons with dR<0.2 (-1 if unmatched leg). Non-finite dR skipped."
      : "disabled");
    fs->make<TNamed>("pfHitMatch", pfHitMatch_ ?
      "Valid hf fit only, vectors in gen-index order over the two fitted hf legs. PF muons = slimmedMuons isPFMuon (index "
      "into slimmedMuons); pfSlimN, pfSlimLinked = packed abs(pdgId)13 candidates with exactly one unused slimmed PF partner "
      "(dR<1e-3, |dpT|<=1e-3 pT+0.02, same charge). Leg hit = innerPosition() of the leg component track (hfLegHitOk 0 if the "
      "extra is unavailable or non-finite; hfLegHitRho/Z). PF point: slimmed inner track initialFreeState, "
      "SteppingHelixPropagatorAny to 3D PCA to the hit; dR of the two positions' eta/phi. Nearest (ties: lower index): "
      "hfLegHitPfIdx (-1), hfLegHitDR/PfPt (-999), hfLegHitPfQ (0), hfLegHitStatus/Path/Miss of that propagation; "
      "hfLegHitN01/N02 = PF muons with dR<0.1/<0.2; hfLegHitNFail = PF muons without inner track or failed propagation. "
      "Segments: DT/CSC segmentMatches of the leg's displacedMuons object and of each PF muon, fingerprint rawId, "
      "hasPhi/hasZed, |dx|,|dy|<1e-4 cm, non-finite dropped (segDropped), de-duplicated. hfLegNSeg; best PF by shared "
      "fraction, then smaller hit dR (unavailable last), then lower index: hfLegSegPfIdx (-1), hfLegSegFrac (-1), "
      "hfLegSegPfPt/PfDR (-999), hfLegSegPfQ (0). Arbitrated (BestInChamberByDR and BelongsToTrackByDR, both lists): "
      "hfLegNSegArb, hfLegSegFracArb."
      : "disabled");
    fs->make<TNamed>("pfStation2", pfStation2_ ?
      "Valid hf fit only, vectors in gen-index order over the two fitted hf legs. hfLegPt/Q at the track reference point; "
      "hfLegEta2/Phi2 = stored station-2 position (propagatorSt2). PF muons (packedPFCandidates abs(pdgId)13): "
      "FreeTrajectoryState(vertex, momentum, charge) through propagatorSt2; pfMuPt/pfMuSt2Ok per PF muon (event order). "
      "Nearest by station-2 position dR among propagated PF muons: hfLegPfIdx (event-local packed index, -1 if none), "
      "hfLegPfDR/Pt (-999), hfLegPfQ (0), hfLegPfN02 = propagated PF muons with dR<0.2. Reference-point nearest over all "
      "PF muons (track vs candidate eta/phi, as pfNearest): hfLegPfRefIdx/DR/Pt/Q. Non-finite dR skipped."
      : "disabled");
    const char* typeName[3] = {"inner", "outer(DSA)", "global"};
    fs->make<TNamed>("hierarchyOrder", (std::string(typeName[order_[0]]) + ">" + typeName[order_[1]] + ">" +
                                        typeName[order_[2]]).c_str());
    fs->make<TNamed>("definitions",
      "All original events. Signal gen: status1,lastCopy,abs(pdgId)13,firstDifferentMother abs1000023; accepted: "
      "abs(eta)<2.4,0<=Lxy<740,abs(vz)<960,no pT cut. twoAcc: exactly two signal muons, opposite charge, both accepted. "
      "Comp types 0 inner,1 outer(DSA),2 global; eligible = own ID and own |eta|<2.4. Hierarchy order = TNamed hierarchyOrder (matching stages and per-object best component), "
      "station-2 position eta/phi, dR<0.06 (I,G) <0.10 (O), same charge, whole-object claim, no fallback. "
      "Best-best = fit of the hierarchy-matched components' tracks (one set). Typed combos: independent per-type matching, ordered (gen0 type a, gen1 type b), types 0 I,1 O,2 G; "
      "Pairs = all unordered pairs of distinct eligible-object best components in configured hierarchy order. Legacy mode: twoAcc only. recoOnly mode: all events, fixed component charges, OS before fit, no fit fallback; legacy best-best/typed/hf fits disabled. pIsBB/pMatchedLegs label objects via staged truth matching independently of reco selection; the labeling component may differ from the fitted component. pPcaOK/DR/MuMuPt/MuMuPhi use original-track momentum at 3D PCA to the fitted vertex; -999 on failure. "
      "pPcaPt/Eta/PhiA/B: the same per-leg momenta at the PCA, -999 unless pPcaOK. pRecoPt/Eta/PhiA/B: per-leg paired component "
      "track at its reference point (as pRecoDR/MuMuPt), every pair. "
      "KalmanVertexFitter(refit), status 0 not fitted,1 shared,2 failed,3 valid. *NoEta = cross-check without reco-eta gate. "
      "Reco dR/pT from track reference-point momenta. dPhi = phi(MET)-phi(mumu) wrapped. "
      "puppiMet* = slimmedMETsPuppi[0] corPt/corPhi(Type1); -999 if the collection is empty. "
      "hf* = hierarchy fallback: legs = eligible components of the two best-best matched objects that each pass the same-charge "
      "station-2 dR match to that gen; pairs ordered by (better leg rank, worse leg rank, gen0 leg rank), fitted in order until "
      "one is valid; hf fields describe the first valid pair, else the first pair; hfTypeA/B = gen0/gen1 leg types. "
      "pfMu* = packedPFCandidates abs(pdgId)13 (all), count and vector px/py sum. chi* = prunedGenParticles abs(pdgId)1000022, "
      "status 1, lastCopy; chiOK = exactly two with first different-PDG parents abs 32 and 1000023; chiPx/Py vector sum. "
      "sigV* = gen muon production vertex [cm]; sigB*, bbB* = MagneticField::inTesla at the gen / bb fitted vertex [T]; "
      "bbSigmaLxy = sqrt(g^T C g), C = fitted vertex 3x3 covariance, g = (dx,dy,-dxdz*dx-dydz*dy)/Lxy, dx,dy = vertex - "
      "beamspot(vz), beamspot fixed; -999 if Lxy=0 or the variance is non-finite or not positive (valid fits only).");
    evt_ = fs->make<TTree>("events", "One row per event");
    pair_ = fs->make<TTree>("pairs", "Any-pair fits; recoOnly includes all events; pEvent = entry in events");
    combo_ = fs->make<TTree>("combos", "Typed ordered fits (gen0 type a, gen1 type b); one row per existing combo; event = entry in events");
#define E(x) evt_->Branch(#x, &x##_)
    E(run); E(lumi); E(event); E(nSig); E(twoAcc); E(nObj); E(nElig);
    E(sigIndex); E(sigQ); E(sigAcc); E(sigPt); E(sigEta); E(sigPhi); E(sigLxy); E(sigD0); E(sigProp); E(sigEta2); E(sigPhi2);
    E(sigVx); E(sigVy); E(sigVz); E(sigBx); E(sigBy); E(sigBz);
    E(cObj); E(cType); E(cQ); E(cIdPass); E(cInEta); E(cProp); E(cEta2); E(cPhi2);
    E(matchComp); E(matchType); E(matchObj); E(matchDR);
    E(matchCompNoEta); E(matchTypeNoEta); E(matchObjNoEta);
    E(indObj); E(indObjNoEta);
    E(genDR); E(genMuMuPt); E(genMuMuPhi); E(genMetPt); E(genMetPhi); E(pfMetPt); E(pfMetPhi); E(puppiMetPt); E(puppiMetPhi);
    E(bbStatus); E(bbChi2); E(bbNdof); E(bbProb); E(bbVx); E(bbVy); E(bbVz); E(bbLxyBS); E(bbMass);
    E(bbRecoDR); E(bbRecoMuMuPt); E(bbRecoMuMuPhi); E(bbSigmaLxy); E(bbBx); E(bbBy); E(bbBz);
    E(hfN); E(hfTried); E(hfFirstIsBB); E(hfFirstStatus); E(hfStatus); E(hfTypeA); E(hfTypeB); E(hfChi2); E(hfNdof);
    E(hfProb); E(hfVx); E(hfVy); E(hfVz); E(hfLxyBS); E(hfMass); E(hfRecoDR); E(hfRecoMuMuPt); E(hfRecoMuMuPhi);
    E(hfVertexValid); E(hfVertexStatus); E(hfVertexPath); E(hfVertexDistance);
    E(hfRefEta); E(hfRefPhi); E(hfVertexEta); E(hfVertexPhi); E(hfVertexPt);
    E(pfMuN); E(pfMuPx); E(pfMuPy); E(nChi); E(chiOK); E(chiPx); E(chiPy);
    E(legPt); E(legEta); E(legPhi); E(legQ); E(legPfN02); E(legPfIdx); E(legPfDR); E(legPfPt); E(legPfEta);
    E(legPfPhi); E(legPfQ);
    E(hfLegPt); E(hfLegQ); E(hfLegEta2); E(hfLegPhi2); E(hfLegPfIdx); E(hfLegPfDR); E(hfLegPfN02); E(hfLegPfPt);
    E(hfLegPfQ); E(hfLegPfRefIdx); E(hfLegPfRefDR); E(hfLegPfRefPt); E(hfLegPfRefQ); E(pfMuPt); E(pfMuSt2Ok);
    E(pfSlimN); E(pfSlimLinked); E(segDropped); E(hfLegHitOk); E(hfLegHitRho); E(hfLegHitZ); E(hfLegHitPfIdx);
    E(hfLegHitDR); E(hfLegHitPfPt); E(hfLegHitPfQ); E(hfLegHitStatus); E(hfLegHitPath); E(hfLegHitMiss); E(hfLegHitN01);
    E(hfLegHitN02); E(hfLegHitNFail); E(hfLegNSeg); E(hfLegSegPfIdx); E(hfLegSegFrac); E(hfLegSegPfPt); E(hfLegSegPfQ);
    E(hfLegSegPfDR); E(hfLegNSegArb); E(hfLegSegFracArb);
    E(anyNPairs); E(anyNValid);
#undef E
#define P(x) pair_->Branch(#x, &x##_)
    P(pEvent); P(pObjA); P(pObjB); P(pTypeA); P(pTypeB); P(pCat); P(pIsBB); P(pStatus); P(pChi2); P(pNdof); P(pProb);
    P(pQA); P(pQB); P(pMatchedLegs); P(pPcaOK); P(pPcaDR); P(pPcaMuMuPt); P(pPcaMuMuPhi);
    P(pVx); P(pVy); P(pVz); P(pLxyBS); P(pMass); P(pRecoDR); P(pRecoMuMuPt); P(pRecoMuMuPhi);
    P(pPcaPtA); P(pPcaEtaA); P(pPcaPhiA); P(pPcaPtB); P(pPcaEtaB); P(pPcaPhiB);
    P(pRecoPtA); P(pRecoEtaA); P(pRecoPhiA); P(pRecoPtB); P(pRecoEtaB); P(pRecoPhiB);
#undef P
#define K(x) combo_->Branch(#x, &x##_)
    K(kEvent); K(kTypeA); K(kTypeB); K(kStatus); K(kChi2); K(kNdof); K(kProb); K(kVx); K(kVy); K(kVz); K(kLxyBS);
    K(kMass); K(kRecoDR); K(kRecoMuMuPt); K(kRecoMuMuPhi);
#undef K
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
      g.vx = p.vx(); g.vy = p.vy(); g.vz = p.vz();
      const auto gb = field->inTesla(GlobalPoint(p.vx(), p.vy(), p.vz()));
      g.bx = gb.x(); g.by = gb.y(); g.bz = gb.z();
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
      sigVx_.push_back(g.vx); sigVy_.push_back(g.vy); sigVz_.push_back(g.vz);
      sigBx_.push_back(g.bx); sigBy_.push_back(g.by); sigBz_.push_back(g.bz);
    }
    twoAcc_ = sig.size() == 2 && sig[0].q * sig[1].q < 0 && sig[0].accepted && sig[1].accepted;

    // Components of cleaned objects (all components that exist; ID/eta flags stored; station 2 only if ID passes).
    const auto& muons = ev.get(muons_);
    const auto& indices = ev.get(indices_);
    if (indices.size() != muons.size()) throw cms::Exception("PreselectedVertexing") << "Cleaner index size";
    std::vector<Comp> comps;
    std::vector<int> objBest(muons.size(), -1);  // best eligible component per cleaned-object position (hierarchy order)
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
        if (objBest[pos] < 0 || rank_[comps[c].type] < rank_[comps[objBest[pos]].type]) objBest[pos] = static_cast<int>(c);
      }
    nObj_ = static_cast<int>(indices.size());
    nElig_ = static_cast<int>(eligibleObjs.size());

    const auto& mets = ev.get(met_);
    if (!mets.empty()) {
      pfMetPt_ = mets[0].corPt(pat::MET::Type1);
      pfMetPhi_ = mets[0].corPhi(pat::MET::Type1);
    }
    const auto& puppiMets = ev.get(puppiMet_);
    if (!puppiMets.empty()) {
      puppiMetPt_ = puppiMets[0].corPt(pat::MET::Type1);
      puppiMetPhi_ = puppiMets[0].corPhi(pat::MET::Type1);
    }
    const auto& gmets = ev.get(genMet_);
    if (!gmets.empty()) {
      genMetPt_ = gmets[0].pt();
      genMetPhi_ = gmets[0].phi();
    }
    const auto& packed = ev.get(packed_);
    std::vector<int> pfMuIdx;
    for (const auto& c : packed) {
      if (std::abs(c.pdgId()) != 13) continue;
      pfMuIdx.push_back(static_cast<int>(&c - &packed[0]));
      ++pfMuN_;
      pfMuPx_ += c.px();
      pfMuPy_ += c.py();
    }
    {
      std::vector<const reco::GenParticle*> chis;
      for (const auto& p : ev.get(gen_))
        if (std::abs(p.pdgId()) == 1000022 && p.status() == 1 && p.isLastCopy()) chis.push_back(&p);
      nChi_ = static_cast<int>(chis.size());
      if (chis.size() == 2) {
        std::array<int, 2> par{{0, 0}};
        for (int i = 0; i < 2; ++i) {
          const auto* mom = firstDifferentMother(*chis[i]);
          par[i] = mom ? std::abs(mom->pdgId()) : 0;
        }
        std::sort(par.begin(), par.end());
        chiOK_ = par[0] == 32 && par[1] == 1000023;
        if (chiOK_) {
          chiPx_ = chis[0]->px() + chis[1]->px();
          chiPy_ = chis[0]->py() + chis[1]->py();
        }
      }
    }

    KalmanVertexFitter kvf(true);
    int eventRow = tree_entries_;
    // Truth labels never control reco-only component selection or pair enumeration.
    Match labels;
    if (sig.size() == 2 && sig[0].q * sig[1].q < 0)
      labels = hierarchy(std::array<Gen, 2>{{sig[0], sig[1]}}, comps, true, order_);
    if (twoAcc_ && !recoOnly_) {
      const std::array<Gen, 2> gen{{sig[0], sig[1]}};
      const double sx = gen[0].px + gen[1].px, sy = gen[0].py + gen[1].py;
      genMuMuPt_ = std::hypot(sx, sy);
      genMuMuPhi_ = std::atan2(sy, sx);
      genDR_ = reco::deltaR(gen[0].eta, gen[0].phi, gen[1].eta, gen[1].phi);

      const Match m = hierarchy(gen, comps, true, order_);
      const Match mn = hierarchy(gen, comps, false, order_);
      for (int g = 0; g < 2; ++g) {
        matchComp_.push_back(m.comp[g]); matchType_.push_back(m.type[g]); matchObj_.push_back(m.obj[g]);
        matchDR_.push_back(m.dr[g]);
        matchCompNoEta_.push_back(mn.comp[g]); matchTypeNoEta_.push_back(mn.type[g]);
        matchObjNoEta_.push_back(mn.obj[g]);
      }
      // Nearest PF muon to each hierarchy-matched leg (track direction at its reference point).
      for (int g = 0; pfNearest_ && g < 2; ++g) {
        int n02 = -1, idx = -1, q = 0, pfQ = 0;
        float pt = -999, eta = -999, phi = -999, pfPt = -999, pfEta = -999, pfPhi = -999;
        double dr = -999;  // double, as the 0.2 cut, so the stored dR and legPfN02 agree
        if (m.comp[g] >= 0) {
          const auto& t = *comps[m.comp[g]].ref;
          pt = t.pt(); eta = t.eta(); phi = t.phi(); q = t.charge();
          n02 = 0;
          for (const int i : pfMuIdx) {
            const double d = reco::deltaR(t.eta(), t.phi(), packed[i].eta(), packed[i].phi());
            if (!std::isfinite(d)) continue;
            if (d < 0.2) ++n02;
            if (idx < 0 || d < dr) {
              idx = i; dr = d; pfPt = packed[i].pt(); pfEta = packed[i].eta(); pfPhi = packed[i].phi();
              pfQ = packed[i].charge();
            }
          }
        }
        legPt_.push_back(pt); legEta_.push_back(eta); legPhi_.push_back(phi); legQ_.push_back(q);
        legPfN02_.push_back(n02); legPfIdx_.push_back(idx); legPfDR_.push_back(dr); legPfPt_.push_back(pfPt);
        legPfEta_.push_back(pfEta); legPfPhi_.push_back(pfPhi); legPfQ_.push_back(pfQ);
      }
      // Independent per-type matches (gated and, for validation, ungated), objects reported by original index.
      std::array<std::array<int, 2>, 3> ind, indn;
      for (int t = 0; t < 3; ++t) {
        ind[t] = independentMatch(gen, comps, t, true);
        indn[t] = independentMatch(gen, comps, t, false);
        for (int g = 0; g < 2; ++g) {
          indObj_.push_back(ind[t][g] >= 0 ? comps[ind[t][g]].obj : -1);
          indObjNoEta_.push_back(indn[t][g] >= 0 ? comps[indn[t][g]].obj : -1);
        }
      }
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
          if (ind[a][0] < 0 || ind[b][1] < 0) continue;
          const auto& ca = comps[ind[a][0]];
          const auto& cb = comps[ind[b][1]];
          const auto ka = keys(muons[std::lower_bound(indices.begin(), indices.end(), ca.obj) - indices.begin()]);
          const auto kb = keys(muons[std::lower_bound(indices.begin(), indices.end(), cb.obj) - indices.begin()]);
          bool shared = ca.obj == cb.obj;
          for (const auto& k : ka) shared |= kb.count(k) > 0;
          Fit f;
          if (shared)
            f.status = kShared;
          else
            f = fitLegs(ca.ref, cb.ref, ttb, kvf, bs, *field);
          const double rx = ca.ref->px() + cb.ref->px(), ry = ca.ref->py() + cb.ref->py();
          kEvent_ = eventRow; kTypeA_ = a; kTypeB_ = b; kStatus_ = f.status; kChi2_ = f.chi2; kNdof_ = f.ndof;
          kProb_ = f.prob; kVx_ = f.vx; kVy_ = f.vy; kVz_ = f.vz; kLxyBS_ = f.lxyBS; kMass_ = f.mass;
          kRecoDR_ = reco::deltaR(ca.ref->eta(), ca.ref->phi(), cb.ref->eta(), cb.ref->phi());
          kRecoMuMuPt_ = std::hypot(rx, ry);
          kRecoMuMuPhi_ = std::atan2(ry, rx);
          combo_->Fill();
        }
      const bool both = m.comp[0] >= 0 && m.comp[1] >= 0;
      if (both) {
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
          f = fitLegs(a.ref, b.ref, ttb, kvf, bs, *field);
        bbStatus_ = f.status;
        bbChi2_ = f.chi2; bbNdof_ = f.ndof; bbProb_ = f.prob; bbVx_ = f.vx; bbVy_ = f.vy; bbVz_ = f.vz;
        bbLxyBS_ = f.lxyBS; bbMass_ = f.mass;
        bbSigmaLxy_ = f.sigmaLxy; bbBx_ = f.bx; bbBy_ = f.by; bbBz_ = f.bz;
        bbRecoDR_ = reco::deltaR(a.ref->eta(), a.ref->phi(), b.ref->eta(), b.ref->phi());
        const double rx = a.ref->px() + b.ref->px(), ry = a.ref->py() + b.ref->py();
        bbRecoMuMuPt_ = std::hypot(rx, ry);
        bbRecoMuMuPhi_ = std::atan2(ry, rx);

        // Hierarchy fallback: same two objects, other components that are themselves matched to the same gen muon.
        std::array<std::vector<int>, 2> legs;
        for (int g = 0; g < 2; ++g)
          for (size_t c = 0; c < comps.size(); ++c) {
            const auto& k = comps[c];
            if (k.obj == m.obj[g] && k.idPass && k.inEta && matchesGen(gen[g], k)) legs[g].push_back(static_cast<int>(c));
          }
        std::vector<std::tuple<int, int, int, int, int>> cand;  // better rank, worse rank, gen0 rank, comp0, comp1
        for (const int c0 : legs[0])
          for (const int c1 : legs[1]) {
            const int r0 = rank_[comps[c0].type], r1 = rank_[comps[c1].type];
            cand.emplace_back(std::min(r0, r1), std::max(r0, r1), r0, c0, c1);
          }
        std::sort(cand.begin(), cand.end());
        hfN_ = static_cast<int>(cand.size());
        hfFirstIsBB_ = !cand.empty() && std::get<3>(cand[0]) == m.comp[0] && std::get<4>(cand[0]) == m.comp[1];
        for (size_t i = 0; i < cand.size(); ++i) {
          const int c0 = std::get<3>(cand[i]), c1 = std::get<4>(cand[i]);
          Fit h;
          if (c0 == m.comp[0] && c1 == m.comp[1]) {  // the best-best fit above
            h.status = bbStatus_; h.chi2 = bbChi2_; h.ndof = bbNdof_; h.prob = bbProb_; h.vx = bbVx_; h.vy = bbVy_;
            h.vz = bbVz_; h.lxyBS = bbLxyBS_; h.mass = bbMass_;
          } else if (shared) {  // sharing is a property of the two objects, the same for every pair
            h.status = kShared;
          } else {
            h = fitLegs(comps[c0].ref, comps[c1].ref, ttb, kvf, bs, *field);
          }
          ++hfTried_;
          if (i == 0) hfFirstStatus_ = h.status;
          if (i == 0 || h.status == kValid) {
            const auto& ra = comps[c0].ref;
            const auto& rb = comps[c1].ref;
            hfStatus_ = h.status; hfTypeA_ = comps[c0].type; hfTypeB_ = comps[c1].type;
            hfChi2_ = h.chi2; hfNdof_ = h.ndof; hfProb_ = h.prob; hfVx_ = h.vx; hfVy_ = h.vy; hfVz_ = h.vz;
            hfLxyBS_ = h.lxyBS; hfMass_ = h.mass;
            hfRecoDR_ = reco::deltaR(ra->eta(), ra->phi(), rb->eta(), rb->phi());
            const double hx = ra->px() + rb->px(), hy = ra->py() + rb->py();
            hfRecoMuMuPt_ = std::hypot(hx, hy);
            hfRecoMuMuPhi_ = std::atan2(hy, hx);
            if (h.status == kValid && vertexDirections_) {
              const auto* prop = dynamic_cast<const SteppingHelixPropagator*>(&es.getData(vertexPropagator_));
              if (!prop) throw cms::Exception("PreselectedVertexing") << "Expected SteppingHelixPropagatorAny";
              const GlobalPoint target(h.vx, h.vy, h.vz);
              for (const auto& ref : {ra, rb}) {
                hfRefEta_.push_back(ref->eta());
                hfRefPhi_.push_back(ref->phi());
                const auto start = trajectoryStateTransform::initialFreeState(*ref, field);
                SteppingHelixStateInfo end;
                prop->propagate(SteppingHelixStateInfo(start), target, end);
                const auto momentum = end.isValid() ? end.momentum() : GlobalVector();
                const bool ok = end.isValid() && std::isfinite(momentum.eta()) && std::isfinite(momentum.phi()) &&
                                momentum.perp() > 0 && std::isfinite(end.path()) &&
                                std::isfinite((end.position() - target).mag());
                hfVertexValid_.push_back(ok);
                hfVertexStatus_.push_back(static_cast<int>(end.status()));
                hfVertexPath_.push_back(ok ? end.path() : -999.f);
                hfVertexDistance_.push_back(ok ? (end.position() - target).mag() : -999.f);
                hfVertexEta_.push_back(ok ? momentum.eta() : -999.f);
                hfVertexPhi_.push_back(ok ? static_cast<float>(momentum.phi()) : -999.f);
                hfVertexPt_.push_back(ok ? momentum.perp() : -999.f);
              }
            }
            // Nearest PF muon to each hf leg by station-2 position dR, plus the reference-point nearest (as pfNearest).
            if (h.status == kValid && pfStation2_) {
              std::vector<Pos> pfSt2;
              for (const int k : pfMuIdx) {
                const auto& p = packed[k];
                const FreeTrajectoryState start(GlobalPoint(p.vx(), p.vy(), p.vz()),
                                                GlobalVector(p.px(), p.py(), p.pz()), TrackCharge(p.charge()), field);
                pfSt2.push_back(position(propagator.extrapolate(start)));
                pfMuPt_.push_back(p.pt());
                pfMuSt2Ok_.push_back(pfSt2.back().ok);
              }
              for (const int c : {c0, c1}) {
                const auto& k = comps[c];
                const auto& t = *k.ref;
                int n02 = 0, idx = -1, q = 0, refIdx = -1, refQ = 0;
                float pt = -999, refPt = -999;
                double dr = -999, refDr = -999;  // double, as the 0.2 cut
                for (size_t j = 0; j < pfMuIdx.size(); ++j) {
                  const auto& p = packed[pfMuIdx[j]];
                  const double d0 = reco::deltaR(t.eta(), t.phi(), p.eta(), p.phi());
                  if (std::isfinite(d0) && (refIdx < 0 || d0 < refDr)) {
                    refIdx = pfMuIdx[j]; refDr = d0; refQ = p.charge(); refPt = p.pt();
                  }
                  if (!pfSt2[j].ok) continue;
                  const double d = std::hypot(static_cast<double>(k.st2.eta) - pfSt2[j].eta,
                                              reco::deltaPhi(static_cast<double>(k.st2.phi), static_cast<double>(pfSt2[j].phi)));
                  if (!std::isfinite(d)) continue;
                  if (d < 0.2) ++n02;
                  if (idx < 0 || d < dr) {
                    idx = pfMuIdx[j]; dr = d; q = p.charge(); pt = p.pt();
                  }
                }
                hfLegPt_.push_back(t.pt()); hfLegQ_.push_back(t.charge());
                hfLegEta2_.push_back(k.st2.eta); hfLegPhi2_.push_back(k.st2.phi);
                hfLegPfIdx_.push_back(idx); hfLegPfDR_.push_back(dr); hfLegPfN02_.push_back(n02);
                hfLegPfPt_.push_back(pt); hfLegPfQ_.push_back(q);
                hfLegPfRefIdx_.push_back(refIdx); hfLegPfRefDR_.push_back(refDr); hfLegPfRefPt_.push_back(refPt);
                hfLegPfRefQ_.push_back(refQ);
              }
            }
            // AN-style association of each hf leg with slimmedMuons PF muons: proximity at the leg's innermost hit and
            // shared segment fingerprints.
            if (h.status == kValid && pfHitMatch_) {
              const auto* prop = dynamic_cast<const SteppingHelixPropagator*>(&es.getData(vertexPropagator_));
              if (!prop) throw cms::Exception("PreselectedVertexing") << "Expected SteppingHelixPropagatorAny";
              const auto& slim = ev.get(slimmed_);
              std::vector<int> pfs;
              for (size_t j = 0; j < slim.size(); ++j)
                if (slim[j].isPFMuon()) pfs.push_back(static_cast<int>(j));
              pfSlimN_ = static_cast<int>(pfs.size());
              std::vector<bool> used(pfs.size(), false);
              for (const int k : pfMuIdx) {
                const auto& p = packed[k];
                int n = 0, first = -1;
                for (size_t j = 0; j < pfs.size(); ++j) {
                  const auto& m = slim[pfs[j]];
                  if (!used[j] && reco::deltaR(p.eta(), p.phi(), m.eta(), m.phi()) < 1e-3 &&
                      std::abs(p.pt() - m.pt()) <= 1e-3 * m.pt() + 0.02 && p.charge() == m.charge()) {
                    ++n;
                    if (first < 0) first = static_cast<int>(j);
                  }
                }
                if (n == 1) {
                  ++pfSlimLinked_;
                  used[first] = true;
                }
              }
              std::vector<std::vector<Seg>> pfSeg, pfSegArb;
              for (const int j : pfs) {
                pfSeg.push_back(segments(slim[j], false, segDropped_));
                int arbDropped = 0;  // the arbitrated list is a subset: its drops are already counted
                pfSegArb.push_back(segments(slim[j], true, arbDropped));
              }
              for (const int c : {c0, c1}) {
                const auto& k = comps[c];
                const auto& t = *k.ref;
                const auto& obj = muons[std::lower_bound(indices.begin(), indices.end(), k.obj) - indices.begin()];
                bool hitOk = false;
                GlobalPoint hit;
                if (t.extra().isNonnull() && t.extra().isAvailable()) {
                  const auto& ip = t.innerPosition();
                  hit = GlobalPoint(ip.x(), ip.y(), ip.z());
                  hitOk = std::isfinite(hit.eta()) && std::isfinite(hit.phi()) && std::isfinite(hit.z());
                }
                std::vector<double> dHit(pfs.size(), -1);  // -1: unavailable
                int idx = -1, n01 = 0, n02 = 0, nFail = 0, status = -1;
                float path = -999, miss = -999;
                for (size_t j = 0; hitOk && j < pfs.size(); ++j) {
                  const auto& in = slim[pfs[j]].innerTrack();
                  if (in.isNull() || !in.isAvailable()) {
                    ++nFail;
                    continue;
                  }
                  SteppingHelixStateInfo end;
                  prop->propagate(SteppingHelixStateInfo(trajectoryStateTransform::initialFreeState(*in, field)), hit, end);
                  const auto pos = end.isValid() ? end.position() : GlobalPoint();
                  const double d = end.isValid() ? std::hypot(static_cast<double>(hit.eta()) - pos.eta(),
                                                              reco::deltaPhi(static_cast<double>(hit.phi()), static_cast<double>(pos.phi())))
                                                 : -1;
                  if (!end.isValid() || !std::isfinite(pos.eta()) || !std::isfinite(pos.phi()) || !std::isfinite(d)) {
                    ++nFail;
                    continue;
                  }
                  dHit[j] = d;
                  if (d < 0.1) ++n01;
                  if (d < 0.2) ++n02;
                  if (idx < 0 || d < dHit[idx]) {
                    idx = static_cast<int>(j);
                    status = static_cast<int>(end.status());
                    path = end.path();
                    miss = (pos - hit).mag();
                  }
                }
                int segDummy = 0;
                const auto legSeg = segments(obj, false, segDropped_), legSegArb = segments(obj, true, segDummy);
                int best = -1;
                double bestFrac = -1, bestFracArb = -1;
                for (size_t j = 0; j < pfs.size(); ++j) {
                  const double f = sharedFraction(legSeg, pfSeg[j]);
                  const auto better = [&](size_t a, size_t b) {  // a beats current best b: smaller dR, unavailable last
                    if (dHit[a] < 0) return false;
                    return dHit[b] < 0 || dHit[a] < dHit[b];
                  };
                  if (f >= 0 && (best < 0 || f > bestFrac || (f == bestFrac && better(j, best)))) {
                    best = static_cast<int>(j);
                    bestFrac = f;
                  }
                  bestFracArb = std::max(bestFracArb, sharedFraction(legSegArb, pfSegArb[j]));
                }
                hfLegHitOk_.push_back(hitOk);
                hfLegHitRho_.push_back(hitOk ? hit.perp() : -999.f);
                hfLegHitZ_.push_back(hitOk ? hit.z() : -999.f);
                hfLegHitPfIdx_.push_back(idx >= 0 ? pfs[idx] : -1);
                hfLegHitDR_.push_back(idx >= 0 ? dHit[idx] : -999.);
                hfLegHitPfPt_.push_back(idx >= 0 ? slim[pfs[idx]].pt() : -999.f);
                hfLegHitPfQ_.push_back(idx >= 0 ? slim[pfs[idx]].charge() : 0);
                hfLegHitStatus_.push_back(status); hfLegHitPath_.push_back(path); hfLegHitMiss_.push_back(miss);
                hfLegHitN01_.push_back(hitOk ? n01 : -1); hfLegHitN02_.push_back(hitOk ? n02 : -1);
                hfLegHitNFail_.push_back(hitOk ? nFail : -1);
                hfLegNSeg_.push_back(static_cast<int>(legSeg.size()));
                hfLegSegPfIdx_.push_back(best >= 0 ? pfs[best] : -1);
                hfLegSegFrac_.push_back(bestFrac);
                hfLegSegPfPt_.push_back(best >= 0 ? slim[pfs[best]].pt() : -999.f);
                hfLegSegPfQ_.push_back(best >= 0 ? slim[pfs[best]].charge() : 0);
                hfLegSegPfDR_.push_back(best >= 0 && dHit[best] >= 0 ? dHit[best] : -999.);
                hfLegNSegArb_.push_back(static_cast<int>(legSegArb.size()));
                hfLegSegFracArb_.push_back(bestFracArb);
              }
            }
          }
          if (h.status == kValid || h.status == kShared) break;
        }
      }

    }
    if (twoAcc_ || recoOnly_) {
      // All events in recoOnly mode; no gen-dependent pair selection or component retry.
      // Legacy mode retains its original truth-accepted Any-pair population.
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
          else if (!recoOnly_ || a.q * b.q < 0)
            f = fitLegs(a.ref, b.ref, ttb, kvf, bs, *field);
          ++anyNPairs_;
          if (f.status == kValid) ++anyNValid_;
          const double rx = a.ref->px() + b.ref->px(), ry = a.ref->py() + b.ref->py();
          pEvent_ = eventRow; pObjA_ = a.obj; pObjB_ = b.obj; pTypeA_ = a.type; pTypeB_ = b.type;
          pCat_ = categoryOf(a.type, b.type);
          pIsBB_ = labels.obj[0] >= 0 && labels.obj[1] >= 0 &&
            ((a.obj == labels.obj[0] && b.obj == labels.obj[1]) ||
             (a.obj == labels.obj[1] && b.obj == labels.obj[0]));  // same OBJECTS
          pQA_ = a.q; pQB_ = b.q;
          pMatchedLegs_ = (a.obj == labels.obj[0] || a.obj == labels.obj[1]) +
                          (b.obj == labels.obj[0] || b.obj == labels.obj[1]);
          pPcaOK_ = false;
          pPcaDR_ = pPcaMuMuPt_ = pPcaMuMuPhi_ = -999.f;
          pPcaPtA_ = pPcaEtaA_ = pPcaPhiA_ = pPcaPtB_ = pPcaEtaB_ = pPcaPhiB_ = -999.f;  // every pair; set only if pPcaOK
          if (recoOnly_ && f.status == kValid) {
            const auto* prop = dynamic_cast<const SteppingHelixPropagator*>(&es.getData(vertexPropagator_));
            if (!prop) throw cms::Exception("PreselectedVertexing") << "Expected SteppingHelixPropagatorAny";
            const GlobalPoint target(f.vx, f.vy, f.vz);
            std::array<GlobalVector, 2> momenta;
            bool ok = true;
            int leg = 0;
            for (const auto& ref : {a.ref, b.ref}) {
              const auto start = trajectoryStateTransform::initialFreeState(*ref, field);
              SteppingHelixStateInfo end;
              prop->propagate(SteppingHelixStateInfo(start), target, end);
              if (!end.isValid()) { ok = false; break; }
              const auto mom = end.momentum();
              if (!std::isfinite(mom.eta()) || !std::isfinite(mom.phi()) || !(mom.perp() > 0) ||
                  !std::isfinite(end.path()) || !std::isfinite((end.position() - target).mag())) {
                ok = false; break;
              }
              momenta[leg++] = mom;
            }
            if (ok) {
              const auto sum = momenta[0] + momenta[1];
              pPcaDR_ = reco::deltaR(momenta[0].eta(), momenta[0].phi(), momenta[1].eta(), momenta[1].phi());
              pPcaMuMuPt_ = sum.perp();
              pPcaMuMuPhi_ = sum.phi();
              pPcaOK_ = std::isfinite(pPcaDR_) && std::isfinite(pPcaMuMuPt_) && std::isfinite(pPcaMuMuPhi_);
              if (pPcaOK_) {
                pPcaPtA_ = momenta[0].perp(); pPcaEtaA_ = momenta[0].eta(); pPcaPhiA_ = momenta[0].phi();
                pPcaPtB_ = momenta[1].perp(); pPcaEtaB_ = momenta[1].eta(); pPcaPhiB_ = momenta[1].phi();
              }
            }
          }
          pStatus_ = f.status; pChi2_ = f.chi2; pNdof_ = f.ndof; pProb_ = f.prob; pVx_ = f.vx; pVy_ = f.vy;
          pVz_ = f.vz; pLxyBS_ = f.lxyBS; pMass_ = f.mass;
          pRecoDR_ = reco::deltaR(a.ref->eta(), a.ref->phi(), b.ref->eta(), b.ref->phi());
          pRecoMuMuPt_ = std::hypot(rx, ry);
          pRecoMuMuPhi_ = std::atan2(ry, rx);
          pRecoPtA_ = a.ref->pt(); pRecoEtaA_ = a.ref->eta(); pRecoPhiA_ = a.ref->phi();
          pRecoPtB_ = b.ref->pt(); pRecoEtaB_ = b.ref->eta(); pRecoPhiB_ = b.ref->phi();
          pair_->Fill();
        }
    }
    evt_->Fill();
    ++tree_entries_;
  }

private:
  void reset() {
    hfVertexValid_.clear(); hfVertexStatus_.clear();
    for (auto* v : {&legQ_, &legPfN02_, &legPfIdx_, &legPfQ_}) v->clear();
    for (auto* v : {&legPt_, &legEta_, &legPhi_, &legPfPt_, &legPfEta_, &legPfPhi_}) v->clear();
    legPfDR_.clear();
    for (auto* v : {&hfLegQ_, &hfLegPfIdx_, &hfLegPfN02_, &hfLegPfQ_, &hfLegPfRefIdx_, &hfLegPfRefQ_, &pfMuSt2Ok_}) v->clear();
    for (auto* v : {&hfLegPt_, &hfLegEta2_, &hfLegPhi2_, &hfLegPfPt_, &hfLegPfRefPt_, &pfMuPt_}) v->clear();
    hfLegPfDR_.clear(); hfLegPfRefDR_.clear();
    for (auto* v : {&hfLegHitOk_, &hfLegHitPfIdx_, &hfLegHitPfQ_, &hfLegHitStatus_, &hfLegHitN01_, &hfLegHitN02_,
                    &hfLegHitNFail_, &hfLegNSeg_, &hfLegSegPfIdx_, &hfLegSegPfQ_, &hfLegNSegArb_}) v->clear();
    for (auto* v : {&hfLegHitRho_, &hfLegHitZ_, &hfLegHitPfPt_, &hfLegHitPath_, &hfLegHitMiss_, &hfLegSegPfPt_}) v->clear();
    for (auto* v : {&hfLegHitDR_, &hfLegSegFrac_, &hfLegSegPfDR_, &hfLegSegFracArb_}) v->clear();
    pfSlimN_ = pfSlimLinked_ = segDropped_ = 0;
    for (auto* v : {&hfVertexPath_, &hfVertexDistance_, &hfRefEta_, &hfRefPhi_, &hfVertexEta_, &hfVertexPhi_, &hfVertexPt_}) v->clear();
    nSig_ = 0; twoAcc_ = false; nObj_ = nElig_ = 0;
    for (auto* v : {&sigIndex_, &sigQ_, &sigAcc_, &sigProp_, &cObj_, &cType_, &cQ_, &cIdPass_, &cInEta_, &cProp_,
                    &matchComp_, &matchType_, &matchObj_, &indObj_, &indObjNoEta_, &matchCompNoEta_, &matchTypeNoEta_, &matchObjNoEta_})
      v->clear();
    for (auto* v : {&sigPt_, &sigEta_, &sigPhi_, &sigLxy_, &sigD0_, &sigEta2_, &sigPhi2_, &cEta2_, &cPhi2_, &matchDR_,
                    &sigVx_, &sigVy_, &sigVz_, &sigBx_, &sigBy_, &sigBz_})
      v->clear();
    genDR_ = genMuMuPt_ = genMuMuPhi_ = genMetPt_ = genMetPhi_ = pfMetPt_ = pfMetPhi_ = puppiMetPt_ = puppiMetPhi_ = -999.f;
    bbStatus_ = kNotFitted;
    bbChi2_ = bbNdof_ = bbProb_ = bbVx_ = bbVy_ = bbVz_ = bbLxyBS_ = bbMass_ = bbRecoDR_ = bbRecoMuMuPt_ =
        bbRecoMuMuPhi_ = bbSigmaLxy_ = bbBx_ = bbBy_ = bbBz_ = -999.f;
    anyNPairs_ = anyNValid_ = 0;
    hfN_ = hfTried_ = 0; hfFirstIsBB_ = false; hfFirstStatus_ = hfStatus_ = kNotFitted; hfTypeA_ = hfTypeB_ = -1;
    hfChi2_ = hfNdof_ = hfProb_ = hfVx_ = hfVy_ = hfVz_ = hfLxyBS_ = hfMass_ = hfRecoDR_ = hfRecoMuMuPt_ = hfRecoMuMuPhi_ =
        -999.f;
    pfMuN_ = nChi_ = 0; pfMuPx_ = pfMuPy_ = 0; chiOK_ = false; chiPx_ = chiPy_ = -999;
  }

  const edm::EDGetTokenT<std::vector<reco::GenParticle>> gen_;
  const edm::EDGetTokenT<std::vector<reco::Muon>> muons_;
  const edm::EDGetTokenT<std::vector<int>> indices_;
  const edm::EDGetTokenT<reco::BeamSpot> beamspot_;
  const edm::EDGetTokenT<std::vector<pat::MET>> met_, puppiMet_;
  const edm::EDGetTokenT<std::vector<reco::GenMET>> genMet_;
  const edm::EDGetTokenT<std::vector<pat::PackedCandidate>> packed_;
  const edm::EDGetTokenT<std::vector<pat::Muon>> slimmed_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> field_;
  const edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> ttb_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> vertexPropagator_;
  const bool recoOnly_, vertexDirections_, pfNearest_, pfStation2_, pfHitMatch_;
  const PropagateToMuonSetup setup_;
  std::array<int, 3> order_{{2, 0, 1}}, rank_{{0, 0, 0}};  // types best first; rank_[type]
  TTree *evt_ = nullptr, *pair_ = nullptr, *combo_ = nullptr;
  int tree_entries_ = 0;

  unsigned int run_ = 0, lumi_ = 0;
  unsigned long long event_ = 0;
  int nSig_ = 0, nObj_ = 0, nElig_ = 0;
  bool twoAcc_ = false;
  std::vector<int> sigIndex_, sigQ_, sigAcc_, sigProp_, cObj_, cType_, cQ_, cIdPass_, cInEta_, cProp_;
  std::vector<float> sigPt_, sigEta_, sigPhi_, sigLxy_, sigD0_, sigEta2_, sigPhi2_, cEta2_, cPhi2_;
  std::vector<float> sigVx_, sigVy_, sigVz_, sigBx_, sigBy_, sigBz_;
  std::vector<int> hfVertexValid_, hfVertexStatus_;
  std::vector<float> hfVertexPath_, hfVertexDistance_, hfRefEta_, hfRefPhi_, hfVertexEta_, hfVertexPhi_, hfVertexPt_;
  std::vector<int> legQ_, legPfN02_, legPfIdx_, legPfQ_;
  std::vector<float> legPt_, legEta_, legPhi_, legPfPt_, legPfEta_, legPfPhi_;
  std::vector<double> legPfDR_;
  std::vector<int> hfLegQ_, hfLegPfIdx_, hfLegPfN02_, hfLegPfQ_, hfLegPfRefIdx_, hfLegPfRefQ_, pfMuSt2Ok_;
  std::vector<float> hfLegPt_, hfLegEta2_, hfLegPhi2_, hfLegPfPt_, hfLegPfRefPt_, pfMuPt_;
  std::vector<double> hfLegPfDR_, hfLegPfRefDR_;
  std::vector<int> hfLegHitOk_, hfLegHitPfIdx_, hfLegHitPfQ_, hfLegHitStatus_, hfLegHitN01_, hfLegHitN02_, hfLegHitNFail_,
      hfLegNSeg_, hfLegSegPfIdx_, hfLegSegPfQ_, hfLegNSegArb_;
  std::vector<float> hfLegHitRho_, hfLegHitZ_, hfLegHitPfPt_, hfLegHitPath_, hfLegHitMiss_, hfLegSegPfPt_;
  std::vector<double> hfLegHitDR_, hfLegSegFrac_, hfLegSegPfDR_, hfLegSegFracArb_;
  int pfSlimN_ = 0, pfSlimLinked_ = 0, segDropped_ = 0;
  std::vector<int> indObj_, indObjNoEta_;
  int kEvent_ = 0, kTypeA_ = 0, kTypeB_ = 0, kStatus_ = 0;
  float kChi2_ = 0, kNdof_ = 0, kProb_ = 0, kVx_ = 0, kVy_ = 0, kVz_ = 0, kLxyBS_ = 0, kMass_ = 0, kRecoDR_ = 0,
        kRecoMuMuPt_ = 0, kRecoMuMuPhi_ = 0;
  std::vector<int> matchComp_, matchType_, matchObj_, matchCompNoEta_, matchTypeNoEta_, matchObjNoEta_;
  std::vector<float> matchDR_;
  float genDR_ = -999, genMuMuPt_ = -999, genMuMuPhi_ = -999, genMetPt_ = -999, genMetPhi_ = -999,
        pfMetPt_ = -999, pfMetPhi_ = -999, puppiMetPt_ = -999, puppiMetPhi_ = -999;
  int bbStatus_ = 0, anyNPairs_ = 0, anyNValid_ = 0;
  float bbChi2_ = -999, bbNdof_ = -999, bbProb_ = -999, bbVx_ = -999, bbVy_ = -999, bbVz_ = -999,
        bbLxyBS_ = -999, bbMass_ = -999, bbRecoDR_ = -999, bbRecoMuMuPt_ = -999, bbRecoMuMuPhi_ = -999,
        bbSigmaLxy_ = -999, bbBx_ = -999, bbBy_ = -999, bbBz_ = -999;
  int hfN_ = 0, hfTried_ = 0, hfFirstStatus_ = 0, hfStatus_ = 0, hfTypeA_ = -1, hfTypeB_ = -1, pfMuN_ = 0, nChi_ = 0;
  bool hfFirstIsBB_ = false, chiOK_ = false;
  float hfChi2_ = -999, hfNdof_ = -999, hfProb_ = -999, hfVx_ = -999, hfVy_ = -999, hfVz_ = -999, hfLxyBS_ = -999,
        hfMass_ = -999, hfRecoDR_ = -999, hfRecoMuMuPt_ = -999, hfRecoMuMuPhi_ = -999;
  double pfMuPx_ = 0, pfMuPy_ = 0, chiPx_ = -999, chiPy_ = -999;
  int pEvent_ = 0, pObjA_ = 0, pObjB_ = 0, pTypeA_ = 0, pTypeB_ = 0, pCat_ = 0, pStatus_ = 0;
  int pQA_ = 0, pQB_ = 0, pMatchedLegs_ = 0;
  bool pPcaOK_ = false;
  float pPcaDR_ = -999, pPcaMuMuPt_ = -999, pPcaMuMuPhi_ = -999;
  float pPcaPtA_ = -999, pPcaEtaA_ = -999, pPcaPhiA_ = -999, pPcaPtB_ = -999, pPcaEtaB_ = -999, pPcaPhiB_ = -999;
  float pRecoPtA_ = -999, pRecoEtaA_ = -999, pRecoPhiA_ = -999, pRecoPtB_ = -999, pRecoEtaB_ = -999, pRecoPhiB_ = -999;
  bool pIsBB_ = false;
  float pChi2_ = 0, pNdof_ = 0, pProb_ = 0, pVx_ = 0, pVy_ = 0, pVz_ = 0, pLxyBS_ = 0, pMass_ = 0, pRecoDR_ = 0,
        pRecoMuMuPt_ = 0, pRecoMuMuPhi_ = 0;
};
DEFINE_FWK_MODULE(PreselectedVertexing);
