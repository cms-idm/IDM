// Dimuon pair candidates (mumuvtx), fixed vertexing component per cleaned displaced muon and signal truth labels
// (mumusig) for the ntuple. Ported unchanged from the recoOnly path of PreselectedVertexing (Alaa_Dev,
// AODSkimmer/plugins/PreselectedVertexing.cc; kb/studies/reco-pair-recovery.md, kb/studies/coffea-ntuple-content.md):
// - components of each cleaned displacedMuons object: 0 inner, 1 outer (embedded DSA), 2 global; eligible = passes its
//   ID (inner: isTrackerMuon; outer: DSA ID; global: isGlobalMuon) AND own track |eta| < 2.4;
// - fixed component = the eligible component highest in the hierarchy order (default G > I > DSA = 2,0,1);
// - pairs: all unordered pairs i < j of objects with a fixed component, in cleaned-object order, leg 1 = i. Shared
//   track keys -> status 1 (no fit); opposite sign -> KalmanVertexFitter(refit) on the two fixed components, status 3
//   valid or 2 failed; same sign -> status 0 (not fitted). Fit outputs and PCA quantities are rounded to float where
//   the producer rounds them; PCA = propagation of each original track to the (float) fitted vertex with
//   SteppingHelixPropagatorAny;
// - truth (MC with prunedGenParticles): signal muons = status-1 last-copy muons whose first different mother is
//   chi2 (|pdgId| 1000023), in gen-collection order; labels only when exactly two opposite-sign signal muons exist
//   (no acceptance requirement): staged hierarchy matching at station 2 (dR < 0.06 I/G, < 0.10 DSA, same charge,
//   globally nearest greedy one-to-one, whole object claimed).
// Products absent from the input (no displacedMuons in standard MiniAOD) leave mumuvtxAvailable = 0 and everything
// empty. No cut is applied here.
#ifndef iDMe_AODSkimmer_MuMuVtx_h
#define iDMe_AODSkimmer_MuMuVtx_h

#include <algorithm>
#include <array>
#include <cmath>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MuonAnalysis/MuonAssociators/interface/PropagateToMuon.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "TMath.h"

#include "iDMe/CustomTools/interface/NtupleContainerV2.hh"

namespace mumuvtx {

  constexpr double kMuonMass = 0.1056583755;
  enum Status { kNotFitted = 0, kShared = 1, kFailed = 2, kValid = 3 };

  inline const reco::GenParticle* firstDifferentMother(const reco::GenParticle& p) {
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
  inline bool outerID(const reco::Track& t) {
    const int csc = t.hitPattern().numberOfValidMuonCSCHits(), dt = t.hitPattern().numberOfValidMuonDTHits();
    return csc + dt > 12 && (csc > 0 || dt > 18) && t.normalizedChi2() < 2.5 && t.pt() > 0 &&
           t.ptError() / t.pt() < 1;
  }
  struct Pos {
    bool ok = false;
    float eta = -999.f, phi = -999.f;
  };
  inline Pos position(const TrajectoryStateOnSurface& state) {
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
    bool accepted = false;
    Pos st2;
  };
  struct Comp {
    int obj = -1;  // raw displacedMuons index (originalIndex), as the producer's tie-break
    int pos = -1;  // local position in the cleaned collection (RecoDisplacedMuon)
    int type = -1, q = 0;
    bool idPass = false, inEta = false;
    Pos st2;
    reco::TrackRef ref;
  };
  struct Match {
    std::array<int, 2> comp{{-1, -1}};
    std::array<int, 2> obj{{-1, -1}};
  };

  // Staged (types in the given order, best first) globally-nearest greedy one-to-one; a matched object is claimed
  // whole. Sorted by dR, then gen index, then original object index.
  inline Match hierarchy(const std::array<Gen, 2>& gen, const std::vector<Comp>& comps,
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
          if (k.type != type || !k.idPass || !k.inEta || !k.st2.ok || k.q != gen[g].q || claimed.count(k.obj))
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
        m.obj[g] = obj;
        claimed.insert(obj);
      }
    }
    return m;
  }

  struct Fit {
    int status = kNotFitted;
    float vx = -999, vy = -999, vz = -999, chi2 = -999, ndof = -999, prob = -999, lxyBS = -999, mass = -999;
  };
  inline Fit fitLegs(const reco::TrackRef& a, const reco::TrackRef& b, const TransientTrackBuilder& ttb,
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
  inline std::set<std::pair<unsigned int, unsigned int>> keys(const reco::Muon& mu) {
    std::set<std::pair<unsigned int, unsigned int>> s;
    for (const auto& r : {mu.innerTrack(), mu.outerTrack(), mu.globalTrack()})
      if (r.isNonnull()) s.emplace(r.id().id(), r.key());
    return s;
  }

  // muons/indices: cleanedDisplacedMuons and its originalIndex (both null if absent); gen: prunedGenParticles (null for
  // data or if absent). The ntuple's RecoDisplacedMuon collection must already be filled from the same muons.
  inline void fill(NtupleContainerV2& nt, const std::vector<reco::Muon>* muons, const std::vector<int>* indices,
                   const std::vector<reco::GenParticle>* genParticles, const reco::BeamSpot& bs,
                   const MagneticField* field, const TransientTrackBuilder& ttb, const SteppingHelixPropagator& prop,
                   const PropagateToMuon& propagatorSt2, const std::array<int, 3>& order) {
    std::array<int, 3> rank{};
    for (int r = 0; r < 3; ++r) rank[order[r]] = r;

    // Signal gen muons (MC only), in gen-collection order; acceptance is a flag.
    std::vector<Gen> sig;
    if (genParticles) {
      for (size_t i = 0; i < genParticles->size(); ++i) {
        const auto& p = (*genParticles)[i];
        if (std::abs(p.pdgId()) != 13 || p.status() != 1 || !p.isLastCopy()) continue;
        const auto* mom = firstDifferentMother(p);
        if (!mom || std::abs(mom->pdgId()) != 1000023) continue;
        Gen g;
        g.index = static_cast<int>(i);
        g.q = p.charge();
        const double lxy = p.vertex().rho();
        g.accepted = std::abs(p.eta()) < 2.4 && lxy >= 0 && lxy < 740 && std::abs(p.vz()) < 960;
        const FreeTrajectoryState start(GlobalPoint(p.vx(), p.vy(), p.vz()), GlobalVector(p.px(), p.py(), p.pz()),
                                        TrackCharge(p.charge()), field);
        g.st2 = position(propagatorSt2.extrapolate(start));
        sig.push_back(g);
      }
      nt.nmumusig_ = static_cast<int>(sig.size());
      for (const auto& g : sig) {
        nt.mumusig_genIdx_.push_back(g.index);
        nt.mumusig_charge_.push_back(g.q);
        nt.mumusig_accepted_.push_back(g.accepted);
      }
      nt.mumusigTwoAcc_ = sig.size() == 2 && sig[0].q * sig[1].q < 0 && sig[0].accepted && sig[1].accepted;
    }

    if (!muons || !indices) return;  // mumuvtxAvailable stays 0
    if (indices->size() != muons->size()) throw cms::Exception("MuMuVtx") << "cleaner index size";
    nt.mumuvtxAvailable_ = 1;

    // Components of cleaned objects (all that exist; station 2 only if the ID passes).
    std::vector<Comp> comps;
    std::vector<int> objBest(muons->size(), -1);  // fixed component per cleaned position (index into comps)
    std::array<std::vector<int>*, 3> idPass{{&nt.recoDisplacedMuonInnerTrackIdPass_,
                                             &nt.recoDisplacedMuonOuterTrackIdPass_,
                                             &nt.recoDisplacedMuonGlobalTrackIdPass_}};
    std::array<std::vector<int>*, 3> inEta{{&nt.recoDisplacedMuonInnerTrackInEta_,
                                            &nt.recoDisplacedMuonOuterTrackInEta_,
                                            &nt.recoDisplacedMuonGlobalTrackInEta_}};
    for (size_t i = 0; i < muons->size(); ++i) {
      const auto& mu = (*muons)[i];
      const std::array<reco::TrackRef, 3> refs{{mu.innerTrack(), mu.outerTrack(), mu.globalTrack()}};
      for (int t = 0; t < 3; ++t) {
        int pass = 0, eta = 0;
        if (refs[t].isNonnull()) {
          if (!refs[t].isAvailable()) throw cms::Exception("MuMuVtx") << "Unavailable TrackRef";
          Comp c;
          c.obj = (*indices)[i];
          c.pos = static_cast<int>(i);
          c.type = t;
          c.q = refs[t]->charge();
          c.ref = refs[t];
          c.idPass = t == 0 ? mu.isTrackerMuon() : t == 1 ? outerID(*refs[t]) : mu.isGlobalMuon();
          const double teta = refs[t]->eta();
          c.inEta = std::isfinite(teta) && std::abs(teta) < 2.4;
          if (c.idPass) c.st2 = position(propagatorSt2.extrapolate(*refs[t]));
          pass = c.idPass;
          eta = c.inEta;
          comps.push_back(c);
        }
        idPass[t]->push_back(pass);
        inEta[t]->push_back(eta);
      }
    }
    for (size_t c = 0; c < comps.size(); ++c)
      if (comps[c].idPass && comps[c].inEta) {
        const int pos = comps[c].pos;
        if (objBest[pos] < 0 || rank[comps[c].type] < rank[comps[objBest[pos]].type]) objBest[pos] = static_cast<int>(c);
      }
    for (size_t i = 0; i < objBest.size(); ++i)
      nt.recoDisplacedMuonFixedComp_.push_back(objBest[i] >= 0 ? comps[objBest[i]].type : -1);

    // Truth labels (never used for components or pairs).
    nt.recoDisplacedMuonMumusigIdx_.assign(muons->size(), -1);
    if (sig.size() == 2 && sig[0].q * sig[1].q < 0) {
      const Match labels = hierarchy(std::array<Gen, 2>{{sig[0], sig[1]}}, comps, order);
      for (int g = 0; g < 2; ++g)
        if (labels.comp[g] >= 0) nt.recoDisplacedMuonMumusigIdx_[comps[labels.comp[g]].pos] = g;
    }

    // Pairs of distinct objects, each at its fixed component.
    KalmanVertexFitter kvf(true);
    std::vector<int> best;
    for (size_t i = 0; i < objBest.size(); ++i)
      if (objBest[i] >= 0) best.push_back(objBest[i]);
    for (size_t i = 0; i < best.size(); ++i)
      for (size_t j = i + 1; j < best.size(); ++j) {
        const auto& a = comps[best[i]];
        const auto& b = comps[best[j]];
        const auto ka = keys((*muons)[a.pos]);
        const auto kb = keys((*muons)[b.pos]);
        bool shared = false;
        for (const auto& k : ka) shared |= kb.count(k) > 0;
        Fit f;
        if (shared)
          f.status = kShared;
        else if (a.q * b.q < 0)
          f = fitLegs(a.ref, b.ref, ttb, kvf, bs);
        bool pcaOK = false;
        float pcaDR = -999.f, pcaMuMuPt = -999.f, pcaMuMuPhi = -999.f;
        std::array<float, 2> pcaPt{{-999.f, -999.f}}, pcaEta{{-999.f, -999.f}}, pcaPhi{{-999.f, -999.f}};
        if (f.status == kValid) {
          const GlobalPoint target(f.vx, f.vy, f.vz);
          std::array<GlobalVector, 2> momenta;
          bool ok = true;
          int leg = 0;
          for (const auto& ref : {a.ref, b.ref}) {
            const auto start = trajectoryStateTransform::initialFreeState(*ref, field);
            SteppingHelixStateInfo end;
            prop.propagate(SteppingHelixStateInfo(start), target, end);
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
            pcaDR = reco::deltaR(momenta[0].eta(), momenta[0].phi(), momenta[1].eta(), momenta[1].phi());
            pcaMuMuPt = sum.perp();
            pcaMuMuPhi = sum.phi();
            pcaOK = std::isfinite(pcaDR) && std::isfinite(pcaMuMuPt) && std::isfinite(pcaMuMuPhi);
            if (pcaOK)
              for (int l = 0; l < 2; ++l) {
                pcaPt[l] = momenta[l].perp(); pcaEta[l] = momenta[l].eta(); pcaPhi[l] = momenta[l].phi();
              }
          }
        }
        nt.mumuvtx_m1_idx_.push_back(a.pos);  nt.mumuvtx_m2_idx_.push_back(b.pos);
        nt.mumuvtx_m1_typ_.push_back(a.type); nt.mumuvtx_m2_typ_.push_back(b.type);
        nt.mumuvtx_m1_charge_.push_back(a.q); nt.mumuvtx_m2_charge_.push_back(b.q);
        nt.mumuvtx_status_.push_back(f.status);
        nt.mumuvtx_chi2_.push_back(f.chi2); nt.mumuvtx_ndof_.push_back(f.ndof); nt.mumuvtx_prob_.push_back(f.prob);
        nt.mumuvtx_vx_.push_back(f.vx); nt.mumuvtx_vy_.push_back(f.vy); nt.mumuvtx_vz_.push_back(f.vz);
        nt.mumuvtx_lxyBS_.push_back(f.lxyBS); nt.mumuvtx_mass_.push_back(f.mass);
        nt.mumuvtx_pcaOK_.push_back(pcaOK);
        nt.mumuvtx_pcaDR_.push_back(pcaDR); nt.mumuvtx_pcaMuMuPt_.push_back(pcaMuMuPt);
        nt.mumuvtx_pcaMuMuPhi_.push_back(pcaMuMuPhi);
        nt.mumuvtx_m1_pcaPt_.push_back(pcaPt[0]); nt.mumuvtx_m1_pcaEta_.push_back(pcaEta[0]);
        nt.mumuvtx_m1_pcaPhi_.push_back(pcaPhi[0]);
        nt.mumuvtx_m2_pcaPt_.push_back(pcaPt[1]); nt.mumuvtx_m2_pcaEta_.push_back(pcaEta[1]);
        nt.mumuvtx_m2_pcaPhi_.push_back(pcaPhi[1]);
        ++nt.nmumuvtx_;
      }
  }

}  // namespace mumuvtx

#endif
