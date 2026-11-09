// Object-level charge-flip check (Level 1, exploratory).
//
// Usage (CMSSW env; one xrootd path per line in the file list):
//   root -l -b -q 'charge_flip_check.C+("files.txt","out.root")'
// Input used 2026-10-13: signal_CMSSW_13_0_13_iDMmu_All_Collections_v1_Propagated (720k events).
//
// Gen set : signal muons (GenSigMuon / GenSigAntiMuon: |pdgId|==13, status 1, last copy,
//           first different-PDG ancestor 1000023), accepted: pT>5, |eta|<2.4,
//           0<=vxy<740 cm, |vz|<960 cm.
// Surface : muon station 2 (gen and reco propagated eta/phi). Reco tracks are propagated
//           with their own reconstructed charge.
// Reco    : RecoDisplacedMuon innerTrack / globalTrack (valid track, St2 status==2),
//           recoDSAMuon (St2 valid). No further reco cuts; DSA displacedId stored.
// Matching: strict one-to-one global-nearest (greedy, with fallback), strict dR < thr.
//           thr = 0.06 inner/global, 0.10 DSA.
//           off   : sign NOT required (the flip measurement)
//           on    : same sign required (efficiency comparison only)
//           loose : sign not required, thr 0.20 inner/global, 0.30 DSA (bias check)
//           vtx   : sign not required, dR of gen production direction vs reco track
//                   momentum direction at its reference point, thr 0.10 inner/global,
//                   0.30 DSA. Much less charge-dependent than station 2 for tracker
//                   tracks (short extrapolation); DSA direction is still extrapolated.
// Output  : one row per accepted signal gen muon.

#include <TChain.h>
#include <TFile.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <TString.h>
#include <Math/Vector4D.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using LV = ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double>>;

namespace {

float deltaPhi(float a, float b) {
  float d = a - b;
  while (d > M_PI) d -= 2 * M_PI;
  while (d <= -M_PI) d += 2 * M_PI;
  return d;
}
float deltaR(float eta1, float phi1, float eta2, float phi2) {
  const float de = eta1 - eta2, dp = deltaPhi(phi1, phi2);
  return std::sqrt(de * de + dp * dp);
}

struct Gen {
  bool use;  // accepted and St2 propagation valid
  bool acc;  // accepted (vertex-direction matching needs no propagation)
  float eta2, phi2, eta, phi;
  int q;
};
struct Cand {
  float eta2, phi2, vtxEta, vtxPhi, pt, ptErr;
  int q, extra;
};

// Greedy one-to-one: all compatible pairs sorted by dR, accept if neither is used.
void greedy(const std::vector<Gen>& g, const std::vector<Cand>& c, float thr, bool sameSign,
            int idx[2], float dr[2], bool atVertex = false) {
  struct P { float dr; int ig, ic; };
  std::vector<P> pairs;
  for (int ig = 0; ig < (int)g.size(); ++ig) {
    idx[ig] = -1;
    dr[ig] = -1;
    if (!(atVertex ? g[ig].acc : g[ig].use)) continue;
    for (int ic = 0; ic < (int)c.size(); ++ic) {
      if (sameSign && c[ic].q != g[ig].q) continue;
      const float d = atVertex ? deltaR(g[ig].eta, g[ig].phi, c[ic].vtxEta, c[ic].vtxPhi)
                               : deltaR(g[ig].eta2, g[ig].phi2, c[ic].eta2, c[ic].phi2);
      if (d < thr) pairs.push_back({d, ig, ic});
    }
  }
  std::sort(pairs.begin(), pairs.end(), [](const P& a, const P& b) { return a.dr < b.dr; });
  std::vector<char> usedC(c.size(), 0);
  for (const auto& p : pairs) {
    if (idx[p.ig] >= 0 || usedC[p.ic]) continue;
    idx[p.ig] = p.ic;
    dr[p.ig] = p.dr;
    usedC[p.ic] = 1;
  }
}

struct ColOut {
  int nCand;
  int offIdx, offQ, offExtra;
  float offDr, offPt, offPtErr, offVtxDr;
  int onMatched, otherOnIdx;
  int vtxIdx, vtxQ, offOtherQ, vtxOtherQ;  // *OtherQ: charge of the other gen muon's match (0: none)
  float vtxDr, vtxPt;
  int looseIdx, looseQ;
  float looseDr, loosePt, looseVtxDr;
  void branch(TTree* t, const std::string& p) {
    t->Branch((p + "_nCand").c_str(), &nCand);
    t->Branch((p + "_offIdx").c_str(), &offIdx);
    t->Branch((p + "_offQ").c_str(), &offQ);
    t->Branch((p + "_offExtra").c_str(), &offExtra);
    t->Branch((p + "_offDr").c_str(), &offDr);
    t->Branch((p + "_offPt").c_str(), &offPt);
    t->Branch((p + "_offPtErr").c_str(), &offPtErr);
    t->Branch((p + "_offVtxDr").c_str(), &offVtxDr);
    t->Branch((p + "_onMatched").c_str(), &onMatched);
    t->Branch((p + "_otherOnIdx").c_str(), &otherOnIdx);  // same-sign match of the other gen muon
    t->Branch((p + "_vtxIdx").c_str(), &vtxIdx);
    t->Branch((p + "_offOtherQ").c_str(), &offOtherQ);
    t->Branch((p + "_vtxOtherQ").c_str(), &vtxOtherQ);
    t->Branch((p + "_vtxQ").c_str(), &vtxQ);
    t->Branch((p + "_vtxDr").c_str(), &vtxDr);
    t->Branch((p + "_vtxPt").c_str(), &vtxPt);
    t->Branch((p + "_looseIdx").c_str(), &looseIdx);
    t->Branch((p + "_looseQ").c_str(), &looseQ);
    t->Branch((p + "_looseDr").c_str(), &looseDr);
    t->Branch((p + "_loosePt").c_str(), &loosePt);
    t->Branch((p + "_looseVtxDr").c_str(), &looseVtxDr);
  }
};

// Fill matching results of one collection for gen ig.
struct ColMatch {
  int off[2], on[2], loose[2], vtx[2];
  float offDr[2], onDr[2], looseDr[2], vtxDr[2];
  void run(const std::vector<Gen>& g, const std::vector<Cand>& c, float thr, float looseThr,
           float vtxThr) {
    greedy(g, c, vtxThr, false, vtx, vtxDr, true);
    greedy(g, c, thr, false, off, offDr);
    greedy(g, c, thr, true, on, onDr);
    greedy(g, c, looseThr, false, loose, looseDr);
  }
  void fill(ColOut& o, const std::vector<Gen>& g, const std::vector<Cand>& c, int ig) const {
    o.nCand = c.size();
    o.offIdx = off[ig];
    o.offDr = offDr[ig];
    o.onMatched = on[ig] >= 0;
    o.otherOnIdx = on[1 - ig];
    o.vtxIdx = vtx[ig];
    o.offOtherQ = off[1 - ig] >= 0 ? c[off[1 - ig]].q : 0;
    o.vtxOtherQ = vtx[1 - ig] >= 0 ? c[vtx[1 - ig]].q : 0;
    o.vtxDr = vtxDr[ig];
    o.vtxQ = vtx[ig] >= 0 ? c[vtx[ig]].q : 0;
    o.vtxPt = vtx[ig] >= 0 ? c[vtx[ig]].pt : -1;
    o.looseIdx = loose[ig];
    o.looseDr = looseDr[ig];
    o.offQ = o.offExtra = o.looseQ = 0;
    o.offPt = o.offPtErr = o.offVtxDr = o.loosePt = o.looseVtxDr = -1;
    if (off[ig] >= 0) {
      const Cand& k = c[off[ig]];
      o.offQ = k.q;
      o.offExtra = k.extra;
      o.offPt = k.pt;
      o.offPtErr = k.ptErr;
      o.offVtxDr = deltaR(g[ig].eta, g[ig].phi, k.vtxEta, k.vtxPhi);
    }
    if (loose[ig] >= 0) {
      const Cand& k = c[loose[ig]];
      o.looseQ = k.q;
      o.loosePt = k.pt;
      o.looseVtxDr = deltaR(g[ig].eta, g[ig].phi, k.vtxEta, k.vtxPhi);
    }
  }
};

}  // namespace

void charge_flip_check(const char* fileList, const char* outName, Long64_t maxEvents = -1) {
  TChain chain("ntuples/outT");
  std::ifstream in(fileList);
  std::string line;
  while (std::getline(in, line))
    if (!line.empty()) chain.Add(line.c_str());

  TTreeReader r(&chain);
  // Gen signal muons (scalars, one muon and one antimuon per event).
  TTreeReaderValue<Bool_t> gValid[2] = {{r, "GenSigMuon_isValid"}, {r, "GenSigAntiMuon_isValid"}};
  TTreeReaderValue<Int_t> gQ[2] = {{r, "GenSigMuon_charge"}, {r, "GenSigAntiMuon_charge"}};
  TTreeReaderValue<Float_t> gPt[2] = {{r, "GenSigMuon_pt"}, {r, "GenSigAntiMuon_pt"}};
  TTreeReaderValue<Float_t> gEta[2] = {{r, "GenSigMuon_eta"}, {r, "GenSigAntiMuon_eta"}};
  TTreeReaderValue<Float_t> gPhi[2] = {{r, "GenSigMuon_phi"}, {r, "GenSigAntiMuon_phi"}};
  TTreeReaderValue<Float_t> gVxy[2] = {{r, "GenSigMuon_vxy"}, {r, "GenSigAntiMuon_vxy"}};
  TTreeReaderValue<Float_t> gVz[2] = {{r, "GenSigMuon_vz"}, {r, "GenSigAntiMuon_vz"}};
  TTreeReaderValue<Float_t> gVx[2] = {{r, "GenSigMuon_vx"}, {r, "GenSigAntiMuon_vx"}};
  TTreeReaderValue<Float_t> gVy[2] = {{r, "GenSigMuon_vy"}, {r, "GenSigAntiMuon_vy"}};
  TTreeReaderValue<Int_t> gProp[2] = {{r, "GenSigMuon_propSt2Valid"}, {r, "GenSigAntiMuon_propSt2Valid"}};
  TTreeReaderValue<Float_t> gEta2[2] = {{r, "GenSigMuon_propSt2Eta"}, {r, "GenSigAntiMuon_propSt2Eta"}};
  TTreeReaderValue<Float_t> gPhi2[2] = {{r, "GenSigMuon_propSt2Phi"}, {r, "GenSigAntiMuon_propSt2Phi"}};

  // displacedMuons inner and global tracks (vectors parallel to RecoDisplacedMuon).
  struct TrackBranches {
    TTreeReaderValue<std::vector<int>> valid, q, st;
    TTreeReaderValue<std::vector<LV>> p4;
    TTreeReaderValue<std::vector<float>> ptErr, eta2, phi2;
    TrackBranches(TTreeReader& r, const std::string& p)
        : valid(r, (p + "_valid").c_str()), q(r, (p + "_charge").c_str()),
          st(r, (p + "_propSt2Status").c_str()), p4(r, (p + "_p4").c_str()),
          ptErr(r, (p + "_ptError").c_str()), eta2(r, (p + "_propSt2PositionEta").c_str()),
          phi2(r, (p + "_propSt2PositionPhi").c_str()) {}
  };
  TrackBranches inner(r, "RecoDisplacedMuon_innerTrack"), global(r, "RecoDisplacedMuon_globalTrack");

  TTreeReaderValue<std::vector<int>> dsaQ(r, "recoDSAMuonCharge"), dsaId(r, "recoDSAMuonDisplacedId"),
      dsaPropValid(r, "recoDSAMuonPropSt2Valid");
  TTreeReaderValue<std::vector<float>> dsaPt(r, "recoDSAMuonPt"), dsaPtErr(r, "recoDSAMuonPtErr"),
      dsaEta(r, "recoDSAMuonEta"), dsaPhi(r, "recoDSAMuonPhi"), dsaEta2(r, "recoDSAMuonPropSt2Eta"),
      dsaPhi2(r, "recoDSAMuonPropSt2Phi");

  TFile out(outName, "RECREATE");
  TTree t("genMuons", "accepted signal gen muons with sign-off/on/loose station-2 matches");
  Long64_t entry;
  int sample, genQ, genProp;
  float genPt, genEta, genPhi, genVxy, genVz, genVx, genVy, genD0, mchi, dmchi, ctau;
  t.Branch("entry", &entry);
  t.Branch("sample", &sample);  // index into the 'samples' list written below
  t.Branch("genQ", &genQ);
  t.Branch("genProp", &genProp);
  t.Branch("genPt", &genPt);
  t.Branch("genEta", &genEta);
  t.Branch("genPhi", &genPhi);
  t.Branch("genVxy", &genVxy);
  t.Branch("genVz", &genVz);
  t.Branch("genVx", &genVx);
  t.Branch("genVy", &genVy);
  t.Branch("genD0", &genD0);  // straight-line transverse IP to (0,0): vx*sin(phi) - vy*cos(phi)
  t.Branch("mchi", &mchi);
  t.Branch("dmchi", &dmchi);
  t.Branch("ctau", &ctau);  // mm, from the sample directory name
  ColOut oIn, oGl, oDsa;
  oIn.branch(&t, "in");
  oGl.branch(&t, "gl");
  oDsa.branch(&t, "dsa");

  std::vector<std::string> samples;
  int curTree = -1;
  Long64_t nEvents = 0, nSizeMismatch = 0;
  while (r.Next()) {
    if (maxEvents >= 0 && nEvents >= maxEvents) break;
    ++nEvents;
    entry = r.GetCurrentEntry();
    if (chain.GetTreeNumber() != curTree) {
      curTree = chain.GetTreeNumber();
      TString f = chain.GetFile()->GetName();
      TString s = f(f.Index("Mchi-"), f.Last('/') - f.Index("Mchi-"));  // Mchi-X_dMchi-Y/ctau-Z
      if (samples.empty() || samples.back() != s.Data()) samples.push_back(s.Data());
      // s = Mchi-10p5_dMchi-1p0/ctau-100
      TString v = s;
      v.ReplaceAll("p", ".");
      v.ReplaceAll("Mchi-", "");
      v.ReplaceAll("_d", " ");
      v.ReplaceAll("/ctau-", " ");
      std::sscanf(v.Data(), "%f %f %f", &mchi, &dmchi, &ctau);
    }
    sample = samples.size() - 1;

    std::vector<Gen> g(2);
    bool acc[2];
    for (int i = 0; i < 2; ++i) {
      acc[i] = *gValid[i] && *gPt[i] > 5 && std::abs(*gEta[i]) < 2.4 && *gVxy[i] >= 0 &&
               *gVxy[i] < 740 && std::abs(*gVz[i]) < 960 && std::abs(*gQ[i]) == 1;
      g[i] = {acc[i] && *gProp[i] == 1, acc[i], *gEta2[i], *gPhi2[i], *gEta[i], *gPhi[i], *gQ[i]};
    }
    if (!acc[0] && !acc[1]) continue;

    auto tracks = [&](TrackBranches& b) {
      std::vector<Cand> c;
      if (b.valid->size() != b.st->size()) ++nSizeMismatch;
      for (size_t i = 0; i < b.valid->size() && i < b.st->size(); ++i) {
        if ((*b.valid)[i] != 1 || (*b.st)[i] != 2) continue;
        const LV& p = (*b.p4)[i];
        c.push_back({(*b.eta2)[i], (*b.phi2)[i], (float)p.Eta(), (float)p.Phi(), (float)p.Pt(),
                     (*b.ptErr)[i], (*b.q)[i], (int)i});
      }
      return c;
    };
    const std::vector<Cand> cIn = tracks(inner), cGl = tracks(global);
    std::vector<Cand> cDsa;
    if (dsaQ->size() != dsaPropValid->size()) ++nSizeMismatch;
    for (size_t i = 0; i < dsaQ->size() && i < dsaPropValid->size(); ++i) {
      if ((*dsaPropValid)[i] != 1) continue;
      cDsa.push_back({(*dsaEta2)[i], (*dsaPhi2)[i], (*dsaEta)[i], (*dsaPhi)[i], (*dsaPt)[i],
                      (*dsaPtErr)[i], (*dsaQ)[i], (*dsaId)[i]});
    }

    ColMatch mIn, mGl, mDsa;
    mIn.run(g, cIn, 0.06, 0.20, 0.10);
    mGl.run(g, cGl, 0.06, 0.20, 0.10);
    mDsa.run(g, cDsa, 0.10, 0.30, 0.30);

    for (int i = 0; i < 2; ++i) {
      if (!acc[i]) continue;
      genQ = *gQ[i];
      genProp = *gProp[i];
      genPt = *gPt[i];
      genEta = *gEta[i];
      genPhi = *gPhi[i];
      genVxy = *gVxy[i];
      genVz = *gVz[i];
      genVx = *gVx[i];
      genVy = *gVy[i];
      genD0 = genVx * std::sin(genPhi) - genVy * std::cos(genPhi);
      mIn.fill(oIn, g, cIn, i);
      mGl.fill(oGl, g, cGl, i);
      mDsa.fill(oDsa, g, cDsa, i);
      t.Fill();
    }
  }
  out.cd();
  t.Write();
  TTree meta("samples", "sample names, row index = genMuons.sample");
  std::string name;
  meta.Branch("name", &name);
  for (auto& s : samples) { name = s; meta.Fill(); }
  meta.Write();
  std::cout << "events " << nEvents << " acceptedGen " << t.GetEntries() << " samples "
            << samples.size() << " sizeMismatchEvents " << nSizeMismatch << std::endl;
}
