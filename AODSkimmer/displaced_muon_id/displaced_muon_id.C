// Signal-matched component and displacedMuon ID comparison. See README.md.
// Run in CMSSW on LPC: root -l -b -q 'displaced_muon_id.C+("files.txt","out.root",1000)'
#include <TChain.h>
#include <TFile.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <TString.h>
#include <TROOT.h>
#include <Math/Vector4D.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace displaced_id {
using LV = ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double>>;
struct Gen { bool accepted, propagated; float eta, phi; int charge; };
struct Component { bool present, propagated, pass; float eta, phi; int charge; };
using Muon = std::array<Component, 3>; // inner=bit1, outer=bit2, global=bit4
constexpr float thresholds[3] = {0.06f, 0.10f, 0.06f};
float dr(float eta1, float phi1, float eta2, float phi2) {
  return std::hypot(eta1-eta2, std::remainder(phi1-phi2, 2*M_PI));
}
struct Match { int idx=-1, component=-1; float distance=-1; };
// policy: -1=no ID; 0=all components; 1..4=hierarchy mixed both/tracker/DSA/either.
bool passes(const Muon& mu,int policy) {
  if(policy<0) return true;
  if(policy==0) {
    bool any=false;
    for(const auto& c:mu) if(c.present) { any=true; if(!c.pass) return false; }
    return any;
  }
  if(mu[2].present) return mu[2].pass;
  if(mu[0].present && mu[1].present) {
    if(policy==1) return mu[0].pass && mu[1].pass;
    if(policy==2) return mu[0].pass;
    if(policy==3) return mu[1].pass;
    return mu[0].pass || mu[1].pass;
  }
  if(mu[0].present) return mu[0].pass;
  if(mu[1].present) return mu[1].pass;
  return false;
}
// With component=-1 use the smallest compatible component distance per gen/object pair.
// Separate passes for inner/outer/global; one common pre-ID pass for whole objects.
std::array<Match,2> match(const std::array<Gen,2>& gen, const std::vector<Muon>& muons,
                          int component,int policy=-1,bool priority=false,bool componentID=false) {
  struct Pair { float distance; int gen, muon, component; };
  std::vector<Pair> pairs;
  for (int g=0; g<2; ++g) {
    if (!gen[g].accepted || !gen[g].propagated) continue;
    for (size_t m=0; m<muons.size(); ++m) {
      if(!passes(muons[m],policy)) continue;
      int first=-1;
      if(priority) for(int c: {2,0,1}) if(muons[m][c].present) { first=c; break; }
      Pair best{1e9f, g, int(m), -1};
      for (int c=0; c<3; ++c) {
        if (component>=0 && c!=component) continue;
        if(priority && c!=first) continue;
        const auto& trk=muons[m][c];
        if (!trk.present || !trk.propagated || trk.charge!=gen[g].charge) continue;
        if(componentID && !trk.pass) continue;
        float distance=dr(gen[g].eta,gen[g].phi,trk.eta,trk.phi);
        if (distance<thresholds[c] && distance<best.distance)
          best={distance,g,int(m),c};
      }
      if (best.component>=0) pairs.push_back(best);
    }
  }
  std::sort(pairs.begin(),pairs.end(),[](const Pair& a,const Pair& b) {
    return std::tie(a.distance,a.gen,a.muon,a.component)<std::tie(b.distance,b.gen,b.muon,b.component);
  });
  std::array<Match,2> result;
  std::vector<bool> used(muons.size(),false);
  for (const auto& p:pairs) {
    if (result[p.gen].idx>=0 || used[p.muon]) continue;
    result[p.gen]={p.muon,p.component,p.distance}; used[p.muon]=true;
  }
  return result;
}
struct TrackBranches {
  TTreeReaderValue<std::vector<int>> valid, charge, status, csc, dt;
  TTreeReaderValue<std::vector<float>> eta, phi, ptError, chi2;
  TTreeReaderValue<std::vector<LV>> p4;
  TrackBranches(TTreeReader& r,const std::string& prefix)
    : valid(r,(prefix+"_valid").c_str()), charge(r,(prefix+"_charge").c_str()),
      status(r,(prefix+"_propSt2Status").c_str()), csc(r,(prefix+"_nCSCHits").c_str()),
      dt(r,(prefix+"_nDTHits").c_str()), eta(r,(prefix+"_propSt2PositionEta").c_str()),
      phi(r,(prefix+"_propSt2PositionPhi").c_str()), ptError(r,(prefix+"_ptError").c_str()),
      chi2(r,(prefix+"_normalizedChi2").c_str()), p4(r,(prefix+"_p4").c_str()) {}
  void check(size_t n) {
    if(valid->size()!=n || charge->size()!=n || status->size()!=n || csc->size()!=n ||
       dt->size()!=n || eta->size()!=n || phi->size()!=n || ptError->size()!=n ||
       chi2->size()!=n || p4->size()!=n) throw std::runtime_error("Nonparallel displacedMuon branches");
  }
  Component get(size_t i,bool flag,int c) {
    bool present=(*valid)[i]==1;
    bool pass=flag;
    if(c==1) { // exact ElectronSkimmer::passesDisplacedID, applied to embedded outerTrack
      double pt=(*p4)[i].Pt();
      pass=((*csc)[i]+(*dt)[i]>12) && ((*csc)[i]>0 || (*dt)[i]>18) &&
           (*chi2)[i]<2.5 && pt>0 && (*ptError)[i]/pt<1;
    }
    return {present,(*status)[i]==2,present && pass,(*eta)[i],(*phi)[i],(*charge)[i]};
  }
};
}

void displaced_muon_id(const char* fileList,const char* outName,Long64_t maxEvents=-1) {
  using namespace displaced_id;
  TChain chain("ntuples/outT");
  std::ifstream input(fileList);
  if(!input) throw std::runtime_error("Cannot open file list");
  std::string line;
  std::vector<std::string> sourceFiles;
  while(std::getline(input,line)) if(!line.empty()) { chain.Add(line.c_str()); sourceFiles.push_back(line); }
  if(chain.GetNtrees()==0) throw std::runtime_error("Empty input chain");
  TTreeReader r(&chain);
  TTreeReaderValue<Bool_t> valid[2]={{r,"GenSigMuon_isValid"},{r,"GenSigAntiMuon_isValid"}};
  TTreeReaderValue<Int_t> charge[2]={{r,"GenSigMuon_charge"},{r,"GenSigAntiMuon_charge"}};
  TTreeReaderValue<Int_t> status[2]={{r,"GenSigMuon_status"},{r,"GenSigAntiMuon_status"}};
  TTreeReaderValue<Int_t> lastCopy[2]={{r,"GenSigMuon_isLastCopy"},{r,"GenSigAntiMuon_isLastCopy"}};
  TTreeReaderValue<Int_t> mother[2]={{r,"GenSigMuon_firstDifferentMotherID"},{r,"GenSigAntiMuon_firstDifferentMotherID"}};
  TTreeReaderValue<Int_t> propagated[2]={{r,"GenSigMuon_propSt2Valid"},{r,"GenSigAntiMuon_propSt2Valid"}};
  TTreeReaderValue<Float_t> pt[2]={{r,"GenSigMuon_pt"},{r,"GenSigAntiMuon_pt"}};
  TTreeReaderValue<Float_t> eta[2]={{r,"GenSigMuon_eta"},{r,"GenSigAntiMuon_eta"}};
  TTreeReaderValue<Float_t> phi[2]={{r,"GenSigMuon_phi"},{r,"GenSigAntiMuon_phi"}};
  TTreeReaderValue<Float_t> vx[2]={{r,"GenSigMuon_vx"},{r,"GenSigAntiMuon_vx"}};
  TTreeReaderValue<Float_t> vy[2]={{r,"GenSigMuon_vy"},{r,"GenSigAntiMuon_vy"}};
  TTreeReaderValue<Float_t> vxy[2]={{r,"GenSigMuon_vxy"},{r,"GenSigAntiMuon_vxy"}};
  TTreeReaderValue<Float_t> vz[2]={{r,"GenSigMuon_vz"},{r,"GenSigAntiMuon_vz"}};
  TTreeReaderValue<Float_t> eta2[2]={{r,"GenSigMuon_propSt2Eta"},{r,"GenSigAntiMuon_propSt2Eta"}};
  TTreeReaderValue<Float_t> phi2[2]={{r,"GenSigMuon_propSt2Phi"},{r,"GenSigAntiMuon_propSt2Phi"}};
  TTreeReaderValue<std::vector<int>> tracker(r,"RecoDisplacedMuon_isTracker"), global(r,"RecoDisplacedMuon_isGlobal");
  TTreeReaderValue<ULong64_t> eventNum(r,"eventNum");
  TTreeReaderValue<ULong64_t> runNum(r,"runNum"), lumiSec(r,"lumiSec");
  TrackBranches inner(r,"RecoDisplacedMuon_innerTrack"), outer(r,"RecoDisplacedMuon_outerTrack"),
                gl(r,"RecoDisplacedMuon_globalTrack");
  TFile out(outName,"RECREATE");
  if(out.IsZombie()) throw std::runtime_error("Cannot create output ROOT");
  TTree tree("genMuons","accepted signal gen, pre-ID and post-ID same-charge matches");
  float genPt,genEta,genVxy,genVz,genD0,mchi,dmchi,ctau;
  int genProp,genQ,objIdx,objMask,objIdBits,objComponent;
  float objDr;
  int compIdx[3],compPass[3];
  int compSelectedIdx[3],objSelectedIdx[5],objSelectedMask[5],prioritySelectedIdx[5],prioritySelectedMask[5];
  int priorityIdx,priorityComponent;
  std::string sourceFile;
  ULong64_t event;
  ULong64_t run,lumi;
  Long64_t entry;
  tree.Branch("entry",&entry);
  tree.Branch("sourceFile",&sourceFile); tree.Branch("run",&run); tree.Branch("lumi",&lumi); tree.Branch("event",&event);
  tree.Branch("genPt",&genPt); tree.Branch("genEta",&genEta);
  tree.Branch("genVxy",&genVxy); tree.Branch("genVz",&genVz); tree.Branch("genD0",&genD0);
  tree.Branch("genProp",&genProp); tree.Branch("genQ",&genQ);
  tree.Branch("mchi",&mchi); tree.Branch("dmchi",&dmchi); tree.Branch("ctau",&ctau);
  tree.Branch("objIdx",&objIdx); tree.Branch("objMask",&objMask); tree.Branch("objIdBits",&objIdBits);
  tree.Branch("objComponent",&objComponent); tree.Branch("objDr",&objDr);
  tree.Branch("priorityIdx",&priorityIdx); tree.Branch("priorityComponent",&priorityComponent);
  tree.Branch("objSelectedIdx",objSelectedIdx,"objSelectedIdx[5]/I");
  tree.Branch("objSelectedMask",objSelectedMask,"objSelectedMask[5]/I");
  tree.Branch("prioritySelectedIdx",prioritySelectedIdx,"prioritySelectedIdx[5]/I");
  tree.Branch("prioritySelectedMask",prioritySelectedMask,"prioritySelectedMask[5]/I");
  const char* names[3]={"inner","outer","global"};
  for(int c=0;c<3;++c) {
    tree.Branch((std::string(names[c])+"_idx").c_str(),&compIdx[c]);
    tree.Branch((std::string(names[c])+"_pass").c_str(),&compPass[c]);
    tree.Branch((std::string(names[c])+"_selectedIdx").c_str(),&compSelectedIdx[c]);
  }
  // Complete reco record, including events with no accepted signal gen.
  TTree reco("displacedMuons","every displacedMuon and its component inputs; no truth filtering");
  int muonIdx,isTracker,isGlobal,presenceMask,present[3],propStatus[3],idPass[3],compCharge[3],cscHits[3],dtHits[3];
  float compPt[3],compEta[3],compPhi[3],st2Eta[3],st2Phi[3],ptError[3],chi2[3];
  reco.Branch("sourceFile",&sourceFile); reco.Branch("entry",&entry);
  reco.Branch("run",&run); reco.Branch("lumi",&lumi); reco.Branch("event",&event);
  reco.Branch("mchi",&mchi); reco.Branch("dmchi",&dmchi); reco.Branch("ctau",&ctau);
  reco.Branch("muonIdx",&muonIdx); reco.Branch("isTracker",&isTracker); reco.Branch("isGlobal",&isGlobal);
  reco.Branch("presenceMask",&presenceMask);
  reco.Branch("present",present,"present[3]/I"); reco.Branch("propStatus",propStatus,"propStatus[3]/I");
  reco.Branch("idPass",idPass,"idPass[3]/I"); reco.Branch("compCharge",compCharge,"compCharge[3]/I");
  reco.Branch("cscHits",cscHits,"cscHits[3]/I"); reco.Branch("dtHits",dtHits,"dtHits[3]/I");
  reco.Branch("compPt",compPt,"compPt[3]/F"); reco.Branch("compEta",compEta,"compEta[3]/F");
  reco.Branch("compPhi",compPhi,"compPhi[3]/F"); reco.Branch("st2Eta",st2Eta,"st2Eta[3]/F");
  reco.Branch("st2Phi",st2Phi,"st2Phi[3]/F"); reco.Branch("ptError",ptError,"ptError[3]/F");
  reco.Branch("chi2",chi2,"chi2[3]/F");
  TrackBranches* trackBranches[3]={&inner,&outer,&gl};
  int curTree=-1;
  Long64_t nEvents=0,nAccepted=0,nProp=0;
  std::array<Long64_t,8> objectCounts{};
  while(r.Next()) {
    if(maxEvents>=0 && nEvents>=maxEvents) break;
    ++nEvents; entry=r.GetCurrentEntry();
    run=*runNum; lumi=*lumiSec; event=*eventNum;
    if(chain.GetTreeNumber()!=curTree) {
      curTree=chain.GetTreeNumber();
      TString path=chain.GetFile()->GetName();
      sourceFile=path.Data();
      int start=path.Index("Mchi-");
      if(start<0) throw std::runtime_error("Sample mass/lifetime missing from input path");
      TString sample=path(start,path.Last('/')-start);
      sample.ReplaceAll("p",".");
      if(std::sscanf(sample.Data(),"Mchi-%f_dMchi-%f/ctau-%f",&mchi,&dmchi,&ctau)!=3)
        throw std::runtime_error("Cannot parse sample mass/lifetime");
    }
    std::array<Gen,2> gen;
    for(int g=0;g<2;++g) {
      bool acc=*valid[g] && *status[g]==1 && *lastCopy[g]==1 && std::abs(*mother[g])==1000023 &&
               std::abs(*eta[g])<2.4 && *vxy[g]>=0 && *vxy[g]<740 &&
               std::abs(*vz[g])<960 && std::abs(*charge[g])==1;
      gen[g]={acc,*propagated[g]==1,*eta2[g],*phi2[g],*charge[g]};
    }
    size_t n=tracker->size();
    if(global->size()!=n) throw std::runtime_error("Nonparallel muon flags");
    inner.check(n); outer.check(n); gl.check(n);
    std::vector<Muon> muons;
    for(size_t m=0;m<n;++m) {
      Muon mu={inner.get(m,(*tracker)[m]==1,0),outer.get(m,false,1),gl.get(m,(*global)[m]==1,2)};
      int mask=0; for(int c=0;c<3;++c) if(mu[c].present) mask|=1<<c;
      ++objectCounts[mask]; muons.push_back(mu);
      muonIdx=m; isTracker=(*tracker)[m]; isGlobal=(*global)[m]; presenceMask=mask;
      for(int c=0;c<3;++c) {
        auto& b=*trackBranches[c];
        present[c]=mu[c].present; propStatus[c]=(*b.status)[m]; idPass[c]=mu[c].pass;
        compCharge[c]=mu[c].charge; cscHits[c]=(*b.csc)[m]; dtHits[c]=(*b.dt)[m];
        compPt[c]=(*b.p4)[m].Pt(); compEta[c]=mu[c].present ? (*b.p4)[m].Eta() : -999;
        compPhi[c]=mu[c].present ? (*b.p4)[m].Phi() : -999;
        st2Eta[c]=mu[c].eta; st2Phi[c]=mu[c].phi; ptError[c]=(*b.ptError)[m]; chi2[c]=(*b.chi2)[m];
      }
      reco.Fill();
    }
    if(!gen[0].accepted && !gen[1].accepted) continue;
    std::array<std::array<Match,2>,3> components;
    std::array<std::array<Match,2>,3> selectedComponents;
    for(int c=0;c<3;++c) components[c]=match(gen,muons,c);
    for(int c=0;c<3;++c) selectedComponents[c]=match(gen,muons,c,-1,false,true);
    auto objects=match(gen,muons,-1);
    auto priorityObjects=match(gen,muons,-1,-1,true);
    std::array<std::array<Match,2>,5> selectedObjects,selectedPriority;
    for(int p=0;p<5;++p) { selectedObjects[p]=match(gen,muons,-1,p); selectedPriority[p]=match(gen,muons,-1,p,true); }
    for(int g=0;g<2;++g) {
      if(!gen[g].accepted) continue;
      ++nAccepted; if(gen[g].propagated) ++nProp;
      genPt=*pt[g]; genEta=*eta[g]; genVxy=*vxy[g]; genVz=*vz[g];
      genD0=*vx[g]*std::sin(*phi[g])-*vy[g]*std::cos(*phi[g]);
      genProp=*propagated[g]; genQ=*charge[g];
      for(int c=0;c<3;++c) {
        compIdx[c]=components[c][g].idx;
        compPass[c]=compIdx[c]>=0 && muons[compIdx[c]][c].pass;
        compSelectedIdx[c]=selectedComponents[c][g].idx;
      }
      objIdx=objects[g].idx; objComponent=objects[g].component; objDr=objects[g].distance;
      priorityIdx=priorityObjects[g].idx; priorityComponent=priorityObjects[g].component;
      for(int p=0;p<5;++p) {
        objSelectedIdx[p]=selectedObjects[p][g].idx; prioritySelectedIdx[p]=selectedPriority[p][g].idx;
        objSelectedMask[p]=prioritySelectedMask[p]=0;
        for(int c=0;c<3;++c) {
          if(objSelectedIdx[p]>=0 && muons[objSelectedIdx[p]][c].present) objSelectedMask[p]|=1<<c;
          if(prioritySelectedIdx[p]>=0 && muons[prioritySelectedIdx[p]][c].present) prioritySelectedMask[p]|=1<<c;
        }
      }
      objMask=objIdBits=0;
      if(objIdx>=0) for(int c=0;c<3;++c) {
        if(muons[objIdx][c].present) objMask|=1<<c;
        if(muons[objIdx][c].pass) objIdBits|=1<<c;
      }
      tree.Fill();
    }
  }
  // A missing/incompatible branch must not silently produce a partial result.
  if(!(maxEvents>=0 && nEvents>=maxEvents) && r.GetEntryStatus()!=TTreeReader::kEntryBeyondEnd)
    throw std::runtime_error("TTreeReader stopped before end of input");
  out.cd(); tree.Write(); reco.Write();
  TTree totals("inputSummary","one row per processed chunk");
  std::string revision=std::getenv("ID_STUDY_REVISION") ? std::getenv("ID_STUDY_REVISION") : "unspecified";
  std::string rootVersion=gROOT->GetVersion();
  totals.Branch("sourceFiles",&sourceFiles); totals.Branch("revision",&revision); totals.Branch("rootVersion",&rootVersion);
  totals.Branch("maxEvents",&maxEvents);
  totals.Branch("events",&nEvents); totals.Branch("acceptedGen",&nAccepted); totals.Branch("propagatedGen",&nProp);
  totals.Fill(); totals.Write(); out.Close();
  std::cout<<"events="<<nEvents<<" acceptedGen="<<nAccepted<<" propagatedGen="<<nProp<<std::endl;
  std::cout<<"object masks in all events (inner=1 outer=2 global=4):";
  for(int m=0;m<8;++m) std::cout<<" "<<m<<":"<<objectCounts[m];
  std::cout<<std::endl;
}
