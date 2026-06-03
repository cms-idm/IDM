// -*- C++ -*-
//
// Package:    iDMe/ElectronSkimmer
// Class:      ElectronSkimmer
//
/**\class ElectronSkimmer ElectronSkimmer.cc iDMe/ElectronSkimmer/plugins/ElectronSkimmer.cc

 Description: MiniAOD skimmer for iDM analysis with electrons
*/
//
// Original Author:  Samuel Bright-Thonney
//         Created:  Tue, 21 Sep 2021 17:00:38 GMT
//

#include <algorithm>
#include <cmath> 
#include <memory>
#include <random>
#include <vector>
#include <boost/format.hpp>
#include <boost/any.hpp>

// user include files
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectionUncertainty.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Common/interface/Provenance.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Conversion.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"
#include "DataFormats/PatCandidates/interface/IsolatedTrack.h"
#include "DataFormats/PatCandidates/interface/PFIsolation.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/Common/interface/RefVector.h"

#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"
#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"
#include "JetMETCorrections/Modules/interface/JetResolution.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"

// stuff for kinematic vertex fit
#include "RecoVertex/KinematicFitPrimitives/interface/ParticleMass.h"
#include "RecoVertex/KinematicFitPrimitives/interface/MultiTrackKinematicConstraint.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicParticleFactoryFromTransientTrack.h"
#include "RecoVertex/KinematicFit/interface/KinematicConstrainedVertexFitter.h"
#include "RecoVertex/KinematicFit/interface/TwoTrackMassKinematicConstraint.h"
#include "RecoVertex/KinematicFit/interface/KinematicParticleVertexFitter.h"
#include "RecoVertex/KinematicFit/interface/KinematicParticleFitter.h"
#include "RecoVertex/KinematicFit/interface/MassKinematicConstraint.h"

#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

// Run3 updated paths, Run2 used RecoEgamma/EgammaTools/...
#include "CommonTools/Egamma/interface/EffectiveAreas.h"
#include "CommonTools/Egamma/interface/ConversionTools.h"

#include "iDMe/CustomTools/interface/DisplacedDileptonAOD.hh"
#include "iDMe/CustomTools/interface/JetCorrections.hh"
#include "iDMe/CustomTools/interface/NtupleContainerV2.hh"
//#include "iDMe/CustomTools/interface/IsolationCalculator.hh"
#include "iDMe/CustomTools/interface/Helpers.hh"

#include "DataFormats/MuonReco/interface/Muon.h"  // for SF computation, add prompt muon channels

#include "TTree.h"
#include "TMath.h"

class ElectronSkimmer : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources>  {
   public:
      explicit ElectronSkimmer(const edm::ParameterSet&);
      ~ElectronSkimmer();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      bool getCollections(const edm::Event&);
      // Run3 added
      bool passesDisplacedID(const reco::Track&) const;
      virtual void beginJob() override;
      virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
      virtual void endJob() override;

      // ----------member data ---------------------------
      TTree *outT;
      NtupleContainerV2 nt;
      edm::Service<TFileService> fs;

      std::mt19937 m_random_generator;

      bool isData;
      bool isSignal;
      std::string year;
      const std::string triggerProcessName_;
      // Run3 modified
      std::string metFilterName_;
      std::vector<std::string> metFilters_;
      std::vector<std::string> trigPaths_;
      // Electron isolation effective areas
      EffectiveAreas effectiveAreas_;

      // Tokens 
      const edm::EDGetTokenT<vector<pat::Electron> > recoElectronToken_;
      const edm::EDGetTokenT<vector<pat::Electron> > recoNanoElectronToken_;
      const edm::EDGetTokenT<vector<pat::Electron> >lowPtElectronToken_;
      const edm::EDGetTokenT<vector<pat::Electron> >lowPtNanoElectronToken_;
      const edm::EDGetTokenT<vector<pat::PackedCandidate> > packedPFCandToken_;
      const edm::EDGetTokenT<vector<pat::Jet> > recoJetToken_;
      const edm::EDGetTokenT<GenEventInfoProduct> genEvtInfoToken_;
      const edm::EDGetTokenT<std::vector<PileupSummaryInfo> > pileupInfosToken_;
      const edm::EDGetTokenT<double> rhoToken_;
      const edm::EDGetTokenT<vector<reco::GenParticle> > genParticleToken_;
      const edm::EDGetTokenT<vector<reco::GenJet> > genJetToken_;
      const edm::EDGetTokenT<vector<reco::GenMET> > genMETToken_;
      const edm::EDGetTokenT<vector<reco::Vertex> > primaryVertexToken_;
      const edm::EDGetTokenT<reco::BeamSpot> beamspotToken_;
      const edm::EDGetTokenT<vector<reco::Conversion> > conversionsToken_;
      const edm::EDGetTokenT<vector<pat::Photon> > photonsToken_;
      const edm::EDGetTokenT<vector<pat::Photon> > ootPhotonsToken_;
      const edm::EDGetTokenT<vector<pat::MET> > METToken_;
      const edm::EDGetTokenT<vector<pat::MET> > puppiMETToken_;
      const edm::EDGetTokenT<edm::TriggerResults> trigResultsToken_;
      const edm::EDGetTokenT<edm::TriggerResults> metFilterResultsToken_;
      const edm::EDGetTokenT<vector<pat::IsolatedTrack> > isoTrackToken_;
      const edm::EDGetTokenT<vector<pat::Muon> > pfRecoMuToken_;
      // Run3 additions
      const edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> ttkToken_;
      const edm::EDGetTokenT<vector<reco::Track> > dsaMuonToken_;
      // Added to allow "RECO" or "PAT" tags
      edm::EDGetTokenT<vector<reco::Conversion> > conversionsAltToken_;
      edm::EDGetTokenT<edm::TriggerResults> metFilterResultsAltToken_;
      edm::EDGetTokenT<vector<pat::IsolatedTrack> > isoTrackAltToken_;

      // Handles
      edm::Handle<vector<pat::Electron> > recoElectronHandle_;
      edm::Handle<vector<pat::Electron> > recoNanoElectronHandle_;
      edm::Handle<vector<pat::Electron> > lowPtElectronHandle_;
      edm::Handle<vector<pat::Electron> >lowPtNanoElectronHandle_;
      edm::Handle<vector<pat::PackedCandidate> > packedPFCandHandle_;
      edm::Handle<vector<pat::Jet> > recoJetHandle_;
      edm::Handle<GenEventInfoProduct> genEvtInfoHandle_;
      edm::Handle<std::vector<PileupSummaryInfo> > pileupInfosHandle_;
      edm::Handle<double> rhoHandle_;
      edm::Handle<vector<reco::GenParticle> > genParticleHandle_;
      edm::Handle<vector<reco::GenJet> > genJetHandle_;
      edm::Handle<vector<reco::GenMET> > genMETHandle_;
      edm::Handle<vector<reco::Vertex> > primaryVertexHandle_;
      edm::Handle<reco::BeamSpot> beamspotHandle_;
      edm::Handle<vector<reco::Conversion> > conversionsHandle_;
      edm::Handle<vector<pat::Photon> > photonsHandle_;
      edm::Handle<vector<pat::Photon> > ootPhotonsHandle_;
      edm::Handle<vector<pat::MET> > METHandle_;
      edm::Handle<vector<pat::MET> > puppiMETHandle_;
      edm::Handle<edm::TriggerResults> trigResultsHandle_;
      edm::Handle<edm::TriggerResults> metFilterResultsHandle_;
      edm::Handle<vector<pat::IsolatedTrack> > isoTrackHandle_;
      edm::Handle<vector<pat::Muon> > pfRecoMuHandle_;
      // Run3 addition
      edm::Handle<vector<reco::Track>> dsaMuonHandle_;
  
      // Trigger variables
      std::vector<std::string> trigPathsWithVersion_;
      std::vector<bool> trigExist_;
      HLTConfigProvider hltConfig_;
      HLTConfigProvider metFilterConfig_;
};
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
ElectronSkimmer::ElectronSkimmer(const edm::ParameterSet& ps)
 :
   isData(ps.getParameter<bool>("isData")),
   isSignal(ps.getParameter<bool>("isSignal")),
   year(ps.getParameter<std::string>("year")),
   triggerProcessName_(ps.getParameter<std::string>("triggerProcessName")),
   metFilterName_(ps.getParameter<std::string>("metFilterName")),
   metFilters_(ps.getParameter<std::vector<std::string> >("metFilters")),
   trigPaths_(ps.getParameter<std::vector<std::string> >("triggerPaths")),
   effectiveAreas_((ps.getParameter<edm::FileInPath>("effAreasConfigFile")).fullPath()),
   recoElectronToken_(consumes<vector<pat::Electron> >(ps.getParameter<edm::InputTag>("recoElectron"))),
   recoNanoElectronToken_(consumes<vector<pat::Electron> >(ps.getParameter<edm::InputTag>("nanoElectron"))),
   lowPtElectronToken_(consumes<vector<pat::Electron> >(ps.getParameter<edm::InputTag>("lowPtElectron"))),
   lowPtNanoElectronToken_(consumes<vector<pat::Electron> >(ps.getParameter<edm::InputTag>("lowPtNanoElectron"))),
   packedPFCandToken_(consumes<vector<pat::PackedCandidate> >(ps.getParameter<edm::InputTag>("pfCands"))),
   recoJetToken_(consumes<vector<pat::Jet> >(ps.getParameter<edm::InputTag>("jets"))),
   genEvtInfoToken_(consumes<GenEventInfoProduct>(ps.getParameter<edm::InputTag>("genEvt"))),
   pileupInfosToken_(consumes<std::vector<PileupSummaryInfo> >(ps.getParameter<edm::InputTag>("pileups"))),
   rhoToken_(consumes<double>(ps.getParameter<edm::InputTag>("rho"))),
   genParticleToken_(consumes<vector<reco::GenParticle> >(ps.getParameter<edm::InputTag>("genParticle"))),
   genJetToken_(consumes<vector<reco::GenJet> >(ps.getParameter<edm::InputTag>("genJet"))),
   genMETToken_(consumes<vector<reco::GenMET> >(ps.getParameter<edm::InputTag>("genMET"))),
   primaryVertexToken_(consumes<vector<reco::Vertex> >(ps.getParameter<edm::InputTag>("primaryVertex"))),
   beamspotToken_(consumes<reco::BeamSpot>(ps.getParameter<edm::InputTag>("beamspot"))),
   conversionsToken_(consumes<vector<reco::Conversion> >(ps.getParameter<edm::InputTag>("conversions"))),
   photonsToken_(consumes<vector<pat::Photon> >(ps.getParameter<edm::InputTag>("photons"))),
   ootPhotonsToken_(consumes<vector<pat::Photon> >(ps.getParameter<edm::InputTag>("ootPhotons"))),
   METToken_(consumes<vector<pat::MET> >(ps.getParameter<edm::InputTag>("MET"))),
   puppiMETToken_(consumes<vector<pat::MET> >(ps.getParameter<edm::InputTag>("puppiMET"))),
   trigResultsToken_(consumes<edm::TriggerResults>(ps.getParameter<edm::InputTag>("trigResults"))),
   metFilterResultsToken_(consumes<edm::TriggerResults>(ps.getParameter<edm::InputTag>("metFilterResults"))),
   isoTrackToken_(consumes<vector<pat::IsolatedTrack> >(ps.getParameter<edm::InputTag>("isoTracks"))),
   pfRecoMuToken_(consumes<vector<pat::Muon> >(ps.getParameter<edm::InputTag>("pfRecoMu"))),
   // Run3 additions
   ttkToken_(esConsumes(edm::ESInputTag{"", "TransientTrackBuilder"})),
   dsaMuonToken_(consumes<vector<reco::Track> >(ps.getParameter<edm::InputTag>("displacedStandAloneMuons"))),
   // Added to allow "RECO" or "PAT" tags
   conversionsAltToken_(mayConsume<vector<reco::Conversion> >(edm::InputTag("reducedEgamma","reducedConversions",
       ps.getParameter<edm::InputTag>("conversions").process() == "PAT" ? "RECO" : "PAT"))),
   metFilterResultsAltToken_(mayConsume<edm::TriggerResults>(edm::InputTag("TriggerResults", "",
       ps.getParameter<edm::InputTag>("metFilterResults").process() == "PAT" ? "RECO" : "PAT"))),
   isoTrackAltToken_(mayConsume<vector<pat::IsolatedTrack> >(edm::InputTag("isolatedTracks","",
       ps.getParameter<edm::InputTag>("isoTracks").process() == "PAT" ? "RECO" : "PAT")))
{
   usesResource("TFileService");
   m_random_generator = std::mt19937(37428479);

}


ElectronSkimmer::~ElectronSkimmer() = default;


//
// member functions
//

void
ElectronSkimmer::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup)
{
   using namespace edm;

   // Set up HLT config
   bool changed = true;
   if (hltConfig_.init(iRun, iSetup, triggerProcessName_, changed)) {
      if (changed) {
         LogInfo("HLTConfig") << "iDMAnalyzer::beginRun: " << "hltConfig init for Run" << iRun.run();
         hltConfig_.dump("ProcessName");
         hltConfig_.dump("GlobalTag");
         hltConfig_.dump("TableName");
      }
   }
   else {
      LogError("HLTConfig") << "iDMAnalyzer::beginRun: config extraction failure with triggerProcessName -> " << triggerProcessName_;
      return;
   }

   if (metFilterConfig_.init(iRun,iSetup,metFilterName_,changed)) {
      if (changed) {
         LogInfo("HLTConfig") << "iDMAnalyzer::beginRun: " << "metFilterConfig init for Run" << iRun.run();
         metFilterConfig_.dump("ProcessName");
         metFilterConfig_.dump("GlobalTag");
         metFilterConfig_.dump("TableName");
      }
   }
   else {
      // Run3 modified to allow "RECO" or "PAT" tags
      std::string altName = (metFilterName_ == "PAT") ? "RECO" : "PAT";
      if (metFilterConfig_.init(iRun,iSetup,altName,changed)) {
         metFilterName_ = altName;
         LogInfo("HLTConfig") << "iDMAnalyzer::beginRun: metFilterConfig init succeeded with alternate process name: " << altName;
      }
      else {
         LogError("HLTConfig") << "iDMAnalyzer::beginRun: config extraction failure for both PAT and RECO process names";
         return;
      }
   }

   // Add trigger paths if they exist
   trigPathsWithVersion_.clear();
   trigExist_.clear();
   
   const std::vector<std::string>& pathNames = hltConfig_.triggerNames();

   /*std::cout << " AVAILABLE TRIGGERS" << std::endl;
   for (auto s : pathNames) {
      std::cout << s << std::endl;
   }*/
   
   // All trigger paths
   for (auto trigPathNoVersion : trigPaths_) {
      auto matchedPaths(hltConfig_.restoreVersion(pathNames, trigPathNoVersion));
      if (matchedPaths.size() == 0) {
         LogWarning("TriggerNotFound") << "Could not find matched full trigger path with --> " << trigPathNoVersion;
         trigPathsWithVersion_.push_back("None");
         trigExist_.push_back(false);
      }
      else {
         trigExist_.push_back(true);
         trigPathsWithVersion_.push_back(matchedPaths[0]);
         if (hltConfig_.triggerIndex(matchedPaths[0]) >= hltConfig_.size()) {
               LogError("TriggerError") << "Cannot find trigger path --> " << matchedPaths[0];
               return;
         }
      }
   }
}


// Run3 displaced muon Id as recommended by Muon POG
bool ElectronSkimmer::passesDisplacedID(const reco::Track& dsaMuon) const {
  float validHits =  dsaMuon.hitPattern().numberOfValidMuonCSCHits() + dsaMuon.hitPattern().numberOfValidMuonDTHits();
  if(validHits > 12){
    if(dsaMuon.hitPattern().numberOfValidMuonCSCHits() != 0 || (dsaMuon.hitPattern().numberOfValidMuonCSCHits() == 0 && dsaMuon.hitPattern().numberOfValidMuonDTHits() > 18)){
      if(dsaMuon.normalizedChi2() < 2.5) {
	if(dsaMuon.ptError()/dsaMuon.pt() < 1){
          return true;
        }
      }
    }
  }
  return false;
}

// ------------ method called once each job just before starting event loop  ------------
void ElectronSkimmer::beginJob()
{
   outT = fs->make<TTree>("outT", "outT");
   nt.isData_ = isData;
   nt.isSignal_ = isSignal;
   nt.SetTree(outT);
   for (size_t i = 0; i < trigPaths_.size(); i++) {
      nt.trigNames_[i] = trigPaths_[i];
      nt.numTrigs_++;
   }
   nt.CreateTreeBranches();
}

// ------------ method called once each job just after ending the event loop  ------------
void ElectronSkimmer::endJob() {}

void ElectronSkimmer::endRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
ElectronSkimmer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
   edm::ParameterSetDescription desc;

   // Inputs from the run_ntuplizer_cfg python (cmsRun inputs)
   desc.add<bool>("isData", 0);
   desc.add<bool>("isSignal",0);
   desc.add<std::string>("year","none"); // placeholder, is updated dynamically later
   desc.add<std::string>("triggerProcessName", "HLT");
   desc.add<std::string>("metFilterName","PAT");
   desc.add<std::vector<std::string> >("metFilters",{});
   desc.add<std::vector<std::string> >("triggerPaths",{});
   desc.add<edm::FileInPath>("effAreasConfigFile");
   desc.add<edm::InputTag>("recoElectron",edm::InputTag("slimmedElectrons"));
   desc.add<edm::InputTag>("nanoElectron",edm::InputTag("slimmedElectronsWithUserDataMinimal"));
   desc.add<edm::InputTag>("lowPtElectron",edm::InputTag("slimmedLowPtElectrons"));
   desc.add<edm::InputTag>("lowPtNanoElectron",edm::InputTag("updatedLowPtElectronsWithUserData"));
   desc.add<edm::InputTag>("pfCands",edm::InputTag("packedPFCandidates"));
   desc.add<edm::InputTag>("jets",edm::InputTag("slimmedJets"));
   desc.add<edm::InputTag>("genEvt", edm::InputTag("generator"));
   desc.add<edm::InputTag>("pileups", edm::InputTag("slimmedAddPileupInfo"));
   desc.add<edm::InputTag>("rho", edm::InputTag("fixedGridRhoFastjetAll"));   
   desc.add<edm::InputTag>("genParticle",edm::InputTag("prunedGenParticles"));
   desc.add<edm::InputTag>("genJet",edm::InputTag("slimmedGenJets"));
   desc.add<edm::InputTag>("genMET",edm::InputTag("genMetTrue"));
   desc.add<edm::InputTag>("primaryVertex",edm::InputTag("offlineSlimmedPrimaryVertices"));
   desc.add<edm::InputTag>("beamspot",edm::InputTag("offlineBeamSpot"));
   desc.add<edm::InputTag>("conversions",edm::InputTag("reducedEgamma","reducedConversions","PAT"));
   desc.add<edm::InputTag>("photons",edm::InputTag("slimmedPhotons"));
   desc.add<edm::InputTag>("ootPhotons",edm::InputTag("slimmedOOTPhotons"));
   desc.add<edm::InputTag>("MET",edm::InputTag("slimmedMETs"));
   desc.add<edm::InputTag>("puppiMET",edm::InputTag("slimmedMETsPuppi"));
   desc.add<edm::InputTag>("trigResults",edm::InputTag("TriggerResults","","HLT"));
   desc.add<edm::InputTag>("metFilterResults",edm::InputTag("TriggerResults","","PAT"));
   desc.add<edm::InputTag>("isoTracks",edm::InputTag("isolatedTracks","","PAT"));
   desc.add<edm::InputTag>("pfRecoMu", edm::InputTag("slimmedMuons"));
   
   // Run3 additions
   desc.add<edm::InputTag>("displacedStandAloneMuons",edm::InputTag("displacedStandAloneMuons"));

   descriptions.add("ElectronSkimmer", desc);
}

// ------------ method called for each event  ------------
void
ElectronSkimmer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using std::cout, std::endl;

   // Retrieving event data and assigning to handles
   iEvent.getByToken(recoElectronToken_,recoElectronHandle_);
   iEvent.getByToken(recoNanoElectronToken_,recoNanoElectronHandle_);
   iEvent.getByToken(lowPtElectronToken_,lowPtElectronHandle_);
   iEvent.getByToken(lowPtNanoElectronToken_,lowPtNanoElectronHandle_);
   iEvent.getByToken(packedPFCandToken_,packedPFCandHandle_);
   iEvent.getByToken(recoJetToken_,recoJetHandle_);
   iEvent.getByToken(pileupInfosToken_,pileupInfosHandle_);
   iEvent.getByToken(rhoToken_,rhoHandle_);
   iEvent.getByToken(primaryVertexToken_,primaryVertexHandle_);
   iEvent.getByToken(beamspotToken_,beamspotHandle_);
   iEvent.getByToken(conversionsToken_,conversionsHandle_);
   iEvent.getByToken(photonsToken_,photonsHandle_);
   iEvent.getByToken(ootPhotonsToken_,ootPhotonsHandle_);
   iEvent.getByToken(METToken_,METHandle_);
   iEvent.getByToken(puppiMETToken_,puppiMETHandle_);
   iEvent.getByToken(trigResultsToken_,trigResultsHandle_);
   iEvent.getByToken(metFilterResultsToken_,metFilterResultsHandle_);
   iEvent.getByToken(isoTrackToken_,isoTrackHandle_);
   iEvent.getByToken(pfRecoMuToken_,pfRecoMuHandle_);
   // Run3 additions
   iEvent.getByToken(dsaMuonToken_,dsaMuonHandle_);
   // Added to allow "RECO" or "PAT" tags
   if (!conversionsHandle_.isValid())
      iEvent.getByToken(conversionsAltToken_,conversionsHandle_);
   if (!metFilterResultsHandle_.isValid())
      iEvent.getByToken(metFilterResultsAltToken_,metFilterResultsHandle_);
   if (!isoTrackHandle_.isValid())
      iEvent.getByToken(isoTrackAltToken_,isoTrackHandle_);
   
   if (!isData) { 
      iEvent.getByToken(genEvtInfoToken_,genEvtInfoHandle_);
      iEvent.getByToken(genParticleToken_,genParticleHandle_);
      iEvent.getByToken(genJetToken_,genJetHandle_);
      iEvent.getByToken(genMETToken_,genMETHandle_);
   }

   // Clear tree branches before filling
   nt.ClearTreeBranches();

   /////////////////////////////////////////////////////////////
   // Computing derived quantities and filling the trees ///////
   /////////////////////////////////////////////////////////////

   // Creating helper function object
   Helper helper;

   // Prearing basic info
   nt.eventNum_ = iEvent.id().event();
   nt.lumiSec_ = iEvent.luminosityBlock();
   nt.runNum_ = iEvent.id().run();
   reco::Vertex pv = (*primaryVertexHandle_).at(0);
   nt.PV_x_ = pv.x();
   nt.PV_y_ = pv.y();
   nt.PV_z_ = pv.z();
      
   double nPV = 0;
   for (const auto & ele : *primaryVertexHandle_) {
     nPV++;
   }
   nt.numPV_ = nPV;

   auto beamspot = *beamspotHandle_;
   // Set up objects for vertex reco - different for Run3
   const TransientTrackBuilder* theB = &iSetup.getData(ttkToken_);
   KalmanVertexFitter kvf(true);

   // MET Filters (as recommended here https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2#UL_data)
   for (size_t i = 0; i < metFilters_.size(); i++) {
      //std::cout << "MET filter " << metFilters_[i] << " is at index " << hltConfig_.triggerIndex(metFilters_[i]) << std::endl;
      nt.METFiltersFailBits_ |= ((!(metFilterResultsHandle_->accept(metFilterConfig_.triggerIndex(metFilters_[i])))) << i);
   }

   // All triggers
   nt.fired_ = 0;
   for (size_t i = 0; i < trigPathsWithVersion_.size(); i++) {
      if (trigExist_.at(i)) {
         std::string trigPath = trigPathsWithVersion_[i];
         nt.fired_ |= (trigResultsHandle_->accept(hltConfig_.triggerIndex(trigPath)) << i);
         nt.trigPassed_[i] = trigResultsHandle_->accept(hltConfig_.triggerIndex(trigPath));
      }
      else {
         nt.fired_ |= (0 <<i);
         nt.trigPassed_[i] = false;
      }
   }

   // Handling MET //
   if (METHandle_->size() > 0) {
      auto met = (*METHandle_).at(0);
      // Current recommendation from JetMET is Type 1 : https://twiki.cern.ch/twiki/bin/view/CMS/MissingET#Recommendations_and_important_li
      auto metType = pat::MET::Type1;
      // PF MET
      nt.PFMET_Pt_ = met.corPt(metType);
      nt.PFMET_Phi_ = met.corPhi(metType);
      nt.PFMET_ET_ = met.corSumEt(metType);
      nt.PFMETJESUpPt_ = met.shiftedPt(pat::MET::JetEnUp,metType);
      nt.PFMETJESUpPhi_ = met.shiftedPhi(pat::MET::JetEnUp,metType);
      nt.PFMETJESDownPt_ = met.shiftedPt(pat::MET::JetEnDown,metType);
      nt.PFMETJESDownPhi_ = met.shiftedPhi(pat::MET::JetEnDown,metType);
      nt.PFMETJERUpPt_ = met.shiftedPt(pat::MET::JetResUp,metType);
      nt.PFMETJERUpPhi_ = met.shiftedPhi(pat::MET::JetResUp,metType);
      nt.PFMETJERDownPt_ = met.shiftedPt(pat::MET::JetResDown,metType);
      nt.PFMETJERDownPhi_ = met.shiftedPhi(pat::MET::JetResDown,metType);
      nt.PFMETUnclusteredUpPt_ = met.shiftedPt(pat::MET::UnclusteredEnUp,metType);
      nt.PFMETUnclusteredUpPhi_ = met.shiftedPhi(pat::MET::UnclusteredEnUp,metType);
      nt.PFMETUnclusteredDownPt_ = met.shiftedPt(pat::MET::UnclusteredEnDown,metType);
      nt.PFMETUnclusteredDownPhi_ = met.shiftedPhi(pat::MET::UnclusteredEnDown,metType);
      
      // Calo MET
      nt.CaloMET_Pt_ = met.caloMETPt();
      nt.CaloMET_Phi_ = met.caloMETPhi();
      nt.CaloMET_ET_ = met.caloMETSumEt();
   }

   // Handling Jets
   for (auto & jet : *recoJetHandle_) {
      nt.PFNJetAll_++;
      if (helper.JetID(jet,year) && jet.pt() > 30) {
         nt.PFNJet_++;
         nt.PFJetPt_.push_back(jet.pt());
         nt.PFJetEta_.push_back(jet.eta());
         nt.PFJetPhi_.push_back(jet.phi());
         auto bTag = jet.bDiscriminator("pfDeepFlavourJetTags:probb") + 

                     jet.bDiscriminator("pfDeepFlavourJetTags:probbb") + 
                     jet.bDiscriminator("pfDeepFlavourJetTags:problepb");
         nt.PFJetBTag_.push_back(bTag);
         // btagging efficiencies
         if (!isData) {
 	   nt.PFJetTruth_.push_back(jet.hadronFlavour());
           if (jet.hadronFlavour() == 5) { 
             nt.PFNbJetTrue_++; 
             nt.PFJetEffDenomPt_.push_back(jet.pt());
 
             float pt_btagEff_num = -999;
 
             if ((year == "2018") && (bTag > 0.2783)) { nt.PFNbJetTagged_++; pt_btagEff_num = jet.pt(); } 
             else if ((year == "2017") && (bTag > 0.3040)) { nt.PFNbJetTagged_++; pt_btagEff_num = jet.pt(); } 
             else if ((year == "2016") && (bTag > 0.2489)) { nt.PFNbJetTagged_++; pt_btagEff_num = jet.pt(); } 
             else if ((year == "2016APV") && (bTag > 0.2598)) { nt.PFNbJetTagged_++; pt_btagEff_num = jet.pt(); }
             nt.PFJetEffNumPt_.push_back(pt_btagEff_num); 
           }
         } 

         // For JEC
         nt.PFJetRawFactor_.push_back(jet.jecFactor("Uncorrected"));
         nt.PFJetMass_.push_back(jet.mass());
         nt.PFJetEnergy_.push_back(jet.energy());
 
         nt.PFJetArea_.push_back(jet.jetArea());
         nt.PFJetPtRaw_.push_back((1 - jet.jecFactor("Uncorrected"))*jet.pt());
         nt.PFJetEnergyRaw_.push_back((1 - jet.jecFactor("Uncorrected"))*jet.energy());
         nt.PFJetMassRaw_.push_back((1 - jet.jecFactor("Uncorrected"))*jet.mass());
 
         nt.fixedGridRhoFastjetAll_ = rhoHandle_.isValid() ? *(rhoHandle_.product()) : -999;

         // For JER
         if (!isData) {
           double min_deltaR = 999;
           double matched_jet_pT = -999;
           double matched_jet_eta = -999;
           double matched_jet_phi = -999;
           for (const auto & genJet : *genJetHandle_) {
             double deltaR = reco::deltaR( genJet.p4(), jet.p4() );
             double deltaPt = fabs( genJet.pt() - jet.pt() );
 
             if (deltaR < min_deltaR) {
               min_deltaR = deltaR;

               if (deltaR < 0.2){
                 matched_jet_pT = genJet.pt();
                 matched_jet_eta = genJet.eta();
                 matched_jet_phi = genJet.phi();
               }
             }
           }
           nt.PFJet_matchedGenJetPt_.push_back(matched_jet_pT);
           nt.PFJet_matchedGenJetEta_.push_back(matched_jet_eta);
           nt.PFJet_matchedGenJetPhi_.push_back(matched_jet_phi);
         } 

         // METdPhi
         nt.PFJetMETdPhi_.push_back(reco::deltaPhi(jet.phi(),nt.PFMET_Phi_));
         if ((jet.pt() > 30) && (jet.eta() > -3.0) && (jet.eta() < -1.3) && (jet.phi() > -1.57) && (jet.phi() < -0.87)) {
            nt.PFHEMFlag_ = true;
         }
      }
   }

   // Record all electrons that are not part of PF -- either regulars that don't pass PF ID
   // or low-pT that aren't reconstructed as PF
   vector<math::XYZTLorentzVector> nonPF_ele_p4s;
  
   // Handling muons: used only for SF measurement with Z/gamma events
   for (const auto & mu : *pfRecoMuHandle_) {
      if (mu.pt() < 3) continue;
      else {
         nt.nMuon_++;
         nt.recoMuonPt_.push_back(mu.pt());     
         nt.recoMuonEta_.push_back(mu.eta());     
         nt.recoMuonPhi_.push_back(mu.phi());     
         nt.recoMuonEnergy_.push_back(mu.energy());
         nt.recoMuonCharge_.push_back(mu.charge());     
         nt.recoMuonIDcutBasedLoose_.push_back(mu.passed('CutBasedIdLoose'));
         nt.recoMuonIDcutBasedMedium_.push_back(mu.passed('CutBasedIdMedium'));
         nt.recoMuonIDcutBasedMediumPrompt_.push_back(mu.passed('CutBasedIdMediumPrompt'));
         nt.recoMuonIDcutBasedTight_.push_back(mu.passed('CutBasedIdTight'));
         nt.recoMuonIsPFMuon_.push_back(mu.isPFMuon());
         nt.recoMuonIsGlobalMuon_.push_back(mu.isGlobalMuon());
         nt.recoMuonIsStandAloneMuon_.push_back(mu.isStandAloneMuon());
      }
   } 
   
   ////////////////////////////////
   // Handling default electrons // 
   ////////////////////////////////
   vector<math::XYZTLorentzVector> reg_ele_p4s;
   vector<const pat::Electron*> reg_good_eles;
   vector<int> iSaved_ele; // record indices of saved electrons for later veto during isolation correction calculations
   int iele = 0;
   for (const auto & ele : *recoNanoElectronHandle_) {
      // require pT > 5 & pass loose ID to consider GED electron
      //if (ele.pt() < 5 || !ele.electronID("cutBasedElectronID-Fall17-94X-V2-loose")) {
      //if (ele.pt() < 2 || !ele.electronID("mvaEleID-Fall17-noIso-V2-wp90")) {
      // Run3 - no pt cut; will change later with ID studies. for now, equivalent of Run2 choice
      if (!ele.electronID("mvaEleID-RunIIIWinter22-noIso-V1-wp90")) {
	 iele++;
         continue;
      }
      iSaved_ele.push_back(iele);
      iele++;
      nt.nElectronDefault_++;
      reco::GsfTrackRef track = ele.gsfTrack();
      reg_ele_p4s.push_back(ele.p4());
      reg_good_eles.push_back(&ele);
      // Filling basic info
      nt.recoElectronPt_.push_back(ele.pt());
      nt.recoElectronEta_.push_back(ele.eta());
      nt.recoElectronEtaError_.push_back(track->etaError());
      nt.recoElectronPhi_.push_back(ele.phi());
      nt.recoElectronPhiError_.push_back(track->phiError());
      nt.recoElectronIsPF_.push_back(ele.isPF());
      nt.recoElectronGenMatched_.push_back(false);
      nt.recoElectronMatchType_.push_back(0);

      // Run3 cut tags updated
      nt.recoElectronID_cutVeto_.push_back(ele.electronID("cutBasedElectronID-RunIIIWinter22-V1-veto"));
      nt.recoElectronID_cutLoose_.push_back(ele.electronID("cutBasedElectronID-RunIIIWinter22-V1-loose"));
      nt.recoElectronID_cutMed_.push_back(ele.electronID("cutBasedElectronID-RunIIIWinter22-V1-medium"));
      nt.recoElectronID_cutTight_.push_back(ele.electronID("cutBasedElectronID-RunIIIWinter22-V1-tight"));
      nt.recoElectronID_cutVetoInt_.push_back(ele.userInt("cutBasedElectronID-RunIIIWinter22-V1-veto"));
      nt.recoElectronID_cutLooseInt_.push_back(ele.userInt("cutBasedElectronID-RunIIIWinter22-V1-loose"));
      nt.recoElectronID_cutMedInt_.push_back(ele.userInt("cutBasedElectronID-RunIIIWinter22-V1-medium"));
      nt.recoElectronID_cutTightInt_.push_back(ele.userInt("cutBasedElectronID-RunIIIWinter22-V1-tight"));
      nt.recoElectronID_mvaIso90_.push_back(ele.electronID("mvaEleID-RunIIIWinter22-iso-V1-wp90"));
      nt.recoElectronID_mvaIso80_.push_back(ele.electronID("mvaEleID-RunIIIWinter22-iso-V1-wp80"));
      // nt.recoElectronID_mvaIsoLoose_.push_back(ele.electronID("mvaEleID-RunIIIWinter22-iso-V1-wpLoose"));
      nt.recoElectronID_mva90_.push_back(ele.electronID("mvaEleID-RunIIIWinter22-noIso-V1-wp90"));
      nt.recoElectronID_mva80_.push_back(ele.electronID("mvaEleID-RunIIIWinter22-noIso-V1-wp80"));
      // nt.recoElectronID_mvaLoose_.push_back(ele.electronID("mvaEleID-RunIIIWinter22-noIso-V1-wpLoose"));
      
      nt.recoElectronAngularRes_.push_back(sqrt(track->phiError()*track->phiError() + track->etaError()*track->etaError()));
      nt.recoElectronE_.push_back(ele.energy());
      nt.recoElectronVxy_.push_back(ele.trackPositionAtVtx().rho());
      nt.recoElectronVz_.push_back(ele.trackPositionAtVtx().z());
      nt.recoElectronTrkIso_.push_back(ele.trackIso());
      nt.recoElectronTrkRelIso_.push_back(ele.trackIso()/ele.pt());
      nt.recoElectronCaloIso_.push_back(ele.caloIso());
      nt.recoElectronCaloRelIso_.push_back(ele.caloIso()/ele.pt());
      nt.recoElectronCharge_.push_back(ele.charge());
      // Calculating "official" dR03 PF Isolation based on https://github.com/cms-sw/cmssw/blob/CMSSW_10_6_X/RecoEgamma/ElectronIdentification/plugins/cuts/GsfEleRelPFIsoScaledCut.cc#L62
      // For Run3: https://github.com/cms-sw/cmssw/blob/CMSSW_13_0_X/RecoEgamma/ElectronIdentification/plugins/cuts/GsfEleRelPFIsoScaledCut.cc
      auto pfIso = ele.pfIsolationVariables();
      const float rho = rhoHandle_.isValid() ? (float)(*rhoHandle_) : 0.0;
      const float eA = effectiveAreas_.getEffectiveArea(std::abs(ele.superCluster()->eta()));
      float iso = pfIso.sumChargedHadronPt + std::max(0.0f,pfIso.sumNeutralHadronEt + pfIso.sumPhotonEt  - rho*eA);
      nt.recoElectronPFIso_.push_back(iso);
      nt.recoElectronPFRelIso_.push_back(iso/ele.pt());
      nt.recoElectronMiniIso_.push_back(ele.pt()*ele.userFloat("miniIsoAll"));
      nt.recoElectronMiniRelIso_.push_back(ele.userFloat("miniIsoAll"));
      // dummy values for corrected isolation
      nt.recoElectronPFIsoEleCorr_.push_back(-999.);
      nt.recoElectronPFRelIsoEleCorr_.push_back(-999.);
      nt.recoElectronMiniIsoEleCorr_.push_back(-999.);
      nt.recoElectronMiniRelIsoEleCorr_.push_back(-999.);
      // Saving individual isolation components
      nt.recoElectronChadIso_.push_back(pfIso.sumChargedHadronPt);
      nt.recoElectronNhadIso_.push_back(pfIso.sumNeutralHadronEt);
      nt.recoElectronPhoIso_.push_back(pfIso.sumPhotonEt);
      nt.recoElectronRhoEA_.push_back(rho*eA);
      // Filling track info
      nt.recoElectronDxy_.push_back(abs(track->dxy(pv.position())));
      nt.recoElectronDxyError_.push_back(track->dxyError());
      nt.recoElectronDz_.push_back(track->dz(pv.position()));
      nt.recoElectronDzError_.push_back(track->dzError());
      nt.recoElectronTrkChi2_.push_back(track->normalizedChi2());
      nt.recoElectronTrkProb_.push_back(TMath::Prob(track->chi2(),(int)track->ndof()));
      nt.recoElectronTrkNumTrackerHits_.push_back(track->hitPattern().numberOfValidTrackerHits());
      nt.recoElectronTrkNumPixHits_.push_back(track->hitPattern().numberOfValidPixelHits());
      nt.recoElectronTrkNumStripHits_.push_back(track->hitPattern().numberOfValidStripHits());
      // Calculating distance to jets
      vector<float> dRtoJets; vector<float> dPhitoJets;
      for (int ij = 0; ij < nt.PFNJet_; ij++) {
         dRtoJets.push_back(sqrt(pow((ele.eta() - nt.PFJetEta_[ij]),2) + pow(reco::deltaPhi(ele.phi(),nt.PFJetPhi_[ij]),2)));
         dPhitoJets.push_back(reco::deltaPhi(ele.phi(),nt.PFJetPhi_[ij]));
      }
      nt.recoElectronDrToJets_.push_back(dRtoJets);
      nt.recoElectronDphiToJets_.push_back(dPhitoJets);
      // Electron ID variables
      nt.recoElectronFull5x5_sigmaIetaIeta_.push_back(ele.full5x5_sigmaIetaIeta());
      float dEtaInSeed = ele.superCluster().isNonnull() && ele.superCluster()->seed().isNonnull() ? ele.deltaEtaSuperClusterTrackAtVtx() - ele.superCluster()->eta() + ele.superCluster()->seed()->eta() : std::numeric_limits<float>::max();
      nt.recoElectronAbsdEtaSeed_.push_back(std::abs(dEtaInSeed));
      nt.recoElectronAbsdPhiIn_.push_back(std::abs(ele.deltaPhiSuperClusterTrackAtVtx()));
      nt.recoElectronHoverE_.push_back(ele.hadronicOverEm());
      const float ecal_energy_inverse = 1.0/ele.ecalEnergy();
      const float eSCoverP = ele.eSuperClusterOverP();
      nt.recoElectronAbs1overEm1overP_.push_back(std::abs(1.0 - eSCoverP)*ecal_energy_inverse);
      constexpr auto missingHitType =reco::HitPattern::MISSING_INNER_HITS;
      nt.recoElectronExpMissingInnerHits_.push_back(ele.gsfTrack()->hitPattern().numberOfLostHits(missingHitType));
      nt.recoElectronConversionVeto_.push_back(!ConversionTools::hasMatchedConversion(ele,*conversionsHandle_,beamspot.position()));
      nt.recoElectronIsEE_.push_back(ele.isEE());
      // x-cleaning study
      nt.recoElectronHasLptMatch_.push_back(false);
      nt.recoElectronLptMatchIdx_.push_back(-999);
      nt.recoElectronHasAllLptMatch_.push_back(false);
      nt.recoElectronAllLptMatchIdx_.push_back(-999);
   }

   /////////////////////////////////
   /// Handling low-pT electrons ///
   /////////////////////////////////
   vector<math::XYZTLorentzVector> lowpt_ele_p4s;
   vector<math::XYZTLorentzVector> allLowPt_ele_p4s;
   vector<const pat::Electron*> lowpt_good_eles;
   int ilpt = 0; // track index (in output tree) of lpt electrons for x-cleaning purposes
   vector<int> iSaved_lpt;
   int ilpt_all = 0;
   for (auto & ele : *lowPtNanoElectronHandle_) {
      // basic cut (should be applied by default in miniAOD stage, but repeating here)
      // Run3 syntax updated - below cuts are legacy sanity check from Run2, likely will change later
      if (ele.pt() < 1 || ele.electronID("ID") < -0.25) {
         ilpt_all++;
         continue;
      }

      // Checking against GED electrons
      float mindR = 999;
      reco::GsfTrackRef track = ele.gsfTrack();
      float PFmatch_threshold = 0.05; // dR threshold for throwing away low-pT electron in favor of PF electron
      int iMatch_reg;
      for (size_t ireg = 0; ireg < reg_good_eles.size(); ireg++) {
         float dR = reco::deltaR(ele.p4(), reg_good_eles[ireg]->p4());
         if (dR < mindR) {
            mindR = dR;
            iMatch_reg = ireg;
         }
      }
      bool isXCleaned = (mindR < PFmatch_threshold);

      if (isXCleaned) {
         nt.recoElectronHasAllLptMatch_[iMatch_reg] = true;
         nt.recoElectronAllLptMatchIdx_[iMatch_reg] = nt.nElectronAllLowPt_;
      }

      // Fill AllLowPt branches for every LowPt electron regardless of XC status
      nt.nElectronAllLowPt_++;
      allLowPt_ele_p4s.push_back(ele.p4());
      nt.recoAllLowPtElectronIsXCleaned_.push_back(isXCleaned);
      nt.recoAllLowPtElectronGEDidx_.push_back(isXCleaned ? iMatch_reg : -999);
      nt.recoAllLowPtElectronMinDrToReg_.push_back(mindR);
      nt.recoAllLowPtElectronPt_.push_back(ele.pt());
      nt.recoAllLowPtElectronPhi_.push_back(ele.phi());
      nt.recoAllLowPtElectronPhiError_.push_back(track->phiError());
      nt.recoAllLowPtElectronEta_.push_back(ele.eta());
      nt.recoAllLowPtElectronEtaError_.push_back(track->etaError());
      nt.recoAllLowPtElectronIsPF_.push_back(ele.isPF());
      nt.recoAllLowPtElectronGenMatched_.push_back(false);
      nt.recoAllLowPtElectronMatchType_.push_back(0);
      nt.recoAllLowPtElectronGEDisMatched_.push_back(false);
      // Run3 syntax updated
      nt.recoAllLowPtElectronID_.push_back(ele.electronID("ID"));
      nt.recoAllLowPtElectronAngularRes_.push_back(sqrt(track->phiError()*track->phiError() + track->etaError()*track->etaError()));
      nt.recoAllLowPtElectronE_.push_back(ele.energy());
      nt.recoAllLowPtElectronVxy_.push_back(ele.trackPositionAtVtx().rho());
      nt.recoAllLowPtElectronVz_.push_back(ele.trackPositionAtVtx().z());
      nt.recoAllLowPtElectronTrkIso_.push_back(ele.trackIso());
      nt.recoAllLowPtElectronTrkRelIso_.push_back(ele.trackIso()/ele.pt());
      nt.recoAllLowPtElectronCaloIso_.push_back(ele.caloIso());
      nt.recoAllLowPtElectronCaloRelIso_.push_back(ele.caloIso()/ele.pt());
      nt.recoAllLowPtElectronCharge_.push_back(ele.charge());
      // Calculating "official" dR03 PF Isolation based on https://github.com/cms-sw/cmssw/blob/CMSSW_10_6_X/RecoEgamma/ElectronIdentification/plugins/cuts/GsfEleRelPFIsoScaledCut.cc#L62
      {
         auto pfIso = ele.pfIsolationVariables();
         const float rho = rhoHandle_.isValid() ? (float)(*rhoHandle_) : 0.0;
         const float eA = effectiveAreas_.getEffectiveArea(std::abs(ele.superCluster()->eta()));
         float iso = pfIso.sumChargedHadronPt + std::max(0.0f,pfIso.sumNeutralHadronEt + pfIso.sumPhotonEt  - rho*eA);
         nt.recoAllLowPtElectronPFIso_.push_back(iso);
         nt.recoAllLowPtElectronPFRelIso_.push_back(iso/ele.pt());
         nt.recoAllLowPtElectronMiniIso_.push_back(ele.pt()*ele.userFloat("miniIsoAll"));
         nt.recoAllLowPtElectronMiniRelIso_.push_back(ele.userFloat("miniIsoAll"));
         // dummy values for corrected isolation
         nt.recoAllLowPtElectronPFIsoEleCorr_.push_back(-999.);
         nt.recoAllLowPtElectronPFRelIsoEleCorr_.push_back(-999.);
         nt.recoAllLowPtElectronMiniIsoEleCorr_.push_back(-999.);
         nt.recoAllLowPtElectronMiniRelIsoEleCorr_.push_back(-999.);
         // Saving individual isolation components
         nt.recoAllLowPtElectronChadIso_.push_back(pfIso.sumChargedHadronPt);
         nt.recoAllLowPtElectronNhadIso_.push_back(pfIso.sumNeutralHadronEt);
         nt.recoAllLowPtElectronPhoIso_.push_back(pfIso.sumPhotonEt);
         nt.recoAllLowPtElectronRhoEA_.push_back(rho*eA);
      }
      // Filling tracks
      nt.recoAllLowPtElectronDxy_.push_back(abs(track->dxy(pv.position())));
      nt.recoAllLowPtElectronDxyError_.push_back(track->dxyError());
      nt.recoAllLowPtElectronDz_.push_back(track->dz(pv.position()));
      nt.recoAllLowPtElectronDzError_.push_back(track->dzError());
      nt.recoAllLowPtElectronTrkChi2_.push_back(track->normalizedChi2());
      nt.recoAllLowPtElectronTrkProb_.push_back(TMath::Prob(track->chi2(),(int)track->ndof()));
      nt.recoAllLowPtElectronTrkNumTrackerHits_.push_back(track->hitPattern().numberOfValidTrackerHits());
      nt.recoAllLowPtElectronTrkNumPixHits_.push_back(track->hitPattern().numberOfValidPixelHits());
      nt.recoAllLowPtElectronTrkNumStripHits_.push_back(track->hitPattern().numberOfValidStripHits());
      // Calculating distance to jets
      {
         vector<float> dRtoJets; vector<float> dPhitoJets;
         for (int ij = 0; ij < nt.PFNJet_; ij++) {
            dRtoJets.push_back(sqrt(pow(ele.eta() - nt.PFJetEta_[ij],2) + pow(reco::deltaPhi(ele.phi(),nt.PFJetPhi_[ij]),2)));
            dPhitoJets.push_back(reco::deltaPhi(ele.phi(),nt.PFJetPhi_[ij]));
         }
         nt.recoAllLowPtElectronDrToJets_.push_back(dRtoJets);
         nt.recoAllLowPtElectronDphiToJets_.push_back(dPhitoJets);
      }
      // Electron ID variables
      nt.recoAllLowPtElectronFull5x5_sigmaIetaIeta_.push_back(ele.full5x5_sigmaIetaIeta());
      {
         float dEtaInSeed = ele.superCluster().isNonnull() && ele.superCluster()->seed().isNonnull() ? ele.deltaEtaSuperClusterTrackAtVtx() - ele.superCluster()->eta() + ele.superCluster()->seed()->eta() : std::numeric_limits<float>::max();
         nt.recoAllLowPtElectronAbsdEtaSeed_.push_back(std::abs(dEtaInSeed));
      }
      nt.recoAllLowPtElectronAbsdPhiIn_.push_back(std::abs(ele.deltaPhiSuperClusterTrackAtVtx()));
      nt.recoAllLowPtElectronHoverE_.push_back(ele.hadronicOverEm());
      {
         const float ecal_energy_inverse = 1.0/ele.ecalEnergy();
         const float eSCoverP = ele.eSuperClusterOverP();
         nt.recoAllLowPtElectronAbs1overEm1overP_.push_back(std::abs(1.0 - eSCoverP)*ecal_energy_inverse);
      }
      {
         constexpr auto missingHitType = reco::HitPattern::MISSING_INNER_HITS;
         nt.recoAllLowPtElectronExpMissingInnerHits_.push_back(ele.gsfTrack()->hitPattern().numberOfLostHits(missingHitType));
      }
      nt.recoAllLowPtElectronConversionVeto_.push_back(!ConversionTools::hasMatchedConversion(ele,*conversionsHandle_,beamspot.position()));
      nt.recoAllLowPtElectronIsEE_.push_back(ele.isEE());

      if (!isXCleaned) {
         // passes cross cleaning — fill surviving LowPt branches
         nt.recoLowPtElectronIsXCleaned_.push_back(false);
         nt.recoLowPtElectronGEDidx_.push_back(-999);

         ilpt++;
         iSaved_lpt.push_back(ilpt_all);

         nt.nElectronLowPt_++;
         nt.recoLowPtElectronMinDrToReg_.push_back(mindR);
         lowpt_ele_p4s.push_back(ele.p4());
         lowpt_good_eles.push_back(&ele);
         nt.recoLowPtElectronPt_.push_back(ele.pt());
         nt.recoLowPtElectronPhi_.push_back(ele.phi());
         nt.recoLowPtElectronPhiError_.push_back(track->phiError());
         nt.recoLowPtElectronEta_.push_back(ele.eta());
         nt.recoLowPtElectronEtaError_.push_back(track->etaError());
         nt.recoLowPtElectronIsPF_.push_back(ele.isPF());
         nt.recoLowPtElectronGenMatched_.push_back(false);
         nt.recoLowPtElectronMatchType_.push_back(0);
         // Run3 syntax updated
         nt.recoLowPtElectronID_.push_back(ele.electronID("ID"));
         nt.recoLowPtElectronAngularRes_.push_back(sqrt(track->phiError()*track->phiError() + track->etaError()*track->etaError()));
         nt.recoLowPtElectronE_.push_back(ele.energy());
         nt.recoLowPtElectronVxy_.push_back(ele.trackPositionAtVtx().rho());
         nt.recoLowPtElectronVz_.push_back(ele.trackPositionAtVtx().z());
         nt.recoLowPtElectronTrkIso_.push_back(ele.trackIso());
         nt.recoLowPtElectronTrkRelIso_.push_back(ele.trackIso()/ele.pt());
         nt.recoLowPtElectronCaloIso_.push_back(ele.caloIso());
         nt.recoLowPtElectronCaloRelIso_.push_back(ele.caloIso()/ele.pt());
         nt.recoLowPtElectronCharge_.push_back(ele.charge());
         // Calculating "official" dR03 PF Isolation based on https://github.com/cms-sw/cmssw/blob/CMSSW_10_6_X/RecoEgamma/ElectronIdentification/plugins/cuts/GsfEleRelPFIsoScaledCut.cc#L62
         auto pfIso = ele.pfIsolationVariables();
         const float rho = rhoHandle_.isValid() ? (float)(*rhoHandle_) : 0.0;
         const float eA = effectiveAreas_.getEffectiveArea(std::abs(ele.superCluster()->eta()));
         float iso = pfIso.sumChargedHadronPt + std::max(0.0f,pfIso.sumNeutralHadronEt + pfIso.sumPhotonEt  - rho*eA);
         nt.recoLowPtElectronPFIso_.push_back(iso);
         nt.recoLowPtElectronPFRelIso_.push_back(iso/ele.pt());
         nt.recoLowPtElectronMiniIso_.push_back(ele.pt()*ele.userFloat("miniIsoAll"));
         nt.recoLowPtElectronMiniRelIso_.push_back(ele.userFloat("miniIsoAll"));
         // dummy values for corrected isolation
         nt.recoLowPtElectronPFIsoEleCorr_.push_back(-999.);
         nt.recoLowPtElectronPFRelIsoEleCorr_.push_back(-999.);
         nt.recoLowPtElectronMiniIsoEleCorr_.push_back(-999.);
         nt.recoLowPtElectronMiniRelIsoEleCorr_.push_back(-999.);
         // Saving individual isolation components
         nt.recoLowPtElectronChadIso_.push_back(pfIso.sumChargedHadronPt);
         nt.recoLowPtElectronNhadIso_.push_back(pfIso.sumNeutralHadronEt);
         nt.recoLowPtElectronPhoIso_.push_back(pfIso.sumPhotonEt);
         nt.recoLowPtElectronRhoEA_.push_back(rho*eA);
         // Filling tracks
         nt.recoLowPtElectronDxy_.push_back(abs(track->dxy(pv.position())));
         nt.recoLowPtElectronDxyError_.push_back(track->dxyError());
         nt.recoLowPtElectronDz_.push_back(track->dz(pv.position()));
         nt.recoLowPtElectronDzError_.push_back(track->dzError());
         nt.recoLowPtElectronTrkChi2_.push_back(track->normalizedChi2());
         nt.recoLowPtElectronTrkProb_.push_back(TMath::Prob(track->chi2(),(int)track->ndof()));
         nt.recoLowPtElectronTrkNumTrackerHits_.push_back(track->hitPattern().numberOfValidTrackerHits());
         nt.recoLowPtElectronTrkNumPixHits_.push_back(track->hitPattern().numberOfValidPixelHits());
         nt.recoLowPtElectronTrkNumStripHits_.push_back(track->hitPattern().numberOfValidStripHits());
         // Calculating distance to jets
         vector<float> dRtoJets; vector<float> dPhitoJets;
         for (int ij = 0; ij < nt.PFNJet_; ij++) {
            dRtoJets.push_back(sqrt(pow(ele.eta() - nt.PFJetEta_[ij],2) + pow(reco::deltaPhi(ele.phi(),nt.PFJetPhi_[ij]),2)));
            dPhitoJets.push_back(reco::deltaPhi(ele.phi(),nt.PFJetPhi_[ij]));
         }
         nt.recoLowPtElectronDrToJets_.push_back(dRtoJets);
         nt.recoLowPtElectronDphiToJets_.push_back(dPhitoJets);
         // Electron ID variables
         nt.recoLowPtElectronFull5x5_sigmaIetaIeta_.push_back(ele.full5x5_sigmaIetaIeta());
         float dEtaInSeed = ele.superCluster().isNonnull() && ele.superCluster()->seed().isNonnull() ? ele.deltaEtaSuperClusterTrackAtVtx() - ele.superCluster()->eta() + ele.superCluster()->seed()->eta() : std::numeric_limits<float>::max();
         nt.recoLowPtElectronAbsdEtaSeed_.push_back(std::abs(dEtaInSeed));
         nt.recoLowPtElectronAbsdPhiIn_.push_back(std::abs(ele.deltaPhiSuperClusterTrackAtVtx()));
         nt.recoLowPtElectronHoverE_.push_back(ele.hadronicOverEm());
         const float ecal_energy_inverse = 1.0/ele.ecalEnergy();
         const float eSCoverP = ele.eSuperClusterOverP();
         nt.recoLowPtElectronAbs1overEm1overP_.push_back(std::abs(1.0 - eSCoverP)*ecal_energy_inverse);
         constexpr auto missingHitType =reco::HitPattern::MISSING_INNER_HITS;
         nt.recoLowPtElectronExpMissingInnerHits_.push_back(ele.gsfTrack()->hitPattern().numberOfLostHits(missingHitType));
         nt.recoLowPtElectronConversionVeto_.push_back(!ConversionTools::hasMatchedConversion(ele,*conversionsHandle_,beamspot.position()));
         nt.recoLowPtElectronIsEE_.push_back(ele.isEE());
         // additional x-cleaning study variables
         nt.recoLowPtElectronGEDisMatched_.push_back(false);
      }

      ilpt_all++;
   }

   // Handling DSA Muons
   std::vector<reco::Track> dsa_muonTracks{};
   std::vector<math::XYZTLorentzVector> dsa_muon_p4s;

   for (const auto & track : *dsaMuonHandle_) {
      dsa_muonTracks.push_back(track);
      nt.nDSAMuon_++;
      // Construct TLorentzVector from track (muon mass assumed)
      float mass = 0.10566; // GeV
      float p = track.p();
      float energy = sqrt(p*p + mass*mass);
      math::XYZTLorentzVector p4(track.px(), track.py(), track.pz(), energy);
      dsa_muon_p4s.push_back(p4);

      // Basic kinematics
      nt.recoDSAMuonPt_.push_back(track.pt());
      nt.recoDSAMuonEta_.push_back(track.eta());
      nt.recoDSAMuonPhi_.push_back(track.phi());
      nt.recoDSAMuonE_.push_back(energy);
      nt.recoDSAMuonPx_.push_back(track.px());
      nt.recoDSAMuonPy_.push_back(track.py());
      nt.recoDSAMuonPz_.push_back(track.pz());

      // Vertex info
      nt.recoDSAMuonVxy_.push_back(track.vertex().rho());
      nt.recoDSAMuonVz_.push_back(track.vertex().z());

      // Tracking info
      nt.recoDSAMuonDxy_.push_back(track.dxy(pv.position()));
      nt.recoDSAMuonDxyError_.push_back(track.dxyError());
      nt.recoDSAMuonDz_.push_back(track.dz(pv.position()));
      nt.recoDSAMuonDzError_.push_back(track.dzError());
      nt.recoDSAMuonTrkChi2_.push_back(track.normalizedChi2());
      nt.recoDSAMuonTrkProb_.push_back(TMath::Prob(track.chi2(), (int)track.ndof()));
      nt.recoDSAMuonTrkNumTrackerHits_.push_back(track.hitPattern().numberOfValidTrackerHits());
      nt.recoDSAMuonTrkNumPixHits_.push_back(track.hitPattern().numberOfValidPixelHits());
      nt.recoDSAMuonTrkNumStripHits_.push_back(track.hitPattern().numberOfValidStripHits());

      // Charge
      nt.recoDSAMuonCharge_.push_back(track.charge());
      nt.recoDSAMuonTrkNumCSCHits_.push_back(track.hitPattern().numberOfValidMuonCSCHits());
      nt.recoDSAMuonTrkNumDTHits_.push_back(track.hitPattern().numberOfValidMuonDTHits());
      nt.recoDSAMuonTrkNumHits_.push_back(track.hitPattern().numberOfValidMuonHits());
      nt.recoDSAMuonTrkNumPlanes_.push_back(track.hitPattern().muonStationsWithValidHits());


      int passesDisplacedId = 0;
      if (passesDisplacedID(track)) {
	passesDisplacedId = 1;
      }
      nt.recoDSAMuonDisplacedId_.push_back(passesDisplacedId);
   }   
      
   // computing dR between low-pT and GED electrons for *all* electrons in each collection.
   // to be used for determining whether any lpt electron is x-cleaned (so it can be neglected when
   // computing the isolation corrections using electrons in the event)
   vector<bool> allLptEles_isXcleaned;
   for (auto & ele : *lowPtNanoElectronHandle_) {
      float mindR = 999;
      reco::GsfTrackRef track = ele.gsfTrack();
      float PFmatch_threshold = 0.05; // dR threshold for throwing away low-pT electron in favor of PF electron
      //int iMatch_reg;
      for (auto & ele2 : *recoNanoElectronHandle_) {
         float dR = reco::deltaR(ele.p4(), ele2.p4());
         if (dR < mindR) {
            mindR = dR;
         }
      }
      if (mindR < PFmatch_threshold) {
         allLptEles_isXcleaned.push_back(true);
      }
      else {
         allLptEles_isXcleaned.push_back(false);
      }
   }

   // Computing corrections to PFIso and MiniIso
   float mindr = 0.05; float maxdr = 0.2; float kt_scale = 10.0; // for miniIso
   // correcting for regular electrons
   for (size_t i = 0; i < reg_good_eles.size(); i++) {
      auto ele = *(reg_good_eles[i]);
      float R_pf = 0.3;
      float R_mini = std::max(mindr, std::min(maxdr, float(kt_scale / ele.pt())));
      float pfIsoCorrection = 0.0;
      float miniIsoCorrection = 0.0;
      for (size_t iged = 0; iged < recoNanoElectronHandle_->size(); iged++) {
         if ((ele.isEE()) && (iSaved_ele[i] == (int)iged)) continue; // have deadcone rejection in EE         
         auto cand_ele = (*recoNanoElectronHandle_)[iged];
         if (cand_ele.isPF()) continue;
         float dR = reco::deltaR(ele.p4(),cand_ele.p4());
         if (dR < R_pf) {
            pfIsoCorrection += (*cand_ele.gsfTrack()).pt();
         }
         if (dR < R_mini) {
            miniIsoCorrection += (*cand_ele.gsfTrack()).pt();
         }
      }
      for (size_t il = 0; il < lowPtNanoElectronHandle_->size(); il++) {
	 // even when the cross cleaning is removed, this still needs to be present due to overlap removal in the isolation calculation
	 if (allLptEles_isXcleaned[il]) continue;
         auto cand_ele = (*lowPtNanoElectronHandle_)[il];
         float dR = reco::deltaR(ele.p4(),cand_ele.p4());
         if (dR < R_pf) {
            pfIsoCorrection += (*cand_ele.gsfTrack()).pt();
         }
         if (dR < R_mini) {
            miniIsoCorrection += (*cand_ele.gsfTrack()).pt();
         }
      }
      float isoCutoff = 0.0;

      float pfIsoCorr = nt.recoElectronPFIso_[i] - pfIsoCorrection;
      nt.recoElectronPFIsoEleCorr_[i] = std::max(pfIsoCorr,isoCutoff);
      nt.recoElectronPFRelIsoEleCorr_[i] = nt.recoElectronPFIsoEleCorr_[i]/ele.pt();

      float miniIsoCorr = nt.recoElectronMiniIso_[i] - miniIsoCorrection;
      nt.recoElectronMiniIsoEleCorr_[i] = std::max(miniIsoCorr,isoCutoff);
      nt.recoElectronMiniRelIsoEleCorr_[i] = nt.recoElectronMiniIsoEleCorr_[i]/ele.pt();
   }

   // correcting for low-pt electrons
   for (size_t i = 0; i < lowpt_good_eles.size(); i++) {
      auto ele = *(lowpt_good_eles[i]);
      float R_pf = 0.3;
      float R_mini = std::max(mindr, std::min(maxdr, float(kt_scale / ele.pt())));
      float pfIsoCorrection = 0.0;
      float miniIsoCorrection = 0.0;
      for (size_t iged = 0; iged < recoNanoElectronHandle_->size(); iged++) {
         auto cand_ele = (*recoNanoElectronHandle_)[iged];
         if (cand_ele.isPF()) continue;
         float dR = reco::deltaR(ele.p4(),cand_ele.p4());
         if (dR < R_pf) {
            pfIsoCorrection += (*cand_ele.gsfTrack()).pt();
         }
         if (dR < R_mini) {
            miniIsoCorrection += (*cand_ele.gsfTrack()).pt();
         }
      }
      for (size_t il = 0; il < lowPtNanoElectronHandle_->size(); il++) {
         if ((ele.isEE()) && (iSaved_lpt[i] == (int)il)) continue; // have deadcone rejection in EE         
	 // even when the cross cleaning is removed, this still needs to be present due to overlap removal in the isolation calculation
         if (allLptEles_isXcleaned[il]) continue;
         auto cand_ele = (*lowPtNanoElectronHandle_)[il];
         float dR = reco::deltaR(ele.p4(),cand_ele.p4());
         if (dR < R_pf) {
            pfIsoCorrection += (*cand_ele.gsfTrack()).pt();
         }
         if (dR < R_mini) {
            miniIsoCorrection += (*cand_ele.gsfTrack()).pt();
         }
      }
      float isoCutoff = 0.0;

      float pfIsoCorr = nt.recoLowPtElectronPFIso_[i] - pfIsoCorrection;
      nt.recoLowPtElectronPFIsoEleCorr_[i] = std::max(pfIsoCorr,isoCutoff);
      nt.recoLowPtElectronPFRelIsoEleCorr_[i] = nt.recoLowPtElectronPFIsoEleCorr_[i]/ele.pt();

      float miniIsoCorr = nt.recoLowPtElectronMiniIso_[i] - miniIsoCorrection;
      nt.recoLowPtElectronMiniIsoEleCorr_[i] = std::max(miniIsoCorr,isoCutoff);
      nt.recoLowPtElectronMiniRelIsoEleCorr_[i] = nt.recoLowPtElectronMiniIsoEleCorr_[i]/ele.pt();
   }
   
   // Handling photons
   for (const auto & ph : *photonsHandle_) {
      nt.nPhotons_++;
      nt.PhotonEt_.push_back(ph.et());
      nt.PhotonEta_.push_back(ph.eta());
      nt.PhotonPhi_.push_back(ph.phi());
   }

   // Handling OOT photons
   for (const auto & ph : *ootPhotonsHandle_) {
      nt.nOOTPhotons_++;
      nt.ootPhotonEt_.push_back(ph.et());
      nt.ootPhotonEta_.push_back(ph.eta());
      nt.ootPhotonPhi_.push_back(ph.phi());
   }

   /*std::cout << "filling conversions" << std::endl;
   for (const auto & conv : *conversionsHandle_) {
      if (conv.nTracks() < 2) continue;
      nt.nConversions_++;

      // fitted pair momentum
      auto conv_p4 = conv.refittedPair4Momentum();
      nt.conversionPt_.push_back(conv_p4.pt());
      nt.conversionEta_.push_back(conv_p4.eta());
      nt.conversionPhi_.push_back(conv_p4.phi());
      nt.conversionE_.push_back(conv_p4.E());
      nt.conversionPx_.push_back(conv_p4.px());
      nt.conversionPy_.push_back(conv_p4.py());
      nt.conversionPz_.push_back(conv_p4.py());

      // conversion vertex info
      auto conv_vtx = conv.conversionVertex();
      nt.conversionVxy_.push_back(sqrt(conv_vtx.x()*conv_vtx.x() + conv_vtx.y()*conv_vtx.y()));
      nt.conversionVz_.push_back(conv_vtx.z());
      nt.conversionX_.push_back(conv_vtx.x());
      nt.conversionY_.push_back(conv_vtx.y());
      nt.conversionZ_.push_back(conv_vtx.z());

      // conversion lxy/lz/dxy/dz
      nt.conversionLxy_.push_back(conv.lxy());
      nt.conversionLz_.push_back(conv.lz());
      nt.conversionLxyPV_.push_back(conv.lxy(pv.position()));
      nt.conversionLzPV_.push_back(conv.lz(pv.position()));
      nt.conversionDxy_.push_back(conv.dxy());
      nt.conversionDz_.push_back(conv.dz());
      nt.conversionDxyPV_.push_back(conv.dxy(pv.position()));
      nt.conversionDzPV_.push_back(conv.dz(pv.position()));

      // other conversion properties
      nt.conversionEoverP_.push_back(conv.EoverP());
      nt.conversionEoverPrefit_.push_back(conv.EoverPrefittedTracks());
      nt.conversionNSharedHits_.push_back(conv.nSharedHits());
      nt.conversionM_.push_back(conv.pairInvariantMass());
      nt.conversionChi2_.push_back(conv_vtx.normalizedChi2());

      auto t1 = *(conv.tracks().at(0));
      auto t2 = *(conv.tracks().at(1));
      nt.conversionDr_.push_back(reco::deltaR(t1,t2));

      nt.conversion_Trk1nHitsVtx_.push_back(conv.nHitsBeforeVtx().at(0));
      nt.conversion_Trk1Pt_.push_back(t1.pt());
      nt.conversion_Trk1Eta_.push_back(t1.eta());
      nt.conversion_Trk1Phi_.push_back(t1.phi());
      nt.conversion_Trk1Chi2_.push_back(t1.normalizedChi2());
      nt.conversion_Trk1NValidHits_.push_back(t1.numberOfValidHits());
      nt.conversion_Trk1numLostHits_.push_back(t1.numberOfLostHits());
      nt.conversion_Trk1dxy_.push_back(t1.dxy());
      nt.conversion_Trk1dxyPV_.push_back(t1.dxy(pv.position()));
      nt.conversion_Trk1dxyBS_.push_back(t1.dxy(beamspot));
      nt.conversion_Trk1dz_.push_back(t1.dz());
      nt.conversion_Trk1dzPV_.push_back(t1.dz(pv.position()));

      nt.conversion_Trk2nHitsVtx_.push_back(conv.nHitsBeforeVtx().at(1));
      nt.conversion_Trk2Pt_.push_back(t2.pt());
      nt.conversion_Trk2Eta_.push_back(t2.eta());
      nt.conversion_Trk2Phi_.push_back(t2.phi());
      nt.conversion_Trk2Chi2_.push_back(t2.normalizedChi2());
      nt.conversion_Trk2NValidHits_.push_back(t2.numberOfValidHits());
      nt.conversion_Trk2numLostHits_.push_back(t2.numberOfLostHits());
      nt.conversion_Trk2dxy_.push_back(t2.dxy());
      nt.conversion_Trk2dxyPV_.push_back(t2.dxy(pv.position()));
      nt.conversion_Trk2dxyBS_.push_back(t2.dxy(beamspot));
      nt.conversion_Trk2dz_.push_back(t2.dz());
      nt.conversion_Trk2dzPV_.push_back(t2.dz(pv.position()));      
   }*/

   // Define vertex reco function 
   auto computeVertices = [&](vector<const pat::Electron*> coll_1, vector<const pat::Electron*> coll_2, std::string type1, std::string type2) {
      for (size_t i = 0; i < coll_1.size(); i++) {
         for (size_t j = 0; j < coll_2.size(); j++) {
            if ( (type1==type2) && (j <= i) ) continue; // don't vertex ele with itself or ones prior (if vertexing with same type)
            
            // don't vertex a GED electron with a matching low-pT (only for x-clean study where we keep xcleaned lpt)
	    // even if the cross cleaning is removed; this part needs to be done because you dont want to vertex an electron with itself
            if (type1 == "L" && type2 == "R") {
               if (nt.recoLowPtElectronIsXCleaned_[i]) continue; // nested if b/c will error if checking condition with i > n_lpt 
            }
            if (type1 == "R" && type2 == "L") {
               if (nt.recoLowPtElectronIsXCleaned_[j]) continue; // nested if b/c will error if checking condition with j > n_lpt 
            }

            pat::Electron ei = *coll_1[i];
            pat::Electron ej = *coll_2[j];
            math::XYZTLorentzVector ll = ei.p4() + ej.p4();
            reco::GsfTrackRef ele_i = ei.gsfTrack();
            reco::GsfTrackRef ele_j = ej.gsfTrack();
            if (ele_i == ele_j) continue; // skip if same ele is in reg and low-pT collections
            if (!ele_i.isNonnull() || !ele_j.isNonnull()) continue; // skip if there's a bad track
            if (reco::deltaR(ei,ej) < 0.01) continue; // skip if they're likely to be the same electron un-cross-cleaned

            TransientVertex tv;
            vector<reco::TransientTrack> transient_tracks{};
            transient_tracks.push_back(theB->build(ele_i));
            transient_tracks.push_back(theB->build(ele_j));
            tv = kvf.vertex(transient_tracks);

            if (!tv.isValid()) continue; // skip if the vertex is bad

            reco::Vertex vertex = reco::Vertex(tv);
            float vx = vertex.x(); 
            float vy = vertex.y(); 
            float vz = vertex.z();
            float vxy = sqrt(vertex.x()*vertex.x() + vertex.y()*vertex.y());
            float sigma_vxy = (1/vxy)*sqrt(vertex.x()*vertex.x()*vertex.xError()*vertex.xError() +
                     vertex.y()*vertex.y()*vertex.yError()*vertex.yError());
            float vtx_chi2 = vertex.normalizedChi2();
            float vtx_prob = TMath::Prob(vertex.chi2(),(int)vertex.ndof());
            float dr = reco::deltaR(ei,ej);
            std::string vtxType = type1+type2;
            float dxy1 = (type1 == "R") ? nt.recoElectronDxy_[i] : nt.recoLowPtElectronDxy_[i];
            float dxy2 = (type2 == "R") ? nt.recoElectronDxy_[j] : nt.recoLowPtElectronDxy_[j];
            float mindxy = std::min(abs(dxy1),abs(dxy2));

            nt.vtx_type_.push_back(vtxType);
            nt.vtx_recoVtxReducedChi2_.push_back(vtx_chi2);
            nt.vtx_prob_.push_back(vtx_prob);
            nt.vtx_recoVtxVxy_.push_back(vxy);
            nt.vtx_recoVtxSigmaVxy_.push_back(sigma_vxy);
            nt.vtx_recoVtxVx_.push_back(vx);
            nt.vtx_recoVtxVy_.push_back(vy);
            nt.vtx_recoVtxVz_.push_back(vz);
            nt.vtx_recoVtxDr_.push_back(dr);
            nt.vtx_recoVtxSign_.push_back(ei.charge()*ej.charge());
            nt.vtx_minDxy_.push_back(mindxy);
            nt.vtx_METdPhi_.push_back(reco::deltaPhi(ll.phi(),nt.PFMET_Phi_));
            nt.vtx_ll_pt_.push_back(ll.pt());
            nt.vtx_ll_eta_.push_back(ll.eta());
            nt.vtx_ll_phi_.push_back(ll.phi());
            nt.vtx_ll_e_.push_back(ll.e());
            nt.vtx_ll_m_.push_back(ll.M());
            nt.vtx_ll_px_.push_back(ll.px());
            nt.vtx_ll_py_.push_back(ll.py());
            nt.vtx_ll_pz_.push_back(ll.pz());
            nt.vtx_isMatched_.push_back(false);
            nt.vtx_matchSign_.push_back(0);
            
            nt.vtx_e1_type_.push_back(type1);
            nt.vtx_e1_idx_.push_back(i);
            nt.vtx_e1_isMatched_.push_back(false);
            nt.vtx_e1_matchType_.push_back(0);
            nt.vtx_e2_type_.push_back(type2);
            nt.vtx_e2_idx_.push_back(j);
            nt.vtx_e2_isMatched_.push_back(false);
            nt.vtx_e2_matchType_.push_back(0);

            // Calculating distance to jets
            vector<float> dRtoJets; vector<float> dPhitoJets;
            for (int ij = 0; ij < nt.PFNJet_; ij++) {
               dRtoJets.push_back(sqrt(pow(ll.eta() - nt.PFJetEta_[ij],2) + pow(reco::deltaPhi(ll.phi(),nt.PFJetPhi_[ij]),2)));
               dPhitoJets.push_back(reco::deltaPhi(ll.phi(),nt.PFJetPhi_[ij]));
            }
            nt.vtx_dRtoJets_.push_back(dRtoJets);
            nt.vtx_dPhiToJets_.push_back(dPhitoJets);

            // get refitted tracks from KVF
            auto refit_tks = tv.refittedTracks();
            if (refit_tks.size() != 2) {
               cout << "Only has " << refit_tks.size() << " refitted tracks!" << endl;
            }
            GlobalPoint gp_pv(pv.position().x(),pv.position().y(),pv.position().z());
            if (refit_tks.size() == 2) {
               auto tk1 = refit_tks.at(0);
               auto tk2 = refit_tks.at(1);
               auto traj1 = tk1.trajectoryStateClosestToPoint(gp_pv);
               auto traj2 = tk2.trajectoryStateClosestToPoint(gp_pv);
               
               nt.vtx_e1_refitDxy_.push_back(traj1.perigeeParameters().transverseImpactParameter());
               nt.vtx_e1_refitDxyErr_.push_back(traj1.perigeeError().transverseImpactParameterError());
               nt.vtx_e1_refitDz_.push_back(traj1.perigeeParameters().longitudinalImpactParameter());
               nt.vtx_e1_refitDzErr_.push_back(traj1.perigeeError().longitudinalImpactParameterError());
               nt.vtx_e1_refitChi2_.push_back(tk1.normalizedChi2());
               
               nt.vtx_e2_refitDxy_.push_back(traj2.perigeeParameters().transverseImpactParameter());
               nt.vtx_e2_refitDxyErr_.push_back(traj2.perigeeError().transverseImpactParameterError());
               nt.vtx_e2_refitDz_.push_back(traj2.perigeeParameters().longitudinalImpactParameter());
               nt.vtx_e2_refitDzErr_.push_back(traj2.perigeeError().longitudinalImpactParameterError());
               nt.vtx_e2_refitChi2_.push_back(tk2.normalizedChi2());

               nt.vtx_refit_dr_.push_back(reco::deltaR(tk1.track(),tk2.track()));
            }
            else if (refit_tks.size() == 1) {
               auto tk1 = refit_tks.at(0);
               auto traj1 = tk1.trajectoryStateClosestToPoint(gp_pv);
               
               nt.vtx_e1_refitDxy_.push_back(traj1.perigeeParameters().transverseImpactParameter());
               nt.vtx_e1_refitDxyErr_.push_back(traj1.perigeeError().transverseImpactParameterError());
               nt.vtx_e1_refitDz_.push_back(traj1.perigeeParameters().longitudinalImpactParameter());
               nt.vtx_e1_refitDzErr_.push_back(traj1.perigeeError().longitudinalImpactParameterError());
               nt.vtx_e1_refitChi2_.push_back(tk1.normalizedChi2());
               
               nt.vtx_e2_refitDxy_.push_back(-999.0);
               nt.vtx_e2_refitDxyErr_.push_back(-999.0);
               nt.vtx_e2_refitDz_.push_back(-999.0);
               nt.vtx_e2_refitDzErr_.push_back(-999.0);
               nt.vtx_e2_refitChi2_.push_back(-999.0);

               nt.vtx_refit_dr_.push_back(-999.0);
            }
            else {
               nt.vtx_e1_refitDxy_.push_back(-999.0);
               nt.vtx_e1_refitDxyErr_.push_back(-999.0);
               nt.vtx_e1_refitDz_.push_back(-999.0);
               nt.vtx_e1_refitDzErr_.push_back(-999.0);
               nt.vtx_e1_refitChi2_.push_back(-999.0);

               nt.vtx_e2_refitDxy_.push_back(-999.0);
               nt.vtx_e2_refitDxyErr_.push_back(-999.0);
               nt.vtx_e2_refitDz_.push_back(-999.0);
               nt.vtx_e2_refitDzErr_.push_back(-999.0);
               nt.vtx_e2_refitChi2_.push_back(-999.0);

               nt.vtx_refit_dr_.push_back(-999.0);
            }

            // Perform kinematic fit to re-compute dielectron mass, pT, dR
            KinematicParticleFactoryFromTransientTrack pFactory;
            ParticleMass e_mass = 0.000511;
            float e_sigma = 0.00000001;
            float chi = 0.;
            float ndf = 0.;
            vector<RefCountedKinematicParticle> eleParticles;
            eleParticles.push_back(pFactory.particle(transient_tracks[0],e_mass,chi,ndf,e_sigma));
            eleParticles.push_back(pFactory.particle(transient_tracks[1],e_mass,chi,ndf,e_sigma));
            KinematicParticleVertexFitter fitter;
            try {
               RefCountedKinematicTree vertexFitTree = fitter.fit(eleParticles);
               if (vertexFitTree->isValid()) {
                  vertexFitTree->movePointerToTheTop();
                  auto diele_part = vertexFitTree->currentParticle();
                  auto diele_state = diele_part->currentState();
                  auto daughters = vertexFitTree->daughterParticles();
                  nt.vtx_refit_m_.push_back(diele_state.mass());
                  nt.vtx_refit_pt_.push_back(diele_state.globalMomentum().transverse());
                  nt.vtx_refit_eta_.push_back(diele_state.globalMomentum().eta());
                  nt.vtx_refit_phi_.push_back(diele_state.globalMomentum().phi());
               }
               else {
                  nt.vtx_refit_m_.push_back(-999.0);
                  nt.vtx_refit_pt_.push_back(-999.0);
                  nt.vtx_refit_eta_.push_back(-999.0);
                  nt.vtx_refit_phi_.push_back(-999.0);
               }
            }
            catch (std::exception ex) {
               cout << "kinematic vertex fit failed!" << endl;
               nt.vtx_refit_m_.push_back(-999.0);
               nt.vtx_refit_pt_.push_back(-999.0);
               nt.vtx_refit_eta_.push_back(-999.0);
               nt.vtx_refit_phi_.push_back(-999.0);
            }
         }
      }
   };

   // Reconstructing electron vertices
   // regular-regular
   computeVertices(reg_good_eles, reg_good_eles, "R", "R");
   // lowpT-lowpT
   computeVertices(lowpt_good_eles, lowpt_good_eles, "L", "L");
   // lowpT-regular
   computeVertices(lowpt_good_eles, reg_good_eles, "L", "R");
   // count vertices
   nt.nvtx_ = nt.vtx_recoVtxVxy_.size();

   
   // Computing electron & vertex PF Isolations OBSOLETE
   //IsolationCalculator isoCalc(recoElectronHandle_,lowPtElectronHandle_,packedPFCandHandle_,nt);
   //isoCalc.calcIso();

   // extra info from MC
   if (!isData) {
      // Gen weight
      nt.genwgt_ = genEvtInfoHandle_->weight();

      // Gen pileup
      for (const auto & pileupInfo : *pileupInfosHandle_) {
         if (pileupInfo.getBunchCrossing() == 0) {
               nt.genpuobs_ = pileupInfo.getPU_NumInteractions();
               nt.genputrue_ = pileupInfo.getTrueNumInteractions();
               break;
         }
      }

      // Lead gen MET
      if (genMETHandle_->size() > 0) {
         auto met = (*genMETHandle_).at(0);
         nt.genLeadMETPt_ = met.pt();
         nt.genLeadMETPhi_ = met.phi();
         nt.genLeadMETET_ = met.sumEt();
         nt.genLeadMETPx_ = met.px();
         nt.genLeadMETPy_ = met.py();
      }

      // Gen Jets
      nt.nGenJet_ = (int)genJetHandle_->size();
      for (const auto & jet : *genJetHandle_) {
         nt.genJetPt_.push_back(jet.pt());
         nt.genJetEta_.push_back(jet.eta());
         nt.genJetPhi_.push_back(jet.phi());
         nt.genJetMETdPhi_.push_back(reco::deltaPhi(jet.phi(),nt.genLeadMETPhi_));
      }

      // Handling gen particles
      // First pass: save all gen particles; save all gen muons/anti-muons by charge;
      // and identify the status-1 chi2-daughter signal muon / anti-muon.
      math::XYZTLorentzVector gen_sig_muon_p4;
      math::XYZTLorentzVector gen_sig_antimuon_p4;
      bool foundGenSigMuon = false;
      bool foundGenSigAntiMuon = false;

      for (const auto & genParticle : *genParticleHandle_) {
         int motherID = -999;
         if (genParticle.numberOfMothers() > 0 && genParticle.mother(0) != nullptr) {
            motherID = genParticle.mother(0)->pdgId();
         }

         nt.genPartID_.push_back(genParticle.pdgId());
         nt.genPartMotherID_.push_back(motherID);
         nt.genPartStatus_.push_back(genParticle.status());
         nt.genPartCharge_.push_back(genParticle.charge());
         nt.genPartPt_.push_back(genParticle.pt());
         nt.genPartEta_.push_back(genParticle.eta());
         nt.genPartPhi_.push_back(genParticle.phi());
         nt.genPartEn_.push_back(genParticle.energy());
         nt.genPartPx_.push_back(genParticle.px());
         nt.genPartPy_.push_back(genParticle.py());
         nt.genPartPz_.push_back(genParticle.pz());
         nt.genPartVxy_.push_back(std::sqrt(genParticle.vx()*genParticle.vx() + genParticle.vy()*genParticle.vy()));
         nt.genPartVx_.push_back(genParticle.vx());
         nt.genPartVy_.push_back(genParticle.vy());
         nt.genPartVz_.push_back(genParticle.vz());
         nt.genPartMass_.push_back(genParticle.mass());

         // All gen-level muons by charge. No status requirement. No mother-ID requirement.
         if (genParticle.pdgId() == 13) {
            nt.nGenMuon_++;
            nt.genMuonMotherID_.push_back(motherID);
            nt.genMuonStatus_.push_back(genParticle.status());
            nt.genMuonCharge_.push_back(genParticle.charge());
            nt.genMuonPt_.push_back(genParticle.pt());
            nt.genMuonEta_.push_back(genParticle.eta());
            nt.genMuonPhi_.push_back(genParticle.phi());
            nt.genMuonEn_.push_back(genParticle.energy());
            nt.genMuonPx_.push_back(genParticle.px());
            nt.genMuonPy_.push_back(genParticle.py());
            nt.genMuonPz_.push_back(genParticle.pz());
            nt.genMuonVxy_.push_back(genParticle.vertex().rho());
            nt.genMuonVz_.push_back(genParticle.vertex().z());
            nt.genMuonVx_.push_back(genParticle.vertex().x());
            nt.genMuonVy_.push_back(genParticle.vertex().y());
         }
         else if (genParticle.pdgId() == -13) {
            nt.nGenAntiMuon_++;
            nt.genAntiMuonMotherID_.push_back(motherID);
            nt.genAntiMuonStatus_.push_back(genParticle.status());
            nt.genAntiMuonCharge_.push_back(genParticle.charge());
            nt.genAntiMuonPt_.push_back(genParticle.pt());
            nt.genAntiMuonEta_.push_back(genParticle.eta());
            nt.genAntiMuonPhi_.push_back(genParticle.phi());
            nt.genAntiMuonEn_.push_back(genParticle.energy());
            nt.genAntiMuonPx_.push_back(genParticle.px());
            nt.genAntiMuonPy_.push_back(genParticle.py());
            nt.genAntiMuonPz_.push_back(genParticle.pz());
            nt.genAntiMuonVxy_.push_back(genParticle.vertex().rho());
            nt.genAntiMuonVz_.push_back(genParticle.vertex().z());
            nt.genAntiMuonVx_.push_back(genParticle.vertex().x());
            nt.genAntiMuonVy_.push_back(genParticle.vertex().y());
         }

         // Keep the requested immediate-mother signal definition unchanged.
         if (isSignal && std::abs(genParticle.pdgId()) == 13 && genParticle.status() == 1 && motherID == 1000023) {
            if (genParticle.pdgId() == 13) {
               foundGenSigMuon = true;
               gen_sig_muon_p4 = genParticle.p4();
               nt.genSigMuonCharge_ = genParticle.charge();
               nt.genSigMuonMotherID_ = motherID;
               nt.genSigMuonStatus_ = genParticle.status();
               nt.genSigMuonPt_ = genParticle.pt();
               nt.genSigMuonEta_ = genParticle.eta();
               nt.genSigMuonPhi_ = genParticle.phi();
               nt.genSigMuonEn_ = genParticle.energy();
               nt.genSigMuonPx_ = genParticle.px();
               nt.genSigMuonPy_ = genParticle.py();
               nt.genSigMuonPz_ = genParticle.pz();
               nt.genSigMuonVxy_ = genParticle.vertex().rho();
               nt.genSigMuonVz_ = genParticle.vertex().z();
               nt.genSigMuonVx_ = genParticle.vertex().x();
               nt.genSigMuonVy_ = genParticle.vertex().y();
            }
            else if (genParticle.pdgId() == -13) {
               foundGenSigAntiMuon = true;
               gen_sig_antimuon_p4 = genParticle.p4();
               nt.genSigAntiMuonCharge_ = genParticle.charge();
               nt.genSigAntiMuonMotherID_ = motherID;
               nt.genSigAntiMuonStatus_ = genParticle.status();
               nt.genSigAntiMuonPt_ = genParticle.pt();
               nt.genSigAntiMuonEta_ = genParticle.eta();
               nt.genSigAntiMuonPhi_ = genParticle.phi();
               nt.genSigAntiMuonEn_ = genParticle.energy();
               nt.genSigAntiMuonPx_ = genParticle.px();
               nt.genSigAntiMuonPy_ = genParticle.py();
               nt.genSigAntiMuonPz_ = genParticle.pz();
               nt.genSigAntiMuonVxy_ = genParticle.vertex().rho();
               nt.genSigAntiMuonVz_ = genParticle.vertex().z();
               nt.genSigAntiMuonVx_ = genParticle.vertex().x();
               nt.genSigAntiMuonVy_ = genParticle.vertex().y();
            }
         }
      }

      // Second pass: save the original reduced gen collection used by the electron analysis.
      // This keeps the original hard-process/status-1-lepton logic intact.
      math::XYZTLorentzVector gen_ele_p4, gen_pos_p4;
      for (const auto & genParticle : *genParticleHandle_) {
         int absID = std::abs(genParticle.pdgId());
         // veto anything that isn't a lepton or a hard process particle
         if ((!genParticle.isHardProcess()) && (genParticle.status() != 1 || (absID < 11) || (absID > 16))) {
            continue;
         }
         nt.nGen_++;
         int motherID = -999;
         if (genParticle.numberOfMothers() > 0 && genParticle.mother(0) != nullptr) {
            motherID = genParticle.mother(0)->pdgId();
         }

         nt.genID_.push_back(genParticle.pdgId());
         nt.genMotherID_.push_back(motherID);
         nt.genCharge_.push_back(genParticle.charge());
         nt.genPt_.push_back(genParticle.pt());
         nt.genEta_.push_back(genParticle.eta());
         nt.genPhi_.push_back(genParticle.phi());
         nt.genEn_.push_back(genParticle.energy());
         nt.genPx_.push_back(genParticle.px());
         nt.genPy_.push_back(genParticle.py());
         nt.genPz_.push_back(genParticle.pz());
         nt.genVxy_.push_back(std::sqrt(genParticle.vx()*genParticle.vx() + genParticle.vy()*genParticle.vy()));
         nt.genVx_.push_back(genParticle.vx());
         nt.genVy_.push_back(genParticle.vy());
         nt.genVz_.push_back(genParticle.vz());
         nt.genMass_.push_back(genParticle.mass());

         if (isSignal) {
            if ((std::abs(genParticle.pdgId()) == 11) && (motherID == 1000023)) {
               // Recording basic info
               if (genParticle.pdgId() == 11) {
                  gen_ele_p4 = genParticle.p4();
                  nt.genEleCharge_ = genParticle.charge();
                  nt.genEleMotherID_ = motherID;
                  nt.genElePt_ = genParticle.pt();
                  nt.genEleEta_ = genParticle.eta();
                  nt.genElePhi_ = genParticle.phi();
                  nt.genEleEn_ = genParticle.energy();
                  nt.genElePx_ = genParticle.px();
                  nt.genElePy_ = genParticle.py();
                  nt.genElePz_ = genParticle.pz();
                  nt.genEleVxy_ = genParticle.vertex().rho();
                  nt.genEleVz_ = genParticle.vertex().z();
                  nt.genEleVx_ = genParticle.vertex().x();
                  nt.genEleVy_ = genParticle.vertex().y();
               }
               else {
                  gen_pos_p4 = genParticle.p4();
                  nt.genPosCharge_ = genParticle.charge();
                  nt.genPosMotherID_ = motherID;
                  nt.genPosPt_ = genParticle.pt();
                  nt.genPosEta_ = genParticle.eta();
                  nt.genPosPhi_ = genParticle.phi();
                  nt.genPosEn_ = genParticle.energy();
                  nt.genPosPx_ = genParticle.px();
                  nt.genPosPy_ = genParticle.py();
                  nt.genPosPz_ = genParticle.pz();
                  nt.genPosVxy_ = genParticle.vertex().rho();
                  nt.genPosVz_ = genParticle.vertex().z();
                  nt.genPosVx_ = genParticle.vertex().x();
                  nt.genPosVy_ = genParticle.vertex().y();
               }
            }
         }
      }

      if (isSignal && foundGenSigMuon && foundGenSigAntiMuon) {
         auto gen_mumu = gen_sig_muon_p4 + gen_sig_antimuon_p4;
         nt.genSigDimuonPt_ = gen_mumu.pt();
         nt.genSigDimuonEta_ = gen_mumu.eta();
         nt.genSigDimuonPhi_ = gen_mumu.phi();
         nt.genSigDimuonEn_ = gen_mumu.energy();
         nt.genSigDimuonMass_ = gen_mumu.mass();
         nt.genSigDimuonDr_ = reco::deltaR(gen_sig_muon_p4, gen_sig_antimuon_p4);
         nt.genSigDimuonMETdPhi_ = reco::deltaPhi(gen_mumu.phi(), nt.genLeadMETPhi_);
         nt.genSigDimuonVxy_ = nt.genSigMuonVxy_;
         nt.genSigDimuonVz_ = nt.genSigMuonVz_;
         nt.genSigDimuonVx_ = nt.genSigMuonVx_;
         nt.genSigDimuonVy_ = nt.genSigMuonVy_;
      }

      if (isSignal) {
         // Gen-matching electrons to reco objects for iDM signal
         // Strategy: merge "good" electrons + low-pT electrons (i.e. the ones saved to ntuples & used in vertexing)
         vector<math::XYZTLorentzVector> all_eles(reg_ele_p4s);
         all_eles.insert(all_eles.end(),lowpt_ele_p4s.begin(),lowpt_ele_p4s.end());
         int n_reg_eles = reg_ele_p4s.size();
         
	 float min_dRe = 999.;
         float min_dRp = 999.;
         int iMatch_e = -1;
         int iMatch_p = -1;
         for (size_t icount = 0; icount < all_eles.size(); icount++) {
	    // don't try gen-matching x-cleaned low-pt electrons
            if (icount >= (size_t)n_reg_eles) {
	       // comment this out for removing cross-cleaning and doing efficiency studies (gen-matching needed)
               if (nt.recoLowPtElectronIsXCleaned_[icount - n_reg_eles]) continue;
            }
            auto ele = all_eles[icount];
            float dRe = reco::deltaR(ele,gen_ele_p4);
            float dRp = reco::deltaR(ele,gen_pos_p4);
            if (dRe > 0.1 && dRp > 0.1) continue;
            
            if (dRe < 0.1 && dRp > 0.1 && dRe < min_dRe) {
               min_dRe = dRe;
               iMatch_e = icount;
            }
            else if (dRe > 0.1 && dRp < 0.1 && dRp < min_dRp) {
               min_dRp = dRp;
               iMatch_p = icount;
            }
            else if (dRe < 0.1 && dRp < 0.1 && (dRe < min_dRe || dRp < min_dRp)) {
               if (dRe < min_dRe && dRp > min_dRp) {
                  min_dRe = dRe;
                  iMatch_e = icount;
               }
               else if (dRe > min_dRe && dRp < min_dRp) {
                  min_dRp = dRp;
                  iMatch_p = icount;
               }
	       else {
                  if (dRe < dRp) {
                     min_dRe = dRe;
                     iMatch_e = icount;
                  }
                  else {
                     min_dRp = dRp;
                     iMatch_p = icount;
                  }
               }
            }
         }
         // check if full signal reconstructed
         if (iMatch_e != -1 && iMatch_p != -1) {
            nt.signalReconstructed_ = true;
         }

         // assign match flags to electrons & vertices
         int iTarg_e = -1; int iTarg_p = -1;
         std::string mType_e = "None"; std::string mType_p = "None";
         if (iMatch_e != -1) {
            nt.genEleMatched_ = true;
            if (iMatch_e < n_reg_eles) {
               nt.recoElectronGenMatched_[iMatch_e] = true;
               nt.recoElectronMatchType_[iMatch_e] = -1;
	       iTarg_e = iMatch_e;
               mType_e = "R";
	       if (nt.recoElectronHasLptMatch_[iMatch_e]) {
                  nt.recoLowPtElectronGEDisMatched_[nt.recoElectronLptMatchIdx_[iMatch_e]] = true;
               }
               for (size_t k = 0; k < nt.recoAllLowPtElectronGEDidx_.size(); k++) {
                  if (nt.recoAllLowPtElectronGEDidx_[k] == iMatch_e) {
                     nt.recoAllLowPtElectronGEDisMatched_[k] = true;
                  }
               }
            }
            else {
               nt.recoLowPtElectronGenMatched_[iMatch_e - n_reg_eles] = true;
               nt.recoLowPtElectronMatchType_[iMatch_e - n_reg_eles] = -1;
               iTarg_e = iMatch_e - n_reg_eles;
               mType_e = "L";
            }
            nt.genEleMatchType_ = mType_e;
            nt.genEleMatchIdxGlobal_ = iMatch_e;
            nt.genEleMatchIdxLocal_ = iTarg_e;
         }
         if (iMatch_p != -1) {
            nt.genPosMatched_ = true;
            if (iMatch_p < n_reg_eles) {
               nt.recoElectronGenMatched_[iMatch_p] = true;
               nt.recoElectronMatchType_[iMatch_p] = 1;
               iTarg_p = iMatch_p;
               mType_p = "R";
               if (nt.recoElectronHasLptMatch_[iMatch_p]) {
		  nt.recoLowPtElectronGEDisMatched_[nt.recoElectronLptMatchIdx_[iMatch_p]] = true;
               }
               for (size_t k = 0; k < nt.recoAllLowPtElectronGEDidx_.size(); k++) {
                  if (nt.recoAllLowPtElectronGEDidx_[k] == iMatch_p) {
                     nt.recoAllLowPtElectronGEDisMatched_[k] = true;
                  }
               }
            }
            else {
               nt.recoLowPtElectronGenMatched_[iMatch_p - n_reg_eles] = true;
               nt.recoLowPtElectronMatchType_[iMatch_p - n_reg_eles] = 1;
               iTarg_p = iMatch_p - n_reg_eles;
               mType_p = "L";
            }
            nt.genPosMatchType_ = mType_p;
            nt.genPosMatchIdxGlobal_ = iMatch_p;
            nt.genPosMatchIdxLocal_ = iTarg_p;
         }

         for (int iv = 0; iv < nt.nvtx_; iv++) {
            if (nt.vtx_e1_type_[iv] == mType_e && nt.vtx_e1_idx_[iv] == iTarg_e) {
               nt.vtx_e1_isMatched_[iv] = true;
               nt.vtx_e1_matchType_[iv] = -1;
            }
            if (nt.vtx_e1_type_[iv] == mType_p && nt.vtx_e1_idx_[iv] == iTarg_p) {
               nt.vtx_e1_isMatched_[iv] = true;
               nt.vtx_e1_matchType_[iv] = 1;
            }

            if (nt.vtx_e2_type_[iv] == mType_e && nt.vtx_e2_idx_[iv] == iTarg_e) {
               nt.vtx_e2_isMatched_[iv] = true;
               nt.vtx_e2_matchType_[iv] = -1;
            }
            if (nt.vtx_e2_type_[iv] == mType_p && nt.vtx_e2_idx_[iv] == iTarg_p) {
               nt.vtx_e2_isMatched_[iv] = true;
               nt.vtx_e2_matchType_[iv] = 1;
            }

            if (nt.vtx_e1_isMatched_[iv] && nt.vtx_e2_isMatched_[iv]) {
               nt.vtx_isMatched_[iv] = true;
               nt.vtx_matchSign_[iv] = nt.vtx_e1_matchType_[iv]*nt.vtx_e2_matchType_[iv];
            }
         }

         // AllLowPt gen-matching (AllLowPt electrons only, no regular electrons in pool)
         float min_dRe_all = 999.;
         float min_dRp_all = 999.;
         int iMatch_e_all = -1;
         int iMatch_p_all = -1;
         for (size_t i = 0; i < allLowPt_ele_p4s.size(); i++) {
            auto alp = allLowPt_ele_p4s[i];
            float dRe = reco::deltaR(alp, gen_ele_p4);
            float dRp = reco::deltaR(alp, gen_pos_p4);
            if (dRe > 0.1 && dRp > 0.1) continue;
            if (dRe < 0.1 && dRp > 0.1 && dRe < min_dRe_all) {
               min_dRe_all = dRe; iMatch_e_all = i;
            }
            else if (dRe > 0.1 && dRp < 0.1 && dRp < min_dRp_all) {
               min_dRp_all = dRp; iMatch_p_all = i;
            }
            else if (dRe < 0.1 && dRp < 0.1 && (dRe < min_dRe_all || dRp < min_dRp_all)) {
               if (dRe < min_dRe_all && dRp > min_dRp_all) {
                  min_dRe_all = dRe; iMatch_e_all = i;
               }
               else if (dRe > min_dRe_all && dRp < min_dRp_all) {
                  min_dRp_all = dRp; iMatch_p_all = i;
               }
               else {
                  if (dRe < dRp) { min_dRe_all = dRe; iMatch_e_all = i; }
                  else           { min_dRp_all = dRp; iMatch_p_all = i; }
               }
            }
         }
         if (iMatch_e_all != -1) {
            nt.recoAllLowPtElectronGenMatched_[iMatch_e_all] = true;
            nt.recoAllLowPtElectronMatchType_[iMatch_e_all] = -1;
            nt.genEleMatchedAllLowPt_ = true;
            nt.genEleMatchIdxAllLowPt_ = iMatch_e_all;
         }
         if (iMatch_p_all != -1) {
            nt.recoAllLowPtElectronGenMatched_[iMatch_p_all] = true;
            nt.recoAllLowPtElectronMatchType_[iMatch_p_all] = 1;
            nt.genPosMatchedAllLowPt_ = true;
            nt.genPosMatchIdxAllLowPt_ = iMatch_p_all;
         }

	 // constructing gen dilepton object
         auto gen_ll = gen_ele_p4 + gen_pos_p4;
         nt.genEEPt_ = gen_ll.pt();
         nt.genEEEta_ = gen_ll.eta();
         nt.genEEPhi_ = gen_ll.phi();
         nt.genEEEn_ = gen_ll.energy();
         nt.genEEMass_ = gen_ll.mass();
         nt.genEEdR_ = reco::deltaR(gen_ele_p4,gen_pos_p4);
         nt.genEEMETdPhi_ = reco::deltaPhi(gen_ll.phi(),nt.genLeadMETPhi_);
         nt.genEEVxy_ = nt.genEleVxy_;
         nt.genEEVz_ = nt.genEleVz_;
         nt.genEEVx_ = nt.genEleVx_;
         nt.genEEVy_ = nt.genEleVy_;
      }
   }

   outT->Fill();
   return;
}

//define this as a plug-in
DEFINE_FWK_MODULE(ElectronSkimmer);
