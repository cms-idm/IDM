// -*- C++ -*-
//
// Package:    iDMe/AODSkimmer
// Class:      AODSkimmer
//
/**\class AODSkimmer AODSkimmer.cc iDMe/AODSkimmer/plugins/AODSkimmer.cc

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

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
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

#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"
#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"
#include "JetMETCorrections/Modules/interface/JetResolution.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"

#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "iDMe/CustomTools/interface/DisplacedDileptonAOD.hh"
#include "iDMe/CustomTools/interface/JetCorrections.hh"
#include "iDMe/CustomTools/interface/NtupleContainer.hh"

#include "TTree.h"
#include "TMath.h"

class AODSkimmer : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources>  {
   public:
      explicit AODSkimmer(const edm::ParameterSet&);
      ~AODSkimmer();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      bool getCollections(const edm::Event&);
      virtual void beginJob() override;
      virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
      virtual void endJob() override;

      // ----------member data ---------------------------
      TTree *outT;
      NtupleContainer nt;
      edm::Service<TFileService> fs;

      std::mt19937 m_random_generator;

      bool isData;
      int year;
      const std::string triggerProcessName_;
      std::vector<std::string> trigPaths16_;
      std::vector<std::string> trigPaths17_;
      std::vector<std::string> trigPaths18_;
      std::vector<std::string> allTrigPaths_;

      // Jet Corrector helpers
      edm::ESHandle<JetCorrectorParametersCollection> JetCorParCollHandle_;
      JetCorrectionUncertainty * jecUnc;

      // Tokens 
      const edm::EDGetTokenT<reco::JetTagCollection> bTagProbbToken_;
      const edm::EDGetTokenT<reco::JetTagCollection> bTagProbbbToken_;
      const edm::EDGetTokenT<vector<reco::GsfElectron> > recoElectronToken_;
      const edm::EDGetTokenT<pat::ElectronCollection> lowPtElectronToken_;
      const edm::EDGetTokenT<vector<pat::IsolatedTrack> > isoTracksToken_;
      const edm::EDGetTokenT<vector<pat::PackedCandidate> > packedPFCandToken_;
      const edm::EDGetTokenT<vector<reco::GenParticle> > genParticleToken_;
      const edm::EDGetTokenT<vector<reco::PFJet> > recoJetToken_; 
      const edm::EDGetTokenT<vector<reco::GenJet> > genJetToken_;
      const edm::EDGetTokenT<vector<reco::GenMET> > genMETToken_;
      const edm::EDGetTokenT<GenEventInfoProduct   > genEvtInfoToken_;
      const edm::EDGetTokenT<vector<reco::Vertex> > primaryVertexToken_;
      const edm::EDGetTokenT<reco::BeamSpot> beamspotToken_;
      const edm::EDGetTokenT<vector<reco::Conversion> > conversionsToken_;
      const edm::EDGetTokenT<vector<reco::Photon> > photonsToken_;
      const edm::EDGetTokenT<vector<reco::Photon> > ootPhotonsToken_;
      const edm::EDGetTokenT<vector<reco::PFMET> > PFMETToken_;
      const edm::EDGetTokenT<vector<reco::CaloMET> > CaloMETToken_;
      const edm::EDGetTokenT<reco::JetCorrector> jetCorrectorToken_;
      const edm::EDGetTokenT<std::vector<PileupSummaryInfo> > pileupInfosToken_;
      const edm::EDGetTokenT<double> rhoToken_;
      const edm::EDGetTokenT<int> primaryVertexFilterToken_;
      const edm::EDGetTokenT<bool> globalSuperTightHalo2016FilterToken_;
      const edm::EDGetTokenT<bool> HBHENoiseFilterToken_;
      const edm::EDGetTokenT<bool> HBHENoiseIsoFilterToken_;
      const edm::EDGetTokenT<bool> EcalDeadCellTriggerPrimitiveFilterToken_;
      const edm::EDGetTokenT<bool> BadPFMuonFilterToken_;
      const edm::EDGetTokenT<bool> BadPFMuonDzFilterToken_;
      const edm::EDGetTokenT<bool> hfNoisyHitsFilterToken_;
      const edm::EDGetTokenT<bool> eeBadScFilterToken_;
      const edm::EDGetTokenT<bool> ecalBadCalibFilterToken_;
      const edm::EDGetTokenT<edm::TriggerResults> trigResultsToken_;
      const edm::EDGetTokenT<trigger::TriggerEvent> trigEventToken_;
      // Run3
      edm::ESGetToken<JetCorrectorParametersCollection, JetCorrectionsRecord> jetCorParCollToken_;
      edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> transientTrackToken_;
      JME::JetResolution::Token jetResolutionToken_;
      JME::JetResolutionScaleFactor::Token jetResolutionSFToken_;
  
      // Handles
      edm::Handle<reco::JetTagCollection> bTagProbbHandle_;
      edm::Handle<reco::JetTagCollection> bTagProbbbHandle_;
      edm::Handle<vector<reco::GsfElectron> > recoElectronHandle_;
      edm::Handle<pat::ElectronCollection> lowPtElectronHandle_;
      edm::Handle<vector<pat::IsolatedTrack> > isoTracksHandle_;
      edm::Handle<vector<pat::PackedCandidate> > packedPFCandHandle_;
      edm::Handle<vector<reco::PFJet> > recoJetHandle_;
      edm::Handle<vector<reco::GenParticle> > genParticleHandle_;
      edm::Handle<vector<reco::GenJet> > genJetHandle_;
      edm::Handle<vector<reco::GenMET> > genMETHandle_;
      edm::Handle<GenEventInfoProduct> genEvtInfoHandle_;
      edm::Handle<vector<reco::Vertex> > primaryVertexHandle_;
      edm::Handle<reco::BeamSpot> beamspotHandle_;
      edm::Handle<vector<reco::Conversion> > conversionsHandle_;
      edm::Handle<vector<reco::Photon> > photonsHandle_;
      edm::Handle<vector<reco::Photon> > ootPhotonsHandle_;
      edm::Handle<vector<reco::PFMET> > PFMETHandle_;
      edm::Handle<vector<reco::CaloMET> > CaloMETHandle_;
      edm::Handle<reco::JetCorrector> jetCorrectorHandle_;
      edm::Handle<std::vector<PileupSummaryInfo> > pileupInfosHandle_;
      edm::Handle<double> rhoHandle_;
      edm::Handle<int> primaryVertexFilterHandle_;
      edm::Handle<bool> globalSuperTightHalo2016FilterHandle_;
      edm::Handle<bool> HBHENoiseFilterHandle_;
      edm::Handle<bool> HBHENoiseIsoFilterHandle_;
      edm::Handle<bool> EcalDeadCellTriggerPrimitiveFilterHandle_;
      edm::Handle<bool> BadPFMuonFilterHandle_;
      edm::Handle<bool> BadPFMuonDzFilterHandle_;
      edm::Handle<bool> hfNoisyHitsFilterHandle_;
      edm::Handle<bool> eeBadScFilterHandle_;
      edm::Handle<bool> ecalBadCalibFilterHandle_;
      edm::Handle<edm::TriggerResults> trigResultsHandle_;
      edm::Handle<trigger::TriggerEvent> trigEventHandle_;

      std::vector<std::string> trigPaths16WithVersion_;
      std::vector<std::string> trigPaths17WithVersion_;
      std::vector<std::string> trigPaths18WithVersion_;
      std::vector<std::string> allTrigPathsWithVersion_;
      std::vector<bool> trigExist16_;
      std::vector<bool> trigExist17_;
      std::vector<bool> trigExist18_;
      std::vector<bool> trigExist_;
      HLTConfigProvider hltConfig_;
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
AODSkimmer::AODSkimmer(const edm::ParameterSet& ps)
 :
   isData(ps.getParameter<bool>("isData")),
   year(ps.getParameter<int>("year")),
   triggerProcessName_(ps.getParameter<std::string>("triggerProcessName")),
   trigPaths16_(ps.getParameter<std::vector<std::string> >("triggerPaths16")),
   trigPaths17_(ps.getParameter<std::vector<std::string> >("triggerPaths17")),
   trigPaths18_(ps.getParameter<std::vector<std::string> >("triggerPaths18")),
   allTrigPaths_(ps.getParameter<std::vector<std::string> >("allTriggerPaths")),
   bTagProbbToken_(consumes<reco::JetTagCollection>(ps.getParameter<edm::InputTag>("bTagProbb"))),
   bTagProbbbToken_(consumes<reco::JetTagCollection>(ps.getParameter<edm::InputTag>("bTagProbbb"))),
   recoElectronToken_(consumes<vector<reco::GsfElectron> >(ps.getParameter<edm::InputTag>("recoElectron"))),
   lowPtElectronToken_(consumes<pat::ElectronCollection>(ps.getParameter<edm::InputTag>("lowPtElectron"))),
   isoTracksToken_(consumes<vector<pat::IsolatedTrack> >(ps.getParameter<edm::InputTag>("isoTracks"))),
   packedPFCandToken_(consumes<vector<pat::PackedCandidate> >(ps.getParameter<edm::InputTag>("packedPFCands"))),
   genParticleToken_(consumes<vector<reco::GenParticle> >(ps.getParameter<edm::InputTag>("genParticle"))),
   recoJetToken_(consumes<vector<reco::PFJet> >(ps.getParameter<edm::InputTag>("recoJetsCHS"))),
   genJetToken_(consumes<vector<reco::GenJet> >(ps.getParameter<edm::InputTag>("genJet"))),
   genMETToken_(consumes<vector<reco::GenMET> >(ps.getParameter<edm::InputTag>("genMET"))),
   genEvtInfoToken_(consumes<GenEventInfoProduct>(ps.getParameter<edm::InputTag>("genEvt"))),
   primaryVertexToken_(consumes<vector<reco::Vertex> >(ps.getParameter<edm::InputTag>("primaryVertex"))),
   beamspotToken_(consumes<reco::BeamSpot>(ps.getParameter<edm::InputTag>("beamSpot"))),
   conversionsToken_(consumes<vector<reco::Conversion> >(ps.getParameter<edm::InputTag>("conversions"))),
   photonsToken_(consumes<vector<reco::Photon> >(ps.getParameter<edm::InputTag>("photons"))),
   ootPhotonsToken_(consumes<vector<reco::Photon> >(ps.getParameter<edm::InputTag>("ootPhotons"))),
   PFMETToken_(consumes<vector<reco::PFMET> >(ps.getParameter<edm::InputTag>("PFMET"))),
   CaloMETToken_(consumes<vector<reco::CaloMET> >(ps.getParameter<edm::InputTag>("CaloMET"))),
   jetCorrectorToken_(consumes<reco::JetCorrector>(ps.getParameter<edm::InputTag>("jetCorrector"))),
   pileupInfosToken_(consumes<std::vector<PileupSummaryInfo>>(ps.getParameter<edm::InputTag>("pileups"))),
   rhoToken_(consumes<double>(ps.getParameter<edm::InputTag>("rho"))),
   primaryVertexFilterToken_(consumes<int>(ps.getParameter<edm::InputTag>("primaryVertexFilter"))),
   globalSuperTightHalo2016FilterToken_(consumes<bool>(ps.getParameter<edm::InputTag>("globalSuperTight2016HaloFilter"))),
   HBHENoiseFilterToken_(consumes<bool>(ps.getParameter<edm::InputTag>("HBHENoiseFilter"))),
   HBHENoiseIsoFilterToken_(consumes<bool>(ps.getParameter<edm::InputTag>("HBHENoiseIsoFilter"))),
   EcalDeadCellTriggerPrimitiveFilterToken_(consumes<bool>(ps.getParameter<edm::InputTag>("EcalDeadCellTrigPrimFilter"))),
   BadPFMuonFilterToken_(consumes<bool>(ps.getParameter<edm::InputTag>("BadPFMuonFilter"))),
   BadPFMuonDzFilterToken_(consumes<bool>(ps.getParameter<edm::InputTag>("BadPFMuonDzFilter"))),
   hfNoisyHitsFilterToken_(consumes<bool>(ps.getParameter<edm::InputTag>("hfNoisyHitsFilter"))),
   eeBadScFilterToken_(consumes<bool>(ps.getParameter<edm::InputTag>("eeBadScFilter"))),
   ecalBadCalibFilterToken_(consumes<bool>(ps.getParameter<edm::InputTag>("ecalBadCalibFilter"))),
   trigResultsToken_(consumes<edm::TriggerResults>(ps.getParameter<edm::InputTag>("trigResult"))),
   trigEventToken_(consumes<trigger::TriggerEvent>(ps.getParameter<edm::InputTag>("trigEvent"))),
   // Run3 additions
   jetCorParCollToken_(esConsumes(edm::ESInputTag("", "AK4PFchs"))),
   transientTrackToken_(esConsumes<TransientTrackBuilder, TransientTrackRecord>()),
   jetResolutionToken_(esConsumes(edm::ESInputTag("", "AK4PFchs_pt"))),
   jetResolutionSFToken_(esConsumes(edm::ESInputTag("", "AK4PFchs")))
{
   usesResource("TFileService");
   m_random_generator = std::mt19937(37428479);

}


AODSkimmer::~AODSkimmer() = default;


//
// member functions
//

void
AODSkimmer::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup)
{
   
   // Set up HLT config
   using namespace edm;

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

   // Add trigger paths if they exist
   allTrigPathsWithVersion_.clear();
   trigPaths16WithVersion_.clear();
   trigPaths17WithVersion_.clear();
   trigPaths18WithVersion_.clear();
   trigExist_.clear();
   trigExist16_.clear();
   trigExist17_.clear();
   trigExist18_.clear();
   
   const std::vector<std::string>& pathNames = hltConfig_.triggerNames();
   
   // All trigger paths (2016+2017+2018)
   for (auto trigPathNoVersion : allTrigPaths_) {
      auto matchedPaths(hltConfig_.restoreVersion(pathNames, trigPathNoVersion));
      if (matchedPaths.size() == 0) {
         LogWarning("TriggerNotFound") << "Could not find matched full trigger path with --> " << trigPathNoVersion;
         allTrigPathsWithVersion_.push_back("None");
         trigExist_.push_back(false);
      }
      else {
         trigExist_.push_back(true);
         allTrigPathsWithVersion_.push_back(matchedPaths[0]);
         if (hltConfig_.triggerIndex(matchedPaths[0]) >= hltConfig_.size()) {
               LogError("TriggerError") << "Cannot find trigger path --> " << matchedPaths[0];
               return;
         }
      }
   }

   // 2016 trigger paths
   for (auto trigPathNoVersion : trigPaths16_) {
      auto matchedPaths(hltConfig_.restoreVersion(pathNames, trigPathNoVersion));
      if (matchedPaths.size() == 0) {
         LogWarning("TriggerNotFound") << "Could not find matched full 2016 trigger path with --> " << trigPathNoVersion;
         trigPaths16WithVersion_.push_back("None");
         trigExist16_.push_back(false);
      }
      else {
         trigExist16_.push_back(true);
         trigPaths16WithVersion_.push_back(matchedPaths[0]);
         if (hltConfig_.triggerIndex(matchedPaths[0]) >= hltConfig_.size()) {
               LogError("TriggerError") << "Cannot find 2016 trigger path --> " << matchedPaths[0];
               return;
         }
      }
   }

   // 2017 trigger paths
   for (auto trigPathNoVersion : trigPaths17_) {
      auto matchedPaths(hltConfig_.restoreVersion(pathNames, trigPathNoVersion));
      if (matchedPaths.size() == 0) {
         LogWarning("TriggerNotFound") << "Could not find matched full 2017 trigger path with --> " << trigPathNoVersion;
         trigPaths17WithVersion_.push_back("None");
         trigExist17_.push_back(false);
      }
      else {
         trigExist17_.push_back(true);
         trigPaths17WithVersion_.push_back(matchedPaths[0]);
         if (hltConfig_.triggerIndex(matchedPaths[0]) >= hltConfig_.size()) {
               LogError("TriggerError") << "Cannot find 2017 trigger path --> " << matchedPaths[0];
               return;
         }
      }
   }

   // 2018 trigger paths
   for (auto trigPathNoVersion : trigPaths18_) {
      auto matchedPaths(hltConfig_.restoreVersion(pathNames, trigPathNoVersion));
      if (matchedPaths.size() == 0) {
         LogWarning("TriggerNotFound") << "Could not find matched full 2018 trigger path with --> " << trigPathNoVersion;
         trigPaths18WithVersion_.push_back("None");
         trigExist18_.push_back(false);
      }
      else {
         trigExist18_.push_back(true);
         trigPaths18WithVersion_.push_back(matchedPaths[0]);
         if (hltConfig_.triggerIndex(matchedPaths[0]) >= hltConfig_.size()) {
               LogError("TriggerError") << "Cannot find 2018 trigger path --> " << matchedPaths[0];
               return;
         }
      }
   }

   // JEC Uncertainty object
   // Run3 modified
   auto JetCorParCollHandle_ = iSetup.getHandle(jetCorParCollToken_);
   JetCorrectorParameters const & JetCorPar = (*JetCorParCollHandle_)["Uncertainty"];
   jecUnc = new JetCorrectionUncertainty(JetCorPar);
   if (!jecUnc) {
      edm::LogError("JECUncertainty") << "iDMAnalyzer::beginRun: failed to get jecUnc object!";
   }
}

// ------------ method called once each job just before starting event loop  ------------
void AODSkimmer::beginJob()
{
   outT = fs->make<TTree>("outT", "outT");
   nt.isData_ = isData;
   nt.SetTree(outT);
   nt.CreateTreeBranches();
}

// ------------ method called once each job just after ending the event loop  ------------
void AODSkimmer::endJob() {}

void AODSkimmer::endRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
AODSkimmer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
   edm::ParameterSetDescription desc;

   // Inputs from the run_ntuplizer_cfg python (cmsRun inputs)
   desc.add<bool>("isData", 0);
   desc.add<int>("year",2018);
   desc.add<std::string>("triggerProcessName", "HLT");
   desc.add<std::vector<std::string> >("triggerPaths16",{});
   desc.add<std::vector<std::string> >("triggerPaths17",{});
   desc.add<std::vector<std::string> >("triggerPaths18",{});
   desc.add<std::vector<std::string> >("allTriggerPaths",{});

   desc.add<edm::InputTag>("jetCorrector", edm::InputTag("nodefault"));

   desc.add<edm::InputTag>("bTagProbb",edm::InputTag("pfDeepCSVJetTags","probb","RECO"));
   desc.add<edm::InputTag>("bTagProbbb",edm::InputTag("pfDeepCSVJetTags","probbb","RECO"));
   desc.add<edm::InputTag>("recoElectron",edm::InputTag("gedGsfElectrons"));
   desc.add<edm::InputTag>("lowPtElectron",edm::InputTag("slimmedLowPtElectrons"));
   desc.add<edm::InputTag>("isoTracks",edm::InputTag("isolatedTracks"));
   desc.add<edm::InputTag>("packedPFCands",edm::InputTag("packedPFCandidates"));
   desc.add<edm::InputTag>("recoJetsCHS",edm::InputTag("ak4PFJetsCHS"));
   desc.add<edm::InputTag>("genParticle",edm::InputTag("genParticles"));
   desc.add<edm::InputTag>("genJet",edm::InputTag("ak4GenJets"));
   desc.add<edm::InputTag>("genMET",edm::InputTag("genMetTrue"));
   desc.add<edm::InputTag>("genEvt", edm::InputTag("generator"));
   desc.add<edm::InputTag>("primaryVertex",edm::InputTag("offlinePrimaryVertices"));
   desc.add<edm::InputTag>("beamSpot",edm::InputTag("offlineBeamSpot"));
   desc.add<edm::InputTag>("conversions",edm::InputTag("allConversions"));
   desc.add<edm::InputTag>("photons",edm::InputTag("photons"));
   desc.add<edm::InputTag>("ootPhotons",edm::InputTag("ootPhotons"));
   desc.add<edm::InputTag>("PFMET",edm::InputTag("pfMetT0rtT1Txy"));
   desc.add<edm::InputTag>("CaloMET",edm::InputTag("caloMet"));
   desc.add<edm::InputTag>("rho", edm::InputTag("fixedGridRhoFastjetAll"));
   desc.add<edm::InputTag>("pileups", edm::InputTag("addPileupInfo"));
   desc.add<edm::InputTag>("primaryVertexFilter",edm::InputTag("myPrimaryVertexFilter"));
   desc.add<edm::InputTag>("globalSuperTight2016HaloFilter",edm::InputTag("globalSuperTightHalo2016Filter"));
   desc.add<edm::InputTag>("HBHENoiseFilter",edm::InputTag("HBHENoiseFilterResultProducer","HBHENoiseFilterResult"));
   desc.add<edm::InputTag>("HBHENoiseIsoFilter",edm::InputTag("HBHENoiseFilterResultProducer","HBHEIsoNoiseFilterResult"));
   desc.add<edm::InputTag>("EcalDeadCellTrigPrimFilter",edm::InputTag("EcalDeadCellTriggerPrimitiveFilter"));
   desc.add<edm::InputTag>("BadPFMuonFilter",edm::InputTag("BadPFMuonFilter"));
   desc.add<edm::InputTag>("BadPFMuonDzFilter",edm::InputTag("BadPFMuonDzFilter"));
   desc.add<edm::InputTag>("hfNoisyHitsFilter",edm::InputTag("hfNoisyHitsFilter"));
   desc.add<edm::InputTag>("eeBadScFilter",edm::InputTag("eeBadScFilter"));
   desc.add<edm::InputTag>("ecalBadCalibFilter",edm::InputTag("ecalBadCalibFilter"));
   desc.add<edm::InputTag>("trigResult",edm::InputTag("TriggerResults","","HLT"));
   desc.add<edm::InputTag>("trigEvent",edm::InputTag("hltTriggerSummaryAOD"));
   
   descriptions.add("AODSkimmer", desc);
}

// ------------ method called for each event  ------------
void
AODSkimmer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using std::cout, std::endl; 

   // Retrieving event data and assigning to handles
   iEvent.getByToken(bTagProbbToken_,bTagProbbHandle_);
   iEvent.getByToken(bTagProbbbToken_,bTagProbbbHandle_);
   iEvent.getByToken(recoElectronToken_,recoElectronHandle_);
   iEvent.getByToken(lowPtElectronToken_,lowPtElectronHandle_);
   iEvent.getByToken(isoTracksToken_,isoTracksHandle_);
   iEvent.getByToken(packedPFCandToken_,packedPFCandHandle_);
   iEvent.getByToken(recoJetToken_,recoJetHandle_);
   iEvent.getByToken(primaryVertexToken_,primaryVertexHandle_);
   iEvent.getByToken(beamspotToken_,beamspotHandle_);
   iEvent.getByToken(conversionsToken_,conversionsHandle_);
   iEvent.getByToken(photonsToken_,photonsHandle_);
   iEvent.getByToken(ootPhotonsToken_,ootPhotonsHandle_);
   iEvent.getByToken(PFMETToken_,PFMETHandle_);
   iEvent.getByToken(CaloMETToken_,CaloMETHandle_);
   iEvent.getByToken(jetCorrectorToken_,jetCorrectorHandle_);
   iEvent.getByToken(pileupInfosToken_,pileupInfosHandle_);
   iEvent.getByToken(rhoToken_,rhoHandle_);
   iEvent.getByToken(primaryVertexFilterToken_,primaryVertexFilterHandle_);
   iEvent.getByToken(globalSuperTightHalo2016FilterToken_,globalSuperTightHalo2016FilterHandle_);
   iEvent.getByToken(HBHENoiseFilterToken_,HBHENoiseFilterHandle_);
   iEvent.getByToken(HBHENoiseIsoFilterToken_,HBHENoiseIsoFilterHandle_);
   iEvent.getByToken(EcalDeadCellTriggerPrimitiveFilterToken_,EcalDeadCellTriggerPrimitiveFilterHandle_);
   iEvent.getByToken(BadPFMuonFilterToken_,BadPFMuonFilterHandle_);
   iEvent.getByToken(BadPFMuonDzFilterToken_,BadPFMuonDzFilterHandle_);
   iEvent.getByToken(hfNoisyHitsFilterToken_,hfNoisyHitsFilterHandle_);
   iEvent.getByToken(eeBadScFilterToken_,eeBadScFilterHandle_);
   iEvent.getByToken(ecalBadCalibFilterToken_,ecalBadCalibFilterHandle_);
   iEvent.getByToken(trigResultsToken_,trigResultsHandle_);
   iEvent.getByToken(trigEventToken_,trigEventHandle_);

   if (!isData) { 
      iEvent.getByToken(genParticleToken_,genParticleHandle_);
      iEvent.getByToken(genJetToken_,genJetHandle_);
      iEvent.getByToken(genMETToken_,genMETHandle_);
      iEvent.getByToken(genEvtInfoToken_,genEvtInfoHandle_);
   }

   // Clear tree branches before filling
   nt.ClearTreeBranches();

   /////////////////////////////////////////////////////////////
   // Computing derived quantities and filling the trees ///////
   /////////////////////////////////////////////////////////////

   // Prearing basic info
   nt.eventNum_ = iEvent.id().event();
   nt.lumiSec_ = iEvent.luminosityBlock();
   nt.runNum_ = iEvent.id().run();
   //Vertex & beamspot information
   reco::Vertex pv = (*primaryVertexHandle_).at(0);
   //reco::BeamSpot beamspot = *beamspotHandle_;
   // Set up objects for vertex reco
   //edm::ESHandle<TransientTrackBuilder> theB;
   //iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder", theB);
   // Run3 modified
   auto theB = iSetup.getHandle(transientTrackToken_);
   KalmanVertexFitter kvf(true);

   // MET Filters (as recommended here https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2#UL_data)
   if ((*primaryVertexFilterHandle_) == 0) nt.METFiltersFailBits_ |= (1<<0);
   if (!(*globalSuperTightHalo2016FilterHandle_)) nt.METFiltersFailBits_ |= (1<<1);
   if (!(*HBHENoiseFilterHandle_)) nt.METFiltersFailBits_ |= (1<<2);
   if (!(*HBHENoiseIsoFilterHandle_)) nt.METFiltersFailBits_ |= (1<<3);
   if (!(*EcalDeadCellTriggerPrimitiveFilterHandle_)) nt.METFiltersFailBits_ |= (1<<4);
   if (!(*BadPFMuonFilterHandle_)) nt.METFiltersFailBits_ |= (1<<5);
   if (!(*BadPFMuonDzFilterHandle_)) nt.METFiltersFailBits_ |= (1<<6);
   if (!(*hfNoisyHitsFilterHandle_)) nt.METFiltersFailBits_ |= (1<<7);
   if (!(*eeBadScFilterHandle_)) nt.METFiltersFailBits_ |= (1<<8);
   if ((year == 2017) || (year == 2018)) { // ecalBadCalib filter only recommended for 2017/18
      if (!(*ecalBadCalibFilterHandle_)) nt.METFiltersFailBits_ |= (1<<9);
   }
   
   // Triggers

   // All triggers
   nt.fired_ = 0;
   for (size_t i = 0; i < allTrigPathsWithVersion_.size(); i++) {
      if (trigExist_.at(i)) {
         std::string trigPath = allTrigPathsWithVersion_[i];
         nt.fired_ |= (trigResultsHandle_->accept(hltConfig_.triggerIndex(trigPath)) << i);
      }
      else {
         nt.fired_ |= (0 <<i);
      }
   }

   // 2016 triggers
   nt.fired16_ = 0;
   for (size_t i = 0; i < trigPaths16WithVersion_.size(); i++) {
      if (trigExist16_.at(i)) {
         std::string trigPath = trigPaths16WithVersion_[i];
         nt.fired16_ |= (trigResultsHandle_->accept(hltConfig_.triggerIndex(trigPath)) << i);
      }
      else {
         nt.fired16_ |= (0 <<i);
      }
   }

   // 2017 triggers
   nt.fired17_ = 0;
   for (size_t i = 0; i < trigPaths17WithVersion_.size(); i++) {
      if (trigExist17_.at(i)) {
         std::string trigPath = trigPaths17WithVersion_[i];
         nt.fired17_ |= (trigResultsHandle_->accept(hltConfig_.triggerIndex(trigPath)) << i);
      }
      else {
         nt.fired17_ |= (0 <<i);
      }
   }

   // 2018 triggers
   nt.fired18_ = 0;
   for (size_t i = 0; i < trigPaths18WithVersion_.size(); i++) {
      if (trigExist18_.at(i)) {
         std::string trigPath = trigPaths18WithVersion_[i];
         nt.fired18_ |= (trigResultsHandle_->accept(hltConfig_.triggerIndex(trigPath)) << i);
      }
      else {
         nt.fired18_ |= (0 <<i);
      }
   }

   ////////////////////////////////
   // Handling default electrons // 
   ////////////////////////////////
   nt.nElectronDefault_ = recoElectronHandle_->size();
   vector<reco::GsfTrackRef> reg_eleTracks{};
   vector<math::XYZTLorentzVector> reg_ele_p4s;
   for (const auto & ele : *recoElectronHandle_) {
      reco::GsfTrackRef track = ele.gsfTrack();
      reg_eleTracks.push_back(track);
      reg_ele_p4s.push_back(ele.p4());
      // Filling basic info
      nt.recoElectronPt_.push_back(ele.pt());
      nt.recoElectronEta_.push_back(ele.eta());
      nt.recoElectronEtaError_.push_back(track->etaError());
      nt.recoElectronPhi_.push_back(ele.phi());
      nt.recoElectronPhiError_.push_back(track->phiError());
      nt.recoElectronAngularRes_.push_back(sqrt(track->phiError()*track->phiError() + track->etaError()*track->etaError()));
      nt.recoElectronE_.push_back(ele.energy());
      nt.recoElectronPx_.push_back(ele.px());
      nt.recoElectronPy_.push_back(ele.py());
      nt.recoElectronPz_.push_back(ele.pz());
      nt.recoElectronVxy_.push_back(ele.trackPositionAtVtx().rho());
      nt.recoElectronVz_.push_back(ele.trackPositionAtVtx().z());
      nt.recoElectronTrkIso_.push_back(ele.dr04TkSumPt());
      nt.recoElectronTrkRelIso_.push_back(ele.dr04TkSumPt()/ele.pt());
      nt.recoElectronCharge_.push_back(ele.charge());
      // Filling track info
      nt.recoElectronDxy_.push_back(track->dxy(pv.position()));
      nt.recoElectronDxyError_.push_back(track->dxyError());
      nt.recoElectronDz_.push_back(track->dz(pv.position()));
      nt.recoElectronDzError_.push_back(track->dzError());
      nt.recoElectronTrkChi2_.push_back(track->normalizedChi2());
      nt.recoElectronTrkProb_.push_back(TMath::Prob(track->chi2(),(int)track->ndof()));
      nt.recoElectronTrkNumTrackerHits_.push_back(track->hitPattern().numberOfValidTrackerHits());
      nt.recoElectronTrkNumPixHits_.push_back(track->hitPattern().numberOfValidPixelHits());
      nt.recoElectronTrkNumStripHits_.push_back(track->hitPattern().numberOfValidStripHits());
   }

   /////////////////////////////////
   /// Handling low-pT electrons ///
   /////////////////////////////////
   std::vector<reco::GsfTrackRef> lowpt_eleTracks{};
   vector<math::XYZTLorentzVector> lowpt_ele_p4s;
   for (unsigned int i = 0; i < lowPtElectronHandle_->size(); i++) {
      pat::ElectronRef ele(lowPtElectronHandle_,i);
      // Cross-cleaning with regular electrons
      float mindR = 999;
      for (auto & reg_ele : *recoElectronHandle_) {
         float dR = reco::deltaR(reg_ele,*ele);
         if (dR < mindR) mindR = dR;
      }
      if (mindR < 0.01) continue;
      reco::GsfTrackRef track = ele->gsfTrack();
      lowpt_eleTracks.push_back(track);
      lowpt_ele_p4s.push_back(ele->p4());
      nt.nElectronLowPt_++;
      // Filling basic info, if electron passes cross cleaning
      nt.recoLowPtElectronPt_.push_back(ele->pt());
      nt.recoLowPtElectronPhi_.push_back(ele->phi());
      nt.recoLowPtElectronPhiError_.push_back(track->phiError());
      nt.recoLowPtElectronEta_.push_back(ele->eta());
      nt.recoLowPtElectronEtaError_.push_back(track->etaError());
      nt.recoLowPtElectronAngularRes_.push_back(sqrt(track->phiError()*track->phiError() + track->etaError()*track->etaError()));
      nt.recoLowPtElectronE_.push_back(ele->energy());
      nt.recoLowPtElectronPx_.push_back(ele->px());
      nt.recoLowPtElectronPy_.push_back(ele->py());
      nt.recoLowPtElectronPz_.push_back(ele->pz());
      nt.recoLowPtElectronVxy_.push_back(ele->trackPositionAtVtx().rho());
      nt.recoLowPtElectronVz_.push_back(ele->trackPositionAtVtx().z());
      nt.recoLowPtElectronTrkIso_.push_back(ele->trackIso());
      nt.recoLowPtElectronTrkRelIso_.push_back(ele->trackIso()/ele->pt());
      nt.recoLowPtElectronCharge_.push_back(ele->charge());
      // Filling tracks
      nt.recoLowPtElectronDxy_.push_back(track->dxy(pv.position()));
      nt.recoLowPtElectronDxyError_.push_back(track->dxyError());
      nt.recoLowPtElectronDz_.push_back(track->dz(pv.position()));
      nt.recoLowPtElectronDzError_.push_back(track->dzError());
      nt.recoLowPtElectronTrkChi2_.push_back(track->normalizedChi2());
      nt.recoLowPtElectronTrkProb_.push_back(TMath::Prob(track->chi2(),(int)track->ndof()));
      nt.recoLowPtElectronTrkNumTrackerHits_.push_back(track->hitPattern().numberOfValidTrackerHits());
      nt.recoLowPtElectronTrkNumPixHits_.push_back(track->hitPattern().numberOfValidPixelHits());
      nt.recoLowPtElectronTrkNumStripHits_.push_back(track->hitPattern().numberOfValidStripHits());
   }

   // Handling photons
   for (const auto & ph : *photonsHandle_) {
      nt.nPhotons_++;
      nt.PhotonEt_.push_back(ph.et());
      nt.PhotonEt_.push_back(ph.eta());
      nt.PhotonEt_.push_back(ph.phi());
   }

   // Handling OOT photons
   for (const auto & ph : *ootPhotonsHandle_) {
      nt.nOOTPhotons_++;
      nt.ootPhotonEt_.push_back(ph.et());
      nt.ootPhotonEt_.push_back(ph.eta());
      nt.ootPhotonEt_.push_back(ph.phi());
   }

   //Handling photon conversions
   for (const auto & conv : *conversionsHandle_) {
      if (conv.nTracks() == 0) continue;
      nt.nConversions_++;
      auto conv_p4 = conv.refittedPair4Momentum();
      nt.conversionPt_.push_back(conv_p4.pt());
      nt.conversionEta_.push_back(conv_p4.eta());
      nt.conversionPhi_.push_back(conv_p4.phi());
      nt.conversionE_.push_back(conv_p4.E());
      nt.conversionPx_.push_back(conv_p4.px());
      nt.conversionPy_.push_back(conv_p4.py());
      nt.conversionPz_.push_back(conv_p4.py());

      auto conv_vtx = conv.conversionVertex();
      nt.conversionVxy_.push_back(sqrt(conv_vtx.x()*conv_vtx.x() + conv_vtx.y()*conv_vtx.y()));
      nt.conversionVz_.push_back(conv_vtx.z());
      nt.conversionX_.push_back(conv_vtx.x());
      nt.conversionY_.push_back(conv_vtx.y());
      nt.conversionZ_.push_back(conv_vtx.z());

      auto t1_in = conv.tracksPin().at(0);
      auto t1_out = conv.tracksPout().at(0);
      nt.conversion_Trk1_innerPt_.push_back(t1_in.rho());
      nt.conversion_Trk1_innerEta_.push_back(t1_in.eta());
      nt.conversion_Trk1_innerPhi_.push_back(t1_in.phi());
      nt.conversion_Trk1_outerPt_.push_back(t1_out.rho());
      nt.conversion_Trk1_outerEta_.push_back(t1_out.eta());
      nt.conversion_Trk1_outerPhi_.push_back(t1_out.phi());
      
      if (conv.nTracks() > 1) {
         auto t2_in = conv.tracksPin().at(1);
         auto t2_out = conv.tracksPout().at(1);
         nt.conversion_Trk2_innerPt_.push_back(t2_in.rho());
         nt.conversion_Trk2_innerEta_.push_back(t2_in.eta());
         nt.conversion_Trk2_innerPhi_.push_back(t2_in.phi());
         nt.conversion_Trk2_outerPt_.push_back(t2_out.rho());
         nt.conversion_Trk2_outerEta_.push_back(t2_out.eta());
         nt.conversion_Trk2_outerPhi_.push_back(t2_out.phi());
      }
      else {
         nt.conversion_Trk2_innerPt_.push_back(-999);
         nt.conversion_Trk2_innerEta_.push_back(-999);
         nt.conversion_Trk2_innerPhi_.push_back(-999);
         nt.conversion_Trk2_outerPt_.push_back(-999);
         nt.conversion_Trk2_outerEta_.push_back(-999);
         nt.conversion_Trk2_outerPhi_.push_back(-999);
      }
   }

   // Algorithm to reconstruct displaced dilepton vertices from isoTracks
   // Want to use regular and OOT photons for this
   //vector<reco::Photon> allPhotons = *photonsHandle_;
   //allPhotons.insert(allPhotons.end(),(*ootPhotonsHandle_).begin(),(*ootPhotonsHandle_).end());
   //DisplacedDileptonsAOD EleFinder(pv,beamspot,isoTracksHandle_,allPhotons,packedPFCandHandle_,theB);
   //EleFinder.findDileptons();
   
   // Define vertex reco function 
   auto computeVertices = [&](vector<int> coll_1, vector<int> coll_2, std::string type) {
      for (size_t i = 0; i < coll_1.size(); i++) {
         for (size_t j = 0; j < coll_2.size(); j++) {
            int ind1 = coll_1[i];
            int ind2 = coll_2[j];
            reco::GsfTrackRef ele_i, ele_j;
            if (type == "regreg") {
               ele_i = reg_eleTracks[ind1];
               ele_j = reg_eleTracks[ind2];
            }
            else if (type == "lowlow") {
               ele_i = lowpt_eleTracks[ind1];
               ele_j = lowpt_eleTracks[ind2];
            }
            else if (type == "lowreg") {
               ele_i = lowpt_eleTracks[ind1];
               ele_j = reg_eleTracks[ind2];
            }
            if ( (type == "regreg" || type == "lowlow") && (j <= i) ) continue; // don't vertex ele with itself or ones prior (if vertexing with same type)
            if (ele_i == ele_j) continue; // skip if same ele is in reg and low-pT collections
            if (!ele_i.isNonnull() || !ele_j.isNonnull()) continue; // skip if there's a bad track
            if (reco::deltaR(*ele_i,*ele_j) < 0.01) continue; // skip if they're likely to be the same electron un-cross-cleaned

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
            float dr = reco::deltaR(*ele_i, *ele_j);

            if (type == "lowlow") {
               math::XYZTLorentzVector ll = lowpt_ele_p4s[ind1] + lowpt_ele_p4s[ind2];
               nt.LLvtx_idx1_.push_back(ind1);
               nt.LLvtx_idx2_.push_back(ind2);
               nt.LLvtx_recoVtxReducedChi2_.push_back(vtx_chi2);
               nt.LLvtx_prob_.push_back(vtx_prob);
               nt.LLvtx_recoVtxVxy_.push_back(vxy);
               nt.LLvtx_recoVtxSigmaVxy_.push_back(sigma_vxy);
               nt.LLvtx_recoVtxVx_.push_back(vx);
               nt.LLvtx_recoVtxVy_.push_back(vy);
               nt.LLvtx_recoVtxVz_.push_back(vz);
               nt.LLvtx_recoVtxDr_.push_back(dr);
               nt.LLvtx_recoVtxSign_.push_back(ele_i->charge()*ele_j->charge());
               nt.LLvtx_ll_pt_.push_back(ll.pt());
               nt.LLvtx_ll_eta_.push_back(ll.eta());
               nt.LLvtx_ll_phi_.push_back(ll.phi());
               nt.LLvtx_ll_e_.push_back(ll.e());
               nt.LLvtx_ll_m_.push_back(ll.M());
               nt.LLvtx_ll_px_.push_back(ll.px());
               nt.LLvtx_ll_py_.push_back(ll.py());
               nt.LLvtx_ll_pz_.push_back(ll.pz());
            }
            else if (type == "regreg") {
               math::XYZTLorentzVector ll = reg_ele_p4s[ind1] + reg_ele_p4s[ind2];
               nt.RRvtx_idx1_.push_back(ind1); 
               nt.RRvtx_idx2_.push_back(ind2);
               nt.RRvtx_recoVtxReducedChi2_.push_back(vtx_chi2);
               nt.RRvtx_prob_.push_back(vtx_prob);
               nt.RRvtx_recoVtxVxy_.push_back(vxy);
               nt.RRvtx_recoVtxSigmaVxy_.push_back(sigma_vxy);
               nt.RRvtx_recoVtxVx_.push_back(vx);
               nt.RRvtx_recoVtxVy_.push_back(vy);
               nt.RRvtx_recoVtxVz_.push_back(vz);
               nt.RRvtx_recoVtxDr_.push_back(dr);
               nt.RRvtx_recoVtxSign_.push_back(ele_i->charge()*ele_j->charge());
               nt.RRvtx_ll_pt_.push_back(ll.pt());
               nt.RRvtx_ll_eta_.push_back(ll.eta());
               nt.RRvtx_ll_phi_.push_back(ll.phi());
               nt.RRvtx_ll_e_.push_back(ll.e());
               nt.RRvtx_ll_m_.push_back(ll.M());
               nt.RRvtx_ll_px_.push_back(ll.px());
               nt.RRvtx_ll_py_.push_back(ll.py());
               nt.RRvtx_ll_pz_.push_back(ll.pz());
            }
            else if (type == "lowreg") {
               math::XYZTLorentzVector ll = lowpt_ele_p4s[ind1] + reg_ele_p4s[ind2];
               nt.LRvtx_idx1_.push_back(ind1); 
               nt.LRvtx_idx2_.push_back(ind2);
               nt.LRvtx_recoVtxReducedChi2_.push_back(vtx_chi2);
               nt.LRvtx_prob_.push_back(vtx_prob);
               nt.LRvtx_recoVtxVxy_.push_back(vxy);
               nt.LRvtx_recoVtxSigmaVxy_.push_back(sigma_vxy);
               nt.LRvtx_recoVtxVx_.push_back(vx);
               nt.LRvtx_recoVtxVy_.push_back(vy);
               nt.LRvtx_recoVtxVz_.push_back(vz);
               nt.LRvtx_recoVtxDr_.push_back(dr);
               nt.LRvtx_recoVtxSign_.push_back(ele_i->charge()*ele_j->charge());
               nt.LRvtx_ll_pt_.push_back(ll.pt());
               nt.LRvtx_ll_eta_.push_back(ll.eta());
               nt.LRvtx_ll_phi_.push_back(ll.phi());
               nt.LRvtx_ll_e_.push_back(ll.e());
               nt.LRvtx_ll_m_.push_back(ll.M());
               nt.LRvtx_ll_px_.push_back(ll.px());
               nt.LRvtx_ll_py_.push_back(ll.py());
               nt.LRvtx_ll_pz_.push_back(ll.pz());
            }
         }
      }
   };

   // Reconstructing electron vertices
   // Sort electron tracks by pT
   vector<int> ind_ele(reg_eleTracks.size());
   std::iota(ind_ele.begin(),ind_ele.end(),0); // create vector of electron indices
   std::sort(ind_ele.begin(), ind_ele.end(), [&](const int & l, const int & r) { // sort indices by ele pT
            return reg_ele_p4s[l].pt() > reg_ele_p4s[r].pt();
            });
   // Sort low-pT electron tracks by pT
   vector<int> ind_lptele(lowpt_eleTracks.size());
   std::iota(ind_lptele.begin(),ind_lptele.end(),0); // create vector of low-pT electron indices
   std::sort(ind_lptele.begin(), ind_lptele.end(), [&](const int & l, const int & r) { // sort indices by ele pT
            return lowpt_ele_p4s[l].pt() > lowpt_ele_p4s[r].pt();
            });

   // lowpT-lowpT
   computeVertices(ind_lptele, ind_lptele, "lowlow");
   nt.nEleVertex_LL_ = nt.LLvtx_recoVtxVxy_.size();
   // regular-regular
   computeVertices(ind_ele, ind_ele, "regreg");
   nt.nEleVertex_RR_ = nt.RRvtx_recoVtxVxy_.size();
   // lowpT-regular
   computeVertices(ind_lptele, ind_ele, "lowreg");
   nt.nEleVertex_LR_ = nt.LRvtx_recoVtxVxy_.size();

   // Handling MET //
   auto PFMET = PFMETHandle_->at(0);
   if (PFMETHandle_->size() > 0) {
      auto pfmet = (*PFMETHandle_).at(0);
      nt.PFMET_ET_ = pfmet.sumEt();
      nt.PFMET_Px_ = pfmet.px();
      nt.PFMET_Py_ = pfmet.py();
      nt.PFMET_Pt_ = pfmet.pt();
      nt.PFMET_Phi_ = pfmet.phi();
   }
   if (CaloMETHandle_->size() > 0) {
      auto calomet = (*CaloMETHandle_).at(0);
      nt.CaloMET_ET_ = calomet.sumEt();
      nt.CaloMET_Px_ = calomet.px();
      nt.CaloMET_Py_ = calomet.py();
      nt.CaloMET_Pt_ = calomet.pt();
      nt.CaloMET_Phi_ = calomet.phi();
   }

   // Pileup //
   nt.rho_ = *rhoHandle_;

   ///////////////////
   // Handling Jets //
   ///////////////////
   
   // Loading/executing module to compute & apply jet corrections, and write to output tree
   // Run3 modified
   JME::JetResolution resolution = JME::JetResolution::get(iSetup, jetResolutionToken_);
   JME::JetResolutionScaleFactor resolution_sf = JME::JetResolutionScaleFactor::get(iSetup, jetResolutionSFToken_);
   
   JetCorrections jc(recoJetHandle_, jetCorrectorHandle_, nt, bTagProbbHandle_, bTagProbbbHandle_, recoElectronHandle_, year, isData);
   jc.Correct(resolution, resolution_sf, *jecUnc, rhoHandle_, genJetHandle_, PFMET);
   ///////////////////////////////////////

   //Handling gen particles
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

      math::XYZTLorentzVector gen_ele_p4, gen_pos_p4;
      for (const auto & genParticle : *genParticleHandle_) {
         if (!genParticle.isHardProcess()) continue;
         nt.nGen_++;
         nt.genID_.push_back(genParticle.pdgId());
         nt.genCharge_.push_back(genParticle.charge());
         nt.genPt_.push_back(genParticle.pt());
         nt.genEta_.push_back(genParticle.eta());
         nt.genPhi_.push_back(genParticle.phi());
         nt.genEn_.push_back(genParticle.energy());
         nt.genPx_.push_back(genParticle.px());
         nt.genPy_.push_back(genParticle.py());
         nt.genPz_.push_back(genParticle.pz());
         nt.genVxy_.push_back(sqrt(genParticle.vx()*genParticle.vx() + genParticle.vy()*genParticle.vy()));
         nt.genVx_.push_back(genParticle.vx());
         nt.genVy_.push_back(genParticle.vy());
         nt.genVz_.push_back(genParticle.vz());
         nt.genMass_.push_back(genParticle.mass());

         if (abs(genParticle.pdgId()) == 11) {
            // Recording basic info
            if (genParticle.pdgId() == 11) {
               gen_ele_p4 = genParticle.p4();
               nt.genEleCharge_ = genParticle.charge();
               nt.genElePt_ = genParticle.pt();
               nt.genEleEta_ = genParticle.eta();
               nt.genElePhi_ = genParticle.phi();
               nt.genEleEn_ = genParticle.energy();
               nt.genElePx_ = genParticle.px();
               nt.genElePy_ = genParticle.py();
               nt.genElePz_ = genParticle.pz();
               nt.genEleVx_ = genParticle.vx();
               nt.genEleVy_ = genParticle.vy();
               nt.genEleVz_ = genParticle.vz();
               nt.genEleVxy_ = sqrt(genParticle.vx()*genParticle.vx() + genParticle.vy()*genParticle.vy());
            }
            else {
               gen_pos_p4 = genParticle.p4();
               nt.genPosCharge_ = genParticle.charge();
               nt.genPosPt_ = genParticle.pt();
               nt.genPosEta_ = genParticle.eta();
               nt.genPosPhi_ = genParticle.phi();
               nt.genPosEn_ = genParticle.energy();
               nt.genPosPx_ = genParticle.px();
               nt.genPosPy_ = genParticle.py();
               nt.genPosPz_ = genParticle.pz();
               nt.genPosVx_ = genParticle.vx();
               nt.genPosVy_ = genParticle.vy();
               nt.genPosVz_ = genParticle.vz();
               nt.genPosVxy_ = sqrt(genParticle.vx()*genParticle.vx() + genParticle.vy()*genParticle.vy());
            }

            // Finding reco-level matches
            float mindR = 999;
            int bestMatchType = 0;
            int idxBestMatch = -1;
            int ie = 0, ile = 0;
            // looping through regular electrons
            for (auto & ele : reg_ele_p4s) {
               float dRe = reco::deltaR(ele,genParticle.p4());
               if (dRe < 0.1) {
                  if (dRe < mindR) {
                     mindR = dRe;
                     bestMatchType = 1;
                     idxBestMatch = ie;
                  }
               }
               ie++;
            }
            // looping through low-pT electrons
            for (auto & ele : lowpt_ele_p4s) {
               float dRe = reco::deltaR(ele,genParticle.p4());
               if (dRe < 0.1) {
                  if (dRe < mindR) {
                     mindR = dRe;
                     bestMatchType = 2;
                     idxBestMatch = ile;
                  }
               }
               ile++;
            }
            if (genParticle.pdgId() == 11) {
               nt.genEleClosestType_ = bestMatchType;
               nt.genEleClosestInd_ = idxBestMatch;
               nt.genEleClosestDr_ = mindR;
            }
            else {
               nt.genPosClosestType_ = bestMatchType;
               nt.genPosClosestInd_ = idxBestMatch;
               nt.genPosClosestDr_ = mindR;
            }
         }
      }
      auto gen_ll = gen_ele_p4 + gen_pos_p4;
      nt.genEEPt_ = gen_ll.pt();
      nt.genEEEta_ = gen_ll.eta();
      nt.genEEPhi_ = gen_ll.phi();
      nt.genEEEn_ = gen_ll.energy();
      nt.genEEMass_ = gen_ll.mass();
      nt.genEEdR_ = reco::deltaR(gen_ele_p4,gen_pos_p4);

      // all gen jets
      nt.nGenJet_ = (int)genJetHandle_->size();
      for (const auto & jet : *genJetHandle_) {
         nt.genJetPt_.push_back(jet.pt());
         nt.genJetEta_.push_back(jet.eta());
         nt.genJetPhi_.push_back(jet.phi());
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
   }

   outT->Fill();
   return;
}

//define this as a plug-in
DEFINE_FWK_MODULE(AODSkimmer);
