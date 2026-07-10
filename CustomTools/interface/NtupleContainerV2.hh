#ifndef NTUPLECONTAINERV2_HH
#define NTUPLECONTAINERV2_HH

#include <vector>
#include <map>
#include <string>
using std::vector;
using std::map;
using std::string;
#include <iostream>

#include <TTree.h>

class NtupleContainerV2 {

public:
    NtupleContainerV2();
    virtual ~NtupleContainerV2();
    void SetTree(TTree *tree);
    void CreateTreeBranches();
    void ClearTreeBranches();

    // Trigger and event-level branches
    unsigned int fired_;
    //unsigned int fired16_;
    //unsigned int fired17_;
    //unsigned int fired18_;
    unsigned long long eventNum_;
    unsigned long long runNum_;
    unsigned long long lumiSec_;
    bool isData_;
    bool isSignal_;
    string trigNames_[100];
    bool trigPassed_[100]; // more than we need to be safe
    int numTrigs_ = 0;
    
    float fixedGridRhoFastjetAll_;
    // MET Filters
    uint32_t METFiltersFailBits_;

    //////////////////////
    //// Gen branches ////
    //////////////////////
    
    // Gen particles
    int nGen_;
    float genwgt_;
    int genpuobs_;
    int genputrue_;
    vector<int> genID_;
    vector<int> genMotherID_;
    // Only save hard-process gen particles
    vector<int> genCharge_;
    vector<float> genPt_;
    vector<float> genEta_;
    vector<float> genPhi_;
    vector<float> genEn_;
    vector<float> genPx_;
    vector<float> genPy_;
    vector<float> genPz_;
    vector<float> genVxy_;
    vector<float> genVz_;
    vector<float> genVx_;
    vector<float> genVy_;
    vector<float> genMass_;

    // Gen Electron & Positron from iDM signal
    int genEleCharge_;
    int genEleMotherID_;
    float genElePt_;
    float genEleEta_;
    float genElePhi_;
    float genEleEn_;
    float genElePx_;
    float genElePy_;
    float genElePz_;
    float genEleVxy_;
    float genEleVz_;
    float genEleVx_;
    float genEleVy_;
    bool genEleMatched_;
    std::string genEleMatchType_;
    int genEleMatchIdxLocal_;
    int genEleMatchIdxGlobal_;
    bool genEleMatchedAllLowPt_;
    int genEleMatchIdxAllLowPt_;

    int genPosCharge_;
    int genPosMotherID_;
    float genPosPt_;
    float genPosEta_;
    float genPosPhi_;
    float genPosEn_;
    float genPosPx_;
    float genPosPy_;
    float genPosPz_;
    float genPosVxy_;
    float genPosVz_;
    float genPosVx_;
    float genPosVy_;
    bool genPosMatched_;
    std::string genPosMatchType_;
    int genPosMatchIdxLocal_;
    int genPosMatchIdxGlobal_;
    bool genPosMatchedAllLowPt_;
    int genPosMatchIdxAllLowPt_;

    // Gen Electron + Positron info
    float genEEPt_;
    float genEEEta_;
    float genEEPhi_;
    float genEEEn_;
    float genEEMass_;
    float genEEdR_;
    float genEEMETdPhi_;
    float genEEVxy_;
    float genEEVz_;
    float genEEVx_;
    float genEEVy_;

    // Track whether full signal (e and p) are reconstructed
    bool signalReconstructed_;
    
    // Gen jet
    int nGenJet_;
    vector<float> genJetPt_;
    vector<float> genJetEta_;
    vector<float> genJetPhi_;
    vector<float> genJetMETdPhi_;
    
    // Gen MET
    float genLeadMETPt_;
    float genLeadMETPhi_;
    float genLeadMETPx_;
    float genLeadMETPy_;
    float genLeadMETET_;

    // Reco Particles
    // Muons: used only for SF measurement with Z/Gamma events
    int nMuon_;
    vector<float> recoMuonPt_;
    vector<float> recoMuonEta_;
    vector<float> recoMuonPhi_;
    vector<float> recoMuonEnergy_;
    vector<float> recoMuonCharge_;
    vector<int> recoMuonIDcutBasedLoose_;
    vector<int> recoMuonIDcutBasedMedium_;
    vector<int> recoMuonIDcutBasedMediumPrompt_;
    vector<int> recoMuonIDcutBasedTight_;
    vector<int> recoMuonIsPFMuon_;
    vector<int> recoMuonIsGlobalMuon_;
    vector<int> recoMuonIsStandAloneMuon_;
    
    // Normal Electrons
    int nElectronDefault_;
    vector<float> recoElectronPt_;
    vector<float> recoElectronEta_;
    vector<float> recoElectronEtaError_;
    vector<float> recoElectronPhi_;
    vector<float> recoElectronPhiError_;
    vector<float> recoElectronID_cutVeto_;
    vector<float> recoElectronID_cutLoose_;
    vector<float> recoElectronID_cutMed_;
    vector<float> recoElectronID_cutTight_;
    vector<int> recoElectronID_cutVetoInt_;
    vector<int> recoElectronID_cutLooseInt_;
    vector<int> recoElectronID_cutMedInt_;
    vector<int> recoElectronID_cutTightInt_;
    vector<float> recoElectronID_mvaIso90_;
    vector<float> recoElectronID_mvaIso80_;
    vector<float> recoElectronID_mvaIsoLoose_;
    vector<float> recoElectronID_mva90_;
    vector<float> recoElectronID_mva80_;
    vector<float> recoElectronID_mvaLoose_;
    vector<float> recoElectronAngularRes_;
    vector<float> recoElectronE_;
    vector<float> recoElectronVxy_;
    vector<float> recoElectronVz_;
    vector<float> recoElectronDxy_;
    vector<float> recoElectronDxyError_;
    vector<float> recoElectronDz_;
    vector<float> recoElectronDzError_;
    vector<float> recoElectronTrkChi2_;
    vector<float> recoElectronTrkIso_;
    vector<float> recoElectronTrkRelIso_;
    vector<float> recoElectronCaloIso_;
    vector<float> recoElectronCaloRelIso_;
    vector<float> recoElectronPFIso_;
    vector<float> recoElectronPFRelIso_;
    vector<float> recoElectronMiniIso_;
    vector<float> recoElectronMiniRelIso_;
    vector<float> recoElectronPFIsoEleCorr_;
    vector<float> recoElectronPFRelIsoEleCorr_;
    vector<float> recoElectronMiniIsoEleCorr_;
    vector<float> recoElectronMiniRelIsoEleCorr_;
    vector<float> recoElectronChadIso_;
    vector<float> recoElectronNhadIso_;
    vector<float> recoElectronPhoIso_;
    vector<float> recoElectronRhoEA_;
    vector<float> recoElectronTrkProb_;
    vector<int> recoElectronTrkNumTrackerHits_;
    vector<int> recoElectronTrkNumPixHits_;
    vector<int> recoElectronTrkNumStripHits_;
    vector<int> recoElectronCharge_;
    vector<bool> recoElectronIsPF_;
    vector<bool> recoElectronGenMatched_;
    vector<int> recoElectronMatchType_;
    vector<vector<float> > recoElectronDrToJets_;
    vector<vector<float> > recoElectronDphiToJets_;
    vector<float> recoElectronFull5x5_sigmaIetaIeta_;
    vector<float> recoElectronAbsdEtaSeed_;
    vector<float> recoElectronAbsdPhiIn_;
    vector<float> recoElectronHoverE_;
    vector<float> recoElectronAbs1overEm1overP_;
    vector<int> recoElectronExpMissingInnerHits_;
    vector<bool> recoElectronConversionVeto_;
    vector<bool> recoElectronIsEE_;
    // special variables for the x-cleaning study
    vector<bool> recoElectronHasLptMatch_;
    vector<int> recoElectronLptMatchIdx_;
    vector<bool> recoElectronHasAllLptMatch_;
    vector<int> recoElectronAllLptMatchIdx_;

    // Low pT electrons
    int nElectronLowPt_;
    vector<float> recoLowPtElectronPt_;
    vector<float> recoLowPtElectronEta_;
    vector<float> recoLowPtElectronEtaError_;
    vector<float> recoLowPtElectronPhi_;
    vector<float> recoLowPtElectronPhiError_;
    vector<float> recoLowPtElectronID_;
    vector<float> recoLowPtElectronAngularRes_;
    vector<float> recoLowPtElectronE_;
    vector<float> recoLowPtElectronVxy_;
    vector<float> recoLowPtElectronVz_;
    vector<float> recoLowPtElectronDxy_;
    vector<float> recoLowPtElectronDxyError_;
    vector<float> recoLowPtElectronDz_;
    vector<float> recoLowPtElectronDzError_;
    vector<float> recoLowPtElectronTrkChi2_;
    vector<float> recoLowPtElectronTrkIso_;
    vector<float> recoLowPtElectronTrkRelIso_;
    vector<float> recoLowPtElectronCaloIso_;
    vector<float> recoLowPtElectronCaloRelIso_;
    vector<float> recoLowPtElectronPFIso_;
    vector<float> recoLowPtElectronPFRelIso_;
    vector<float> recoLowPtElectronMiniIso_;
    vector<float> recoLowPtElectronMiniRelIso_;
    vector<float> recoLowPtElectronPFIsoEleCorr_;
    vector<float> recoLowPtElectronPFRelIsoEleCorr_;
    vector<float> recoLowPtElectronMiniIsoEleCorr_;
    vector<float> recoLowPtElectronMiniRelIsoEleCorr_;
    vector<float> recoLowPtElectronTrkProb_;
    vector<float> recoLowPtElectronChadIso_;
    vector<float> recoLowPtElectronNhadIso_;
    vector<float> recoLowPtElectronPhoIso_;
    vector<float> recoLowPtElectronRhoEA_;
    vector<int> recoLowPtElectronTrkNumTrackerHits_;
    vector<int> recoLowPtElectronTrkNumPixHits_;
    vector<int> recoLowPtElectronTrkNumStripHits_;
    vector<int> recoLowPtElectronCharge_;
    vector<float> recoLowPtElectronMinDrToReg_;
    vector<bool> recoLowPtElectronIsPF_;
    vector<bool> recoLowPtElectronGenMatched_;
    vector<int> recoLowPtElectronMatchType_;
    vector<vector<float> > recoLowPtElectronDrToJets_;
    vector<vector<float> > recoLowPtElectronDphiToJets_;
    vector<float> recoLowPtElectronFull5x5_sigmaIetaIeta_;
    vector<float> recoLowPtElectronAbsdEtaSeed_;
    vector<float> recoLowPtElectronAbsdPhiIn_;
    vector<float> recoLowPtElectronHoverE_;
    vector<float> recoLowPtElectronAbs1overEm1overP_;
    vector<int> recoLowPtElectronExpMissingInnerHits_;
    vector<bool> recoLowPtElectronConversionVeto_;
    vector<bool> recoLowPtElectronIsEE_;
    // supercluster
    vector<float> recoLowPtElectronScRawE_;
    vector<float> recoLowPtElectronScEnergy_;
    vector<float> recoLowPtElectronScEta_;
    vector<float> recoLowPtElectronScPhi_;
    vector<float> recoLowPtElectronScEtaWidth_;
    vector<float> recoLowPtElectronScPhiWidth_;
    vector<int> recoLowPtElectronScClustersSize_;
    // additional shower shapes
    vector<float> recoLowPtElectronFull5x5R9_;
    vector<float> recoLowPtElectronFull5x5E5x5_;
    vector<float> recoLowPtElectronFull5x5E1x5_;
    vector<float> recoLowPtElectronFull5x5E2x5Max_;
    vector<float> recoLowPtElectronFull5x5SigmaIphiIphi_;
    vector<float> recoLowPtElectronFull5x5HcalDepth1OverEcal_;
    vector<float> recoLowPtElectronFull5x5HcalDepth2OverEcal_;
    // special variables for the x-cleaning study
    vector<bool> recoLowPtElectronIsXCleaned_;
    vector<int> recoLowPtElectronGEDidx_;
    vector<bool> recoLowPtElectronGEDisMatched_;

    // All Low pT electrons (includes cross-cleaned)
    int nElectronAllLowPt_;
    vector<float> recoAllLowPtElectronPt_;
    vector<float> recoAllLowPtElectronEta_;
    vector<float> recoAllLowPtElectronEtaError_;
    vector<float> recoAllLowPtElectronPhi_;
    vector<float> recoAllLowPtElectronPhiError_;
    vector<float> recoAllLowPtElectronID_;
    vector<float> recoAllLowPtElectronAngularRes_;
    vector<float> recoAllLowPtElectronE_;
    vector<float> recoAllLowPtElectronVxy_;
    vector<float> recoAllLowPtElectronVz_;
    vector<float> recoAllLowPtElectronDxy_;
    vector<float> recoAllLowPtElectronDxyError_;
    vector<float> recoAllLowPtElectronDz_;
    vector<float> recoAllLowPtElectronDzError_;
    vector<float> recoAllLowPtElectronTrkChi2_;
    vector<float> recoAllLowPtElectronTrkIso_;
    vector<float> recoAllLowPtElectronTrkRelIso_;
    vector<float> recoAllLowPtElectronCaloIso_;
    vector<float> recoAllLowPtElectronCaloRelIso_;
    vector<float> recoAllLowPtElectronPFIso_;
    vector<float> recoAllLowPtElectronPFRelIso_;
    vector<float> recoAllLowPtElectronMiniIso_;
    vector<float> recoAllLowPtElectronMiniRelIso_;
    vector<float> recoAllLowPtElectronPFIsoEleCorr_;
    vector<float> recoAllLowPtElectronPFRelIsoEleCorr_;
    vector<float> recoAllLowPtElectronMiniIsoEleCorr_;
    vector<float> recoAllLowPtElectronMiniRelIsoEleCorr_;
    vector<float> recoAllLowPtElectronTrkProb_;
    vector<float> recoAllLowPtElectronChadIso_;
    vector<float> recoAllLowPtElectronNhadIso_;
    vector<float> recoAllLowPtElectronPhoIso_;
    vector<float> recoAllLowPtElectronRhoEA_;
    vector<int> recoAllLowPtElectronTrkNumTrackerHits_;
    vector<int> recoAllLowPtElectronTrkNumPixHits_;
    vector<int> recoAllLowPtElectronTrkNumStripHits_;
    vector<int> recoAllLowPtElectronCharge_;
    vector<float> recoAllLowPtElectronMinDrToReg_;
    vector<bool> recoAllLowPtElectronIsPF_;
    vector<bool> recoAllLowPtElectronGenMatched_;
    vector<int> recoAllLowPtElectronMatchType_;
    vector<vector<float> > recoAllLowPtElectronDrToJets_;
    vector<vector<float> > recoAllLowPtElectronDphiToJets_;
    vector<float> recoAllLowPtElectronFull5x5_sigmaIetaIeta_;
    vector<float> recoAllLowPtElectronAbsdEtaSeed_;
    vector<float> recoAllLowPtElectronAbsdPhiIn_;
    vector<float> recoAllLowPtElectronHoverE_;
    vector<float> recoAllLowPtElectronAbs1overEm1overP_;
    vector<int> recoAllLowPtElectronExpMissingInnerHits_;
    vector<bool> recoAllLowPtElectronConversionVeto_;
    vector<bool> recoAllLowPtElectronIsEE_;
    // supercluster
    vector<float> recoAllLowPtElectronScRawE_;
    vector<float> recoAllLowPtElectronScEnergy_;
    vector<float> recoAllLowPtElectronScEta_;
    vector<float> recoAllLowPtElectronScPhi_;
    vector<float> recoAllLowPtElectronScEtaWidth_;
    vector<float> recoAllLowPtElectronScPhiWidth_;
    vector<int> recoAllLowPtElectronScClustersSize_;
    // additional shower shapes
    vector<float> recoAllLowPtElectronFull5x5R9_;
    vector<float> recoAllLowPtElectronFull5x5E5x5_;
    vector<float> recoAllLowPtElectronFull5x5E1x5_;
    vector<float> recoAllLowPtElectronFull5x5E2x5Max_;
    vector<float> recoAllLowPtElectronFull5x5SigmaIphiIphi_;
    vector<float> recoAllLowPtElectronFull5x5HcalDepth1OverEcal_;
    vector<float> recoAllLowPtElectronFull5x5HcalDepth2OverEcal_;
    // x-cleaning status and GED match info
    vector<bool> recoAllLowPtElectronIsXCleaned_;
    vector<int> recoAllLowPtElectronGEDidx_;
    vector<bool> recoAllLowPtElectronGEDisMatched_;

    // DSA Muons
    int nDSAMuon_;
    std::vector<float> recoDSAMuonPt_;
    std::vector<float> recoDSAMuonPtErr_;
    std::vector<float> recoDSAMuonEta_;
    std::vector<float> recoDSAMuonEtaErr_;
    std::vector<float> recoDSAMuonPhi_;
    std::vector<float> recoDSAMuonPhiErr_;
    std::vector<float> recoDSAMuonOuterPhi_;
    std::vector<float> recoDSAMuonOuterEta_;
    std::vector<float> recoDSAMuonE_;
    std::vector<float> recoDSAMuonPx_;
    std::vector<float> recoDSAMuonPy_;
    std::vector<float> recoDSAMuonPz_;
    std::vector<float> recoDSAMuonVxy_;
    std::vector<float> recoDSAMuonVz_;
    std::vector<float> recoDSAMuonDxy_;
    std::vector<float> recoDSAMuonDxyError_;
    std::vector<float> recoDSAMuonDz_;
    std::vector<float> recoDSAMuonDzError_;
    std::vector<float> recoDSAMuonTrkChi2_;
    std::vector<float> recoDSAMuonTrkProb_;
    std::vector<int>   recoDSAMuonTrkNumTrackerHits_;
    std::vector<int>   recoDSAMuonTrkNumPixHits_;
    std::vector<int>   recoDSAMuonTrkNumStripHits_;
    std::vector<int>   recoDSAMuonCharge_;
    std::vector<int>   recoDSAMuonDisplacedId_;
    std::vector<int>   recoDSAMuonTrkNumCSCHits_;
    std::vector<int>   recoDSAMuonTrkNumHits_;
    std::vector<int>   recoDSAMuonTrkNumPlanes_;
    std::vector<int>   recoDSAMuonTrkNumDTHits_;
    std::vector<int>   recoDSAMuonIdx_;

    // Photons
    int nPhotons_;
    vector<float> PhotonEt_;
    vector<float> PhotonEta_;
    vector<float> PhotonPhi_;
    vector<float> PhotonPt_;
    vector<float> PhotonEnergy_;
    // supercluster
    vector<float> PhotonScRawE_;
    vector<float> PhotonScEta_;
    vector<float> PhotonScPhi_;
    vector<float> PhotonScEtaWidth_;
    vector<float> PhotonScPhiWidth_;
    // shower shapes
    vector<float> PhotonR9_;
    vector<float> PhotonFull5x5R9_;
    vector<float> PhotonSIeIe_;
    vector<float> PhotonFull5x5SIeIe_;
    vector<float> PhotonHoE_;
    vector<float> PhotonFull5x5HoE_;
    vector<float> PhotonE1x5_;
    vector<float> PhotonE2x5_;
    vector<float> PhotonE5x5_;
    vector<float> PhotonFull5x5E1x5_;
    vector<float> PhotonFull5x5E2x5_;
    vector<float> PhotonFull5x5E5x5_;
    // regression inputs (pat-level)
    vector<float> PhotonSeedE_;
    vector<float> PhotonEMax_;
    vector<float> PhotonE2nd_;
    vector<float> PhotonE3x3_;
    vector<float> PhotonETop_;
    vector<float> PhotonEBottom_;
    vector<float> PhotonELeft_;
    vector<float> PhotonERight_;
    // PF isolation
    vector<float> PhotonChIso_;
    vector<float> PhotonNhIso_;
    vector<float> PhotonPhIso_;
    vector<float> PhotonPuChIso_;
    // PUPPI isolation
    vector<float> PhotonPuppiChIso_;
    vector<float> PhotonPuppiNhIso_;
    vector<float> PhotonPuppiPhIso_;
    // detector-based isolation
    vector<float> PhotonTrkIso_;
    vector<float> PhotonEcalIso_;
    vector<float> PhotonHcalIso_;
    // flags
    vector<int> PhotonPassElectronVeto_;
    vector<int> PhotonHasPixelSeed_;
    vector<int> PhotonIsEB_;
    vector<int> PhotonIsEE_;
    vector<int> PhotonIsEBEEGap_;

    // OOT Photons
    int nOOTPhotons_;
    vector<float> ootPhotonEt_;
    vector<float> ootPhotonEta_;
    vector<float> ootPhotonPhi_;
    vector<float> ootPhotonPt_;
    vector<float> ootPhotonEnergy_;
    // supercluster
    vector<float> ootPhotonScRawE_;
    vector<float> ootPhotonScEta_;
    vector<float> ootPhotonScPhi_;
    vector<float> ootPhotonScEtaWidth_;
    vector<float> ootPhotonScPhiWidth_;
    // shower shapes
    vector<float> ootPhotonR9_;
    vector<float> ootPhotonFull5x5R9_;
    vector<float> ootPhotonSIeIe_;
    vector<float> ootPhotonFull5x5SIeIe_;
    vector<float> ootPhotonHoE_;
    vector<float> ootPhotonFull5x5HoE_;
    vector<float> ootPhotonE1x5_;
    vector<float> ootPhotonE2x5_;
    vector<float> ootPhotonE5x5_;
    vector<float> ootPhotonFull5x5E1x5_;
    vector<float> ootPhotonFull5x5E2x5_;
    vector<float> ootPhotonFull5x5E5x5_;
    // regression inputs (pat-level)
    vector<float> ootPhotonSeedE_;
    vector<float> ootPhotonEMax_;
    vector<float> ootPhotonE2nd_;
    vector<float> ootPhotonE3x3_;
    vector<float> ootPhotonETop_;
    vector<float> ootPhotonEBottom_;
    vector<float> ootPhotonELeft_;
    vector<float> ootPhotonERight_;
    // PF isolation
    vector<float> ootPhotonChIso_;
    vector<float> ootPhotonNhIso_;
    vector<float> ootPhotonPhIso_;
    vector<float> ootPhotonPuChIso_;
    // PUPPI isolation
    vector<float> ootPhotonPuppiChIso_;
    vector<float> ootPhotonPuppiNhIso_;
    vector<float> ootPhotonPuppiPhIso_;
    // detector-based isolation
    vector<float> ootPhotonTrkIso_;
    vector<float> ootPhotonEcalIso_;
    vector<float> ootPhotonHcalIso_;
    // flags
    vector<int> ootPhotonPassElectronVeto_;
    vector<int> ootPhotonHasPixelSeed_;
    vector<int> ootPhotonIsEB_;
    vector<int> ootPhotonIsEE_;
    vector<int> ootPhotonIsEBEEGap_;

    // Isolated Tracks
    int nIsoTrack_;
    vector<float> isoTrackPt_;
    vector<float> isoTrackEta_;
    vector<float> isoTrackPhi_;
    vector<float> isoTrackP_;
    vector<int>   isoTrackCharge_;
    vector<float> isoTrackDxy_;
    vector<float> isoTrackDz_;
    vector<float> isoTrackDxyErr_;
    vector<float> isoTrackDzErr_;
    // PF isolation DR03
    vector<float> isoTrackPfIso03ChHad_;
    vector<float> isoTrackPfIso03NhHad_;
    vector<float> isoTrackPfIso03Pho_;
    vector<float> isoTrackPfIso03Pu_;
    // mini PF isolation
    vector<float> isoTrackMiniIsoChHad_;
    vector<float> isoTrackMiniIsoNhHad_;
    vector<float> isoTrackMiniIsoPho_;
    vector<float> isoTrackMiniIsoPu_;
    // calo matching
    vector<float> isoTrackMatchedCaloJetEmE_;
    vector<float> isoTrackMatchedCaloJetHadE_;
    // track quality
    vector<int>   isoTrackIsHighPurity_;
    vector<int>   isoTrackIsTight_;
    vector<int>   isoTrackIsLoose_;
    // hit pattern
    vector<int>   isoTrackNValidHits_;
    vector<int>   isoTrackNValidPixHits_;
    vector<int>   isoTrackNValidStripHits_;
    vector<int>   isoTrackLostInnerLayers_;
    vector<int>   isoTrackLostLayers_;
    vector<int>   isoTrackLostOuterLayers_;
    // dEdx
    vector<float> isoTrackDEdxStrip_;
    vector<float> isoTrackDEdxPixel_;
    // other
    vector<int>   isoTrackFromPV_;
    vector<float> isoTrackDeltaEta_;
    vector<float> isoTrackDeltaPhi_;
    vector<int>   isoTrackPfLepOverlap_;
    vector<float> isoTrackPfNeutralSum_;

    // PF Candidates (charged only, cross-cleaned against electrons)
    int nPFCand_;
    vector<float> pfCandPt_;
    vector<float> pfCandEta_;
    vector<float> pfCandPhi_;
    vector<float> pfCandEnergy_;
    vector<int>   pfCandCharge_;
    vector<int>   pfCandPdgId_;
    vector<bool>  pfCandHasTrackDetails_;
    // impact parameter / track quality (only meaningful if HasTrackDetails)
    vector<float> pfCandDxy_;
    vector<float> pfCandDxyErr_;
    vector<float> pfCandDz_;
    vector<float> pfCandDzErr_;
    vector<float> pfCandTrkChi2_;
    vector<int>   pfCandNumHits_;
    vector<int>   pfCandNumPixHits_;
    vector<int>   pfCandPixelLayers_;
    vector<int>   pfCandStripLayers_;
    vector<int>   pfCandTrackerLayers_;
    vector<int>   pfCandLostInnerHits_;
    vector<bool>  pfCandTrkHighPurity_;
    vector<int>   pfCandTrkAlgo_;
    // vertex association
    vector<int>   pfCandFromPV_;
    vector<int>   pfCandPvAssocQuality_;
    vector<float> pfCandDzAssocPV_;
    // PF-specific (calo/puppi/egamma) info
    vector<float> pfCandCaloFrac_;
    vector<float> pfCandHcalFrac_;
    vector<float> pfCandRawCaloFrac_;
    vector<float> pfCandRawHcalFrac_;
    vector<float> pfCandPuppiWeight_;
    vector<float> pfCandPuppiWeightNoLep_;
    vector<bool>  pfCandIsGoodEgamma_;
    vector<bool>  pfCandIsIsolatedChHad_;

    // Lost Tracks (charged pat::PackedCandidates not promoted to PF candidates;
    // cross-cleaned against electrons). Same schema as PFCand; the PF-specific
    // calo/puppi/egamma fields are never set by particle flow for these and are
    // filled with their PackedCandidate defaults (0/false).
    int nLostTrack_;
    vector<float> lostTrackPt_;
    vector<float> lostTrackEta_;
    vector<float> lostTrackPhi_;
    vector<float> lostTrackEnergy_;
    vector<int>   lostTrackCharge_;
    vector<int>   lostTrackPdgId_;
    vector<bool>  lostTrackHasTrackDetails_;
    vector<float> lostTrackDxy_;
    vector<float> lostTrackDxyErr_;
    vector<float> lostTrackDz_;
    vector<float> lostTrackDzErr_;
    vector<float> lostTrackTrkChi2_;
    vector<int>   lostTrackNumHits_;
    vector<int>   lostTrackNumPixHits_;
    vector<int>   lostTrackPixelLayers_;
    vector<int>   lostTrackStripLayers_;
    vector<int>   lostTrackTrackerLayers_;
    vector<int>   lostTrackLostInnerHits_;
    vector<bool>  lostTrackTrkHighPurity_;
    vector<int>   lostTrackTrkAlgo_;
    vector<int>   lostTrackFromPV_;
    vector<int>   lostTrackPvAssocQuality_;
    vector<float> lostTrackDzAssocPV_;
    vector<float> lostTrackCaloFrac_;
    vector<float> lostTrackHcalFrac_;
    vector<float> lostTrackRawCaloFrac_;
    vector<float> lostTrackRawHcalFrac_;
    vector<float> lostTrackPuppiWeight_;
    vector<float> lostTrackPuppiWeightNoLep_;
    vector<bool>  lostTrackIsGoodEgamma_;
    vector<bool>  lostTrackIsIsolatedChHad_;

    // Photon conversions
    int nConversions_;
    vector<float> conversionPt_;
    vector<float> conversionEta_;
    vector<float> conversionPhi_;
    vector<float> conversionE_;
    vector<float> conversionPx_;
    vector<float> conversionPy_;
    vector<float> conversionPz_;
    vector<float> conversionX_;
    vector<float> conversionY_;
    vector<float> conversionZ_;
    vector<float> conversionVxy_;
    vector<float> conversionVz_;
    vector<float> conversionLxy_;
    vector<float> conversionLz_;
    vector<float> conversionLxyPV_;
    vector<float> conversionLzPV_;
    vector<float> conversionDxy_;
    vector<float> conversionDz_;
    vector<float> conversionDxyPV_;
    vector<float> conversionDzPV_;
    vector<float> conversionEoverP_;
    vector<float> conversionEoverPrefit_;
    vector<int> conversionNSharedHits_;
    vector<float> conversionM_;
    vector<float> conversionDr_;
    vector<float> conversionChi2_;
    vector<int> conversion_Trk1nHitsVtx_;
    vector<float> conversion_Trk1Pt_;
    vector<float> conversion_Trk1Eta_;
    vector<float> conversion_Trk1Phi_;
    vector<float> conversion_Trk1Chi2_;
    vector<int> conversion_Trk1NValidHits_;
    vector<int> conversion_Trk1numLostHits_;
    vector<float> conversion_Trk1dxy_;
    vector<float> conversion_Trk1dxyBS_;
    vector<float> conversion_Trk1dxyPV_;
    vector<float> conversion_Trk1dz_;
    vector<float> conversion_Trk1dzPV_;
    vector<int> conversion_Trk2nHitsVtx_;
    vector<float> conversion_Trk2Pt_;
    vector<float> conversion_Trk2Eta_;
    vector<float> conversion_Trk2Phi_;
    vector<float> conversion_Trk2Chi2_;
    vector<int> conversion_Trk2NValidHits_;
    vector<int> conversion_Trk2numLostHits_;
    vector<float> conversion_Trk2dxy_;
    vector<float> conversion_Trk2dxyBS_;
    vector<float> conversion_Trk2dxyPV_;
    vector<float> conversion_Trk2dz_;
    vector<float> conversion_Trk2dzPV_;

    // Jets
    int PFNJet_;
    int PFNJetAll_;
    int PFNbJetTrue_;
    int PFNbJetTagged_;
    vector<float> PFJet_matchedGenJetPt_;
    vector<float> PFJet_matchedGenJetEta_;
    vector<float> PFJet_matchedGenJetPhi_;
    vector<float> PFJetRawFactor_;
    vector<float> PFJetMass_;
    vector<float> PFJetMassRaw_;
    vector<float> PFJetPtRaw_;
    vector<float> PFJetArea_;
    vector<float> PFJetEnergy_;
    vector<float> PFJetEnergyRaw_;    
    vector<float> PFJetPt_;
    vector<float> PFJetEta_;
    vector<float> PFJetPhi_;
    vector<float> PFJetBTag_;
    vector<int> PFJetTruth_;
    vector<float> PFJetEffDenomPt_;
    vector<float> PFJetEffNumPt_;
    vector<float> PFJetMETdPhi_;
    vector<float> PFJetCorrectedCHEF_;
    vector<float> PFJetCorrectedNHEF_;
    vector<float> PFJetCorrectedCEEF_;
    vector<float> PFJetCorrectedNEEF_;
    vector<float> PFJetCorrectedNumDaughters_;
    vector<float> PFJetCorrectedChargedMultiplicity_;
    vector<float> PFJetCorrectedPt_;
    vector<float> PFJetCorrectedJESUpPt_;
    vector<float> PFJetCorrectedJESUpEta_;
    vector<float> PFJetCorrectedJESUpPhi_;
    vector<float> PFJetCorrectedJESDownPt_;
    vector<float> PFJetCorrectedJESDownEta_;
    vector<float> PFJetCorrectedJESDownPhi_;
    vector<float> PFJetCorrectedJERUpPt_;
    vector<float> PFJetCorrectedJERUpEta_;
    vector<float> PFJetCorrectedJERUpPhi_;
    vector<float> PFJetCorrectedJERDownPt_;
    vector<float> PFJetCorrectedJERDownEta_;
    vector<float> PFJetCorrectedJERDownPhi_;
    bool PFHEMFlag_;

    // MET
    float PFMET_ET_;
    float PFMET_Pt_;
    float PFMET_Phi_;
    float PFMETJESUpPt_;
    float PFMETJESUpPhi_;
    float PFMETJESDownPt_;
    float PFMETJESDownPhi_;
    float PFMETJERUpPt_;
    float PFMETJERUpPhi_;
    float PFMETJERDownPt_;
    float PFMETJERDownPhi_;
    float PFMETUnclusteredUpPt_;
    float PFMETUnclusteredUpPhi_;
    float PFMETUnclusteredDownPt_;
    float PFMETUnclusteredDownPhi_;    
    float PFMETJetResUpSmearPt_;
    float PFMETJetResUpSmearPhi_;
    float PFMETJetResDownSmearPt_;
    float PFMETJetResDownSmearPhi_;
    
    float CaloMET_ET_;
    float CaloMET_Pt_;
    float CaloMET_Phi_;

    // Pileup density
    float rho_;

    // PV information
    float PV_x_;
    float PV_y_;
    float PV_z_;
    int numPV_;

    // Electron-positron vertices
    int nvtx_;
    vector<std::string> vtx_type_;
    vector<float> vtx_recoVtxVxy_;
    vector<float> vtx_recoVtxSigmaVxy_;
    vector<float> vtx_recoVtxVx_;
    vector<float> vtx_recoVtxVy_;
    vector<float> vtx_recoVtxVz_;
    vector<float> vtx_recoVtxReducedChi2_;
    vector<float> vtx_prob_;
    vector<float> vtx_recoVtxDr_;
    vector<int> vtx_recoVtxSign_;
    vector<float> vtx_minDxy_;
    vector<float> vtx_METdPhi_;
    vector<float> vtx_ll_pt_;
    vector<float> vtx_ll_eta_;
    vector<float> vtx_ll_phi_;
    vector<float> vtx_ll_e_;
    vector<float> vtx_ll_m_;
    vector<float> vtx_ll_px_;
    vector<float> vtx_ll_py_;
    vector<float> vtx_ll_pz_;
    vector<float> vtx_refit_m_;
    vector<float> vtx_refit_pt_;
    vector<float> vtx_refit_eta_;
    vector<float> vtx_refit_phi_;
    vector<float> vtx_refit_dr_;
    vector<bool> vtx_isMatched_;
    vector<int> vtx_matchSign_;
    vector<vector<float> > vtx_dRtoJets_;
    vector<vector<float> > vtx_dPhiToJets_;

    vector<std::string> vtx_e1_type_;
    vector<int> vtx_e1_idx_;
    vector<bool> vtx_e1_isMatched_;
    vector<int> vtx_e1_matchType_;
    vector<float> vtx_e1_refitDxy_;
    vector<float> vtx_e1_refitDxyErr_;
    vector<float> vtx_e1_refitDz_;
    vector<float> vtx_e1_refitDzErr_;
    vector<float> vtx_e1_refitChi2_;

    vector<std::string> vtx_e2_type_;
    vector<int> vtx_e2_idx_;
    vector<bool> vtx_e2_isMatched_;
    vector<int> vtx_e2_matchType_;
    vector<float> vtx_e2_refitDxy_;
    vector<float> vtx_e2_refitDxyErr_;
    vector<float> vtx_e2_refitDz_;
    vector<float> vtx_e2_refitDzErr_;
    vector<float> vtx_e2_refitChi2_;

    // LowPt-only Electron-positron vertices
    int nlptvtx_;
    vector<std::string> lptvtx_type_;
    vector<float> lptvtx_recoVtxVxy_;
    vector<float> lptvtx_recoVtxSigmaVxy_;
    vector<float> lptvtx_recoVtxVx_;
    vector<float> lptvtx_recoVtxVy_;
    vector<float> lptvtx_recoVtxVz_;
    vector<float> lptvtx_recoVtxReducedChi2_;
    vector<float> lptvtx_prob_;
    vector<float> lptvtx_recoVtxDr_;
    vector<int> lptvtx_recoVtxSign_;
    vector<float> lptvtx_minDxy_;
    vector<float> lptvtx_METdPhi_;
    vector<float> lptvtx_ll_pt_;
    vector<float> lptvtx_ll_eta_;
    vector<float> lptvtx_ll_phi_;
    vector<float> lptvtx_ll_e_;
    vector<float> lptvtx_ll_m_;
    vector<float> lptvtx_ll_px_;
    vector<float> lptvtx_ll_py_;
    vector<float> lptvtx_ll_pz_;
    vector<float> lptvtx_refit_m_;
    vector<float> lptvtx_refit_pt_;
    vector<float> lptvtx_refit_eta_;
    vector<float> lptvtx_refit_phi_;
    vector<float> lptvtx_refit_dr_;
    vector<bool> lptvtx_isMatched_;
    vector<int> lptvtx_matchSign_;
    vector<vector<float> > lptvtx_dRtoJets_;
    vector<vector<float> > lptvtx_dPhiToJets_;

    vector<std::string> lptvtx_e1_type_;
    vector<int> lptvtx_e1_idx_;
    vector<bool> lptvtx_e1_isMatched_;
    vector<int> lptvtx_e1_matchType_;
    vector<float> lptvtx_e1_refitDxy_;
    vector<float> lptvtx_e1_refitDxyErr_;
    vector<float> lptvtx_e1_refitDz_;
    vector<float> lptvtx_e1_refitDzErr_;
    vector<float> lptvtx_e1_refitChi2_;

    vector<std::string> lptvtx_e2_type_;
    vector<int> lptvtx_e2_idx_;
    vector<bool> lptvtx_e2_isMatched_;
    vector<int> lptvtx_e2_matchType_;
    vector<float> lptvtx_e2_refitDxy_;
    vector<float> lptvtx_e2_refitDxyErr_;
    vector<float> lptvtx_e2_refitDz_;
    vector<float> lptvtx_e2_refitDzErr_;
    vector<float> lptvtx_e2_refitChi2_;

protected:
    // Reco and gen TTrees
    TTree * outT;

};


#endif
