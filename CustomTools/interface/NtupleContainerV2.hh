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


    // All gen particles (including non-hard processes)
    vector<int> genPartID_;
    vector<int> genPartMotherID_;
    vector<int> genPartStatus_;
    vector<int> genPartCharge_;
    vector<float> genPartPt_;
    vector<float> genPartEta_;
    vector<float> genPartPhi_;
    vector<float> genPartEn_;
    vector<float> genPartPx_;
    vector<float> genPartPy_;
    vector<float> genPartPz_;
    vector<float> genPartVxy_;
    vector<float> genPartVz_;
    vector<float> genPartVx_;
    vector<float> genPartVy_;
    vector<float> genPartMass_;

    // All gen muons and anti-muons (not restricted by status or mother ID)
    int nGenMuon_;
    vector<int> genMuonMotherID_;
    vector<int> genMuonStatus_;
    vector<int> genMuonCharge_;
    vector<float> genMuonPt_;
    vector<float> genMuonEta_;
    vector<float> genMuonPhi_;
    vector<float> genMuonEn_;
    vector<float> genMuonPx_;
    vector<float> genMuonPy_;
    vector<float> genMuonPz_;
    vector<float> genMuonVxy_;
    vector<float> genMuonVz_;
    vector<float> genMuonVx_;
    vector<float> genMuonVy_;

    int nGenAntiMuon_;
    vector<int> genAntiMuonMotherID_;
    vector<int> genAntiMuonStatus_;
    vector<int> genAntiMuonCharge_;
    vector<float> genAntiMuonPt_;
    vector<float> genAntiMuonEta_;
    vector<float> genAntiMuonPhi_;
    vector<float> genAntiMuonEn_;
    vector<float> genAntiMuonPx_;
    vector<float> genAntiMuonPy_;
    vector<float> genAntiMuonPz_;
    vector<float> genAntiMuonVxy_;
    vector<float> genAntiMuonVz_;
    vector<float> genAntiMuonVx_;
    vector<float> genAntiMuonVy_;

    // Gen Signal Muon & Anti-Muon from iDM signal only: status == 1 and motherID == 1000023
    int genSigMuonCharge_;
    int genSigMuonMotherID_;
    int genSigMuonStatus_;
    float genSigMuonPt_;
    float genSigMuonEta_;
    float genSigMuonPhi_;
    float genSigMuonEn_;
    float genSigMuonPx_;
    float genSigMuonPy_;
    float genSigMuonPz_;
    float genSigMuonVxy_;
    float genSigMuonVz_;
    float genSigMuonVx_;
    float genSigMuonVy_;
    bool genSigMuonMatched_;
    std::string genSigMuonMatchType_;
    int genSigMuonMatchIdxLocal_;
    int genSigMuonMatchIdxGlobal_;

    int genSigAntiMuonCharge_;
    int genSigAntiMuonMotherID_;
    int genSigAntiMuonStatus_;
    float genSigAntiMuonPt_;
    float genSigAntiMuonEta_;
    float genSigAntiMuonPhi_;
    float genSigAntiMuonEn_;
    float genSigAntiMuonPx_;
    float genSigAntiMuonPy_;
    float genSigAntiMuonPz_;
    float genSigAntiMuonVxy_;
    float genSigAntiMuonVz_;
    float genSigAntiMuonVx_;
    float genSigAntiMuonVy_;
    bool genSigAntiMuonMatched_;
    std::string genSigAntiMuonMatchType_;
    int genSigAntiMuonMatchIdxLocal_;
    int genSigAntiMuonMatchIdxGlobal_;

    // Gen Signal Dimuon (from iDM signal only)

    float genSigDimuonPt_;
    float genSigDimuonEta_;
    float genSigDimuonPhi_;
    float genSigDimuonEn_;
    float genSigDimuonMass_;
    float genSigDimuonDr_;
    float genSigDimuonMETdPhi_;
    float genSigDimuonVxy_;
    float genSigDimuonVz_;
    float genSigDimuonVx_;
    float genSigDimuonVy_;

     // Track whether full signal (mu anti-mu) are reconstructed
    bool signalDimuonReconstructed_;

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
    // Muons
    int nMuon_;
    vector<float> recoMuonPt_;
    vector<float> recoMuonPtErr_;
    vector<float> recoMuonEta_;
    vector<float> recoMuonEtaErr_;
    vector<float> recoMuonPhi_;
    vector<float> recoMuonPhiErr_;
    vector<float> recoMuonEnergy_;
    vector<float> recoMuonE_;
    vector<float> recoMuonVxy_;
    vector<float> recoMuonVz_;
    vector<float> recoMuonVx_;
    vector<float> recoMuonVy_;
    vector<float> recoMuonDxy_;
    vector<float> recoMuonDxyError_;
    vector<float> recoMuonDz_;
    vector<float> recoMuonDzError_;
    vector<float> recoMuonTrkChi2_;
    vector<float> recoMuonTrkProb_;
    vector<int> recoMuonTrkNumTrackerHits_;
    vector<int> recoMuonTrkNumPixHits_;
    vector<int> recoMuonTrkNumStripHits_;
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

    // OOT Photons
    int nOOTPhotons_;
    vector<float> ootPhotonEt_;
    vector<float> ootPhotonEta_;
    vector<float> ootPhotonPhi_;

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

protected:
    // Reco and gen TTrees
    TTree * outT;

};


#endif
