#ifndef NTUPLECONTAINERV2_HH
#define NTUPLECONTAINERV2_HH

#include <vector>
#include <map>
#include <string>
using std::vector;
using std::map;
using std::string;
#include <iostream>

#include "DataFormats/Math/interface/LorentzVector.h"
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
    unsigned long long eventNum_;
    unsigned long long runNum_;
    unsigned long long lumiSec_;
    bool isData_;
    bool isSignal_;
    string trigNames_[100];
    bool trigPassed_[100];
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

    // All gen particles. This is the single complete gen-particle collection.
    int nGenParticle_;
    vector<int> genPartID_;
    vector<int> genPartMotherID_;                 // immediate mother PDG ID
    vector<int> genPartFirstDifferentMotherID_;   // first mother after walking through same-PDG copies
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
    vector<int> genPartIsFirstCopy_;
    vector<int> genPartIsLastCopy_;
    vector<int> genPartIsLastCopyBeforeFSR_;
    vector<int> genPartIsHardProcess_;
    vector<int> genPartFromHardProcessFinalState_;
    vector<int> genPartFromHardProcessBeforeFSR_;
    vector<int> genPartIsPromptFinalState_;

    // All gen-lepton entries available in prunedGenParticles:
    // 11 <= abs(pdgId) <= 16.
    //
    // This includes charged leptons and neutrinos. No kinematic, geometric,
    // hard-process, status, or copy requirement is applied.
    // genLeptonGenParticleIdx_ links each entry back to the authoritative
    // GenParticle_* collection.
    //
    // Station-2 propagation status:
    //   0 = not attempted because the lepton is neutral
    //   1 = charged-lepton propagation attempted but failed
    //   2 = propagation succeeded
    //
    // genLeptonPropSt2Idx_ is -1 unless status == 2; otherwise it indexes
    // PropGenLeptonSt2_*.
    int nGenLepton_;
    vector<int> genLeptonGenParticleIdx_;
    vector<int> genLeptonID_;
    vector<int> genLeptonMotherID_;
    vector<int> genLeptonFirstDifferentMotherID_;
    vector<int> genLeptonStatus_;
    vector<int> genLeptonCharge_;
    vector<math::XYZTLorentzVector> genLeptonP4_;
    vector<float> genLeptonVxy_;
    vector<float> genLeptonVz_;
    vector<float> genLeptonVx_;
    vector<float> genLeptonVy_;
    vector<int> genLeptonIsFirstCopy_;
    vector<int> genLeptonIsLastCopy_;
    vector<int> genLeptonIsLastCopyBeforeFSR_;
    vector<int> genLeptonIsHardProcess_;
    vector<int> genLeptonFromHardProcessFinalState_;
    vector<int> genLeptonFromHardProcessBeforeFSR_;
    vector<int> genLeptonIsPromptFinalState_;
    vector<int> genLeptonIsSignal_;
    vector<int> genLeptonPropSt2Status_;
    vector<int> genLeptonPropSt2Idx_;

    // Successfully propagated gen leptons at Station 2. The p4 is constructed
    // from the propagated momentum and the source gen-lepton mass.
    int nPropGenLeptonSt2_;
    vector<int> propGenLeptonSt2GenLeptonIdx_;
    vector<math::XYZTLorentzVector> propGenLeptonSt2P4_;
    vector<float> propGenLeptonSt2PositionEta_;
    vector<float> propGenLeptonSt2PositionPhi_;

    // Signal-gen-lepton view: last-copy entries in GenLepton_* whose first
    // different mother has abs(PDG ID) == 1000023. The propagated collection
    // points both to this signal view and directly to the GenLepton entry.
    int nGenSigLepton_;
    vector<int> genSigLeptonGenLeptonIdx_;
    vector<int> genSigLeptonPropSt2Idx_;

    int nPropGenSigLeptonSt2_;
    vector<int> propGenSigLeptonSt2GenSigLeptonIdx_;
    vector<int> propGenSigLeptonSt2GenLeptonIdx_;
    vector<math::XYZTLorentzVector> propGenSigLeptonSt2P4_;
    vector<float> propGenSigLeptonSt2PositionEta_;
    vector<float> propGenSigLeptonSt2PositionPhi_;

    // Gen Signal Muon from iDM signal only: status == 1 and motherID == 1000023
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
    float genSigMuonMass_;
    int genSigMuonImmediateMotherID_;
    int genSigMuonFirstDifferentMotherID_;
    int genSigMuonIsFirstCopy_;
    int genSigMuonIsLastCopy_;
    int genSigMuonIsLastCopyBeforeFSR_;
    int genSigMuonIsHardProcess_;
    int genSigMuonFromHardProcessFinalState_;
    int genSigMuonFromHardProcessBeforeFSR_;
    int genSigMuonIsPromptFinalState_;

    // New nearest-reco matching diagnostics
    float genSigMuonMinDrToRecoMuon_;
    int genSigMuonMatchRecoMuonIdx_;
    float genSigMuonMinDrToDSAMuon_;
    int genSigMuonMatchDSAMuonIdx_;

    // Propagated gen signal muon to muon-station surfaces
    int   genSigMuonPropSt1Valid_;
    float genSigMuonPropSt1Eta_;
    float genSigMuonPropSt1Phi_;
    float genSigMuonPropSt1MomEta_;
    float genSigMuonPropSt1MomPhi_;

    int   genSigMuonPropSt2Valid_;
    float genSigMuonPropSt2Eta_;
    float genSigMuonPropSt2Phi_;
    float genSigMuonPropSt2MomEta_;
    float genSigMuonPropSt2MomPhi_;

    int   genSigMuonPropSt3Valid_;
    float genSigMuonPropSt3Eta_;
    float genSigMuonPropSt3Phi_;
    float genSigMuonPropSt3MomEta_;
    float genSigMuonPropSt3MomPhi_;

    int   genSigMuonPropSt4Valid_;
    float genSigMuonPropSt4Eta_;
    float genSigMuonPropSt4Phi_;
    float genSigMuonPropSt4MomEta_;
    float genSigMuonPropSt4MomPhi_;

    // Propagated same-sign DSA matching diagnostics
    float genSigMuonMinDrToDSAMuonPropSt1_;
    int   genSigMuonMatchDSAMuonPropSt1Idx_;
    float genSigMuonMinDrToDSAMuonPropSt2_;
    int   genSigMuonMatchDSAMuonPropSt2Idx_;
    float genSigMuonMinDrToDSAMuonPropSt3_;
    int   genSigMuonMatchDSAMuonPropSt3Idx_;
    float genSigMuonMinDrToDSAMuonPropSt4_;
    int   genSigMuonMatchDSAMuonPropSt4Idx_;

    // Gen signal muon propagated to each DSA muon's actual outermost valid hit surface
    float genSigMuonMinDrToDSAMuonOuterHit_;
    int   genSigMuonMatchDSAMuonOuterHitIdx_;

    // Gen Signal Anti-Muon from iDM signal only: status == 1 and motherID == 1000023
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
    float genSigAntiMuonMass_;
    int genSigAntiMuonImmediateMotherID_;
    int genSigAntiMuonFirstDifferentMotherID_;
    int genSigAntiMuonIsFirstCopy_;
    int genSigAntiMuonIsLastCopy_;
    int genSigAntiMuonIsLastCopyBeforeFSR_;
    int genSigAntiMuonIsHardProcess_;
    int genSigAntiMuonFromHardProcessFinalState_;
    int genSigAntiMuonFromHardProcessBeforeFSR_;
    int genSigAntiMuonIsPromptFinalState_;

    // New nearest-reco matching diagnostics
    float genSigAntiMuonMinDrToRecoMuon_;
    int genSigAntiMuonMatchRecoMuonIdx_;
    float genSigAntiMuonMinDrToDSAMuon_;
    int genSigAntiMuonMatchDSAMuonIdx_;


    // Propagated gen signal anti-muon to muon-station surfaces
    int   genSigAntiMuonPropSt1Valid_;
    float genSigAntiMuonPropSt1Eta_;
    float genSigAntiMuonPropSt1Phi_;
    float genSigAntiMuonPropSt1MomEta_;
    float genSigAntiMuonPropSt1MomPhi_;

    int   genSigAntiMuonPropSt2Valid_;
    float genSigAntiMuonPropSt2Eta_;
    float genSigAntiMuonPropSt2Phi_;
    float genSigAntiMuonPropSt2MomEta_;
    float genSigAntiMuonPropSt2MomPhi_;

    int   genSigAntiMuonPropSt3Valid_;
    float genSigAntiMuonPropSt3Eta_;
    float genSigAntiMuonPropSt3Phi_;
    float genSigAntiMuonPropSt3MomEta_;
    float genSigAntiMuonPropSt3MomPhi_;

    int   genSigAntiMuonPropSt4Valid_;
    float genSigAntiMuonPropSt4Eta_;
    float genSigAntiMuonPropSt4Phi_;
    float genSigAntiMuonPropSt4MomEta_;
    float genSigAntiMuonPropSt4MomPhi_;

    // Propagated same-sign DSA matching diagnostics
    float genSigAntiMuonMinDrToDSAMuonPropSt1_;
    int   genSigAntiMuonMatchDSAMuonPropSt1Idx_;
    float genSigAntiMuonMinDrToDSAMuonPropSt2_;
    int   genSigAntiMuonMatchDSAMuonPropSt2Idx_;
    float genSigAntiMuonMinDrToDSAMuonPropSt3_;
    int   genSigAntiMuonMatchDSAMuonPropSt3Idx_;
    float genSigAntiMuonMinDrToDSAMuonPropSt4_;
    int   genSigAntiMuonMatchDSAMuonPropSt4Idx_;

    // Gen signal anti-muon propagated to each DSA muon's actual outermost valid hit surface
    float genSigAntiMuonMinDrToDSAMuonOuterHit_;
    int   genSigAntiMuonMatchDSAMuonOuterHitIdx_;

    // Gen Signal Dimuon
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

    int nGenSigMuonFinal_;
    bool genSigMuonIsValid_;
    bool genSigAntiMuonIsValid_;
    bool genSigDimuonIsValid_;
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

    ///////////////////////
    //// Reco branches ////
    ///////////////////////

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

    // Complete PF-muon view of slimmedMuons, with no pT or acceptance cuts.
    // pfMuonPatIdx_ is the index in the input slimmedMuons collection.
    //
    // Propagation-track type:
    //   0 = no usable track
    //   1 = globalTrack
    //   2 = outerTrack / standAloneMuon
    //   3 = innerTrack
    //
    // Station-2 status:
    //   0 = no usable propagation track
    //   1 = propagation attempted but failed
    //   2 = propagation succeeded
    int nPFMuon_;
    vector<int> pfMuonPatIdx_;
    vector<math::XYZTLorentzVector> pfMuonP4_;
    vector<int> pfMuonCharge_;
    vector<int> pfMuonIDcutBasedLoose_;
    vector<int> pfMuonIDcutBasedMedium_;
    vector<int> pfMuonIDcutBasedMediumPrompt_;
    vector<int> pfMuonIDcutBasedTight_;
    vector<int> pfMuonIsGlobalMuon_;
    vector<int> pfMuonIsStandAloneMuon_;
    vector<int> pfMuonPropagationTrackType_;
    vector<int> pfMuonTrkNumValidMuonHits_;
    vector<int> pfMuonTrkNumValidTrackerHits_;
    vector<int> pfMuonTrkNumValidPixelHits_;
    vector<int> pfMuonTrkNumValidStripHits_;
    vector<int> pfMuonNumMatchedStations_;
    vector<int> pfMuonPropSt2Status_;
    vector<int> pfMuonPropSt2Idx_;

    // Successfully propagated PF muons at Station 2.
    int nPropPFMuonSt2_;
    vector<int> propPFMuonSt2PFMuonIdx_;
    vector<math::XYZTLorentzVector> propPFMuonSt2P4_;
    vector<float> propPFMuonSt2PositionEta_;
    vector<float> propPFMuonSt2PositionPhi_;
    
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
    vector<bool> recoLowPtElectronIsXCleaned_;
    vector<int> recoLowPtElectronGEDidx_;
    vector<bool> recoLowPtElectronGEDisMatched_;

    // All Low pT electrons
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
    std::vector<int> recoDSAMuonTrkNumTrackerHits_;
    std::vector<int> recoDSAMuonTrkNumPixHits_;
    std::vector<int> recoDSAMuonTrkNumStripHits_;
    std::vector<int> recoDSAMuonCharge_;
    std::vector<int> recoDSAMuonDisplacedId_;
    std::vector<int> recoDSAMuonTrkNumCSCHits_;
    std::vector<int> recoDSAMuonTrkNumHits_;
    std::vector<int> recoDSAMuonTrkNumPlanes_;
    std::vector<int> recoDSAMuonTrkNumDTHits_;
    std::vector<int> recoDSAMuonIdx_;
    std::vector<math::XYZTLorentzVector> recoDSAMuonP4_;
    
    // DSA reco tracks propagated to muon-station surfaces
    std::vector<int>   recoDSAMuonPropSt1Valid_;
    std::vector<float> recoDSAMuonPropSt1Eta_;
    std::vector<float> recoDSAMuonPropSt1Phi_;
    std::vector<float> recoDSAMuonPropSt1MomEta_;
    std::vector<float> recoDSAMuonPropSt1MomPhi_;

    std::vector<int>   recoDSAMuonPropSt2Valid_;
    std::vector<float> recoDSAMuonPropSt2Eta_;
    std::vector<float> recoDSAMuonPropSt2Phi_;
    std::vector<float> recoDSAMuonPropSt2MomEta_;
    std::vector<float> recoDSAMuonPropSt2MomPhi_;
    std::vector<int>   recoDSAMuonPropSt2Idx_;

    std::vector<int>   recoDSAMuonPropSt3Valid_;
    std::vector<float> recoDSAMuonPropSt3Eta_;
    std::vector<float> recoDSAMuonPropSt3Phi_;
    std::vector<float> recoDSAMuonPropSt3MomEta_;
    std::vector<float> recoDSAMuonPropSt3MomPhi_;

    std::vector<int>   recoDSAMuonPropSt4Valid_;
    std::vector<float> recoDSAMuonPropSt4Eta_;
    std::vector<float> recoDSAMuonPropSt4Phi_;
    std::vector<float> recoDSAMuonPropSt4MomEta_;
    std::vector<float> recoDSAMuonPropSt4MomPhi_;

    // Successfully propagated DSA muons at Station 2.
    int nPropDSAMuonSt2_;
    std::vector<int> propDSAMuonSt2DSAMuonIdx_;
    std::vector<math::XYZTLorentzVector> propDSAMuonSt2P4_;
    std::vector<float> propDSAMuonSt2PositionEta_;
    std::vector<float> propDSAMuonSt2PositionPhi_;

    // DSA outermost valid muon-hit diagnostics
    std::vector<int> recoDSAMuonOuterHitValid_;
    std::vector<int> recoDSAMuonOuterHitTrackExtraAvailable_;
    std::vector<int> recoDSAMuonOuterHitRecHitsSize_;
    std::vector<int> recoDSAMuonOuterHitNValidMuonHits_;
    std::vector<unsigned int> recoDSAMuonOuterHitDetId_;
    std::vector<float> recoDSAMuonOuterHitX_;
    std::vector<float> recoDSAMuonOuterHitY_;
    std::vector<float> recoDSAMuonOuterHitZ_;
    std::vector<float> recoDSAMuonOuterHitR_;
    std::vector<float> recoDSAMuonOuterHitEta_;
    std::vector<float> recoDSAMuonOuterHitPhi_;

    // DSA-centric closest propagated signal-gen match at that DSA outer-hit surface
    std::vector<float> recoDSAMuonMinDrToGenSigMuonOuterHit_;
    std::vector<int> recoDSAMuonMatchGenSigMuonOuterHitIdx_;
    std::vector<int> recoDSAMuonMatchGenSigMuonOuterHitPdgId_;
    std::vector<int> recoDSAMuonMatchGenSigMuonOuterHitPropValid_;
    std::vector<float> recoDSAMuonMatchGenSigMuonOuterHitPropEta_;
    std::vector<float> recoDSAMuonMatchGenSigMuonOuterHitPropPhi_;
    std::vector<float> recoDSAMuonMatchGenSigMuonOuterHitPropMomEta_;
    std::vector<float> recoDSAMuonMatchGenSigMuonOuterHitPropMomPhi_;

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
    TTree * outT;

};

#endif
