#include "iDMe/CustomTools/interface/NtupleContainerV2.hh"

NtupleContainerV2::NtupleContainerV2() {}

NtupleContainerV2::~NtupleContainerV2() {}

void NtupleContainerV2::SetTree(TTree *tree) { outT = tree; }

void NtupleContainerV2::CreateTreeBranches() {

    // Data or MC
    //outT->Branch("isData",&isData_);

    // Reco information
    outT->Branch("trigFired",&fired_);
    //outT->Branch("trigFired16",&fired16_);
    //outT->Branch("trigFired17",&fired17_);
    //outT->Branch("trigFired18",&fired18_);
    outT->Branch("eventNum", &eventNum_);
    outT->Branch("lumiSec", &lumiSec_);
    outT->Branch("runNum", &runNum_);
    for (int i = 0; i < numTrigs_; i++) {
        TString branchName = TString::Format("trig_%s", trigNames_[i].c_str());
        outT->Branch(branchName,&trigPassed_[i],branchName+"/O");
    }

    // MET Filters
    outT->Branch("METFiltersFailBits",&METFiltersFailBits_);

    // Muons
    outT->Branch("nMuon",&nMuon_);
    outT->Branch("Muon_pt",&recoMuonPt_);
    outT->Branch("Muon_eta",&recoMuonEta_);
    outT->Branch("Muon_phi",&recoMuonPhi_);
    outT->Branch("Muon_e",&recoMuonEnergy_);
    outT->Branch("Muon_charge",&recoMuonCharge_);
    outT->Branch("Muon_IDcutLoose",&recoMuonIDcutBasedLoose_);
    outT->Branch("Muon_IDcutMedium",&recoMuonIDcutBasedMedium_);
    outT->Branch("Muon_IDcutMediumPrompt",&recoMuonIDcutBasedMediumPrompt_);
    outT->Branch("Muon_IDcutTight",&recoMuonIDcutBasedTight_);
    outT->Branch("Muon_isPFMuon",&recoMuonIsPFMuon_);
    outT->Branch("Muon_isGlobalMuon",&recoMuonIsGlobalMuon_);
    outT->Branch("Muon_isStandAloneMuon",&recoMuonIsStandAloneMuon_);

    // Normal Electrons
    outT->Branch("nElectron",&nElectronDefault_);
    outT->Branch("Electron_pt",&recoElectronPt_);
    outT->Branch("Electron_eta",&recoElectronEta_);
    outT->Branch("Electron_etaErr",&recoElectronEtaError_);
    outT->Branch("Electron_phi",&recoElectronPhi_);
    outT->Branch("Electron_phiErr",&recoElectronPhiError_);
    outT->Branch("Electron_IDcutVeto",&recoElectronID_cutVeto_);
    outT->Branch("Electron_IDcutLoose",&recoElectronID_cutLoose_);
    outT->Branch("Electron_IDcutMed",&recoElectronID_cutMed_);
    outT->Branch("Electron_IDcutTight",&recoElectronID_cutTight_);
    outT->Branch("Electron_IDcutVetoInt",&recoElectronID_cutVetoInt_);
    outT->Branch("Electron_IDcutLooseInt",&recoElectronID_cutLooseInt_);
    outT->Branch("Electron_IDcutMedInt",&recoElectronID_cutMedInt_);
    outT->Branch("Electron_IDcutTightInt",&recoElectronID_cutTightInt_);
    outT->Branch("Electron_IDmvaIso90",&recoElectronID_mvaIso90_);
    outT->Branch("Electron_IDmvaIso80",&recoElectronID_mvaIso80_);
    outT->Branch("Electron_IDmvaIsoLoose",&recoElectronID_mvaIsoLoose_);
    outT->Branch("Electron_IDmva90",&recoElectronID_mva90_);
    outT->Branch("Electron_IDmva80",&recoElectronID_mva80_);
    outT->Branch("Electron_IDmvaLoose",&recoElectronID_mvaLoose_);
    outT->Branch("Electron_angRes",&recoElectronAngularRes_);
    outT->Branch("Electron_e",&recoElectronE_);
    outT->Branch("Electron_vxy",&recoElectronVxy_);
    outT->Branch("Electron_vz",&recoElectronVz_);
    outT->Branch("Electron_dxy",&recoElectronDxy_);
    outT->Branch("Electron_dxyErr",&recoElectronDxyError_);
    outT->Branch("Electron_dz",&recoElectronDz_);
    outT->Branch("Electron_dzErr",&recoElectronDzError_);
    outT->Branch("Electron_trkChi2",&recoElectronTrkChi2_);
    outT->Branch("Electron_trkIso",&recoElectronTrkIso_);
    outT->Branch("Electron_trkRelIso",&recoElectronTrkRelIso_);
    outT->Branch("Electron_calIso",&recoElectronCaloIso_);
    outT->Branch("Electron_calRelIso",&recoElectronCaloRelIso_);
    outT->Branch("Electron_PFIso",&recoElectronPFIso_);
    outT->Branch("Electron_PFRelIso",&recoElectronPFRelIso_);
    outT->Branch("Electron_miniIso",&recoElectronMiniIso_);
    outT->Branch("Electron_miniRelIso",&recoElectronMiniRelIso_);
    outT->Branch("Electron_PFIsoEleCorr",&recoElectronPFIsoEleCorr_);
    outT->Branch("Electron_PFRelIsoEleCorr",&recoElectronPFRelIsoEleCorr_);
    outT->Branch("Electron_miniIsoEleCorr",&recoElectronMiniIsoEleCorr_);
    outT->Branch("Electron_miniRelIsoEleCorr",&recoElectronMiniRelIsoEleCorr_);
    outT->Branch("Electron_chadIso",&recoElectronChadIso_);
    outT->Branch("Electron_nhadIso",&recoElectronNhadIso_);
    outT->Branch("Electron_phoIso",&recoElectronPhoIso_);
    outT->Branch("Electron_rhoEA",&recoElectronRhoEA_);
    outT->Branch("Electron_trkProb",&recoElectronTrkProb_);
    outT->Branch("Electron_numTrackerHits",&recoElectronTrkNumTrackerHits_);
    outT->Branch("Electron_numPixHits",&recoElectronTrkNumPixHits_);
    outT->Branch("Electron_numStripHits",&recoElectronTrkNumStripHits_);
    outT->Branch("Electron_charge",&recoElectronCharge_);
    outT->Branch("Electron_isPF",&recoElectronIsPF_);
    outT->Branch("Electron_genMatched",&recoElectronGenMatched_);
    outT->Branch("Electron_matchType",&recoElectronMatchType_);
    outT->Branch("Electron_dRJets",&recoElectronDrToJets_);
    outT->Branch("Electron_dPhiJets",&recoElectronDphiToJets_);
    outT->Branch("Electron_full55sigmaIetaIeta",&recoElectronFull5x5_sigmaIetaIeta_);
    outT->Branch("Electron_absdEtaSeed",&recoElectronAbsdEtaSeed_);
    outT->Branch("Electron_absdPhiIn",&recoElectronAbsdPhiIn_);
    outT->Branch("Electron_HoverE",&recoElectronHoverE_);
    outT->Branch("Electron_abs1overEm1overP",&recoElectronAbs1overEm1overP_);
    outT->Branch("Electron_expMissingInnerHits",&recoElectronExpMissingInnerHits_);
    outT->Branch("Electron_conversionVeto",&recoElectronConversionVeto_);
    outT->Branch("Electron_isEE",&recoElectronIsEE_);
    // special vars for x-clean study
    outT->Branch("Electron_hasLptMatch",&recoElectronHasLptMatch_);
    outT->Branch("Electron_lptMatchIdx",&recoElectronLptMatchIdx_);
    outT->Branch("Electron_hasAllLptMatch",&recoElectronHasAllLptMatch_);
    outT->Branch("Electron_AllLptMatchIdx",&recoElectronAllLptMatchIdx_);

    // Low pT electrons
    outT->Branch("nLptElectron",&nElectronLowPt_);
    outT->Branch("LptElectron_pt",&recoLowPtElectronPt_);
    outT->Branch("LptElectron_eta",&recoLowPtElectronEta_);
    outT->Branch("LptElectron_etaErr",&recoLowPtElectronEtaError_);
    outT->Branch("LptElectron_phi",&recoLowPtElectronPhi_);
    outT->Branch("LptElectron_phiErr",&recoLowPtElectronPhiError_);
    outT->Branch("LptElectron_ID",&recoLowPtElectronID_);
    outT->Branch("LptElectron_angRes",&recoLowPtElectronAngularRes_);
    outT->Branch("LptElectron_e",&recoLowPtElectronE_);
    outT->Branch("LptElectron_vxy",&recoLowPtElectronVxy_);
    outT->Branch("LptElectron_vz",&recoLowPtElectronVz_);
    outT->Branch("LptElectron_dxy",&recoLowPtElectronDxy_);
    outT->Branch("LptElectron_dxyErr",&recoLowPtElectronDxyError_);
    outT->Branch("LptElectron_dz",&recoLowPtElectronDz_);
    outT->Branch("LptElectron_dzErr",&recoLowPtElectronDzError_);
    outT->Branch("LptElectron_trkChi2",&recoLowPtElectronTrkChi2_);
    outT->Branch("LptElectron_trkIso",&recoLowPtElectronTrkIso_);
    outT->Branch("LptElectron_trkRelIso",&recoLowPtElectronTrkRelIso_);
    outT->Branch("LptElectron_calIso",&recoLowPtElectronCaloIso_);
    outT->Branch("LptElectron_calRelIso",&recoLowPtElectronCaloRelIso_);
    outT->Branch("LptElectron_PFIso",&recoLowPtElectronPFIso_);
    outT->Branch("LptElectron_PFRelIso",&recoLowPtElectronPFRelIso_);
    outT->Branch("LptElectron_miniIso",&recoLowPtElectronMiniIso_);
    outT->Branch("LptElectron_miniRelIso",&recoLowPtElectronMiniRelIso_);
    outT->Branch("LptElectron_PFIsoEleCorr",&recoLowPtElectronPFIsoEleCorr_);
    outT->Branch("LptElectron_PFRelIsoEleCorr",&recoLowPtElectronPFRelIsoEleCorr_);
    outT->Branch("LptElectron_miniIsoEleCorr",&recoLowPtElectronMiniIsoEleCorr_);
    outT->Branch("LptElectron_miniRelIsoEleCorr",&recoLowPtElectronMiniRelIsoEleCorr_);
    outT->Branch("LptElectron_chadIso",&recoLowPtElectronChadIso_);
    outT->Branch("LptElectron_nhadIso",&recoLowPtElectronNhadIso_);
    outT->Branch("LptElectron_phoIso",&recoLowPtElectronPhoIso_);
    outT->Branch("LptElectron_rhoEA",&recoLowPtElectronRhoEA_);
    outT->Branch("LptElectron_trkProb",&recoLowPtElectronTrkProb_);
    outT->Branch("LptElectron_numTrackerHits",&recoLowPtElectronTrkNumTrackerHits_);
    outT->Branch("LptElectron_numPixHits",&recoLowPtElectronTrkNumPixHits_);
    outT->Branch("LptElectron_numStripHits",&recoLowPtElectronTrkNumStripHits_);
    outT->Branch("LptElectron_charge",&recoLowPtElectronCharge_);
    outT->Branch("LptElectron_minDRtoReg",&recoLowPtElectronMinDrToReg_);
    outT->Branch("LptElectron_isPF",&recoLowPtElectronIsPF_);
    outT->Branch("LptElectron_genMatched",&recoLowPtElectronGenMatched_);
    outT->Branch("LptElectron_matchType",&recoLowPtElectronMatchType_);
    outT->Branch("LptElectron_dRJets",&recoLowPtElectronDrToJets_);
    outT->Branch("LptElectron_dPhiJets",&recoLowPtElectronDphiToJets_);
    outT->Branch("LptElectron_full55sigmaIetaIeta",&recoLowPtElectronFull5x5_sigmaIetaIeta_);
    outT->Branch("LptElectron_absdEtaSeed",&recoLowPtElectronAbsdEtaSeed_);
    outT->Branch("LptElectron_absdPhiIn",&recoLowPtElectronAbsdPhiIn_);
    outT->Branch("LptElectron_HoverE",&recoLowPtElectronHoverE_);
    outT->Branch("LptElectron_abs1overEm1overP",&recoLowPtElectronAbs1overEm1overP_);
    outT->Branch("LptElectron_expMissingInnerHits",&recoLowPtElectronExpMissingInnerHits_);
    outT->Branch("LptElectron_conversionVeto",&recoLowPtElectronConversionVeto_);
    outT->Branch("LptElectron_isEE",&recoLowPtElectronIsEE_);
    // special vars for x-cleaning study
    outT->Branch("LptElectron_xCleaned",&recoLowPtElectronIsXCleaned_);
    outT->Branch("LptElectron_gedIdx",&recoLowPtElectronGEDidx_);
    outT->Branch("LptElectron_gedIsMatched",&recoLowPtElectronGEDisMatched_);

    // All Low pT electrons (includes cross-cleaned)
    outT->Branch("nAllLptElectron",&nElectronAllLowPt_);
    outT->Branch("AllLptElectron_pt",&recoAllLowPtElectronPt_);
    outT->Branch("AllLptElectron_eta",&recoAllLowPtElectronEta_);
    outT->Branch("AllLptElectron_etaErr",&recoAllLowPtElectronEtaError_);
    outT->Branch("AllLptElectron_phi",&recoAllLowPtElectronPhi_);
    outT->Branch("AllLptElectron_phiErr",&recoAllLowPtElectronPhiError_);
    outT->Branch("AllLptElectron_ID",&recoAllLowPtElectronID_);
    outT->Branch("AllLptElectron_angRes",&recoAllLowPtElectronAngularRes_);
    outT->Branch("AllLptElectron_e",&recoAllLowPtElectronE_);
    outT->Branch("AllLptElectron_vxy",&recoAllLowPtElectronVxy_);
    outT->Branch("AllLptElectron_vz",&recoAllLowPtElectronVz_);
    outT->Branch("AllLptElectron_dxy",&recoAllLowPtElectronDxy_);
    outT->Branch("AllLptElectron_dxyErr",&recoAllLowPtElectronDxyError_);
    outT->Branch("AllLptElectron_dz",&recoAllLowPtElectronDz_);
    outT->Branch("AllLptElectron_dzErr",&recoAllLowPtElectronDzError_);
    outT->Branch("AllLptElectron_trkChi2",&recoAllLowPtElectronTrkChi2_);
    outT->Branch("AllLptElectron_trkIso",&recoAllLowPtElectronTrkIso_);
    outT->Branch("AllLptElectron_trkRelIso",&recoAllLowPtElectronTrkRelIso_);
    outT->Branch("AllLptElectron_calIso",&recoAllLowPtElectronCaloIso_);
    outT->Branch("AllLptElectron_calRelIso",&recoAllLowPtElectronCaloRelIso_);
    outT->Branch("AllLptElectron_PFIso",&recoAllLowPtElectronPFIso_);
    outT->Branch("AllLptElectron_PFRelIso",&recoAllLowPtElectronPFRelIso_);
    outT->Branch("AllLptElectron_miniIso",&recoAllLowPtElectronMiniIso_);
    outT->Branch("AllLptElectron_miniRelIso",&recoAllLowPtElectronMiniRelIso_);
    outT->Branch("AllLptElectron_PFIsoEleCorr",&recoAllLowPtElectronPFIsoEleCorr_);
    outT->Branch("AllLptElectron_PFRelIsoEleCorr",&recoAllLowPtElectronPFRelIsoEleCorr_);
    outT->Branch("AllLptElectron_miniIsoEleCorr",&recoAllLowPtElectronMiniIsoEleCorr_);
    outT->Branch("AllLptElectron_miniRelIsoEleCorr",&recoAllLowPtElectronMiniRelIsoEleCorr_);
    outT->Branch("AllLptElectron_chadIso",&recoAllLowPtElectronChadIso_);
    outT->Branch("AllLptElectron_nhadIso",&recoAllLowPtElectronNhadIso_);
    outT->Branch("AllLptElectron_phoIso",&recoAllLowPtElectronPhoIso_);
    outT->Branch("AllLptElectron_rhoEA",&recoAllLowPtElectronRhoEA_);
    outT->Branch("AllLptElectron_trkProb",&recoAllLowPtElectronTrkProb_);
    outT->Branch("AllLptElectron_numTrackerHits",&recoAllLowPtElectronTrkNumTrackerHits_);
    outT->Branch("AllLptElectron_numPixHits",&recoAllLowPtElectronTrkNumPixHits_);
    outT->Branch("AllLptElectron_numStripHits",&recoAllLowPtElectronTrkNumStripHits_);
    outT->Branch("AllLptElectron_charge",&recoAllLowPtElectronCharge_);
    outT->Branch("AllLptElectron_minDRtoReg",&recoAllLowPtElectronMinDrToReg_);
    outT->Branch("AllLptElectron_isPF",&recoAllLowPtElectronIsPF_);
    outT->Branch("AllLptElectron_genMatched",&recoAllLowPtElectronGenMatched_);
    outT->Branch("AllLptElectron_matchType",&recoAllLowPtElectronMatchType_);
    outT->Branch("AllLptElectron_dRJets",&recoAllLowPtElectronDrToJets_);
    outT->Branch("AllLptElectron_dPhiJets",&recoAllLowPtElectronDphiToJets_);
    outT->Branch("AllLptElectron_full55sigmaIetaIeta",&recoAllLowPtElectronFull5x5_sigmaIetaIeta_);
    outT->Branch("AllLptElectron_absdEtaSeed",&recoAllLowPtElectronAbsdEtaSeed_);
    outT->Branch("AllLptElectron_absdPhiIn",&recoAllLowPtElectronAbsdPhiIn_);
    outT->Branch("AllLptElectron_HoverE",&recoAllLowPtElectronHoverE_);
    outT->Branch("AllLptElectron_abs1overEm1overP",&recoAllLowPtElectronAbs1overEm1overP_);
    outT->Branch("AllLptElectron_expMissingInnerHits",&recoAllLowPtElectronExpMissingInnerHits_);
    outT->Branch("AllLptElectron_conversionVeto",&recoAllLowPtElectronConversionVeto_);
    outT->Branch("AllLptElectron_isEE",&recoAllLowPtElectronIsEE_);
    // x-cleaning status and GED match info
    outT->Branch("AllLptElectron_xCleaned",&recoAllLowPtElectronIsXCleaned_);
    outT->Branch("AllLptElectron_gedIdx",&recoAllLowPtElectronGEDidx_);
    outT->Branch("AllLptElectron_gedIsMatched",&recoAllLowPtElectronGEDisMatched_);

    // DSA muons
    outT->Branch("nDSAMuon", &nDSAMuon_);
    outT->Branch("recoDSAMuonPt", &recoDSAMuonPt_);
    outT->Branch("recoDSAMuonPtErr", &recoDSAMuonPtErr_);
    outT->Branch("recoDSAMuonEta", &recoDSAMuonEta_);
    outT->Branch("recoDSAMuonEtaErr", &recoDSAMuonEtaErr_);
    outT->Branch("recoDSAMuonPhi", &recoDSAMuonPhi_);
    outT->Branch("recoDSAMuonPhiErr", &recoDSAMuonPhiErr_);
    outT->Branch("recoDSAMuonOuterEta", &recoDSAMuonOuterEta_);
    outT->Branch("recoDSAMuonOuterPhi", &recoDSAMuonOuterPhi_);
    outT->Branch("recoDSAMuonE", &recoDSAMuonE_);
    outT->Branch("recoDSAMuonPx", &recoDSAMuonPx_);
    outT->Branch("recoDSAMuonPy", &recoDSAMuonPy_);
    outT->Branch("recoDSAMuonPz", &recoDSAMuonPz_);
    outT->Branch("recoDSAMuonVxy", &recoDSAMuonVxy_);
    outT->Branch("recoDSAMuonVz", &recoDSAMuonVz_);
    outT->Branch("recoDSAMuonDxy", &recoDSAMuonDxy_);
    outT->Branch("recoDSAMuonDxyError", &recoDSAMuonDxyError_);
    outT->Branch("recoDSAMuonDz", &recoDSAMuonDz_);
    outT->Branch("recoDSAMuonDzError", &recoDSAMuonDzError_);
    outT->Branch("recoDSAMuonTrkChi2", &recoDSAMuonTrkChi2_);
    outT->Branch("recoDSAMuonTrkProb", &recoDSAMuonTrkProb_);
    outT->Branch("recoDSAMuonTrkNumTrackerHits", &recoDSAMuonTrkNumTrackerHits_);
    outT->Branch("recoDSAMuonTrkNumPixHits", &recoDSAMuonTrkNumPixHits_);
    outT->Branch("recoDSAMuonTrkNumStripHits", &recoDSAMuonTrkNumStripHits_);
    outT->Branch("recoDSAMuonCharge", &recoDSAMuonCharge_);
    outT->Branch("recoDSAMuonDisplacedId", &recoDSAMuonDisplacedId_);
    outT->Branch("recoDSAMuonTrkNumCSCHits", &recoDSAMuonTrkNumCSCHits_);
    outT->Branch("recoDSAMuonTrkNumHits", &recoDSAMuonTrkNumHits_);
    outT->Branch("recoDSAMuonTrkNumPlanes", &recoDSAMuonTrkNumPlanes_);
    outT->Branch("recoDSAMuonTrkNumDTHits", &recoDSAMuonTrkNumDTHits_);
    outT->Branch("recoDSAMuonIdx", &recoDSAMuonIdx_);
    
    // Photons
    outT->Branch("nPhoton",&nPhotons_);
    outT->Branch("Photon_et",&PhotonEt_);
    outT->Branch("Photon_eta",&PhotonEta_);
    outT->Branch("Photon_phi",&PhotonPhi_);

    // OOT Photons
    outT->Branch("nootPhoton",&nOOTPhotons_);
    outT->Branch("ootPhoton_et",&ootPhotonEt_);
    outT->Branch("ootPhoton_eta",&ootPhotonEta_);
    outT->Branch("ootPhoton_phi",&ootPhotonPhi_);

    // Photon conversions
    outT->Branch("nConversion",&nConversions_);
    outT->Branch("Conversion_pt",&conversionPt_);
    outT->Branch("Conversion_eta",&conversionEta_);
    outT->Branch("Conversion_phi",&conversionPhi_);
    outT->Branch("Conversion_energy",&conversionE_);
    outT->Branch("Conversion_px",&conversionPx_);
    outT->Branch("Conversion_py",&conversionPy_);
    outT->Branch("Conversion_pz",&conversionPz_);
    outT->Branch("Conversion_vxy",&conversionVxy_);
    outT->Branch("Conversion_vz",&conversionVz_);
    outT->Branch("Conversion_x",&conversionX_);
    outT->Branch("Conversion_y",&conversionY_);
    outT->Branch("Conversion_z",&conversionZ_);
    outT->Branch("Conversion_lxy",&conversionLxy_);
    outT->Branch("Conversion_lz",&conversionLz_);
    outT->Branch("Conversion_lxyPV",&conversionLxyPV_);
    outT->Branch("Conversion_lzPV",&conversionLzPV_);
    outT->Branch("Conversion_dxy",&conversionDxy_);
    outT->Branch("Conversion_dz",&conversionDz_);
    outT->Branch("Conversion_dxyPV",&conversionDxyPV_);
    outT->Branch("Conversion_dzPV",&conversionDzPV_);
    outT->Branch("Conversion_EoverP",&conversionEoverP_);
    outT->Branch("Conversion_EoverPrefit",&conversionEoverPrefit_);
    outT->Branch("Conversion_nSharedHits",&conversionNSharedHits_);
    outT->Branch("Conversion_m",&conversionM_);
    outT->Branch("Conversion_dR",&conversionDr_);
    outT->Branch("Conversion_chi2",&conversionChi2_);
    outT->Branch("Conversion_tk1_nHitsBeforeVtx",&conversion_Trk1nHitsVtx_);
    outT->Branch("Conversion_tk1_pt",&conversion_Trk1Pt_);
    outT->Branch("Conversion_tk1_eta",&conversion_Trk1Eta_);
    outT->Branch("Conversion_tk1_phi",&conversion_Trk1Phi_);
    outT->Branch("Conversion_tk1_chi2",&conversion_Trk1Chi2_);
    outT->Branch("Conversion_tk1_nValidHits",&conversion_Trk1NValidHits_);
    outT->Branch("Conversion_tk1_nLostHits",&conversion_Trk1numLostHits_);
    outT->Branch("Conversion_tk1_dxy",&conversion_Trk1dxy_);
    outT->Branch("Conversion_tk1_dxyBS",&conversion_Trk1dxyBS_);
    outT->Branch("Conversion_tk1_dxyPV",&conversion_Trk1dxyPV_);
    outT->Branch("Conversion_tk1_dz",&conversion_Trk1dz_);
    outT->Branch("Conversion_tk1_dzPV",&conversion_Trk1dzPV_);
    outT->Branch("Conversion_tk2_nHitsBeforeVtx",&conversion_Trk2nHitsVtx_);
    outT->Branch("Conversion_tk2_pt",&conversion_Trk2Pt_);
    outT->Branch("Conversion_tk2_eta",&conversion_Trk2Eta_);
    outT->Branch("Conversion_tk2_phi",&conversion_Trk2Phi_);
    outT->Branch("Conversion_tk2_chi2",&conversion_Trk2Chi2_);
    outT->Branch("Conversion_tk2_nValidHits",&conversion_Trk2NValidHits_);
    outT->Branch("Conversion_tk2_nLostHits",&conversion_Trk2numLostHits_);
    outT->Branch("Conversion_tk2_dxy",&conversion_Trk2dxy_);
    outT->Branch("Conversion_tk2_dxyBS",&conversion_Trk2dxyBS_);
    outT->Branch("Conversion_tk2_dxyPV",&conversion_Trk2dxyPV_);
    outT->Branch("Conversion_tk2_dz",&conversion_Trk2dz_);
    outT->Branch("Conversion_tk2_dzPV",&conversion_Trk2dzPV_);
    

    // Jets
    outT->Branch("nPFJetAll",&PFNJetAll_);
    outT->Branch("nPFJet",&PFNJet_);
    outT->Branch("numPFJetTrueB",&PFNbJetTrue_);
    outT->Branch("numPFJetTaggedB",&PFNbJetTagged_);
    outT->Branch("PFJet_matchedGenJetPt",&PFJet_matchedGenJetPt_);
    outT->Branch("PFJet_matchedGenJetEta",&PFJet_matchedGenJetEta_);
    outT->Branch("PFJet_matchedGenJetPhi",&PFJet_matchedGenJetPhi_);
    outT->Branch("PFJet_rawFactor",&PFJetRawFactor_);
    outT->Branch("PFJet_massRaw",&PFJetMassRaw_);
    outT->Branch("PFJet_ptRaw",&PFJetPtRaw_);
    outT->Branch("PFJet_area",&PFJetArea_);
    outT->Branch("PFJet_energy",&PFJetEnergy_);
    outT->Branch("PFJet_energyRaw",&PFJetEnergyRaw_);
    outT->Branch("PFJet_mass",&PFJetMass_);
    outT->Branch("PFJet_pt",&PFJetPt_);
    outT->Branch("PFJet_eta",&PFJetEta_);
    outT->Branch("PFJet_phi",&PFJetPhi_);
    outT->Branch("PFJet_bTag",&PFJetBTag_);
    outT->Branch("PFJet_truth",&PFJetTruth_);
    outT->Branch("PFJet_bTagEffDenomPt",&PFJetEffDenomPt_);
    outT->Branch("PFJet_bTagEffNumPt",&PFJetEffNumPt_);
    outT->Branch("PFJet_METdPhi",&PFJetMETdPhi_);
    outT->Branch("PFJet_CHEF",&PFJetCorrectedCHEF_);
    outT->Branch("PFJet_NHEF",&PFJetCorrectedNHEF_);
    outT->Branch("PFJet_CEEF",&PFJetCorrectedCEEF_);
    outT->Branch("PFJet_NEEF",&PFJetCorrectedNEEF_);
    outT->Branch("PFJet_corrNumDaughters",&PFJetCorrectedNumDaughters_);
    outT->Branch("PFJet_corrCHM",&PFJetCorrectedChargedMultiplicity_);
    outT->Branch("PFJet_corr_pt",&PFJetCorrectedPt_);
    outT->Branch("PFJet_corrJESUp_pt",&PFJetCorrectedJESUpPt_);
    outT->Branch("PFJet_corrJESUp_eta",&PFJetCorrectedJESUpEta_);
    outT->Branch("PFJet_corrJESUp_phi",&PFJetCorrectedJESUpPhi_);
    outT->Branch("PFJet_corrJESDown_pt",&PFJetCorrectedJESDownPt_);
    outT->Branch("PFJet_corrJESDown_eta",&PFJetCorrectedJESDownEta_);
    outT->Branch("PFJet_corrJESDown_phi",&PFJetCorrectedJESDownPhi_);
    outT->Branch("PFJet_corrJERUp_pt",&PFJetCorrectedJERUpPt_);
    outT->Branch("PFJet_corrJERUp_eta",&PFJetCorrectedJERUpEta_);
    outT->Branch("PFJet_corrJERUp_phi",&PFJetCorrectedJERUpPhi_);
    outT->Branch("PFJet_corrJERDown_pt",&PFJetCorrectedJERDownPt_);
    outT->Branch("PFJet_corrJERDown_eta",&PFJetCorrectedJERDownEta_);
    outT->Branch("PFJet_corrJERDown_phi",&PFJetCorrectedJERDownPhi_);
    outT->Branch("HEM_flag",&PFHEMFlag_);

    // MET
    outT->Branch("PFMET_ET",&PFMET_ET_);
    outT->Branch("PFMET_pt",&PFMET_Pt_);
    outT->Branch("PFMET_phi",&PFMET_Phi_);
    outT->Branch("PFMET_JESUpPt",&PFMETJESUpPt_);
    outT->Branch("PFMET_JESUpPhi",&PFMETJESUpPhi_);
    outT->Branch("PFMET_JESDownPt",&PFMETJESDownPt_);
    outT->Branch("PFMET_JESDownPhi",&PFMETJESDownPhi_);
    outT->Branch("PFMET_JERUpPt",&PFMETJERUpPt_);
    outT->Branch("PFMET_JERUpPhi",&PFMETJERUpPhi_);
    outT->Branch("PFMET_JERDownPt",&PFMETJERDownPt_);
    outT->Branch("PFMET_JERDownPhi",&PFMETJERDownPhi_);
    outT->Branch("PFMET_UnclusteredUpPt",&PFMETUnclusteredUpPt_);
    outT->Branch("PFMET_UnclusteredUpPhi",&PFMETUnclusteredUpPhi_);
    outT->Branch("PFMET_UnclusteredDownPt",&PFMETUnclusteredDownPt_);
    outT->Branch("PFMET_UnclusteredDownPhi",&PFMETUnclusteredDownPhi_);
    outT->Branch("PFMET_JetResUpSmearPt",&PFMETJetResUpSmearPt_);
    outT->Branch("PFMET_JetResUpSmearPhi",&PFMETJetResUpSmearPhi_);
    outT->Branch("PFMET_JetResDownSmearPt",&PFMETJetResDownSmearPt_);
    outT->Branch("PFMET_JetResDownSmearPhi",&PFMETJetResDownSmearPhi_);

    outT->Branch("CaloMET_ET",&CaloMET_ET_);
    outT->Branch("CaloMET_pt",&CaloMET_Pt_);
    outT->Branch("CaloMET_phi",&CaloMET_Phi_);
    
    // Pileup density
    outT->Branch("rho",&rho_);
    outT->Branch("fixedGridRhoFastjetAll",&fixedGridRhoFastjetAll_);

    // PV
    outT->Branch("PV_x",&PV_x_);
    outT->Branch("PV_y",&PV_y_);
    outT->Branch("PV_z",&PV_z_);
    outT->Branch("numPV",&numPV_);

    // Electron-positron vertex branches
    outT->Branch("nvtx",&nvtx_);
    outT->Branch("vtx_typ",&vtx_type_);
    outT->Branch("vtx_vxy", &vtx_recoVtxVxy_);
    outT->Branch("vtx_sigmavxy", &vtx_recoVtxSigmaVxy_);
    outT->Branch("vtx_vx",&vtx_recoVtxVx_);
    outT->Branch("vtx_vy",&vtx_recoVtxVy_);
    outT->Branch("vtx_vz",&vtx_recoVtxVz_);
    outT->Branch("vtx_reduced_chi2", &vtx_recoVtxReducedChi2_);
    outT->Branch("vtx_prob", &vtx_prob_);
    outT->Branch("vtx_dR",  &vtx_recoVtxDr_);
    outT->Branch("vtx_sign",&vtx_recoVtxSign_);
    outT->Branch("vtx_minDxy",&vtx_minDxy_);
    outT->Branch("vtx_METdPhi",&vtx_METdPhi_);
    outT->Branch("vtx_pt",&vtx_ll_pt_);
    outT->Branch("vtx_eta",&vtx_ll_eta_);
    outT->Branch("vtx_phi",&vtx_ll_phi_);
    outT->Branch("vtx_energy",&vtx_ll_e_);
    outT->Branch("vtx_m",&vtx_ll_m_);
    outT->Branch("vtx_px",&vtx_ll_px_);
    outT->Branch("vtx_py",&vtx_ll_py_);
    outT->Branch("vtx_pz",&vtx_ll_pz_);
    outT->Branch("vtx_refit_m",&vtx_refit_m_);
    outT->Branch("vtx_refit_pt",&vtx_refit_pt_);
    outT->Branch("vtx_refit_eta",&vtx_refit_eta_);
    outT->Branch("vtx_refit_phi",&vtx_refit_phi_);
    outT->Branch("vtx_refit_dR",&vtx_refit_dr_);
    outT->Branch("vtx_isMatched",&vtx_isMatched_);
    outT->Branch("vtx_matchSign",&vtx_matchSign_);
    outT->Branch("vtx_dRJets",&vtx_dRtoJets_);
    outT->Branch("vtx_dPhiJets",&vtx_dPhiToJets_);

    outT->Branch("vtx_e1_typ",&vtx_e1_type_);
    outT->Branch("vtx_e1_idx",&vtx_e1_idx_);
    outT->Branch("vtx_e1_isMatched",&vtx_e1_isMatched_);
    outT->Branch("vtx_e1_matchType",&vtx_e1_matchType_);
    outT->Branch("vtx_e1_refit_dxy",&vtx_e1_refitDxy_);
    outT->Branch("vtx_e1_refit_dxyErr",&vtx_e1_refitDxyErr_);
    outT->Branch("vtx_e1_refit_dz",&vtx_e1_refitDz_);
    outT->Branch("vtx_e1_refit_dzErr",&vtx_e1_refitDzErr_);
    outT->Branch("vtx_e1_refit_chi2",&vtx_e1_refitChi2_);

    outT->Branch("vtx_e2_typ",&vtx_e2_type_);
    outT->Branch("vtx_e2_idx",&vtx_e2_idx_);
    outT->Branch("vtx_e2_isMatched",&vtx_e2_isMatched_);
    outT->Branch("vtx_e2_matchType",&vtx_e2_matchType_);
    outT->Branch("vtx_e2_refit_dxy",&vtx_e2_refitDxy_);
    outT->Branch("vtx_e2_refit_dxyErr",&vtx_e2_refitDxy_);
    outT->Branch("vtx_e2_refit_dz",&vtx_e2_refitDz_);
    outT->Branch("vtx_e2_refit_dzErr",&vtx_e2_refitDz_);
    outT->Branch("vtx_e2_refit_chi2",&vtx_e2_refitChi2_);

    // Gen information
    if (!isData_) {
        outT->Branch("genWgt", &genwgt_);
        outT->Branch("genPU_obs", &genpuobs_);
        outT->Branch("genPU_true", &genputrue_);
        outT->Branch("nGenJet",&nGenJet_);
        outT->Branch("GenJet_pt", &genJetPt_);
        outT->Branch("GenJet_eta", &genJetEta_);
        outT->Branch("GenJet_phi", &genJetPhi_);
        outT->Branch("GenJet_METdPhi",&genJetMETdPhi_);
        outT->Branch("GenMET_pt", &genLeadMETPt_);
        outT->Branch("GenMET_phi", &genLeadMETPhi_);
        outT->Branch("GenMET_px",&genLeadMETPx_);
        outT->Branch("GenMET_py",&genLeadMETPy_);
        outT->Branch("GenMET_ET",&genLeadMETET_);
    }
    if (!isData_) {
        outT->Branch("nGenPart",&nGen_);
        outT->Branch("GenPart_ID", &genID_);
        outT->Branch("GenPart_motherID",&genMotherID_);
        outT->Branch("GenPart_charge", &genCharge_);
        outT->Branch("GenPart_pt", &genPt_);
        outT->Branch("GenPart_eta", &genEta_);
        outT->Branch("GenPart_phi", &genPhi_);
        outT->Branch("GenPart_e", &genEn_);
        outT->Branch("GenPart_px",&genPx_);
        outT->Branch("GenPart_py",&genPy_);
        outT->Branch("GenPart_pz", &genPz_);
        outT->Branch("GenPart_vxy", &genVxy_);
        outT->Branch("GenPart_vx",&genVx_);
        outT->Branch("GenPart_vy",&genVy_);
        outT->Branch("GenPart_vz",&genVz_);
        outT->Branch("GenPart_mass", &genMass_);

        if (isSignal_) {
            outT->Branch("GenEle_charge",&genEleCharge_);
            outT->Branch("GenEle_motherID",&genEleMotherID_);
            outT->Branch("GenEle_pt",&genElePt_);
            outT->Branch("GenEle_eta",&genEleEta_);
            outT->Branch("GenEle_phi",&genElePhi_);
            outT->Branch("GenEle_energy",&genEleEn_);
            outT->Branch("GenEle_px",&genElePx_);
            outT->Branch("GenEle_py",&genElePy_);
            outT->Branch("GenEle_pz",&genElePz_);
            outT->Branch("GenEle_vxy",&genEleVxy_);
            outT->Branch("GenEle_vz",&genEleVz_);
            outT->Branch("GenEle_vx",&genEleVx_);
            outT->Branch("GenEle_vy",&genEleVy_);
            outT->Branch("GenEle_matched",&genEleMatched_);
            outT->Branch("GenEle_matchType",&genEleMatchType_);
            outT->Branch("GenEle_matchIdxLocal",&genEleMatchIdxLocal_);
            outT->Branch("GenEle_matchIdxGlobal",&genEleMatchIdxGlobal_);
            outT->Branch("GenEle_matchedAllLowPt",&genEleMatchedAllLowPt_);
            outT->Branch("GenEle_matchIdxAllLowPt",&genEleMatchIdxAllLowPt_);

            outT->Branch("GenEle_GEDPtRes",&genEleGEDPtRes_);
            outT->Branch("GenEle_GEDPtResRel",&genEleGEDPtResRel_);
            outT->Branch("GenEle_LptPtRes",&genEleLptPtRes_);
            outT->Branch("GenEle_LptPtResRel",&genEleLptPtResRel_);

            outT->Branch("GenPos_charge",&genPosCharge_);
            outT->Branch("GenPos_motherID",&genPosMotherID_);
            outT->Branch("GenPos_pt",&genPosPt_);
            outT->Branch("GenPos_eta",&genPosEta_);
            outT->Branch("GenPos_phi",&genPosPhi_);
            outT->Branch("GenPos_energy",&genPosEn_);
            outT->Branch("GenPos_px",&genPosPx_);
            outT->Branch("GenPos_py",&genPosPy_);
            outT->Branch("GenPos_pz",&genPosPz_);
            outT->Branch("GenPos_vxy",&genPosVxy_);
            outT->Branch("GenPos_vz",&genPosVz_);
            outT->Branch("GenPos_vx",&genPosVx_);
            outT->Branch("GenPos_vy",&genPosVy_);
            outT->Branch("GenPos_matched",&genPosMatched_);
            outT->Branch("GenPos_matchType",&genPosMatchType_);
            outT->Branch("GenPos_matchIdxLocal",&genPosMatchIdxLocal_);
            outT->Branch("GenPos_matchIdxGlobal",&genPosMatchIdxGlobal_);
            outT->Branch("GenPos_matchedAllLowPt",&genPosMatchedAllLowPt_);
            outT->Branch("GenPos_matchIdxAllLowPt",&genPosMatchIdxAllLowPt_);

            outT->Branch("GenPos_GEDPtRes",&genPosGEDPtRes_);
            outT->Branch("GenPos_GEDPtResRel",&genPosGEDPtResRel_);
            outT->Branch("GenPos_LptPtRes",&genPosLptPtRes_);
            outT->Branch("GenPos_LptPtResRel",&genPosLptPtResRel_);

            // Signal reco info
            outT->Branch("signalReconstructed",&signalReconstructed_);
            outT->Branch("signalVertexReconstructed",&signalVertexReconstructed_);

            // Gen Electron + Positron info
            outT->Branch("genEE_pt",&genEEPt_);
            outT->Branch("genEE_eta",&genEEEta_);
            outT->Branch("genEE_phi",&genEEPhi_);
            outT->Branch("genEE_energy",&genEEEn_);
            outT->Branch("genEE_mass",&genEEMass_);
            outT->Branch("genEE_dr",&genEEdR_);
            outT->Branch("genEE_METdPhi",&genEEMETdPhi_);
            outT->Branch("genEE_vxy",&genEEVxy_);
            outT->Branch("genEE_vz",&genEEVz_);
            outT->Branch("genEE_vx",&genEEVx_);
            outT->Branch("genEE_vy",&genEEVy_);
        }
    }

}

void NtupleContainerV2::ClearTreeBranches() {
    // Reset trigger
    fired_ = 0;
    for (int i = 0; i < numTrigs_; i++) {
        trigPassed_[i] = false;
    }

    // MET Filters
    METFiltersFailBits_ = 0;

    // Gen particles
    nGen_ = 0;
    genID_.clear();
    genMotherID_.clear();
    genCharge_.clear();
    genPt_.clear();
    genEta_.clear();
    genPhi_.clear();
    genEn_.clear();
    genPx_.clear();
    genPy_.clear();
    genPz_.clear();
    genVxy_.clear();
    genVx_.clear();
    genVy_.clear();
    genVz_.clear();
    genMass_.clear();

    // Gen Electron & Positron from iDM signal
    genEleCharge_ = 0;
    genEleMotherID_ = 0;
    genElePt_ = -999;
    genEleEta_ = -999;
    genElePhi_ = -999;
    genEleEn_ = -999;
    genElePx_ = -999;
    genElePy_ = -999;
    genElePz_ = -999;
    genEleVxy_ = -999;
    genEleVz_ = -999;
    genEleVx_ = -999;
    genEleVy_ = -999;
    genEleMatched_ = false;
    genEleMatchType_ = "None";
    genEleMatchIdxLocal_ = -999;
    genEleMatchIdxGlobal_ = -999;
    genEleMatchedAllLowPt_ = false;
    genEleMatchIdxAllLowPt_ = -999;

    genPosCharge_ = 0;
    genPosMotherID_ = 0;
    genPosPt_ = -999;
    genPosEta_ = -999;
    genPosPhi_ = -999;
    genPosEn_ = -999;
    genPosPx_ = -999;
    genPosPy_ = -999;
    genPosPz_ = -999;
    genPosVxy_ = -999;
    genPosVz_ = -999;
    genPosVx_ = -999;
    genPosVy_ = -999;
    genPosMatched_ = false;
    genPosMatchType_ = "None";
    genPosMatchIdxLocal_ = -999;
    genPosMatchIdxGlobal_ = -999;
    genPosMatchedAllLowPt_ = false;
    genPosMatchIdxAllLowPt_ = -999;

    genEleGEDPtRes_ = -999;
    genEleGEDPtResRel_ = -999;
    genEleLptPtRes_ = -999;
    genEleLptPtResRel_ = -999;
    genPosGEDPtRes_ = -999;
    genPosGEDPtResRel_ = -999;
    genPosLptPtRes_ = -999;
    genPosLptPtResRel_ = -999;

    // Signal reconstruction info
    signalReconstructed_ = false;
    signalVertexReconstructed_ = false;

    // Gen Electron + Positron info
    genEEPt_ = -999;
    genEEEta_ = -999;
    genEEPhi_ = -999;
    genEEEn_ = -999;
    genEEMass_ = -999;
    genEEdR_ = -999;
    genEEMETdPhi_ = -999;
    genEEVxy_ = -999;
    genEEVz_ = -999;
    genEEVx_ = -999;
    genEEVy_ = -999;
    
    // Gen jet
    nGenJet_ = 0;
    genJetPt_.clear();
    genJetEta_.clear();
    genJetPhi_.clear();
    genJetMETdPhi_.clear();

    //Gen MET
    genLeadMETPt_ = -9999;
    genLeadMETPhi_ = -9999;
    genLeadMETET_ = -9999;
    genLeadMETPx_ = -9999;
    genLeadMETPy_ = -9999;

    // Muons
    nMuon_ = 0;
    recoMuonPt_.clear();
    recoMuonEta_.clear();
    recoMuonPhi_.clear();
    recoMuonEnergy_.clear();
    recoMuonCharge_.clear();
    recoMuonIDcutBasedLoose_.clear();
    recoMuonIDcutBasedMedium_.clear();
    recoMuonIDcutBasedMediumPrompt_.clear();
    recoMuonIDcutBasedTight_.clear();
    recoMuonIsPFMuon_.clear();
    recoMuonIsGlobalMuon_.clear();
    recoMuonIsStandAloneMuon_.clear();

    // Electrons
    nElectronDefault_ = 0;
    recoElectronPt_.clear();
    recoElectronEta_.clear();
    recoElectronEtaError_.clear();
    recoElectronPhi_.clear();
    recoElectronPhiError_.clear();
    recoElectronID_cutVeto_.clear();
    recoElectronID_cutLoose_.clear();
    recoElectronID_cutMed_.clear();
    recoElectronID_cutTight_.clear();
    recoElectronID_cutVetoInt_.clear();
    recoElectronID_cutLooseInt_.clear();
    recoElectronID_cutMedInt_.clear();
    recoElectronID_cutTightInt_.clear();
    recoElectronID_mvaIso90_.clear();
    recoElectronID_mvaIso80_.clear();
    recoElectronID_mvaIsoLoose_.clear();
    recoElectronID_mva90_.clear();
    recoElectronID_mva80_.clear();
    recoElectronID_mvaLoose_.clear();
    recoElectronAngularRes_.clear();
    recoElectronE_.clear();
    recoElectronVxy_.clear();
    recoElectronVz_.clear();
    recoElectronDxy_.clear();
    recoElectronDxyError_.clear();
    recoElectronDz_.clear();
    recoElectronDzError_.clear();
    recoElectronTrkChi2_.clear();
    recoElectronTrkIso_.clear();
    recoElectronTrkRelIso_.clear();
    recoElectronCaloIso_.clear();
    recoElectronCaloRelIso_.clear();
    recoElectronPFIso_.clear();
    recoElectronPFRelIso_.clear();
    recoElectronMiniIso_.clear();
    recoElectronMiniRelIso_.clear();
    recoElectronPFIsoEleCorr_.clear();
    recoElectronPFRelIsoEleCorr_.clear();
    recoElectronMiniIsoEleCorr_.clear();
    recoElectronMiniRelIsoEleCorr_.clear();
    recoElectronChadIso_.clear();
    recoElectronNhadIso_.clear();
    recoElectronPhoIso_.clear();
    recoElectronRhoEA_.clear();
    recoElectronTrkProb_.clear();
    recoElectronTrkNumTrackerHits_.clear();
    recoElectronTrkNumPixHits_.clear();
    recoElectronTrkNumStripHits_.clear();
    recoElectronCharge_.clear();
    recoElectronIsPF_.clear();
    recoElectronGenMatched_.clear();
    recoElectronMatchType_.clear();
    recoElectronDrToJets_.clear();
    recoElectronDphiToJets_.clear();
    recoElectronFull5x5_sigmaIetaIeta_.clear();
    recoElectronAbsdEtaSeed_.clear();
    recoElectronAbsdPhiIn_.clear();
    recoElectronHoverE_.clear();
    recoElectronAbs1overEm1overP_.clear();
    recoElectronExpMissingInnerHits_.clear();
    recoElectronConversionVeto_.clear();
    recoElectronIsEE_.clear();
    // special vars for x-cleaning study
    recoElectronHasLptMatch_.clear();
    recoElectronLptMatchIdx_.clear();
    recoElectronHasAllLptMatch_.clear();
    recoElectronAllLptMatchIdx_.clear();

    // Low pT electrons
    nElectronLowPt_ = 0;
    recoLowPtElectronPt_.clear();
    recoLowPtElectronEta_.clear();
    recoLowPtElectronEtaError_.clear();
    recoLowPtElectronPhi_.clear();
    recoLowPtElectronPhiError_.clear();
    recoLowPtElectronID_.clear();
    recoLowPtElectronAngularRes_.clear();
    recoLowPtElectronE_.clear();
    recoLowPtElectronVxy_.clear();
    recoLowPtElectronVz_.clear();
    recoLowPtElectronDxy_.clear();
    recoLowPtElectronDxyError_.clear();
    recoLowPtElectronDz_.clear();
    recoLowPtElectronDzError_.clear();
    recoLowPtElectronTrkChi2_.clear();
    recoLowPtElectronTrkIso_.clear();
    recoLowPtElectronTrkRelIso_.clear();
    recoLowPtElectronCaloIso_.clear();
    recoLowPtElectronCaloRelIso_.clear();
    recoLowPtElectronPFIso_.clear();
    recoLowPtElectronPFRelIso_.clear();
    recoLowPtElectronMiniIso_.clear();
    recoLowPtElectronMiniRelIso_.clear();
    recoLowPtElectronPFIsoEleCorr_.clear();
    recoLowPtElectronPFRelIsoEleCorr_.clear();
    recoLowPtElectronMiniIsoEleCorr_.clear();
    recoLowPtElectronMiniRelIsoEleCorr_.clear();
    recoLowPtElectronChadIso_.clear();
    recoLowPtElectronNhadIso_.clear();
    recoLowPtElectronPhoIso_.clear();
    recoLowPtElectronRhoEA_.clear();
    recoLowPtElectronTrkProb_.clear();
    recoLowPtElectronTrkNumTrackerHits_.clear();
    recoLowPtElectronTrkNumPixHits_.clear();
    recoLowPtElectronTrkNumStripHits_.clear();
    recoLowPtElectronCharge_.clear();
    recoLowPtElectronMinDrToReg_.clear();
    recoLowPtElectronIsPF_.clear();
    recoLowPtElectronGenMatched_.clear();
    recoLowPtElectronMatchType_.clear();
    recoLowPtElectronDrToJets_.clear();
    recoLowPtElectronDphiToJets_.clear();
    recoLowPtElectronFull5x5_sigmaIetaIeta_.clear();
    recoLowPtElectronAbsdEtaSeed_.clear();
    recoLowPtElectronAbsdPhiIn_.clear();
    recoLowPtElectronHoverE_.clear();
    recoLowPtElectronAbs1overEm1overP_.clear();
    recoLowPtElectronExpMissingInnerHits_.clear();
    recoLowPtElectronConversionVeto_.clear();
    recoLowPtElectronIsEE_.clear();
    // special vars for x-cleaning study
    recoLowPtElectronIsXCleaned_.clear();
    recoLowPtElectronGEDidx_.clear();
    recoLowPtElectronGEDisMatched_.clear();

    // Low pT electrons
    nElectronAllLowPt_ = 0;
    recoAllLowPtElectronPt_.clear();
    recoAllLowPtElectronEta_.clear();
    recoAllLowPtElectronEtaError_.clear();
    recoAllLowPtElectronPhi_.clear();
    recoAllLowPtElectronPhiError_.clear();
    recoAllLowPtElectronID_.clear();
    recoAllLowPtElectronAngularRes_.clear();
    recoAllLowPtElectronE_.clear();
    recoAllLowPtElectronVxy_.clear();
    recoAllLowPtElectronVz_.clear();
    recoAllLowPtElectronDxy_.clear();
    recoAllLowPtElectronDxyError_.clear();
    recoAllLowPtElectronDz_.clear();
    recoAllLowPtElectronDzError_.clear();
    recoAllLowPtElectronTrkChi2_.clear();
    recoAllLowPtElectronTrkIso_.clear();
    recoAllLowPtElectronTrkRelIso_.clear();
    recoAllLowPtElectronCaloIso_.clear();
    recoAllLowPtElectronCaloRelIso_.clear();
    recoAllLowPtElectronPFIso_.clear();
    recoAllLowPtElectronPFRelIso_.clear();
    recoAllLowPtElectronMiniIso_.clear();
    recoAllLowPtElectronMiniRelIso_.clear();
    recoAllLowPtElectronPFIsoEleCorr_.clear();
    recoAllLowPtElectronPFRelIsoEleCorr_.clear();
    recoAllLowPtElectronMiniIsoEleCorr_.clear();
    recoAllLowPtElectronMiniRelIsoEleCorr_.clear();
    recoAllLowPtElectronChadIso_.clear();
    recoAllLowPtElectronNhadIso_.clear();
    recoAllLowPtElectronPhoIso_.clear();
    recoAllLowPtElectronRhoEA_.clear();
    recoAllLowPtElectronTrkProb_.clear();
    recoAllLowPtElectronTrkNumTrackerHits_.clear();
    recoAllLowPtElectronTrkNumPixHits_.clear();
    recoAllLowPtElectronTrkNumStripHits_.clear();
    recoAllLowPtElectronCharge_.clear();
    recoAllLowPtElectronMinDrToReg_.clear();
    recoAllLowPtElectronIsPF_.clear();
    recoAllLowPtElectronGenMatched_.clear();
    recoAllLowPtElectronMatchType_.clear();
    recoAllLowPtElectronDrToJets_.clear();
    recoAllLowPtElectronDphiToJets_.clear();
    recoAllLowPtElectronFull5x5_sigmaIetaIeta_.clear();
    recoAllLowPtElectronAbsdEtaSeed_.clear();
    recoAllLowPtElectronAbsdPhiIn_.clear();
    recoAllLowPtElectronHoverE_.clear();
    recoAllLowPtElectronAbs1overEm1overP_.clear();
    recoAllLowPtElectronExpMissingInnerHits_.clear();
    recoAllLowPtElectronConversionVeto_.clear();
    recoAllLowPtElectronIsEE_.clear();
    recoAllLowPtElectronIsXCleaned_.clear();
    recoAllLowPtElectronGEDidx_.clear();
    recoAllLowPtElectronGEDisMatched_.clear();

    // Gen weight and pileup
    genwgt_ = 0;
    genpuobs_ = -9999;
    genputrue_ = -9999;

    // DSA muons
    nDSAMuon_ = 0;
    recoDSAMuonPt_.clear();
    recoDSAMuonEta_.clear();
    recoDSAMuonPtErr_.clear();
    recoDSAMuonEtaErr_.clear();
    recoDSAMuonPhi_.clear();
    recoDSAMuonPhiErr_.clear();
    recoDSAMuonOuterEta_.clear();
    recoDSAMuonOuterPhi_.clear();
    recoDSAMuonE_.clear();
    recoDSAMuonPx_.clear();
    recoDSAMuonPy_.clear();
    recoDSAMuonPz_.clear();
    recoDSAMuonVxy_.clear();
    recoDSAMuonVz_.clear();
    recoDSAMuonDxy_.clear();
    recoDSAMuonDxyError_.clear();
    recoDSAMuonDz_.clear();
    recoDSAMuonDzError_.clear();
    recoDSAMuonTrkChi2_.clear();
    recoDSAMuonTrkProb_.clear();
    recoDSAMuonTrkNumTrackerHits_.clear();
    recoDSAMuonTrkNumPixHits_.clear();
    recoDSAMuonTrkNumStripHits_.clear();
    recoDSAMuonCharge_.clear();
    recoDSAMuonDisplacedId_.clear();
    recoDSAMuonTrkNumCSCHits_.clear();
    recoDSAMuonTrkNumHits_.clear();
    recoDSAMuonTrkNumPlanes_.clear();
    recoDSAMuonTrkNumDTHits_.clear();
    recoDSAMuonIdx_.clear();

    // Photons
    nPhotons_ = 0;
    PhotonEt_.clear();
    PhotonEta_.clear();
    PhotonPhi_.clear();

    // OOT Photons
    nOOTPhotons_ = 0;
    ootPhotonEt_.clear();
    ootPhotonEta_.clear();
    ootPhotonPhi_.clear();

    // Photon conversions
    nConversions_ = 0;
    conversionPt_.clear();
    conversionEta_.clear();
    conversionPhi_.clear();
    conversionE_.clear();
    conversionPx_.clear();
    conversionPy_.clear();
    conversionPz_.clear();
    conversionVxy_.clear();
    conversionVz_.clear();
    conversionX_.clear();
    conversionY_.clear();
    conversionZ_.clear();
    conversionLxy_.clear();
    conversionLz_.clear();
    conversionLxyPV_.clear();
    conversionLzPV_.clear();
    conversionDxy_.clear();
    conversionDz_.clear();
    conversionDxyPV_.clear();
    conversionDzPV_.clear();
    conversionEoverP_.clear();
    conversionEoverPrefit_.clear();
    conversionNSharedHits_.clear();
    conversionM_.clear();
    conversionDr_.clear();
    conversionChi2_.clear();
    conversion_Trk1nHitsVtx_.clear();
    conversion_Trk1Pt_.clear();
    conversion_Trk1Eta_.clear();
    conversion_Trk1Phi_.clear();
    conversion_Trk1Chi2_.clear();
    conversion_Trk1NValidHits_.clear();
    conversion_Trk1numLostHits_.clear();
    conversion_Trk1dxy_.clear();
    conversion_Trk1dxyBS_.clear();
    conversion_Trk1dxyPV_.clear();
    conversion_Trk1dz_.clear();
    conversion_Trk1dzPV_.clear();
    conversion_Trk2nHitsVtx_.clear();
    conversion_Trk2Pt_.clear();
    conversion_Trk2Eta_.clear();
    conversion_Trk2Phi_.clear();
    conversion_Trk2Chi2_.clear();
    conversion_Trk2NValidHits_.clear();
    conversion_Trk2numLostHits_.clear();
    conversion_Trk2dxy_.clear();
    conversion_Trk2dxyBS_.clear();
    conversion_Trk2dxyPV_.clear();
    conversion_Trk2dz_.clear();
    conversion_Trk2dzPV_.clear();

    // Jets
    PFNJet_ = 0;
    PFNJetAll_ = 0;
    PFNbJetTrue_ = 0;
    PFNbJetTagged_ = 0;
    PFJet_matchedGenJetPt_.clear();
    PFJet_matchedGenJetEta_.clear();
    PFJet_matchedGenJetPhi_.clear();
    PFJetRawFactor_.clear();
    PFJetMassRaw_.clear();
    PFJetPtRaw_.clear();
    PFJetArea_.clear();
    PFJetMass_.clear();
    PFJetEnergy_.clear();
    PFJetEnergyRaw_.clear();
    PFJetPt_.clear();
    PFJetEta_.clear();
    PFJetPhi_.clear();
    PFJetBTag_.clear();
    PFJetTruth_.clear();
    PFJetEffDenomPt_.clear();
    PFJetEffNumPt_.clear();
    PFJetMETdPhi_.clear();
    PFJetCorrectedCHEF_.clear();
    PFJetCorrectedNHEF_.clear();
    PFJetCorrectedCEEF_.clear();
    PFJetCorrectedNEEF_.clear();
    PFJetCorrectedNumDaughters_.clear();
    PFJetCorrectedChargedMultiplicity_.clear();
    PFJetCorrectedPt_.clear();
    PFJetCorrectedJESUpPt_.clear();
    PFJetCorrectedJESUpEta_.clear();
    PFJetCorrectedJESUpPhi_.clear();
    PFJetCorrectedJESDownPt_.clear();
    PFJetCorrectedJESDownEta_.clear();
    PFJetCorrectedJESDownPhi_.clear();
    PFJetCorrectedJERUpPt_.clear();
    PFJetCorrectedJERUpEta_.clear();
    PFJetCorrectedJERUpPhi_.clear();
    PFJetCorrectedJERDownPt_.clear();
    PFJetCorrectedJERDownEta_.clear();
    PFJetCorrectedJERDownPhi_.clear();
    PFHEMFlag_ = false;

    // MET 
    PFMET_ET_ = -999;
    PFMET_Pt_ = -999;
    PFMET_Phi_ = -999;
    PFMETJESUpPt_ = -9999;
    PFMETJESUpPhi_ = -9999;
    PFMETJESDownPt_ = -9999;
    PFMETJESDownPhi_ = -9999;
    PFMETJERUpPt_ = -9999;
    PFMETJERUpPhi_ = -9999;
    PFMETJERDownPt_ = -9999;
    PFMETJERDownPhi_ = -9999;
    PFMETUnclusteredUpPt_ = -9999;
    PFMETUnclusteredUpPhi_ = -9999;
    PFMETUnclusteredDownPt_ = -9999;
    PFMETUnclusteredDownPhi_ = -9999;
    PFMETJetResUpSmearPt_ = -9999;
    PFMETJetResUpSmearPhi_ = -9999;
    PFMETJetResDownSmearPt_ = -9999;
    PFMETJetResDownSmearPhi_ = -9999;
    
    CaloMET_ET_ = -999;
    CaloMET_Pt_ = -999;
    CaloMET_Phi_ = -999;

    // Pileup density
    rho_ = -9999;

    fixedGridRhoFastjetAll_ = -999.;

    // PV
    PV_x_ = -999.;
    PV_y_ = -999.;
    PV_z_ = -999.;
    numPV_ = 0;

    // Electron-positron vertices
    nvtx_ = 0;
    vtx_type_.clear();
    vtx_recoVtxVxy_.clear();
    vtx_recoVtxVz_.clear();
    vtx_recoVtxSigmaVxy_.clear();
    vtx_recoVtxVx_.clear();
    vtx_recoVtxVy_.clear();
    vtx_recoVtxVz_.clear();
    vtx_recoVtxReducedChi2_.clear();
    vtx_prob_.clear();
    vtx_recoVtxDr_.clear();
    vtx_recoVtxSign_.clear();
    vtx_minDxy_.clear();
    vtx_METdPhi_.clear();
    vtx_ll_pt_.clear();
    vtx_ll_eta_.clear();
    vtx_ll_phi_.clear();
    vtx_ll_e_.clear();
    vtx_ll_m_.clear();
    vtx_ll_px_.clear();
    vtx_ll_py_.clear();
    vtx_ll_pz_.clear();
    vtx_refit_m_.clear();
    vtx_refit_pt_.clear();
    vtx_refit_eta_.clear();
    vtx_refit_phi_.clear();
    vtx_refit_dr_.clear();
    vtx_isMatched_.clear();
    vtx_matchSign_.clear();
    vtx_dRtoJets_.clear();
    vtx_dPhiToJets_.clear();

    vtx_e1_type_.clear();
    vtx_e1_idx_.clear();
    vtx_e1_isMatched_.clear();
    vtx_e1_matchType_.clear();
    vtx_e1_refitDxy_.clear();
    vtx_e1_refitDxyErr_.clear();
    vtx_e1_refitDz_.clear();
    vtx_e1_refitDzErr_.clear();
    vtx_e1_refitChi2_.clear();

    vtx_e2_type_.clear();
    vtx_e2_idx_.clear();
    vtx_e2_isMatched_.clear();
    vtx_e2_matchType_.clear();
    vtx_e2_refitDxy_.clear();
    vtx_e2_refitDxyErr_.clear();
    vtx_e2_refitDz_.clear();
    vtx_e2_refitDzErr_.clear();
    vtx_e2_refitChi2_.clear();
}
