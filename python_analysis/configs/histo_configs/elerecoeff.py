try:
    from histobins import *
except ModuleNotFoundError:
    from configs.hists.histobins import *
from hist import Hist
import hist
import numpy as np
import awkward as ak

def make_histograms():
    histograms = {
        "ele_reco_lpt_pt_lxy":  Hist(samp, cut, ele_pt_zoom_fine, vxy_zoom_fine, storage=hist.storage.Weight()),
        "ele_reco_ged_pt_lxy":  Hist(samp, cut, ele_pt_zoom_fine, vxy_zoom_fine, storage=hist.storage.Weight()),
        "ele_reco_none_pt_lxy": Hist(samp, cut, ele_pt_zoom_fine, vxy_zoom_fine, storage=hist.storage.Weight()),
        "ele_reco_lpt_pt_eta":  Hist(samp, cut, ele_pt_zoom_fine, ele_eta_fine,  storage=hist.storage.Weight()),
        "ele_reco_ged_pt_eta":  Hist(samp, cut, ele_pt_zoom_fine, ele_eta_fine,  storage=hist.storage.Weight()),
        "ele_reco_none_pt_eta": Hist(samp, cut, ele_pt_zoom_fine, ele_eta_fine,  storage=hist.storage.Weight()),
        "ele_reco_lpt_pt_vz":   Hist(samp, cut, ele_pt_zoom_fine, vz_fine,       storage=hist.storage.Weight()),
        "ele_reco_ged_pt_vz":   Hist(samp, cut, ele_pt_zoom_fine, vz_fine,       storage=hist.storage.Weight()),
        "ele_reco_none_pt_vz":  Hist(samp, cut, ele_pt_zoom_fine, vz_fine,       storage=hist.storage.Weight()),
    }
    return histograms

subroutines = []
    
#['vtx', 'nDSAMuon', 'recoDSAMuonDzError', 'numPFJetTrueB', 'genPU', 'signalReconstructed', 'fixedGridRhoFastjetAll', 'recoDSAMuonTrkProb', 'recoDSAMuonDz', 'recoDSAMuonCharge', 'trigFired','recoDSAMuonPy', 'genEE', 'CaloMET', 'GenPart', 'recoDSAMuonOuterEta', 'trig', 'recoDSAMuonDisplacedId', 'recoDSAMuonTrkNumTrackerHits', 'recoDSAMuonTrkNumHits', 'recoDSAMuonTrkNumDTHits', 'recoDSAMuonE', 'GenPos', 'GenMET', 'rho', 'PV', 'numPV', 'recoDSAMuonTrkNumStripHits', 'recoDSAMuonPhiErr', 'recoDSAMuonTrkNumPlanes', 'ootPhoton', 'GenJet', 'recoDSAMuonEta', 'recoDSAMuonVz', 'PFJet', 'Electron', 'runNum', 'recoDSAMuonOuterPhi', 'METFiltersFailBits', 'nPFJetAll', 'numPFJetTaggedB', 'HEM', 'genWgt', 'recoDSAMuonPz', 'recoDSAMuonVxy', 'recoDSAMuonTrkChi2', 'recoDSAMuonPtErr', 'recoDSAMuonPt', 'GenEle', 'recoDSAMuonTrkNumPixHits', 'recoDSAMuonDxyError', 'Muon', 'recoDSAMuonTrkNumCSCHits', 'LptElectron', 'recoDSAMuonPhi', 'eventNum', 'recoDSAMuonDxy', 'recoDSAMuonIdx', 'recoDSAMuonEtaErr', 'Photon', 'Conversion', 'PFMET', 'recoDSAMuonPx', 'lumiSec', 'eventWgt', 'hasHEMjet', 'hasHEMjetBug', 'hasHEMelecPF', 'hasHEMelecLpt', 'GenJetMETdPhi', 'nJets', 'good_vtx', 'nGoodVtx', 'sel_vtx']
#Electron ['pt', 'eta', 'etaErr', 'phi', 'phiErr', 'IDcutVeto', 'IDcutLoose', 'IDcutMed', 'IDcutTight', 'IDcutVetoInt', 'IDcutLooseInt', 'IDcutMedInt', 'IDcutTightInt', 'IDmvaIso90', 'IDmvaIso80', 'IDmvaIsoLoose', 'IDmva90', 'IDmva80', 'IDmvaLoose', 'angRes', 'e', 'vxy', 'vz', 'dxy', 'dxyErr', 'dz', 'dzErr', 'trkChi2', 'trkIso', 'trkRelIso', 'calIso', 'calRelIso', 'PFIso', 'PFRelIso', 'miniIso', 'miniRelIso', 'PFIsoEleCorr', 'PFRelIsoEleCorr', 'miniIsoEleCorr', 'miniRelIsoEleCorr', 'chadIso', 'nhadIso', 'phoIso', 'rhoEA', 'trkProb', 'numTrackerHits', 'numPixHits', 'numStripHits', 'charge', 'isPF', 'genMatched', 'matchType', 'dRJets', 'dPhiJets', 'full55sigmaIetaIeta', 'absdEtaSeed', 'absdPhiIn', 'HoverE', 'abs1overEm1overP', 'expMissingInnerHits', 'conversionVeto', 'isEE', 'hasLptMatch', 'lptMatchIdx', 'mindRj', 'mindPhiJ', 'IDscore', 'passID', 'passIDBasic']
#sel_vtx ['typ', 'vxy', 'sigmavxy', 'vx', 'vy', 'vz', 'reduced_chi2', 'prob', 'dR', 'sign', 'minDxy', 'METdPhi', 'pt', 'eta', 'phi', 'energy', 'm', 'px', 'py', 'pz', 'refit_m', 'refit_pt', 'refit_eta', 'refit_phi', 'refit_dR', 'isMatched', 'matchSign', 'dRJets', 'dPhiJets', 'e1_typ', 'e1_idx', 'e1_isMatched', 'e1_matchType', 'e1_refit_dxy', 'e1_refit_dxyErr', 'e1_refit_dz', 'e1_refit_dzErr', 'e1_refit_chi2', 'e2_typ', 'e2_idx', 'e2_isMatched', 'e2_matchType', 'e2_refit_dxy', 'e2_refit_dxyErr', 'e2_refit_dz', 'e2_refit_dzErr', 'e2_refit_chi2', 'mindRj', 'mindPhiJ', 'corrMinDxy', 'cos_collinear', 'projectedLxy', 'cos_collinear_fromPV', 'cos_collinear_fromPV_refit', 'e1', 'e2', 'min_dxy', 'eleDphi', 'vxy_fromPV', 'gen_cos_collinear_fromPV', 'isGood']
#LptElectron ['pt', 'eta', 'etaErr', 'phi', 'phiErr', 'ID', 'angRes', 'e', 'vxy', 'vz', 'dxy', 'dxyErr', 'dz', 'dzErr', 'trkChi2', 'trkIso', 'trkRelIso', 'calIso', 'calRelIso', 'PFIso', 'PFRelIso', 'miniIso', 'miniRelIso', 'PFIsoEleCorr', 'PFRelIsoEleCorr', 'miniIsoEleCorr', 'miniRelIsoEleCorr', 'chadIso', 'nhadIso', 'phoIso', 'rhoEA', 'trkProb', 'numTrackerHits', 'numPixHits', 'numStripHits', 'charge', 'minDRtoReg', 'isPF', 'genMatched', 'matchType', 'dRJets', 'dPhiJets', 'full55sigmaIetaIeta', 'absdEtaSeed', 'absdPhiIn', 'HoverE', 'abs1overEm1overP', 'expMissingInnerHits', 'conversionVeto', 'isEE', 'xCleaned', 'gedIdx', 'gedIsMatched', 'mindRj', 'mindPhiJ', 'IDscore', 'passID', 'passIDBasic']
# GenEle/Pos ['charge', 'motherID', 'pt', 'eta', 'phi', 'energy', 'px', 'py', 'pz', 'vxy', 'vz', 'vx', 'vy', 'matched', 'matchType', 'matchIdxLocal', 'matchIdxGlobal', 'dr', 'dRbin', 'vxyBin', 'ptBin', 'matchPassID', 'matchPassIDBasic']

def fillHistos(events, hists, samp, cut, info, sum_wgt=1):
    #wgt = events.eventWgt/sum_wgt
    #wgt = events.eventWgt/np.sum(events.eventWgt)
    wgt = events.eventWgt
    #print(sum_wgt, np.sum(events.eventWgt), np.sum(wgt))

    LptMatchedEle = events.GenEle[events.GenEle.matchType == 'L']
    GEDMatchedEle = events.GenEle[events.GenEle.matchType == 'R']
    unMatchedEle = events.GenEle[events.GenEle.matchType == 'None']
    LptMatchedPos = events.GenPos[events.GenPos.matchType == 'L']
    GEDMatchedPos = events.GenPos[events.GenPos.matchType == 'R']
    unMatchedPos = events.GenPos[events.GenPos.matchType == 'None']

    LptMatchedEleWgt = wgt[events.GenEle.matchType == 'L']
    GEDMatchedEleWgt = wgt[events.GenEle.matchType == 'R']
    unMatchedEleWgt = wgt[events.GenEle.matchType == 'None']
    LptMatchedPosWgt = wgt[events.GenPos.matchType == 'L']
    GEDMatchedPosWgt = wgt[events.GenPos.matchType == 'R']
    unMatchedPosWgt = wgt[events.GenPos.matchType == 'None']

    for ele, wgt, prefix in zip([LptMatchedEle, GEDMatchedEle, unMatchedEle],
                                [LptMatchedEleWgt, GEDMatchedEleWgt, unMatchedEleWgt],
                                ["ele_reco_lpt", "ele_reco_ged", "ele_reco_none"]):
        hists[f"{prefix}_pt_lxy"].fill(samp=samp, cut=cut, pt=ele.pt, vxy=ele.vxy, weight=wgt)
        hists[f"{prefix}_pt_eta"].fill(samp=samp, cut=cut, pt=ele.pt, eta=ele.eta, weight=wgt)
        hists[f"{prefix}_pt_vz" ].fill(samp=samp, cut=cut, pt=ele.pt, vz=ele.vz,  weight=wgt)

    for ele, wgt, prefix in zip([LptMatchedPos, GEDMatchedPos, unMatchedPos],
                                [LptMatchedPosWgt, GEDMatchedPosWgt, unMatchedPosWgt],
                                ["ele_reco_lpt", "ele_reco_ged", "ele_reco_none"]):
        hists[f"{prefix}_pt_lxy"].fill(samp=samp, cut=cut, pt=ele.pt, vxy=ele.vxy, weight=wgt)
        hists[f"{prefix}_pt_eta"].fill(samp=samp, cut=cut, pt=ele.pt, eta=ele.eta, weight=wgt)
        hists[f"{prefix}_pt_vz" ].fill(samp=samp, cut=cut, pt=ele.pt, vz=ele.vz,  weight=wgt)
   

        
        
