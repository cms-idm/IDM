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
        "ele_reco_lpt_pt_lxy":      Hist(samp, cut, ele_pt_special, ele_lxy_special, storage=hist.storage.Weight()),
        "ele_reco_ged_pt_lxy":      Hist(samp, cut, ele_pt_special, ele_lxy_special, storage=hist.storage.Weight()),
        "ele_reco_none_pt_lxy":     Hist(samp, cut, ele_pt_special, ele_lxy_special, storage=hist.storage.Weight()),
        "ele_reco_lpt_eta":         Hist(samp, cut, ele_eta_fine,  storage=hist.storage.Weight()),
        "ele_reco_ged_eta":         Hist(samp, cut, ele_eta_fine,  storage=hist.storage.Weight()),
        "ele_reco_none_eta":        Hist(samp, cut, ele_eta_fine,  storage=hist.storage.Weight()),
        "ele_reco_lpt_lz":          Hist(samp, cut, Lz_fine,       storage=hist.storage.Weight()),
        "ele_reco_ged_lz":          Hist(samp, cut, Lz_fine,       storage=hist.storage.Weight()),
        "ele_reco_none_lz":         Hist(samp, cut, Lz_fine,       storage=hist.storage.Weight()),
        "ele_reco_alllpt_pt_lxy":   Hist(samp, cut, ele_pt_special, ele_lxy_special, storage=hist.storage.Weight()),
        "ele_reco_noalllpt_pt_lxy": Hist(samp, cut, ele_pt_special, ele_lxy_special, storage=hist.storage.Weight()),
        "ele_reco_alllpt_eta":      Hist(samp, cut, ele_eta_fine,  storage=hist.storage.Weight()),
        "ele_reco_noalllpt_eta":    Hist(samp, cut, ele_eta_fine,  storage=hist.storage.Weight()),
        "ele_reco_alllpt_lz":       Hist(samp, cut, Lz_fine,       storage=hist.storage.Weight()),
        "ele_reco_noalllpt_lz":     Hist(samp, cut, Lz_fine,       storage=hist.storage.Weight()),
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

    # lxy/lz are measured from the chi2 production vertex (the true, unsmeared
    # primary vertex) rather than the reconstructed PV, which carries ~10-15um
    # of resolution/bias that would otherwise leak into a "truth" quantity.
    chi2 = ak.firsts(events.GenPart[np.abs(events.GenPart.ID) == 1000023])
    lxy_ele = np.sqrt((events.GenEle.vx - chi2.vx)**2 + (events.GenEle.vy - chi2.vy)**2)
    lxy_pos = np.sqrt((events.GenPos.vx - chi2.vx)**2 + (events.GenPos.vy - chi2.vy)**2)
    lz_ele = np.abs(events.GenEle.vz - chi2.vz)
    lz_pos = np.abs(events.GenPos.vz - chi2.vz)
    
    all_gen           = ak.concatenate([events.GenEle,                    events.GenPos],                   axis=0)
    all_lxy           = ak.concatenate([lxy_ele,                          lxy_pos],                         axis=0)
    all_lz            = ak.concatenate([lz_ele,                           lz_pos],                          axis=0)
    all_wgt           = ak.concatenate([wgt,                              wgt],                             axis=0)
    all_matchedalllpt = ak.concatenate([events.GenEle.matchedAllLowPt,    events.GenPos.matchedAllLowPt],   axis=0)

    for mtype, prefix in [('L', 'ele_reco_lpt'), ('R', 'ele_reco_ged'), ('None', 'ele_reco_none')]:
        mask = all_gen.matchType == mtype
        ele  = all_gen[mask]
        lxy  = all_lxy[mask]
        lz   = all_lz[mask]
        w    = all_wgt[mask]
        hists[f"{prefix}_pt_lxy"].fill(samp=samp, cut=cut, pt=ele.pt, lxy=lxy, weight=w)
        hists[f"{prefix}_eta"].fill(   samp=samp, cut=cut, eta=ele.eta,        weight=w)
        hists[f"{prefix}_lz" ].fill(   samp=samp, cut=cut, lz=lz,              weight=w)

    for matched, prefix in [(True, 'ele_reco_alllpt'), (False, 'ele_reco_noalllpt')]:
        mask = all_matchedalllpt == matched
        ele  = all_gen[mask]
        lxy  = all_lxy[mask]
        lz   = all_lz[mask]
        w    = all_wgt[mask]
        hists[f"{prefix}_pt_lxy"].fill(samp=samp, cut=cut, pt=ele.pt, lxy=lxy, weight=w)
        hists[f"{prefix}_eta"].fill(   samp=samp, cut=cut, eta=ele.eta,        weight=w)
        hists[f"{prefix}_lz" ].fill(   samp=samp, cut=cut, lz=lz,              weight=w)
   

        
        
