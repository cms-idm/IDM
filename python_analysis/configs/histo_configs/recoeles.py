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
        # quantities associated w/ selected vertex
        "reco_N_pairs":                   Hist(samp, cut, nvtxs,         storage=hist.storage.Weight()),
        "reco_N_vtx":                     Hist(samp, cut, nvtxs,         storage=hist.storage.Weight()),
        "reco_N_good_vtx":                Hist(samp, cut, nvtxs,         storage=hist.storage.Weight()),
        "reco_N_jets":                    Hist(samp, cut, njets,         storage=hist.storage.Weight()),
        "reco_N_photons":                 Hist(samp, cut, nphos,         storage=hist.storage.Weight()),
        "reco_N_Lpt_eles":                Hist(samp, cut, neles,         storage=hist.storage.Weight()),
        "reco_N_GED_eles":                Hist(samp, cut, neles,         storage=hist.storage.Weight()),
        "reco_ele_lpt_leading_pt" :       Hist(samp, cut, ele_pt_zoom_fine,        storage=hist.storage.Weight()),
        "reco_ele_lpt_subleading_pt" :    Hist(samp, cut, ele_pt_zoom_fine,        storage=hist.storage.Weight()),
        "reco_ele_lpt_leading_eta" :      Hist(samp, cut, ele_eta,       storage=hist.storage.Weight()),
        "reco_ele_lpt_subleading_eta" :   Hist(samp, cut, ele_eta,       storage=hist.storage.Weight()),
        "reco_ele_lpt_leading_dxy" :      Hist(samp, cut, ele_dxy,       storage=hist.storage.Weight()),
        "reco_ele_lpt_subleading_dxy" :   Hist(samp, cut, ele_dxy,       storage=hist.storage.Weight()),
        "reco_ele_lpt_leading_dz" :       Hist(samp, cut, ele_dz,        storage=hist.storage.Weight()),
        "reco_ele_lpt_subleading_dz" :    Hist(samp, cut, ele_dz,        storage=hist.storage.Weight()),
        "reco_ele_lpt_leading_vxy" :      Hist(samp, cut, vxy_coarse,    storage=hist.storage.Weight()),
        "reco_ele_lpt_subleading_vxy" :   Hist(samp, cut, vxy_coarse,    storage=hist.storage.Weight()),
        "reco_ele_lpt_leading_vz" :       Hist(samp, cut, vz_coarse,     storage=hist.storage.Weight()),
        "reco_ele_lpt_subleading_vz" :    Hist(samp, cut, vz_coarse,     storage=hist.storage.Weight()),
        "reco_ele_ged_leading_pt" :       Hist(samp, cut, ele_pt_zoom_fine,        storage=hist.storage.Weight()),
        "reco_ele_ged_subleading_pt" :    Hist(samp, cut, ele_pt_zoom_fine,        storage=hist.storage.Weight()),
        "reco_ele_ged_leading_eta" :      Hist(samp, cut, ele_eta,       storage=hist.storage.Weight()),
        "reco_ele_ged_subleading_eta" :   Hist(samp, cut, ele_eta,       storage=hist.storage.Weight()),
        "reco_ele_ged_leading_dxy" :      Hist(samp, cut, ele_dxy,       storage=hist.storage.Weight()),
        "reco_ele_ged_subleading_dxy" :   Hist(samp, cut, ele_dxy,       storage=hist.storage.Weight()),
        "reco_ele_ged_leading_dz" :       Hist(samp, cut, ele_dz,        storage=hist.storage.Weight()),
        "reco_ele_ged_subleading_dz" :    Hist(samp, cut, ele_dz,        storage=hist.storage.Weight()),
        "reco_ele_ged_leading_vxy" :      Hist(samp, cut, vxy_coarse,    storage=hist.storage.Weight()),
        "reco_ele_ged_subleading_vxy" :   Hist(samp, cut, vxy_coarse,    storage=hist.storage.Weight()),
        "reco_ele_ged_leading_vz" :       Hist(samp, cut, vz_coarse,     storage=hist.storage.Weight()),
        "reco_ele_ged_subleading_vz" :    Hist(samp, cut, vz_coarse,     storage=hist.storage.Weight()),
    }
    #print('init', histograms["gen_leading_ele_pt"].to_numpy())
    return histograms

subroutines = []
    
#['vtx', 'nDSAMuon', 'recoDSAMuonDzError', 'numPFJetTrueB', 'genPU', 'signalReconstructed', 'fixedGridRhoFastjetAll', 'recoDSAMuonTrkProb', 'recoDSAMuonDz', 'recoDSAMuonCharge', 'trigFired','recoDSAMuonPy', 'genEE', 'CaloMET', 'GenPart', 'recoDSAMuonOuterEta', 'trig', 'recoDSAMuonDisplacedId', 'recoDSAMuonTrkNumTrackerHits', 'recoDSAMuonTrkNumHits', 'recoDSAMuonTrkNumDTHits', 'recoDSAMuonE', 'GenPos', 'GenMET', 'rho', 'PV', 'numPV', 'recoDSAMuonTrkNumStripHits', 'recoDSAMuonPhiErr', 'recoDSAMuonTrkNumPlanes', 'ootPhoton', 'GenJet', 'recoDSAMuonEta', 'recoDSAMuonVz', 'PFJet', 'Electron', 'runNum', 'recoDSAMuonOuterPhi', 'METFiltersFailBits', 'nPFJetAll', 'numPFJetTaggedB', 'HEM', 'genWgt', 'recoDSAMuonPz', 'recoDSAMuonVxy', 'recoDSAMuonTrkChi2', 'recoDSAMuonPtErr', 'recoDSAMuonPt', 'GenEle', 'recoDSAMuonTrkNumPixHits', 'recoDSAMuonDxyError', 'Muon', 'recoDSAMuonTrkNumCSCHits', 'LptElectron', 'recoDSAMuonPhi', 'eventNum', 'recoDSAMuonDxy', 'recoDSAMuonIdx', 'recoDSAMuonEtaErr', 'Photon', 'Conversion', 'PFMET', 'recoDSAMuonPx', 'lumiSec', 'eventWgt', 'hasHEMjet', 'hasHEMjetBug', 'hasHEMelecPF', 'hasHEMelecLpt', 'GenJetMETdPhi', 'nJets', 'good_vtx', 'nGoodVtx', 'sel_vtx']
#Electron ['pt', 'eta', 'etaErr', 'phi', 'phiErr', 'IDcutVeto', 'IDcutLoose', 'IDcutMed', 'IDcutTight', 'IDcutVetoInt', 'IDcutLooseInt', 'IDcutMedInt', 'IDcutTightInt', 'IDmvaIso90', 'IDmvaIso80', 'IDmvaIsoLoose', 'IDmva90', 'IDmva80', 'IDmvaLoose', 'angRes', 'e', 'vxy', 'vz', 'dxy', 'dxyErr', 'dz', 'dzErr', 'trkChi2', 'trkIso', 'trkRelIso', 'calIso', 'calRelIso', 'PFIso', 'PFRelIso', 'miniIso', 'miniRelIso', 'PFIsoEleCorr', 'PFRelIsoEleCorr', 'miniIsoEleCorr', 'miniRelIsoEleCorr', 'chadIso', 'nhadIso', 'phoIso', 'rhoEA', 'trkProb', 'numTrackerHits', 'numPixHits', 'numStripHits', 'charge', 'isPF', 'genMatched', 'matchType', 'dRJets', 'dPhiJets', 'full55sigmaIetaIeta', 'absdEtaSeed', 'absdPhiIn', 'HoverE', 'abs1overEm1overP', 'expMissingInnerHits', 'conversionVeto', 'isEE', 'hasLptMatch', 'lptMatchIdx', 'mindRj', 'mindPhiJ', 'IDscore', 'passID', 'passIDBasic']
#sel_vtx ['typ', 'vxy', 'sigmavxy', 'vx', 'vy', 'vz', 'reduced_chi2', 'prob', 'dR', 'sign', 'minDxy', 'METdPhi', 'pt', 'eta', 'phi', 'energy', 'm', 'px', 'py', 'pz', 'refit_m', 'refit_pt', 'refit_eta', 'refit_phi', 'refit_dR', 'isMatched', 'matchSign', 'dRJets', 'dPhiJets', 'e1_typ', 'e1_idx', 'e1_isMatched', 'e1_matchType', 'e1_refit_dxy', 'e1_refit_dxyErr', 'e1_refit_dz', 'e1_refit_dzErr', 'e1_refit_chi2', 'e2_typ', 'e2_idx', 'e2_isMatched', 'e2_matchType', 'e2_refit_dxy', 'e2_refit_dxyErr', 'e2_refit_dz', 'e2_refit_dzErr', 'e2_refit_chi2', 'mindRj', 'mindPhiJ', 'corrMinDxy', 'cos_collinear', 'projectedLxy', 'cos_collinear_fromPV', 'cos_collinear_fromPV_refit', 'e1', 'e2', 'min_dxy', 'eleDphi', 'vxy_fromPV', 'gen_cos_collinear_fromPV', 'isGood']
#LptElectron ['pt', 'eta', 'etaErr', 'phi', 'phiErr', 'ID', 'angRes', 'e', 'vxy', 'vz', 'dxy', 'dxyErr', 'dz', 'dzErr', 'trkChi2', 'trkIso', 'trkRelIso', 'calIso', 'calRelIso', 'PFIso', 'PFRelIso', 'miniIso', 'miniRelIso', 'PFIsoEleCorr', 'PFRelIsoEleCorr', 'miniIsoEleCorr', 'miniRelIsoEleCorr', 'chadIso', 'nhadIso', 'phoIso', 'rhoEA', 'trkProb', 'numTrackerHits', 'numPixHits', 'numStripHits', 'charge', 'minDRtoReg', 'isPF', 'genMatched', 'matchType', 'dRJets', 'dPhiJets', 'full55sigmaIetaIeta', 'absdEtaSeed', 'absdPhiIn', 'HoverE', 'abs1overEm1overP', 'expMissingInnerHits', 'conversionVeto', 'isEE', 'xCleaned', 'gedIdx', 'gedIsMatched', 'mindRj', 'mindPhiJ', 'IDscore', 'passID', 'passIDBasic']

def fillHistos(events, hists, samp, cut, info, sum_wgt=1):
    #wgt = events.eventWgt/sum_wgt
    wgt = events.eventWgt/np.sum(events.eventWgt)
    #print(sum_wgt, np.sum(events.eventWgt), np.sum(wgt))
    
    eles_ged = events.Electron
    sortbypt = ak.argsort(eles_ged.pt, ascending=False)
    eles_ged = eles_ged[sortbypt]
    eles_lpt = events.LptElectron
    sortbypt = ak.argsort(eles_lpt.pt, ascending=False)
    eles_lpt = eles_lpt[sortbypt]
    jets = events.PFJet
    sortbypt = ak.argsort(jets.pt, ascending=False)
    jets = jets[sortbypt]
    photons = events.Photon
    #sortbypt = ak.argsort(photons.pt, ascending=False)
    #photons = photons[sortbypt]

    cut_1_lptele = ak.num(eles_lpt) >= 1
    cut_2_lptele = ak.num(eles_lpt) >= 2
    cut_1_gedele = ak.num(eles_ged) >= 1
    cut_2_gedele = ak.num(eles_ged) >= 2
    
    n_total = ak.num(eles_ged) + ak.num(eles_lpt)
    n_pairs = n_total * (n_total - 1) // 2   # C(N,2) = N*(N-1)/2
    hists["reco_N_pairs"                 ].fill(samp = samp, cut = cut, nvtxs = n_pairs, weight = wgt)
    hists["reco_N_vtx"                   ].fill(samp = samp, cut = cut, nvtxs = ak.num(events.vtx), weight = wgt)
    hists["reco_N_good_vtx"              ].fill(samp = samp, cut = cut, nvtxs = events.nGoodVtx, weight = wgt)
    hists["reco_N_jets"                  ].fill(samp = samp, cut = cut, njets = ak.num(jets), weight = wgt)
    hists["reco_N_photons"               ].fill(samp = samp, cut = cut, nphos = ak.num(jets), weight = wgt)
    hists["reco_N_GED_eles"              ].fill(samp = samp, cut = cut, neles = ak.num(eles_ged), weight = wgt)
    hists["reco_N_Lpt_eles"              ].fill(samp = samp, cut = cut, neles = ak.num(eles_lpt), weight = wgt)
    
    hists["reco_ele_lpt_leading_pt"      ].fill(samp = samp, cut = cut, pt = eles_lpt[cut_1_lptele].pt[:, 0], weight = wgt[cut_1_lptele])
    hists["reco_ele_lpt_subleading_pt"   ].fill(samp = samp, cut = cut, pt = eles_lpt[cut_2_lptele].pt[:, 1], weight = wgt[cut_2_lptele])
    hists["reco_ele_lpt_leading_eta"     ].fill(samp = samp, cut = cut, eta = eles_lpt[cut_1_lptele].eta[:, 0], weight = wgt[cut_1_lptele])
    hists["reco_ele_lpt_subleading_eta"  ].fill(samp = samp, cut = cut, eta = eles_lpt[cut_2_lptele].eta[:, 1], weight = wgt[cut_2_lptele])
    hists["reco_ele_lpt_leading_dxy"     ].fill(samp = samp, cut = cut, dxy = eles_lpt[cut_1_lptele].dxy[:, 0], weight = wgt[cut_1_lptele])
    hists["reco_ele_lpt_subleading_dxy"  ].fill(samp = samp, cut = cut, dxy = eles_lpt[cut_2_lptele].dxy[:, 1], weight = wgt[cut_2_lptele])
    hists["reco_ele_lpt_leading_dz"      ].fill(samp = samp, cut = cut, dz = eles_lpt[cut_1_lptele].dz[:, 0], weight = wgt[cut_1_lptele])
    hists["reco_ele_lpt_subleading_dz"   ].fill(samp = samp, cut = cut, dz = eles_lpt[cut_2_lptele].dz[:, 1], weight = wgt[cut_2_lptele])
    hists["reco_ele_lpt_leading_vxy"     ].fill(samp = samp, cut = cut, vxy = eles_lpt[cut_1_lptele].vxy[:, 0], weight = wgt[cut_1_lptele])
    hists["reco_ele_lpt_subleading_vxy"  ].fill(samp = samp, cut = cut, vxy = eles_lpt[cut_2_lptele].vxy[:, 1], weight = wgt[cut_2_lptele])
    hists["reco_ele_lpt_leading_vz"      ].fill(samp = samp, cut = cut, vz = eles_lpt[cut_1_lptele].vz[:, 0], weight = wgt[cut_1_lptele])
    hists["reco_ele_lpt_subleading_vz"   ].fill(samp = samp, cut = cut, vz = eles_lpt[cut_2_lptele].vz[:, 1], weight = wgt[cut_2_lptele])

    hists["reco_ele_ged_leading_pt"      ].fill(samp = samp, cut = cut, pt = eles_ged[cut_1_gedele].pt[:, 0], weight = wgt[cut_1_gedele])
    hists["reco_ele_ged_subleading_pt"   ].fill(samp = samp, cut = cut, pt = eles_ged[cut_2_gedele].pt[:, 1], weight = wgt[cut_2_gedele])
    hists["reco_ele_ged_leading_eta"     ].fill(samp = samp, cut = cut, eta = eles_ged[cut_1_gedele].eta[:, 0], weight = wgt[cut_1_gedele])
    hists["reco_ele_ged_subleading_eta"  ].fill(samp = samp, cut = cut, eta = eles_ged[cut_2_gedele].eta[:, 1], weight = wgt[cut_2_gedele])
    hists["reco_ele_ged_leading_dxy"     ].fill(samp = samp, cut = cut, dxy = eles_ged[cut_1_gedele].dxy[:, 0], weight = wgt[cut_1_gedele])
    hists["reco_ele_ged_subleading_dxy"  ].fill(samp = samp, cut = cut, dxy = eles_ged[cut_2_gedele].dxy[:, 1], weight = wgt[cut_2_gedele])
    hists["reco_ele_ged_leading_dz"      ].fill(samp = samp, cut = cut, dz = eles_ged[cut_1_gedele].dz[:, 0], weight = wgt[cut_1_gedele])
    hists["reco_ele_ged_subleading_dz"   ].fill(samp = samp, cut = cut, dz = eles_ged[cut_2_gedele].dz[:, 1], weight = wgt[cut_2_gedele])
    hists["reco_ele_ged_leading_vxy"     ].fill(samp = samp, cut = cut, vxy = eles_ged[cut_1_gedele].vxy[:, 0], weight = wgt[cut_1_gedele])
    hists["reco_ele_ged_subleading_vxy"  ].fill(samp = samp, cut = cut, vxy = eles_ged[cut_2_gedele].vxy[:, 1], weight = wgt[cut_2_gedele])
    hists["reco_ele_ged_leading_vz"      ].fill(samp = samp, cut = cut, vz = eles_ged[cut_1_gedele].vz[:, 0], weight = wgt[cut_1_gedele])
    hists["reco_ele_ged_subleading_vz"   ].fill(samp = samp, cut = cut, vz = eles_ged[cut_2_gedele].vz[:, 1], weight = wgt[cut_2_gedele])

        
        
