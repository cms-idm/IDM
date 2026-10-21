try:
    from histobins import *
except ModuleNotFoundError:
    from configs.hists.histobins import *
from hist import Hist
import hist
import numpy as np
import awkward as ak
from hist.axis import Variable

# coarse binning for the 4D per-electron reco efficiency hists (no flow bins; overflow is clipped into the top bin at fill time):
# single-electron gen pt and |eta|, gen ee pair lxy and dR
ee_pt_4d     = Variable([0, 5, 10, 20, 100],          name="pt",     label=r"Gen $p_T(e)$ [GeV]",        flow=False)
ee_lxy_4d    = Variable([0, 1, 5, 10, 15, 100],       name="lxy",    label=r"Gen $L_{xy}(e^+e^-)$ [cm]", flow=False)
ee_dr_4d     = Variable([0, 0.01, 0.1, 1, 5],         name="dr",     label=r"Gen $\Delta R(e^+e^-)$",    flow=False)
ee_abseta_4d = Variable([0, 1, 1.5, 2, 2.5, 3],       name="abseta", label=r"Gen $|\eta(e)|$",           flow=False)
ee4d_cats    = ['lpt', 'ged', 'alpt', 'xlpt', 'none', 'sumnoalpt', 'alptnosum']

# log-spaced gen ee dR binning for the 1D per-electron hists (first bin [0, 1e-3] catches the near-collinear pairs)
ee_dr_1d     = Variable(np.concatenate([[0], np.logspace(-3, np.log10(6), 41)]), name="dr", label=r"Gen $\Delta R(e^+e^-)$")

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
        "ele_reco_xlpt_pt_lxy":     Hist(samp, cut, ele_pt_special, ele_lxy_special, storage=hist.storage.Weight()),
        "ele_reco_xlpt_eta":        Hist(samp, cut, ele_eta_fine,  storage=hist.storage.Weight()),
        "ele_reco_xlpt_lz":         Hist(samp, cut, Lz_fine,       storage=hist.storage.Weight()),
        "ele_reco_lpt_dr":          Hist(samp, cut, ee_dr_1d,      storage=hist.storage.Weight()),
        "ele_reco_ged_dr":          Hist(samp, cut, ee_dr_1d,      storage=hist.storage.Weight()),
        "ele_reco_none_dr":         Hist(samp, cut, ee_dr_1d,      storage=hist.storage.Weight()),
        "ele_reco_alllpt_dr":       Hist(samp, cut, ee_dr_1d,      storage=hist.storage.Weight()),
        "ele_reco_noalllpt_dr":     Hist(samp, cut, ee_dr_1d,      storage=hist.storage.Weight()),
        "ele_reco_xlpt_dr":         Hist(samp, cut, ee_dr_1d,      storage=hist.storage.Weight()),
    }
    for c in ee4d_cats:
        histograms[f"ele_reco_{c}_ee4d"] = Hist(samp, cut, ee_pt_4d, ee_lxy_4d, ee_dr_4d, ee_abseta_4d, storage=hist.storage.Weight())
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
    all_ee            = ak.concatenate([events.genEE,                     events.genEE],                    axis=0)

    # xlpt: gen electron matched to an AllLptElectron that was cross-cleaned against a GED electron.
    # matchIdxAllLowPt is a per-event local index into AllLptElectron (-1/-999 when unmatched).
    all_alpt_xc = ak.concatenate([events.AllLptElectron.xCleaned, events.AllLptElectron.xCleaned], axis=0)
    all_idxalllpt = ak.concatenate([events.GenEle.matchIdxAllLowPt, events.GenPos.matchIdxAllLowPt], axis=0)
    match_xc = ak.fill_none(ak.firsts(all_alpt_xc[ak.local_index(all_alpt_xc, axis=1) == all_idxalllpt]), False)
    all_matchedxlpt = all_matchedalllpt & match_xc

    for mtype, prefix in [('L', 'ele_reco_lpt'), ('R', 'ele_reco_ged'), ('None', 'ele_reco_none')]:
        mask = all_gen.matchType == mtype
        ele  = all_gen[mask]
        lxy  = all_lxy[mask]
        lz   = all_lz[mask]
        w    = all_wgt[mask]
        hists[f"{prefix}_pt_lxy"].fill(samp=samp, cut=cut, pt=ele.pt, lxy=lxy, weight=w)
        hists[f"{prefix}_eta"].fill(   samp=samp, cut=cut, eta=ele.eta,        weight=w)
        hists[f"{prefix}_lz" ].fill(   samp=samp, cut=cut, lz=lz,              weight=w)
        hists[f"{prefix}_dr" ].fill(   samp=samp, cut=cut, dr=all_ee.dr[mask], weight=w)

    for matched, prefix in [(True, 'ele_reco_alllpt'), (False, 'ele_reco_noalllpt')]:
        mask = all_matchedalllpt == matched
        ele  = all_gen[mask]
        lxy  = all_lxy[mask]
        lz   = all_lz[mask]
        w    = all_wgt[mask]
        hists[f"{prefix}_pt_lxy"].fill(samp=samp, cut=cut, pt=ele.pt, lxy=lxy, weight=w)
        hists[f"{prefix}_eta"].fill(   samp=samp, cut=cut, eta=ele.eta,        weight=w)
        hists[f"{prefix}_lz" ].fill(   samp=samp, cut=cut, lz=lz,              weight=w)
        hists[f"{prefix}_dr" ].fill(   samp=samp, cut=cut, dr=all_ee.dr[mask], weight=w)

    mask = all_matchedxlpt
    ele  = all_gen[mask]
    w    = all_wgt[mask]
    hists["ele_reco_xlpt_pt_lxy"].fill(samp=samp, cut=cut, pt=ele.pt, lxy=all_lxy[mask], weight=w)
    hists["ele_reco_xlpt_eta"   ].fill(samp=samp, cut=cut, eta=ele.eta,                 weight=w)
    hists["ele_reco_xlpt_lz"    ].fill(samp=samp, cut=cut, lz=all_lz[mask],             weight=w)
    hists["ele_reco_xlpt_dr"    ].fill(samp=samp, cut=cut, dr=all_ee.dr[mask],          weight=w)

    # 4D hists: one entry per gen electron, binned in its own gen pt and |eta| and its pair's gen ee dR and lxy.
    # e and e+ share the decay vertex, so all_lxy is the ee lxy (from the chi2 production vertex).
    # sum = reconstructed as GED or Lpt (matchType R or L)
    all_matchedsum = (all_gen.matchType == 'L') | (all_gen.matchType == 'R')
    ee4d_masks = {
        'lpt':  all_gen.matchType == 'L',
        'ged':  all_gen.matchType == 'R',
        'alpt': all_matchedalllpt,
        'xlpt': all_matchedxlpt,
        'none': all_gen.matchType == 'None',
        'sumnoalpt': all_matchedsum & ~all_matchedalllpt,
        'alptnosum': all_matchedalllpt & ~all_matchedsum,
    }
    # values above the last edge are clipped into the top bin (e.g. the top pt bin is pt > 20)
    top = lambda x, ax: np.minimum(x, ax.edges[-2])
    for c, mask in ee4d_masks.items():
        hists[f"ele_reco_{c}_ee4d"].fill(samp=samp, cut=cut,
                                         pt=top(all_gen.pt[mask], ee_pt_4d), lxy=top(all_lxy[mask], ee_lxy_4d),
                                         dr=top(all_ee.dr[mask], ee_dr_4d), abseta=top(np.abs(all_gen.eta[mask]), ee_abseta_4d),
                                         weight=all_wgt[mask])


        
        
