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
        # per-electron quantities: both electrons in each vertex filled into unified histos
        "vtx_ele_ID":             Hist(samp, cut, ele_id,             storage=hist.storage.Weight()),
        "vtx_ele_miniIso":        Hist(samp, cut, ele_miniIso,        storage=hist.storage.Weight()),
        "vtx_ele_miniRelIso":     Hist(samp, cut, ele_miniRelIso,     storage=hist.storage.Weight()),
        "vtx_ele_miniIsoCorr":    Hist(samp, cut, ele_miniIsoCorr,    storage=hist.storage.Weight()),
        "vtx_ele_miniRelIsoCorr": Hist(samp, cut, ele_miniRelIsoCorr, storage=hist.storage.Weight()),
        "vtx_ele_track_chi2":     Hist(samp, cut, vtx_chi2,           storage=hist.storage.Weight()),
        "vtx_ele_refit_chi2":     Hist(samp, cut, vtx_chi2,           storage=hist.storage.Weight()),
        "vtx_ele_dxy":            Hist(samp, cut, dxy_fine,           storage=hist.storage.Weight()),
        "vtx_ele_refit_dxy":      Hist(samp, cut, dxy_fine,           storage=hist.storage.Weight()),
        "vtx_ele_dxydz":          Hist(samp, cut, ele_dxydz,          storage=hist.storage.Weight()),
        "vtx_ele_logdxydz":       Hist(samp, cut, ele_logdxydz,       storage=hist.storage.Weight()),
        # per-electron leading/subleading by pT
        "vtx_leading_ele_pt":        Hist(samp, cut, ele_pt,  storage=hist.storage.Weight()),
        "vtx_subleading_ele_pt":     Hist(samp, cut, ele_pt,  storage=hist.storage.Weight()),
        "vtx_leading_ele_eta":       Hist(samp, cut, ele_eta, storage=hist.storage.Weight()),
        "vtx_subleading_ele_eta":    Hist(samp, cut, ele_eta, storage=hist.storage.Weight()),
        "vtx_leading_ele_mindRj":    Hist(samp, cut, dRj,     storage=hist.storage.Weight()),
        "vtx_subleading_ele_mindRj": Hist(samp, cut, dRj,     storage=hist.storage.Weight()),
        # per-vertex aggregates
        "vtx_maxMiniIso":         Hist(samp, cut, ele_miniRelIsoCorr, storage=hist.storage.Weight()),
        "vtx_min_dxy":            Hist(samp, cut, dxy_fine,           storage=hist.storage.Weight()),
        "vtx_min_refit_dxy":      Hist(samp, cut, dxy_fine,           storage=hist.storage.Weight()),
        "vtx_min_logdxydz":       Hist(samp, cut, ele_logdxydz,       storage=hist.storage.Weight()),
        # vertex-level quantities
        "vtx_reduced_chi2":       Hist(samp, cut, vtx_chi2,           storage=hist.storage.Weight()),
        "vtx_mass":               Hist(samp, cut, vtx_mass,           storage=hist.storage.Weight()),
        "vtx_refit_mass":         Hist(samp, cut, vtx_mass,           storage=hist.storage.Weight()),
        "vtx_dr":                 Hist(samp, cut, ee_dr,              storage=hist.storage.Weight()),
        "vtx_refit_dr":           Hist(samp, cut, ee_dr,              storage=hist.storage.Weight()),
        "vtx_pt":                 Hist(samp, cut, vtx_pt,             storage=hist.storage.Weight()),
        "vtx_refit_pt":           Hist(samp, cut, vtx_pt,             storage=hist.storage.Weight()),
        "vtx_eta":                Hist(samp, cut, ele_eta,            storage=hist.storage.Weight()),
        "vtx_refit_eta":          Hist(samp, cut, ele_eta,            storage=hist.storage.Weight()),
        "vtx_phi":                Hist(samp, cut, ele_phi,            storage=hist.storage.Weight()),
        "vtx_refit_phi":          Hist(samp, cut, ele_phi,            storage=hist.storage.Weight()),
    }
    return histograms

subroutines = []

#['vtx', 'nDSAMuon', 'recoDSAMuonDzError', 'numPFJetTrueB', 'genPU', 'signalReconstructed', 'fixedGridRhoFastjetAll', 'recoDSAMuonTrkProb', 'recoDSAMuonDz', 'recoDSAMuonCharge', 'trigFired','recoDSAMuonPy', 'genEE', 'CaloMET', 'GenPart', 'recoDSAMuonOuterEta', 'trig', 'recoDSAMuonDisplacedId', 'recoDSAMuonTrkNumTrackerHits', 'recoDSAMuonTrkNumHits', 'recoDSAMuonTrkNumDTHits', 'recoDSAMuonE', 'GenPos', 'GenMET', 'rho', 'PV', 'numPV', 'recoDSAMuonTrkNumStripHits', 'recoDSAMuonPhiErr', 'recoDSAMuonTrkNumPlanes', 'ootPhoton', 'GenJet', 'recoDSAMuonEta', 'recoDSAMuonVz', 'PFJet', 'Electron', 'runNum', 'recoDSAMuonOuterPhi', 'METFiltersFailBits', 'nPFJetAll', 'numPFJetTaggedB', 'HEM', 'genWgt', 'recoDSAMuonPz', 'recoDSAMuonVxy', 'recoDSAMuonTrkChi2', 'recoDSAMuonPtErr', 'recoDSAMuonPt', 'GenEle', 'recoDSAMuonTrkNumPixHits', 'recoDSAMuonDxyError', 'Muon', 'recoDSAMuonTrkNumCSCHits', 'LptElectron', 'recoDSAMuonPhi', 'eventNum', 'recoDSAMuonDxy', 'recoDSAMuonIdx', 'recoDSAMuonEtaErr', 'Photon', 'Conversion', 'PFMET', 'recoDSAMuonPx', 'lumiSec', 'eventWgt', 'hasHEMjet', 'hasHEMjetBug', 'hasHEMelecPF', 'hasHEMelecLpt', 'GenJetMETdPhi', 'nJets', 'good_vtx', 'nGoodVtx', 'sel_vtx']
# vtx ['typ', 'vxy', 'sigmavxy', 'vx', 'vy', 'vz', 'reduced_chi2', 'prob', 'dR', 'sign', 'minDxy', 'METdPhi', 'pt', 'eta', 'phi', 'energy', 'm', 'px', 'py', 'pz', 'refit_m', 'refit_pt', 'refit_eta', 'refit_phi', 'refit_dR', 'isMatched', 'matchSign', 'dRJets', 'dPhiJets', 'e1_typ', 'e1_idx', 'e1_isMatched', 'e1_matchType', 'e1_refit_dxy', 'e1_refit_dxyErr', 'e1_refit_dz', 'e1_refit_dzErr', 'e1_refit_chi2', 'e2_typ', 'e2_idx', 'e2_isMatched', 'e2_matchType', 'e2_refit_dxy', 'e2_refit_dxyErr', 'e2_refit_dz', 'e2_refit_dzErr', 'e2_refit_chi2', 'mindRj', 'mindPhiJ', 'corrMinDxy', 'cos_collinear', 'projectedLxy', 'cos_collinear_fromPV', 'cos_collinear_fromPV_refit', 'e1', 'e2', 'min_dxy', 'eleDphi', 'vxy_fromPV', 'gen_cos_collinear_fromPV', 'isGood']
# e1/2 ['pt', 'eta', 'etaErr', 'phi', 'phiErr', 'angRes', 'e', 'vxy', 'vz', 'dxy', 'dxyErr', 'dz', 'dzErr', 'trkChi2', 'trkIso', 'trkRelIso', 'calIso', 'calRelIso', 'PFIso', 'PFRelIso', 'miniIso', 'miniRelIso', 'PFIsoEleCorr', 'PFRelIsoEleCorr', 'miniIsoEleCorr', 'miniRelIsoEleCorr', 'chadIso', 'nhadIso', 'phoIso', 'rhoEA', 'trkProb', 'numTrackerHits', 'numPixHits', 'numStripHits', 'charge', 'isPF', 'genMatched', 'matchType', 'dRJets', 'dPhiJets', 'full55sigmaIetaIeta', 'absdEtaSeed', 'absdPhiIn', 'HoverE', 'abs1overEm1overP', 'expMissingInnerHits', 'conversionVeto', 'isEE', 'mindRj', 'mindPhiJ', 'IDscore', 'passID', 'passIDBasic', 'refit_dxy']

def fillHistos(events, hists, samp, cut, info, sum_wgt=1):

    wgt = events.eventWgt / sum_wgt

    # broadcast per-event weight to per-vertex, then flatten alongside vtx
    wgt_vtx = ak.flatten(ak.broadcast_arrays(wgt, events.vtx.reduced_chi2)[0])
    vtx = ak.flatten(events.vtx)
    e1  = vtx.e1
    e2  = vtx.e2

    # fill both electrons into each unified per-electron histo
    for (e, e_refit_dxy, e_refit_chi2) in [
        (e1, vtx.e1_refit_dxy, vtx.e1_refit_chi2),
        (e2, vtx.e2_refit_dxy, vtx.e2_refit_chi2),
    ]:
        e_dxydz    = np.abs(e.dxy) / (np.abs(e.dz) + 1e-9)
        e_logdxydz = np.log(e_dxydz + 1e-9)

        hists["vtx_ele_ID"            ].fill(samp=samp, cut=cut, ele_id=e.IDscore,                    weight=wgt_vtx)
        hists["vtx_ele_miniIso"       ].fill(samp=samp, cut=cut, iso=e.miniIso,                       weight=wgt_vtx)
        hists["vtx_ele_miniRelIso"    ].fill(samp=samp, cut=cut, iso=e.miniRelIso,                    weight=wgt_vtx)
        hists["vtx_ele_miniIsoCorr"   ].fill(samp=samp, cut=cut, iso=e.miniIsoEleCorr,                weight=wgt_vtx)
        hists["vtx_ele_miniRelIsoCorr"].fill(samp=samp, cut=cut, iso=e.miniRelIsoEleCorr,             weight=wgt_vtx)
        hists["vtx_ele_track_chi2"    ].fill(samp=samp, cut=cut, chi2=e.trkChi2,                      weight=wgt_vtx)
        hists["vtx_ele_refit_chi2"    ].fill(samp=samp, cut=cut, chi2=e_refit_chi2,                   weight=wgt_vtx)
        hists["vtx_ele_dxy"           ].fill(samp=samp, cut=cut, dxy=np.abs(e.dxy),                   weight=wgt_vtx)
        hists["vtx_ele_refit_dxy"     ].fill(samp=samp, cut=cut, dxy=np.abs(e_refit_dxy),             weight=wgt_vtx)
        hists["vtx_ele_dxydz"         ].fill(samp=samp, cut=cut, dxydz=e_dxydz,                      weight=wgt_vtx)
        hists["vtx_ele_logdxydz"      ].fill(samp=samp, cut=cut, logdxydz=e_logdxydz,                weight=wgt_vtx)

    # --- per-vertex aggregates ---
    e1_dxydz    = np.abs(e1.dxy) / (np.abs(e1.dz) + 1e-9)
    e2_dxydz    = np.abs(e2.dxy) / (np.abs(e2.dz) + 1e-9)
    e1_logdxydz = np.log(e1_dxydz + 1e-9)
    e2_logdxydz = np.log(e2_dxydz + 1e-9)

    hists["vtx_maxMiniIso"   ].fill(samp=samp, cut=cut, iso=np.maximum(e1.miniRelIsoEleCorr, e2.miniRelIsoEleCorr),          weight=wgt_vtx)
    hists["vtx_min_dxy"      ].fill(samp=samp, cut=cut, dxy=np.minimum(np.abs(vtx.e1.dxy), np.abs(vtx.e2.dxy)), weight=wgt_vtx)
    hists["vtx_min_refit_dxy"].fill(samp=samp, cut=cut, dxy=np.minimum(np.abs(vtx.e1_refit_dxy), np.abs(vtx.e2_refit_dxy)), weight=wgt_vtx)
    hists["vtx_min_logdxydz" ].fill(samp=samp, cut=cut, logdxydz=np.minimum(e1_logdxydz, e2_logdxydz),                     weight=wgt_vtx)

    # --- vertex-level ---
    hists["vtx_reduced_chi2"].fill(samp=samp, cut=cut, chi2=vtx.reduced_chi2, weight=wgt_vtx)
    hists["vtx_mass"        ].fill(samp=samp, cut=cut, mass=vtx.m,            weight=wgt_vtx)
    hists["vtx_refit_mass"  ].fill(samp=samp, cut=cut, mass=vtx.refit_m,      weight=wgt_vtx)
    hists["vtx_dr"          ].fill(samp=samp, cut=cut, dr=vtx.dR,             weight=wgt_vtx)
    hists["vtx_refit_dr"    ].fill(samp=samp, cut=cut, dr=vtx.refit_dR,       weight=wgt_vtx)
    hists["vtx_pt"          ].fill(samp=samp, cut=cut, pt=vtx.pt,             weight=wgt_vtx)
    hists["vtx_refit_pt"    ].fill(samp=samp, cut=cut, pt=vtx.refit_pt,       weight=wgt_vtx)
    hists["vtx_eta"         ].fill(samp=samp, cut=cut, eta=vtx.eta,           weight=wgt_vtx)
    hists["vtx_refit_eta"   ].fill(samp=samp, cut=cut, eta=vtx.refit_eta,     weight=wgt_vtx)
    hists["vtx_phi"         ].fill(samp=samp, cut=cut, phi=vtx.phi,           weight=wgt_vtx)
    hists["vtx_refit_phi"   ].fill(samp=samp, cut=cut, phi=vtx.refit_phi,     weight=wgt_vtx)

    # --- leading/subleading by pT ---
    e1_is_lead = e1.pt >= e2.pt
    def lead(a, b):    return ak.where(e1_is_lead, a, b)
    def sublead(a, b): return ak.where(e1_is_lead, b, a)

    hists["vtx_leading_ele_pt"       ].fill(samp=samp, cut=cut, pt=lead(e1.pt, e2.pt),                   weight=wgt_vtx)
    hists["vtx_subleading_ele_pt"    ].fill(samp=samp, cut=cut, pt=sublead(e1.pt, e2.pt),                weight=wgt_vtx)
    hists["vtx_leading_ele_eta"      ].fill(samp=samp, cut=cut, eta=lead(e1.eta, e2.eta),                weight=wgt_vtx)
    hists["vtx_subleading_ele_eta"   ].fill(samp=samp, cut=cut, eta=sublead(e1.eta, e2.eta),             weight=wgt_vtx)
    hists["vtx_leading_ele_mindRj"   ].fill(samp=samp, cut=cut, drj=lead(e1.mindRj, e2.mindRj),         weight=wgt_vtx)
    hists["vtx_subleading_ele_mindRj"].fill(samp=samp, cut=cut, drj=sublead(e1.mindRj, e2.mindRj),      weight=wgt_vtx)
