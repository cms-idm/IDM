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
        "res_pt_lpt":              Hist(samp, cut, res_pt,                   storage=hist.storage.Weight()),
        "res_pt_ged":              Hist(samp, cut, res_pt,                   storage=hist.storage.Weight()),
        "res_e_lpt":               Hist(samp, cut, res_e,                    storage=hist.storage.Weight()),
        "res_e_ged":               Hist(samp, cut, res_e,                    storage=hist.storage.Weight()),
        "res_pt_vs_genpt_lpt":     Hist(samp, cut, ele_pt_special,  res_pt, storage=hist.storage.Weight()),
        "res_pt_vs_genpt_ged":     Hist(samp, cut, ele_pt_special,  res_pt, storage=hist.storage.Weight()),
        "res_e_vs_genpt_lpt":      Hist(samp, cut, ele_pt_special,  res_e,  storage=hist.storage.Weight()),
        "res_e_vs_genpt_ged":      Hist(samp, cut, ele_pt_special,  res_e,  storage=hist.storage.Weight()),
        "res_pt_vs_genlxy_lpt":    Hist(samp, cut, ele_lxy_special, res_pt, storage=hist.storage.Weight()),
        "res_pt_vs_genlxy_ged":    Hist(samp, cut, ele_lxy_special, res_pt, storage=hist.storage.Weight()),
        "res_e_vs_genlxy_lpt":     Hist(samp, cut, ele_lxy_special, res_e,  storage=hist.storage.Weight()),
        "res_e_vs_genlxy_ged":     Hist(samp, cut, ele_lxy_special, res_e,  storage=hist.storage.Weight()),
        "res_pt_vs_geneta_lpt":    Hist(samp, cut, ele_eta,         res_pt, storage=hist.storage.Weight()),
        "res_pt_vs_geneta_ged":    Hist(samp, cut, ele_eta,         res_pt, storage=hist.storage.Weight()),
        "res_e_vs_geneta_lpt":     Hist(samp, cut, ele_eta,         res_e,  storage=hist.storage.Weight()),
        "res_e_vs_geneta_ged":     Hist(samp, cut, ele_eta,         res_e,  storage=hist.storage.Weight()),
    }
    return histograms

subroutines = []

# GenEle/Pos ['charge', 'motherID', 'pt', 'eta', 'phi', 'energy', 'px', 'py', 'pz', 'vxy', 'vz', 'vx', 'vy', 'matched', 'matchType', 'matchIdxLocal', 'matchIdxGlobal', 'dr', 'dRbin', 'vxyBin', 'ptBin', 'matchPassID', 'matchPassIDBasic']
# Electron ['pt', 'eta', 'etaErr', 'phi', 'phiErr', 'IDcutVeto', 'IDcutLoose', 'IDcutMed', 'IDcutTight', 'IDcutVetoInt', 'IDcutLooseInt', 'IDcutMedInt', 'IDcutTightInt', 'IDmvaIso90', 'IDmvaIso80', 'IDmvaIsoLoose', 'IDmva90', 'IDmva80', 'IDmvaLoose', 'angRes', 'e', 'vxy', 'vz', 'dxy', 'dxyErr', 'dz', 'dzErr', 'trkChi2', 'trkIso', 'trkRelIso', 'calIso', 'calRelIso', 'PFIso', 'PFRelIso', 'miniIso', 'miniRelIso', 'PFIsoEleCorr', 'PFRelIsoEleCorr', 'miniIsoEleCorr', 'miniRelIsoEleCorr', 'chadIso', 'nhadIso', 'phoIso', 'rhoEA', 'trkProb', 'numTrackerHits', 'numPixHits', 'numStripHits', 'charge', 'isPF', 'genMatched', 'matchType', 'dRJets', 'dPhiJets', 'full55sigmaIetaIeta', 'absdEtaSeed', 'absdPhiIn', 'HoverE', 'abs1overEm1overP', 'expMissingInnerHits', 'conversionVeto', 'isEE', 'hasLptMatch', 'lptMatchIdx', 'mindRj', 'mindPhiJ', 'IDscore', 'passID', 'passIDBasic']
# LptElectron ['pt', 'eta', 'etaErr', 'phi', 'phiErr', 'ID', 'angRes', 'e', 'vxy', 'vz', 'dxy', 'dxyErr', 'dz', 'dzErr', 'trkChi2', 'trkIso', 'trkRelIso', 'calIso', 'calRelIso', 'PFIso', 'PFRelIso', 'miniIso', 'miniRelIso', 'PFIsoEleCorr', 'PFRelIsoEleCorr', 'miniIsoEleCorr', 'miniRelIsoEleCorr', 'chadIso', 'nhadIso', 'phoIso', 'rhoEA', 'trkProb', 'numTrackerHits', 'numPixHits', 'numStripHits', 'charge', 'minDRtoReg', 'isPF', 'genMatched', 'matchType', 'dRJets', 'dPhiJets', 'full55sigmaIetaIeta', 'absdEtaSeed', 'absdPhiIn', 'HoverE', 'abs1overEm1overP', 'expMissingInnerHits', 'conversionVeto', 'isEE', 'xCleaned', 'gedIdx', 'gedIsMatched', 'mindRj', 'mindPhiJ', 'IDscore', 'passID', 'passIDBasic']

def fillHistos(events, hists, samp, cut, info, sum_wgt=1):
    wgt = events.eventWgt

    lxy_map = {
        'ele': np.sqrt((events.GenEle.vx - events.PV.x)**2 + (events.GenEle.vy - events.PV.y)**2),
        'pos': np.sqrt((events.GenPos.vx - events.PV.x)**2 + (events.GenPos.vy - events.PV.y)**2),
    }

    for gen_key, gen_coll in [('ele', events.GenEle), ('pos', events.GenPos)]:
        gen_lxy_all = lxy_map[gen_key]
        for mtype, suffix, reco_coll in [('L', 'lpt', events.LptElectron), ('R', 'ged', events.Electron)]:
            mask = gen_coll.matchType == mtype
            if ak.sum(mask) == 0:
                continue
            gen  = gen_coll[mask]
            reco = reco_coll[mask]
            w    = wgt[mask]
            lxy  = gen_lxy_all[mask]
            idx   = gen.matchIdxLocal
            local = ak.local_index(reco.pt, axis=1)
            # select the one matched reco electron per event via matchIdxLocal
            sel_pt = reco.pt[local == idx]
            sel_e  = reco.e[local == idx]
            valid  = ak.num(sel_pt) > 0
            reco_pt  = ak.flatten(sel_pt[valid])
            reco_e   = ak.flatten(sel_e[valid])
            gen_pt   = gen.pt[valid]
            gen_e    = gen.energy[valid]
            gen_eta  = gen.eta[valid]
            gen_lxy  = lxy[valid]
            w_v      = w[valid]
            res_pt   = (reco_pt - gen_pt) / gen_pt
            res_e    = (reco_e  - gen_e)  / gen_e

            hists[f"res_pt_{suffix}"            ].fill(samp=samp, cut=cut, res=res_pt,                weight=w_v)
            hists[f"res_e_{suffix}"             ].fill(samp=samp, cut=cut, res=res_e,                 weight=w_v)
            hists[f"res_pt_vs_genpt_{suffix}"   ].fill(samp=samp, cut=cut, pt=gen_pt,  res=res_pt,    weight=w_v)
            hists[f"res_e_vs_genpt_{suffix}"    ].fill(samp=samp, cut=cut, pt=gen_pt,  res=res_e,     weight=w_v)
            hists[f"res_pt_vs_genlxy_{suffix}"  ].fill(samp=samp, cut=cut, lxy=gen_lxy, res=res_pt,   weight=w_v)
            hists[f"res_e_vs_genlxy_{suffix}"   ].fill(samp=samp, cut=cut, lxy=gen_lxy, res=res_e,    weight=w_v)
            hists[f"res_pt_vs_geneta_{suffix}"  ].fill(samp=samp, cut=cut, eta=gen_eta, res=res_pt,   weight=w_v)
            hists[f"res_e_vs_geneta_{suffix}"   ].fill(samp=samp, cut=cut, eta=gen_eta, res=res_e,    weight=w_v)
