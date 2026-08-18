try:
    from histobins import *
except ModuleNotFoundError:
    from configs.hists.histobins import *
from hist import Hist
from hist.axis import Variable, Regular, StrCategory
import hist
import numpy as np
import awkward as ak

def make_histograms():
    histograms = {
        "res_pt_lpt":                      Hist(samp, cut, res_pt,                                storage=hist.storage.Weight()),
        "res_pt_ged":                      Hist(samp, cut, res_pt,                                storage=hist.storage.Weight()),
        "res_e_lpt":                       Hist(samp, cut, res_e,                                 storage=hist.storage.Weight()),
        "res_e_ged":                       Hist(samp, cut, res_e,                                 storage=hist.storage.Weight()),
        "res_pt_vs_genpt_lpt":             Hist(samp, cut, ele_pt_res,     res_pt,                storage=hist.storage.Weight()),
        "res_pt_vs_genpt_ged":             Hist(samp, cut, ele_pt_res,     res_pt,                storage=hist.storage.Weight()),
        "res_e_vs_genpt_lpt":              Hist(samp, cut, ele_pt_res,     res_e,                 storage=hist.storage.Weight()),
        "res_e_vs_genpt_ged":              Hist(samp, cut, ele_pt_res,     res_e,                 storage=hist.storage.Weight()),
        "res_pt_vs_genlxy_lpt":            Hist(samp, cut, ele_lxy_res,    res_pt,                storage=hist.storage.Weight()),
        "res_pt_vs_genlxy_ged":            Hist(samp, cut, ele_lxy_res,    res_pt,                storage=hist.storage.Weight()),
        "res_e_vs_genlxy_lpt":             Hist(samp, cut, ele_lxy_res,    res_e,                 storage=hist.storage.Weight()),
        "res_e_vs_genlxy_ged":             Hist(samp, cut, ele_lxy_res,    res_e,                 storage=hist.storage.Weight()),
        "res_pt_vs_geneta_lpt":            Hist(samp, cut, ele_eta_res,    res_pt,                storage=hist.storage.Weight()),
        "res_pt_vs_geneta_ged":            Hist(samp, cut, ele_eta_res,    res_pt,                storage=hist.storage.Weight()),
        "res_e_vs_geneta_lpt":             Hist(samp, cut, ele_eta_res,    res_e,                 storage=hist.storage.Weight()),
        "res_e_vs_geneta_ged":             Hist(samp, cut, ele_eta_res,    res_e,                 storage=hist.storage.Weight()),
        "res_pt_vs_genpt_genlxy_lpt":      Hist(samp, cut, genpt_coarse,   genlxy_coarse, res_pt, storage=hist.storage.Weight()),
        "res_pt_vs_genpt_genlxy_ged":      Hist(samp, cut, genpt_coarse,   genlxy_coarse, res_pt, storage=hist.storage.Weight()),
        "res_e_vs_genpt_genlxy_lpt":       Hist(samp, cut, genpt_coarse,   genlxy_coarse, res_e,  storage=hist.storage.Weight()),
        "res_e_vs_genpt_genlxy_ged":       Hist(samp, cut, genpt_coarse,   genlxy_coarse, res_e,  storage=hist.storage.Weight()),
        "res_pt_vs_genpt_geneta_lpt":      Hist(samp, cut, genpt_coarse,   geneta_coarse, res_pt, storage=hist.storage.Weight()),
        "res_pt_vs_genpt_geneta_ged":      Hist(samp, cut, genpt_coarse,   geneta_coarse, res_pt, storage=hist.storage.Weight()),
        "res_e_vs_genpt_geneta_lpt":       Hist(samp, cut, genpt_coarse,   geneta_coarse, res_e,  storage=hist.storage.Weight()),
        "res_e_vs_genpt_geneta_ged":       Hist(samp, cut, genpt_coarse,   geneta_coarse, res_e,  storage=hist.storage.Weight()),
        "res_pt_alpt":                     Hist(samp, cut, res_pt,                                storage=hist.storage.Weight()),
        "res_e_alpt":                      Hist(samp, cut, res_e,                                 storage=hist.storage.Weight()),
        "res_pt_vs_genpt_alpt":            Hist(samp, cut, ele_pt_res,     res_pt,                storage=hist.storage.Weight()),
        "res_e_vs_genpt_alpt":             Hist(samp, cut, ele_pt_res,     res_e,                 storage=hist.storage.Weight()),
        "res_pt_vs_genlxy_alpt":           Hist(samp, cut, ele_lxy_res,    res_pt,                storage=hist.storage.Weight()),
        "res_e_vs_genlxy_alpt":            Hist(samp, cut, ele_lxy_res,    res_e,                 storage=hist.storage.Weight()),
        "res_pt_vs_geneta_alpt":           Hist(samp, cut, ele_eta_res,    res_pt,                storage=hist.storage.Weight()),
        "res_e_vs_geneta_alpt":            Hist(samp, cut, ele_eta_res,    res_e,                 storage=hist.storage.Weight()),
        "res_pt_vs_genpt_genlxy_alpt":     Hist(samp, cut, genpt_coarse,   genlxy_coarse, res_pt, storage=hist.storage.Weight()),
        "res_e_vs_genpt_genlxy_alpt":      Hist(samp, cut, genpt_coarse,   genlxy_coarse, res_e,  storage=hist.storage.Weight()),
        "res_pt_vs_genpt_geneta_alpt":     Hist(samp, cut, genpt_coarse,   geneta_coarse, res_pt, storage=hist.storage.Weight()),
        "res_e_vs_genpt_geneta_alpt":      Hist(samp, cut, genpt_coarse,   geneta_coarse, res_e,  storage=hist.storage.Weight()),
        # cross-cleaned LowPt: in AllLptElectron but removed by cross-cleaning (not in LptElectron)
        "res_pt_xlpt":                     Hist(samp, cut, res_pt,                                storage=hist.storage.Weight()),
        "res_e_xlpt":                      Hist(samp, cut, res_e,                                 storage=hist.storage.Weight()),
        "res_pt_vs_genpt_xlpt":            Hist(samp, cut, ele_pt_res,     res_pt,                storage=hist.storage.Weight()),
        "res_e_vs_genpt_xlpt":             Hist(samp, cut, ele_pt_res,     res_e,                 storage=hist.storage.Weight()),
        "res_pt_vs_genlxy_xlpt":           Hist(samp, cut, ele_lxy_res,    res_pt,                storage=hist.storage.Weight()),
        "res_e_vs_genlxy_xlpt":            Hist(samp, cut, ele_lxy_res,    res_e,                 storage=hist.storage.Weight()),
        "res_pt_vs_geneta_xlpt":           Hist(samp, cut, ele_eta_res,    res_pt,                storage=hist.storage.Weight()),
        "res_e_vs_geneta_xlpt":            Hist(samp, cut, ele_eta_res,    res_e,                 storage=hist.storage.Weight()),
        "res_pt_vs_genpt_genlxy_xlpt":     Hist(samp, cut, genpt_coarse,   genlxy_coarse, res_pt, storage=hist.storage.Weight()),
        "res_e_vs_genpt_genlxy_xlpt":      Hist(samp, cut, genpt_coarse,   genlxy_coarse, res_e,  storage=hist.storage.Weight()),
        "res_pt_vs_genpt_geneta_xlpt":     Hist(samp, cut, genpt_coarse,   geneta_coarse, res_pt, storage=hist.storage.Weight()),
        "res_e_vs_genpt_geneta_xlpt":      Hist(samp, cut, genpt_coarse,   geneta_coarse, res_e,  storage=hist.storage.Weight()),
        # dxy resolution
        "res_dxy_lpt":                     Hist(samp, cut, res_dxy,                                storage=hist.storage.Weight()),
        "res_dxy_ged":                     Hist(samp, cut, res_dxy,                                storage=hist.storage.Weight()),
        "res_dxy_vs_genpt_lpt":            Hist(samp, cut, ele_pt_res,     res_dxy,                storage=hist.storage.Weight()),
        "res_dxy_vs_genpt_ged":            Hist(samp, cut, ele_pt_res,     res_dxy,                storage=hist.storage.Weight()),
        "res_dxy_vs_genlxy_lpt":           Hist(samp, cut, ele_lxy_res,    res_dxy,                storage=hist.storage.Weight()),
        "res_dxy_vs_genlxy_ged":           Hist(samp, cut, ele_lxy_res,    res_dxy,                storage=hist.storage.Weight()),
        "res_dxy_vs_geneta_lpt":           Hist(samp, cut, ele_eta_res,    res_dxy,                storage=hist.storage.Weight()),
        "res_dxy_vs_geneta_ged":           Hist(samp, cut, ele_eta_res,    res_dxy,                storage=hist.storage.Weight()),
        "res_dxy_vs_genpt_genlxy_lpt":     Hist(samp, cut, genpt_coarse,   genlxy_coarse, res_dxy, storage=hist.storage.Weight()),
        "res_dxy_vs_genpt_genlxy_ged":     Hist(samp, cut, genpt_coarse,   genlxy_coarse, res_dxy, storage=hist.storage.Weight()),
        "res_dxy_vs_genpt_geneta_lpt":     Hist(samp, cut, genpt_coarse,   geneta_coarse, res_dxy, storage=hist.storage.Weight()),
        "res_dxy_vs_genpt_geneta_ged":     Hist(samp, cut, genpt_coarse,   geneta_coarse, res_dxy, storage=hist.storage.Weight()),
        "res_dxy_alpt":                    Hist(samp, cut, res_dxy,                                storage=hist.storage.Weight()),
        "res_dxy_vs_genpt_alpt":           Hist(samp, cut, ele_pt_res,     res_dxy,                storage=hist.storage.Weight()),
        "res_dxy_vs_genlxy_alpt":          Hist(samp, cut, ele_lxy_res,    res_dxy,                storage=hist.storage.Weight()),
        "res_dxy_vs_geneta_alpt":          Hist(samp, cut, ele_eta_res,    res_dxy,                storage=hist.storage.Weight()),
        "res_dxy_vs_genpt_genlxy_alpt":    Hist(samp, cut, genpt_coarse,   genlxy_coarse, res_dxy, storage=hist.storage.Weight()),
        "res_dxy_vs_genpt_geneta_alpt":    Hist(samp, cut, genpt_coarse,   geneta_coarse, res_dxy, storage=hist.storage.Weight()),
        "res_dxy_xlpt":                    Hist(samp, cut, res_dxy,                                storage=hist.storage.Weight()),
        "res_dxy_vs_genpt_xlpt":           Hist(samp, cut, ele_pt_res,     res_dxy,                storage=hist.storage.Weight()),
        "res_dxy_vs_genlxy_xlpt":          Hist(samp, cut, ele_lxy_res,    res_dxy,                storage=hist.storage.Weight()),
        "res_dxy_vs_geneta_xlpt":          Hist(samp, cut, ele_eta_res,    res_dxy,                storage=hist.storage.Weight()),
        "res_dxy_vs_genpt_genlxy_xlpt":    Hist(samp, cut, genpt_coarse,   genlxy_coarse, res_dxy, storage=hist.storage.Weight()),
        "res_dxy_vs_genpt_geneta_xlpt":    Hist(samp, cut, genpt_coarse,   geneta_coarse, res_dxy, storage=hist.storage.Weight()),
    }
    return histograms

subroutines = []

# GenEle/Pos ['charge', 'motherID', 'pt', 'eta', 'phi', 'energy', 'px', 'py', 'pz', 'vxy', 'vz', 'vx', 'vy', 'matched', 'matchType', 'matchIdxLocal', 'matchIdxGlobal', 'dr', 'dRbin', 'vxyBin', 'ptBin', 'matchPassID', 'matchPassIDBasic']
# Electron ['pt', 'eta', 'etaErr', 'phi', 'phiErr', 'IDcutVeto', 'IDcutLoose', 'IDcutMed', 'IDcutTight', 'IDcutVetoInt', 'IDcutLooseInt', 'IDcutMedInt', 'IDcutTightInt', 'IDmvaIso90', 'IDmvaIso80', 'IDmvaIsoLoose', 'IDmva90', 'IDmva80', 'IDmvaLoose', 'angRes', 'e', 'vxy', 'vz', 'dxy', 'dxyErr', 'dz', 'dzErr', 'trkChi2', 'trkIso', 'trkRelIso', 'calIso', 'calRelIso', 'PFIso', 'PFRelIso', 'miniIso', 'miniRelIso', 'PFIsoEleCorr', 'PFRelIsoEleCorr', 'miniIsoEleCorr', 'miniRelIsoEleCorr', 'chadIso', 'nhadIso', 'phoIso', 'rhoEA', 'trkProb', 'numTrackerHits', 'numPixHits', 'numStripHits', 'charge', 'isPF', 'genMatched', 'matchType', 'dRJets', 'dPhiJets', 'full55sigmaIetaIeta', 'absdEtaSeed', 'absdPhiIn', 'HoverE', 'abs1overEm1overP', 'expMissingInnerHits', 'conversionVeto', 'isEE', 'hasLptMatch', 'lptMatchIdx', 'mindRj', 'mindPhiJ', 'IDscore', 'passID', 'passIDBasic']
# LptElectron ['pt', 'eta', 'etaErr', 'phi', 'phiErr', 'ID', 'angRes', 'e', 'vxy', 'vz', 'dxy', 'dxyErr', 'dz', 'dzErr', 'trkChi2', 'trkIso', 'trkRelIso', 'calIso', 'calRelIso', 'PFIso', 'PFRelIso', 'miniIso', 'miniRelIso', 'PFIsoEleCorr', 'PFRelIsoEleCorr', 'miniIsoEleCorr', 'miniRelIsoEleCorr', 'chadIso', 'nhadIso', 'phoIso', 'rhoEA', 'trkProb', 'numTrackerHits', 'numPixHits', 'numStripHits', 'charge', 'minDRtoReg', 'isPF', 'genMatched', 'matchType', 'dRJets', 'dPhiJets', 'full55sigmaIetaIeta', 'absdEtaSeed', 'absdPhiIn', 'HoverE', 'abs1overEm1overP', 'expMissingInnerHits', 'conversionVeto', 'isEE', 'xCleaned', 'gedIdx', 'gedIsMatched', 'mindRj', 'mindPhiJ', 'IDscore', 'passID', 'passIDBasic']

def fillHistos(events, hists, samp, cut, info, sum_wgt=1):
    wgt = events.eventWgt/sum_wgt

    # lxy_map is measured from the chi2 production vertex (the true, unsmeared
    # primary vertex) rather than the reconstructed PV, which carries ~10-15um
    # of resolution/bias that would otherwise leak into a "truth" quantity.
    # dxy_gen_map is intentionally left relative to the reco PV below, since it's
    # compared directly against reco dxy (also PV-relative) for resolution studies.
    chi2 = ak.firsts(events.GenPart[np.abs(events.GenPart.ID) == 1000023])
    lxy_map = {
        'ele': np.sqrt((events.GenEle.vx - chi2.vx)**2 + (events.GenEle.vy - chi2.vy)**2),
        'pos': np.sqrt((events.GenPos.vx - chi2.vx)**2 + (events.GenPos.vy - chi2.vy)**2),
    }
    dxy_gen_map = {
        'ele': (-(events.GenEle.vx - events.PV.x) * events.GenEle.py + (events.GenEle.vy - events.PV.y) * events.GenEle.px) / events.GenEle.pt,
        'pos': (-(events.GenPos.vx - events.PV.x) * events.GenPos.py + (events.GenPos.vy - events.PV.y) * events.GenPos.px) / events.GenPos.pt,
    }

    fill_specs = [
        ('lpt',  lambda g: g.matchType == 'L',                       events.LptElectron,   lambda g: g.matchIdxLocal),
        ('ged',  lambda g: g.matchType == 'R',                       events.Electron,       lambda g: g.matchIdxLocal),
        ('alpt', lambda g: g.matchedAllLowPt,                         events.AllLptElectron, lambda g: g.matchIdxAllLowPt),
        ('xlpt', lambda g: g.matchedAllLowPt & (g.matchType != 'L'), events.AllLptElectron, lambda g: g.matchIdxAllLowPt),
    ]

    for suffix, get_mask, reco_coll, get_idx in fill_specs:
        for gen_key, gen_coll in [('ele', events.GenEle), ('pos', events.GenPos)]:
            gen_lxy_all = lxy_map[gen_key]
            mask = get_mask(gen_coll)
            if ak.sum(mask) == 0:
                continue
            gen  = gen_coll[mask]
            reco = reco_coll[mask]
            w    = wgt[mask]
            lxy  = gen_lxy_all[mask]
            idx   = get_idx(gen)
            local = ak.local_index(reco.pt, axis=1)
            sel_pt  = reco.pt[ local == idx]
            sel_e   = reco.e[  local == idx]
            sel_dxy = reco.dxy[local == idx]
            valid  = ak.num(sel_pt) > 0
            reco_pt  = ak.flatten(sel_pt[ valid])
            reco_e   = ak.flatten(sel_e[  valid])
            reco_dxy = ak.flatten(sel_dxy[valid])
            gen_pt   = gen.pt[valid]
            gen_e    = gen.energy[valid]
            gen_eta  = gen.eta[valid]
            gen_lxy  = lxy[valid]
            gen_dxy  = dxy_gen_map[gen_key][mask][valid]
            w_v      = w[valid]
            res_pt   = (reco_pt  - gen_pt)  / gen_pt
            res_e    = (reco_e   - gen_e)   / gen_e
            res_dxy  = (reco_dxy - gen_dxy) / gen_dxy

            hists[f"res_pt_{suffix}"                 ].fill(samp=samp, cut=cut, res=res_pt,                           weight=w_v)
            hists[f"res_e_{suffix}"                  ].fill(samp=samp, cut=cut, res=res_e,                            weight=w_v)
            hists[f"res_pt_vs_genpt_{suffix}"        ].fill(samp=samp, cut=cut, pt=gen_pt,   res=res_pt,              weight=w_v)
            hists[f"res_e_vs_genpt_{suffix}"         ].fill(samp=samp, cut=cut, pt=gen_pt,   res=res_e,               weight=w_v)
            hists[f"res_pt_vs_genlxy_{suffix}"       ].fill(samp=samp, cut=cut, lxy=gen_lxy, res=res_pt,              weight=w_v)
            hists[f"res_e_vs_genlxy_{suffix}"        ].fill(samp=samp, cut=cut, lxy=gen_lxy, res=res_e,               weight=w_v)
            hists[f"res_pt_vs_geneta_{suffix}"       ].fill(samp=samp, cut=cut, eta=gen_eta, res=res_pt,              weight=w_v)
            hists[f"res_e_vs_geneta_{suffix}"        ].fill(samp=samp, cut=cut, eta=gen_eta, res=res_e,               weight=w_v)
            hists[f"res_pt_vs_genpt_genlxy_{suffix}" ].fill(samp=samp, cut=cut, pt=gen_pt,   lxy=gen_lxy, res=res_pt, weight=w_v)
            hists[f"res_e_vs_genpt_genlxy_{suffix}"  ].fill(samp=samp, cut=cut, pt=gen_pt,   lxy=gen_lxy, res=res_e,  weight=w_v)
            hists[f"res_pt_vs_genpt_geneta_{suffix}" ].fill(samp=samp, cut=cut, pt=gen_pt,   eta=gen_eta, res=res_pt, weight=w_v)
            hists[f"res_e_vs_genpt_geneta_{suffix}"  ].fill(samp=samp, cut=cut, pt=gen_pt,   eta=gen_eta, res=res_e,  weight=w_v)
            hists[f"res_dxy_{suffix}"                ].fill(samp=samp, cut=cut, res=res_dxy,                          weight=w_v)
            hists[f"res_dxy_vs_genpt_{suffix}"       ].fill(samp=samp, cut=cut, pt=gen_pt,   res=res_dxy,             weight=w_v)
            hists[f"res_dxy_vs_genlxy_{suffix}"      ].fill(samp=samp, cut=cut, lxy=gen_lxy, res=res_dxy,             weight=w_v)
            hists[f"res_dxy_vs_geneta_{suffix}"      ].fill(samp=samp, cut=cut, eta=gen_eta, res=res_dxy,             weight=w_v)
            hists[f"res_dxy_vs_genpt_genlxy_{suffix}"].fill(samp=samp, cut=cut, pt=gen_pt,   lxy=gen_lxy, res=res_dxy, weight=w_v)
            hists[f"res_dxy_vs_genpt_geneta_{suffix}"].fill(samp=samp, cut=cut, pt=gen_pt,   eta=gen_eta, res=res_dxy, weight=w_v)
