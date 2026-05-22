try:
    from histobins import *
except ModuleNotFoundError:
    from configs.hists.histobins import *
from hist import Hist
import hist
import numpy as np
import awkward as ak
import vector
vector.register_awkward()

def make_histograms():
    histograms = {
        # quantities associated w/ selected vertex
        "gen_leading_ele_pt" :       Hist(samp, cut, ele_pt,        storage=hist.storage.Weight()),
        "gen_subleading_ele_pt" :    Hist(samp, cut, ele_pt,        storage=hist.storage.Weight()),
        "gen_leading_ele_eta" :      Hist(samp, cut, ele_eta,       storage=hist.storage.Weight()),
        "gen_subleading_ele_eta" :   Hist(samp, cut, ele_eta,       storage=hist.storage.Weight()),
        "gen_leading_ele_vxy" :      Hist(samp, cut, vxy_coarse,    storage=hist.storage.Weight()),
        "gen_subleading_ele_vxy" :   Hist(samp, cut, vxy_coarse,    storage=hist.storage.Weight()),
        "gen_leading_ele_vz" :       Hist(samp, cut, vz_coarse,     storage=hist.storage.Weight()),
        "gen_subleading_ele_vz" :    Hist(samp, cut, vz_coarse,     storage=hist.storage.Weight()),
        "gen_diele_dR" :             Hist(samp, cut, ee_dr,         storage=hist.storage.Weight()),
        "gen_diele_mass" :           Hist(samp, cut, ee_mass,       storage=hist.storage.Weight()),
        "gen_diele_pt" :             Hist(samp, cut, ele_pt,        storage=hist.storage.Weight()),
        "gen_diele_pt_vs_dr" :       Hist(samp, cut, ele_pt, ee_dr_narrow, storage=hist.storage.Weight()),
        "gen_diele_ctau" :           Hist(samp, cut, ee_ctau,       storage=hist.storage.Weight()),
        "gen_diele_ctau_proper" :    Hist(samp, cut, ee_ctau_pr,    storage=hist.storage.Weight()),
        "gen_diele_met_dphi" :       Hist(samp, cut, ee_met_dphi,   storage=hist.storage.Weight()),
        "gen_met" :                  Hist(samp, cut, met_pt,        storage=hist.storage.Weight()),
        "gen_eechi1_ctau_proper" :   Hist(samp, cut, ee_ctau_pr,    storage=hist.storage.Weight()),
        "gen_chi2_ctau_proper" :     Hist(samp, cut, ee_ctau_pr,    storage=hist.storage.Weight()),
    }
    #print('init', histograms["gen_leading_ele_pt"].to_numpy())
    return histograms

subroutines = []

def fillHistos(events, hists, samp, cut, info, sum_wgt=1):
    wgt = events.eventWgt/sum_wgt

    if info['type'] == "signal":

        gen_eles = events.GenPart[np.abs(events.GenPart.ID) == 11]
        sortGenEbypt = ak.argsort(gen_eles.pt, ascending=False)
        GenE = gen_eles[sortGenEbypt]

        chi2 = events.GenPart[np.abs(events.GenPart.ID) == 1000023]
        chi1 = events.GenPart[np.abs(events.GenPart.ID) == 1000022]
        chi1_prompt = chi1[chi1.motherID != 1000023]
        chi1_displaced = chi1[chi1.motherID == 1000023]

        assert all(ak.num(chi1_prompt) == 1)
        assert all(ak.num(chi1_displaced) == 1)
        assert all(ak.num(chi2) == 1)
        
        chi2 = chi2[:, 0]
        chi1_prompt = chi1_prompt[:, 0]
        chi1_displaced = chi1_displaced[:, 0]
        
        # GenE ['ID', 'motherID', 'charge', 'pt', 'eta', 'phi', 'e', 'px', 'py', 'pz', 'vxy', 'vx', 'vy', 'vz', 'mass']
        # genEE ['pt', 'eta', 'phi', 'energy', 'mass', 'dr', 'METdPhi', 'vxy', 'vz', 'vx', 'vy']

        #print(GenE.pt[:, 0][0], len(GenE.pt[:, 0]))
        
        hists["gen_leading_ele_pt"     ].fill(samp = samp, cut = cut, pt = GenE.pt[:, 0], weight = wgt)
        #print(hists["gen_leading_ele_pt"     ])
        #print(hists["gen_leading_ele_pt"     ].to_numpy())
        #print('fill', GenE.pt[:, 0], events.eventWgt, sum_wgt, wgt)
        hists["gen_subleading_ele_pt"  ].fill(samp = samp, cut = cut, pt = GenE.pt[:, 1], weight = wgt)
        hists["gen_leading_ele_eta"    ].fill(samp = samp, cut = cut, eta = GenE.eta[:, 0], weight = wgt)
        hists["gen_subleading_ele_eta" ].fill(samp = samp, cut = cut, eta = GenE.eta[:, 1], weight = wgt)
        hists["gen_leading_ele_vxy"    ].fill(samp = samp, cut = cut, vxy = GenE.vxy[:, 0], weight = wgt)
        hists["gen_subleading_ele_vxy" ].fill(samp = samp, cut = cut, vxy = GenE.vxy[:, 1], weight = wgt)
        hists["gen_leading_ele_vz"     ].fill(samp = samp, cut = cut, vz = GenE.vz[:, 0], weight = wgt)
        hists["gen_subleading_ele_vz"  ].fill(samp = samp, cut = cut, vz = GenE.vz[:, 1], weight = wgt)

        hists["gen_diele_dR"           ].fill(samp = samp, cut = cut, dr = events.genEE.dr, weight = wgt)
        hists["gen_diele_mass"         ].fill(samp = samp, cut = cut, mass = events.genEE.mass, weight = wgt)
        hists["gen_diele_pt"           ].fill(samp = samp, cut = cut, pt = events.genEE.pt, weight = wgt)
        hists["gen_diele_pt_vs_dr"     ].fill(samp = samp, cut = cut, pt = events.genEE.pt, dr = events.genEE.dr, weight = wgt)
        ctau_lab = np.sqrt(events.genEE.vxy**2 + events.genEE.vz**2) * 10
        hists["gen_diele_ctau"         ].fill(samp = samp, cut = cut, ctau = ctau_lab, weight = wgt)
        hists["gen_diele_met_dphi"     ].fill(samp = samp, cut = cut, dphi = events.genEE.METdPhi, weight = wgt)
        hists["gen_met"                ].fill(samp = samp, cut = cut, met_pt = events.GenMET.pt, weight = wgt)
        ee_p3 = events.genEE.pt * np.cosh(events.genEE.eta)
        ee_betagamma = ee_p3 / events.genEE.mass
        ee_ctau_proper = ctau_lab / ee_betagamma
        hists["gen_diele_ctau_proper"  ].fill(samp = samp, cut = cut, ctau = ee_ctau_proper, weight = wgt)
        
        chi2_p3 = chi2.pt * np.cosh(chi2.eta)
        chi2_betagamma = chi2_p3 / chi2.mass
        chi2_ctau_proper = ctau_lab / chi2_betagamma
        hists["gen_chi2_ctau_proper"  ].fill(samp = samp, cut = cut, ctau = chi2_ctau_proper, weight = wgt)

        eechi1_p3 = np.sqrt((GenE[:,0].px + GenE[:,1].px + chi1_displaced.px)**2 +
                            (GenE[:,0].py + GenE[:,1].py + chi1_displaced.py)**2 +
                            (GenE[:,0].pz + GenE[:,1].pz + chi1_displaced.pz)**2)
        eechi1_e = GenE[:,0].e + GenE[:,1].e + chi1_displaced.e
        eechi1_mass2 = eechi1_e**2 - eechi1_p3**2
        eechi1_mask = (eechi1_p3 > 0) & (eechi1_e > 0) & (eechi1_mass2 > 0)
        eechi1_mass = np.sqrt(eechi1_mass2[eechi1_mask])
        eechi1_betagamma = eechi1_p3[eechi1_mask] / eechi1_mass
        eechi1_ctau_proper = ctau_lab[eechi1_mask] / eechi1_betagamma
        #print(eechi1_p3, eechi1_e, eechi1_mass, eechi1_betagamma, eechi1_ctau_proper)
        hists["gen_eechi1_ctau_proper"  ].fill(samp = samp, cut = cut, ctau = eechi1_ctau_proper, weight = wgt[eechi1_mask])

        #print(chi2.mass, chi2_betagamma)
        #print(eechi1_mass, eechi1_betagamma)
        # These should be identical event-by-event if reconstruction is consistent
        #print(ak.mean(eechi1_mass))
        #print(ak.mean(chi2.mass))
        # Also check the boost
        #print(ak.mean(eechi1_betagamma))
        #print(ak.mean(chi2_betagamma))
