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

# --- axes for vertex-offset diagnostics ---
# Floor is set far below any physical scale (down to 1e-9 cm) so that a
# vertex pair that's numerically coincident (as opposed to genuinely close)
# shows up as a visible peak near the floor rather than vanishing into
# underflow (see gen_diele_lxy_flawed's low-mass-sample issue).
voffset_xy_coarse = Regular(100, 0, 50,      name="voffset_xy", label=r"$|\Delta v_{xy}|$ [cm]")
voffset_xy_log    = Regular(130, 1e-9, 100,  name="voffset_xy", label=r"$|\Delta v_{xy}|$ [cm]", transform=transform.log)
voffset_z_coarse  = Regular(100, 0, 50,      name="voffset_z",  label=r"$|\Delta v_{z}|$ [cm]")
voffset_z_log     = Regular(130, 1e-9, 100,  name="voffset_z",  label=r"$|\Delta v_{z}|$ [cm]", transform=transform.log)

def make_histograms():
    histograms = {
        # quantities associated w/ selected vertex
        "gen_leading_ele_pt" :       Hist(samp, cut, ele_pt,        storage=hist.storage.Weight()),
        "gen_subleading_ele_pt" :    Hist(samp, cut, ele_pt,        storage=hist.storage.Weight()),
        "gen_leading_ele_eta" :      Hist(samp, cut, ele_eta,       storage=hist.storage.Weight()),
        "gen_subleading_ele_eta" :   Hist(samp, cut, ele_eta,       storage=hist.storage.Weight()),
        "gen_leading_ele_lxy" :      Hist(samp, cut, lxy_coarse,    storage=hist.storage.Weight()),
        "gen_subleading_ele_lxy" :   Hist(samp, cut, lxy_coarse,    storage=hist.storage.Weight()),
        "gen_leading_ele_lxy_log" :      Hist(samp, cut, lxy_log,   storage=hist.storage.Weight()),
        "gen_subleading_ele_lxy_log" :   Hist(samp, cut, lxy_log,   storage=hist.storage.Weight()),
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
        # old, flawed gen Lxy: gen ee vertex measured relative to the reco PV instead of the true chi2 production vertex
        "gen_diele_lxy_flawed" :     Hist(samp, cut, lxy_coarse,    storage=hist.storage.Weight()),
        "gen_diele_lxy_flawed_log" : Hist(samp, cut, lxy_log,       storage=hist.storage.Weight()),
        # |PV - chi2 production vertex|: quantifies how much the flawed calculation above
        # (which uses PV in place of the chi2 vertex) actually differs from the fixed one.
        "pv_chi2_voffset_xy" :         Hist(samp, cut, voffset_xy_coarse, storage=hist.storage.Weight()),
        "pv_chi2_voffset_xy_log" :     Hist(samp, cut, voffset_xy_log,    storage=hist.storage.Weight()),
        "pv_chi2_voffset_z" :          Hist(samp, cut, voffset_z_coarse,  storage=hist.storage.Weight()),
        "pv_chi2_voffset_z_log" :      Hist(samp, cut, voffset_z_log,     storage=hist.storage.Weight()),
        # |GenEle vertex - chi2 vertex|: diagnostic for the separate issue found in the
        # lightest-mass signal samples, where GenPart's chi2 vx/vy/vz numerically coincides
        # with GenEle's own vertex (i.e. chi2's branch is recording its decay point, not its
        # production point) -- this should peak at a real, nonzero displacement for a properly
        # long-lived chi2, and instead pile up near the floor for the affected samples.
        "genele_chi2_voffset_xy" :     Hist(samp, cut, voffset_xy_coarse, storage=hist.storage.Weight()),
        "genele_chi2_voffset_xy_log" : Hist(samp, cut, voffset_xy_log,    storage=hist.storage.Weight()),
        "genele_chi2_voffset_z" :      Hist(samp, cut, voffset_z_coarse,  storage=hist.storage.Weight()),
        "genele_chi2_voffset_z_log" :  Hist(samp, cut, voffset_z_log,     storage=hist.storage.Weight()),
    }
    #print('init', histograms["gen_leading_ele_pt"].to_numpy())
    return histograms

subroutines = []

def fillHistos(events, hists, samp, cut, info, sum_wgt=1):
    wgt = events.eventWgt/sum_wgt

    if info['type'] == "signal":

        # Use the dedicated GenEle/GenPos branches (always motherID == chi2) instead of
        # re-deriving "the two electrons" from the generic GenPart collection, which also
        # contains final-state electrons from hadron decays/conversions elsewhere in the event.
        gen_ele = events.GenEle
        gen_pos = events.GenPos
        ele_leads = gen_ele.pt > gen_pos.pt

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
        
        # GenEle/GenPos fields: ['charge', 'motherID', 'pt', 'eta', 'phi', 'energy', 'px', 'py', 'pz',
        #                        'vxy', 'vz', 'vx', 'vy', 'matched', 'matchType', ...]
        # genEE ['pt', 'eta', 'phi', 'energy', 'mass', 'dr', 'METdPhi', 'vxy', 'vz', 'vx', 'vy']

        lead_pt  = ak.where(ele_leads, gen_ele.pt,  gen_pos.pt)
        sub_pt   = ak.where(ele_leads, gen_pos.pt,  gen_ele.pt)
        lead_eta = ak.where(ele_leads, gen_ele.eta, gen_pos.eta)
        sub_eta  = ak.where(ele_leads, gen_pos.eta, gen_ele.eta)
        lead_vx  = ak.where(ele_leads, gen_ele.vx,  gen_pos.vx)
        sub_vx   = ak.where(ele_leads, gen_pos.vx,  gen_ele.vx)
        lead_vy  = ak.where(ele_leads, gen_ele.vy,  gen_pos.vy)
        sub_vy   = ak.where(ele_leads, gen_pos.vy,  gen_ele.vy)
        lead_vz  = ak.where(ele_leads, gen_ele.vz,  gen_pos.vz)
        sub_vz   = ak.where(ele_leads, gen_pos.vz,  gen_ele.vz)

        hists["gen_leading_ele_pt"     ].fill(samp = samp, cut = cut, pt = lead_pt, weight = wgt)
        hists["gen_subleading_ele_pt"  ].fill(samp = samp, cut = cut, pt = sub_pt, weight = wgt)
        hists["gen_leading_ele_eta"    ].fill(samp = samp, cut = cut, eta = lead_eta, weight = wgt)
        hists["gen_subleading_ele_eta" ].fill(samp = samp, cut = cut, eta = sub_eta, weight = wgt)
        lxy_leading = np.sqrt((lead_vx - chi2.vx)**2 + (lead_vy - chi2.vy)**2)
        lxy_subleading = np.sqrt((sub_vx - chi2.vx)**2 + (sub_vy - chi2.vy)**2)
        hists["gen_leading_ele_lxy"    ].fill(samp = samp, cut = cut, lxy = lxy_leading, weight = wgt)
        hists["gen_subleading_ele_lxy" ].fill(samp = samp, cut = cut, lxy = lxy_subleading, weight = wgt)
        hists["gen_leading_ele_lxy_log"    ].fill(samp = samp, cut = cut, lxy = lxy_leading, weight = wgt)
        hists["gen_subleading_ele_lxy_log" ].fill(samp = samp, cut = cut, lxy = lxy_subleading, weight = wgt)
        hists["gen_leading_ele_vz"     ].fill(samp = samp, cut = cut, vz = lead_vz, weight = wgt)
        hists["gen_subleading_ele_vz"  ].fill(samp = samp, cut = cut, vz = sub_vz, weight = wgt)

        # old, flawed calculation: gen ee vertex relative to the reco PV instead of the true chi2 production vertex
        lxy_flawed = np.sqrt((events.genEE.vx - events.PV.x)**2 + (events.genEE.vy - events.PV.y)**2)
        hists["gen_diele_lxy_flawed"    ].fill(samp = samp, cut = cut, lxy = lxy_flawed, weight = wgt)
        hists["gen_diele_lxy_flawed_log"].fill(samp = samp, cut = cut, lxy = lxy_flawed, weight = wgt)

        pv_chi2_voffset_xy = np.sqrt((events.PV.x - chi2.vx)**2 + (events.PV.y - chi2.vy)**2)
        pv_chi2_voffset_z  = np.abs(events.PV.z - chi2.vz)
        hists["pv_chi2_voffset_xy"    ].fill(samp = samp, cut = cut, voffset_xy = pv_chi2_voffset_xy, weight = wgt)
        hists["pv_chi2_voffset_xy_log"].fill(samp = samp, cut = cut, voffset_xy = pv_chi2_voffset_xy, weight = wgt)
        hists["pv_chi2_voffset_z"     ].fill(samp = samp, cut = cut, voffset_z  = pv_chi2_voffset_z,  weight = wgt)
        hists["pv_chi2_voffset_z_log" ].fill(samp = samp, cut = cut, voffset_z  = pv_chi2_voffset_z,  weight = wgt)

        genele_chi2_voffset_xy = np.sqrt((gen_ele.vx - chi2.vx)**2 + (gen_ele.vy - chi2.vy)**2)
        genele_chi2_voffset_z  = np.abs(gen_ele.vz - chi2.vz)
        hists["genele_chi2_voffset_xy"    ].fill(samp = samp, cut = cut, voffset_xy = genele_chi2_voffset_xy, weight = wgt)
        hists["genele_chi2_voffset_xy_log"].fill(samp = samp, cut = cut, voffset_xy = genele_chi2_voffset_xy, weight = wgt)
        hists["genele_chi2_voffset_z"     ].fill(samp = samp, cut = cut, voffset_z  = genele_chi2_voffset_z,  weight = wgt)
        hists["genele_chi2_voffset_z_log" ].fill(samp = samp, cut = cut, voffset_z  = genele_chi2_voffset_z,  weight = wgt)

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

        eechi1_p3 = np.sqrt((gen_ele.px + gen_pos.px + chi1_displaced.px)**2 +
                            (gen_ele.py + gen_pos.py + chi1_displaced.py)**2 +
                            (gen_ele.pz + gen_pos.pz + chi1_displaced.pz)**2)
        eechi1_e = gen_ele.energy + gen_pos.energy + chi1_displaced.e
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
