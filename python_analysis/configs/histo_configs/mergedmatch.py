try:
    from histobins import *
except ModuleNotFoundError:
    from configs.histo_configs.histobins import *
from hist import Hist
from hist.axis import Variable, Regular, StrCategory
import hist
import numpy as np
import awkward as ak

# Two event topologies, selected per collection:
#
#   merged: exactly one reco electron is within dR < 0.1 of BOTH gen signal
#           particles simultaneously; no second reco electron is matched to either.
#     merged_i  = dR(reco_i, GenEle) < 0.1  AND  dR(reco_i, GenPos) < 0.1
#     matched_i = dR(reco_i, GenEle) < 0.1  OR   dR(reco_i, GenPos) < 0.1
#     keep iff: any(merged_i) AND sum(matched_i) == 1
#
#   sep (classical): one reco electron matched exclusively to GenEle, a different one
#        exclusively to GenPos; no reco matched to both.
#     only_ele_i = dR(reco_i, GenEle) < 0.1  AND  dR(reco_i, GenPos) >= 0.1
#     only_pos_i = dR(reco_i, GenPos) < 0.1  AND  dR(reco_i, GenEle) >= 0.1
#     keep iff: any(only_ele_i) AND any(only_pos_i) AND NOT any(merged_i)
#
#   sep (ambiguous): a merged reco exists but a SECOND reco is also within dR < 0.1
#        of at least one gen particle. Even though the merged_i condition is satisfied,
#        the presence of a second matched reco indicates the event is better described
#        as separated at the reco level.
#     keep iff: any(merged_i) AND sum(matched_i) > 1

# Axis for reco-gen dR: by construction both are < 0.1, so zoom in past that threshold.
# gen-gen dR uses ee_dr_narrow (0-0.2) from histobins since it can reach up to ~0.2.
_dr_reco_gen = Regular(100, 0, 0.15, name='dr', label=r'$\Delta R$')

def make_histograms():
    histograms = {}
    for coll in ['gedlpt', 'alpt']:
        # ── merged-only hists ─────────────────────────────────────────────────
        # pt
        histograms[f'gen_lead_pt_{coll}']    = Hist(samp, cut, ele_pt, storage=hist.storage.Weight())
        histograms[f'gen_sublead_pt_{coll}'] = Hist(samp, cut, ele_pt, storage=hist.storage.Weight())
        histograms[f'reco_pt_{coll}']        = Hist(samp, cut, ele_pt, storage=hist.storage.Weight())
        # eta
        histograms[f'gen_lead_eta_{coll}']    = Hist(samp, cut, ele_eta, storage=hist.storage.Weight())
        histograms[f'gen_sublead_eta_{coll}'] = Hist(samp, cut, ele_eta, storage=hist.storage.Weight())
        histograms[f'reco_eta_{coll}']        = Hist(samp, cut, ele_eta, storage=hist.storage.Weight())
        # dR pairs
        histograms[f'dr_reco_gen_lead_{coll}']    = Hist(samp, cut, _dr_reco_gen, storage=hist.storage.Weight())
        histograms[f'dr_reco_gen_sublead_{coll}'] = Hist(samp, cut, _dr_reco_gen, storage=hist.storage.Weight())
        histograms[f'dr_gen_gen_{coll}']          = Hist(samp, cut, ee_dr_narrow,  storage=hist.storage.Weight())
        # displacement (merged only)
        histograms[f'reco_dxy_{coll}'] = Hist(samp, cut, ele_dxy, storage=hist.storage.Weight())

        # ── merged/sep comparison hists ───────────────────────────────────────
        for scenario in ['merged', 'sep']:
            s = f'{scenario}_{coll}'
            histograms[f'gen_lxy_{s}']       = Hist(samp, cut, ele_lxy_res,          storage=hist.storage.Weight())
            histograms[f'gen_ee_dr_{s}']     = Hist(samp, cut, ee_dr,                storage=hist.storage.Weight())
            histograms[f'gen_ee_pt_{s}']     = Hist(samp, cut, ele_pt,               storage=hist.storage.Weight())
            histograms[f'reco_trkchi2_{s}']  = Hist(samp, cut, ele_chi2,             storage=hist.storage.Weight())
            histograms[f'dr_gen_vs_lxy_{s}'] = Hist(samp, cut, ele_lxy_res, ee_dr,   storage=hist.storage.Weight())
    return histograms

subroutines = []

def fillHistos(events, hists, samp, cut, info, sum_wgt=1):
    wgt = events.eventWgt / sum_wgt

    def _project(coll):
        return ak.zip({
            'pt':                coll.pt,
            'eta':               coll.eta,
            'phi':               coll.phi,
            'dxy':               coll.dxy,
            'dz':                coll.dz,
            'trkChi2':           coll.trkChi2,
            'PFRelIso':          coll.PFRelIso,
            'miniRelIsoEleCorr': coll.miniRelIsoEleCorr,
            'mindRj':            coll.mindRj,
        })

    events['AllLptElectron', 'mindRj'] = ak.fill_none(
        ak.min(events.AllLptElectron.dRJets, axis=-1), 999
    )

    gedlpt_coll = ak.concatenate(
        [_project(events.Electron), _project(events.LptElectron)], axis=1
    )
    alpt_coll = _project(events.AllLptElectron)

    coll_map = {'gedlpt': gedlpt_coll, 'alpt': alpt_coll}

    gen_lxy = np.sqrt(
        (events.GenEle.vx - events.PV.x)**2 +
        (events.GenEle.vy - events.PV.y)**2
    )

    for coll_label, coll in coll_map.items():

        def _dphi(a, b):
            d = np.abs(a - b)
            return ak.where(d > np.pi, 2 * np.pi - d, d)

        dphi_e = _dphi(coll.phi, events.GenEle.phi)
        dphi_p = _dphi(coll.phi, events.GenPos.phi)
        dr_to_ele = np.sqrt((coll.eta - events.GenEle.eta)**2 + dphi_e**2)
        dr_to_pos = np.sqrt((coll.eta - events.GenPos.eta)**2 + dphi_p**2)

        merged_i   = (dr_to_ele < 0.1) & (dr_to_pos < 0.1)
        matched_i  = (dr_to_ele < 0.1) | (dr_to_pos < 0.1)
        only_ele_i = (dr_to_ele < 0.1) & (dr_to_pos >= 0.1)
        only_pos_i = (dr_to_pos < 0.1) & (dr_to_ele >= 0.1)

        # ── merged selection ──────────────────────────────────────────────────
        has_merged = ak.any(merged_i,  axis=1)
        n_matched  = ak.sum(matched_i, axis=1)
        keep       = has_merged & (n_matched == 1)

        if ak.sum(keep) > 0:
            merged_ele  = ak.firsts(coll[merged_i])[keep]
            w           = wgt[keep]
            lxy         = gen_lxy[keep]
            gen_ele     = events.GenEle[keep]
            gen_pos     = events.GenPos[keep]

            lead_is_ele     = gen_ele.pt >= gen_pos.pt
            gen_lead_pt     = ak.where(lead_is_ele, gen_ele.pt,  gen_pos.pt)
            gen_sublead_pt  = ak.where(lead_is_ele, gen_pos.pt,  gen_ele.pt)
            gen_lead_eta    = ak.where(lead_is_ele, gen_ele.eta, gen_pos.eta)
            gen_sublead_eta = ak.where(lead_is_ele, gen_pos.eta, gen_ele.eta)

            dr_reco_ele        = ak.firsts(dr_to_ele[merged_i])[keep]
            dr_reco_pos        = ak.firsts(dr_to_pos[merged_i])[keep]
            dr_reco_gen_lead   = ak.where(lead_is_ele, dr_reco_ele, dr_reco_pos)
            dr_reco_gen_sublead = ak.where(lead_is_ele, dr_reco_pos, dr_reco_ele)

            dphi_gg = _dphi(gen_ele.phi, gen_pos.phi)
            dr_gg   = np.sqrt((gen_ele.eta - gen_pos.eta)**2 + dphi_gg**2)

            m = f'merged_{coll_label}'

            hists[f'gen_lead_pt_{coll_label}'        ].fill(samp=samp, cut=cut, pt=gen_lead_pt,            weight=w)
            hists[f'gen_sublead_pt_{coll_label}'     ].fill(samp=samp, cut=cut, pt=gen_sublead_pt,         weight=w)
            hists[f'reco_pt_{coll_label}'            ].fill(samp=samp, cut=cut, pt=merged_ele.pt,          weight=w)
            hists[f'gen_lead_eta_{coll_label}'       ].fill(samp=samp, cut=cut, eta=gen_lead_eta,          weight=w)
            hists[f'gen_sublead_eta_{coll_label}'    ].fill(samp=samp, cut=cut, eta=gen_sublead_eta,       weight=w)
            hists[f'reco_eta_{coll_label}'           ].fill(samp=samp, cut=cut, eta=merged_ele.eta,        weight=w)
            hists[f'dr_reco_gen_lead_{coll_label}'   ].fill(samp=samp, cut=cut, dr=dr_reco_gen_lead,       weight=w)
            hists[f'dr_reco_gen_sublead_{coll_label}'].fill(samp=samp, cut=cut, dr=dr_reco_gen_sublead,    weight=w)
            hists[f'dr_gen_gen_{coll_label}'         ].fill(samp=samp, cut=cut, dr=dr_gg,                  weight=w)
            hists[f'reco_dxy_{coll_label}'           ].fill(samp=samp, cut=cut, dxy=np.abs(merged_ele.dxy), weight=w)
            hists[f'gen_lxy_{m}'                     ].fill(samp=samp, cut=cut, lxy=lxy,                   weight=w)
            hists[f'gen_ee_dr_{m}'                   ].fill(samp=samp, cut=cut, dr=events.genEE.dr[keep],  weight=w)
            hists[f'gen_ee_pt_{m}'                   ].fill(samp=samp, cut=cut, pt=events.genEE.pt[keep],  weight=w)
            hists[f'reco_trkchi2_{m}'                ].fill(samp=samp, cut=cut, chi2=merged_ele.trkChi2,   weight=w)
            hists[f'dr_gen_vs_lxy_{m}'               ].fill(samp=samp, cut=cut, lxy=lxy, dr=dr_gg,         weight=w)

        # ── separated selection ───────────────────────────────────────────────
        has_ele_match = ak.any(only_ele_i, axis=1)
        has_pos_match = ak.any(only_pos_i, axis=1)
        # classical: exclusive one-to-one matching, no merged reco
        sep_classical = has_ele_match & has_pos_match & ~has_merged
        # ambiguous: merged reco exists but a second reco is also close to a gen particle
        sep_ambiguous = has_merged & (n_matched > 1)
        sep_keep      = sep_classical | sep_ambiguous

        if ak.sum(sep_keep) > 0:
            w_sep   = wgt[sep_keep]
            lxy_sep = gen_lxy[sep_keep]

            dphi_gg_sep = _dphi(events.GenEle.phi[sep_keep], events.GenPos.phi[sep_keep])
            dr_gg_sep   = np.sqrt(
                (events.GenEle.eta[sep_keep] - events.GenPos.eta[sep_keep])**2 + dphi_gg_sep**2
            )

            s = f'sep_{coll_label}'

            hists[f'gen_lxy_{s}'      ].fill(samp=samp, cut=cut, lxy=lxy_sep,                  weight=w_sep)
            hists[f'gen_ee_dr_{s}'    ].fill(samp=samp, cut=cut, dr=events.genEE.dr[sep_keep], weight=w_sep)
            hists[f'gen_ee_pt_{s}'    ].fill(samp=samp, cut=cut, pt=events.genEE.pt[sep_keep], weight=w_sep)
            hists[f'dr_gen_vs_lxy_{s}'].fill(samp=samp, cut=cut, lxy=lxy_sep, dr=dr_gg_sep,    weight=w_sep)

            # trkChi2: classical events use dedicated ele/pos recos; ambiguous events
            # use the merged reco and the second matched reco
            if ak.sum(sep_classical) > 0:
                sep_ele_reco = ak.firsts(coll[only_ele_i])[sep_classical]
                sep_pos_reco = ak.firsts(coll[only_pos_i])[sep_classical]
                w_classical  = wgt[sep_classical]
                hists[f'reco_trkchi2_{s}'].fill(samp=samp, cut=cut, chi2=sep_ele_reco.trkChi2, weight=w_classical)
                hists[f'reco_trkchi2_{s}'].fill(samp=samp, cut=cut, chi2=sep_pos_reco.trkChi2, weight=w_classical)
            if ak.sum(sep_ambiguous) > 0:
                amb_merged_reco = ak.firsts(coll[merged_i])[sep_ambiguous]
                amb_second_reco = ak.firsts(coll[matched_i & ~merged_i])[sep_ambiguous]
                w_amb = wgt[sep_ambiguous]
                hists[f'reco_trkchi2_{s}'].fill(samp=samp, cut=cut, chi2=amb_merged_reco.trkChi2, weight=w_amb)
                hists[f'reco_trkchi2_{s}'].fill(samp=samp, cut=cut, chi2=amb_second_reco.trkChi2, weight=w_amb)
