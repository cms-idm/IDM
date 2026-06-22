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
    histograms = {}
    for coll in ['gedlpt', 'alpt']:
        for rank in ['lead', 'sublead']:
            tag = f'{coll}_{rank}'
            # 1D: dR to nearest gen particle (GenEle or GenPos, whichever is closer)
            histograms[f'dr_to_gen_{tag}']                   = Hist(samp, cut, dR,                      storage=hist.storage.Weight())
            # 2D: dR vs secondary reco/gen quantities
            histograms[f'dr_to_gen_vs_dRj_{tag}']            = Hist(samp, cut, dRj,             dR,     storage=hist.storage.Weight())
            histograms[f'dr_to_gen_vs_PFRelIso_{tag}']       = Hist(samp, cut, ele_PFRelIso,    dR,     storage=hist.storage.Weight())
            histograms[f'dr_to_gen_vs_miniRelIsoCorr_{tag}'] = Hist(samp, cut, ele_miniRelIsoCorr, dR,  storage=hist.storage.Weight())
            histograms[f'dr_to_gen_vs_pt_{tag}']             = Hist(samp, cut, ele_pt,          dR,     storage=hist.storage.Weight())
            histograms[f'dr_to_gen_vs_eta_{tag}']            = Hist(samp, cut, ele_eta,         dR,     storage=hist.storage.Weight())

            # ── Suggested additional histograms ───────────────────────────────
            # Enable to study matching quality as a function of gen-level quantities
            # and track quality. These can identify regimes where the dR < 0.1
            # threshold is too loose or where confusion between candidates is high.

            # dR vs gen Lxy: does matching degrade for more displaced electrons?
            histograms[f'dr_to_gen_vs_genlxy_{tag}'] = Hist(samp, cut, ele_lxy_res, dR, storage=hist.storage.Weight())

            # dR vs gen pT: does matching degrade at low gen pT?
            histograms[f'dr_to_gen_vs_genpt_{tag}']  = Hist(samp, cut, ele_pt_res,  dR, storage=hist.storage.Weight())

            # dR vs reco track chi2: poorly fit tracks may be angularly offset
            histograms[f'dr_to_gen_vs_trkChi2_{tag}'] = Hist(samp, cut, ele_chi2,   dR, storage=hist.storage.Weight())

            # dR vs reco dxy: do displaced tracks have larger reco-gen angular offsets?
            histograms[f'dr_to_gen_vs_dxy_{tag}'] = Hist(samp, cut, ele_dxy,        dR, storage=hist.storage.Weight())

            # dR to nearest OTHER reco electron in the same collection:
            # identifies cases where a nearby clone could confuse the match
            histograms[f'dr_to_nearest_other_reco_{tag}'] = Hist(samp, cut, dR,     storage=hist.storage.Weight())

        # Gen-to-reco: for each gen particle, min dR to any reco electron in the collection.
        # Measures how well the collection covers each gen particle as a function of its kinematics.
        for genpart in ['ele', 'pos']:
            gtag = f'{coll}_{genpart}'
            histograms[f'dr_gen_to_reco_{gtag}']            = Hist(samp, cut, dR,          storage=hist.storage.Weight())
            histograms[f'dr_gen_to_reco_vs_genpt_{gtag}']   = Hist(samp, cut, ele_pt_res,  dR, storage=hist.storage.Weight())
            histograms[f'dr_gen_to_reco_vs_geneta_{gtag}']  = Hist(samp, cut, ele_eta_res, dR, storage=hist.storage.Weight())
            histograms[f'dr_gen_to_reco_vs_genlxy_{gtag}']  = Hist(samp, cut, ele_lxy_res, dR, storage=hist.storage.Weight())

        # Number of reco electrons within dR < 0.1 of the nearest gen particle:
        # directly measures matching ambiguity (how often is there > 1 candidate?)
        histograms[f'n_reco_near_gen_{coll}'] = Hist(samp, cut, neles, storage=hist.storage.Weight())
        # Flipped: number of gen particles (0, 1, or 2) within dR < 0.1 of any reco electron.
        histograms[f'n_gen_near_reco_{coll}'] = Hist(samp, cut, neles, storage=hist.storage.Weight())

    # 2D: gen dielectron dR vs reco dielectron dR.
    # Two versions filled independently so differences between vertex choices are visible.
    _dr_gen  = Regular(100, 0, 1,   name='dr_gen',  label=r'Gen $\Delta R(e^+e^-)$')
    _dr_reco = Regular(100, 0, 1,   name='dr_reco', label=r'Reco $\Delta R(e^+e^-)$')
    histograms['dr_gen_vs_dr_reco_ee_matched'] = Hist(samp, cut, _dr_gen, _dr_reco, storage=hist.storage.Weight())
    histograms['dr_gen_vs_dr_reco_ee_selvtx']  = Hist(samp, cut, _dr_gen, _dr_reco, storage=hist.storage.Weight())

    _pt_gen  = Regular(100, 0, 100, name='pt_gen',  label=r'Gen $p_T(e^+e^-)$ [GeV]')
    _pt_reco = Regular(100, 0, 100, name='pt_reco', label=r'Reco $p_T(e^+e^-)$ [GeV]')
    histograms['pt_gen_vs_pt_reco_ee_matched'] = Hist(samp, cut, _pt_gen, _pt_reco, storage=hist.storage.Weight())
    histograms['pt_gen_vs_pt_reco_ee_selvtx']  = Hist(samp, cut, _pt_gen, _pt_reco, storage=hist.storage.Weight())
            
    return histograms

subroutines = []

# GenEle/GenPos: scalar per event — eta, phi, pt, vxy, etc.
# gedlpt collection: GED + cross-cleaned LowPt concatenated, projected to common fields.
# alpt collection:   AllLowPt electrons (incl. x-cleaned); mindRj derived from dRJets.

def fillHistos(events, hists, samp, cut, info, sum_wgt=1):
    wgt = events.eventWgt / sum_wgt

    # AllLptElectron does not get mindRj in the standard subroutines; compute it here
    events['AllLptElectron', 'mindRj'] = ak.fill_none(
        ak.min(events.AllLptElectron.dRJets, axis=-1), 999
    )

    def min_dr_to_gen(eta, phi, gen_ele, gen_pos):
        """Min dR from a reco electron to either signal gen particle."""
        dphi_e = np.abs(phi - gen_ele.phi)
        dphi_e = ak.where(dphi_e > np.pi, 2*np.pi - dphi_e, dphi_e)
        dphi_p = np.abs(phi - gen_pos.phi)
        dphi_p = ak.where(dphi_p > np.pi, 2*np.pi - dphi_p, dphi_p)
        dr_e = np.sqrt((eta - gen_ele.eta)**2 + dphi_e**2)
        dr_p = np.sqrt((eta - gen_pos.eta)**2 + dphi_p**2)
        return np.minimum(dr_e, dr_p)

    def _project(coll):
        """Extract the common fields needed for fills into a plain record array."""
        return ak.zip({
            'pt':                coll.pt,
            'eta':               coll.eta,
            'phi':               coll.phi,
            'dxy':               coll.dxy,
            'PFRelIso':          coll.PFRelIso,
            'miniRelIsoEleCorr': coll.miniRelIsoEleCorr,
            'mindRj':            coll.mindRj,
            'trkChi2':           coll.trkChi2,
        })

    # Build the two analysis collections.
    # gedlpt: GED + cross-cleaned LowPt (mirrors the vtx_ electron pool).
    # alpt:   all LowPt including x-cleaned (mirrors the lptvtx_ pool).
    gedlpt_coll = ak.concatenate(
        [_project(events.Electron), _project(events.LptElectron)], axis=1
    )
    alpt_coll = _project(events.AllLptElectron)

    coll_map = {
        'gedlpt': gedlpt_coll,
        'alpt':   alpt_coll,
    }
    rank_map = {
        'lead':    0,
        'sublead': 1,
    }

    for coll_label, coll in coll_map.items():
        # Sort each event's electrons by pt descending once, reuse for both ranks
        sorted_idx  = ak.argsort(coll.pt, axis=1, ascending=False)
        sorted_coll = coll[sorted_idx]

        # n_reco_near_gen: property of the collection per event, not per rank.
        # Use events with at least one electron (the lead mask).
        has_any = ak.num(coll.pt) > 0
        if ak.sum(has_any) > 0:
            sel_any   = sorted_coll[has_any]
            w_any     = wgt[has_any]
            ge_any    = events.GenEle[has_any]
            gp_any    = events.GenPos[has_any]
            dphi_e_all = np.abs(sel_any.phi - ge_any.phi)
            dphi_e_all = ak.where(dphi_e_all > np.pi, 2*np.pi - dphi_e_all, dphi_e_all)
            dphi_p_all = np.abs(sel_any.phi - gp_any.phi)
            dphi_p_all = ak.where(dphi_p_all > np.pi, 2*np.pi - dphi_p_all, dphi_p_all)
            dr_all_to_ele = np.sqrt((sel_any.eta - ge_any.eta)**2 + dphi_e_all**2)
            dr_all_to_pos = np.sqrt((sel_any.eta - gp_any.eta)**2 + dphi_p_all**2)
            dr_all_to_gen = np.minimum(dr_all_to_ele, dr_all_to_pos)
            n_near = ak.sum(dr_all_to_gen < 0.1, axis=1)
            hists[f'n_reco_near_gen_{coll_label}'].fill(samp=samp, cut=cut, neles=n_near, weight=w_any)
            n_gen_near = (ak.any(dr_all_to_ele < 0.1, axis=1).to_numpy().astype(int)
                        + ak.any(dr_all_to_pos < 0.1, axis=1).to_numpy().astype(int))
            hists[f'n_gen_near_reco_{coll_label}'].fill(samp=samp, cut=cut, neles=n_gen_near, weight=w_any)

        for rank_label, rank in rank_map.items():
            tag = f'{coll_label}_{rank_label}'

            # Select events that have at least rank+1 electrons
            has_rank = ak.num(coll.pt) > rank
            if ak.sum(has_rank) == 0:
                continue

            sel     = sorted_coll[has_rank]
            w       = wgt[has_rank]
            gen_ele = events.GenEle[has_rank]
            gen_pos = events.GenPos[has_rank]

            ele = sel[:, rank]  # rank-th electron (0 = leading, 1 = subleading)

            dr_gen = min_dr_to_gen(ele.eta, ele.phi, gen_ele, gen_pos)

            hists[f'dr_to_gen_{tag}'].fill(samp=samp, cut=cut, dr=dr_gen, weight=w)
            hists[f'dr_to_gen_vs_dRj_{tag}'].fill(samp=samp, cut=cut, drj=ele.mindRj, dr=dr_gen, weight=w)
            hists[f'dr_to_gen_vs_PFRelIso_{tag}'].fill(samp=samp, cut=cut, relIso=ele.PFRelIso, dr=dr_gen, weight=w)
            hists[f'dr_to_gen_vs_miniRelIsoCorr_{tag}'].fill(samp=samp, cut=cut, iso=ele.miniRelIsoEleCorr, dr=dr_gen, weight=w)
            hists[f'dr_to_gen_vs_pt_{tag}'].fill(samp=samp, cut=cut, pt=ele.pt, dr=dr_gen, weight=w)
            hists[f'dr_to_gen_vs_eta_{tag}'].fill(samp=samp, cut=cut, eta=ele.eta, dr=dr_gen, weight=w)

            # ── Suggested fills ──
            gen_lxy = np.sqrt((gen_ele.vx - events.PV.x[has_rank])**2
                             +(gen_ele.vy - events.PV.y[has_rank])**2)
            gen_pt  = np.minimum(gen_ele.pt, gen_pos.pt)
            hists[f'dr_to_gen_vs_genlxy_{tag}'].fill(samp=samp, cut=cut, lxy=gen_lxy,       dr=dr_gen, weight=w)
            hists[f'dr_to_gen_vs_genpt_{tag}' ].fill(samp=samp, cut=cut, pt=gen_pt,         dr=dr_gen, weight=w)
            hists[f'dr_to_gen_vs_trkChi2_{tag}'].fill(samp=samp, cut=cut, chi2=ele.trkChi2, dr=dr_gen, weight=w)
            hists[f'dr_to_gen_vs_dxy_{tag}'    ].fill(samp=samp, cut=cut, dxy=ele.dxy,      dr=dr_gen, weight=w)

            # dr_to_nearest_other_reco: dR from the rank-th electron to the nearest
            # OTHER electron in the same collection. Exclude self by local index.
            local_idx = ak.local_index(sel.pt, axis=1)
            others    = sel[local_idx != rank]
            dphi_o    = np.abs(ele.phi - others.phi)
            dphi_o    = ak.where(dphi_o > np.pi, 2*np.pi - dphi_o, dphi_o)
            dr_others = np.sqrt((ele.eta - others.eta)**2 + dphi_o**2)
            dr_nearest_other = ak.fill_none(ak.min(dr_others, axis=1), 999.0)
            hists[f'dr_to_nearest_other_reco_{tag}'].fill(samp=samp, cut=cut, dr=dr_nearest_other, weight=w)

    # ── Gen-to-reco: min dR from each gen particle to any reco electron ───────
    # Complements the reco-to-gen fills above: here we ask, for each gen particle,
    # how close is the nearest reco electron in each collection?
    gen_lxy_all = np.sqrt(
        (events.GenEle.vx - events.PV.x)**2 +
        (events.GenEle.vy - events.PV.y)**2
    )
    gen_parts = {'ele': events.GenEle, 'pos': events.GenPos}

    for coll_label, coll in coll_map.items():
        has_reco = ak.num(coll.pt) > 0
        if ak.sum(has_reco) == 0:
            continue
        reco_sel = coll[has_reco]
        w_reco   = wgt[has_reco]
        lxy_sel  = gen_lxy_all[has_reco]

        for genpart_label, gen_part_full in gen_parts.items():
            gtag    = f'{coll_label}_{genpart_label}'
            gen_sel = gen_part_full[has_reco]
            dphi_g  = np.abs(reco_sel.phi - gen_sel.phi)
            dphi_g  = ak.where(dphi_g > np.pi, 2*np.pi - dphi_g, dphi_g)
            dr_g    = np.sqrt((reco_sel.eta - gen_sel.eta)**2 + dphi_g**2)
            dr_min  = ak.fill_none(ak.min(dr_g, axis=1), 999.0)
            hists[f'dr_gen_to_reco_{gtag}'           ].fill(samp=samp, cut=cut, dr=dr_min,         weight=w_reco)
            hists[f'dr_gen_to_reco_vs_genpt_{gtag}'  ].fill(samp=samp, cut=cut, pt=gen_sel.pt,     dr=dr_min, weight=w_reco)
            hists[f'dr_gen_to_reco_vs_geneta_{gtag}' ].fill(samp=samp, cut=cut, eta=gen_sel.eta,   dr=dr_min, weight=w_reco)
            hists[f'dr_gen_to_reco_vs_genlxy_{gtag}' ].fill(samp=samp, cut=cut, lxy=lxy_sel,       dr=dr_min, weight=w_reco)

    # ── Gen vs reco dielectron dR and pT ──────────────────────────────────────
    matched_vtxs = events.vtx[events.vtx.isMatched]
    has_matched  = ak.num(matched_vtxs) > 0

    if ak.sum(has_matched) > 0:
        m        = has_matched
        dr_match = ak.fill_none(ak.firsts(matched_vtxs.refit_dR), -1.0)
        pt_match = ak.fill_none(ak.firsts(matched_vtxs.refit_pt), -1.0)
        hists['dr_gen_vs_dr_reco_ee_matched'].fill(
            samp=samp, cut=cut,
            dr_gen=events.genEE.dr[m], dr_reco=dr_match[m], weight=wgt[m],
        )
        hists['pt_gen_vs_pt_reco_ee_matched'].fill(
            samp=samp, cut=cut,
            pt_gen=events.genEE.pt[m], pt_reco=pt_match[m], weight=wgt[m],
        )

    if 'sel_vtx' in events.fields:
        has_sel = ~ak.is_none(events.sel_vtx.refit_dR)
        if ak.sum(has_sel) > 0:
            s = has_sel
            hists['dr_gen_vs_dr_reco_ee_selvtx'].fill(
                samp=samp, cut=cut,
                dr_gen=events.genEE.dr[s],
                dr_reco=ak.fill_none(events.sel_vtx.refit_dR[s], 0.0),
                weight=wgt[s],
            )
            hists['pt_gen_vs_pt_reco_ee_selvtx'].fill(
                samp=samp, cut=cut,
                pt_gen=events.genEE.pt[s],
                pt_reco=ak.fill_none(events.sel_vtx.refit_pt[s], 0.0),
                weight=wgt[s],
            )
