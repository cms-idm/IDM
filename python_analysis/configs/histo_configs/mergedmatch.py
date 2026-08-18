try:
    from histobins import *
except ModuleNotFoundError:
    from configs.histo_configs.histobins import *
from hist import Hist
from hist.axis import Variable, Regular, StrCategory
import hist
import numpy as np
import awkward as ak

# Quality match: dR < 0.1, charge agrees with gen, and |pt_reco - pt_gen|/pt_gen < 0.2.
#
# Two event topologies, selected per collection:
#
#   merged: one reco quality-matched to one gen (dR + charge + pt); the other gen
#           is within dR < 0.1 of the same reco but is NOT a full quality match
#           (fails charge or pt). The primary pt check accepts either the individual
#           gen pt or the gen system sum pt (pt_ele + pt_pos) within 20%.
#     full_match_ele_i = dR(reco_i, GenEle) < 0.1  AND  charge == -1  AND  pt within 20%
#     full_match_pos_i = dR(reco_i, GenPos) < 0.1  AND  charge == +1  AND  pt within 20%
#     merged_i  = both gens within dR < 0.1, reco quality-matched to one, NOT the other
#     keep iff: any(merged_i) AND sum(matched_i) == 1
#
#   resolved (classical): one reco quality-matched exclusively to GenEle, a different one
#        exclusively to GenPos; no reco is merged.
#     keep iff: any(only_ele_i) AND any(only_pos_i) AND NOT any(merged_i)
#
#   resolved (ambiguous): a merged reco exists but a SECOND reco is also quality-matched
#        to at least one gen particle. The presence of a second matched reco indicates the
#        event is better described as resolved at the reco level.
#     keep iff: any(merged_i) AND sum(matched_i) > 1

# Axis for reco-gen dR: by construction both are < 0.1, so zoom in past that threshold.
# gen-gen dR uses ee_dr_narrow (0-0.2) from histobins since it can reach up to ~0.2.
_dr_reco_gen = Regular(50, 0, 0.15, name='dr', label=r'$\Delta R$')

# Axes for reco-gen pt difference (merged case).
# The merged reco pt ≈ sum of both gen pts, so the difference relative to either
# individual gen particle can be large (especially for the subleading).
# Variable-width bins: fine (0.25 GeV) in [-2, 2] and coarse outside to keep
# the region around 0 well-resolved without bloating the histogram size.
_dpt_abs_edges = np.concatenate([
    np.linspace(-10,  -2,  9)[:-1],  # 8 bins × 1.0 GeV
    np.linspace( -2,   2, 17)[:-1],  # 16 bins × 0.25 GeV  (dense near 0)
    np.linspace(  2,  10,  9)[:-1],  # 8 bins × 1.0 GeV
    np.linspace( 10,  40,  7),       # 6 bins × 5.0 GeV
])
_dpt_abs = Variable(_dpt_abs_edges, name='dpt_abs', label=r'$p_T^{\rm reco} - p_T^{\rm gen}$ [GeV]')

# Variable-width bins: fine (0.05) in [-0.5, 0.5] and coarser outside.
_dpt_rel_edges = np.concatenate([
    np.linspace(-2.0, -0.5,  7)[:-1],  # 6 bins × 0.25
    np.linspace(-0.5,  0.5, 21)[:-1],  # 20 bins × 0.05  (dense near 0)
    np.linspace( 0.5,  2.0,  7)[:-1],  # 6 bins × 0.25
    np.linspace( 2.0,  8.0,  7),       # 6 bins × 1.0
])
_dpt_rel = Variable(_dpt_rel_edges, name='dpt_rel', label=r'$(p_T^{\rm reco} - p_T^{\rm gen})/p_T^{\rm gen}$')

# Temporary switch: the dpt_* histograms (1D + 2D vs genpt/eedr, merged +
# resolved) make up the bulk of the histogram memory/merge footprint.
# Set to True to restore them.
FILL_DPT_HISTS = False

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
        if FILL_DPT_HISTS:
            # reco-gen pt difference: 1D (merged only)
            histograms[f'dpt_abs_lead_{coll}']    = Hist(samp, cut, _dpt_abs, storage=hist.storage.Weight())
            histograms[f'dpt_abs_sublead_{coll}'] = Hist(samp, cut, _dpt_abs, storage=hist.storage.Weight())
            histograms[f'dpt_rel_lead_{coll}']    = Hist(samp, cut, _dpt_rel, storage=hist.storage.Weight())
            histograms[f'dpt_rel_sublead_{coll}'] = Hist(samp, cut, _dpt_rel, storage=hist.storage.Weight())
            # reco-gen pt difference vs gen ele pt (2D)
            histograms[f'dpt_abs_lead_vs_genpt_{coll}']    = Hist(samp, cut, ele_pt_res, _dpt_abs, storage=hist.storage.Weight())
            histograms[f'dpt_abs_sublead_vs_genpt_{coll}'] = Hist(samp, cut, ele_pt_res, _dpt_abs, storage=hist.storage.Weight())
            histograms[f'dpt_rel_lead_vs_genpt_{coll}']    = Hist(samp, cut, ele_pt_res, _dpt_rel, storage=hist.storage.Weight())
            histograms[f'dpt_rel_sublead_vs_genpt_{coll}'] = Hist(samp, cut, ele_pt_res, _dpt_rel, storage=hist.storage.Weight())
            # reco-gen pt difference vs gen ee dR (2D)
            histograms[f'dpt_abs_lead_vs_eedr_{coll}']    = Hist(samp, cut, ee_dr_narrow, _dpt_abs, storage=hist.storage.Weight())
            histograms[f'dpt_abs_sublead_vs_eedr_{coll}'] = Hist(samp, cut, ee_dr_narrow, _dpt_abs, storage=hist.storage.Weight())
            histograms[f'dpt_rel_lead_vs_eedr_{coll}']    = Hist(samp, cut, ee_dr_narrow, _dpt_rel, storage=hist.storage.Weight())
            histograms[f'dpt_rel_sublead_vs_eedr_{coll}'] = Hist(samp, cut, ee_dr_narrow, _dpt_rel, storage=hist.storage.Weight())

            # ── resolved dpt hists (classical sub-type: clean one-to-one assignment) ─
            # 1D
            histograms[f'dpt_abs_lead_resolved_{coll}']    = Hist(samp, cut, _dpt_abs, storage=hist.storage.Weight())
            histograms[f'dpt_abs_sublead_resolved_{coll}'] = Hist(samp, cut, _dpt_abs, storage=hist.storage.Weight())
            histograms[f'dpt_rel_lead_resolved_{coll}']    = Hist(samp, cut, _dpt_rel, storage=hist.storage.Weight())
            histograms[f'dpt_rel_sublead_resolved_{coll}'] = Hist(samp, cut, _dpt_rel, storage=hist.storage.Weight())
            # vs gen ele pt (2D)
            histograms[f'dpt_abs_lead_resolved_vs_genpt_{coll}']    = Hist(samp, cut, ele_pt_res, _dpt_abs, storage=hist.storage.Weight())
            histograms[f'dpt_abs_sublead_resolved_vs_genpt_{coll}'] = Hist(samp, cut, ele_pt_res, _dpt_abs, storage=hist.storage.Weight())
            histograms[f'dpt_rel_lead_resolved_vs_genpt_{coll}']    = Hist(samp, cut, ele_pt_res, _dpt_rel, storage=hist.storage.Weight())
            histograms[f'dpt_rel_sublead_resolved_vs_genpt_{coll}'] = Hist(samp, cut, ele_pt_res, _dpt_rel, storage=hist.storage.Weight())
            # vs gen ee dR (2D) — use full ee_dr range since resolved dR > 0.1
            histograms[f'dpt_abs_lead_resolved_vs_eedr_{coll}']    = Hist(samp, cut, ee_dr, _dpt_abs, storage=hist.storage.Weight())
            histograms[f'dpt_abs_sublead_resolved_vs_eedr_{coll}'] = Hist(samp, cut, ee_dr, _dpt_abs, storage=hist.storage.Weight())
            histograms[f'dpt_rel_lead_resolved_vs_eedr_{coll}']    = Hist(samp, cut, ee_dr, _dpt_rel, storage=hist.storage.Weight())
            histograms[f'dpt_rel_sublead_resolved_vs_eedr_{coll}'] = Hist(samp, cut, ee_dr, _dpt_rel, storage=hist.storage.Weight())

        # ── merged/resolved comparison hists ─────────────────────────────────
        for scenario in ['merged', 'resolved']:
            s = f'{scenario}_{coll}'
            histograms[f'gen_lxy_{s}']       = Hist(samp, cut, ele_lxy_res,          storage=hist.storage.Weight())
            histograms[f'gen_ee_dr_{s}']     = Hist(samp, cut, ee_dr,                storage=hist.storage.Weight())
            histograms[f'gen_ee_pt_{s}']     = Hist(samp, cut, ele_pt,               storage=hist.storage.Weight())
            histograms[f'reco_trkchi2_{s}']  = Hist(samp, cut, ele_chi2,             storage=hist.storage.Weight())
            histograms[f'dr_gen_vs_lxy_{s}'] = Hist(samp, cut, ele_lxy_res, ee_dr,   storage=hist.storage.Weight())

        # ── gen ee kinematics per reco category ──────────────────────────────
        # Categories: zero_matched, merged (ele+photrk combined), resolved.
        _bycat = StrCategory(['zero_matched', 'merged', 'resolved'], name='cat', label='Reco category')
        histograms[f'gen_ee_dr_cat_{coll}']  = Hist(samp, cut, _bycat, ee_dr,   storage=hist.storage.Weight())
        histograms[f'gen_ee_pt_cat_{coll}']  = Hist(samp, cut, _bycat, ele_pt,  storage=hist.storage.Weight())
        histograms[f'gen_ee_eta_cat_{coll}'] = Hist(samp, cut, _bycat, ele_eta, storage=hist.storage.Weight())

        # ── reco category hists ───────────────────────────────────────────────
        # Categories (orthogonal, exhaustive):
        #   zero_matched   : neither gen matched to any reco object
        #   one_ele        : one gen matched to a reco electron; other unmatched
        #   one_photrk     : one gen matched to photon/track/losttrack/pfcand only; other unmatched
        #   merged         : both gens matched to the same reco electron
        #   merged_photrk  : both gens matched to the same photon/track/losttrack/pfcand (nothing else nearby)
        #   ele_photrk     : one gen matched to reco electron, other to photon/track/losttrack/pfcand
        #   resolved       : each gen matched to a different reco electron
        #   two_photrk     : each gen matched to a different photon/track/losttrack/pfcand (no reco electrons)
        # Electron match takes priority over photon/track/losttrack/pfcand match.
        # When more than one non-electron type is present for a single split, the
        # type label is chosen by priority: photon > track > conversion > ootphoton
        # > pfcand > losttrack.
        _cat_labels = [
            'zero_matched', 'one_ele',
            'one_photon', 'one_track', 'one_conversion', 'one_ootphoton', 'one_pfcand', 'one_losttrack',
            'merged',
            'merged_photon', 'merged_track', 'merged_conversion', 'merged_ootphoton', 'merged_pfcand', 'merged_losttrack',
            'ele_photon', 'ele_track', 'ele_conversion', 'ele_ootphoton', 'ele_pfcand', 'ele_losttrack',
            'resolved',
            'two_photon', 'two_track', 'two_conversion', 'two_ootphoton', 'two_pfcand', 'two_losttrack',
        ]
        histograms[f'reco_cat_{coll}'] = Hist(
            samp, cut,
            StrCategory(_cat_labels, name='cat', label='Reco category'),
            storage=hist.storage.Weight(),
        )
    return histograms

subroutines = []

def fillHistos(events, hists, samp, cut, info, sum_wgt=1):
    wgt = events.eventWgt / sum_wgt

    def _project(coll):
        return ak.zip({
            'pt':                coll.pt,
            'eta':               coll.eta,
            'phi':               coll.phi,
            'charge':            coll.charge,
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

    # gen_lxy is measured from the chi2 production vertex (the true, unsmeared
    # primary vertex) rather than the reconstructed PV, which carries ~10-15um
    # of resolution/bias that would otherwise leak into a "truth" quantity.
    chi2 = ak.firsts(events.GenPart[np.abs(events.GenPart.ID) == 1000023])
    gen_lxy = np.sqrt(
        (events.GenEle.vx - chi2.vx)**2 +
        (events.GenEle.vy - chi2.vy)**2
    )
    gen_sum_pt = events.GenEle.pt + events.GenPos.pt

    # Photon/track dR to gen particles — computed once, reused per collection.
    def _dphi_gen(a, b):
        d = np.abs(a - b)
        return ak.where(d > np.pi, 2 * np.pi - d, d)

    def _dr_to_gen(obj_eta, obj_phi, gen_eta, gen_phi):
        dp = _dphi_gen(obj_phi, gen_phi)
        return np.sqrt((obj_eta - gen_eta)**2 + dp**2)

    _pho_dr_genele    = _dr_to_gen(events.Photon.eta,      events.Photon.phi,      events.GenEle.eta, events.GenEle.phi)
    _pho_dr_genpos    = _dr_to_gen(events.Photon.eta,      events.Photon.phi,      events.GenPos.eta, events.GenPos.phi)
    _ootpho_dr_genele = _dr_to_gen(events.ootPhoton.eta,   events.ootPhoton.phi,   events.GenEle.eta, events.GenEle.phi)
    _ootpho_dr_genpos = _dr_to_gen(events.ootPhoton.eta,   events.ootPhoton.phi,   events.GenPos.eta, events.GenPos.phi)
    _trk_dr_genele    = _dr_to_gen(events.IsoTrack.eta,    events.IsoTrack.phi,    events.GenEle.eta, events.GenEle.phi)
    _trk_dr_genpos    = _dr_to_gen(events.IsoTrack.eta,    events.IsoTrack.phi,    events.GenPos.eta, events.GenPos.phi)
    _conv_dr_genele   = _dr_to_gen(events.Conversion.eta,  events.Conversion.phi,  events.GenEle.eta, events.GenEle.phi)
    _conv_dr_genpos   = _dr_to_gen(events.Conversion.eta,  events.Conversion.phi,  events.GenPos.eta, events.GenPos.phi)
    _pf_dr_genele     = _dr_to_gen(events.PFCand.eta,      events.PFCand.phi,      events.GenEle.eta, events.GenEle.phi)
    _pf_dr_genpos     = _dr_to_gen(events.PFCand.eta,      events.PFCand.phi,      events.GenPos.eta, events.GenPos.phi)
    _lost_dr_genele   = _dr_to_gen(events.LostTrack.eta,   events.LostTrack.phi,   events.GenEle.eta, events.GenEle.phi)
    _lost_dr_genpos   = _dr_to_gen(events.LostTrack.eta,   events.LostTrack.phi,   events.GenPos.eta, events.GenPos.phi)

    # 20% relative pt matching. For merged objects (both gens close) we also accept
    # a match to the gen system pt (pt_obj ≈ pt_ele + pt_pos).
    def _pt_rel(obj_pt, gen_pt):
        return np.abs(obj_pt - gen_pt) / gen_pt < 0.2

    _pho_pt_genele    = _pt_rel(events.Photon.pt,     events.GenEle.pt)
    _pho_pt_genpos    = _pt_rel(events.Photon.pt,     events.GenPos.pt)
    _pho_pt_sum       = _pt_rel(events.Photon.pt,     gen_sum_pt)
    _ootpho_pt_genele = _pt_rel(events.ootPhoton.pt,  events.GenEle.pt)
    _ootpho_pt_genpos = _pt_rel(events.ootPhoton.pt,  events.GenPos.pt)
    _ootpho_pt_sum    = _pt_rel(events.ootPhoton.pt,  gen_sum_pt)
    _trk_pt_genele    = _pt_rel(events.IsoTrack.pt,   events.GenEle.pt)
    _trk_pt_genpos    = _pt_rel(events.IsoTrack.pt,   events.GenPos.pt)
    _trk_pt_sum       = _pt_rel(events.IsoTrack.pt,   gen_sum_pt)
    _conv_pt_genele   = _pt_rel(events.Conversion.pt, events.GenEle.pt)
    _conv_pt_genpos   = _pt_rel(events.Conversion.pt, events.GenPos.pt)
    _conv_pt_sum      = _pt_rel(events.Conversion.pt, gen_sum_pt)
    _pf_pt_genele     = _pt_rel(events.PFCand.pt,     events.GenEle.pt)
    _pf_pt_genpos     = _pt_rel(events.PFCand.pt,     events.GenPos.pt)
    _pf_pt_sum        = _pt_rel(events.PFCand.pt,     gen_sum_pt)
    _lost_pt_genele   = _pt_rel(events.LostTrack.pt,  events.GenEle.pt)
    _lost_pt_genpos   = _pt_rel(events.LostTrack.pt,  events.GenPos.pt)
    _lost_pt_sum      = _pt_rel(events.LostTrack.pt,  gen_sum_pt)

    # True if GenEle/GenPos has a quality match (dR < 0.1, pt within 20%) to any
    # photon, OOT photon, track/PFCand/LostTrack (charge-constrained), or conversion.
    _genele_pho    = ak.any((_pho_dr_genele    < 0.1) & _pho_pt_genele,                                   axis=1)
    _genele_ootpho = ak.any((_ootpho_dr_genele < 0.1) & _ootpho_pt_genele,                                axis=1)
    _genele_trk    = ak.any((_trk_dr_genele    < 0.1) & (events.IsoTrack.charge == -1) & _trk_pt_genele,  axis=1)
    _genele_conv   = ak.any((_conv_dr_genele   < 0.1) & _conv_pt_genele,                                  axis=1)
    _genele_pf     = ak.any((_pf_dr_genele     < 0.1) & (events.PFCand.charge   == -1) & _pf_pt_genele,   axis=1)
    _genele_lost   = ak.any((_lost_dr_genele   < 0.1) & (events.LostTrack.charge == -1) & _lost_pt_genele, axis=1)
    _genpos_pho    = ak.any((_pho_dr_genpos    < 0.1) & _pho_pt_genpos,                                   axis=1)
    _genpos_ootpho = ak.any((_ootpho_dr_genpos < 0.1) & _ootpho_pt_genpos,                                axis=1)
    _genpos_trk    = ak.any((_trk_dr_genpos    < 0.1) & (events.IsoTrack.charge == +1) & _trk_pt_genpos,  axis=1)
    _genpos_conv   = ak.any((_conv_dr_genpos   < 0.1) & _conv_pt_genpos,                                  axis=1)
    _genpos_pf     = ak.any((_pf_dr_genpos     < 0.1) & (events.PFCand.charge   == +1) & _pf_pt_genpos,   axis=1)
    _genpos_lost   = ak.any((_lost_dr_genpos   < 0.1) & (events.LostTrack.charge == +1) & _lost_pt_genpos, axis=1)

    _genele_any_photrk = _genele_pho | _genele_ootpho | _genele_trk | _genele_conv | _genele_pf | _genele_lost
    _genpos_any_photrk = _genpos_pho | _genpos_ootpho | _genpos_trk | _genpos_conv | _genpos_pf | _genpos_lost

    # ── merged-photrk detection ───────────────────────────────────────────────
    # Find photons/oot-photons/tracks within dR < 0.1 of BOTH gen particles
    # simultaneously and passing a pt quality check. The pt check requires
    # the object to match EITHER individual gen OR the gen system sum pt
    # (pt_ele + pt_pos). Track charge is NOT constrained here (analogous to
    # the merged-electron case).
    _pho_both    = ((_pho_dr_genele    < 0.1) & (_pho_dr_genpos    < 0.1) & (_pho_pt_genele    | _pho_pt_genpos    | _pho_pt_sum   ))
    _ootpho_both = ((_ootpho_dr_genele < 0.1) & (_ootpho_dr_genpos < 0.1) & (_ootpho_pt_genele | _ootpho_pt_genpos | _ootpho_pt_sum))
    _trk_both    = ((_trk_dr_genele    < 0.1) & (_trk_dr_genpos    < 0.1) & (_trk_pt_genele    | _trk_pt_genpos    | _trk_pt_sum   ))
    _conv_both   = ((_conv_dr_genele   < 0.1) & (_conv_dr_genpos   < 0.1) & (_conv_pt_genele   | _conv_pt_genpos   | _conv_pt_sum  ))
    _pf_both     = ((_pf_dr_genele     < 0.1) & (_pf_dr_genpos     < 0.1) & (_pf_pt_genele     | _pf_pt_genpos     | _pf_pt_sum    ))
    _lost_both   = ((_lost_dr_genele   < 0.1) & (_lost_dr_genpos   < 0.1) & (_lost_pt_genele   | _lost_pt_genpos   | _lost_pt_sum  ))

    _pho_both_any    = ak.any(_pho_both,    axis=1)
    _ootpho_both_any = ak.any(_ootpho_both, axis=1)
    _trk_both_any    = ak.any(_trk_both,    axis=1)
    _conv_both_any   = ak.any(_conv_both,   axis=1)
    _pf_both_any     = ak.any(_pf_both,     axis=1)
    _lost_both_any   = ak.any(_lost_both,   axis=1)
    _has_photrk_both = _pho_both_any | _ootpho_both_any | _trk_both_any | _conv_both_any | _pf_both_any | _lost_both_any

    # Exclusivity: every object quality-matched (dR < 0.1, pt within 20% of that
    # gen OR the gen sum) to EITHER gen must also quality-match BOTH gens. This
    # ensures no object is exclusively assigned to just one gen.
    _photrk_excl = (
        (ak.sum((_pho_dr_genele    < 0.1) & (_pho_pt_genele    | _pho_pt_sum   ), axis=1) == ak.sum(_pho_both,    axis=1)) &
        (ak.sum((_pho_dr_genpos    < 0.1) & (_pho_pt_genpos    | _pho_pt_sum   ), axis=1) == ak.sum(_pho_both,    axis=1)) &
        (ak.sum((_ootpho_dr_genele < 0.1) & (_ootpho_pt_genele | _ootpho_pt_sum), axis=1) == ak.sum(_ootpho_both, axis=1)) &
        (ak.sum((_ootpho_dr_genpos < 0.1) & (_ootpho_pt_genpos | _ootpho_pt_sum), axis=1) == ak.sum(_ootpho_both, axis=1)) &
        (ak.sum((_trk_dr_genele    < 0.1) & (_trk_pt_genele    | _trk_pt_sum   ), axis=1) == ak.sum(_trk_both,    axis=1)) &
        (ak.sum((_trk_dr_genpos    < 0.1) & (_trk_pt_genpos    | _trk_pt_sum   ), axis=1) == ak.sum(_trk_both,    axis=1)) &
        (ak.sum((_conv_dr_genele   < 0.1) & (_conv_pt_genele   | _conv_pt_sum  ), axis=1) == ak.sum(_conv_both,   axis=1)) &
        (ak.sum((_conv_dr_genpos   < 0.1) & (_conv_pt_genpos   | _conv_pt_sum  ), axis=1) == ak.sum(_conv_both,   axis=1)) &
        (ak.sum((_pf_dr_genele     < 0.1) & (_pf_pt_genele     | _pf_pt_sum    ), axis=1) == ak.sum(_pf_both,     axis=1)) &
        (ak.sum((_pf_dr_genpos     < 0.1) & (_pf_pt_genpos     | _pf_pt_sum    ), axis=1) == ak.sum(_pf_both,     axis=1)) &
        (ak.sum((_lost_dr_genele   < 0.1) & (_lost_pt_genele   | _lost_pt_sum  ), axis=1) == ak.sum(_lost_both,   axis=1)) &
        (ak.sum((_lost_dr_genpos   < 0.1) & (_lost_pt_genpos   | _lost_pt_sum  ), axis=1) == ak.sum(_lost_both,   axis=1))
    )

    for coll_label, coll in coll_map.items():

        def _dphi(a, b):
            d = np.abs(a - b)
            return ak.where(d > np.pi, 2 * np.pi - d, d)

        dphi_e = _dphi(coll.phi, events.GenEle.phi)
        dphi_p = _dphi(coll.phi, events.GenPos.phi)
        dr_to_ele = np.sqrt((coll.eta - events.GenEle.eta)**2 + dphi_e**2)
        dr_to_pos = np.sqrt((coll.eta - events.GenPos.eta)**2 + dphi_p**2)

        # Full quality match: dR < 0.1, charge agreement, and pt within 20%.
        pt_match_ele = _pt_rel(coll.pt, events.GenEle.pt)
        pt_match_pos = _pt_rel(coll.pt, events.GenPos.pt)
        pt_match_sum = _pt_rel(coll.pt, gen_sum_pt)

        dr_match_ele = (dr_to_ele < 0.1) & (coll.charge == -1) & pt_match_ele
        dr_match_pos = (dr_to_pos < 0.1) & (coll.charge == +1) & pt_match_pos

        # Merged: a single reco within dR < 0.1 of both gen particles, quality-matched
        # to one (dR + charge + pt), while the other gen is NOT a full quality match
        # (fails the dR+charge+pt triple). The pt check for the primary gen accepts
        # either the individual gen pt OR the gen system sum pt (pt_ele + pt_pos).
        # The other gen is denied a photon/track/conversion alternative.
        pt_ok_as_ele = pt_match_ele | pt_match_sum
        pt_ok_as_pos = pt_match_pos | pt_match_sum

        merged_i = (dr_to_ele < 0.1) & (dr_to_pos < 0.1) & (
            ((coll.charge == -1) & pt_ok_as_ele & ~dr_match_pos & ~_genpos_any_photrk) |
            ((coll.charge == +1) & pt_ok_as_pos & ~dr_match_ele & ~_genele_any_photrk)
        )

        # Exclusive charge-matched pairs, excluding recos already in merged_i.
        only_ele_i = dr_match_ele & ~merged_i
        only_pos_i = dr_match_pos & ~merged_i
        matched_i  = merged_i | dr_match_ele | dr_match_pos

        has_merged = ak.any(merged_i,  axis=1)
        n_matched  = ak.sum(matched_i, axis=1)

        # ── reco category fill ────────────────────────────────────────────────
        # Exclusive-assignment matching: gen particles compete for reco objects.
        # Electron matches take priority over photon/track, but a reco electron
        # already claimed by one gen cannot force "merged" if the second gen has
        # an alternative (photon/track or a different reco electron).
        #
        # Key flags:
        #   A = exclusive reco electron for GenEle (dR<0.1 to GenEle, not GenPos)
        #   B = exclusive reco electron for GenPos (dR<0.1 to GenPos, not GenEle)
        #   M = has_merged (reco electron dR<0.1 to both simultaneously)
        #   P_e / P_p = photon or track within dR<0.1 of GenEle / GenPos
        #
        # Sep: both gens can be assigned DIFFERENT reco electrons.
        #   Classical: each gen has an exclusive reco electron (no shared one).
        #   Ambiguous: there is a merged reco electron but n_matched > 1, so a second
        #              reco electron exists and can be given to the other gen.
        #   (Two fully-merged reco electrons → A=F, B=F, M=T, n>1 → sep_ambiguous.)
        # Merged: only one reco electron matched and it is shared (n_matched==1, M=T),
        #         and neither gen has a photrk escape.
        _A  = ak.any(only_ele_i, axis=1)
        _B  = ak.any(only_pos_i, axis=1)
        _Pe = _genele_any_photrk
        _Pp = _genpos_any_photrk

        _cat_sep    = (_A & _B & ~has_merged) | (has_merged & (n_matched > 1))
        # Under ~_cat_sep: if has_merged then n_matched==1 (single shared reco, no exclusive)
        _cat_merged = ~_cat_sep & has_merged & ~_Pe & ~_Pp
        _cat_ele_pt = ~_cat_sep & (
                          (_A & ~has_merged & _Pp) |        # GenEle exclusive reco, GenPos photrk
                          (_B & ~has_merged & _Pe) |        # GenPos exclusive reco, GenEle photrk
                          (has_merged & (_Pe | _Pp))        # only shared reco; photrk frees second gen
                      )
        _cat_one_ele = ~_cat_sep & ((_A & ~has_merged & ~_Pp) | (_B & ~has_merged & ~_Pe))
        _no_ele      = ~_cat_sep & ~_A & ~_B & ~has_merged
        _cat_two_pt  = _no_ele & _Pe & _Pp
        _cat_one_pt  = _no_ele & ((_Pe & ~_Pp) | (~_Pe & _Pp))
        _cat_zero    = _no_ele & ~_Pe & ~_Pp

        # merged_photrk: no reco electron matches either gen (_no_ele); a single
        # photon/track/losttrack/pfcand is within dR < 0.1 of BOTH gen particles;
        # and no other photon/track/losttrack/pfcand is exclusively close to just
        # one gen.
        # Overlaps with one_photrk (merged track sets only _Pe or only _Pp due to
        # charge-constrained _genXXX_any_photrk) and two_photrk (merged photon sets
        # both), so subtract it from those masks below.
        _cat_merged_photrk = _no_ele & _has_photrk_both & _photrk_excl

        # ── split photrk categories by object type ──────────────────────────
        # Priority (used only to pick one label when multiple types apply):
        #   photon > track > conversion > ootphoton > pfcand > losttrack

        # merged_photrk split: type of the object that matches both gen particles
        _cat_merged_photon     = _no_ele & _pho_both_any & _photrk_excl
        _cat_merged_track      = _no_ele & ~_pho_both_any & _trk_both_any & _photrk_excl
        _cat_merged_conversion = _no_ele & ~_pho_both_any & ~_trk_both_any & _conv_both_any & _photrk_excl
        _cat_merged_ootphoton  = _no_ele & ~_pho_both_any & ~_trk_both_any & ~_conv_both_any & _ootpho_both_any & _photrk_excl
        _cat_merged_pfcand     = _no_ele & ~_pho_both_any & ~_trk_both_any & ~_conv_both_any & ~_ootpho_both_any & _pf_both_any & _photrk_excl
        _cat_merged_losttrack  = _no_ele & ~_pho_both_any & ~_trk_both_any & ~_conv_both_any & ~_ootpho_both_any & ~_pf_both_any & _lost_both_any & _photrk_excl

        # one_photrk split: type of the single photrk-matched gen
        _one_pt_base = _cat_one_pt & ~_cat_merged_photrk
        _cat_one_photon    = _one_pt_base & (
            (_Pe & ~_Pp & _genele_pho) | (~_Pe & _Pp & _genpos_pho))
        _cat_one_track     = _one_pt_base & (
            (_Pe & ~_Pp & ~_genele_pho & _genele_trk) |
            (~_Pe & _Pp & ~_genpos_pho & _genpos_trk))
        _cat_one_conversion = _one_pt_base & (
            (_Pe & ~_Pp & ~_genele_pho & ~_genele_trk & _genele_conv) |
            (~_Pe & _Pp & ~_genpos_pho & ~_genpos_trk & _genpos_conv))
        _cat_one_ootphoton = _one_pt_base & (
            (_Pe & ~_Pp & ~_genele_pho & ~_genele_trk & ~_genele_conv & _genele_ootpho) |
            (~_Pe & _Pp & ~_genpos_pho & ~_genpos_trk & ~_genpos_conv & _genpos_ootpho))
        _cat_one_pfcand = _one_pt_base & (
            (_Pe & ~_Pp & ~_genele_pho & ~_genele_trk & ~_genele_conv & ~_genele_ootpho & _genele_pf) |
            (~_Pe & _Pp & ~_genpos_pho & ~_genpos_trk & ~_genpos_conv & ~_genpos_ootpho & _genpos_pf))
        _cat_one_losttrack = _one_pt_base & (
            (_Pe & ~_Pp & ~_genele_pho & ~_genele_trk & ~_genele_conv & ~_genele_ootpho & ~_genele_pf & _genele_lost) |
            (~_Pe & _Pp & ~_genpos_pho & ~_genpos_trk & ~_genpos_conv & ~_genpos_ootpho & ~_genpos_pf & _genpos_lost))

        # ele_photrk split: highest-priority type across all photrk-matched gen particles
        _elept_pho    = _genele_pho    | _genpos_pho
        _elept_trk    = ~_elept_pho & (_genele_trk | _genpos_trk)
        _elept_conv   = ~_elept_pho & ~_elept_trk & (_genele_conv | _genpos_conv)
        _elept_ootpho = ~_elept_pho & ~_elept_trk & ~_elept_conv & (_genele_ootpho | _genpos_ootpho)
        _elept_pf     = ~_elept_pho & ~_elept_trk & ~_elept_conv & ~_elept_ootpho & (_genele_pf | _genpos_pf)
        _cat_ele_photon     = _cat_ele_pt & _elept_pho
        _cat_ele_track      = _cat_ele_pt & _elept_trk
        _cat_ele_conversion = _cat_ele_pt & _elept_conv
        _cat_ele_ootphoton  = _cat_ele_pt & _elept_ootpho
        _cat_ele_pfcand     = _cat_ele_pt & _elept_pf
        _cat_ele_losttrack  = _cat_ele_pt & ~_elept_pho & ~_elept_trk & ~_elept_conv & ~_elept_ootpho & ~_elept_pf

        # two_photrk split: highest-priority type across both photrk-matched gen particles
        _two_pt_base = _cat_two_pt & ~_cat_merged_photrk
        _twopt_pho    = _genele_pho    | _genpos_pho
        _twopt_trk    = ~_twopt_pho & (_genele_trk | _genpos_trk)
        _twopt_conv   = ~_twopt_pho & ~_twopt_trk & (_genele_conv | _genpos_conv)
        _twopt_ootpho = ~_twopt_pho & ~_twopt_trk & ~_twopt_conv & (_genele_ootpho | _genpos_ootpho)
        _twopt_pf     = ~_twopt_pho & ~_twopt_trk & ~_twopt_conv & ~_twopt_ootpho & (_genele_pf | _genpos_pf)
        _cat_two_photon     = _two_pt_base & _twopt_pho
        _cat_two_track      = _two_pt_base & _twopt_trk
        _cat_two_conversion = _two_pt_base & _twopt_conv
        _cat_two_ootphoton  = _two_pt_base & _twopt_ootpho
        _cat_two_pfcand     = _two_pt_base & _twopt_pf
        _cat_two_losttrack  = _two_pt_base & ~_twopt_pho & ~_twopt_trk & ~_twopt_conv & ~_twopt_ootpho & ~_twopt_pf

        _cat_masks = [
            (_cat_zero,              'zero_matched'      ),
            (_cat_one_ele,           'one_ele'           ),
            (_cat_one_photon,        'one_photon'        ),
            (_cat_one_track,         'one_track'         ),
            (_cat_one_conversion,    'one_conversion'    ),
            (_cat_one_ootphoton,     'one_ootphoton'     ),
            (_cat_one_pfcand,        'one_pfcand'        ),
            (_cat_one_losttrack,     'one_losttrack'     ),
            (_cat_merged,            'merged'            ),
            (_cat_merged_photon,     'merged_photon'     ),
            (_cat_merged_track,      'merged_track'      ),
            (_cat_merged_conversion, 'merged_conversion' ),
            (_cat_merged_ootphoton,  'merged_ootphoton'  ),
            (_cat_merged_pfcand,     'merged_pfcand'     ),
            (_cat_merged_losttrack,  'merged_losttrack'  ),
            (_cat_ele_photon,        'ele_photon'        ),
            (_cat_ele_track,         'ele_track'         ),
            (_cat_ele_conversion,    'ele_conversion'    ),
            (_cat_ele_ootphoton,     'ele_ootphoton'     ),
            (_cat_ele_pfcand,        'ele_pfcand'        ),
            (_cat_ele_losttrack,     'ele_losttrack'     ),
            (_cat_sep,               'resolved'          ),
            (_cat_two_photon,        'two_photon'        ),
            (_cat_two_track,         'two_track'         ),
            (_cat_two_conversion,    'two_conversion'    ),
            (_cat_two_ootphoton,     'two_ootphoton'     ),
            (_cat_two_pfcand,        'two_pfcand'        ),
            (_cat_two_losttrack,     'two_losttrack'     ),
        ]
        for _cmask, _cname in _cat_masks:
            _n = int(ak.sum(_cmask))
            if _n > 0:
                hists[f'reco_cat_{coll_label}'].fill(
                    samp=samp, cut=cut,
                    cat=np.full(_n, _cname),
                    weight=wgt[_cmask],
                )

        _covered = _cat_masks[0][0]
        for _cm, _ in _cat_masks[1:]:
            _covered = _covered | _cm
        _uncat = ~_covered
        if ak.sum(_uncat) > 0:
            print(f"[mergedmatch/{coll_label}] WARNING: {int(ak.sum(_uncat))} uncategorized events")
            print(f"  A={int(ak.sum(_A[_uncat]))}, B={int(ak.sum(_B[_uncat]))}, M={int(ak.sum(has_merged[_uncat]))}, n_matched={list(ak.to_list(n_matched[_uncat]))}")
            print(f"  Pe={int(ak.sum(_Pe[_uncat]))}, Pp={int(ak.sum(_Pp[_uncat]))}")

        # ── merged selection ──────────────────────────────────────────────────
        keep = _cat_merged

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

            if FILL_DPT_HISTS:
                dpt_abs_lead    = merged_ele.pt - gen_lead_pt
                dpt_abs_sublead = merged_ele.pt - gen_sublead_pt
                dpt_rel_lead    = dpt_abs_lead    / gen_lead_pt
                dpt_rel_sublead = dpt_abs_sublead / gen_sublead_pt
                hists[f'dpt_abs_lead_{coll_label}'            ].fill(samp=samp, cut=cut, dpt_abs=dpt_abs_lead,    weight=w)
                hists[f'dpt_abs_sublead_{coll_label}'         ].fill(samp=samp, cut=cut, dpt_abs=dpt_abs_sublead, weight=w)
                hists[f'dpt_rel_lead_{coll_label}'            ].fill(samp=samp, cut=cut, dpt_rel=dpt_rel_lead,    weight=w)
                hists[f'dpt_rel_sublead_{coll_label}'         ].fill(samp=samp, cut=cut, dpt_rel=dpt_rel_sublead, weight=w)
                hists[f'dpt_abs_lead_vs_genpt_{coll_label}'   ].fill(samp=samp, cut=cut, pt=gen_lead_pt,    dpt_abs=dpt_abs_lead,    weight=w)
                hists[f'dpt_abs_sublead_vs_genpt_{coll_label}'].fill(samp=samp, cut=cut, pt=gen_sublead_pt, dpt_abs=dpt_abs_sublead, weight=w)
                hists[f'dpt_rel_lead_vs_genpt_{coll_label}'   ].fill(samp=samp, cut=cut, pt=gen_lead_pt,    dpt_rel=dpt_rel_lead,    weight=w)
                hists[f'dpt_rel_sublead_vs_genpt_{coll_label}'].fill(samp=samp, cut=cut, pt=gen_sublead_pt, dpt_rel=dpt_rel_sublead, weight=w)
                hists[f'dpt_abs_lead_vs_eedr_{coll_label}'    ].fill(samp=samp, cut=cut, dr=dr_gg,          dpt_abs=dpt_abs_lead,    weight=w)
                hists[f'dpt_abs_sublead_vs_eedr_{coll_label}' ].fill(samp=samp, cut=cut, dr=dr_gg,          dpt_abs=dpt_abs_sublead, weight=w)
                hists[f'dpt_rel_lead_vs_eedr_{coll_label}'    ].fill(samp=samp, cut=cut, dr=dr_gg,          dpt_rel=dpt_rel_lead,    weight=w)
                hists[f'dpt_rel_sublead_vs_eedr_{coll_label}' ].fill(samp=samp, cut=cut, dr=dr_gg,          dpt_rel=dpt_rel_sublead, weight=w)

        # ── resolved selection ────────────────────────────────────────────────
        # Classical sub-type: exclusive one-to-one reco matching, no merged reco.
        # Ambiguous sub-type: merged reco exists but a second reco is also matched.
        # Together these equal _cat_sep, computed above.
        res_classical = _A & _B & ~has_merged
        res_ambiguous = has_merged & (n_matched > 1)

        if ak.sum(_cat_sep) > 0:
            w_res   = wgt[_cat_sep]
            lxy_res = gen_lxy[_cat_sep]

            dphi_gg_res = _dphi(events.GenEle.phi[_cat_sep], events.GenPos.phi[_cat_sep])
            dr_gg_res   = np.sqrt(
                (events.GenEle.eta[_cat_sep] - events.GenPos.eta[_cat_sep])**2 + dphi_gg_res**2
            )

            s = f'resolved_{coll_label}'

            hists[f'gen_lxy_{s}'      ].fill(samp=samp, cut=cut, lxy=lxy_res,                   weight=w_res)
            hists[f'gen_ee_dr_{s}'    ].fill(samp=samp, cut=cut, dr=events.genEE.dr[_cat_sep],  weight=w_res)
            hists[f'gen_ee_pt_{s}'    ].fill(samp=samp, cut=cut, pt=events.genEE.pt[_cat_sep],  weight=w_res)
            hists[f'dr_gen_vs_lxy_{s}'].fill(samp=samp, cut=cut, lxy=lxy_res, dr=dr_gg_res,    weight=w_res)

            # trkChi2: classical events use dedicated ele/pos recos; ambiguous events
            # use the merged reco and the second matched reco
            if ak.sum(res_classical) > 0:
                res_ele_reco = ak.firsts(coll[only_ele_i])[res_classical]
                res_pos_reco = ak.firsts(coll[only_pos_i])[res_classical]
                w_classical  = wgt[res_classical]
                hists[f'reco_trkchi2_{s}'].fill(samp=samp, cut=cut, chi2=res_ele_reco.trkChi2, weight=w_classical)
                hists[f'reco_trkchi2_{s}'].fill(samp=samp, cut=cut, chi2=res_pos_reco.trkChi2, weight=w_classical)

                if FILL_DPT_HISTS:
                    gen_ele_r = events.GenEle[res_classical]
                    gen_pos_r = events.GenPos[res_classical]
                    lead_is_ele_r    = gen_ele_r.pt >= gen_pos_r.pt
                    gen_lead_pt_r    = ak.where(lead_is_ele_r, gen_ele_r.pt, gen_pos_r.pt)
                    gen_sublead_pt_r = ak.where(lead_is_ele_r, gen_pos_r.pt, gen_ele_r.pt)
                    reco_lead_pt_r    = ak.where(lead_is_ele_r, res_ele_reco.pt, res_pos_reco.pt)
                    reco_sublead_pt_r = ak.where(lead_is_ele_r, res_pos_reco.pt, res_ele_reco.pt)

                    dpt_abs_lead_r    = reco_lead_pt_r    - gen_lead_pt_r
                    dpt_abs_sublead_r = reco_sublead_pt_r - gen_sublead_pt_r
                    dpt_rel_lead_r    = dpt_abs_lead_r    / gen_lead_pt_r
                    dpt_rel_sublead_r = dpt_abs_sublead_r / gen_sublead_pt_r
                    dr_gg_r = events.genEE.dr[res_classical]

                    c = coll_label
                    hists[f'dpt_abs_lead_resolved_{c}'            ].fill(samp=samp, cut=cut, dpt_abs=dpt_abs_lead_r,    weight=w_classical)
                    hists[f'dpt_abs_sublead_resolved_{c}'         ].fill(samp=samp, cut=cut, dpt_abs=dpt_abs_sublead_r, weight=w_classical)
                    hists[f'dpt_rel_lead_resolved_{c}'            ].fill(samp=samp, cut=cut, dpt_rel=dpt_rel_lead_r,    weight=w_classical)
                    hists[f'dpt_rel_sublead_resolved_{c}'         ].fill(samp=samp, cut=cut, dpt_rel=dpt_rel_sublead_r, weight=w_classical)
                    hists[f'dpt_abs_lead_resolved_vs_genpt_{c}'   ].fill(samp=samp, cut=cut, pt=gen_lead_pt_r,    dpt_abs=dpt_abs_lead_r,    weight=w_classical)
                    hists[f'dpt_abs_sublead_resolved_vs_genpt_{c}'].fill(samp=samp, cut=cut, pt=gen_sublead_pt_r, dpt_abs=dpt_abs_sublead_r, weight=w_classical)
                    hists[f'dpt_rel_lead_resolved_vs_genpt_{c}'   ].fill(samp=samp, cut=cut, pt=gen_lead_pt_r,    dpt_rel=dpt_rel_lead_r,    weight=w_classical)
                    hists[f'dpt_rel_sublead_resolved_vs_genpt_{c}'].fill(samp=samp, cut=cut, pt=gen_sublead_pt_r, dpt_rel=dpt_rel_sublead_r, weight=w_classical)
                    hists[f'dpt_abs_lead_resolved_vs_eedr_{c}'    ].fill(samp=samp, cut=cut, dr=dr_gg_r, dpt_abs=dpt_abs_lead_r,    weight=w_classical)
                    hists[f'dpt_abs_sublead_resolved_vs_eedr_{c}' ].fill(samp=samp, cut=cut, dr=dr_gg_r, dpt_abs=dpt_abs_sublead_r, weight=w_classical)
                    hists[f'dpt_rel_lead_resolved_vs_eedr_{c}'    ].fill(samp=samp, cut=cut, dr=dr_gg_r, dpt_rel=dpt_rel_lead_r,    weight=w_classical)
                    hists[f'dpt_rel_sublead_resolved_vs_eedr_{c}' ].fill(samp=samp, cut=cut, dr=dr_gg_r, dpt_rel=dpt_rel_sublead_r, weight=w_classical)
            if ak.sum(res_ambiguous) > 0:
                amb_merged_reco = ak.firsts(coll[merged_i])[res_ambiguous]
                w_amb = wgt[res_ambiguous]
                hists[f'reco_trkchi2_{s}'].fill(samp=samp, cut=cut, chi2=amb_merged_reco.trkChi2, weight=w_amb)
                # second reco: prefer matched-but-not-merged; fall back to second merged
                # for low-mass events where all matched electrons are also merged
                amb_second_reco = ak.firsts(coll[matched_i & ~merged_i])[res_ambiguous]
                has_unmerged_second = ~ak.is_none(amb_second_reco)
                if ak.sum(has_unmerged_second) > 0:
                    hists[f'reco_trkchi2_{s}'].fill(samp=samp, cut=cut, chi2=amb_second_reco[has_unmerged_second].trkChi2, weight=w_amb[has_unmerged_second])
                if ak.sum(~has_unmerged_second) > 0:
                    second_merged = ak.pad_none(coll[merged_i], 2, axis=1)[:, 1][res_ambiguous][~has_unmerged_second]
                    ok = ~ak.is_none(second_merged)
                    if ak.sum(ok) > 0:
                        hists[f'reco_trkchi2_{s}'].fill(samp=samp, cut=cut, chi2=second_merged[ok].trkChi2, weight=w_amb[~has_unmerged_second][ok])

        # ── gen ee kinematics per reco category ──────────────────────────────
        _merged_both = _cat_merged | _cat_merged_photrk
        for _mask, _cname in [
            (_cat_zero,    'zero_matched'),
            (_merged_both, 'merged'),
            (_cat_sep,     'resolved'),
        ]:
            if ak.sum(_mask) > 0:
                hists[f'gen_ee_dr_cat_{coll_label}'].fill(
                    samp=samp, cut=cut, cat=_cname,
                    dr=events.genEE.dr[_mask], weight=wgt[_mask],
                )
                hists[f'gen_ee_pt_cat_{coll_label}'].fill(
                    samp=samp, cut=cut, cat=_cname,
                    pt=events.genEE.pt[_mask], weight=wgt[_mask],
                )
                hists[f'gen_ee_eta_cat_{coll_label}'].fill(
                    samp=samp, cut=cut, cat=_cname,
                    eta=events.genEE.eta[_mask], weight=wgt[_mask],
                )
