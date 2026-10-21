try:
    from histobins import *
except ModuleNotFoundError:
    from configs.hists.histobins import *
# share the 4D binning with elerecoeff so the reason breakdowns line up bin-for-bin with its alptnosum/sumnoalpt hists
from elerecoeff import ee_pt_4d, ee_lxy_4d, ee_dr_4d, ee_abseta_4d
from hist import Hist
import hist
import numpy as np
import awkward as ak
from hist.axis import StrCategory

# Reasons a gen electron lands in each of the elerecoeff "disagreement" categories:
#   alptnosum: matched to an AllLptElectron, but matchType None (not matched in the GED + LptElectron pool)
#   sumnoalpt: matchType L or R, but not matched to any AllLptElectron
# Every gen electron in a category gets exactly one reason; "other" is the remainder.
cat_reasons = {
    'alptnosum': ['GED xcleaned two lpts', '1 GED + 1 ALPT matched, opp charge', '1 GED + 1 ALPT matched, same charge', 'other'],
    'sumnoalpt': ['no LPT ele', '1 GED + 1 ALPT matched, opp charge', '1 GED + 1 ALPT matched, same charge', '2 GED + 1 ALPT matched', 'other'],
}

# same dR window as the gen matching in ElectronSkimmer.cc (both the GED+Lpt and the AllLowPt pools)
gen_match_dr = 0.1

def make_histograms():
    histograms = {}
    for c, reasons in cat_reasons.items():
        reason = StrCategory(reasons, name="reason", label="Reason")
        histograms[f"ele_reco_{c}_reason_ee4d"] = Hist(samp, cut, reason, ee_pt_4d, ee_lxy_4d, ee_dr_4d, ee_abseta_4d, storage=hist.storage.Weight())
    return histograms

subroutines = []

def fillHistos(events, hists, samp, cut, info, sum_wgt=1):
    wgt = events.eventWgt

    # lxy from the chi2 production vertex, as in elerecoeff
    chi2 = ak.firsts(events.GenPart[np.abs(events.GenPart.ID) == 1000023])
    lxy_ele = np.sqrt((events.GenEle.vx - chi2.vx)**2 + (events.GenEle.vy - chi2.vy)**2)
    lxy_pos = np.sqrt((events.GenPos.vx - chi2.vx)**2 + (events.GenPos.vy - chi2.vy)**2)

    # one row per gen electron (GenEle rows, then GenPos rows); per-event collections are duplicated to line up
    all_gen  = ak.concatenate([events.GenEle,         events.GenPos],         axis=0)
    all_lxy  = ak.concatenate([lxy_ele,               lxy_pos],               axis=0)
    all_wgt  = ak.concatenate([wgt,                   wgt],                   axis=0)
    all_ee   = ak.concatenate([events.genEE,          events.genEE],          axis=0)
    all_alpt = ak.concatenate([events.AllLptElectron, events.AllLptElectron], axis=0)

    all_matchedalllpt = all_gen.matchedAllLowPt
    all_matchedsum    = (all_gen.matchType == 'L') | (all_gen.matchType == 'R')

    # --- alptnosum reasons ---
    # The GED+Lpt matching pool drops x-cleaned AllLptElectrons (R < 0.05 to a GED electron), so an alptnosum
    # electron's AllLpt match is normally x-cleaned and the GED that absorbed it was not matched to this gen electron.
    # "GED xcleaned two lpts": the matched AllLptElectron is x-cleaned, and at least one other AllLptElectron in the
    # event was x-cleaned against the same GED electron (gedIdx), i.e. one GED electron swallowed two LPT electrons,
    # regardless of how the partner gen electron was matched.
    alpt_idx   = ak.local_index(all_alpt, axis=1)
    is_match   = alpt_idx == all_gen.matchIdxAllLowPt
    match_xc   = ak.fill_none(ak.firsts(all_alpt.xCleaned[is_match]), False)
    match_ged  = ak.fill_none(ak.firsts(all_alpt.gedIdx[is_match]), -999)
    n_same_ged = ak.sum(all_alpt.xCleaned & (all_alpt.gedIdx == match_ged), axis=1)
    ged_xc_two = match_xc & (n_same_ged >= 2)

    # "1 GED + 1 ALPT matched": exactly one gen-matched GED electron and exactly one gen-matched AllLptElectron in the
    # event, matched to different gen electrons (matchType -1 = e-, +1 = e+), split by whether their reco charges are
    # opposite or the same. The GED is not required to be the one that x-cleaned the ALPT.
    # (Electron fields are concatenated individually: concatenating the full record materializes every branch)
    ged_gm  = ak.concatenate([events.Electron.genMatched, events.Electron.genMatched], axis=0)
    ged_mt  = ak.concatenate([events.Electron.matchType,  events.Electron.matchType],  axis=0)
    ged_q   = ak.concatenate([events.Electron.charge,     events.Electron.charge],     axis=0)
    one_one = (ak.sum(ged_gm, axis=1) == 1) & (ak.sum(all_alpt.genMatched, axis=1) == 1)
    m_ged_mt  = ak.fill_none(ak.firsts(ged_mt[ged_gm]), 0)
    m_ged_q   = ak.fill_none(ak.firsts(ged_q[ged_gm]), 0)
    m_alpt_mt = ak.fill_none(ak.firsts(all_alpt.matchType[all_alpt.genMatched]), 0)
    m_alpt_q  = ak.fill_none(ak.firsts(all_alpt.charge[all_alpt.genMatched]), 0)
    one_one   = one_one & (m_ged_mt != m_alpt_mt)
    one_one_opp  = one_one & (m_ged_q == -m_alpt_q)
    one_one_same = one_one & (m_ged_q == m_alpt_q)

    # --- sumnoalpt reasons ---
    # "no LPT ele": no AllLptElectron (pt > 1 GeV, any x-cleaning status) within the gen-matching dR of this gen
    # electron, so the AllLowPt matching had nothing to pick. Otherwise a candidate existed but the matching
    # did not assign it to this electron (e.g. it went to the partner) -> "other".
    dphi = (all_alpt.phi - all_gen.phi + np.pi) % (2*np.pi) - np.pi
    dr_alpt = np.sqrt((all_alpt.eta - all_gen.eta)**2 + dphi**2)
    no_lpt = ~ak.any(dr_alpt < gen_match_dr, axis=1)

    # "1 GED + 1 ALPT matched" (sumnoalpt): same one-GED + one-ALPT event requirement as for alptnosum, with the ALPT
    # matched to the partner gen electron (GenEle rows are e-, GenPos rows e+), so the GED is the one matched to
    # this electron. Split by whether the GED and ALPT reco charges are opposite or the same.
    gen_mt = np.concatenate([-np.ones(len(events)), np.ones(len(events))])
    alpt_partner = one_one & (m_alpt_mt == -gen_mt)

    # "2 GED + 1 ALPT matched": both gen electrons have a gen-matched GED, and the event's single gen-matched ALPT
    # went to the partner. Typically a close pair sharing one ALPT within the matching dR of both gen electrons,
    # which the greedy AllLowPt matching gives to the closer one.
    ged_self = ak.sum(ged_gm & (ged_mt == gen_mt), axis=1)
    ged_part = ak.sum(ged_gm & (ged_mt == -gen_mt), axis=1)
    two_one  = (ged_self == 1) & (ged_part == 1) & (ak.sum(all_alpt.genMatched, axis=1) == 1) & (m_alpt_mt == -gen_mt)

    alptnosum = all_matchedalllpt & ~all_matchedsum
    sumnoalpt = all_matchedsum & ~all_matchedalllpt
    reason_masks = {
        'alptnosum': {
            'GED xcleaned two lpts':               alptnosum & ged_xc_two,
            '1 GED + 1 ALPT matched, opp charge':  alptnosum & ~ged_xc_two & one_one_opp,
            '1 GED + 1 ALPT matched, same charge': alptnosum & ~ged_xc_two & one_one_same,
            'other':                               alptnosum & ~ged_xc_two & ~one_one_opp & ~one_one_same,
        },
        'sumnoalpt': {
            'no LPT ele':                          sumnoalpt & no_lpt,
            '1 GED + 1 ALPT matched, opp charge':  sumnoalpt & ~no_lpt & alpt_partner & (m_ged_q == -m_alpt_q),
            '1 GED + 1 ALPT matched, same charge': sumnoalpt & ~no_lpt & alpt_partner & (m_ged_q == m_alpt_q),
            '2 GED + 1 ALPT matched':              sumnoalpt & ~no_lpt & two_one,
            'other':                               sumnoalpt & ~no_lpt & ~alpt_partner & ~two_one,
        },
    }

    # values above the last edge are clipped into the top bin (e.g. the top pt bin is pt > 20)
    top = lambda x, ax: np.minimum(x, ax.edges[-2])
    for c, reasons in reason_masks.items():
        for r, mask in reasons.items():
            hists[f"ele_reco_{c}_reason_ee4d"].fill(samp=samp, cut=cut, reason=r,
                                                    pt=top(all_gen.pt[mask], ee_pt_4d), lxy=top(all_lxy[mask], ee_lxy_4d),
                                                    dr=top(all_ee.dr[mask], ee_dr_4d), abseta=top(np.abs(all_gen.eta[mask]), ee_abseta_4d),
                                                    weight=all_wgt[mask])
