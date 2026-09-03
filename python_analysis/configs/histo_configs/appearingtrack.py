try:
    from histobins import *
except ModuleNotFoundError:
    from configs.histo_configs.histobins import *
from hist import Hist
from hist.axis import Regular, StrCategory, IntCategory, Integer
import hist
import numpy as np
import awkward as ak

# "Appearing track" selection: probes single displaced-electron reconstruction
# (as opposed to the merged-electron category in mergedcats.py/mergedeles.py,
# which requires two nearby gen electrons collapsing into one reco electron).
# An AllLptElectron is a candidate iff:
#   - |dxy| >= 1e-4 cm
#   - dR >= 1 to every other AllLptElectron in the same event
#   - dR >= 1 to every PFJet with pt > 30 GeV in the same event
# At most one candidate is kept per event: if several AllLptElectrons in the
# same event pass, the one with the largest |dxy| is taken.
# Works for both signal and background; the isMerged flag (signal only, see
# _compute_merged_flag) additionally tags whether the selected electron
# individually qualifies as a genuine merged electron under the mergedcats.py
# definition.

_LOG = hist.axis.transform.log

# ── Extra axes not in histobins ──────────────────────────────────────────────
_e_energy  = Regular(100,  0, 100,  name='e',        label='E [GeV]')
_calIso    = Regular(100,  0, 100,  name='iso',       label='Calo Iso [GeV]')
_calRelIso = Regular(100,  0,   5,  name='relIso',    label='Calo Relative Iso')
_rhoEA     = Regular(100,  0,  20,  name='rhoEA',     label=r'$\rho \times EA$ [GeV]')
_sieie     = Regular(100,  0,0.05,  name='sieie',     label=r'$\sigma_{i\eta i\eta}$')
_dEtaSeed  = Regular(100,  0,0.01,  name='dEtaSeed',  label=r'$|\Delta\eta_\mathrm{seed}|$')
_dPhiIn    = Regular(100,  0, 0.1,  name='dPhiIn',    label=r'$|\Delta\phi_\mathrm{in}|$')
_HoE       = Regular(100,  0,   1,  name='HoE',       label='H/E')
_invEmP    = Regular(100,  0, 0.1,  name='invEmP',    label=r'$|1/E - 1/p|$ [GeV$^{-1}$]')
_missHits  = Integer(0,    5,       name='missHits',  label='Exp. Missing Inner Hits')
_charge    = IntCategory([-1, 1],   name='charge',    label='Charge')

# dxy floored at the selection cut (1e-4 cm) so the axis stays log-scale down
# to that value; anything below the floor (shouldn't occur post-selection)
# lands in underflow rather than being dropped.
_dxy_log = Regular(140, 1e-4, 100, name='dxy', label=r'Electron Track $d_{xy}$ [cm]', transform=_LOG)

# Full-range linear dphi(e, ptmiss), plus a log-scale zoom toward the
# collinear region (small dphi is the signature of a genuinely displaced
# track pointing away from the hard-scatter PV).
_dphi_log = Regular(100, 1e-4, 3.2, name='dphi', label=r'$\Delta\phi(e, p_{T}^{miss})$', transform=_LOG)

# lxyEst = dxy / sin(dphi(e, ptmiss)): the same reco-only Lxy proxy used in
# mergedeles.py's mele_gen_lxy_vs_lxyEst, without the gen-Lxy correlation
# (which doesn't exist for background).
_lxyEst_log = Regular(100, 1e-4, 100, name='lxyEst', label=r'$d_{xy}/\sin(\Delta\phi(e, p_{T}^{miss}))$ [cm]', transform=_LOG)

# Tags whether the selected electron is a genuine merged electron (signal
# only, via _compute_merged_flag below); always 'n/a' for background since
# there's no gen truth to check against.
isMerged = StrCategory(['merged', 'notMerged', 'n/a'], name='isMerged', label='Genuine Merged Electron (signal only)')

# (field_name_on_AllLptElectron, hist_axis) -- same reco variables used for
# the merged-electron category in mergedcats.py/mergedeles.py, so this config
# fills the full set needed to assess single displaced-electron reconstruction
# on equal footing.
_ELE_HISTS = [
    ('pt',                  ele_pt),
    ('eta',                 ele_eta),
    ('phi',                 ele_phi),
    #('e',                   _e_energy),
    ('ID',                  ele_id),
    ('angRes',              ele_angRes),
    #('vxy',                 vxy_coarse),
    #('vz',                  vz_coarse),
    ('dxy',                 _dxy_log),
    ('dz',                  ele_dz),
    ('trkChi2',             ele_chi2),
    ('trkIso',              ele_trkIso),
    ('trkRelIso',           ele_trkRelIso),
    ('calIso',              _calIso),
    ('calRelIso',           _calRelIso),
    #('PFIso',               ele_PFIso),
    #('PFRelIso',            ele_PFRelIso),
    ('miniIso',             ele_miniIso),
    ('miniRelIso',          ele_miniRelIso),
    #('PFIsoEleCorr',        ele_PFIso),
    #('PFRelIsoEleCorr',     ele_PFRelIso),
    #('miniIsoEleCorr',      ele_miniIsoCorr),
    #('miniRelIsoEleCorr',   ele_miniRelIsoCorr),
    #('chadIso',             ele_PFIso),
    #('nhadIso',             ele_PFIso),
    #('phoIso',              ele_PFIso),
    ('rhoEA',               _rhoEA),
    ('trkProb',             ele_prob),
    ('numTrackerHits',      ele_trkHits),
    ('numPixHits',          ele_pixHits),
    ('numStripHits',        ele_stripHits),
    #('charge',              _charge),
    ('minDRtoReg',          dR),
    ('mindRj',              dRj),
    ('mindPhiJ',            dphiJ),
    ('full55sigmaIetaIeta', _sieie),
    ('absdEtaSeed',         _dEtaSeed),
    ('absdPhiIn',           _dPhiIn),
    ('HoverE',              _HoE),
    ('abs1overEm1overP',    _invEmP),
    ('expMissingInnerHits', _missHits),
    #('IDscore',             ele_id),
]

# Fields that are filled as np.abs(...) (signed in the ntuple, unsigned on
# the axis) -- mirrors the mele_*/res_* convention in mergedcats.py.
_ABS_FIELDS = {'vxy', 'vz', 'dxy', 'dz'}

# 'ID' and 'IDscore' both read the same underlying 'ID' field (mirrors
# mergedcats.py's mele_ID/mele_IDscore, which both pull from me.ID).
_FIELD_ALIASES = {'IDscore': 'ID'}

def _fill_single(hists, prefix, obj, evt_weight, isMerged_label, samp, cut, field_hists):
    """Fill `{prefix}_{field}` hists with one (already-selected, non-jagged)
    reco electron per event.
    """
    for field, axis in field_hists:
        val = getattr(obj, _FIELD_ALIASES.get(field, field))
        if field in _ABS_FIELDS:
            val = np.abs(val)
        hists[f'{prefix}_{field}'].fill(
            samp=samp, cut=cut, isMerged=isMerged_label, weight=evt_weight,
            **{axis.name: val},
        )

def make_histograms():
    histograms = {}
    for field, axis in _ELE_HISTS:
        histograms[f'AT_{field}'] = Hist(samp, cut, isMerged, axis, storage=hist.storage.Weight())

    histograms['AT_ptmiss']            = Hist(samp, cut, isMerged, met_pt,       storage=hist.storage.Weight())
    histograms['AT_dphi_e_ptmiss']     = Hist(samp, cut, isMerged, dphi_generic, storage=hist.storage.Weight())
    histograms['AT_dphi_e_ptmiss_log'] = Hist(samp, cut, isMerged, _dphi_log,    storage=hist.storage.Weight())
    histograms['AT_lxyEst']            = Hist(samp, cut, isMerged, _lxyEst_log,  storage=hist.storage.Weight())
    histograms['AT_dxy_vs_dphi']       = Hist(samp, cut, isMerged, _dxy_log, _dphi_log, storage=hist.storage.Weight())
    return histograms

subroutines = []

def _compute_merged_flag(events):
    """Per-electron bool (jagged, same shape as events.AllLptElectron): is
    this particular AllLptElectron a genuine merged electron, using the same
    definition as the merged-electron category in mergedcats.py (quality-
    matched to one gen electron, dR<0.1 to both gens, and no photon/track/
    conversion alternative for either gen). Signal only -- requires
    GenEle/GenPos and the photon/track/conversion collections.

    Unlike mergedcats.py's event-level _cat_merged, this doesn't additionally
    exclude events where a second AllLptElectron is also gen-matched
    (_cat_sep there) -- that's an event-topology concept, not a property of
    this one electron. The caller reduces to at most one candidate electron
    per event (by |dxy|) before reading this flag off the chosen electron.
    """
    def _dphi(a, b):
        d = np.abs(a - b)
        return ak.where(d > np.pi, 2 * np.pi - d, d)

    def _dr(eta1, phi1, eta2, phi2):
        return np.sqrt((eta1 - eta2)**2 + _dphi(phi1, phi2)**2)

    def _pt_rel(obj_pt, gen_pt):
        return np.abs(obj_pt - gen_pt) / gen_pt < 0.2

    gen_sum_pt = events.GenEle.pt + events.GenPos.pt

    _pho_dr_ge  = _dr(events.Photon.eta,     events.Photon.phi,     events.GenEle.eta, events.GenEle.phi)
    _pho_dr_gp  = _dr(events.Photon.eta,     events.Photon.phi,     events.GenPos.eta, events.GenPos.phi)
    _oot_dr_ge  = _dr(events.ootPhoton.eta,  events.ootPhoton.phi,  events.GenEle.eta, events.GenEle.phi)
    _oot_dr_gp  = _dr(events.ootPhoton.eta,  events.ootPhoton.phi,  events.GenPos.eta, events.GenPos.phi)
    _trk_dr_ge  = _dr(events.IsoTrack.eta,   events.IsoTrack.phi,   events.GenEle.eta, events.GenEle.phi)
    _trk_dr_gp  = _dr(events.IsoTrack.eta,   events.IsoTrack.phi,   events.GenPos.eta, events.GenPos.phi)
    _conv_dr_ge = _dr(events.Conversion.eta, events.Conversion.phi, events.GenEle.eta, events.GenEle.phi)
    _conv_dr_gp = _dr(events.Conversion.eta, events.Conversion.phi, events.GenPos.eta, events.GenPos.phi)

    _pho_pt_ge  = _pt_rel(events.Photon.pt,     events.GenEle.pt)
    _pho_pt_gp  = _pt_rel(events.Photon.pt,     events.GenPos.pt)
    _oot_pt_ge  = _pt_rel(events.ootPhoton.pt,  events.GenEle.pt)
    _oot_pt_gp  = _pt_rel(events.ootPhoton.pt,  events.GenPos.pt)
    _trk_pt_ge  = _pt_rel(events.IsoTrack.pt,   events.GenEle.pt)
    _trk_pt_gp  = _pt_rel(events.IsoTrack.pt,   events.GenPos.pt)
    _conv_pt_ge = _pt_rel(events.Conversion.pt, events.GenEle.pt)
    _conv_pt_gp = _pt_rel(events.Conversion.pt, events.GenPos.pt)

    _ge_any = (
        ak.any((_pho_dr_ge  < 0.1) & _pho_pt_ge,                                  axis=1) |
        ak.any((_oot_dr_ge  < 0.1) & _oot_pt_ge,                                  axis=1) |
        ak.any((_trk_dr_ge  < 0.1) & (events.IsoTrack.charge == -1) & _trk_pt_ge, axis=1) |
        ak.any((_conv_dr_ge < 0.1) & _conv_pt_ge,                                 axis=1)
    )
    _gp_any = (
        ak.any((_pho_dr_gp  < 0.1) & _pho_pt_gp,                                  axis=1) |
        ak.any((_oot_dr_gp  < 0.1) & _oot_pt_gp,                                  axis=1) |
        ak.any((_trk_dr_gp  < 0.1) & (events.IsoTrack.charge == +1) & _trk_pt_gp, axis=1) |
        ak.any((_conv_dr_gp < 0.1) & _conv_pt_gp,                                 axis=1)
    )

    coll = events.AllLptElectron
    dphi_e   = _dphi(coll.phi, events.GenEle.phi)
    dphi_p   = _dphi(coll.phi, events.GenPos.phi)
    dr_to_ge = np.sqrt((coll.eta - events.GenEle.eta)**2 + dphi_e**2)
    dr_to_gp = np.sqrt((coll.eta - events.GenPos.eta)**2 + dphi_p**2)

    pt_me = _pt_rel(coll.pt, events.GenEle.pt)
    pt_mp = _pt_rel(coll.pt, events.GenPos.pt)
    pt_ms = _pt_rel(coll.pt, gen_sum_pt)

    dr_match_ge = (dr_to_ge < 0.1) & (coll.charge == -1) & pt_me
    dr_match_gp = (dr_to_gp < 0.1) & (coll.charge == +1) & pt_mp

    pt_ok_ge = pt_me | pt_ms
    pt_ok_gp = pt_mp | pt_ms

    merged_i = (dr_to_ge < 0.1) & (dr_to_gp < 0.1) & (
        ((coll.charge == -1) & pt_ok_ge & ~dr_match_gp & ~_gp_any) |
        ((coll.charge == +1) & pt_ok_gp & ~dr_match_ge & ~_ge_any)
    )

    # ~_ge_any/~_gp_any broadcast from event-level to per-electron here, on
    # top of the ~_gp_any/~_ge_any already embedded above for the gen the
    # electron did NOT match to -- this adds the check for the gen it DID
    # match to, completing the full mergedcats.py exclusivity condition.
    return merged_i & ~_ge_any & ~_gp_any

def fillHistos(events, hists, samp, cut, info, sum_wgt=1):
    wgt_all = events.eventWgt / sum_wgt

    # mindRj/mindPhiJ derived fields needed by _ELE_HISTS (same convention as
    # mergedcats.py).
    events['AllLptElectron', 'mindRj']   = ak.fill_none(ak.min(events.AllLptElectron.dRJets,   axis=-1), 999)
    events['AllLptElectron', 'mindPhiJ'] = ak.fill_none(ak.min(events.AllLptElectron.dPhiJets, axis=-1), 999)

    coll = events.AllLptElectron

    # Compute the merged-electron flag (signal only) up front, since it needs
    # the full (unfiltered) AllLptElectron collection for the exclusivity
    # checks -- same as mergedcats.py. Jagged, same shape as coll.
    if info['type'] == 'signal':
        per_ele_merged = _compute_merged_flag(events)

    def _dphi(a, b):
        d = np.abs(a - b)
        return ak.where(d > np.pi, 2 * np.pi - d, d)

    def _dr(eta1, phi1, eta2, phi2):
        return np.sqrt((eta1 - eta2)**2 + _dphi(phi1, phi2)**2)

    # ── Per-electron isolation from other electrons and jets ─────────────────
    # dR to every other AllLptElectron in the same event (self excluded).
    idx = ak.local_index(coll, axis=1)
    dr_ee = _dr(coll.eta[:, :, None], coll.phi[:, :, None], coll.eta[:, None, :], coll.phi[:, None, :])
    dr_ee = ak.where(idx[:, :, None] == idx[:, None, :], np.inf, dr_ee)
    min_dr_ee = ak.fill_none(ak.min(dr_ee, axis=-1), 999)
    ele_iso = min_dr_ee >= 1.0

    # dR to every PFJet with pt > 30 GeV in the same event.
    jets = events.PFJet
    jets = jets[jets.pt > 30]
    dr_ej = _dr(coll.eta[:, :, None], coll.phi[:, :, None], jets.eta[:, None, :], jets.phi[:, None, :])
    min_dr_ej = ak.fill_none(ak.min(dr_ej, axis=-1), 999)
    jet_iso = min_dr_ej >= 1.0

    dxy_pass = np.abs(coll.dxy) >= 1e-4

    ele_sel = ele_iso & jet_iso & dxy_pass

    # ── "Appearing track" selection ───────────────────────────────────────────
    # At most one candidate per event: if several AllLptElectrons in the same
    # event pass the cuts above, keep the one with the largest |dxy|.
    cand = coll[ele_sel]
    if ak.sum(ak.num(cand, axis=1)) == 0:
        return

    order     = ak.argsort(np.abs(cand.dxy), axis=1, ascending=False)
    ele       = ak.firsts(cand[order])
    has_cand  = ~ak.is_none(ele)
    if ak.sum(has_cand) == 0:
        return

    ele  = ele[has_cand]
    evts = events[has_cand]
    wgt  = wgt_all[has_cand]

    if info['type'] == 'signal':
        merged_best   = ak.firsts(per_ele_merged[ele_sel][order])
        is_merged_sel = merged_best[has_cand]
        isMerged_label = np.where(ak.to_numpy(is_merged_sel), 'merged', 'notMerged')
    else:
        isMerged_label = np.full(len(ele), 'n/a', dtype=object)

    # ── Reco-quantity fills ────────────────────────────────────────────────
    _fill_single(hists, 'AT', ele, wgt, isMerged_label, samp, cut, _ELE_HISTS)

    ptmiss     = evts.PFMET.pt
    dphi_e_met = _dphi(ele.phi, evts.PFMET.phi)
    lxyEst     = np.abs(ele.dxy) / np.sin(dphi_e_met)

    hists['AT_ptmiss'           ].fill(samp=samp, cut=cut, isMerged=isMerged_label, met_pt=ptmiss,     weight=wgt)
    hists['AT_dphi_e_ptmiss'    ].fill(samp=samp, cut=cut, isMerged=isMerged_label, dphi=dphi_e_met,   weight=wgt)
    hists['AT_dphi_e_ptmiss_log'].fill(samp=samp, cut=cut, isMerged=isMerged_label, dphi=dphi_e_met,   weight=wgt)
    hists['AT_lxyEst'           ].fill(samp=samp, cut=cut, isMerged=isMerged_label, lxyEst=lxyEst,     weight=wgt)
    hists['AT_dxy_vs_dphi'      ].fill(samp=samp, cut=cut, isMerged=isMerged_label, dxy=np.abs(ele.dxy), dphi=dphi_e_met, weight=wgt)
