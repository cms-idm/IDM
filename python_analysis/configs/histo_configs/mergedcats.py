try:
    from histobins import *
except ModuleNotFoundError:
    from configs.histo_configs.histobins import *
from hist import Hist
from hist.axis import Variable, Regular, StrCategory, IntCategory, Integer
import hist
import numpy as np
import awkward as ak

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
_r9        = Regular(100,  0, 1.5,  name='r9',        label='R9')

# Same variable-width style as ele_lxy_res (histobins.py), extended out to 150 cm.
_gen_lxy = Variable(
    [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1, 1.5, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 50, 70, 100, 150],
    name='lxy', label='$L_{xy}$ [cm]',
)

# (field_name_on_AllLptElectron, hist_axis)
_ELE_HISTS = [
    ('pt',                  ele_pt),
    ('eta',                 ele_eta),
    ('phi',                 ele_phi),
    ('e',                   _e_energy),
    ('ID',                  ele_id),
    ('angRes',              ele_angRes),
    ('vxy',                 vxy_coarse),
    ('vz',                  vz_coarse),
    ('dxy',                 ele_dxy),
    ('dz',                  ele_dz),
    ('trkChi2',             ele_chi2),
    ('trkIso',              ele_trkIso),
    ('trkRelIso',           ele_trkRelIso),
    ('calIso',              _calIso),
    ('calRelIso',           _calRelIso),
    ('PFIso',               ele_PFIso),
    ('PFRelIso',            ele_PFRelIso),
    ('miniIso',             ele_miniIso),
    ('miniRelIso',          ele_miniRelIso),
    ('PFIsoEleCorr',        ele_PFIso),
    ('PFRelIsoEleCorr',     ele_PFRelIso),
    ('miniIsoEleCorr',      ele_miniIsoCorr),
    ('miniRelIsoEleCorr',   ele_miniRelIsoCorr),
    ('chadIso',             ele_PFIso),
    ('nhadIso',             ele_PFIso),
    ('phoIso',              ele_PFIso),
    ('rhoEA',               _rhoEA),
    ('trkProb',             ele_prob),
    ('numTrackerHits',      ele_trkHits),
    ('numPixHits',          ele_pixHits),
    ('numStripHits',        ele_stripHits),
    ('charge',              _charge),
    ('minDRtoReg',          dR),
    ('mindRj',              dRj),
    ('mindPhiJ',            dphiJ),
    ('full55sigmaIetaIeta', _sieie),
    ('absdEtaSeed',         _dEtaSeed),
    ('absdPhiIn',           _dPhiIn),
    ('HoverE',              _HoE),
    ('abs1overEm1overP',    _invEmP),
    ('expMissingInnerHits', _missHits),
    ('IDscore',             ele_id),
]

# (field_name_on_Photon, hist_axis)
_PHO_HISTS = [
    ('pt',           ele_pt),
    ('eta',          ele_eta),
    ('phi',          ele_phi),
    ('energy',       _e_energy),
    ('r9',           _r9),
    ('full5x5_r9',   _r9),
    ('sIeIe',        _sieie),
    ('full5x5_sIeIe',_sieie),
    ('HoE',          _HoE),
    ('full5x5_HoE',  _HoE),
    ('chIso',        ele_PFIso),
    ('nhIso',        ele_PFIso),
    ('phIso',        ele_PFIso),
    ('puChIso',      ele_PFIso),
    ('trkIso',       ele_trkIso),
    ('ecalIso',      ele_PFIso),
    ('hcalIso',      ele_PFIso),
    ('mindRj',       dRj),
    ('mindPhiJ',     dphiJ),
]

# For the merged-electron category: once the reco electron is tightly matched
# to one gen electron (dr<0.1, charge match, delta_pt_rel<0.1), these are the
# other reco collections we check the *second* (unmatched) gen electron's
# proximity to. (suffix, events field name)
_OTHERGEN_DR_TYPES = [
    ('lptele',     'AllLptElectron'),
    ('photon',     'Photon'),
    ('conversion', 'Conversion'),
    ('ootphoton',  'ootPhoton'),
    ('pfcand',     'PFCand'),
    ('losttrack',  'LostTrack'),
]

# Reco variables that have a corresponding 2D gen-Lxy correlation histogram
# (mpho_gen_lxy_vs_*, defined below): also produce 1D versions of just these,
# sliced above/below the Lxy cut. gen_lxy (and vx/vy/PV) are in cm throughout
# this codebase, so 1000 mm -> 100 cm.
_LXY_SLICE_VARS = [
    ('mpho', 'HoE',           _HoE),
    ('mpho', 'full5x5_sIeIe', _sieie),
]
_LXY_CUT_CM    = 100.0
_LXY_HI_SUFFIX = '_lxyGt1000mm'
_LXY_LO_SUFFIX = '_lxyLt1000mm'

def _fill_lxy_slice(hists, prefix, obj, mask, weight, samp, cut, suffix):
    if ak.sum(mask) == 0:
        return
    w_m = weight[mask]
    for p, field, axis in _LXY_SLICE_VARS:
        if p != prefix:
            continue
        hists[f'{prefix}_{field}{suffix}'].fill(
            samp=samp, cut=cut, weight=w_m,
            **{axis.name: getattr(obj, field)[mask]},
        )

# Fields that are filled as np.abs(...) in the matched mele_*/res_* hists
# (signed in the ntuple, but histogrammed unsigned) — replicate that here so
# the zero/other comparison hists land on the same axis convention.
_ABS_FIELDS = {'vxy', 'vz', 'dxy', 'dz'}

# _ELE_HISTS has two hist-key entries ('ID' and 'IDscore') that both read the
# same underlying 'ID' field (mirrors the existing mele_ID/mele_IDscore fills,
# which both pull from me.ID) — 'IDscore' isn't itself a field on the ntuple.
_FIELD_ALIASES = {'IDscore': 'ID'}

def _fill_multi(hists, prefix, obj, evt_weight, samp, cut, field_hists):
    """Fill `{prefix}_{field}` hists with every entry of a jagged reco
    collection (0, 1, or many objects per event), broadcasting the per-event
    weight to match. No-op if the (masked) collection has no entries at all.
    """
    if ak.sum(ak.num(obj, axis=1)) == 0:
        return
    w_flat = ak.flatten(ak.broadcast_arrays(evt_weight, obj.pt)[0])
    for field, axis in field_hists:
        val = getattr(obj, _FIELD_ALIASES.get(field, field))
        if field in _ABS_FIELDS:
            val = np.abs(val)
        hists[f'{prefix}_{field}'].fill(
            samp=samp, cut=cut, weight=w_flat,
            **{axis.name: ak.flatten(val)},
        )

def _fill_single(hists, prefix, obj, evt_weight, samp, cut, field_hists):
    """Fill `{prefix}_{field}` hists with one (already-selected, non-jagged)
    reco object per event.
    """
    for field, axis in field_hists:
        val = getattr(obj, _FIELD_ALIASES.get(field, field))
        if field in _ABS_FIELDS:
            val = np.abs(val)
        hists[f'{prefix}_{field}'].fill(
            samp=samp, cut=cut, weight=evt_weight,
            **{axis.name: val},
        )

def make_histograms():
    histograms = {}

    # ── Merged-electron category (AllLptElectron reco variables) ─────────────
    for field, axis in _ELE_HISTS:
        histograms[f'mele_{field}'] = Hist(samp, cut, axis, storage=hist.storage.Weight())
    histograms['mele_gen_lxy']        = Hist(samp, cut, _gen_lxy,  storage=hist.storage.Weight())
    histograms['mele_gen_ee_pt']      = Hist(samp, cut, ele_pt,        storage=hist.storage.Weight())
    histograms['mele_gen_lead_pt']    = Hist(samp, cut, ele_pt,        storage=hist.storage.Weight())
    histograms['mele_gen_sublead_pt'] = Hist(samp, cut, ele_pt,        storage=hist.storage.Weight())
    histograms['mele_gen_ee_dr']      = Hist(samp, cut, ee_dr,  storage=hist.storage.Weight())
    histograms['mele_gen_ee_eta']     = Hist(samp, cut, ele_eta,       storage=hist.storage.Weight())
    histograms['mele_gen_lead_eta']   = Hist(samp, cut, ele_eta,       storage=hist.storage.Weight())
    histograms['mele_gen_sublead_eta']= Hist(samp, cut, ele_eta,       storage=hist.storage.Weight())
    for _suffix, _ in _OTHERGEN_DR_TYPES:
        histograms[f'mele_othergen_dr_{_suffix}'] = Hist(samp, cut, dR, storage=hist.storage.Weight())

    # ── Merged-electron comparison hists: same reco variables (_ELE_HISTS),
    # but for (a) AllLptElectron objects in zero-match events — split into the
    # leading-pt electron (zerolead) and the rest (zeroothers) in the same
    # event — and (b) other (non-matched) AllLptElectron objects present
    # alongside the merged one.
    for field, axis in _ELE_HISTS:
        histograms[f'mele_zerolead_{field}']   = Hist(samp, cut, axis, storage=hist.storage.Weight())
        histograms[f'mele_zeroothers_{field}'] = Hist(samp, cut, axis, storage=hist.storage.Weight())
        histograms[f'mele_other_{field}']      = Hist(samp, cut, axis, storage=hist.storage.Weight())

    # ── Merged-photon category (Photon reco variables) ────────────────────────
    for field, axis in _PHO_HISTS:
        histograms[f'mpho_{field}'] = Hist(samp, cut, axis, storage=hist.storage.Weight())
    histograms['mpho_gen_lxy']        = Hist(samp, cut, _gen_lxy,  storage=hist.storage.Weight())
    histograms['mpho_gen_ee_pt']      = Hist(samp, cut, ele_pt,        storage=hist.storage.Weight())
    histograms['mpho_gen_lead_pt']    = Hist(samp, cut, ele_pt,        storage=hist.storage.Weight())
    histograms['mpho_gen_sublead_pt'] = Hist(samp, cut, ele_pt,        storage=hist.storage.Weight())
    histograms['mpho_gen_ee_dr']      = Hist(samp, cut, ee_dr,  storage=hist.storage.Weight())
    histograms['mpho_gen_ee_eta']     = Hist(samp, cut, ele_eta,       storage=hist.storage.Weight())
    histograms['mpho_gen_lead_eta']   = Hist(samp, cut, ele_eta,       storage=hist.storage.Weight())
    histograms['mpho_gen_sublead_eta']= Hist(samp, cut, ele_eta,       storage=hist.storage.Weight())

    # ── Merged-photon comparison hists: same reco variables (_PHO_HISTS), but
    # for (a) Photon objects in zero-match events, and (b) other (non-matched)
    # Photon objects present alongside the merged one.
    for field, axis in _PHO_HISTS:
        histograms[f'mpho_zero_{field}']  = Hist(samp, cut, axis, storage=hist.storage.Weight())
        histograms[f'mpho_other_{field}'] = Hist(samp, cut, axis, storage=hist.storage.Weight())

    # ── Merged-photon 2D correlations: gen Lxy vs shower-shape/ID variables ───
    histograms['mpho_gen_lxy_vs_HoE']   = Hist(samp, cut, _gen_lxy, _HoE,   storage=hist.storage.Weight())
    histograms['mpho_gen_lxy_vs_sieie'] = Hist(samp, cut, _gen_lxy, _sieie, storage=hist.storage.Weight())

    # ── 1D versions of the above, sliced above/below the Lxy cut ──────────────
    for prefix, field, axis in _LXY_SLICE_VARS:
        histograms[f'{prefix}_{field}{_LXY_HI_SUFFIX}'] = Hist(samp, cut, axis, storage=hist.storage.Weight())
        histograms[f'{prefix}_{field}{_LXY_LO_SUFFIX}'] = Hist(samp, cut, axis, storage=hist.storage.Weight())

    # ── Resolved category (AllLptElectron reco variables) ─────────────────────
    for field, axis in _ELE_HISTS:
        histograms[f'res_{field}'] = Hist(samp, cut, axis, storage=hist.storage.Weight())
    histograms['res_gen_lxy']        = Hist(samp, cut, _gen_lxy, storage=hist.storage.Weight())
    histograms['res_gen_ee_pt']      = Hist(samp, cut, ele_pt,      storage=hist.storage.Weight())
    histograms['res_gen_lead_pt']    = Hist(samp, cut, ele_pt,      storage=hist.storage.Weight())
    histograms['res_gen_sublead_pt'] = Hist(samp, cut, ele_pt,      storage=hist.storage.Weight())
    histograms['res_gen_ee_dr']      = Hist(samp, cut, ee_dr,       storage=hist.storage.Weight())
    histograms['res_gen_ee_eta']     = Hist(samp, cut, ele_eta,     storage=hist.storage.Weight())
    histograms['res_gen_lead_eta']   = Hist(samp, cut, ele_eta,     storage=hist.storage.Weight())
    histograms['res_gen_sublead_eta']= Hist(samp, cut, ele_eta,     storage=hist.storage.Weight())

    # ── Zero matches category (gen kinematics only — no matched reco object) ──
    histograms['zero_gen_lxy']        = Hist(samp, cut, _gen_lxy, storage=hist.storage.Weight())
    histograms['zero_gen_ee_pt']      = Hist(samp, cut, ele_pt,      storage=hist.storage.Weight())
    histograms['zero_gen_lead_pt']    = Hist(samp, cut, ele_pt,      storage=hist.storage.Weight())
    histograms['zero_gen_sublead_pt'] = Hist(samp, cut, ele_pt,      storage=hist.storage.Weight())
    histograms['zero_gen_ee_dr']      = Hist(samp, cut, ee_dr,       storage=hist.storage.Weight())
    histograms['zero_gen_ee_eta']     = Hist(samp, cut, ele_eta,     storage=hist.storage.Weight())
    histograms['zero_gen_lead_eta']   = Hist(samp, cut, ele_eta,     storage=hist.storage.Weight())
    histograms['zero_gen_sublead_eta']= Hist(samp, cut, ele_eta,     storage=hist.storage.Weight())

    return histograms

subroutines = []

# When True, fill histograms with raw event counts (weight=1) instead of the
# physical xsec/lumi/genWgt weight — useful for diagnosing bins dominated by
# a small number of high-weighted events (low MC effective statistics).
RAW_COUNTS = True

def fillHistos(events, hists, samp, cut, info, sum_wgt=1):
    if RAW_COUNTS:
        wgt = ak.ones_like(events.eventWgt)
    else:
        wgt = events.eventWgt / sum_wgt

    # ── Derived fields ────────────────────────────────────────────────────────
    events['AllLptElectron', 'mindRj']   = ak.fill_none(ak.min(events.AllLptElectron.dRJets,   axis=-1), 999)
    events['AllLptElectron', 'mindPhiJ'] = ak.fill_none(ak.min(events.AllLptElectron.dPhiJets, axis=-1), 999)

    # ── Gen kinematics ────────────────────────────────────────────────────────
    # gen_lxy is measured from the chi2 production vertex (the true, unsmeared
    # primary vertex) rather than the reconstructed PV, which carries ~10-15um
    # of resolution/bias that would otherwise leak into a "truth" quantity.
    chi2       = ak.firsts(events.GenPart[np.abs(events.GenPart.ID) == 1000023])
    gen_lxy    = np.sqrt((events.GenEle.vx - chi2.vx)**2 + (events.GenEle.vy - chi2.vy)**2)
    gen_sum_pt = events.GenEle.pt + events.GenPos.pt
    lxy_hi     = gen_lxy > _LXY_CUT_CM

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _dphi(a, b):
        d = np.abs(a - b)
        return ak.where(d > np.pi, 2 * np.pi - d, d)

    def _dr(eta1, phi1, eta2, phi2):
        return np.sqrt((eta1 - eta2)**2 + _dphi(phi1, phi2)**2)

    def _pt_rel(obj_pt, gen_pt):
        return np.abs(obj_pt - gen_pt) / gen_pt < 0.2

    # Photon has no precomputed dRJets/dPhiJets branch (unlike AllLptElectron),
    # so compute min dR/dPhi to the nearest PFJet directly.
    _pho_dphi_j = _dphi(events.Photon.phi[:, :, None], events.PFJet.phi[:, None, :])
    _pho_dr_j   = _dr(events.Photon.eta[:, :, None], events.Photon.phi[:, :, None],
                       events.PFJet.eta[:, None, :], events.PFJet.phi[:, None, :])
    events['Photon', 'mindRj']   = ak.fill_none(ak.min(_pho_dr_j,   axis=-1), 999)
    events['Photon', 'mindPhiJ'] = ak.fill_none(ak.min(_pho_dphi_j, axis=-1), 999)

    # ── Photon / OOT-photon / track / conversion matching to gen ─────────────
    # (Needed to determine whether gens have non-electron alternatives and for
    # the exclusivity condition that defines merged-photon category.)
    _pho_dr_ge    = _dr(events.Photon.eta,     events.Photon.phi,     events.GenEle.eta, events.GenEle.phi)
    _pho_dr_gp    = _dr(events.Photon.eta,     events.Photon.phi,     events.GenPos.eta, events.GenPos.phi)
    _oot_dr_ge    = _dr(events.ootPhoton.eta,  events.ootPhoton.phi,  events.GenEle.eta, events.GenEle.phi)
    _oot_dr_gp    = _dr(events.ootPhoton.eta,  events.ootPhoton.phi,  events.GenPos.eta, events.GenPos.phi)
    _trk_dr_ge    = _dr(events.IsoTrack.eta,   events.IsoTrack.phi,   events.GenEle.eta, events.GenEle.phi)
    _trk_dr_gp    = _dr(events.IsoTrack.eta,   events.IsoTrack.phi,   events.GenPos.eta, events.GenPos.phi)
    _conv_dr_ge   = _dr(events.Conversion.eta, events.Conversion.phi, events.GenEle.eta, events.GenEle.phi)
    _conv_dr_gp   = _dr(events.Conversion.eta, events.Conversion.phi, events.GenPos.eta, events.GenPos.phi)

    _pho_pt_ge    = _pt_rel(events.Photon.pt,     events.GenEle.pt)
    _pho_pt_gp    = _pt_rel(events.Photon.pt,     events.GenPos.pt)
    _pho_pt_sum   = _pt_rel(events.Photon.pt,     gen_sum_pt)
    _oot_pt_ge    = _pt_rel(events.ootPhoton.pt,  events.GenEle.pt)
    _oot_pt_gp    = _pt_rel(events.ootPhoton.pt,  events.GenPos.pt)
    _oot_pt_sum   = _pt_rel(events.ootPhoton.pt,  gen_sum_pt)
    _trk_pt_ge    = _pt_rel(events.IsoTrack.pt,   events.GenEle.pt)
    _trk_pt_gp    = _pt_rel(events.IsoTrack.pt,   events.GenPos.pt)
    _trk_pt_sum   = _pt_rel(events.IsoTrack.pt,   gen_sum_pt)
    _conv_pt_ge   = _pt_rel(events.Conversion.pt, events.GenEle.pt)
    _conv_pt_gp   = _pt_rel(events.Conversion.pt, events.GenPos.pt)
    _conv_pt_sum  = _pt_rel(events.Conversion.pt, gen_sum_pt)

    _ge_pho    = ak.any((_pho_dr_ge  < 0.1) & _pho_pt_ge,                                   axis=1)
    _ge_oot    = ak.any((_oot_dr_ge  < 0.1) & _oot_pt_ge,                                   axis=1)
    _ge_trk    = ak.any((_trk_dr_ge  < 0.1) & (events.IsoTrack.charge == -1) & _trk_pt_ge,  axis=1)
    _ge_conv   = ak.any((_conv_dr_ge < 0.1) & _conv_pt_ge,                                  axis=1)
    _gp_pho    = ak.any((_pho_dr_gp  < 0.1) & _pho_pt_gp,                                   axis=1)
    _gp_oot    = ak.any((_oot_dr_gp  < 0.1) & _oot_pt_gp,                                   axis=1)
    _gp_trk    = ak.any((_trk_dr_gp  < 0.1) & (events.IsoTrack.charge == +1) & _trk_pt_gp,  axis=1)
    _gp_conv   = ak.any((_conv_dr_gp < 0.1) & _conv_pt_gp,                                  axis=1)

    _ge_any = _ge_pho | _ge_oot | _ge_trk | _ge_conv
    _gp_any = _gp_pho | _gp_oot | _gp_trk | _gp_conv

    # Merged-photon detection: a single photon within dR < 0.1 of both gens,
    # pt-matching either gen or the system sum, with exclusivity (no other
    # photon/track closer to just one gen).
    _pho_both  = (_pho_dr_ge < 0.1) & (_pho_dr_gp < 0.1) & (_pho_pt_ge | _pho_pt_gp | _pho_pt_sum)
    _oot_both  = (_oot_dr_ge < 0.1) & (_oot_dr_gp < 0.1) & (_oot_pt_ge | _oot_pt_gp | _oot_pt_sum)
    _trk_both  = (_trk_dr_ge < 0.1) & (_trk_dr_gp < 0.1) & (_trk_pt_ge | _trk_pt_gp | _trk_pt_sum)
    _conv_both = (_conv_dr_ge < 0.1) & (_conv_dr_gp < 0.1) & (_conv_pt_ge | _conv_pt_gp | _conv_pt_sum)

    _pho_both_any = ak.any(_pho_both, axis=1)

    _photrk_excl = (
        (ak.sum((_pho_dr_ge  < 0.1) & (_pho_pt_ge  | _pho_pt_sum),  axis=1) == ak.sum(_pho_both,  axis=1)) &
        (ak.sum((_pho_dr_gp  < 0.1) & (_pho_pt_gp  | _pho_pt_sum),  axis=1) == ak.sum(_pho_both,  axis=1)) &
        (ak.sum((_oot_dr_ge  < 0.1) & (_oot_pt_ge  | _oot_pt_sum),  axis=1) == ak.sum(_oot_both,  axis=1)) &
        (ak.sum((_oot_dr_gp  < 0.1) & (_oot_pt_gp  | _oot_pt_sum),  axis=1) == ak.sum(_oot_both,  axis=1)) &
        (ak.sum((_trk_dr_ge  < 0.1) & (_trk_pt_ge  | _trk_pt_sum),  axis=1) == ak.sum(_trk_both,  axis=1)) &
        (ak.sum((_trk_dr_gp  < 0.1) & (_trk_pt_gp  | _trk_pt_sum),  axis=1) == ak.sum(_trk_both,  axis=1)) &
        (ak.sum((_conv_dr_ge < 0.1) & (_conv_pt_ge | _conv_pt_sum),  axis=1) == ak.sum(_conv_both, axis=1)) &
        (ak.sum((_conv_dr_gp < 0.1) & (_conv_pt_gp | _conv_pt_sum),  axis=1) == ak.sum(_conv_both, axis=1))
    )

    # ── AllLptElectron matching ────────────────────────────────────────────────
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

    only_ge_i = dr_match_ge & ~merged_i
    only_gp_i = dr_match_gp & ~merged_i
    matched_i = merged_i | dr_match_ge | dr_match_gp

    has_merged = ak.any(merged_i,  axis=1)
    n_matched  = ak.sum(matched_i, axis=1)

    _A = ak.any(only_ge_i, axis=1)
    _B = ak.any(only_gp_i, axis=1)

    _cat_sep    = (_A & _B & ~has_merged) | (has_merged & (n_matched > 1))
    _cat_merged = ~_cat_sep & has_merged & ~_ge_any & ~_gp_any
    _no_ele     = ~_cat_sep & ~_A & ~_B & ~has_merged

    _cat_merged_photon = _no_ele & _pho_both_any & _photrk_excl

    # ── Merged-electron fills ─────────────────────────────────────────────────
    if ak.sum(_cat_merged) > 0:
        me  = ak.firsts(coll[merged_i])[_cat_merged]
        w   = wgt[_cat_merged]
        lxy = gen_lxy[_cat_merged]
        ge  = events.GenEle[_cat_merged]
        gp  = events.GenPos[_cat_merged]

        lead_is_e   = ge.pt >= gp.pt
        lead_pt     = ak.where(lead_is_e, ge.pt,  gp.pt)
        sublead_pt  = ak.where(lead_is_e, gp.pt,  ge.pt)
        lead_eta    = ak.where(lead_is_e, ge.eta, gp.eta)
        sublead_eta = ak.where(lead_is_e, gp.eta, ge.eta)

        hists['mele_pt'               ].fill(samp=samp, cut=cut, pt=me.pt,                              weight=w)
        hists['mele_eta'              ].fill(samp=samp, cut=cut, eta=me.eta,                            weight=w)
        hists['mele_phi'              ].fill(samp=samp, cut=cut, phi=me.phi,                            weight=w)
        hists['mele_e'                ].fill(samp=samp, cut=cut, e=me.e,                                weight=w)
        hists['mele_ID'               ].fill(samp=samp, cut=cut, ele_id=me.ID,                          weight=w)
        hists['mele_angRes'           ].fill(samp=samp, cut=cut, angRes=me.angRes,                      weight=w)
        hists['mele_vxy'              ].fill(samp=samp, cut=cut, vxy=np.abs(me.vxy),                    weight=w)
        hists['mele_vz'               ].fill(samp=samp, cut=cut, vz=np.abs(me.vz),                     weight=w)
        hists['mele_dxy'              ].fill(samp=samp, cut=cut, dxy=np.abs(me.dxy),                    weight=w)
        hists['mele_dz'               ].fill(samp=samp, cut=cut, dz=np.abs(me.dz),                     weight=w)
        hists['mele_trkChi2'          ].fill(samp=samp, cut=cut, chi2=me.trkChi2,                      weight=w)
        hists['mele_trkIso'           ].fill(samp=samp, cut=cut, trkIso=me.trkIso,                     weight=w)
        hists['mele_trkRelIso'        ].fill(samp=samp, cut=cut, relIso=me.trkRelIso,                  weight=w)
        hists['mele_calIso'           ].fill(samp=samp, cut=cut, iso=me.calIso,                        weight=w)
        hists['mele_calRelIso'        ].fill(samp=samp, cut=cut, relIso=me.calRelIso,                  weight=w)
        hists['mele_PFIso'            ].fill(samp=samp, cut=cut, iso=me.PFIso,                         weight=w)
        hists['mele_PFRelIso'         ].fill(samp=samp, cut=cut, relIso=me.PFRelIso,                   weight=w)
        hists['mele_miniIso'          ].fill(samp=samp, cut=cut, iso=me.miniIso,                       weight=w)
        hists['mele_miniRelIso'       ].fill(samp=samp, cut=cut, iso=me.miniRelIso,                    weight=w)
        hists['mele_PFIsoEleCorr'     ].fill(samp=samp, cut=cut, iso=me.PFIsoEleCorr,                  weight=w)
        hists['mele_PFRelIsoEleCorr'  ].fill(samp=samp, cut=cut, relIso=me.PFRelIsoEleCorr,            weight=w)
        hists['mele_miniIsoEleCorr'   ].fill(samp=samp, cut=cut, iso=me.miniIsoEleCorr,                weight=w)
        hists['mele_miniRelIsoEleCorr'].fill(samp=samp, cut=cut, iso=me.miniRelIsoEleCorr,             weight=w)
        hists['mele_chadIso'          ].fill(samp=samp, cut=cut, iso=me.chadIso,                       weight=w)
        hists['mele_nhadIso'          ].fill(samp=samp, cut=cut, iso=me.nhadIso,                       weight=w)
        hists['mele_phoIso'           ].fill(samp=samp, cut=cut, iso=me.phoIso,                        weight=w)
        hists['mele_rhoEA'            ].fill(samp=samp, cut=cut, rhoEA=me.rhoEA,                       weight=w)
        hists['mele_trkProb'          ].fill(samp=samp, cut=cut, prob=me.trkProb,                      weight=w)
        hists['mele_numTrackerHits'   ].fill(samp=samp, cut=cut, numTrkHits=me.numTrackerHits,         weight=w)
        hists['mele_numPixHits'       ].fill(samp=samp, cut=cut, numPixHits=me.numPixHits,             weight=w)
        hists['mele_numStripHits'     ].fill(samp=samp, cut=cut, numStripHits=me.numStripHits,         weight=w)
        hists['mele_charge'           ].fill(samp=samp, cut=cut, charge=me.charge,                     weight=w)
        hists['mele_minDRtoReg'       ].fill(samp=samp, cut=cut, dr=me.minDRtoReg,                     weight=w)
        hists['mele_mindRj'           ].fill(samp=samp, cut=cut, drj=me.mindRj,                        weight=w)
        hists['mele_mindPhiJ'         ].fill(samp=samp, cut=cut, dphiJ=me.mindPhiJ,                    weight=w)
        hists['mele_full55sigmaIetaIeta'].fill(samp=samp, cut=cut, sieie=me.full55sigmaIetaIeta,       weight=w)
        hists['mele_absdEtaSeed'      ].fill(samp=samp, cut=cut, dEtaSeed=me.absdEtaSeed,              weight=w)
        hists['mele_absdPhiIn'        ].fill(samp=samp, cut=cut, dPhiIn=me.absdPhiIn,                  weight=w)
        hists['mele_HoverE'           ].fill(samp=samp, cut=cut, HoE=me.HoverE,                        weight=w)
        hists['mele_abs1overEm1overP' ].fill(samp=samp, cut=cut, invEmP=me.abs1overEm1overP,           weight=w)
        hists['mele_expMissingInnerHits'].fill(samp=samp, cut=cut, missHits=me.expMissingInnerHits,    weight=w)
        hists['mele_IDscore'          ].fill(samp=samp, cut=cut, ele_id=me.ID,                         weight=w)

        hists['mele_gen_lxy'        ].fill(samp=samp, cut=cut, lxy=lxy,                                weight=w)
        hists['mele_gen_ee_pt'      ].fill(samp=samp, cut=cut, pt=events.genEE.pt[_cat_merged],        weight=w)
        hists['mele_gen_lead_pt'    ].fill(samp=samp, cut=cut, pt=lead_pt,                             weight=w)
        hists['mele_gen_sublead_pt' ].fill(samp=samp, cut=cut, pt=sublead_pt,                          weight=w)
        hists['mele_gen_ee_dr'      ].fill(samp=samp, cut=cut, dr=events.genEE.dr[_cat_merged],        weight=w)
        hists['mele_gen_ee_eta'     ].fill(samp=samp, cut=cut, eta=events.genEE.eta[_cat_merged],       weight=w)
        hists['mele_gen_lead_eta'   ].fill(samp=samp, cut=cut, eta=lead_eta,                           weight=w)
        hists['mele_gen_sublead_eta'].fill(samp=samp, cut=cut, eta=sublead_eta,                        weight=w)

        # dR from the *second* (unmatched) gen electron to other reco object
        # types. Tighter match than merged_i itself: dr<0.1, charge match, and
        # delta_pt_rel<0.1 (vs the 0.2 used for the merged_i selection) picks
        # out which gen the reco electron corresponds to.
        dr_me_ge   = _dr(me.eta, me.phi, ge.eta, ge.phi)
        dr_me_gp   = _dr(me.eta, me.phi, gp.eta, gp.phi)
        dpt_rel_ge = np.abs(me.pt - ge.pt) / ge.pt
        dpt_rel_gp = np.abs(me.pt - gp.pt) / gp.pt

        matched_ge = (dr_me_ge < 0.1) & (dpt_rel_ge < 0.1) & (me.charge == -1)
        matched_gp = (dr_me_gp < 0.1) & (dpt_rel_gp < 0.1) & (me.charge == +1)
        _has_tight_match = matched_ge | matched_gp

        if ak.sum(_has_tight_match) > 0:
            other_eta = ak.where(matched_ge, gp.eta, ge.eta)[_has_tight_match]
            other_phi = ak.where(matched_ge, gp.phi, ge.phi)[_has_tight_match]
            w_other   = w[_has_tight_match]

            for _suffix, _field in _OTHERGEN_DR_TYPES:
                obj    = getattr(events, _field)[_cat_merged][_has_tight_match]
                obj_dr = ak.fill_none(
                    ak.min(_dr(obj.eta, obj.phi, other_eta, other_phi), axis=1), 999
                )
                hists[f'mele_othergen_dr_{_suffix}'].fill(samp=samp, cut=cut, dr=obj_dr, weight=w_other)

        # Other (non-matched) AllLptElectron objects present in these events.
        _fill_multi(hists, 'mele_other', coll[~merged_i][_cat_merged], w, samp, cut, _ELE_HISTS)

    # ── Merged-photon fills ───────────────────────────────────────────────────
    if ak.sum(_cat_merged_photon) > 0:
        mp  = ak.firsts(events.Photon[_pho_both])[_cat_merged_photon]
        w   = wgt[_cat_merged_photon]
        lxy = gen_lxy[_cat_merged_photon]
        ge  = events.GenEle[_cat_merged_photon]
        gp  = events.GenPos[_cat_merged_photon]

        lead_is_e   = ge.pt >= gp.pt
        lead_pt     = ak.where(lead_is_e, ge.pt,  gp.pt)
        sublead_pt  = ak.where(lead_is_e, gp.pt,  ge.pt)
        lead_eta    = ak.where(lead_is_e, ge.eta, gp.eta)
        sublead_eta = ak.where(lead_is_e, gp.eta, ge.eta)

        hists['mpho_pt'          ].fill(samp=samp, cut=cut, pt=mp.pt,              weight=w)
        hists['mpho_eta'         ].fill(samp=samp, cut=cut, eta=mp.eta,            weight=w)
        hists['mpho_phi'         ].fill(samp=samp, cut=cut, phi=mp.phi,            weight=w)
        hists['mpho_energy'      ].fill(samp=samp, cut=cut, e=mp.energy,           weight=w)
        hists['mpho_r9'          ].fill(samp=samp, cut=cut, r9=mp.r9,              weight=w)
        hists['mpho_full5x5_r9'  ].fill(samp=samp, cut=cut, r9=mp.full5x5_r9,     weight=w)
        hists['mpho_sIeIe'       ].fill(samp=samp, cut=cut, sieie=mp.sIeIe,        weight=w)
        hists['mpho_full5x5_sIeIe'].fill(samp=samp, cut=cut, sieie=mp.full5x5_sIeIe, weight=w)
        hists['mpho_HoE'         ].fill(samp=samp, cut=cut, HoE=mp.HoE,            weight=w)
        hists['mpho_full5x5_HoE' ].fill(samp=samp, cut=cut, HoE=mp.full5x5_HoE,   weight=w)
        hists['mpho_chIso'       ].fill(samp=samp, cut=cut, iso=mp.chIso,          weight=w)
        hists['mpho_nhIso'       ].fill(samp=samp, cut=cut, iso=mp.nhIso,          weight=w)
        hists['mpho_phIso'       ].fill(samp=samp, cut=cut, iso=mp.phIso,          weight=w)
        hists['mpho_puChIso'     ].fill(samp=samp, cut=cut, iso=mp.puChIso,        weight=w)
        hists['mpho_trkIso'      ].fill(samp=samp, cut=cut, trkIso=mp.trkIso,      weight=w)
        hists['mpho_ecalIso'     ].fill(samp=samp, cut=cut, iso=mp.ecalIso,        weight=w)
        hists['mpho_hcalIso'     ].fill(samp=samp, cut=cut, iso=mp.hcalIso,        weight=w)
        hists['mpho_mindRj'      ].fill(samp=samp, cut=cut, drj=mp.mindRj,        weight=w)
        hists['mpho_mindPhiJ'    ].fill(samp=samp, cut=cut, dphiJ=mp.mindPhiJ,    weight=w)

        hists['mpho_gen_lxy'        ].fill(samp=samp, cut=cut, lxy=lxy,                                      weight=w)
        hists['mpho_gen_lxy_vs_HoE']  .fill(samp=samp, cut=cut, lxy=lxy, HoE=mp.HoE,           weight=w)
        hists['mpho_gen_lxy_vs_sieie'].fill(samp=samp, cut=cut, lxy=lxy, sieie=mp.full5x5_sIeIe, weight=w)
        hists['mpho_gen_ee_pt'      ].fill(samp=samp, cut=cut, pt=events.genEE.pt[_cat_merged_photon],       weight=w)
        hists['mpho_gen_lead_pt'    ].fill(samp=samp, cut=cut, pt=lead_pt,                                   weight=w)
        hists['mpho_gen_sublead_pt' ].fill(samp=samp, cut=cut, pt=sublead_pt,                                weight=w)
        hists['mpho_gen_ee_dr'      ].fill(samp=samp, cut=cut, dr=events.genEE.dr[_cat_merged_photon],       weight=w)
        hists['mpho_gen_ee_eta'     ].fill(samp=samp, cut=cut, eta=events.genEE.eta[_cat_merged_photon],      weight=w)
        hists['mpho_gen_lead_eta'   ].fill(samp=samp, cut=cut, eta=lead_eta,                                 weight=w)
        hists['mpho_gen_sublead_eta'].fill(samp=samp, cut=cut, eta=sublead_eta,                              weight=w)

        # HoE/sieie versions sliced above/below the Lxy cut (mirrors the 2D
        # mpho_gen_lxy_vs_HoE / mpho_gen_lxy_vs_sieie correlation histograms).
        _mpho_hi = lxy_hi[_cat_merged_photon]
        _fill_lxy_slice(hists, 'mpho', mp, _mpho_hi,  w, samp, cut, _LXY_HI_SUFFIX)
        _fill_lxy_slice(hists, 'mpho', mp, ~_mpho_hi, w, samp, cut, _LXY_LO_SUFFIX)

        # Other (non-matched) Photon objects present in these events.
        _fill_multi(hists, 'mpho_other', events.Photon[~_pho_both][_cat_merged_photon], w, samp, cut, _PHO_HISTS)

    # ── Resolved fills ─────────────────────────────────────────────────────────
    # Gen kinematics for all resolved events (classical + ambiguous).
    if ak.sum(_cat_sep) > 0:
        w_res   = wgt[_cat_sep]
        lxy_res = gen_lxy[_cat_sep]
        ge_res  = events.GenEle[_cat_sep]
        gp_res  = events.GenPos[_cat_sep]

        lead_is_e_res   = ge_res.pt >= gp_res.pt
        lead_pt_res     = ak.where(lead_is_e_res, ge_res.pt,  gp_res.pt)
        sublead_pt_res  = ak.where(lead_is_e_res, gp_res.pt,  ge_res.pt)
        lead_eta_res    = ak.where(lead_is_e_res, ge_res.eta, gp_res.eta)
        sublead_eta_res = ak.where(lead_is_e_res, gp_res.eta, ge_res.eta)

        hists['res_gen_lxy'        ].fill(samp=samp, cut=cut, lxy=lxy_res,                          weight=w_res)
        hists['res_gen_ee_pt'      ].fill(samp=samp, cut=cut, pt=events.genEE.pt[_cat_sep],         weight=w_res)
        hists['res_gen_lead_pt'    ].fill(samp=samp, cut=cut, pt=lead_pt_res,                       weight=w_res)
        hists['res_gen_sublead_pt' ].fill(samp=samp, cut=cut, pt=sublead_pt_res,                    weight=w_res)
        hists['res_gen_ee_dr'      ].fill(samp=samp, cut=cut, dr=events.genEE.dr[_cat_sep],         weight=w_res)
        hists['res_gen_ee_eta'     ].fill(samp=samp, cut=cut, eta=events.genEE.eta[_cat_sep],        weight=w_res)
        hists['res_gen_lead_eta'   ].fill(samp=samp, cut=cut, eta=lead_eta_res,                     weight=w_res)
        hists['res_gen_sublead_eta'].fill(samp=samp, cut=cut, eta=sublead_eta_res,                  weight=w_res)

        # Reco fills: closest matched reco to each gen particle.
        # Covers both classical (exclusive one-to-one) and ambiguous (merged + extra) subcases.
        dr_ge_masked = ak.where(matched_i, dr_to_ge, np.inf)
        dr_gp_masked = ak.where(matched_i, dr_to_gp, np.inf)
        ge_reco = ak.firsts(coll[ak.argmin(dr_ge_masked, axis=1, keepdims=True)])[_cat_sep]
        gp_reco = ak.firsts(coll[ak.argmin(dr_gp_masked, axis=1, keepdims=True)])[_cat_sep]

        for re in [ge_reco, gp_reco]:
            hists['res_pt'               ].fill(samp=samp, cut=cut, pt=re.pt,                              weight=w_res)
            hists['res_eta'              ].fill(samp=samp, cut=cut, eta=re.eta,                            weight=w_res)
            hists['res_phi'              ].fill(samp=samp, cut=cut, phi=re.phi,                            weight=w_res)
            hists['res_e'                ].fill(samp=samp, cut=cut, e=re.e,                                weight=w_res)
            hists['res_ID'               ].fill(samp=samp, cut=cut, ele_id=re.ID,                          weight=w_res)
            hists['res_angRes'           ].fill(samp=samp, cut=cut, angRes=re.angRes,                      weight=w_res)
            hists['res_vxy'              ].fill(samp=samp, cut=cut, vxy=np.abs(re.vxy),                    weight=w_res)
            hists['res_vz'               ].fill(samp=samp, cut=cut, vz=np.abs(re.vz),                     weight=w_res)
            hists['res_dxy'              ].fill(samp=samp, cut=cut, dxy=np.abs(re.dxy),                    weight=w_res)
            hists['res_dz'               ].fill(samp=samp, cut=cut, dz=np.abs(re.dz),                     weight=w_res)
            hists['res_trkChi2'          ].fill(samp=samp, cut=cut, chi2=re.trkChi2,                      weight=w_res)
            hists['res_trkIso'           ].fill(samp=samp, cut=cut, trkIso=re.trkIso,                     weight=w_res)
            hists['res_trkRelIso'        ].fill(samp=samp, cut=cut, relIso=re.trkRelIso,                  weight=w_res)
            hists['res_calIso'           ].fill(samp=samp, cut=cut, iso=re.calIso,                        weight=w_res)
            hists['res_calRelIso'        ].fill(samp=samp, cut=cut, relIso=re.calRelIso,                  weight=w_res)
            hists['res_PFIso'            ].fill(samp=samp, cut=cut, iso=re.PFIso,                         weight=w_res)
            hists['res_PFRelIso'         ].fill(samp=samp, cut=cut, relIso=re.PFRelIso,                   weight=w_res)
            hists['res_miniIso'          ].fill(samp=samp, cut=cut, iso=re.miniIso,                       weight=w_res)
            hists['res_miniRelIso'       ].fill(samp=samp, cut=cut, iso=re.miniRelIso,                    weight=w_res)
            hists['res_PFIsoEleCorr'     ].fill(samp=samp, cut=cut, iso=re.PFIsoEleCorr,                  weight=w_res)
            hists['res_PFRelIsoEleCorr'  ].fill(samp=samp, cut=cut, relIso=re.PFRelIsoEleCorr,            weight=w_res)
            hists['res_miniIsoEleCorr'   ].fill(samp=samp, cut=cut, iso=re.miniIsoEleCorr,                weight=w_res)
            hists['res_miniRelIsoEleCorr'].fill(samp=samp, cut=cut, iso=re.miniRelIsoEleCorr,             weight=w_res)
            hists['res_chadIso'          ].fill(samp=samp, cut=cut, iso=re.chadIso,                       weight=w_res)
            hists['res_nhadIso'          ].fill(samp=samp, cut=cut, iso=re.nhadIso,                       weight=w_res)
            hists['res_phoIso'           ].fill(samp=samp, cut=cut, iso=re.phoIso,                        weight=w_res)
            hists['res_rhoEA'            ].fill(samp=samp, cut=cut, rhoEA=re.rhoEA,                       weight=w_res)
            hists['res_trkProb'          ].fill(samp=samp, cut=cut, prob=re.trkProb,                      weight=w_res)
            hists['res_numTrackerHits'   ].fill(samp=samp, cut=cut, numTrkHits=re.numTrackerHits,         weight=w_res)
            hists['res_numPixHits'       ].fill(samp=samp, cut=cut, numPixHits=re.numPixHits,             weight=w_res)
            hists['res_numStripHits'     ].fill(samp=samp, cut=cut, numStripHits=re.numStripHits,         weight=w_res)
            hists['res_charge'           ].fill(samp=samp, cut=cut, charge=re.charge,                     weight=w_res)
            hists['res_minDRtoReg'       ].fill(samp=samp, cut=cut, dr=re.minDRtoReg,                     weight=w_res)
            hists['res_mindRj'           ].fill(samp=samp, cut=cut, drj=re.mindRj,                        weight=w_res)
            hists['res_mindPhiJ'         ].fill(samp=samp, cut=cut, dphiJ=re.mindPhiJ,                    weight=w_res)
            hists['res_full55sigmaIetaIeta'].fill(samp=samp, cut=cut, sieie=re.full55sigmaIetaIeta,       weight=w_res)
            hists['res_absdEtaSeed'      ].fill(samp=samp, cut=cut, dEtaSeed=re.absdEtaSeed,              weight=w_res)
            hists['res_absdPhiIn'        ].fill(samp=samp, cut=cut, dPhiIn=re.absdPhiIn,                  weight=w_res)
            hists['res_HoverE'           ].fill(samp=samp, cut=cut, HoE=re.HoverE,                        weight=w_res)
            hists['res_abs1overEm1overP' ].fill(samp=samp, cut=cut, invEmP=re.abs1overEm1overP,           weight=w_res)
            hists['res_expMissingInnerHits'].fill(samp=samp, cut=cut, missHits=re.expMissingInnerHits,    weight=w_res)
            hists['res_IDscore'          ].fill(samp=samp, cut=cut, ele_id=re.ID,                         weight=w_res)

    # ── Zero matches fills ─────────────────────────────────────────────────────
    # Definition from mergedmatch.py: no electron matched to either gen AND
    # neither gen has a photon/track/conversion alternative.
    _cat_zero = _no_ele & ~_ge_any & ~_gp_any
    if ak.sum(_cat_zero) > 0:
        w_z   = wgt[_cat_zero]
        lxy_z = gen_lxy[_cat_zero]
        ge_z  = events.GenEle[_cat_zero]
        gp_z  = events.GenPos[_cat_zero]

        lead_is_e_z   = ge_z.pt >= gp_z.pt
        lead_pt_z     = ak.where(lead_is_e_z, ge_z.pt,  gp_z.pt)
        sublead_pt_z  = ak.where(lead_is_e_z, gp_z.pt,  ge_z.pt)
        lead_eta_z    = ak.where(lead_is_e_z, ge_z.eta, gp_z.eta)
        sublead_eta_z = ak.where(lead_is_e_z, gp_z.eta, ge_z.eta)

        hists['zero_gen_lxy'        ].fill(samp=samp, cut=cut, lxy=lxy_z,                          weight=w_z)
        hists['zero_gen_ee_pt'      ].fill(samp=samp, cut=cut, pt=events.genEE.pt[_cat_zero],      weight=w_z)
        hists['zero_gen_lead_pt'    ].fill(samp=samp, cut=cut, pt=lead_pt_z,                       weight=w_z)
        hists['zero_gen_sublead_pt' ].fill(samp=samp, cut=cut, pt=sublead_pt_z,                    weight=w_z)
        hists['zero_gen_ee_dr'      ].fill(samp=samp, cut=cut, dr=events.genEE.dr[_cat_zero],      weight=w_z)
        hists['zero_gen_ee_eta'     ].fill(samp=samp, cut=cut, eta=events.genEE.eta[_cat_zero],     weight=w_z)
        hists['zero_gen_lead_eta'   ].fill(samp=samp, cut=cut, eta=lead_eta_z,                     weight=w_z)
        hists['zero_gen_sublead_eta'].fill(samp=samp, cut=cut, eta=sublead_eta_z,                  weight=w_z)

        # AllLptElectron objects present in these (otherwise unmatched) events,
        # split into the leading-pt electron (zerolead) and any other
        # electrons in the same event (zeroothers).
        coll_z      = coll[_cat_zero]
        order_z     = ak.argsort(coll_z.pt, axis=1, ascending=False)
        coll_z_sort = coll_z[order_z]
        has_lead    = ak.num(coll_z_sort, axis=1) > 0
        if ak.sum(has_lead) > 0:
            lead_z   = ak.firsts(coll_z_sort)[has_lead]
            others_z = coll_z_sort[:, 1:][has_lead]
            w_lead_z = w_z[has_lead]
            _fill_single(hists, 'mele_zerolead',   lead_z,   w_lead_z, samp, cut, _ELE_HISTS)
            _fill_multi (hists, 'mele_zeroothers', others_z, w_lead_z, samp, cut, _ELE_HISTS)

        # Same-type Photon objects present in these (otherwise unmatched) events.
        _fill_multi(hists, 'mpho_zero', events.Photon[_cat_zero],  w_z, samp, cut, _PHO_HISTS)
