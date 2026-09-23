from hist import Hist
from hist.axis import Variable, Regular, StrCategory, IntCategory, Integer
import hist
import numpy as np
import awkward as ak

# When True, fill histograms with raw event counts (weight=1) instead of the
# physical xsec/lumi/genWgt weight -- useful for diagnosing bins dominated by
# a small number of high-weighted events (low MC effective statistics).
RAW_COUNTS = True

# With RAW_COUNTS=True every fill weight is exactly 1, so a Weight() storage's
# sumw2 accumulator is just a duplicate of sumw -- use the cheaper Double()
# storage in that case. Falls back to Weight() (proper variance tracking) if
# RAW_COUNTS is ever set back to False.
_STORAGE = hist.storage.Double() if RAW_COUNTS else hist.storage.Weight()

# ── Axes (copied from configs/histo_configs/histobins.py, standalone so this
# module carries no dependency on it; rebinned to ~50 bins each) ─────────────
samp = StrCategory([], name="samp", label="Sample Name", growth=True)
cut  = StrCategory([], name="cut",  label="Cut Applied",  growth=True)

vtx_sign = IntCategory([-1, 1], name="sign", label="Vertex sign (q1*q2)")

ele_pt        = Regular(50, 0,   50,  name="pt",          label="$p_{T}$ [GeV]")
ele_eta       = Regular(50, -3,   3,  name="eta",         label=r"$\eta$")
ele_phi       = Regular(50, -3.2, 3.2, name="phi",        label=r"$\phi$")
ele_trkHits   = Regular(50, 0,   30,  name="numTrkHits",   label="Number of Tracker Hits")
ele_pixHits   = Regular(50, 0,   10,  name="numPixHits",   label="Number of Pixel Hits")
ele_stripHits = Regular(50, 0,   25,  name="numStripHits", label="Number of Strip Hits")
ele_chi2      = Regular(50, 0,  100,  name="chi2",         label=r"Track $\chi^2/df$")
ele_trkIso    = Regular(50, 0,  100,  name="trkIso",       label="Tracker Iso")
ele_trkRelIso = Regular(50, 0,    5,  name="relIso",       label="Tracker Relative Iso")
ele_PFRelIso  = Regular(50, 0,   10,  name="relIso",       label="PF Relative Iso")
ele_PFIso     = Regular(50, 0,   20,  name="iso",          label="PF Isolation")
ele_miniIso        = Regular(50, 0, 100, name="iso", label="Mini Iso")
ele_miniRelIso     = Regular(50, 0,  10, name="iso", label="Mini Relative Iso")
ele_miniIsoCorr    = Regular(50, 0, 100, name="iso", label="Corrected Mini Iso")
ele_miniRelIsoCorr = Regular(50, 0,  10, name="iso", label="Corrected Mini Relative Iso")
ele_prob    = Regular(50, 0, 1,   name="prob",   label=r"Electron Track $\chi^2$ Probability")
ele_angRes  = Regular(50, 0, 0.1, name="angRes", label=r"Angular Resolution $\sqrt{\sigma_\eta^2 + \sigma_\phi^2}$")
ele_dxy     = Regular(50, 0, 40,  name="dxy",    label="Electron Track $d_{xy}$ [cm]")
ele_dz      = Regular(50, 0, 5,   name="dz",     label="Electron Track $d_{z}$ [cm]")
ele_id      = Regular(50, -1, 4,  name="ele_id", label="Low $p_T$ electron ID Score")

ee_dr = Regular(50, 0, 1, name='dr', label=r"$\Delta R$")

dphiJ    = Regular(50, 0, 3.2, name="dphiJ",   label=r"$\Delta \phi$")
dR       = Regular(50, 0, 6,   name='dr',      label=r"$\Delta R$")
dR_zoom  = Regular(50, 0, 1,   name='dr_zoom', label=r"$\Delta R$")
dRj      = Regular(50, 0, 6,   name='drj',     label=r"$\Delta R$")

# ── Extra axes not in histobins ──────────────────────────────────────────────
_e_energy  = Regular(50,  0, 100,  name='e',        label='E [GeV]')
_calIso    = Regular(50,  0, 100,  name='iso',       label='Calo Iso [GeV]')
_calRelIso = Regular(50,  0,   5,  name='relIso',    label='Calo Relative Iso')
_rhoEA     = Regular(50,  0,  20,  name='rhoEA',     label=r'$\rho \times EA$ [GeV]')
_sieie     = Regular(50,  0,0.05,  name='sieie',     label=r'$\sigma_{i\eta i\eta}$')
_dEtaSeed  = Regular(50,  0,0.01,  name='dEtaSeed',  label=r'$|\Delta\eta_\mathrm{seed}|$')
_dPhiIn    = Regular(50,  0, 0.1,  name='dPhiIn',    label=r'$|\Delta\phi_\mathrm{in}|$')
_HoE       = Regular(50,  0,   1,  name='HoE',       label='H/E')
_invEmP    = Regular(50,  0, 0.1,  name='invEmP',    label=r'$|1/E - 1/p|$ [GeV$^{-1}$]')
_missHits  = Integer(0,    5,       name='missHits',  label='Exp. Missing Inner Hits')
_charge    = IntCategory([-1, 1],   name='charge',    label='Charge')
_r9        = Regular(50,  0, 1.5,  name='r9',        label='R9')
_drNearestEle = Regular(50, 0, 6,  name='drNearestEle', label=r'Min $\Delta R(e, e_{\mathrm{other\ lpt}})$')
_angRes    = Regular(50,0,0.02,name="angRes",label=r"Angular Resolution $\sqrt{\sigma_\eta^2 + \sigma_\phi^2}$")

# Relational axes: selected merged electron vs. its nearest other AllLptElectron
# -- large dPt/dVxy/dVz/opposite charge point to a genuine second electron,
# while near-zero values across the board point to a duplicate GSF track.
_dPt      = Regular(50, -50, 50, name='dpt',    label=r'$p_T(e_{\mathrm{sel}}) - p_T(e_{\mathrm{near}})$ [GeV]')
_dPtRel   = Regular(50,  -2,  2, name='dptRel', label=r'$(p_T(e_{\mathrm{sel}}) - p_T(e_{\mathrm{near}}))/p_T(e_{\mathrm{sel}})$')
_dVxyNear = Regular(50,   0, 10, name='dvxy',   label=r'$|v_{xy}(e_{\mathrm{sel}}) - v_{xy}(e_{\mathrm{near}})|$ [cm]')
_dVzNear  = Regular(50,   0, 10, name='dvz',    label=r'$|v_z(e_{\mathrm{sel}}) - v_z(e_{\mathrm{near}})|$ [cm]')
_dDxyNear = Regular(50,   0,  5, name='ddxy',   label=r'$|d_{xy}(e_{\mathrm{sel}}) - d_{xy}(e_{\mathrm{near}})|$ [cm]')
_dDzNear  = Regular(50,   0,  5, name='ddz',    label=r'$|d_z(e_{\mathrm{sel}}) - d_z(e_{\mathrm{near}})|$ [cm]')

# Diagnostic for the chargeProd==-1 (oppositely charged) subset of
# mele_nearestEle: the near track's *only* possible gen match is the lepton
# sharing its own charge sign ("target gen" -- gen ele if near.charge==-1,
# gen pos if near.charge==+1). If it also had dR<0.1 and |dPt|/pt<0.2 to that
# gen, it would already have satisfied dr_match_ge/dr_match_gp on its own and
# the event would have been pulled into the ambiguous/resolved (_cat_sep)
# category instead of landing here as "merged" -- these hists show which of
# those two cuts (if either) it actually failed.
_dPtRelTargetGen = Regular(
    50, -2, 2, name='dptRelTargetGen',
    label=r'$(p_T(e_{\mathrm{near}}) - p_T(\mathrm{gen}_{\mathrm{target}}))/p_T(\mathrm{gen}_{\mathrm{target}})$',
)
_oppQFailReason = StrCategory(
    ['fail_dR_only', 'fail_pt_only', 'fail_both', 'pass_both_unexpected'],
    name='reason', label='Why not gen-matched to target lepton',
)

# Same variable-width style as histobins.py's ele_lxy_res, extended out to 150 cm.
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
    #('vxy',                 vxy_coarse),
    #('vz',                  vz_coarse),
    ('dxy',                 ele_dxy),
    ('dz',                  ele_dz),
    #('trkChi2',             ele_chi2),
    #('trkIso',              ele_trkIso),
    #('trkRelIso',           ele_trkRelIso),
    #('calIso',              _calIso),
    #('calRelIso',           _calRelIso),
    #('PFIso',               ele_PFIso),
    #('PFRelIso',            ele_PFRelIso),
    #('miniIso',             ele_miniIso),
    #('miniRelIso',          ele_miniRelIso),
    #('PFIsoEleCorr',        ele_PFIso),
    #('PFRelIsoEleCorr',     ele_PFRelIso),
    #('miniIsoEleCorr',      ele_miniIsoCorr),
    #('miniRelIsoEleCorr',   ele_miniRelIsoCorr),
    #('chadIso',             ele_PFIso),
    #('nhadIso',             ele_PFIso),
    #('phoIso',              ele_PFIso),
    #('rhoEA',               _rhoEA),
    #('trkProb',             ele_prob),
    ('numTrackerHits',      ele_trkHits),
    ('numPixHits',          ele_pixHits),
    ('numStripHits',        ele_stripHits),
    ('charge',              _charge),
    ('minDRtoReg',          dR),
    ('drNearestEle',        _drNearestEle),
    ('mindRj',              dRj),
    ('mindPhiJ',            dphiJ),
    ('full5x5sigmaIetaIeta', _sieie),
    #('absdEtaSeed',         _dEtaSeed),
    #('absdPhiIn',           _dPhiIn),
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
    ('full5x5sigmaIetaIeta', _sieie),
    ('HoE',          _HoE),
    ('full5x5_HoE',  _HoE),
    #('chIso',        ele_PFIso),
    #('nhIso',        ele_PFIso),
    #('phIso',        ele_PFIso),
    #('puChIso',      ele_PFIso),
    #('trkIso',       ele_trkIso),
    #('ecalIso',      ele_PFIso),
    #('hcalIso',      ele_PFIso),
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
    ('mpho', 'HoE',                  _HoE),
    ('mpho', 'full5x5sigmaIetaIeta', _sieie),
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
            **{axis.name: getattr(obj, _PHO_FIELD_ALIASES.get(field, field))[mask]},
        )

# Fields that are filled as np.abs(...) in the matched mele_* hists (signed
# in the ntuple, but histogrammed unsigned) — replicate that here so the
# zero/other comparison hists land on the same axis convention.
_ABS_FIELDS = {'vxy', 'vz', 'dxy', 'dz'}

# _ELE_HISTS has two hist-key entries ('ID' and 'IDscore') that both read the
# same underlying 'ID' field (mirrors the existing mele_ID/mele_IDscore fills,
# which both pull from me.ID) — 'IDscore' isn't itself a field on the ntuple.
# full5x5sigmaIetaIeta is likewise a hist-key alias, not an AllLptElectron
# field on its own -- the real (legacy-named) field is full55sigmaIetaIeta;
# renamed at the hist-key level only so it matches the mpho-side name for the
# same quantity (see _PHO_FIELD_ALIASES) and can be overlaid/compared 1:1.
_FIELD_ALIASES = {'IDscore': 'ID', 'full5x5sigmaIetaIeta': 'full55sigmaIetaIeta'}

# _PHO_HISTS' equivalent of _FIELD_ALIASES, kept separate because the two
# object types alias the *same* new hist-key ('full5x5sigmaIetaIeta') to
# *different* real fields -- and because Photon has no analog of the
# ID/IDscore case, so it can't just reuse _FIELD_ALIASES as-is.
_PHO_FIELD_ALIASES = {'full5x5sigmaIetaIeta': 'full5x5_sIeIe'}

def _fill_multi(hists, prefix, obj, evt_weight, samp, cut, field_hists, aliases=_FIELD_ALIASES):
    """Fill `{prefix}_{field}` hists with every entry of a jagged reco
    collection (0, 1, or many objects per event), broadcasting the per-event
    weight to match. No-op if the (masked) collection has no entries at all.
    Pass aliases=_PHO_FIELD_ALIASES when field_hists is _PHO_HISTS.
    """
    if ak.sum(ak.num(obj, axis=1)) == 0:
        return
    w_flat = ak.flatten(ak.broadcast_arrays(evt_weight, obj.pt)[0])
    for field, axis in field_hists:
        val = getattr(obj, aliases.get(field, field))
        if field in _ABS_FIELDS:
            val = np.abs(val)
        hists[f'{prefix}_{field}'].fill(
            samp=samp, cut=cut, weight=w_flat,
            **{axis.name: ak.flatten(val)},
        )

def _fill_single(hists, prefix, obj, evt_weight, samp, cut, field_hists, aliases=_FIELD_ALIASES):
    """Fill `{prefix}_{field}` hists with one (already-selected, non-jagged)
    reco object per event. Pass aliases=_PHO_FIELD_ALIASES when field_hists
    is _PHO_HISTS.
    """
    for field, axis in field_hists:
        val = getattr(obj, aliases.get(field, field))
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
        histograms[f'mele_{field}'] = Hist(samp, cut, axis, storage=_STORAGE)
    histograms['mele_gen_lxy']        = Hist(samp, cut, _gen_lxy,  storage=_STORAGE)
    histograms['mele_gen_ee_pt']      = Hist(samp, cut, ele_pt,        storage=_STORAGE)
    histograms['mele_gen_lead_pt']    = Hist(samp, cut, ele_pt,        storage=_STORAGE)
    histograms['mele_gen_sublead_pt'] = Hist(samp, cut, ele_pt,        storage=_STORAGE)
    histograms['mele_gen_ee_dr']      = Hist(samp, cut, ee_dr,  storage=_STORAGE)
    histograms['mele_gen_ee_eta']     = Hist(samp, cut, ele_eta,       storage=_STORAGE)
    histograms['mele_gen_lead_eta']   = Hist(samp, cut, ele_eta,       storage=_STORAGE)
    histograms['mele_gen_sublead_eta']= Hist(samp, cut, ele_eta,       storage=_STORAGE)
    for _suffix, _ in _OTHERGEN_DR_TYPES:
        histograms[f'mele_othergen_dr_{_suffix}'] = Hist(samp, cut, dR, storage=_STORAGE)

    # ── Merged-electron comparison hists: same reco variables (_ELE_HISTS),
    # but for (a) AllLptElectron objects in zero-match events — split into the
    # leading-pt electron (zerolead) and the rest (zeroothers) in the same
    # event — and (b) other (non-matched) AllLptElectron objects present
    # alongside the merged one.
    for field, axis in _ELE_HISTS:
        histograms[f'mele_zerolead_{field}']   = Hist(samp, cut, axis, storage=_STORAGE)
        histograms[f'mele_zeroothers_{field}'] = Hist(samp, cut, axis, storage=_STORAGE)
        histograms[f'mele_other_{field}']      = Hist(samp, cut, axis, storage=_STORAGE)

    # ── Merged-electron "nearest other electron" realism check: same reco
    # variables (_ELE_HISTS), but only for the single AllLptElectron closest
    # to the merged one (the object responsible for the low-dR entries in
    # mele_drNearestEle) — pt/trkChi2/hit-count/IDscore etc. tell us whether
    # that nearby object looks like a real, well-reconstructed electron or a
    # low-quality/duplicate GSF track candidate.
    for field, axis in _ELE_HISTS:
        histograms[f'mele_nearestEle_{field}'] = Hist(samp, cut, axis, storage=_STORAGE)

    # ── Merged-electron / nearest-electron relational variables: compare the
    # selected merged electron directly to its nearest other AllLptElectron,
    # to help distinguish a genuine second nearby electron from a duplicate
    # GSF track built off the same underlying seed (near-zero dPt, matching
    # charge, and ~identical vertex parameters would point to a duplicate).
    histograms['mele_nearestEle_dPt']        = Hist(samp, cut, _dPt,      storage=_STORAGE)
    histograms['mele_nearestEle_dPtRel']     = Hist(samp, cut, _dPtRel,   storage=_STORAGE)
    histograms['mele_nearestEle_chargeProd'] = Hist(samp, cut, vtx_sign,  storage=_STORAGE)
    histograms['mele_nearestEle_dVxy']       = Hist(samp, cut, _dVxyNear, storage=_STORAGE)
    histograms['mele_nearestEle_dVz']        = Hist(samp, cut, _dVzNear,  storage=_STORAGE)
    histograms['mele_nearestEle_dDxy']       = Hist(samp, cut, _dDxyNear, storage=_STORAGE)
    histograms['mele_nearestEle_dDz']        = Hist(samp, cut, _dDzNear,  storage=_STORAGE)
    histograms['mele_nearestEle_drZoom']     = Hist(samp, cut, dR_zoom,   storage=_STORAGE)

    # ── Oppositely-charged mele_nearestEle diagnostic: why isn't this pair
    # counted as "resolved"? (see _dPtRelTargetGen/_oppQFailReason comment)
    histograms['mele_nearestEle_oppQ_dr_to_targetgen']     = Hist(samp, cut, dR_zoom,           storage=_STORAGE)
    histograms['mele_nearestEle_oppQ_dPtRel_to_targetgen'] = Hist(samp, cut, _dPtRelTargetGen,  storage=_STORAGE)
    histograms['mele_nearestEle_oppQ_failreason']          = Hist(samp, cut, _oppQFailReason,   storage=_STORAGE)

    # ── Merged-photon category (Photon reco variables) ────────────────────────
    for field, axis in _PHO_HISTS:
        histograms[f'mpho_{field}'] = Hist(samp, cut, axis, storage=_STORAGE)
    histograms['mpho_gen_lxy']        = Hist(samp, cut, _gen_lxy,  storage=_STORAGE)
    histograms['mpho_gen_ee_pt']      = Hist(samp, cut, ele_pt,        storage=_STORAGE)
    histograms['mpho_gen_lead_pt']    = Hist(samp, cut, ele_pt,        storage=_STORAGE)
    histograms['mpho_gen_sublead_pt'] = Hist(samp, cut, ele_pt,        storage=_STORAGE)
    histograms['mpho_gen_ee_dr']      = Hist(samp, cut, ee_dr,  storage=_STORAGE)
    histograms['mpho_gen_ee_eta']     = Hist(samp, cut, ele_eta,       storage=_STORAGE)
    histograms['mpho_gen_lead_eta']   = Hist(samp, cut, ele_eta,       storage=_STORAGE)
    histograms['mpho_gen_sublead_eta']= Hist(samp, cut, ele_eta,       storage=_STORAGE)
    histograms['mpho_drNearestEle']   = Hist(samp, cut, _drNearestEle, storage=_STORAGE)

    # ── Merged-photon "nearest electron" realism check: same rationale as the
    # mele_nearestEle_* hists — quality variables for the single AllLptElectron
    # closest to the merged photon (candidate cause of low-dR mpho_drNearestEle
    # entries: e.g. a low-pT GSF electron built from the same supercluster).
    for field, axis in _ELE_HISTS:
        histograms[f'mpho_nearestEle_{field}'] = Hist(samp, cut, axis, storage=_STORAGE)

    # ── Merged-photon comparison hists: same reco variables (_PHO_HISTS), but
    # for (a) Photon objects in zero-match events, and (b) other (non-matched)
    # Photon objects present alongside the merged one.
    for field, axis in _PHO_HISTS:
        histograms[f'mpho_zero_{field}']  = Hist(samp, cut, axis, storage=_STORAGE)
        histograms[f'mpho_other_{field}'] = Hist(samp, cut, axis, storage=_STORAGE)

    # ── Merged-photon 2D correlations: gen Lxy vs shower-shape/ID variables ───
    histograms['mpho_gen_lxy_vs_HoE']   = Hist(samp, cut, _gen_lxy, _HoE,   storage=_STORAGE)
    histograms['mpho_gen_lxy_vs_sieie'] = Hist(samp, cut, _gen_lxy, _sieie, storage=_STORAGE)

    # ── 1D versions of the above, sliced above/below the Lxy cut ──────────────
    for prefix, field, axis in _LXY_SLICE_VARS:
        histograms[f'{prefix}_{field}{_LXY_HI_SUFFIX}'] = Hist(samp, cut, axis, storage=_STORAGE)
        histograms[f'{prefix}_{field}{_LXY_LO_SUFFIX}'] = Hist(samp, cut, axis, storage=_STORAGE)

    # ── Zero matches category (gen kinematics only — no matched reco object) ──
    histograms['zero_gen_lxy']        = Hist(samp, cut, _gen_lxy, storage=_STORAGE)
    histograms['zero_gen_ee_pt']      = Hist(samp, cut, ele_pt,      storage=_STORAGE)
    histograms['zero_gen_lead_pt']    = Hist(samp, cut, ele_pt,      storage=_STORAGE)
    histograms['zero_gen_sublead_pt'] = Hist(samp, cut, ele_pt,      storage=_STORAGE)
    histograms['zero_gen_ee_dr']      = Hist(samp, cut, ee_dr,       storage=_STORAGE)
    histograms['zero_gen_ee_eta']     = Hist(samp, cut, ele_eta,     storage=_STORAGE)
    histograms['zero_gen_lead_eta']   = Hist(samp, cut, ele_eta,     storage=_STORAGE)
    histograms['zero_gen_sublead_eta']= Hist(samp, cut, ele_eta,     storage=_STORAGE)

    return histograms

# computeMergedCatVars (analysisTools/analysisSubroutines.py) runs once per
# chunk, before the cut loop, and attaches the vertexed/merged-electron/
# merged-photon categorization (cat_vertexed, cat_merged, cat_merged_photon,
# cat_zero, gen_lxy, lxy_hi, AllLptElectron.{mindRj,mindPhiJ,drNearestEle,
# is_merged,is_matched,dr_to_ge,dr_to_gp}, Photon.{mindRj,mindPhiJ,
# both_match}) that fillHistos below reads back at each cut stage, instead of
# recomputing it from scratch every time (an_selection.py currently fills at
# 6 stages). cat_vertexed (a good, truth-matched ee vertex, v15acr -- see
# analysisSubroutines.computeMergedCatVars) supersedes all of the other
# categories: cat_merged/cat_merged_photon/cat_zero are only True for events
# that are *not* vertexed. The vertexed/resolved (cat_sep) categories
# themselves are filled by the partner config configs/histo_configs/
# resolvedcats.py.
subroutines = ['computeMergedCatVars']

def fillHistos(events, hists, samp, cut, info, sum_wgt=1):
    if RAW_COUNTS:
        wgt = ak.ones_like(events.eventWgt)
    else:
        wgt = events.eventWgt / sum_wgt

    # gen_lxy/lxy_hi and the merged-electron/merged-photon
    # categorization (cat_merged, cat_merged_photon, plus the
    # AllLptElectron.{mindRj,mindPhiJ,drNearestEle,is_merged,is_matched,
    # dr_to_ge,dr_to_gp} and Photon.{mindRj,mindPhiJ,both_match} fields used
    # below) are computed once per chunk by computeMergedCatVars
    # (analysisTools/analysisSubroutines.py, registered in this module's
    # `subroutines` list) rather than recomputed here on every
    # savePlots=True cut stage (an_selection.py currently has 6).
    gen_lxy = events.gen_lxy
    lxy_hi  = events.lxy_hi
    coll    = events.AllLptElectron

    _cat_merged        = events.cat_merged
    _cat_merged_photon = events.cat_merged_photon

    # ── Helpers (used below on already-category-selected, small subsets) ────
    def _dphi(a, b):
        d = np.abs(a - b)
        return ak.where(d > np.pi, 2 * np.pi - d, d)

    def _dr(eta1, phi1, eta2, phi2):
        return np.sqrt((eta1 - eta2)**2 + _dphi(phi1, phi2)**2)

    # ── Merged-electron fills ─────────────────────────────────────────────────
    if ak.sum(_cat_merged) > 0:
        me  = ak.firsts(coll[coll.is_merged])[_cat_merged]
        w   = wgt[_cat_merged]
        lxy = gen_lxy[_cat_merged]
        ge  = events.GenEle[_cat_merged]
        gp  = events.GenPos[_cat_merged]

        lead_is_e   = ge.pt >= gp.pt
        lead_pt     = ak.where(lead_is_e, ge.pt,  gp.pt)
        sublead_pt  = ak.where(lead_is_e, gp.pt,  ge.pt)
        lead_eta    = ak.where(lead_is_e, ge.eta, gp.eta)
        sublead_eta = ak.where(lead_is_e, gp.eta, ge.eta)

        _fill_single(hists, 'mele', me, w, samp, cut, _ELE_HISTS)

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
        _fill_multi(hists, 'mele_other', coll[~coll.is_merged][_cat_merged], w, samp, cut, _ELE_HISTS)

        # Realism check on the single nearest other AllLptElectron (the one
        # realizing the min in mele_drNearestEle), restricted to dR<0.2 to
        # isolate the near-duplicate population behind the mele_drNearestEle
        # spike at dR~0.
        _other_e_mele = coll[~coll.is_merged][_cat_merged]
        if ak.sum(ak.num(_other_e_mele, axis=1)) > 0:
            _dr_me_other     = _dr(me.eta, me.phi, _other_e_mele.eta, _other_e_mele.phi)
            _nearest_idx     = ak.argmin(_dr_me_other, axis=1, keepdims=True)
            _min_dr_me_other = ak.fill_none(ak.min(_dr_me_other, axis=1), 999)
            _near_mele_mask  = _min_dr_me_other < 0.2
            if ak.sum(_near_mele_mask) > 0:
                _nearest_mele = ak.firsts(_other_e_mele[_nearest_idx])[_near_mele_mask]
                w_near        = w[_near_mele_mask]
                _fill_single(hists, 'mele_nearestEle', _nearest_mele, w_near, samp, cut, _ELE_HISTS)

                # Relational variables vs. the selected merged electron itself.
                _me_near     = me[_near_mele_mask]
                _dpt         = _me_near.pt - _nearest_mele.pt
                _chargeProd  = _me_near.charge * _nearest_mele.charge
                hists['mele_nearestEle_dPt'       ].fill(samp=samp, cut=cut, dpt=_dpt,          weight=w_near)
                hists['mele_nearestEle_dPtRel'    ].fill(samp=samp, cut=cut, dptRel=_dpt / _me_near.pt, weight=w_near)
                hists['mele_nearestEle_chargeProd'].fill(samp=samp, cut=cut, sign=_chargeProd, weight=w_near)
                hists['mele_nearestEle_dVxy'      ].fill(samp=samp, cut=cut, dvxy=np.abs(_me_near.vxy - _nearest_mele.vxy), weight=w_near)
                hists['mele_nearestEle_dVz'       ].fill(samp=samp, cut=cut, dvz=np.abs(_me_near.vz  - _nearest_mele.vz),   weight=w_near)
                hists['mele_nearestEle_dDxy'      ].fill(samp=samp, cut=cut, ddxy=np.abs(_me_near.dxy - _nearest_mele.dxy), weight=w_near)
                hists['mele_nearestEle_dDz'       ].fill(samp=samp, cut=cut, ddz=np.abs(_me_near.dz  - _nearest_mele.dz),   weight=w_near)
                hists['mele_nearestEle_drZoom'    ].fill(samp=samp, cut=cut, dr_zoom=_min_dr_me_other[_near_mele_mask],     weight=w_near)

                # ── Oppositely-charged subset: is the near track's own gen
                # match (dR<0.1, |dPt|/pt<0.2 to whichever gen shares its
                # charge sign) failing on dR, on pt, or both? If it passed
                # both, dr_match_ge/dr_match_gp would already have claimed
                # it and this event would be _cat_sep (resolved), not
                # _cat_merged -- so "pass_both_unexpected" should be ~empty.
                _opp_mask = _chargeProd < 0
                if ak.sum(_opp_mask) > 0:
                    _near_opp = _nearest_mele[_opp_mask]
                    w_opp     = w_near[_opp_mask]
                    ge_opp    = ge[_near_mele_mask][_opp_mask]
                    gp_opp    = gp[_near_mele_mask][_opp_mask]

                    _target_eta = ak.where(_near_opp.charge == -1, ge_opp.eta, gp_opp.eta)
                    _target_phi = ak.where(_near_opp.charge == -1, ge_opp.phi, gp_opp.phi)
                    _target_pt  = ak.where(_near_opp.charge == -1, ge_opp.pt,  gp_opp.pt)

                    _dr_opp_target     = _dr(_near_opp.eta, _near_opp.phi, _target_eta, _target_phi)
                    _dptrel_opp_target = (_near_opp.pt - _target_pt) / _target_pt

                    # ak.where mis-broadcasts plain python string literals against
                    # an array condition in this awkward version (it zips the
                    # strings' characters instead of treating them as scalars),
                    # so build this one with np.where on plain numpy bool arrays.
                    _pass_dr = np.asarray(ak.fill_none(_dr_opp_target < 0.1, False))
                    _pass_pt = np.asarray(ak.fill_none(np.abs(_dptrel_opp_target) < 0.2, False))

                    _reason = ak.Array(np.where(_pass_dr & _pass_pt,   'pass_both_unexpected',
                                        np.where(_pass_dr & ~_pass_pt, 'fail_pt_only',
                                        np.where(~_pass_dr & _pass_pt, 'fail_dR_only',
                                                                        'fail_both'))))

                    hists['mele_nearestEle_oppQ_dr_to_targetgen'    ].fill(samp=samp, cut=cut, dr_zoom=_dr_opp_target, weight=w_opp)
                    hists['mele_nearestEle_oppQ_dPtRel_to_targetgen'].fill(samp=samp, cut=cut, dptRelTargetGen=_dptrel_opp_target, weight=w_opp)
                    hists['mele_nearestEle_oppQ_failreason'         ].fill(samp=samp, cut=cut, reason=_reason, weight=w_opp)

    # ── Merged-photon fills ───────────────────────────────────────────────────
    if ak.sum(_cat_merged_photon) > 0:
        mp  = ak.firsts(events.Photon[events.Photon.both_match])[_cat_merged_photon]
        w   = wgt[_cat_merged_photon]
        lxy = gen_lxy[_cat_merged_photon]
        ge  = events.GenEle[_cat_merged_photon]
        gp  = events.GenPos[_cat_merged_photon]

        lead_is_e   = ge.pt >= gp.pt
        lead_pt     = ak.where(lead_is_e, ge.pt,  gp.pt)
        sublead_pt  = ak.where(lead_is_e, gp.pt,  ge.pt)
        lead_eta    = ak.where(lead_is_e, ge.eta, gp.eta)
        sublead_eta = ak.where(lead_is_e, gp.eta, ge.eta)

        _fill_single(hists, 'mpho', mp, w, samp, cut, _PHO_HISTS, aliases=_PHO_FIELD_ALIASES)

        # dR from the selected (merged) photon to the nearest AllLptElectron
        # in the same event.
        _mpho_coll = coll[_cat_merged_photon]
        mpho_dr_e  = ak.fill_none(ak.min(_dr(_mpho_coll.eta, _mpho_coll.phi, mp.eta, mp.phi), axis=1), 999)
        hists['mpho_drNearestEle'].fill(samp=samp, cut=cut, drNearestEle=mpho_dr_e, weight=w)

        # Realism check on the single nearest AllLptElectron to the merged
        # photon (the one realizing the min in mpho_drNearestEle), restricted
        # to dR<0.2 to isolate the near-duplicate population behind the
        # mpho_drNearestEle spike at dR~0.
        _near_mpho_mask = mpho_dr_e < 0.2
        if ak.sum(_near_mpho_mask) > 0:
            _dr_mp_e       = _dr(_mpho_coll.eta, _mpho_coll.phi, mp.eta, mp.phi)
            _nearest_idx_p = ak.argmin(_dr_mp_e, axis=1, keepdims=True)
            _nearest_mpho  = ak.firsts(_mpho_coll[_nearest_idx_p])[_near_mpho_mask]
            _fill_single(hists, 'mpho_nearestEle', _nearest_mpho, w[_near_mpho_mask], samp, cut, _ELE_HISTS)

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
        _fill_multi(hists, 'mpho_other', events.Photon[~events.Photon.both_match][_cat_merged_photon], w, samp, cut, _PHO_HISTS, aliases=_PHO_FIELD_ALIASES)

    # ── Zero matches fills ─────────────────────────────────────────────────────
    # Definition from mergedmatch.py: no electron matched to either gen AND
    # neither gen has a photon/track/conversion alternative.
    _cat_zero = events.cat_zero
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
        #coll_z      = coll[_cat_zero]
        #order_z     = ak.argsort(coll_z.pt, axis=1, ascending=False)
        #coll_z_sort = coll_z[order_z]
        #has_lead    = ak.num(coll_z_sort, axis=1) > 0
        #if ak.sum(has_lead) > 0:
        #    lead_z   = ak.firsts(coll_z_sort)[has_lead]
        #    others_z = coll_z_sort[:, 1:][has_lead]
        #    w_lead_z = w_z[has_lead]
        #    _fill_single(hists, 'mele_zerolead',   lead_z,   w_lead_z, samp, cut, _ELE_HISTS)
        #    _fill_multi (hists, 'mele_zeroothers', others_z, w_lead_z, samp, cut, _ELE_HISTS)
    
        # Same-type Photon objects present in these (otherwise unmatched) events.
        #_fill_multi(hists, 'mpho_zero', events.Photon[_cat_zero],  w_z, samp, cut, _PHO_HISTS)
