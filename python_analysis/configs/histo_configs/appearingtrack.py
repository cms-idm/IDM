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
# The plotted electron is events.best_merged: the one merged-electron
# candidate per event picked by selectBestMergedCandidate from
# events.merged_candidates (defineGoodMergedCandidates, run for every chunk by
# computeMergedVars with the processor's good_merged version) -- i.e. the same
# electron the aptrk_selection.py cuts act on. Events without one are skipped.
# Works for both signal and background; the isMerged flag (signal only, from
# AllLptElectron.is_merged -- computeMergedCatVars in
# analysisTools/analysisSubroutines.py, registered below via `subroutines`)
# additionally tags whether the selected electron individually qualifies as a
# genuine merged electron under the mergedcats.py definition.

_LOG = hist.axis.transform.log

# ── Extra axes not in histobins ──────────────────────────────────────────────
_e_energy  = Regular(50,  0, 100,  name='e',        label='E [GeV]')
_calIso    = Regular(50,  0,  50,  name='iso',       label='Calo Iso [GeV]')
_calRelIso = Regular(50,  0,   5,  name='relIso',    label='Calo Relative Iso')
_rhoEA     = Regular(50,  0,  20,  name='rhoEA',     label=r'$\rho \times EA$ [GeV]')
_sieie     = Regular(50,  0,0.05,  name='sieie',     label=r'$\sigma_{i\eta i\eta}$')
_dEtaSeed  = Regular(50,  0,0.01,  name='dEtaSeed',  label=r'$|\Delta\eta_\mathrm{seed}|$')
_dPhiIn    = Regular(50,  0, 0.1,  name='dPhiIn',    label=r'$|\Delta\phi_\mathrm{in}|$')
_HoE       = Regular(50,  0,   1,  name='HoE',       label='H/E')
_invEmP    = Regular(50,  0, 0.1,  name='invEmP',    label=r'$|1/E - 1/p|$ [GeV$^{-1}$]')
_missHits  = Integer(0,    5,       name='missHits',  label='Exp. Missing Inner Hits')
_charge    = IntCategory([-1, 1],   name='charge',    label='Charge')
_ele_chi2  = Regular(50,0,25,name="chi2",label=r"Track $\chi^2/df$")
_ele_angRes = Regular(50,0,0.04,name="angRes",label=r"Angular Resolution $\sqrt{\sigma_\eta^2 + \sigma_\phi^2}$")

# Local overrides of the shared dR/dRj/dphiJ axes (histobins.py) with labels
# spelling out what the delta is between, rather than the shared generic
# "$\Delta R$"/"$\Delta \phi$" -- those objects are reused elsewhere for
# different object pairs, so relabel locally instead of editing them.
_minDRtoReg = Regular(50, 0, 5,   name='dr',    label=r'Min $\Delta R(e, e_{\mathrm{reg}})$')
_drNearestEle = Regular(50, 0, 5, name='drNearestEle', label=r'Min $\Delta R(e, e_{\mathrm{other\ lpt}})$')
_dRj        = Regular(50, 0, 5,   name='drj',   label=r'Min $\Delta R(e, j)$')
_dPhiJ      = Regular(64,  0, 3.2, name='dphiJ', label=r'Min $\Delta \phi(e, j)$')

# dxy floored at the selection cut (1e-4 cm) so the axis stays log-scale down
# to that value; anything below the floor (shouldn't occur post-selection)
# lands in underflow rather than being dropped.
_dxy_log = Regular(60, 1e-4, 100, name='dxy', label=r'Electron Track $d_{xy}$ [cm]', transform=_LOG)

# Full-range linear dphi(e, ptmiss), plus a log-scale zoom toward the
# collinear region (small dphi is the signature of a genuinely displaced
# track pointing away from the hard-scatter PV). Local override of the
# shared dphi_generic axis (histobins.py) so the label names both objects
# instead of the shared generic "$\Delta \phi$".
_dphi_lin = Regular(32, 0,    3.2, name='dphi', label=r'$\Delta\phi(e, p_{T}^{miss})$')
_dphi_log = Regular(50, 1e-4, 3.2, name='dphi', label=r'$\Delta\phi(e, p_{T}^{miss})$', transform=_LOG)

_cosdphi  = Regular(50, -1,   1,   name='cosdphi', label=r'$\cos(\Delta\phi(e, p_{T}^{miss}))$')

# lxyEst = dxy / sin(dphi(e, ptmiss)): the same reco-only Lxy proxy used in
# mergedeles.py's mele_gen_lxy_vs_lxyEst, without the gen-Lxy correlation
# (which doesn't exist for background).
_lxyEst_log = Regular(60, 1e-4, 100, name='lxyEst', label=r'$d_{xy}/\sin(\Delta\phi(e, p_{T}^{miss}))$ [cm]', transform=_LOG)

# log10(dxy/dz) -- named 'log10dxydz' (rather than reusing histobins.py's
# 'logdxydz', which vtxvars.py fills with a natural log) to be explicit about
# the base. A tiny epsilon guards the dz denominator against division by zero.
_log10dxydz = Regular(60, -3, 3, name='log10dxydz', label=r'$\log_{10}(d_{xy}/d_{z})$')

# Tags the selected electron's relationship to the event's gen-matched merged
# electron (signal only, via AllLptElectron.is_merged -- see fillHistos
# below); always 'n/a' for background since there's no gen truth to check
# against.
#   mergedSelected    -- the event has a gen-matched merged electron and it IS
#                        the selected reco candidate.
#   mergedNotSelected -- the event has a gen-matched merged electron, but the
#                        selected reco candidate is a different electron.
#   notMerged         -- the event has no gen-matched merged electron at all.
isMerged = StrCategory(
    ['mergedSelected', 'mergedNotSelected', 'notMerged', 'n/a'],
    name='isMerged', label='Genuine Merged Electron (signal only)',
)

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
    ('trkChi2',             _ele_chi2),
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
    ('minDRtoReg',          _minDRtoReg),
    ('drNearestEle',        _drNearestEle),
    ('mindRj',              _dRj),
    ('mindPhiJ',            _dPhiJ),
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
    histograms['AT_dphi_e_ptmiss']     = Hist(samp, cut, isMerged, _dphi_lin,    storage=hist.storage.Weight())
    histograms['AT_dphi_e_ptmiss_log'] = Hist(samp, cut, isMerged, _dphi_log,    storage=hist.storage.Weight())
    histograms['AT_cosdphi_e_ptmiss']  = Hist(samp, cut, isMerged, _cosdphi,     storage=hist.storage.Weight())
    histograms['AT_lxyEst']            = Hist(samp, cut, isMerged, _lxyEst_log,  storage=hist.storage.Weight())
    histograms['AT_log10dxydz']        = Hist(samp, cut, isMerged, _log10dxydz,  storage=hist.storage.Weight())
    histograms['AT_dxy_vs_dphi']       = Hist(samp, cut, isMerged, _dxy_log, _dphi_log, storage=hist.storage.Weight())
    return histograms

subroutines = ['computeMergedCatVars']

def fillHistos(events, hists, samp, cut, info, sum_wgt=1):
    wgt_all = events.eventWgt / sum_wgt

    # best_merged is only set when AllLptElectron is in the ntuple (see
    # computeMergedVars).
    if 'best_merged' not in events.fields:
        return

    def _dphi(a, b):
        d = np.abs(a - b)
        return ak.where(d > np.pi, 2 * np.pi - d, d)

    def _dr(eta1, phi1, eta2, phi2):
        return np.sqrt((eta1 - eta2)**2 + _dphi(phi1, phi2)**2)

    # ── Selected electron: events.best_merged ─────────────────────────────────
    has_cand = ~ak.is_none(events.best_merged)
    if ak.sum(has_cand) == 0:
        return

    evts = events[has_cand]
    wgt  = wgt_all[has_cand]
    ele  = evts.best_merged
    coll = evts.AllLptElectron

    # Derived fields needed by _ELE_HISTS. best_merged is a copy of the
    # AllLptElectron record, so compute them on it directly: mindRj/mindPhiJ
    # (same convention as mergedcats.py) and drNearestEle, the dR to the
    # nearest other AllLptElectron (self excluded by matching pt/eta/phi).
    is_self = (coll.pt == ele.pt) & (coll.eta == ele.eta) & (coll.phi == ele.phi)
    dr_ee   = ak.where(is_self, np.inf, _dr(ele.eta, ele.phi, coll.eta, coll.phi))
    ele = ak.with_field(ele, ak.fill_none(ak.min(ele.dRJets,   axis=-1), 999), 'mindRj')
    ele = ak.with_field(ele, ak.fill_none(ak.min(ele.dPhiJets, axis=-1), 999), 'mindPhiJ')
    ele = ak.with_field(ele, ak.fill_none(ak.min(dr_ee,        axis=-1), 999), 'drNearestEle')

    # Merged-electron flag (signal only), computed by computeMergedCatVars
    # (analysisTools/analysisSubroutines.py) using the same definition as the
    # merged-electron category in mergedcats.py.
    if info['type'] == 'signal':
        is_merged_sel     = ak.to_numpy(ele.is_merged)
        event_has_merged  = ak.to_numpy(ak.any(coll.is_merged, axis=1))
        isMerged_label = np.where(
            is_merged_sel, 'mergedSelected',
            np.where(event_has_merged, 'mergedNotSelected', 'notMerged'),
        )
    else:
        isMerged_label = np.full(len(ele), 'n/a', dtype=object)

    # ── Reco-quantity fills ────────────────────────────────────────────────
    _fill_single(hists, 'AT', ele, wgt, isMerged_label, samp, cut, _ELE_HISTS)

    ptmiss     = evts.PFMET.pt
    dphi_e_met = _dphi(ele.phi, evts.PFMET.phi)
    lxyEst     = np.abs(ele.dxy) / np.sin(dphi_e_met)
    # Epsilon guards the dz denominator against division by zero (same
    # convention as vtxvars.py's e_dxydz).
    log10dxydz = np.log10(np.abs(ele.dxy) / (np.abs(ele.dz) + 1e-9))

    hists['AT_ptmiss'           ].fill(samp=samp, cut=cut, isMerged=isMerged_label, met_pt=ptmiss,     weight=wgt)
    hists['AT_dphi_e_ptmiss'    ].fill(samp=samp, cut=cut, isMerged=isMerged_label, dphi=dphi_e_met,   weight=wgt)
    hists['AT_dphi_e_ptmiss_log'].fill(samp=samp, cut=cut, isMerged=isMerged_label, dphi=dphi_e_met,   weight=wgt)
    hists['AT_cosdphi_e_ptmiss' ].fill(samp=samp, cut=cut, isMerged=isMerged_label, cosdphi=np.cos(dphi_e_met), weight=wgt)
    hists['AT_lxyEst'           ].fill(samp=samp, cut=cut, isMerged=isMerged_label, lxyEst=lxyEst,     weight=wgt)
    hists['AT_log10dxydz'       ].fill(samp=samp, cut=cut, isMerged=isMerged_label, log10dxydz=log10dxydz, weight=wgt)
    hists['AT_dxy_vs_dphi'      ].fill(samp=samp, cut=cut, isMerged=isMerged_label, dxy=np.abs(ele.dxy), dphi=dphi_e_met, weight=wgt)
