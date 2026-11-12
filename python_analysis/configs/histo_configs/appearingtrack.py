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
_e_energy  = Regular(50,  0, 100,  name='e',        label='E (GeV)')
_calIso    = Regular(50,  0,  50,  name='iso',       label='Calo Iso (GeV)')
_calRelIso = Regular(50,  0,   5,  name='relIso',    label='Calo Relative Iso')
_rhoEA     = Regular(50,  0,  20,  name='rhoEA',     label=r'$\rho \times EA$ (GeV)')
_sieie     = Regular(50,  0,0.05,  name='sieie',     label=r'$\sigma_{i\eta i\eta}$')
_scEtaW    = Regular(50,  0, 0.5,  name='scEtaWidth', label=r'SC $\eta$ Width')
_scPhiW    = Regular(50,  0, 0.5,  name='scPhiWidth', label=r'SC $\phi$ Width')
# Shower shape (full 5x5). The e1x5/e5x5 and e2x5Max/e5x5 ratios are -1
# (underflow) when e5x5 <= 0 -- see _safe_ratio in fillHistos. Both ratios are
# exactly 1 for a large fraction of electrons (~30-40% in signal), so the axes
# extend one bin past 1 to keep those out of overflow. sigmaIphiIphi has a
# sentinel at ~31.6 in the ntuple, which lands in overflow here.
_r9        = Regular(50,  0, 1.2,  name='r9',        label=r'Full5x5 $R_9$')
_sipip     = Regular(50,  0, 0.1,  name='sipip',     label=r'Full5x5 $\sigma_{i\phi i\phi}$')
_e1x5Frac  = Regular(51,  0, 1.02, name='e1x5Frac',  label=r'Full5x5 $E_{1\times5}/E_{5\times5}$')
_e2x5Frac  = Regular(51,  0, 1.02, name='e2x5Frac',  label=r'Full5x5 $E_{2\times5}^{max}/E_{5\times5}$')
# Supercluster
_scNClus   = Integer(0,   10,      name='scNClus',   label='N Clusters in SC')
_scEoverP  = Regular(50,  0, 5,    name='scEoverP',  label=r'$E_{SC}/p$')
_scRawFrac = Regular(50,  0, 1.5,  name='scRawFrac', label=r'$E_{SC}^{raw}/E_{SC}$')
_dEtaSCTrk = Regular(50, 1e-4, 1,  name='dEtaSCTrk', label=r'$|\eta_{SC} - \eta_{trk}|$', transform=hist.axis.transform.log)
# (full55hcalDepth1/2OverEcal not included: identically 0 in the current ntuples)
# Jets / event level
_dRj0      = Regular(50,  0,   5,  name='drj0',      label=r'$\Delta R(e, j_0)$')
_metLeadPt = Regular(50,  0,  10,  name='metLeadPt', label=r'$p_T^{miss}/p_T^{j_0}$')
_HT        = Regular(50,  0, 1000, name='HT',        label=r'$H_T$ (GeV)')
_nJets     = Integer(0,   10,      name='nJets',     label=r'$N_{jets}$')
_dEtaSeed  = Regular(50,  0,0.01,  name='dEtaSeed',  label=r'$|\Delta\eta_\mathrm{seed}|$')
_dPhiIn    = Regular(50,  0, 0.1,  name='dPhiIn',    label=r'$|\Delta\phi_\mathrm{in}|$')
_HoE       = Regular(50,  0,   1,  name='HoE',       label='H/E')
_invEmP    = Regular(50,  0, 0.1,  name='invEmP',    label=r'$|1/E - 1/p|$ (GeV$^{-1}$)')
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

# ATgm_mindRj_vs_ptRatioJ: dR to the nearest PFJet vs pT(e)/pT(that jet),
# with NO self-jet veto -- the diagnostic for the veto below. Finer dR bins
# than _dRj (0.02) so the dR < 0.1 self-jet region spans several bins; both
# veto thresholds (dR = 0.1, ratio = 0.3) fall on bin edges. Electrons in
# events with no PFJet land in overflow (dR = 999) / underflow (ratio = -1).
_dRjFine    = Regular(100, 0, 2,  name='drj',      label=r'Min $\Delta R(e, j)$ (no self-jet veto)')
_ptRatioJ   = Regular(40,  0, 2,  name='ptRatioJ', label=r'$p_T(e)/p_T(j_{\mathrm{nearest}})$')

# Self-jet veto for the ATgm_* jet quantities: a PFJet with dR(e, j) < 0.1 and
# pT(e)/pT(j) > 0.3 is taken to be the merged electron itself clustered into a
# jet, and is ignored. Same definition as defineGoodMergedCandidates
# (analysisTools/analysisSubroutines.py).
_SELF_JET_DR     = 0.1
_SELF_JET_PTFRAC = 0.3

# dxy floored at the selection cut (1e-4 cm) so the axis stays log-scale down
# to that value; anything below the floor (shouldn't occur post-selection)
# lands in underflow rather than being dropped.
_dxy_log = Regular(80, 1e-6, 100, name='dxy', label=r'Electron Track $d_{xy}$ (cm)', transform=_LOG)

# Full-range linear dphi(e, ptmiss), plus a log-scale zoom toward the
# collinear region (small dphi is the signature of a genuinely displaced
# track pointing away from the hard-scatter PV). Local override of the
# shared dphi_generic axis (histobins.py) so the label names both objects
# instead of the shared generic "$\Delta \phi$".
_dphi_lin = Regular(32, 0,    3.2, name='dphi', label=r'$\Delta\phi(e, p_{T}^{miss})$')
_dphi_log = Regular(50, 1e-4, 3.2, name='dphi', label=r'$\Delta\phi(e, p_{T}^{miss})$', transform=_LOG)

_cosdphi  = Regular(50, -1,   1,   name='cosdphi', label=r'$\cos(\Delta\phi(e, p_{T}^{miss}))$')

# Jet--ptmiss dphi, from the same jet collection as the HT/nJets hists (i.e.
# self-jet-vetoed for ATgm_*). Events with no jets are filled with 999 and
# land in overflow.
_dphi_j0_met   = Regular(32, 0, 3.2, name='dphi', label=r'$\Delta\phi(j_0, p_{T}^{miss})$')
_mindphi_j_met = Regular(32, 0, 3.2, name='dphi', label=r'Min $\Delta\phi(j, p_{T}^{miss})$')

# lxyEst = dxy / sin(dphi(e, ptmiss)): the same reco-only Lxy proxy used in
# mergedeles.py's mele_gen_lxy_vs_lxyEst, without the gen-Lxy correlation
# (which doesn't exist for background).
_lxyEst_log = Regular(80, 1e-8, 100, name='lxyEst', label=r'$d_{xy}/\sin(\Delta\phi(e, p_{T}^{miss}))$ (cm)', transform=_LOG)

# log10(dxy/dz) -- named 'log10dxydz' (rather than reusing histobins.py's
# 'logdxydz', which vtxvars.py fills with a natural log) to be explicit about
# the base. A tiny epsilon guards the dz denominator against division by zero.
_log10dxydz = Regular(80, -5, 3, name='log10dxydz', label=r'$\log_{10}(d_{xy}/d_{z})$')

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
    ('scEtaWidth',          _scEtaW),
    ('scPhiWidth',          _scPhiW),
    ('full55r9',            _r9),
    ('full55sigmaIphiIphi', _sipip),
    ('e1x5Frac',            _e1x5Frac),   # derived in fillHistos
    ('e2x5Frac',            _e2x5Frac),   # derived in fillHistos
    ('scClustersSize',      _scNClus),
    ('scEoverP',            _scEoverP),   # derived in fillHistos
    ('scRawFrac',           _scRawFrac),  # derived in fillHistos
    ('dEtaSCTrk',           _dEtaSCTrk),  # derived in fillHistos
    ('dRj0',                _dRj0),
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

def _fill_single(hists, prefix, obj, evt_weight, samp, cut, field_hists, cat_kw):
    """Fill `{prefix}_{field}` hists with one (already-selected, non-jagged)
    reco electron per event. `cat_kw` holds any extra category-axis values
    (e.g. {'isMerged': labels} for the AT_* hists, {} for ATgm_*).
    """
    for field, axis in field_hists:
        val = getattr(obj, _FIELD_ALIASES.get(field, field))
        if field in _ABS_FIELDS:
            val = np.abs(val)
        hists[f'{prefix}_{field}'].fill(
            samp=samp, cut=cut, weight=evt_weight, **cat_kw,
            **{axis.name: val},
        )

def _book(histograms, prefix, cat_axes):
    """Book the full set of `{prefix}_*` hists with extra category axes
    `cat_axes` inserted between (samp, cut) and the variable axis/axes.
    """
    def _h(*axes):
        return Hist(samp, cut, *cat_axes, *axes, storage=hist.storage.Weight())
    for field, axis in _ELE_HISTS:
        histograms[f'{prefix}_{field}'] = _h(axis)

    histograms[f'{prefix}_ptmiss']            = _h(met_pt)
    histograms[f'{prefix}_dphi_e_ptmiss']     = _h(_dphi_lin)
    histograms[f'{prefix}_dphi_e_ptmiss_log'] = _h(_dphi_log)
    histograms[f'{prefix}_cosdphi_e_ptmiss']  = _h(_cosdphi)
    histograms[f'{prefix}_lxyEst']            = _h(_lxyEst_log)
    histograms[f'{prefix}_log10dxydz']        = _h(_log10dxydz)
    histograms[f'{prefix}_dxy_vs_dphi']       = _h(_dxy_log, _dphi_log)
    histograms[f'{prefix}_met_leadPt_ratio']  = _h(_metLeadPt)
    histograms[f'{prefix}_HT']                = _h(_HT)
    histograms[f'{prefix}_nJets']             = _h(_nJets)
    histograms[f'{prefix}_dphi_j0_ptmiss']    = _h(_dphi_j0_met)
    histograms[f'{prefix}_mindphi_j_ptmiss']  = _h(_mindphi_j_met)

def make_histograms():
    histograms = {}
    # AT_*: the selected candidate (events.best_merged), signal + background,
    # split by its relationship to the gen-matched merged electron.
    _book(histograms, 'AT', [isMerged])
    # ATgm_*: the gen-matched merged electron itself (AllLptElectron with
    # isGenMerged), signal only, for every event that has one -- regardless of
    # whether a best_merged candidate exists or which electron it is.
    _book(histograms, 'ATgm', [])
    histograms['ATgm_mindRj_vs_ptRatioJ'] = Hist(samp, cut, _dRjFine, _ptRatioJ,
                                                 storage=hist.storage.Weight())
    return histograms

subroutines = ['computeMergedCatVars']

def _dphi(a, b):
    d = np.abs(a - b)
    return ak.where(d > np.pi, 2 * np.pi - d, d)

def _dr(eta1, phi1, eta2, phi2):
    return np.sqrt((eta1 - eta2)**2 + _dphi(phi1, phi2)**2)

def _safe_ratio(num, den):
    # A non-positive denominator gives -1 (lands in underflow) rather than inf/nan.
    return ak.where(den > 0, num / ak.where(den > 0, den, 1), -1)

def _add_derived(ele, coll):
    """Derived fields needed by _ELE_HISTS, for one AllLptElectron record per
    event (`ele`) out of that event's AllLptElectron collection (`coll`):
    mindRj/mindPhiJ (same convention as mergedcats.py), drNearestEle (the dR
    to the nearest other AllLptElectron, self excluded by matching
    pt/eta/phi), and the shower-shape / supercluster ratios.
    """
    is_self = (coll.pt == ele.pt) & (coll.eta == ele.eta) & (coll.phi == ele.phi)
    dr_ee   = ak.where(is_self, np.inf, _dr(ele.eta, ele.phi, coll.eta, coll.phi))
    ele = ak.with_field(ele, ak.fill_none(ak.min(ele.dRJets,   axis=-1), 999), 'mindRj')
    ele = ak.with_field(ele, ak.fill_none(ak.min(np.abs(ele.dPhiJets), axis=-1), 999), 'mindPhiJ')
    ele = ak.with_field(ele, ak.fill_none(ak.min(dr_ee,        axis=-1), 999), 'drNearestEle')

    ele = ak.with_field(ele, _safe_ratio(ele.full55e1x5,    ele.full55e5x5), 'e1x5Frac')
    ele = ak.with_field(ele, _safe_ratio(ele.full55e2x5Max, ele.full55e5x5), 'e2x5Frac')
    ele = ak.with_field(ele, _safe_ratio(ele.scEnergy, ele.pt * np.cosh(ele.eta)), 'scEoverP')
    ele = ak.with_field(ele, _safe_ratio(ele.scRawE,   ele.scEnergy),            'scRawFrac')
    ele = ak.with_field(ele, np.abs(ele.scEta - ele.eta), 'dEtaSCTrk')
    return ele

def _nearest_jet_ratio(ele, jets):
    """dR from `ele` (one per event) to its nearest jet in `jets`, and
    pT(e)/pT(that jet). No self-jet veto. Events with no jets give
    (999, -1)."""
    dr  = _dr(ele.eta, ele.phi, jets.eta, jets.phi)
    idx = ak.argmin(dr, axis=1, keepdims=True)
    near_dr    = ak.fill_none(ak.firsts(dr[idx]), 999)
    near_ratio = ak.fill_none(ele.pt / ak.firsts(jets.pt[idx]), -1)
    return near_dr, near_ratio

def _veto_self_jets(ele, jets):
    """Drop the electron's own jet (dR < _SELF_JET_DR and pT(e)/pT(j) >
    _SELF_JET_PTFRAC) from `jets`, and recompute ele's jet fields (mindRj,
    mindPhiJ, dRj0) from the remaining jets. Returns (ele, kept_jets)."""
    dr      = _dr(ele.eta, ele.phi, jets.eta, jets.phi)
    is_self = (dr < _SELF_JET_DR) & (ele.pt / jets.pt > _SELF_JET_PTFRAC)
    kept    = jets[~is_self]
    lead    = kept[ak.argmax(kept.pt, axis=1, keepdims=True)]
    ele = ak.with_field(ele, ak.fill_none(ak.min(dr[~is_self], axis=1), 999), 'mindRj')
    ele = ak.with_field(ele, ak.fill_none(ak.min(_dphi(ele.phi, kept.phi), axis=1), 999), 'mindPhiJ')
    ele = ak.with_field(ele, ak.fill_none(ak.firsts(_dr(ele.eta, ele.phi, lead.eta, lead.phi)), 999), 'dRj0')
    return ele, kept

def _fill_all(hists, prefix, evts, ele, jets, wgt, samp, cut, cat_kw):
    """Fill every `{prefix}_*` hist booked by _book, for one electron per
    event (`ele`, already passed through _add_derived) in events `evts`.
    `jets` is the PFJet collection used for the event-level jet quantities
    (leading-jet pT, HT, nJets).
    """
    # ── Reco-quantity fills ────────────────────────────────────────────────
    _fill_single(hists, prefix, ele, wgt, samp, cut, _ELE_HISTS, cat_kw)

    ptmiss     = evts.PFMET.pt
    dphi_e_met = _dphi(ele.phi, evts.PFMET.phi)
    lxyEst     = np.abs(ele.dxy) / np.sin(dphi_e_met)
    # Epsilon guards the dz denominator against division by zero (same
    # convention as vtxvars.py's e_dxydz).
    log10dxydz = np.log10(np.abs(ele.dxy) / (np.abs(ele.dz) + 1e-9))

    common = dict(samp=samp, cut=cut, weight=wgt, **cat_kw)
    hists[f'{prefix}_ptmiss'           ].fill(met_pt=ptmiss, **common)
    hists[f'{prefix}_dphi_e_ptmiss'    ].fill(dphi=dphi_e_met, **common)
    hists[f'{prefix}_dphi_e_ptmiss_log'].fill(dphi=dphi_e_met, **common)
    hists[f'{prefix}_cosdphi_e_ptmiss' ].fill(cosdphi=np.cos(dphi_e_met), **common)
    hists[f'{prefix}_lxyEst'           ].fill(lxyEst=lxyEst, **common)
    hists[f'{prefix}_log10dxydz'       ].fill(log10dxydz=log10dxydz, **common)
    hists[f'{prefix}_dxy_vs_dphi'      ].fill(dxy=np.abs(ele.dxy), dphi=dphi_e_met, **common)

    # Jet/event-level. MET/leading-jet pT is -1 (underflow) for events with no PFJet.
    lead_jet_pt = ak.fill_none(ak.firsts(jets.pt), 0)
    met_lead    = ak.where(lead_jet_pt > 0, ptmiss / ak.where(lead_jet_pt > 0, lead_jet_pt, 1), -1)
    hists[f'{prefix}_met_leadPt_ratio' ].fill(metLeadPt=met_lead, **common)
    hists[f'{prefix}_HT'               ].fill(HT=ak.sum(jets.pt, axis=1), **common)
    hists[f'{prefix}_nJets'            ].fill(nJets=ak.num(jets.pt, axis=1), **common)

    # Jet--ptmiss dphi. Leading jet by pT (jets aren't assumed pT-ordered after
    # the ATgm_* self-jet veto).
    dphi_j_met  = _dphi(jets.phi, evts.PFMET.phi)
    lead_idx    = ak.argmax(jets.pt, axis=1, keepdims=True)
    dphi_j0_met = ak.fill_none(ak.firsts(dphi_j_met[lead_idx]), 999)
    min_dphi    = ak.fill_none(ak.min(dphi_j_met, axis=1), 999)
    hists[f'{prefix}_dphi_j0_ptmiss'   ].fill(dphi=dphi_j0_met, **common)
    hists[f'{prefix}_mindphi_j_ptmiss' ].fill(dphi=min_dphi, **common)

def fillHistos(events, hists, samp, cut, info, sum_wgt=1):
    wgt_all = events.eventWgt / sum_wgt

    # ── Gen-matched merged electron (signal only): ATgm_* ─────────────────────
    # Filled before the best_merged early-return below, since these don't
    # depend on a selected candidate existing. isGenMerged (= is_merged &
    # cat_merged) is set by defineGoodMergedCandidates via computeMergedVars
    # (analysisTools/analysisSubroutines.py): the electron matched to both
    # gens, in an event that is neither resolved (cat_sep) nor vertexed. At
    # most one AllLptElectron per event is expected to carry it, so take the
    # first.
    if (info['type'] == 'signal' and 'AllLptElectron' in events.fields
            and 'isGenMerged' in events.AllLptElectron.fields):
        has_gm = ak.any(events.AllLptElectron.isGenMerged, axis=1)
        if ak.sum(has_gm) > 0:
            evts_gm = events[has_gm]
            coll_gm = evts_gm.AllLptElectron
            ele_gm  = _add_derived(ak.firsts(coll_gm[coll_gm.isGenMerged]), coll_gm)
            wgt_gm  = wgt_all[has_gm]
            # 2D self-jet diagnostic: filled BEFORE the veto, from all PFJets.
            near_dr, near_ratio = _nearest_jet_ratio(ele_gm, evts_gm.PFJet)
            hists['ATgm_mindRj_vs_ptRatioJ'].fill(samp=samp, cut=cut, weight=wgt_gm,
                                                  drj=near_dr, ptRatioJ=near_ratio)
            # All other ATgm_* jet quantities ignore the electron's own jet.
            ele_gm, jets_gm = _veto_self_jets(ele_gm, evts_gm.PFJet)
            _fill_all(hists, 'ATgm', evts_gm, ele_gm, jets_gm, wgt_gm, samp, cut, {})

    # best_merged is only set when AllLptElectron is in the ntuple (see
    # computeMergedVars).
    if 'best_merged' not in events.fields:
        return

    # ── Selected electron: events.best_merged ─────────────────────────────────
    has_cand = ~ak.is_none(events.best_merged)
    if ak.sum(has_cand) == 0:
        return

    evts = events[has_cand]
    wgt  = wgt_all[has_cand]
    coll = evts.AllLptElectron
    # best_merged is a copy of the AllLptElectron record, so the derived
    # fields can be computed on it directly.
    ele  = _add_derived(evts.best_merged, coll)

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

    _fill_all(hists, 'AT', evts, ele, evts.PFJet, wgt, samp, cut, {'isMerged': isMerged_label})
