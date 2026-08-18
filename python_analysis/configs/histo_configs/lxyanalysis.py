try:
    from histobins import *
except ModuleNotFoundError:
    from configs.histo_configs.histobins import *
from hist import Hist
from hist.axis import Regular, StrCategory
import hist
import numpy as np
import awkward as ak
from analysisTools.analysisSubroutines import defineGoodVertices

# Two event topologies, compared side-by-side via the 'evttype' axis on every hist below:
#
#   vtx:  the event has >=1 reconstructed vertex passing the standard "good vertex"
#         definition (defineGoodVertices, v15acr) AND gen-matched to both GenEle and
#         GenPos (vtx.isMatched = e1_isMatched & e2_isMatched). Among good+matched
#         vertices, the one with lowest reduced_chi2 is chosen (mirrors
#         selectBestVertex). Both vertex legs (e1, e2) are filled as separate
#         "electron" entries, each paired with the gen electron matching its charge.
#
#   mele: the event has an AllLptElectron quality-matched (dR<0.1, charge, pt within
#         20%) to one gen electron while also being within dR<0.1 of the other gen
#         (but not a full quality match to it), with neither gen having a
#         photon/track/conversion alternative -- identical to mergedcats.py/
#         mergedeles.py's merged-electron category. There is only ONE reco object for
#         two gen electrons here, so a single "electron" entry per event is filled,
#         pairing that reco object with whichever gen electron (GenEle/GenPos) has
#         the higher pt.
#
# For every "electron" entry (see above), the following are filled:
#   gen_dxy, reco_dxy, gen_betagamma, reco_betagamma,
#   gen_dphi_e_chi2 (electron vs gen chi2), reco_dphi_e_met (electron vs PFMET),
#   gen_alpha, gen_ctau, gen_lxy, reco_lxy
# gen_dphi_chi2_met (chi2 vs gen ptmiss) is event-level (not per-electron) and is
# filled once per event instead.
#
# gen_alpha = beta*gamma (of the electron) * sin(gen_dphi_e_chi2) * sin(theta), where
# theta is the electron's polar angle from the beam axis (sin(theta) = 1/cosh(eta)).
# This follows from ctau = L_3D/(beta*gamma), L_3D = L_xy/sin(theta), and the
# impact-parameter relation L_xy = dxy/sin(dphi): so gen_ctau = dxy/alpha and
# gen_lxy = dxy/sin(dphi_e_chi2). reco_lxy uses the analogous reco-side proxy angle,
# dphi(electron, PFMET), in place of dphi(electron, chi2) since chi2's direction isn't
# directly reconstructable (same substitution mergedeles.py makes for its lxyEst).

ELECTRON_MASS = 0.000511  # GeV (PDG)

_LOG = hist.axis.transform.log

evttype = StrCategory(['vtx', 'mele'], name='evttype', label='Event topology')

_dxy_log       = Regular(100, 1e-4, 40,  name='dxy',       label=r'Electron Track $d_{xy}$ [cm]',  transform=_LOG)
_dxy_lin       = Regular(100, 0,    40,  name='dxy',       label=r'Electron Track $d_{xy}$ [cm]')
_betagamma_log = Regular(100, 1,    1e6, name='betagamma', label=r'$\beta\gamma$',                 transform=_LOG)
_betagamma_lin = Regular(100, 0,    1e6, name='betagamma', label=r'$\beta\gamma$')
_dphi_lin      = Regular(100, 0,    np.pi, name='dphi',    label=r'$\Delta\phi$')
_dphi_log      = Regular(100, 1e-4, 1,   name='dphi',      label=r'$\Delta\phi$',                  transform=_LOG)
_alpha_log     = Regular(100, 1e-3, 1e6, name='alpha',     label=r'$\alpha = \beta\gamma\sin(\Delta\phi)\sin(\theta)$', transform=_LOG)
_alpha_lin     = Regular(100, 0,    1e6, name='alpha',     label=r'$\alpha = \beta\gamma\sin(\Delta\phi)\sin(\theta)$')
_ctau_log      = Regular(100, 1e-6, 1e3, name='ctau',      label=r'$c\tau$ [cm]',                  transform=_LOG)
_ctau_lin      = Regular(100, 0,    1e3, name='ctau',      label=r'$c\tau$ [cm]')
_lxy_log       = Regular(100, 1e-4, 50,  name='lxy',       label=r'$L_{xy}$ [cm]',                 transform=_LOG)
_lxy_lin       = Regular(100, 0,    50,  name='lxy',       label=r'$L_{xy}$ [cm]')

def make_histograms():
    histograms = {
        'gen_dxy':              Hist(samp, cut, evttype, _dxy_lin,       storage=hist.storage.Weight()),
        'gen_dxy_log':          Hist(samp, cut, evttype, _dxy_log,       storage=hist.storage.Weight()),
        'reco_dxy':             Hist(samp, cut, evttype, _dxy_lin,       storage=hist.storage.Weight()),
        'reco_dxy_log':         Hist(samp, cut, evttype, _dxy_log,       storage=hist.storage.Weight()),
        'gen_betagamma':        Hist(samp, cut, evttype, _betagamma_lin, storage=hist.storage.Weight()),
        'gen_betagamma_log':    Hist(samp, cut, evttype, _betagamma_log, storage=hist.storage.Weight()),
        'reco_betagamma':       Hist(samp, cut, evttype, _betagamma_lin, storage=hist.storage.Weight()),
        'reco_betagamma_log':   Hist(samp, cut, evttype, _betagamma_log, storage=hist.storage.Weight()),
        'gen_dphi_e_chi2':      Hist(samp, cut, evttype, _dphi_lin,      storage=hist.storage.Weight()),
        'gen_dphi_e_chi2_log':  Hist(samp, cut, evttype, _dphi_log,      storage=hist.storage.Weight()),
        'gen_dphi_chi2_met':    Hist(samp, cut, evttype, _dphi_lin,      storage=hist.storage.Weight()),
        'gen_dphi_chi2_met_log':Hist(samp, cut, evttype, _dphi_log,      storage=hist.storage.Weight()),
        'reco_dphi_e_met':      Hist(samp, cut, evttype, _dphi_lin,      storage=hist.storage.Weight()),
        'reco_dphi_e_met_log':  Hist(samp, cut, evttype, _dphi_log,      storage=hist.storage.Weight()),
        'gen_alpha':            Hist(samp, cut, evttype, _alpha_lin,     storage=hist.storage.Weight()),
        'gen_alpha_log':        Hist(samp, cut, evttype, _alpha_log,     storage=hist.storage.Weight()),
        'gen_ctau':             Hist(samp, cut, evttype, _ctau_lin,      storage=hist.storage.Weight()),
        'gen_ctau_log':         Hist(samp, cut, evttype, _ctau_log,      storage=hist.storage.Weight()),
        'gen_lxy':              Hist(samp, cut, evttype, _lxy_lin,       storage=hist.storage.Weight()),
        'gen_lxy_log':          Hist(samp, cut, evttype, _lxy_log,       storage=hist.storage.Weight()),
        'reco_lxy':             Hist(samp, cut, evttype, _lxy_lin,       storage=hist.storage.Weight()),
        'reco_lxy_log':         Hist(samp, cut, evttype, _lxy_log,       storage=hist.storage.Weight()),
    }
    return histograms

subroutines = []

def _dphi(a, b):
    d = np.abs(a - b)
    return ak.where(d > np.pi, 2 * np.pi - d, d)

def _fill_electron(hists, samp, cut, evttype_label, w, gen_g, reco_r, pv_x, pv_y, chi2_phi, met_phi):
    """Fill every per-electron hist for one (gen electron, reco electron) pairing."""
    gen_dxy = np.abs((-(gen_g.vx - pv_x) * gen_g.py + (gen_g.vy - pv_y) * gen_g.px) / gen_g.pt)
    gen_p3  = np.sqrt(gen_g.px**2 + gen_g.py**2 + gen_g.pz**2)
    gen_betagamma = gen_p3 / ELECTRON_MASS
    gen_dphi_e_chi2 = _dphi(gen_g.phi, chi2_phi)
    gen_sintheta = 1 / np.cosh(gen_g.eta)
    gen_alpha = gen_betagamma * np.sin(gen_dphi_e_chi2) * gen_sintheta
    gen_ctau  = gen_dxy / gen_alpha
    gen_lxy   = gen_dxy / np.sin(gen_dphi_e_chi2)

    reco_dxy = np.abs(reco_r.dxy)
    reco_p3  = reco_r.pt * np.cosh(reco_r.eta)
    reco_betagamma  = reco_p3 / ELECTRON_MASS
    reco_dphi_e_met = _dphi(reco_r.phi, met_phi)
    reco_lxy = reco_dxy / np.sin(reco_dphi_e_met)

    hists['gen_dxy'         ].fill(samp=samp, cut=cut, evttype=evttype_label, dxy=gen_dxy,           weight=w)
    hists['reco_dxy'        ].fill(samp=samp, cut=cut, evttype=evttype_label, dxy=reco_dxy,          weight=w)
    hists['gen_betagamma'   ].fill(samp=samp, cut=cut, evttype=evttype_label, betagamma=gen_betagamma,  weight=w)
    hists['reco_betagamma'  ].fill(samp=samp, cut=cut, evttype=evttype_label, betagamma=reco_betagamma, weight=w)
    hists['gen_dphi_e_chi2' ].fill(samp=samp, cut=cut, evttype=evttype_label, dphi=gen_dphi_e_chi2,   weight=w)
    hists['reco_dphi_e_met' ].fill(samp=samp, cut=cut, evttype=evttype_label, dphi=reco_dphi_e_met,   weight=w)
    hists['gen_alpha'       ].fill(samp=samp, cut=cut, evttype=evttype_label, alpha=gen_alpha,        weight=w)
    hists['gen_ctau'        ].fill(samp=samp, cut=cut, evttype=evttype_label, ctau=gen_ctau,          weight=w)
    hists['gen_lxy'         ].fill(samp=samp, cut=cut, evttype=evttype_label, lxy=gen_lxy,            weight=w)
    hists['reco_lxy'        ].fill(samp=samp, cut=cut, evttype=evttype_label, lxy=reco_lxy,           weight=w)

def fillHistos(events, hists, samp, cut, info, sum_wgt=1):
    wgt = ak.ones_like(events.eventWgt)

    chi2 = ak.firsts(events.GenPart[np.abs(events.GenPart.ID) == 1000023])
    chi1 = events.GenPart[np.abs(events.GenPart.ID) == 1000022]
    gen_met_phi = np.arctan2(ak.sum(chi1.py, axis=1), ak.sum(chi1.px, axis=1))

    # ── vtx category: good + gen-matched (both legs) vertex, lowest chi2 among them ──
    defineGoodVertices(events, version='v15acr')
    good_matched = events.vtx.isGood & events.vtx.isMatched
    gm_vtx = events.vtx[good_matched]
    has_vtx = ak.num(gm_vtx, axis=1) > 0

    if ak.sum(has_vtx) > 0:
        gm_vtx_sel = gm_vtx[has_vtx]
        sel_vtx = ak.flatten(gm_vtx_sel[ak.argmin(gm_vtx_sel.reduced_chi2, axis=1, keepdims=True)])

        w        = wgt[has_vtx]
        pv_x     = events.PV.x[has_vtx]
        pv_y     = events.PV.y[has_vtx]
        chi2_phi = chi2.phi[has_vtx]
        met_phi  = events.PFMET.phi[has_vtx]

        gen_ele_sel = events.GenEle[has_vtx]
        gen_pos_sel = events.GenPos[has_vtx]
        e1_is_ele = sel_vtx.e1.charge == -1
        gen_for_e1 = ak.where(e1_is_ele,  gen_ele_sel, gen_pos_sel)
        gen_for_e2 = ak.where(~e1_is_ele, gen_ele_sel, gen_pos_sel)

        _fill_electron(hists, samp, cut, 'vtx', w, gen_for_e1, sel_vtx.e1, pv_x, pv_y, chi2_phi, met_phi)
        _fill_electron(hists, samp, cut, 'vtx', w, gen_for_e2, sel_vtx.e2, pv_x, pv_y, chi2_phi, met_phi)

        hists['gen_dphi_chi2_met'].fill(
            samp=samp, cut=cut, evttype='vtx',
            dphi=_dphi(chi2_phi, gen_met_phi[has_vtx]), weight=w,
        )

    # ── mele category: merged-electron selection, identical to mergedeles.py ────────
    def _pt_rel(obj_pt, gen_pt):
        return np.abs(obj_pt - gen_pt) / gen_pt < 0.2

    gen_sum_pt = events.GenEle.pt + events.GenPos.pt

    def _dr(eta1, phi1, eta2, phi2):
        return np.sqrt((eta1 - eta2)**2 + _dphi(phi1, phi2)**2)

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

    only_ge_i = dr_match_ge & ~merged_i
    only_gp_i = dr_match_gp & ~merged_i
    matched_i = merged_i | dr_match_ge | dr_match_gp

    has_merged = ak.any(merged_i,  axis=1)
    n_matched  = ak.sum(matched_i, axis=1)

    _A = ak.any(only_ge_i, axis=1)
    _B = ak.any(only_gp_i, axis=1)

    _cat_sep    = (_A & _B & ~has_merged) | (has_merged & (n_matched > 1))
    _cat_merged = ~_cat_sep & has_merged & ~_ge_any & ~_gp_any

    if ak.sum(_cat_merged) > 0:
        me = ak.firsts(coll[merged_i])[_cat_merged]
        w  = wgt[_cat_merged]

        gen_ele_sel = events.GenEle[_cat_merged]
        gen_pos_sel = events.GenPos[_cat_merged]
        lead_is_ele = gen_ele_sel.pt >= gen_pos_sel.pt
        lead_gen    = ak.where(lead_is_ele, gen_ele_sel, gen_pos_sel)

        pv_x     = events.PV.x[_cat_merged]
        pv_y     = events.PV.y[_cat_merged]
        chi2_phi = chi2.phi[_cat_merged]
        met_phi  = events.PFMET.phi[_cat_merged]

        _fill_electron(hists, samp, cut, 'mele', w, lead_gen, me, pv_x, pv_y, chi2_phi, met_phi)

        hists['gen_dphi_chi2_met'].fill(
            samp=samp, cut=cut, evttype='mele',
            dphi=_dphi(chi2_phi, gen_met_phi[_cat_merged]), weight=w,
        )
