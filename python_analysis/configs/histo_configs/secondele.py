from hist import Hist
from hist.axis import Variable, Regular, StrCategory
import hist
import numpy as np
import awkward as ak

# What is actually inside a "merged" electron? Compares the selected merged
# AllLptElectron in cat_merged events against single-electron references
# (gen-matched legs of the truth-matched vertex in cat_vertexed events) in:
#
#   (1) energy sharing   -- reco energy/pT relative to the charge-matched
#                           ("own") gen leg and to the gen pair sum, vs. the
#                           gen energy fraction carried by the own leg
#   (2) shower shape     -- SC width / cluster multiplicity / R9 / 2nd-max
#                           fractions, 1D and vs. gen dR(e+e-) and reco pT
#   (4) track quality    -- hit counts, chi2, missing inner hits, 1D and vs.
#                           gen dR(e+e-)
#   (3a) partner gen leg -- is the gen leg the merged candidate is *not*
#                           charge-matched to even in acceptance?
#
# Interpretation of (1): if the candidate is truly calorimeter-merged, its
# SC energy tracks the pair sum (E_SC/E_sum ~ 1 at any own-leg fraction); if
# only one leg was reconstructed, E_SC/E_sum follows the own-leg fraction
# (the diagonal), which is where the resolved reference legs sit by
# construction. There is no raw track-momentum branch on AllLptElectron, so
# E/p mismatch is accessed via abs1overEm1overP (|1/E_ecal - 1/p_trk|).
#
# Prefixes: 'mele' = merged candidate, 'refele' = vertexed reference legs.

# Same convention as mergedcats.py/resolvedcats.py.
RAW_COUNTS = True
_STORAGE = hist.storage.Double() if RAW_COUNTS else hist.storage.Weight()

# Partner-leg acceptance (3a): tracker |eta| coverage and a rough low-pT
# electron seeding threshold. Adjust as needed.
_ACC_ETA_MAX = 2.5
_ACC_PT_MIN  = 1.0   # GeV

# Reference-leg gen match: dR to same-charge gen and relative pT window.
_REF_DR_MAX     = 0.1
_REF_PT_REL_MAX = 0.2

_PREFIXES = ['mele', 'refele']

samp = StrCategory([], name="samp", label="Sample Name", growth=True)
cut  = StrCategory([], name="cut",  label="Cut Applied",  growth=True)

def _int_axis(n, name, label):
    # Integer-valued quantity on a Regular axis with bins centered on integers
    # (robust to the branch being stored as float).
    return Regular(n, -0.5, n - 0.5, name=name, label=label)

# ── Slicing axes ─────────────────────────────────────────────────────────────
_gen_ee_dr = Variable(
    [0, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0],
    name='geneedr', label=r'Gen $\Delta R(e^+, e^-)$',
)
_reco_pt = Variable(
    [0, 2, 4, 6, 8, 10, 15, 20, 30, 50],
    name='recopt', label=r'Reco $p_T$ (GeV)',
)
_gen_lxy = Variable(
    [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1, 1.5, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 50, 70, 100, 150],
    name='lxy', label='Gen $L_{xy}$ (cm)',
)

# ── (1) Energy sharing ───────────────────────────────────────────────────────
_own_frac = Regular(50, 0, 1, name='ownfrac', label=r'$E^{\rm gen}_{\rm own} / (E^{\rm gen}_{e^-} + E^{\rm gen}_{e^+})$')

# (hist key, axis); values computed in _energy_vars
_ENERGY_VARS = [
    ('scE_over_genEown',   Regular(60, 0, 3,   name='ratio', label=r'$E_{\rm SC} / E^{\rm gen}_{\rm own}$')),
    ('scE_over_genEsum',   Regular(60, 0, 1.5, name='ratio', label=r'$E_{\rm SC} / E^{\rm gen}_{\rm sum}$')),
    ('scRawE_over_genEsum',Regular(60, 0, 1.5, name='ratio', label=r'$E_{\rm SC}^{\rm raw} / E^{\rm gen}_{\rm sum}$')),
    ('pt_over_genPtown',   Regular(60, 0, 3,   name='ratio', label=r'$p_T / p_T^{\rm gen,own}$')),
    ('pt_over_genPtsum',   Regular(60, 0, 1.5, name='ratio', label=r'$p_T / p_T^{\rm gen,sum}$')),
    ('scE_over_e',         Regular(60, 0, 3,   name='ratio', label=r'$E_{\rm SC} / E_{\rm ele}$')),
    ('invEmP',             Regular(50, 0, 0.5, name='invEmP', label=r'$|1/E - 1/p|$ (GeV$^{-1}$)')),
]
# Energy vars that also get a 2D vs. own-leg gen energy fraction.
_ENERGY_VS_FRAC = ['scE_over_genEown', 'scE_over_genEsum', 'pt_over_genPtsum', 'invEmP']

# ── (2) Shower shape: (field on AllLptElectron, axis) ────────────────────────
_SHAPE_VARS = [
    ('scClustersSize',       _int_axis(10, 'nclus',   'SC Number of Clusters')),
    ('full55e2ndOverScRawE', Regular(50, 0, 1,    name='e2nd',   label=r'$E_{\rm 2nd} / E_{\rm SC}^{\rm raw}$')),
    ('full55eMaxOverScRawE', Regular(50, 0, 1.2,  name='emax',   label=r'$E_{\rm max} / E_{\rm SC}^{\rm raw}$')),
    ('seedEoverScRawE',      Regular(50, 0, 1.2,  name='seedE',  label=r'$E_{\rm seed} / E_{\rm SC}^{\rm raw}$')),
    ('scPhiWidth',           Regular(50, 0, 0.15, name='phiw',   label=r'SC $\phi$ Width')),
    ('scEtaWidth',           Regular(50, 0, 0.05, name='etaw',   label=r'SC $\eta$ Width')),
    ('full55r9',             Regular(50, 0, 1.2,  name='r9',     label='R9 (5x5)')),
    ('full55sigmaIetaIeta',  Regular(50, 0, 0.05, name='sieie',  label=r'$\sigma_{i\eta i\eta}$ (5x5)')),
    ('full55sigmaIphiIphi',  Regular(50, 0, 0.08, name='sipip',  label=r'$\sigma_{i\phi i\phi}$ (5x5)')),
]

# ── (4) Track quality: (field on AllLptElectron, axis) ───────────────────────
_TRK_VARS = [
    ('numTrackerHits',      _int_axis(35, 'ntrk',   'Number of Tracker Hits')),
    ('numPixHits',          _int_axis(10, 'npix',   'Number of Pixel Hits')),
    ('numStripHits',        _int_axis(30, 'nstrip', 'Number of Strip Hits')),
    ('trkChi2',             Regular(50, 0, 50, name='chi2', label=r'Track $\chi^2/df$')),
    ('trkProb',             Regular(50, 0, 1,  name='prob', label=r'Track $\chi^2$ Probability')),
    ('expMissingInnerHits', _int_axis(6, 'missHits', 'Exp. Missing Inner Hits')),
]

# ── (3a) Partner-leg acceptance ──────────────────────────────────────────────
_acc_cat = StrCategory(['in_acc', 'fail_pt', 'fail_eta', 'fail_both'], name='acc', label='Partner gen leg acceptance')
_partner_pt      = Regular(50, 0, 25,  name='pt',      label=r'Partner gen $p_T$ (GeV)')
_partner_abseta  = Regular(50, 0, 5,   name='abseta',  label=r'Partner gen $|\eta|$')
_partner_ptfrac  = Regular(50, 0, 1,   name='ptfrac',  label=r'$p_T^{\rm gen,partner} / p_T^{\rm gen,sum}$')


def make_histograms():
    histograms = {}
    for p in _PREFIXES:
        # (1)
        for key, axis in _ENERGY_VARS:
            histograms[f'{p}_{key}'] = Hist(samp, cut, axis, storage=_STORAGE)
            if key in _ENERGY_VS_FRAC:
                histograms[f'{p}_{key}_vs_ownfrac'] = Hist(samp, cut, _own_frac, axis, storage=_STORAGE)
        histograms[f'{p}_ownfrac'] = Hist(samp, cut, _own_frac, storage=_STORAGE)
        histograms[f'{p}_gen_ee_dr'] = Hist(samp, cut, _gen_ee_dr, storage=_STORAGE)
        histograms[f'{p}_pt']        = Hist(samp, cut, _reco_pt,   storage=_STORAGE)

        # (2)
        for field, axis in _SHAPE_VARS:
            histograms[f'{p}_{field}']             = Hist(samp, cut, axis,             storage=_STORAGE)
            histograms[f'{p}_{field}_vs_geneedr']  = Hist(samp, cut, _gen_ee_dr, axis, storage=_STORAGE)
            histograms[f'{p}_{field}_vs_recopt']   = Hist(samp, cut, _reco_pt,   axis, storage=_STORAGE)

        # (4)
        for field, axis in _TRK_VARS:
            histograms[f'{p}_{field}']            = Hist(samp, cut, axis,             storage=_STORAGE)
            histograms[f'{p}_{field}_vs_geneedr'] = Hist(samp, cut, _gen_ee_dr, axis, storage=_STORAGE)

    # (3a) merged only: for the reference, the partner is the other
    # (reconstructed) vertex leg, so it is trivially in acceptance.
    histograms['mele_partner_acc']           = Hist(samp, cut, _acc_cat,                 storage=_STORAGE)
    histograms['mele_partner_acc_vs_lxy']    = Hist(samp, cut, _gen_lxy, _acc_cat,       storage=_STORAGE)
    histograms['mele_partner_acc_vs_geneedr']= Hist(samp, cut, _gen_ee_dr, _acc_cat,     storage=_STORAGE)
    histograms['mele_partner_pt']            = Hist(samp, cut, _partner_pt,              storage=_STORAGE)
    histograms['mele_partner_abseta']        = Hist(samp, cut, _partner_abseta,          storage=_STORAGE)
    histograms['mele_partner_ptfrac']        = Hist(samp, cut, _partner_ptfrac,          storage=_STORAGE)
    histograms['mele_partner_pt_vs_abseta']  = Hist(samp, cut, _partner_abseta, _partner_pt, storage=_STORAGE)
    return histograms


def _energy_vars(ele, own, partner):
    e_sum  = own.energy + partner.energy
    pt_sum = own.pt + partner.pt
    return {
        'scE_over_genEown':    ele.scEnergy / own.energy,
        'scE_over_genEsum':    ele.scEnergy / e_sum,
        'scRawE_over_genEsum': ele.scRawE / e_sum,
        'pt_over_genPtown':    ele.pt / own.pt,
        'pt_over_genPtsum':    ele.pt / pt_sum,
        'scE_over_e':          ele.scEnergy / ele.e,
        'invEmP':              ele.abs1overEm1overP,
    }, own.energy / e_sum


def _fill_ele(hists, p, ele, own, partner, gen_ee_dr, w, samp, cut):
    """Fill items (1), (2), (4) for one reco electron per entry, given its
    charge-matched ("own") gen leg and the other ("partner") gen leg."""
    evars, own_frac = _energy_vars(ele, own, partner)
    for key, axis in _ENERGY_VARS:
        hists[f'{p}_{key}'].fill(samp=samp, cut=cut, weight=w, **{axis.name: evars[key]})
        if key in _ENERGY_VS_FRAC:
            hists[f'{p}_{key}_vs_ownfrac'].fill(samp=samp, cut=cut, weight=w,
                                                ownfrac=own_frac, **{axis.name: evars[key]})
    hists[f'{p}_ownfrac'  ].fill(samp=samp, cut=cut, weight=w, ownfrac=own_frac)
    hists[f'{p}_gen_ee_dr'].fill(samp=samp, cut=cut, weight=w, geneedr=gen_ee_dr)
    hists[f'{p}_pt'       ].fill(samp=samp, cut=cut, weight=w, recopt=ele.pt)

    # Skip fields absent from older ntuples (e.g. full55e2ndOverScRawE,
    # full55eMaxOverScRawE, seedEoverScRawE predate the Nov 2026 skimmer);
    # their histograms stay booked but empty.
    for field, axis in _SHAPE_VARS:
        if field not in ele.fields:
            continue
        val = getattr(ele, field)
        hists[f'{p}_{field}'           ].fill(samp=samp, cut=cut, weight=w, **{axis.name: val})
        hists[f'{p}_{field}_vs_geneedr'].fill(samp=samp, cut=cut, weight=w, geneedr=gen_ee_dr, **{axis.name: val})
        hists[f'{p}_{field}_vs_recopt' ].fill(samp=samp, cut=cut, weight=w, recopt=ele.pt,     **{axis.name: val})

    for field, axis in _TRK_VARS:
        if field not in ele.fields:
            continue
        val = getattr(ele, field)
        hists[f'{p}_{field}'           ].fill(samp=samp, cut=cut, weight=w, **{axis.name: val})
        hists[f'{p}_{field}_vs_geneedr'].fill(samp=samp, cut=cut, weight=w, geneedr=gen_ee_dr, **{axis.name: val})


def _own_partner(charge, ge, gp):
    """Own = gen leg with the same charge as the reco electron; partner = other."""
    is_neg  = charge == -1
    own     = ak.where(is_neg, ge, gp)
    partner = ak.where(is_neg, gp, ge)
    return own, partner


def _dr(eta1, phi1, eta2, phi2):
    d = np.abs(phi1 - phi2)
    d = ak.where(d > np.pi, 2 * np.pi - d, d)
    return np.sqrt((eta1 - eta2)**2 + d**2)


# Provides cat_merged, cat_vertexed, vertexed_vtx (with e1/e2 legs),
# gen_lxy and AllLptElectron.is_merged -- see
# analysisTools/analysisSubroutines.computeMergedCatVars.
subroutines = ['computeMergedCatVars']


def fillHistos(events, hists, samp, cut, info, sum_wgt=1):
    if RAW_COUNTS:
        wgt = ak.ones_like(events.eventWgt)
    else:
        wgt = events.eventWgt / sum_wgt

    coll = events.AllLptElectron

    # ── Merged candidates ────────────────────────────────────────────────────
    _cat_merged = events.cat_merged
    if ak.sum(_cat_merged) > 0:
        me  = ak.firsts(coll[coll.is_merged])[_cat_merged]
        w   = wgt[_cat_merged]
        ge  = events.GenEle[_cat_merged]
        gp  = events.GenPos[_cat_merged]
        dr  = events.genEE.dr[_cat_merged]
        lxy = events.gen_lxy[_cat_merged]

        own, partner = _own_partner(me.charge, ge, gp)
        _fill_ele(hists, 'mele', me, own, partner, dr, w, samp, cut)

        # (3a) Is the partner gen leg even in acceptance?
        pass_pt  = np.asarray(partner.pt > _ACC_PT_MIN)
        pass_eta = np.asarray(np.abs(partner.eta) < _ACC_ETA_MAX)
        acc = np.where(pass_pt & pass_eta, 'in_acc',
              np.where(~pass_pt & pass_eta, 'fail_pt',
              np.where(pass_pt & ~pass_eta, 'fail_eta', 'fail_both')))
        hists['mele_partner_acc'           ].fill(samp=samp, cut=cut, weight=w, acc=acc)
        hists['mele_partner_acc_vs_lxy'    ].fill(samp=samp, cut=cut, weight=w, lxy=lxy, acc=acc)
        hists['mele_partner_acc_vs_geneedr'].fill(samp=samp, cut=cut, weight=w, geneedr=dr, acc=acc)
        hists['mele_partner_pt'            ].fill(samp=samp, cut=cut, weight=w, pt=partner.pt)
        hists['mele_partner_abseta'        ].fill(samp=samp, cut=cut, weight=w, abseta=np.abs(partner.eta))
        hists['mele_partner_ptfrac'        ].fill(samp=samp, cut=cut, weight=w, ptfrac=partner.pt / (own.pt + partner.pt))
        hists['mele_partner_pt_vs_abseta'  ].fill(samp=samp, cut=cut, weight=w, abseta=np.abs(partner.eta), pt=partner.pt)

    # ── Single-electron references: gen-matched vertexed legs ────────────────
    _cat_vertexed = events.cat_vertexed
    if ak.sum(_cat_vertexed) > 0:
        v  = events.vertexed_vtx[_cat_vertexed]
        w  = wgt[_cat_vertexed]
        ge = events.GenEle[_cat_vertexed]
        gp = events.GenPos[_cat_vertexed]
        dr = events.genEE.dr[_cat_vertexed]

        for leg in (v.e1, v.e2):
            own, partner = _own_partner(leg.charge, ge, gp)
            matched = (
                (_dr(leg.eta, leg.phi, own.eta, own.phi) < _REF_DR_MAX) &
                (np.abs(leg.pt - own.pt) / own.pt < _REF_PT_REL_MAX)
            )
            matched = ak.fill_none(matched, False)
            if ak.sum(matched) == 0:
                continue
            _fill_ele(hists, 'refele', leg[matched], own[matched], partner[matched],
                      dr[matched], w[matched], samp, cut)
