try:
    from histobins import *
except ModuleNotFoundError:
    from configs.histo_configs.histobins import *
from hist import Hist
import hist
import numpy as np
import awkward as ak

# Inputs for correlation matrices of the vertexed-dielectron and merged-electron
# variable sets (e.g. for BDT feature selection).
#
# Rather than one binned 2D hist per variable pair, each collection gets a single
# "moment matrix": a 2D hist over (var_i, var_j) string categories whose bin
# (i,j) holds sum_w(x_i * x_j). The extra category _CONST is the constant 1, so
# row/column _CONST holds sum_w(x_i) and (_CONST,_CONST) holds sum_w. That is
# enough to compute the exact weighted Pearson correlation of every pair -- see
# correlation_matrix() below. Storage is Double since the per-bin "weights" are
# moments, not event weights (a Weight storage's variance would be meaningless).
#
# Event selection (the two matrices are filled independently):
#   vertexed: events with >= 1 entry in events.vtx (no good-vertex requirement);
#             uses the lowest-reduced_chi2 vertex among ALL vertices, since
#             sel_vtx only exists for events with a good vertex.
#   merged:   events with a merged candidate, i.e. events.best_merged (from
#             selectBestMergedCandidate) is not None.
# Events with any non-finite variable (e.g. no PFJet, log10 of dz=0) are dropped
# from that matrix entirely, so all moments are computed on the same events.

_CONST = '_const'
_LOG_FLOOR = 1e-4

def _log10(x, floor=_LOG_FLOOR):
    # heavy-tailed positive variables are log-scaled so Pearson isn't dominated
    # by the tails; values at/below the floor are clamped to it
    return np.log10(np.maximum(x, floor))

def _dphi(a, b):
    d = np.abs(a - b)
    return ak.where(d > np.pi, 2 * np.pi - d, d)

def _lead_jet(ev, field):
    return ak.firsts(ev.PFJet[field])

# (name, extractor(obj, ev)) -- obj is the selected vertex or merged candidate
_VTX_VARS = [
    ('log10_chi2',          lambda v, ev: _log10(v.reduced_chi2)),
    ('METdPhi',             lambda v, ev: np.abs(v.METdPhi)),
    ('refit_m',             lambda v, ev: v.refit_m),
    ('refit_dR',            lambda v, ev: v.refit_dR),
    ('log10_corrMinDxy',    lambda v, ev: _log10(v.corrMinDxy)),
    ('log10_minDxy',        lambda v, ev: _log10(v.min_dxy)),
    ('log10_vxy',           lambda v, ev: _log10(v.vxy)),
    ('log10_Lxy',           lambda v, ev: _log10(np.sqrt((v.vx - ev.PV.x)**2 + (v.vy - ev.PV.y)**2))),
    ('log10_vxy_signif',    lambda v, ev: _log10(v.vxy / v.sigmavxy)),
    ('cos_collinear',       lambda v, ev: v.cos_collinear),
    ('prod_eta',            lambda v, ev: v.e1.eta * v.e2.eta),
    ('met_leadPt_ratio',    lambda v, ev: ev.PFMET.pt / _lead_jet(ev, 'pt')),
    ('jetMETdPhi',          lambda v, ev: np.abs(_lead_jet(ev, 'METdPhi'))),
    ('minJetMETdPhi',       lambda v, ev: ak.min(np.abs(ev.PFJet.METdPhi), axis=1)),
    ('refit_pt',            lambda v, ev: v.refit_pt),
    ('minLogDxyDz',         lambda v, ev: np.minimum(np.log10(np.abs(v.e1.dxy / v.e1.dz)), np.log10(np.abs(v.e2.dxy / v.e2.dz)))),
    ('log10_maxMiniIso',    lambda v, ev: _log10(np.maximum(v.e1.miniRelIsoEleCorr, v.e2.miniRelIsoEleCorr))),
    ('mindRj',              lambda v, ev: v.mindRj),
    ('met_pt',              lambda v, ev: ev.PFMET.pt),
]

_MELE_VARS = [
    ('pt',                  lambda me, ev: me.pt),
    ('eta',                 lambda me, ev: me.eta),
    ('log10_dxy',           lambda me, ev: _log10(np.abs(me.dxy))),
    ('log10_dxy_signif',    lambda me, ev: _log10(np.abs(me.dxy) / me.dxyErr)),
    ('log10_dz',            lambda me, ev: _log10(np.abs(me.dz))),
    ('log10_dz_signif',     lambda me, ev: _log10(np.abs(me.dz) / me.dzErr)),
    ('logDxyDz',            lambda me, ev: np.log10(np.abs(me.dxy / me.dz))),
    ('METdPhi',             lambda me, ev: _dphi(me.phi, ev.PFMET.phi)),
    ('cosMETdPhi',          lambda me, ev: np.cos(_dphi(me.phi, ev.PFMET.phi))),
    # Lxy estimate |dxy|/sin(dPhi(e, ptmiss)); sin=0 gives inf and drops the event
    ('log10_lxyEst',        lambda me, ev: _log10(np.abs(me.dxy) / np.sin(_dphi(me.phi, ev.PFMET.phi)))),
    ('mindRj',              lambda me, ev: me.mindRj),
    ('mindPhiJ',            lambda me, ev: me.mindPhiJ),
    ('dRj0',                lambda me, ev: me.dRj0),
    ('log10_trkChi2',       lambda me, ev: _log10(me.trkChi2)),
    ('trkProb',             lambda me, ev: me.trkProb),
    ('numTrackerHits',      lambda me, ev: me.numTrackerHits),
    ('numPixHits',          lambda me, ev: me.numPixHits),
    ('ID',                  lambda me, ev: me.ID),
    ('log10_PFRelIso',      lambda me, ev: _log10(me.PFRelIso)),
    ('log10_miniRelIso',    lambda me, ev: _log10(me.miniRelIso)),
    ('log10_trkRelIso',     lambda me, ev: _log10(me.trkRelIso)),
    ('HoverE',              lambda me, ev: me.HoverE),
    ('EoverPInv',           lambda me, ev: me.abs1overEm1overP),
    ('dEtaSeed',            lambda me, ev: me.absdEtaSeed),
    ('dPhiIn',              lambda me, ev: me.absdPhiIn),
    ('sigmaIetaIeta',       lambda me, ev: me.full55sigmaIetaIeta),
    ('scEtaWidth',          lambda me, ev: me.scEtaWidth),
    ('scPhiWidth',          lambda me, ev: me.scPhiWidth),
    ('missingInnerHits',    lambda me, ev: me.expMissingInnerHits),
    ('conversionVeto',      lambda me, ev: me.conversionVeto),
    ('svProxyVxy',          lambda me, ev: me.svProxyVxy),
    ('met_leadPt_ratio',    lambda me, ev: ev.PFMET.pt / _lead_jet(ev, 'pt')),
    ('met_pt',              lambda me, ev: ev.PFMET.pt),
]

def _moment_hist(var_spec):
    names = [name for name, _ in var_spec] + [_CONST]
    return Hist(
        samp, cut,
        StrCategory(names, name='var_i', label='Variable'),
        StrCategory(names, name='var_j', label='Variable'),
        storage=hist.storage.Double(),
    )

def make_histograms():
    histograms = {}
    histograms['corr_vtx']  = _moment_hist(_VTX_VARS)
    histograms['corr_mele'] = _moment_hist(_MELE_VARS)
    return histograms

subroutines = []

def _fill_moments(h, samp, cut, var_spec, obj, ev, wgt):
    if len(ev) == 0:
        return
    names = [name for name, _ in var_spec] + [_CONST]
    # values_astype before fill_none so bool/int option-type fields don't turn
    # into a union type that to_numpy can't handle; to_numpy can still hand back
    # a numpy masked array for option types, which breaks the matmul below, so
    # masked entries are filled with nan (and dropped by the finite cut)
    cols = [np.ma.filled(ak.to_numpy(ak.fill_none(ak.values_astype(get_val(obj, ev), np.float64), np.nan)), np.nan) for _, get_val in var_spec]
    X = np.stack(cols + [np.ones(len(ev))], axis=1)
    w = np.ma.filled(ak.to_numpy(wgt).astype(np.float64), np.nan)

    finite = np.all(np.isfinite(X), axis=1) & np.isfinite(w)
    X, w = X[finite], w[finite]

    M = X.T @ (X * w[:, None])
    ii, jj = np.meshgrid(names, names, indexing='ij')
    h.fill(samp=samp, cut=cut, var_i=ii.ravel(), var_j=jj.ravel(), weight=M.ravel())

def fillHistos(events, hists, samp, cut, info, sum_wgt=1):
    wgt = events.eventWgt / sum_wgt

    # ── Vertexed: lowest-chi2 vertex among all vertices ──────────────────────
    has_vtx = ak.num(events.vtx.vxy, axis=1) > 0
    ev_v = events[has_vtx]
    v = ak.firsts(ev_v.vtx[ak.argmin(ev_v.vtx.reduced_chi2, axis=1, keepdims=True)])
    _fill_moments(hists['corr_vtx'], samp, cut, _VTX_VARS, v, ev_v, wgt[has_vtx])

    # ── Merged: events with a merged candidate ───────────────────────────────
    if 'best_merged' in events.fields:
        has_me = ~ak.is_none(events.best_merged)
        ev_m = events[has_me]
        _fill_moments(hists['corr_mele'], samp, cut, _MELE_VARS, ev_m.best_merged, ev_m, wgt[has_me])

def correlation_matrix(h, samp, cut):
    """
    Decode a moment-matrix hist from this config into (names, corr), the
    weighted Pearson correlation matrix for one sample/cut.
    """
    M = h[{'samp': hist.loc(samp), 'cut': hist.loc(cut)}].values()
    names = list(h.axes['var_i'])
    k = names.index(_CONST)
    keep = [i for i in range(len(names)) if i != k]

    S = M / M[k, k]
    mean = S[keep, k]
    cov = S[np.ix_(keep, keep)] - np.outer(mean, mean)
    sd = np.sqrt(np.diag(cov))
    return [names[i] for i in keep], cov / np.outer(sd, sd)
