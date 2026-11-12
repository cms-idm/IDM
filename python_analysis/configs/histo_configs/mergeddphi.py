try:
    from histobins import *
except ModuleNotFoundError:
    from configs.histo_configs.histobins import *
from hist import Hist
from hist.axis import Regular
import hist
import numpy as np
import awkward as ak

# |dPhi| between every pair of five objects, for signal events with a
# truth-matched merged electron:
#
#   chi2    : gen chi2 (GenPart with |ID| == 1000023)
#   genee   : gen dielectron system (genEE)
#   mele    : reco merged electron -- the AllLptElectron with is_merged
#   genmet  : GenMET
#   recomet : PFMET
#
# Event selection: events.cat_merged from computeMergedCatVars, i.e. a single
# AllLptElectron within dR < 0.1 of both gen electrons, quality-matched to one
# of them, with no second matched reco electron and no photon/track/conversion
# alternative for either gen, and the event not vertexed. This is the truth
# definition only -- no good-merged-candidate (isGoodMerged) requirement.

_OBJECTS = ['chi2', 'genee', 'mele', 'genmet', 'recomet']
_LABELS = {
    'chi2':    r'\chi_2^{\rm gen}',
    'genee':   r'(ee)^{\rm gen}',
    'mele':    r'e_{\rm merged}^{\rm reco}',
    'genmet':  r'p_T^{\rm miss,gen}',
    'recomet': r'p_T^{\rm miss,reco}',
}
_PAIRS = [(a, b) for i, a in enumerate(_OBJECTS) for b in _OBJECTS[i+1:]]

def _dphi_axis(a, b):
    upplim = 0.05
    if a == 'recomet' or b == 'recomet':
        upplim = 0.5
    elif a == 'mele' or b == 'mele':
        upplim = 0.1
    return Regular(20, 0, upplim, name='dphi', label=rf'$|\Delta\phi({_LABELS[a]}, {_LABELS[b]})|$')

def make_histograms():
    histograms = {}
    for a, b in _PAIRS:
        histograms[f'dphi_{a}_{b}'] = Hist(samp, cut, _dphi_axis(a, b), storage=hist.storage.Weight())
    return histograms

# Sets events.cat_merged and AllLptElectron.is_merged once per chunk, before
# the cut loop (no-op for background, which has no GenEle).
subroutines = ['computeMergedCatVars']

def _absdphi(a, b):
    d = np.abs(a - b) % (2 * np.pi)
    return ak.where(d > np.pi, 2 * np.pi - d, d)

def fillHistos(events, hists, samp, cut, info, sum_wgt=1):
    if 'cat_merged' not in events.fields:
        return
    sel = events.cat_merged
    if ak.sum(sel) == 0:
        return

    ev  = events[sel]
    wgt = ev.eventWgt / sum_wgt

    chi2 = ak.firsts(ev.GenPart[np.abs(ev.GenPart.ID) == 1000023])
    mele = ak.firsts(ev.AllLptElectron[ev.AllLptElectron.is_merged])

    phis = {
        'chi2':    chi2.phi,
        'genee':   ev.genEE.phi,
        'mele':    mele.phi,
        'genmet':  ev.GenMET.phi,
        'recomet': ev.PFMET.phi,
    }

    for a, b in _PAIRS:
        dphi = _absdphi(phis[a], phis[b])
        ok = ~ak.is_none(dphi)
        hists[f'dphi_{a}_{b}'].fill(
            samp=samp, cut=cut,
            dphi=ak.to_numpy(dphi[ok]),
            weight=ak.to_numpy(wgt[ok]),
        )
