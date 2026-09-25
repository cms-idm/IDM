# imports
import os
import copy
import numpy as np
import matplotlib.pyplot as plt

from hist import Hist
from hist.storage import Weight

import analysisTools.utils as utils
import coffea.util as util

outdir = 'workarea'
plotdir = './plots/vtxrecoeff'
os.makedirs(plotdir, exist_ok=True)

sig_vers = 'Jul2026noID'
selection = 'an'
hists = 'genstudy'
saved_signal_hists = f'{outdir}/hists_sig{sig_vers}_{selection}-sel_{hists}.coffea'
plottag = f'sig{sig_vers}_{selection}-sel'

# Substring (case-insensitive) used to find the vertex-reco cutflow stage by its
# description -- matches cut9's `desc` in configs/cut_configs/an_selection.py
# ("Has Good ee Vertex").
VERTEX_CUT_KEYWORD = 'good ee vertex'

# genstudy's histograms aren't filled at the true, pre-selection 'all' stage
# (see analysisTools.Analyzer._process -- histoFill only runs on cuts with
# savePlots=True, and 'all' never gets a fill), so cut1 (MET-filters only,
# ~99.96% efficient for signal) is used as the denominator instead.
DENOM_CUT_IDX = 'cut1'


def n_eff(h):
    """
    Per-bin effective raw MC event count: N_eff = sum(w)^2 / sum(w^2), i.e.
    vals^2/variances. This undoes each sample's xsec*lumi/sum_wgt event weight,
    recovering (approximately) how many simulated events actually landed in
    each bin, regardless of that sample's cross section.
    """
    vals, var = h.values(), h.variances()
    return np.where(var > 0, vals**2 / np.where(var > 0, var, 1), 0)


def combine_neff_ratio(h, samples, num_cut, den_cut):
    """
    Combine per-sample dR histograms into a single efficiency curve by summing
    each sample's per-bin N_eff (effective raw MC counts) rather than its
    xsec-weighted yield -- so samples with more simulated statistics dominate
    the combination instead of samples with higher cross section.

    Returns a 1D Hist (same dr axis as `h`) holding the combined efficiency,
    with a binomial variance based on the combined N_eff.
    """
    dr_axis = copy.deepcopy(h[{"cut": den_cut, "samp": samples[0]}].axes[0])
    n_bins = len(dr_axis)

    num_neff_tot = np.zeros(n_bins)
    den_neff_tot = np.zeros(n_bins)
    for s in samples:
        num_neff_tot += n_eff(h[{"cut": num_cut, "samp": s}])
        den_neff_tot += n_eff(h[{"cut": den_cut, "samp": s}])

    with np.errstate(invalid='ignore', divide='ignore'):
        ratio = np.where(den_neff_tot > 0, num_neff_tot / den_neff_tot, 0)
        ratio_var = np.where(den_neff_tot > 0, ratio * (1 - ratio) / den_neff_tot, 0)

    h_ratio = Hist(dr_axis, storage=Weight())
    h_ratio.view(flow=False)[...] = np.stack([ratio, ratio_var], axis=-1)
    return h_ratio


def plot_vtx_eff_vs_gendr(sig_histo, plot_dict):
    """
    Plot vertex-reco efficiency vs gen ee deltaR, combining all signal samples
    into a single curve via N_eff (effective raw MC count) weighting -- see
    combine_neff_ratio -- rather than a straight xsec-weighted sum, so the
    combination isn't dominated by whichever mass point has the largest
    cross section.

    Example plot_dict:

    plot_dict = {
        'ylim': [0, None],
        'title': '',
        'doSave': True,
        'outDir': plotdir,
        'outName': 'vtxrecoeff_vs_gendr.png',
    }
    """
    vertex_idx = utils.find_cut_idx_by_desc(sig_histo, VERTEX_CUT_KEYWORD)
    samples = list(sig_histo['cutflow'].keys())

    h = sig_histo['gen_diele_dR']
    h_eff = combine_neff_ratio(h, samples, num_cut=vertex_idx, den_cut=DENOM_CUT_IDX)

    fig, ax = plt.subplots(figsize=(10, 8))
    centers = h_eff.axes[0].centers
    vals = h_eff.values()
    errs = np.sqrt(h_eff.variances())
    ax.errorbar(centers, vals, yerr=errs, fmt='-o', markersize=4, color='C0', capsize=3)

    ax.grid()
    ax.set_xlabel(r'Gen $\Delta R(e^+e^-)$', fontsize=20)
    ax.set_ylabel('Vertex Reco Efficiency', fontsize=20)
    ax.set_title(plot_dict.get('title', ''), fontsize=18)
    ax.tick_params(axis='both', labelsize=14)

    ylim = plot_dict.get('ylim')
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    if plot_dict.get('doSave', False):
        os.makedirs(plot_dict['outDir'], exist_ok=True)
        plt.tight_layout()
        outpath = f"{plot_dict['outDir']}/{plot_dict['outName']}"
        plt.savefig(outpath)
        print(f"Saved: {outpath}")

    plt.close(fig)


# ---- Load histograms and make plot -----------------------------------------

s_hists = util.load(saved_signal_hists)[0]

plot_dict = {
    'ylim': [0, None],
    'title': r'Vertex Reco Efficiency vs Gen $ee$ $\Delta R$ [AN Selection, All Signal Samples]',
    'doSave': True,
    'outDir': plotdir,
    'outName': f'vtxrecoeff_vs_gendr_{plottag}_allsamples.png',
}
plot_vtx_eff_vs_gendr(s_hists, plot_dict)
