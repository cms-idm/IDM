# imports
import os
import copy
import numpy as np
import matplotlib.pyplot as plt

from hist import Hist
from hist.storage import Weight

import coffea.util as util

outdir = 'workarea'
plotdir = './plots/vtxreco'
os.makedirs(plotdir, exist_ok=True)

sig_vers = 'Jul2026noID'
selection = 'an'
hists = 'vtxreco'
saved_signal_hists = f'{outdir}/hists_sig{sig_vers}_{selection}-sel_{hists}.coffea'
plottag = f'sig{sig_vers}_{selection}-sel'

# Cut stages to plot at -- both are before the good-vertex cut (cut9), so the
# efficiency isn't trivially 1. See configs/cut_configs/an_selection.py.
CUTS = {
    'cut1': 'inclusive',
    'cut8': 'preselected',
}

# Gen variables filled in configs/histo_configs/vtxreco.py: name -> (x label, log x)
VARIABLES = {
    'gen_ee_pt'     : (r'Gen $p_{T}(e^+e^-)$ [GeV]', False),
    'gen_lxy'       : (r'Gen $L_{xy}$ [cm]',         False),
    'gen_lxy_log'   : (r'Gen $L_{xy}$ [cm]',         True),
    'gen_ee_dr'     : (r'Gen $\Delta R(e^+e^-)$',    False),
    'gen_ee_dr_log' : (r'Gen $\Delta R(e^+e^-)$',    True),
    'gen_ee_eta'    : (r'Gen $\eta(e^+e^-)$',        False),
}

# Plot groups: each entry is one plot per (variable, cut); each curve is
# (numerator hist prefix, numerator suffix, legend label, color). All curves
# share the vtxreco_<var>_den denominator.
PLOTS = {
    'lptvtx': {
        'ylabel': 'lptvtx Reco Efficiency',
        'curves': [
            ('vtxreco_lptvtx', 'num',          'Good lptvtx',              'C0'),
            ('vtxreco_lptvtx', 'num_genmatch', 'Good lptvtx (gen-matched)', 'C1'),
        ],
    },
    'vtx': {
        'ylabel': 'vtx Reco Efficiency',
        'curves': [
            ('vtxreco_vtx', 'num',          'Good vtx',              'C4'),
            ('vtxreco_vtx', 'num_genmatch', 'Good vtx (gen-matched)', 'C5'),
        ],
    },
    'merged': {
        'ylabel': 'Merged Reco Efficiency',
        'curves': [
            ('vtxreco_merged', 'num',          'Good merged',              'C2'),
            ('vtxreco_merged', 'num_genmatch', 'Good merged (gen-matched)', 'C3'),
        ],
    },
}


def samp_slice(h, s, cut=None):
    """
    Slice hist `h` to sample `s` (and `cut`, if given). The samp axis only grows
    when a sample is filled, so a sample with no entries in `h` (e.g. no
    gen-matched vtx at all) is missing -- return an empty hist for it instead.
    """
    idx = {} if cut is None else {"cut": cut}
    if s in list(h.axes["samp"]):
        return h[{**idx, "samp": s}]
    return h[{**idx, "samp": sum}] * 0


def n_eff(h):
    """
    Per-bin effective raw MC event count: N_eff = sum(w)^2 / sum(w^2), i.e.
    vals^2/variances -- see vtxrecoeff_vs_gendr_plots.py.
    """
    vals, var = h.values(), h.variances()
    return np.where(var > 0, vals**2 / np.where(var > 0, var, 1), 0)


def combine_neff_ratio(h_num, h_den, samples, cut):
    """
    Combine all signal samples into one efficiency curve at `cut` by summing
    per-bin N_eff of numerator and denominator over samples (so the combination
    isn't dominated by the highest-xsec mass point). Returns a 1D Hist holding
    the efficiency with a binomial variance based on the combined N_eff (NaN
    in bins with an empty denominator).
    """
    axis = copy.deepcopy(h_den[{"cut": cut, "samp": samples[0]}].axes[0])
    n_bins = len(axis)

    num_tot = np.zeros(n_bins)
    den_tot = np.zeros(n_bins)
    for s in samples:
        num_tot += n_eff(samp_slice(h_num, s, cut))
        den_tot += n_eff(samp_slice(h_den, s, cut))

    with np.errstate(invalid='ignore', divide='ignore'):
        # empty-denominator bins -> NaN so they're skipped when plotted
        ratio = np.where(den_tot > 0, num_tot / den_tot, np.nan)
        ratio_var = np.where(den_tot > 0, ratio * (1 - ratio) / den_tot, np.nan)

    h_ratio = Hist(axis, storage=Weight())
    h_ratio.view(flow=False)[...] = np.stack([ratio, ratio_var], axis=-1)
    return h_ratio


def combine_weighted_ratio(h_num, h_den, samples, cut):
    """
    Combine all signal samples into one efficiency curve at `cut` using the
    event weights directly: eff = sum(w_num) / sum(w_den) per bin, summed over
    samples. The numerator is a subset of the denominator, so the variance is
    [(1 - 2 eff) sum(w_num^2) + eff^2 sum(w_den^2)] / sum(w_den)^2. Returns a
    1D Hist (NaN in bins with an empty denominator).
    """
    axis = copy.deepcopy(h_den[{"cut": cut, "samp": samples[0]}].axes[0])
    n_bins = len(axis)

    num_w, num_w2 = np.zeros(n_bins), np.zeros(n_bins)
    den_w, den_w2 = np.zeros(n_bins), np.zeros(n_bins)
    for s in samples:
        hn = samp_slice(h_num, s, cut)
        hd = samp_slice(h_den, s, cut)
        num_w += hn.values(); num_w2 += hn.variances()
        den_w += hd.values(); den_w2 += hd.variances()

    with np.errstate(invalid='ignore', divide='ignore'):
        # empty-denominator bins -> NaN so they're skipped when plotted
        ratio = np.where(den_w > 0, num_w / den_w, np.nan)
        ratio_var = np.where(den_w > 0, ((1 - 2*ratio)*num_w2 + ratio**2*den_w2) / den_w**2, np.nan)
        ratio_var = np.clip(ratio_var, 0, None)

    h_ratio = Hist(axis, storage=Weight())
    h_ratio.view(flow=False)[...] = np.stack([ratio, ratio_var], axis=-1)
    return h_ratio


def plot_eff(sig_histo, var, cut, group):
    xlabel, logx = VARIABLES[var]
    cfg = PLOTS[group]
    samples = list(sig_histo['cutflow'].keys())
    h_den = sig_histo[f'vtxreco_{var}_den']

    fig, ax = plt.subplots(figsize=(10, 8))
    for prefix, suffix, label, color in cfg['curves']:
        h_eff = combine_neff_ratio(sig_histo[f'{prefix}_{var}_{suffix}'], h_den, samples, cut)
        ax.errorbar(h_eff.axes[0].centers, h_eff.values(), yerr=np.sqrt(h_eff.variances()),
                    fmt='-o', markersize=4, color=color, capsize=3, label=label)

    if logx:
        ax.set_xscale('log')
    ax.set_ylim(0, 1.05)
    ax.grid()
    ax.legend(fontsize=16)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(cfg['ylabel'], fontsize=20)
    ax.set_title(f'{cfg["ylabel"]} [{CUTS[cut].capitalize()}, All Signal Samples]', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)

    plt.tight_layout()
    outpath = f'{plotdir}/vtxreco_{group}_vs_{var}_{CUTS[cut]}_{plottag}.png'
    plt.savefig(outpath)
    print(f'Saved: {outpath}')
    plt.close(fig)


# 2D gen Lxy (x) vs gen ee pT (y) efficiency maps: one plot per numerator
# (prefix, suffix) -> (plot tag, title)
VAR_2D = 'gen_lxy_vs_pt'
PLOTS_2D = {
    ('vtxreco_lptvtx', 'num')          : ('lptvtx',          'lptvtx Reco Efficiency'),
    ('vtxreco_lptvtx', 'num_genmatch') : ('lptvtx_genmatch', 'Gen-matched lptvtx Reco Efficiency'),
    ('vtxreco_vtx',    'num')          : ('vtx',             'vtx Reco Efficiency'),
    ('vtxreco_vtx',    'num_genmatch') : ('vtx_genmatch',    'Gen-matched vtx Reco Efficiency'),
    ('vtxreco_merged', 'num')          : ('merged',          'Merged Reco Efficiency'),
    ('vtxreco_merged', 'num_genmatch') : ('merged_genmatch', 'Gen-matched Merged Reco Efficiency'),
}


def plot_eff_2d(sig_histo, cut, prefix, suffix):
    """
    Weighted reco efficiency in bins of gen Lxy x gen ee pT, all signal samples
    summed. Bins are drawn evenly sized regardless of their edge values (the
    real edges are printed as tick labels), with the efficiency printed in each
    bin.
    """
    tag, title = PLOTS_2D[(prefix, suffix)]
    samples = list(sig_histo['cutflow'].keys())
    h_num = sig_histo[f'{prefix}_{VAR_2D}_{suffix}'][{"cut": cut}]
    h_den = sig_histo[f'vtxreco_{VAR_2D}_den'][{"cut": cut}]

    num = sum(n_eff(samp_slice(h_num, s)) for s in samples)
    den = sum(n_eff(samp_slice(h_den, s)) for s in samples)
    with np.errstate(invalid='ignore', divide='ignore'):
        eff = np.where(den > 0, num / den, np.nan)   # shape (n_lxy, n_pt)

    lxy_axis, pt_axis = h_den.axes['lxy'], h_den.axes['pt']
    n_lxy, n_pt = len(lxy_axis), len(pt_axis)

    fig, ax = plt.subplots(figsize=(10, 8))
    mesh = ax.pcolormesh(np.arange(n_pt + 1), np.arange(n_lxy + 1), eff,
                         cmap='viridis', vmin=0, vmax=1)
    fig.colorbar(mesh, ax=ax).set_label('Reco Efficiency', fontsize=18)

    for i in range(n_lxy):
        for j in range(n_pt):
            if np.isnan(eff[i, j]):
                continue
            ax.text(j + 0.5, i + 0.5, f'{eff[i, j]:.3f}', ha='center', va='center', fontsize=14,
                    color='black' if eff[i, j] > 0.5 else 'white')

    ax.set_xticks(np.arange(n_pt + 1))
    ax.set_xticklabels([f'{e:g}' for e in pt_axis.edges])
    ax.set_yticks(np.arange(n_lxy + 1))
    ax.set_yticklabels([f'{e:g}' for e in lxy_axis.edges])
    ax.set_xlabel(r'Gen $p_{T}(e^+e^-)$ [GeV]', fontsize=20)
    ax.set_ylabel(r'Gen $L_{xy}$ [cm]', fontsize=20)
    ax.set_title(f'{title} [{CUTS[cut].capitalize()}, All Signal Samples]', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)

    plt.tight_layout()
    outpath = f'{plotdir}/vtxreco2d_{tag}_{VAR_2D}_{CUTS[cut]}_{plottag}.png'
    plt.savefig(outpath)
    print(f'Saved: {outpath}')
    plt.close(fig)


# ---- Load histograms and make plots ----------------------------------------

s_hists = util.load(saved_signal_hists)[0]

for cut in CUTS:
    for var in VARIABLES:
        for group in PLOTS:
            plot_eff(s_hists, var, cut, group)
    for prefix, suffix in PLOTS_2D:
        plot_eff_2d(s_hists, cut, prefix, suffix)
