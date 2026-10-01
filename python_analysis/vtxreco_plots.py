# imports
import os
import re
import copy
import numpy as np
import matplotlib.pyplot as plt

from hist import Hist
from hist.storage import Weight

import coffea.util as util

outdir = 'workarea'
plotdir = './plots/vtxreco'
os.makedirs(plotdir, exist_ok=True)

sig_vers = 'Preapp2018'
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


def cut_slice(h, cut):
    """
    Slice hist `h` to `cut` on its cut axis. The cut (and samp) axes only grow
    when filled -- numerators that are never filled at all (e.g. lptvtx/merged on
    eras without AllLptElectron, like 2018: see has_lpt in
    configs/histo_configs/vtxreco.py) have an empty cut axis -- return a hist of
    zeros with the same samp/var binning instead of KeyError-ing.
    """
    if cut in list(h.axes["cut"]):
        return h[{"cut": cut}]
    return Hist(*[ax for ax in h.axes if ax.name != "cut"], storage=Weight())


def samp_slice(h, s, cut=None):
    """
    Slice hist `h` to sample `s` (and `cut`, if given). The samp axis only grows
    when a sample is filled, so a sample with no entries in `h` (e.g. no
    gen-matched vtx at all) is missing -- return an empty hist for it instead.
    """
    if cut is not None:
        h = cut_slice(h, cut)
    if s in list(h.axes["samp"]):
        return h[{"samp": s}]
    return h[{"samp": sum}] * 0


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


# 2D gen Lxy (x) vs gen ee pT (y) efficiency maps: one plot per numerator per gen ee dR bin
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


def sample_m1(samp):
    """
    Parse m1 = Mchi - dMchi/2 (the lighter mass eigenstate) from a signal
    sample name like 'sig_2018_Mchi-105p0_dMchi-10p0_ctau-1' (decimal points
    are replaced with 'p', per Analyzer.loadFiles in analysisTools.py).
    """
    mchi, dmchi = re.search(r'Mchi-([\d]+p[\d]+)_dMchi-([\d]+p[\d]+)', samp).groups()
    return float(mchi.replace('p', '.')) - float(dmchi.replace('p', '.')) / 2


def plot_eff_2d(sig_histo, cut, prefix, suffix, idr, run2range=False):
    """
    Weighted reco efficiency in bins of gen Lxy x gen ee pT, for gen ee dR bin
    `idr`, all signal samples summed. Bins are drawn evenly sized regardless of
    their edge values (the real edges are printed as tick labels), with the
    efficiency printed in each bin.

    If `idr` is None, all gen ee dR bins are summed together (no dR cut).

    If `run2range`, only signal samples with m1 > 5 GeV (the mass range
    covered by the Run 2 analysis) are combined, and the output is tagged
    'run2range'. `run2range` is only meaningful with `idr=None`.
    """
    tag, title = PLOTS_2D[(prefix, suffix)]
    samples = list(sig_histo['cutflow'].keys())
    if run2range:
        samples = [s for s in samples if sample_m1(s) >= 5]
    h_num = cut_slice(sig_histo[f'{prefix}_{VAR_2D}_{suffix}'], cut)
    h_den = cut_slice(sig_histo[f'vtxreco_{VAR_2D}_den'], cut)

    num_full = sum(n_eff(samp_slice(h_num, s)) for s in samples)  # shape (n_lxy, n_pt, n_dr)
    den_full = sum(n_eff(samp_slice(h_den, s)) for s in samples)
    if idr is None:
        num, den = num_full.sum(axis=-1), den_full.sum(axis=-1)
    else:
        num, den = num_full[:, :, idr], den_full[:, :, idr]
    with np.errstate(invalid='ignore', divide='ignore'):
        eff = np.where(den > 0, num / den, np.nan)   # shape (n_lxy, n_pt)

    lxy_axis, pt_axis = h_den.axes['lxy'], h_den.axes['pt']
    n_lxy, n_pt = len(lxy_axis), len(pt_axis)
    if idr is not None:
        dr_lo, dr_hi = h_den.axes['dr'][idr]
        dr_subtitle = '\n' + rf'${dr_lo:g} <$ Gen $\Delta R(e^+e^-) < {dr_hi:g}$'
        dr_tag = f'dr{dr_lo:g}to{dr_hi:g}'
    else:
        dr_subtitle = ''
        dr_tag = 'alldr'

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
    sample_tag = 'Run 2 Mass Range' if run2range else 'All Signal Samples'
    ax.set_title(f'{title} [{CUTS[cut].capitalize()}, {sample_tag}]' + dr_subtitle, fontsize=16)
    ax.tick_params(axis='both', labelsize=14)

    plt.tight_layout()
    outtag = f'{plottag}_run2range' if run2range else plottag
    outpath = f'{plotdir}/vtxreco2d_{tag}_{VAR_2D}_{dr_tag}_{CUTS[cut]}_{outtag}.png'
    plt.savefig(outpath)
    print(f'Saved: {outpath}')
    plt.close(fig)


# ---- Load histograms and make plots ----------------------------------------

s_hists = util.load(saved_signal_hists)[0]

for cut in CUTS:
    for var in VARIABLES:
        for group in PLOTS:
            plot_eff(s_hists, var, cut, group)
    n_dr = len(s_hists[f'vtxreco_{VAR_2D}_den'].axes['dr'])
    for prefix, suffix in PLOTS_2D:
        for idr in range(n_dr):
            plot_eff_2d(s_hists, cut, prefix, suffix, idr)
        plot_eff_2d(s_hists, cut, prefix, suffix, None)
        plot_eff_2d(s_hists, cut, prefix, suffix, None, run2range=True)
