# imports
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak

import sys
import os

# Make the script runnable regardless of the caller's current working
# directory by anchoring paths to the repo root (two levels up from this file).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from analysisTools.analysisTools import Analyzer
from analysisTools.analysisTools import loadSchema
import analysisTools.analysisTools as tools
import analysisTools.analysisSubroutines as routines
import importlib
import coffea.util as util
import analysisTools.utils as utils
import analysisTools.plotTools as ptools
import time
import json
import glob
from hist import Hist, loc as hloc
from hist.axis import Variable
from configs.histo_configs.histobins import ele_pt
from hist.storage import Weight
import copy
import mplhep as hep

#cuts_config = "configs/selections/minimal_cuts.py"
#hists_config = "configs/hists/genstudy.py"
#sample_config = "configs/samples/signal_2024_Apr2026_aEM.json"
outdir = os.path.join(REPO_ROOT, 'workarea')
plotdir = os.path.join(REPO_ROOT, 'plots', 'elerecoeff')
os.makedirs(plotdir, exist_ok=True)
saved_signal_hists = f"{outdir}/hists_sigJul2026noID_anmatchvtx-sel_elerecoeff.coffea"

title = 'Electron Reco'
seltag = 'nocuts'
# Plot settings
plot_dict = {
    'variable': ['ele_reco_lpt_pt_lxy', 'ele_reco_ged_pt_lxy', 'ele_reco_none_pt_lxy'], 
    'year': 2024,
    'cut': 'cut1',
}
_pt_edges  = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10, 12.5, 15, 17.5, 20, 22.5, 25, 30, 40, 50]
_lxy_edges = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 7.5, 10, 12.5, 15, 20, 25, 30, 40]
_eta_edges = list(np.linspace(-3, 3, 61))
_vz_edges  = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2, 3, 4, 5, 7.5, 10, 12.5, 15, 20, 25, 30, 40, 50]

proj_configs = [
    {
        'variables':        ['ele_reco_lpt_pt_lxy',   'ele_reco_ged_pt_lxy',   'ele_reco_none_pt_lxy'],
        'alllpt_variables': ['ele_reco_alllpt_pt_lxy','ele_reco_noalllpt_pt_lxy'],
        'project': {"samp": sum, "lxy": sum}, 'rebin_edges': _pt_edges,
        'plottag': f'{seltag}_ele-reco_only-pt',  'var': r'$p_T$ [GeV]',
    },
    {
        'variables':        ['ele_reco_lpt_pt_lxy',   'ele_reco_ged_pt_lxy',   'ele_reco_none_pt_lxy'],
        'alllpt_variables': ['ele_reco_alllpt_pt_lxy','ele_reco_noalllpt_pt_lxy'],
        'project': {"samp": sum, "pt":  sum}, 'rebin_edges': _lxy_edges,
        'plottag': f'{seltag}_ele-reco_only-xy', 'var': r'$L_{xy}$ [cm]', 'doLogx': True,
    },
    {
        'variables':        ['ele_reco_lpt_eta',   'ele_reco_ged_eta',   'ele_reco_none_eta'],
        'alllpt_variables': ['ele_reco_alllpt_eta','ele_reco_noalllpt_eta'],
        'project': {"samp": sum}, 'rebin_edges': _eta_edges,
        'plottag': f'{seltag}_ele-reco_only-eta', 'var': r'$\eta$',
    },
    {
        'variables':        ['ele_reco_lpt_lz',   'ele_reco_ged_lz',   'ele_reco_none_lz'],
        'alllpt_variables': ['ele_reco_alllpt_lz','ele_reco_noalllpt_lz'],
        'project': {"samp": sum}, 'rebin_edges': _vz_edges,
        'plottag': f'{seltag}_ele-reco_only-vz',  'var': r'$L_z$ [cm]', 'doLogx': True,
    },
]

# Signal
s_hists = util.load(saved_signal_hists)[0]
s_pts = utils.get_signal_point_dict(s_hists)
s_histnames = utils.get_signal_list_of_histograms(s_hists)
s_cutsidx = utils.get_signal_list_of_cuts(s_hists, get_cut_idx = True)
s_cutsname = utils.get_signal_list_of_cuts(s_hists, get_cut_idx = False)

df = utils.get_signal_cutflow_dict(s_hists, 'cutflow')
size = (16, 12)
fig, ax = plt.subplots(figsize=size)

style_dict = {
    'fig': fig, 'ax': ax,
    'rebin': 1j, 'xlim': None,   # if None, the default will show up; otherwise give as a list, i.e. [0, 10]
    'doLogy': False, 'doLogx': False, 'doDensity': False,
    'xlabel': None,   # if None, the default will show up; otherwise give as a string, i.e. 'Electron dxy'
    'ylabel': None,   # if None, the default will show up; otherwise give as a string, i.e. 'Efficiency'
    'zlabel': None,   # if None, the default will show up; otherwise give as a string, i.e. 'Efficiency'
    'label': None,    # if None, the default will show up; otherwise give as a string, i.e. 'Highest ctau signal samples'
    'flow': None,     # overflow
    'doSave': False,
    'ls': ['-', '--', ':'],
}

def rebin_variable(h, new_edges):
    """Rebin a 1D Hist to arbitrary bin edges by summing values and variances."""
    old_edges = h.axes[0].edges
    new_edges = np.array(new_edges)
    vals, vars_ = h.values(), h.variances()
    n_new = len(new_edges) - 1
    new_vals = np.zeros(n_new)
    new_vars = np.zeros(n_new)
    for i in range(n_new):
        lo = np.searchsorted(old_edges, new_edges[i],     side='left')
        hi = np.searchsorted(old_edges, new_edges[i + 1], side='left')
        new_vals[i] = vals[lo:hi].sum()
        new_vars[i] = vars_[lo:hi].sum()
    ax = Variable(new_edges, name=h.axes[0].name, label=h.axes[0].label)
    h_new = Hist(ax, storage=Weight())
    h_new.view(flow=False)[...] = np.stack([new_vals, new_vars], axis=-1)
    return h_new

def ratio_hist(h_num, h_den):
    """
    Returns a new Hist object representing h_num / (h_den1 + h_den2)
    with correct variance propagation.
    All histograms must have compatible axes.
    """
    num = h_num.values()
    den = h_den.values()
    num_var = h_num.variances()
    den_var = h_den.variances()

    with np.errstate(invalid='ignore', divide='ignore'):
        ratio = np.where(den > 0, num / den, 0)
        ratio_var = np.where(
            den > 0,
            ratio**2 * (num_var / np.where(num > 0, num**2, 1)
                      + den_var / np.where(den > 0, den**2, 1)),
            0
        )

    # create new hist with same axis and Weight storage (to hold variances)
    h_ratio = Hist(copy.deepcopy(h_num.axes[0]), storage=Weight())
    h_ratio.view(flow=False)[...] = np.stack([ratio, ratio_var], axis=-1)

    return h_ratio

plt.close(fig)

def pick_representative_samples(pts_df, max_samples=6):
    """Select evenly-spaced samples across the sorted (mchi, ctau) grid."""
    df = pts_df.sort_values(['mchi', 'delta', 'ctau'])
    n = len(df)
    if n <= max_samples:
        return df.index.tolist()
    indices = np.round(np.linspace(0, n - 1, max_samples)).astype(int)
    return df.index[indices].tolist()

def pick_pair_varying_variable(pts_df, vary_col, fix_cols):
    """Pick 2 samples that differ maximally in vary_col within a fixed group of fix_cols."""
    groups = pts_df.groupby([pts_df[c] for c in fix_cols])
    best_pair, best_spread = None, -1
    for _, grp in groups:
        if len(grp) < 2:
            continue
        sorted_grp = grp.sort_values(vary_col)
        spread = sorted_grp[vary_col].iloc[-1] - sorted_grp[vary_col].iloc[0]
        if spread > best_spread:
            best_spread = spread
            best_pair = [sorted_grp.index[0], sorted_grp.index[-1]]
    return best_pair if best_pair is not None else pts_df.index[:2].tolist()

vary_configs = [
    {'vary': 'm1',    'fix': ['delta', 'ctau'], 'tag': 'varyM1',    'label': r'vary $M_1$'},
    {'vary': 'delta', 'fix': ['m1',    'ctau'], 'tag': 'varyDelta', 'label': r'vary $\Delta$'},
    {'vary': 'ctau',  'fix': ['m1',   'delta'], 'tag': 'varyCTau',  'label': r'vary $c\tau$'},
]
sample_pairs = {vc['tag']: pick_pair_varying_variable(s_pts, vc['vary'], vc['fix']) for vc in vary_configs}
representative_samples = pick_representative_samples(s_pts, max_samples=6)

def sample_label(row):
    return rf"$M_1$={row['m1']:.3g}, $\Delta$={row['delta']:.3g}, $c\tau$={row['ctau']:.0f} mm"

for cfg in proj_configs:
    sumhists = [
        s_hists[v][{"cut": plot_dict['cut'], **cfg['project']}]
        for v in cfg['variables']
    ]

    fig, ax = plt.subplots(figsize=size)
    hep.cms.label('Private Work', data=True, year=plot_dict['year'], com='13.6')
    hep.histplot(sumhists, yerr=[np.sqrt(h.variances()) for h in sumhists], density=style_dict['doDensity'], ax=ax,
                 histtype='step', flow=style_dict['flow'], label=['Lpt', 'GED', 'None'])
    binwidth = sumhists[0].axes.widths[0][0]
    if style_dict['doDensity']:
        ax.set_ylabel(f'A.U./{binwidth:.3f}')
    else:
        ax.set_ylabel(f'Events/{binwidth:.3f}')
    ax.set_yscale('log')
    if cfg.get('doLogx'):
        ax.set_xscale('log')
    plt.title(rf"{title}: {cfg['var']}")
    plt.legend()
    plt.savefig(f"{plotdir}/hist_{cfg['plottag']}_hist_allpts.png")
    plt.close(fig)

    alllpt_hists = [
        s_hists[v][{"cut": plot_dict['cut'], **cfg['project']}]
        for v in cfg['alllpt_variables']
    ]

    sumhists_rb    = [rebin_variable(h, cfg['rebin_edges']) for h in sumhists]
    alllpt_hists_rb = [rebin_variable(h, cfg['rebin_edges']) for h in alllpt_hists]

    h_lpt_eff    = ratio_hist(sumhists_rb[0], sumhists_rb[0] + sumhists_rb[1] + sumhists_rb[2])
    h_ged_eff    = ratio_hist(sumhists_rb[1], sumhists_rb[0] + sumhists_rb[1] + sumhists_rb[2])
    h_both_eff   = ratio_hist(sumhists_rb[0] + sumhists_rb[1], sumhists_rb[0] + sumhists_rb[1] + sumhists_rb[2])
    h_alllpt_eff = ratio_hist(alllpt_hists_rb[0], alllpt_hists_rb[0] + alllpt_hists_rb[1])
    effhists = [h_lpt_eff, h_ged_eff, h_both_eff, h_alllpt_eff]

    fig, ax = plt.subplots(figsize=size)
    hep.cms.label('Private Work', data=True, year=plot_dict['year'], com='13.6')
    hep.histplot(effhists, yerr=[np.sqrt(h.variances()) for h in effhists], density=style_dict['doDensity'], ax=ax,
                 histtype='step', flow=style_dict['flow'], label=['Lpt', 'GED', 'Both', 'AllLowPt'])
    binwidth = sumhists[0].axes.widths[0][0]
    if style_dict['doDensity']:
        ax.set_ylabel(f'A.U./{binwidth:.3f}')
    else:
        ax.set_ylabel(f'Events/{binwidth:.3f}')
    ax.set_ylim(-0.05, 1.05)
    if cfg.get('doLogx'):
        ax.set_xscale('log')
    plt.title(rf"{title} Efficiency by {cfg['var']}")
    plt.legend()
    plt.savefig(f"{plotdir}/hist_{cfg['plottag']}_ratio_allpts.png")
    plt.close(fig)

    # Per-sample efficiency plots: 3 plots, each pair controls for 2 of (m1, delta, ctau)
    for vc in vary_configs:
        pair = sample_pairs[vc['tag']]
        fig, ax = plt.subplots(figsize=size)
        hep.cms.label('Private Work', data=True, year=plot_dict['year'], com='13.6')
        colors = [f'C{i}' for i in range(len(pair))]
        for sname, color in zip(pair, colors):
            samp_slice = {k: (sname if k == "samp" else v) for k, v in cfg['project'].items()}
            ps_hists = [
                s_hists[v][{"cut": plot_dict['cut'], **samp_slice}]
                for v in cfg['variables']
            ]
            ps_alllpt_hists = [
                s_hists[v][{"cut": plot_dict['cut'], **samp_slice}]
                for v in cfg['alllpt_variables']
            ]
            ps_hists        = [rebin_variable(h, cfg['rebin_edges']) for h in ps_hists]
            ps_alllpt_hists = [rebin_variable(h, cfg['rebin_edges']) for h in ps_alllpt_hists]
            h_lpt, h_ged, h_none = ps_hists
            den          = h_lpt + h_ged + h_none
            h_lpt_eff    = ratio_hist(h_lpt, den)
            h_ged_eff    = ratio_hist(h_ged, den)
            h_alllpt_eff = ratio_hist(ps_alllpt_hists[0], ps_alllpt_hists[0] + ps_alllpt_hists[1])
            row = s_pts.loc[sname]
            slabel = sample_label(row)
            hep.histplot(h_lpt_eff,    yerr=np.sqrt(h_lpt_eff.variances()),    ax=ax,
                         histtype='step', linestyle='-',  color=color, label=f"{slabel} — Lpt")
            hep.histplot(h_ged_eff,    yerr=np.sqrt(h_ged_eff.variances()),    ax=ax,
                         histtype='step', linestyle='--', color=color, label=f"{slabel} — GED")
            hep.histplot(h_alllpt_eff, yerr=np.sqrt(h_alllpt_eff.variances()), ax=ax,
                         histtype='step', linestyle=':',  color=color, label=f"{slabel} — AllLowPt")
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel('Reco Efficiency')
        ax.set_xlabel(cfg['var'])
        if cfg.get('doLogx'):
            ax.set_xscale('log')
        plt.title(rf"{title} Efficiency by {cfg['var']} — {vc['label']}")
        plt.legend(fontsize=11)
        plt.savefig(f"{plotdir}/hist_{cfg['plottag']}_ratio_persamp_{vc['tag']}.png")
        plt.close(fig)

    # Multi-sample comparison: one efficiency type per plot, all samples as solid colored lines
    eff_types = [
        {'tag': 'lpt',    'label': 'LowPt'},
        {'tag': 'ged',    'label': 'GED'},
        {'tag': 'both',   'label': 'Both'},
        {'tag': 'alllpt', 'label': 'AllLowPt'},
    ]
    rep_colors = plt.cm.tab10(np.linspace(0, 0.9, len(representative_samples)))
    for et in eff_types:
        fig, ax = plt.subplots(figsize=size)
        hep.cms.label('Private Work', data=True, year=plot_dict['year'], com='13.6')
        for sname, color in zip(representative_samples, rep_colors):
            samp_slice = {k: (sname if k == "samp" else v) for k, v in cfg['project'].items()}
            ps_hists = [
                s_hists[v][{"cut": plot_dict['cut'], **samp_slice}]
                for v in cfg['variables']
            ]
            ps_alllpt_hists = [
                s_hists[v][{"cut": plot_dict['cut'], **samp_slice}]
                for v in cfg['alllpt_variables']
            ]
            ps_hists        = [rebin_variable(h, cfg['rebin_edges']) for h in ps_hists]
            ps_alllpt_hists = [rebin_variable(h, cfg['rebin_edges']) for h in ps_alllpt_hists]
            h_lpt, h_ged, h_none = ps_hists
            den = h_lpt + h_ged + h_none
            if et['tag'] == 'lpt':
                h_eff = ratio_hist(h_lpt, den)
            elif et['tag'] == 'ged':
                h_eff = ratio_hist(h_ged, den)
            elif et['tag'] == 'both':
                h_eff = ratio_hist(h_lpt + h_ged, den)
            else:
                h_eff = ratio_hist(ps_alllpt_hists[0], ps_alllpt_hists[0] + ps_alllpt_hists[1])
            row = s_pts.loc[sname]
            hep.histplot(h_eff, yerr=np.sqrt(h_eff.variances()), ax=ax,
                         histtype='step', linestyle='-', color=color, label=sample_label(row))
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel('Reco Efficiency')
        ax.set_xlabel(cfg['var'])
        if cfg.get('doLogx'):
            ax.set_xscale('log')
        plt.title(rf"{title} {et['label']} Efficiency by {cfg['var']}")
        plt.legend(fontsize=10)
        plt.savefig(f"{plotdir}/hist_{cfg['plottag']}_ratio_multisamp_{et['tag']}.png")
        plt.close(fig)

# ── Comparison plots ──────────────────────────────────────────────────────────
cmap = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]
_comp_pt_edges  = list(range(0, 31))   # 30 × 1-GeV bins
_comp_lxy_edges = list(range(0, 21))   # 20 × 1-cm bins
_comp_pt_vars     = ['ele_reco_lpt_pt_lxy', 'ele_reco_ged_pt_lxy', 'ele_reco_none_pt_lxy']
_comp_alllpt_vars = ['ele_reco_alllpt_pt_lxy', 'ele_reco_noalllpt_pt_lxy']
_comp_eff_labels  = ['LowPt', 'GED', 'Both', 'AllLowPt']
_comp_colors      = cmap[0:4] #['C0', 'C1', 'C2', 'C3']
_cut = plot_dict['cut']

def _comp_effhists(hists_rb, alllpt_rb):
    h_lpt, h_ged, h_none = hists_rb
    den = h_lpt + h_ged + h_none
    return [
        ratio_hist(h_lpt,         den),
        ratio_hist(h_ged,         den),
        ratio_hist(h_lpt + h_ged, den),
        ratio_hist(alllpt_rb[0],  alllpt_rb[0] + alllpt_rb[1]),
    ]

def _comp_plot(effhists, xlabel, title_str, savepath):
    fig, ax = plt.subplots(figsize=size)
    hep.cms.label('Private Work', data=True, year=plot_dict['year'], com='13.6')
    for h, label, color in zip(effhists, _comp_eff_labels, _comp_colors):
        hep.histplot(h, yerr=np.sqrt(h.variances()), ax=ax,
                     histtype='step', linestyle='-', color=color, label=label)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel('Reco Efficiency')
    ax.set_xlabel(xlabel)
    plt.title(title_str)
    plt.legend(fontsize=11)
    plt.savefig(savepath)
    plt.close(fig)

# Plot 1: efficiency vs pT, all Lxy
_h  = [s_hists[v][{"cut": _cut, "samp": sum, "lxy": sum}] for v in _comp_pt_vars]
_ha = [s_hists[v][{"cut": _cut, "samp": sum, "lxy": sum}] for v in _comp_alllpt_vars]
_comp_plot(_comp_effhists([rebin_variable(h, _comp_pt_edges) for h in _h],
                          [rebin_variable(h, _comp_pt_edges) for h in _ha]),
           r'$p_T$ [GeV]', rf'{title} Efficiency vs $p_T$',
           f'plots/hist_{seltag}_comp_eff_pt.png')

# Plot 1b: efficiency vs pT, all Lxy, no rebinning
_h  = [s_hists[v][{"cut": _cut, "samp": sum, "lxy": sum}] for v in _comp_pt_vars]
_ha = [s_hists[v][{"cut": _cut, "samp": sum, "lxy": sum}] for v in _comp_alllpt_vars]
_comp_plot(_comp_effhists(_h, _ha),
           r'$p_T$ [GeV]', rf'{title} Efficiency vs $p_T$ (no rebin)',
           f'plots/hist_{seltag}_comp_eff_pt_norebin.png')

# Plot 2: efficiency vs Lxy, all pT
_h  = [s_hists[v][{"cut": _cut, "samp": sum, "pt": sum}] for v in _comp_pt_vars]
_ha = [s_hists[v][{"cut": _cut, "samp": sum, "pt": sum}] for v in _comp_alllpt_vars]
_comp_plot(_comp_effhists([rebin_variable(h, _comp_lxy_edges) for h in _h],
                          [rebin_variable(h, _comp_lxy_edges) for h in _ha]),
           r'$L_{xy}$ [cm]', rf'{title} Efficiency vs $L_{{xy}}$',
           f'plots/hist_{seltag}_comp_eff_lxy.png')

# Plots 3-7: efficiency vs pT in Lxy slices
_lxy_slices = [
    {'lo': None, 'hi': 1,    'tag': 'lxy0to1',   'label': r'$L_{xy} < 1$ cm'},
    {'lo': 1,    'hi': 5,    'tag': 'lxy1to5',   'label': r'$1 < L_{xy} < 5$ cm'},
    {'lo': 5,    'hi': 10,   'tag': 'lxy5to10',  'label': r'$5 < L_{xy} < 10$ cm'},
    {'lo': 10,   'hi': 15,   'tag': 'lxy10to15', 'label': r'$10 < L_{xy} < 15$ cm'},
    {'lo': 15,   'hi': None, 'tag': 'lxygt15',   'label': r'$L_{xy} > 15$ cm'},
]
for slc in _lxy_slices:
    lo, hi = slc['lo'], slc['hi']
    if lo is None:
        lxy_sl = slice(None, hloc(hi), sum)
    elif hi is None:
        lxy_sl = slice(hloc(lo), None, sum)
    else:
        lxy_sl = slice(hloc(lo), hloc(hi), sum)
    _h  = [s_hists[v][{"cut": _cut, "samp": sum, "lxy": lxy_sl}] for v in _comp_pt_vars]
    _ha = [s_hists[v][{"cut": _cut, "samp": sum, "lxy": lxy_sl}] for v in _comp_alllpt_vars]
    _comp_plot(_comp_effhists([rebin_variable(h, _comp_pt_edges) for h in _h],
                              [rebin_variable(h, _comp_pt_edges) for h in _ha]),
               r'$p_T$ [GeV]',
               rf'{title} Efficiency vs $p_T$, {slc["label"]}',
               f'plots/hist_{seltag}_comp_eff_pt_{slc["tag"]}.png')

# Plots 8-11: efficiency vs Lxy in pT slices
_pt_slices = [
    {'lo': None, 'hi': 5,    'tag': 'pt0to5',   'label': r'$p_T < 5$ cm'},
    {'lo': 5,    'hi': 10,   'tag': 'pt5to10',  'label': r'$5 < p_T < 10$ cm'},
    {'lo': 10,   'hi': 20,   'tag': 'pt10to20', 'label': r'$10 < p_T < 20$ cm'},
    {'lo': 20,   'hi': None, 'tag': 'ptgt20',   'label': r'$p_T > 20$ cm'},
]
for slc in _pt_slices:
    lo, hi = slc['lo'], slc['hi']
    if lo is None:
        pt_sl = slice(None, hloc(hi), sum)
    elif hi is None:
        pt_sl = slice(hloc(lo), None, sum)
    else:
        pt_sl = slice(hloc(lo), hloc(hi), sum)
    axlab = r'$L_{xy}$'
    _h  = [s_hists[v][{"cut": _cut, "samp": sum, "pt": pt_sl}] for v in _comp_pt_vars]
    _ha = [s_hists[v][{"cut": _cut, "samp": sum, "pt": pt_sl}] for v in _comp_alllpt_vars]
    _comp_plot(_comp_effhists([rebin_variable(h, _comp_lxy_edges) for h in _h],
                              [rebin_variable(h, _comp_lxy_edges) for h in _ha]),
               r'$L_{xy}$ [cm]',
               rf'{title} Efficiency vs {axlab}, {slc["label"]}',
               f'plots/hist_{seltag}_comp_eff_lxy_{slc["tag"]}.png')
