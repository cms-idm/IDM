# imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import awkward as ak

import sys
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
import os
import glob
from hist import Hist
from hist.axis import Variable
from hist.storage import Weight
import copy
import mplhep as hep

outdir = 'workarea'
saved_signal_hists = f"{outdir}/hists_sigMay2026_an-sel_elerecoeff.coffea"

title = 'Electron Reco'
plottag = 'prevtx_ele-reco_pt-lxy'

_pt_edges  = [0, 1, 2, 5, 10, 20, 50]
_lxy_edges = [0, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 40]

histvars        = ['ele_reco_lpt_pt_lxy', 'ele_reco_ged_pt_lxy', 'ele_reco_none_pt_lxy']
alllpt_histvars = ['ele_reco_alllpt_pt_lxy', 'ele_reco_noalllpt_pt_lxy']

plot_dict = {
    'year': 2024,
    'cut': 'cut8',
}

# Signal
s_hists = util.load(saved_signal_hists)[0]
s_pts   = utils.get_signal_point_dict(s_hists)

size = (8, 6)

_cmap = plt.cm.viridis.copy()
_cmap.set_bad('white')  # nan bins (zero denominator) render white


def rebin_variable_2D(h, new_edges_x, new_edges_y):
    """Rebin a 2D Hist to arbitrary variable bin edges by summing values and variances."""
    old_ex = h.axes[0].edges
    old_ey = h.axes[1].edges
    new_ex = np.array(new_edges_x)
    new_ey = np.array(new_edges_y)
    vals  = h.values()
    vars_ = h.variances()
    nx = len(new_ex) - 1
    ny = len(new_ey) - 1
    new_vals = np.zeros((nx, ny))
    new_vars = np.zeros((nx, ny))
    lox = np.searchsorted(old_ex, new_ex[:-1], side='left')
    hix = np.searchsorted(old_ex, new_ex[1:],  side='left')
    loy = np.searchsorted(old_ey, new_ey[:-1], side='left')
    hiy = np.searchsorted(old_ey, new_ey[1:],  side='left')
    for i in range(nx):
        for j in range(ny):
            new_vals[i, j] = vals[lox[i]:hix[i], loy[j]:hiy[j]].sum()
            new_vars[i, j] = vars_[lox[i]:hix[i], loy[j]:hiy[j]].sum()
    ax_x = Variable(new_ex, name=h.axes[0].name, label=h.axes[0].label)
    ax_y = Variable(new_ey, name=h.axes[1].name, label=h.axes[1].label)
    h_new = Hist(ax_x, ax_y, storage=Weight())
    h_new.view(flow=False)[...] = np.stack([new_vals, new_vars], axis=-1)
    return h_new


def ratio_hist_2D(h_num, h_den):
    """2D efficiency ratio with proper variance propagation. Zero-denominator bins set to nan."""
    num     = h_num.values()
    den     = h_den.values()
    num_var = h_num.variances()
    den_var = h_den.variances()
    with np.errstate(invalid='ignore', divide='ignore'):
        ratio = np.where(den > 0, num / den, np.nan)
        ratio_var = np.where(
            den > 0,
            ratio**2 * (num_var / np.where(num > 0, num**2, 1)
                      + den_var / np.where(den > 0, den**2, 1)),
            0
        )
    h_ratio = Hist(copy.deepcopy(h_num.axes[0]), copy.deepcopy(h_num.axes[1]), storage=Weight())
    h_ratio.view(flow=False)[...] = np.stack([ratio, ratio_var], axis=-1)
    return h_ratio


def pick_representative_samples(pts_df, max_samples=6):
    """Select evenly-spaced samples across the sorted (mchi, ctau) grid."""
    df = pts_df.sort_values(['mchi', 'delta', 'ctau'])
    n = len(df)
    if n <= max_samples:
        return df.index.tolist()
    indices = np.round(np.linspace(0, n - 1, max_samples)).astype(int)
    return df.index[indices].tolist()


def sample_label(row):
    return rf"$M_\chi$={row['mchi']:.0f}, $\Delta M$={row['dmchi']:.1f}, $c\tau$={row['ctau']:.0f} mm"


def plot_2D_eff(h_eff, ax, title_str):
    hep.hist2dplot(h_eff, ax=ax, cmap=_cmap, cmin=0, cmax=1, cbarextend=False)
    ax.set_xlabel(r'$p_T$ [GeV]')
    ax.set_ylabel(r'$v_{xy}$ [cm]')
    ax.set_title(title_str, fontsize=11)


selected_samples = pick_representative_samples(s_pts)

# ── All-samples efficiency plots ───────────────────────────────────────────────

sumhists = [
    s_hists[v][{"cut": plot_dict['cut'], "samp": sum}]
    for v in histvars
]
alllpt_hists = [
    s_hists[v][{"cut": plot_dict['cut'], "samp": sum}]
    for v in alllpt_histvars
]

sumhists_rb     = [rebin_variable_2D(h, _pt_edges, _lxy_edges) for h in sumhists]
alllpt_hists_rb = [rebin_variable_2D(h, _pt_edges, _lxy_edges) for h in alllpt_hists]

den_all      = sumhists_rb[0] + sumhists_rb[1] + sumhists_rb[2]
den_alllpt   = alllpt_hists_rb[0] + alllpt_hists_rb[1]

eff_all = {
    'lpt':    ratio_hist_2D(sumhists_rb[0],                  den_all),
    'ged':    ratio_hist_2D(sumhists_rb[1],                  den_all),
    'both':   ratio_hist_2D(sumhists_rb[0] + sumhists_rb[1], den_all),
    'alllpt': ratio_hist_2D(alllpt_hists_rb[0],              den_alllpt),
}
labels_all = {
    'lpt': 'Lpt', 'ged': 'GED', 'both': 'Both', 'alllpt': 'AllLowPt',
}

for tag, h_eff in eff_all.items():
    fig, ax = plt.subplots(figsize=size)
    hep.cms.label('Private Work', data=True, year=plot_dict['year'], com='13.6', ax=ax)
    plot_2D_eff(h_eff, ax, rf'{title}: {labels_all[tag]} Efficiency ($p_T$ vs $v_{{xy}}$) — all samples')
    plt.tight_layout()
    plt.savefig(f"plots/hist_{plottag}_{tag}_allsamps.png")
    plt.close(fig)

# ── Per-sample efficiency plots (2×2 grid per sample) ─────────────────────────

for sname in selected_samples:
    ps_hists = [
        s_hists[v][{"cut": plot_dict['cut'], "samp": sname}]
        for v in histvars
    ]
    ps_alllpt_hists = [
        s_hists[v][{"cut": plot_dict['cut'], "samp": sname}]
        for v in alllpt_histvars
    ]
    ps_rb     = [rebin_variable_2D(h, _pt_edges, _lxy_edges) for h in ps_hists]
    ps_alllpt_rb = [rebin_variable_2D(h, _pt_edges, _lxy_edges) for h in ps_alllpt_hists]

    den_ps       = ps_rb[0] + ps_rb[1] + ps_rb[2]
    den_alllpt_ps = ps_alllpt_rb[0] + ps_alllpt_rb[1]

    eff_ps = {
        'lpt':    ratio_hist_2D(ps_rb[0],            den_ps),
        'ged':    ratio_hist_2D(ps_rb[1],            den_ps),
        'both':   ratio_hist_2D(ps_rb[0] + ps_rb[1], den_ps),
        'alllpt': ratio_hist_2D(ps_alllpt_rb[0],     den_alllpt_ps),
    }

    row = s_pts.loc[sname]
    slabel = sample_label(row)

    fig, axes = plt.subplots(2, 2, figsize=(size[0] * 2, size[1] * 2))
    hep.cms.label('Private Work', data=True, year=plot_dict['year'], com='13.6', ax=axes[0, 0])
    plot_2D_eff(eff_ps['lpt'],    axes[0, 0], rf'Lpt — {slabel}')
    plot_2D_eff(eff_ps['ged'],    axes[0, 1], rf'GED — {slabel}')
    plot_2D_eff(eff_ps['both'],   axes[1, 0], rf'Both — {slabel}')
    plot_2D_eff(eff_ps['alllpt'], axes[1, 1], rf'AllLowPt — {slabel}')
    fig.suptitle(rf'{title} Efficiency ($p_T$ vs $v_{{xy}}$)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"plots/hist_{plottag}_persamp_{sname}.png")
    plt.close(fig)
