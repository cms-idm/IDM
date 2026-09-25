# imports
import os
import numpy as np
import matplotlib.pyplot as plt

import analysisTools.utils as utils
import analysisTools.plotTools as ptools
import coffea.util as util

outdir = 'workarea'
plotdir = './plots/vtxrecoeff'
os.makedirs(plotdir, exist_ok=True)

sig_vers = 'Jul2026noID'
selection = 'an'
hists = 'mergedcats'
saved_signal_hists = f'{outdir}/hists_sig{sig_vers}_{selection}-sel_{hists}.coffea'
plottag = f'sig{sig_vers}_{selection}-sel'

# Substring (case-insensitive) used to find the vertex-reco cutflow stage by its
# description, rather than hardcoding a cut index -- matches cut9's `desc` in
# configs/cut_configs/an_selection.py ("Has Good ee Vertex").
VERTEX_CUT_KEYWORD = 'good ee vertex'


def get_vtx_reco_efficiency(sig_histo):
    """
    Per-sample cumulative efficiency of surviving through the "Has Good ee Vertex"
    cutflow stage -- i.e. the fraction of generated signal events that still have
    a good ee vertex after all upstream event-selection cuts.

    Returns a pandas Series indexed by sample name.
    """
    vertex_idx = utils.find_cut_idx_by_desc(sig_histo, VERTEX_CUT_KEYWORD)

    # Position of the vertex cut among the cut indices, in the same order used
    # to label get_signal_cutflow_dict's columns -- selecting by position sidesteps
    # get_signal_list_of_cuts()'s description text mangling (see ptools.getCut).
    cut_idx_list = utils.get_signal_list_of_cuts(sig_histo, get_cut_idx=True)
    col_pos = cut_idx_list.index(vertex_idx)

    cutflow_df = utils.get_signal_cutflow_dict(sig_histo, 'cutflow')
    return cutflow_df.iloc[:, col_pos]


def plot_vtx_eff_vs_m1(sig_histo, s_pts, plot_dict):
    """
    Plot vertex-reco efficiency vs m1, one curve per (delta, ctau) combination.

    Example plot_dict:

    plot_dict = {
        'deltas': [0.1],        # dMchi/m1 values to draw curves for
        'ctaus':  [1, 10, 100], # ctau [mm] values to draw curves for

        'ylim': None,   # None to auto-scale, or [ymin, ymax]
        'doLog': True,
        'title': '',

        'doSave': True,
        'outDir': plotdir,
        'outName': 'vtxrecoeff_vs_m1.png',
    }
    """
    eff = get_vtx_reco_efficiency(sig_histo)

    df = s_pts.copy()
    df['vtx_eff'] = eff
    # m1 = 0.05 GeV samples are excluded -- edge-of-grid point with poor stats
    df = df[~np.isclose(df['m1'], 0.05)]

    fig, ax = plt.subplots(figsize=(10, 8))

    deltas = plot_dict['deltas']
    ctaus = plot_dict['ctaus']
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(len(deltas) * len(ctaus), 1)))

    color_idx = 0
    for delta in deltas:
        for ctau in ctaus:
            sel = df[np.isclose(df['delta'], delta) & np.isclose(df['ctau'], ctau)]
            sel = sel.sort_values('m1')
            if sel.empty:
                print(f"No signal points found for delta={delta}, ctau={ctau}")
                continue
            label = rf"$\Delta$ = {delta}, c$\tau$ = {int(ctau)}mm"
            ax.plot(sel['m1'], sel['vtx_eff'], label=label, color=colors[color_idx],
                     marker='o', markersize=5, lw=2)
            color_idx += 1

    if plot_dict.get('doLog', True):
        ax.set_yscale('log')
        ax.set_xscale('log')

    ylim = plot_dict.get('ylim')
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    ax.grid()
    ax.set_xlabel(r'$m_1$ [GeV]', fontsize=20)
    ax.set_ylabel('Vertex Reco Efficiency', fontsize=20)
    ax.set_title(plot_dict.get('title', ''), fontsize=18)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(loc='best', fontsize=14)

    if plot_dict.get('doSave', False):
        os.makedirs(plot_dict['outDir'], exist_ok=True)
        plt.tight_layout()
        outpath = f"{plot_dict['outDir']}/{plot_dict['outName']}"
        plt.savefig(outpath)
        print(f"Saved: {outpath}")

    plt.close(fig)


# ---- Load histograms -------------------------------------------------------

s_hists = util.load(saved_signal_hists)[0]
s_pts = utils.get_signal_point_dict(s_hists)

plot_dict = {
    'deltas': [0.05, 0.1, 0.2],
    'ctaus': [1, 10, 100],
    'ylim': None,
    'doLog': True,
    'title': '',
    'doSave': True,
    'outDir': plotdir,
    'outName': '',
}

# All (delta, ctau) combinations overlaid
plot_dict['deltas'] = [0.05, 0.1, 0.2]
plot_dict['ctaus'] = [1, 10, 100]
plot_dict['title'] = 'Vertex Reco Efficiency [AN Selection]'
plot_dict['outName'] = f'vtxrecoeff_vs_m1_{plottag}_allpoints.png'
plot_vtx_eff_vs_m1(s_hists, s_pts, plot_dict)

# Delta dependence at fixed ctau
plot_dict['deltas'] = [0.05, 0.1, 0.2]
plot_dict['ctaus'] = [10]
plot_dict['title'] = rf'Vertex Reco Efficiency [AN Selection]: c$\tau$ = {plot_dict["ctaus"][0]}mm'
plot_dict['outName'] = f'vtxrecoeff_vs_m1_{plottag}_ctau-{utils.stringfy_friendly(plot_dict["ctaus"][0])}.png'
plot_vtx_eff_vs_m1(s_hists, s_pts, plot_dict)

# ctau dependence at fixed delta
plot_dict['deltas'] = [0.1]
plot_dict['ctaus'] = [1, 10, 100]
plot_dict['title'] = rf'Vertex Reco Efficiency [AN Selection]: $\Delta$ = {plot_dict["deltas"][0]}'
plot_dict['outName'] = f'vtxrecoeff_vs_m1_{plottag}_delta-{utils.stringfy_friendly(plot_dict["deltas"][0])}.png'
plot_vtx_eff_vs_m1(s_hists, s_pts, plot_dict)
