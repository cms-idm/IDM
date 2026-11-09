# imports
import re
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
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

plt.rcParams.update({
    'font.size':        24,
    'axes.titlesize':   32,
    # Lift the title clear of the CMS / lumi label hep.cms.label draws just
    # above the axes.
    'axes.titlepad':    44,
    'axes.labelsize':   34,
    'xtick.labelsize':  30,
    'ytick.labelsize':  30,
    'legend.fontsize':  24,
    'figure.titlesize': 32,
})

# Cut-name labels on the x-axis stay small so the long descriptions fit.
CUT_LABEL_FS = 17
CMS_LABEL_FS = 26
LEGEND_FS = 20
# Background labels are short, so they're packed into columns.
BKG_LEGEND_NCOL = 3

# Display name for each cut config (configs/cut_configs/<selection>_selection.py)
SELECTION_NAMES = {
    'an':    'Resolved Selection',
    'aptrk': 'Merged Selection',
}

outdir = 'workarea'
bkg_vers = 'Aug2026'
sig_vers = 'Jul2026noID'
selection = 'aptrk'
year = 2024
hists = 'cutflow_only'

saved_signal_hists = f"{outdir}/hists_sig{sig_vers}_{selection}-sel_{hists}.coffea"
saved_bkg_hists = f"{outdir}/hists_bkg{bkg_vers}_{selection}-sel_{hists}.coffea"
plottag = f'sb{bkg_vers}_{selection}-sel'

# All samples here are processed as year 2024; 'cutflow_cts' is already
# xsec * getLumi(2024) * efficiency, so it's plotted as-is (no rescaling).
LUMI_FB, _ = tools.getLumi(year)


def _new_fig():
    fig, ax = plt.subplots(figsize=(16, 12))
    hep.cms.label('Private Work', data=True, year=year, lumi=round(LUMI_FB),
                  com='13.6', ax=ax, fontsize=CMS_LABEL_FS)
    return fig, ax


# ---- Cut-column alignment ----------------------------------------------
#
# Signal and background cutflows aren't guaranteed to share the same set/order
# of cut columns: skimmed background samples carry an extra 'passHLT'
# preselection step (applied before cut1) that signal never goes through, and
# raw dict insertion order isn't guaranteed to put 'all' (the true baseline)
# first. These helpers build a canonical column order keyed by cut index
# (all, cut1..cutN) so both dataframes can be reindexed onto the same x-axis.
#
# 'passHLT' (background's skim-level HLT/MET preselection, applied before the
# full-event ntuple even starts) is treated as the same step as 'cut2' (the
# MET-trigger cut in the shared full-event cut chain) -- both represent
# "passed the MET trigger" -- so it's dropped rather than given its own x-axis
# slot; background's own recorded cut2 value is plotted at that position
# instead. The two aren't literally the same computation, so their event
# counts can differ slightly; that's expected and can be ignored.

def _cut_sort_key(idx):
    if idx == 'all':
        return (0, 0)
    m = re.match(r'^cut(\d+)$', idx)
    if m:
        return (1, int(m.group(1)))
    return (2, idx)

def align_sig_bkg_cut_columns(sig_histo, bkg_histos):
    """
    Returns (union_idx, cut_labels, sig_desc_to_idx, bkg_desc_to_idx):
      - union_idx: canonical list of cut indices (e.g. 'all','cut1',...)
      - cut_labels: matching human-readable descriptions for the x-axis
      - sig_desc_to_idx / bkg_desc_to_idx: maps to rename a cutflow dataframe's
        description-named columns back to their cut index before reindexing
    """
    sig_idx = utils.get_signal_list_of_cuts(sig_histo, get_cut_idx=True)
    sig_desc = utils.get_signal_list_of_cuts(sig_histo, get_cut_idx=False)
    bkg_idx = utils.get_bkg_list_of_cuts(bkg_histos, get_cut_idx=True)
    bkg_desc = utils.get_bkg_list_of_cuts(bkg_histos, get_cut_idx=False)

    # Drop 'passHLT' -- it's treated as the same step as 'cut2' (see module
    # note above), so it gets no x-axis slot of its own.
    bkg_idx, bkg_desc = ([i for i in bkg_idx if i != 'passHLT'],
                          [d for i, d in zip(bkg_idx, bkg_desc) if i != 'passHLT'])

    sig_desc_to_idx = {d: i for i, d in zip(sig_idx, sig_desc)}
    bkg_desc_to_idx = {d: i for i, d in zip(bkg_idx, bkg_desc)}

    idx_to_desc = dict(zip(bkg_idx, bkg_desc))
    idx_to_desc.update(dict(zip(sig_idx, sig_desc)))  # prefer signal's wording where both exist

    union_idx = sorted(set(sig_idx) | set(bkg_idx), key=_cut_sort_key)
    cut_labels = [idx_to_desc[i] for i in union_idx]

    return union_idx, cut_labels, sig_desc_to_idx, bkg_desc_to_idx

def realign_cutflow_df(df, desc_to_idx, union_idx):
    """Rename a cutflow dataframe's description columns to cut indices, then
    reindex onto the canonical union of cut indices (missing ones -> NaN)."""
    return df.rename(columns=desc_to_idx).reindex(columns=union_idx)


# ---- Combined signal+background cutflow plot ---------------------------

def _auto_ylim(values, pad_decades=0.3):
    """Pick log-scale y-limits from the actual plotted values, padded by
    `pad_decades` orders of magnitude on each side."""
    finite = [v for v in values if np.isfinite(v) and v > 0]
    if not finite:
        return None
    lo = 10 ** (np.floor(np.log10(min(finite))) - pad_decades)
    hi = 10 ** (np.ceil(np.log10(max(finite))) + pad_decades)
    return [lo, hi]

def _make_room_for_legends(fig, ax, legends, values, gap=0.03):
    """Raise the top of the y-axis so every plotted value sits below the
    lowest legend edge (legends stay inside the axes, above the data)."""
    fig.tight_layout()
    fig.canvas.draw()
    to_axes = ax.transAxes.inverted()
    legend_bottom = min(to_axes.transform(leg.get_window_extent())[0][1] for leg in legends)
    frac = legend_bottom - gap      # fraction of axes height left for the data
    if frac <= 0.2:
        return

    log = ax.get_yscale() == 'log'
    finite = [v for v in values if np.isfinite(v) and (v > 0 or not log)]
    if not finite:
        return
    lo, _ = ax.get_ylim()
    hi = max(finite)
    if log:
        lo_l, hi_l = np.log10(lo), np.log10(hi)
        ax.set_ylim(lo, 10 ** (lo_l + (hi_l - lo_l) / frac))
    else:
        ax.set_ylim(lo, lo + (hi - lo) / frac)

def _select_sig_points(sig_df, plot_dict):
    """Return [(point, sig_dict), ...] for the signal points matching
    plot_dict's m1s/deltas/ctaus, sorted by m1 so distinct shades of red can
    be assigned across exactly that many lines -- lightest shade for the
    lowest mass, darkest for the highest."""
    sig_points = []
    for point in sig_df.index.values:
        sig_dict = ptools.signalPoint(point)
        m1 = round(sig_dict['m1'], 5)
        delta = round(sig_dict['delta'], 5)
        ctau = int(sig_dict['ctau'])

        m1_disp = int(m1) if m1.is_integer() else m1
        delta_disp = int(delta) if delta.is_integer() else delta

        if (m1_disp in plot_dict['m1s']) and (delta_disp in plot_dict['deltas']) and (ctau in plot_dict['ctaus']):
            sig_points.append((point, sig_dict))

    return sorted(sig_points, key=lambda p: round(p[1]['m1'], 5))

def _sig_label(sig_dict):
    m1 = round(sig_dict['m1'], 5)
    dmchi = round(sig_dict['dmchi'], 5)
    ctau = int(sig_dict['ctau'])
    return rf"($M_1$, $\Delta$) = ({m1}, {dmchi}) GeV, c$\tau$ = {ctau}mm"

def plot_sig_bkg_yields(sig_histo, bkg_histos, sig_df_eff, sig_df_cts, bkg_df_cts, plot_dict,
                         sig_df_wts=None, show=False):
    """
    Overlay combined-background cutflow yields (one line per background
    process category -- e.g. all QCD subprocesses summed into a single "QCD"
    line, all WJets subprocesses summed into a single "WJets" line, etc --
    plus a "Total Bkg" line) with a selected subset of signal points, all on
    the same axes. The vertical axis is absolute event yield (xsec * cut
    efficiency * integrated luminosity), not efficiency.

    sig_df_eff: utils.get_signal_cutflow_dict(sig_histo, 'cutflow')          -- used only to build the yield uncertainty
    sig_df_cts: utils.get_signal_cutflow_dict(sig_histo, 'cutflow_cts')      -- xsec * lumi * efficiency
    bkg_df_cts: utils.get_bkg_cutflow_df(bkg_histos, 'cutflow_cts', process='all')

    Example plot_dict:

    plot_dict = {
        # Select signal points to display
        'm1s': [0.05, 0.5, 5, 50],
        'deltas': [0.1],
        'ctaus': [10],

        # Select background process categories to display ('all', or a list
        # such as ['QCD', 'WJets', 'Total'])
        'bkg_processes': 'all',

        # Plot display styling
        'ylim': None,   # None to auto-scale to the plotted data, or [ymin, ymax]
        'doLog': True,
        'ylabel': 'Events',
        'title': rf"Cutflow: $\Delta$ = {deltas}, c$\tau$ = {ctaus}mm",

        # Plot saving
        'doSave': False,
        'outDir': './plots/',
        'outName': 'cutflow_sig_bkg.png'
    }
    """

    union_idx, cut_labels, sig_desc_to_idx, bkg_desc_to_idx = align_sig_bkg_cut_columns(sig_histo, bkg_histos)
    x = np.arange(len(union_idx))

    bkg_df_cts = realign_cutflow_df(bkg_df_cts, bkg_desc_to_idx, union_idx)
    sig_df_eff = realign_cutflow_df(sig_df_eff, sig_desc_to_idx, union_idx)
    sig_df_cts = realign_cutflow_df(sig_df_cts, sig_desc_to_idx, union_idx)
    if sig_df_wts is not None:
        sig_df_wts = realign_cutflow_df(sig_df_wts, sig_desc_to_idx, union_idx)

    fig, ax = _new_fig()

    plotted_vals = []

    # background lines, combined per process category
    bkg_selection = plot_dict.get('bkg_processes', 'all')
    if bkg_selection == 'all':
        bkg_processes = [p for p in bkg_df_cts.index.values if p != 'Total']
        doTotal = True
    else:
        bkg_processes = [p for p in bkg_selection if p != 'Total']
        doTotal = 'Total' in bkg_selection

    bkg_handles, sig_handles = [], []

    for process in bkg_processes:
        color = ptools.bkg_cmap.get(process, 'gray')
        vals = bkg_df_cts.loc[process].values.astype(float)
        bkg_handles += ax.plot(x, vals, label=process, color=color, ls='--', lw=1.5, alpha=0.6)
        plotted_vals.extend(vals)

    if doTotal:
        vals = bkg_df_cts.loc['Total'].values.astype(float)
        bkg_handles += ax.plot(x, vals, label='Total Bkg', color='black', ls='--', lw=3)
        plotted_vals.extend(vals)

    # signal lines, subset selected by m1/delta/ctau
    sig_points = _select_sig_points(sig_df_cts, plot_dict)
    sig_colors = plt.cm.Reds(np.linspace(0.4, 0.95, max(len(sig_points), 1)))

    for color_idx, (point, sig_dict) in enumerate(sig_points):
        color = sig_colors[color_idx]

        label = _sig_label(sig_dict)
        counts = sig_df_cts.loc[point].values.astype(float)
        if sig_df_wts is not None:
            eff = sig_df_eff.loc[point].values.astype(float)
            W_0 = sig_df_eff.loc[point].iloc[0]         # cutflow['all'], guaranteed first in union_idx
            sw2_all = sig_df_wts.loc[point].iloc[0]
            yerr_eff = np.sqrt(eff * (1 - eff) * sw2_all) / W_0
            norm = counts[0] / W_0                      # xsec * lumi for this signal point
            yerr = yerr_eff * norm
            sig_handles.append(ax.errorbar(x, counts, yerr=yerr, label=label, color=color,
                                           fmt='-o', markersize=4, capsize=3, lw=2))
            plotted_vals.extend(counts + yerr)
        else:
            sig_handles += ax.plot(x, counts, label=label, color=color,
                                   lw=2, marker='o', markersize=4)
        plotted_vals.extend(counts)

    if plot_dict.get('doLog', True):
        ax.set_yscale('log')

    ylim = plot_dict.get('ylim')
    if ylim is None:
        ylim = _auto_ylim(plotted_vals)
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    ax.grid()

    ax.set_ylabel(plot_dict.get('ylabel', 'Events'))
    ax.set_title(plot_dict.get('title', ''))

    ax.set_xticks(ticks=x, labels=cut_labels, rotation=45, ha='right')
    ax.tick_params(axis='x', labelsize=CUT_LABEL_FS)

    # Background (short labels, packed in columns) upper left; signal upper right.
    legends = []
    if bkg_handles:
        legends.append(ax.legend(handles=bkg_handles, loc='upper left',
                                 ncol=BKG_LEGEND_NCOL, fontsize=LEGEND_FS))
        ax.add_artist(legends[-1])
    if sig_handles:
        legends.append(ax.legend(handles=sig_handles, loc='upper right',
                                 ncol=1, fontsize=LEGEND_FS))
    if legends and plot_dict.get('ylim') is None:
        _make_room_for_legends(fig, ax, legends, plotted_vals)

    if plot_dict.get('doSave', False):
        os.makedirs(plot_dict['outDir'], exist_ok=True)
        plt.tight_layout()
        outpath = f"{plot_dict['outDir']}/{plot_dict['outName']}"
        plt.savefig(outpath)
        print(f"Saved: {outpath}")

    if show:
        plt.show()

    plt.close(fig)


# ---- Signal / background ratio plot ------------------------------------

def plot_sig_bkg_ratios(sig_histo, bkg_histos, sig_df_cts, bkg_df_cts, plot_dict, show=False):
    """
    For each selected signal point, plot S/B (solid) and S/sqrt(B) (dotted)
    at every cut step, where B is the total background yield. Uses the same
    signal-point selection, cut-column alignment and plot_dict keys as
    plot_sig_bkg_yields ('bkg_processes' is ignored -- only 'Total' is used).
    Cut steps where the total background or the signal is zero are left out (NaN).
    """

    union_idx, cut_labels, sig_desc_to_idx, bkg_desc_to_idx = align_sig_bkg_cut_columns(sig_histo, bkg_histos)
    x = np.arange(len(union_idx))

    bkg_df_cts = realign_cutflow_df(bkg_df_cts, bkg_desc_to_idx, union_idx)
    sig_df_cts = realign_cutflow_df(sig_df_cts, sig_desc_to_idx, union_idx)

    bkg_total = bkg_df_cts.loc['Total'].values.astype(float)
    bkg_total = np.where(bkg_total > 0, bkg_total, np.nan)

    fig, ax = _new_fig()

    plotted_vals = []

    sig_points = _select_sig_points(sig_df_cts, plot_dict)
    sig_colors = plt.cm.Reds(np.linspace(0.4, 0.95, max(len(sig_points), 1)))

    for color_idx, (point, sig_dict) in enumerate(sig_points):
        color = sig_colors[color_idx]
        counts = sig_df_cts.loc[point].values.astype(float)
        counts = np.where(counts > 0, counts, np.nan)   # zero signal can't be shown on a log axis

        s_over_b = counts / bkg_total
        s_over_sqrtb = counts / np.sqrt(bkg_total)

        ax.plot(x, s_over_b, label=_sig_label(sig_dict), color=color,
                ls='-', lw=2, marker='o', markersize=4)
        ax.plot(x, s_over_sqrtb, color=color,
                ls=':', lw=2.5, marker='o', markersize=4)
        plotted_vals.extend(s_over_b)
        plotted_vals.extend(s_over_sqrtb)

    if plot_dict.get('doLog', True):
        ax.set_yscale('log')

    ylim = plot_dict.get('ylim')
    if ylim is None:
        ylim = _auto_ylim(plotted_vals)
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    ax.grid()

    ax.set_ylabel(plot_dict.get('ylabel', r'$S/B$, $S/\sqrt{B}$'))
    ax.set_title(plot_dict.get('title', ''))

    ax.set_xticks(ticks=x, labels=cut_labels, rotation=45, ha='right')
    ax.tick_params(axis='x', labelsize=CUT_LABEL_FS)

    # line-style key for the two ratios upper left; color legend for signal points upper right
    style_handles = [plt.Line2D([], [], color='gray', ls='-', lw=2, label=r'$S/B$'),
                     plt.Line2D([], [], color='gray', ls=':', lw=2.5, label=r'$S/\sqrt{B}$')]
    legends = [ax.legend(handles=style_handles, loc='upper left', ncol=1, fontsize=LEGEND_FS)]
    ax.add_artist(legends[-1])
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        legends.append(ax.legend(handles=handles, loc='upper right', ncol=1, fontsize=LEGEND_FS))
    if plot_dict.get('ylim') is None:
        _make_room_for_legends(fig, ax, legends, plotted_vals)

    if plot_dict.get('doSave', False):
        os.makedirs(plot_dict['outDir'], exist_ok=True)
        plt.tight_layout()
        outpath = f"{plot_dict['outDir']}/{plot_dict['outName']}"
        plt.savefig(outpath)
        print(f"Saved: {outpath}")

    if show:
        plt.show()

    plt.close(fig)


def plot_sig_bkg_all(sig_histo, bkg_histos, sig_df_eff, sig_df_cts, bkg_df_cts, plot_dict, sig_df_wts=None):
    """Make the yield cutflow plot and the matching S/B, S/sqrt(B) ratio plot
    for the same signal points. The ratio plot is saved alongside the yield
    plot with 'cutflow_sig-bkg' -> 'cutflow_sig-bkg-ratio' in its file name."""
    plot_sig_bkg_yields(sig_histo, bkg_histos, sig_df_eff, sig_df_cts, bkg_df_cts, plot_dict, sig_df_wts=sig_df_wts)

    ratio_dict = dict(plot_dict)
    ratio_dict['ylabel'] = r'$S/B$ (solid), $S/\sqrt{B}$ (dotted)'
    ratio_dict['ylim'] = None
    ratio_dict['title'] = plot_dict['title'].replace('Signal + Background Cutflow', r'$S/B$ and $S/\sqrt{B}$')
    ratio_dict['outName'] = plot_dict['outName'].replace('cutflow_sig-bkg', 'cutflow_sig-bkg-ratio')
    plot_sig_bkg_ratios(sig_histo, bkg_histos, sig_df_cts, bkg_df_cts, ratio_dict)


# ---- Load histograms -----------------------------------------------------

# Signal
s_hists = util.load(saved_signal_hists)[0]
s_pts = utils.get_signal_point_dict(s_hists)
sig_df_eff = utils.get_signal_cutflow_dict(s_hists, 'cutflow')
sig_df_cts = utils.get_signal_cutflow_dict(s_hists, 'cutflow_cts')
sig_df_wts = utils.get_signal_cutflow_dict(s_hists, 'cutflow_wgts2')

# Background - combine all subprocesses of a given category (QCD, WJets, ZJets, ...)
# into a single line, plus a "Total" line summed over all categories
b_hists = util.load(saved_bkg_hists)[0]
bkg_df_cts = utils.get_bkg_cutflow_df(b_hists, 'cutflow_cts', process='all')

# 'all' already comes from the slim (true, unskimmed) tree -- see Analyzer._processSlim.
# Overwrite the MET-filters and MET-trigger columns with the independent measurement
# made directly on that same slim tree (rather than the full-tree dataset, which sits
# downstream of the background HLT/MET preselection) -- the two should agree at the
# MET-trigger step, since that's where the full tree's preselection is reproduced.
# Every cut after the MET trigger keeps using the full-tree dataset's own recorded values.
bkg_slim_overrides = utils.get_bkg_slim_cut_overrides(b_hists)
for col in bkg_slim_overrides.columns:
    bkg_df_cts[col] = bkg_slim_overrides[col]

plot_dict = {

    # Select signal points to display
    'm1s': [0.05, 0.5, 5, 50],
    'deltas': [0.1],
    'ctaus': [10],

    # Select background process categories to display
    'bkg_processes': 'all',  # or e.g. ['QCD', 'WJets', 'Total']

    # Plot display styling
    'ylim': None,   # auto-scaled to the plotted data
    'doLog': True,

    'ylabel': 'Events',
    # Signal-point parameters are shown in the legend; CMS label carries lumi/year
    'title': f"Signal + Background Cutflow [{SELECTION_NAMES[selection]}]",

    # Plot saving
    'doSave': True,
    'outDir': './plots/',
    'outName': ''
}

m1s = [0.05, 0.5, 5, 50]
deltas = [0.1]
ctaus = [10]

plot_dict['m1s'] = m1s; plot_dict['deltas'] = deltas; plot_dict['ctaus'] = ctaus
plot_dict['outName'] = f'cutflow/cutflow_sig-bkg_{plottag}_ctau-{utils.stringfy_friendly(ctaus[0])}_delta-{utils.stringfy_friendly(deltas[0])}_m1-wide.png'

plot_sig_bkg_all(s_hists, b_hists, sig_df_eff, sig_df_cts, bkg_df_cts, plot_dict, sig_df_wts=sig_df_wts)

m1s = [0.5, 1, 2, 5]
deltas = [0.1]
ctaus = [10]

plot_dict['m1s'] = m1s; plot_dict['deltas'] = deltas; plot_dict['ctaus'] = ctaus
plot_dict['outName'] = f'cutflow/cutflow_sig-bkg_{plottag}_ctau-{utils.stringfy_friendly(ctaus[0])}_delta-{utils.stringfy_friendly(deltas[0])}_m1-narrow.png'

plot_sig_bkg_all(s_hists, b_hists, sig_df_eff, sig_df_cts, bkg_df_cts, plot_dict, sig_df_wts=sig_df_wts)

m1s = [0.5]
deltas = [0.1]
ctaus = [1, 10, 100]

plot_dict['m1s'] = m1s; plot_dict['deltas'] = deltas; plot_dict['ctaus'] = ctaus
plot_dict['outName'] = f'cutflow/cutflow_sig-bkg_{plottag}_m1-{utils.stringfy_friendly(m1s[0])}_delta-{utils.stringfy_friendly(deltas[0])}.png'

plot_sig_bkg_all(s_hists, b_hists, sig_df_eff, sig_df_cts, bkg_df_cts, plot_dict, sig_df_wts=sig_df_wts)

m1s = [5]
deltas = [0.1]
ctaus = [1, 10, 100]

plot_dict['m1s'] = m1s; plot_dict['deltas'] = deltas; plot_dict['ctaus'] = ctaus
plot_dict['outName'] = f'cutflow/cutflow_sig-bkg_{plottag}_m1-{utils.stringfy_friendly(m1s[0])}_delta-{utils.stringfy_friendly(deltas[0])}.png'

plot_sig_bkg_all(s_hists, b_hists, sig_df_eff, sig_df_cts, bkg_df_cts, plot_dict, sig_df_wts=sig_df_wts)
