# imports
import re
import numpy as np
import matplotlib.pyplot as plt
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

outdir = 'workarea'
bkg_vers = 'Aug2026'
sig_vers = 'Jul2026noID'
selection = 'an'
hists = 'appearingtrack'

saved_signal_hists = f"{outdir}/hists_sig{sig_vers}_{selection}-sel_{hists}.coffea"
saved_bkg_hists = f"{outdir}/hists_bkg{bkg_vers}_{selection}-sel_{hists}.coffea"
plottag = f'sb{bkg_vers}_{selection}-sel'

# All samples here are processed as year 2024; 'cutflow_cts' is already
# xsec * getLumi(2024) * efficiency, so it's plotted as-is (no rescaling).
LUMI_FB, _ = tools.getLumi(2024)


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

def plot_sig_bkg_yields(sig_histo, bkg_histos, sig_df_eff, sig_df_cts, bkg_df_cts, plot_dict,
                         sig_df_wts=None, show=True):
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

    size = (16, 12)
    fig, ax = plt.subplots(figsize=size)

    plotted_vals = []

    # background lines, combined per process category
    bkg_selection = plot_dict.get('bkg_processes', 'all')
    if bkg_selection == 'all':
        bkg_processes = [p for p in bkg_df_cts.index.values if p != 'Total']
        doTotal = True
    else:
        bkg_processes = [p for p in bkg_selection if p != 'Total']
        doTotal = 'Total' in bkg_selection

    for process in bkg_processes:
        color = ptools.bkg_cmap.get(process, 'gray')
        vals = bkg_df_cts.loc[process].values.astype(float)
        ax.plot(x, vals, label=process, color=color, ls='--', lw=2.5)
        plotted_vals.extend(vals)

    if doTotal:
        vals = bkg_df_cts.loc['Total'].values.astype(float)
        ax.plot(x, vals, label='Total Bkg', color='black', ls='--', lw=3)
        plotted_vals.extend(vals)

    # signal lines, subset selected by m1/delta/ctau
    sig_df_cts = sig_df_cts.copy()
    m1_list = [round(ptools.signalPoint(point)['m1'], 5) for point in sig_df_cts.index.values]
    sig_df_cts['m1'] = m1_list
    sig_df_cts = sig_df_cts.sort_values(by=['m1'])
    sig_df_cts.pop('m1')

    color_idx = 0
    for point in sig_df_cts.index.values:
        sig_dict = ptools.signalPoint(point)
        m1 = round(sig_dict['m1'], 5)
        delta = round(sig_dict['delta'], 5)
        dmchi = round(sig_dict['dmchi'], 5)
        ctau = int(sig_dict['ctau'])

        m1_disp = int(m1) if m1.is_integer() else m1
        delta_disp = int(delta) if delta.is_integer() else delta

        if (m1_disp in plot_dict['m1s']) and (delta_disp in plot_dict['deltas']) and (ctau in plot_dict['ctaus']):
            label = rf"($M_1$, $\Delta$) = ({round(m1, 5)}, {round(dmchi, 5)}) GeV, c$\tau$ = {ctau}mm"
            counts = sig_df_cts.loc[point].values.astype(float)
            if sig_df_wts is not None:
                eff = sig_df_eff.loc[point].values.astype(float)
                W_0 = sig_df_eff.loc[point].iloc[0]         # cutflow['all'], guaranteed first in union_idx
                sw2_all = sig_df_wts.loc[point].iloc[0]
                yerr_eff = np.sqrt(eff * (1 - eff) * sw2_all) / W_0
                norm = counts[0] / W_0                      # xsec * lumi for this signal point
                yerr = yerr_eff * norm
                ax.errorbar(x, counts, yerr=yerr, label=label, color=ptools.cmap[color_idx % len(ptools.cmap)],
                            fmt='-o', markersize=4, capsize=3, lw=2)
            else:
                ax.plot(x, counts, label=label, color=ptools.cmap[color_idx % len(ptools.cmap)],
                        lw=2, marker='o', markersize=4)
            plotted_vals.extend(counts)
            color_idx += 1

    if plot_dict.get('doLog', True):
        ax.set_yscale('log')

    ylim = plot_dict.get('ylim')
    if ylim is None:
        ylim = _auto_ylim(plotted_vals)
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    ax.grid()

    ax.set_ylabel(plot_dict.get('ylabel', 'Events'), fontsize=28)
    ax.set_title(plot_dict.get('title', ''), fontsize=24)

    ax.set_xticks(ticks=x, labels=cut_labels, rotation=45, ha='right')
    ax.tick_params(axis='x', labelsize=17)
    ax.tick_params(axis='y', labelsize=28)

    ax.legend(loc='upper right', fontsize=14, ncol=1)

    if plot_dict.get('doSave', False):
        os.makedirs(plot_dict['outDir'], exist_ok=True)
        plt.tight_layout()
        outpath = f"{plot_dict['outDir']}/{plot_dict['outName']}"
        plt.savefig(outpath)
        print(f"Saved: {outpath}")

    if show:
        plt.show()

    plt.close(fig)


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

lumi_tag = rf"$\mathcal{{L}}$ = {LUMI_FB:.0f} fb$^{{-1}}$ (2024)"

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
    'title': '',

    # Plot saving
    'doSave': True,
    'outDir': './plots/',
    'outName': ''
}

m1s = [0.05, 0.5, 5, 50]
deltas = [0.1]
ctaus = [10]

plot_dict['m1s'] = m1s; plot_dict['deltas'] = deltas; plot_dict['ctaus'] = ctaus
plot_dict['title'] = rf"Signal + Background Cutflow [AN Selection, {lumi_tag}]: $M_1$ = {m1s}, $\Delta$ = {deltas}, c$\tau$ = {ctaus}mm"
plot_dict['outName'] = f'cutflow/cutflow_sig-bkg_{plottag}_ctau-{utils.stringfy_friendly(ctaus[0])}_delta-{utils.stringfy_friendly(deltas[0])}_m1-wide.png'

plot_sig_bkg_yields(s_hists, b_hists, sig_df_eff, sig_df_cts, bkg_df_cts, plot_dict, sig_df_wts=sig_df_wts)

m1s = [0.5, 1, 2, 5]
deltas = [0.1]
ctaus = [10]

plot_dict['m1s'] = m1s; plot_dict['deltas'] = deltas; plot_dict['ctaus'] = ctaus
plot_dict['title'] = rf"Signal + Background Cutflow [AN Selection, {lumi_tag}]: $M_1$ = {m1s}, $\Delta$ = {deltas}, c$\tau$ = {ctaus}mm"
plot_dict['outName'] = f'cutflow/cutflow_sig-bkg_{plottag}_ctau-{utils.stringfy_friendly(ctaus[0])}_delta-{utils.stringfy_friendly(deltas[0])}_m1-narrow.png'

plot_sig_bkg_yields(s_hists, b_hists, sig_df_eff, sig_df_cts, bkg_df_cts, plot_dict, sig_df_wts=sig_df_wts)

m1s = [0.5]
deltas = [0.1]
ctaus = [1, 10, 100]

plot_dict['m1s'] = m1s; plot_dict['deltas'] = deltas; plot_dict['ctaus'] = ctaus
plot_dict['title'] = rf"Signal + Background Cutflow [AN Selection, {lumi_tag}]: $M_1$ = {m1s}, $\Delta$ = {deltas}, c$\tau$ = {ctaus}mm"
plot_dict['outName'] = f'cutflow/cutflow_sig-bkg_{plottag}_m1-{utils.stringfy_friendly(m1s[0])}_delta-{utils.stringfy_friendly(deltas[0])}.png'

plot_sig_bkg_yields(s_hists, b_hists, sig_df_eff, sig_df_cts, bkg_df_cts, plot_dict, sig_df_wts=sig_df_wts)

m1s = [5]
deltas = [0.1]
ctaus = [1, 10, 100]

plot_dict['m1s'] = m1s; plot_dict['deltas'] = deltas; plot_dict['ctaus'] = ctaus
plot_dict['title'] = rf"Signal + Background Cutflow [AN Selection, {lumi_tag}]: $M_1$ = {m1s}, $\Delta$ = {deltas}, c$\tau$ = {ctaus}mm"
plot_dict['outName'] = f'cutflow/cutflow_sig-bkg_{plottag}_m1-{utils.stringfy_friendly(m1s[0])}_delta-{utils.stringfy_friendly(deltas[0])}.png'

plot_sig_bkg_yields(s_hists, b_hists, sig_df_eff, sig_df_cts, bkg_df_cts, plot_dict, sig_df_wts=sig_df_wts)
