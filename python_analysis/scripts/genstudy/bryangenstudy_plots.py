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

outdir = os.path.join(REPO_ROOT, 'workarea')
plotdir = os.path.join(REPO_ROOT, 'plots', 'genstudy', 'bryan')
os.makedirs(plotdir, exist_ok=True)
saved_signal_hists = f"{outdir}/hists_sigJul2026noID_anmatchvtx-sel_genstudy.coffea"

# Signal
s_hists = util.load(saved_signal_hists)[0]
s_pts = utils.get_signal_point_dict(s_hists)
s_histnames = utils.get_signal_list_of_histograms(s_hists)
s_cutsidx = utils.get_signal_list_of_cuts(s_hists, get_cut_idx = True)
s_cutsname = utils.get_signal_list_of_cuts(s_hists, get_cut_idx = False)

df = utils.get_signal_cutflow_dict(s_hists, 'cutflow')

size = (16, 12)
sel_label = 'prevtx'

# Requested plots, one entry per variable. 'log_variables'/'doLogx' are only
# set for lxy, which has a dedicated log-spaced-binning histogram (matching
# the convention in genele_vxy_plots.py); the other variables reuse their
# linear histogram with just the y-axis switched to log scale.
VARIABLES = [
    dict(variables=['gen_diele_dR'], log_variables=None, cut='cut8',
         title=r'Gen EE $\Delta R$', label='gen-ee-dr', ls=['-']),
    dict(variables=['gen_diele_pt'], log_variables=None, cut='cut1',
         title=r'Gen EE $p_T$', label='gen-ee-pt', ls=['-']),
    dict(variables=['gen_leading_ele_pt'], log_variables=None, cut='cut1',
         title=r'Gen Leading Electron $p_T$', label='gen-e-leading-pt', ls=['-']),
    dict(variables=['gen_subleading_ele_pt'], log_variables=None, cut='cut1',
         title=r'Gen Subleading Electron $p_T$', label='gen-e-subleading-pt', ls=['-']),
    dict(variables=['gen_leading_ele_lxy', 'gen_subleading_ele_lxy'],
         log_variables=['gen_leading_ele_lxy_log', 'gen_subleading_ele_lxy_log'], cut='cut1',
         title=r'Gen Electron $L_{xy}$ (electron vtx $-$ $\chi_2$ vtx)', label='gen-e-lxy', ls=['-', '--']),
]

# Requested parameter sets
PARAM_SETS = [
    dict(m1s=[0.5, 1, 2, 5], deltas=[0.1],              ctaus=[10]),
    dict(m1s=[5],            deltas=[0.05, 0.1, 0.2],   ctaus=[10]),
    dict(m1s=[0.5],          deltas=[0.05, 0.1, 0.2],   ctaus=[10]),
    dict(m1s=[5],            deltas=[0.1],              ctaus=[1, 10, 100]),
    dict(m1s=[0.5],          deltas=[0.1],              ctaus=[1, 10, 100]),
]

def param_tag(m1s, deltas, ctaus):
    fmt = lambda vals: "-".join(utils.stringfy_friendly(v) for v in vals)
    return f"m1-{fmt(m1s)}_delta-{fmt(deltas)}_ctau-{fmt(ctaus)}"

def make_plot(varcfg, m1s, deltas, ctaus, doLog):
    use_dedicated_log_hist = doLog and varcfg['log_variables'] is not None
    variables = varcfg['log_variables'] if use_dedicated_log_hist else varcfg['variables']

    fig, ax = plt.subplots(figsize=size)
    plot_dict = {
        'variable': variables,
        'year': 2024,
        'cut': varcfg['cut'],
    }
    style_dict = {
        'fig': fig, 'ax': ax,
        'rebin': 1j, 'xlim': None,
        'doLogy': doLog, 'doLogx': use_dedicated_log_hist, 'doDensity': True, 'doYerr': False,
        'xlabel': None, 'ylabel': None, 'label': None,
        'flow': None,
        'doSave': False,
        'ls': varcfg['ls'],
    }

    cmap_idx = 0
    for m1 in m1s:
        for delta in deltas:
            for ctau in ctaus:
                ptools.plot_signal_1D(s_hists, m1, delta, ctau, plot_dict, style_dict, cmap_idx=cmap_idx)
                cmap_idx += 1

    scale_suffix = '-log' if doLog else ''
    title_suffix = ' (log scale)' if doLog else ''
    plt.title(rf"{varcfg['title']}{title_suffix}: $M_1$ = {m1s}, $\Delta$ = {deltas}, c$\tau$ = {ctaus}mm")
    plt.legend()
    tag = param_tag(m1s, deltas, ctaus)
    plt.savefig(f"{plotdir}/hist_{sel_label}_{varcfg['label']}{scale_suffix}_{tag}.png")
    plt.close(fig)

for pset in PARAM_SETS:
    for varcfg in VARIABLES:
        make_plot(varcfg, pset['m1s'], pset['deltas'], pset['ctaus'], doLog=False)
        make_plot(varcfg, pset['m1s'], pset['deltas'], pset['ctaus'], doLog=True)
