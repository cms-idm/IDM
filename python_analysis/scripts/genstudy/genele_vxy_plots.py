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

#cuts_config = "configs/selections/minimal_cuts.py"
#hists_config = "configs/hists/genstudy.py"
#sample_config = "configs/samples/signal_2024_Apr2026_aEM.json"
outdir = os.path.join(REPO_ROOT, 'workarea')
plotdir = os.path.join(REPO_ROOT, 'plots', 'genstudy')
os.makedirs(plotdir, exist_ok=True)
saved_signal_hists = f"{outdir}/hists_sigJul2026noID_anmatchvtx-sel_genstudy.coffea"

# Signal
s_hists = util.load(saved_signal_hists)[0]
s_pts = utils.get_signal_point_dict(s_hists)
s_histnames = utils.get_signal_list_of_histograms(s_hists)
s_cutsidx = utils.get_signal_list_of_cuts(s_hists, get_cut_idx = True)
s_cutsname = utils.get_signal_list_of_cuts(s_hists, get_cut_idx = False)
#print(s_pts)
#print(s_histnames)
#print(s_cutsidx)
#print(s_cutsname)

df = utils.get_signal_cutflow_dict(s_hists, 'cutflow')

#m1s = [0.05, 0.5, 5, 50]
#deltas = [0.1, 0.2]
#ctaus = [1]
m1s = [0.05, 0.5, 5, 50]
deltas = [0.1, 0.2]
ctaus = [10]

size = (16, 12)

sel_label = 'prevtx'

def make_scan_plots(plot_dict, style_dict, plot_label, plot_title):
    # signal points
    m1s = [0.5, 1, 2, 5]
    deltas = [0.1]
    ctaus = [10]

    fig, ax = plt.subplots(figsize=size)
    style_dict['fig'] = fig; style_dict['ax'] = ax

    cmap_idx = 0
    for m1 in m1s:
        for delta in deltas:
            for ctau in ctaus:
                ptools.plot_signal_1D(s_hists, m1, delta, ctau, plot_dict, style_dict, cmap_idx = cmap_idx)
                cmap_idx += 1

    plt.title(rf'{plot_title}: $M_1$ = {m1s}, $\Delta$ = {deltas}, c$\tau$ = {ctaus}mm')
    plt.legend()
    plt.savefig(f"{plotdir}/hist_{sel_label}_{plot_label}_delta-{utils.stringfy_friendly(deltas[0])}_ctau-{utils.stringfy_friendly(ctaus[0])}_m1-narrow.png")

    # signal points
    m1s = [0.05, 0.5, 5, 50]
    deltas = [0.1]
    ctaus = [10]

    fig, ax = plt.subplots(figsize=size)
    style_dict['fig'] = fig; style_dict['ax'] = ax

    cmap_idx = 0
    for m1 in m1s:
        for delta in deltas:
            for ctau in ctaus:
                ptools.plot_signal_1D(s_hists, m1, delta, ctau, plot_dict, style_dict, cmap_idx = cmap_idx)
                cmap_idx += 1

    plt.title(rf'{plot_title}: $M_1$ = {m1s}, $\Delta$ = {deltas}, c$\tau$ = {ctaus}mm')
    plt.legend()
    plt.savefig(f"{plotdir}/hist_{sel_label}_{plot_label}_delta-{utils.stringfy_friendly(deltas[0])}_ctau-{utils.stringfy_friendly(ctaus[0])}_m1-wide.png")

    # signal points
    m1s = [5]
    deltas = [0.1]
    ctaus = [1, 10, 100]

    fig, ax = plt.subplots(figsize=size)
    style_dict['fig'] = fig; style_dict['ax'] = ax

    cmap_idx = 0
    for m1 in m1s:
        for delta in deltas:
            for ctau in ctaus:
                ptools.plot_signal_1D(s_hists, m1, delta, ctau, plot_dict, style_dict, cmap_idx = cmap_idx)
                cmap_idx += 1

    plt.title(rf'{plot_title}: $M_1$ = {m1s}, $\Delta$ = {deltas}, c$\tau$ = {ctaus}mm')
    plt.legend()
    plt.savefig(f"{plotdir}/hist_{sel_label}_{plot_label}_m1-{utils.stringfy_friendly(m1s[0])}_delta-{utils.stringfy_friendly(deltas[0])}.png")

    # signal points
    m1s = [5]
    deltas = [0.05, 0.1, 0.2]
    ctaus = [10]

    fig, ax = plt.subplots(figsize=size)
    style_dict['fig'] = fig; style_dict['ax'] = ax

    cmap_idx = 0
    for m1 in m1s:
        for delta in deltas:
            for ctau in ctaus:
                ptools.plot_signal_1D(s_hists, m1, delta, ctau, plot_dict, style_dict, cmap_idx = cmap_idx)
                cmap_idx += 1

    plt.title(rf'{plot_title}: $M_1$ = {m1s}, $\Delta$ = {deltas}, c$\tau$ = {ctaus}mm')
    plt.legend()
    plt.savefig(f"{plotdir}/hist_{sel_label}_{plot_label}_m1-{utils.stringfy_friendly(m1s[0])}_ctau-{utils.stringfy_friendly(ctaus[0])}.png")

# Linear-scale L_xy plots
plot_dict = {
    'variable': ['gen_leading_ele_lxy', 'gen_subleading_ele_lxy'],
    'year': 2024,
    'cut': 'cut1',
}
style_dict = {
    'fig': None, 'ax': None,
    'rebin': 1j, 'xlim': None,     # if None, the default will show up; otherwise give as a list, i.e. [0, 10]
    'doLogy': False, 'doLogx': False, 'doDensity': True, 'doYerr': False,
    'xlabel': None,   # if None, the default will show up; otherwise give as a string, i.e. 'Electron dxy'
    'ylabel': None,   # if None, the default will show up; otherwise give as a string, i.e. 'Efficiency'
    'label': None,    # if None, the default will show up; otherwise give as a string, i.e. 'Highest ctau signal samples'
    'flow': None,     # overflow
    'doSave': False,
    'ls': ['-', '--'],
}
make_scan_plots(plot_dict, style_dict, 'gen-ele-lxy', r'Gen Electron $L_{xy}$')

# Log-scale L_xy plots (1um - 100cm)
log_plot_dict = {
    'variable': ['gen_leading_ele_lxy_log', 'gen_subleading_ele_lxy_log'],
    'year': 2024,
    'cut': 'cut1',
}
log_style_dict = dict(style_dict)
log_style_dict['doLogx'] = True
log_style_dict['doLogy'] = True
make_scan_plots(log_plot_dict, log_style_dict, 'gen-ele-lxy-log', r'Gen Electron $L_{xy}$ (log scale)')

# Flawed L_xy plots: gen ee vertex measured relative to the reco PV instead of the true chi2 production vertex
flawed_plot_dict = {
    'variable': 'gen_diele_lxy_flawed',
    'year': 2024,
    'cut': 'cut1',
}
flawed_style_dict = dict(style_dict)
flawed_style_dict['ls'] = '-'
make_scan_plots(flawed_plot_dict, flawed_style_dict, 'gen-diele-lxy-flawed', r'Gen Diele $L_{xy}$ (flawed, rel. to PV)')

# Flawed L_xy plots (log scale)
flawed_log_plot_dict = {
    'variable': 'gen_diele_lxy_flawed_log',
    'year': 2024,
    'cut': 'cut1',
}
flawed_log_style_dict = dict(flawed_style_dict)
flawed_log_style_dict['doLogx'] = True
flawed_log_style_dict['doLogy'] = True
make_scan_plots(flawed_log_plot_dict, flawed_log_style_dict, 'gen-diele-lxy-flawed-log', r'Gen Diele $L_{xy}$ (flawed, rel. to PV, log scale)')

# |PV - chi2 vertex| (transverse): how far the flawed calculation's PV stand-in actually
# sits from the true chi2 production vertex.
pv_chi2_xy_plot_dict = {
    'variable': 'pv_chi2_voffset_xy',
    'year': 2024,
    'cut': 'cut1',
}
pv_chi2_xy_style_dict = dict(flawed_style_dict)
make_scan_plots(pv_chi2_xy_plot_dict, pv_chi2_xy_style_dict, 'pv-chi2-voffset-xy', r'$|PV - \chi_2\ \mathrm{vtx}|_{xy}$')

# |PV - chi2 vertex| (transverse, log scale)
pv_chi2_xy_log_plot_dict = {
    'variable': 'pv_chi2_voffset_xy_log',
    'year': 2024,
    'cut': 'cut1',
}
pv_chi2_xy_log_style_dict = dict(pv_chi2_xy_style_dict)
pv_chi2_xy_log_style_dict['doLogx'] = True
pv_chi2_xy_log_style_dict['doLogy'] = True
make_scan_plots(pv_chi2_xy_log_plot_dict, pv_chi2_xy_log_style_dict, 'pv-chi2-voffset-xy-log', r'$|PV - \chi_2\ \mathrm{vtx}|_{xy}$ (log scale)')

# |GenEle vertex - chi2 vertex| (transverse): diagnostic for the light-mass-sample issue
# where GenPart's chi2 vx/vy numerically coincides with GenEle's own vertex.
genele_chi2_xy_plot_dict = {
    'variable': 'genele_chi2_voffset_xy',
    'year': 2024,
    'cut': 'cut1',
}
genele_chi2_xy_style_dict = dict(flawed_style_dict)
make_scan_plots(genele_chi2_xy_plot_dict, genele_chi2_xy_style_dict, 'genele-chi2-voffset-xy', r'$|\mathrm{GenEle\ vtx} - \chi_2\ \mathrm{vtx}|_{xy}$')

# |GenEle vertex - chi2 vertex| (transverse, log scale)
genele_chi2_xy_log_plot_dict = {
    'variable': 'genele_chi2_voffset_xy_log',
    'year': 2024,
    'cut': 'cut1',
}
genele_chi2_xy_log_style_dict = dict(genele_chi2_xy_style_dict)
genele_chi2_xy_log_style_dict['doLogx'] = True
genele_chi2_xy_log_style_dict['doLogy'] = True
make_scan_plots(genele_chi2_xy_log_plot_dict, genele_chi2_xy_log_style_dict, 'genele-chi2-voffset-xy-log', r'$|\mathrm{GenEle\ vtx} - \chi_2\ \mathrm{vtx}|_{xy}$ (log scale)')

