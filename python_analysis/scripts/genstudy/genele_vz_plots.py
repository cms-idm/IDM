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
fig, ax = plt.subplots(figsize=size)

# Plot settings
plot_label = 'gen-ele-vz'
sel_label = 'prevtx'
plot_title = r'Gen Electron $v_z$'
plot_dict = {
    'variable': ['gen_leading_ele_vz', 'gen_subleading_ele_vz'],
    'year': 2024,
    'cut': 'cut1',
}

style_dict = {
    'fig': fig, 'ax': ax,
    'rebin': 1j, 'xlim': None,     # if None, the default will show up; otherwise give as a list, i.e. [0, 10]
    'doLogy': False, 'doLogx': False, 'doDensity': True, 'doYerr': False, 
    'xlabel': None,   # if None, the default will show up; otherwise give as a string, i.e. 'Electron dxy'
    'ylabel': None,   # if None, the default will show up; otherwise give as a string, i.e. 'Efficiency'
    'label': None,    # if None, the default will show up; otherwise give as a string, i.e. 'Highest ctau signal samples'
    'flow': None,     # overflow
    'doSave': False,
    'ls': ['-', '--'],
}

# signal points
m1s = [0.5, 1, 2, 5]
deltas = [0.1]
ctaus = [10]

# Plot for variables signal points
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

# Plot for variables signal points
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

# Plot for variables signal points
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

# Plot for variables signal points
cmap_idx = 0
for m1 in m1s:
    for delta in deltas:
        for ctau in ctaus:
            ptools.plot_signal_1D(s_hists, m1, delta, ctau, plot_dict, style_dict, cmap_idx = cmap_idx)
            cmap_idx +=	1

plt.title(rf'{plot_title}: $M_1$ = {m1s}, $\Delta$ = {deltas}, c$\tau$ = {ctaus}mm')
plt.legend()
plt.savefig(f"{plotdir}/hist_{sel_label}_{plot_label}_m1-{utils.stringfy_friendly(m1s[0])}_ctau-{utils.stringfy_friendly(ctaus[0])}.png")

def make_single_plot(variable, plot_label, plot_title, doLogx=False):
    single_plot_dict = {
        'variable': variable,
        'year': 2024,
        'cut': 'cut1',
    }
    single_style_dict = dict(style_dict)
    single_style_dict['ls'] = '-'
    single_style_dict['doLogx'] = doLogx
    single_style_dict['doLogy'] = doLogx

    fig, ax = plt.subplots(figsize=size)
    single_style_dict['fig'] = fig; single_style_dict['ax'] = ax

    m1s = [0.5, 1, 2, 5]
    deltas = [0.1]
    ctaus = [10]

    cmap_idx = 0
    for m1 in m1s:
        for delta in deltas:
            for ctau in ctaus:
                ptools.plot_signal_1D(s_hists, m1, delta, ctau, single_plot_dict, single_style_dict, cmap_idx = cmap_idx)
                cmap_idx += 1

    plt.title(rf'{plot_title}: $M_1$ = {m1s}, $\Delta$ = {deltas}, c$\tau$ = {ctaus}mm')
    plt.legend()
    plt.savefig(f"{plotdir}/hist_{sel_label}_{plot_label}_delta-{utils.stringfy_friendly(deltas[0])}_ctau-{utils.stringfy_friendly(ctaus[0])}.png")

# |PV - chi2 vertex| (z): how far the flawed lxy calculation's PV stand-in actually
# sits from the true chi2 production vertex, along z.
make_single_plot('pv_chi2_voffset_z',     'pv-chi2-voffset-z',     r'$|PV - \chi_2\ \mathrm{vtx}|_{z}$')
make_single_plot('pv_chi2_voffset_z_log', 'pv-chi2-voffset-z-log', r'$|PV - \chi_2\ \mathrm{vtx}|_{z}$ (log scale)', doLogx=True)

# |GenEle vertex - chi2 vertex| (z): diagnostic for the light-mass-sample issue where
# GenPart's chi2 vz numerically coincides with GenEle's own vertex.
make_single_plot('genele_chi2_voffset_z',     'genele-chi2-voffset-z',     r'$|\mathrm{GenEle\ vtx} - \chi_2\ \mathrm{vtx}|_{z}$')
make_single_plot('genele_chi2_voffset_z_log', 'genele-chi2-voffset-z-log', r'$|\mathrm{GenEle\ vtx} - \chi_2\ \mathrm{vtx}|_{z}$ (log scale)', doLogx=True)

