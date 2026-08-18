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
import mplhep as hep
import analysisTools.utils as utils
import analysisTools.plotTools as ptools
import time
import json
import glob

#cuts_config = "configs/selections/minimal_cuts.py"
#hists_config = "configs/hists/genstudy.py"
#sample_config = "configs/samples/signal_2024_Apr2026_aEM.json"
outdir = os.path.join(REPO_ROOT, 'workarea')
plotdir = os.path.join(REPO_ROOT, 'plots', 'recoele')
os.makedirs(plotdir, exist_ok=True)
saved_signal_hists = f"{outdir}/hists_sigJul2026noID_anmatchvtx-sel_recoeles.coffea"

title = 'Reco Lpt Electron $p_T$'
plottag = 'prevtx_reco-lpt-ele-pt'

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
plot_dict = {
    'variable': ['reco_ele_lpt_leading_pt', 'reco_ele_lpt_subleading_pt'],
    'year': 2024,
    'cut': 'cut8',
}

style_dict = {
    'fig': fig, 'ax': ax,
    'rebin': 1j, 'xlim': None,     # if None, the default will show up; otherwise give as a list, i.e. [0, 10]
    'doLogy': False, 'doLogx': False, 'doDensity': False, 'doYerr': False, 
    'xlabel': None,   # if None, the default will show up; otherwise give as a string, i.e. 'Electron dxy'
    'ylabel': None,   # if None, the default will show up; otherwise give as a string, i.e. 'Efficiency'
    'label': None,    # if None, the default will show up; otherwise give as a string, i.e. 'Highest ctau signal samples'
    'flow': None,     # overflow
    'doSave': False,
    'ls': ['-', '--'],
}

# signal points
m1s = [1, 2, 5]
deltas = [0.1]
ctaus = [10]

# Plot for variables signal points
cmap_idx = 0
for m1 in m1s:
    for delta in deltas:
        for ctau in ctaus:
            ptools.plot_signal_1D(s_hists, m1, delta, ctau, plot_dict, style_dict, cmap_idx = cmap_idx)
            cmap_idx += 1

plt.title(rf'{title}: $M_1$ = {m1s}, $\Delta$ = {deltas}, c$\tau$ = {ctaus}mm')
plt.legend()
plt.savefig(f"{plotdir}/hist_{plottag}_delta-{utils.stringfy_friendly(deltas[0])}_ctau-{utils.stringfy_friendly(ctaus[0])}_m1-narrow.png")

# signal points
m1s = [0.05, 0.5, 5]
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

plt.title(rf'{title}: $M_1$ = {m1s}, $\Delta$ = {deltas}, c$\tau$ = {ctaus}mm')
plt.legend()
plt.savefig(f"{plotdir}/hist_{plottag}_delta-{utils.stringfy_friendly(deltas[0])}_ctau-{utils.stringfy_friendly(ctaus[0])}_m1-wide.png")

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

plt.title(rf'{title}: $M_1$ = {m1s}, $\Delta$ = {deltas}, c$\tau$ = {ctaus}mm')
plt.legend()
plt.savefig(f"{plotdir}/hist_{plottag}_m1-{utils.stringfy_friendly(m1s[0])}_delta-{utils.stringfy_friendly(deltas[0])}.png")

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

plt.title(rf'{title}: $M_1$ = {m1s}, $\Delta$ = {deltas}, c$\tau$ = {ctaus}mm')
plt.legend()
plt.savefig(f"{plotdir}/hist_{plottag}_m1-{utils.stringfy_friendly(m1s[0])}_ctau-{utils.stringfy_friendly(ctaus[0])}.png")

# All signal points summed
fig, ax = plt.subplots(figsize=size)
style_dict['fig'] = fig; style_dict['ax'] = ax

histo_lead_sum = None
histo_sublead_sum = None
for _, row in s_pts.iterrows():
    samp = row['name']
    h_lead = s_hists['reco_ele_lpt_leading_pt'][{"samp": samp, "cut": plot_dict['cut']}]
    h_sub  = s_hists['reco_ele_lpt_subleading_pt'][{"samp": samp, "cut": plot_dict['cut']}]
    histo_lead_sum    = h_lead if histo_lead_sum    is None else histo_lead_sum    + h_lead
    histo_sublead_sum = h_sub  if histo_sublead_sum is None else histo_sublead_sum + h_sub

hep.cms.label('Private Work', data=True, year=plot_dict['year'], com='13.6')
hep.histplot([histo_lead_sum, histo_sublead_sum], ax=ax, histtype='step',
             label=[r'Leading $e$', r'Subleading $e$'], ls=['-', '--'], yerr=True)
binwidth = histo_lead_sum.axes.widths[0][0]
ax.set_ylabel(f'Events/{binwidth:.3f}')
plt.title(rf'{title}: All Signal Points')
plt.legend()
plt.savefig(f"{plotdir}/hist_{plottag}_allpoints.png")

