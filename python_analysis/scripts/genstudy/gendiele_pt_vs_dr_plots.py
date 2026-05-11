# imports
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak

import sys
#sys.path.append("../../analysisTools/")
from tools.analysisTools import Analyzer
from tools.analysisTools import loadSchema
import tools.analysisTools as tools
import tools.analysisSubroutines as routines
import importlib
import coffea.util as util
import tools.utils as utils
import tools.plotTools as ptools
import time
import json
import os
import glob

#cuts_config = "configs/selections/minimal_cuts.py"
#hists_config = "configs/hists/genstudy.py"
#sample_config = "configs/samples/signal_2024_Apr2026_aEM.json"
outdir = 'workarea'
saved_signal_hists = f"{outdir}/step1_genstudy_signal_minimalselection.coffea"

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

size = (16, 12)

# Plot settings
plot_label = 'gen-diele-pt-vs-dr'
sel_label = 'prevtx'
plot_title = r'Gen EE $p_T$ vs $\Delta R$'
plot_dict = {
    'variable': 'gen_diele_pt_vs_dr',
    'year': 2024,
    'cut': 'cut5',
}

style_dict = {
    'xrebin': 2j, 'yrebin': 2j, 'xlim': None, 'ylim': None,    # if None, the default will show up; otherwise give as a list, i.e. [0, 10]
    'doLogy': False, 'doLogx': False, 'doLogz': False,  
    'xlabel': None,   # if None, the default will show up; otherwise give as a string, i.e. 'Electron dxy'
    'ylabel': None,   # if None, the default will show up; otherwise give as a string, i.e. 'Efficiency'
    'zlabel': None,
    'label': None,    # if None, the default will show up; otherwise give as a string, i.e. 'Highest ctau signal samples'
    'flow': None,     # overflow
    'doSave': False,
}

# signal points
m1s = [0.05, 0.5, 1, 2, 5, 50]
deltas = [0.05, 0.1, 0.2]
ctaus = [1, 10, 100]

# Plot for variables signal points
for m1 in m1s:
    for delta in deltas:
        for ctau in ctaus:
            fig, ax = plt.subplots(figsize=size)
            style_dict['fig'] = fig; style_dict['ax'] = ax
            ptools.plot_signal_2D(s_hists, m1, delta, ctau, plot_dict, style_dict)

            plt.title(rf'{plot_title}: $M_1$ = {m1}, $\Delta$ = {delta}, c$\tau$ = {ctau}mm')
            #plt.show()
            sm1 = utils.stringfy_friendly(m1)
            sdm = utils.stringfy_friendly(delta)
            sct = utils.stringfy_friendly(ctau)
            plt.savefig(f"plots/pt_vs_dr_2D/hist_{sel_label}_{plot_label}_m1-{sm1}_delta-{sdm}_ctau-{sct}.png")
            plt.close()
