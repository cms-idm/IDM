# imports
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak

import sys
#sys.path.append("../../analysisTools/")
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

#cuts_config = "configs/selections/minimal_cuts.py"
#hists_config = "configs/hists/genstudy.py"
#sample_config = "configs/samples/signal_2024_May2026_aEM.json"
outdir = 'workarea'
vers = 'Jun2026noID'
selection = 'anmatchvtx'
hists = 'mergedmatch'
saved_signal_hists = f"{outdir}/hists_sig{vers}_{selection}-sel_{hists}.coffea"
plottag = f'sig{vers}_{selection}-sel'

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
df_wts = utils.get_signal_cutflow_dict(s_hists, 'cutflow_wgts2')

#m1s = [0.05, 0.5, 5, 50]
#deltas = [0.1, 0.2]
#ctaus = [1]
m1s = [0.05, 0.5, 5, 50]
deltas = [0.1, 0.2]
ctaus = [10]

plot_dict_sig_eff = {
    
    # Select signal points to display
    'm1s': m1s,
    'deltas': deltas,
    'ctaus': ctaus,

    # Plot display styling
    'ylim': [1e-5, 1], # None for default
    'doLog': True,
    
    'ylabel': 'Efficiency', # None for default
    'title': rf"Signal Cutflow [AN Selection]: $M_1$ = {m1s}, $\Delta$ = {deltas}, c$\tau$ = {ctaus}mm",
    'label': None,

    # Plot saving
    'doSave': True,
    'outDir': './plots/',
    'outName': f'cutflow/cutflow_sigMay2026_an-sel_ctau-{utils.stringfy_friendly(ctaus[0])}.png'
    }

m1s = [0.05, 0.5, 5, 50]
deltas = [0.1]
ctaus = [10]

plot_dict_sig_eff['m1s'] = m1s; plot_dict_sig_eff['deltas'] = deltas; plot_dict_sig_eff['ctaus'] = ctaus
plot_dict_sig_eff['title'] = rf"Signal Cutflow [AN Selection]: $M_1$ = {m1s}, $\Delta$ = {deltas}, c$\tau$ = {ctaus}mm"
plot_dict_sig_eff['outName'] = f'cutflow/cutflow_{plottag}_ctau-{utils.stringfy_friendly(ctaus[0])}_delta-{utils.stringfy_friendly(deltas[0])}_m1-wide.png'

ptools.plot_signal_efficiency(s_hists, df, plot_dict_sig_eff, df_wts=df_wts)

m1s = [0.5, 1, 2, 5]
deltas = [0.1]
ctaus = [10]

plot_dict_sig_eff['m1s'] = m1s; plot_dict_sig_eff['deltas'] = deltas; plot_dict_sig_eff['ctaus'] = ctaus
plot_dict_sig_eff['title'] = rf"Signal Cutflow [AN Selection]: $M_1$ = {m1s}, $\Delta$ = {deltas}, c$\tau$ = {ctaus}mm"
plot_dict_sig_eff['outName'] = f'cutflow/cutflow_{plottag}_ctau-{utils.stringfy_friendly(ctaus[0])}_delta-{utils.stringfy_friendly(deltas[0])}_m1-narrow.png'

ptools.plot_signal_efficiency(s_hists, df, plot_dict_sig_eff, df_wts=df_wts)

m1s = [0.5]
deltas = [0.1]
ctaus = [1, 10, 100]

plot_dict_sig_eff['m1s'] = m1s; plot_dict_sig_eff['deltas'] = deltas; plot_dict_sig_eff['ctaus'] = ctaus
plot_dict_sig_eff['title'] = rf"Signal Cutflow [AN Selection]: $M_1$ = {m1s}, $\Delta$ = {deltas}, c$\tau$ = {ctaus}mm"
plot_dict_sig_eff['outName'] = f'cutflow/cutflow_{plottag}_m1-{utils.stringfy_friendly(m1s[0])}_delta-{utils.stringfy_friendly(deltas[0])}.png'

ptools.plot_signal_efficiency(s_hists, df, plot_dict_sig_eff, df_wts=df_wts)

m1s = [0.5]
deltas = [0.05, 0.1, 0.2]
ctaus = [10]

plot_dict_sig_eff['m1s'] = m1s; plot_dict_sig_eff['deltas'] = deltas; plot_dict_sig_eff['ctaus'] = ctaus
plot_dict_sig_eff['title'] = rf"Signal Cutflow [AN Selection]: $M_1$ = {m1s}, $\Delta$ = {deltas}, c$\tau$ = {ctaus}mm"
plot_dict_sig_eff['outName'] = f'cutflow/cutflow_{plottag}_m1-{utils.stringfy_friendly(m1s[0])}_ctau-{utils.stringfy_friendly(ctaus[0])}.png'

ptools.plot_signal_efficiency(s_hists, df, plot_dict_sig_eff, df_wts=df_wts)

m1s = [5]
deltas = [0.1]
ctaus = [1, 10, 100]

plot_dict_sig_eff['m1s'] = m1s; plot_dict_sig_eff['deltas'] = deltas; plot_dict_sig_eff['ctaus'] = ctaus
plot_dict_sig_eff['title'] = rf"Signal Cutflow [AN Selection]: $M_1$ = {m1s}, $\Delta$ = {deltas}, c$\tau$ = {ctaus}mm"
plot_dict_sig_eff['outName'] = f'cutflow/cutflow_{plottag}_m1-{utils.stringfy_friendly(m1s[0])}_delta-{utils.stringfy_friendly(deltas[0])}.png'

ptools.plot_signal_efficiency(s_hists, df, plot_dict_sig_eff, df_wts=df_wts)

m1s = [5]
deltas = [0.05, 0.1, 0.2]
ctaus = [10]

plot_dict_sig_eff['m1s'] = m1s; plot_dict_sig_eff['deltas'] = deltas; plot_dict_sig_eff['ctaus'] = ctaus
plot_dict_sig_eff['title'] = rf"Signal Cutflow [AN Selection]: $M_1$ = {m1s}, $\Delta$ = {deltas}, c$\tau$ = {ctaus}mm"
plot_dict_sig_eff['outName'] = f'cutflow/cutflow_{plottag}_m1-{utils.stringfy_friendly(m1s[0])}_ctau-{utils.stringfy_friendly(ctaus[0])}.png'

ptools.plot_signal_efficiency(s_hists, df, plot_dict_sig_eff, df_wts=df_wts)



