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
from hist import Hist
from configs.histo_configs.histobins import ele_pt
from hist.storage import Weight
from hist.axis import Regular, Variable
import copy
import mplhep as hep

#cuts_config = "configs/selections/minimal_cuts.py"
#hists_config = "configs/hists/genstudy.py"
#sample_config = "configs/samples/signal_2024_Apr2026_aEM.json"
outdir = 'workarea'
saved_signal_hists = f"{outdir}/hists_sigMay2026_an-sel_elerecoeff.coffea"

title = 'Electron Reco'
plottag = 'prevtx_ele-reco_pt-lxy'
var = '[$p_T$, $v_{xy}$]'

# Signal
s_hists = util.load(saved_signal_hists)[0]
s_pts = utils.get_signal_point_dict(s_hists)
s_histnames = utils.get_signal_list_of_histograms(s_hists)
s_cutsidx = utils.get_signal_list_of_cuts(s_hists, get_cut_idx = True)
s_cutsname = utils.get_signal_list_of_cuts(s_hists, get_cut_idx = False)

df = utils.get_signal_cutflow_dict(s_hists, 'cutflow')
size = (16, 12)
fig, ax = plt.subplots(figsize=size)

histvars = [f'ele_reco_lpt_pt_lxy', f'ele_reco_ged_pt_lxy', f'ele_reco_none_pt_lxy']

# Plot settings
plot_dict = {
    'variable': None, 
    'year': 2024,
    'cut': 'cut8',
}

style_dict = {
    'fig': fig, 'ax': ax,
    'xrebin': 1j, 'yrebin': 1j, 'xlim': None,  'ylim': None,   # if None, the default will show up; otherwise give as a list, i.e. [0, 10]
    'doLogz': False, 'doLogy': False, 'doLogx': False, 
    'xlabel': None,   # if None, the default will show up; otherwise give as a string, i.e. 'Electron dxy'
    'ylabel': None,   # if None, the default will show up; otherwise give as a string, i.e. 'Efficiency'
    'zlabel': None,   # if None, the default will show up; otherwise give as a string, i.e. 'Efficiency'
    'label': None,    # if None, the default will show up; otherwise give as a string, i.e. 'Highest ctau signal samples'
    'flow': None,     # overflow
    'doSave': False,
}

def ratio_hist(h_num, h_den):
    """
    Returns a new Hist object representing h_num / (h_den1 + h_den2)
    with correct variance propagation.
    All histograms must have compatible axes.
    """
    num = h_num.values()
    den = h_den.values()
    num_var = h_num.variances()
    den_var = h_den.variances()

    with np.errstate(invalid='ignore', divide='ignore'):
        ratio = np.where(den > 0, num / den, 0)
        ratio_var = np.where(
            den > 0,
            ratio**2 * (num_var / np.where(num > 0, num**2, 1)
                      + den_var / np.where(den > 0, den**2, 1)),
            0
        )

    # create new hist with same axis and Weight storage (to hold variances)
    h_ratio = Hist(copy.deepcopy(h_num.axes[0]), storage=Weight())
    h_ratio.view(flow=False)[...] = np.stack([ratio, ratio_var], axis=-1)

    return h_ratio

# signal points
m1s = [0.05, 0.5, 1, 2, 5, 50]
deltas = [0.05, 0.1, 0.2]
ctaus = [1, 10, 100]

sumcounts = [None, None, None]
sumvarias = [None, None, None]
edges0 = None
edges1 = None

# Plot for variables signal points
for m1 in m1s:
    for delta in deltas:
        for ctau in ctaus:
            #hists, _ = ptools.plot_signal_1D(s_hists, m1, delta, ctau, plot_dict, style_dict, cmap_idx = 0)
            for iv, hv in enumerate(histvars):
                plot_dict['variable'] = v
                counts, variances, edges0, edges1 = ptools.plot_signal_2D(s_hists, m1, delta, ctau, plot_dict, style_dict)

                if sumcounts[iv] is None:
                    sumcounts[iv] = counts
                    sumvarias[iv] = variances
                else:
                    sumcounts[iv] += counts
                    sumvarias[iv] += variances

plt.close(fig)

for iv, hv in enumerate(histvars):
    fig, ax = plt.subplots(figsize=size)
    hep.cms.label('Private Work', data=True, year=plot_dict['year'], com='13.6')
    
    h_new = hist.Hist(
        #hist.axis.Variable(new_bins, name=h.axes[0].name),
        #hist.axis.Variable(new_bins, name=h.axes[1].name),
        Regular(500,0,25,name="pt",label=r"$p_{T}$ [GeV]"),
        Regular(500,0,25,name="vxy",label=r"$v_{xy}$ [cm]"), 
        storage=hist.storage.Weight(),
    )
    h_new.view()["value"][:] = sumcounts[iv]
    h_new.view()["variance"][:] = sumvarias[iv]
    
    if style_dict['doLogz']:
        hep.hist2dplot(h_new, flow=style_dict['flow'], norm=mpl.colors.LogNorm(), ax=ax, cbarextend=True)
    else:
        hep.hist2dplot(h_new, flow=style_dict['flow'], ax=ax, cbarextend=True)

    xbinwidth = h_new.axes.widths[0][0]
    xbinwidth = h_new.axes.widths[y][0]
    ax.set_zlabel(f'Events/{xbinwidth:.3f}/{ybinwidth:.3f}')
    ax.set_zscale('log')
    plt.title(rf'{title}: {var}')
    plt.legend()
    plt.savefig(f"plots/hist_{plottag}_hist_.png")
    plt.close(fig)

# find lpt eff
h_lpt_eff = ratio_hist(sumhists[0], sumhists[0] + sumhists[1] + sumhists[2])
h_ged_eff = ratio_hist(sumhists[1], sumhists[0] + sumhists[1] + sumhists[2])
h_both_eff = ratio_hist(sumhists[0] + sumhists[1], sumhists[0] + sumhists[1] + sumhists[2])
h_none_eff = ratio_hist(sumhists[2], sumhists[0] + sumhists[1] + sumhists[2])
effhists = [h_lpt_eff, h_ged_eff, h_both_eff, h_none_eff]

fig, ax = plt.subplots(figsize=size)
hep.cms.label('Private Work', data=True, year=plot_dict['year'], com='13.6')
hep.histplot(effhists, yerr=[np.sqrt(h.variances()) for h in effhists], density=style_dict['doDensity'], ax=ax,
             histtype='step', flow=style_dict['flow'], label = ['Lpt', 'GED', 'Both', 'Neither'])
binwidth = sumhists[0].axes.widths[0][0]
if style_dict['doDensity']:
    ax.set_ylabel(f'A.U./{binwidth:.3f}')
else:
    ax.set_ylabel(f'Events/{binwidth:.3f}')
ax.set_ylim(-0.05, 1.05)
plt.title(rf'{title} Efficiency by {var}')
plt.legend()
plt.savefig(f"plots/hist_{plottag}_ratio_allpts.png")
plt.close(fig)
