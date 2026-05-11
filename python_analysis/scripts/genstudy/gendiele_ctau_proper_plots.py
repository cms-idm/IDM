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

import numpy as np
from scipy.optimize import curve_fit

def exponential(x, A, lam):
    return A * np.exp(-lam * x)

def fit_and_plot_exponential(histo, ax, color, density=False, label=None):
    """
    Fit an exponential to a histogram and overlay it on the given axes.
    χ²/dof is computed from bin errors via histo.variances() and shown in the legend.

    Parameters
    ----------
    histo   : boost_histogram / hist Histogram object
    ax      : matplotlib Axes to draw on
    color   : line color (should match the histogram)
    density : whether the histogram was plotted with density=True
    label   : optional override for the legend label
    """
    edges    = histo.axes[0].edges
    counts   = histo.values()
    variances = histo.variances()

    # Bin centers; mask out empty bins (zero count or zero variance)
    centers  = 0.5 * (edges[:-1] + edges[1:])
    mask     = (counts > 0) & (variances > 0)
    x, y     = centers[mask], counts[mask]
    yerr     = np.sqrt(variances[mask])
    p0 = [y.max(), 1.0 / np.average(x, weights=y)]

    if len(x) < len(p0) + 1:  # need at least nparams+1 points for dof > 0
        print(f"Skipping fit: only {len(x)} valid bin(s) after masking")
        return None, None, None
    
    if density:
        norm = (y * np.diff(edges)[mask]).sum()
        y    = y / norm
        yerr = yerr / norm

    try:
        p0 = [y.max(), 1.0 / np.average(x, weights=y)]
        popt, pcov = curve_fit(exponential, x, y, p0=p0, sigma=yerr,
                               absolute_sigma=True, maxfev=5000)
        A, lam = popt
        perr   = np.sqrt(np.diag(pcov))

        x_fit = np.linspace(x.min(), x.max(), 300)
        y_fit = exponential(x_fit, *popt)

        # χ²/dof: residuals weighted by bin errors, dof = nbins - nparams
        y_to_check = y[y > 0]
        x_to_check = x[y > 0]
        yerr_to_check = yerr[y > 0]
        residuals = (y_to_check - exponential(x_to_check, *popt)) / yerr_to_check
        chi2      = np.sum(residuals**2)
        dof       = len(x_to_check) - len(popt) 
        chi2_dof  = chi2 / dof if dof > 0 else np.inf

        fit_label = label or (
            rf"exp fit: $c\tau={1/lam:.3f} \pm {perr[1]/lam**2:.3f}$"
            rf", $\chi^2/\mathrm{{dof}}={chi2_dof:.2f}$"
        )
        ax.plot(x_fit, y_fit, color=color, linestyle='--', linewidth=1.5, label=fit_label)
        return popt, perr, chi2_dof

    except RuntimeError as e:
        print(f"Fit failed: {e}")
        return None, None, None

def set_log_ylimits(ax, histo_list, padding_factor=3.0, density=False):
    """
    Set y-axis limits for a log-scale histogram plot based on the actual
    histogram contents, ignoring where the fit curves extend to.

    Parameters
    ----------
    ax             : matplotlib Axes
    histo_list     : list of histogram objects that were plotted
    padding_factor : multiplier for top/bottom padding in log space
    """
    all_counts = []
    for histo in histo_list:
        counts    = histo.values()
        variances = histo.variances()
        edges     = histo.axes[0].edges
        bin_widths = np.diff(edges)
        if variances is not None:
            errs = np.sqrt(np.where(variances > 0, variances, 0))
        else:
            errs = np.sqrt(np.where(counts > 0, counts, 0))
            
        if density:
            norm   = (counts[counts > 0] * bin_widths[counts > 0]).sum()
            counts = counts / norm
            errs   = errs   / norm

        # Lower edge of error bars for filled bins
        lower = counts - errs
        upper = counts + errs
        all_counts.extend(upper[counts > 0].tolist())
        all_counts.extend(lower[lower > 0].tolist())  # only positive (log scale)

    if not all_counts:
        return

    ymin = min(all_counts)
    ymax = max(all_counts)

    # Pad in log space
    ax.set_ylim(
        ymin / padding_factor,
        ymax * padding_factor
    )

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
fig, ax = plt.subplots(figsize=size)

# Plot settings
plot_label = 'gen-chi2-ctau-proper'
sel_label = 'prevtx'
plot_title = r'Gen $\chi_2$ Proper $c\tau$'
plot_dict = {
    'variable': ['gen_chi2_ctau_proper'],
    'year': 2024,
    'cut': 'cut5',
}

style_dict = {
    'fig': fig, 'ax': ax,
    'rebin': 1j, 'xlim': None,     # if None, the default will show up; otherwise give as a list, i.e. [0, 10]
    'doLogy': True, 'doLogx': False, 'doDensity': True, 'doYerr': True, 
    'xlabel': None,   # if None, the default will show up; otherwise give as a string, i.e. 'Electron dxy'
    'ylabel': None,   # if None, the default will show up; otherwise give as a string, i.e. 'Efficiency'
    'label': None,    # if None, the default will show up; otherwise give as a string, i.e. 'Highest ctau signal samples'
    'flow': None,     # overflow
    'doSave': False,
    'ls': ['-'],
}

# signal points
m1s = [1, 2, 5]
deltas = [0.1, 0.2]
ctaus = [10]

# Plot for variables signal points
hlist = []
cmap_idx = 0
for m1 in m1s:
    for delta in deltas:
        for ctau in ctaus:
            histo, color = ptools.plot_signal_1D(s_hists, m1, delta, ctau, plot_dict, style_dict, cmap_idx = cmap_idx)
            hlist += histo
            for ih, h in enumerate(histo):
                fit_and_plot_exponential(
                    h, ax,
                    color=color,
                    density=style_dict['doDensity'],
                )
            cmap_idx += 1

plt.title(rf'{plot_title}: $M_1$ = {m1s}, $\Delta$ = {deltas}, c$\tau$ = {ctaus}mm')
set_log_ylimits(ax, hlist, density=style_dict['doDensity'])
plt.legend()
plt.savefig(f"plots/genstudy/hist_{sel_label}_{plot_label}_ctau-{utils.stringfy_friendly(ctaus[0])}_m1-narrow.png")

# signal points
m1s = [0.05, 0.5, 5]
deltas = [0.1, 0.2]
ctaus = [10]

fig, ax = plt.subplots(figsize=size)
style_dict['fig'] = fig; style_dict['ax'] = ax

# Plot for variables signal points
cmap_idx = 0
hlist = []
for m1 in m1s:
    for delta in deltas:
        for ctau in ctaus:
            histo, color = ptools.plot_signal_1D(s_hists, m1, delta, ctau, plot_dict, style_dict, cmap_idx = cmap_idx)
            hlist += histo
            for ih, h in enumerate(histo):
                fit_and_plot_exponential(
                    h, ax,
                    color=color,
                    density=style_dict['doDensity'],
                )
            cmap_idx += 1

plt.title(rf'{plot_title}: $M_1$ = {m1s}, $\Delta$ = {deltas}, c$\tau$ = {ctaus}mm')
set_log_ylimits(ax, hlist, density=style_dict['doDensity'])
plt.legend()
plt.savefig(f"plots/genstudy/hist_{sel_label}_{plot_label}_ctau-{utils.stringfy_friendly(ctaus[0])}_m1-wide.png")

# signal points
m1s = [1, 2, 5]
deltas = [0.1]
ctaus = [1, 10, 100]

fig, ax = plt.subplots(figsize=size)
style_dict['fig'] = fig; style_dict['ax'] = ax

# Plot for variables signal points
cmap_idx = 0
hlist = []
for m1 in m1s:
    for delta in deltas:
        for ctau in ctaus:
            histo, color = ptools.plot_signal_1D(s_hists, m1, delta, ctau, plot_dict, style_dict, cmap_idx = cmap_idx)
            hlist += histo
            for ih, h in enumerate(histo):
                popt, perr, chi2_dof = fit_and_plot_exponential(
                    h, ax,
                    color=color,
                    density=style_dict['doDensity'],
                )
            try:
                print('m1', m1, 'delta', delta, 'ctau', ctau, 'fitted ctau', round(1/popt[1],5), 'pm', round(perr[1]/popt[1]**2,5), 'x2/dof', round(chi2_dof,3))
            except:
                pass
            cmap_idx += 1

plt.title(rf'{plot_title}: $M_1$ = {m1s}, $\Delta$ = {deltas}, c$\tau$ = {ctaus}mm')
set_log_ylimits(ax, hlist, density=style_dict['doDensity'])
plt.legend()
plt.savefig(f"plots/genstudy/hist_{sel_label}_{plot_label}_delta-{utils.stringfy_friendly(deltas[0])}.png")

# signal points
m1s = [5]
deltas = [0.05, 0.1, 0.2]
ctaus = [1, 10, 100]

fig, ax = plt.subplots(figsize=size)
style_dict['fig'] = fig; style_dict['ax'] = ax

# Plot for variables signal points
cmap_idx = 0
hlist = []
for m1 in m1s:
    for delta in deltas:
        for ctau in ctaus:
            histo, color = ptools.plot_signal_1D(s_hists, m1, delta, ctau, plot_dict, style_dict, cmap_idx = cmap_idx)
            hlist += histo
            for ih, h in enumerate(histo):
                fit_and_plot_exponential(
                    h, ax,
                    color=color,
                    density=style_dict['doDensity'],
                )
            cmap_idx +=	1

plt.title(rf'{plot_title}: $M_1$ = {m1s}, $\Delta$ = {deltas}, c$\tau$ = {ctaus}mm')
set_log_ylimits(ax, hlist, density=style_dict['doDensity'])
plt.legend()
plt.savefig(f"plots/genstudy/hist_{sel_label}_{plot_label}_m1-{utils.stringfy_friendly(m1s[0])}.png")

