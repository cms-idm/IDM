# imports
import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt

from hist import Hist
from hist.storage import Weight

import coffea.util as util

# Make the script runnable regardless of the caller's current working
# directory by anchoring paths to the repo root (two levels up from this file).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

outdir = os.path.join(REPO_ROOT, 'workarea')
plotdir = os.path.join(REPO_ROOT, 'plots', 'elerecoeff')
os.makedirs(plotdir, exist_ok=True)

sig_vers = 'Jul2026noID'
selection = 'anabrv'
hists = 'elerecoeff'
saved_signal_hists = f'{outdir}/hists_sig{sig_vers}_{selection}-sel_{hists}.coffea'
plottag = f'sig{sig_vers}_{selection}-sel'

# Cut stages to plot at -- see configs/cut_configs/anmatchvtx_selection.py.
CUTS = {
    'cut1': 'inclusive',
    #'cut3': 'preselected',
}

# 4D per-electron hists filled in configs/histo_configs/elerecoeff.py, axes (samp, cut, pt, lxy, dr, abseta).
# lpt/ged/none partition the gen electrons by matchType, so their sum is the denominator.
# Numerators: plot tag -> (list of ee4d categories summed, title)
NUMS_4D = {
    'lpt':  (['lpt'],        'Lpt Electron Reco Efficiency'),
    'ged':  (['ged'],        'GED Electron Reco Efficiency'),
    'sum': (['lpt', 'ged'], 'Lpt + GED Electron Reco Efficiency'),
    'alpt': (['alpt'],       'AllLowPt Electron Reco Efficiency'),
    'xlpt': (['xlpt'],       'xLowPt Electron Reco Efficiency'),
    'sumnoalpt': (['sumnoalpt'], 'Lpt + GED, not AllLowPt Electron Fraction'),
    'alptnosum': (['alptnosum'], 'AllLowPt, not Lpt + GED Electron Fraction'),
}
DEN_4D = ['lpt', 'ged', 'none']

# single signal point (m1 [GeV], delta, ctau [mm]) to also make the maps for alone
SINGLE_POINT = (50, 0.2, 100)


def ee4d(sig_histo, c):
    return sig_histo[f'ele_reco_{c}_ee4d']


def cut_slice(h, cut):
    """
    Slice hist `h` to `cut` on its cut axis. The cut (and samp) axes only grow
    when filled, so a hist never filled at `cut` returns zeros with the same
    samp/var binning instead of KeyError-ing.
    """
    if cut in list(h.axes["cut"]):
        return h[{"cut": cut}]
    return Hist(*[ax for ax in h.axes if ax.name != "cut"], storage=Weight())


def samp_slice(h, s):
    """
    Slice hist `h` to sample `s`. The samp axis only grows when a sample is
    filled, so a sample with no entries in `h` returns an empty hist instead.
    """
    if s in list(h.axes["samp"]):
        return h[{"samp": s}]
    return h[{"samp": sum}] * 0


def n_eff(h):
    """
    Per-bin effective raw MC event count: N_eff = sum(w)^2 / sum(w^2), i.e.
    vals^2/variances -- see vtxreco_plots.py.
    """
    vals, var = h.values(), h.variances()
    return np.where(var > 0, vals**2 / np.where(var > 0, var, 1), 0)


def combine_neff(sig_histo, cats, cut, samples):
    """
    Per-bin N_eff of the sum of ee4d categories `cats` at `cut`, summed over
    signal samples (so the combination isn't dominated by the highest-xsec
    mass point). Returns an array of shape (n_pt, n_lxy, n_dr, n_abseta).
    """
    tot = 0
    for s in samples:
        h = sum(samp_slice(cut_slice(ee4d(sig_histo, c), cut), s) for c in cats)
        tot = tot + n_eff(h)
    return tot


def sample_m1(samp):
    """
    Parse m1 = Mchi - dMchi/2 (the lighter mass eigenstate) from a signal
    sample name like 'sig_2018_Mchi-105p0_dMchi-10p0_ctau-1' -- see vtxreco_plots.py.
    """
    mchi, dmchi = re.search(r'Mchi-([\d]+p[\d]+)_dMchi-([\d]+p[\d]+)', samp).groups()
    return float(mchi.replace('p', '.')) - float(dmchi.replace('p', '.')) / 2


def sample_point(samp):
    """
    Parse the signal point (m1, delta = dMchi/m1, ctau [mm]) from a signal
    sample name, independent of year (see sample_m1).
    """
    dmchi = float(re.search(r'dMchi-([\d]+p[\d]+)', samp).group(1).replace('p', '.'))
    ctau = float(re.search(r'ctau-([\dp]+)', samp).group(1).replace('p', '.'))
    m1 = sample_m1(samp)
    return m1, dmchi / m1, ctau


def edge_labels(axis):
    """
    Tick labels for the bin edges of `axis`. Overflow is clipped into the top
    bin at fill time, so the last edge is drawn as infinity.
    """
    return [f'{e:g}' for e in axis.edges[:-1]] + [r'$\infty$']


def bin_range(axis, i, var, unit=''):
    """Title string for bin `i` of `axis`; the top bin is open-ended (see edge_labels)."""
    lo, hi = axis[i]
    if i == len(axis) - 1:
        return rf'{var} $> {lo:g}${unit}'
    return rf'${lo:g} <$ {var} $< {hi:g}${unit}'


def plot_eff_4d(num, den, axes, cut, tag, title, ilxy, idr, point=None):
    """
    Reco efficiency in bins of gen electron pT x gen electron |eta|, for gen ee
    Lxy bin `ilxy` and gen ee dR bin `idr`, all signal samples combined. Bins
    are drawn evenly sized regardless of their edge values (the real edges are
    printed as tick labels), with the efficiency printed in each bin.

    If `idr` is None, the dR axis is flattened (num and den summed over all dR
    bins) instead of sliced.

    If `point` = (m1, delta, ctau) is given, `num`/`den` are taken to be from
    only the signal samples at that point, and the plot is labeled with it.
    """
    pt_axis, lxy_axis, dr_axis, eta_axis = axes
    n_pt, n_eta = len(pt_axis), len(eta_axis)

    lxy_lo, lxy_hi = lxy_axis[ilxy]
    bin_subtitle = '\n' + bin_range(lxy_axis, ilxy, r'Gen $L_{xy}(e^+e^-)$', ' cm')
    bin_tag = f'lxy{lxy_lo:g}to{lxy_hi:g}'
    if idr is None:
        num, den = num[:, ilxy, :, :].sum(axis=1), den[:, ilxy, :, :].sum(axis=1)
        bin_tag += '_drall'
    else:
        num, den = num[:, ilxy, idr, :], den[:, ilxy, idr, :]
        dr_lo, dr_hi = dr_axis[idr]
        bin_subtitle += ', ' + bin_range(dr_axis, idr, r'Gen $\Delta R(e^+e^-)$')
        bin_tag += f'_dr{dr_lo:g}to{dr_hi:g}'
    with np.errstate(invalid='ignore', divide='ignore'):
        eff = np.where(den > 0, num / den, np.nan).T   # shape (n_eta, n_pt)

    fig, ax = plt.subplots(figsize=(10, 8))
    mesh = ax.pcolormesh(np.arange(n_pt + 1), np.arange(n_eta + 1), eff,
                         cmap='viridis', vmin=0, vmax=1)
    fig.colorbar(mesh, ax=ax).set_label('Reco Efficiency', fontsize=18)

    for i in range(n_eta):
        for j in range(n_pt):
            if np.isnan(eff[i, j]):
                continue
            ax.text(j + 0.5, i + 0.5, f'{eff[i, j]:.3f}', ha='center', va='center', fontsize=14,
                    color='black' if eff[i, j] > 0.5 else 'white')

    ax.set_xticks(np.arange(n_pt + 1))
    ax.set_xticklabels(edge_labels(pt_axis))
    ax.set_yticks(np.arange(n_eta + 1))
    ax.set_yticklabels(edge_labels(eta_axis))
    ax.set_xlabel(r'Gen $p_{T}(e)$ [GeV]', fontsize=20)
    ax.set_ylabel(r'Gen $|\eta(e)|$', fontsize=20)
    if point is not None:
        m1, delta, ctau = point
        sample_title = rf'$m_1$ = {m1:g} GeV, $\Delta$ = {delta:g}, $c\tau$ = {ctau:g} mm'
        outtag = f'{plottag}_m1-{m1:g}_delta-{delta:g}_ctau-{ctau:g}'
    else:
        sample_title = 'All Signal Samples'
        outtag = plottag
    ax.set_title(f'{title} [{CUTS[cut].capitalize()}, {sample_title}]' + bin_subtitle, fontsize=16)
    ax.tick_params(axis='both', labelsize=14)

    plt.tight_layout()
    tagdir = os.path.join(plotdir, tag)
    os.makedirs(tagdir, exist_ok=True)
    outpath = f'{tagdir}/elerecoeff4d_{tag}_pt_vs_abseta_{bin_tag}_{CUTS[cut]}_{outtag}.png'
    plt.savefig(outpath)
    print(f'Saved: {outpath}')
    plt.close(fig)


# ---- Load histograms and make plots ----------------------------------------

s_hists = util.load(saved_signal_hists)[0]
samples = list(s_hists['cutflow'].keys())
point_samples = [s for s in samples if np.allclose(sample_point(s), SINGLE_POINT)]
h_ref = ee4d(s_hists, DEN_4D[0])
axes_4d = tuple(h_ref.axes[name] for name in ('pt', 'lxy', 'dr', 'abseta'))

for cut in CUTS:
    for samps, point in ((samples, None), (point_samples, SINGLE_POINT)):
        den = combine_neff(s_hists, DEN_4D, cut, samps)
        for tag, (cats, title) in NUMS_4D.items():
            num = combine_neff(s_hists, cats, cut, samps)
            for ilxy in range(len(axes_4d[1])):
                for idr in [*range(len(axes_4d[2])), None]:
                    plot_eff_4d(num, den, axes_4d, cut, tag, title, ilxy, idr, point=point)
