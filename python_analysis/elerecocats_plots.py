# imports
import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt

import coffea.util as util

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

outdir = os.path.join(REPO_ROOT, 'workarea')
plotdir = os.path.join(REPO_ROOT, 'plots', 'elerecoeff', 'reasons')
os.makedirs(plotdir, exist_ok=True)

sig_vers = 'Jul2026noID'
selection = 'anabrv'
hists = 'elerecocats'
saved_signal_hists = f'{outdir}/hists_sig{sig_vers}_{selection}-sel_{hists}.coffea'
# elerecoeff hists for the overall gen electron distributions (same 4D binning, see configs/histo_configs/elerecoeff.py)
saved_eff_hists = f'{outdir}/hists_sig{sig_vers}_{selection}-sel_elerecoeff.coffea'
plottag = f'sig{sig_vers}_{selection}-sel'

# Cut stages to plot at -- see configs/cut_configs/anabrv_selection.py.
CUTS = {
    'cut1': 'inclusive',
    #'cut3': 'preselected',
}

# Reason-breakdown hists filled in configs/histo_configs/elerecocats.py, axes (samp, cut, reason, pt, lxy, dr, abseta).
# The reasons partition each category, so the sum over reasons is the category total.
CATS = {
    'alptnosum': 'AllLowPt, not Lpt + GED',
    'sumnoalpt': 'Lpt + GED, not AllLowPt',
}

VAR_LABELS = {
    'pt':     r'Gen $p_{T}(e)$ [GeV]',
    'lxy':    r'Gen $L_{xy}(e^+e^-)$ [cm]',
    'dr':     r'Gen $\Delta R(e^+e^-)$',
    'abseta': r'Gen $|\eta(e)|$',
}

# elerecoeff ee4d categories that partition the gen electrons by matchType, so their sum is all gen electrons
ALL_CATS = ['lpt', 'ged', 'none']

# single signal point (m1 [GeV], delta, ctau [mm]) to also make the plots for alone
SINGLE_POINT = (50, 0.2, 100)


def reason_hist(sig_histo, c):
    return sig_histo[f'ele_reco_{c}_reason_ee4d']


def n_eff(h):
    """
    Per-bin effective raw MC event count: N_eff = sum(w)^2 / sum(w^2), i.e.
    vals^2/variances -- see vtxreco_plots.py.
    """
    vals, var = h.values(), h.variances()
    return np.where(var > 0, vals**2 / np.where(var > 0, var, 1), 0)


def reason_neff(h, cut, samples, axes):
    """
    Per-bin N_eff of each reason in hist `h` at `cut`, projected onto `axes`
    and summed over signal samples (so the combination isn't dominated by the
    highest-xsec mass point). Samples/cuts never filled in `h` contribute zero.
    Returns an array of shape (n_reason, *[len(axis) for axis in axes]).
    """
    reasons = list(h.axes['reason'])
    shape = (len(reasons), *[len(h.axes[a]) for a in axes])
    tot = np.zeros(shape)
    if cut not in list(h.axes['cut']):
        return tot
    filled = list(h.axes['samp'])
    for s in samples:
        if s not in filled:
            continue
        hs = h[{'samp': s, 'cut': cut}]
        for ir, r in enumerate(reasons):
            tot[ir] += n_eff(hs[{'reason': r}].project(*axes))
    return tot


def overall_neff(eff_histo, cut, samples, var):
    """
    Per-bin N_eff of all gen electrons (sum of the ALL_CATS elerecoeff ee4d hists)
    at `cut`, projected onto `var` and summed over signal samples as in reason_neff.
    """
    hs = [eff_histo[f'ele_reco_{c}_ee4d'] for c in ALL_CATS]
    tot = np.zeros(len(hs[0].axes[var]))
    if cut not in list(hs[0].axes['cut']):
        return tot
    for s in samples:
        if s not in list(hs[0].axes['samp']):
            continue
        tot += n_eff(sum(h[{'samp': s, 'cut': cut}].project(var) for h in hs))
    return tot


def category_fraction(neff):
    """Fraction of the category in each bin due to each reason (reason axis first)."""
    den = neff.sum(axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, neff / den, np.nan), den


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


def point_labels(point):
    """(title string, output filename tag) for signal point `point`, or all signal if None."""
    if point is None:
        return 'All Signal Samples', plottag
    m1, delta, ctau = point
    return (rf'$m_1$ = {m1:g} GeV, $\Delta$ = {delta:g}, $c\tau$ = {ctau:g} mm',
            f'{plottag}_m1-{m1:g}_delta-{delta:g}_ctau-{ctau:g}')


def reason_tag(r):
    """Filename-safe tag for reason `r`, e.g. '1 GED + 1 ALPT matched, opp charge' -> '1GED1ALPTmatched_oppcharge'."""
    return re.sub(r'[^\w,]', '', r).replace(',', '_')


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


def save(fig, subdir, name):
    d = os.path.join(plotdir, subdir)
    os.makedirs(d, exist_ok=True)
    outpath = f'{d}/{name}.png'
    fig.savefig(outpath)
    print(f'Saved: {outpath}')
    plt.close(fig)


def overall_scale(cat_total, overall_total):
    """
    Constant factor to divide the overall gen electron N_eff by so it is drawn on
    the category's scale: the ratio of the totals, rounded up to 1, 2 or 5 x 10^k.
    """
    if cat_total <= 0 or overall_total <= 0:
        return 1
    ratio = overall_total / cat_total
    k = 10 ** np.floor(np.log10(ratio))
    return next(m * k for m in (1, 2, 5, 10) if m * k >= ratio)


def plot_reason_stack_1d(neff, overall, scale, reasons, axis, cat, cut, point=None):
    """
    Stacked N_eff of category `cat` from each reason vs. one gen variable (all
    other axes summed), not normalized per bin. Bins are drawn evenly sized
    regardless of their edge values (the real edges are printed as tick labels),
    with the category's total N_eff printed above each bin. The overall gen
    electron N_eff `overall` is drawn as a gray line, divided by `scale`.
    """
    den = neff.sum(axis=0)
    n = len(axis)
    x = np.arange(n) + 0.5

    fig, ax = plt.subplots(figsize=(10, 8))
    bottom = np.zeros(n)
    colors = plt.cm.tab10(np.arange(len(reasons)))
    for ir, r in enumerate(reasons):
        ax.bar(x, neff[ir], width=1, bottom=bottom, color=colors[ir], edgecolor='black', label=r)
        bottom += neff[ir]
    for j in range(n):
        ax.text(x[j], den[j], f'{den[j]:.0f}', ha='center', va='bottom', fontsize=11)
    overall = overall / scale
    ax.stairs(overall, np.arange(n + 1), color='gray', linewidth=2.5, zorder=3,
              label=rf'All gen $e$ ($\div${scale:g})')

    ax.set_xlim(0, n)
    ax.set_ylim(0, 1.4 * max(den.max(), overall.max(), 1))
    ax.set_xticks(np.arange(n + 1))
    ax.set_xticklabels(edge_labels(axis))
    ax.set_xlabel(VAR_LABELS[axis.name], fontsize=20)
    ax.set_ylabel(rf'{cat} electrons ($N_{{eff}}$)', fontsize=20)
    sample_title, outtag = point_labels(point)
    ax.set_title(f'{CATS[cat]} Electron Reasons [{CUTS[cut].capitalize()}, {sample_title}]', fontsize=16)
    ax.legend(loc='upper center', ncol=2, fontsize=12, frameon=False)
    ax.tick_params(axis='both', labelsize=14)

    plt.tight_layout()
    save(fig, f'{cat}/1D', f'elerecocats_{cat}_reasons_vs_{axis.name}_{CUTS[cut]}_{outtag}')


def plot_reason_map(frac, axes, cat, reason, cut, ilxy, idr, point=None):
    """
    Fraction of category `cat` due to `reason` in bins of gen electron pT x gen
    electron |eta|, for gen ee Lxy bin `ilxy` and gen ee dR bin `idr` (dR summed
    if `idr` is None), with the fraction printed in each bin. `frac` is this
    reason's fraction over (pt, lxy, dr, abseta), or (pt, lxy, abseta) if dR is summed.
    """
    pt_axis, lxy_axis, dr_axis, eta_axis = axes
    n_pt, n_eta = len(pt_axis), len(eta_axis)

    lxy_lo, lxy_hi = lxy_axis[ilxy]
    bin_subtitle = '\n' + bin_range(lxy_axis, ilxy, r'Gen $L_{xy}(e^+e^-)$', ' cm')
    bin_tag = f'lxy{lxy_lo:g}to{lxy_hi:g}'
    if idr is None:
        frac = frac[:, ilxy, :]
        bin_tag += '_drall'
    else:
        frac = frac[:, ilxy, idr, :]
        dr_lo, dr_hi = dr_axis[idr]
        bin_subtitle += ', ' + bin_range(dr_axis, idr, r'Gen $\Delta R(e^+e^-)$')
        bin_tag += f'_dr{dr_lo:g}to{dr_hi:g}'
    frac = frac.T   # shape (n_eta, n_pt)

    fig, ax = plt.subplots(figsize=(10, 8))
    mesh = ax.pcolormesh(np.arange(n_pt + 1), np.arange(n_eta + 1), frac,
                         cmap='viridis', vmin=0, vmax=1)
    fig.colorbar(mesh, ax=ax).set_label(f'Fraction of {cat} electrons', fontsize=18)

    for i in range(n_eta):
        for j in range(n_pt):
            if np.isnan(frac[i, j]):
                continue
            ax.text(j + 0.5, i + 0.5, f'{frac[i, j]:.3f}', ha='center', va='center', fontsize=14,
                    color='black' if frac[i, j] > 0.5 else 'white')

    ax.set_xticks(np.arange(n_pt + 1))
    ax.set_xticklabels(edge_labels(pt_axis))
    ax.set_yticks(np.arange(n_eta + 1))
    ax.set_yticklabels(edge_labels(eta_axis))
    ax.set_xlabel(VAR_LABELS['pt'], fontsize=20)
    ax.set_ylabel(VAR_LABELS['abseta'], fontsize=20)
    sample_title, outtag = point_labels(point)
    ax.set_title(f'{CATS[cat]}: {reason} [{CUTS[cut].capitalize()}, {sample_title}]' + bin_subtitle, fontsize=14)
    ax.tick_params(axis='both', labelsize=14)

    plt.tight_layout()
    rtag = reason_tag(reason)
    save(fig, f'{cat}/{rtag}', f'elerecocats_{cat}_{rtag}_pt_vs_abseta_{bin_tag}_{CUTS[cut]}_{outtag}')


# ---- Load histograms and make plots ----------------------------------------

s_hists = util.load(saved_signal_hists)[0]
e_hists = util.load(saved_eff_hists)[0]
samples = list(s_hists['cutflow'].keys())
point_samples = [s for s in samples if np.allclose(sample_point(s), SINGLE_POINT)]

for cut in CUTS:
    for samps, point in ((samples, None), (point_samples, SINGLE_POINT)):
        for cat in CATS:
            h = reason_hist(s_hists, cat)
            reasons = list(h.axes['reason'])
            axes_4d = tuple(h.axes[name] for name in ('pt', 'lxy', 'dr', 'abseta'))

            neff_1d = {var: reason_neff(h, cut, samps, (var,)) for var in VAR_LABELS}
            overall_1d = {var: overall_neff(e_hists, cut, samps, var) for var in VAR_LABELS}
            # one scale for all four variables, from the totals (the same for every variable's projection)
            scale = overall_scale(neff_1d['pt'].sum(), overall_1d['pt'].sum())
            for var in VAR_LABELS:
                plot_reason_stack_1d(neff_1d[var], overall_1d[var], scale, reasons, h.axes[var], cat, cut, point=point)

            frac_4d, _ = category_fraction(reason_neff(h, cut, samps, ('pt', 'lxy', 'dr', 'abseta')))
            frac_3d, _ = category_fraction(reason_neff(h, cut, samps, ('pt', 'lxy', 'abseta')))
            for ir, r in enumerate(reasons):
                for ilxy in range(len(axes_4d[1])):
                    for idr in range(len(axes_4d[2])):
                        plot_reason_map(frac_4d[ir], axes_4d, cat, r, cut, ilxy, idr, point=point)
                    plot_reason_map(frac_3d[ir], axes_4d, cat, r, cut, ilxy, None, point=point)
