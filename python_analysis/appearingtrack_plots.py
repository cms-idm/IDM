import os
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import coffea.util as util
import analysisTools.utils as utils
import analysisTools.plotTools as ptools

plt.rcParams.update({
    'font.size':        20,
    'axes.titlesize':   24,
    'axes.labelsize':   20,
    'xtick.labelsize':  20,
    'ytick.labelsize':  20,
    'legend.fontsize':  16,
    'figure.titlesize': 24,
})

# ---- FIELDS: the two coffea files to plot from -----------------------------
outdir      = 'workarea'
bkg_vers    = 'Aug2026'
sig_vers    = 'Jul2026noID'
selection   = 'an'
hists_tag   = 'appearingtrack'

saved_signal_hists = f"{outdir}/hists_sig{sig_vers}_{selection}-sel_{hists_tag}.coffea"
saved_bkg_hists    = f"{outdir}/hists_bkg{bkg_vers}_{selection}-sel_{hists_tag}.coffea"

# ---- Plot settings ----------------------------------------------------------
cut       = 'cut8'      # cut index (key in the 'cut' axis) to plot at
m1_values = [0.5, 5]    # signal mass points to show, combined over delta/ctau
year      = 2024
doLogy    = True
plotdir   = 'plots/appearingtrack'

sig_colors = {0.5: 'black', 5: 'red', 0.05: 'blue', 50: 'darkgreen'}
_SIG_COLOR_FALLBACK = ['magenta', 'orange', 'cyan', 'purple']

_NONPHYS_AXES = ('samp', 'cut', 'isMerged')


def _get_1d_hist_names(hists_dict):
    """Names of 1D 'AT_*' histograms (samp/cut/isMerged + exactly one physical axis)."""
    names = []
    for key, h in hists_dict.items():
        if not key.startswith('AT_') or not hasattr(h, 'axes'):
            continue
        phys_axes = [a for a in h.axes.name if a not in _NONPHYS_AXES]
        if len(phys_axes) == 1:
            names.append(key)
    return names


def _combine_group(h, samp_names):
    """Sum the given (present) sample names on the 'samp' axis of h into one hist."""
    present = ptools.getPresentSamples(h, samp_names)
    if not present:
        return None
    return h[{'samp': present}][{'samp': sum}]


def plot_variable(s_hists, b_hists, bkg_cats, s_pts, var, cut, outdir):
    """Build and save a single normalized-shape overlay plot for one 'AT_*' variable."""
    if var not in s_hists or var not in b_hists:
        print(f"Skipping {var}: missing from one of the two files")
        return

    hs, hb = s_hists[var], b_hists[var]

    try:
        hs_cut = hs[{'cut': cut, 'isMerged': sum}]
        hb_cut = hb[{'cut': cut, 'isMerged': sum}]
    except Exception as e:
        print(f"Skipping {var}: couldn't select cut='{cut}' ({e})")
        return

    # ---- combine background sub-samples into per-category hists (xsec+wgt already
    # baked into each bin's fill weight, see appearingtrack.fillHistos) -----------
    bkg_hists, bkg_integrals = {}, {}
    for cat, samps in bkg_cats.items():
        h_cat = _combine_group(hb_cut, samps)
        if h_cat is None:
            continue
        integral = h_cat.sum(flow=False).value
        if integral <= 0:
            continue
        bkg_hists[cat] = h_cat
        bkg_integrals[cat] = integral

    # ---- combine signal sub-samples into per-m1 hists --------------------------
    sig_hists, sig_integrals = {}, {}
    for m1 in m1_values:
        samps = s_pts[np.isclose(s_pts['m1'], m1)].index.tolist()
        h_m1 = _combine_group(hs_cut, samps)
        if h_m1 is None:
            continue
        integral = h_m1.sum(flow=False).value
        if integral <= 0:
            continue
        sig_hists[m1] = h_m1
        sig_integrals[m1] = integral

    if not bkg_hists and not sig_hists:
        print(f"Skipping {var}: nothing to plot at cut='{cut}'")
        return

    # ---- normalize every displayed histogram by ONE shared factor, so their
    # relative (xsec+wgt-weighted) normalizations stay comparable to each other --
    total = sum(bkg_integrals.values()) + sum(sig_integrals.values())
    if total <= 0:
        print(f"Skipping {var}: zero total yield at cut='{cut}'")
        return

    bkg_hists = {cat: h / total for cat, h in bkg_hists.items()}
    sig_hists = {m1: h / total for m1, h in sig_hists.items()}

    fig, ax = plt.subplots(figsize=(12, 10))
    hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)

    if bkg_hists:
        cats_sorted = sorted(bkg_hists, key=lambda c: bkg_integrals[c])
        colors = [ptools.bkg_cmap.get(c, 'gray') for c in cats_sorted]
        hep.histplot([bkg_hists[c] for c in cats_sorted], ax=ax, label=cats_sorted,
                     color=colors, histtype='fill', stack=True, edgecolor='black', linewidth=0.5)

    for i, m1 in enumerate(m1_values):
        if m1 not in sig_hists:
            continue
        color = sig_colors.get(m1, _SIG_COLOR_FALLBACK[i % len(_SIG_COLOR_FALLBACK)])
        hep.histplot(sig_hists[m1], ax=ax, histtype='step', linewidth=2.5, yerr=True,
                     color=color, label=rf'Signal $m_1$ = {m1} GeV')

    phys_axis = [a for a in hs_cut.axes.name if a not in _NONPHYS_AXES][0]
    ax.set_xlabel(hs_cut.axes[phys_axis].label)
    ax.set_ylabel('Normalized Events')
    ax.set_title(f'Appearing Track: cut = {cut}')
    if doLogy:
        ax.set_yscale('log')
    ax.legend(loc='best', fontsize=14, ncol=1)

    os.makedirs(outdir, exist_ok=True)
    plt.tight_layout()
    outpath = f'{outdir}/{var}_{cut}.png'
    plt.savefig(outpath)
    plt.close(fig)
    print(f"Saved: {outpath}")


if __name__ == '__main__':
    s_hists = util.load(saved_signal_hists)[0]
    b_hists = util.load(saved_bkg_hists)[0]

    bkg_cats, _ = utils.bkg_categories(b_hists['cutflow'])
    s_pts = utils.get_signal_point_dict(s_hists)

    variables = sorted(set(_get_1d_hist_names(s_hists)) & set(_get_1d_hist_names(b_hists)))
    print(f"Plotting {len(variables)} appearing-track variables at cut='{cut}': {variables}")

    for var in variables:
        plot_variable(s_hists, b_hists, bkg_cats, s_pts, var, cut, plotdir)
