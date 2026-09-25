import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import coffea.util as util
import analysisTools.utils as utils
import analysisTools.plotTools as ptools

# samp list-selection is immediately summed, so dropped bins not going to overflow is fine
warnings.filterwarnings('ignore', message='List indexing selection is experimental')

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
plottag     = f'sb{bkg_vers}_{selection}-sel'

saved_signal_hists = f"{outdir}/hists_sig{sig_vers}_{selection}-sel_{hists_tag}.coffea"
saved_bkg_hists    = f"{outdir}/hists_bkg{bkg_vers}_{selection}-sel_{hists_tag}.coffea"

# ---- Plot settings ----------------------------------------------------------
cut       = 'cut8'      # cut index (key in the 'cut' axis) to plot at
m1_values = [0.5, 5]    # signal mass points to show, combined over delta/ctau
year      = 2024
doLogy    = True
plotdir   = 'plots/appearingtrack'

sig_linestyles = ['-', '--']  # one per entry in m1_values
sig_merge_labels = ('mergedSelected', 'mergedNotSelected', 'notMerged')  # isMerged categories, signal only
sig_colors = {
    'mergedSelected':    'red',
    'mergedNotSelected': 'orange',
    'notMerged':         'black',
}  # by isMerged category

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


def _prep_bkg(hb_cut, bkg_cats):
    """Per-category combined (raw, xsec-weighted) hists + integrals, plus a hist summing
    every present background sample together -- built directly from the raw samples (not
    from the per-category hists after they've been shape-normalized) so the relative
    cross-sections between background types are preserved in the sum."""
    bkg_hists, bkg_integrals = {}, {}
    all_bkg_samps = []
    for cat, samps in bkg_cats.items():
        h_cat = _combine_group(hb_cut, samps)
        if h_cat is None:
            continue
        integral = h_cat.sum(flow=False).value
        if integral <= 0:
            continue
        bkg_hists[cat] = h_cat
        bkg_integrals[cat] = integral
        all_bkg_samps.extend(samps)

    bkg_total, bkg_total_integral = None, 0
    h_all = _combine_group(hb_cut, all_bkg_samps)
    if h_all is not None:
        integral = h_all.sum(flow=False).value
        if integral > 0:
            bkg_total, bkg_total_integral = h_all, integral

    return bkg_hists, bkg_integrals, bkg_total, bkg_total_integral


def _cutflow_weighted_total(hists_dict, samples, cut):
    """Sum of xsec/lumi-weighted event counts (the 'cutflow_cts' branch filled by
    analysisTools.py, keyed by sample name then cut label) entering `cut`, for the
    given sample names. This is the true per-event denominator *before*
    appearingtrack.py's own internal candidate-selection (see fillHistos) -- i.e. it
    includes events for which no usable AllLptElectron candidate was found and which
    therefore never appear in any isMerged category / AT_* histogram. Returns 0 if
    'cutflow_cts' or the requested cut isn't present (e.g. older saved hist files)."""
    cutflow_cts = hists_dict.get('cutflow_cts', {})
    return sum(cutflow_cts.get(s, {}).get(cut, 0.0) for s in samples)


def _prep_sig(hs_cut, s_hists, s_pts, cut):
    """Per-(m1, isMerged) combined hists + integrals + category fractions, for the
    illustrative mass points in m1_values. Fractions are relative to ALL events
    entering `cut` at that mass point (via _cutflow_weighted_total), not just the
    ones landing in one of the three isMerged categories -- so together with
    sig_no_cand_fracs (the complementary "no candidate electron at all" fraction)
    they sum to ~100%. Falls back to normalizing against the three categories alone,
    and omits sig_no_cand_fracs, if cutflow_cts isn't available."""
    sig_hists, sig_integrals, sig_fracs, sig_no_cand_fracs = {}, {}, {}, {}
    for m1 in m1_values:
        samps = s_pts[np.isclose(s_pts['m1'], m1)].index.tolist()
        h_m1 = _combine_group(hs_cut, samps)
        if h_m1 is None:
            continue
        splits = {ml: h_m1[{'isMerged': ml}] for ml in sig_merge_labels}
        integrals = {ml: h.sum(flow=False).value for ml, h in splits.items()}
        cat_total = sum(integrals.values())
        total_entering = _cutflow_weighted_total(s_hists, samps, cut)
        denom = total_entering if total_entering > 0 else cat_total
        if denom <= 0:
            continue
        for merge_label, integral in integrals.items():
            if integral <= 0:
                continue
            sig_hists[(m1, merge_label)] = splits[merge_label]
            sig_integrals[(m1, merge_label)] = integral
            sig_fracs[(m1, merge_label)] = integral / denom
        if total_entering > 0:
            sig_no_cand_fracs[m1] = max(0.0, (total_entering - cat_total) / total_entering)
    return sig_hists, sig_integrals, sig_fracs, sig_no_cand_fracs


def _prep_sig_total(hs_cut, s_hists, s_pts, cut):
    """All signal mass points summed together (preserving relative cross-sections, same
    idea as _prep_bkg's all-backgrounds hist), split by isMerged, plus each category's
    fraction of all events entering `cut` (see _prep_sig) and the complementary
    "no candidate electron at all" fraction."""
    all_samps = s_pts.index.tolist()
    h_all = _combine_group(hs_cut, all_samps)
    sig_total, sig_total_integrals, sig_total_fracs, sig_total_no_cand_frac = {}, {}, {}, None
    if h_all is None:
        return sig_total, sig_total_integrals, sig_total_fracs, sig_total_no_cand_frac
    splits = {ml: h_all[{'isMerged': ml}] for ml in sig_merge_labels}
    integrals = {ml: h.sum(flow=False).value for ml, h in splits.items()}
    cat_total = sum(integrals.values())
    total_entering = _cutflow_weighted_total(s_hists, all_samps, cut)
    denom = total_entering if total_entering > 0 else cat_total
    if denom <= 0:
        return sig_total, sig_total_integrals, sig_total_fracs, sig_total_no_cand_frac
    for merge_label, integral in integrals.items():
        if integral <= 0:
            continue
        sig_total[merge_label] = splits[merge_label]
        sig_total_integrals[merge_label] = integral
        sig_total_fracs[merge_label] = integral / denom
    if total_entering > 0:
        sig_total_no_cand_frac = max(0.0, (total_entering - cat_total) / total_entering)
    return sig_total, sig_total_integrals, sig_total_fracs, sig_total_no_cand_frac


def _finalize_plot(ax, h_ref, var, cut, outdir, tag, title_suffix):
    """Shared axis labeling/scaling/saving for all three plot variants."""
    phys_axis = [a for a in h_ref.axes.name if a not in _NONPHYS_AXES][0]
    ax.set_xlabel(h_ref.axes[phys_axis].label)
    ax.set_ylabel('Normalized Events')
    ax.set_title(f'Appearing Track: cut = {cut} ({title_suffix})')
    if doLogy:
        ax.set_yscale('log')
    # axes built with a log transform already have bins equally spaced in
    # log(x) (i.e. constant-width in the space the plot is about to use), so
    # the raw per-bin heights from histplot need no bin-width/area correction
    # -- just switch the x-axis to match the axis's native spacing.
    if getattr(h_ref.axes[phys_axis], 'transform', None) is not None:
        ax.set_xscale('log')
    ax.legend(loc='best', fontsize=14, ncol=1)

    os.makedirs(outdir, exist_ok=True)
    plt.tight_layout()
    outpath = f'{outdir}/hist_{plottag}_{var}_{cut}_{tag}.png'
    plt.savefig(outpath)
    plt.close(ax.figure)
    print(f"Saved: {outpath}")


def plot_background(b_hists, bkg_cats, var, cut, outdir):
    """Background-only plot: individual background categories plus their combined sum,
    all shown normalized to unit area for shape comparison."""
    if var not in b_hists:
        print(f"Skipping {var} (background): missing from background file")
        return

    hb = b_hists[var]
    try:
        hb_cut = hb[{'cut': cut, 'isMerged': sum}]
    except Exception as e:
        print(f"Skipping {var} (background): couldn't select cut='{cut}' ({e})")
        return

    bkg_hists, bkg_integrals, bkg_total, bkg_total_integral = _prep_bkg(hb_cut, bkg_cats)
    if not bkg_hists:
        print(f"Skipping {var} (background): nothing to plot at cut='{cut}'")
        return

    fig, ax = plt.subplots(figsize=(12, 10))
    hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)

    cats_sorted = sorted(bkg_hists, key=lambda c: bkg_integrals[c], reverse=True)
    for i, cat in enumerate(cats_sorted):
        hep.histplot(bkg_hists[cat] / bkg_integrals[cat], ax=ax, histtype='step',
                     linewidth=2, color=f'C{i}', label=cat)
    if bkg_total is not None:
        hep.histplot(bkg_total / bkg_total_integral, ax=ax, histtype='step', linewidth=2.5,
                     color='black', linestyle='--', label='All backgrounds')

    _finalize_plot(ax, hb_cut, var, cut, outdir, 'bkg', 'Backgrounds')


_MERGE_TAGS = {
    'mergedSelected':    'merged, selected',
    'mergedNotSelected': 'merged, not selected',
    'notMerged':         'not merged',
}


def plot_signal(s_hists, s_pts, var, cut, outdir):
    """Signal-only plot: the signal hists (m1 in m1_values, split by isMerged category)."""
    if var not in s_hists:
        print(f"Skipping {var} (signal): missing from signal file")
        return

    hs = s_hists[var]
    try:
        hs_cut = hs[{'cut': cut}]  # isMerged kept: split by category below
    except Exception as e:
        print(f"Skipping {var} (signal): couldn't select cut='{cut}' ({e})")
        return

    sig_hists, sig_integrals, sig_fracs, sig_no_cand_fracs = _prep_sig(hs_cut, s_hists, s_pts, cut)
    if not sig_hists and not sig_no_cand_fracs:
        print(f"Skipping {var} (signal): nothing to plot at cut='{cut}'")
        return

    fig, ax = plt.subplots(figsize=(12, 10))
    hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)

    # signal: color by isMerged category (sig_colors), one linestyle per mass
    # point (shared across all categories). Legend gives each category's
    # fraction of all events entering `cut` at that mass point, plus (as a
    # label-only entry with no histogram, since those events are dropped
    # before any AT_* fill -- see _prep_sig) the fraction with no candidate
    # electron at all.
    for i, m1 in enumerate(m1_values):
        linestyle = sig_linestyles[i % len(sig_linestyles)]
        for merge_label in sig_merge_labels:
            key = (m1, merge_label)
            if key not in sig_hists:
                continue
            hep.histplot(sig_hists[key] / sig_integrals[key], ax=ax, histtype='step',
                         linewidth=1.5, yerr=True, color=sig_colors[merge_label],
                         linestyle=linestyle,
                         label=rf'Signal $m_1$ = {m1} GeV ({_MERGE_TAGS[merge_label]}, {sig_fracs[key]:.1%})')
        if m1 in sig_no_cand_fracs:
            ax.plot([], [], linestyle='none',
                     label=rf'Signal $m_1$ = {m1} GeV (no candidate electron, {sig_no_cand_fracs[m1]:.1%})')

    _finalize_plot(ax, hs_cut, var, cut, outdir, 'sig', 'Signal')


def plot_summary(s_hists, b_hists, bkg_cats, s_pts, var, cut, outdir):
    """Summary plot: total background vs. total signal (summed over all mass points),
    split by isMerged category."""
    if var not in s_hists or var not in b_hists:
        print(f"Skipping {var} (summary): missing from one of the two files")
        return

    hs, hb = s_hists[var], b_hists[var]
    try:
        hs_cut = hs[{'cut': cut}]  # isMerged kept: split by category below
        hb_cut = hb[{'cut': cut, 'isMerged': sum}]
    except Exception as e:
        print(f"Skipping {var} (summary): couldn't select cut='{cut}' ({e})")
        return

    _, _, bkg_total, bkg_total_integral = _prep_bkg(hb_cut, bkg_cats)
    sig_total, sig_total_integrals, sig_total_fracs, sig_total_no_cand_frac = _prep_sig_total(hs_cut, s_hists, s_pts, cut)

    if bkg_total is None and not sig_total and sig_total_no_cand_frac is None:
        print(f"Skipping {var} (summary): nothing to plot at cut='{cut}'")
        return

    fig, ax = plt.subplots(figsize=(12, 10))
    hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)

    if bkg_total is not None:
        hep.histplot(bkg_total / bkg_total_integral, ax=ax, histtype='step', linewidth=2,
                     color='C0', label='All backgrounds')

    # Legend gives each category's fraction of all signal events entering `cut`
    # (summed over every mass point), plus (as a label-only entry with no
    # histogram -- see _prep_sig_total) the fraction with no candidate electron
    # at all.
    for merge_label in sig_merge_labels:
        if merge_label not in sig_total:
            continue
        hep.histplot(sig_total[merge_label] / sig_total_integrals[merge_label], ax=ax,
                     histtype='step', linewidth=1.5, yerr=True,
                     color=sig_colors[merge_label], linestyle='-',
                     label=f'All signal ({_MERGE_TAGS[merge_label]}, {sig_total_fracs[merge_label]:.1%})')
    if sig_total_no_cand_frac is not None:
        ax.plot([], [], linestyle='none',
                 label=f'All signal (no candidate electron, {sig_total_no_cand_frac:.1%})')

    _finalize_plot(ax, hs_cut, var, cut, outdir, 'summary', 'Summary')


if __name__ == '__main__':
    s_hists = util.load(saved_signal_hists)[0]
    b_hists = util.load(saved_bkg_hists)[0]

    bkg_cats, _ = utils.bkg_categories(b_hists['cutflow'])
    s_pts = utils.get_signal_point_dict(s_hists)
    s_pts = s_pts[~np.isclose(s_pts['m1'], 0.05)]

    variables = sorted(set(_get_1d_hist_names(s_hists)) & set(_get_1d_hist_names(b_hists)))
    print(f"Plotting {len(variables)} appearing-track variables at cut='{cut}': {variables}")

    for var in variables:
        plot_background(b_hists, bkg_cats, var, cut, plotdir)
        plot_signal(s_hists, s_pts, var, cut, plotdir)
        plot_summary(s_hists, b_hists, bkg_cats, s_pts, var, cut, plotdir)
