import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import mplhep as hep
import coffea.util as util
import analysisTools.utils as utils
import analysisTools.plotTools as ptools

# samp list-selection is immediately summed, so dropped bins not going to overflow is fine
warnings.filterwarnings('ignore', message='List indexing selection is experimental')

ptools.set_plot_style()

# ---- FIELDS: the two coffea files to plot from -----------------------------
outdir      = 'workarea'
bkg_vers    = 'Aug2026'
sig_vers    = 'Jul2026noID'
selection   = 'aptrk'
hists_tag   = 'appearingtrack'
plottag     = f'sb{bkg_vers}_{selection}-sel'

saved_signal_hists = f"{outdir}/hists_sig{sig_vers}_{selection}-sel_{hists_tag}.coffea"
saved_bkg_hists    = f"{outdir}/hists_bkg{bkg_vers}_{selection}-sel_{hists_tag}.coffea"

# ---- Plot settings ----------------------------------------------------------
cut       = 'cut9'      # cut index (key in the 'cut' axis) to plot the AT_* hists at
# Cut for the ATgm_* hists (gen-matched merged electron, signal only). These
# are filled for every signal event with an isGenMerged AllLptElectron,
# independent of the merged-candidate requirement -- so they're plotted at the
# last saved stage before cut9 (which requires a good merged candidate).
gm_cut    = 'cut3'
m1_values = [0.5, 5]    # signal mass points to show, combined over delta/ctau
year      = 2024
doLogy    = True
plotdir   = 'plots/appearingtrack'

sig_linestyles = ['-', '--']  # one per entry in m1_values
sig_merge_labels = ('mergedSelected', 'mergedNotSelected', 'notMerged')  # isMerged categories, signal only
# by isMerged category; CMS 6-colour palette, leaving index 0 (blue) for 'All Backgrounds'
sig_colors = {
    'mergedSelected':    ptools.CMS_COLORS_6[2],   # red
    'mergedNotSelected': ptools.CMS_COLORS_6[1],   # orange
    'notMerged':         ptools.CMS_COLORS_6[3],   # purple
}
bkg_total_color = ptools.CMS_COLORS_6[0]           # blue

_NONPHYS_AXES = ('samp', 'cut', 'isMerged')

# human-readable plot-title names for the AT_* hists; unlisted vars fall back to the raw name
_VAR_TITLES = {
    'AT_pt':                  r'$p_T$',
    'AT_eta':                 r'$\eta$',
    'AT_phi':                 r'$\phi$',
    'AT_ID':                  'ID Score',
    'AT_angRes':              'Angular Resolution',
    'AT_dxy':                 r'Track $d_{xy}$',
    'AT_dz':                  r'Track $d_z$',
    'AT_trkChi2':             r'Track $\chi^2/\mathrm{ndf}$',
    'AT_trkIso':              'Track Isolation',
    'AT_trkRelIso':           'Relative Track Isolation',
    'AT_calIso':              'Calorimeter Isolation',
    'AT_calRelIso':           'Relative Calorimeter Isolation',
    'AT_miniIso':             'Mini Isolation',
    'AT_miniRelIso':          'Relative Mini Isolation',
    'AT_rhoEA':               r'Pileup Correction ($\rho \times$ EA)',
    'AT_trkProb':             'Track Probability',
    'AT_numTrackerHits':      'N Tracker Hits',
    'AT_numPixHits':          'N Pixel Hits',
    'AT_numStripHits':        'N Strip Hits',
    'AT_minDRtoReg':          r'Min $\Delta R$ to GED',
    'AT_drNearestEle':        r'Min $\Delta R$ to Other Low-$p_T$',
    'AT_mindRj':              r'Min $\Delta R$ to Jet',
    'AT_mindPhiJ':            r'Min $\Delta\phi$ to Jet',
    'AT_mindRj_vs_ptRatioJ':  r'Nearest Jet $\Delta R$ vs $p_T(e)/p_T(j)$',
    'AT_full55sigmaIetaIeta': r'$\sigma_{i\eta i\eta}$ (Full 5x5)',
    'AT_scEtaWidth':          r'SC $\eta$ Width',
    'AT_scPhiWidth':          r'SC $\phi$ Width',
    'AT_full55r9':            r'$R_9$ (Full 5x5)',
    'AT_full55sigmaIphiIphi': r'$\sigma_{i\phi i\phi}$ (Full 5x5)',
    'AT_e1x5Frac':            r'$E_{1\times5}/E_{5\times5}$',
    'AT_e2x5Frac':            r'$E_{2\times5}^{max}/E_{5\times5}$',
    'AT_scClustersSize':      'N Clusters in SC',
    'AT_scEoverP':            r'$E_{SC}/p$',
    'AT_scRawFrac':           r'$E_{SC}^{raw}/E_{SC}$',
    'AT_dEtaSCTrk':           r'$|\eta_{SC} - \eta_{trk}|$',
    'AT_dRj0':                r'$\Delta R$ to Leading Jet',
    'AT_met_leadPt_ratio':    r'$p_T^\mathrm{miss}$ / Leading Jet $p_T$',
    'AT_HT':                  r'$H_T$',
    'AT_nJets':               r'$N_\mathrm{jets}$',
    'AT_absdEtaSeed':         r'$|\Delta\eta_\mathrm{seed}|$',
    'AT_absdPhiIn':           r'$|\Delta\phi_\mathrm{in}|$',
    'AT_HoverE':              'H/E',
    'AT_abs1overEm1overP':    r'$|1/E - 1/p|$',
    'AT_expMissingInnerHits': 'Expected Missing Inner Hits',
    'AT_ptmiss':              r'$p_T^\mathrm{miss}$',
    'AT_dphi_e_ptmiss':       r'$\Delta\phi$(e, $p_T^\mathrm{miss}$)',
    'AT_dphi_e_ptmiss_log':   r'$\Delta\phi$(e, $p_T^\mathrm{miss}$) (Log Scale)',
    'AT_cosdphi_e_ptmiss':    r'$\cos\Delta\phi$(e, $p_T^\mathrm{miss}$)',
    'AT_lxyEst':              r'Estimated $L_{xy}$ ($d_{xy}/\sin\Delta\phi$)',
    'AT_log10dxydz':          r'$\log_{10}(d_{xy}/d_z)$',
}


def _var_title(var):
    """Plot-title name for an AT_*/ATgm_* hist; ATgm_* reuse the AT_* titles."""
    if var.startswith('ATgm_'):
        var = 'AT_' + var[len('ATgm_'):]
    return _VAR_TITLES.get(var, var)


def _get_1d_hist_names(hists_dict, prefix='AT_'):
    """Names of 1D `prefix`* histograms (samp/cut[/isMerged] + exactly one physical axis)."""
    names = []
    for key, h in hists_dict.items():
        if not key.startswith(prefix) or not hasattr(h, 'axes'):
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


def _finalize_plot(ax, h_ref, var, cut, outdir, tag, title_suffix, logy=doLogy, legend_fs=None,
                   title_prefix='Merged Electron'):
    """Shared axis labeling/scaling/saving for all plot variants."""
    phys_axis = [a for a in h_ref.axes.name if a not in _NONPHYS_AXES][0]
    ax.set_xlabel(h_ref.axes[phys_axis].label)
    ax.set_ylabel('Normalized Events')
    ax.set_title(f'{title_prefix}: {_var_title(var)}')
    if logy:
        ax.set_yscale('log')
    # axes built with a log transform already have bins equally spaced in
    # log(x) (i.e. constant-width in the space the plot is about to use), so
    # the raw per-bin heights from histplot need no bin-width/area correction
    # -- just switch the x-axis to match the axis's native spacing.
    if getattr(h_ref.axes[phys_axis], 'transform', None) is not None:
        ax.set_xscale('log')
    # Legend pinned top-right with headroom added above the data for it (loc='best'
    # can't avoid the data when the curves fill the whole frame).
    leg = ax.legend(loc='upper right', fontsize=legend_fs, ncol=1)
    plt.tight_layout()
    ptools.legend_headroom(ax, leg)

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

    fig, ax = ptools.new_fig(year)

    cats_sorted = sorted(bkg_hists, key=lambda c: bkg_integrals[c], reverse=True)
    for i, cat in enumerate(cats_sorted):
        hep.histplot(bkg_hists[cat] / bkg_integrals[cat], ax=ax, histtype='step',
                     linewidth=2, color=ptools.bkg_cmap.get(cat, ptools.CMS_COLORS_10[3]), label=cat)
    if bkg_total is not None:
        hep.histplot(bkg_total / bkg_total_integral, ax=ax, histtype='step', linewidth=2.5,
                     color='black', linestyle='--', label='All Backgrounds')

    _finalize_plot(ax, hb_cut, var, cut, f'{outdir}/background', 'bkg', 'Backgrounds')


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

    fig, ax = ptools.new_fig(year)

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

    # 8 long entries: a size down from the default so the legend doesn't eat half the frame.
    _finalize_plot(ax, hs_cut, var, cut, f'{outdir}/signal', 'sig', 'Signal', legend_fs=20)


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

    fig, ax = ptools.new_fig(year)

    if bkg_total is not None:
        hep.histplot(bkg_total / bkg_total_integral, ax=ax, histtype='step', linewidth=2,
                     color=bkg_total_color, label='All Backgrounds')

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

    _finalize_plot(ax, hs_cut, var, cut, f'{outdir}/summary', 'summary', 'Summary')


def plot_summarysimple(s_hists, b_hists, bkg_cats, s_pts, var, cut, outdir, logy=doLogy):
    """Simplified summary plot: total background vs. total signal (summed over all mass
    points), signal restricted to the merged-and-selected isMerged category. logy=False
    gives a linear-y version, saved with a '_liny' tag suffix."""
    if var not in s_hists or var not in b_hists:
        print(f"Skipping {var} (summarysimple): missing from one of the two files")
        return

    hs, hb = s_hists[var], b_hists[var]
    try:
        hs_cut = hs[{'cut': cut}]  # isMerged kept: mergedSelected picked out below
        hb_cut = hb[{'cut': cut, 'isMerged': sum}]
    except Exception as e:
        print(f"Skipping {var} (summarysimple): couldn't select cut='{cut}' ({e})")
        return

    _, _, bkg_total, bkg_total_integral = _prep_bkg(hb_cut, bkg_cats)
    sig_total, sig_total_integrals, _, _ = _prep_sig_total(hs_cut, s_hists, s_pts, cut)

    if bkg_total is None and 'mergedSelected' not in sig_total:
        print(f"Skipping {var} (summarysimple): nothing to plot at cut='{cut}'")
        return

    fig, ax = ptools.new_fig(year)

    if bkg_total is not None:
        hep.histplot(bkg_total / bkg_total_integral, ax=ax, histtype='step', linewidth=2,
                     color=bkg_total_color, label='All Backgrounds')
    if 'mergedSelected' in sig_total:
        hep.histplot(sig_total['mergedSelected'] / sig_total_integrals['mergedSelected'], ax=ax,
                     histtype='step', linewidth=1.5, yerr=True,
                     color=sig_colors['mergedSelected'], linestyle='-',
                     label='Signal (All Samples)')

    tag = 'summarysimple' if logy else 'summarysimple_liny'
    _finalize_plot(ax, hs_cut, var, cut, f'{outdir}/summarysimple', tag, 'Summary', logy=logy)


def plot_genmerged_total(s_hists, s_pts, var, cut, outdir):
    """ATgm_* plot: the gen-matched merged electron (AllLptElectron.isGenMerged),
    all signal mass points summed (preserving relative cross-sections),
    normalized to unit area. Signal only, and a different event set from the
    AT_* plots (every event with a gen-matched merged electron at `cut`, not
    just those with a selected candidate) -- hence separate plots. The legend
    gives the fraction of all signal events entering `cut` that have one."""
    if var not in s_hists:
        print(f"Skipping {var} (genmerged): missing from signal file")
        return

    try:
        hs_cut = s_hists[var][{'cut': cut}]
    except Exception as e:
        print(f"Skipping {var} (genmerged): couldn't select cut='{cut}' ({e})")
        return

    all_samps = s_pts.index.tolist()
    h_all = _combine_group(hs_cut, all_samps)
    integral = h_all.sum(flow=False).value if h_all is not None else 0
    if integral <= 0:
        print(f"Skipping {var} (genmerged): nothing to plot at cut='{cut}'")
        return

    # Fraction uses the flow-inclusive count (every event fills exactly once),
    # so it isn't biased by entries outside the axis range.
    total_entering = _cutflow_weighted_total(s_hists, all_samps, cut)
    frac_str = f', {h_all.sum(flow=True).value / total_entering:.1%} of events' if total_entering > 0 else ''

    fig, ax = ptools.new_fig(year)
    hep.histplot(h_all / integral, ax=ax, histtype='step', linewidth=1.5, yerr=True,
                 color=sig_colors['mergedSelected'], linestyle='-',
                 label=f'All signal (gen-matched merged{frac_str})')

    _finalize_plot(ax, hs_cut, var, cut, f'{outdir}/genmerged/total', 'genmerged_total', 'Signal',
                   title_prefix='Gen-Matched Merged Electron')


# 2D ATgm_* hists to plot (all signal summed, unit-normalized, log color scale).
gm_2d_variables = ['ATgm_mindRj_vs_ptRatioJ']
# Dashed guide lines (x, y) marking the self-jet veto region (dR < 0.1 and
# pT(e)/pT(j) > 0.3; see appearingtrack.py) on the 2D self-jet diagnostic.
_GM_2D_GUIDES = {'ATgm_mindRj_vs_ptRatioJ': (0.1, 0.3)}


def plot_genmerged_2d(s_hists, s_pts, var, cut, outdir):
    """2D ATgm_* plot: all signal mass points summed, normalized to unit area
    (in-range bins), log color scale."""
    if var not in s_hists:
        print(f"Skipping {var} (genmerged 2D): missing from signal file")
        return
    try:
        hs_cut = s_hists[var][{'cut': cut}]
    except Exception as e:
        print(f"Skipping {var} (genmerged 2D): couldn't select cut='{cut}' ({e})")
        return

    h_all = _combine_group(hs_cut, s_pts.index.tolist())
    integral = h_all.sum(flow=False).value if h_all is not None else 0
    if integral <= 0:
        print(f"Skipping {var} (genmerged 2D): nothing to plot at cut='{cut}'")
        return

    fig, ax = ptools.new_fig(year)
    art = hep.hist2dplot(h_all / integral, ax=ax, norm=LogNorm())
    art.cbar.set_label('Normalized Events')
    xax, yax = h_all.axes
    ax.set_xlabel(xax.label)
    ax.set_ylabel(yax.label)
    if var in _GM_2D_GUIDES:
        gx, gy = _GM_2D_GUIDES[var]
        ax.axvline(gx, color='white', linestyle='--', linewidth=2)
        ax.axhline(gy, color='white', linestyle='--', linewidth=2)
    ax.set_title(f'Gen-Matched Merged Electron: {_var_title(var)}')

    outdir = f'{outdir}/genmerged/2d'
    os.makedirs(outdir, exist_ok=True)
    plt.tight_layout()
    outpath = f'{outdir}/hist_{plottag}_{var}_{cut}_genmerged_2d.png'
    plt.savefig(outpath)
    plt.close(fig)
    print(f"Saved: {outpath}")


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
        plot_summarysimple(s_hists, b_hists, bkg_cats, s_pts, var, cut, plotdir)
        plot_summarysimple(s_hists, b_hists, bkg_cats, s_pts, var, cut, plotdir, logy=False)

    # ATgm_*: gen-matched merged electron, signal only, at gm_cut.
    gm_variables = sorted(_get_1d_hist_names(s_hists, prefix='ATgm_'))
    print(f"Plotting {len(gm_variables)} gen-matched merged-electron variables at cut='{gm_cut}': {gm_variables}")
    for var in gm_variables:
        plot_genmerged_total(s_hists, s_pts, var, gm_cut, plotdir)
    for var in gm_2d_variables:
        plot_genmerged_2d(s_hists, s_pts, var, gm_cut, plotdir)
