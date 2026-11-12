import numpy as np
import matplotlib.pyplot as plt
import hist
import coffea.util as util
import analysisTools.utils as utils
import analysisTools.plotTools as ptools
from trainBDT.feature_sets import FEATURE_SETS
import os

# Smaller fonts than the house default: the matrix has one tick label per
# variable and long two-line titles.
ptools.set_plot_style(**{
    'font.size':        20,
    'axes.titlesize':   24,
    'axes.labelsize':   20,
    'xtick.labelsize':  16,
    'ytick.labelsize':  16,
    'figure.titlesize': 24,
})

outdir    = 'workarea'
sig_vers  = 'Jul2026noID'
bkg_vers  = 'Aug2026'
selection = 'aptrk'
hists_tag = 'corrmatrix'
saved_signal_hists = f"{outdir}/hists_sig{sig_vers}_{selection}-sel_{hists_tag}.coffea"
saved_bkg_hists    = f"{outdir}/hists_bkg{bkg_vers}_{selection}-sel_{hists_tag}.coffea"
# cut3 = jet cuts, the last cut before the good-vertex requirement (cut4), so
# both matrices still see their full category populations
cut    = 'cut9'
year   = 2024

plotdir = 'plots/corrmatrix'
os.makedirs(plotdir, exist_ok=True)

# ── Metadata ───────────────────────────────────────────────────────────────────

# hist key -> (file tag, title label); see configs/histo_configs/corrmatrix.py
categories = {
    'corr_vtx':  ('vtx',  'Vertexed (lowest-$\\chi^2$ vertex)'),
    'corr_mele': ('mele', 'Merged electron'),
}

# hist key -> BDT feature sets (trainBDT/feature_sets.py) to also plot as
# matrices restricted to just those variables
feature_subsets = {
    'corr_mele': ['merged7'],
}

# individual signal points to plot: (m1, delta, ctau)
indiv_samps = [
    (0.5, 0.1, 10),
    (5,   0.1, 1),
    (5,   0.1, 10),
    (5,   0.1, 100),
    (50,  0.1, 10),
]

_CONST = '_const'


def _sig_label(m1, delta, ctau):
    return rf'$M_1$={m1}, $\delta$={ptools.fmt_delta(delta)}, $c\tau$={ctau} mm'


def _sig_tag(m1, delta, ctau):
    return f'm1-{m1}_delta-{delta}_ctau-{ctau}'.replace('.', 'p')


# ── Moment-matrix decoding ─────────────────────────────────────────────────────

def _moments(h, samp):
    """Moment matrix M for one sample at the chosen cut (see corrmatrix.py)."""
    return h[{'samp': hist.loc(samp), 'cut': hist.loc(cut)}].values()


def _corr_from_moments(M, names):
    """Weighted Pearson correlation from a moment matrix; returns (names, corr)."""
    k = names.index(_CONST)
    keep = [i for i in range(len(names)) if i != k]
    S = M / M[k, k]
    mean = S[keep, k]
    cov = S[np.ix_(keep, keep)] - np.outer(mean, mean)
    sd = np.sqrt(np.diag(cov))
    with np.errstate(invalid='ignore', divide='ignore'):
        corr = cov / np.outer(sd, sd)
    return [names[i] for i in keep], corr


def _aggregate_moments(h, samps=None, normalize=True):
    """
    Pool samples (default: all on the samp axis). With normalize=True each sample
    is scaled to unit total weight so every signal point contributes equally (raw
    xsec weights would let the light-mass points dominate). For background use
    normalize=False, so the pooled moments follow the physical xsec-weighted mix.
    """
    names = list(h.axes['var_i'])
    k = names.index(_CONST)
    on_axis = set(h.axes['samp'])
    samps = list(h.axes['samp']) if samps is None else [s for s in samps if s in on_axis]
    M_tot = np.zeros((len(names), len(names)))
    n_used = 0
    for samp in samps:
        M = _moments(h, samp)
        if M[k, k] > 0:
            M_tot += M / M[k, k] if normalize else M
            n_used += 1
    return M_tot, n_used


def _feature_to_var(feat):
    """BDT feature name -> corrmatrix variable name (merged features drop 'mele_')."""
    return feat[len('mele_'):] if feat.startswith('mele_') else feat


def _subset_corr(names, corr, feature_set):
    """Restrict (names, corr) to the variables of a FEATURE_SETS entry, in its order."""
    vars_ = [_feature_to_var(f) for f in FEATURE_SETS[feature_set]]
    missing = [v for v in vars_ if v not in names]
    if missing:
        raise KeyError(f'{feature_set}: variables {missing} not in the corrmatrix hist')
    idx = [names.index(v) for v in vars_]
    return vars_, corr[np.ix_(idx, idx)]


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_corr(names, corr, title, fname):
    n = len(names)
    size = max(12, 0.55 * n + 4)
    fig, ax = ptools.new_fig(year, cms_label=False, figsize=(size + 2, size))

    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Weighted Pearson correlation')
    ptools.add_cms_label(ax, year)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=90)
    ax.set_yticklabels(names)

    # annotate cells; white text on strongly-colored cells
    fs = 11 if n <= 20 else 8
    for i in range(n):
        for j in range(n):
            c = corr[i, j]
            if not np.isfinite(c):
                continue
            ax.text(j, i, f'{c:.2f}', ha='center', va='center', fontsize=fs,
                    color='white' if abs(c) > 0.6 else 'black')

    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(f'{plotdir}/{fname}.png', dpi=150)
    plt.close(fig)
    print(f'Saved {plotdir}/{fname}.png')


def plot_corr_views(key, names, corr, title, fname):
    """
    Plot the full matrix, plus one per feature set in feature_subsets[key].
    title(label) and fname(tag) build the title/filename from the category label
    and file tag, so subset plots get e.g. 'Merged electron, merged7 features'
    and 'mele-merged7'.
    """
    tag, cat_label = categories[key]
    plot_corr(names, corr, title(cat_label), fname(tag))
    for fset in feature_subsets.get(key, []):
        sub_names, sub_corr = _subset_corr(names, corr, fset)
        plot_corr(sub_names, sub_corr,
                  title(f'{cat_label}, {fset} features'), fname(f'{tag}-{fset}'))


def plot_signal(s_hists):
    s_pts    = utils.get_signal_point_dict(s_hists)
    cut_desc = s_hists['cutDesc'][cut]

    for key, (tag, cat_label) in categories.items():
        h = s_hists[key]
        names_all = list(h.axes['var_i'])

        # all samples, aggregated
        M_tot, n_used = _aggregate_moments(h)
        names, corr = _corr_from_moments(M_tot, names_all)
        plot_corr_views(key, names, corr,
                        lambda lbl: f'{lbl}, all signal ({n_used} pts)\n{year}, {cut}: {cut_desc}',
                        lambda t: f'corr_sig_{t}_{cut}_allsamps')

        # individual samples
        for m1, delta, ctau in indiv_samps:
            samp = ptools.samp_name(s_pts, m1, delta, ctau)
            if samp is None:
                print(f'No sample for m1={m1}, delta={delta}, ctau={ctau}; skipping')
                continue
            M = _moments(h, samp)
            if M[names_all.index(_CONST), names_all.index(_CONST)] <= 0:
                print(f'{key}: {samp} has no entries at {cut}; skipping')
                continue
            names, corr = _corr_from_moments(M, names_all)
            plot_corr_views(key, names, corr,
                            lambda lbl: f'{lbl}, signal {_sig_label(m1, delta, ctau)}\n{year}, {cut}: {cut_desc}',
                            lambda t: f'corr_sig_{t}_{cut}_{_sig_tag(m1, delta, ctau)}')


def plot_background(b_hists):
    bkg_cats, _ = utils.bkg_categories(b_hists['cutflow'])
    cut_desc    = b_hists['cutDesc'][cut]

    for key, (tag, cat_label) in categories.items():
        h = b_hists[key]
        names_all = list(h.axes['var_i'])

        # total background, xsec-weighted
        M_tot, n_used = _aggregate_moments(h, normalize=False)
        names, corr = _corr_from_moments(M_tot, names_all)
        plot_corr_views(key, names, corr,
                        lambda lbl: f'{lbl}, total background ({n_used} samples)\n{year}, {cut}: {cut_desc}',
                        lambda t: f'corr_bkg_{t}_{cut}_total')

        # each background process, xsec-weighted over its subsamples
        for bkg_cat, samps in sorted(bkg_cats.items()):
            M_cat, n_used = _aggregate_moments(h, samps, normalize=False)
            if n_used == 0:
                print(f'{key}: {bkg_cat} has no entries at {cut}; skipping')
                continue
            names, corr = _corr_from_moments(M_cat, names_all)
            plot_corr_views(key, names, corr,
                            lambda lbl: f'{lbl}, {bkg_cat} background ({n_used} samples)\n{year}, {cut}: {cut_desc}',
                            lambda t: f'corr_bkg_{t}_{cut}_{bkg_cat}')


if os.path.exists(saved_signal_hists):
    plot_signal(util.load(saved_signal_hists)[0])
else:
    print(f'{saved_signal_hists} not found; skipping signal')

if os.path.exists(saved_bkg_hists):
    plot_background(util.load(saved_bkg_hists)[0])
else:
    print(f'{saved_bkg_hists} not found; skipping background')
