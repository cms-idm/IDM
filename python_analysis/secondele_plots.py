import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import mplhep as hep
import coffea.util as util
import analysisTools.utils as utils
import analysisTools.plotTools as ptools
# Hist definitions (variable lists, acceptance thresholds) come straight from
# the histo config so this script stays in sync with what was filled.
from configs.histo_configs import secondele as cfg

# samp list-selection is immediately summed, so dropped bins not going to overflow is fine
warnings.filterwarnings('ignore', message='List indexing selection is experimental')

ptools.set_plot_style()

# ---- Config -----------------------------------------------------------------
outdir    = 'workarea'
vers      = 'Jul2026noID'
selection = 'aptrk'
hists_tag = 'secondele'
cut       = 'cut8'
year      = 2024
# m1 values dropped from the all-samples sums (e.g. [0.05]; empty = keep all)
exclude_m1s = []

saved_signal_hists = f"{outdir}/hists_sig{vers}_{selection}-sel_{hists_tag}.coffea"
plottag = f'sig{vers}_{selection}-sel'
plotdir = 'plots/secondele'

# ---- Display metadata -------------------------------------------------------
_OBJ_TITLES = {
    'mele':   'Merged Electron',
    'refele': 'Reference Electron',
}
# overlay styles: merged candidate vs single-electron reference
_PREFIX_STYLES = {
    'mele':   {'label': 'Merged candidate',         'color': 'C0'},
    'refele': {'label': 'Reference (vertexed leg)', 'color': 'C1'},
}

# readable quantity names for titles; unlisted keys fall back to the axis label
_VAR_TITLES = {
    'scE_over_genEown':     'SC Energy / Own Gen Leg Energy',
    'scE_over_genEsum':     'SC Energy / Gen Pair Energy',
    'scRawE_over_genEsum':  'Raw SC Energy / Gen Pair Energy',
    'pt_over_genPtown':     r'$p_T$ / Own Gen Leg $p_T$',
    'pt_over_genPtsum':     r'$p_T$ / Gen Pair $p_T$',
    'scE_over_e':           'SC Energy / Electron Energy',
    'invEmP':               r'$|1/E - 1/p|$',
    'ownfrac':              'Own Gen Leg Energy Fraction',
    'gen_ee_dr':            r'Gen $\Delta R(e^+, e^-)$',
    'pt':                   r'$p_T$',
    'scClustersSize':       'SC Cluster Multiplicity',
    'full55e2ndOverScRawE': r'$E_{\rm 2nd} / E_{\rm SC}^{\rm raw}$',
    'full55eMaxOverScRawE': r'$E_{\rm max} / E_{\rm SC}^{\rm raw}$',
    'seedEoverScRawE':      r'$E_{\rm seed} / E_{\rm SC}^{\rm raw}$',
    'scPhiWidth':           r'SC $\phi$ Width',
    'scEtaWidth':           r'SC $\eta$ Width',
    'full55r9':             r'$R_9$ (Full 5x5)',
    'full55sigmaIetaIeta':  r'$\sigma_{i\eta i\eta}$ (Full 5x5)',
    'full55sigmaIphiIphi':  r'$\sigma_{i\phi i\phi}$ (Full 5x5)',
    'numTrackerHits':       'Tracker Hits',
    'numPixHits':           'Pixel Hits',
    'numStripHits':         'Strip Hits',
    'trkChi2':              r'Track $\chi^2/\mathrm{ndf}$',
    'trkProb':              r'Track $\chi^2$ Probability',
    'expMissingInnerHits':  'Expected Missing Inner Hits',
    'partner_pt':           r'Partner Gen Leg $p_T$',
    'partner_abseta':       r'Partner Gen Leg $|\eta|$',
    'partner_ptfrac':       r'Partner Gen Leg $p_T$ Fraction',
}
# x-axis (slicing variable) names for 2D titles, keyed by axis name
_AXIS_TITLES = {
    'ownfrac': 'Own Gen Leg Energy Fraction',
    'geneedr': r'Gen $\Delta R(e^+, e^-)$',
    'recopt':  r'Reco $p_T$',
    'lxy':     r'Gen $L_{xy}$',
    'abseta':  r'Partner Gen $|\eta|$',
}
# Axes binned ~logarithmically from 0: drawn on symlog x, linear below linthresh
# (= first bin edge above 0).
_SYMLOGX_AXES = {'geneedr': 0.01, 'lxy': 0.05}

# 1D vars drawn with log y
_LOGY_VARS = {'invEmP', 'trkChi2', 'expMissingInnerHits', 'scClustersSize', 'gen_ee_dr'}
# energy / pT ratios: guide line at ratio = 1
_RATIO_VARS = {'scE_over_genEown', 'scE_over_genEsum', 'scRawE_over_genEsum',
               'pt_over_genPtown', 'pt_over_genPtsum', 'scE_over_e'}

_ACC_LABELS = {
    'in_acc':    'Partner in acceptance',
    'fail_pt':   rf'Fails $p_T > {cfg._ACC_PT_MIN:g}$ GeV only',
    'fail_eta':  rf'Fails $|\eta| < {cfg._ACC_ETA_MAX:g}$ only',
    'fail_both': r'Fails $p_T$ and $|\eta|$',
}
_ACC_TICKS = {
    'in_acc':    'In\nacceptance',
    'fail_pt':   'Fails\n$p_T$ only',
    'fail_eta':  r'Fails' + '\n' + r'$|\eta|$ only',
    'fail_both': 'Fails\nboth',
}
_ACC_ORDER  = ['in_acc', 'fail_pt', 'fail_eta', 'fail_both']
_ACC_COLORS = {c: ptools.cms_colors(len(_ACC_ORDER))[i] for i, c in enumerate(_ACC_ORDER)}

_count_label = 'Electrons' if cfg.RAW_COUNTS else 'Weighted Electrons'

# ---- Load -------------------------------------------------------------------
s_hists = util.load(saved_signal_hists)[0]
s_pts   = utils.get_signal_point_dict(s_hists)
_excl   = set(s_pts.name[np.isin(np.round(s_pts.m1, 4), exclude_m1s)])
_samps  = [s for s in s_pts.name if s not in _excl]
if _excl:
    print(f"Excluded {len(_excl)} samples with m1 in {exclude_m1s}")

for sub in ('1d', '2d', 'partner'):
    os.makedirs(f'{plotdir}/{sub}', exist_ok=True)


# ---- Helpers ----------------------------------------------------------------
def _get(name):
    """Hist `name` at `cut`, summed over the kept signal samples (None if absent)."""
    if name not in s_hists:
        print(f"Missing hist {name}, skipping")
        return None
    h = s_hists[name]
    if len(h.axes['cut']) == 0:
        print(f"Unfilled hist {name} (field absent from ntuples?), skipping")
        return None
    if cut not in h.axes['cut']:
        raise KeyError(f"cut {cut!r} not in {name}; available: {list(h.axes['cut'])}")
    h = h[{'cut': cut}]
    present = [s for s in _samps if s in h.axes['samp']]
    return h[{'samp': present}][{'samp': sum}]


def _var_title(key, axis=None):
    return _VAR_TITLES.get(key, axis.label if axis is not None else key)


def _apply_xscale(ax, axis_name):
    if axis_name in _SYMLOGX_AXES:
        ax.set_xscale('symlog', linthresh=_SYMLOGX_AXES[axis_name], linscale=1)


def _place_legend_fixed(ax, **kw):
    """Legend for a fixed y-range (fractions / 2D maps): first clear corner of
    upper right -> lower left -> upper left -> lower right, else place_legend."""
    for loc in ('upper right', 'lower left', 'upper left', 'lower right'):
        leg = ax.legend(loc=loc, **kw)
        if not ptools.legend_covers_data(ax, leg):
            return leg
        leg.remove()
    return ptools.place_legend(ax, **kw)


def _finish(fig, ax, outpath, xlabel, ylabel, title, logy=False, legend='headroom', **legkw):
    """Labels, title, log y, legend ('headroom' / 'fixed' / None), save, close."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if logy:
        ax.set_yscale('log')
    plt.tight_layout()
    if legend == 'headroom':
        ptools.place_legend(ax, **legkw)
    elif legend == 'fixed':
        _place_legend_fixed(ax, **legkw)
    plt.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"Saved: {outpath}")


def _normalize(h):
    """Unit-area normalized (values, edges, ylabel). Uniform bins: fraction per
    bin; variable bins: fraction / bin width, so the shape isn't distorted."""
    vals, edges = h.to_numpy()
    widths = np.diff(edges)
    total = vals.sum()
    if np.allclose(widths, widths[0]):
        return vals / total, edges, 'Fraction of Electrons'
    return vals / total / widths, edges, 'Fraction / Bin Width'


def _binomial(num, den):
    with np.errstate(divide='ignore', invalid='ignore'):
        f = np.where(den > 0, num / den, np.nan)
        e = np.where(den > 0, np.sqrt(f * (1 - f) / den), np.nan)
    return f, e


# ---- (1, 2, 4) 1D: merged candidate vs reference ----------------------------
_1d_keys = ([k for k, _ in cfg._ENERGY_VARS] + ['ownfrac', 'gen_ee_dr', 'pt']
            + [f for f, _ in cfg._SHAPE_VARS] + [f for f, _ in cfg._TRK_VARS])

for key in _1d_keys:
    hs = {p: _get(f'{p}_{key}') for p in _PREFIX_STYLES}
    hs = {p: h for p, h in hs.items() if h is not None}
    if not hs or all(h.values().sum() == 0 for h in hs.values()):
        print(f"Skipping empty {key}")
        continue
    axis = next(iter(hs.values())).axes[0]
    fig, ax = ptools.new_fig(year)
    ylabel = None
    for p, h in hs.items():
        n = h.values().sum()
        if n == 0:
            print(f"Skipping empty {p}_{key}")
            continue
        y, edges, ylabel = _normalize(h)
        st = _PREFIX_STYLES[p]
        hep.histplot(y, edges, ax=ax, histtype='step', linewidth=2,
                     color=st['color'], label=f"{st['label']} (N = {n:.0f})")
    if key in _RATIO_VARS:
        ax.axvline(1, color='black', ls=':', lw=1.5, label='Ratio = 1')
    _apply_xscale(ax, axis.name)
    _finish(fig, ax, f'{plotdir}/1d/hist_{plottag}_{key}_{cut}_mele_vs_refele.png',
            axis.label, ylabel,
            f'Merged vs. Reference Electron: {_var_title(key, axis)}',
            logy=key in _LOGY_VARS)


# ---- (1, 2, 4) 2D colormaps, one per prefix ---------------------------------
def _guides_ownfrac(ax, key):
    """Hypothesis lines on the energy-ratio vs own-leg-fraction maps:
    one leg reconstructed vs. full pair in the SC."""
    if key in ('scE_over_genEsum', 'pt_over_genPtsum'):
        ax.axline((0, 0), slope=1, color='black', ls=':', lw=2, label='One leg ($y = x$)')
        ax.axhline(1, color='black', ls='--', lw=2, label='Full pair ($y = 1$)')
        return True
    if key == 'scE_over_genEown':
        ax.axhline(1, color='black', ls=':', lw=2, label='One leg ($y = 1$)')
        x = np.linspace(1 / ax.get_ylim()[1], 1, 200)
        ax.plot(x, 1 / x, color='black', ls='--', lw=2, label='Full pair ($y = 1/x$)')
        return True
    return False


_2d_specs = []  # (hist name, prefix, y-var key, x axis name)
for p in _PREFIX_STYLES:
    _2d_specs += [(f'{p}_{k}_vs_ownfrac', p, k, 'ownfrac') for k in cfg._ENERGY_VS_FRAC]
    _2d_specs += [(f'{p}_{f}_vs_{x}', p, f, x) for f, _ in cfg._SHAPE_VARS for x in ('geneedr', 'recopt')]
    _2d_specs += [(f'{p}_{f}_vs_geneedr', p, f, 'geneedr') for f, _ in cfg._TRK_VARS]
_2d_specs.append(('mele_partner_pt_vs_abseta', 'mele', 'partner_pt', 'abseta'))

for name, p, key, xname in _2d_specs:
    h = _get(name)
    if h is None:
        continue
    if h.values().sum() == 0:
        print(f"Skipping empty {name}")
        continue
    xaxis, yaxis = h.axes[0], h.axes[1]
    fig, ax = ptools.new_fig(year, cms_label=False)
    art = hep.hist2dplot(h, ax=ax, norm=LogNorm(), flow=None)
    art.cbar.set_label(_count_label)
    ptools.add_cms_label(ax, year)
    _apply_xscale(ax, xname)

    has_leg = False
    if xname == 'ownfrac':
        has_leg = _guides_ownfrac(ax, key)
    elif name == 'mele_partner_pt_vs_abseta':
        # acceptance boundaries (cut lines on a 2D map: white dashed)
        ax.axvline(cfg._ACC_ETA_MAX, color='white', ls='--', lw=2)
        ax.axhline(cfg._ACC_PT_MIN,  color='white', ls='--', lw=2)

    title = f'{_OBJ_TITLES[p]}: {_var_title(key, yaxis)}\nvs {_AXIS_TITLES.get(xname, xaxis.label)}'
    _finish(fig, ax, f'{plotdir}/2d/hist_{plottag}_{name}_{cut}.png',
            xaxis.label, yaxis.label, title,
            legend='fixed' if has_leg else None, fontsize=20)


# ---- (3a) Partner gen leg -----------------------------------------------------
# Inclusive acceptance fractions
h = _get('mele_partner_acc')
if h is not None and h.values().sum() > 0:
    _vals  = dict(zip(h.axes['acc'], h.values()))
    counts = np.array([_vals.get(c, 0.) for c in _ACC_ORDER], dtype=float)
    n_tot = counts.sum()
    frac, err = _binomial(counts, n_tot)
    acc_incl = dict(zip(_ACC_ORDER, frac))
    fig, ax = ptools.new_fig(year)
    x = np.arange(len(_ACC_ORDER))
    ax.bar(x, frac, yerr=err, color=[_ACC_COLORS[c] for c in _ACC_ORDER],
           edgecolor='black', linewidth=0.8,
           error_kw={'ecolor': 'black', 'capsize': 4, 'elinewidth': 1.2})
    for xi, f in zip(x, frac):
        ax.text(xi, f + 0.02, f'{100 * f:.1f}%', ha='center', va='bottom', fontsize=22)
    ax.set_xticks(x)
    ax.set_xticklabels([_ACC_TICKS[c] for c in _ACC_ORDER], fontsize=22)
    ax.set_ylim(0, 1.1)
    _finish(fig, ax, f'{plotdir}/partner/hist_{plottag}_partner_acc_{cut}.png',
            '', 'Fraction of Merged Candidates',
            f'Merged Electron: Partner Gen Leg Acceptance\n(N = {n_tot:.0f})',
            legend=None)
else:
    acc_incl = {}
    print("Skipping empty mele_partner_acc")

# Acceptance fractions vs Lxy / gen dR: per-bin fraction of each category
for xname in ('lxy', 'geneedr'):
    name = f'mele_partner_acc_vs_{xname}'
    h = _get(name)
    if h is None:
        continue
    if h.values().sum() == 0:
        print(f"Skipping empty {name}")
        continue
    xaxis = h.axes[0]
    edges = xaxis.edges
    n_all = sum(h[{'acc': c}].values() for c in _ACC_ORDER)
    fig, ax = ptools.new_fig(year)
    for c in _ACC_ORDER:
        f, e = _binomial(h[{'acc': c}].values(), n_all)
        lbl = _ACC_LABELS[c] + (f' ({100 * acc_incl[c]:.1f}%)' if c in acc_incl else '')
        hep.histplot(f, edges, yerr=e, ax=ax, histtype='step', linewidth=2,
                     color=_ACC_COLORS[c], label=lbl)
    _apply_xscale(ax, xname)
    ax.set_ylim(0, 1.05)
    _finish(fig, ax, f'{plotdir}/partner/hist_{plottag}_partner_acc_vs_{xname}_{cut}.png',
            xaxis.label, f'Fraction of Merged Candidates in Bin',
            f'Merged Electron: Partner Gen Leg Acceptance\nvs {_AXIS_TITLES[xname]}',
            legend='fixed', fontsize=20)

# Partner gen leg kinematics (1D)
_partner_guides = {
    'partner_pt':     cfg._ACC_PT_MIN,
    'partner_abseta': cfg._ACC_ETA_MAX,
}
for key in ('partner_pt', 'partner_abseta', 'partner_ptfrac'):
    h = _get(f'mele_{key}')
    if h is None:
        continue
    n = h.values().sum()
    if n == 0:
        print(f"Skipping empty mele_{key}")
        continue
    y, edges, ylabel = _normalize(h)
    fig, ax = ptools.new_fig(year)
    hep.histplot(y, edges, ax=ax, histtype='step', linewidth=2, color='C0',
                 label=f'Merged candidate (N = {n:.0f})')
    if key in _partner_guides:
        ax.axvline(_partner_guides[key], color='black', ls='--', lw=1.5,
                   label='Acceptance boundary')
    _finish(fig, ax, f'{plotdir}/partner/hist_{plottag}_{key}_{cut}.png',
            h.axes[0].label, ylabel, f'Merged Electron: {_var_title(key)}')
