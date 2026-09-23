import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import coffea.util as util
import analysisTools.utils as utils
import os

plt.rcParams.update({
    'font.size':        20,
    'axes.titlesize':   24,
    'axes.labelsize':   20,
    'xtick.labelsize':  20,
    'ytick.labelsize':  20,
    'legend.fontsize':  18,
    'figure.titlesize': 24,
})

outdir    = 'workarea'
vers      = 'Jul2026noID'
selection = 'an'
seltag    = 'minjdphi'
cut       = 'cut8'
year      = 2024
size      = (16, 12)

# Signal-only: no background run currently targets the mergedcats/resolvedcats
# hist configs (histmaker_background.py is on 'appearingtrack'), so there is
# no background-equivalent file to merge in here.
merged_file   = f"{outdir}/hists_sig{vers}_{selection}-sel_mergedcats.coffea"
resolved_file = f"{outdir}/hists_sig{vers}_{selection}-sel_resolvedcats.coffea"

m_hists = util.load(merged_file)[0]
r_hists = util.load(resolved_file)[0]
# mele_*/mpho_*/zero_* (mergedcats.py) and res_*/vtx_* (resolvedcats.py) never
# collide -- only generic bookkeeping keys (cutflow, cutDesc, ...) are shared
# between the two files, and those are equivalent (same samples/cuts) either way.
s_hists = {**m_hists, **r_hists}

s_pts = utils.get_signal_point_dict(s_hists)

plotdir = 'plots/allcats'
os.makedirs(plotdir, exist_ok=True)

# ── Metadata ──────────────────────────────────────────────────────────────────

# Five reco categories, distinguished by color/line style when overlaid.
# 'vtx' (vertexed) supersedes the other four -- see computeMergedCatVars
# (analysisTools/analysisSubroutines.py).
cat_styles = {
    'vtx':  {'label': 'Vertexed',        'color': 'C4', 'ls': '-'},
    'mele': {'label': 'Merged electron', 'color': 'C0', 'ls': '-'},
    'mpho': {'label': 'Merged photon',   'color': 'C1', 'ls': '--'},
    'res':  {'label': 'Resolved',        'color': 'C2', 'ls': '-.'},
    'zero': {'label': 'Zero match',      'color': 'C3', 'ls': ':'},
}

# Same signal-point series as scripts/genstudy/bryangenstudy_plots.py, used
# for the per-series breakdowns of the two "reason" plots (Sections 2/3).
PARAM_SETS = [
    dict(m1s=[0.5, 1, 2, 5], deltas=[0.1],            ctaus=[10]),
    dict(m1s=[5],            deltas=[0.05, 0.1, 0.2],  ctaus=[10]),
    dict(m1s=[0.5],          deltas=[0.05, 0.1, 0.2],  ctaus=[10]),
    dict(m1s=[5],            deltas=[0.1],             ctaus=[1, 10, 100]),
    dict(m1s=[0.5],          deltas=[0.1],             ctaus=[1, 10, 100]),
]


def _samp_name(m1, delta, ctau):
    row = s_pts[np.isclose(s_pts.m1, m1) & np.isclose(s_pts.delta, delta) & (s_pts.ctau == ctau)]
    return row['name'].iloc[0] if not row.empty else None


def _sig_label(m1, delta, ctau):
    return rf'$M_1$={m1}, $\Delta$={delta}, $c\tau$={ctau} mm'


def _param_tag(m1s, deltas, ctaus):
    fmt = lambda vals: "-".join(utils.stringfy_friendly(v) for v in vals)
    return f"m1-{fmt(m1s)}_delta-{fmt(deltas)}_ctau-{fmt(ctaus)}"


def _series_points(pset):
    """Every (m1, delta, ctau) combo covered by a PARAM_SETS entry -- mirrors
    bryangenstudy_plots.py's nested m1/delta/ctau loop."""
    for m1 in pset['m1s']:
        for delta in pset['deltas']:
            for ctau in pset['ctaus']:
                yield m1, delta, ctau


def _varbin_histplot(h, ax, **kwargs):
    """Density-correct plot for variable-width bins (area = 1)."""
    vals, edges = h.to_numpy()
    widths = np.diff(edges)
    area = float(np.sum(vals * widths))
    dens = vals / area / widths if area > 0 else np.zeros_like(vals)
    hep.histplot((dens, edges), ax=ax, **kwargs)


def _histplot(h, ax, varbin=False, **kwargs):
    """Unified plot: variable-bin density or regular density."""
    if varbin:
        _varbin_histplot(h, ax, **kwargs)
    else:
        if np.sum(h.values()) == 0:
            return
        hep.histplot(h, ax=ax, density=True, **kwargs)


def _select(hkey, samp_sel):
    """s_hists[hkey] sliced to (cut, samp_sel), or None if that hist doesn't
    exist or was never filled at this cut stage (a StrCategory with
    growth=True only gains a category label once something is filled into
    it, so a rare category/sample combo can simply be absent from the cut
    axis -- not just empty)."""
    if hkey not in s_hists:
        return None
    h = s_hists[hkey]
    if cut not in list(h.axes['cut']):
        return None
    return h[{'cut': cut, 'samp': samp_sel}]


# ── Section 1: category overlays, one plot per variable (all-samples only) ────
# One "_cats_" plot per variable present in the hists, overlaying every
# category that has a matching histogram. cat_prefix maps a category name to
# the hist-key prefix to use for that section -- reco/nearestEle sections use
# only the *leading* electron for the two-electron categories (res, vtx), per
# instructions; gen-level sections use the bare per-category prefix (an event
# has exactly one gen ee pair regardless of category).

def _cats_overlay(field, cat_prefix, field_override=None, doLogy=False, varbin=False,
                   fname_key=None, title_prefix=''):
    """One category-overlay plot for `field`. cat_prefix: {cat: hist-key
    prefix}. field_override: optional {cat: field-name-to-use-instead}, for
    the handful of fields where mpho (Photon) uses a different literal field
    name than the AllLptElectron-based categories (E vs energy, HoverE vs HoE).
    The plot title/xlabel are taken from the histogram's own axis label
    (set in mergedcats.py/resolvedcats.py), not re-derived from `field`."""
    field_override = field_override or {}
    fname_key = fname_key or field

    fig, ax = plt.subplots(figsize=size)
    hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
    any_drawn = False
    xlabel = None
    for cat, prefix in cat_prefix.items():
        fname = field_override.get(cat, field)
        hkey = f'{prefix}_{fname}'
        h = _select(hkey, sum)
        if h is None or np.sum(h.values()) == 0:
            continue
        if xlabel is None:
            xlabel = h.axes[-1].label
        cstyle = cat_styles[cat]
        _histplot(h, ax, varbin=varbin, histtype='step', label=cstyle['label'],
                  color=cstyle['color'], linestyle=cstyle['ls'], linewidth=2)
        any_drawn = True
    if not any_drawn:
        plt.close(fig)
        return False

    xlabel = xlabel or field
    ax.set_xlabel(xlabel)
    ax.set_ylabel('A.U.')
    if doLogy:
        ax.set_yscale('log')
    ax.set_title(rf'{title_prefix}{xlabel} — all samples')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{plotdir}/hist_{seltag}_{fname_key}_cats_allsamps.png')
    plt.close(fig)
    return True


# ── 1a: Gen ee kinematics -- every category (incl. zero match) ────────────────
_GEN_PREFIX = {'zero': 'zero', 'mele': 'mele', 'mpho': 'mpho', 'res': 'res', 'vtx': 'vtx'}
_GEN_FIELDS = [
    ('gen_lxy',         True,  True),
    ('gen_ee_pt',       False, False),
    ('gen_lead_pt',     False, False),
    ('gen_sublead_pt',  False, False),
    ('gen_ee_dr',       True,  False),
    ('gen_ee_eta',      False, False),
    ('gen_lead_eta',    False, False),
    ('gen_sublead_eta', False, False),
]

print("Section 1a: gen ee kinematics, all categories (all-samples)")
n_made = 0
for field, doLogy, varbin in _GEN_FIELDS:
    n_made += _cats_overlay(field, _GEN_PREFIX, doLogy=doLogy, varbin=varbin,
                             fname_key=f'gen_{field}', title_prefix='Merged categories: gen ')
print(f"  -> {n_made} plots")

# ── 1b: Per-reco-object variables -- mele/mpho use their single reco object,
# res/vtx use only the leading (by reco pt) electron ─────────────────────────
_RECO_PREFIX = {'mele': 'mele', 'mpho': 'mpho', 'res': 'res_lead', 'vtx': 'vtx_ele_lead'}

# Shared AllLptElectron-style fields (mele/res/vtx all fill these from the same
# _ELE_HISTS list). 'ID' is dropped -- it and 'IDscore' both read the same
# underlying field (mergedcats.py/resolvedcats.py _FIELD_ALIASES), so keeping
# both would violate "exactly one plot per variable"; IDscore is kept since
# it's the one available under all four categories (vtx_ele's _VTXELE_EXCLUDE
# drops the plain 'ID' key but keeps IDscore).
_ELE_FIELDS = [
    ('pt',                  False),
    ('eta',                 False),
    ('phi',                 False),
    ('e',                   False),  # mpho: 'energy' (see _FIELD_OVERRIDE)
    ('angRes',              False),
    ('dxy',                 True),
    ('dz',                  False),
    ('numTrackerHits',      False),
    ('numPixHits',          False),
    ('numStripHits',        False),
    ('charge',              False),
    ('minDRtoReg',          True),   # AllLptElectron-only: mele/res, not mpho/vtx
    ('drNearestEle',        True),
    ('mindRj',              True),
    ('mindPhiJ',            False),
    ('full5x5sigmaIetaIeta', False),
    ('HoverE',              True),   # mpho: 'HoE' (see _FIELD_OVERRIDE)
    ('abs1overEm1overP',    True),   # AllLptElectron-only: mele/res, not mpho/vtx
    ('expMissingInnerHits', False),
    ('IDscore',             False),
]
_FIELD_OVERRIDE = {
    'e':      {'mpho': 'energy'},
    'HoverE': {'mpho': 'HoE'},
}

# Photon-only fields (no AllLptElectron analog -- single-category plots).
_MPHO_ONLY_FIELDS = [
    ('r9',           False),
    ('full5x5_r9',   False),
    ('sIeIe',        False),
    ('full5x5_HoE',  True),
]

# Vertexed-only per-leg fields: the vertex-constrained refit quantities plus
# the leg-type/gen-match flags (_VTXLEG_HISTS, resolvedcats.py) -- no
# mele/mpho/res analog since these require an actual vertex fit.
_VTX_ONLY_LEG_FIELDS = [
    ('typ',          False),
    ('isMatched',    False),
    ('matchType',    False),
    ('refit_dxy',    True),
    ('refit_dxyErr', False),
    ('refit_dz',     False),
    ('refit_dzErr',  False),
    ('refit_chi2',   True),
]

print("Section 1b: per-reco-object variables (leading electron for res/vtx), all categories (all-samples)")
n_made = 0
for field, doLogy in _ELE_FIELDS:
    n_made += _cats_overlay(field, _RECO_PREFIX, field_override=_FIELD_OVERRIDE, doLogy=doLogy,
                             fname_key=f'reco_{field}', title_prefix='Merged categories: reco ')
for field, doLogy in _MPHO_ONLY_FIELDS:
    n_made += _cats_overlay(field, {'mpho': 'mpho'}, doLogy=doLogy,
                             fname_key=f'reco_{field}', title_prefix='Merged photon: reco ')
for field, doLogy in _VTX_ONLY_LEG_FIELDS:
    n_made += _cats_overlay(field, {'vtx': 'vtx_ele_lead'}, doLogy=doLogy,
                             fname_key=f'reco_{field}', title_prefix='Vertexed leading leg: ')
print(f"  -> {n_made} plots")

# ── 1c: Pair/vertex-level variables -- res (pair 4-vector, no fit) vs vtx
# (fitted vertex); plus vtx-only quantities with no fitted-vertex-free analog ─
_PAIR_PREFIX = {'res': 'res_ee', 'vtx': 'vtx'}
_PAIR_FIELDS = [
    ('dr',      True,  False),
    ('sign',    False, False),
    ('eleDphi', False, False),
    ('mass',    False, False),
    ('pt',      False, False),
    ('eta',     False, False),
    ('phi',     False, False),
    ('METdPhi', False, False),
]

_VTX_ONLY_PAIR_FIELDS = [
    ('vtx_chi2',                    True),
    ('vtx_min_dxy',                 True),
    ('vtx_vxy',                     True),
    ('vtx_sigmavxy',                True),
    ('vtx_vz',                      True),
    ('vtx_prob',                    False),
    ('vtx_dr_raw',                  True),
    ('vtx_min_dxy_raw',             True),
    ('vtx_mass_raw',                False),
    ('vtx_pt_raw',                  False),
    ('vtx_eta_raw',                 False),
    ('vtx_phi_raw',                 False),
    ('vtx_mindRj',                  True),
    ('vtx_mindPhiJ',                False),
    ('vtx_cos_collinear',           False),
    ('vtx_cos_collinear_fromPV',    False),
    ('vtx_cos_collinear_fromPV_refit', False),
    ('vtx_gen_cos_collinear_fromPV', False),
    ('vtx_projectedLxy',            True),
    ('vtx_vxy_fromPV',              True),
    ('vtx_typ',                     False),
    ('vtx_isGood',                  False),
    ('vtx_isMatched',               False),
    ('vtx_matchSign',               False),
    ('vtx_bothElePassID',           False),
    ('vtx_bothElePassIDBasic',      False),
]

print("Section 1c: pair/vertex-level variables (all-samples)")
n_made = 0
for field, doLogy, varbin in _PAIR_FIELDS:
    n_made += _cats_overlay(field, _PAIR_PREFIX, doLogy=doLogy, varbin=varbin,
                             fname_key=f'pair_{field}', title_prefix='Resolved vs. vertexed: ')
for hkey, doLogy in _VTX_ONLY_PAIR_FIELDS:
    h = _select(hkey, sum)
    if h is None or np.sum(h.values()) == 0:
        continue
    fig, ax = plt.subplots(figsize=size)
    hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
    xlabel = h.axes[-1].label
    _histplot(h, ax, histtype='step', color=cat_styles['vtx']['color'], linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('A.U.')
    if doLogy:
        ax.set_yscale('log')
    ax.set_title(rf'{xlabel} — all samples')
    plt.tight_layout()
    plt.savefig(f'{plotdir}/hist_{seltag}_{hkey}_cats_allsamps.png')
    plt.close(fig)
    n_made += 1
print(f"  -> {n_made} plots")

# ── 1d: Nearest-electron realism-check variables -- same field-overlay logic
# as 1b, but for the single nearest *other* AllLptElectron next to each
# category's selected object (mele_nearestEle_*/mpho_nearestEle_*/
# res_lead_nearestEle_*/vtx_ele_lead_nearestEle_*). The nearest object is
# always an AllLptElectron here (even for mpho), so no field-name overrides
# are needed like in 1b. ──────────────────────────────────────────────────────
_NEARESTELE_PREFIX = {
    'mele': 'mele_nearestEle', 'mpho': 'mpho_nearestEle',
    'res':  'res_lead_nearestEle', 'vtx': 'vtx_ele_lead_nearestEle',
}

print("Section 1d: nearest-electron realism-check variables, all categories (all-samples)")
n_made = 0
for field, doLogy in _ELE_FIELDS:
    n_made += _cats_overlay(field, _NEARESTELE_PREFIX, doLogy=doLogy,
                             fname_key=f'nearestEle_{field}', title_prefix='Nearest-electron realism check: ')
print(f"  -> {n_made} plots")

# ── 1e: Nearest-electron *relational* variables (selected object vs. its
# nearest other AllLptElectron) -- only defined for mele and vtx (both legs). ─
_RELATIONAL_FIELDS = [
    ('dPt',        False),
    ('dPtRel',     False),
    ('chargeProd', False),
    ('dVxy',       True),
    ('dVz',        True),
    ('dDxy',       True),
    ('dDz',        True),
    ('drZoom',     False),
]

print("Section 1e: nearest-electron relational variables (all-samples)")
n_made = 0
for field, doLogy in _RELATIONAL_FIELDS:
    n_made += _cats_overlay(field, _NEARESTELE_PREFIX, doLogy=doLogy,
                             fname_key=f'nearestEle_{field}', title_prefix='Nearest-electron relational: ')
print(f"  -> {n_made} plots")


# ── Section 2: category fractions (zero/mele/mpho/res/vtx) ───────────────────
# Same style as mergedmatch_plots.py's Section 8 (hist_minjdphi_reco_cat_alpt_*):
# a fraction-of-events bar chart with Poisson error bars, one all-samples
# version plus one grouped-bar version per PARAM_SETS series. Each category's
# event count comes from the sum of its {cat}_gen_lxy histogram -- every event
# in a category fills that hist exactly once, so its total is exactly that
# category's (raw, unweighted) event count at this cut stage. Errors are
# simple Poisson (sqrt(N)/total): RAW_COUNTS=True in both hist configs means
# these are unweighted counts (Double storage, no separate variance to read
# back, unlike mergedmatch_plots.py's Weight-storage reco_cat hist).

_CAT_ORDER = ['zero', 'mele', 'mpho', 'res', 'vtx']
_CAT_GEN_LXY_KEY = {c: f'{c}_gen_lxy' for c in _CAT_ORDER}


def _cat_event_fracs(samp_sel):
    counts = {}
    for cat in _CAT_ORDER:
        h = _select(_CAT_GEN_LXY_KEY[cat], samp_sel)
        counts[cat] = float(np.sum(h.values())) if h is not None else 0.0
    total = sum(counts.values())
    if total <= 0:
        zero = {c: 0.0 for c in _CAT_ORDER}
        return zero, zero
    fracs = {c: counts[c] / total for c in _CAT_ORDER}
    errs  = {c: np.sqrt(counts[c]) / total for c in _CAT_ORDER}
    return fracs, errs


print("Section 2: category fractions (all-samples)")
fracs, errs = _cat_event_fracs(sum)
if sum(fracs.values()) > 0:
    fig, ax = plt.subplots(figsize=size)
    hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
    x      = np.arange(len(_CAT_ORDER))
    vals   = [fracs[c] for c in _CAT_ORDER]
    errs_  = [errs[c]  for c in _CAT_ORDER]
    colors = [cat_styles[c]['color'] for c in _CAT_ORDER]
    xlbls  = [cat_styles[c]['label'] for c in _CAT_ORDER]
    ax.bar(x, vals, yerr=errs_,
           color=colors, alpha=0.8, edgecolor='black', linewidth=0.8,
           error_kw={'ecolor': 'black', 'capsize': 4, 'elinewidth': 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels(xlbls, fontsize=15)
    ax.set_ylabel('Fraction of events')
    ax.set_ylim(0, None)
    ax.set_title('Reco category distribution — all samples')
    plt.tight_layout()
    plt.savefig(f'{plotdir}/hist_{seltag}_cat_frac_allsamps.png')
    plt.close(fig)
    print("  -> 1 plot")
else:
    print("  -> 0 plots")


def _cat_frac_series(pset):
    points = list(_series_points(pset))
    n = len(points)
    if n == 0:
        return False

    fig, ax = plt.subplots(figsize=size)
    hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
    x     = np.arange(len(_CAT_ORDER))
    bar_w = 0.8 / n
    any_drawn = False
    for idx, (m1, delta, ctau) in enumerate(points):
        sname = _samp_name(m1, delta, ctau)
        if sname is None:
            continue
        fracs, errs = _cat_event_fracs(sname)
        if sum(fracs.values()) == 0:
            continue
        vals  = [fracs[c] for c in _CAT_ORDER]
        errs_ = [errs[c]  for c in _CAT_ORDER]
        offsets = x - 0.4 + bar_w * (idx + 0.5)
        ax.bar(offsets, vals, width=bar_w, yerr=errs_, label=_sig_label(m1, delta, ctau),
               color=f'C{idx}', alpha=0.8, edgecolor='black', linewidth=0.5,
               error_kw={'ecolor': 'black', 'capsize': 3, 'elinewidth': 1.0})
        any_drawn = True
    if not any_drawn:
        plt.close(fig)
        return False

    xlbls = [cat_styles[c]['label'] for c in _CAT_ORDER]
    ax.set_xticks(x)
    ax.set_xticklabels(xlbls, fontsize=15)
    ax.set_ylabel('Fraction of events')
    ax.set_ylim(0, None)
    ax.set_title('Reco category distribution')
    ax.legend(fontsize=13)
    plt.tight_layout()
    tag = _param_tag(pset['m1s'], pset['deltas'], pset['ctaus'])
    plt.savefig(f'{plotdir}/hist_{seltag}_cat_frac_{tag}.png')
    plt.close(fig)
    return True


print("Section 2: category fractions (per-series)")
n_made = 0
for pset in PARAM_SETS:
    n_made += _cat_frac_series(pset)
print(f"  -> {n_made} per-series plots")


# ── Sections 3 & 4: categorical "reason" plots ────────────────────────────────
# mele_nearestEle_oppQ_failreason (Section 3) and res_notVtx_reason (Section 4)
# are each single-category diagnostics (no cross-category overlay to make),
# plotted as a fraction-of-total bar chart -- one bar per reason category.
# Each gets an all-samples-summed version plus one grouped-bar version per
# PARAM_SETS series (one color per signal point in the series).

def _reason_fractions(hkey, samp_sel):
    h = _select(hkey, samp_sel)
    if h is None:
        return None, None
    total = np.sum(h.values())
    if total == 0:
        return None, None
    cats = list(h.axes[-1])
    fracs = h.values() / total
    return cats, fracs


def _reason_plot_allsamples(hkey, title, ylabel, fname):
    cats, fracs = _reason_fractions(hkey, sum)
    if cats is None:
        return False
    fig, ax = plt.subplots(figsize=size)
    hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
    ax.bar(cats, fracs, color='C0')
    for i, f in enumerate(fracs):
        ax.text(i, f, f'{f:.2f}', ha='center', va='bottom')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{title} — all samples')
    plt.setp(ax.get_xticklabels(), rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(f'{plotdir}/hist_{seltag}_{fname}_allsamps.png')
    plt.close(fig)
    return True


def _reason_plot_series(hkey, pset, title, ylabel, fname):
    points = list(_series_points(pset))
    n = len(points)
    if n == 0:
        return False

    # Reference category ordering from the all-samples histogram so every bar
    # group uses the same x-axis regardless of which reasons a given signal
    # point happens to populate.
    ref_cats, _ = _reason_fractions(hkey, sum)
    if ref_cats is None:
        return False

    fig, ax = plt.subplots(figsize=size)
    hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
    x = np.arange(len(ref_cats))
    width = 0.8 / n
    any_drawn = False
    for i, (m1, delta, ctau) in enumerate(points):
        sname = _samp_name(m1, delta, ctau)
        if sname is None:
            continue
        cats, fracs = _reason_fractions(hkey, sname)
        if cats is None:
            continue
        frac_by_cat = dict(zip(cats, fracs))
        heights = [frac_by_cat.get(c, 0.0) for c in ref_cats]
        offset = (i - (n - 1) / 2) * width
        ax.bar(x + offset, heights, width=width, color=f'C{i}', label=_sig_label(m1, delta, ctau))
        any_drawn = True
    if not any_drawn:
        plt.close(fig)
        return False

    ax.set_xticks(x)
    ax.set_xticklabels(ref_cats, rotation=20, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=13)
    plt.tight_layout()
    tag = _param_tag(pset['m1s'], pset['deltas'], pset['ctaus'])
    plt.savefig(f'{plotdir}/hist_{seltag}_{fname}_{tag}.png')
    plt.close(fig)
    return True


print("Section 3: why isn't this merged-electron event resolved? (all-samples + per-series)")
_reason_plot_allsamples(
    'mele_nearestEle_oppQ_failreason',
    'Why is this pair not resolved?',
    'Fraction of opposite-charge mele_nearestEle pairs',
    'mele_nearestEle_oppQ_failreason',
)
n_made = 0
for pset in PARAM_SETS:
    n_made += _reason_plot_series(
        'mele_nearestEle_oppQ_failreason', pset,
        'Why is this pair not resolved?',
        'Fraction of opposite-charge mele_nearestEle pairs',
        'mele_nearestEle_oppQ_failreason',
    )
print(f"  -> {n_made} per-series plots")

print("Section 4: why isn't this resolved event vertexed? (all-samples + per-series)")
_reason_plot_allsamples(
    'res_notVtx_reason',
    'Why did this event fail vertexing?',
    'Fraction of resolved events',
    'res_notVtx_reason',
)
n_made = 0
for pset in PARAM_SETS:
    n_made += _reason_plot_series(
        'res_notVtx_reason', pset,
        'Why did this event fail vertexing?',
        'Fraction of resolved events',
        'res_notVtx_reason',
    )
print(f"  -> {n_made} per-series plots")

print("Done.")
