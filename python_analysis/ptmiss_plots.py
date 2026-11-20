import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import hist
from matplotlib.colors import LogNorm
from scipy.stats import beta
import coffea.util as util
import analysisTools.utils as utils
import analysisTools.plotTools as ptools
from configs.histo_configs.ptmiss import (MET_LABELS, TRIGGERS_MAIN, TRIG_EFF_CUTS, SLIM_CUTS,
                                          CORR_PAIRS, TRIG_MIN_MET)

# Plots for configs/histo_configs/ptmiss.py: ptmiss trigger efficiencies, ptmiss
# distributions / definition comparison, and resolution/response vs GenMET.
#
# Takes a signal coffea file (histmaker_signal.py) and, optionally, a background
# one (histmaker_background.py with process_slim = True). If the background file
# is missing or bkg_vers is None, only the signal plots are made.
#
# Trigger efficiencies: signal from the full tree at TRIG_EFF_CUTS (after the
# ptmiss filters, before the trigger); background from the slim tree at
# SLIM_CUTS (every generated event passing the filters), since the bkg full
# trees already sit behind the METNoMu120 OR. The hists are weighted, so the
# Clopper-Pearson errors use effective counts n_eff = sum(w)^2 / sum(w^2).

ptools.set_plot_style()
# samples are selected by list and then summed, so bins dropped from the samp
# axis never matter
warnings.filterwarnings('ignore', message='List indexing selection is experimental')

# ── Configuration ─────────────────────────────────────────────────────────────

outdir    = 'workarea'
sig_vers  = 'Nov2026'
bkg_vers  = 'Nov2026'      # None -> signal only
selection = 'an'
hists_tag = 'ptmiss'
year      = 2024

saved_signal_hists = f"{outdir}/hists_sig{sig_vers}_{selection}-sel_{hists_tag}.coffea"
saved_bkg_hists    = (f"{outdir}/hists_bkg{bkg_vers}_{selection}-sel_{hists_tag}.coffea"
                      if bkg_vers is not None else None)
seltag  = f'{hists_tag}_{selection}-sel'
plotdir = 'plots/ptmiss'

SIG_TRIG_CUT = TRIG_EFF_CUTS[0]     # signal trigger-efficiency cut (full tree)
BKG_TRIG_CUT = SLIM_CUTS[0]         # background trigger-efficiency cut (slim tree)
DIST_CUTS    = ['cut1', 'cut2', 'cut3']   # cuts for the distribution / response plots

# signal points (m1 GeV, delta, ctau mm) overlaid in the signal-vs-background plots
SIG_POINTS = [(0.5, 0.1, 10), (5, 0.1, 10), (50, 0.1, 10)]
# single signal point used where the series are definitions / triggers / slices
SIG_REF_POINT = (5, 0.1, 10)

EFF_METVARS = ['pf', 'trignomu']   # definitions for the per-definition efficiency plots
RESP_GENMET_MIN = 100              # 1D response distributions: GenMET >= this (GeV)
MIN_NEFF = 10                      # profile points need this many effective events

# ── Lookups ───────────────────────────────────────────────────────────────────

MET_NAMES = {
    'pf':       'PF',
    'pfraw':    'Raw PF',
    'puppi':    'PUPPI',
    'calo':     'Calo',
    'pfnomu':   r'PF No-$\mu$',
    'mhtnomu':  r'PF $H_T^{miss}$ No-$\mu$',
    'trignomu': r'min(No-$\mu$ $p_T^{miss}$, $H_T^{miss}$)',
    'gen':      'Gen',
}
TRIG_TITLES = {'METNoMu120_OR': 'PFMETNoMu120 OR'}
GENLEP_LABELS = {
    '0l': 'No prompt lepton', '1e': r'1 prompt $e$', '1mu': r'1 prompt $\mu$',
    '1tau': r'1 prompt $\tau$', '2l+': r'$\geq$2 prompt leptons', 'na': 'No gen info',
}
REF_VARIANTS = [('Plain MC',                    {'passIsoMu27': sum, 'passEle32': sum}),
                ('Ref.: HLT_IsoMu27',           {'passIsoMu27': 1,   'passEle32': sum}),
                ('Ref.: HLT_Ele32_WPTight_Gsf', {'passIsoMu27': sum, 'passEle32': 1})]
NOREF = REF_VARIANTS[0][1]

POINT_STYLE = dict(fmt='o', markersize=6, capsize=0, lw=2)
EFF_Y = 'Trigger efficiency'
NORM_Y = 'Normalized events'


def _met_x(var):
    return MET_LABELS.get(var, var) + ' (GeV)'


def _sig_label(pt):
    m1, delta, ctau = pt
    return rf'$m_1$={m1:g} GeV, $\delta$={ptools.fmt_delta(delta)}, c$\tau$={ctau} mm'


def _sig_tag(pt):
    m1, delta, ctau = pt
    f = lambda x: f'{x:g}'.replace('.', 'p')
    return f'sig-m1-{f(m1)}_delta-{f(delta)}_ctau-{ctau}'


# ── Load ──────────────────────────────────────────────────────────────────────

s_hists = util.load(saved_signal_hists)[0]
s_pts   = utils.get_signal_point_dict(s_hists)

b_hists = None
if saved_bkg_hists is not None and os.path.exists(saved_bkg_hists):
    b_hists = util.load(saved_bkg_hists)[0]
else:
    print(f'No background file ({saved_bkg_hists}): making signal-only plots')
HISTS = {'sig': s_hists, 'bkg': b_hists}

# background processes from the sample names (bkg_<year>_<process>_<sub>)
BKG_PROCS = {}
for s in (b_hists['met_dist'].axes['samp'] if b_hists else []):
    BKG_PROCS.setdefault(s.split('_')[2], []).append(s)
BKG_PROCS = dict(sorted(BKG_PROCS.items()))


# A source is (src, samples to sum, legend label, file tag).
def _sig_source(pt):
    name = ptools.samp_name(s_pts, *pt)
    if name is None:
        print(f'Signal point {pt} not in {saved_signal_hists}, skipped')
        return None
    return ('sig', [name], _sig_label(pt), _sig_tag(pt))


SIG_SOURCES = [s for s in map(_sig_source, SIG_POINTS) if s]
BKG = ('bkg', sum(BKG_PROCS.values(), []), 'Total background', 'bkg') if b_hists else None
EACH = [s for s in (_sig_source(SIG_REF_POINT), BKG) if s]   # plots made once per source
ALL = SIG_SOURCES + ([BKG] if BKG else [])                   # plots overlaying the sources
SIGBKG = 'sigbkg' if BKG else 'sig'


def _proj(source, hname, cut, **sel):
    """Histogram `hname` at `cut`, summed over the source's samples, with the extra
    axis selections in `sel` (category value, or sum). None if anything is missing."""
    hd = HISTS[source[0]]
    if hd is None or hname not in hd:
        return None
    h = hd[hname]
    samps = [s for s in source[1] if s in h.axes['samp']]
    if cut not in h.axes['cut'] or not samps:
        return None
    if len(samps) == 1:
        h = h[{'cut': cut, 'samp': samps[0]}]
    else:
        h = h[{'cut': cut, 'samp': samps}][{'samp': sum}]
    if any(v is not sum and v not in h.axes[a] for a, v in sel.items()):
        return None
    return h[{a: v if v is sum else hist.loc(v) for a, v in sel.items()}] if sel else h


def _split(h, axis, labels={}):
    """[(slice, label)] for each category of `axis`; [] if h is None."""
    return [] if h is None else [(h[{axis: v}], labels.get(v, v)) for v in h.axes[axis]]


def _empty(h):
    return h is None or h.sum().value <= 0


def _trig_cut(source):
    return SIG_TRIG_CUT if source[0] == 'sig' else BKG_TRIG_CUT


def _out(subdir, var, cut, variant=None):
    name = f'hist_{seltag}_{var}_{cut}' + (f'_{variant}' if variant else '') + '.png'
    return f'{plotdir}/{subdir}/{name}'


# ── Drawing ───────────────────────────────────────────────────────────────────

def _clopper_pearson(k, n, cl=0.683):
    """Central Clopper-Pearson interval; k and n may be non-integer effective counts.
    (PLOTTING.md points to ptools.clopper_pearson, which doesn't exist yet.)"""
    alpha = 1 - cl
    with np.errstate(divide='ignore', invalid='ignore'):
        lo = np.where(k > 0, beta.ppf(alpha / 2, k, n - k + 1), 0.0)
        hi = np.where(k < n, beta.ppf(1 - alpha / 2, k + 1, n - k), 1.0)
    return np.nan_to_num(lo), np.nan_to_num(hi, nan=1.0)


def _draw_eff(ax, h, label, color):
    """Efficiency vs ptmiss from a (met, passTrig) histogram."""
    num, den = h[{'passTrig': hist.loc(1)}], h[{'passTrig': sum}]
    sw, sw2 = den.values(), den.variances()
    ax_met = den.axes[0]
    ok = (sw > 0) & (sw2 > 0) & (ax_met.edges[:-1] >= TRIG_MIN_MET)
    if not ok.any():
        return
    eff = num.values()[ok] / sw[ok]
    n_eff = sw[ok] ** 2 / sw2[ok]
    lo, hi = _clopper_pearson(np.clip(eff, 0, 1) * n_eff, n_eff)
    ax.errorbar(ax_met.centers[ok], eff, xerr=0.5 * ax_met.widths[ok],
                yerr=(np.clip(eff - lo, 0, None), np.clip(hi - eff, 0, None)),
                color=color, label=label, **POINT_STYLE)


def _draw_step(ax, h, label, color, density=True, linestyle='-'):
    """Step line with sqrt(sum w^2) errors, unit area if density."""
    if _empty(h):
        return False
    vals, err = h.values(), np.sqrt(h.variances())
    norm = vals.sum() if density else 1
    hep.histplot(vals / norm, bins=h.axes[0].edges, yerr=err / norm, ax=ax, histtype='step',
                 linewidth=2, label=label, color=color, linestyle=linestyle)
    return True


def _draw_prof(ax, h, label, color, qty):
    """Weighted mean or RMS ('mean'/'rms') of axis 1 in bins of axis 0 (GenMET).
    Bins with fewer than MIN_NEFF effective events are dropped."""
    v, y = h.values(), h.axes[1].centers
    sw, sw2 = v.sum(axis=1), h.variances().sum(axis=1)
    n_eff = np.divide(sw ** 2, sw2, out=np.zeros_like(sw), where=sw2 > 0)
    ok = (sw > 0) & (n_eff >= MIN_NEFF)
    if not ok.any():
        return
    v, sw, n_eff = v[ok], sw[ok], n_eff[ok]
    mean = (v * y).sum(axis=1) / sw
    rms = np.sqrt((v * (y - mean[:, None]) ** 2).sum(axis=1) / sw)
    val, err = (mean, rms / np.sqrt(n_eff)) if qty == 'mean' else (rms, rms / np.sqrt(2 * n_eff))
    ax.errorbar(h.axes[0].centers[ok], val, yerr=err, xerr=0.5 * h.axes[0].widths[ok],
                color=color, label=label, **POINT_STYLE)


def _legend_kw(n):
    """Smaller font / two columns for crowded legends."""
    return {'fontsize': 18, 'ncol': 2} if n > 6 else {'fontsize': 20} if n > 4 else {}


def _place_corner_legend(ax, **kw):
    """Fixed 0-1 y-range: first legend corner that doesn't cover the points,
    else upper right with headroom."""
    for loc in ('upper right', 'lower left', 'upper left', 'lower right'):
        leg = ax.legend(loc=loc, **kw)
        if not ptools.legend_covers_data(ax, leg):
            return
        leg.remove()
    ptools.place_legend(ax, **kw)


def _finish(fig, ax, path, xlabel, ylabel, title, logx=False, logy=False, legend='headroom',
            legend_kw=None):
    """Labels, scales, legend; then save and close. legend: 'headroom' (upper right
    with headroom), 'eff' (0-1 y-range, corner search), or None."""
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    legend_kw = legend_kw or {}
    plt.tight_layout()
    if legend == 'headroom':
        ptools.legend_headroom(ax, ax.legend(loc='upper right', **legend_kw))
    elif legend == 'eff':
        ax.set_ylim(0, 1)
        _place_corner_legend(ax, **legend_kw)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    print(f'Saved: {path}')
    plt.close(fig)


def _plot(series, draw, path, xlabel, ylabel, title, hline=None, **kw):
    """One figure with a colour per (hist, label) in `series`, drawn with `draw`;
    empty hists are dropped, and nothing is saved if all are empty."""
    series = [(h, label) for h, label in series if not _empty(h)]
    if not series:
        print(f'Nothing to plot for {os.path.basename(path)}, skipped')
        return
    fig, ax = ptools.new_fig(year)
    for (h, label), color in zip(series, ptools.cms_colors(len(series))):
        draw(ax, h, label, color)
    if hline is not None:
        ax.axhline(hline, color='black', linestyle=':', linewidth=2)
    _finish(fig, ax, path, xlabel, ylabel, title, legend_kw=_legend_kw(len(series)), **kw)


def _plot_2d(h, path, xlabel, ylabel, title, logy=False, diagonal=False):
    if _empty(h):
        return
    fig, ax = ptools.new_fig(year, cms_label=False)
    art = hep.hist2dplot(h * (1.0 / h.sum().value), ax=ax, norm=LogNorm())
    art.cbar.set_label('Fraction of events')
    ptools.add_cms_label(ax, year)
    if diagonal:
        ax.axline((0, 0), (1, 1), color='black', linestyle=':', linewidth=2)
    _finish(fig, ax, path, xlabel, ylabel, title, logy=logy, legend=None)


# ── Trigger efficiency ────────────────────────────────────────────────────────

def plot_trigger():
    for trig in TRIGGERS_MAIN:
        ttl = TRIG_TITLES.get(trig, trig)

        # signal points and total background, per definition
        for var in EFF_METVARS:
            series = [(_proj(s, 'trig_eff', _trig_cut(s), trigger=trig, metvar=var, **NOREF), s[2])
                      for s in ALL]
            _plot(series, _draw_eff, _out('trigger', f'trigeff-{trig}-{var}', SIG_TRIG_CUT, SIGBKG),
                  _met_x(var), EFF_Y, f'{ttl}: Efficiency\n({MET_NAMES[var]} $p_T^{{miss}}$)',
                  legend='eff')

        for s in EACH:
            cut, sfx = _trig_cut(s), f'\n({s[2]})'
            out = lambda tag: _out('trigger', tag, cut, s[3])

            # one series per definition
            h = _proj(s, 'trig_eff', cut, trigger=trig, **NOREF)
            _plot(_split(h, 'metvar', MET_NAMES), _draw_eff, out(f'trigeff-{trig}'),
                  r'Offline $p_T^{miss}$ (GeV)', EFF_Y, f'{ttl}: Efficiency{sfx}', legend='eff')

            # reference-trigger method vs plain MC (closure), analysis trigger only
            if trig == TRIGGERS_MAIN[0]:
                for var in EFF_METVARS:
                    series = [(_proj(s, 'trig_eff', cut, trigger=trig, metvar=var, **sel), label)
                              for label, sel in REF_VARIANTS]
                    _plot(series, _draw_eff, out(f'trigeff-refs-{trig}-{var}'), _met_x(var), EFF_Y,
                          f'{ttl}: Reference-Trigger Efficiency{sfx}', legend='eff')

            # HT slices
            for var in ('pf', 'trignomu'):
                h = _proj(s, 'trig_eff_ht', cut, trigger=trig, metvar=var)
                e = h.axes['ht'].edges if h is not None else []
                series = [(h[{'ht': i}], rf'$H_T$ {e[i]:g}-{e[i + 1]:g} GeV') for i in range(len(e) - 1)]
                _plot(series, _draw_eff, out(f'trigeff-ht-{trig}-{var}'), _met_x(var), EFF_Y,
                      f'{ttl}: Efficiency in $H_T$ Slices{sfx}', legend='eff')

            # prompt gen-lepton categories
            for var in ('pf', 'pfnomu', 'trignomu'):
                h = _proj(s, 'trig_eff_genlep', cut, trigger=trig, metvar=var)
                _plot(_split(h, 'genlep', GENLEP_LABELS), _draw_eff, out(f'trigeff-genlep-{trig}-{var}'),
                      _met_x(var), EFF_Y, f'{ttl}: Efficiency by Gen Leptons{sfx}', legend='eff')


# ── Distributions ─────────────────────────────────────────────────────────────

# signal points vs total background, unit area: (hist, file tag, x label, title, selection)
SIGBKG_1D = [
    *[('met_dist', f'met-{v}', _met_x(v), f'$p_T^{{miss}}$\n({MET_NAMES[v]})', dict(genlep=sum, metvar=v))
      for v in MET_NAMES if v != 'gen'],
    ('genmet_dist', 'genmet', _met_x('gen'), r'Gen $p_T^{miss}$', dict(genlep=sum)),
    ('met_significance', 'met-significance', r'$p_T^{miss}$ significance', r'$p_T^{miss}$ Significance',
     dict(genlep=sum)),
    ('pf_calo_ratio', 'pf-calo-ratio', r'$|p_T^{miss,PF} - p_T^{miss,Calo}|/p_T^{miss,Calo}$',
     r'PF vs. Calo $p_T^{miss}$', {}),
]


def plot_dists(cut):
    for hname, tag, xlabel, title, sel in SIGBKG_1D:
        logx = any(getattr(a, 'transform', None) is not None for a in s_hists[hname].axes)
        _plot([(_proj(s, hname, cut, **sel), s[2]) for s in ALL], _draw_step,
              _out('dist', tag, cut, SIGBKG), xlabel, NORM_Y, title, logx=logx, logy=True)

    # all definitions overlaid, per source
    for s in EACH:
        _plot(_split(_proj(s, 'met_dist', cut, genlep=sum), 'metvar', MET_NAMES), _draw_step,
              _out('dist', 'met-defs', cut, s[3]), r'$p_T^{miss}$ (GeV)', NORM_Y,
              f'$p_T^{{miss}}$: Definition Comparison\n({s[2]}, {cut})', logy=True)
        _plot(_split(_proj(s, 'met_dphi_gen', cut, genmet=sum), 'metvar', MET_NAMES), _draw_step,
              _out('dist', 'dphi-gen', cut, s[3]),
              r'$|\Delta\phi(\vec{p}_T^{miss}, \vec{p}_T^{miss,gen})|$', NORM_Y,
              f'$p_T^{{miss}}$ Direction vs. Gen\n({s[2]}, {cut})', logy=True)


def plot_bkg_composition(cut):
    """Weighted yields per background process (bkg_cmap) with the signal points overlaid."""
    if not BKG:
        return
    for var in ('pf', 'trignomu'):
        sel = dict(genlep=sum, metvar=var)
        fig, ax = ptools.new_fig(year)
        drawn = [_draw_step(ax, _proj(('bkg', samps), 'met_dist', cut, **sel), proc,
                            ptools.bkg_cmap.get(proc, ptools.CMS_COLORS_10[3]), density=False)
                 for proc, samps in BKG_PROCS.items()]
        for s, color in zip(SIG_SOURCES, ptools.SIGNAL_COLORS_WITH_BKG):
            _draw_step(ax, _proj(s, 'met_dist', cut, **sel), s[2], color, density=False, linestyle='--')
        if not any(drawn):
            plt.close(fig)
            continue
        _finish(fig, ax, _out('dist', f'met-{var}-bycomp', cut, 'sigbkg'), _met_x(var), 'Weighted events',
                f'$p_T^{{miss}}$: Background Composition\n({MET_NAMES[var]}, {cut})', logy=True,
                legend_kw=_legend_kw(len(BKG_PROCS) + len(SIG_SOURCES)))


def plot_corr(cut):
    for s in EACH:
        for a, b in CORR_PAIRS:
            _plot_2d(_proj(s, 'met_corr', cut, pair=f'{a}_{b}'), _out('corr', f'corr-{a}-{b}', cut, s[3]),
                     _met_x(a), _met_x(b), f'$p_T^{{miss}}$: {MET_NAMES[a]} vs. {MET_NAMES[b]}\n({s[2]}, {cut})',
                     diagonal=True)
        h = _proj(s, 'met_sig_vs_met', cut)
        _plot_2d(h, _out('corr', 'sig-vs-met', cut, s[3]), _met_x('pf'), r'$p_T^{miss}$ significance',
                 f'$p_T^{{miss}}$: Significance vs. $p_T^{{miss}}$\n({s[2]}, {cut})',
                 logy=h is not None and h.axes['sig'].transform is not None)


# ── Resolution / response vs GenMET ───────────────────────────────────────────
# The NoMu definitions only use events without a prompt gen muon (see the histo config).

# (hist, quantity, y label, title, file tag)
PROFILES = [
    ('met_response', 'mean', r'Mean $p_T^{miss}/p_T^{miss,gen}$', r'$p_T^{miss}$: Response', 'response-mean'),
    ('met_diff',     'mean', r'Mean $p_T^{miss} - p_T^{miss,gen}$ (GeV)', r'$p_T^{miss}$: Bias', 'diff-mean'),
    ('met_diff',     'rms',  r'RMS($p_T^{miss} - p_T^{miss,gen}$) (GeV)', r'$p_T^{miss}$: Resolution', 'diff-rms'),
    ('met_upar',     'mean', r'Mean $p_{T,\parallel}^{miss} - p_T^{miss,gen}$ (GeV)',
     r'$p_T^{miss}$: Parallel Bias', 'upar-mean'),
    ('met_upar',     'rms',  r'RMS($p_{T,\parallel}^{miss} - p_T^{miss,gen}$) (GeV)',
     r'$p_T^{miss}$: Parallel Resolution', 'upar-rms'),
    ('met_uperp',    'rms',  r'RMS($p_{T,\perp}^{miss}$) (GeV)', r'$p_T^{miss}$: Perpendicular Resolution',
     'uperp-rms'),
]


def plot_response(cut):
    for s in EACH:
        for hname, qty, ylabel, title, tag in PROFILES:
            hline = 1 if hname == 'met_response' else 0 if qty == 'mean' else None
            _plot(_split(_proj(s, hname, cut), 'metvar', MET_NAMES),
                  lambda ax, h, label, color, q=qty: _draw_prof(ax, h, label, color, q),
                  _out('response', tag, cut, s[3]), _met_x('gen'), ylabel, f'{title}\n({s[2]}, {cut})',
                  hline=hline)

        h = _proj(s, 'met_response', cut)
        if h is not None:
            h = h[{'genmet': slice(hist.loc(RESP_GENMET_MIN), None, sum)}]
        _plot(_split(h, 'metvar', MET_NAMES), _draw_step,
              _out('response', 'response', cut, f'genmet{RESP_GENMET_MIN}_{s[3]}'),
              r'$p_T^{miss}/p_T^{miss,gen}$', NORM_Y,
              f'$p_T^{{miss}}$: Response\n({s[2]}, {cut}, $p_T^{{miss,gen}} \\geq$ {RESP_GENMET_MIN} GeV)',
              logy=True)


# ── Run ───────────────────────────────────────────────────────────────────────

plot_trigger()
for cut in DIST_CUTS:
    plot_dists(cut)
    plot_bkg_composition(cut)
    plot_corr(cut)
    plot_response(cut)
