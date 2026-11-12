import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import coffea.util as util
import analysisTools.utils as utils
import analysisTools.plotTools as ptools
import os

# Plots for configs/histo_configs/mergeddphi.py: |dPhi| between every pair of
# {gen chi2, gen ee, reco merged electron, GenMET, PFMET} for signal events in
# the truth-defined merged-electron category (events.cat_merged).

ptools.set_plot_style()

outdir    = 'workarea'
vers      = 'Jul2026noID'
selection = 'anabrv'
hists_tag = 'mergeddphi'
saved_signal_hists = f"{outdir}/hists_sig{vers}_{selection}-sel_{hists_tag}.coffea"
seltag = 'mergeddphi'
cut    = 'cut3'          # None -> last cut on the histogram's cut axis
year   = 2024

s_hists = util.load(saved_signal_hists)[0]
s_pts   = utils.get_signal_point_dict(s_hists)

plotdir = 'plots/mergeddphi'
os.makedirs(plotdir, exist_ok=True)

# ── Metadata (mirrors the histo config) ───────────────────────────────────────

objects = ['chi2', 'genee', 'mele', 'genmet', 'recomet']
obj_labels = {
    'chi2':    r'\chi_2^{\rm gen}',
    'genee':   r'(ee)^{\rm gen}',
    'mele':    r'e_{\rm merged}^{\rm reco}',
    'genmet':  r'p_T^{\rm miss,gen}',
    'recomet': r'p_T^{\rm miss,reco}',
}
pairs = [(a, b) for i, a in enumerate(objects) for b in objects[i+1:]]


def _hname(a, b):
    return f'dphi_{a}_{b}'


def _pair_label(a, b):
    return rf'$|\Delta\phi({obj_labels[a]}, {obj_labels[b]})|$'


# human-readable object names for plot titles
obj_titles = {
    'chi2':    r'Gen $\chi_2$',
    'genee':   'Gen ee Pair',
    'mele':    'Reco Merged Electron',
    'genmet':  r'Gen $p_T^\mathrm{miss}$',
    'recomet': r'Reco $p_T^\mathrm{miss}$',
}


def _pair_title(a, b):
    return rf'$|\Delta\phi|$: {obj_titles[a]} vs. {obj_titles[b]}'


if cut is None:
    cut = list(s_hists[_hname(*pairs[0])].axes['cut'])[-1]
print(f'Plotting cut: {cut}')

series_list = [
    {'m1s': [0.05, 0.5, 5, 50], 'deltas': [0.1],            'ctaus': [10],         'tag': 'ctau-10_delta-0p1_m1-wide'},
    {'m1s': [0.5, 1, 2, 5],     'deltas': [0.1],            'ctaus': [10],         'tag': 'ctau-10_delta-0p1_m1-narrow'},
    {'m1s': [0.5],              'deltas': [0.1],            'ctaus': [1, 10, 100], 'tag': 'm1-0p5_delta-0p1'},
    {'m1s': [0.5],              'deltas': [0.05, 0.1, 0.2], 'ctaus': [10],         'tag': 'm1-0p5_ctau-10'},
    {'m1s': [5],                'deltas': [0.1],            'ctaus': [1, 10, 100], 'tag': 'm1-5_delta-0p1'},
    {'m1s': [5],                'deltas': [0.05, 0.1, 0.2], 'ctaus': [10],         'tag': 'm1-5_ctau-10'},
]


def _sig_label(m1, delta, ctau):
    return rf'$M_1$={m1}, $\delta$={ptools.fmt_delta(delta)}, $c\tau$={ctau} mm'


def _series_points(series):
    return [(m1, d, c) for m1 in series['m1s'] for d in series['deltas'] for c in series['ctaus']]


def _get(a, b, samp):
    """1D dphi histogram at `cut` for one sample name, or summed over all (samp=sum)."""
    h = s_hists[_hname(a, b)]
    if samp is not sum and samp not in h.axes['samp']:
        return None
    return h[{'samp': samp, 'cut': cut}]


def _draw(h, ax, **kwargs):
    """Unit-area step histogram; skips empty histograms."""
    if h is None or h.sum().value <= 0:
        return False
    hep.histplot(h, ax=ax, histtype='step', density=True, linewidth=2, **kwargs)
    return True


def _xrange(*pair_list):
    """x-range spanning the dphi axes of the given (a, b) pairs, read from the saved
    histograms so it follows whatever binning the histo config defines."""
    edges = [s_hists[_hname(a, b)].axes['dphi'].edges for a, b in pair_list]
    return min(e[0] for e in edges), max(e[-1] for e in edges)


def _finish(ax, xlabel, title, xlim, logy=False, legend_fs=None, ncol=1):
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Normalized Events')
    if logy:
        ax.set_yscale('log')
    ax.set_title(title)
    # Legend pinned top-right with headroom added above the data for it (loc='best'
    # can't avoid the data when the curves fill the whole frame).
    leg = ax.legend(loc='upper right', fontsize=legend_fs, ncol=ncol)
    plt.tight_layout()
    ptools.legend_headroom(ax, leg)
    plt.tight_layout()


# ── Section 1: all pairs, summed over signal samples ─────────────────────────
# One overlay of all ten pairs, plus one plot per pair.

fig, ax = ptools.new_fig(year)
for i, (a, b) in enumerate(pairs):
    _draw(_get(a, b, sum), ax, label=_pair_label(a, b), color=ptools.cms_colors(len(pairs))[i],
          linestyle='-' if i < 5 else '--')
# 10 long entries: two columns at a smaller size so the legend doesn't eat the frame.
_finish(ax, r'$|\Delta\phi|$', r'$|\Delta\phi|$ Between Object Pairs', _xrange(*pairs),
        legend_fs=18, ncol=2)
plt.savefig(f'{plotdir}/hist_{seltag}_allpairs_allsamps_{cut}.png')
plt.close(fig)

for a, b in pairs:
    fig, ax = ptools.new_fig(year)
    _draw(_get(a, b, sum), ax, label='All signal samples', color='C0')
    _finish(ax, _pair_label(a, b), _pair_title(a, b), _xrange((a, b)))
    plt.savefig(f'{plotdir}/hist_{seltag}_{_hname(a, b)}_allsamps_{cut}.png')
    plt.close(fig)


# ── Section 2: per-pair signal-point overlays ────────────────────────────────

for a, b in pairs:
    for series in series_list:
        fig, ax = ptools.new_fig(year)
        drawn = 0
        points = _series_points(series)
        colors = ptools.cms_colors(len(points))
        for i, (m1, delta, ctau) in enumerate(points):
            samp = ptools.samp_name(s_pts, m1, delta, ctau)
            if samp is None:
                print(f'No sample for m1={m1}, delta={delta}, ctau={ctau}')
                continue
            drawn += _draw(_get(a, b, samp), ax, label=_sig_label(m1, delta, ctau), color=colors[i])
        if drawn == 0:
            plt.close(fig)
            continue
        _finish(ax, _pair_label(a, b), _pair_title(a, b), _xrange((a, b)))
        plt.savefig(f'{plotdir}/hist_{seltag}_{_hname(a, b)}_{series["tag"]}_{cut}.png')
        plt.close(fig)


print(f'Plots written to {plotdir}/')
