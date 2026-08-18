import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import LogNorm
import mplhep as hep
import coffea.util as util
import analysisTools.utils as utils
import analysisTools.plotTools as ptools
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
vers      = 'Jun2026noID'
selection = 'anmatchvtx'
hists_tag = 'mergedmatch'
saved_signal_hists = f"{outdir}/hists_sig{vers}_{selection}-sel_{hists_tag}.coffea"
seltag   = 'minjdphi'
cut      = 'cut8'
year     = 2024
size     = (16, 12)

s_hists = util.load(saved_signal_hists)[0]
s_pts   = utils.get_signal_point_dict(s_hists)

os.makedirs('plots', exist_ok=True)

# ── display metadata ──────────────────────────────────────────────────────────

coll_styles = {
    'gedlpt': {'label': 'GED+LowPt', 'color': 'C0'},
    'alpt':   {'label': 'AllLowPt',  'color': 'C1'},
}
scenario_styles = {
    'merged': {'label': 'Merged',    'ls': '-'},
    'resolved': {'label': 'Resolved', 'ls': '--'},
}
# gen/reco object styles for pt and eta overlays within the merged selection
obj_styles = [
    ('gen_lead',    'Gen leading',    'C0', '-'),
    ('gen_sublead', 'Gen subleading', 'C1', '-'),
    ('reco',        'Reco merged',    'C2', '-'),
]
rank_styles = {
    'lead':    {'label': 'To gen leading',    'ls': '-'},
    'sublead': {'label': 'To gen subleading', 'ls': '--'},
}

plottag = f'sig{vers}_{selection}-sel'


def _samp_name(m1, delta, ctau):
    row = s_pts[np.isclose(s_pts.m1, m1) & np.isclose(s_pts.delta, delta) & (s_pts.ctau == ctau)]
    return row['name'].iloc[0] if not row.empty else None


def _sig_label(m1, delta, ctau):
    return rf'$M_1$={m1}, $\Delta$={delta}, $c\tau$={ctau} mm'


def _varbin_histplot(h, ax, **kwargs):
    """Plot a variable-bin histogram as a proper density (area = 1).

    hep.histplot's density=True divides by a single representative bin width
    in some mplhep versions, which leaves a visible step at bin-width transitions.
    This helper pre-computes count / (total_area * bin_width) per bin so every
    bin is correctly normalised regardless of its individual width.
    """
    vals, edges = h.to_numpy()
    widths = np.diff(edges)
    area = float(np.sum(vals * widths))
    dens = vals / area / widths if area > 0 else np.zeros_like(vals)
    hep.histplot((dens, edges), ax=ax, **kwargs)

style_dict_base_sig = {
    'rebin': 1j, 'xlim': None,
    'doLogy': False, 'doLogx': False, 'doDensity': True, 'doYerr': False,
    'xlabel': None, 'ylabel': None, 'label': None, 'flow': None,
    'doSave': False,
    'doCMSLabel': False,
}

series_list = [
    {'m1s': [0.05, 0.5, 5, 50], 'deltas': [0.1],            'ctaus': [10],         'tag': 'ctau-10_delta-0p1_m1-wide'},
    {'m1s': [0.5, 1, 2, 5],     'deltas': [0.1],            'ctaus': [10],         'tag': 'ctau-10_delta-0p1_m1-narrow'},
    {'m1s': [0.5],              'deltas': [0.1],            'ctaus': [1, 10, 100], 'tag': 'm1-0p5_delta-0p1'},
    {'m1s': [0.5],              'deltas': [0.05, 0.1, 0.2], 'ctaus': [10],         'tag': 'm1-0p5_ctau-10'},
    {'m1s': [5],                'deltas': [0.1],            'ctaus': [1, 10, 100], 'tag': 'm1-5_delta-0p1'},
    {'m1s': [5],                'deltas': [0.05, 0.1, 0.2], 'ctaus': [10],         'tag': 'm1-5_ctau-10'},
]

def _run_sig_plots(specs):
    for title, outname, variables, doLogy, solid_label, dotted_label in specs:
        is_paired = len(variables) > 1
        plot_dict = {
            'variable': variables if is_paired else variables[0],
            'year': year,
            'cut': cut,
        }
        ls = ['-', '--'] if is_paired else ['-']
        for series in series_list:
            m1s    = series['m1s']
            deltas = series['deltas']
            ctaus  = series['ctaus']
            tag    = series['tag']
            fig, ax = plt.subplots(figsize=size)
            style_dict = {
                **style_dict_base_sig,
                'fig': fig, 'ax': ax,
                'ls': ls,
                'doLogy': doLogy,
            }
            cmap_idx = 0
            for m1 in m1s:
                for delta in deltas:
                    for ctau in ctaus:
                        ptools.plot_signal_1D(s_hists, m1, delta, ctau, plot_dict, style_dict, cmap_idx=cmap_idx)
                        cmap_idx += 1
            hep.cms.label('Private Work', data=True, year=plot_dict['year'], com='13.6', ax=ax)
            handles, labels = ax.get_legend_handles_labels()
            if is_paired and solid_label is not None:
                handles += [
                    mlines.Line2D([], [], color='black', ls='-',  label=solid_label),
                    mlines.Line2D([], [], color='black', ls='--', label=dotted_label),
                ]
                labels += [solid_label, dotted_label]
            ax.legend(handles=handles, labels=labels)
            plt.title(rf'{title}: $M_1$ = {m1s}, $\Delta$ = {deltas}, c$\tau$ = {ctaus}mm')
            plt.tight_layout()
            plt.savefig(f'plots/hist_{seltag}_{outname}_{tag}.png')
            plt.close()


# ── Section 1: pt distributions (merged only) ─────────────────────────────────
# One plot per collection: overlay gen leading, gen subleading, and reco pt.

for coll, cstyle in coll_styles.items():
    fig, ax = plt.subplots(figsize=size)
    hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
    for obj_key, obj_label, color, ls in obj_styles:
        h = s_hists[f'{obj_key}_pt_{coll}'][{"cut": cut, "samp": sum}]
        hep.histplot(h, ax=ax, histtype='step', density=True,
                     label=obj_label, color=color, linestyle=ls, linewidth=2)
    ax.set_xlabel(r'$p_T$ [GeV]')
    ax.set_ylabel('A.U.')
    ax.set_title(rf'{cstyle["label"]} merged: gen & reco $p_T$ — all samples')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'plots/hist_{seltag}_merged_pt_{coll}_allsamps.png')
    plt.close(fig)

_run_sig_plots([
    (
        rf'{coll_styles["gedlpt"]["label"]} merged: gen $p_T$ leading vs subleading',
        'merged_gen_pt_gedlpt',
        ['gen_lead_pt_gedlpt', 'gen_sublead_pt_gedlpt'],
        False, 'Gen leading', 'Gen subleading',
    ),
    (
        rf'{coll_styles["alpt"]["label"]} merged: gen $p_T$ leading vs subleading',
        'merged_gen_pt_alpt',
        ['gen_lead_pt_alpt', 'gen_sublead_pt_alpt'],
        False, 'Gen leading', 'Gen subleading',
    ),
    (
        r'Merged: reco $p_T$ — GED+LowPt vs AllLowPt',
        'merged_reco_pt',
        ['reco_pt_gedlpt', 'reco_pt_alpt'],
        False, 'GED+LowPt', 'AllLowPt',
    ),
])


# ── Section 2: eta distributions (merged only) ────────────────────────────────
# One plot per collection: overlay gen leading, gen subleading, and reco eta.

for coll, cstyle in coll_styles.items():
    fig, ax = plt.subplots(figsize=size)
    hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
    for obj_key, obj_label, color, ls in obj_styles:
        h = s_hists[f'{obj_key}_eta_{coll}'][{"cut": cut, "samp": sum}]
        hep.histplot(h, ax=ax, histtype='step', density=True,
                     label=obj_label, color=color, linestyle=ls, linewidth=2)
    ax.set_xlabel(r'$\eta$')
    ax.set_ylabel('A.U.')
    ax.set_title(rf'{cstyle["label"]} merged: gen & reco $\eta$ — all samples')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'plots/hist_{seltag}_merged_eta_{coll}_allsamps.png')
    plt.close(fig)


# ── Section 3: reco-gen dR distributions (merged only) ───────────────────────
# One plot per collection: overlay dR to gen leading and gen subleading.

for coll, cstyle in coll_styles.items():
    fig, ax = plt.subplots(figsize=size)
    hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
    for rank, rstyle in rank_styles.items():
        h = s_hists[f'dr_reco_gen_{rank}_{coll}'][{"cut": cut, "samp": sum}]
        hep.histplot(h, ax=ax, histtype='step', density=True,
                     label=rstyle['label'], color=cstyle['color'],
                     linestyle=rstyle['ls'], linewidth=2)
    ax.axvline(0.1, color='gray', ls=':', lw=1.5, label=r'$\Delta R = 0.1$')
    ax.set_xlabel(r'$\Delta R$(reco, gen)')
    ax.set_ylabel('A.U.')
    ax.set_title(rf'{cstyle["label"]} merged: reco-gen $\Delta R$ — all samples')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'plots/hist_{seltag}_dr_reco_gen_{coll}_allsamps.png')
    plt.close(fig)

_run_sig_plots([
    (
        rf'{coll_styles["gedlpt"]["label"]} merged: reco-gen $\Delta R$ leading vs subleading',
        'merged_dr_reco_gen_gedlpt',
        ['dr_reco_gen_lead_gedlpt', 'dr_reco_gen_sublead_gedlpt'],
        True, 'To gen leading', 'To gen subleading',
    ),
    (
        rf'{coll_styles["alpt"]["label"]} merged: reco-gen $\Delta R$ leading vs subleading',
        'merged_dr_reco_gen_alpt',
        ['dr_reco_gen_lead_alpt', 'dr_reco_gen_sublead_alpt'],
        True, 'To gen leading', 'To gen subleading',
    ),
])


# ── Section 4: reco dxy (merged only) ────────────────────────────────────────
# One plot overlaying both collections.

fig, ax = plt.subplots(figsize=size)
hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
for coll, cstyle in coll_styles.items():
    h = s_hists[f'reco_dxy_{coll}'][{"cut": cut, "samp": sum}]
    hep.histplot(h, ax=ax, histtype='step', density=True,
                 label=cstyle['label'], color=cstyle['color'], linewidth=2)
ax.set_xlabel(r'Reco $|d_{xy}|$ [cm]')
ax.set_ylabel('A.U.')
ax.set_title(r'Merged reco electron $|d_{xy}|$ — all samples')
ax.legend()
plt.tight_layout()
plt.savefig(f'plots/hist_{seltag}_reco_dxy_allcolls_allsamps.png')
plt.close(fig)

_run_sig_plots([
    (
        r'Merged: reco $|d_{xy}|$ — GED+LowPt vs AllLowPt',
        'merged_reco_dxy',
        ['reco_dxy_gedlpt', 'reco_dxy_alpt'],
        False, 'GED+LowPt', 'AllLowPt',
    ),
])


# ── Section 5: reco-gen pt difference (merged only) ──────────────────────────
# Guard: these histograms only exist after the coffea job is re-run with the
# updated mergedmatch.py config that fills dpt_* histograms.
# Temporarily disabled: mergedmatch.py's FILL_DPT_HISTS is currently False, so
# these histograms don't exist in the saved output. Flip both flags back on
# together to restore.
PLOT_DPT_HISTS = False

if PLOT_DPT_HISTS:
    _dpt_specs = [
        ('abs', r'$p_T^{\rm reco} - p_T^{\rm gen}$ [GeV]'),
        ('rel', r'$(p_T^{\rm reco} - p_T^{\rm gen})/p_T^{\rm gen}$'),
    ]

    # All-samples overlay: leading vs subleading, per (collection, quantity type)
    for qty, xlabel in _dpt_specs:
        for coll, cstyle in coll_styles.items():
            fig, ax = plt.subplots(figsize=size)
            hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
            for rank, rstyle in rank_styles.items():
                h = s_hists[f'dpt_{qty}_{rank}_{coll}'][{"cut": cut, "samp": sum}]
                _varbin_histplot(h, ax, histtype='step',
                                 label=rstyle['label'], color=cstyle['color'],
                                 linestyle=rstyle['ls'], linewidth=2)
            ax.set_xlabel(xlabel)
            ax.set_ylabel('A.U.')
            ax.set_title(rf'{cstyle["label"]} merged: $\Delta p_T$ ({qty}) — all samples')
            ax.legend()
            plt.tight_layout()
            plt.savefig(f'plots/hist_{seltag}_dpt_{qty}_{coll}_allsamps.png')
            plt.close(fig)

    for qty, xlabel in _dpt_specs:
        for coll, cstyle in coll_styles.items():
            for series in series_list:
                m1s    = series['m1s']
                deltas = series['deltas']
                ctaus  = series['ctaus']
                tag    = series['tag']
                fig, ax = plt.subplots(figsize=size)
                hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
                cmap_idx = 0
                for m1 in m1s:
                    for delta in deltas:
                        for ctau in ctaus:
                            sname = _samp_name(m1, delta, ctau)
                            if sname is None:
                                cmap_idx += 1
                                continue
                            color = f'C{cmap_idx}'
                            label = _sig_label(m1, delta, ctau)
                            for rank, rstyle in rank_styles.items():
                                h = s_hists[f'dpt_{qty}_{rank}_{coll}'][{'cut': cut, 'samp': sname}]
                                _varbin_histplot(h, ax, histtype='step',
                                                 label=f'{label} — {rstyle["label"]}',
                                                 color=color, linestyle=rstyle['ls'], linewidth=2)
                            cmap_idx += 1
                ax.set_xlabel(xlabel)
                ax.set_ylabel('A.U.')
                ax.legend(fontsize=13)
                plt.title(rf'{cstyle["label"]} merged: $\Delta p_T$ ({qty}) — {tag}')
                plt.tight_layout()
                plt.savefig(f'plots/hist_{seltag}_merged_dpt_{qty}_{coll}_{tag}.png')
                plt.close(fig)

    # 2D colormesh: pt difference vs gen ele pt (all samples)
    # Divide by dpt bin widths so the colour axis is events / (dpt unit) — this
    # compensates for the variable-width dpt bins and keeps the colour semi-continuous
    # across the dense region near 0.
    for qty, ylabel in _dpt_specs:
        for rank in ['lead', 'sublead']:
            rank_label = 'leading' if rank == 'lead' else 'subleading'
            for coll, cstyle in coll_styles.items():
                h2d  = s_hists[f'dpt_{qty}_{rank}_vs_genpt_{coll}'][{"cut": cut, "samp": sum}]
                pt_edges  = h2d.axes[0].edges
                dpt_edges = h2d.axes[1].edges
                dpt_widths = np.diff(dpt_edges)
                vals = h2d.values() / dpt_widths[np.newaxis, :]  # events / (dpt bin width)
                if not np.any(vals > 0):
                    continue
                fig, ax = plt.subplots(figsize=size)
                hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
                pcm = ax.pcolormesh(pt_edges, dpt_edges, np.where(vals > 0, vals, np.nan).T,
                                    cmap='viridis')
                clabel = r'Events / GeV' if qty == 'abs' else r'Events / (unit $\Delta p_T/p_T$)'
                plt.colorbar(pcm, ax=ax, label=clabel)
                ax.set_xlabel(rf'Gen $p_T$ ({rank_label}) [GeV]')
                ax.set_ylabel(ylabel)
                ax.set_title(
                    rf'{cstyle["label"]} merged — $\Delta p_T$ ({qty}) vs gen $p_T$ ({rank_label}) — all samples'
                )
                plt.tight_layout()
                plt.savefig(f'plots/hist_{seltag}_dpt_{qty}_{rank}_vs_genpt_{coll}_2D_allsamps.png')
                plt.close(fig)

    # 2D colormesh: pt difference vs gen ee dR (all samples)
    for qty, ylabel in _dpt_specs:
        for rank in ['lead', 'sublead']:
            rank_label = 'leading' if rank == 'lead' else 'subleading'
            for coll, cstyle in coll_styles.items():
                h2d  = s_hists[f'dpt_{qty}_{rank}_vs_eedr_{coll}'][{"cut": cut, "samp": sum}]
                dr_edges  = h2d.axes[0].edges
                dpt_edges = h2d.axes[1].edges
                dpt_widths = np.diff(dpt_edges)
                vals = h2d.values() / dpt_widths[np.newaxis, :]  # events / (dpt bin width)
                if not np.any(vals > 0):
                    continue
                fig, ax = plt.subplots(figsize=size)
                hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
                pcm = ax.pcolormesh(dr_edges, dpt_edges, np.where(vals > 0, vals, np.nan).T,
                                    cmap='viridis')
                clabel = r'Events / GeV' if qty == 'abs' else r'Events / (unit $\Delta p_T/p_T$)'
                plt.colorbar(pcm, ax=ax, label=clabel)
                ax.set_xlabel(r'Gen $\Delta R(e^+e^-)$')
                ax.set_ylabel(ylabel)
                ax.set_title(
                    rf'{cstyle["label"]} merged — $\Delta p_T$ ({qty}) vs gen $\Delta R$ ({rank_label}) — all samples'
                )
                plt.tight_layout()
                plt.savefig(f'plots/hist_{seltag}_dpt_{qty}_{rank}_vs_eedr_{coll}_2D_allsamps.png')
                plt.close(fig)


# ── Section 6: merged vs resolved comparison — 1D hists ──────────────────────
# For each comparison quantity, one plot per collection overlaying merged and resolved.

comp_specs = [
    ('gen_lxy',      r'Gen $L_{xy}$ [cm]',                True),
    ('gen_ee_dr',    r'Gen $\Delta R(e^+e^-)$',            True),
    ('gen_ee_pt',    r'Gen $p_T(e^+e^-)$ [GeV]',          False),
    ('reco_trkchi2', r'Reco track $\chi^2/\mathrm{dof}$', True),
]

for histbase, xlabel, do_logy in comp_specs:
    for coll, cstyle in coll_styles.items():
        fig, ax = plt.subplots(figsize=size)
        hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
        for scenario, sstyle in scenario_styles.items():
            h = s_hists[f'{histbase}_{scenario}_{coll}'][{"cut": cut, "samp": sum}]
            hep.histplot(h, ax=ax, histtype='step', density=True,
                         label=sstyle['label'], color=cstyle['color'],
                         linestyle=sstyle['ls'], linewidth=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('A.U.')
        if do_logy:
            ax.set_yscale('log')
        ax.set_title(rf'{cstyle["label"]} — {xlabel}: merged vs resolved — all samples')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'plots/hist_{seltag}_{histbase}_{coll}_merged_vs_resolved.png')
        plt.close(fig)

_run_sig_plots([
    (
        rf'{cstyle["label"]} {xlabel}: merged vs resolved',
        f'{histbase}_{coll}_merged_vs_resolved',
        [f'{histbase}_merged_{coll}', f'{histbase}_resolved_{coll}'],
        do_logy, 'Merged', 'Resolved',
    )
    for histbase, xlabel, do_logy in comp_specs
    for coll, cstyle in coll_styles.items()
])


# ── Section 7: 2D colormesh — gen dR vs gen Lxy ──────────────────────────────
# One plot per (scenario, collection).

for scenario, sstyle in scenario_styles.items():
    for coll, cstyle in coll_styles.items():
        histname = f'dr_gen_vs_lxy_{scenario}_{coll}'
        h2d  = s_hists[histname][{"cut": cut, "samp": sum}]
        lxy_edges = h2d.axes[0].edges
        dr_edges  = h2d.axes[1].edges
        vals = h2d.values()
        if not np.any(vals > 0):
            continue

        fig, ax = plt.subplots(figsize=size)
        hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
        vals_pos = np.where(vals > 0, vals, np.nan)
        pcm = ax.pcolormesh(lxy_edges, dr_edges, vals_pos.T, cmap='viridis', norm=LogNorm())
        plt.colorbar(pcm, ax=ax, label='Events')
        ax.set_xlabel(r'Gen $L_{xy}$ [cm]')
        ax.set_ylabel(r'Gen $\Delta R(e^+e^-)$')
        ax.set_xscale('log')
        ax.set_title(
            rf'{cstyle["label"]} ({sstyle["label"]}) — gen $\Delta R$ vs $L_{{xy}}$ — all samples'
        )
        plt.tight_layout()
        plt.savefig(f'plots/hist_{seltag}_dr_gen_vs_lxy_{scenario}_{coll}_2D_allsamps.png')
        plt.close(fig)


# ── Section 8: reco category distributions ───────────────────────────────────

_cat_order = [
    'zero_matched', 'one_ele',
    'one_photon', 'one_track', 'one_conversion', 'one_ootphoton', 'one_pfcand', 'one_losttrack',
    'merged',
    'merged_photon', 'merged_track', 'merged_conversion', 'merged_ootphoton', 'merged_pfcand', 'merged_losttrack',
    'ele_photon', 'ele_track', 'ele_conversion', 'ele_ootphoton', 'ele_pfcand', 'ele_losttrack',
    'resolved',
    'two_photon', 'two_track', 'two_conversion', 'two_ootphoton', 'two_pfcand', 'two_losttrack',
]
_CAT_LABELS = {
    'zero_matched':      'zero\nmatched',
    'one_ele':           'one\nele',
    'one_photon':        'one\nphoton',
    'one_track':         'one\ntrack',
    'one_conversion':    'one\nconv',
    'one_ootphoton':     'one\nootpho',
    'one_pfcand':        'one\npfcand',
    'one_losttrack':     'one\nlosttrk',
    'merged':            'merged\n(ele)',
    'merged_photon':     'merged\n(pho)',
    'merged_track':      'merged\n(trk)',
    'merged_conversion': 'merged\n(conv)',
    'merged_ootphoton':  'merged\n(ootpho)',
    'merged_pfcand':     'merged\n(pfcand)',
    'merged_losttrack':  'merged\n(losttrk)',
    'ele_photon':        'ele+\nphoton',
    'ele_track':         'ele+\ntrack',
    'ele_conversion':    'ele+\nconv',
    'ele_ootphoton':     'ele+\nootpho',
    'ele_pfcand':        'ele+\npfcand',
    'ele_losttrack':     'ele+\nlosttrk',
    'resolved':          'resolved',
    'two_photon':        'two\nphoton',
    'two_track':         'two\ntrack',
    'two_conversion':    'two\nconv',
    'two_ootphoton':     'two\nootpho',
    'two_pfcand':        'two\npfcand',
    'two_losttrack':     'two\nlosttrk',
}


def _cat_fracs(h_cat):
    """Normalised fraction and uncertainty per category from a reco_cat hist.

    Returns (fracs, errs) dicts keyed by category name.
    Errors are sqrt(variances) / total_weight (simple propagation, normalization held fixed).
    """
    cats  = list(h_cat.axes['cat'])
    vals  = h_cat.values()
    variances = h_cat.variances()
    d    = dict(zip(cats, vals))
    dvar = dict(zip(cats, variances))
    total = sum(d.values())
    if total <= 0:
        zero = {c: 0.0 for c in _cat_order}
        return zero, zero
    fracs = {c: d.get(c, 0.0) / total for c in _cat_order}
    errs  = {c: np.sqrt(dvar.get(c, 0.0)) / total for c in _cat_order}
    return fracs, errs


def _top6_cats(fracs_list):
    """Return the 6 category names with highest aggregate fraction across the
    given list of fracs dicts (one entry per signal point in the plot)."""
    agg = {c: sum(f.get(c, 0.0) for f in fracs_list) for c in _cat_order}
    return sorted(_cat_order, key=lambda c: agg[c], reverse=True)[:6]


def _display_data(fracs, errs, top6):
    """Collapse all categories not in top6 into a single 'others' bin.

    Returns (vals, errs, xlabels) lists of length 7, with 'others' last.
    Error on 'others' is the quadrature sum of the individual errors.
    """
    others_cats = [c for c in _cat_order if c not in top6]
    others_frac = sum(fracs.get(c, 0.0) for c in others_cats)
    others_err  = np.sqrt(sum(errs.get(c, 0.0)**2 for c in others_cats))
    vals  = [fracs.get(c, 0.0) for c in top6] + [others_frac]
    errs_ = [errs.get(c, 0.0)  for c in top6] + [others_err]
    xlbls = [_CAT_LABELS[c]    for c in top6] + ['others']
    return vals, errs_, xlbls


# ─ All-samples aggregate (one plot per collection) ───────────────────────────

for coll, cstyle in coll_styles.items():
    fig, ax = plt.subplots(figsize=size)
    hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
    h           = s_hists[f'reco_cat_{coll}'][{'cut': cut, 'samp': sum}]
    fracs, errs = _cat_fracs(h)
    top6        = _top6_cats([fracs])
    vals, errs_, xlbls = _display_data(fracs, errs, top6)
    x = np.arange(7)
    ax.bar(x, vals, yerr=errs_,
           color=cstyle['color'], alpha=0.8, edgecolor='black', linewidth=0.8,
           error_kw={'ecolor': 'black', 'capsize': 4, 'elinewidth': 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels(xlbls, fontsize=15)
    ax.set_ylabel('Fraction of events')
    ax.set_ylim(0, None)
    ax.set_title(rf'{cstyle["label"]}: reco category distribution — all samples')
    plt.tight_layout()
    plt.savefig(f'plots/hist_{seltag}_reco_cat_{coll}_allsamps.png')
    plt.close(fig)

# ─ Per-series (one plot per series × collection) ─────────────────────────────

for coll, cstyle in coll_styles.items():
    for series in series_list:
        m1s    = series['m1s']
        deltas = series['deltas']
        ctaus  = series['ctaus']
        tag    = series['tag']

        sig_pts = [(m1, d, ct) for m1 in m1s for d in deltas for ct in ctaus]
        sig_pts = [(m1, d, ct) for m1, d, ct in sig_pts if _samp_name(m1, d, ct) is not None]
        if not sig_pts:
            continue

        # Collect per-sample fracs first so top6 is shared across all bars in
        # the plot, keeping the x-axis consistent for visual comparison.
        pt_data   = []
        all_fracs = []
        for m1, delta, ctau in sig_pts:
            sname = _samp_name(m1, delta, ctau)
            h     = s_hists[f'reco_cat_{coll}'][{'cut': cut, 'samp': sname}]
            f, e  = _cat_fracs(h)
            pt_data.append((m1, delta, ctau, f, e))
            all_fracs.append(f)

        top6  = _top6_cats(all_fracs)
        n_pts = len(sig_pts)
        bar_w = 0.8 / n_pts
        x     = np.arange(7)

        fig, ax = plt.subplots(figsize=size)
        hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)

        for idx, (m1, delta, ctau, fracs, errs) in enumerate(pt_data):
            vals, errs_, xlbls = _display_data(fracs, errs, top6)
            offsets = x - 0.4 + bar_w * (idx + 0.5)
            ax.bar(offsets, vals, width=bar_w, label=_sig_label(m1, delta, ctau),
                   yerr=errs_, color=f'C{idx}', alpha=0.8, edgecolor='black', linewidth=0.5,
                   error_kw={'ecolor': 'black', 'capsize': 3, 'elinewidth': 1.0})

        ax.set_xticks(x)
        ax.set_xticklabels(xlbls, fontsize=15)
        ax.set_ylabel('Fraction of events')
        ax.set_ylim(0, None)
        ax.set_title(rf'{cstyle["label"]}: reco category distribution — {tag}')
        ax.legend(fontsize=13)
        plt.tight_layout()
        plt.savefig(f'plots/hist_{seltag}_reco_cat_{coll}_{tag}.png')
        plt.close(fig)


# ── Section 8b: print reco-category fractions — m1-narrow series only ────────
# Full per-category breakdown (not collapsed to top6 + others like the bar plot).

_narrow_series = next(s for s in series_list if s['tag'] == 'ctau-10_delta-0p1_m1-narrow')

for coll, cstyle in coll_styles.items():
    print(f"\n{'='*70}")
    print(f"{cstyle['label']} ({coll}) — reco category fractions — {_narrow_series['tag']}")
    print(f"{'='*70}")
    for m1 in _narrow_series['m1s']:
        for delta in _narrow_series['deltas']:
            for ctau in _narrow_series['ctaus']:
                sname = _samp_name(m1, delta, ctau)
                if sname is None:
                    print(f"\n{_sig_label(m1, delta, ctau)}: sample not found, skipping")
                    continue
                h = s_hists[f'reco_cat_{coll}'][{'cut': cut, 'samp': sname}]
                fracs, errs = _cat_fracs(h)
                print(f"\n{_sig_label(m1, delta, ctau)}  (sample: {sname})")
                for c in _cat_order:
                    print(f"  {c:20s} {fracs[c]*100:6.2f}%  +- {errs[c]*100:5.2f}%")


# ── Section 9: gen ee kinematics (dR, pT) by reco category ───────────────────

_bycat_styles = {
    'zero_matched': {'label': 'Zero matched', 'color': 'C3', 'ls': ':'},
    'merged':       {'label': 'Merged',        'color': 'C1', 'ls': '-'},
    'resolved':     {'label': 'Resolved',      'color': 'C2', 'ls': '--'},
}

_bycat_vars = [
    ('gen_ee_dr',  r'Gen $\Delta R(e^+e^-)$'),
    ('gen_ee_pt',  r'Gen $p_T(e^+e^-)$ [GeV]'),
    ('gen_ee_eta', r'Gen $\eta(e^+e^-)$'),
]

# Representative individual signal points
_bycat_indiv = [
    (0.5, 0.1,   1),
    (0.5, 0.1,  10),
    (0.5, 0.1, 100),
    (5,   0.1,  10),
]


def _plot_bycat(ax, h_full):
    """Overlay the three categories on ax (density-normalised steps)."""
    for cat, kstyle in _bycat_styles.items():
        h = h_full[{'cat': cat}]
        if np.sum(h.values()) == 0:
            continue
        hep.histplot(h, ax=ax, histtype='step', density=True,
                     label=kstyle['label'], color=kstyle['color'],
                     linestyle=kstyle['ls'], linewidth=2)


# ─ All-samples aggregate ─────────────────────────────────────────────────────

for coll, cstyle in coll_styles.items():
    for histbase, xlabel in _bycat_vars:
        fig, ax = plt.subplots(figsize=size)
        hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
        h_full = s_hists[f'{histbase}_cat_{coll}'][{'cut': cut, 'samp': sum}]
        _plot_bycat(ax, h_full)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('A.U.')
        ax.set_title(rf'{cstyle["label"]}: {xlabel} — by reco category, all samples')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'plots/hist_{seltag}_{histbase}_bycat_{coll}_allsamps.png')
        plt.close(fig)

# ─ Individual signal samples ─────────────────────────────────────────────────

for coll, cstyle in coll_styles.items():
    for histbase, xlabel in _bycat_vars:
        for m1, delta, ctau in _bycat_indiv:
            sname = _samp_name(m1, delta, ctau)
            if sname is None:
                continue
            fig, ax = plt.subplots(figsize=size)
            hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
            h_full = s_hists[f'{histbase}_cat_{coll}'][{'cut': cut, 'samp': sname}]
            _plot_bycat(ax, h_full)
            ax.set_xlabel(xlabel)
            ax.set_ylabel('A.U.')
            ax.set_title(
                rf'{cstyle["label"]}: {xlabel} — by reco category, {_sig_label(m1, delta, ctau)}'
            )
            ax.legend()
            plt.tight_layout()
            m1s = str(m1).replace('.', 'p')
            ds  = str(delta).replace('.', 'p')
            plt.savefig(
                f'plots/hist_{seltag}_{histbase}_bycat_{coll}_m1-{m1s}_d-{ds}_ctau-{ctau}.png'
            )
            plt.close(fig)
