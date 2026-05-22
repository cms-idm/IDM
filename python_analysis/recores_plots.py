import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import mplhep as hep
import coffea.util as util
import analysisTools.utils as utils
import analysisTools.plotTools as ptools
import os

outdir    = 'workarea'
vers      = 'May2026'
selection = 'ansplitvtx'    # change to match the selection used when running the histmaker
hists_tag = 'recores'
saved_signal_hists = f"{outdir}/hists_sig{vers}_{selection}-sel_{hists_tag}.coffea"
seltag = 'prevtx'
cut    = 'cut8'
year   = 2024
size   = (16, 12)

s_hists = util.load(saved_signal_hists)[0]
s_pts   = utils.get_signal_point_dict(s_hists)

os.makedirs('plots', exist_ok=True)

# ── helpers ───────────────────────────────────────────────────────────────────

def profile_from_2D(h2d):
    """Weighted mean and std of the resolution axis (axis 1) in each x bin (axis 0)."""
    vals        = h2d.values()
    x_centers   = h2d.axes[0].centers
    res_centers = h2d.axes[1].centers
    total = vals.sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean  = np.where(total > 0, (vals * res_centers[np.newaxis, :]).sum(axis=1)    / total, np.nan)
        mean2 = np.where(total > 0, (vals * res_centers[np.newaxis, :]**2).sum(axis=1) / total, np.nan)
        std   = np.sqrt(np.maximum(mean2 - mean**2, 0))
    return x_centers, mean, std, total > 0

def get_samp_name(m1, delta, ctau):
    row = s_pts[np.isclose(s_pts.m1, m1) & np.isclose(s_pts.delta, delta) & (s_pts.ctau == ctau)]
    return row.index[0] if not row.empty else None

def sample_label(row):
    return rf"$M_1$={row['m1']:.3g}, $\Delta$={row['delta']:.3g}, $c\tau$={row['ctau']:.0f} mm"

series_list = [
    {'m1s': [0.05, 0.5, 5, 50], 'deltas': [0.1],            'ctaus': [10],         'tag': 'ctau-10_delta-0p1_m1-wide'   },
    {'m1s': [0.5, 1, 2, 5],     'deltas': [0.1],            'ctaus': [10],         'tag': 'ctau-10_delta-0p1_m1-narrow' },
    {'m1s': [0.5],              'deltas': [0.1],            'ctaus': [1, 10, 100], 'tag': 'm1-0p5_delta-0p1'            },
    {'m1s': [0.5],              'deltas': [0.05, 0.1, 0.2], 'ctaus': [10],         'tag': 'm1-0p5_ctau-10'              },
    {'m1s': [5],                'deltas': [0.1],            'ctaus': [1, 10, 100], 'tag': 'm1-5_delta-0p1'              },
    {'m1s': [5],                'deltas': [0.05, 0.1, 0.2], 'ctaus': [10],         'tag': 'm1-5_ctau-10'                },
]

qty_configs = [
    {'qty': 'pt', 'title': r'$p_T$ Resolution',  'xlabel': r'$(p_T^\mathrm{reco} - p_T^\mathrm{gen})/p_T^\mathrm{gen}$'},
    {'qty': 'e',  'title': r'Energy Resolution',  'xlabel': r'$(E^\mathrm{reco} - E^\mathrm{gen})/E^\mathrm{gen}$'},
]

gen_var_configs = [
    {'var': 'genpt',  'label': r'$p_T^\mathrm{gen}$ [GeV]', 'doLogx': False},
    {'var': 'genlxy', 'label': r'$L_{xy}$ [cm]',             'doLogx': True},
    {'var': 'geneta', 'label': r'$\eta^\mathrm{gen}$',        'doLogx': False},
]

style_dict_base = {
    'rebin': 1j, 'xlim': None,
    'doLogy': False, 'doLogx': False, 'doDensity': True, 'doYerr': False,
    'xlabel': None, 'ylabel': None, 'label': None, 'flow': None,
    'doSave': False, 'doCMSLabel': False,
    'ls': ['-', '--'],
}

# ── Section 1: 1D resolution distributions ───────────────────────────────────

for qcfg in qty_configs:
    qty = qcfg['qty']
    plot_dict = {
        'variable': [f'res_{qty}_lpt', f'res_{qty}_ged'],
        'year': year,
        'cut': cut,
    }

    # Summed over all samples
    h_lpt = s_hists[f'res_{qty}_lpt'][{"cut": cut, "samp": sum}]
    h_ged = s_hists[f'res_{qty}_ged'][{"cut": cut, "samp": sum}]
    fig, ax = plt.subplots(figsize=size)
    hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
    hep.histplot([h_lpt, h_ged], ax=ax, histtype='step', density=True,
                 label=['LowPt', 'GED'], color=['C0', 'C1'])
    ax.set_xlabel(qcfg['xlabel'])
    ax.set_ylabel('A.U.')
    ax.set_yscale('log')
    ax.set_title(f'Electron {qcfg["title"]} — all samples')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'plots/hist_{seltag}_res_{qty}_1D_allsamps.png')
    plt.close(fig)

    # Per-series: one plot per series, all samples overlaid, LowPt solid / GED dashed
    for series in series_list:
        m1s, deltas, ctaus, tag = series['m1s'], series['deltas'], series['ctaus'], series['tag']
        fig, ax = plt.subplots(figsize=size)
        style_dict = {**style_dict_base, 'fig': fig, 'ax': ax}
        cmap_idx = 0
        for m1 in m1s:
            for delta in deltas:
                for ctau in ctaus:
                    ptools.plot_signal_1D(s_hists, m1, delta, ctau, plot_dict, style_dict, cmap_idx=cmap_idx)
                    cmap_idx += 1
        hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
        handles, labels = ax.get_legend_handles_labels()
        handles += [
            mlines.Line2D([], [], color='black', ls='-',  label='LowPt'),
            mlines.Line2D([], [], color='black', ls='--', label='GED'),
        ]
        labels += ['LowPt', 'GED']
        ax.legend(handles=handles, labels=labels, fontsize=11)
        ax.set_xlabel(qcfg['xlabel'])
        ax.set_yscale('log')
        ax.set_title(rf'Electron {qcfg["title"]}: $M_1$={m1s}, $\Delta$={deltas}, $c\tau$={ctaus} mm')
        plt.tight_layout()
        plt.savefig(f'plots/hist_{seltag}_res_{qty}_1D_{tag}.png')
        plt.close(fig)

# ── Section 2: Profile plots (mean ± RMS vs gen variable) ────────────────────

for qcfg in qty_configs:
    qty = qcfg['qty']
    for gvcfg in gen_var_configs:
        gvar = gvcfg['var']

        # Summed over all samples
        x_l, mn_l, sd_l, vl = profile_from_2D(s_hists[f'res_{qty}_vs_{gvar}_lpt'][{"cut": cut, "samp": sum}])
        x_g, mn_g, sd_g, vg = profile_from_2D(s_hists[f'res_{qty}_vs_{gvar}_ged'][{"cut": cut, "samp": sum}])
        fig, ax = plt.subplots(figsize=size)
        hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
        ax.errorbar(x_l[vl], mn_l[vl], yerr=sd_l[vl], fmt='o-', color='C0', capsize=3, label='LowPt (mean ± RMS)')
        ax.errorbar(x_g[vg], mn_g[vg], yerr=sd_g[vg], fmt='s-', color='C1', capsize=3, label='GED (mean ± RMS)')
        ax.axhline(0, color='gray', linestyle=':', linewidth=1)
        ax.set_xlabel(gvcfg['label'])
        ax.set_ylabel(qcfg['xlabel'])
        if gvcfg['doLogx']:
            ax.set_xscale('log')
        ax.set_title(f'Electron {qcfg["title"]} vs {gvcfg["label"]} — all samples')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'plots/hist_{seltag}_res_{qty}_vs_{gvar}_profile_allsamps.png')
        plt.close(fig)

        # Per-series: LowPt solid, GED dashed; same color per sample
        for series in series_list:
            m1s, deltas, ctaus, tag = series['m1s'], series['deltas'], series['ctaus'], series['tag']
            fig, ax = plt.subplots(figsize=size)
            hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
            cmap_idx = 0
            for m1 in m1s:
                for delta in deltas:
                    for ctau in ctaus:
                        sname = get_samp_name(m1, delta, ctau)
                        if sname is None:
                            cmap_idx += 1
                            continue
                        color  = ptools.cmap[cmap_idx]
                        slabel = sample_label(s_pts.loc[sname])
                        x_l, mn_l, sd_l, vl = profile_from_2D(s_hists[f'res_{qty}_vs_{gvar}_lpt'][{"cut": cut, "samp": sname}])
                        x_g, mn_g, sd_g, vg = profile_from_2D(s_hists[f'res_{qty}_vs_{gvar}_ged'][{"cut": cut, "samp": sname}])
                        ax.errorbar(x_l[vl], mn_l[vl], yerr=sd_l[vl], fmt='o-',  color=color, capsize=3, label=slabel)
                        ax.errorbar(x_g[vg], mn_g[vg], yerr=sd_g[vg], fmt='s--', color=color, capsize=3, label='_nolegend_')
                        cmap_idx += 1
            ax.axhline(0, color='gray', linestyle=':', linewidth=1)
            handles, labels = ax.get_legend_handles_labels()
            handles += [
                mlines.Line2D([], [], color='black', ls='-',  marker='o', label='LowPt'),
                mlines.Line2D([], [], color='black', ls='--', marker='s', label='GED'),
            ]
            labels += ['LowPt', 'GED']
            ax.legend(handles=handles, labels=labels, fontsize=11)
            ax.set_xlabel(gvcfg['label'])
            ax.set_ylabel(qcfg['xlabel'])
            if gvcfg['doLogx']:
                ax.set_xscale('log')
            ax.set_title(rf'Electron {qcfg["title"]} vs {gvcfg["label"]}: $M_1$={m1s}, $\Delta$={deltas}, $c\tau$={ctaus} mm')
            plt.tight_layout()
            plt.savefig(f'plots/hist_{seltag}_res_{qty}_vs_{gvar}_profile_{tag}.png')
            plt.close(fig)

# ── Section 3: 2D colormesh plots (summed over all samples) ──────────────────

_2d_specs = [
    ('res_pt_vs_genpt_lpt',  r'LowPt $p_T$ Res vs Gen $p_T$',         False),
    ('res_pt_vs_genpt_ged',  r'GED $p_T$ Res vs Gen $p_T$',           False),
    ('res_e_vs_genpt_lpt',   r'LowPt Energy Res vs Gen $p_T$',         False),
    ('res_e_vs_genpt_ged',   r'GED Energy Res vs Gen $p_T$',           False),
    ('res_pt_vs_genlxy_lpt', r'LowPt $p_T$ Res vs Gen $L_{xy}$',      True),
    ('res_pt_vs_genlxy_ged', r'GED $p_T$ Res vs Gen $L_{xy}$',        True),
    ('res_e_vs_genlxy_lpt',  r'LowPt Energy Res vs Gen $L_{xy}$',     True),
    ('res_e_vs_genlxy_ged',  r'GED Energy Res vs Gen $L_{xy}$',       True),
    ('res_pt_vs_geneta_lpt', r'LowPt $p_T$ Res vs Gen $\eta$',        False),
    ('res_pt_vs_geneta_ged', r'GED $p_T$ Res vs Gen $\eta$',          False),
    ('res_e_vs_geneta_lpt',  r'LowPt Energy Res vs Gen $\eta$',       False),
    ('res_e_vs_geneta_ged',  r'GED Energy Res vs Gen $\eta$',         False),
]

for histname, title, doLogx in _2d_specs:
    h2d       = s_hists[histname][{"cut": cut, "samp": sum}]
    x_edges   = h2d.axes[0].edges
    res_edges = h2d.axes[1].edges
    vals      = h2d.values()

    fig, ax = plt.subplots(figsize=size)
    hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
    pcm = ax.pcolormesh(x_edges, res_edges, vals.T, cmap='viridis')
    plt.colorbar(pcm, ax=ax, label='Events')
    ax.set_xlabel(h2d.axes[0].label)
    ax.set_ylabel(h2d.axes[1].label)
    if doLogx:
        ax.set_xscale('log')
    ax.set_title(f'{title} — all samples')
    plt.tight_layout()
    plt.savefig(f'plots/hist_{seltag}_{histname}_2D_allsamps.png')
    plt.close(fig)
