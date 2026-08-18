import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import LogNorm
import mplhep as hep
import sys
import os

# Make the script runnable regardless of the caller's current working
# directory by anchoring paths to the repo root (one level up from this file).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import coffea.util as util
import analysisTools.utils as utils
import analysisTools.plotTools as ptools

plt.rcParams.update({
    'font.size':        20,
    'axes.titlesize':   24,
    'axes.labelsize':   20,
    'xtick.labelsize':  20,
    'ytick.labelsize':  20,
    'legend.fontsize':  18,
    'figure.titlesize': 24,
})

outdir    = os.path.join(REPO_ROOT, 'workarea')
vers      = 'Jul2026noID'
selection = 'anmatchvtx'
hists_tag = 'genmatching'
saved_signal_hists = f"{outdir}/hists_sig{vers}_{selection}-sel_{hists_tag}.coffea"
seltag   = 'nocuts'
cut      = 'cut1'
year     = 2024
size     = (16, 12)
ann_mode = None   # 'wgt_sum': annotate with weighted bin yield
                       # 'n_eff':   annotate with N_eff = values²/variances (raw count for uniform-weight MC)
                       # None:      no annotation

s_hists = util.load(saved_signal_hists)[0]
s_pts   = utils.get_signal_point_dict(s_hists)

plotdir = os.path.join(REPO_ROOT, 'plots', 'genmatching')
os.makedirs(plotdir, exist_ok=True)

# ── display metadata ──────────────────────────────────────────────────────────

coll_styles = {
    'gedlpt': {'label': 'GED+LowPt', 'color': 'C0'},
    'alpt':   {'label': 'AllLowPt',  'color': 'C1'},
}
rank_styles = {
    'lead':    {'label': 'Leading',    'ls': '-'},
    'sublead': {'label': 'Subleading', 'ls': '--'},
}

# ── bin annotation helpers ────────────────────────────────────────────────────

def _ann_values(h):
    """Per-bin annotation value selected by ann_mode."""
    vals = h.values()
    if ann_mode == 'n_eff':
        var      = h.variances()
        safe_var = np.where(var > 0, var, 1)
        return np.where(var > 0, np.round(vals**2 / safe_var), 0).astype(int)
    return vals  # 'wgt_sum'

def _ann_fmt(v):
    """Format a single annotation value."""
    if ann_mode == 'n_eff':
        return str(int(v))
    return f'{v:.3g}'

def _annotate_counts(ax, h, color, x_offset=0.0, fontsize=11):
    """Print per-bin annotation just above the top of each non-empty bin."""
    if ann_mode is None:
        return
    vals  = h.values()
    total = vals.sum()
    if total <= 0:
        return
    dens = vals / total  # bin_width = 1 for Integer axis
    ann  = _ann_values(h)
    for x, y, v in zip(h.axes[0].centers, dens, ann):
        if v > 0:
            ax.text(x + x_offset, y, _ann_fmt(v),
                    ha='center', va='bottom', color=color, fontsize=fontsize)

# Secondary variable specs for 2D hists: (histname_suffix, x-axis label, log-x?)
_2d_var_specs = [
    ('dRj',           r'Min $\Delta R$(ele, jet)',              False),
    ('PFRelIso',      r'PF Relative Isolation',                 False),
    ('miniRelIsoCorr',r'Corrected Mini Relative Isolation',     False),
    ('pt',            r'Reco $p_T$ [GeV]',                      False),
    ('eta',           r'Reco $\eta$',                           False),
    ('genlxy',        r'Gen $L_{xy}$ [cm]',                     True),
    ('genpt',         r'Gen $p_T$ [GeV]',                       False),
    ('trkChi2',       r'Track $\chi^2/\mathrm{dof}$',           False),
    ('dxy',           r'Reco $d_{xy}$ [cm]',                    False),
]

# ── helpers ───────────────────────────────────────────────────────────────────

def median_profile_from_2D(h2d):
    """Median and 90th-percentile of the dR (axis 1) per x bin (axis 0).

    Returns (x_centers, x_edges, medians, p90s, valid).
    """
    vals       = h2d.values()
    x_centers  = h2d.axes[0].centers
    x_edges    = h2d.axes[0].edges
    dr_centers = h2d.axes[1].centers
    n_x        = len(x_centers)
    medians    = np.full(n_x, np.nan)
    p90s       = np.full(n_x, np.nan)
    valid      = np.zeros(n_x, dtype=bool)
    for i, counts in enumerate(vals):
        total = counts.sum()
        if total <= 0:
            continue
        cdf        = np.cumsum(counts) / total
        i50        = min(np.searchsorted(cdf, 0.50), len(dr_centers) - 1)
        i90        = min(np.searchsorted(cdf, 0.90), len(dr_centers) - 1)
        medians[i] = dr_centers[i50]
        p90s[i]    = dr_centers[i90]
        valid[i]   = True
    return x_centers, x_edges, medians, p90s, valid

# ── Section 1: 1D dR distributions ───────────────────────────────────────────
# 1a: one plot per rank, overlay all three collections
for rank, rstyle in rank_styles.items():
    fig, ax = plt.subplots(figsize=size)
    hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
    for coll, cstyle in coll_styles.items():
        h = s_hists[f'dr_to_gen_{coll}_{rank}'][{"cut": cut, "samp": sum}]
        hep.histplot(h, ax=ax, histtype='step', density=True,
                     label=cstyle['label'], color=cstyle['color'], linewidth=2)
    ax.axvline(0.1, color='gray', ls=':', lw=1.5, label=r'$\Delta R = 0.1$ threshold')
    ax.set_xlabel(r'$\Delta R$(reco, gen)')
    ax.set_ylabel('A.U.')
    ax.set_yscale('log')
    ax.set_title(rf'{rstyle["label"]} electron min $\Delta R$(reco, gen) — all samples')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{plotdir}/hist_{seltag}_dr_to_gen_{rank}_allcolls.png')
    plt.close(fig)

# 1b: one plot per collection, overlay both ranks
for coll, cstyle in coll_styles.items():
    fig, ax = plt.subplots(figsize=size)
    hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
    for rank, rstyle in rank_styles.items():
        h = s_hists[f'dr_to_gen_{coll}_{rank}'][{"cut": cut, "samp": sum}]
        hep.histplot(h, ax=ax, histtype='step', density=True,
                     label=rstyle['label'], color=cstyle['color'],
                     linestyle=rstyle['ls'], linewidth=2)
    ax.axvline(0.1, color='gray', ls=':', lw=1.5, label=r'$\Delta R = 0.1$ threshold')
    ax.set_xlabel(r'$\Delta R$(reco, gen)')
    ax.set_ylabel('A.U.')
    ax.set_yscale('log')
    ax.set_title(rf'{cstyle["label"]} min $\Delta R$(reco, gen): leading vs subleading — all samples')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{plotdir}/hist_{seltag}_dr_to_gen_{coll}_bothranks.png')
    plt.close(fig)

# 1c: cumulative dR distributions — fraction of electrons within dR cone
#     useful for choosing/validating the dR < 0.1 threshold

for rank, rstyle in rank_styles.items():
    fig, ax = plt.subplots(figsize=size)
    hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
    for coll, cstyle in coll_styles.items():
        h    = s_hists[f'dr_to_gen_{coll}_{rank}'][{"cut": cut, "samp": sum}]
        vals = h.values()
        edges = h.axes[0].edges
        centers = h.axes[0].centers
        total = vals.sum()
        if total > 0:
            cdf = np.cumsum(vals) / total
            ax.step(centers, cdf, where='mid',
                    label=cstyle['label'], color=cstyle['color'], linewidth=2)
    ax.axvline(0.1, color='gray', ls=':', lw=1.5, label=r'$\Delta R = 0.1$')
    ax.axhline(0.9, color='lightgray', ls=':', lw=1)
    ax.set_xlabel(r'$\Delta R$(reco, gen)')
    ax.set_ylabel('Cumulative fraction')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(rf'{rstyle["label"]} electron cumulative $\Delta R$(reco, gen) — all samples')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{plotdir}/hist_{seltag}_dr_to_gen_{rank}_cdf_allcolls.png')
    plt.close(fig)


# ── Section 2: 2D colormesh plots ─────────────────────────────────────────────

for coll, cstyle in coll_styles.items():
    for rank, rstyle in rank_styles.items():
        for varname, varlabel, doLogx in _2d_var_specs:
            histname = f'dr_to_gen_vs_{varname}_{coll}_{rank}'
            h2d      = s_hists[histname][{"cut": cut, "samp": sum}]
            x_edges  = h2d.axes[0].edges
            dr_edges = h2d.axes[1].edges
            vals     = h2d.values()
            if not np.any(vals > 0):
                continue

            fig, ax = plt.subplots(figsize=size)
            hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
            vals_pos = np.where(vals > 0, vals, np.nan)
            pcm = ax.pcolormesh(x_edges, dr_edges, vals_pos.T, cmap='viridis', norm=LogNorm())
            plt.colorbar(pcm, ax=ax, label='Events')
            ax.axhline(0.1, color='red', ls='--', lw=1.2, label=r'$\Delta R = 0.1$')
            ax.set_xlabel(varlabel)
            ax.set_ylabel(r'$\Delta R$(reco, gen)')
            if doLogx:
                ax.set_xscale('log')
            ax.set_title(
                rf'{cstyle["label"]} ({rstyle["label"]}) $\Delta R$(reco, gen) vs {varlabel} — all samples'
            )
            ax.legend(fontsize=16)
            plt.tight_layout()
            plt.savefig(f'{plotdir}/hist_{seltag}_{histname}_2D_allsamps.png')
            plt.close(fig)


# ── Section 3: Profile plots (median + 90th pct of dR vs secondary variable) ──
# One plot per (rank, secondary variable): overlay all three collections.

for rank, rstyle in rank_styles.items():
    for varname, varlabel, doLogx in _2d_var_specs:
        fig, ax = plt.subplots(figsize=size)
        hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
        for coll, cstyle in coll_styles.items():
            histname = f'dr_to_gen_vs_{varname}_{coll}_{rank}'
            h2d      = s_hists[histname][{"cut": cut, "samp": sum}]
            x_centers, x_edges, medians, p90s, valid = median_profile_from_2D(h2d)
            if not valid.any():
                continue
            xv = x_centers[valid]
            ax.plot(xv, medians[valid], color=cstyle['color'],
                    ls='-',  lw=2,   label=rf'{cstyle["label"]} median')
            ax.plot(xv, p90s[valid],   color=cstyle['color'],
                    ls='--', lw=1.5, label=rf'{cstyle["label"]} 90th pct')
            ax.fill_between(xv, medians[valid], p90s[valid],
                            color=cstyle['color'], alpha=0.12, label='_')
        ax.axhline(0.1, color='gray', ls=':', lw=1.5, label=r'$\Delta R = 0.1$ threshold')
        ax.set_xlabel(varlabel)
        ax.set_ylabel(r'$\Delta R$(reco, gen)')
        ax.set_ylim(bottom=0)
        if doLogx:
            ax.set_xscale('log')
        ax.set_title(
            rf'{rstyle["label"]} electron $\Delta R$(reco, gen) vs {varlabel} — all samples'
        )
        ax.legend(fontsize=16, ncol=2)
        plt.tight_layout()
        plt.savefig(f'{plotdir}/hist_{seltag}_dr_to_gen_vs_{varname}_{rank}_profile_allsamps.png')
        plt.close(fig)


# ── Section 4: n_reco_near_gen and dr_to_nearest_other_reco ──────────────────
# Note: these histograms require fill logic in genmatching.py that is not yet
# implemented. The plots below will be empty until those fills are added.

# 4a: n_reco_near_gen — matching ambiguity: how many reco electrons land within
#     dR < 0.1 of the nearest gen particle?

fig, ax = plt.subplots(figsize=size)
hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
for i, (coll, cstyle) in enumerate(coll_styles.items()):
    h = s_hists[f'n_reco_near_gen_{coll}'][{"cut": cut, "samp": sum}]
    hep.histplot(h, ax=ax, histtype='step', density=True,
                 label=cstyle['label'], color=cstyle['color'], linewidth=2)
    _annotate_counts(ax, h, cstyle['color'], x_offset=(i - 0.5) * 0.3)
ax.set_xlabel(r'$N$ reco electrons within $\Delta R < 0.1$ of nearest gen')
ax.set_ylabel('A.U.')
ax.set_title(r'Reco electron matching ambiguity — all samples')
ax.legend()
plt.tight_layout()
plt.savefig(f'{plotdir}/hist_{seltag}_n_reco_near_gen_allcolls.png')
plt.close(fig)

# 4b: dr_to_nearest_other_reco — intra-collection clone distance

for rank, rstyle in rank_styles.items():
    fig, ax = plt.subplots(figsize=size)
    hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
    for coll, cstyle in coll_styles.items():
        h = s_hists[f'dr_to_nearest_other_reco_{coll}_{rank}'][{"cut": cut, "samp": sum}]
        hep.histplot(h, ax=ax, histtype='step', density=True,
                     label=cstyle['label'], color=cstyle['color'], linewidth=2)
    ax.axvline(0.1, color='gray', ls=':', lw=1.5, label=r'$\Delta R = 0.1$ threshold')
    ax.set_xlabel(r'$\Delta R$ to nearest other reco electron')
    ax.set_ylabel('A.U.')
    ax.set_yscale('log')
    ax.set_title(rf'{rstyle["label"]} electron intra-collection clone distance — all samples')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{plotdir}/hist_{seltag}_dr_to_nearest_other_reco_{rank}_allcolls.png')
    plt.close(fig)


# ── Section 5: n_gen_near_reco ────────────────────────────────────────────────
# Flipped coverage: how many gen particles (0, 1, or 2) are within dR < 0.1
# of any reco electron in the collection?

fig, ax = plt.subplots(figsize=size)
hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
for i, (coll, cstyle) in enumerate(coll_styles.items()):
    h = s_hists[f'n_gen_near_reco_{coll}'][{"cut": cut, "samp": sum}]
    hep.histplot(h, ax=ax, histtype='step', density=True,
                 label=cstyle['label'], color=cstyle['color'], linewidth=2)
    _annotate_counts(ax, h, cstyle['color'], x_offset=(i - 0.5) * 0.3)
ax.set_xlabel(r'$N$ gen particles within $\Delta R < 0.1$ of any reco electron')
ax.set_ylabel('A.U.')
ax.set_title(r'Gen particle coverage by reco collection — all samples')
ax.legend()
plt.tight_layout()
plt.savefig(f'{plotdir}/hist_{seltag}_n_gen_near_reco_allcolls.png')
plt.close(fig)


# ── Section 6: gen-to-reco dR (inverse matching direction) ───────────────────

genpart_styles = {
    'ele': {'label': r'$e^-$', 'ls': '-'},
    'pos': {'label': r'$e^+$', 'ls': '--'},
}

# Secondary variable specs for gen-to-reco 2D hists
_gen2reco_var_specs = [
    ('genpt',  r'Gen $p_T$ [GeV]',   False),
    ('geneta', r'Gen $\eta$',         False),
    ('genlxy', r'Gen $L_{xy}$ [cm]', True),
]

# 6a: one plot per collection, overlay ele and pos
for coll, cstyle in coll_styles.items():
    fig, ax = plt.subplots(figsize=size)
    hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
    for genpart, gpstyle in genpart_styles.items():
        h = s_hists[f'dr_gen_to_reco_{coll}_{genpart}'][{"cut": cut, "samp": sum}]
        hep.histplot(h, ax=ax, histtype='step', density=True,
                     label=gpstyle['label'], color=cstyle['color'],
                     linestyle=gpstyle['ls'], linewidth=2)
    ax.axvline(0.1, color='gray', ls=':', lw=1.5, label=r'$\Delta R = 0.1$ threshold')
    ax.set_xlabel(r'Min $\Delta R$(gen, reco)')
    ax.set_ylabel('A.U.')
    ax.set_yscale('log')
    ax.set_title(rf'{cstyle["label"]} min $\Delta R$(gen, reco): $e^-$ vs $e^+$ — all samples')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{plotdir}/hist_{seltag}_dr_gen_to_reco_{coll}_bothparts.png')
    plt.close(fig)

# 6b: one plot per genpart, overlay both collections
for genpart, gpstyle in genpart_styles.items():
    fig, ax = plt.subplots(figsize=size)
    hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
    for coll, cstyle in coll_styles.items():
        h = s_hists[f'dr_gen_to_reco_{coll}_{genpart}'][{"cut": cut, "samp": sum}]
        hep.histplot(h, ax=ax, histtype='step', density=True,
                     label=cstyle['label'], color=cstyle['color'], linewidth=2)
    ax.axvline(0.1, color='gray', ls=':', lw=1.5, label=r'$\Delta R = 0.1$ threshold')
    ax.set_xlabel(r'Min $\Delta R$(gen, reco)')
    ax.set_ylabel('A.U.')
    ax.set_yscale('log')
    ax.set_title(rf'{gpstyle["label"]} min $\Delta R$(gen, reco) — all samples')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{plotdir}/hist_{seltag}_dr_gen_to_reco_{genpart}_allcolls.png')
    plt.close(fig)

# 6c: 2D colormesh — dR vs secondary variable for each (coll, genpart)
for coll, cstyle in coll_styles.items():
    for genpart, gpstyle in genpart_styles.items():
        gtag = f'{coll}_{genpart}'
        for varname, varlabel, doLogx in _gen2reco_var_specs:
            histname = f'dr_gen_to_reco_vs_{varname}_{gtag}'
            h2d      = s_hists[histname][{"cut": cut, "samp": sum}]
            x_edges  = h2d.axes[0].edges
            dr_edges = h2d.axes[1].edges
            vals     = h2d.values()
            if not np.any(vals > 0):
                continue

            fig, ax = plt.subplots(figsize=size)
            hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
            vals_pos = np.where(vals > 0, vals, np.nan)
            pcm = ax.pcolormesh(x_edges, dr_edges, vals_pos.T, cmap='viridis', norm=LogNorm())
            plt.colorbar(pcm, ax=ax, label='Events')
            ax.axhline(0.1, color='red', ls='--', lw=1.2, label=r'$\Delta R = 0.1$')
            ax.set_xlabel(varlabel)
            ax.set_ylabel(r'Min $\Delta R$(gen, reco)')
            if doLogx:
                ax.set_xscale('log')
            ax.set_title(
                rf'{cstyle["label"]} {gpstyle["label"]} $\Delta R$(gen, reco) vs {varlabel} — all samples'
            )
            ax.legend(fontsize=16)
            plt.tight_layout()
            plt.savefig(f'{plotdir}/hist_{seltag}_{histname}_2D_allsamps.png')
            plt.close(fig)

# 6d: profile plots (median + 90th pct) — one plot per variable, overlay colls and genparts
for varname, varlabel, doLogx in _gen2reco_var_specs:
    fig, ax = plt.subplots(figsize=size)
    hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
    for coll, cstyle in coll_styles.items():
        for genpart, gpstyle in genpart_styles.items():
            gtag     = f'{coll}_{genpart}'
            histname = f'dr_gen_to_reco_vs_{varname}_{gtag}'
            h2d      = s_hists[histname][{"cut": cut, "samp": sum}]
            x_centers, x_edges, medians, p90s, valid = median_profile_from_2D(h2d)
            if not valid.any():
                continue
            xv    = x_centers[valid]
            label = rf'{cstyle["label"]} {gpstyle["label"]}'
            ax.plot(xv, medians[valid], color=cstyle['color'],
                    ls=gpstyle['ls'], lw=2, label=rf'{label} median')
            ax.plot(xv, p90s[valid], color=cstyle['color'],
                    ls=gpstyle['ls'], lw=1.5, alpha=0.5, label=rf'{label} 90th pct')
    ax.axhline(0.1, color='gray', ls=':', lw=1.5, label=r'$\Delta R = 0.1$ threshold')
    ax.set_xlabel(varlabel)
    ax.set_ylabel(r'Min $\Delta R$(gen, reco)')
    ax.set_ylim(bottom=0)
    if doLogx:
        ax.set_xscale('log')
    ax.set_title(rf'Gen-to-reco $\Delta R$ vs {varlabel} — all samples')
    ax.legend(fontsize=14, ncol=2)
    plt.tight_layout()
    plt.savefig(f'{plotdir}/hist_{seltag}_dr_gen_to_reco_vs_{varname}_profile_allsamps.png')
    plt.close(fig)


# ── Section 7: ee-level gen vs reco correlation (dR and pT) ──────────────────

_ee_var_specs = [
    ('dr', r'$\Delta R(e^+e^-)$', 'dr_gen_vs_dr_reco_ee'),
    ('pt', r'$p_T(e^+e^-)$ [GeV]', 'pt_gen_vs_pt_reco_ee'),
]
_ee_fill_styles = {
    'matched': {'label': 'Matched vtx'},
    'selvtx':  {'label': 'Selected vtx'},
}

# 7a: 2D colormesh — one plot per (variable, fill-type)
for varkey, varlabel, histbase in _ee_var_specs:
    for filltype, fstyle in _ee_fill_styles.items():
        histname = f'{histbase}_{filltype}'
        h2d      = s_hists[histname][{"cut": cut, "samp": sum}]
        gen_edges  = h2d.axes[0].edges
        reco_edges = h2d.axes[1].edges
        vals       = h2d.values()
        if not np.any(vals > 0):
            continue

        fig, ax = plt.subplots(figsize=size)
        hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
        vals_pos = np.where(vals > 0, vals, np.nan)
        pcm = ax.pcolormesh(gen_edges, reco_edges, vals_pos.T, cmap='viridis', norm=LogNorm())
        plt.colorbar(pcm, ax=ax, label='Events')
        # diagonal reference line
        lim = min(gen_edges[-1], reco_edges[-1])
        ax.plot([0, lim], [0, lim], 'r--', lw=1.2, label='Gen = Reco')
        ax.set_xlabel(rf'Gen {varlabel}')
        ax.set_ylabel(rf'Reco {varlabel}')
        ax.set_title(rf'Gen vs reco {varlabel} ({fstyle["label"]}) — all samples')
        ax.legend(fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{plotdir}/hist_{seltag}_{histname}_2D_allsamps.png')
        plt.close(fig)

# 7b: overlay matched vs selvtx — 1D projections (gen axis and reco axis)
for varkey, varlabel, histbase in _ee_var_specs:
    for axis_idx, axis_label in [(0, 'Gen'), (1, 'Reco')]:
        fig, ax = plt.subplots(figsize=size)
        hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
        for filltype, fstyle in _ee_fill_styles.items():
            histname = f'{histbase}_{filltype}'
            h2d      = s_hists[histname][{"cut": cut, "samp": sum}]
            h1d      = h2d.project(h2d.axes[axis_idx].name)
            hep.histplot(h1d, ax=ax, histtype='step', density=True,
                         label=fstyle['label'], linewidth=2)
        ax.set_xlabel(rf'{axis_label} {varlabel}')
        ax.set_ylabel('A.U.')
        ax.set_title(rf'{axis_label} {varlabel}: matched vs selected vtx — all samples')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'{plotdir}/hist_{seltag}_{histbase}_{axis_label.lower()}_proj_allsamps.png')
        plt.close(fig)


# ── Section 8: signal point comparisons ───────────────────────────────────────
# Same series slices as vtxvars_plots.py; one plot per (series, histogram spec).
# Covers: n_reco_near_gen, n_gen_near_reco, dr_to_gen (both ranks).

plottag = f'sig{vers}_{selection}-sel'

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

# (title, outname, variables, doLogy, solid_label, dotted_label, do_ann)
# do_ann=True adds raw event count text at the top of each bin
sig_plot_specs = [
    (
        r'$N$ reco near gen',
        'n_reco_near_gen',
        ['n_reco_near_gen_gedlpt', 'n_reco_near_gen_alpt'],
        False, 'GED+LowPt', 'AllLowPt', True,
    ),
    (
        r'$N$ gen near reco',
        'n_gen_near_reco',
        ['n_gen_near_reco_gedlpt', 'n_gen_near_reco_alpt'],
        False, 'GED+LowPt', 'AllLowPt', True,
    ),
    (
        r'Leading $\Delta R$(reco, gen)',
        'dr_to_gen_lead',
        ['dr_to_gen_gedlpt_lead', 'dr_to_gen_alpt_lead'],
        True, 'GED+LowPt', 'AllLowPt', False,
    ),
    (
        r'Subleading $\Delta R$(reco, gen)',
        'dr_to_gen_sublead',
        ['dr_to_gen_gedlpt_sublead', 'dr_to_gen_alpt_sublead'],
        True, 'GED+LowPt', 'AllLowPt', False,
    ),
]

for title, outname, variables, doLogy, solid_label, dotted_label, do_ann in sig_plot_specs:
    is_paired = len(variables) > 1
    plot_dict = {
        'variable': variables if is_paired else variables[0],
        'year': 2024,
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
                    result = ptools.plot_signal_1D(s_hists, m1, delta, ctau, plot_dict, style_dict, cmap_idx=cmap_idx)
                    if result is not None and do_ann:
                        histo, clr = result
                        if isinstance(histo, list):
                            for h, xoff in zip(histo, [-0.15, 0.15]):
                                _annotate_counts(ax, h, clr, x_offset=xoff, fontsize=9)
                        else:
                            _annotate_counts(ax, histo, clr, fontsize=9)
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
        plt.savefig(f'{plotdir}/hist_{seltag}_{outname}_{tag}.png')
        plt.close()
