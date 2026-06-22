import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
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
vers      = 'May2026'
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
    'sep':    {'label': 'Separated', 'ls': '--'},
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


# ── Section 6: merged vs sep comparison — 1D hists ───────────────────────────
# For each comparison quantity, one plot per collection overlaying merged and sep.

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
        ax.set_title(rf'{cstyle["label"]} — {xlabel}: merged vs separated — all samples')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'plots/hist_{seltag}_{histbase}_{coll}_merged_vs_sep.png')
        plt.close(fig)


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
