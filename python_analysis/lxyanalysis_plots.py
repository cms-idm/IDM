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
selection = 'anmatchvtx'
hists_tag = 'lxyanalysis'
saved_signal_hists = f"{outdir}/hists_sig{vers}_{selection}-sel_{hists_tag}.coffea"
seltag = 'all'
cut    = 'cut1'
year   = 2024
size   = (16, 12)

s_hists = util.load(saved_signal_hists)[0]
s_pts   = utils.get_signal_point_dict(s_hists)

os.makedirs('plots', exist_ok=True)

# ── vtx-vs-mele overlay for every lxyanalysis hist, all-samples and a few
# individual signal points ───────────────────────────────────────────────────

_lxyanalysis_vars = [
    ('gen_dxy',           r'Gen Electron $d_{xy}$ [cm]',                                True),
    ('reco_dxy',          r'Reco Electron $d_{xy}$ [cm]',                               True),
    ('gen_betagamma',     r'Gen $\beta\gamma$',                                         True),
    ('reco_betagamma',    r'Reco $\beta\gamma$',                                        True),
    ('gen_dphi_e_chi2',   r'Gen $\Delta\phi(e, \chi_2)$',                               False),
    ('gen_dphi_chi2_met', r'Gen $\Delta\phi(\chi_2, p_{T}^{miss})$',                     False),
    ('reco_dphi_e_met',   r'Reco $\Delta\phi(e, p_{T}^{miss})$',                         False),
    ('gen_alpha',         r'Gen $\alpha = \beta\gamma\sin(\Delta\phi)\sin(\theta)$',     True),
    ('gen_ctau',          r'Gen $c\tau$ [cm]',                                          True),
    ('gen_lxy',           r'Gen $L_{xy}$ [cm]',                                         True),
    ('reco_lxy',          r'Reco $L_{xy}$ [cm]',                                        True),
]

def _plot_evttype_overlay(key, xlabel, samp_mask, out_suffix, title_suffix, log_x=False):
    if key not in s_hists:
        return
    if samp_mask is None:
        h = s_hists[key][{'cut': cut, 'samp': sum}]
    else:
        samps = s_pts[samp_mask].index.tolist()
        if not samps:
            print(f"No samples matched for {out_suffix}")
            return
        h = s_hists[key][{'cut': cut, 'samp': samps}][{'samp': sum}]
    h_vtx  = h[{'evttype': 'vtx'}]
    h_mele = h[{'evttype': 'mele'}]
    if np.sum(h_vtx.values()) == 0 and np.sum(h_mele.values()) == 0:
        return
    fig, ax = plt.subplots(figsize=size)
    hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
    if np.sum(h_vtx.values()) > 0:
        hep.histplot(h_vtx, ax=ax, histtype='step', density=True, yerr=False,
                     label='Vertex (vtx)', color='C0', linewidth=2)
    if np.sum(h_mele.values()) > 0:
        hep.histplot(h_mele, ax=ax, histtype='step', density=True, yerr=False,
                     label='Merged electron (mele)', color='C1', linewidth=2)
    if log_x:
        ax.set_xscale('log')
    # vtx and mele can differ by orders of magnitude in per-bin density, so a
    # linear y-axis hides whichever category has the smaller dynamic range.
    ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('A.U.')
    ax.set_title(f'{xlabel} ({title_suffix})')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'plots/hist_{seltag}_{key}_{out_suffix}.png')
    plt.close(fig)

print("lxyanalysis: vtx-vs-mele overlay (all-samples)")
for key, xlabel, log_x in _lxyanalysis_vars:
    _plot_evttype_overlay(key, xlabel, None, 'allsamps', 'all samples', log_x=log_x)

# ── Same overlay for a few individual (unsummed) signal points ──────────────

def _samp_name(m1, delta, ctau):
    row = s_pts[np.isclose(s_pts.m1, m1) & np.isclose(s_pts.delta, delta) & (s_pts.ctau == ctau)]
    return row['name'].iloc[0] if not row.empty else None

_indiv_pts = [
    (0.5, 0.1,   1),
    (0.5, 0.1,  10),
    (0.5, 0.1, 100),
    (5,   0.1,  10),
]

print("lxyanalysis: vtx-vs-mele overlay, individual signal points")
for m1, delta, ctau in _indiv_pts:
    sname = _samp_name(m1, delta, ctau)
    if sname is None:
        print(f"No sample matched for m1={m1}, delta={delta}, ctau={ctau}")
        continue
    out_suffix   = f'm1-{utils.stringfy_friendly(m1)}_d-{utils.stringfy_friendly(delta)}_ctau-{ctau}'
    title_suffix = rf'$m_1$={m1} GeV, $\Delta$={delta}, c$\tau$={ctau} mm'
    for key, xlabel, log_x in _lxyanalysis_vars:
        _plot_evttype_overlay(key, xlabel, s_pts.name == sname, out_suffix, title_suffix, log_x=log_x)

print("Done.")
