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
hists_tag = 'mergedeles'
saved_signal_hists = f"{outdir}/hists_sig{vers}_{selection}-sel_{hists_tag}.coffea"
seltag = 'minjdphi'
cut    = 'cut8'
year   = 2024
size   = (16, 12)

s_hists = util.load(saved_signal_hists)[0]

os.makedirs('plots', exist_ok=True)

# mele_gen_lxy_vs_svProxyVxy has linear-binned axes (not log), so its plot
# is excluded from the log-scale axis treatment applied to the others.
_LINEAR_2D_KEYS = {'mele_gen_lxy_vs_svProxyVxy'}

def _plot_2d_group(varlist):
    for key, xlabel, ylabel in varlist:
        if key not in s_hists:
            continue
        h = s_hists[key][{'cut': cut, 'samp': sum}]
        if np.sum(h.values()) == 0:
            continue
        fig, ax = plt.subplots(figsize=size)
        hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
        hep.hist2dplot(h, ax=ax, norm=plt.matplotlib.colors.LogNorm(), cbarextend=True)
        ax.axline((0, 0), slope=1, color='k', linestyle=':', linewidth=2)
        if key not in _LINEAR_2D_KEYS:
            ax.set_xscale('log')
            ax.set_yscale('log')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'Merged electron: {ylabel} vs {xlabel}')
        plt.tight_layout()
        plt.savefig(f'plots/hist_{seltag}_{key}_allsamps.png')
        plt.close(fig)

# ── Merged-electron gen Lxy vs reco displacement correlations (all-samples) ──

print("Merged electron: gen Lxy vs reco displacement (all-samples)")
_plot_2d_group([
    ('mele_gen_lxy_vs_dxy',        r'Gen $L_{xy}$ [cm]', r'Reco $|d_{xy}|$ [cm]'),
    ('mele_gen_lxy_vs_svProxyVxy', r'Gen $L_{xy}$ [cm]', r'Reco SV proxy $v_{xy}$ [cm]'),
    ('mele_gen_lxy_vs_lxyEst',     r'Gen $L_{xy}$ [cm]', r'Reco $|d_{xy}|\sin(\Delta\phi(e, p_{T}^{miss}))$ [cm]'),
])

# ── Merged-electron gen Lxy vs reco displacement correlations, sliced by
# signal point (one plot per ctau value summed over all m1/delta at that
# ctau, and one plot per m1 value summed over all ctau/delta at that m1) ─────

s_pts = utils.get_signal_point_dict(s_hists)

_mele_lxy_corr_vars = [
    ('mele_gen_lxy_vs_dxy',    r'Gen $L_{xy}$ [cm]', r'Reco $|d_{xy}|$ [cm]'),
    ('mele_gen_lxy_vs_lxyEst', r'Gen $L_{xy}$ [cm]', r'Reco $|d_{xy}|\sin(\Delta\phi(e, p_{T}^{miss}))$ [cm]'),
]

def _plot_2d_group_sliced(varlist, samp_mask, out_suffix, title_suffix):
    samps = s_pts[samp_mask].index.tolist()
    if not samps:
        print(f"No samples matched for {out_suffix}")
        return
    for key, xlabel, ylabel in varlist:
        if key not in s_hists:
            continue
        h = s_hists[key][{'cut': cut, 'samp': samps}][{'samp': sum}]
        if np.sum(h.values()) == 0:
            continue
        fig, ax = plt.subplots(figsize=size)
        hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
        hep.hist2dplot(h, ax=ax, norm=plt.matplotlib.colors.LogNorm(), cbarextend=True)
        ax.axline((0, 0), slope=1, color='k', linestyle=':', linewidth=2)
        if key not in _LINEAR_2D_KEYS:
            ax.set_xscale('log')
            ax.set_yscale('log')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'Merged electron: {ylabel} vs {xlabel} ({title_suffix})')
        plt.tight_layout()
        plt.savefig(f'plots/hist_{seltag}_{key}_{out_suffix}.png')
        plt.close(fig)

print("Merged electron: gen Lxy vs reco displacement, sliced by ctau")
for ctau in [1, 10, 100]:
    _plot_2d_group_sliced(
        _mele_lxy_corr_vars,
        s_pts.ctau == ctau,
        f'ctau-{utils.stringfy_friendly(ctau)}',
        rf'c$\tau$ = {ctau} mm',
    )

print("Merged electron: gen Lxy vs reco displacement, sliced by m1")
for m1 in [0.05, 0.5, 5, 50]:
    _plot_2d_group_sliced(
        _mele_lxy_corr_vars,
        np.isclose(s_pts.m1, m1),
        f'm1-{utils.stringfy_friendly(m1)}',
        rf'$m_1$ = {m1} GeV',
    )

# ── Merged-electron gen Lxy vs reco displacement correlations, for a few
# individual (unsummed) signal points ────────────────────────────────────────

def _samp_name(m1, delta, ctau):
    row = s_pts[np.isclose(s_pts.m1, m1) & np.isclose(s_pts.delta, delta) & (s_pts.ctau == ctau)]
    return row['name'].iloc[0] if not row.empty else None

_mele_lxy_corr_indiv_pts = [
    (0.5, 0.1,   1),
    (0.5, 0.1,  10),
    (0.5, 0.1, 100),
    (5,   0.1,  10),
]

print("Merged electron: gen Lxy vs reco displacement, individual signal points")
for m1, delta, ctau in _mele_lxy_corr_indiv_pts:
    sname = _samp_name(m1, delta, ctau)
    if sname is None:
        print(f"No sample matched for m1={m1}, delta={delta}, ctau={ctau}")
        continue
    _plot_2d_group_sliced(
        _mele_lxy_corr_vars,
        s_pts.name == sname,
        f'm1-{utils.stringfy_friendly(m1)}_d-{utils.stringfy_friendly(delta)}_ctau-{ctau}',
        rf'$m_1$={m1} GeV, $\Delta$={delta}, c$\tau$={ctau} mm',
    )

# ── Merged-electron reco vs truth angular correlation (all-samples) ─────────

print("Merged electron: reco vs truth dphi (all-samples)")
_plot_2d_group([
    ('mele_dphi_reco_vs_truth',        r'Reco $\Delta\phi(e, p_{T}^{miss})$', r'Truth $\Delta\phi(ee, \chi_2)$'),
    ('mele_dphi_reco_vs_truth_genmet', r'Reco $\Delta\phi(e, p_{T}^{miss})$', r'Truth $\Delta\phi(ee, p_{T}^{miss,gen})$'),
])

# ── Merged-electron reco-vs-gen dphi overlay (all-samples) ──────────────────

print("Merged electron: reco-vs-gen dphi overlay (all-samples)")
_mele_1d_dphi_vars = [
    ('mele_dphi_recoMET_vs_genMET',  r'$\Delta\phi$(reco $p_{T}^{miss}$, gen $p_{T}^{miss}$)', 'C0'),
    ('mele_dphi_recoele_vs_genee',   r'$\Delta\phi$(reco electron, gen ee)',                    'C1'),
    ('mele_dphi_genee_vs_chi2',      r'$\Delta\phi$(gen ee, gen $\chi_2$)',                     'C2'),
    ('mele_dphi_chi2_vs_genmet',     r'$\Delta\phi$(gen $\chi_2$, gen $p_{T}^{miss}$)',         'C3'),
]

fig, ax = plt.subplots(figsize=size)
hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
any_drawn = False
for key, label, color in _mele_1d_dphi_vars:
    if key not in s_hists:
        continue
    h = s_hists[key][{'cut': cut, 'samp': sum}]
    if np.sum(h.values()) == 0:
        continue
    hep.histplot(h, ax=ax, histtype='step', density=True,
                 label=label, color=color, linewidth=2)
    any_drawn = True
if any_drawn:
    ax.set_xscale('log')
    ax.set_xlabel(r'$\Delta\phi$')
    ax.set_ylabel('A.U.')
    ax.set_title(r'Merged electron: reco-vs-gen $\Delta\phi$ comparison')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'plots/hist_{seltag}_mele_dphi_reco_vs_gen_allsamps.png')
plt.close(fig)

# ── Reco-quality comparison between the two gen-Lxy populations seen in the
# gen-lxy-vs-dxy plot (prompt vs displaced, split at gen Lxy = 0.01 cm; see
# configs/histo_configs/mergedeles.py) ───────────────────────────────────────

# dxySignif/dzSignif are now log-binned in the config (see mergedeles.py's
# _dxySignif/_dzSignif axes), so their overlay plots need a log x-axis to
# render sensibly.
_LOG_X_POPCOMPARE_VARS = {'dxySignif', 'dzSignif'}

def _plot_pop_overlay(key, xlabel, out_name, log_x=False):
    if key not in s_hists:
        return
    h = s_hists[key][{'cut': cut, 'samp': sum}]
    h_prompt    = h[{'pop': 'prompt'}]
    h_displaced = h[{'pop': 'displaced'}]
    if np.sum(h_prompt.values()) == 0 and np.sum(h_displaced.values()) == 0:
        return
    fig, ax = plt.subplots(figsize=size)
    hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
    if np.sum(h_prompt.values()) > 0:
        hep.histplot(h_prompt, ax=ax, histtype='step', density=True, yerr=False,
                     label=r'Prompt (Gen $L_{xy} < 0.01$ cm)', color='C0', linewidth=2)
    if np.sum(h_displaced.values()) > 0:
        hep.histplot(h_displaced, ax=ax, histtype='step', density=True, yerr=False,
                     label=r'Displaced (Gen $L_{xy} \geq 0.01$ cm)', color='C1', linewidth=2)
    if log_x:
        ax.set_xscale('log')
    # Prompt and displaced can differ by orders of magnitude in per-bin density
    # (few entries in narrow bins vs. many entries spread over wide bins), so a
    # linear y-axis hides whichever population has the smaller dynamic range.
    ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('A.U.')
    ax.set_title(f'Merged electron: {xlabel}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'plots/hist_{seltag}_{out_name}_allsamps.png')
    plt.close(fig)

print("Merged electron: reco-quality comparison between gen-Lxy populations (all-samples)")
_mele_popcompare_vars = [
    ('pt',             r'Reco $p_{T}$ [GeV]'),
    ('eta',            r'Reco $\eta$'),
    ('trkChi2',        r'Track $\chi^2/df$'),
    ('trkProb',        r'Track $\chi^2$ Probability'),
    ('numTrackerHits', r'Number of Tracker Hits'),
    ('numPixHits',     r'Number of Pixel Hits'),
    ('numStripHits',   r'Number of Strip Hits'),
    ('trkRelIso',      r'Tracker Relative Iso'),
    ('PFRelIso',       r'PF Relative Iso'),
    ('miniRelIso',     r'Mini Relative Iso'),
    ('angRes',         r'Angular Resolution $\sqrt{\sigma_\eta^2 + \sigma_\phi^2}$'),
    ('ID',             r'Low $p_T$ Electron ID Score'),
    ('dxySignif',      r'Track $d_{xy}/\sigma_{d_{xy}}$'),
    ('HoverE',           r'$H/E$'),
    ('EoverPInv',        r'$|1/E - 1/p|$ [GeV$^{-1}$]'),
    ('dEtaSeed',         r'$|\Delta\eta_{seed}|$'),
    ('dPhiIn',           r'$|\Delta\phi_{in}|$'),
    ('sigmaIetaIeta',    r'Full5x5 $\sigma_{i\eta i\eta}$'),
    ('scEtaWidth',       r'SC $\eta$ Width'),
    ('scPhiWidth',       r'SC $\phi$ Width'),
    ('missingInnerHits', r'Expected Missing Inner Hits'),
    ('dzSignif',         r'Track $d_{z}/\sigma_{d_{z}}$'),
    ('minDRtoReg',       r'Min $\Delta R$ to Seeding Region'),
    ('svProxyValid',     r'SV Proxy Fit Valid'),
    ('conversionVeto',   r'Conversion Veto'),
    ('isPF',             r'Reconstructed as PF Electron'),
]

for var, xlabel in _mele_popcompare_vars:
    _plot_pop_overlay(
        f'mele_popcompare_{var}', xlabel,
        f'mele_popcompare_{var}',
        log_x=(var in _LOG_X_POPCOMPARE_VARS),
    )

# ── Lead/sublead resolution & separation, same prompt/displaced split ───────

# mele_dr_ee_reco/mele_dr_lead_reco/mele_dr_sublead_reco sit on
# _dr_reco_zoom (mergedeles.py), which is now log-binned, so their overlay
# plots need a log x-axis to render sensibly. mele_gen_lxy sits on _lxy_log
# (also log-binned) for the same reason.
_LOG_X_LEADSUBLEAD_VARS = {'mele_dr_ee_reco', 'mele_dr_lead_reco', 'mele_dr_sublead_reco', 'mele_gen_lxy'}

print("Merged electron: lead/sublead resolution & separation by gen-Lxy population (all-samples)")
_mele_leadsublead_vars = [
    ('mele_gen_lxy',          r'Gen $L_{xy}$ [cm]'),
    ('mele_dr_ee_reco',       r'$\Delta R$(gen $ee$, reco)'),
    ('mele_dr_lead_reco',     r'$\Delta R$(gen lead, reco)'),
    ('mele_dr_sublead_reco',  r'$\Delta R$(gen sublead, reco)'),
    ('mele_ptres_ee',         r'$(p_T^{reco} - p_T^{gen,ee})/p_T^{gen,ee}$'),
    ('mele_ptres_lead',       r'$(p_T^{reco} - p_T^{gen,lead})/p_T^{gen,lead}$'),
    ('mele_dpt_lead_sublead', r'Gen $p_T^{lead} - p_T^{sublead}$ [GeV]'),
    ('mele_dxyres_lead',      r'$(d_{xy}^{reco} - d_{xy}^{gen,lead})/d_{xy}^{gen,lead}$'),
    ('mele_dxyres_sublead',   r'$(d_{xy}^{reco} - d_{xy}^{gen,sublead})/d_{xy}^{gen,sublead}$'),
]

for key, xlabel in _mele_leadsublead_vars:
    _plot_pop_overlay(
        key, xlabel,
        key,
        log_x=(key in _LOG_X_LEADSUBLEAD_VARS),
    )

print("Done.")
