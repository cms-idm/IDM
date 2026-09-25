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
hists_tag = 'mergedcats'
saved_signal_hists = f"{outdir}/hists_sig{vers}_{selection}-sel_{hists_tag}.coffea"
seltag = 'minjdphi'
cut    = 'cut8'
year   = 2024
size   = (16, 12)

s_hists = util.load(saved_signal_hists)[0]
s_pts   = utils.get_signal_point_dict(s_hists)

plotdir = 'plots/mergedcats'
os.makedirs(plotdir, exist_ok=True)

# ── Metadata ───────────────────────────────────────────────────────────────────

# Five reco categories, distinguished by color/line style when overlaid.
# 'vtx' (vertexed) supersedes the other four -- see computeMergedCatVars
# (analysisTools/analysisSubroutines.py) and mergedcats.py.
cat_styles = {
    'vtx':  {'label': 'Vertexed',        'color': 'C4', 'ls': '-'},
    'mele': {'label': 'Merged electron', 'color': 'C0', 'ls': '-'},
    'mpho': {'label': 'Merged photon',   'color': 'C1', 'ls': '--'},
    'res':  {'label': 'Resolved',        'color': 'C2', 'ls': '-.'},
    'zero': {'label': 'Zero match',      'color': 'C3', 'ls': ':'},
}

series_list = [
    {'m1s': [0.05, 0.5, 5, 50], 'deltas': [0.1],            'ctaus': [10],         'tag': 'ctau-10_delta-0p1_m1-wide'},
    {'m1s': [0.5, 1, 2, 5],     'deltas': [0.1],            'ctaus': [10],         'tag': 'ctau-10_delta-0p1_m1-narrow'},
    {'m1s': [0.5],              'deltas': [0.1],            'ctaus': [1, 10, 100], 'tag': 'm1-0p5_delta-0p1'},
    {'m1s': [0.5],              'deltas': [0.05, 0.1, 0.2], 'ctaus': [10],         'tag': 'm1-0p5_ctau-10'},
    {'m1s': [5],                'deltas': [0.1],            'ctaus': [1, 10, 100], 'tag': 'm1-5_delta-0p1'},
    {'m1s': [5],                'deltas': [0.05, 0.1, 0.2], 'ctaus': [10],         'tag': 'm1-5_ctau-10'},
]


def _samp_name(m1, delta, ctau):
    row = s_pts[np.isclose(s_pts.m1, m1) & np.isclose(s_pts.delta, delta) & (s_pts.ctau == ctau)]
    return row['name'].iloc[0] if not row.empty else None


def _sig_label(m1, delta, ctau):
    return rf'$M_1$={m1}, $\Delta$={delta}, $c\tau$={ctau} mm'


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


# ── Variable definitions ───────────────────────────────────────────────────────

# (hist_key_suffix, xlabel, doLogy, varbin)
# hist keys: {cat}_{suffix}  e.g. mele_gen_lxy, mpho_gen_lxy
_gen_vars = [
    ('gen_lxy',         r'Gen $L_{xy}$ [cm]',                True,  True),
    ('gen_ee_pt',       r'Gen $p_T(e^+e^-)$ [GeV]',          False, False),
    ('gen_lead_pt',     r'Gen leading $p_T$ [GeV]',           False, False),
    ('gen_sublead_pt',  r'Gen subleading $p_T$ [GeV]',        False, False),
    ('gen_ee_dr',       r'Gen $\Delta R(e^+e^-)$',             True,  False),
    ('gen_ee_eta',      r'Gen $\eta(e^+e^-)$',                False, False),
    ('gen_lead_eta',    r'Gen leading $\eta$',                 False, False),
    ('gen_sublead_eta', r'Gen subleading $\eta$',              False, False),
]

# (hist_key, xlabel, doLogy, varbin)
_mele_vars = [
    ('mele_pt',                  r'Reco $p_T$ [GeV]',                          False, False),
    ('mele_eta',                 r'Reco $\eta$',                                False, False),
    ('mele_phi',                 r'Reco $\phi$',                                False, False),
    ('mele_e',                   r'Reco $E$ [GeV]',                             False, False),
    ('mele_ID',                  r'Low-$p_T$ ele ID score',                     False, False),
    ('mele_IDscore',             r'Low-$p_T$ ele ID score',                     False, False),
    ('mele_angRes',              r'Angular resolution $\sqrt{\sigma_\eta^2+\sigma_\phi^2}$', False, False),
    ('mele_vxy',                 r'$|v_{xy}|$ [cm]',                            True,  False),
    ('mele_vz',                  r'$|v_z|$ [cm]',                               True,  False),
    ('mele_dxy',                 r'$|d_{xy}|$ [cm]',                            True,  False),
    ('mele_dz',                  r'$|d_z|$ [cm]',                               False, False),
    ('mele_trkChi2',             r'Track $\chi^2/\mathrm{dof}$',                True,  False),
    ('mele_trkProb',             r'Track $\chi^2$ probability',                  True,  False),
    ('mele_numTrackerHits',      r'Number of tracker hits',                      False, False),
    ('mele_numPixHits',          r'Number of pixel hits',                        False, False),
    ('mele_numStripHits',        r'Number of strip hits',                        False, False),
    ('mele_charge',              r'Charge',                                      False, False),
    ('mele_trkIso',              r'Tracker iso [GeV]',                           True,  False),
    ('mele_trkRelIso',           r'Tracker relative iso',                        True,  False),
    ('mele_calIso',              r'Calo iso [GeV]',                              True,  False),
    ('mele_calRelIso',           r'Calo relative iso',                           True,  False),
    ('mele_PFIso',               r'PF iso [GeV]',                                True,  False),
    ('mele_PFRelIso',            r'PF relative iso',                             True,  False),
    ('mele_miniIso',             r'Mini iso [GeV]',                              True,  False),
    ('mele_miniRelIso',          r'Mini relative iso',                           True,  False),
    ('mele_PFIsoEleCorr',        r'PF iso (ele-corr) [GeV]',                     True,  False),
    ('mele_PFRelIsoEleCorr',     r'PF relative iso (ele-corr)',                  True,  False),
    ('mele_miniIsoEleCorr',      r'Mini iso (ele-corr) [GeV]',                   True,  False),
    ('mele_miniRelIsoEleCorr',   r'Mini relative iso (ele-corr)',                True,  False),
    ('mele_chadIso',             r'Charged hadron iso [GeV]',                    True,  False),
    ('mele_nhadIso',             r'Neutral hadron iso [GeV]',                    True,  False),
    ('mele_phoIso',              r'Photon iso [GeV]',                            True,  False),
    ('mele_rhoEA',               r'$\rho \times EA$ [GeV]',                      False, False),
    ('mele_minDRtoReg',          r'Min $\Delta R$ to reg. electron',             True,  False),
    ('mele_drNearestEle',        r'Min $\Delta R$ to other low-$p_T$ ele',       True,  False),
    ('mele_mindRj',              r'Min $\Delta R$ to jet',                       True,  False),
    ('mele_mindPhiJ',            r'Min $\Delta\phi$ to jet',                     True,  False),
    ('mele_full55sigmaIetaIeta', r'$\sigma_{i\eta i\eta}^{5\times5}$',           False, False),
    ('mele_absdEtaSeed',         r'$|\Delta\eta_\mathrm{seed}|$',                False, False),
    ('mele_absdPhiIn',           r'$|\Delta\phi_\mathrm{in}|$',                  False, False),
    ('mele_HoverE',              r'H/E',                                         True,  False),
    ('mele_abs1overEm1overP',    r'$|1/E - 1/p|$ [GeV$^{-1}$]',                 True,  False),
    ('mele_expMissingInnerHits', r'Exp. missing inner hits',                     False, False),
]

_mpho_vars = [
    ('mpho_pt',            r'Reco $p_T$ [GeV]',                         False, False),
    ('mpho_eta',           r'Reco $\eta$',                               False, False),
    ('mpho_phi',           r'Reco $\phi$',                               False, False),
    ('mpho_energy',        r'Reco $E$ [GeV]',                            False, False),
    ('mpho_r9',            r'R9',                                        False, False),
    ('mpho_full5x5_r9',    r'Full 5$\times$5 R9',                        False, False),
    ('mpho_sIeIe',         r'$\sigma_{i\eta i\eta}$',                    False, False),
    ('mpho_full5x5_sIeIe', r'Full 5$\times$5 $\sigma_{i\eta i\eta}$',    False, False),
    ('mpho_HoE',           r'H/E',                                       True,  False),
    ('mpho_full5x5_HoE',   r'Full 5$\times$5 H/E',                      True,  False),
    ('mpho_chIso',         r'Charged hadron iso [GeV]',                  True,  False),
    ('mpho_nhIso',         r'Neutral hadron iso [GeV]',                  True,  False),
    ('mpho_phIso',         r'Photon iso [GeV]',                          True,  False),
    ('mpho_puChIso',       r'PU-subtracted charged hadron iso [GeV]',   True,  False),
    ('mpho_trkIso',        r'Tracker iso [GeV]',                         True,  False),
    ('mpho_ecalIso',       r'ECAL iso [GeV]',                            True,  False),
    ('mpho_hcalIso',       r'HCAL iso [GeV]',                            True,  False),
    ('mpho_mindRj',        r'Min $\Delta R$ to jet',                     True,  False),
    ('mpho_mindPhiJ',      r'Min $\Delta\phi$ to jet',                   True,  False),
]

# Resolved category reco variables mirror the merged-electron set (same
# AllLptElectron fields), just with the 'res_' prefix.
_res_vars = [(f'res_{key[len("mele_"):]}', xlabel, doLogy, varbin) for key, xlabel, doLogy, varbin in _mele_vars]


# ── Section 1: Gen ee kinematics — overlay all categories ─────────────────────
# All-samples aggregate: one line per category, one plot per variable.

print("Section 1: gen ee kinematics (all-samples)")
for var, xlabel, doLogy, varbin in _gen_vars:
    fig, ax = plt.subplots(figsize=size)
    hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
    any_drawn = False
    for cat, cstyle in cat_styles.items():
        hkey = f'{cat}_{var}'
        if hkey not in s_hists:
            continue
        h = s_hists[hkey][{'cut': cut, 'samp': sum}]
        if np.sum(h.values()) == 0:
            continue
        _histplot(h, ax, varbin=varbin,
                  histtype='step', label=cstyle['label'],
                  color=cstyle['color'], linestyle=cstyle['ls'], linewidth=2)
        any_drawn = True
    if not any_drawn:
        plt.close(fig)
        continue
    ax.set_xlabel(xlabel)
    ax.set_ylabel('A.U.')
    if doLogy:
        ax.set_yscale('log')
    ax.set_title(rf'Merged categories: {xlabel} — all samples')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{plotdir}/hist_{seltag}_{var}_cats_allsamps.png')
    plt.close(fig)

# Per-series: one plot per variable per series per category — one category at
# a time (each signal point is a color) to keep multi-sample plots readable.
# --- TEMPORARILY DISABLED: only "_cats_" overlay plots are being made right now ---
"""
print("Section 1: gen ee kinematics (per-series)")
for var, xlabel, doLogy, varbin in _gen_vars:
    for cat, cstyle in cat_styles.items():
        hkey = f'{cat}_{var}'
        if hkey not in s_hists:
            continue
        for series in series_list:
            m1s, deltas, ctaus, tag = series['m1s'], series['deltas'], series['ctaus'], series['tag']
            fig, ax = plt.subplots(figsize=size)
            hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
            cmap_idx = 0
            any_drawn = False
            for m1 in m1s:
                for delta in deltas:
                    for ctau in ctaus:
                        sname = _samp_name(m1, delta, ctau)
                        if sname is None:
                            cmap_idx += 1
                            continue
                        h = s_hists[hkey][{'cut': cut, 'samp': sname}]
                        if np.sum(h.values()) == 0:
                            cmap_idx += 1
                            continue
                        _histplot(h, ax, varbin=varbin,
                                  histtype='step', label=_sig_label(m1, delta, ctau),
                                  color=f'C{cmap_idx}', linewidth=2)
                        any_drawn = True
                        cmap_idx += 1
            if not any_drawn:
                plt.close(fig)
                continue
            ax.set_xlabel(xlabel)
            ax.set_ylabel('A.U.')
            if doLogy:
                ax.set_yscale('log')
            ax.legend(fontsize=13)
            plt.title(rf'{cstyle["label"]}: {xlabel} — {tag}')
            plt.tight_layout()
            plt.savefig(f'{plotdir}/hist_{seltag}_{hkey}_{tag}.png')
            plt.close(fig)
"""


# ── Section 2: Merged-electron reco variables ─────────────────────────────────
# --- TEMPORARILY DISABLED: only "_cats_" overlay plots are being made right now ---
"""
print("Section 2: merged-electron reco variables (all-samples)")
for key, xlabel, doLogy, varbin in _mele_vars:
    if key not in s_hists:
        continue
    h = s_hists[key][{'cut': cut, 'samp': sum}]
    if np.sum(h.values()) == 0:
        continue
    fig, ax = plt.subplots(figsize=size)
    hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
    _histplot(h, ax, varbin=varbin,
              histtype='step', color='C0', linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('A.U.')
    if doLogy:
        ax.set_yscale('log')
    ax.set_title(rf'Merged electron: {xlabel} — all samples')
    plt.tight_layout()
    plt.savefig(f'{plotdir}/hist_{seltag}_{key}_allsamps.png')
    plt.close(fig)

print("Section 2: merged-electron reco variables (per-series)")
for key, xlabel, doLogy, varbin in _mele_vars:
    if key not in s_hists:
        continue
    for series in series_list:
        m1s, deltas, ctaus, tag = series['m1s'], series['deltas'], series['ctaus'], series['tag']
        fig, ax = plt.subplots(figsize=size)
        hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
        cmap_idx = 0
        any_drawn = False
        for m1 in m1s:
            for delta in deltas:
                for ctau in ctaus:
                    sname = _samp_name(m1, delta, ctau)
                    if sname is None:
                        cmap_idx += 1
                        continue
                    h = s_hists[key][{'cut': cut, 'samp': sname}]
                    if np.sum(h.values()) == 0:
                        cmap_idx += 1
                        continue
                    _histplot(h, ax, varbin=varbin,
                              histtype='step', label=_sig_label(m1, delta, ctau),
                              color=f'C{cmap_idx}', linewidth=2)
                    any_drawn = True
                    cmap_idx += 1
        if not any_drawn:
            plt.close(fig)
            continue
        ax.set_xlabel(xlabel)
        ax.set_ylabel('A.U.')
        if doLogy:
            ax.set_yscale('log')
        ax.legend(fontsize=13)
        plt.title(rf'Merged electron: {xlabel} — {tag}')
        plt.tight_layout()
        plt.savefig(f'{plotdir}/hist_{seltag}_{key}_{tag}.png')
        plt.close(fig)
"""


# ── Section 2b: Merged-electron — other-gen dR to other reco object types ────
# (mele_othergen_dr_*: with the reco electron tightly matched to one gen
# electron, how close is the *other*, unmatched gen electron to the nearest
# reco object of each non-electron type? Events with no such object in the
# collection were filled at dR=999, landing in the axis overflow bin.)

_mele_othergen_dr_vars = [
    ('mele_othergen_dr_lptele',     r'$\Delta R$(other gen $e$, nearest low-$p_T$ ele)', True, False),
    ('mele_othergen_dr_photon',     r'$\Delta R$(other gen $e$, nearest photon)',         True, False),
    ('mele_othergen_dr_conversion', r'$\Delta R$(other gen $e$, nearest conversion)',     True, False),
    ('mele_othergen_dr_ootphoton',  r'$\Delta R$(other gen $e$, nearest OOT photon)',     True, False),
    ('mele_othergen_dr_pfcand',     r'$\Delta R$(other gen $e$, nearest PF candidate)',   True, False),
    ('mele_othergen_dr_losttrack',  r'$\Delta R$(other gen $e$, nearest lost track)',     True, False),
]

# --- TEMPORARILY DISABLED: only "_cats_" overlay plots are being made right now ---
"""
print("Section 2b: merged-electron other-gen dR to other objects (all-samples)")
for key, xlabel, doLogy, varbin in _mele_othergen_dr_vars:
    if key not in s_hists:
        continue
    h = s_hists[key][{'cut': cut, 'samp': sum}]
    if np.sum(h.values()) == 0:
        continue
    fig, ax = plt.subplots(figsize=size)
    hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
    _histplot(h, ax, varbin=varbin,
              histtype='step', color='C0', linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('A.U.')
    if doLogy:
        ax.set_yscale('log')
    ax.set_title(rf'Merged electron: {xlabel} — all samples')
    plt.tight_layout()
    plt.savefig(f'{plotdir}/hist_{seltag}_{key}_allsamps.png')
    plt.close(fig)

print("Section 2b: merged-electron other-gen dR to other objects (per-series)")
for key, xlabel, doLogy, varbin in _mele_othergen_dr_vars:
    if key not in s_hists:
        continue
    for series in series_list:
        m1s, deltas, ctaus, tag = series['m1s'], series['deltas'], series['ctaus'], series['tag']
        fig, ax = plt.subplots(figsize=size)
        hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
        cmap_idx = 0
        any_drawn = False
        for m1 in m1s:
            for delta in deltas:
                for ctau in ctaus:
                    sname = _samp_name(m1, delta, ctau)
                    if sname is None:
                        cmap_idx += 1
                        continue
                    h = s_hists[key][{'cut': cut, 'samp': sname}]
                    if np.sum(h.values()) == 0:
                        cmap_idx += 1
                        continue
                    _histplot(h, ax, varbin=varbin,
                              histtype='step', label=_sig_label(m1, delta, ctau),
                              color=f'C{cmap_idx}', linewidth=2)
                    any_drawn = True
                    cmap_idx += 1
        if not any_drawn:
            plt.close(fig)
            continue
        ax.set_xlabel(xlabel)
        ax.set_ylabel('A.U.')
        if doLogy:
            ax.set_yscale('log')
        ax.legend(fontsize=13)
        plt.title(rf'Merged electron: {xlabel} — {tag}')
        plt.tight_layout()
        plt.savefig(f'{plotdir}/hist_{seltag}_{key}_{tag}.png')
        plt.close(fig)

# Overlay: all six object types on one axis (all-samples only), for direct
# comparison of which object type the other gen electron tends to sit nearest.
print("Section 2b: merged-electron other-gen dR overlay across object types (all-samples)")
_othergen_dr_styles = [
    ('mele_othergen_dr_lptele',     'Low-$p_T$ electron', 'C0'),
    ('mele_othergen_dr_photon',     'Photon',             'C1'),
    ('mele_othergen_dr_conversion', 'Conversion',         'C2'),
    ('mele_othergen_dr_ootphoton',  'OOT photon',         'C3'),
    ('mele_othergen_dr_pfcand',     'PF candidate',       'C4'),
    ('mele_othergen_dr_losttrack',  'Lost track',         'C5'),
]

fig, ax = plt.subplots(figsize=size)
hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
any_drawn = False
for key, label, color in _othergen_dr_styles:
    if key not in s_hists:
        continue
    h = s_hists[key][{'cut': cut, 'samp': sum}]
    if np.sum(h.values()) == 0:
        continue
    hep.histplot(h, ax=ax, histtype='step', density=True,
                 label=label, color=color, linewidth=2)
    any_drawn = True
if any_drawn:
    ax.set_xlabel(r'$\Delta R$(other gen $e$, nearest object)')
    ax.set_ylabel('A.U.')
    ax.set_yscale('log')
    ax.set_title(r'Merged electron: other-gen $\Delta R$ by object type — all samples')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{plotdir}/hist_{seltag}_mele_othergen_dr_overlay_allsamps.png')
plt.close(fig)
"""


# ── Section 3: Merged-photon reco variables ───────────────────────────────────
# --- TEMPORARILY DISABLED: only "_cats_" overlay plots are being made right now ---
"""
print("Section 3: merged-photon reco variables (all-samples)")
for key, xlabel, doLogy, varbin in _mpho_vars:
    if key not in s_hists:
        continue
    h = s_hists[key][{'cut': cut, 'samp': sum}]
    if np.sum(h.values()) == 0:
        continue
    fig, ax = plt.subplots(figsize=size)
    hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
    _histplot(h, ax, varbin=varbin,
              histtype='step', color='C1', linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('A.U.')
    if doLogy:
        ax.set_yscale('log')
    ax.set_title(rf'Merged photon: {xlabel} — all samples')
    plt.tight_layout()
    plt.savefig(f'{plotdir}/hist_{seltag}_{key}_allsamps.png')
    plt.close(fig)

print("Section 3: merged-photon reco variables (per-series)")
for key, xlabel, doLogy, varbin in _mpho_vars:
    if key not in s_hists:
        continue
    for series in series_list:
        m1s, deltas, ctaus, tag = series['m1s'], series['deltas'], series['ctaus'], series['tag']
        fig, ax = plt.subplots(figsize=size)
        hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
        cmap_idx = 0
        any_drawn = False
        for m1 in m1s:
            for delta in deltas:
                for ctau in ctaus:
                    sname = _samp_name(m1, delta, ctau)
                    if sname is None:
                        cmap_idx += 1
                        continue
                    h = s_hists[key][{'cut': cut, 'samp': sname}]
                    if np.sum(h.values()) == 0:
                        cmap_idx += 1
                        continue
                    _histplot(h, ax, varbin=varbin,
                              histtype='step', label=_sig_label(m1, delta, ctau),
                              color=f'C{cmap_idx}', linewidth=2)
                    any_drawn = True
                    cmap_idx += 1
        if not any_drawn:
            plt.close(fig)
            continue
        ax.set_xlabel(xlabel)
        ax.set_ylabel('A.U.')
        if doLogy:
            ax.set_yscale('log')
        ax.legend(fontsize=13)
        plt.title(rf'Merged photon: {xlabel} — {tag}')
        plt.tight_layout()
        plt.savefig(f'{plotdir}/hist_{seltag}_{key}_{tag}.png')
        plt.close(fig)
"""


# ── Section 4: Resolved category reco variables ───────────────────────────────
# --- TEMPORARILY DISABLED: only "_cats_" overlay plots are being made right now ---
"""
print("Section 4: resolved reco variables (all-samples)")
for key, xlabel, doLogy, varbin in _res_vars:
    if key not in s_hists:
        continue
    h = s_hists[key][{'cut': cut, 'samp': sum}]
    if np.sum(h.values()) == 0:
        continue
    fig, ax = plt.subplots(figsize=size)
    hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
    _histplot(h, ax, varbin=varbin,
              histtype='step', color='C2', linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('A.U.')
    if doLogy:
        ax.set_yscale('log')
    ax.set_title(rf'Resolved: {xlabel} — all samples')
    plt.tight_layout()
    plt.savefig(f'{plotdir}/hist_{seltag}_{key}_allsamps.png')
    plt.close(fig)

print("Section 4: resolved reco variables (per-series)")
for key, xlabel, doLogy, varbin in _res_vars:
    if key not in s_hists:
        continue
    for series in series_list:
        m1s, deltas, ctaus, tag = series['m1s'], series['deltas'], series['ctaus'], series['tag']
        fig, ax = plt.subplots(figsize=size)
        hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
        cmap_idx = 0
        any_drawn = False
        for m1 in m1s:
            for delta in deltas:
                for ctau in ctaus:
                    sname = _samp_name(m1, delta, ctau)
                    if sname is None:
                        cmap_idx += 1
                        continue
                    h = s_hists[key][{'cut': cut, 'samp': sname}]
                    if np.sum(h.values()) == 0:
                        cmap_idx += 1
                        continue
                    _histplot(h, ax, varbin=varbin,
                              histtype='step', label=_sig_label(m1, delta, ctau),
                              color=f'C{cmap_idx}', linewidth=2)
                    any_drawn = True
                    cmap_idx += 1
        if not any_drawn:
            plt.close(fig)
            continue
        ax.set_xlabel(xlabel)
        ax.set_ylabel('A.U.')
        if doLogy:
            ax.set_yscale('log')
        ax.legend(fontsize=13)
        plt.title(rf'Resolved: {xlabel} — {tag}')
        plt.tight_layout()
        plt.savefig(f'{plotdir}/hist_{seltag}_{key}_{tag}.png')
        plt.close(fig)
"""


# ── Cross-category field-name mapping (used by Sections 5 and 6) ─────────────
# mele/res (AllLptElectron) and mpho (Photon) use different field names for a
# few physically-equivalent quantities, so they can't be matched by the plain
# f'{cat}_{var}' pattern. Defined once here: Section 6 uses it to build the
# proper 3-way overlay, and Section 5 uses it to skip these hist keys entirely
# (rather than making an incomplete mele/res-only or mpho-only duplicate).
_xcat_vars = [
    ('E',             {'mele': 'mele_e',                      'mpho': 'mpho_energy', 'res': 'res_e'},
     r'Reco $E$ [GeV]',         False),
    ('trkIso',        {'mele': 'mele_trkIso',                 'mpho': 'mpho_trkIso', 'res': 'res_trkIso'},
     r'Tracker iso [GeV]',      True),
    ('chadIso',       {'mele': 'mele_chadIso',                'mpho': 'mpho_chIso',  'res': 'res_chadIso'},
     r'Charged hadron iso [GeV]', True),
    ('nhadIso',       {'mele': 'mele_nhadIso',                'mpho': 'mpho_nhIso',  'res': 'res_nhadIso'},
     r'Neutral hadron iso [GeV]', True),
    ('phoIso',        {'mele': 'mele_phoIso',                 'mpho': 'mpho_phIso',  'res': 'res_phoIso'},
     r'Photon iso [GeV]',       True),
    ('HoE',           {'mele': 'mele_HoverE',                 'mpho': 'mpho_HoE',    'res': 'res_HoverE'},
     r'H/E',                    True),
    ('sigmaIetaIeta', {'mele': 'mele_full55sigmaIetaIeta',    'mpho': 'mpho_full5x5_sIeIe', 'res': 'res_full55sigmaIetaIeta'},
     r'$\sigma_{i\eta i\eta}^{5\times5}$', False),
]
# vtx (vertexed category) uses the same AllLptElectron-style field names as
# mele/res, but under the vtx_ele_ prefix (both vertex legs filled into the
# same histos -- see mergedcats.py's _VTXELE_HISTS/_fill_vtxele) rather than
# the plain vtx_ prefix, which is reserved for genuine vertex-level quantities
# (vtx_mass, vtx_chi2, ...) that have no per-category counterpart to overlay.
for _tag, _key_map, _, _ in _xcat_vars:
    _key_map.setdefault('vtx', f'vtx_ele_{_key_map["mele"][len("mele_"):]}')
_xcat_handled_keys = {hkey for _, key_map, _, _ in _xcat_vars for hkey in key_map.values()}


# ── Section 5: Overlay reco quantities across categories (all-samples only) ──
# One "_cats_" plot per reco variable, overlaying whichever of merged-electron,
# merged-photon, resolved, and zero-match categories have a matching histogram
# (mele_*/mpho_*/res_*/zero_* sharing the same suffix, e.g. mele_pt/mpho_pt/
# res_pt). Covers every variable in _mele_vars, _mpho_vars, and
# _mele_othergen_dr_vars (res_vars mirror mele_vars' suffixes automatically),
# except the hist keys already covered by the explicit cross-mapping above,
# which get their proper 3-way overlay in Section 6 instead — including them
# here too would just produce a redundant, incomplete (mele/res-only or
# mpho-only) duplicate under a different filename.

print("Section 5: reco/gen overlays across categories (all-samples)")

# vtx (vertexed) stores its per-electron analogs of these AllLptElectron
# variables under vtx_ele_ (both vertex legs, see mergedcats.py), not the
# plain vtx_ prefix -- which is reserved for genuine vertex-level quantities
# (vtx_pt/eta/phi/... are the *vertex's* pt/eta/phi, not an electron's, so
# they must not be silently substituted in here).
_cat_hist_prefix = {'vtx': 'vtx_ele'}

_cats_overlay_vars = {}
for key, xlabel, doLogy, varbin in _mele_vars + _mpho_vars + _mele_othergen_dr_vars:
    if key in _xcat_handled_keys:
        continue
    suffix = key.split('_', 1)[1]
    _cats_overlay_vars.setdefault(suffix, (xlabel, doLogy))

for var, (xlabel, doLogy) in _cats_overlay_vars.items():
    fig, ax = plt.subplots(figsize=size)
    hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
    any_drawn = False
    for cat, cstyle in cat_styles.items():
        hkey = f'{_cat_hist_prefix.get(cat, cat)}_{var}'
        if hkey not in s_hists:
            continue
        h = s_hists[hkey][{'cut': cut, 'samp': sum}]
        if np.sum(h.values()) == 0:
            continue
        hep.histplot(h, ax=ax, histtype='step', density=True,
                     label=cstyle['label'], color=cstyle['color'],
                     linestyle=cstyle['ls'], linewidth=2)
        any_drawn = True
    if not any_drawn:
        plt.close(fig)
        continue
    ax.set_xlabel(xlabel)
    ax.set_ylabel('A.U.')
    if doLogy:
        ax.set_yscale('log')
    ax.set_title(rf'Merged categories: reco {xlabel} — all samples')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{plotdir}/hist_{seltag}_reco_{var}_cats_allsamps.png')
    plt.close(fig)


# ── Section 6: Isolation / shower-shape overlays across all three reco categories
# (all-samples only) ───────────────────────────────────────────────────────────
# mele/res (AllLptElectron) and mpho (Photon) use different field names for the
# same physical quantity, so each entry (defined above, shared with Section 5)
# maps category -> its own hist key instead of the single f'{cat}_{var}'
# pattern used in Section 5.

print("Section 6: isolation/shower-shape overlays across categories (all-samples)")
for tag, key_map, xlabel, doLogy in _xcat_vars:
    fig, ax = plt.subplots(figsize=size)
    hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
    any_drawn = False
    for cat, cstyle in cat_styles.items():
        hkey = key_map.get(cat)
        if hkey is None or hkey not in s_hists:
            continue
        h = s_hists[hkey][{'cut': cut, 'samp': sum}]
        if np.sum(h.values()) == 0:
            continue
        hep.histplot(h, ax=ax, histtype='step', density=True,
                     label=cstyle['label'], color=cstyle['color'],
                     linestyle=cstyle['ls'], linewidth=2)
        any_drawn = True
    if not any_drawn:
        plt.close(fig)
        continue
    ax.set_xlabel(xlabel)
    ax.set_ylabel('A.U.')
    if doLogy:
        ax.set_yscale('log')
    ax.set_title(rf'Merged categories: reco {xlabel} — all samples')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{plotdir}/hist_{seltag}_reco_{tag}_cats_allsamps.png')
    plt.close(fig)


# ── Section 7: Merged-photon 2D correlations (all-samples only) ──────────────
# --- TEMPORARILY DISABLED (through Section 9): only "_cats_" overlay plots are
# being made right now ---
"""
print("Section 7: merged-photon gen Lxy vs shower-shape/ID (all-samples)")
_mpho_2d_vars = [
    ('mpho_gen_lxy_vs_HoE',   r'Gen $L_{xy}$ [cm]', 'H/E'),
    ('mpho_gen_lxy_vs_sieie', r'Gen $L_{xy}$ [cm]', r'$\sigma_{i\eta i\eta}^{5\times5}$'),
]

for key, xlabel, ylabel in _mpho_2d_vars:
    if key not in s_hists:
        continue
    h = s_hists[key][{'cut': cut, 'samp': sum}]
    if np.sum(h.values()) == 0:
        continue
    fig, ax = plt.subplots(figsize=size)
    hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
    hep.hist2dplot(h, ax=ax, norm=plt.matplotlib.colors.LogNorm(), cbarextend=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title('Merged photon: all samples')
    plt.tight_layout()
    plt.savefig(f'{plotdir}/hist_{seltag}_{key}_allsamps.png')
    plt.close(fig)


# ── Section 8: Merged-photon HoE/sieie sliced above/below gen Lxy cut ────────
# (all-samples only) — 1D projections of the Section 7 2D correlations, split
# at the same gen-Lxy cut used in histmaker (mergedcats.py: _LXY_CUT_CM = 100 cm).

print("Section 8: merged-photon HoE/sieie sliced by gen Lxy (all-samples)")
_lxy_slice_vars = [
    ('mpho_HoE',           r'H/E',                                    True),
    ('mpho_full5x5_sIeIe', r'Full 5$\times$5 $\sigma_{i\eta i\eta}$', False),
]
_lxy_slice_styles = [
    ('_lxyLt1000mm', r'Gen $L_{xy}$ < 100 cm', 'C1', '-'),
    ('_lxyGt1000mm', r'Gen $L_{xy}$ > 100 cm', 'C4', '--'),
]

for base_key, xlabel, doLogy in _lxy_slice_vars:
    fig, ax = plt.subplots(figsize=size)
    hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
    any_drawn = False
    for suffix, label, color, ls in _lxy_slice_styles:
        hkey = f'{base_key}{suffix}'
        if hkey not in s_hists:
            continue
        h = s_hists[hkey][{'cut': cut, 'samp': sum}]
        if np.sum(h.values()) == 0:
            continue
        hep.histplot(h, ax=ax, histtype='step', density=True,
                     label=label, color=color, linestyle=ls, linewidth=2)
        any_drawn = True
    if not any_drawn:
        plt.close(fig)
        continue
    ax.set_xlabel(xlabel)
    ax.set_ylabel('A.U.')
    if doLogy:
        ax.set_yscale('log')
    ax.set_title(rf'Merged photon: {xlabel} by gen $L_{{xy}}$ — all samples')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{plotdir}/hist_{seltag}_{base_key}_lxyslices_allsamps.png')
    plt.close(fig)


# ── Section 9: Merged electron/photon vs zero-match & non-matched-object
# comparisons (all-samples only) ──────────────────────────────────────────────
# For each matched mele_*/mpho_* reco variable, overlay:
#   - the matched (merged) object                              [mele_*          / mpho_*]
#   - the leading-pt lpt electron in zero-match events           [mele_zerolead_*]
#   - the other (non-leading) lpt electrons in the same events   [mele_zeroothers_*]
#   - same-type objects present in zero-match events                            [mpho_zero_*]
#   - other, non-matched same-type objects present alongside
#     the merged one, in merged-category events              [mele_other_*/ mpho_other_*]
# Per-series breakdowns are intentionally not repeated here, same rationale as
# the cross-category overlays in Section 5.

_mele_comparison_styles = [
    ('',            'Merged Events - Matched',    'C0', '-'),
    ('zerolead_',   'Unmatched Events - Leading', 'C3', ':'),
    ('zeroothers_', 'Unmatched Events - Other',   'C5', '-.'),
    ('other_',      'Merged Events - Other',      'C4', '--'),
]

_mpho_comparison_styles = [
    ('',       'Merged Events - Matched',  'C0', '-'),
    ('zero_',  'Unmatched Events - All',   'C3', ':'),
    ('other_', 'Merged Events - Other',    'C4', '--'),
]

print("Section 9: merged-electron vs zero-match/other-object comparisons (all-samples)")
for key, xlabel, doLogy, varbin in _mele_vars:
    base = key[len('mele_'):]
    fig, ax = plt.subplots(figsize=size)
    hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
    any_drawn = False
    for infix, label, color, ls in _mele_comparison_styles:
        hkey = f'mele_{infix}{base}'
        if hkey not in s_hists:
            continue
        h = s_hists[hkey][{'cut': cut, 'samp': sum}]
        if np.sum(h.values()) == 0:
            continue
        _histplot(h, ax, varbin=varbin,
                  histtype='step', label=label, color=color, linestyle=ls, linewidth=2)
        any_drawn = True
    if not any_drawn:
        plt.close(fig)
        continue
    ax.set_xlabel(xlabel)
    ax.set_ylabel('A.U.')
    if doLogy:
        ax.set_yscale('log')
    ax.set_title(rf'Merged electron: {xlabel} — matched vs zero-match (leading/other) vs other — all samples')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{plotdir}/hist_{seltag}_mele_{base}_matched_vs_zero_vs_other_allsamps.png')
    plt.close(fig)

print("Section 9: merged-photon vs zero-match/other-object comparisons (all-samples)")
for key, xlabel, doLogy, varbin in _mpho_vars:
    base = key[len('mpho_'):]
    fig, ax = plt.subplots(figsize=size)
    hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
    any_drawn = False
    for infix, label, color, ls in _mpho_comparison_styles:
        hkey = f'mpho_{infix}{base}'
        if hkey not in s_hists:
            continue
        h = s_hists[hkey][{'cut': cut, 'samp': sum}]
        if np.sum(h.values()) == 0:
            continue
        _histplot(h, ax, varbin=varbin,
                  histtype='step', label=label, color=color, linestyle=ls, linewidth=2)
        any_drawn = True
    if not any_drawn:
        plt.close(fig)
        continue
    ax.set_xlabel(xlabel)
    ax.set_ylabel('A.U.')
    if doLogy:
        ax.set_yscale('log')
    ax.set_title(rf'Merged photon: {xlabel} — matched vs zero-match vs other — all samples')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{plotdir}/hist_{seltag}_mpho_{base}_matched_vs_zero_vs_other_allsamps.png')
    plt.close(fig)
"""

# ── Section 10: Nearest-electron realism check (all-samples only) ────────────
# For each AllLptElectron reco variable, compare the matched merged electron,
# merged photon, and resolved electron themselves against the single nearest
# *other* AllLptElectron next to each (mele_nearestEle_*/mpho_nearestEle_*/
# res_nearestEle_*, restricted to dR<0.2 -- the near-duplicate population
# behind the mele_drNearestEle/mpho_drNearestEle/res_drNearestEle spike at
# dR~0) -- do these nearby "duplicate" candidates look like real,
# well-reconstructed electrons, or low-quality/duplicate GSF tracks?

_nearestEle_comparison_styles = [
    ('mele',            'Merged electron (matched)',     'C0', '-'),
    ('mele_nearestEle', 'Nearest other ele (near mele)', 'C3', '--'),
    ('mpho_nearestEle', 'Nearest ele (near mpho)',       'C1', '-.'),
    ('res',             'Resolved electron (matched)',   'C2', '-'),
    #('res_nearestEle',  'Nearest other ele (near res)',  'C4', ':'),
]

print("Section 10: nearest-electron realism check (all-samples)")
for key, xlabel, doLogy, varbin in _mele_vars:
    base = key[len('mele_'):]
    fig, ax = plt.subplots(figsize=size)
    hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
    any_drawn = False
    for prefix, label, color, ls in _nearestEle_comparison_styles:
        hkey = f'{prefix}_{base}'
        if hkey not in s_hists:
            continue
        h = s_hists[hkey][{'cut': cut, 'samp': sum}]
        if np.sum(h.values()) == 0:
            continue
        _histplot(h, ax, varbin=varbin,
                  histtype='step', label=label, color=color, linestyle=ls, linewidth=2)
        any_drawn = True
    if not any_drawn:
        plt.close(fig)
        continue
    ax.set_xlabel(xlabel)
    ax.set_ylabel('A.U.')
    if doLogy:
        ax.set_yscale('log')
    ax.set_title(rf'Nearest-electron realism check: {xlabel} — all samples')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{plotdir}/hist_{seltag}_{base}_nearestEle_realism_allsamps.png')
    plt.close(fig)


# ── Section 10b: Merged-electron / nearest-electron relational variables
# (all-samples only) ───────────────────────────────────────────────────────────
# Direct comparisons between the selected merged electron and its nearest
# other AllLptElectron (mele_nearestEle_dPt/dPtRel/chargeProd/dVxy/dVz/dDxy/
# dDz/drZoom) -- near-zero dPt/dVxy/dVz/dDxy/dDz with matching charge would
# point to a duplicate GSF track rather than a genuine second nearby electron.

_mele_nearestEle_relational_vars = [
    ('mele_nearestEle_dPt',        r'$p_T(e_{\mathrm{sel}}) - p_T(e_{\mathrm{near}})$ [GeV]',                    False, False),
    ('mele_nearestEle_dPtRel',     r'$(p_T(e_{\mathrm{sel}}) - p_T(e_{\mathrm{near}}))/p_T(e_{\mathrm{sel}})$',   False, False),
    ('mele_nearestEle_chargeProd', r'Charge product ($q_{\mathrm{sel}} \times q_{\mathrm{near}}$)',               False, False),
    ('mele_nearestEle_dVxy',       r'$|v_{xy}(e_{\mathrm{sel}}) - v_{xy}(e_{\mathrm{near}})|$ [cm]',              True,  False),
    ('mele_nearestEle_dVz',        r'$|v_z(e_{\mathrm{sel}}) - v_z(e_{\mathrm{near}})|$ [cm]',                    True,  False),
    ('mele_nearestEle_dDxy',       r'$|d_{xy}(e_{\mathrm{sel}}) - d_{xy}(e_{\mathrm{near}})|$ [cm]',              True,  False),
    ('mele_nearestEle_dDz',        r'$|d_z(e_{\mathrm{sel}}) - d_z(e_{\mathrm{near}})|$ [cm]',                    True,  False),
    ('mele_nearestEle_drZoom',     r'$\Delta R(e_{\mathrm{sel}}, e_{\mathrm{near}})$ (zoom)',                     False, False),
]

print("Section 10b: merged-electron / nearest-electron relational variables (all-samples)")
for key, xlabel, doLogy, varbin in _mele_nearestEle_relational_vars:
    if key not in s_hists:
        continue
    h = s_hists[key][{'cut': cut, 'samp': sum}]
    if np.sum(h.values()) == 0:
        continue
    fig, ax = plt.subplots(figsize=size)
    hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
    _histplot(h, ax, varbin=varbin,
              histtype='step', color='C0', linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('A.U.')
    if doLogy:
        ax.set_yscale('log')
    ax.set_title(rf'Merged electron vs. nearest other electron: {xlabel} — all samples')
    plt.tight_layout()
    plt.savefig(f'{plotdir}/hist_{seltag}_{key}_allsamps.png')
    plt.close(fig)


# ── Section 10c: why isn't the oppositely-charged mele_nearestEle subset
# counted as "resolved"? (all-samples only) ─────────────────────────────────
# For chargeProd==-1 entries of mele_nearestEle (the near track's charge is
# opposite the selected merged electron's), the near track's only possible
# gen match is the lepton sharing its own charge sign. If it also passed
# dR<0.1 and |dPt|/pt<0.2 to that gen, it would already have satisfied
# dr_match_ge/dr_match_gp on its own and the event would have landed in the
# ambiguous/resolved (_cat_sep) category instead of here as "merged" -- these
# plots show which of those two cuts (if either) actually failed.

print("Section 10c: oppositely-charged mele_nearestEle -- why not resolved (all-samples)")

_oppQ_dist_vars = [
    ('mele_nearestEle_oppQ_dr_to_targetgen',
     r'$\Delta R(e_{\mathrm{near}}, \mathrm{gen}_{\mathrm{target}})$', False),
    ('mele_nearestEle_oppQ_dPtRel_to_targetgen',
     r'$(p_T(e_{\mathrm{near}}) - p_T(\mathrm{gen}_{\mathrm{target}}))/p_T(\mathrm{gen}_{\mathrm{target}})$', False),
]

for key, xlabel, doLogy in _oppQ_dist_vars:
    if key not in s_hists:
        continue
    h = s_hists[key][{'cut': cut, 'samp': sum}]
    if np.sum(h.values()) == 0:
        continue
    fig, ax = plt.subplots(figsize=size)
    hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
    _histplot(h, ax, histtype='step', color='C0', linewidth=2)
    ax.axvline(0.1 if 'dr_to_targetgen' in key else 0.2, color='k', linestyle=':', linewidth=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('A.U.')
    if doLogy:
        ax.set_yscale('log')
    ax.set_title(rf'Opposite-charge mele\_nearestEle vs. target gen: {xlabel} — all samples')
    plt.tight_layout()
    plt.savefig(f'{plotdir}/hist_{seltag}_{key}_allsamps.png')
    plt.close(fig)

# Categorical breakdown: fraction of the oppositely-charged subset failing
# the dR cut only, the pt cut only, both, or (unexpectedly) neither.
_reason_key = 'mele_nearestEle_oppQ_failreason'
if _reason_key in s_hists:
    h = s_hists[_reason_key][{'cut': cut, 'samp': sum}]
    total = np.sum(h.values())
    if total > 0:
        cats = list(h.axes[0])
        vals = h.values()
        fracs = vals / total
        fig, ax = plt.subplots(figsize=size)
        hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
        ax.bar(cats, fracs, color='C0')
        for i, f in enumerate(fracs):
            ax.text(i, f, f'{f:.2f}', ha='center', va='bottom')
        ax.set_ylabel('Fraction of opposite-charge mele_nearestEle pairs')
        ax.set_title(r'Why isn\'t this pair "resolved"? — all samples')
        plt.tight_layout()
        plt.savefig(f'{plotdir}/hist_{seltag}_{_reason_key}_allsamps.png')
        plt.close(fig)


print("Done.")
