import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
from scipy.special import erf as sp_erf
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
    'legend.fontsize':  20,
    'figure.titlesize': 24,
})

outdir    = os.path.join(REPO_ROOT, 'workarea')
vers      = 'Jul2026noID'
selection = 'anmatchvtx'
hists_tag = 'recores'
saved_signal_hists = f"{outdir}/hists_sig{vers}_{selection}-sel_{hists_tag}.coffea"
seltag = 'nocuts'
cut    = 'cut1'
year   = 2024
size   = (16, 12)

s_hists = util.load(saved_signal_hists)[0]
s_pts   = utils.get_signal_point_dict(s_hists)

plotdir = os.path.join(REPO_ROOT, 'plots', 'recores')
os.makedirs(plotdir, exist_ok=True)

# ── helpers ───────────────────────────────────────────────────────────────────

# ── Double-sided crystal ball helpers ────────────────────────────────────────

def _dscb(x, mu, sigma, alpha_l, n_l, alpha_r, n_r):
    """Unnormalized double-sided crystal ball."""
    t   = (x - mu) / sigma
    A_l = (n_l / alpha_l) ** n_l * np.exp(-0.5 * alpha_l**2)
    B_l = n_l / alpha_l - alpha_l
    A_r = (n_r / alpha_r) ** n_r * np.exp(-0.5 * alpha_r**2)
    B_r = n_r / alpha_r - alpha_r
    # np.where evaluates both branches before selecting; clamp bases to 1
    # for the non-selected branch to avoid invalid/overflow in the power.
    base_l = np.where(t < -alpha_l,  B_l - t,  1.0)
    base_r = np.where(t >  alpha_r,  B_r + t,  1.0)
    return np.where(
        t < -alpha_l,
        A_l * base_l ** (-n_l),
        np.where(t > alpha_r, A_r * base_r ** (-n_r), np.exp(-0.5 * t**2))
    )

def _dscb_norm(alpha_l, n_l, alpha_r, n_r):
    """Analytic normalization integral of the unnormalized DSCB in t-space."""
    tail_l = np.exp(-0.5 * alpha_l**2) * n_l / (alpha_l * (n_l - 1))
    tail_r = np.exp(-0.5 * alpha_r**2) * n_r / (alpha_r * (n_r - 1))
    core   = np.sqrt(np.pi / 2) * (sp_erf(alpha_r / np.sqrt(2)) + sp_erf(alpha_l / np.sqrt(2)))
    return tail_l + core + tail_r


def dscb_density(x, mu, sigma, alpha_l, n_l, alpha_r, n_r):
    """Normalized DSCB PDF (integrates to 1 over x)."""
    return _dscb(x, mu, sigma, alpha_l, n_l, alpha_r, n_r) / (_dscb_norm(alpha_l, n_l, alpha_r, n_r) * sigma)

def _equalize_res_bins(edges, counts, variances=None):
    """Normalize bin counts to innermost-bin-width equivalent density.

    Divides each bin by (bin_width / min_bin_width) so that all bins are
    comparable regardless of the variable-width res axis structure.
    """
    widths  = np.diff(edges).astype(float)
    factors = widths / widths.min()
    counts  = counts.astype(float).copy()
    counts /= factors
    if variances is not None:
        variances = variances.astype(float).copy()
        variances /= factors**2
    return counts, variances

def fit_dscb(centers, counts, variances=None, res_edges=None, fit_range=None):
    """
    Fit DSCB to a 1-D histogram given bin centers and counts.
    Returns (mu, sigma, alpha_l, n_l, alpha_r, n_r, amp, chi2/dof) or None on failure.
    Pass variances (sumw2) for proper weighted-histogram errors; falls back to sqrt(N).
    fit_range: optional (lo, hi) tuple; only bins with centers in [lo, hi] are used in the fit.
    """
    if res_edges is None:
        mids = (centers[:-1] + centers[1:]) / 2
        res_edges = np.concatenate([[2*centers[0] - mids[0]], mids, [2*centers[-1] - mids[-1]]])
    counts, variances = _equalize_res_bins(res_edges, counts, variances)
    if fit_range is not None:
        rng_mask  = (centers >= fit_range[0]) & (centers <= fit_range[1])
        centers   = centers[rng_mask]
        counts    = counts[rng_mask]
        if variances is not None:
            variances = variances[rng_mask]
    total = counts.sum()
    if total <= 0:
        return None
    w    = counts / total
    mu0  = float((w * centers).sum())
    sig0 = float(max(np.sqrt((w * (centers - mu0)**2).sum()), 1e-3))
    p0 = [mu0, sig0, 1.0, 5.0, 1.0, 5.0, float(total)]
    lo = [centers[0], 1e-4, 0.1, 1.01, 0.1, 1.01, 0.0]
    hi = [centers[-1], 10.0, 10.0, 100.0, 10.0, 100.0, np.inf]
    if variances is not None:
        mask = variances > 0
        errs = np.sqrt(variances[mask])
    else:
        mask = counts > 0
        errs = np.sqrt(counts[mask])
    c_fit = centers[mask]
    y_fit = counts[mask]
    try:
        popt, pcov = curve_fit(
            lambda x, mu, s, al, nl, ar, nr, amp: amp * _dscb(x, mu, s, al, nl, ar, nr),
            c_fit, y_fit, p0=p0, sigma=errs, bounds=(lo, hi), maxfev=10000,
            absolute_sigma=True,
        )
        perr     = np.where(np.isfinite(np.diag(pcov)), np.sqrt(np.abs(np.diag(pcov))), np.nan)
        fitted   = popt[6] * _dscb(c_fit, *popt[:6])
        chi2     = float(np.sum(((y_fit - fitted) / errs)**2))
        ndof     = int(mask.sum()) - 7
        chi2_dof = chi2 / ndof if ndof > 0 else np.nan
        return np.concatenate([popt, [chi2_dof], perr])
    except Exception:
        return None

def _gauss(x, mu, sigma, amp):
    return amp * np.exp(-0.5 * ((x - mu) / sigma)**2)

def fit_gaussian(centers, counts, variances=None, res_edges=None, fit_range=None,
                 method='iterative_chi2', chi2_threshold=2.0, peak_fraction=0.2,
                 sigma_multiple=2.0):
    """
    Fit a Gaussian to a 1-D histogram, auto-selecting the range where the fit is reasonable.
    Drop-in replacement for fit_dscb — same 15-element return array; DSCB-only slots are NaN.

    method:
      'iterative_chi2' — expand window around peak bin until chi2/dof > chi2_threshold;
                         returns the widest range that still satisfies the threshold.
      'peak_fraction'  — include bins whose counts >= peak_fraction * peak_count.
      'sigma_multiple' — first-pass fit over the available range to estimate sigma,
                         then refit within +-sigma_multiple*sigma of the peak.
    """
    if res_edges is None:
        mids = (centers[:-1] + centers[1:]) / 2
        res_edges = np.concatenate([[2*centers[0] - mids[0]], mids, [2*centers[-1] - mids[-1]]])
    counts, variances = _equalize_res_bins(res_edges, counts, variances)
    if fit_range is not None:
        rng_mask  = (centers >= fit_range[0]) & (centers <= fit_range[1])
        centers   = centers[rng_mask]
        counts    = counts[rng_mask]
        if variances is not None:
            variances = variances[rng_mask]

    def _try_fit(mask):
        """Gaussian fit on selected bins; returns (popt, perr, chi2_dof) or None."""
        if variances is not None:
            valid = mask & (variances > 0)
            errs  = np.sqrt(variances[valid])
        else:
            valid = mask & (counts > 0)
            errs  = np.sqrt(counts[valid])
        if valid.sum() < 4:
            return None
        c_fit = centers[valid]
        y_fit = counts[valid]
        total = y_fit.sum()
        if total <= 0:
            return None
        w    = y_fit / total
        mu0  = float((w * c_fit).sum())
        sig0 = float(max(np.sqrt((w * (c_fit - mu0)**2).sum()), 1e-3))
        try:
            popt, pcov = curve_fit(
                _gauss, c_fit, y_fit,
                p0=[mu0, sig0, float(total)],
                sigma=errs,
                bounds=([c_fit[0], 1e-4, 0.0], [c_fit[-1], 10.0, np.inf]),
                maxfev=5000, absolute_sigma=True,
            )
            perr     = np.where(np.isfinite(np.diag(pcov)), np.sqrt(np.abs(np.diag(pcov))), np.nan)
            fitted   = _gauss(c_fit, *popt)
            chi2     = float(np.sum(((y_fit - fitted) / errs)**2))
            ndof     = int(valid.sum()) - 3
            chi2_dof = chi2 / ndof if ndof > 0 else np.nan
            return popt, perr, chi2_dof
        except Exception:
            return None

    if method == 'iterative_chi2':
        peak_idx = int(np.argmax(counts))
        best = None
        for radius in range(2, len(centers) // 2 + 2):
            lo_i = max(0, peak_idx - radius)
            hi_i = min(len(centers) - 1, peak_idx + radius)
            mask = np.zeros(len(centers), dtype=bool)
            mask[lo_i:hi_i + 1] = True
            res  = _try_fit(mask)
            if res is None:
                continue
            _, _, chi2_dof = res
            if not np.isnan(chi2_dof) and chi2_dof > chi2_threshold:
                break
            best = res
        if best is None:
            return None
        popt, perr, chi2_dof = best

    elif method == 'peak_fraction':
        peak_val = counts.max()
        if peak_val <= 0:
            return None
        mask = counts >= peak_fraction * peak_val
        res  = _try_fit(mask)
        if res is None:
            return None
        popt, perr, chi2_dof = res

    elif method == 'sigma_multiple':
        all_mask = np.ones(len(centers), dtype=bool)
        first    = _try_fit(all_mask)
        if first is None:
            return None
        mu0, sig0, _ = first[0]
        mask = (centers >= mu0 - sigma_multiple * sig0) & (centers <= mu0 + sigma_multiple * sig0)
        res  = _try_fit(mask) if mask.sum() >= 4 else None
        popt, perr, chi2_dof = res if res is not None else first

    else:
        raise ValueError(f"Unknown method {method!r}; choose 'iterative_chi2', 'peak_fraction', or 'sigma_multiple'")

    mu, sigma, amp = popt
    mu_err, sigma_err, amp_err = perr
    return np.array([mu, sigma, np.nan, np.nan, np.nan, np.nan, amp, chi2_dof,
                     mu_err, sigma_err, np.nan, np.nan, np.nan, np.nan, amp_err])

# ─────────────────────────────────────────────────────────────────────────────

def profile_from_3D(h3d, fit_range=None, fitter='dscb', fitter_kw=None):
    """DSCB-fitted (or Gaussian-fitted) mu and sigma in each (axis 0, axis 1) cell, with moment fallback.

    fitter: 'dscb' (default) or 'gauss'. fitter_kw is forwarded to fit_gaussian when fitter='gauss'.
    """
    _fit_fn  = fit_gaussian if fitter == 'gauss' else fit_dscb
    _fit_kw  = fitter_kw or {}
    vals        = h3d.values()       # (n0, n1, n_res)
    variances   = h3d.variances()
    res_centers = h3d.axes[2].centers
    res_edges   = h3d.axes[2].edges
    n0, n1 = vals.shape[0], vals.shape[1]
    mu_arr    = np.full((n0, n1), np.nan)
    sigma_arr = np.full((n0, n1), np.nan)
    for i in range(n0):
        for j in range(n1):
            counts  = vals[i, j, :]
            vars_ij = variances[i, j, :] if variances is not None else None
            result = _fit_fn(res_centers, counts, vars_ij, res_edges=res_edges, fit_range=fit_range, **_fit_kw)
            if result is not None:
                mu_arr[i, j]    = result[0]
                sigma_arr[i, j] = result[1]
            else:
                total = counts.sum()
                if total > 0:
                    mu0  = (counts * res_centers).sum() / total
                    var0 = np.maximum((counts * (res_centers - mu0)**2).sum() / total, 0)
                    mu_arr[i, j]    = mu0
                    sigma_arr[i, j] = np.sqrt(var0)
    return h3d.axes[0].edges, h3d.axes[1].edges, mu_arr, sigma_arr

def profile_from_2D(h2d, rebin=1, target_edges=None, fit_range=None, fitter='dscb', fitter_kw=None):
    """DSCB-fitted (or Gaussian-fitted) profile: median and sigma of resolution axis (axis 1) per x bin (axis 0).

    fitter: 'dscb' (default) or 'gauss'. fitter_kw is forwarded to fit_gaussian when fitter='gauss'.
    """
    _fit_fn = fit_gaussian if fitter == 'gauss' else fit_dscb
    _fit_kw = fitter_kw or {}
    vals        = h2d.values()
    variances   = h2d.variances()
    x_edges     = h2d.axes[0].edges
    res_centers = h2d.axes[1].centers
    res_edges   = h2d.axes[1].edges
    if target_edges is not None:
        target_edges = np.asarray(target_edges, dtype=float)
        x_centers    = 0.5 * (target_edges[:-1] + target_edges[1:])
        grouped      = np.zeros((len(x_centers), vals.shape[1]))
        grouped_vars = np.zeros((len(x_centers), vals.shape[1])) if variances is not None else None
        for i, (lo, hi) in enumerate(zip(target_edges[:-1], target_edges[1:])):
            idx_lo = np.searchsorted(x_edges, lo, 'left')
            idx_hi = np.searchsorted(x_edges, hi, 'right') - 1
            if idx_hi > idx_lo:
                grouped[i] = vals[idx_lo:idx_hi].sum(axis=0)
                if grouped_vars is not None:
                    grouped_vars[i] = variances[idx_lo:idx_hi].sum(axis=0)
        vals      = grouped
        variances = grouped_vars
        x_edges   = target_edges
    elif rebin > 1:
        n = vals.shape[0]
        n_new = n // rebin
        vals = vals[:n_new * rebin].reshape(n_new, rebin, -1).sum(axis=1)
        if variances is not None:
            variances = variances[:n_new * rebin].reshape(n_new, rebin, -1).sum(axis=1)
        x_edges_rb = x_edges[::rebin][:n_new + 1]
        x_centers = 0.5 * (x_edges_rb[:-1] + x_edges_rb[1:])
        x_edges   = x_edges_rb
    else:
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    totw  = vals.sum(axis=1)
    sumw2 = variances.sum(axis=1) if variances is not None else totw
    eff_n = np.divide(totw**2, sumw2, out=np.zeros_like(totw), where=sumw2 > 0)
    medians    = np.full(len(x_centers), np.nan)
    sigmas     = np.full(len(x_centers), np.nan)
    sigma_errs = np.full(len(x_centers), np.nan)
    valid      = np.zeros(len(x_centers), dtype=bool)
    for i, counts in enumerate(vals):
        vars_i = variances[i] if variances is not None else None
        result = _fit_fn(res_centers, counts, vars_i, res_edges=res_edges, fit_range=fit_range, **_fit_kw)
        if result is not None:
            medians[i]    = result[0]
            sigmas[i]     = result[1]
            sigma_errs[i] = result[9]   # index 9 = sigma_err
            valid[i]      = True
        else:
            total = counts.sum()
            if total > 0:
                mu0  = (counts * res_centers).sum() / total
                var0 = np.maximum((counts * (res_centers - mu0)**2).sum() / total, 0)
                medians[i] = mu0
                sigmas[i]  = np.sqrt(var0)
                valid[i]   = True
    return x_centers, medians, sigmas, sigma_errs, valid, x_edges, totw, eff_n

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
    {'qty': 'pt',  'title': r'$p_T$ Resolution',
     'xlabel': r'$(p_T^\mathrm{reco} - p_T^\mathrm{gen})/p_T^\mathrm{gen}$',
     'ylabel': r'DSCB $\mu$ ($\pm\,\sigma$ band)  of  $(p_T^\mathrm{reco} - p_T^\mathrm{gen})/p_T^\mathrm{gen}$',
     'fit_range': (-0.2, 0.2), 'xlim': (-0.2, 0.2), 'sigma_top': 0.1},
    # {'qty': 'e',  'title': r'Energy Resolution',
    #  'xlabel': r'$(E^\mathrm{reco} - E^\mathrm{gen})/E^\mathrm{gen}$',
    #  'ylabel': r'DSCB $\mu$ ($\pm\,\sigma$ band)  of  $(E^\mathrm{reco} - E^\mathrm{gen})/E^\mathrm{gen}$',
    #  'fit_range': (-0.2, 0.2), 'xlim': (-0.2, 0.2), 'sigma_top': 0.1},
    {'qty': 'dxy', 'title': r'$d_{xy}$ Resolution',
     'xlabel': r'$(d_{xy}^\mathrm{reco} - d_{xy}^\mathrm{gen})/d_{xy}^\mathrm{gen}$',
     'ylabel': r'DSCB $\mu$ ($\pm\,\sigma$ band)  of  $(d_{xy}^\mathrm{reco} - d_{xy}^\mathrm{gen})/d_{xy}^\mathrm{gen}$',
     'fit_range': (-0.5, 0.5), 'xlim': (-0.5, 0.5), 'sigma_top': 0.3},
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
    #h_lpt  = s_hists[f'res_{qty}_lpt' ][{"cut": cut, "samp": sum}]
    h_ged  = s_hists[f'res_{qty}_ged' ][{"cut": cut, "samp": sum}]
    #h_alpt = s_hists[f'res_{qty}_alpt'][{"cut": cut, "samp": sum}]
    h_xlpt = s_hists[f'res_{qty}_xlpt'][{"cut": cut, "samp": sum}]
    res_centers = h_ged.axes[0].centers
    res_edges   = h_ged.axes[0].edges
    c_ged,  v_ged  = _equalize_res_bins(res_edges, h_ged.values(),  h_ged.variances())
    c_xlpt, v_xlpt = _equalize_res_bins(res_edges, h_xlpt.values(), h_xlpt.variances())
    fig, ax = plt.subplots(figsize=size)
    hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
    hep.histplot([c_ged, c_xlpt], bins=res_edges, ax=ax, histtype='errorbar', density=True,
                 yerr=[np.sqrt(v_ged), np.sqrt(v_xlpt)],
                 label=['GED', 'XCleaned LowPt'], color=['C0', 'C1'],
                 markersize=6, elinewidth=0.8, capsize=2)
    x_dense = np.linspace(res_edges[0], res_edges[-1], 1000)
    param_lines = []
    for h, color, lbl in [(h_ged, 'C0', 'GED'), (h_xlpt, 'C1', 'XCleaned LowPt')]:
        result = fit_dscb(h.axes[0].centers, h.values(), h.variances(), res_edges=h.axes[0].edges, fit_range=qcfg['fit_range'])
        #result = fit_gaussian(h.axes[0].centers, h.values(), h.variances(), res_edges=h.axes[0].edges,
        #                      fit_range=qcfg['fit_range'], method='sigma_multiple')
        if result is not None:
            mu, sigma, al, nl, ar, nr = result[:6]
            #mu, sigma, amp = result[0], result[1], result[6]
            chi2_dof = result[7]
            ax.plot(x_dense, dscb_density(x_dense, mu, sigma, al, nl, ar, nr),
                    #amp * np.exp(-0.5 * ((x_dense - mu) / sigma)**2) / norm_eq,
                    color=color, lw=2, ls='--',
                    label=rf'{lbl} fit: $\mu$={mu:.3f}, $\sigma$={sigma:.3f}')
            param_lines.append(
                rf'{lbl}: $\mu$={mu:.2g}, $\sigma$={sigma:.2g}' + '\n'
                rf'    $\alpha_L$={al:.2f}, $n_L$={nl:.1f}, $\alpha_R$={ar:.2f}, $n_R$={nr:.1f}, $\chi^2$/dof={chi2_dof:.2f}'
            )
            #param_lines.append(
            #    rf'{lbl}: $\mu$={mu:.2g}, $\sigma$={sigma:.2g}, $\chi^2$/dof={chi2_dof:.2f}'
            #)
    ax.set_xlabel(qcfg['xlabel'])
    ax.set_ylabel('A.U.')
    ax.set_xlim(*qcfg['xlim'])
    ax.set_title(f'Electron {qcfg["title"]} — all samples')
    ax.legend()
    if param_lines:
        ax.text(0.02, 0.98, '\n'.join(param_lines), transform=ax.transAxes,
                fontsize=18, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig(f'{plotdir}/hist_{seltag}_res_{qty}_1D_allsamps.png')
    plt.close(fig)


# ── Section 2: Profile plots (DSCB median ± σ vs gen variable) ───────────────

for qcfg in qty_configs:
    qty = qcfg['qty']
    for gvcfg in gen_var_configs:
        gvar = gvcfg['var']

        # Summed over all samples
        #x_l, mn_l, sd_l, _, vl = profile_from_2D(s_hists[f'res_{qty}_vs_{gvar}_lpt' ][{"cut": cut, "samp": sum}], rebin=2)
        x_g, mn_g, sd_g, _, vg, *_ = profile_from_2D(s_hists[f'res_{qty}_vs_{gvar}_ged' ][{"cut": cut, "samp": sum}], rebin=1, fit_range=qcfg['fit_range'])
        #x_a, mn_a, sd_a, _, va, *_ = profile_from_2D(s_hists[f'res_{qty}_vs_{gvar}_alpt'][{"cut": cut, "samp": sum}], rebin=2, fit_range=qcfg['fit_range'])
        x_x, mn_x, sd_x, _, vx, *_ = profile_from_2D(s_hists[f'res_{qty}_vs_{gvar}_xlpt'][{"cut": cut, "samp": sum}], rebin=1, fit_range=qcfg['fit_range'])
        fig, ax = plt.subplots(figsize=size)
        hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
        #ax.plot(x_l[vl], mn_l[vl], '-', color='C0', label=r'LowPt ($\mu \pm\,\sigma$)')
        #ax.fill_between(x_l[vl], mn_l[vl] - sd_l[vl], mn_l[vl] + sd_l[vl], color='C0', alpha=0.3, label='_')
        ax.plot(x_g[vg], mn_g[vg], '-', color='C0', label=r'GED ($\mu \pm\,\sigma$)')
        ax.fill_between(x_g[vg], mn_g[vg] - sd_g[vg], mn_g[vg] + sd_g[vg], color='C0', alpha=0.3, label='_')
        #ax.plot(x_a[va], mn_a[va], '-', color='C2', label=r'AllLowPt ($\mu \pm\,\sigma$)')
        #ax.fill_between(x_a[va], mn_a[va] - sd_a[va], mn_a[va] + sd_a[va], color='C2', alpha=0.3, label='_')
        ax.plot(x_x[vx], mn_x[vx], '-', color='C1', label=r'XCleaned LowPt ($\mu \pm\,\sigma$)')
        ax.fill_between(x_x[vx], mn_x[vx] - sd_x[vx], mn_x[vx] + sd_x[vx], color='C1', alpha=0.3, label='_')
        ax.axhline(0, color='gray', linestyle=':', linewidth=1)
        ax.set_xlabel(gvcfg['label'])
        ax.set_ylabel(qcfg['ylabel'])
        ax.set_ylim(*qcfg['xlim'])
        if gvcfg['doLogx']:
            ax.set_xscale('log')
        ax.set_title(f'Electron {qcfg["title"]} vs {gvcfg["label"]} — all samples')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'{plotdir}/hist_{seltag}_res_{qty}_vs_{gvar}_profile_allsamps.png')
        plt.close(fig)


# ── Section 3: 2D colormesh plots (summed over all samples) ──────────────────

_2d_specs = [
    #('res_pt_vs_genpt_lpt',  r'LowPt $p_T$ Res vs Gen $p_T$',                False),
    ('res_pt_vs_genpt_ged',  r'GED $p_T$ Res vs Gen $p_T$',                  False),
    #('res_pt_vs_genpt_alpt', r'AllLowPt $p_T$ Res vs Gen $p_T$',             False),
    ('res_pt_vs_genpt_xlpt', r'XCleaned LowPt $p_T$ Res vs Gen $p_T$',       False),
    # ('res_e_vs_genpt_lpt',   r'LowPt Energy Res vs Gen $p_T$',              False),
    # ('res_e_vs_genpt_ged',   r'GED Energy Res vs Gen $p_T$',                False),
    # ('res_e_vs_genpt_alpt',  r'AllLowPt Energy Res vs Gen $p_T$',           False),
    # ('res_e_vs_genpt_xlpt',  r'XCleaned LowPt Energy Res vs Gen $p_T$',     False),
    #('res_pt_vs_genlxy_lpt', r'LowPt $p_T$ Res vs Gen $L_{xy}$',             True),
    ('res_pt_vs_genlxy_ged', r'GED $p_T$ Res vs Gen $L_{xy}$',               True),
    #('res_pt_vs_genlxy_alpt',r'AllLowPt $p_T$ Res vs Gen $L_{xy}$',          True),
    ('res_pt_vs_genlxy_xlpt',r'XCleaned LowPt $p_T$ Res vs Gen $L_{xy}$',    True),
    # ('res_e_vs_genlxy_lpt',  r'LowPt Energy Res vs Gen $L_{xy}$',          True),
    # ('res_e_vs_genlxy_ged',  r'GED Energy Res vs Gen $L_{xy}$',            True),
    # ('res_e_vs_genlxy_alpt', r'AllLowPt Energy Res vs Gen $L_{xy}$',       True),
    # ('res_e_vs_genlxy_xlpt', r'XCleaned LowPt Energy Res vs Gen $L_{xy}$', True),
    #('res_pt_vs_geneta_lpt', r'LowPt $p_T$ Res vs Gen $\eta$',               False),
    ('res_pt_vs_geneta_ged', r'GED $p_T$ Res vs Gen $\eta$',                 False),
    #('res_pt_vs_geneta_alpt',r'AllLowPt $p_T$ Res vs Gen $\eta$',            False),
    ('res_pt_vs_geneta_xlpt',r'XCleaned LowPt $p_T$ Res vs Gen $\eta$',      False),
    # ('res_e_vs_geneta_lpt',  r'LowPt Energy Res vs Gen $\eta$',            False),
    # ('res_e_vs_geneta_ged',  r'GED Energy Res vs Gen $\eta$',              False),
    # ('res_e_vs_geneta_alpt', r'AllLowPt Energy Res vs Gen $\eta$',         False),
    # ('res_e_vs_geneta_xlpt', r'XCleaned LowPt Energy Res vs Gen $\eta$',   False),
    ('res_dxy_vs_genpt_ged',  r'GED $d_{xy}$ Res vs Gen $p_T$',                  False),
    ('res_dxy_vs_genpt_xlpt', r'XCleaned LowPt $d_{xy}$ Res vs Gen $p_T$',       False),
    ('res_dxy_vs_genlxy_ged', r'GED $d_{xy}$ Res vs Gen $L_{xy}$',               True),
    ('res_dxy_vs_genlxy_xlpt',r'XCleaned LowPt $d_{xy}$ Res vs Gen $L_{xy}$',    True),
    ('res_dxy_vs_geneta_ged', r'GED $d_{xy}$ Res vs Gen $\eta$',                 False),
    ('res_dxy_vs_geneta_xlpt',r'XCleaned LowPt $d_{xy}$ Res vs Gen $\eta$',      False),
]

for histname, title, doLogx in _2d_specs:
    h2d       = s_hists[histname][{"cut": cut, "samp": sum}]
    x_edges   = h2d.axes[0].edges
    res_edges = h2d.axes[1].edges
    vals      = h2d.values()

    fig, ax = plt.subplots(figsize=size)
    hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
    vals_pos = np.where(vals > 0, vals, np.nan)
    pcm = ax.pcolormesh(x_edges, res_edges, vals_pos.T, cmap='viridis', norm=LogNorm())
    plt.colorbar(pcm, ax=ax, label='Events')
    ax.set_xlabel(h2d.axes[0].label)
    ax.set_ylabel(h2d.axes[1].label)
    if doLogx:
        ax.set_xscale('log')
    ax.set_title(f'{title} — all samples')
    plt.tight_layout()
    plt.savefig(f'{plotdir}/hist_{seltag}_{histname}_2D_allsamps.png')
    plt.close(fig)

# ── Section 4: 2D mean-resolution maps (genpt × genlxy, genpt × geneta) ──────

_cmap_res = plt.cm.RdBu_r.copy()
_cmap_res.set_bad('lightgray')

_mean2d_specs = [
    ('res_pt_vs_genpt_genlxy',  r'DSCB $\mu$: $p_T$ Resolution',    r'DSCB $\mu$ of $(p_T^\mathrm{reco}-p_T^\mathrm{gen})/p_T^\mathrm{gen}$',          (-0.2, 0.2)),
    # ('res_e_vs_genpt_genlxy',  r'DSCB $\mu$: Energy Resolution',   r'DSCB $\mu$ of $(E^\mathrm{reco}-E^\mathrm{gen})/E^\mathrm{gen}$',                  (-0.2, 0.2)),
    ('res_pt_vs_genpt_geneta',  r'DSCB $\mu$: $p_T$ Resolution',    r'DSCB $\mu$ of $(p_T^\mathrm{reco}-p_T^\mathrm{gen})/p_T^\mathrm{gen}$',          (-0.2, 0.2)),
    # ('res_e_vs_genpt_geneta',  r'DSCB $\mu$: Energy Resolution',   r'DSCB $\mu$ of $(E^\mathrm{reco}-E^\mathrm{gen})/E^\mathrm{gen}$',                  (-0.2, 0.2)),
    ('res_dxy_vs_genpt_genlxy', r'DSCB $\mu$: $d_{xy}$ Resolution', r'DSCB $\mu$ of $(d_{xy}^\mathrm{reco}-d_{xy}^\mathrm{gen})/d_{xy}^\mathrm{gen}$', (-0.5, 0.5)),
    ('res_dxy_vs_genpt_geneta', r'DSCB $\mu$: $d_{xy}$ Resolution', r'DSCB $\mu$ of $(d_{xy}^\mathrm{reco}-d_{xy}^\mathrm{gen})/d_{xy}^\mathrm{gen}$', (-0.5, 0.5)),
]

for histbase, title_base, cbar_label, fit_range in _mean2d_specs:
    for reco_type, reco_label in [('ged', 'GED'), ('xlpt', 'XCleaned LowPt')]:  #('lpt', 'LowPt'), ('alpt', 'AllLowPt'),
        histname = f'{histbase}_{reco_type}'
        h3d = s_hists[histname][{"cut": cut, "samp": sum}]
        pt_edges, y_edges, mu, _ = profile_from_3D(h3d, fit_range=fit_range)

        vmax = np.nanmax(np.abs(mu))
        fig, ax = plt.subplots(figsize=size)
        hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
        pcm = ax.pcolormesh(pt_edges, y_edges, mu.T, cmap=_cmap_res, vmin=-vmax, vmax=vmax)
        plt.colorbar(pcm, ax=ax, label=cbar_label)
        ax.set_xlabel(h3d.axes[0].label)
        ax.set_ylabel(h3d.axes[1].label)
        ax.set_title(f'{title_base} ({reco_label}) — all samples')
        plt.tight_layout()
        plt.savefig(f'{plotdir}/hist_{seltag}_{histname}_meanres_allsamps.png')
        plt.close(fig)

# ── Section 5: 2D RMS-resolution maps (genpt × genlxy, genpt × geneta) ───────

_cmap_rms = plt.cm.viridis.copy()
_cmap_rms.set_bad('lightgray')

_sigma2d_specs = [
    ('res_pt_vs_genpt_genlxy',  r'DSCB $\sigma$: $p_T$ Resolution',    r'DSCB $\sigma$ of $(p_T^\mathrm{reco}-p_T^\mathrm{gen})/p_T^\mathrm{gen}$',          (-0.2, 0.2)),
    # ('res_e_vs_genpt_genlxy',  r'DSCB $\sigma$: Energy Resolution',   r'DSCB $\sigma$ of $(E^\mathrm{reco}-E^\mathrm{gen})/E^\mathrm{gen}$',                  (-0.2, 0.2)),
    ('res_pt_vs_genpt_geneta',  r'DSCB $\sigma$: $p_T$ Resolution',    r'DSCB $\sigma$ of $(p_T^\mathrm{reco}-p_T^\mathrm{gen})/p_T^\mathrm{gen}$',          (-0.2, 0.2)),
    # ('res_e_vs_genpt_geneta',  r'DSCB $\sigma$: Energy Resolution',   r'DSCB $\sigma$ of $(E^\mathrm{reco}-E^\mathrm{gen})/E^\mathrm{gen}$',                  (-0.2, 0.2)),
    ('res_dxy_vs_genpt_genlxy', r'DSCB $\sigma$: $d_{xy}$ Resolution', r'DSCB $\sigma$ of $(d_{xy}^\mathrm{reco}-d_{xy}^\mathrm{gen})/d_{xy}^\mathrm{gen}$', (-0.5, 0.5)),
    ('res_dxy_vs_genpt_geneta', r'DSCB $\sigma$: $d_{xy}$ Resolution', r'DSCB $\sigma$ of $(d_{xy}^\mathrm{reco}-d_{xy}^\mathrm{gen})/d_{xy}^\mathrm{gen}$', (-0.5, 0.5)),
]

for histbase, title_base, cbar_label, fit_range in _sigma2d_specs:
    for reco_type, reco_label in [('ged', 'GED'), ('xlpt', 'XCleaned LowPt')]:  #('lpt', 'LowPt'), ('alpt', 'AllLowPt'),
        histname = f'{histbase}_{reco_type}'
        h3d = s_hists[histname][{"cut": cut, "samp": sum}]
        pt_edges, y_edges, _, sigma = profile_from_3D(h3d, fit_range=fit_range)

        fig, ax = plt.subplots(figsize=size)
        hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
        pcm = ax.pcolormesh(pt_edges, y_edges, sigma.T, cmap=_cmap_rms)
        plt.colorbar(pcm, ax=ax, label=cbar_label)
        ax.set_xlabel(h3d.axes[0].label)
        ax.set_ylabel(h3d.axes[1].label)
        ax.set_title(f'{title_base} ({reco_label}) — all samples')
        plt.tight_layout()
        plt.savefig(f'{plotdir}/hist_{seltag}_{histname}_sigmares_allsamps.png')
        plt.close(fig)

# ── Section 6: 1D DSCB σ vs gen variable (LowPt and GED overlaid) ────────────

for qcfg in qty_configs:
    qty = qcfg['qty']
    for gvcfg in gen_var_configs:
        gvar = gvcfg['var']
        _pt_edges = np.concatenate([np.arange(0, 21, dtype=float), np.arange(22, 51, 2, dtype=float)])
        prof_kw = dict(target_edges=_pt_edges) if gvar == 'genpt' else dict(rebin=1)
        #x_l, _, sd_l, se_l, vl, xedg_l, totw_l, effn_l = profile_from_2D(s_hists[f'res_{qty}_vs_{gvar}_lpt' ][{"cut": cut, "samp": sum}], **prof_kw)
        x_g, _, sd_g, se_g, vg, xedg_g, totw_g, effn_g = profile_from_2D(s_hists[f'res_{qty}_vs_{gvar}_ged' ][{"cut": cut, "samp": sum}], **prof_kw, fit_range=qcfg['fit_range'])
        #x_a, _, sd_a, se_a, va, xedg_a, totw_a, effn_a = profile_from_2D(s_hists[f'res_{qty}_vs_{gvar}_alpt'][{"cut": cut, "samp": sum}], **prof_kw, fit_range=qcfg['fit_range'])
        x_x, _, sd_x, se_x, vx, xedg_x, totw_x, effn_x = profile_from_2D(s_hists[f'res_{qty}_vs_{gvar}_xlpt'][{"cut": cut, "samp": sum}], **prof_kw, fit_range=qcfg['fit_range'])

        for _lbl, _sd, _se, _valid, _xedg, _totw, _effn in [
            ('GED',            sd_g, se_g, vg, xedg_g, totw_g, effn_g),
            ('XCleaned LowPt', sd_x, se_x, vx, xedg_x, totw_x, effn_x),
        ]:
            _mask = _valid & np.isfinite(_se) & (_se > 0.05)
            if _mask.any():
                print(f"\n{qty} vs {gvar} — {_lbl}: bins with sigma_err > 0.05")
                print(f"  {'[lo, hi]':>22}  {'eff_n':>10}  {'tot_weight':>12}  {'sigma':>8}  {'sigma_err':>10}")
                for _i in np.where(_mask)[0]:
                    print(f"  [{_xedg[_i]:8.4g}, {_xedg[_i+1]:8.4g}]"
                          f"  {_effn[_i]:10.1f}  {_totw[_i]:12.4g}"
                          f"  {_sd[_i]:8.4f}  {_se[_i]:10.4f}")

        sd_g_plot = np.where(vg, sd_g, np.nan)
        se_g_plot = np.where(vg & np.isfinite(se_g), se_g, np.nan)
        sd_x_plot = np.where(vx, sd_x, np.nan)
        se_x_plot = np.where(vx & np.isfinite(se_x), se_x, np.nan)
        fig, ax = plt.subplots(figsize=size)
        hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
        hep.histplot(
            [sd_g_plot, sd_x_plot], bins=xedg_g,
            histtype='step', yerr=[se_g_plot, se_x_plot],
            color=['C0', 'C1'], label=['GED', 'XCleaned LowPt'],
            ax=ax,
        )
        ax.set_xlabel(gvcfg['label'])
        ax.set_ylabel(r'DSCB $\sigma$  of  ' + qcfg['xlabel'])
        ax.set_ylim(bottom=0, top=qcfg['sigma_top'])
        if gvcfg['doLogx']:
            ax.set_xscale('log')
        ax.set_title(f'Electron {qcfg["title"]} DSCB $\\sigma$ vs {gvcfg["label"]} — all samples')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'{plotdir}/hist_{seltag}_res_{qty}_vs_{gvar}_sigma1D_allsamps.png')
        plt.close(fig)

# ── Section 7: 1D resolution slices for individual gen-variable bins ──────────

_slice_specs = [
    {'var': 'genpt',  'label': r'$p_T^\mathrm{gen}$',         'unit': 'GeV', 'abs': False,
     'bins': [(2, 3), (5, 6), (10, 12)]},
    {'var': 'genlxy', 'label': r'$L_{xy}^\mathrm{gen}$',      'unit': 'cm',  'abs': False,
     'bins': [(1, 2), (5, 10)]},
    {'var': 'geneta', 'label': r'$|\eta^\mathrm{gen}|$',       'unit': '',    'abs': True,
     'bins': [(0.0, 0.1), (2.1, 2.2)]},
]

for qcfg in qty_configs:
    qty = qcfg['qty']
    for sspec in _slice_specs:
        gvar      = sspec['var']
        #h_lpt     = s_hists[f'res_{qty}_vs_{gvar}_lpt' ][{"cut": cut, "samp": sum}]
        h_ged     = s_hists[f'res_{qty}_vs_{gvar}_ged' ][{"cut": cut, "samp": sum}]
        #h_alpt    = s_hists[f'res_{qty}_vs_{gvar}_alpt'][{"cut": cut, "samp": sum}]
        h_xlpt    = s_hists[f'res_{qty}_vs_{gvar}_xlpt'][{"cut": cut, "samp": sum}]
        x_edges   = h_ged.axes[0].edges
        res_centers = h_ged.axes[1].centers
        res_edges   = h_ged.axes[1].edges
        x_dense     = np.linspace(res_edges[0], res_edges[-1], 1000)

        for lo, hi in sspec['bins']:
            il = np.searchsorted(x_edges, lo, 'left')
            ir = np.searchsorted(x_edges, hi, 'right') - 1
            #counts_l = h_lpt.values()[il:ir].sum(axis=0)
            #var_l    = h_lpt.variances()[il:ir].sum(axis=0)
            counts_g_raw = h_ged.values()[il:ir].sum(axis=0)
            var_g_raw    = h_ged.variances()[il:ir].sum(axis=0)
            #counts_a_raw = h_alpt.values()[il:ir].sum(axis=0)
            #var_a_raw    = h_alpt.variances()[il:ir].sum(axis=0)
            counts_x_raw = h_xlpt.values()[il:ir].sum(axis=0)
            var_x_raw    = h_xlpt.variances()[il:ir].sum(axis=0)
            if sspec['abs']:
                il2 = np.searchsorted(x_edges, -hi, 'left')
                ir2 = np.searchsorted(x_edges, -lo, 'right') - 1
                if ir2 > il2:
                    counts_g_raw += h_ged.values()[il2:ir2].sum(axis=0)
                    var_g_raw    += h_ged.variances()[il2:ir2].sum(axis=0)
                    counts_x_raw += h_xlpt.values()[il2:ir2].sum(axis=0)
                    var_x_raw    += h_xlpt.variances()[il2:ir2].sum(axis=0)
            counts_g, var_g = _equalize_res_bins(res_edges, counts_g_raw, var_g_raw)
            counts_x, var_x = _equalize_res_bins(res_edges, counts_x_raw, var_x_raw)

            fig, ax = plt.subplots(figsize=size)
            hep.cms.label('Private Work', data=True, year=year, com='13.6', ax=ax)
            hep.histplot(
                [counts_g, counts_x], bins=res_edges, ax=ax,
                histtype='errorbar', density=True,
                yerr=[np.sqrt(var_g), np.sqrt(var_x)],
                label=['GED', 'XCleaned LowPt'], color=['C0', 'C1'],
                markersize=6, elinewidth=0.8, capsize=2,
            )
            param_lines = []
            for counts, variances, color, lbl in [
                #(counts_l_raw, var_l_raw, 'C0', 'LowPt'),
                (counts_g_raw, var_g_raw, 'C0', 'GED'),
                #(counts_a_raw, var_a_raw, 'C2', 'AllLowPt'),
                (counts_x_raw, var_x_raw, 'C1', 'XCleaned LowPt'),
            ]:
                result = fit_dscb(res_centers, counts, variances, res_edges=res_edges, fit_range=qcfg['fit_range'])
                #result = fit_gaussian(res_centers, counts, variances, res_edges=res_edges,
                #                      fit_range=qcfg['fit_range'], method='sigma_multiple')
                if result is not None:
                    mu, sigma, al, nl, ar, nr = result[:6]
                    #mu, sigma, amp = result[0], result[1], result[6]
                    chi2_dof = result[7]
                    ax.plot(x_dense, dscb_density(x_dense, mu, sigma, al, nl, ar, nr),
                            #amp * np.exp(-0.5 * ((x_dense - mu) / sigma)**2) / norm_eq,
                            color=color, lw=2, ls='--')
                    param_lines.append(
                        rf'{lbl}: $\mu$={mu:.2g}, $\sigma$={sigma:.2g}' + '\n'
                        rf'    $\alpha_L$={al:.2f}, $n_L$={nl:.1f}, $\alpha_R$={ar:.2f}, $n_R$={nr:.1f}, '
                        rf'$\chi^2$/dof={chi2_dof:.2f}'
                    )
                    #param_lines.append(
                    #    rf'{lbl}: $\mu$={mu:.2g}, $\sigma$={sigma:.2g}, $\chi^2$/dof={chi2_dof:.2f}'
                    #)
            ax.set_xlabel(qcfg['xlabel'])
            ax.set_ylabel('A.U.')
            ax.set_xlim(*qcfg['xlim'])
            if param_lines:
                ax.text(0.02, 0.98, '\n'.join(param_lines), transform=ax.transAxes,
                        fontsize=18, va='top', ha='left',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax.set_title(
                f'Electron {qcfg["title"]}: '
                rf'{sspec["label"]} $\in$ [{lo}, {hi}] {sspec["unit"]} — all samples'
            )
            ax.legend()
            plt.tight_layout()
            lo_str = str(lo).replace('.', 'p').replace('-', 'n')
            hi_str = str(hi).replace('.', 'p').replace('-', 'n')
            plt.savefig(f'{plotdir}/hist_{seltag}_res_{qty}_vs_{gvar}_{lo_str}to{hi_str}_1Dslice.png')
            plt.close(fig)
