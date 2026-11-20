"""
Step 3: optimize the BDT score threshold of a selection, for that selection exactly as it is defined.

Workflow (run from trainBDT/, after step2_trainBDT.py has produced the model):
  0. Find the BDT cut in cut_config (the cutN calling getBDTscore_kfold) and the model JSON it uses (its bdt_json).
     Skim signal and bkg through every cut *before* the BDT cut into new h5 files, so the scan sees the
     population the BDT cut is actually applied to. Each event is scored by models[eventNum % k], the same
     rule getBDTscore_kfold uses in the selection, so every MC event has an out-of-fold score.
  1. Figure of merit: S/sqrt(B), S and B weighted to xsec*lumi (wgt_norm, signed weights).
  2. A threshold is only valid if at least min_bkg_raw raw bkg MC events pass it.
  3. Scan per signal point (m1, delta, ctau); the common threshold maximizes the worst point's
     FOM(t)/FOM_best (minimax regret) over the benchmark points.
  4. Thresholds are chosen on sel_folds and S, B are quoted on eval_folds (yields scaled up to the full
     sample by k/len(folds)). The loosest threshold within plateau_frac of the maximum is taken, not the argmax.
  5. The result goes into the model JSON as thres_opt[<cut_config name>], which the selection's BDT cut reads.
     step2_trainBDT.py rewrites the model JSON on retraining, which drops thres_opt: rerun this script then.
"""
import os
import sys
import re
import json
import time
import inspect
import hashlib
import importlib
sys.path.append("../analysisTools/")
sys.path.append("../configs/cut_configs/")
import numpy as np
import h5py
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import BDTutils_kfold as bdt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for analysisTools.*
import analysisTools.plotTools as ptools

ptools.set_plot_style()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
cut_config = "../configs/cut_configs/aptrk_selection.py"
bdt_type = "merged" # "vertexed" (an_selection) or "merged" (aptrk_selection): which BDT inputs the skim writes
signal_json = "../configs/sample_configs/signal_2024_Jul2026noID_0p1_skimmed.json"
bkg_json = "../configs/sample_configs/bkg_2024_Aug2026_skimmed.json"
bkg_slim_json = "../configs/sample_configs/bkg_2024_Aug2026-slim.json" # all-events companion config, used for bkg normalization
remake_h5 = True # False reuses the h5 files a previous run of this script made (only the scan is redone)
workers = 4

min_bkg_raw = 10          # a threshold is valid only if >= this many raw bkg MC events pass it (in the folds used)
thres_step = 0.001        # scan grid step in BDT score
sel_folds = [0, 1, 2]     # folds used to choose the thresholds
eval_folds = [3, 4]       # folds used to quote S, B at the chosen threshold
plateau_frac = 0.95       # take the loosest threshold whose FOM is >= plateau_frac * max
benchmark_points = None   # list of (m1, delta, ctau) entering the minimax; None = every signal point
write_json = True         # write thres_opt into the model JSON


def find_bdt_cut(cut_config):
    # the cut calling getBDTscore_kfold, the cut before it, and the model JSON (the module's bdt_json)
    sel_name = os.path.splitext(os.path.basename(cut_config))[0]
    lib = importlib.import_module(sel_name)
    cuts = sorted([c for c in dir(lib) if re.fullmatch(r"cut\d+", c)], key=lambda x: int(x[3:]))
    bdt_cuts = [c for c in cuts if 'getBDTscore_kfold' in inspect.getsource(getattr(lib, c))]
    if len(bdt_cuts) != 1:
        raise ValueError(f'{cut_config}: expected exactly one cut calling getBDTscore_kfold, found {bdt_cuts}')
    bdt_cut = bdt_cuts[0]
    i = cuts.index(bdt_cut)
    if i == 0:
        raise ValueError(f'{cut_config}: the BDT cut {bdt_cut} is the first cut, nothing to skim before it')
    if bdt_cut != cuts[-1]:
        print(f'WARNING: {cuts[i+1:]} come after the BDT cut {bdt_cut} and are not included in the optimization')
    return sel_name, cuts[i-1], bdt_cut, os.path.normpath(lib.bdt_json)


def make_h5(sample_json, last_cut, fname, slim_json=None):
    from skimmerTools import Skimmer
    print(f'Skimming {sample_json} through {last_cut} -> {fname}')
    t1 = time.time()
    skim = Skimmer(sample_json, cut_config, last_cut=last_cut, slimFileList=slim_json, bdt_type=bdt_type)
    out = skim.process(execr='futures', workers=workers)
    bdt.write_h5(out[0], fname)
    print('Completed in {:.2f} min\n'.format((time.time()-t1)/60))


def load_models(vars_json):
    with open(vars_json) as fin:
        vars_dict = json.load(fin)
    if vars_dict.get('fold_rule') != 'eventNum % k':
        raise ValueError(f"{vars_json}: unsupported fold_rule {vars_dict.get('fold_rule')}")
    # relative model paths are relative to trainBDT/, as in getBDTscore_kfold
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(vars_json)))
    models = []
    for m in vars_dict['models']:
        bst = xgb.XGBClassifier()
        bst.load_model(m if os.path.isabs(m) else os.path.join(base_dir, m))
        models.append(bst)
    if len(models) != vars_dict['k']:
        raise ValueError(f"{vars_json}: k = {vars_dict['k']} but {len(models)} fold models are listed")
    return vars_dict, models


def load_scored(fname, vars_dict, models, is_sig):
    bdt.check_features([fname], vars_dict['variables'])
    with h5py.File(fname, 'r') as fin:
        X = np.concatenate([fin[v][()].reshape(-1, 1) for v in vars_dict['variables']], axis=1)
        out = {'wgt': fin['wgt_norm'][()].astype(float),
               'fold': bdt.get_fold(bdt.read_event_num(fin, fname), vars_dict['k'])}
        if is_sig:
            # rounded so float noise in the stored mass/splitting does not split one point in two
            out['point'] = np.stack([np.round(fin[k][()], 4) for k in ('m1', 'delta', 'ctau')], axis=1)
    out['score'] = bdt.predict_kfold(models, X, out['fold'])
    if np.any(out['score'] < 0):
        raise ValueError(f'{fname}: {np.sum(out["score"] < 0)} events were not scored (fold outside 0..k-1)')
    return out


def yields_above(score, wgt, grid):
    # weighted and raw counts with score > t for each t in grid
    order = np.argsort(score)
    s = score[order]
    cw = np.concatenate([[0.], np.cumsum(wgt[order])])
    idx = np.searchsorted(s, grid, side='right')
    return cw[-1] - cw[idx], len(s) - idx


def scan(sig, bkg, points, grid, folds, k):
    # S, B (scaled to the full sample) and S/sqrt(B) per point vs threshold, using only events in folds
    scale = k / len(folds)
    in_b = np.isin(bkg['fold'], folds)
    B, B_raw = yields_above(bkg['score'][in_b], bkg['wgt'][in_b], grid)
    B = B * scale
    valid = (B_raw >= min_bkg_raw) & (B > 0)
    S = np.zeros((len(points), len(grid)))
    for ip, p in enumerate(points):
        sel = np.isin(sig['fold'], folds) & np.all(sig['point'] == p, axis=1)
        S[ip] = yields_above(sig['score'][sel], sig['wgt'][sel], grid)[0] * scale
    fom = np.where(valid, S / np.sqrt(np.where(valid, B, 1.)), np.nan)
    return {'S': S, 'B': B, 'B_raw': B_raw, 'valid': valid, 'fom': fom}


def loosest_on_plateau(curve, grid):
    # smallest threshold whose value is within plateau_frac of the curve's maximum
    best = np.nanmax(curve)
    return grid[np.argmax(curve >= plateau_frac * best)] if np.isfinite(best) else np.nan


def at(arr, grid, t):
    # value of a scan curve (last axis = grid) at threshold t; t off the grid rounds up to the next grid point
    return arr[..., min(np.searchsorted(grid, t), len(grid) - 1)]


def md5(fname):
    h = hashlib.md5()
    with open(fname, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def make_plots(plot_dir, grid, points, res_sel, ratio, min_ratio, t_common, best_t, old_t, is_bench):
    os.makedirs(plot_dir, exist_ok=True)
    def vlines(ax):
        ax.axvline(t_common, color=ptools.CMS_COLORS_6[2], ls='--', label=f'optimized ({t_common:.3f})')  # CMS red
        ax.axvline(old_t, color='gray', ls=':', label=f'old medium WP ({old_t:.3f})')

    # background yield vs threshold, with the raw-count requirement
    fig, ax = ptools.new_fig(None)
    ax.plot(grid, res_sel['B'], label='B (xsec*lumi weighted)')
    ax.plot(grid, res_sel['B_raw'], label='B raw MC events')
    ax.axhline(min_bkg_raw, color='k', lw=0.8, label=f'{min_bkg_raw} raw events')
    vlines(ax)
    ax.set_yscale('log')
    ax.set_xlabel('BDT score threshold')
    ax.set_ylabel(f'Events above threshold (folds {sel_folds}, scaled)')
    ptools.place_legend(ax)
    fig.savefig(f'{plot_dir}/bkg_vs_threshold.png', bbox_inches='tight')
    plt.close(fig)

    # FOM / FOM_best per point and the minimax envelope
    fig, ax = ptools.new_fig(None)
    for ip in range(len(points)):
        ax.plot(grid, ratio[ip], color='C0' if is_bench[ip] else 'C1', lw=0.6, alpha=0.5)
    ax.plot(grid, min_ratio, color='k', lw=2, label='min over benchmark points')
    vlines(ax)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('BDT score threshold')
    ax.set_ylabel(r'$(S/\sqrt{B}) \,/\, (S/\sqrt{B})_{best}$ per signal point')
    ptools.place_legend(ax)
    fig.savefig(f'{plot_dir}/fom_ratio_vs_threshold.png', bbox_inches='tight')
    plt.close(fig)

    # best threshold per point
    deltas = np.unique(points[:, 1])
    fig, axes = plt.subplots(1, len(deltas), figsize=(6*len(deltas), 5), squeeze=False)
    for ax, d in zip(axes[0], deltas):
        m = points[:, 1] == d
        sc = ax.scatter(points[m, 0], points[m, 2], c=best_t[m], cmap='viridis', vmin=np.nanmin(best_t), vmax=np.nanmax(best_t), s=200)
        for x, y, v in zip(points[m, 0], points[m, 2], best_t[m]):
            ax.annotate(f'{v:.3f}', (x, y), textcoords='offset points', xytext=(0, 10), ha='center', fontsize=8)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('m1 (GeV)')
        ax.set_ylabel('ctau (mm)')
        ax.set_title(f'delta = {d:g}')
    fig.colorbar(sc, ax=axes[0].tolist(), label='best threshold (loosest on plateau)')
    ptools.add_cms_label(axes[0][0], None)
    fig.savefig(f'{plot_dir}/best_threshold_per_point.png', bbox_inches='tight')
    plt.close(fig)


def main():
    sel_name, pre_cut, bdt_cut, vars_json = find_bdt_cut(cut_config)
    print(f'{sel_name}: BDT cut {bdt_cut}, skimming through {pre_cut}, model {vars_json}')
    vars_dict, models = load_models(vars_json)
    k = vars_dict['k']
    if set(sel_folds) & set(eval_folds) or not set(sel_folds) | set(eval_folds) <= set(range(k)):
        raise ValueError(f'sel_folds {sel_folds} and eval_folds {eval_folds} must be disjoint subsets of 0..{k-1}')

    # 0. skim through the cut before the BDT cut
    sig_h5 = f"h5/{os.path.splitext(os.path.basename(signal_json))[0]}_thropt_{sel_name}_{pre_cut}.h5"
    bkg_h5 = f"h5/{os.path.splitext(os.path.basename(bkg_json))[0]}_thropt_{sel_name}_{pre_cut}.h5"
    for fname, sjson, slim in [(sig_h5, signal_json, None), (bkg_h5, bkg_json, bkg_slim_json)]:
        if remake_h5 or not os.path.exists(fname):
            make_h5(sjson, pre_cut, fname, slim)
        else:
            print(f'Reusing {fname}')
    sig = load_scored(sig_h5, vars_dict, models, is_sig=True)
    bkg = load_scored(bkg_h5, vars_dict, models, is_sig=False)
    print(f'Signal: {len(sig["wgt"])} events, bkg: {len(bkg["wgt"])} events (raw, after {pre_cut})')

    # 1-3. scan
    grid = np.round(np.arange(0., 1., thres_step), 6)
    points = np.unique(sig['point'], axis=0)
    if benchmark_points is None:
        is_bench = np.ones(len(points), dtype=bool)
    else:
        bench = np.round(np.array(benchmark_points, dtype=float), 4)
        is_bench = np.array([np.any(np.all(bench == p, axis=1)) for p in points])
        if is_bench.sum() != len(bench):
            raise ValueError(f'benchmark_points not all found among the signal points {points.tolist()}')
    res_sel = scan(sig, bkg, points, grid, sel_folds, k)
    res_eval = scan(sig, bkg, points, grid, eval_folds, k)
    res_all = scan(sig, bkg, points, grid, list(range(k)), k)
    if not res_sel['valid'].any():
        raise RuntimeError(f'No threshold has >= {min_bkg_raw} raw bkg events in folds {sel_folds}')
    t_max_valid = grid[res_sel['valid']].max()
    print(f'Thresholds up to {t_max_valid:.3f} keep >= {min_bkg_raw} raw bkg events in folds {sel_folds}')

    fom_best = np.nanmax(res_sel['fom'], axis=1)
    best_t = np.array([loosest_on_plateau(f, grid) for f in res_sel['fom']])
    usable = is_bench & (fom_best > 0)
    for p in points[is_bench & ~usable]:
        print(f'WARNING: point {p.tolist()} has no signal above any valid threshold; left out of the minimax')
    with np.errstate(invalid='ignore', divide='ignore'):
        ratio = res_sel['fom'] / fom_best[:, None]
    min_ratio = np.nanmin(np.where(res_sel['valid'], ratio[usable], np.nan), axis=0) if usable.any() else np.full(len(grid), np.nan)
    t_common = loosest_on_plateau(min_ratio, grid)
    t_argmax = grid[np.nanargmax(min_ratio)]
    old_t = vars_dict['thres_WP']['medium']
    print(f'\nCommon threshold: {t_common:.4f} (minimax argmax {t_argmax:.4f}, worst-point FOM/FOM_best = '
          f'{at(min_ratio, grid, t_common):.3f}); old medium WP: {old_t:.4f}')

    # 4. report: choice on sel folds, S and B quoted on eval folds
    B_e, Braw_e = at(res_eval['B'], grid, t_common), at(res_eval['B_raw'], grid, t_common)
    B_a, Braw_a = at(res_all['B'], grid, t_common), at(res_all['B_raw'], grid, t_common)
    old_B_e, old_Braw_e = at(res_eval['B'], grid, old_t), at(res_eval['B_raw'], grid, old_t)
    print(f'B at optimized threshold: eval folds {B_e:.3f} ({Braw_e} raw), all folds {B_a:.3f} ({Braw_a} raw)')
    print(f'B at old medium WP:       eval folds {old_B_e:.3f} ({old_Braw_e} raw)')
    if Braw_e < min_bkg_raw:
        print(f'WARNING: only {Braw_e} raw bkg events in eval folds at the optimized threshold')

    model_name = os.path.splitext(os.path.basename(vars_json))[0].replace('variables_', '')
    plot_dir = f'plots/{model_name}_thropt_{sel_name}'
    os.makedirs(plot_dir, exist_ok=True)
    header = ['m1', 'delta', 'ctau', 'benchmark', 'best_t', 'fom_best_sel', 'ratio_sel_at_opt',
              'S_eval_opt', 'fom_eval_opt', 'S_eval_old', 'fom_eval_old', 'S_all_opt', 'fom_all_opt']
    rows = []
    for ip, p in enumerate(points):
        rows.append([*p, int(is_bench[ip]), best_t[ip], fom_best[ip], at(ratio[ip], grid, t_common),
                     at(res_eval['S'][ip], grid, t_common), at(res_eval['fom'][ip], grid, t_common),
                     at(res_eval['S'][ip], grid, old_t), at(res_eval['fom'][ip], grid, old_t),
                     at(res_all['S'][ip], grid, t_common), at(res_all['fom'][ip], grid, t_common)])
    print('\n' + ' '.join(f'{h:>14}' for h in header))
    for r in rows:
        print(' '.join(f'{v:>14.4g}' for v in r))
    with open(f'{plot_dir}/scan_summary.csv', 'w') as fout:
        fout.write(','.join(header) + '\n')
        for r in rows:
            fout.write(','.join(f'{v:.6g}' for v in r) + '\n')
    np.savez(f'{plot_dir}/scan.npz', grid=grid, points=points, fom_sel=res_sel['fom'], fom_eval=res_eval['fom'],
             S_sel=res_sel['S'], B_sel=res_sel['B'], B_raw_sel=res_sel['B_raw'], min_ratio=min_ratio)
    make_plots(plot_dir, grid, points, res_sel, ratio, min_ratio, t_common, best_t, old_t, is_bench)
    print(f'\nPlots and tables in {plot_dir}/')

    # 5. store the threshold for the selection
    if write_json:
        with open(vars_json) as fin:
            vars_dict = json.load(fin) # re-read so only thres_opt changes
        vars_dict.setdefault('thres_opt', {})[sel_name] = {
            'value': float(t_common),
            'fom': 'S/sqrt(B), S and B weighted with wgt_norm (xsec*lumi)',
            'combination': 'loosest threshold with min over benchmark points of FOM/FOM_best >= plateau_frac * its max',
            'bdt_cut': bdt_cut, 'pre_bdt_cut': pre_cut,
            'min_bkg_raw': min_bkg_raw, 'thres_step': thres_step, 'plateau_frac': plateau_frac,
            'sel_folds': sel_folds, 'eval_folds': eval_folds,
            'benchmark_points': None if benchmark_points is None else [list(map(float, p)) for p in benchmark_points],
            'min_ratio': float(at(min_ratio, grid, t_common)),
            'B_eval': float(B_e), 'B_raw_eval': int(Braw_e), 'B_all': float(B_a), 'B_raw_all': int(Braw_a),
            'signal_h5': sig_h5, 'bkg_h5': bkg_h5, 'signal_h5_md5': md5(sig_h5), 'bkg_h5_md5': md5(bkg_h5),
            'date': time.strftime('%Y-%m-%d %H:%M'),
        }
        with open(vars_json, 'w') as fout:
            json.dump(vars_dict, fout, indent=4)
        print(f'Wrote thres_opt["{sel_name}"] = {t_common:.4f} to {vars_json}')


if __name__ == '__main__':
    main()
