import argparse
import contextlib
import io
import json
import os
import re
import h5py
import numpy as np
from BDTutils_kfold import reweight_bkg_arrays, build_sig_subprocess, reweight_signal, exclude_signal_points

# Prints a per-sample weight summary of a step1_makeh5.py output file. Background samples are identified by
# (xsec, sum_wgt) -- the h5 has no sample name column, and sum_wgt = wgt/wgt_norm is constant per sample.
# Signal samples are identified by (m1, delta, ctau).
#
# Columns:
#   N           raw number of events in the h5
#   wgt/evt     mean per-event weight wgt_norm = xsec*lumi*genWgt/sum_wgt ('*' if not uniform within the sample)
#   total wgt   sum of wgt_norm
#   xs*lumi     xsec*lumi (xsec as stored in the h5, i.e. including filter_eff for signal)
#   eff         weighted eff sum(genWgt)/sum_wgt, with genWgt = wgt/(xsec*lumi) and sum_wgt from --json
#               (as in the analysisTools cutflow); without a json match, sum_wgt = wgt/wgt_norm from the h5,
#               which reduces to total wgt/(xs*lumi)
#   xs*lumi*eff expected yield; should match total wgt when the json sum_wgt is the one the skim was normalized to
#   BDT w/evt   mean per-event BDT weight for the out-of-fold evaluation, computed with the BDTutils_kfold functions
#               exactly as in train_bdt (full sample, native-dtype wgt_norm, abs applied):
#                 bkg:    w+ * N_B/sum(w+), w+ = max(wgt_norm, 0), N_B = raw bkg count
#                 signal: sum(bkg w)/(n_pts*N_point), n_pts = number of signal points not excluded (needs --bkg-h5)
#               the training folds use the same functions on ~4/5 of the events, so per-event values are about the
#               same and totals about 4/5 of these
#   BDT total   sum of the BDT weights

# copy of analysisTools.getLumi (importing analysisTools pulls in coffea and is slow)
LUMI = {'2016': 16.8, '2016APV': 19.5, '2017': 41.48, '2018': 59.83, '2022': 38.01, '2024': 124.0}

CONFIG_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../configs/sample_configs'))

def find_json(h5_file):
    # step1 names the h5 f"{json basename}{suffix}{cut_tag}", so pick the longest config name prefixing it
    base = os.path.basename(h5_file)
    cands = [j[:-5] for j in os.listdir(CONFIG_DIR) if j.endswith('.json') and base.startswith(j[:-5] + '_')]
    if not cands:
        return None, None
    name = max(cands, key=len)
    # bkg skims are normalized against the all-events "-slim" companion (as in step1)
    slim = None
    if name.endswith('_skimmed'):
        slim_path = os.path.join(CONFIG_DIR, name[:-len('_skimmed')] + '-slim.json')
        slim = slim_path if os.path.exists(slim_path) else None
    return os.path.join(CONFIG_DIR, name + '.json'), slim

def load_json_info(json_file, slim_json_file=None):
    # mirrors the Skimmer: bkg samples take num_events/sum_wgt from their slim (all-events) counterpart
    with open(json_file) as f:
        samples = json.load(f)
    slim = {}
    if slim_json_file is not None:
        with open(slim_json_file) as f:
            slim = {s['name']: s for s in json.load(f)}
    info = []
    for s in samples:
        s = dict(s)
        if s['type'] == 'bkg' and s['name'] in slim:
            s['num_events'] = slim[s['name']]['num_events']
            s['sum_wgt'] = slim[s['name']]['sum_wgt']
        info.append(s)
    return info

def match_bkg(info, xsec, sum_wgt):
    cands = [s for s in info if s['type'] == 'bkg' and np.isclose(s['xsec'], xsec, rtol=1e-6)]
    if not cands:
        return None
    return min(cands, key=lambda s: abs(s['sum_wgt'] - sum_wgt))

def match_sig(info, m1, delta, ctau):
    for s in info:
        if s['type'] != 'signal':
            continue
        s_m1 = s['Mchi'] - s['dMchi']/2
        if np.isclose(s_m1, m1) and np.isclose(s['dMchi']/s_m1, delta) and np.isclose(float(s['ctau']), ctau):
            return s
    return None

def main():
    parser = argparse.ArgumentParser(description='Check per-sample weights in a step1_makeh5.py h5 file')
    parser.add_argument('h5_file')
    parser.add_argument('--json', default=None, help='sample config used to make the h5, for sample names and the sum_wgt in eff (default: found from the h5 name)')
    parser.add_argument('--slim-json', default=None, help='all-events bkg companion config, for bkg normalization as in step1 (default: found from --json)')
    parser.add_argument('--year', default=None, help='year for the lumi (default: parsed from the file name)')
    parser.add_argument('--lumi', type=float, default=None, help='lumi in /fb (overrides --year)')
    parser.add_argument('--bkg-h5', default=None, help='bkg h5 trained against, needed for the signal BDT weights')
    parser.add_argument('--exclude-points', default=None,
                        help='signal points excluded from training, as in step2 exclude_points (json, e.g. \'[{"m1": 0.05}]\')')
    args = parser.parse_args()
    exclude_points = json.loads(args.exclude_points) if args.exclude_points is not None else []

    if args.lumi is not None:
        lumi = args.lumi
    else:
        year = args.year
        if year is None:
            m = re.search(r'_(20\d\d(?:APV)?)_', os.path.basename(args.h5_file))
            if m is None:
                parser.error('could not parse the year from the file name; pass --year or --lumi')
            year = m.group(1)
        lumi = LUMI[str(year)]
    print(f'File: {args.h5_file}')
    print(f'Lumi: {lumi} /fb')

    if args.json is None:
        args.json, slim = find_json(args.h5_file)
        if args.slim_json is None:
            args.slim_json = slim
    print(f'Json: {args.json}' + (f' (slim: {args.slim_json})' if args.slim_json is not None else ''))
    info = load_json_info(args.json, args.slim_json) if args.json is not None else None

    with h5py.File(args.h5_file, 'r') as f:
        xsec = f['xsec'][()]
        wgt = f['wgt'][()].astype(np.float64)
        wgt_norm_raw = f['wgt_norm'][()] # native dtype, as read by BDTutils_kfold, for the BDT weights
        wgt_norm = wgt_norm_raw.astype(np.float64)
        isSignal = 'm1' in f
        if isSignal:
            m1, delta, ctau = f['m1'][()], f['delta'][()], f['ctau'][()]

    if isSignal:
        keys = np.stack([m1, delta, ctau], axis=1)
    else:
        # float32 wgt/wgt_norm smears sum_wgt, so within each xsec cluster it on relative gaps and
        # replace each event's value by its cluster's median
        with np.errstate(divide='ignore', invalid='ignore'):
            sum_wgt = np.where(wgt_norm != 0, wgt/wgt_norm, 0.0)
        for xs in np.unique(xsec):
            sel = np.where(xsec == xs)[0]
            order = sel[np.argsort(sum_wgt[sel])]
            vals = sum_wgt[order]
            breaks = np.where(np.diff(vals) > 1e-3*np.abs(vals[1:]))[0] + 1
            for grp in np.split(order, breaks):
                sum_wgt[grp] = np.median(sum_wgt[grp])
        keys = np.stack([xsec, sum_wgt], axis=1)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.reshape(-1)

    # per-event BDT weights (nan where not available), from the same calls as the out-of-fold evaluation weights
    # in BDTutils_kfold.train_bdt (their printout is suppressed)
    quiet = lambda: contextlib.redirect_stdout(io.StringIO())
    bdt_w = np.full(len(wgt_norm), np.nan)
    if not isSignal:
        with quiet():
            _, _, bkg_wgt, _ = reweight_bkg_arrays(wgt_norm_raw)
        bdt_w = np.abs(bkg_wgt).astype(np.float64)
    elif args.bkg_h5 is not None:
        with h5py.File(args.bkg_h5, 'r') as f:
            bkg_xsec_norm = f['wgt_norm'][()]
        # exclude_signal_points on the event indices gives the indices kept for training
        keep = np.arange(len(wgt_norm))
        sig_point = {'m1': m1, 'delta': delta, 'ctau': ctau}
        if exclude_points:
            keep, _, sig_point = exclude_signal_points(keep, wgt_norm_raw, sig_point, exclude_points)
        with quiet():
            bkg_nev_all, _, bkg_wgt, _ = reweight_bkg_arrays(bkg_xsec_norm)
            sig_subprocess_all, nSamp_all = build_sig_subprocess(sig_point)
            sig_wgt = reweight_signal(keep, bkg_nev_all, nSamp_all, bkg_wgt, sig_subprocess_all, bkg_xsec_norm)
        bdt_w[keep] = np.abs(sig_wgt)
        print(f'Bkg h5: {args.bkg_h5} (N_B = {bkg_nev_all}, sum of bkg BDT weights = {bkg_wgt.sum():.6g}), '
              f'{nSamp_all} signal points in training')

    rows = []
    for i, key in enumerate(uniq):
        idx = inv == i
        n = int(idx.sum())
        w = wgt_norm[idx]
        xs = xsec[idx][0]
        tot = w.sum()
        xsl = xs*lumi
        uniform = np.allclose(w, w[0], rtol=1e-4)

        if isSignal:
            label = f'm1={key[0]:g} delta={key[1]:g} ctau={key[2]:g}'
            s = match_sig(info, *key) if info is not None else None
        else:
            label = f'xsec={key[0]:g} sum_wgt={key[1]:.4g}'
            s = match_bkg(info, key[0], key[1]) if info is not None else None

        if s is not None:
            name = s['name']
            eff = (wgt[idx]/xsl).sum()/s['sum_wgt']
        else:
            name = '-' if info is None else '(no json match)'
            eff = tot/xsl
        bw = bdt_w[idx]
        rows.append((name, label, n, w.mean(), uniform, tot, xsl, eff, xsl*eff, bw.mean(), bw.sum()))

    fmt = lambda x, width: f'{"-":>{width}}' if np.isnan(x) else f'{x:>{width}.4g}'
    nw = max(len('name'), max(len(r[0]) for r in rows))
    lw = max(len('sample'), max(len(r[1]) for r in rows))
    header = (f"{'name':<{nw}} {'sample':<{lw}} {'N':>8} {'wgt/evt':>11} {'total wgt':>11} {'xs*lumi':>11} {'eff':>11} {'xs*lumi*eff':>12}"
              f" {'BDT w/evt':>11} {'BDT total':>11}")
    print(header)
    print('-'*len(header))
    for name, label, n, wpe, uniform, tot, xsl, eff, yld, bpe, btot in rows:
        flag = ' ' if uniform else '*'
        print(f'{name:<{nw}} {label:<{lw}} {n:>8d} {wpe:>10.4g}{flag} {tot:>11.4g} {xsl:>11.4g} {eff:>11.4g} {yld:>12.4g}'
              f' {fmt(bpe, 11)} {fmt(btot, 11)}')
    print('-'*len(header))
    bdt_tot = np.nan if np.all(np.isnan(bdt_w)) else np.nansum(bdt_w)
    print(f"{'total':<{nw}} {'':<{lw}} {len(wgt_norm):>8d} {'':>11} {wgt_norm.sum():>11.4g} {'':>11} {'':>11} {'':>12} {'':>11} {fmt(bdt_tot, 11)}")
    if any(not r[4] for r in rows):
        print('* per-event weight not uniform within the sample (genWgt varies); mean shown')
    if info is None:
        print('eff = total wgt/(xs*lumi) (pass --json for sum(genWgt)/sum_wgt with the json sum_wgt)')
    else:
        print('eff = sum(genWgt)/sum_wgt, sum_wgt from the json')
    if isSignal and args.bkg_h5 is None:
        print('BDT weights for signal need --bkg-h5')
    elif isSignal and exclude_points:
        print(f'BDT weight "-" = excluded by {exclude_points}')

if __name__ == '__main__':
    main()
