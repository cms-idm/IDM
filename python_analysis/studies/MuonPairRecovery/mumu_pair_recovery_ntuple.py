#!/usr/bin/env python3
"""Reco-pair-recovery from the ntuple through the shared Analyzer (procType='mumu'), and its decisive test.

kb/studies/coffea-ntuple-content.md, test 3. Subcommands:
  run      Analyzer(sample config).process(procType='mumu') over ntuples/outT. Every input file must be fully
           processed (the shared Runner skips bad files, so coverage is checked here and a gap is an error).
           Writes a new ROOT file (never overwrites): counters per group, per-event records, per-pair identities.
  compare  against the reference reco-pair-recovery output (recovery.root: every counter of every group, every
           per-event (T, nPassing, nCorrect, disjoint) under unique (sample, run, lumi, event) keys) and against
           the producer's preselectedVertexing pairs (per event, the ordered pairs (raw A, raw B) with status, chi2
           and ndof, all exactly equal).
The sample config is a list in the Analyzer format (type 'signal', year, Mchi, dMchi, ctau, location or fileset)
with one extra key per sample, 'reference_name' (the sample name used by recovery.root, e.g.
Mchi-5p25_dMchi-0p5_ctau-1), which sets the grouping: sample, mass (pooled over lifetimes), lifetime (pooled over
masses), overall; unweighted pooling, raw counts.
"""
import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # python_analysis/
from analysisTools.analysisTools import Analyzer  # noqa: E402
from analysisTools.muonProcessors import RECORD_FIELDS, PAIR_RECORD_FIELDS  # noqa: E402

SAMPLE_RE = re.compile(r'(Mchi-[\dp]+_dMchi-[\dp]+)_(ctau-[\dp]+)')
TREE = 'ntuples/outT'


def run(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    with open(args.samples) as f:
        config = json.load(f)
    if args.only:
        config = [s for s in config if s['reference_name'] in set(args.only)]
        missing = set(args.only) - {s['reference_name'] for s in config}
        if missing:
            raise RuntimeError('Requested samples not in config: %s' % sorted(missing))
    elif len(config) != args.expected_samples:
        raise RuntimeError('Expected %d samples, config has %d' % (args.expected_samples, len(config)))
    for s in config:
        if s.get('type') != 'signal' or not SAMPLE_RE.fullmatch(s['reference_name']):
            raise RuntimeError('Bad sample entry: %s' % s.get('reference_name'))
    ana = Analyzer(config, 'none', None)
    by_dataset = {name: ana.sample_info[name]['reference_name'] for name in ana.sample_names}
    if len(set(by_dataset.values())) != len(by_dataset):
        raise RuntimeError('reference_name not unique')
    expected = {}  # filename -> entries
    for name in ana.sample_names:
        files = ana.sample_locs[name]
        if not files:
            raise RuntimeError('No files for ' + name)
        for fn in files:
            with uproot.open(fn) as f:
                expected[(name, fn)] = f[TREE].num_entries
    start = time.time()
    acc = ana.process(treename=TREE, execr='futures' if args.workers > 1 else 'iterative', workers=args.workers,
                      procType='mumu')
    elapsed = time.time() - start
    acc = acc[0] if isinstance(acc, tuple) else acc
    seen = Counter()
    for (dataset, fn, _start), chunk in acc.items():
        seen[(dataset, fn)] += chunk['counts']['events']
    if seen != Counter(expected):
        gaps = sorted(k for k in set(expected) | set(seen) if seen.get(k, 0) != expected.get(k, 0))
        raise RuntimeError('Input coverage incomplete (%d files): %s' % (len(gaps), gaps[:10]))

    groups = defaultdict(Counter)
    cols = defaultdict(list)
    pcols = defaultdict(list)
    names = sorted(set(by_dataset.values()))
    index = {n: i for i, n in enumerate(names)}
    for key in sorted(acc, key=lambda k: (k[0], k[1], k[2])):
        chunk = acc[key]
        ref = by_dataset[key[0]]
        m = SAMPLE_RE.fullmatch(ref)
        for g in ('overall', 'sample/' + ref, 'mass/' + m[1], 'lifetime/' + m[2]):
            groups[g].update(chunk['counts'])
        n = len(chunk['records']['run'])
        cols['sample'].append(np.full(n, index[ref], dtype=np.int64))
        for k in RECORD_FIELDS:
            cols[k].append(np.asarray(chunk['records'][k], dtype=np.int64))
        npair = len(chunk['pairs']['run'])
        pcols['sample'].append(np.full(npair, index[ref], dtype=np.int64))
        for k in PAIR_RECORD_FIELDS:
            pcols[k].append(np.asarray(chunk['pairs'][k]))
    counters = sorted({k for c in groups.values() for k in c})
    group_names = sorted(groups)
    gi, ci, val = [], [], []
    for i, g in enumerate(group_names):
        for j, k in enumerate(counters):
            gi.append(i); ci.append(j); val.append(groups[g][k])
    with uproot.recreate(output) as f:
        f['inputs'] = '\n'.join(by_dataset[name] + ' ' + fn for (name, fn) in sorted(expected))
        f['samples'] = '\n'.join(names)
        f['groups'] = '\n'.join(group_names)
        f['counters'] = '\n'.join(counters)
        f['definitions'] = ('reco_pair_recovery from the ntuple (mumuvtx) via Analyzer procType=mumu; raw unweighted; '
                            'kb/studies/coffea-ntuple-content.md. counts: group, counter, value; events and pairs: '
                            'sample index follows samples')
        f['counts'] = {'group': np.array(gi, dtype=np.int64), 'counter': np.array(ci, dtype=np.int64),
                       'value': np.array(val, dtype=np.int64)}
        f['events'] = {k: np.concatenate(v) for k, v in cols.items()}
        f['pairs'] = {k: np.concatenate(v) for k, v in pcols.items()}
    c = groups['overall']
    print('samples %d files %d events %d wall %.1f s' % (len(names), len(expected), c['events'], elapsed))
    print('Candidate correctness: %d/%d' % (c['correct_pairs'], c['passing_pairs']))
    print('Truth event recovery: %d/%d' % (c['recovered_events'], c['truth_accepted']))


def read_reference_counts(path):
    ref = {}
    with uproot.open(path) as f:
        for key, cls in f.classnames().items():
            if cls == 'TH1D' and key.split(';')[0].endswith('/counts'):
                h = f[key]
                ref[key.split(';')[0][:-len('/counts')]] = dict(zip(h.axis().labels(), (int(v) for v in h.values())))
    return ref


def producer_pairs(producer_dir, names):
    """(sample, run, lumi, event) -> tuple of (pObjA, pObjB, pStatus, pChi2, pNdof) in producer order."""
    out = {}
    for name in names:
        path = Path(producer_dir) / (name + '.root')
        with uproot.open(path) as f:
            ev = f['preselectedVertexing/events'].arrays(['run', 'lumi', 'event'], library='np')
            pr = f['preselectedVertexing/pairs'].arrays(['pEvent', 'pObjA', 'pObjB', 'pStatus', 'pChi2', 'pNdof'],
                                                        library='np')
        rows = defaultdict(list)
        for i in range(len(pr['pEvent'])):
            rows[int(pr['pEvent'][i])].append((int(pr['pObjA'][i]), int(pr['pObjB'][i]), int(pr['pStatus'][i]),
                                               np.float32(pr['pChi2'][i]), np.float32(pr['pNdof'][i])))
        for r in range(len(ev['run'])):
            key = (name, int(ev['run'][r]), int(ev['lumi'][r]), int(ev['event'][r]))
            if key in out:
                raise RuntimeError('Duplicate producer event key %s' % (key,))
            out[key] = tuple(rows.get(r, ()))
    return out


def compare(args):
    with uproot.open(args.ntuple_result) as f:
        groups = str(f['groups']).split('\n')
        counters = str(f['counters']).split('\n')
        names = str(f['samples']).split('\n')
        t = f['counts'].arrays(library='np')
        pe = f['events'].arrays(library='np')
        pp = f['pairs'].arrays(library='np')
    mine = defaultdict(dict)
    for g, c, v in zip(t['group'], t['counter'], t['value']):
        mine[groups[g]][counters[c]] = int(v)
    ref = read_reference_counts(args.reference)
    with uproot.open(args.reference) as f:
        ref_names = [line.split(' ')[0] for line in str(f['inputs'].member('fTitle')).split('\n')]
        re_ = f['events'].arrays(library='np')
    keep = set(names)
    if not keep <= set(ref_names):
        raise RuntimeError('Samples missing from reference: %s' % sorted(keep - set(ref_names)))
    if keep != set(ref_names):  # subset run: per-sample groups only
        ref = {g: c for g, c in ref.items() if g.startswith('sample/') and g[len('sample/'):] in keep}
        mine = {g: c for g, c in mine.items() if g.startswith('sample/')}
    failures = []
    if set(ref) != set(mine):
        failures.append('group sets differ: only ref %s, only ntuple %s' % (sorted(set(ref) - set(mine)),
                                                                          sorted(set(mine) - set(ref))))
    n_counter = 0
    for g in sorted(set(ref) & set(mine)):
        missing = set(ref[g]) - set(mine[g])
        if missing:
            failures.append('%s: reference counters not computed: %s' % (g, sorted(missing)))
        r = {k: v for k, v in ref[g].items() if v}
        p = {k: v for k, v in mine[g].items() if v}
        for k in sorted(set(r) | set(p)):
            n_counter += 1
            if r.get(k, 0) != p.get(k, 0):
                failures.append('%s %s: ref %d ntuple %d' % (g, k, r.get(k, 0), p.get(k, 0)))

    def keyed(arrays, sample_names):
        out = {}
        for i in range(len(arrays['run'])):
            name = sample_names[arrays['sample'][i]]
            if name not in keep:
                continue
            key = (name, int(arrays['run'][i]), int(arrays['lumi'][i]), int(arrays['event'][i]))
            if key in out:
                raise RuntimeError('Duplicate event key %s' % (key,))
            out[key] = tuple(int(arrays[k][i]) for k in ('T', 'nPassing', 'nCorrect', 'disjoint'))
        return out
    rk, mk = keyed(re_, ref_names), keyed(pe, names)
    if set(rk) != set(mk):
        failures.append('event key sets differ: only ref %d, only ntuple %d' % (len(set(rk) - set(mk)),
                                                                              len(set(mk) - set(rk))))
    diff = [k for k in set(rk) & set(mk) if rk[k] != mk[k]]
    for k in sorted(diff)[:20]:
        failures.append('event %s: ref %s ntuple %s (T,nPassing,nCorrect,disjoint)' % (k, rk[k], mk[k]))

    # Pair identity: same pairs, same order, same raw objects, status, chi2, ndof (float32, exact).
    prod = producer_pairs(args.producer_dir, sorted(keep))
    mine_pairs = defaultdict(list)
    for i in range(len(pp['run'])):
        key = (names[pp['sample'][i]], int(pp['run'][i]), int(pp['lumi'][i]), int(pp['event'][i]))
        mine_pairs[key].append((int(pp['rawA'][i]), int(pp['rawB'][i]), int(pp['status'][i]),
                                np.float32(pp['chi2'][i]), np.float32(pp['ndof'][i])))
    if set(prod) != set(mk):
        failures.append('producer event keys differ from ntuple event keys')
    pair_diff = [k for k in set(prod) & set(mk) if tuple(mine_pairs.get(k, ())) != prod[k]]
    for k in sorted(pair_diff)[:20]:
        failures.append('pairs %s: producer %s ntuple %s' % (k, prod[k][:3], tuple(mine_pairs.get(k, ()))[:3]))
    n_pairs = sum(len(v) for v in prod.values())
    print('compared %d groups, %d nonzero counters, %d events (%d differing), %d producer pairs in %d events '
          '(%d events differing)' % (len(set(ref) & set(mine)), n_counter, len(set(rk) & set(mk)), len(diff),
                                     n_pairs, len(prod), len(pair_diff)))
    if 'overall' in mine:
        c = mine['overall']
        print('overall ntuple: recovery %d/%d, correctness %d/%d, selected %d, disjoint %d' %
              (c['recovered_events'], c['truth_accepted'], c['correct_pairs'], c['passing_pairs'],
               c['selected_events'], c['multiple_with_disjoint_pair_pair']))
    if failures:
        print('FAIL (%d differences)' % len(failures))
        for line in failures[:50]:
            print(' ', line)
        return 1
    print('PASS: identical')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--samples', required=True, help='sample config JSON (Analyzer format + reference_name)')
    r.add_argument('--output', required=True)
    r.add_argument('--only', nargs='*', help='subset of reference sample names (test runs)')
    r.add_argument('--expected-samples', type=int, default=72)
    r.add_argument('--workers', type=int, default=1)
    c = sub.add_parser('compare')
    c.add_argument('--ntuple-result', required=True)
    c.add_argument('--reference', required=True, help='recovery.root of the reference reco-pair-recovery run')
    c.add_argument('--producer-dir', required=True, help='directory of the producer <sample>.root files')
    args = parser.parse_args()
    if args.cmd == 'run':
        run(args)
        return 0
    return compare(args)


if __name__ == '__main__':
    sys.exit(main())
