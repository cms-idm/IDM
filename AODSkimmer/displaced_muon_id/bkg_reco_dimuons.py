#!/usr/bin/env python3
"""Summarize recoOnly PreselectedVertexing outputs on 2022EE background MC (kb/studies/bkg-reco-dimuons.md).

One ROOT file per background sample, produced with recoOnly=True. Same cumulative cut chain as
reco_pair_recovery.py (pair_stage); raw unweighted counts, no gen truth. Each pair's stage is also
recomputed with an independent vectorized mask and must agree exactly. Writes ROOT counts, passing-pair
category and Lxy histograms, event and passing-pair trees; prints per-sample/group/overall tables.
Requires CMSSW PyROOT, uproot, awkward and numpy on LPC. Never overwrites an output.
"""
import argparse
from array import array
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from reco_pair_recovery import STAGES, pair_stage

# pCat encoding of PreselectedVertexing.cc categoryOf: 0 GG, 1 II, 2 OO, 3 GO, 4 GI, 5 OI (O = DSA).
CATEGORIES = ('G-G', 'I-I', 'DSA-DSA', 'G-DSA', 'G-I', 'I-DSA')
LXY_EDGES = [0., 0.1, 0.5, 1., 5., 10., 50., 740.]  # cm, pLxyBS
PAIR_BRANCHES = ['pEvent', 'pObjA', 'pObjB', 'pQA', 'pQB', 'pStatus', 'pChi2', 'pNdof', 'pPcaOK', 'pPcaDR',
                 'pPcaMuMuPt', 'pPcaMuMuPhi', 'pCat', 'pTypeA', 'pTypeB', 'pLxyBS', 'pMass']


def vector_stages(p, met_pt, met_phi):
    """Independent re-implementation of pair_stage over flat numpy arrays; returns 0..6 per pair.
    Float branches are promoted to float64 so cut boundaries match pair_stage on Python floats."""
    p = {k: (v.astype(np.float64) if v.dtype.kind == 'f' else v) for k, v in p.items()}
    mpt = met_pt.astype(np.float64)[p['pEvent']]
    mphi = met_phi.astype(np.float64)[p['pEvent']]
    with np.errstate(divide='ignore', invalid='ignore'):
        chi = (np.isfinite(p['pChi2']) & np.isfinite(p['pNdof']) & (p['pChi2'] >= 0) & (p['pNdof'] > 0) &
               (p['pChi2'] / p['pNdof'] < 4))
        angle_ok = (np.isfinite(mpt) & (mpt > 0) & np.isfinite(mphi) & np.isfinite(p['pPcaMuMuPt']) &
                    (p['pPcaMuMuPt'] > 0) & np.isfinite(p['pPcaMuMuPhi']))
        d = mphi - p['pPcaMuMuPhi']
        dphi = np.abs(d - 2 * np.pi * np.round(d / (2 * np.pi)))
    cuts = [p['pQA'] * p['pQB'] < 0, p['pStatus'] == 3, chi, p['pPcaOK'].astype(bool),
            np.isfinite(p['pPcaDR']) & (p['pPcaDR'] >= 0) & (p['pPcaDR'] < .9), angle_ok & (dphi < .5)]
    passed = np.ones(len(p['pEvent']), dtype=bool)
    stage = np.zeros(len(p['pEvent']), dtype=int)
    for c in cuts:
        passed &= c
        stage += passed
    return stage


def read_file_events(path):
    """Trusted full-file event counts: lines '<file> <events>' (from edmFileUtil before processing)."""
    canon = lambda s: s[s.find('/store/'):]
    counts = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            name, n = line.split()
            counts[canon(name)] = int(n)
    return counts


def check_coverage(report_dir, flist_dir, sample, n_events, file_events):
    """FrameworkJobReport: no error, exactly the original file list, every event of every file read."""
    report = ET.parse(report_dir / (sample + '.xml')).getroot()
    if [e for e in report.findall('.//FrameworkError') if e.get('ExitStatus') != '0']:
        raise RuntimeError('FrameworkJobReport reports a processing error: ' + sample)
    flist = flist_dir / (sample + '.txt')
    originals = [l.strip() for l in flist.read_text().splitlines() if l.strip() and not l.lstrip().startswith('#')]
    canon = lambda s: s[s.find('/store/'):]
    observed = [(canon(f.findtext('PFN', '')), int(f.findtext('EventsRead', '-1'))) for f in report.findall('.//InputFile')]
    if (len(observed) != len(originals) or set(p for p, n in observed) != set(map(canon, originals))
            or any(n <= 0 or n != file_events.get(p) for p, n in observed) or sum(n for p, n in observed) != n_events):
        raise RuntimeError('Per-file coverage/events mismatch: ' + sample)
    print('COVERAGE %s: %d original files, %d events verified' % (sample, len(originals), n_events), flush=True)


def summarize(stages, pairs, n_elig):
    """Event-level counters from this event's pair stages; pairs are dicts with pObjA/pObjB."""
    out = Counter(events=1, pairs=len(stages))
    last = max(stages, default=-1)
    out['last_' + ('no_eligible_pair' if last < 0 else STAGES[last])] = 1
    for i in range(last + 1):
        out['events_' + STAGES[i]] = 1
    for s in stages:
        for i in range(s + 1):
            out['candidates_' + STAGES[i]] += 1
    passing = [p for s, p in zip(stages, pairs) if s == 6]
    n = len(passing)
    disjoint = any({a['pObjA'], a['pObjB']}.isdisjoint({b['pObjA'], b['pObjB']})
                   for i, a in enumerate(passing) for b in passing[i + 1:])
    out['selected_events'] = int(n > 0)
    out['one_passing'] = int(n == 1)
    out['multiple_passing'] = int(n > 1)
    out['multiple_with_disjoint_pair_pair'] = int(disjoint)
    out['multiple_all_overlapping'] = int(n > 1 and not disjoint)
    for p in passing:
        out['passing_' + CATEGORIES[p['pCat']]] += 1
        lxy = p['pLxyBS']
        out['passing_lxy_' + ('valid' if math.isfinite(lxy) and lxy >= 0 else 'invalid_or_sentinel')] += 1
    assert out['events_eligible_pair'] == int(n_elig >= 2)
    assert sum(out['passing_' + c] for c in CATEGORIES) == out['candidates_met_angle'] == n
    return out, last, n, disjoint


def ratio(n, d):
    return 'undefined (0/0)' if not d else '%.4f%% (%d/%d)' % (100 * n / d, n, d)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True, help='producer outputs <sample>.root')
    parser.add_argument('--report-dir', required=True, help='cmsRun FrameworkJobReports <sample>.xml')
    parser.add_argument('--flist-dir', required=True, help='original MiniAOD file lists <sample>.txt')
    parser.add_argument('--file-events', required=True, help="'<file> <events>' per input file, full-file counts")
    parser.add_argument('--sample-json', required=True, help='bkg_2022.json; top-level keys are the process groups')
    parser.add_argument('--output', required=True)
    parser.add_argument('--expected-samples', type=int, default=50)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    import awkward as ak
    import uproot
    import ROOT
    process_groups = set(json.loads(Path(args.sample_json).read_text()))
    file_events = read_file_events(args.file_events)
    paths = sorted(Path(args.input_dir).glob('*.root'))
    if len(paths) != args.expected_samples:
        raise RuntimeError('Expected %d sample files, found %d' % (args.expected_samples, len(paths)))
    groups = defaultdict(Counter)
    lxy = defaultdict(list)
    revisions, seen, sources, event_rows, pair_rows = set(), set(), [], [], []
    for sample_index, path in enumerate(paths):
        with uproot.open(path) as source:
            prefix = 'preselectedVertexing/'
            name = str(source[prefix + 'sample'].member('fTitle'))
            group = name.split('_')[0]
            if name != path.stem or name in seen or group not in process_groups:
                raise RuntimeError('Invalid/duplicate sample or unknown process group: ' + name)
            seen.add(name)
            revisions.add(str(source[prefix + 'revision'].member('fTitle')))
            if str(source[prefix + 'recoOnly'].member('fTitle')) == 'disabled':
                raise RuntimeError('Not a reco-only run: ' + name)
            if str(source[prefix + 'hierarchyOrder'].member('fTitle')) != 'global>inner>outer(DSA)':
                raise RuntimeError('Unexpected hierarchy: ' + name)
            events = source[prefix + 'events'].arrays(['run', 'lumi', 'event', 'nElig', 'anyNPairs', 'pfMetPt',
                                                      'pfMetPhi'], library='np')
            n_events = len(events['run'])
            check_coverage(Path(args.report_dir), Path(args.flist_dir), name, n_events, file_events)
            pairs = source[prefix + 'pairs'].arrays(PAIR_BRANCHES, library='np')
        if np.any((pairs['pEvent'] < 0) | (pairs['pEvent'] >= n_events)):
            raise RuntimeError('Invalid pair event index: ' + name)
        if np.any((pairs['pCat'] < 0) | (pairs['pCat'] >= len(CATEGORIES))):
            raise RuntimeError('Invalid pair category: ' + name)
        vstage = vector_stages(pairs, events['pfMetPt'], events['pfMetPhi'])
        records = [dict(zip(PAIR_BRANCHES, values)) for values in zip(*(pairs[b].tolist() for b in PAIR_BRANCHES))]
        by_event = defaultdict(list)
        for k, p in enumerate(records):
            s = pair_stage(p, float(events['pfMetPt'][p['pEvent']]), float(events['pfMetPhi'][p['pEvent']]))
            if s != vstage[k]:
                raise RuntimeError('Stage mismatch between pair_stage and vector mask: %s pair %d' % (name, k))
            by_event[p['pEvent']].append((s, p))
        ids = set(zip(events['run'].tolist(), events['lumi'].tolist(), events['event'].tolist()))
        sample_counts = Counter(duplicate_event_ids=n_events - len(ids))
        for row in range(n_events):
            epairs = by_event[row]
            n_elig = int(events['nElig'][row])
            identities = [(p['pObjA'], p['pObjB']) for s, p in epairs]
            if (len(epairs) != events['anyNPairs'][row] or len(epairs) != n_elig * (n_elig - 1) // 2
                    or len(set(map(frozenset, identities))) != len(identities) or any(a == b for a, b in identities)):
                raise RuntimeError('Incomplete/repeated/self pair enumeration: %s event row %d' % (name, row))
            c, last, n_passing, disjoint = summarize([s for s, p in epairs], [p for s, p in epairs], n_elig)
            sample_counts.update(c)
            event_rows.append((sample_index, int(events['run'][row]), int(events['lumi'][row]), int(events['event'][row]),
                               n_elig, len(epairs), last, n_passing, int(disjoint)))
            for s, p in epairs:
                if s == 6:
                    pair_rows.append((sample_index, row, p['pCat'], p['pLxyBS'], p['pMass'], p['pPcaDR'],
                                      p['pPcaMuMuPt'], p['pChi2'] / p['pNdof']))
                    if math.isfinite(p['pLxyBS']) and p['pLxyBS'] >= 0:
                        for g in ('overall', 'group/' + group, 'sample/' + name):
                            lxy[g].append(p['pLxyBS'])
        assert sum(v for k, v in sample_counts.items() if k.startswith('last_')) == n_events
        for g in ('overall', 'group/' + group, 'sample/' + name):
            groups[g].update(sample_counts)
        sources.append((name, str(path.resolve())))
    if len(revisions) != 1 or 'unspecified' in revisions:
        raise RuntimeError('Mixed/unspecified code revisions')

    root = ROOT.TFile.Open(str(output), 'CREATE')
    if not root or root.IsZombie():
        raise RuntimeError('Cannot create output')
    ROOT.TNamed('revision', next(iter(revisions))).Write()
    ROOT.TNamed('definitions', 'Raw unweighted 2022EE background MC, no gen truth; cleanedDisplacedMuons; fixed G>I>DSA '
                'eligible leg and charge, no fallback; all unordered eligible-object pairs; cumulative cuts OS, valid '
                'KalmanVertexFitter, chi2/ndof<4, PCA valid, PCA dR<0.9, abs(dphi(Type1 PF MET, mumu PCA))<0.5; no trigger. '
                'events_<stage>: events with >=1 pair reaching that stage; last_<stage>: partition by furthest stage. '
                'Category and Lxy (pLxyBS, cm, vertex to beamspot at vertex z) are per passing pair; group/overall '
                'sums are raw bookkeeping over different cross sections, not a physical mix.').Write()
    ROOT.TNamed('inputs', '\n'.join(n + ' ' + p for n, p in sources)).Write()
    tree = ROOT.TTree('events', 'Event accounting; sample index follows inputs metadata; lastStage -1 = no eligible pair')
    fields = ('sample', 'run', 'lumi', 'event', 'nElig', 'nPairs', 'lastStage', 'nPassing', 'disjoint')
    buffers = {k: array('q', [0]) for k in fields}
    for k, b in buffers.items():
        tree.Branch(k, b, k + '/L')
    for r in event_rows:
        for k, v in zip(fields, r):
            buffers[k][0] = v
        tree.Fill()
    tree.Write()
    ptree = ROOT.TTree('passingPairs', 'Passing pairs; eventRow = entry of the sample in producer events')
    pfields = ('sample', 'eventRow', 'cat')
    pbuf = {k: array('q', [0]) for k in pfields}
    dfields = ('lxyBS', 'mass', 'pcaDR', 'pcaMuMuPt', 'chi2ndof')
    dbuf = {k: array('d', [0]) for k in dfields}
    for k, b in pbuf.items():
        ptree.Branch(k, b, k + '/L')
    for k, b in dbuf.items():
        ptree.Branch(k, b, k + '/D')
    for r in pair_rows:
        for k, v in zip(pfields + dfields, r):
            (pbuf if k in pbuf else dbuf)[k][0] = v
        ptree.Fill()
    ptree.Write()
    for g, counts in sorted(groups.items()):
        directory = root
        for part in g.split('/'):
            directory = directory.GetDirectory(part) or directory.mkdir(part)
        directory.cd()
        keys = sorted(counts)
        h = ROOT.TH1D('counts', 'Raw counts;Counter;Count', len(keys), 0, len(keys))
        for i, k in enumerate(keys, 1):
            h.GetXaxis().SetBinLabel(i, k)
            h.SetBinContent(i, counts[k])
        h.Write()
        hc = ROOT.TH1D('passing_category', 'Passing pairs by leg type;Leg-type category;Pairs', len(CATEGORIES), 0,
                       len(CATEGORIES))
        for i, c in enumerate(CATEGORIES, 1):
            hc.GetXaxis().SetBinLabel(i, c)
            hc.SetBinContent(i, counts['passing_' + c])
        hc.Write()
        hl = ROOT.TH1D('passing_lxy', 'Passing pairs;Vertex L_{xy} (beamspot) [cm];Pairs', len(LXY_EDGES) - 1,
                       array('d', LXY_EDGES))
        for v in lxy[g]:
            hl.Fill(v)
        if hl.GetEntries() != counts['passing_lxy_valid']:
            raise RuntimeError('Lxy histogram total mismatch: ' + g)
        hl.Write()
    root.Close()

    print('\n%-48s %8s %8s  %s' % ('group', 'events', 'pairs', 'events with >=1 pair reaching: ' + ', '.join(STAGES)))
    for g, c in sorted(groups.items(), key=lambda x: (x[0] != 'overall', x[0])):
        print('%-48s %8d %8d  %s' % (g, c['events'], c['pairs'],
                                     ' '.join(str(c['events_' + s]) for s in STAGES)))
    print('\nSelected events (>=1 pair passing all stages) / all events:')
    for g, c in sorted(groups.items(), key=lambda x: (x[0] != 'overall', x[0])):
        print('  %-46s %s  passing pairs %d  multiple %d  disjoint %d  dup ids %d  categories %s' % (
            g, ratio(c['selected_events'], c['events']), c['candidates_met_angle'], c['multiple_passing'],
            c['multiple_with_disjoint_pair_pair'], c['duplicate_event_ids'],
            ' '.join('%s=%d' % (k, c['passing_' + k]) for k in CATEGORIES)))


if __name__ == '__main__':
    main()
