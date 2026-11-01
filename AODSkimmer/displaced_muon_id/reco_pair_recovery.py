#!/usr/bin/env python3
"""Summarize recoOnly PreselectedVertexing outputs; raw counts, no truth-based reco cuts.

One ROOT file per sample, produced with recoOnly=True. Writes ROOT numerical counts
and event accounting; prints per-sample/mass/lifetime/overall summaries to stdout.
Requires CMSSW PyROOT, uproot and awkward on LPC. Never overwrites an output.
"""
import argparse
from array import array
from collections import Counter, defaultdict
from pathlib import Path
import math
import re

STAGES = ('eligible_pair', 'opposite_sign', 'valid_vertex', 'chi2', 'pca_valid', 'dr', 'met_angle')


def pair_stage(p, met_pt, met_phi):
    """Last passed cumulative stage; eligibility is guaranteed by producer enumeration."""
    if p['pQA'] * p['pQB'] >= 0:
        return 0
    if p['pStatus'] != 3:
        return 1
    if not (math.isfinite(p['pChi2']) and math.isfinite(p['pNdof']) and
            p['pChi2'] >= 0 and p['pNdof'] > 0 and p['pChi2'] / p['pNdof'] < 4):
        return 2
    if not p['pPcaOK']:
        return 3
    if not (math.isfinite(p['pPcaDR']) and 0 <= p['pPcaDR'] < .9):
        return 4
    valid_angle = (math.isfinite(met_pt) and met_pt > 0 and math.isfinite(met_phi) and
                   math.isfinite(p['pPcaMuMuPt']) and p['pPcaMuMuPt'] > 0 and
                   math.isfinite(p['pPcaMuMuPhi']))
    if not valid_angle:
        return 5
    dphi = abs(math.remainder(met_phi - p['pPcaMuMuPhi'], 2 * math.pi))
    return 6 if dphi < .5 else 5


def summarize(event, pairs):
    t = bool(event['twoAcc'])
    out = Counter(events=1, truth_accepted=int(t),
                  has_signal_pair=int(event['nSig'] == 2 and event['sigQ'][0] * event['sigQ'][1] < 0))
    best_correct = -1
    passing = []
    any_stage = -1
    for p in pairs:
        stage = pair_stage(p, event['pfMetPt'], event['pfMetPhi'])
        any_stage = max(any_stage, stage)
        correct = t and bool(p['pIsBB'])
        if correct:
            best_correct = max(best_correct, stage)
        for i in range(stage + 1):
            out['candidates_' + STAGES[i]] += 1
            out['correct_candidates_' + STAGES[i]] += int(correct)
        if stage == 6:
            passing.append(p)
            category = ('no_qualifying_truth_pair' if not t else 'correct' if correct else
                        'partial' if p['pMatchedLegs'] == 1 else 'wrong')
            out['passing_' + category] += 1
    for i in range(any_stage + 1):
        out['events_' + STAGES[i]] = 1
    for i in range(best_correct + 1):
        out['truth_recovered_' + STAGES[i]] = 1
    ncorrect = sum(t and bool(p['pIsBB']) for p in passing)
    n = len(passing)
    out['passing_pairs'] = n
    out['correct_pairs'] = ncorrect
    out['selected_events'] = int(n > 0)
    out['selected_truth_events'] = int(t and n > 0)
    out['selected_has_signal_pair'] = int(n > 0 and out['has_signal_pair'])
    out['recovered_events'] = int(ncorrect > 0)
    out['selected_without_T'] = int(n > 0 and not t)
    out['zero_passing'] = int(n == 0)
    out['one_passing'] = int(n == 1)
    out['multiple_passing'] = int(n > 1)
    disjoint = any({a['pObjA'], a['pObjB']}.isdisjoint({b['pObjA'], b['pObjB']})
                   for i, a in enumerate(passing) for b in passing[i + 1:])
    out['multiple_with_disjoint_pair_pair'] = int(disjoint)
    out['multiple_all_overlapping'] = int(n > 1 and not disjoint)
    out['correct_only'] = int(ncorrect > 0 and n == ncorrect)
    out['correct_plus_incorrect'] = int(ncorrect > 0 and n > ncorrect)
    out['T_incorrect_only'] = int(t and n > 0 and not ncorrect)
    out['T_no_passing'] = int(t and n == 0)
    if t and best_correct < 6:
        # Exclusive losses follow the correct pair, regardless of unrelated surviving pairs.
        out['loss_' + ('reconstruction_ID_eta_propagation_matching' if best_correct < 0 else
                       STAGES[best_correct + 1])] = 1
    assert out['recovered_events'] + out['T_incorrect_only'] + out['T_no_passing'] == int(t)
    assert out['passing_correct'] + out['passing_partial'] + out['passing_wrong'] + out['passing_no_qualifying_truth_pair'] == n
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--report-dir', required=True, help='cmsRun FrameworkJobReports matching sample names')
    parser.add_argument('--expected-samples', type=int, default=72)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    import awkward as ak
    import uproot
    import ROOT
    from preselected_vertexing_analysis import check_report
    paths = sorted(Path(args.input_dir).glob('*.root'))
    if len(paths) != args.expected_samples:
        raise RuntimeError('Expected %d sample files, found %d' % (args.expected_samples, len(paths)))
    groups = defaultdict(Counter)
    seen = set()
    revisions = set()
    records = []
    sources = []
    for sample_index, path in enumerate(paths):
        with uproot.open(path) as source:
            prefix = 'preselectedVertexing/'
            name = str(source[prefix + 'sample'].member('fTitle'))
            match = re.fullmatch(r'(Mchi-[\dp]+_dMchi-[\dp]+)_(ctau-[\dp]+)', name)
            if not match or name in seen:
                raise RuntimeError('Invalid/duplicate sample: ' + name)
            seen.add(name)
            revisions.add(str(source[prefix + 'revision'].member('fTitle')))
            if str(source[prefix + 'recoOnly'].member('fTitle')) == 'disabled':
                raise RuntimeError('Not a reco-only run: ' + name)
            if str(source[prefix + 'hierarchyOrder'].member('fTitle')) != 'global>inner>outer(DSA)':
                raise RuntimeError('Unexpected hierarchy: ' + name)
            events = source[prefix + 'events'].arrays(['run', 'lumi', 'event', 'nSig', 'sigQ', 'sigD0', 'twoAcc',
                                                      'nElig', 'anyNPairs', 'pfMetPt', 'pfMetPhi'], library='ak')
            check_report(Path(args.report_dir), name, len(events))
            pairs = source[prefix + 'pairs'].arrays(['pEvent', 'pObjA', 'pObjB', 'pQA', 'pQB', 'pIsBB',
                                                    'pMatchedLegs', 'pStatus', 'pChi2', 'pNdof', 'pPcaOK',
                                                    'pPcaDR', 'pPcaMuMuPt', 'pPcaMuMuPhi'], library='ak')
            by_event = defaultdict(list)
            for p in ak.to_list(pairs):
                row = p['pEvent']
                if not 0 <= row < len(events):
                    raise RuntimeError('Invalid pair event index')
                by_event[row].append(p)
            for row, event in enumerate(ak.to_list(events)):
                epairs = by_event[row]
                if len(epairs) != event['anyNPairs'] or len(epairs) != event['nElig'] * (event['nElig'] - 1) // 2:
                    raise RuntimeError('Incomplete pair enumeration: ' + name)
                identities = [(p['pObjA'], p['pObjB']) for p in epairs]
                if len(set(identities)) != len(identities) or any(a == b for a, b in identities):
                    raise RuntimeError('Repeated/self pair: ' + name)
                c = summarize(event, epairs)
                for group in ('overall', 'sample/' + name, 'mass/' + match[1], 'lifetime/' + match[2]):
                    groups[group].update(c)
                records.append((sample_index, event['run'], event['lumi'], event['event'], int(event['twoAcc']),
                                c['passing_pairs'], c['correct_pairs'], c['multiple_with_disjoint_pair_pair'],
                                event['sigD0'][0] if event['twoAcc'] else -999.,
                                event['sigD0'][1] if event['twoAcc'] else -999.))
            sources.append((name, str(path.resolve())))
    if len(revisions) != 1 or 'unspecified' in revisions:
        raise RuntimeError('Mixed/unspecified code revisions')
    root = ROOT.TFile.Open(str(output), 'CREATE')
    if not root or root.IsZombie():
        raise RuntimeError('Cannot create output')
    ROOT.TNamed('revision', next(iter(revisions))).Write()
    ROOT.TNamed('definitions', 'Raw unweighted; T=exactly two opposite-charge accepted signal daughters; fixed G>I>DSA '
                'reco legs; truth object labels from staged same-charge station-2 greedy matching, dR<0.06 G/I,<0.10 DSA; '
                'failed station propagation stays in T; object label component may differ from fixed fitted leg; no fit fallback; PCA original-track momenta; cuts OS,valid,'
                'chi2/ndof<4,PCA valid,dR<0.9,abs(dphi(Type1 PF MET,mumu))<0.5; no angular definition for zero pT').Write()
    ROOT.TNamed('inputs', '\n'.join(name + ' ' + path for name, path in sources)).Write()
    # Store exact event units and d0 for future replotting without rescanning MiniAOD.
    tree = ROOT.TTree('events', 'Event accounting; sample index follows inputs metadata; raw unweighted')
    fields = ('sample', 'run', 'lumi', 'event', 'T', 'nPassing', 'nCorrect', 'disjoint')
    buffers = {k: array('q', [0]) for k in fields}
    for k, buffer in buffers.items():
        tree.Branch(k, buffer, k + '/L')
    d0a, d0b = array('d', [0]), array('d', [0])
    tree.Branch('genD0A', d0a, 'genD0A/D'); tree.Branch('genD0B', d0b, 'genD0B/D')
    for record in records:
        for k, value in zip(fields, record[:8]):
            buffers[k][0] = value
        d0a[0], d0b[0] = record[8:]
        tree.Fill()
    tree.Write()
    for group, counts in sorted(groups.items()):
        root.cd()
        directory = root
        for part in group.split('/'):
            directory = directory.GetDirectory(part) or directory.mkdir(part)
        directory.cd()
        keys = sorted(counts)
        hist = ROOT.TH1D('counts', 'Raw counts;Counter;Count', len(keys), 0, len(keys))
        for i, key in enumerate(keys, 1):
            hist.GetXaxis().SetBinLabel(i, key)
            hist.SetBinContent(i, counts[key])
        hist.Write()
        print(group, ' '.join('%s=%d' % item for item in sorted(counts.items())))
    root.Close()
    c = groups['overall']
    def ratio(n, d):
        return 'undefined (zero denominator)' if not d else '%.6f%% (%d/%d)' % (100 * n / d, n, d)
    print('Candidate correctness:', ratio(c['correct_pairs'], c['passing_pairs']))
    print('Truth event recovery:', ratio(c['recovered_events'], c['truth_accepted']))
    print('Missed truth fraction:', ratio(c['truth_accepted'] - c['recovered_events'], c['truth_accepted']))


if __name__ == '__main__':
    main()
