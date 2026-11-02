#!/usr/bin/env python3
"""Recovery efficiency plots and group-ratio CSV from saved reco-pair ROOT outputs.

Run with CMSSW python3. Build reads recovery.root and its saved producer ROOT
files once; plot reads only the new histogram ROOT file. No MiniAOD processing.
"""
import argparse
from array import array
import csv
from pathlib import Path
import re

import numpy as np

from preselected_vertexing_analysis import cp, flowcounts, write_hist

LABELS = {
    'dimuon_pt': r'Gen dimuon $p_T$ [GeV]',
    'leading_pt': r'Leading gen muon $p_T$ [GeV]',
    'subleading_pt': r'Subleading gen muon $p_T$ [GeV]',
    'leading_abs_eta': r'Leading gen muon $|\eta|$',
    'subleading_abs_eta': r'Subleading gen muon $|\eta|$',
    'leading_abs_d0': r'Leading gen muon $|d_0|$ [cm]',
    'subleading_abs_d0': r'Subleading gen muon $|d_0|$ [cm]',
    'lxy': r'Gen dimuon production $L_{xy}$ [cm]',
}
RATIOS = {
    'candidate_correctness': ('correct_pairs', 'passing_pairs'),
    'truth_event_recovery': ('recovered_events', 'truth_accepted'),
    'truth_event_missed': ('missed_events', 'truth_accepted'),
    'truth_incorrect_only': ('T_incorrect_only', 'truth_accepted'),
    'truth_no_passing': ('T_no_passing', 'truth_accepted'),
    'recovered_with_competitors': ('correct_plus_incorrect', 'recovered_events'),
    'selected_multiple': ('multiple_passing', 'selected_events'),
    'selected_disjoint': ('multiple_with_disjoint_pair_pair', 'selected_events'),
    'selected_all_overlapping': ('multiple_all_overlapping', 'selected_events'),
    'selected_one': ('one_passing', 'selected_events'),
}
DEFINITIONS = (
    'Event recovery = T events with at least one correct passing pair / all T events; '
    'raw unweighted; no trigger; unchanged reco-only fixed G>I>DSA selection. '
    'Axes are gen-level; leading/subleading ordered by gen pT (ties: first stored daughter); '
    'dimuon pT recomputed from saved daughter pT/phi; Lxy common production radius. '
    '|d0|=abs(vx*sin(phi)-vy*cos(phi)), origin reference, cm. '
    'Truth labels: same-charge staged G>I>DSA station-2 greedy one-to-one, strict dR<0.06 G/I,<0.10 DSA; '
    'object-label component may differ from fitted component; failed propagation remains in T. '
    '68.27% Clopper-Pearson intervals; zero-denominator bins omitted, nonzero bins retained. '
    'Numerator and denominator histograms retain under/overflow; final edges include float-rounded acceptance boundaries. '
    'CSV ratios are fractions; empty ratio means zero denominator; mass pooling over lifetimes and '
    'lifetime pooling over masses are raw unweighted.'
)


def group_parts(group):
    if group == 'overall':
        return 'overall', '', '', ''
    kind, name = group.split('/', 1)
    match = re.fullmatch(r'(Mchi-[\dp]+_dMchi-[\dp]+)(?:_(ctau-[\dp]+))?', name)
    if kind == 'sample':
        return kind, name, match[1], match[2]
    return kind, '', name if kind == 'mass' else '', name if kind == 'lifetime' else ''


def build(args):
    import awkward as ak
    import ROOT
    import uproot
    output, csv_path = Path(args.output), Path(args.csv)
    if output.exists() or csv_path.exists():
        raise FileExistsError('Output ROOT/CSV already exists')
    with uproot.open(args.recovery) as source:
        recovery = source['events'].arrays(library='np')
        inputs = [line.split(' ', 1) for line in source['inputs'].member('fTitle').splitlines()]
        processing_revision = source['revision'].member('fTitle')
        counts = {}
        for key in source.keys(recursive=True, cycle=False):
            if re.fullmatch(r'(overall|(?:sample|mass|lifetime)/[^/]+)/counts', key):
                h = source[key]
                counts[key.rsplit('/', 1)[0]] = dict(zip(h.axis().labels(), h.values()))
    columns = {key: [] for key in LABELS}
    columns.update({key: [] for key in ('sample', 'run', 'lumi', 'event', 'recovered')})
    for si, (name, path) in enumerate(inputs):
        with uproot.open(path) as source:
            prefix = 'preselectedVertexing/'
            assert source[prefix + 'sample'].member('fTitle') == name
            assert source[prefix + 'revision'].member('fTitle') == processing_revision
            events = source[prefix + 'events'].arrays(
                ['run', 'lumi', 'event', 'twoAcc', 'sigPt', 'sigEta', 'sigPhi', 'sigD0', 'sigLxy'], library='ak')
        rows = recovery['sample'] == si
        assert int(rows.sum()) == len(events), name
        for key in ('run', 'lumi', 'event'):
            assert np.array_equal(recovery[key][rows], ak.to_numpy(events[key])), (name, key)
        t = ak.to_numpy(events['twoAcc']).astype(bool)
        assert np.array_equal(recovery['T'][rows].astype(bool), t), name
        values = {key: ak.to_numpy(events[key][t]).astype(np.float64)
                  for key in ('sigPt', 'sigEta', 'sigPhi', 'sigD0', 'sigLxy')}
        assert all(v.shape == (int(t.sum()), 2) and np.isfinite(v).all() for v in values.values()), name
        pt, phi, lxy = values['sigPt'], values['sigPhi'], values['sigLxy']
        assert np.all(pt >= 0) and np.all(values['sigD0'] >= 0), name
        assert np.allclose(lxy[:, 0], lxy[:, 1], rtol=1e-6, atol=1e-6), ('Noncommon production Lxy', name)
        lead = np.argmax(pt, axis=1)
        row = np.arange(len(lead))
        columns['dimuon_pt'].append(np.hypot(np.sum(pt * np.cos(phi), axis=1), np.sum(pt * np.sin(phi), axis=1)))
        for prefix, leg in (('leading', lead), ('subleading', 1 - lead)):
            columns[prefix + '_pt'].append(pt[row, leg])
            columns[prefix + '_abs_eta'].append(np.abs(values['sigEta'][row, leg]))
            columns[prefix + '_abs_d0'].append(values['sigD0'][row, leg])
        columns['lxy'].append(lxy[:, 0])
        columns['sample'].append(np.full(int(t.sum()), si, np.int64))
        for key in ('run', 'lumi', 'event'):
            columns[key].append(recovery[key][rows][t])
        columns['recovered'].append((recovery['nCorrect'][rows][t] > 0).astype(np.int64))
    columns = {key: np.concatenate(value) for key, value in columns.items()}
    assert len(columns['sample']) == int(counts['overall']['truth_accepted'])
    assert int(columns['recovered'].sum()) == int(counts['overall']['recovered_events'])
    pt_edges = np.r_[np.arange(0, 26, 1), np.arange(30, 105, 5), np.arange(120, 320, 20),
                     np.arange(400, 1100, 100), 1500, 2000, 3000, 5000]
    d0_edges = np.r_[np.arange(0, 1, .05), np.arange(1, 16, .5), np.arange(16, 52, 2),
                     np.arange(60, 210, 10), np.arange(230, 740, 30), 740]
    lxy_edges = np.r_[np.arange(0, 31, 1), np.arange(35, 105, 5), np.arange(120, 320, 20),
                      np.arange(350, 740, 30), 740]
    edges = {}
    for key in LABELS:
        if key.endswith('_pt'):
            maximum = float(columns[key].max())
            assert maximum < pt_edges[-1], (key, maximum, 'Extend explicit pT binning')
            stop = max(26, int(np.searchsorted(pt_edges, maximum, side='right')) + 1)
            edges[key] = pt_edges[:stop]
        elif key.endswith('_abs_eta'):
            edges[key] = np.linspace(0, 2.4, 25)
        else:
            edges[key] = (d0_edges if key.endswith('_abs_d0') else lxy_edges).copy()
        # Accepted double-precision values can round to the boundary in saved float branches.
        edges[key][-1] = np.nextafter(max(edges[key][-1], float(np.float32(edges[key][-1]))), np.inf)
    groups = {'overall': np.ones(len(columns['sample']), bool)}
    for si, (name, path) in enumerate(inputs):
        mass, lifetime = name.rsplit('_', 1)
        mask = columns['sample'] == si
        groups['sample/' + name] = mask
        for group in ('mass/' + mass, 'lifetime/' + lifetime):
            groups[group] = groups.get(group, np.zeros(len(mask), bool)) | mask
    assert set(groups) == set(counts)
    root = ROOT.TFile.Open(str(output), 'CREATE')
    if not root or root.IsZombie():
        raise RuntimeError('Cannot create ROOT output')
    ROOT.TNamed('definitions', DEFINITIONS).Write()
    ROOT.TNamed('processing_revision', processing_revision).Write()
    ROOT.TNamed('presentation_revision', args.revision).Write()
    ROOT.TNamed('source_recovery', str(Path(args.recovery).resolve())).Write()
    ROOT.TNamed('inputs', '\n'.join(name + ' ' + path for name, path in inputs)).Write()
    tree = ROOT.TTree('events', 'Accepted truth events only; gen variables and unchanged event recovery flag')
    buffers = {}
    for key in columns:
        integer = key not in LABELS
        buffers[key] = array('q' if integer else 'd', [0])
        tree.Branch(key, buffers[key], key + ('/L' if integer else '/D'))
    for i in range(len(columns['sample'])):
        for key in columns:
            buffers[key][0] = int(columns[key][i]) if key not in LABELS else float(columns[key][i])
        tree.Fill()
    tree.Write()
    for group, mask in sorted(groups.items()):
        assert int(mask.sum()) == int(counts[group]['truth_accepted']), group
        selected = mask & columns['recovered'].astype(bool)
        assert int(selected.sum()) == int(counts[group]['recovered_events']), group
        directory = root
        for part in group.split('/'):
            directory = directory.GetDirectory(part) or directory.mkdir(part)
        directory.cd()
        for variable in LABELS:
            den = flowcounts(columns[variable][mask], edges[variable])
            num = flowcounts(columns[variable][selected], edges[variable])
            assert np.all(num <= den) and int(den.sum()) == int(mask.sum()) and int(num.sum()) == int(selected.sum())
            assert den[0] == den[-1] == num[0] == num[-1] == 0, (group, variable, 'Nonzero full-range flow')
            for label, values in (('denominator', den), ('numerator', num)):
                write_hist(label + '__' + variable, ';' + LABELS[variable] + ';Events', edges[variable], values)
        print('VALIDATED', group, 'denominator', int(mask.sum()), 'numerator', int(selected.sum()), flush=True)
    root.Close()
    fields = ['group_type', 'group', 'sample', 'mass_point', 'lifetime', 'pooling']
    fields += [metric + '_' + suffix for metric in RATIOS for suffix in ('numerator', 'denominator', 'ratio')]
    with csv_path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for group, counter in sorted(counts.items()):
            kind, sample, mass, lifetime = group_parts(group)
            row = dict(group_type=kind, group=group, sample=sample, mass_point=mass, lifetime=lifetime,
                       pooling='raw_unweighted_no_trigger')
            counter['missed_events'] = counter['truth_accepted'] - counter['recovered_events']
            for metric, (nkey, dkey) in RATIOS.items():
                n, d = int(counter.get(nkey, 0)), int(counter.get(dkey, 0))
                row.update({metric + '_numerator': n, metric + '_denominator': d,
                            metric + '_ratio': n / d if d else ''})
            writer.writerow(row)
    print('CSV rows', len(counts), 'ROOT', output, 'CSV', csv_path, flush=True)


def plot(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import uproot
    destination = Path(args.plot_dir)
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    with uproot.open(args.output) as source:
        groups = sorted({key.rsplit('/', 1)[0] for key in source.keys(recursive=True, cycle=False)
                         if key.endswith('/denominator__dimuon_pt')})
        overlay_only = args.group in ('overlay_mass', 'overlay_lifetime')
        if args.group and not overlay_only:
            if args.group not in groups:
                raise ValueError('Unknown group: ' + args.group)
            groups = [args.group]
        def curves(group, variable):
            denominator = source[group + '/denominator__' + variable]
            n = source[group + '/numerator__' + variable].values()
            d = denominator.values()
            edges = denominator.axis().edges()
            ok = d > 0
            ratio = np.divide(n, d, out=np.zeros_like(n), where=ok)
            lower, upper = cp(n, d)
            return (edges[:-1] + edges[1:]) / 2, ratio, lower, upper, ok, edges
        def label(group):
            return group.replace('sample/', '').replace('mass/', '').replace('lifetime/', '').replace('p', '.')
        def draw(groups_here, variable, view, tag):
            many = len(groups_here) > 10
            fig, ax = plt.subplots(figsize=(10, 7.8 if many else 6.6))
            for i, group in enumerate(groups_here):
                x, y, lo, hi, ok, edges = curves(group, variable)
                color = plt.get_cmap('turbo')(.05 + .9 * i / max(len(groups_here) - 1, 1)) if many else 'C%d' % (i % 10)
                ax.errorbar(x[ok], y[ok], yerr=[(y-lo)[ok], (hi-y)[ok]], fmt='o', markersize=3,
                            linewidth=.8, color=color, label=label(group))
            ax.set_xlabel(LABELS[variable], fontsize=12)
            ax.set_ylabel('Recovered T events / all T events', fontsize=12)
            ax.set_ylim(0, 1.05)
            xmax = (25 if variable.endswith('_pt') else 15 if variable.endswith('_abs_d0') else 30) if view == 'focused' else edges[-1]
            ax.set_xlim(0, xmax)
            ax.grid(alpha=.25)
            ax.text(0, 1.02, 'CMS Simulation Private Work', transform=ax.transAxes, fontsize=11, fontweight='bold')
            ax.text(1, 1.02, '2022EE (13.6 TeV)', transform=ax.transAxes, ha='right', fontsize=10)
            if len(groups_here) == 1:
                ax.set_title(label(groups_here[0]), fontsize=9, pad=33)
            else:
                fig.legend(*ax.get_legend_handles_labels(), loc='lower center', ncol=3 if many else 2,
                           fontsize=8, bbox_to_anchor=(.5, .07))
            footer = 'Raw unweighted; no trigger; 68.27% Clopper-Pearson intervals; empty denominator bins omitted.'
            if variable.endswith('_abs_d0'):
                footer += '\nGen |d0| = |vx sin(phi) - vy cos(phi)|, relative to origin.'
            fig.text(.08, .015, footer, fontsize=8)
            bottom = .28 if many else .24 if len(groups_here) > 1 else .13
            fig.subplots_adjust(left=.11, right=.97, bottom=bottom, top=.86)
            path = destination / view / tag / (variable + '.png')
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=110)
            if args.log:
                ax.set_yscale('log'); ax.set_ylim(1e-5, 1.2)
                fig.savefig(path.with_name(path.stem + '_log.png'), dpi=110)
            plt.close(fig)
        for variable in LABELS:
            views = ('full',) if variable.endswith('_abs_eta') else ('full', 'focused')
            for view in views:
                for group in ([] if overlay_only else groups):
                    draw([group], variable, view, group)
                for kind in ((args.group.removeprefix('overlay_'),) if overlay_only else () if args.group else ('mass', 'lifetime')):
                    overlay = [group for group in groups if group.startswith(kind + '/')]
                    draw(overlay, variable, view, 'overlay_' + kind)
            print('PLOTTED', variable, flush=True)
    print('Plots:', destination, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    build_parser = sub.add_parser('build', help='Join saved gen variables, fill ROOT counts and export ratios')
    build_parser.add_argument('--recovery', required=True)
    build_parser.add_argument('--output', required=True)
    build_parser.add_argument('--csv', required=True)
    build_parser.add_argument('--revision', required=True)
    plot_parser = sub.add_parser('plot', help='Replot only from saved numerator/denominator ROOT histograms')
    plot_parser.add_argument('--output', required=True)
    plot_parser.add_argument('--plot-dir', required=True)
    plot_parser.add_argument('--group', help='Plot one saved group, overlay_mass or overlay_lifetime')
    plot_parser.add_argument('--log', action='store_true', help='Add log-y copies; linear copies always retained')
    args = parser.parse_args()
    (build if args.mode == 'build' else plot)(args)


if __name__ == '__main__':
    main()
