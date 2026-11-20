"""
List truth-merged signal events (run, lumi, event + gen electron info) at a
chosen cut of a selection, for later event-display ("event map") production
from MiniAOD.

Runs the standard iDMeProcessor (analysisTools/analysisTools.py) over the
signal sample config, so every event gets the usual extra variables and
routines.computeMergedVars (-> events.cat_merged, the truth merged-electron
category from computeMergedCatVars). The selection's cuts are applied in
order up to and including `cut`; events surviving it with cat_merged == True
are written, for every signal point, to one combined CSV.

This file doubles as the processor's "histo config" module (it defines
make_histograms / fillHistos / subroutines), so the processor imports it
by name -- everything that runs the job lives under `if __name__ == '__main__'`.

Run from anywhere:  python eventmaps/merged_eventlist.py
"""
import os
import sys
import time

import numpy as np
import awkward as ak
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)  # python_analysis/
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import analysisTools.analysisTools as tools
import analysisTools.analysisSubroutines as routines

# ---- SETTINGS ----
vers = 'Nov2026'                  # signal ntuple version (sample config tag)
selection = 'anabrv'        # file stem in configs/cut_configs/
cut = 'cut3'                      # cut function name in `selection`; merged events are checked after it.
                                  # NB: in an_selection, cut9 (good ee vertex) removes nearly all truth-merged
                                  # events, since cat_merged excludes vertexed events.
execr = 'futures'                 # 'futures' or 'iterative'
workers = 4
max_files_per_samp = -1           # -1 = all files; small number for quick tests

# ---- FILES ----
sample_config = os.path.join(BASE, f"configs/sample_configs/signal_2024_{vers}_0p1.json")
cuts_config = os.path.join(BASE, f"configs/cut_configs/{selection}_selection.py")
outdir = os.path.join(HERE, 'outputs')
outfile = os.path.join(outdir, f"merged_events_sig{vers}_{selection}-sel_{cut}.csv")

# per-event columns filled in fillHistos (sample metadata columns are added afterwards)
_EVENT_COLUMNS = ['run', 'lumi', 'event', 'gen_ele_pt', 'gen_pos_pt', 'gen_lxy', 'gen_dR_ee', 'ntuple_file']


# ============================================================================
# "Histo config" interface used by iDMeProcessor
# ============================================================================
subroutines = []  # computeMergedVars already runs inside iDMeProcessor.process


def make_histograms():
    # {sample: {column: list}} -- coffea's accumulate() merges nested dicts of lists across chunks
    return {'merged_events': {}}


def _np(arr, fill=np.nan):
    return ak.to_numpy(ak.fill_none(arr, fill))


def fillHistos(events, hists, samp, cut, info, sum_wgt=1):
    """Called by iDMeProcessor only at the target cut (see _SetPlots below)."""
    if 'cat_merged' not in events.fields:
        raise RuntimeError("events.cat_merged is missing: this needs signal ntuples with GenEle/GenPos and "
                           "AllLptElectron (filled by routines.computeMergedVars)")
    filename = events.metadata.get('filename', '') if hasattr(events, 'metadata') else ''

    sel = events[ak.fill_none(events.cat_merged, False)]
    if len(sel) == 0:
        return

    rec = hists['merged_events'].setdefault(samp, {c: [] for c in _EVENT_COLUMNS})
    rec['run'].extend(_np(sel.runNum, -1).tolist())
    rec['lumi'].extend(_np(sel.lumiSec, -1).tolist())
    rec['event'].extend(_np(sel.eventNum, -1).tolist())
    rec['gen_ele_pt'].extend(_np(sel.GenEle.pt).tolist())
    rec['gen_pos_pt'].extend(_np(sel.GenPos.pt).tolist())
    rec['gen_lxy'].extend(_np(sel.gen_lxy).tolist())  # from computeMergedCatVars (chi2 production vertex)
    rec['gen_dR_ee'].extend(_np(routines.deltaR(sel.GenEle.eta, sel.GenEle.phi,
                                                sel.GenPos.eta, sel.GenPos.phi)).tolist())
    rec['ntuple_file'].extend([filename] * len(sel))


# ============================================================================
# Running
# ============================================================================
class _SetPlots:
    """Wrap a cut function, overriding its savePlots flag, so fillHistos runs
    only at the target cut (regardless of the selection's own plots flags)."""
    def __init__(self, cut_fn, plots):
        self.cut_fn = cut_fn
        self.plots = plots

    def __call__(self, events, info):
        events, name, desc, _ = self.cut_fn(events, info)
        return events, name, desc, self.plots


def restrict_to_cut(proc, target):
    """Keep the processor's cuts up to and including `target` (a cut function
    name in the selection), with fillHistos enabled only at `target`."""
    if target not in proc.cutNames:
        raise ValueError(f"cut '{target}' not in {proc.cutFile}; available: {proc.cutNames}")
    i = proc.cutNames.index(target)
    proc.cuts = [_SetPlots(c, False) for c in proc.cuts[:i]] + [_SetPlots(proc.cuts[i], True)]
    proc.cutNames = proc.cutNames[:i + 1]


def build_table(out, analyzer, target):
    frames = []
    for samp in analyzer.sample_names:
        info = analyzer.sample_info[samp]
        n_pass = out['cutflow_nevts'].get(samp, {}).get(target, 0)
        rec = out['merged_events'].get(samp)
        n_merged = len(rec['event']) if rec else 0
        print(f"{samp}: {n_merged} truth-merged / {n_pass} events passing {target}")
        if n_merged == 0:
            continue
        df = pd.DataFrame(rec)
        df.insert(0, 'sample', samp)
        df.insert(1, 'year', info['year'])
        df.insert(2, 'm1', info['M1'])
        df.insert(3, 'delta', info['dM'])  # fractional splitting, e.g. 0.1 = 10%
        df.insert(4, 'mZD', info.get('mZD', np.nan))
        df.insert(5, 'ctau', info['ctau'])
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=['sample', 'year', 'm1', 'delta', 'mZD', 'ctau'] + _EVENT_COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(['m1', 'delta', 'mZD', 'ctau', 'run', 'lumi', 'event']).reset_index(drop=True)


if __name__ == '__main__':
    from coffea import processor

    # Analyzer handles sample-config loading/naming; this file is passed as the histo config
    analyzer = tools.Analyzer(sample_config, os.path.abspath(__file__), cuts_config,
                              max_files_per_samp=max_files_per_samp)
    if analyzer.mode != 'signal':
        sys.exit(f"{sample_config} is not a signal config (type '{analyzer.mode}'); truth merging needs signal")

    proc = tools.iDMeProcessor(analyzer.sample_names, analyzer.sample_info, analyzer.sample_locs,
                               analyzer.histoFile, analyzer.cuts, mode=analyzer.mode,
                               good_vtx=analyzer.good_vtx, good_merged=analyzer.good_merged)
    restrict_to_cut(proc, cut)
    print(f"Applying {selection}_selection.py cuts {proc.cutNames}; listing truth-merged (cat_merged) events after {cut}")

    # same Runner setup as Analyzer.process (old-coffea path), but with the restricted processor
    if execr == 'futures':
        executor = processor.FuturesExecutor(workers=workers)
    else:
        executor = processor.IterativeExecutor()
    runner = processor.Runner(executor=executor, schema=tools.MySchema, savemetrics=True,
                              xrootdtimeout=180, skipbadfiles=True)

    t1 = time.time()
    out, metrics = runner(analyzer.sample_locs, treename='ntuples/outT', processor_instance=proc)
    print("Runtime: {:.2f} minutes".format((time.time() - t1) / 60))

    df = build_table(out, analyzer, cut)
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(outfile, index=False)
    print(f"Wrote {len(df)} events ({df['sample'].nunique()} signal points) to {outfile}")
