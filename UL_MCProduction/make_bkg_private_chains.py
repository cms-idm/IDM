#!/usr/bin/env python3
"""Build per-sample chains for the private 2022EE background production from saved McM requests.

Offline: reads <prep>/samples.tsv, <prep>/prepids.tsv and <prep>/mcm/<sample>_{request,dr_request}.json (McM GS and
DRPremix requests, saved by the lookup scripts) and the tested W+jets chain, whose MiniAOD (iDM PAT) step is reused
verbatim. Writes <out>/<sample>_chain.json (W+jets chain format), <out>/<sample>_fragment.py (McM fragment) and
<out>/samples.tsv (sample, index, McM efficiency, generated events per job, LHE or not, releases).
"""
import argparse
import json
import math
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--prep', required=True)
p.add_argument('--wjets-chain', required=True)
p.add_argument('--output', required=True)
p.add_argument('--events-out-per-job', type=int, default=250)
a = p.parse_args()

GT = '124X_mcRun3_2022_realistic_postEE_v1'
STEPS = ['Run3Summer22EEGENSIM', 'Run3Summer22EEDIGIPremix', 'Run3Summer22EERECO', 'Run3Summer22EEMiniAOD']
# McM sequence fields passed to cmsDriver; list values are comma-joined. 'index' is McM bookkeeping and 'nThreads'
# is set by the job (2 CPUs). Any other non-empty field stops the build for review.
PASS = ['eventcontent', 'datatier', 'conditions', 'beamspot', 'step', 'geometry', 'era', 'procModifiers', 'datamix']
DROP = ['index', 'nThreads']
EMPTY = ('', [], None, False, -1)
PREMIX = 'dbs:/Neutrino_E-10_gun/Run3Summer21PrePremix-Summer22_124X_mcRun3_2022_realistic_v11-v2/PREMIX'


def options(seq, where):
    extra = [k for k, v in seq.items() if v not in EMPTY and k not in PASS + DROP]
    if extra:
        raise SystemExit('%s: unreviewed McM sequence fields %s' % (where, extra))
    opts = {k: ','.join(seq[k]) if isinstance(seq[k], list) else seq[k] for k in PASS if seq.get(k) not in EMPTY}
    if opts.get('conditions') != GT:
        raise SystemExit('%s: conditions %s' % (where, opts.get('conditions')))
    opts['mc'] = None
    return opts


prep = Path(a.prep)
out = Path(a.output)
out.mkdir(exist_ok=False)
wjets = json.loads(Path(a.wjets_chain).read_text())
if wjets['STEPS'] != STEPS:
    raise SystemExit('Unexpected W+jets chain steps')
pat = wjets['WORKFLOWS']['Run3Summer22EEMiniAOD']
prepids = {l.split('\t')[0]: l.rstrip('\n').split('\t') for l in open(prep / 'prepids.tsv')}
rows = []
for index, line in enumerate(open(prep / 'samples.tsv')):
    s = line.split()[0]
    gs = json.loads((prep / 'mcm' / (s + '_request.json')).read_text())
    dr = json.loads((prep / 'mcm' / (s + '_dr_request.json')).read_text())
    if [gs['prepid'], dr['prepid']] != prepids[s][1:3]:
        raise SystemExit('%s: prepids differ from prepids.tsv' % s)
    if len(gs['sequences']) != 1 or len(dr['sequences']) != 2:
        raise SystemExit('%s: expected 1 GS and 2 DRPremix sequences' % s)
    raw = (prep / 'mcm' / (s + '_fragment.py')).read_text()
    # McM returns some fragments as a JSON-quoted string.
    fragment = json.loads(raw) if raw.startswith('"') else raw
    lhe = 'ExternalLHEProducer' in fragment
    gen = options(gs['sequences'][0], s + ' GS')
    if gen['step'] != ('LHE,GEN,SIM' if lhe else 'GEN,SIM'):
        raise SystemExit('%s: GS step %s does not match the fragment (LHE %s)' % (s, gen['step'], lhe))
    digi = options(dr['sequences'][0], s + ' DIGI')
    reco = options(dr['sequences'][1], s + ' RECO')
    if not digi['step'].startswith('DIGI,DATAMIX') or digi.get('datamix') != 'PreMix':
        raise SystemExit('%s: unexpected DIGI sequence %s' % (s, digi))
    if not reco['step'].startswith('RAW2DIGI'):
        raise SystemExit('%s: unexpected RECO sequence %s' % (s, reco))
    digi['pileup_input'] = PREMIX
    g = gs['generator_parameters'][-1]
    eff = g['match_efficiency'] * g['filter_efficiency']
    if not 0 < eff <= 1:
        raise SystemExit('%s: efficiency %s' % (s, eff))
    nevents = int(round(a.events_out_per_job / eff))

    def wf(release, opts):
        return {'SCRAM_ARCH': 'el8_amd64_gcc10', 'CMSSW_VERSION': release, 'OPTIONS': opts,
                'CUSTOMIZES': {'files': [], 'pre-cmsRun': [], 'post-cmsRun': []}}
    chain = {'STEPS': STEPS, 'KEEPS': ['Run3Summer22EEMiniAOD'],
             'SOURCE': {'GS': gs['prepid'], 'DRPremix': dr['prepid']},
             'WORKFLOWS': {'Run3Summer22EEGENSIM': wf(gs['cmssw_release'], gen),
                           'Run3Summer22EEDIGIPremix': wf(dr['cmssw_release'], digi),
                           'Run3Summer22EERECO': wf(dr['cmssw_release'], reco),
                           'Run3Summer22EEMiniAOD': pat}}
    (out / (s + '_chain.json')).write_text(json.dumps(chain, indent=2) + '\n')
    (out / (s + '_fragment.py')).write_text(fragment)
    rows.append([s, index, '%.4g' % eff, nevents, int(lhe), gs['cmssw_release'], dr['cmssw_release'],
                 math.ceil(11000 / (nevents * eff))])
with open(out / 'samples.tsv', 'w') as f:
    f.write('#sample\tindex\teff_mcm\tnevents_per_job\tlhe\tgs_release\tdr_release\tjobs_mcm\n')
    for r in rows:
        f.write('\t'.join(map(str, r)) + '\n')
print('%d samples, %d jobs at McM efficiency' % (len(rows), sum(r[-1] for r in rows)))
