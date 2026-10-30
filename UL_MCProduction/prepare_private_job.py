#!/usr/bin/env python3
"""Prepare one job of the private 2022EE background production (prepare_wjets_test.py generalized).

Preparation only: writes run_test.sh and inputs into a new directory; does not run CMSSW or write to EOS.
Job N of sample index I: lumi section N+1 (one lumi per job, run 1), --nevents generated events, every GEN-SIM and
DIGIPremix engine reseeded from random.Random(seed_base + 10*N + step), seed_base = 30102026000 + 10000*I. For LHE
samples the externalLHEProducer seed is set to lhe_seed(I, N) = 500000000 + 100000*I + 10*N: with 2 threads and
generateConcurrently=True the gridpack runs use lhe_seed + thread (CMSSW ExternalLHEProducer), so each job owns the
block [lhe_seed, lhe_seed + 10).
"""
import argparse
import json
from pathlib import Path
import shlex
import shutil

THREADS = 2
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--chain', required=True)
p.add_argument('--fragment', required=True)
p.add_argument('--pileup-list', required=True)
p.add_argument('--output', required=True)
p.add_argument('--sample-index', type=int, required=True)
p.add_argument('--job', type=int, required=True)
p.add_argument('--nevents', type=int, required=True)
a = p.parse_args()
if not 0 <= a.job < 1000 or not 0 <= a.sample_index < 1000 or not 0 < a.nevents < 100000:
    raise SystemExit('--job, --sample-index or --nevents out of range')
seed_base = 30102026000 + 10000 * a.sample_index
lhe_seed = 500000000 + 100000 * a.sample_index + 10 * a.job


def job_seeds(i):
    # RandomNumberServiceHelper.populate() uses SystemRandom; seed every engine reproducibly instead.
    # cmsDriver exec()s this inside a method, so no comprehension scope: bind the generator as a default.
    return ('from IOMC.RandomEngine.RandomServiceHelper import RandomNumberServiceHelper; import random; '
            'h = RandomNumberServiceHelper(process.RandomNumberGeneratorService); '
            'h.insertSeeds(*map(lambda _, g=random.Random(%d): g.randint(1, 900000000), range(h.countSeeds())))'
            % (seed_base + 10 * a.job + i))
chain = json.loads(Path(a.chain).read_text())
steps = chain['STEPS']
if steps != ['Run3Summer22EEGENSIM', 'Run3Summer22EEDIGIPremix',
             'Run3Summer22EERECO', 'Run3Summer22EEMiniAOD']:
    raise SystemExit('Unexpected chain: review before preparing this job')
lhe = 'ExternalLHEProducer' in Path(a.fragment).read_text()
out = Path(a.output).resolve()
out.mkdir(exist_ok=False)
(out / 'logs').mkdir()
for source, name in [(a.chain, 'chain.json'), (a.fragment, 'fragment.py'),
                     (a.pileup_list, 'pileup.txt')]:
    shutil.copyfile(source, out / name)
(out / 'check_output.py').write_text('''import sys
import ROOT
f = ROOT.TFile.Open(sys.argv[1])
if not f or f.IsZombie():
    raise SystemExit("Cannot reopen output")
t = f.Get("Events")
n = t.GetEntries() if t else 0
if n < 1:
    raise SystemExit("Expected at least 1 event")
if len(sys.argv) > 2:
    r = ROOT.TFile.Open(sys.argv[2])
    if not r or r.IsZombie() or not r.Get("Events"):
        raise SystemExit("Cannot reopen previous output")
    if r.Get("Events").GetEntries() != n:
        raise SystemExit("Event count differs from previous stage")
    r.Close()
print("PASS: reopened output with", n, "events:", sys.argv[1])
f.Close()
''')
q = shlex.quote
lines = ['#!/usr/bin/env bash', 'set -eo pipefail',
         'cd ' + q(str(out)),
         'test ! -e logs/started || { echo "Job already started; use a new directory"; exit 1; }',
         'test -n "${X509_USER_PROXY:-}" && test -r "$X509_USER_PROXY"',
         'voms-proxy-info -exists -valid 1:00',
         'date -u > logs/started',
         'exec > >(tee -a logs/execution.log) 2>&1',
         'source /cvmfs/cms.cern.ch/cmsset_default.sh',
         'trap \'s=$?; echo "TEST_EXIT=$s"; exit "$s"\' EXIT',
         'date -u', 'hostname', 'cat /etc/redhat-release',
         'echo ' + q('JOB=%d SAMPLE_INDEX=%d NEVENTS=%d LHE=%d SEED_BASE=%d LHE_SEED=%d'
                     % (a.job, a.sample_index, a.nevents, lhe, seed_base, lhe_seed))]
previous = None
for i, step in enumerate(steps):
    wf = chain['WORKFLOWS'][step]
    release = wf['CMSSW_VERSION']
    for values in wf.get('CUSTOMIZES', {}).values():
        if values:
            raise SystemExit('Nonempty CUSTOMIZES requires explicit review')
    # One effective option map per step: the chain's options, then the job's settings, so nothing is given twice.
    opts = dict(wf['OPTIONS'])
    if 'nThreads' in opts or 'nStreams' in opts:
        raise SystemExit('Chain sets threads; the job sets --nThreads %d' % THREADS)
    custom = [shlex.split(opts.pop('customise_commands'))[0]] if 'customise_commands' in opts else []
    if 'pileup_input' in opts:
        opts['pileup_input'] = 'filelist:' + str(out / 'pileup.txt')
    lines += ['(', 'export SCRAM_ARCH=' + q(wf['SCRAM_ARCH']),
              'test -d ' + q(release) + ' || scram project CMSSW ' + q(release),
              'cd ' + q(release + '/src'), 'eval "$(scram runtime -sh)"']
    args = ['cmsDriver.py']
    if i == 0:
        lines += ['mkdir -p Configuration/GenProduction/python',
                  'cp ../../fragment.py Configuration/GenProduction/python/private_fragment.py',
                  'scram b -j 2']
        job = ['process.source.firstLuminosityBlock=cms.untracked.uint32(%d)' % (a.job + 1),
               'process.source.numberEventsInLuminosityBlock=cms.untracked.uint32(%d)' % a.nevents,
               job_seeds(i)]
        if lhe:
            # After job_seeds, so this engine gets the reserved block instead of a random seed.
            job += ['process.externalLHEProducer.nEvents=cms.untracked.uint32(%d)' % a.nevents,
                    'process.RandomNumberGeneratorService.externalLHEProducer.initialSeed=cms.untracked.uint32(%d)'
                    % lhe_seed]
        custom += job
        # No --number_out: ExternalLHEProducer fails at end of run unless all LHE events are used.
        args += ['Configuration/GenProduction/python/private_fragment.py', '--number', str(a.nevents)]
    else:
        args += [step, '--number', '-1', '--filein', 'file:' + str(out / (previous + '.root'))]
        if step == 'Run3Summer22EEDIGIPremix':
            custom += [job_seeds(i)]
    if custom:
        args += ['--customise_commands', '; '.join(custom)]
    args += ['--nThreads', str(THREADS), '--no_exec', '--fileout', 'file:' + str(out / (step + '.root')),
             '--python_filename', str(out / (step + '.py'))]
    for key, value in opts.items():
        args.append('--' + key)
        if value is not None:
            args.append(str(value))
    lines += ['cd ' + q(str(out)), 'python3 --version', 'root-config --version',
              'echo ' + q('RELEASE=' + release + ' STEP=' + step),
              'echo ' + q(shlex.join(args)) + ' >> logs/commands.log',
              shlex.join(args) + ' > logs/' + step + '_cmsDriver.log 2>&1',
              'echo ' + q('cmsRun -j ' + step + '_FrameworkJobReport.xml ' + step + '.py') + ' >> logs/commands.log',
              '/usr/bin/time -v cmsRun -j ' + q(step + '_FrameworkJobReport.xml') + ' ' + q(step + '.py') + ' > logs/' + step + '_cmsRun.log 2>&1',
              'python3 check_output.py ' + q(step + '.root') + ('' if previous is None else ' ' + q(previous + '.root'))]
    if i == len(steps) - 1:
        lines += ['edmDumpEventContent ' + q(step + '.root') + ' > logs/MiniAOD_content.log',
                  'grep -F \'"displacedMuons"\' logs/MiniAOD_content.log',
                  'grep -F \'"displacedTracks"\' logs/MiniAOD_content.log',
                  'grep -F \'"displacedStandAloneMuons"\' logs/MiniAOD_content.log',
                  'grep -F \'"displacedGlobalMuons"\' logs/MiniAOD_content.log',
                  'grep -F \'"patUnfilteredDisplacedMuons"\' logs/MiniAOD_content.log']
    lines += [')']
    previous = step
lines += ['echo "ALL_STAGES_PASS"']
(out / 'run_test.sh').write_text('\n'.join(lines) + '\n')
print(out / 'run_test.sh')
