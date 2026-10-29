#!/usr/bin/env python3
"""Prepare one bounded W+jets runtime test from a saved SampleFactory chain.

Preparation only: does not submit jobs, execute CMSSW, or write to EOS.
Run on LPC in a new directory; inputs are explicit snapshots.
"""
import argparse
import json
from pathlib import Path
import shlex
import shutil

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--chain', required=True)
p.add_argument('--fragment', required=True)
p.add_argument('--pileup-list', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
chain = json.loads(Path(a.chain).read_text())
steps = chain['STEPS']
if steps != ['Run3Summer22EEGENSIM', 'Run3Summer22EEDIGIPremix',
             'Run3Summer22EERECO', 'Run3Summer22EEMiniAOD']:
    raise SystemExit('Unexpected chain: review before preparing this test')
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
if not t or t.GetEntries() != 100:
    raise SystemExit("Expected exactly 100 test events")
print("PASS: reopened output with 100 test events:", sys.argv[1])
f.Close()
''')
q = shlex.quote
lines = ['#!/usr/bin/env bash', 'set -eo pipefail',
         'cd ' + q(str(out)),
         'test ! -e logs/started || { echo "Test already started; use a new run directory"; exit 1; }',
         'test -n "${X509_USER_PROXY:-}" && test -r "$X509_USER_PROXY"',
         'voms-proxy-info -exists -valid 1:00',
         'date -u > logs/started',
         'exec > >(tee -a logs/execution.log) 2>&1',
         'source /cvmfs/cms.cern.ch/cmsset_default.sh',
         'trap \'s=$?; echo "TEST_EXIT=$s"; exit "$s"\' EXIT',
         'date -u', 'hostname', 'cat /etc/redhat-release']
previous = None
for i, step in enumerate(steps):
    wf = chain['WORKFLOWS'][step]
    release = wf['CMSSW_VERSION']
    opts = wf['OPTIONS']
    for values in wf.get('CUSTOMIZES', {}).values():
        if values:
            raise SystemExit('Nonempty CUSTOMIZES requires explicit review')
    lines += ['(', 'export SCRAM_ARCH=' + q(wf['SCRAM_ARCH']),
              'test -d ' + q(release) + ' || scram project CMSSW ' + q(release),
              'cd ' + q(release + '/src'), 'eval "$(scram runtime -sh)"']
    args = ['cmsDriver.py']
    if i == 0:
        lines += ['mkdir -p Configuration/GenProduction/python',
                  'cp ../../fragment.py Configuration/GenProduction/python/wjets_fragment.py',
                  'scram b -j 2']
        args += ['Configuration/GenProduction/python/wjets_fragment.py',
                 '--number', '2000', '--number_out', '100',
                 # One stream: an in-flight event on a second stream can be written past --number_out.
                 '--nStreams', '1',
                 '--customise_commands',
                 'process.externalLHEProducer.nEvents=cms.untracked.uint32(2000); '
                 'process.RandomNumberGeneratorService.externalLHEProducer.initialSeed=cms.untracked.uint32(29102026)']
    else:
        args += [step, '--number', '-1', '--filein', 'file:' + str(out / (previous + '.root'))]
    args += ['--nThreads', '2', '--no_exec', '--fileout', 'file:' + str(out / (step + '.root')),
             '--python_filename', str(out / (step + '.py'))]
    for key, value in opts.items():
        args.append('--' + key)
        if value is not None:
            if key == 'pileup_input':
                value = 'filelist:' + str(out / 'pileup.txt')
            elif key == 'customise_commands':
                value = shlex.split(value)[0]
            args.append(str(value))
    lines += ['cd ' + q(str(out)), 'python3 --version', 'root-config --version',
              'echo ' + q('RELEASE=' + release + ' STEP=' + step),
              'echo ' + q(shlex.join(args)) + ' >> logs/commands.log',
              shlex.join(args) + ' > logs/' + step + '_cmsDriver.log 2>&1',
              'echo ' + q('cmsRun -j ' + step + '_FrameworkJobReport.xml ' + step + '.py') + ' >> logs/commands.log',
              '/usr/bin/time -v cmsRun -j ' + q(step + '_FrameworkJobReport.xml') + ' ' + q(step + '.py') + ' > logs/' + step + '_cmsRun.log 2>&1',
              'python3 check_output.py ' + q(step + '.root')]
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
