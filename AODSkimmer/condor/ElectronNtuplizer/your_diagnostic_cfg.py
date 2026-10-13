"""Run DisplacedMuonDoubleCounting with the existing iDM ntuplizer setup.

Place this file at AODSkimmer/condor/ElectronNtuplizer/your_diagnostic_cfg.py.
Run it from CMSSW_13_0_13/src/iDMe/AODSkimmer after scram runtime -sh, passing the
same input file-list and other arguments as ElectronNtuplizer_cfg.py.

Pass the same arguments you normally give scripts/ElectronNtuplizer_cfg.py.
The outfile argument may be required while that config is parsed; its output
module and TFileService are removed from the diagnostic schedule below.
"""
import os
import runpy
import sys

import FWCore.ParameterSet.Config as cms

# The production config supplies the exact input source, conditions, geometry,
# magnetic field, and Station-2 PropagateToMuon settings used by this workflow.
production_cfg = os.environ.get("IDM_NTUPLE_CFG") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../scripts/ElectronNtuplizer_cfg.py")
)
# cmsRun leaves the outer cfg filename in sys.argv. The production cfg's
# VarParsing must see only its own filename and the key=value arguments.
original_argv = sys.argv[:]
outer_cfg = os.path.abspath(__file__)
sys.argv = [sys.argv[0]] + [
    arg for arg in sys.argv[1:]
    if os.path.abspath(arg) != outer_cfg
]
try:
    process = runpy.run_path(production_cfg, run_name="__main__")["process"]
finally:
    sys.argv = original_argv

skimmers = [
    module for module in process.analyzers_().values()
    if module.type_() == "ElectronSkimmer"
]
if len(skimmers) != 1:
    raise RuntimeError(
        "Expected exactly one ElectronSkimmer in the production config; "
        "found {}".format(len(skimmers))
    )
skimmer = skimmers[0]

process.doubleCounting = cms.EDAnalyzer(
    "DisplacedMuonDoubleCounting",
    displacedMuons=skimmer.displacedMuons,
    genParticle=skimmer.genParticle,
    genMuonPropagatorSt2=skimmer.genMuonPropagatorSt2.clone(),
    maxDR=cms.double(0.1),
)
process.doubleCountingPath = cms.Path(process.doubleCounting)
process.schedule = cms.Schedule(process.doubleCountingPath)

# The diagnostic prints only its endJob summary.
if hasattr(process, "TFileService"):
    del process.TFileService
