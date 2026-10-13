"""Run DisplacedMuonDoubleCounting with the existing iDM ntuplizer setup.

From CMSSW_13_0_13/src/iDMe, after scram runtime -sh:
  cmsRun /path/to/your_diagnostic_cfg.py year=2022 data=0 signal=1 \
      flist=fileLists/signal/2022EE/your_sample_flist.txt \
      outfile=/tmp/unused_diagnostic.root

Pass the same arguments you normally give scripts/ElectronNtuplizer_cfg.py.
The outfile argument may be required while that config is parsed; its output
module and TFileService are removed from the diagnostic schedule below.
"""
import os
import runpy

import FWCore.ParameterSet.Config as cms

# The production config supplies the exact input source, conditions, geometry,
# magnetic field, and Station-2 PropagateToMuon settings used by this workflow.
production_cfg = os.environ.get(
    "IDM_NTUPLE_CFG", "scripts/ElectronNtuplizer_cfg.py"
)
process = runpy.run_path(production_cfg, run_name="__main__")["process"]

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
