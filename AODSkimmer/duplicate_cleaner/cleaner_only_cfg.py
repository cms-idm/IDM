"""Run only DisplacedMuonDuplicateCleaner (validate on) and keep its products, for check_cleaner.py --edm.

  cmsRun cleaner_only_cfg.py flist=files.txt outFile=cleaned.root [nEvents=-1]

flist: text file with one input file per line, or one ROOT file (iDMmu MiniAOD keeping displacedMuons).
"""
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

options = VarParsing("analysis")
options.register("flist", "", VarParsing.multiplicity.singleton, VarParsing.varType.string, "file list or one ROOT file")
options.register("outFile", "cleaned.root", VarParsing.multiplicity.singleton, VarParsing.varType.string, "EDM output")
options.register("nEvents", -1, VarParsing.multiplicity.singleton, VarParsing.varType.int, "events to process")
options.parseArguments()

process = cms.Process("DUPCLEAN")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 5000

if options.flist.endswith(".root"):
    files = [options.flist]
else:
    with open(options.flist) as src:
        files = [line.strip() for line in src if line.strip() and not line.startswith("#")]
process.source = cms.Source("PoolSource", fileNames=cms.untracked.vstring(files))
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(options.nEvents))

process.load("iDMe.AODSkimmer.cleanedDisplacedMuons_cff")
process.cleanedDisplacedMuons.validate = True
process.p = cms.Path(process.cleanedDisplacedMuons)
process.out = cms.OutputModule(
    "PoolOutputModule",
    fileName=cms.untracked.string(options.outFile),
    outputCommands=cms.untracked.vstring("drop *", "keep *_cleanedDisplacedMuons_*_DUPCLEAN"),
)
process.e = cms.EndPath(process.out)
