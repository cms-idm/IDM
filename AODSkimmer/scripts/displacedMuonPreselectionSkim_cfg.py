"""Preserve all input products in events with >=1 cleaned ID-passing displacedMuon (no eta/pT cut).

cmsRun displacedMuonPreselectionSkim_cfg.py flist=<list.txt|input.root> outFile=<new.root> [nEvents=-1]
Flags: displacedMuonPreselection:<instance>:PRESKIM, vector<int>, indexed to RAW displacedMuons::RECO.
Instances: {inner,outer,global}{Present,PassID,InEta}, keptAfterCleaning, preselected, preselectedEta.
InEta means finite |component track eta| < 2.4, independently of ID; absent components have all 0.
preselectedEta reproduces the original preselection AFTER cleaning; preselected omits the eta gate.
Original products (including raw muons and their track targets) and cleaner products are all kept.
No conditions/geometry are needed: only stored track fields and muon ID bits are read.
"""
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

options = VarParsing("analysis")
options.register("flist", "", VarParsing.multiplicity.singleton, VarParsing.varType.string, "input list or one ROOT file")
options.register("outFile", "preselectedMiniAOD.root", VarParsing.multiplicity.singleton,
                 VarParsing.varType.string, "new EDM output")
options.register("nEvents", -1, VarParsing.multiplicity.singleton, VarParsing.varType.int, "input events (-1: all)")
options.parseArguments()
if not options.flist:
    raise ValueError("flist is required")
if options.flist.endswith(".root"):
    files = [options.flist]
else:
    with open(options.flist) as source:
        files = [line.strip() for line in source if line.strip() and not line.lstrip().startswith("#")]
if not files:
    raise ValueError("Empty input list")

process = cms.Process("PRESKIM")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 5000
process.options = cms.untracked.PSet(wantSummary=cms.untracked.bool(True))
process.source = cms.Source("PoolSource", fileNames=cms.untracked.vstring(files))
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(options.nEvents))
process.load("iDMe.AODSkimmer.cleanedDisplacedMuons_cff")
process.cleanedDisplacedMuons.validate = True
process.displacedMuonPreselection = cms.EDFilter(
    "DisplacedMuonPreselectionFilter",
    muons=cms.InputTag("displacedMuons", "", "RECO"),
    originalIndex=cms.InputTag("cleanedDisplacedMuons", "originalIndex"),
)
process.skim = cms.Path(process.cleanedDisplacedMuons + process.displacedMuonPreselection)
process.out = cms.OutputModule(
    "PoolOutputModule",
    fileName=cms.untracked.string(options.outFile),
    outputCommands=cms.untracked.vstring("keep *"),
    dropMetaData=cms.untracked.string("NONE"),
    SelectEvents=cms.untracked.PSet(SelectEvents=cms.vstring("skim")),
)
process.end = cms.EndPath(process.out)
