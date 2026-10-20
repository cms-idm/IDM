"""All-event station-2 inputs for cleaned component matching; never use the event skim as sole input.
cmsRun component_efficiency_cfg.py flist=<original-list|root> sample=<name> outFile=<new.root> revision=<sha>
"""
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
from Configuration.Eras.Era_Run3_cff import Run3
options = VarParsing("analysis")
for key, default in (("flist", ""), ("sample", ""), ("outFile", "component_inputs.root"), ("revision", "unspecified")):
    options.register(key, default, VarParsing.multiplicity.singleton, VarParsing.varType.string, key)
options.register("nEvents", -1, VarParsing.multiplicity.singleton, VarParsing.varType.int, "events")
options.parseArguments()
if not options.flist or not options.sample:
    raise ValueError("flist and sample are required")
if options.flist.endswith(".root"):
    files = [options.flist]
else:
    with open(options.flist) as source:
        files = [line.strip() for line in source if line.strip() and not line.lstrip().startswith("#")]
if not files:
    raise ValueError("Empty input list")
process = cms.Process("COMPEFF", Run3)
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 5000
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, "130X_mcRun3_2022_realistic_postEE_v6", "")
for name in ("Along", "Opposite", "Any"):
    process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagator%s_cfi" % name)
process.source = cms.Source("PoolSource", fileNames=cms.untracked.vstring(files))
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(options.nEvents))
process.TFileService = cms.Service("TFileService", fileName=cms.string(options.outFile))
process.cleanedDisplacedMuons = cms.EDProducer("DisplacedMuonDuplicateCleaner",
    muons=cms.InputTag("displacedMuons", "", "RECO"), validate=cms.bool(True))
process.componentInputs = cms.EDAnalyzer("ComponentMatchingInputs",
    genParticles=cms.InputTag("prunedGenParticles"), muons=cms.InputTag("cleanedDisplacedMuons"),
    originalIndex=cms.InputTag("cleanedDisplacedMuons", "originalIndex"), sample=cms.string(options.sample),
    revision=cms.string(options.revision), propagatorSt2=cms.PSet(
        useSimpleGeometry=cms.bool(True), useStation2=cms.bool(True), fallbackToME1=cms.bool(False),
        cosmicPropagationHypothesis=cms.bool(False), useMB2InOverlap=cms.bool(False),
        useTrack=cms.string("none"), useState=cms.string("atVertex"),
        propagatorAlong=cms.ESInputTag("", "SteppingHelixPropagatorAlong"),
        propagatorAny=cms.ESInputTag("", "SteppingHelixPropagatorAny"),
        propagatorOpposite=cms.ESInputTag("", "SteppingHelixPropagatorOpposite")))
process.p = cms.Path(process.cleanedDisplacedMuons + process.componentInputs)
