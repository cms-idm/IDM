"""All-event hierarchy matching and vertexing of pre-selected cleaned displacedMuons (study note: kb/studies/preselected-vertexing-efficiency.md).
cmsRun preselected_vertexing_cfg.py flist=<original-list|root> sample=<name> outFile=<new.root> revision=<sha> [hierarchyOrder=2,0,1]
hierarchyOrder: component types best first (0 inner, 1 outer/DSA, 2 global); default 2,0,1 = Global > Inner > DSA (user decision
2026-10-22). Runs before 2026-10-22 used the then-fixed order 2,1,0 (Global > DSA > Inner).
"""
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
from Configuration.Eras.Era_Run3_cff import Run3
options = VarParsing("analysis")
for key, default in (("flist", ""), ("sample", ""), ("outFile", "component_inputs.root"), ("revision", "unspecified"),
                     ("hierarchyOrder", "2,0,1")):
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
process = cms.Process("PRESELVTX", Run3)
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 5000
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, "130X_mcRun3_2022_realistic_postEE_v6", "")
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")
for name in ("Along", "Opposite", "Any"):
    process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagator%s_cfi" % name)
process.source = cms.Source("PoolSource", fileNames=cms.untracked.vstring(files))
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(options.nEvents))
process.TFileService = cms.Service("TFileService", fileName=cms.string(options.outFile))
process.cleanedDisplacedMuons = cms.EDProducer("DisplacedMuonDuplicateCleaner",
    muons=cms.InputTag("displacedMuons", "", "RECO"), validate=cms.bool(True))
process.preselectedVertexing = cms.EDAnalyzer("PreselectedVertexing",
    genParticles=cms.InputTag("prunedGenParticles"), muons=cms.InputTag("cleanedDisplacedMuons"),
    originalIndex=cms.InputTag("cleanedDisplacedMuons", "originalIndex"), beamspot=cms.InputTag("offlineBeamSpot"),
    met=cms.InputTag("slimmedMETs"), genMet=cms.InputTag("genMetTrue"), packedCandidates=cms.InputTag("packedPFCandidates"),
    hierarchyOrder=cms.vint32([int(x) for x in options.hierarchyOrder.split(",")]), sample=cms.string(options.sample),
    revision=cms.string(options.revision), propagatorSt2=cms.PSet(
        useSimpleGeometry=cms.bool(True), useStation2=cms.bool(True), fallbackToME1=cms.bool(False),
        cosmicPropagationHypothesis=cms.bool(False), useMB2InOverlap=cms.bool(False),
        useTrack=cms.string("none"), useState=cms.string("atVertex"),
        propagatorAlong=cms.ESInputTag("", "SteppingHelixPropagatorAlong"),
        propagatorAny=cms.ESInputTag("", "SteppingHelixPropagatorAny"),
        propagatorOpposite=cms.ESInputTag("", "SteppingHelixPropagatorOpposite")))
process.p = cms.Path(process.cleanedDisplacedMuons + process.preselectedVertexing)
