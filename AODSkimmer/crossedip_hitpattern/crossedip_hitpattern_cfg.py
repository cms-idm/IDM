"""cmsRun config for CrossedIPHitPatternCheck (iDMe/AODSkimmer/plugins/CrossedIPHitPatternCheck.cc).

  cmsRun crossedip_hitpattern_cfg.py flist=files.txt [nEvents=-1] [outFile=crossedip_hitpattern.root]
  python3 summarize_crossedip_hitpattern.py crossedip_hitpattern*.root

Compares the exact crossedIP (TrackExtra) of displacedMuon inner tracks with the HitPattern proxy
(kb/studies/shared-track-duplicates.md). Input: Run3Summer22EE MiniAOD of the iDMmu 09-28 production
(AODSkimmer/fileLists/signal/2022EE/iDMmu/*_flist.txt). flist may also be a single ROOT file.
"""
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
from Configuration.Eras.Era_Run3_cff import Run3
from RecoMuon.MuonIdentification.muons1stStep_cfi import muons1stStep

options = VarParsing("analysis")
options.register("flist", "", VarParsing.multiplicity.singleton, VarParsing.varType.string, "text file, one input file per line, or one ROOT file")
options.register("nEvents", -1, VarParsing.multiplicity.singleton, VarParsing.varType.int, "events to process")
options.register("outFile", "crossedip_hitpattern.root", VarParsing.multiplicity.singleton, VarParsing.varType.string, "output ROOT file")
options.parseArguments()

process = cms.Process("XIPHP", Run3)
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 10000
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, "130X_mcRun3_2022_realistic_postEE_v6", "")

if options.flist.endswith(".root"):
    files = [options.flist]
else:
    with open(options.flist) as src:
        files = [line.strip() for line in src if line.strip() and not line.startswith("#")]
process.source = cms.Source("PoolSource", fileNames=cms.untracked.vstring(files))
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(options.nEvents))
process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi")
process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi")
process.TFileService = cms.Service("TFileService", fileName=cms.string(options.outFile))

tap = muons1stStep.TrackAssociatorParameters
process.crossedIPHitPatternCheck = cms.EDAnalyzer(
    "CrossedIPHitPatternCheck",
    muons=cms.InputTag("displacedMuons"),
    muonMaxDistanceX=tap.muonMaxDistanceX,  # chamber-edge tolerance of the direction-consistency P test
    muonMaxDistanceY=tap.muonMaxDistanceY,
)
process.p = cms.Path(process.crossedIPHitPatternCheck)
