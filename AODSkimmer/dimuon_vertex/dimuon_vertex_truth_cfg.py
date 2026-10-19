"""cmsRun config for DimuonVertexTruth (iDMe/AODSkimmer/plugins/DimuonVertexTruth.cc).

  cmsRun dimuon_vertex_truth_cfg.py flist=files.txt sample=Mchi-10p5_dMchi-1p0_ctau-10 [nEvents=-1] [outFile=dimuon_vertex_truth.root]

Input: Run3Summer22EE MiniAOD of the iDMmu production (2026-09-23 for step 1, 2026-09-28 from step 2;
AODSkimmer/fileLists/signal/2022EE/iDMmu/*_flist.txt), which keeps displacedMuons and the
displaced track collections. flist may also be a single ROOT file.
Station-2 and station-1 propagation use the ntuplizer's genMuonPropagatorSt2/St1 settings.
Fallback for gen muons failing station 2: inner tracks at the ECAL inner boundary, global/outer at station 1.
"""
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
from Configuration.Eras.Era_Run3_cff import Run3

options = VarParsing("analysis")
options.register("flist", "", VarParsing.multiplicity.singleton, VarParsing.varType.string, "text file, one input file per line, or one ROOT file")
options.register("sample", "", VarParsing.multiplicity.singleton, VarParsing.varType.string, "sample label stored in the output")
options.register("nEvents", -1, VarParsing.multiplicity.singleton, VarParsing.varType.int, "events to process")
options.register("outFile", "dimuon_vertex_truth.root", VarParsing.multiplicity.singleton, VarParsing.varType.string, "output ROOT file")
options.parseArguments()

process = cms.Process("DIMUVTX", Run3)
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, "130X_mcRun3_2022_realistic_postEE_v6", "")
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")
process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi")
process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi")
process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi")
process.load("TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff")  # ECAL inner boundary

if options.flist.endswith(".root"):
    files = [options.flist]
else:
    with open(options.flist) as src:
        files = [line.strip() for line in src if line.strip() and not line.startswith("#")]
process.source = cms.Source("PoolSource", fileNames=cms.untracked.vstring(files))
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(options.nEvents))
process.TFileService = cms.Service("TFileService", fileName=cms.string(options.outFile))

process.dimuonVertexTruth = cms.EDAnalyzer(
    "DimuonVertexTruth",
    muons=cms.InputTag("cleanedDisplacedMuons"),  # deduplicated (kb/DECISIONS.md 2026-10-19)
    genParticles=cms.InputTag("prunedGenParticles"),
    beamspot=cms.InputTag("offlineBeamSpot"),
    standAloneMuons=cms.InputTag("displacedStandAloneMuons"),  # raw DSA tracks for the hierarchy merging check
    pfMuons=cms.InputTag("slimmedMuons"),  # PF muon matching (pf* branches)
    met=cms.InputTag("slimmedMETs"),  # Type-1 PF MET (pfMet*)
    sample=cms.string(options.sample),
    drInner=cms.double(0.06),
    drGlobal=cms.double(0.06),
    drOuter=cms.double(0.10),
    # Same as genMuonPropagatorSt2 / genMuonPropagatorSt1 in scripts/ElectronNtuplizer_cfg.py.
    propagatorSt2=cms.PSet(
        useSimpleGeometry=cms.bool(True),
        useStation2=cms.bool(True),
        fallbackToME1=cms.bool(False),
        cosmicPropagationHypothesis=cms.bool(False),
        useMB2InOverlap=cms.bool(False),
        useTrack=cms.string("none"),
        useState=cms.string("atVertex"),
        propagatorAlong=cms.ESInputTag("", "SteppingHelixPropagatorAlong"),
        propagatorAny=cms.ESInputTag("", "SteppingHelixPropagatorAny"),
        propagatorOpposite=cms.ESInputTag("", "SteppingHelixPropagatorOpposite"),
    ),
)
process.dimuonVertexTruth.propagatorSt1 = process.dimuonVertexTruth.propagatorSt2.clone(useStation2=False)
# Same module as python/cleanedDisplacedMuons_cff.py, defined inline because the condor jobs ship
# only lib/ (condor_dimuon_vertex_truth.sh), not the python area.
process.cleanedDisplacedMuons = cms.EDProducer(
    "DisplacedMuonDuplicateCleaner",
    muons=cms.InputTag("displacedMuons", "", "RECO"),
    validate=cms.bool(False),
)
process.p = cms.Path(process.cleanedDisplacedMuons + process.dimuonVertexTruth)
